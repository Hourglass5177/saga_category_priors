from __future__ import annotations

"""Filesystem-backed production hooks for :mod:`v10_orchestrator`.

This adapter contains no training or download fallback.  It consumes complete
V9 S-AM lifting banks, resumes V10 ObjectBank/replay outputs, and runs offline
evaluation.  A deployment that is authorised to *only* materialise missing
lifting may inject ``ensure_lifting(scene_id)`` explicitly.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, read_rows, write_json, write_rows
from .taxonomy import Taxonomy, default_taxonomy_path, load_taxonomy
from .v10_evaluation import audit_v10_associations, evaluate_v10_replays
from .v10_lifting_worker import compatible_lifting_bank_is_complete
from .v10_pipeline import DEV2, DEV8, VIEW_CONSENSUS_ARM
from .v10_replay import replay_v10_priors
from .v10_runner import run_v10_banks
from .v9_evaluation import score_iou_spearman
from .v9_metrics import (
    evaluate_v9_predictions,
    metrics_by_condition,
    paired_scannet_bootstrap_from_predictions,
    scene_metrics,
)


EnsureLifting = Callable[[str], Any]


@dataclass(frozen=True)
class FilesystemV10Config:
    runtime_manifest: Path
    gt_dir: Path
    lifting_root: Path
    bank_root: Path
    replay_root: Path
    artifacts_root: Path
    category_priors: Path
    size_bins: Path
    b1_fixed_prediction_root: Path
    b1_fixed_condition: str
    v9_closeout: Path
    git_commit: str
    taxonomy_path: Path | None = None
    locked_runtime_manifest: Path | None = None
    locked_gt_dir: Path | None = None
    ensure_lifting: EnsureLifting | None = None


def _threshold_label(value: float) -> str:
    if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
        raise ValueError("acceptance threshold must be finite and in [0, 1]")
    return f"{float(value):.2f}".replace(".", "p")


def _scope(scene_ids: Sequence[str]) -> str:
    scenes = tuple(map(str, scene_ids))
    if scenes == DEV2:
        return "dev2"
    if scenes == DEV8:
        return "dev8"
    return f"scenes{len(scenes)}"


def _require_complete_lifting(path: Path, scene_id: str) -> None:
    if not compatible_lifting_bank_is_complete(path, expected_scene_id=str(scene_id)):
        raise RuntimeError(
            f"missing or incomplete registered V9/V10 S-AM lifting for {scene_id}: {path}"
        )


def _condition_metrics(analysis: Mapping[str, Any], condition: str) -> dict[str, Any]:
    values = metrics_by_condition(analysis)
    if str(condition) not in values:
        raise ValueError(f"evaluation is missing condition {condition}")
    return values[str(condition)]


def _cached_json(path: Path) -> dict[str, Any] | None:
    try:
        value = load_json(path)
    except (OSError, ValueError, TypeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _rows_parseable(path: Path) -> bool:
    try:
        read_rows(path)
        return True
    except (OSError, ValueError, TypeError, RuntimeError):
        return False


def _file_identity(path: Path) -> dict[str, Any]:
    target = Path(path).resolve()
    stat = target.stat()
    return {
        "path": str(target),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _files_identity(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [_file_identity(path) for path in sorted(map(Path, paths), key=str)]


def _b1_output_path(root: Path, condition: str, scene_id: str) -> Path:
    direct = Path(root) / str(condition) / str(scene_id)
    if (direct / "output.json").is_file():
        return direct / "output.json"
    seeded = direct / "seed-42" / "output.json"
    if seeded.is_file():
        return seeded
    raise FileNotFoundError(direct / "output.json")


def _candidate_scores(diagnostics: Mapping[str, Any]) -> dict[int, float]:
    result: dict[int, float] = {}
    for row in diagnostics["candidate_scores"]:
        candidate_id = int(row["candidate_id"])
        if candidate_id in result:
            raise ValueError("replay diagnostics contain duplicate candidate scores")
        result[candidate_id] = float(row["score"])
    return result


def _candidate_owner(output: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> np.ndarray:
    labels = np.asarray(output["point_labels"], dtype=np.int32)
    owner = np.full(labels.shape, -1, dtype=np.int32)
    instances = diagnostics["instances"]
    for raw_instance_id, row in instances.items():
        instance_id = int(raw_instance_id)
        candidate_id = int(row["candidate_id"])
        owner[labels == instance_id] = candidate_id
    return owner


class FilesystemV10Hooks:
    """Production implementation of the V10 orchestrator hook protocol."""

    def __init__(self, config: FilesystemV10Config) -> None:
        self.config = config
        self.taxonomy: Taxonomy = load_taxonomy(config.taxonomy_path)
        self.taxonomy_source = Path(
            config.taxonomy_path if config.taxonomy_path is not None else default_taxonomy_path()
        ).resolve()
        self.config.bank_root.mkdir(parents=True, exist_ok=True)
        self.config.replay_root.mkdir(parents=True, exist_ok=True)
        self.config.artifacts_root.mkdir(parents=True, exist_ok=True)
        if not str(config.git_commit).strip():
            raise ValueError("git_commit must be non-empty")

    def _audit_source_identity(
        self,
        *,
        scene_ids: Sequence[str],
        conditions: Sequence[str],
        runtime_manifest: Path | None = None,
        gt_dir: Path | None = None,
    ) -> dict[str, Any]:
        scenes = tuple(map(str, scene_ids))
        manifest = Path(runtime_manifest or self.config.runtime_manifest)
        truth = Path(gt_dir or self.config.gt_dir)
        bank_files = [
            self.config.bank_root / condition / scene_id / name
            for condition in map(str, conditions)
            for scene_id in scenes
            for name in ("object_bank.json", "object_bank.npz")
        ]
        return {
            "runtime_manifest": _file_identity(manifest),
            "ground_truth": _files_identity(
                [truth / f"{scene_id}.npz" for scene_id in scenes]
            ),
            "banks": _files_identity(bank_files),
            "size_bins": _file_identity(self.config.size_bins),
            "taxonomy": _file_identity(self.taxonomy_source),
        }

    def _replay_source_identity(
        self,
        *,
        replay_root: Path,
        scene_ids: Sequence[str],
        classifier: str,
        conditions: Sequence[str],
        runtime_manifest: Path,
        gt_dir: Path,
    ) -> dict[str, Any]:
        scenes = tuple(map(str, scene_ids))
        replay_files = [
            Path(replay_root)
            / VIEW_CONSENSUS_ARM
            / str(classifier)
            / str(condition)
            / scene_id
            / name
            for condition in map(str, conditions)
            for scene_id in scenes
            for name in ("output.json", "diagnostics.json")
        ]
        return {
            "runtime_manifest": _file_identity(runtime_manifest),
            "ground_truth": _files_identity(
                [Path(gt_dir) / f"{scene_id}.npz" for scene_id in scenes]
            ),
            "replays": _files_identity(replay_files),
            "size_bins": _file_identity(self.config.size_bins),
            "taxonomy": _file_identity(self.taxonomy_source),
        }

    def closeout_v9(self) -> Mapping[str, Any]:
        path = Path(self.config.v9_closeout)
        if not path.is_file():
            raise RuntimeError(f"V9 closeout artifact is missing: {path}")
        payload = load_json(path)
        if not isinstance(payload, Mapping):
            raise ValueError("V9 closeout must be a JSON object")
        return dict(payload)

    def _ensure_scene_lifting(self, scene_id: str) -> None:
        target = Path(self.config.lifting_root) / str(scene_id)
        if compatible_lifting_bank_is_complete(target, expected_scene_id=str(scene_id)):
            return
        if str(scene_id) in DEV2:
            raise RuntimeError(
                "DEV2 must reuse the completed V9 S-AM lifting; regeneration is "
                f"not authorised ({scene_id}: {target})"
            )
        callback = self.config.ensure_lifting
        if callback is None:
            raise RuntimeError(
                f"V10 needs lifting for {scene_id}, but no ensure_lifting(scene_id) "
                "callback was supplied. No training or download was started."
            )
        callback(str(scene_id))
        _require_complete_lifting(target, str(scene_id))

    def ensure_banks(
        self,
        *,
        scene_ids: Sequence[str],
        structure_conditions: Sequence[str],
    ) -> Mapping[str, Any]:
        normalized = tuple(map(str, scene_ids))
        for scene_id in normalized:
            self._ensure_scene_lifting(scene_id)
        return run_v10_banks(
            lifting_root=self.config.lifting_root,
            output_root=self.config.bank_root,
            scene_ids=normalized,
            git_commit=self.config.git_commit,
            conditions=tuple(map(str, structure_conditions)),
        )

    def _audit(
        self,
        *,
        scene_ids: Sequence[str],
        conditions: Sequence[str],
        classifiers: Sequence[str],
        stem: str,
    ) -> tuple[dict[str, Any], Path]:
        rows_path = self.config.artifacts_root / f"{stem}.parquet"
        analysis_path = self.config.artifacts_root / f"{stem}.json"
        expected_keys = {
            f"{condition}/{classifier}"
            for condition in map(str, conditions)
            for classifier in map(str, classifiers)
        }
        source_identity = self._audit_source_identity(
            scene_ids=scene_ids,
            conditions=conditions,
        )
        if rows_path.is_file() and analysis_path.is_file():
            cached = _cached_json(analysis_path)
            if (
                cached is not None
                and _rows_parseable(rows_path)
                and cached.get("schema") == "saga-v10-association-audit-v1"
                and cached.get("runtime_git_commit") == self.config.git_commit
                and cached.get("scene_ids") == list(map(str, scene_ids))
                and set(cached.get("conditions", {})) == expected_keys
                and Path(str(cached.get("bank_root", ""))).resolve()
                == self.config.bank_root.resolve()
                and cached.get("source_identity") == source_identity
            ):
                return dict(cached), rows_path
        payload = audit_v10_associations(
            runtime_manifest=self.config.runtime_manifest,
            gt_dir=self.config.gt_dir,
            bank_root=self.config.bank_root,
            scene_ids=tuple(map(str, scene_ids)),
            conditions=tuple(map(str, conditions)),
            classifiers=tuple(map(str, classifiers)),
            taxonomy=self.taxonomy,
            rows_output=rows_path,
            analysis_output=analysis_path,
            size_bins=self.config.size_bins,
        )
        result = {
            **payload,
            "runtime_git_commit": self.config.git_commit,
            "bank_root": str(self.config.bank_root.resolve()),
            "source_identity": source_identity,
        }
        write_json(analysis_path, result)
        return result, rows_path

    def audit_dev2_structures(
        self,
        *,
        scene_ids: Sequence[str],
        structure_conditions: Sequence[str],
    ) -> Mapping[str, Mapping[str, Any]]:
        if tuple(map(str, scene_ids)) != DEV2:
            raise ValueError("DEV2 structure audit must use the registered two scenes")
        payload, _ = self._audit(
            scene_ids=scene_ids,
            conditions=structure_conditions,
            classifiers=("mv-label",),
            stem="v10_association_funnel2",
        )
        result = {
            str(condition): dict(
                payload["conditions"][f"{condition}/mv-label"]["gate_metrics"]
            )
            for condition in map(str, structure_conditions)
        }
        write_rows(
            self.config.artifacts_root / "v10_pair_reconstruction_factorial2.parquet",
            [
                {"condition": condition, **result[condition]}
                for condition in ("P0R0", "P1R0", "P0R1", "P1R1")
                if condition in result
            ],
        )
        if "VC1" in result:
            write_json(
                self.config.artifacts_root / "v10_view_consensus2.json",
                {
                    "schema": "saga-v10-view-consensus2-v1",
                    "scene_ids": list(map(str, scene_ids)),
                    "git_commit": self.config.git_commit,
                    "gate_metrics": result["VC1"],
                },
            )
        return result

    def _dev8_audit(self) -> tuple[dict[str, Any], Path]:
        return self._audit(
            scene_ids=DEV8,
            conditions=(VIEW_CONSENSUS_ARM,),
            classifiers=("mv-label", "codebook"),
            stem="v10_bank8",
        )

    def classifier_metrics_dev8(
        self,
        *,
        scene_ids: Sequence[str],
        structure_condition: str,
    ) -> Mapping[str, Mapping[str, Any]]:
        if tuple(map(str, scene_ids)) != DEV8 or structure_condition != VIEW_CONSENSUS_ARM:
            raise ValueError("late classifier selection is registered on DEV8/VC1")
        payload, _ = self._dev8_audit()
        result: dict[str, Mapping[str, Any]] = {}
        for classifier in ("mv-label", "codebook"):
            final = payload["conditions"][f"VC1/{classifier}"]["stages"][
                "final_candidate"
            ]
            result[classifier] = {
                "geometric_candidate_match_025_count": int(
                    final["candidate_match_025_count"]
                ),
                "late_classifier_correct_025_count": int(
                    final["late_classifier_correct_025_count"]
                ),
            }
        return result

    def _replay_base(self, acceptance_threshold: float) -> Path:
        return self.config.replay_root / f"threshold-{_threshold_label(acceptance_threshold)}"

    def _run_replay(
        self,
        *,
        scene_ids: Sequence[str],
        classifier: str,
        acceptance_threshold: float,
        conditions: Sequence[str],
    ) -> Path:
        root = self._replay_base(acceptance_threshold)
        replay_v10_priors(
            bank_root=self.config.bank_root,
            output_root=root,
            scene_ids=tuple(map(str, scene_ids)),
            structure_conditions=(VIEW_CONSENSUS_ARM,),
            prior_conditions=tuple(map(str, conditions)),
            classifier=str(classifier),
            category_priors=self.config.category_priors,
            acceptance_threshold=float(acceptance_threshold),
            git_commit=self.config.git_commit,
        )
        return root

    def _evaluate_v10(
        self,
        *,
        scene_ids: Sequence[str],
        classifier: str,
        acceptance_threshold: float,
        conditions: Sequence[str],
        stem: str,
        runtime_manifest: Path | None = None,
        gt_dir: Path | None = None,
        viewer_output: Path | None = None,
    ) -> dict[str, Any]:
        replay_root = self._run_replay(
            scene_ids=scene_ids,
            classifier=classifier,
            acceptance_threshold=acceptance_threshold,
            conditions=conditions,
        )
        active_runtime = Path(runtime_manifest or self.config.runtime_manifest)
        active_gt = Path(gt_dir or self.config.gt_dir)
        source_identity = self._replay_source_identity(
            replay_root=replay_root,
            scene_ids=scene_ids,
            classifier=classifier,
            conditions=conditions,
            runtime_manifest=active_runtime,
            gt_dir=active_gt,
        )
        metrics_path = self.config.artifacts_root / f"{stem}.parquet"
        analysis_path = self.config.artifacts_root / f"{stem}.json"
        if metrics_path.is_file() and analysis_path.is_file():
            cached = _cached_json(analysis_path)
            if (
                cached is not None
                and _rows_parseable(metrics_path)
                and cached.get("schema") == "saga-v10-object-system-analysis-v1"
                and cached.get("runtime_git_commit") == self.config.git_commit
                and cached.get("scene_ids") == list(map(str, scene_ids))
                and cached.get("evaluated_conditions") == list(map(str, conditions))
                and cached.get("classifier") == str(classifier)
                and float(cached.get("acceptance_threshold", -1.0))
                == float(acceptance_threshold)
                and (
                    viewer_output is None
                    or (
                        isinstance(cached.get("viewer"), Mapping)
                        and Path(str(cached["viewer"].get("directory", ""))).is_dir()
                    )
                )
                and cached.get("source_identity") == source_identity
            ):
                return dict(cached)
        payload = evaluate_v10_replays(
            runtime_manifest=active_runtime,
            gt_dir=active_gt,
            replay_root=replay_root,
            structure_condition=VIEW_CONSENSUS_ARM,
            classifier=str(classifier),
            scene_ids=tuple(map(str, scene_ids)),
            conditions=tuple(map(str, conditions)),
            taxonomy=self.taxonomy,
            metrics_output=metrics_path,
            analysis_output=analysis_path,
            size_bins=self.config.size_bins,
            viewer_output=viewer_output,
        )
        result = {
            **payload,
            "runtime_git_commit": self.config.git_commit,
            "scene_ids": list(map(str, scene_ids)),
            "evaluated_conditions": list(map(str, conditions)),
            "acceptance_threshold": float(acceptance_threshold),
            "source_identity": source_identity,
        }
        write_json(analysis_path, result)
        return result

    def _evaluate_b1(self, scene_ids: Sequence[str], *, stem: str) -> dict[str, Any]:
        metrics_path = self.config.artifacts_root / f"{stem}.parquet"
        analysis_path = self.config.artifacts_root / f"{stem}.json"
        scenes = tuple(map(str, scene_ids))
        source_identity = {
            "runtime_manifest": _file_identity(self.config.runtime_manifest),
            "ground_truth": _files_identity(
                [self.config.gt_dir / f"{scene_id}.npz" for scene_id in scenes]
            ),
            "prediction_root": str(self.config.b1_fixed_prediction_root.resolve()),
            "condition": self.config.b1_fixed_condition,
            "predictions": _files_identity(
                [
                    _b1_output_path(
                        self.config.b1_fixed_prediction_root,
                        self.config.b1_fixed_condition,
                        scene_id,
                    )
                    for scene_id in scenes
                ]
            ),
            "size_bins": _file_identity(self.config.size_bins),
            "taxonomy": _file_identity(self.taxonomy_source),
        }
        if metrics_path.is_file() and analysis_path.is_file():
            cached = _cached_json(analysis_path)
            if (
                cached is not None
                and _rows_parseable(metrics_path)
                and cached.get("schema") == "saga-v9-object-system-analysis-v1"
                and cached.get("runtime_git_commit") == self.config.git_commit
                and cached.get("scene_ids") == list(map(str, scene_ids))
                and cached.get("source_identity") == source_identity
            ):
                return dict(cached)
        payload = evaluate_v9_predictions(
            runtime_manifest=self.config.runtime_manifest,
            gt_dir=self.config.gt_dir,
            prediction_root=self.config.b1_fixed_prediction_root,
            scene_ids=tuple(map(str, scene_ids)),
            conditions=(self.config.b1_fixed_condition,),
            taxonomy=self.taxonomy,
            metrics_output=metrics_path,
            analysis_output=analysis_path,
            size_bins=self.config.size_bins,
        )
        result = {
            **payload,
            "runtime_git_commit": self.config.git_commit,
            "scene_ids": list(map(str, scene_ids)),
            "source_identity": source_identity,
        }
        write_json(analysis_path, result)
        return result

    def threshold_sweep_dev2(
        self,
        *,
        scene_ids: Sequence[str],
        structure_condition: str,
        classifier: str,
        thresholds: Sequence[float],
    ) -> Sequence[Mapping[str, Any]]:
        if tuple(map(str, scene_ids)) != DEV2 or structure_condition != VIEW_CONSENSUS_ARM:
            raise ValueError("uniform threshold selection is registered on DEV2/VC1")
        b1 = _condition_metrics(
            self._evaluate_b1(DEV2, stem="v10_b1_fixed_dev2"),
            self.config.b1_fixed_condition,
        )
        rows: list[dict[str, Any]] = []
        for threshold in map(float, thresholds):
            analysis = self._evaluate_v10(
                scene_ids=DEV2,
                classifier=classifier,
                acceptance_threshold=threshold,
                conditions=("U000",),
                stem=f"v10_u000_dev2_t{_threshold_label(threshold)}",
            )
            uniform = _condition_metrics(analysis, "U000")
            structure_passed = (
                float(uniform["map_50_95"]) >= float(b1["map_50_95"]) - 0.001
                and float(uniform["ap50"]) >= float(b1["ap50"]) - 0.002
                and int(uniform["predicted_instance_count"])
                <= 1.25 * int(b1["predicted_instance_count"])
            )
            rows.append(
                {
                    "acceptance_threshold": threshold,
                    "map_50_95": float(uniform["map_50_95"]),
                    "ap50": float(uniform["ap50"]),
                    "predicted_instance_count": int(
                        uniform["predicted_instance_count"]
                    ),
                    "structure_passed": bool(structure_passed),
                }
            )
        write_json(
            self.config.artifacts_root / "v10_u000_threshold_selection_inputs.json",
            {"schema": "saga-v10-threshold-inputs-v1", "rows": rows},
        )
        return rows

    def _replay_diagnostics(
        self,
        *,
        scene_id: str,
        classifier: str,
        condition: str,
        acceptance_threshold: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        target = (
            self._replay_base(acceptance_threshold)
            / VIEW_CONSENSUS_ARM
            / str(classifier)
            / str(condition)
            / str(scene_id)
        )
        return load_json(target / "diagnostics.json"), load_json(target / "output.json")

    def _score_iou(self, *, classifier: str, acceptance_threshold: float) -> float:
        _, rows_path = self._dev8_audit()
        audit_rows = read_rows(rows_path)
        scores: list[float] = []
        ious: list[float] = []
        diagnostics = {
            scene_id: self._replay_diagnostics(
                scene_id=scene_id,
                classifier=classifier,
                condition="U000",
                acceptance_threshold=acceptance_threshold,
            )[0]
            for scene_id in DEV8
        }
        score_maps = {
            scene_id: _candidate_scores(row) for scene_id, row in diagnostics.items()
        }
        for row in audit_rows:
            if (
                row.get("row_type") == "stage_candidate"
                and row.get("stage") == "final_candidate"
                and row.get("condition") == VIEW_CONSENSUS_ARM
                and row.get("classifier") == classifier
            ):
                scene_id = str(row["scene_id"])
                candidate_id = int(row["candidate_id"])
                scores.append(score_maps[scene_id][candidate_id])
                ious.append(float(row["best_official_iou"]))
        return score_iou_spearman(scores, ious)

    def uniform_health_inputs_dev8(
        self,
        *,
        scene_ids: Sequence[str],
        structure_condition: str,
        classifier: str,
        acceptance_threshold: float,
    ) -> Mapping[str, Mapping[str, Any]]:
        if tuple(map(str, scene_ids)) != DEV8 or structure_condition != VIEW_CONSENSUS_ARM:
            raise ValueError("uniform health is registered on DEV8/VC1")
        official = self._evaluate_v10(
            scene_ids=DEV8,
            classifier=classifier,
            acceptance_threshold=acceptance_threshold,
            conditions=("U000",),
            stem=f"v10_uniform_health8_{classifier}_t{_threshold_label(acceptance_threshold)}",
            viewer_output=self.config.artifacts_root / "viewer" / "dev8-uniform",
        )
        uniform = _condition_metrics(official, "U000")
        audit, _ = self._dev8_audit()
        final = audit["conditions"][f"VC1/{classifier}"]["stages"]["final_candidate"]
        bank = {
            **uniform,
            "geometric_match_050_count": int(final["candidate_match_050_count"]),
            "geometric_match_050_scene_count": int(
                final["candidate_match_050_scene_count"]
            ),
            "same_class_match_050_count": int(
                final["same_class_candidate_match_050_count"]
            ),
            "same_class_match_050_scene_count": int(
                final["same_class_candidate_match_050_scene_count"]
            ),
            "same_class_candidate_precision_025": float(
                final["same_class_candidate_precision_025"]
            ),
            "tiny_small_recall_025": float(
                final["same_class_tiny_small_recall_025"]
            ),
            "score_iou_spearman": self._score_iou(
                classifier=classifier,
                acceptance_threshold=acceptance_threshold,
            ),
        }
        b1 = _condition_metrics(
            self._evaluate_b1(DEV8, stem="v10_b1_fixed_dev8"),
            self.config.b1_fixed_condition,
        )
        result = {"bank": bank, "b1_fixed": b1}
        write_json(
            self.config.artifacts_root / "v10_uniform_health8.json",
            {
                "schema": "saga-v10-uniform-health8-inputs-v1",
                "scene_ids": list(DEV8),
                "classifier": classifier,
                "acceptance_threshold": float(acceptance_threshold),
                **result,
            },
        )
        return result

    def prior_factorial_dev8(
        self,
        *,
        scene_ids: Sequence[str],
        structure_condition: str,
        classifier: str,
        acceptance_threshold: float,
        prior_conditions: Sequence[str],
    ) -> Mapping[str, Mapping[str, Any]]:
        if tuple(map(str, scene_ids)) != DEV8 or structure_condition != VIEW_CONSENSUS_ARM:
            raise ValueError("prior factorial is registered on DEV8/VC1")
        conditions = tuple(map(str, prior_conditions))
        stem = f"v10_prior_factorial8_{classifier}_t{_threshold_label(acceptance_threshold)}"
        analysis = self._evaluate_v10(
            scene_ids=DEV8,
            classifier=classifier,
            acceptance_threshold=acceptance_threshold,
            conditions=conditions,
            stem=stem,
            viewer_output=self.config.artifacts_root / "viewer" / "dev8-prior",
        )
        source_metrics = self.config.artifacts_root / f"{stem}.parquet"
        if source_metrics.is_file():
            write_rows(
                self.config.artifacts_root / "v10_prior_factorial8.parquet",
                read_rows(source_metrics),
            )
        aggregates = metrics_by_condition(analysis)
        uniform_state = {
            scene_id: self._replay_diagnostics(
                scene_id=scene_id,
                classifier=classifier,
                condition="U000",
                acceptance_threshold=acceptance_threshold,
            )
            for scene_id in DEV8
        }
        result: dict[str, Mapping[str, Any]] = {}
        for condition in conditions:
            score_deltas: list[float] = []
            changed = False
            for scene_id in DEV8:
                diagnostics, output = self._replay_diagnostics(
                    scene_id=scene_id,
                    classifier=classifier,
                    condition=condition,
                    acceptance_threshold=acceptance_threshold,
                )
                uniform_diagnostics, uniform_output = uniform_state[scene_id]
                uniform_scores = _candidate_scores(uniform_diagnostics)
                data_scores = _candidate_scores(diagnostics)
                if set(uniform_scores) != set(data_scores):
                    raise ValueError("prior replay changed the frozen candidate set")
                score_deltas.extend(
                    data_scores[candidate_id] - uniform_scores[candidate_id]
                    for candidate_id in sorted(uniform_scores)
                )
                changed = changed or (
                    set(diagnostics["accepted_candidate_ids"])
                    != set(uniform_diagnostics["accepted_candidate_ids"])
                    or not np.array_equal(
                        _candidate_owner(output, diagnostics),
                        _candidate_owner(uniform_output, uniform_diagnostics),
                    )
                )
            result[condition] = {
                "rows": scene_metrics(analysis, condition),
                "aggregate": aggregates[condition],
                "candidate_score_deltas": score_deltas,
                "accepted_or_ownership_changed": bool(changed),
            }
        return result

    def _comparison(
        self,
        *,
        scene_ids: Sequence[str],
        classifier: str,
        acceptance_threshold: float,
        data_condition: str,
        stem: str,
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        self.ensure_banks(
            scene_ids=scene_ids,
            structure_conditions=(VIEW_CONSENSUS_ARM,),
        )
        analysis = self._evaluate_v10(
            scene_ids=scene_ids,
            classifier=classifier,
            acceptance_threshold=acceptance_threshold,
            conditions=("U000", str(data_condition)),
            stem=stem,
        )
        return {
            "uniform_rows": scene_metrics(analysis, "U000"),
            "data_rows": scene_metrics(analysis, str(data_condition)),
        }

    def holdout5_comparison(
        self,
        *,
        scene_ids: Sequence[str],
        structure_condition: str,
        classifier: str,
        acceptance_threshold: float,
        data_condition: str,
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        if structure_condition != VIEW_CONSENSUS_ARM:
            raise ValueError("holdout5 comparison requires VC1")
        return self._comparison(
            scene_ids=scene_ids,
            classifier=classifier,
            acceptance_threshold=float(acceptance_threshold),
            data_condition=str(data_condition),
            stem="v10_holdout5_metrics",
        )

    def tune24_comparison(
        self,
        *,
        scene_ids: Sequence[str],
        structure_condition: str,
        classifier: str,
        acceptance_threshold: float,
        data_condition: str,
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        if structure_condition != VIEW_CONSENSUS_ARM:
            raise ValueError("tune24 comparison requires VC1")
        return self._comparison(
            scene_ids=scene_ids,
            classifier=classifier,
            acceptance_threshold=float(acceptance_threshold),
            data_condition=str(data_condition),
            stem="v10_tune24_metrics",
        )

    def final48_bootstrap(
        self,
        *,
        scene_ids: Sequence[str],
        structure_condition: str,
        classifier: str,
        acceptance_threshold: float,
        data_condition: str,
    ) -> Mapping[str, Any]:
        if structure_condition != VIEW_CONSENSUS_ARM:
            raise ValueError("final48 comparison requires VC1")
        runtime_manifest = self.config.locked_runtime_manifest
        gt_dir = self.config.locked_gt_dir
        if runtime_manifest is None or gt_dir is None:
            raise RuntimeError(
                "final48 requires locked_runtime_manifest and locked_gt_dir"
            )
        scene_ids = tuple(map(str, scene_ids))
        classifier = str(classifier)
        threshold = float(acceptance_threshold)
        data_condition = str(data_condition)
        self.ensure_banks(
            scene_ids=scene_ids,
            structure_conditions=(VIEW_CONSENSUS_ARM,),
        )
        self._evaluate_v10(
            scene_ids=scene_ids,
            classifier=classifier,
            acceptance_threshold=threshold,
            conditions=("U000", data_condition),
            stem="v10_final_metrics",
            runtime_manifest=runtime_manifest,
            gt_dir=gt_dir,
            viewer_output=self.config.artifacts_root / "viewer" / "final48",
        )
        replay_root = self._replay_base(threshold)
        source_identity = self._replay_source_identity(
            replay_root=replay_root,
            scene_ids=scene_ids,
            classifier=classifier,
            conditions=("U000", data_condition),
            runtime_manifest=runtime_manifest,
            gt_dir=gt_dir,
        )
        output_path = self.config.artifacts_root / "v10_final_bootstrap.json"
        if output_path.is_file():
            cached = _cached_json(output_path)
            if (
                cached is not None
                and cached.get("runtime_git_commit") == self.config.git_commit
                and cached.get("scene_ids") == list(scene_ids)
                and cached.get("classifier") == classifier
                and cached.get("data_condition") == data_condition
                and float(cached.get("acceptance_threshold", -1.0)) == threshold
                and cached.get("source_identity") == source_identity
            ):
                return dict(cached)
        payload = paired_scannet_bootstrap_from_predictions(
            runtime_manifest=runtime_manifest,
            gt_dir=gt_dir,
            prediction_root=replay_root / VIEW_CONSENSUS_ARM / classifier,
            scene_ids=scene_ids,
            reference_condition="U000",
            treatment_condition=data_condition,
            taxonomy=self.taxonomy,
            samples=10_000,
            seed=20_260_804,
        )
        result = {
            **payload,
            "runtime_git_commit": self.config.git_commit,
            "scene_ids": list(scene_ids),
            "classifier": classifier,
            "data_condition": data_condition,
            "acceptance_threshold": threshold,
            "source_identity": source_identity,
        }
        write_json(output_path, result)
        return result

from __future__ import annotations

"""Sequential, resumable runner for section-30 candidate formation.

The worker command receives no GT path.  It performs one semantic selection
and one HDBSCAN run per class, then persists C0/C1/C2 and the lossless trace.
Ground truth is consumed only by the separate diagnose/evaluate commands.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import hashlib
import json

import numpy as np

from .category_candidate_trace import (
    assert_candidate_bank_identity,
    load_candidate_formation_trace,
    validate_candidate_formation_trace,
    validate_candidate_formation_trace_views,
)
from .category_denoise import load_candidate_bank
from .category_denoise_runner import (
    _build_command,
    _normalize_scene_ids,
    _run_command,
    _valid_candidate_bank,
    _valid_prediction,
    _validate_common_inputs,
)
from .io import load_json
from .io import sha256_file, write_json


CANDIDATE_REPAIR_CONDITIONS = (
    "C0-legacy",
    "C1-consistent-envelope",
    "C2-raw-anchored-envelope",
)
TRACE_VIEW_FILES = (
    "formation_trace.npz",
    "formation_trace.json",
    "sample_rank.npz",
    "raw_hdbscan_labels.npz",
    "raw_membership.npz",
    "prethreshold_assignment.npz",
    "distance_components.npz",
    "raw_clusters.json",
    "trace_diagnostics.json",
)


def _reference_scene_root(root: Path, scene_id: str) -> Path:
    direct = root / scene_id
    return direct if direct.is_dir() else root / "bank" / scene_id


def _candidate_scene_complete(
    root: Path,
    scene_id: str,
    *,
    seed: int,
    sample_cap: int = 5_000,
    reference_bank_root: Path | None = None,
) -> bool:
    try:
        prediction = root / "b0" / scene_id / "output.json"
        bank_scene = root / "bank" / scene_id
        trace_scene = root / "candidate_trace" / scene_id
        if not _valid_prediction(prediction):
            return False
        if not (bank_scene / "candidate_repair.log").is_file():
            return False
        if not all((trace_scene / name).is_file() for name in TRACE_VIEW_FILES):
            return False
        trace = load_candidate_formation_trace(trace_scene)
        validate_candidate_formation_trace_views(trace, trace_scene)
        banks = {
            condition: load_candidate_bank(bank_scene / condition)
            for condition in CANDIDATE_REPAIR_CONDITIONS
        }
        for condition, bank in banks.items():
            if not _valid_candidate_bank(
                bank_scene / condition, scene_id=scene_id, seed=seed
            ):
                return False
        validate_candidate_formation_trace(trace, bank=banks["C0-legacy"])
        if (
            trace.scene_id != scene_id
            or trace.seed != int(seed)
            or trace.sample_cap != int(sample_cap)
        ):
            return False
        if reference_bank_root is not None:
            reference = load_candidate_bank(
                _reference_scene_root(reference_bank_root, scene_id)
            )
            assert_candidate_bank_identity(reference, banks["C0-legacy"])
        return True
    except (OSError, EOFError, KeyError, TypeError, ValueError):
        return False


def repair_category_candidates(
    runtime_manifest: str | Path,
    output_root: str | Path,
    repo_root: str | Path,
    category_priors: str | Path,
    scene_ids: str | Sequence[str] | None = None,
    *,
    reference_bank_root: str | Path | None = None,
    seed: int = 42,
    sample_cap: int = 5_000,
    python_bin: str | Path | None = None,
) -> dict[str, Any]:
    """Generate and validate one C0/C1/C2 family per selected scene."""

    if isinstance(sample_cap, bool) or int(sample_cap) <= 0:
        raise ValueError("sample_cap must be a positive integer")
    scenes, pipeline, priors, root = _validate_common_inputs(
        runtime_manifest, repo_root, category_priors, output_root, seed
    )
    selected = _normalize_scene_ids(scene_ids, scenes)
    reference = (
        Path(reference_bank_root).resolve()
        if reference_bank_root is not None
        else None
    )
    if reference is not None and not reference.is_dir():
        raise FileNotFoundError(reference)

    rows: list[dict[str, Any]] = []
    for scene_id in selected:
        bank_scene = root / "bank" / scene_id
        trace_scene = root / "candidate_trace" / scene_id
        b0_scene = root / "b0" / scene_id
        bank_scene.mkdir(parents=True, exist_ok=True)
        trace_scene.mkdir(parents=True, exist_ok=True)
        b0_scene.mkdir(parents=True, exist_ok=True)
        if _candidate_scene_complete(
            root,
            scene_id,
            seed=seed,
            sample_cap=sample_cap,
            reference_bank_root=reference,
        ):
            rows.append(
                {
                    "scene_id": scene_id,
                    "status": "skipped_complete",
                    "bank_root": str(bank_scene),
                    "trace_root": str(trace_scene),
                }
            )
            continue

        command = _build_command(
            pipeline_path=pipeline,
            priors_path=priors,
            scene_id=scene_id,
            scene=scenes[scene_id],
            output_path=b0_scene / "output.json",
            progress_path=b0_scene / "progress.txt",
            diagnostics_path=b0_scene / "diagnostics.json",
            bank_path=bank_scene,
            action="candidate-repair",
            mode="uniform",
            seed=seed,
            python_bin=python_bin,
        )
        command.extend(("--category-candidate-trace-path", str(trace_scene)))
        command.extend(("--category-candidate-sample-cap", str(int(sample_cap))))
        return_code = _run_command(
            command,
            cwd=pipeline.parent,
            log_path=bank_scene / "candidate_repair.log",
        )
        if return_code != 0 or not _candidate_scene_complete(
            root,
            scene_id,
            seed=seed,
            sample_cap=sample_cap,
            reference_bank_root=reference,
        ):
            raise RuntimeError(
                f"candidate repair failed or did not reproduce C0 for {scene_id}; "
                f"inspect {bank_scene / 'candidate_repair.log'}"
            )
        rows.append(
            {
                "scene_id": scene_id,
                "status": "complete",
                "bank_root": str(bank_scene),
                "trace_root": str(trace_scene),
            }
        )
    return {
        "action": "candidate-repair",
        "seed": int(seed),
        "sample_cap": int(sample_cap),
        "reference_identity_required": reference is not None,
        "total": len(rows),
        "complete": len(rows),
        "runs": rows,
    }


def _candidate_replay_complete(
    scene_root: Path,
    *,
    scene_id: str,
    mode: str,
    score_threshold: float,
    expected_identity: Mapping[str, Any],
) -> bool:
    try:
        if not _valid_prediction(scene_root / "output.json"):
            return False
        diagnostics = load_json(scene_root / "diagnostics.json")
        identity = diagnostics.get("category_denoise", diagnostics)
        if not isinstance(identity, Mapping):
            return False
        runner_identity = load_json(scene_root / "runner_identity.json")
        return bool(
            (scene_root / "postprocess.log").is_file()
            and identity.get("action") == "candidate-replay"
            and identity.get("scene_id") == scene_id
            and identity.get("mode") == mode
            and abs(float(identity.get("score_threshold")) - score_threshold)
            <= 1e-12
            and int(identity.get("protected_or_reinserted_point_count", -1)) == 0
            and isinstance(identity.get("decisions"), list)
            and isinstance(identity.get("candidate_survival"), list)
            and runner_identity == dict(expected_identity)
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _update_digest_array(digest: Any, name: str, value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(name.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def _candidate_bank_digest(bank: Any) -> tuple[str, str]:
    """Return content and candidate-ID/Q digests without writing SHA files."""

    content = hashlib.sha256()
    for name in (
        "global_pre_knn",
        "semantic_top1",
        "semantic_top1_score",
        "branch_full_labels",
        "branch_core_labels",
        "assignment_confidence",
    ):
        _update_digest_array(content, name, getattr(bank, name))
    candidate_rows = json.dumps(
        list(bank.candidates),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    content.update(candidate_rows)

    q_identity = hashlib.sha256()
    for row in sorted(bank.candidates, key=lambda item: int(item["candidate_id"])):
        q_identity.update(
            json.dumps(
                [int(row["candidate_id"]), float(row.get("base_score", row.get("Q", 0.0)))],
                separators=(",", ":"),
            ).encode("ascii")
        )
    return content.hexdigest(), q_identity.hexdigest()


def _replay_identity(
    *,
    bank: Any,
    bank_path: Path,
    priors_path: Path,
    scene_id: str,
    mode: str,
    score_threshold: float,
    seed: int,
) -> dict[str, Any]:
    bank_digest, q_digest = _candidate_bank_digest(bank)
    diagnostics = bank.diagnostics if isinstance(bank.diagnostics, Mapping) else {}
    return {
        "schema": "saga-category-candidate-replay-runner-identity-v1",
        "scene_id": scene_id,
        "mode": mode,
        "score_threshold": float(score_threshold),
        "seed": int(seed),
        "legacy_knn_k": 256,
        "legacy_filter_min_count": 10,
        "bank_path": str(bank_path.resolve()),
        "bank_digest": bank_digest,
        "candidate_id_q_digest": q_digest,
        "candidate_count": len(bank.candidates),
        "repair_condition": diagnostics.get("candidate_repair_condition", "C0-legacy"),
        "sample_cap": int(diagnostics.get("sample_cap", -1)),
        "category_priors_path": str(priors_path.resolve()),
        "category_priors_sha256": sha256_file(priors_path),
    }


def _resolve_replay_bank(
    bank_root: Path,
    scene_id: str,
    selected_condition: str | None,
) -> Path:
    direct = bank_root / scene_id
    if selected_condition is None:
        return direct
    candidates = (
        direct / selected_condition,
        bank_root / "bank" / scene_id / selected_condition,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def replay_repaired_category_candidates(
    runtime_manifest: str | Path,
    bank_root: str | Path,
    output_root: str | Path,
    repo_root: str | Path,
    category_priors: str | Path,
    scene_ids: str | Sequence[str] | None = None,
    modes: str | Sequence[str] = ("uniform", "class"),
    *,
    score_threshold: float,
    selected_condition: str | None = None,
    seed: int = 42,
    python_bin: str | Path | None = None,
) -> dict[str, Any]:
    """Run same-bank U/D through the unchanged full-scene legacy denoiser."""

    threshold = float(score_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("score_threshold must be in [0, 1]")
    normalized_modes = (modes,) if isinstance(modes, str) else tuple(map(str, modes))
    if not normalized_modes or len(normalized_modes) != len(set(normalized_modes)):
        raise ValueError("modes must be a non-empty unique sequence")
    if not set(normalized_modes).issubset({"uniform", "class"}):
        raise ValueError("modes may contain only uniform and class")
    normalized_modes = tuple(
        mode for mode in ("uniform", "class") if mode in normalized_modes
    )
    scenes, pipeline, priors, root = _validate_common_inputs(
        runtime_manifest, repo_root, category_priors, output_root, seed
    )
    selected = _normalize_scene_ids(scene_ids, scenes)
    bank_base = Path(bank_root).resolve()
    if not bank_base.is_dir():
        raise FileNotFoundError(bank_base)
    rows: list[dict[str, Any]] = []
    for scene_id in selected:
        scene_bank = _resolve_replay_bank(
            bank_base, scene_id, selected_condition
        )
        if not _valid_candidate_bank(scene_bank, scene_id=scene_id, seed=seed):
            raise ValueError(f"missing or corrupt repaired bank: {scene_bank}")
        bank = load_candidate_bank(scene_bank)
        for mode in normalized_modes:
            scene_root = root / "replay" / mode / scene_id
            scene_root.mkdir(parents=True, exist_ok=True)
            expected_identity = _replay_identity(
                bank=bank,
                bank_path=scene_bank,
                priors_path=priors,
                scene_id=scene_id,
                mode=mode,
                score_threshold=threshold,
                seed=seed,
            )
            if _candidate_replay_complete(
                scene_root,
                scene_id=scene_id,
                mode=mode,
                score_threshold=threshold,
                expected_identity=expected_identity,
            ):
                rows.append(
                    {
                        "scene_id": scene_id,
                        "mode": mode,
                        "status": "skipped_complete",
                        "root": str(scene_root),
                    }
                )
                continue
            command = _build_command(
                pipeline_path=pipeline,
                priors_path=priors,
                scene_id=scene_id,
                scene=scenes[scene_id],
                output_path=scene_root / "output.json",
                progress_path=scene_root / "progress.txt",
                diagnostics_path=scene_root / "diagnostics.json",
                bank_path=scene_bank,
                action="candidate-replay",
                mode=mode,
                seed=seed,
                python_bin=python_bin,
            )
            command.extend(
                ("--category-candidate-score-threshold", str(threshold))
            )
            return_code = _run_command(
                command,
                cwd=pipeline.parent,
                log_path=scene_root / "postprocess.log",
            )
            if return_code == 0:
                write_json(scene_root / "runner_identity.json", expected_identity)
            if return_code != 0 or not _candidate_replay_complete(
                scene_root,
                scene_id=scene_id,
                mode=mode,
                score_threshold=threshold,
                expected_identity=expected_identity,
            ):
                raise RuntimeError(
                    f"candidate replay failed for {mode}/{scene_id}; inspect "
                    f"{scene_root / 'postprocess.log'}"
                )
            rows.append(
                {
                    "scene_id": scene_id,
                    "mode": mode,
                    "status": "complete",
                    "root": str(scene_root),
                }
            )
    return {
        "action": "candidate-replay",
        "seed": int(seed),
        "score_threshold": threshold,
        "selected_condition": selected_condition,
        "total": len(rows),
        "complete": len(rows),
        "runs": rows,
    }


__all__ = [
    "CANDIDATE_REPAIR_CONDITIONS",
    "TRACE_VIEW_FILES",
    "repair_category_candidates",
    "replay_repaired_category_candidates",
]

from __future__ import annotations

import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .io import read_rows, write_json, write_rows
from .runner import load_scene_runtime_manifest
from .taxonomy import Taxonomy


HISTORY_CONDITIONS = ("B0-legacy", "B1-other-classes")
HISTORY_SEEDS = (42, 3407, 20260804)
SIZE_LABELS = ("tiny", "small", "medium", "large")


def _git_commit() -> str:
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _valid_train_instance(row: Mapping[str, Any], classes: set[str]) -> bool:
    if str(row.get("split", "train")) != "train":
        return False
    if str(row.get("canonical_class", "")) not in classes:
        return False
    if not bool(row.get("metric_scale_valid", True)) or not bool(
        row.get("quality_valid", True)
    ):
        return False
    try:
        value = float(row["bbox_diag_m"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(np.isfinite(value) and value > 0)


def build_size_bin_spec(
    train_rows: Sequence[Mapping[str, Any]],
    canonical_classes: Sequence[str],
) -> dict[str, Any]:
    classes = tuple(str(value) for value in canonical_classes)
    allowed = set(classes)
    rows = [row for row in train_rows if _valid_train_instance(row, allowed)]
    if not rows:
        raise ValueError("No valid train instances with positive bbox_diag_m")

    values = np.asarray([float(row["bbox_diag_m"]) for row in rows], dtype=np.float64)
    q25, q50, q75 = np.quantile(values, (0.25, 0.50, 0.75)).tolist()
    boundaries = {
        "tiny_max_m": float(q25),
        "small_max_m": float(q50),
        "medium_max_m": float(q75),
    }

    per_class: dict[str, Any] = {}
    for class_name in classes:
        class_values = np.asarray(
            [
                float(row["bbox_diag_m"])
                for row in rows
                if str(row["canonical_class"]) == class_name
            ],
            dtype=np.float64,
        )
        if class_values.size == 0:
            per_class[class_name] = {"instance_count": 0}
            continue
        class_quantiles = np.quantile(class_values, (0.25, 0.50, 0.75))
        per_class[class_name] = {
            "instance_count": int(class_values.size),
            "bbox_diag_m_q25": float(class_quantiles[0]),
            "bbox_diag_m_q50": float(class_quantiles[1]),
            "bbox_diag_m_q75": float(class_quantiles[2]),
        }

    return {
        "kind": "v3_gt_size_bins",
        "schema_version": "1.0",
        "source_split": "train",
        "source_metric": "bbox_diag_m",
        "quantile_method": "numpy_linear",
        "training_instance_count": int(values.size),
        "boundaries_m": boundaries,
        "definitions": {
            "tiny": "bbox_diag_m <= tiny_max_m",
            "small": "tiny_max_m < bbox_diag_m <= small_max_m",
            "medium": "small_max_m < bbox_diag_m <= medium_max_m",
            "large": "bbox_diag_m > medium_max_m",
        },
        "mapped_point_diagnostic": {
            "below_official_min_region_size": "point_count < 100",
            "official_min_region_size": 100,
        },
        "per_class_train": per_class,
    }


def classify_physical_size(diagonal_m: float, size_spec: Mapping[str, Any]) -> str:
    boundaries = size_spec["boundaries_m"]
    value = float(diagonal_m)
    if value <= float(boundaries["tiny_max_m"]):
        return "tiny"
    if value <= float(boundaries["small_max_m"]):
        return "small"
    if value <= float(boundaries["medium_max_m"]):
        return "medium"
    return "large"


def _oriented_bbox_diagonal(coords: np.ndarray) -> float:
    points = np.asarray(coords, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("Instance coordinates must have shape [N, 3] with N > 0")
    centered = points - points.mean(axis=0, keepdims=True)
    if points.shape[0] >= 3 and np.linalg.matrix_rank(centered) >= 2:
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ axes.T
    else:
        projected = centered
    extents = projected.max(axis=0) - projected.min(axis=0)
    return float(np.linalg.norm(extents))


def load_tune_gt_instances(
    gt_dir: str | Path,
    scenes: Mapping[str, Mapping[str, Any]],
    taxonomy: Taxonomy,
    size_spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    root = Path(gt_dir)
    class_names = tuple(taxonomy.canonical_classes)
    records: list[dict[str, Any]] = []
    for scene_id in sorted(scenes):
        path = root / f"{scene_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing tune GT: {path}")
        with np.load(path, allow_pickle=False) as payload:
            coords = np.asarray(payload["coords"], dtype=np.float64)
            semantic = np.asarray(payload["semantic"], dtype=np.int64)
            instance = np.asarray(payload["instance"], dtype=np.int64)
        if coords.shape != (semantic.size, 3) or instance.shape != semantic.shape:
            raise ValueError(f"{scene_id}: inconsistent GT array shapes")

        valid = (
            (instance >= 0)
            & (semantic >= 0)
            & (semantic < len(class_names))
        )
        physical_scene_id = str(
            scenes[scene_id].get("physical_scene_id")
            or scene_id.rsplit("_", 1)[0]
        )
        for semantic_id, instance_id in sorted(
            set(zip(semantic[valid].tolist(), instance[valid].tolist()))
        ):
            mask = valid & (semantic == semantic_id) & (instance == instance_id)
            point_count = int(mask.sum())
            diagonal_m = _oriented_bbox_diagonal(coords[mask])
            records.append(
                {
                    "scene_id": scene_id,
                    "physical_scene_id": physical_scene_id,
                    "canonical_class": class_names[int(semantic_id)],
                    "semantic_id": int(semantic_id),
                    "instance_id": int(instance_id),
                    "point_count": point_count,
                    "bbox_diag_m": diagonal_m,
                    "physical_size_bin": classify_physical_size(diagonal_m, size_spec),
                    "below_official_min_region_size": point_count < 100,
                }
            )
    return records


def _scene_summaries(
    records: Sequence[Mapping[str, Any]],
    scenes: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_scene[str(row["scene_id"])].append(row)
    for scene_id in sorted(scenes):
        scene_rows = by_scene.get(scene_id, [])
        size_counts = Counter(str(row["physical_size_bin"]) for row in scene_rows)
        class_counts = Counter(str(row["canonical_class"]) for row in scene_rows)
        small_class_counts = Counter(
            str(row["canonical_class"])
            for row in scene_rows
            if str(row["physical_size_bin"]) in {"tiny", "small"}
        )
        summaries[scene_id] = {
            "scene_id": scene_id,
            "physical_scene_id": str(
                scenes[scene_id].get("physical_scene_id")
                or scene_id.rsplit("_", 1)[0]
            ),
            "instance_count": len(scene_rows),
            "size_counts": {label: int(size_counts[label]) for label in SIZE_LABELS},
            "class_counts": dict(sorted(class_counts.items())),
            "tiny_small_class_counts": dict(sorted(small_class_counts.items())),
            "below_official_min_region_size_count": sum(
                bool(row["below_official_min_region_size"]) for row in scene_rows
            ),
        }
    return summaries


def select_diagnostic_scenes(
    records: Sequence[Mapping[str, Any]],
    scenes: Mapping[str, Mapping[str, Any]],
    *,
    budget: int = 8,
    target_small_per_class: int = 2,
) -> dict[str, Any]:
    if budget < 1 or budget > len(scenes):
        raise ValueError("diagnostic budget must be within the available scene count")
    if target_small_per_class < 1:
        raise ValueError("target_small_per_class must be positive")
    summaries = _scene_summaries(records, scenes)
    total_small = Counter()
    for summary in summaries.values():
        total_small.update(summary["tiny_small_class_counts"])
    goals = {
        class_name: min(target_small_per_class, count)
        for class_name, count in sorted(total_small.items())
        if count > 0
    }

    selected: list[str] = []
    selected_physical: set[str] = set()
    achieved = Counter()
    selection_trace: list[dict[str, Any]] = []
    while len(selected) < budget:
        candidates: list[tuple[tuple[float, ...], str, dict[str, Any]]] = []
        for scene_id, summary in summaries.items():
            if scene_id in selected or summary["physical_scene_id"] in selected_physical:
                continue
            counts = summary["tiny_small_class_counts"]
            newly_covered = sum(
                achieved[class_name] == 0 and int(counts.get(class_name, 0)) > 0
                for class_name in goals
            )
            weighted_goal_gain = sum(
                min(
                    max(goals[class_name] - achieved[class_name], 0),
                    int(counts.get(class_name, 0)),
                )
                / max(total_small[class_name], 1)
                for class_name in goals
            )
            tiny_count = int(summary["size_counts"]["tiny"])
            tiny_small_count = tiny_count + int(summary["size_counts"]["small"])
            score = (
                float(newly_covered),
                float(weighted_goal_gain),
                float(tiny_count),
                float(tiny_small_count),
                float(summary["instance_count"]),
            )
            candidates.append((score, scene_id, summary))
        if not candidates:
            raise ValueError("Not enough distinct physical scenes for diagnostic budget")
        best_score = max(item[0] for item in candidates)
        _, scene_id, summary = min(
            (item for item in candidates if item[0] == best_score),
            key=lambda item: item[1],
        )
        selected.append(scene_id)
        selected_physical.add(summary["physical_scene_id"])
        achieved.update(summary["tiny_small_class_counts"])
        selection_trace.append(
            {
                "rank": len(selected),
                "scene_id": scene_id,
                "physical_scene_id": summary["physical_scene_id"],
                "score": {
                    "new_tiny_small_classes": int(best_score[0]),
                    "rare_class_goal_gain": float(best_score[1]),
                    "tiny_instances": int(best_score[2]),
                    "tiny_small_instances": int(best_score[3]),
                    "all_instances": int(best_score[4]),
                },
            }
        )

    for scene_id, summary in summaries.items():
        summary["selected_rank"] = (
            selected.index(scene_id) + 1 if scene_id in selected else None
        )
    return {
        "kind": "v3_diagnostic_scene_selection",
        "schema_version": "1.0",
        "selection_basis": "tune_gt_only",
        "selection_rule": (
            "greedy lexicographic coverage of new tiny/small classes, then "
            "inverse-frequency class goals, tiny count, tiny+small count, and total count"
        ),
        "diagnostic_budget": budget,
        "target_tiny_small_instances_per_class": target_small_per_class,
        "tiny_small_goals": goals,
        "tiny_small_achieved": {
            class_name: int(achieved[class_name]) for class_name in goals
        },
        "selected_scenes": selected,
        "remaining_scenes": [scene_id for scene_id in sorted(scenes) if scene_id not in selected],
        "selection_trace": selection_trace,
        "scene_summaries": [summaries[scene_id] for scene_id in sorted(summaries)],
    }


def prepare_history_anchor(
    rows: Sequence[Mapping[str, Any]],
    *,
    git_commit: str,
) -> list[dict[str, Any]]:
    selected = [
        dict(row)
        for row in rows
        if str(row.get("condition")) in HISTORY_CONDITIONS
        and int(row.get("run_seed")) in HISTORY_SEEDS
    ]
    expected = {(condition, seed) for condition in HISTORY_CONDITIONS for seed in HISTORY_SEEDS}
    observed = {(str(row["condition"]), int(row["run_seed"])) for row in selected}
    if observed != expected or len(selected) != len(expected):
        raise ValueError(
            f"Expected exactly B0/B1 x three seeds; missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )
    protocols = {str(row.get("protocol_version")) for row in selected}
    scene_counts = {int(row.get("scene_count", -1)) for row in selected}
    if len(protocols) != 1 or scene_counts != {48}:
        raise ValueError("History anchor must share one protocol and contain 48 scenes per row")
    for row in selected:
        row["study_role"] = "v3_history_anchor"
        row["git_commit"] = git_commit
    return sorted(selected, key=lambda row: (str(row["condition"]), int(row["run_seed"])))


def prepare_v3_stage0(
    *,
    locked_metrics_path: str | Path,
    train_instance_stats_path: str | Path,
    tune_scene_manifest_path: str | Path,
    tune_gt_dir: str | Path,
    taxonomy: Taxonomy,
    history_output: str | Path,
    size_bins_output: str | Path,
    diagnostic_scenes_output: str | Path,
    diagnostic_budget: int = 8,
    target_small_per_class: int = 2,
    git_commit: str | None = None,
) -> dict[str, Any]:
    commit = git_commit or _git_commit()
    history = prepare_history_anchor(read_rows(locked_metrics_path), git_commit=commit)
    size_spec = build_size_bin_spec(
        read_rows(train_instance_stats_path), taxonomy.canonical_classes
    )
    scenes = load_scene_runtime_manifest(tune_scene_manifest_path)
    instances = load_tune_gt_instances(tune_gt_dir, scenes, taxonomy, size_spec)
    selection = select_diagnostic_scenes(
        instances,
        scenes,
        budget=diagnostic_budget,
        target_small_per_class=target_small_per_class,
    )

    size_spec["git_commit"] = commit
    size_spec["tune_gt_instance_count"] = len(instances)
    size_spec["tune_gt_size_counts"] = dict(
        sorted(Counter(row["physical_size_bin"] for row in instances).items())
    )
    size_spec["tune_gt_below_official_min_region_size_count"] = sum(
        bool(row["below_official_min_region_size"]) for row in instances
    )
    selection["git_commit"] = commit
    selection["size_bin_source"] = str(size_bins_output)

    write_rows(history_output, history)
    write_json(size_bins_output, size_spec)
    write_json(diagnostic_scenes_output, selection)

    b0 = [float(row["map_50_95"]) for row in history if row["condition"] == "B0-legacy"]
    b1 = [float(row["map_50_95"]) for row in history if row["condition"] == "B1-other-classes"]
    return {
        "status": "complete",
        "git_commit": commit,
        "history_rows": len(history),
        "history_mean_map_50_95": {
            "B0-legacy": float(np.mean(b0)),
            "B1-other-classes": float(np.mean(b1)),
            "difference": float(np.mean(b1) - np.mean(b0)),
        },
        "train_instance_count": size_spec["training_instance_count"],
        "tune_gt_instance_count": len(instances),
        "selected_scenes": selection["selected_scenes"],
        "remaining_scene_count": len(selection["remaining_scenes"]),
        "outputs": {
            "history": str(history_output),
            "size_bins": str(size_bins_output),
            "diagnostic_scenes": str(diagnostic_scenes_output),
        },
    }

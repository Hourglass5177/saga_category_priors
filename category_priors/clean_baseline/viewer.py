from __future__ import annotations

"""Evaluation-only 3D viewer exports for the clean alpha-mask baseline.

The functions in this module deliberately accept every GT-derived quantity as
an array supplied by the evaluation caller.  They never load a runtime bank,
never construct or modify an object, and are not imported by the evidence or
consensus workers.  This keeps qualitative inspection on the evaluation side
of the train/runtime boundary.
"""

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


VIEWER_SCHEMA = "saga-clean-alpha-mask-viewer-v1"
FORMAL_CONDITION_ORDER = (
    "C0-no-prior",
    "U-global",
    "D-predicted",
)

# Fixed colors requested by the Gaussian -> GT precision audit.
CORRECT_COLOR = np.asarray((0, 190, 0), dtype=np.uint8)
SAME_CLASS_WRONG_INSTANCE_COLOR = np.asarray((255, 210, 0), dtype=np.uint8)
WRONG_CLASS_COLOR = np.asarray((220, 30, 30), dtype=np.uint8)
UNSUPPORTED_COLOR = np.asarray((128, 128, 128), dtype=np.uint8)
GT_COLOR = np.asarray((30, 100, 255), dtype=np.uint8)


def _as_xyz(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return np.ascontiguousarray(result)


def _as_int_vector(value: Any, *, name: str, length: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or len(raw) != int(length):
        raise ValueError(f"{name} must be a vector of length {length}")
    if raw.dtype == np.bool_:
        raise TypeError(f"{name} must contain integers")
    try:
        result = raw.astype(np.int64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must contain integers") from exc
    if not np.array_equal(raw, result):
        raise TypeError(f"{name} must contain integers")
    return np.ascontiguousarray(result)


def _class_value(class_index: int, class_names: Sequence[str] | None) -> str | int:
    value = int(class_index)
    if class_names is None:
        return value
    if not 0 <= value < len(class_names):
        return value
    return str(class_names[value])


def _metadata_class(metadata: Mapping[str, Any]) -> str | int:
    for key in ("class", "class_id", "class_name"):
        if key in metadata:
            value = metadata[key]
            if isinstance(value, (str, int, np.integer)) and not isinstance(
                value, (bool, np.bool_)
            ):
                return str(value) if isinstance(value, str) else int(value)
            raise TypeError(f"instance metadata {key} must be a string or integer")
    raise ValueError("instance metadata is missing class/class_id")


def _metadata_score(metadata: Mapping[str, Any]) -> float:
    value = float(metadata.get("score", 0.0))
    if not math.isfinite(value):
        raise ValueError("instance score must be finite")
    return value


def _stable_object_key(value: Any) -> tuple[int, str]:
    try:
        return 0, f"{int(value):020d}"
    except (TypeError, ValueError, OverflowError):
        return 1, str(value)


def _condition_key(value: str) -> tuple[int, str]:
    try:
        return FORMAL_CONDITION_ORDER.index(str(value)), str(value)
    except ValueError:
        return len(FORMAL_CONDITION_ORDER), str(value)


def _condition_slug(condition: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(condition)
    ).strip("_")


def _write_ascii_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    points = np.asarray(xyz, dtype=np.float64)
    colors = np.asarray(rgb, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("PLY xyz must have shape (N, 3)")
    if colors.shape != points.shape:
        raise ValueError("PLY RGB must have shape (N, 3)")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for point, color in zip(points, colors):
        lines.append(
            "{:.9g} {:.9g} {:.9g} {} {} {}".format(
                float(point[0]),
                float(point[1]),
                float(point[2]),
                int(color[0]),
                int(color[1]),
                int(color[2]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _normalise_predictions(
    predictions: Mapping[str, Mapping[str, Any]],
    *,
    gaussian_count: int,
) -> dict[str, tuple[np.ndarray, dict[int, Mapping[str, Any]]]]:
    result: dict[str, tuple[np.ndarray, dict[int, Mapping[str, Any]]]] = {}
    for raw_condition, payload in predictions.items():
        condition = str(raw_condition)
        if not isinstance(payload, Mapping):
            raise TypeError(f"{condition}: prediction must be a mapping")
        labels = _as_int_vector(
            payload.get("point_labels"),
            name=f"{condition}.point_labels",
            length=gaussian_count,
        )
        if np.any(labels < -1):
            raise ValueError(f"{condition}: only -1 may represent background")
        raw_instances = payload.get("instances")
        if not isinstance(raw_instances, Mapping):
            raise TypeError(f"{condition}.instances must be a mapping")
        instances: dict[int, Mapping[str, Any]] = {}
        for raw_id, metadata in raw_instances.items():
            try:
                instance_id = int(raw_id)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError(f"{condition}: instance IDs must be integers") from exc
            if instance_id < 0 or instance_id in instances:
                raise ValueError(f"{condition}: invalid or duplicate instance ID")
            if not isinstance(metadata, Mapping):
                raise TypeError(f"{condition}/{instance_id}: metadata must be a mapping")
            _metadata_class(metadata)
            _metadata_score(metadata)
            instances[instance_id] = metadata
        declared = set(instances)
        labelled = {int(value) for value in np.unique(labels[labels >= 0])}
        if labelled != declared:
            raise ValueError(
                f"{condition}: point labels and instance metadata disagree"
            )
        result[condition] = labels, instances
    if not result:
        raise ValueError("at least one prediction condition is required")
    return result


def _gt_catalog(
    gt_semantic: np.ndarray,
    gt_instance: np.ndarray,
    class_names: Sequence[str] | None,
) -> dict[int, dict[str, Any]]:
    catalog: dict[int, dict[str, Any]] = {}
    for raw_id in np.unique(gt_instance[gt_instance >= 0]):
        instance_id = int(raw_id)
        mask = gt_instance == instance_id
        classes, counts = np.unique(gt_semantic[mask & (gt_semantic >= 0)], return_counts=True)
        if not len(classes):
            continue
        maximum = int(counts.max())
        class_index = int(classes[counts == maximum].min())
        point_ids = np.flatnonzero(mask & (gt_semantic == class_index)).astype(
            np.int64, copy=False
        )
        catalog[instance_id] = {
            "class": _class_value(class_index, class_names),
            "class_index": class_index,
            "point_ids": point_ids,
            "point_count": int(len(point_ids)),
        }
    return catalog


def _audit_instance(
    *,
    scene_id: str,
    condition: str,
    instance_id: int,
    metadata: Mapping[str, Any],
    labels: np.ndarray,
    gaussian_to_gt_point: np.ndarray,
    gt_semantic: np.ndarray,
    gt_instance: np.ndarray,
    gt_catalog: Mapping[int, Mapping[str, Any]],
    class_names: Sequence[str] | None,
    tiny_small_instance_ids: frozenset[int],
) -> tuple[dict[str, Any], np.ndarray]:
    gaussian_ids = np.flatnonzero(labels == int(instance_id)).astype(
        np.int64, copy=False
    )
    predicted_class = _metadata_class(metadata)
    mapped_point_ids = gaussian_to_gt_point[gaussian_ids]
    valid = (mapped_point_ids >= 0) & (mapped_point_ids < len(gt_instance))
    safe_point_ids = mapped_point_ids.copy()
    safe_point_ids[~valid] = 0
    mapped_gt_instance = gt_instance[safe_point_ids]
    mapped_gt_semantic = gt_semantic[safe_point_ids]
    valid &= (mapped_gt_instance >= 0) & (mapped_gt_semantic >= 0)

    mapped_classes = np.empty(len(gaussian_ids), dtype=object)
    mapped_classes[:] = None
    for index in np.flatnonzero(valid):
        mapped_classes[index] = _class_value(
            int(mapped_gt_semantic[index]), class_names
        )
    class_correct = valid & (mapped_classes == predicted_class)

    same_class_instances = Counter(
        int(value) for value in mapped_gt_instance[class_correct]
    )
    all_instances = Counter(int(value) for value in mapped_gt_instance[valid])
    source = same_class_instances if same_class_instances else all_instances
    if source:
        target_gt_instance = min(
            source, key=lambda value: (-source[value], int(value))
        )
        target = gt_catalog.get(int(target_gt_instance))
    else:
        target_gt_instance = None
        target = None

    target_class = None if target is None else target["class"]
    target_class_index = None if target is None else int(target["class_index"])
    target_same_class = target is not None and predicted_class == target_class
    correct = (
        valid
        & (mapped_gt_instance == int(target_gt_instance))
        & bool(target_same_class)
        if target_gt_instance is not None
        else np.zeros(len(gaussian_ids), dtype=bool)
    )
    same_class_wrong_instance = class_correct & ~correct
    wrong_class = valid & ~class_correct
    unsupported = ~valid
    categories = np.full(len(gaussian_ids), 3, dtype=np.int8)
    categories[correct] = 0
    categories[same_class_wrong_instance] = 1
    categories[wrong_class] = 2

    projected = set(int(value) for value in mapped_point_ids[valid])
    projected.update(
        len(gt_instance) + int(gaussian_id)
        for gaussian_id in gaussian_ids[unsupported]
    )
    target_points = (
        set(int(value) for value in target["point_ids"])
        if target is not None
        else set()
    )
    intersection = len(projected & target_points) if target_same_class else 0
    union = len(projected | target_points)
    same_class_iou = intersection / union if union else 0.0
    recall = intersection / len(target_points) if target_points else 0.0

    touch_threshold = max(1, int(math.ceil(0.05 * int(valid.sum()))))
    touches = Counter(int(value) for value in mapped_gt_instance[valid])
    substantial_touches = sorted(
        value for value, count in touches.items() if count >= touch_threshold
    )
    count = len(gaussian_ids)
    row = {
        "scene_id": str(scene_id),
        "condition": str(condition),
        "instance_id": int(instance_id),
        "class": predicted_class,
        "score": _metadata_score(metadata),
        "predicted_gaussian_count": int(count),
        "correct_gaussian_count": int(correct.sum()),
        "same_class_wrong_instance_count": int(same_class_wrong_instance.sum()),
        "wrong_class_count": int(wrong_class.sum()),
        "unsupported_count": int(unsupported.sum()),
        "point_precision": float(correct.sum() / count) if count else 0.0,
        "class_purity": float(class_correct.sum() / count) if count else 0.0,
        "unsupported_fraction": float(unsupported.sum() / count) if count else 0.0,
        "dominant_gt_instance": (
            None if target_gt_instance is None else int(target_gt_instance)
        ),
        "dominant_gt_class": target_class,
        "dominant_gt_class_index": target_class_index,
        "same_class_iou": float(same_class_iou),
        "gt_to_gaussian_recall": float(recall),
        "matched_at_025": bool(same_class_iou >= 0.25),
        "matched_at_050": bool(same_class_iou >= 0.50),
        "is_tiny_small": bool(
            target_gt_instance is not None
            and int(target_gt_instance) in tiny_small_instance_ids
        ),
        "pure_false_positive": bool(int(valid.sum()) == 0),
        "mapped_gt_count": int(valid.sum()),
        "gt_instances_touched": substantial_touches,
        "merge_gt_instance_count": int(len(substantial_touches)),
        "merge_candidate": bool(len(substantial_touches) >= 2),
        # Filled after all objects in the condition are known.
        "duplicate_prediction": False,
        "split_candidate": False,
        "gaussian_ids": gaussian_ids.tolist(),
    }
    return row, categories


def _mark_splits_and_duplicates(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        target = row["dominant_gt_instance"]
        if target is None or row["class"] != row["dominant_gt_class"]:
            continue
        groups[(str(row["condition"]), int(target), str(row["class"]))].append(row)
    for group in groups.values():
        if len(group) <= 1:
            continue
        ranked = sorted(
            group,
            key=lambda row: (
                -float(row["same_class_iou"]),
                -float(row["point_precision"]),
                -float(row["score"]),
                _stable_object_key(row["instance_id"]),
            ),
        )
        for index, row in enumerate(ranked):
            row["split_candidate"] = True
            row["duplicate_prediction"] = index > 0


def _condition_comparison_rows(
    rows: Sequence[Mapping[str, Any]],
    gt_catalog: Mapping[int, Mapping[str, Any]],
    tiny_small_instance_ids: frozenset[int],
    conditions: Sequence[str],
    *,
    scene_id: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for gt_instance_id in sorted(gt_catalog):
        target = gt_catalog[gt_instance_id]
        for condition in sorted(conditions, key=_condition_key):
            candidates = [
                row
                for row in rows
                if row["condition"] == condition
                and row["dominant_gt_instance"] == gt_instance_id
                and row["class"] == target["class"]
            ]
            candidates.sort(
                key=lambda row: (
                    -float(row["same_class_iou"]),
                    -float(row["point_precision"]),
                    -float(row["score"]),
                    _stable_object_key(row["instance_id"]),
                )
            )
            best = candidates[0] if candidates else None
            result.append(
                {
                    "scene_id": str(scene_id),
                    "gt_instance_id": int(gt_instance_id),
                    "gt_class": target["class"],
                    "is_tiny_small": gt_instance_id in tiny_small_instance_ids,
                    "condition": str(condition),
                    "predicted_instance_id": (
                        None if best is None else int(best["instance_id"])
                    ),
                    "same_class_iou": 0.0 if best is None else float(best["same_class_iou"]),
                    "point_precision": 0.0 if best is None else float(best["point_precision"]),
                    "gt_to_gaussian_recall": 0.0 if best is None else float(best["gt_to_gaussian_recall"]),
                    "score": None if best is None else float(best["score"]),
                    "matched_at_025": bool(best is not None and best["matched_at_025"]),
                    "matched_at_050": bool(best is not None and best["matched_at_050"]),
                }
            )
    return result


def audit_clean_viewer_scene(
    *,
    scene_id: str,
    gaussian_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    gt_semantic: np.ndarray,
    gt_instance: np.ndarray,
    gaussian_to_gt_point: np.ndarray,
    predictions: Mapping[str, Mapping[str, Any]],
    class_names: Sequence[str] | None = None,
    tiny_small_instance_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Create object precision rows without reading any runtime-side GT data."""

    gaussian_points = _as_xyz(gaussian_xyz, name="gaussian_xyz")
    gt_points = _as_xyz(gt_xyz, name="gt_xyz")
    semantic = _as_int_vector(
        gt_semantic, name="gt_semantic", length=len(gt_points)
    )
    instances = _as_int_vector(
        gt_instance, name="gt_instance", length=len(gt_points)
    )
    mapping = _as_int_vector(
        gaussian_to_gt_point,
        name="gaussian_to_gt_point",
        length=len(gaussian_points),
    )
    if np.any(mapping < -1) or np.any(mapping >= len(gt_points)):
        raise ValueError("gaussian_to_gt_point contains an invalid point ID")
    normalized = _normalise_predictions(
        predictions, gaussian_count=len(gaussian_points)
    )
    tiny_small = frozenset(int(value) for value in tiny_small_instance_ids)
    catalog = _gt_catalog(semantic, instances, class_names)

    rows: list[dict[str, Any]] = []
    point_categories: dict[str, list[int]] = {}
    for condition in sorted(normalized, key=_condition_key):
        labels, metadata_by_id = normalized[condition]
        for instance_id in sorted(metadata_by_id):
            row, categories = _audit_instance(
                scene_id=scene_id,
                condition=condition,
                instance_id=instance_id,
                metadata=metadata_by_id[instance_id],
                labels=labels,
                gaussian_to_gt_point=mapping,
                gt_semantic=semantic,
                gt_instance=instances,
                gt_catalog=catalog,
                class_names=class_names,
                tiny_small_instance_ids=tiny_small,
            )
            rows.append(row)
            point_categories[f"{condition}:{instance_id}"] = categories.tolist()
    _mark_splits_and_duplicates(rows)
    rows.sort(
        key=lambda row: (
            _condition_key(str(row["condition"])),
            _stable_object_key(row["instance_id"]),
        )
    )
    comparison = _condition_comparison_rows(
        rows, catalog, tiny_small, tuple(normalized), scene_id=str(scene_id)
    )
    return {
        "schema": VIEWER_SCHEMA,
        "scene_id": str(scene_id),
        "conditions": sorted(normalized, key=_condition_key),
        "objects": rows,
        "point_categories": point_categories,
        "condition_comparison_rows": comparison,
        "diagnostic_space": "3D Gaussian to GT only",
        "contains_2d_render_metrics": False,
    }


def select_clean_viewer_cases(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select fixed qualitative roles deterministically when examples exist."""

    normalized = [dict(row) for row in rows]
    stable = lambda row: (
        _condition_key(str(row["condition"])),
        _stable_object_key(row["instance_id"]),
    )
    matched = [row for row in normalized if row.get("dominant_gt_instance") is not None]
    selections: list[dict[str, Any]] = []

    def add(role: str, candidates: Sequence[dict[str, Any]], key: Any) -> None:
        if not candidates:
            return
        chosen = sorted(candidates, key=lambda row: (key(row), stable(row)))[0]
        selections.append({"role": role, **chosen})

    add(
        "highest_precision",
        matched,
        lambda row: (-float(row["point_precision"]),),
    )
    if matched:
        median = float(np.median([float(row["point_precision"]) for row in matched]))
        add(
            "median_precision",
            matched,
            lambda row: (abs(float(row["point_precision"]) - median),),
        )
    add(
        "lowest_precision",
        matched,
        lambda row: (float(row["point_precision"]),),
    )
    add(
        "pure_false_positive",
        [row for row in normalized if bool(row.get("pure_false_positive"))],
        lambda row: (-int(row["predicted_gaussian_count"]),),
    )
    add(
        "tiny_small_success",
        [
            row
            for row in normalized
            if bool(row.get("is_tiny_small")) and bool(row.get("matched_at_025"))
        ],
        lambda row: (-float(row["same_class_iou"]), -float(row["point_precision"])),
    )
    add(
        "tiny_small_failure",
        [
            row
            for row in normalized
            if bool(row.get("is_tiny_small")) and not bool(row.get("matched_at_025"))
        ],
        lambda row: (float(row["same_class_iou"]), float(row["point_precision"])),
    )
    add(
        "merge_case",
        [row for row in normalized if bool(row.get("merge_candidate"))],
        lambda row: (-int(row["merge_gt_instance_count"]), -int(row["predicted_gaussian_count"])),
    )
    add(
        "split_case",
        [row for row in normalized if bool(row.get("split_candidate"))],
        lambda row: (
            bool(row.get("duplicate_prediction")),
            -float(row["same_class_iou"]),
        ),
    )
    return selections


def _colors_from_categories(categories: np.ndarray) -> np.ndarray:
    result = np.empty((len(categories), 3), dtype=np.uint8)
    result[categories == 0] = CORRECT_COLOR
    result[categories == 1] = SAME_CLASS_WRONG_INSTANCE_COLOR
    result[categories == 2] = WRONG_CLASS_COLOR
    result[categories == 3] = UNSUPPORTED_COLOR
    if np.any((categories < 0) | (categories > 3)):
        raise ValueError("viewer point categories are out of range")
    return result


def export_clean_viewer_scene(
    *,
    audit: Mapping[str, Any],
    gaussian_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    gt_semantic: np.ndarray,
    gt_instance: np.ndarray,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write selected object PLYs and C0/U/D comparison rows."""

    if audit.get("schema") != VIEWER_SCHEMA:
        raise ValueError("unexpected clean-viewer audit schema")
    gaussian_points = _as_xyz(gaussian_xyz, name="gaussian_xyz")
    gt_points = _as_xyz(gt_xyz, name="gt_xyz")
    semantic = _as_int_vector(
        gt_semantic, name="gt_semantic", length=len(gt_points)
    )
    instances = _as_int_vector(
        gt_instance, name="gt_instance", length=len(gt_points)
    )
    rows = audit.get("objects")
    categories_by_key = audit.get("point_categories")
    if not isinstance(rows, Sequence) or not isinstance(categories_by_key, Mapping):
        raise TypeError("viewer audit is missing object/category rows")
    selections = select_clean_viewer_cases(rows)
    root = Path(output_dir) / str(audit["scene_id"])
    exported: list[dict[str, Any]] = []

    for case in selections:
        condition = str(case["condition"])
        instance_id = int(case["instance_id"])
        gaussian_ids = np.asarray(case["gaussian_ids"], dtype=np.int64)
        if np.any(gaussian_ids < 0) or np.any(gaussian_ids >= len(gaussian_points)):
            raise ValueError("viewer Gaussian ID is out of range")
        key = f"{condition}:{instance_id}"
        if key not in categories_by_key:
            raise ValueError(f"missing point categories for {key}")
        categories = np.asarray(categories_by_key[key], dtype=np.int8)
        if len(categories) != len(gaussian_ids):
            raise ValueError(f"point category length mismatch for {key}")
        predicted_colors = _colors_from_categories(categories)

        target = case.get("dominant_gt_instance")
        target_class_index = case.get("dominant_gt_class_index")
        if target is None:
            gt_mask = np.zeros(len(gt_points), dtype=bool)
        else:
            gt_mask = instances == int(target)
            if target_class_index is not None:
                gt_mask &= semantic == int(target_class_index)
        selected_gt = gt_points[gt_mask]
        gt_colors = np.tile(GT_COLOR, (len(selected_gt), 1))
        case_root = (
            root
            / _condition_slug(condition)
            / f"{case['role']}-instance-{instance_id}"
        )
        predicted_xyz = gaussian_points[gaussian_ids]
        _write_ascii_ply(
            case_root / "predicted_gaussians.ply",
            predicted_xyz,
            predicted_colors,
        )
        _write_ascii_ply(
            case_root / "matched_gt_points.ply",
            selected_gt,
            gt_colors,
        )
        _write_ascii_ply(
            case_root / "overlay.ply",
            np.concatenate((predicted_xyz, selected_gt), axis=0),
            np.concatenate((predicted_colors, gt_colors), axis=0),
        )
        metrics = {
            key: value
            for key, value in case.items()
            if key != "gaussian_ids"
        }
        metrics.update(
            {
                "schema": VIEWER_SCHEMA,
                "qualitative_only": True,
                "contains_2d_render_metrics": False,
                "color_legend": {
                    "correct_gt_instance": CORRECT_COLOR.tolist(),
                    "same_class_wrong_instance": SAME_CLASS_WRONG_INSTANCE_COLOR.tolist(),
                    "wrong_class": WRONG_CLASS_COLOR.tolist(),
                    "no_valid_gt_support": UNSUPPORTED_COLOR.tolist(),
                    "matched_gt_points": GT_COLOR.tolist(),
                },
            }
        )
        _write_json(case_root / "metrics.json", metrics)
        exported.append(
            {
                "role": str(case["role"]),
                "condition": condition,
                "instance_id": instance_id,
                "directory": str(case_root),
            }
        )

    comparison_payload = {
        "schema": VIEWER_SCHEMA,
        "scene_id": str(audit["scene_id"]),
        "conditions": list(audit.get("conditions", [])),
        "rows": list(audit.get("condition_comparison_rows", [])),
        "comparison_space": "3D Gaussian to GT",
        "contains_2d_render_metrics": False,
    }
    _write_json(root / "condition_comparison.json", comparison_payload)
    selection_payload = {
        "schema": VIEWER_SCHEMA,
        "scene_id": str(audit["scene_id"]),
        "selection_rule": (
            "fixed high/median/low precision, pure FP, tiny/small, merge/split roles"
        ),
        "cases": exported,
        "qualitative_only": True,
        "contains_2d_render_metrics": False,
    }
    _write_json(root / "viewer_case_selection.json", selection_payload)
    return selection_payload


def build_clean_baseline_viewer(
    *,
    scene_id: str,
    gaussian_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    gt_semantic: np.ndarray,
    gt_instance: np.ndarray,
    gaussian_to_gt_point: np.ndarray,
    predictions: Mapping[str, Mapping[str, Any]],
    output_dir: str | Path,
    class_names: Sequence[str] | None = None,
    tiny_small_instance_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Audit and export one scene without any 2D rendering diagnostic."""

    audit = audit_clean_viewer_scene(
        scene_id=scene_id,
        gaussian_xyz=gaussian_xyz,
        gt_xyz=gt_xyz,
        gt_semantic=gt_semantic,
        gt_instance=gt_instance,
        gaussian_to_gt_point=gaussian_to_gt_point,
        predictions=predictions,
        class_names=class_names,
        tiny_small_instance_ids=tiny_small_instance_ids,
    )
    return export_clean_viewer_scene(
        audit=audit,
        gaussian_xyz=gaussian_xyz,
        gt_xyz=gt_xyz,
        gt_semantic=gt_semantic,
        gt_instance=gt_instance,
        output_dir=output_dir,
    )


__all__ = [
    "CORRECT_COLOR",
    "FORMAL_CONDITION_ORDER",
    "GT_COLOR",
    "SAME_CLASS_WRONG_INSTANCE_COLOR",
    "UNSUPPORTED_COLOR",
    "VIEWER_SCHEMA",
    "WRONG_CLASS_COLOR",
    "audit_clean_viewer_scene",
    "build_clean_baseline_viewer",
    "export_clean_viewer_scene",
    "select_clean_viewer_cases",
]

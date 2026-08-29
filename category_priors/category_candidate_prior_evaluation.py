from __future__ import annotations

"""Frozen-bank candidate-prior evaluation for the section-30 DEV2/DEV8 gate.

This module is deliberately offline and GT-aware.  It never changes candidate
membership, candidate classes, Q, or the shared legacy post-processing path.
DEV8 candidate AP is threshold-free and scene-equal.  Only after that gate
passes is the hard score threshold selected from the fixed grid on uniform
DEV2 candidate F1, with exact ties resolved toward the higher threshold;
class-derived scores must reuse that threshold.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

DEV2_SCENE_IDS = ("scene0645_00", "scene0025_01")
DEV8_SCENE_IDS = (
    "scene0645_00",
    "scene0025_01",
    "scene0046_00",
    "scene0474_01",
    "scene0591_02",
    "scene0329_02",
    "scene0164_03",
    "scene0064_01",
)
THRESHOLD_GRID = (0.05, 0.10, 0.15, 0.20, 0.25)
IDENTITY_ATOL = 1e-6
SCORE_CHANGE_MIN = 0.01
SCORE_CHANGE_FRACTION_MIN = 0.10
SUPPORT_CHANGE_COUNT_MIN = 5
SUPPORT_CHANGE_CLASS_MIN = 2
SUPPORT_CHANGE_SCENE_MIN = 2
AP25_IMPROVEMENT_MIN = 0.002
AP50_DECLINE_MAX = 0.002
POSITIVE_SCENE_MIN = 5
FP_TP_WORSENING_MAX = 0.20


@dataclass(frozen=True)
class CandidatePriorExample:
    """One immutable candidate with paired U/D scores and offline GT label."""

    scene_id: str
    candidate_id: int
    branch_class: str
    core_point_count: int
    uniform_score: float
    class_score: float
    uniform_support_pass: bool
    class_support_pass: bool
    same_class_iou: float
    matched_gt_class: str | None
    matched_gt_instance_id: int | None
    matched_gt_size_bin: str | None
    q_value: float | None

    @property
    def key(self) -> tuple[str, int]:
        return (self.scene_id, self.candidate_id)

    @property
    def matched_gt_key(self) -> tuple[str, str, int] | None:
        if self.matched_gt_class is None or self.matched_gt_instance_id is None:
            return None
        return (
            self.scene_id,
            self.matched_gt_class,
            self.matched_gt_instance_id,
        )


@dataclass(frozen=True)
class OfficialCandidateGroundTruth:
    """One official-valid GT object used only for recall denominators."""

    scene_id: str
    class_name: str
    instance_id: int
    size_bin: str | None

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.scene_id, self.class_name, self.instance_id)


@dataclass(frozen=True)
class CandidateThresholdSelection:
    """The single U-only threshold frozen on the two registered DEV2 scenes."""

    selected_threshold: float
    scene_ids: tuple[str, ...]
    target_iou: float
    score_source: str
    tie_rule: str
    grid_rows: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_threshold": self.selected_threshold,
            "scene_ids": list(self.scene_ids),
            "target_iou": self.target_iou,
            "score_source": self.score_source,
            "tie_rule": self.tie_rule,
            "grid_rows": [dict(row) for row in self.grid_rows],
        }


@dataclass(frozen=True)
class CandidatePriorDev8Evaluation:
    """JSON-ready DEV8 result and its per-scene experimental units."""

    per_scene: tuple[dict[str, Any], ...]
    analysis: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {**self.analysis, "per_scene": [dict(row) for row in self.per_scene]}


def _validate_examples(rows: Sequence[CandidatePriorExample]) -> None:
    if not rows:
        raise ValueError("candidate examples must not be empty")
    keys: set[tuple[str, int]] = set()
    valid_sizes = {None, "tiny", "small", "medium", "large"}
    for row in rows:
        if not row.scene_id or row.candidate_id < 0 or not row.branch_class:
            raise ValueError("candidate example has invalid identity")
        if row.key in keys:
            raise ValueError(f"candidate examples repeat key {row.key}")
        keys.add(row.key)
        if row.core_point_count < 0:
            raise ValueError("candidate core point count must be non-negative")
        for name, score in (
            ("uniform", row.uniform_score),
            ("class", row.class_score),
        ):
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"{name} candidate score must be in [0, 1]")
        if not isinstance(row.uniform_support_pass, (bool, np.bool_)) or not isinstance(
            row.class_support_pass, (bool, np.bool_)
        ):
            raise TypeError("candidate support-pass fields must be boolean")
        if (
            not math.isfinite(row.same_class_iou)
            or not 0.0 <= row.same_class_iou <= 1.0
        ):
            raise ValueError("candidate same-class IoU must be in [0, 1]")
        if (row.matched_gt_class is None) != (
            row.matched_gt_instance_id is None
        ):
            raise ValueError("candidate GT target identity is incomplete")
        if row.same_class_iou > 0.0 and row.matched_gt_key is None:
            raise ValueError("positive-overlap candidate lacks a GT target")
        if (
            row.matched_gt_class is not None
            and row.matched_gt_class != row.branch_class
        ):
            raise ValueError("same-class candidate targets another class")
        if row.matched_gt_instance_id is not None and row.matched_gt_instance_id < 0:
            raise ValueError("candidate target instance ID must be non-negative")
        if row.matched_gt_size_bin not in valid_sizes:
            raise ValueError("candidate target size bin is invalid")
        if row.matched_gt_key is None and row.matched_gt_size_bin is not None:
            raise ValueError("candidate without a GT target has a target size bin")
        if row.q_value is not None and (
            not math.isfinite(row.q_value) or not 0.0 <= row.q_value <= 1.0
        ):
            raise ValueError("candidate Q must be in [0, 1]")


def _validate_ground_truth(rows: Sequence[OfficialCandidateGroundTruth]) -> None:
    if not rows:
        raise ValueError("official GT rows must not be empty")
    keys: set[tuple[str, str, int]] = set()
    valid_sizes = {None, "tiny", "small", "medium", "large"}
    for row in rows:
        if not row.scene_id or not row.class_name or row.instance_id < 0:
            raise ValueError("official GT object has invalid identity")
        if row.key in keys:
            raise ValueError(f"official GT repeats object {row.key}")
        keys.add(row.key)
        if row.size_bin not in valid_sizes:
            raise ValueError("official GT size bin is invalid")


def _key(row: Mapping[str, Any], table: str) -> tuple[str, int]:
    if "scene_id" not in row or "candidate_id" not in row:
        raise ValueError(f"{table} row is missing scene_id/candidate_id")
    scene_id = str(row["scene_id"])
    candidate_id = int(row["candidate_id"])
    if not scene_id or candidate_id < 0:
        raise ValueError(f"{table} row has an invalid candidate key")
    return scene_id, candidate_id


def _index_rows(
    rows: Sequence[Mapping[str, Any]], table: str
) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        key = _key(row, table)
        if key in result:
            raise ValueError(f"{table} repeats candidate key {key}")
        result[key] = row
    if not result:
        raise ValueError(f"{table} must not be empty")
    return result


def _first_present(
    row: Mapping[str, Any],
    names: Sequence[str],
    *,
    table: str,
    required: bool = True,
) -> Any:
    found = [name for name in names if name in row]
    if not found:
        if required:
            raise ValueError(f"{table} row is missing one of {tuple(names)}")
        return None
    value = row[found[0]]
    for name in found[1:]:
        other = row[name]
        if isinstance(value, (int, float, np.number)) and isinstance(
            other, (int, float, np.number)
        ):
            if abs(float(value) - float(other)) > IDENTITY_ATOL:
                raise ValueError(f"{table} row has conflicting aliases {found}")
        elif value != other:
            raise ValueError(f"{table} row has conflicting aliases {found}")
    return value


def _finite_score(row: Mapping[str, Any], mode: str) -> float:
    prefix = "U" if mode == "uniform" else "D"
    value = _first_present(
        row,
        ("score", f"{prefix}_score"),
        table=f"{mode} score",
    )
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{mode} candidate score must be finite and in [0, 1]")
    return score


def _core_count(candidate: Mapping[str, Any]) -> int:
    value = _first_present(
        candidate,
        ("core_point_count", "core_actual_count", "core_point_count_recorded"),
        table="candidate",
    )
    result = int(value)
    if result < 0:
        raise ValueError("candidate core point count must be non-negative")
    return result


def _support_pass(
    score_row: Mapping[str, Any],
    candidate: Mapping[str, Any],
    mode: str,
) -> bool:
    prefix = "U" if mode == "uniform" else "D"
    direct = _first_present(
        score_row,
        ("support_pass", f"{prefix}_support_pass"),
        table=f"{mode} score",
        required=False,
    )
    threshold = _first_present(
        score_row,
        ("support_threshold", f"{prefix}_support_threshold"),
        table=f"{mode} score",
        required=direct is None,
    )
    score_core = _first_present(
        score_row,
        ("core_point_count", "core_actual_count"),
        table=f"{mode} score",
        required=False,
    )
    candidate_core = _core_count(candidate)
    if score_core is not None and int(score_core) != candidate_core:
        raise ValueError(f"{mode} score row changes the frozen core point count")
    derived: bool | None = None
    if threshold is not None:
        support_threshold = int(threshold)
        if support_threshold < 0:
            raise ValueError(f"{mode} support threshold must be non-negative")
        derived = candidate_core >= support_threshold
    if direct is not None:
        if not isinstance(direct, (bool, np.bool_)):
            raise TypeError(f"{mode} support_pass must be boolean")
        direct_bool = bool(direct)
        if derived is not None and direct_bool != derived:
            raise ValueError(f"{mode} support_pass conflicts with support threshold")
        return direct_bool
    if derived is None:  # Defensive: required=True above should make this unreachable.
        raise ValueError(f"{mode} score row lacks a support rule")
    return derived


def _optional_q(row: Mapping[str, Any]) -> float | None:
    value = _first_present(
        row,
        ("Q", "base_score"),
        table="candidate identity",
        required=False,
    )
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("candidate Q must be finite and in [0, 1]")
    return result


def _validate_score_identity(
    candidate: Mapping[str, Any],
    score_row: Mapping[str, Any],
    *,
    mode: str,
) -> None:
    branch_class = str(candidate["branch_class"])
    if "branch_class" in score_row and str(score_row["branch_class"]) != branch_class:
        raise ValueError(f"{mode} score row changes candidate branch_class")
    exact_aliases = (
        ("branch_class_index", ("branch_class_index",)),
        (
            "full point count",
            ("full_point_count", "full_actual_count"),
        ),
    )
    for label, aliases in exact_aliases:
        candidate_value = _first_present(
            candidate,
            aliases,
            table="candidate",
            required=False,
        )
        score_value = _first_present(
            score_row,
            aliases,
            table=f"{mode} score",
            required=False,
        )
        if (
            candidate_value is not None
            and score_value is not None
            and candidate_value != score_value
        ):
            raise ValueError(f"{mode} score row changes candidate {label}")
    score_core = _first_present(
        score_row,
        ("core_point_count", "core_actual_count", "core_point_count_recorded"),
        table=f"{mode} score",
        required=False,
    )
    if score_core is not None and int(score_core) != _core_count(candidate):
        raise ValueError(f"{mode} score row changes candidate core point count")
    _support_pass(score_row, candidate, mode)


def join_candidate_prior_rows(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    uniform_score_rows: Sequence[Mapping[str, Any]],
    class_score_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
) -> tuple[CandidatePriorExample, ...]:
    """Join four exact-key tables and reject any U/D bank identity drift."""

    candidates = _index_rows(candidate_rows, "candidate")
    uniform = _index_rows(uniform_score_rows, "uniform score")
    per_class = _index_rows(class_score_rows, "class score")
    labels = _index_rows(label_rows, "candidate label")
    key_sets = {
        "candidate": set(candidates),
        "uniform": set(uniform),
        "class": set(per_class),
        "label": set(labels),
    }
    if len({frozenset(value) for value in key_sets.values()}) != 1:
        raise ValueError(
            "candidate/U/D/label tables do not contain the same frozen bank"
        )

    result: list[CandidatePriorExample] = []
    for key in sorted(candidates):
        candidate = candidates[key]
        uniform_row = uniform[key]
        class_row = per_class[key]
        label = labels[key]
        if "branch_class" not in candidate:
            raise ValueError("candidate row is missing branch_class")
        branch_class = str(candidate["branch_class"])
        _validate_score_identity(candidate, uniform_row, mode="uniform")
        _validate_score_identity(candidate, class_row, mode="class")
        q_values = [
            value
            for value in (
                _optional_q(candidate),
                _optional_q(uniform_row),
                _optional_q(class_row),
            )
            if value is not None
        ]
        if q_values and max(q_values) - min(q_values) > IDENTITY_ATOL:
            raise ValueError("U/D score rows change frozen candidate Q")
        q_value = q_values[0] if q_values else None

        iou = float(
            _first_present(
                label,
                ("same_class_iou", "full_best_same_class_iou"),
                table="candidate label",
            )
        )
        if not math.isfinite(iou) or not 0.0 <= iou <= 1.0:
            raise ValueError("candidate same-class IoU must be in [0, 1]")
        target_class_value = _first_present(
            label,
            ("matched_gt_class", "full_best_same_class_gt_class"),
            table="candidate label",
            required=False,
        )
        target_instance_value = _first_present(
            label,
            (
                "matched_gt_instance_id",
                "full_best_same_class_gt_instance",
            ),
            table="candidate label",
            required=False,
        )
        target_class = (
            str(target_class_value) if target_class_value is not None else None
        )
        target_instance = (
            int(target_instance_value)
            if target_instance_value is not None
            else None
        )
        if (target_class is None) != (target_instance is None):
            raise ValueError("candidate GT target identity must be complete or absent")
        if iou > 0.0 and target_class is None:
            raise ValueError("positive-overlap candidate is missing its GT target")
        if target_class is not None and target_class != branch_class:
            raise ValueError("same-class annotation targets another class")
        size_value = _first_present(
            label,
            ("matched_gt_size_bin", "size_bin"),
            table="candidate label",
            required=False,
        )
        result.append(
            CandidatePriorExample(
                scene_id=key[0],
                candidate_id=key[1],
                branch_class=branch_class,
                core_point_count=_core_count(candidate),
                uniform_score=_finite_score(uniform_row, "uniform"),
                class_score=_finite_score(class_row, "class"),
                uniform_support_pass=_support_pass(
                    uniform_row, candidate, "uniform"
                ),
                class_support_pass=_support_pass(class_row, candidate, "class"),
                same_class_iou=iou,
                matched_gt_class=target_class,
                matched_gt_instance_id=target_instance,
                matched_gt_size_bin=(
                    str(size_value) if size_value is not None else None
                ),
                q_value=q_value,
            )
        )
    normalized = tuple(result)
    _validate_examples(normalized)
    return normalized


def normalize_official_gt_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[OfficialCandidateGroundTruth, ...]:
    """Normalize the explicit official-GT universe used for recall."""

    result: list[OfficialCandidateGroundTruth] = []
    keys: set[tuple[str, str, int]] = set()
    for row in rows:
        if "scene_id" not in row:
            raise ValueError("official GT row is missing scene_id")
        scene_id = str(row["scene_id"])
        class_name = str(
            _first_present(
                row,
                ("gt_class", "class_name", "class"),
                table="official GT",
            )
        )
        instance_id = int(
            _first_present(
                row,
                ("gt_instance_id", "instance_id"),
                table="official GT",
            )
        )
        size = _first_present(
            row,
            ("size_bin",),
            table="official GT",
            required=False,
        )
        if size is not None and str(size) not in {
            "tiny",
            "small",
            "medium",
            "large",
        }:
            raise ValueError("official GT size_bin is invalid")
        item = OfficialCandidateGroundTruth(
            scene_id=scene_id,
            class_name=class_name,
            instance_id=instance_id,
            size_bin=str(size) if size is not None else None,
        )
        if not scene_id or not class_name or instance_id < 0:
            raise ValueError("official GT row has invalid identity")
        if item.key in keys:
            raise ValueError(f"official GT repeats object {item.key}")
        keys.add(item.key)
        result.append(item)
    normalized = tuple(sorted(result, key=lambda item: item.key))
    _validate_ground_truth(normalized)
    return normalized


def binary_average_precision(
    scores: Sequence[float],
    positive: Sequence[bool],
    *,
    eligible: Sequence[bool] | None = None,
) -> float:
    """Tie-grouped AP with a hard support gate and full positive denominator."""

    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(positive, dtype=bool)
    if values.ndim != 1 or labels.shape != values.shape:
        raise ValueError("scores and positive labels must be aligned vectors")
    if not np.isfinite(values).all():
        raise ValueError("candidate AP scores must be finite")
    if eligible is None:
        admitted = np.ones(len(values), dtype=bool)
    else:
        admitted = np.asarray(eligible, dtype=bool)
        if admitted.shape != values.shape:
            raise ValueError("eligible must align with candidate scores")
    positive_count = int(np.count_nonzero(labels))
    if not len(values) or not positive_count or not np.any(admitted):
        return 0.0
    values = values[admitted]
    labels = labels[admitted]
    order = np.argsort(-values, kind="mergesort")
    sorted_scores = values[order]
    sorted_labels = labels[order]
    true_positive = 0
    false_positive = 0
    average_precision = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group_positive = int(np.count_nonzero(sorted_labels[start:end]))
        group_count = end - start
        true_positive += group_positive
        false_positive += group_count - group_positive
        precision = true_positive / (true_positive + false_positive)
        average_precision += group_positive / positive_count * precision
        start = end
    return float(average_precision)


def _mode_score(item: CandidatePriorExample, mode: str) -> float:
    if mode == "uniform":
        return item.uniform_score
    if mode == "class":
        return item.class_score
    raise ValueError("mode must be 'uniform' or 'class'")


def _mode_support_pass(item: CandidatePriorExample, mode: str) -> bool:
    if mode == "uniform":
        return bool(item.uniform_support_pass)
    if mode == "class":
        return bool(item.class_support_pass)
    raise ValueError("mode must be 'uniform' or 'class'")


def _classification_counts(
    rows: Sequence[CandidatePriorExample],
    *,
    mode: str,
    score_threshold: float | None,
    iou_threshold: float,
) -> dict[str, Any]:
    accepted = np.asarray(
        [
            _mode_support_pass(row, mode)
            and (
                score_threshold is None
                or _mode_score(row, mode) >= score_threshold
            )
            for row in rows
        ],
        dtype=bool,
    )
    positive = np.asarray(
        [row.same_class_iou >= iou_threshold for row in rows], dtype=bool
    )
    true_positive = int(np.count_nonzero(accepted & positive))
    false_positive = int(np.count_nonzero(accepted & ~positive))
    false_negative = int(np.count_nonzero(~accepted & positive))
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / denominator if denominator else 0.0
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    return {
        "accepted_candidate_count": int(np.count_nonzero(accepted)),
        "true_positive_count": true_positive,
        "false_positive_count": false_positive,
        "false_negative_count": false_negative,
        "precision": (
            true_positive / precision_denominator if precision_denominator else 0.0
        ),
        "recall": true_positive / recall_denominator if recall_denominator else 0.0,
        "f1": float(f1),
    }


def select_uniform_threshold_dev2(
    examples: Sequence[CandidatePriorExample],
) -> CandidateThresholdSelection:
    """Freeze the only hard threshold using U on the registered DEV2 only."""

    rows = tuple(examples)
    _validate_examples(rows)
    scenes = tuple(sorted({row.scene_id for row in rows}))
    if set(scenes) != set(DEV2_SCENE_IDS):
        raise ValueError(
            "threshold selection requires exactly the registered DEV2 scenes"
        )
    if any(not any(row.scene_id == scene for row in rows) for scene in DEV2_SCENE_IDS):
        raise ValueError("each DEV2 scene must contain at least one candidate")
    grid_rows: list[dict[str, Any]] = []
    for threshold in THRESHOLD_GRID:
        scene_metrics = []
        for scene_id in DEV2_SCENE_IDS:
            selected = [row for row in rows if row.scene_id == scene_id]
            scene_metrics.append(
                _classification_counts(
                    selected,
                    mode="uniform",
                    score_threshold=threshold,
                    iou_threshold=0.25,
                )
            )
        pooled = _classification_counts(
            rows,
            mode="uniform",
            score_threshold=threshold,
            iou_threshold=0.25,
        )
        grid_rows.append(
            {
                "threshold": threshold,
                "scene_equal_candidate_f1_025": float(
                    np.mean([metric["f1"] for metric in scene_metrics])
                ),
                "pooled_candidate_f1_025": pooled["f1"],
                "pooled_true_positive_count": pooled["true_positive_count"],
                "pooled_false_positive_count": pooled["false_positive_count"],
                "pooled_false_negative_count": pooled["false_negative_count"],
            }
        )
    best_value = max(
        float(row["scene_equal_candidate_f1_025"]) for row in grid_rows
    )
    selected_threshold = max(
        float(row["threshold"])
        for row in grid_rows
        if float(row["scene_equal_candidate_f1_025"]) == best_value
    )
    return CandidateThresholdSelection(
        selected_threshold=selected_threshold,
        scene_ids=DEV2_SCENE_IDS,
        target_iou=0.25,
        score_source="uniform",
        tie_rule="exact_tie_choose_higher_threshold",
        grid_rows=tuple(grid_rows),
    )


def candidate_prior_mechanical_effect(
    examples: Sequence[CandidatePriorExample],
) -> dict[str, Any]:
    """Apply the pre-registered score-change OR support-change mechanism gate."""

    rows = tuple(examples)
    _validate_examples(rows)
    score_changed = [
        row
        for row in rows
        if abs(row.class_score - row.uniform_score) >= SCORE_CHANGE_MIN
    ]
    support_changed = [
        row
        for row in rows
        if row.class_support_pass != row.uniform_support_pass
    ]
    score_fraction = len(score_changed) / len(rows)
    support_classes = sorted({row.branch_class for row in support_changed})
    support_scenes = sorted({row.scene_id for row in support_changed})
    score_gate = score_fraction >= SCORE_CHANGE_FRACTION_MIN
    support_gate = (
        len(support_changed) >= SUPPORT_CHANGE_COUNT_MIN
        and len(support_classes) >= SUPPORT_CHANGE_CLASS_MIN
        and len(support_scenes) >= SUPPORT_CHANGE_SCENE_MIN
    )
    return {
        "candidate_count": len(rows),
        "score_change_min": SCORE_CHANGE_MIN,
        "score_changed_candidate_count": len(score_changed),
        "score_changed_candidate_fraction": score_fraction,
        "score_change_gate_passed": score_gate,
        "support_changed_candidate_count": len(support_changed),
        "support_changed_classes": support_classes,
        "support_changed_class_count": len(support_classes),
        "support_changed_scenes": support_scenes,
        "support_changed_scene_count": len(support_scenes),
        "support_change_gate_passed": support_gate,
        "mechanically_effective": score_gate or support_gate,
        "gate_logic": "score_change_fraction OR support_change_count_class_scene",
    }


def _tiny_small_recall(
    rows: Sequence[CandidatePriorExample],
    ground_truth: Sequence[OfficialCandidateGroundTruth],
    *,
    mode: str,
    score_threshold: float | None,
    iou_threshold: float,
    scene_ids: Sequence[str],
) -> dict[str, Any]:
    tiny_keys = {
        item.key for item in ground_truth if item.size_bin in {"tiny", "small"}
    }
    recovered = {
        row.matched_gt_key
        for row in rows
        if _mode_support_pass(row, mode)
        and (
            score_threshold is None
            or _mode_score(row, mode) >= score_threshold
        )
        and row.same_class_iou >= iou_threshold
        and row.matched_gt_key in tiny_keys
    }
    per_scene: list[dict[str, Any]] = []
    scene_recalls: list[float] = []
    for scene_id in scene_ids:
        targets = {key for key in tiny_keys if key[0] == scene_id}
        matched = targets.intersection(recovered)
        recall = len(matched) / len(targets) if targets else None
        if recall is not None:
            scene_recalls.append(recall)
        per_scene.append(
            {
                "scene_id": scene_id,
                "tiny_small_gt_count": len(targets),
                "recovered_gt_count": len(matched),
                "recall": recall,
            }
        )
    return {
        "iou_threshold": iou_threshold,
        "tiny_small_gt_count": len(tiny_keys),
        "recovered_gt_count": len(recovered),
        "pooled_recall": len(recovered) / len(tiny_keys) if tiny_keys else None,
        "scene_equal_recall": float(np.mean(scene_recalls))
        if scene_recalls
        else None,
        "evaluable_scene_count": len(scene_recalls),
        "per_scene": per_scene,
    }


def _fp_tp_gate(
    uniform_counts: Mapping[str, Any],
    class_counts: Mapping[str, Any],
) -> dict[str, Any]:
    uniform_tp = int(uniform_counts["true_positive_count"])
    uniform_fp = int(uniform_counts["false_positive_count"])
    class_tp = int(class_counts["true_positive_count"])
    class_fp = int(class_counts["false_positive_count"])
    # Match the repository's registered output gate: floor the TP denominator
    # at one so the all-rejected 0/0 baseline remains a finite safety check.
    uniform_ratio = uniform_fp / max(uniform_tp, 1)
    class_ratio = class_fp / max(class_tp, 1)
    if uniform_ratio == 0.0:
        passed = class_ratio == 0.0
        relative_worsening = 0.0 if passed else None
        reason = None if passed else "uniform_zero_fp_class_nonzero_fp"
    else:
        relative_worsening = (class_ratio - uniform_ratio) / uniform_ratio
        passed = relative_worsening <= FP_TP_WORSENING_MAX
        reason = None if passed else "relative_worsening_above_limit"
    return {
        "iou_threshold": 0.25,
        "aggregation": "pooled_candidate_counts",
        "uniform_fp_tp_ratio": uniform_ratio,
        "class_fp_tp_ratio": class_ratio,
        "true_positive_denominator_floor": 1,
        "relative_worsening": relative_worsening,
        "maximum_relative_worsening": FP_TP_WORSENING_MAX,
        "passed": passed,
        "reason": reason,
    }


def _validate_dev8_inputs(
    rows: Sequence[CandidatePriorExample],
    ground_truth: Sequence[OfficialCandidateGroundTruth],
) -> None:
    candidate_scenes = {row.scene_id for row in rows}
    gt_scenes = {row.scene_id for row in ground_truth}
    expected = set(DEV8_SCENE_IDS)
    if candidate_scenes != expected:
        raise ValueError("candidate evaluation requires exactly the registered DEV8")
    if gt_scenes != expected:
        raise ValueError("official GT universe requires exactly the registered DEV8")
    gt_by_key = {item.key: item for item in ground_truth}
    for row in rows:
        target = row.matched_gt_key
        if target is None:
            continue
        if target not in gt_by_key:
            raise ValueError(f"candidate target is absent from official GT: {target}")
        expected_size = gt_by_key[target].size_bin
        if (
            row.matched_gt_size_bin is not None
            and row.matched_gt_size_bin != expected_size
        ):
            raise ValueError("candidate target size bin disagrees with official GT")


def evaluate_candidate_prior_dev8(
    *,
    examples: Sequence[CandidatePriorExample],
    official_gt: Sequence[OfficialCandidateGroundTruth],
) -> CandidatePriorDev8Evaluation:
    """Evaluate paired U/D scores on DEV8 before any acceptance threshold."""

    rows = tuple(examples)
    ground_truth = tuple(official_gt)
    _validate_examples(rows)
    _validate_ground_truth(ground_truth)
    _validate_dev8_inputs(rows, ground_truth)
    per_scene: list[dict[str, Any]] = []
    for scene_id in DEV8_SCENE_IDS:
        scene_rows = [row for row in rows if row.scene_id == scene_id]
        if not scene_rows:
            raise ValueError(f"DEV8 scene has no candidates: {scene_id}")
        positives_025 = [row.same_class_iou >= 0.25 for row in scene_rows]
        positives_050 = [row.same_class_iou >= 0.50 for row in scene_rows]
        uniform_scores = [row.uniform_score for row in scene_rows]
        class_scores = [row.class_score for row in scene_rows]
        uniform_eligible = [row.uniform_support_pass for row in scene_rows]
        class_eligible = [row.class_support_pass for row in scene_rows]
        uniform_ap25 = binary_average_precision(
            uniform_scores, positives_025, eligible=uniform_eligible
        )
        class_ap25 = binary_average_precision(
            class_scores, positives_025, eligible=class_eligible
        )
        uniform_ap50 = binary_average_precision(
            uniform_scores, positives_050, eligible=uniform_eligible
        )
        class_ap50 = binary_average_precision(
            class_scores, positives_050, eligible=class_eligible
        )
        per_scene.append(
            {
                "scene_id": scene_id,
                "candidate_count": len(scene_rows),
                "positive_candidate_count_025": int(sum(positives_025)),
                "positive_candidate_count_050": int(sum(positives_050)),
                "uniform_candidate_ap_025": uniform_ap25,
                "class_candidate_ap_025": class_ap25,
                "delta_candidate_ap_025": class_ap25 - uniform_ap25,
                "uniform_candidate_ap_050": uniform_ap50,
                "class_candidate_ap_050": class_ap50,
                "delta_candidate_ap_050": class_ap50 - uniform_ap50,
                "positive_direction_ap_025": class_ap25 > uniform_ap25,
                "uniform_support_only_025": _classification_counts(
                    scene_rows,
                    mode="uniform",
                    score_threshold=None,
                    iou_threshold=0.25,
                ),
                "class_support_only_025": _classification_counts(
                    scene_rows,
                    mode="class",
                    score_threshold=None,
                    iou_threshold=0.25,
                ),
            }
        )

    uniform_ap25 = float(
        np.mean([row["uniform_candidate_ap_025"] for row in per_scene])
    )
    class_ap25 = float(
        np.mean([row["class_candidate_ap_025"] for row in per_scene])
    )
    uniform_ap50 = float(
        np.mean([row["uniform_candidate_ap_050"] for row in per_scene])
    )
    class_ap50 = float(
        np.mean([row["class_candidate_ap_050"] for row in per_scene])
    )
    delta_ap25 = class_ap25 - uniform_ap25
    delta_ap50 = class_ap50 - uniform_ap50
    positive_scene_ids = [
        str(row["scene_id"])
        for row in per_scene
        if bool(row["positive_direction_ap_025"])
    ]
    mechanical = candidate_prior_mechanical_effect(rows)
    uniform_acceptance = _classification_counts(
        rows,
        mode="uniform",
        score_threshold=None,
        iou_threshold=0.25,
    )
    class_acceptance = _classification_counts(
        rows,
        mode="class",
        score_threshold=None,
        iou_threshold=0.25,
    )
    tiny_small = {
        "uniform_025": _tiny_small_recall(
            rows,
            ground_truth,
            mode="uniform",
            score_threshold=None,
            iou_threshold=0.25,
            scene_ids=DEV8_SCENE_IDS,
        ),
        "class_025": _tiny_small_recall(
            rows,
            ground_truth,
            mode="class",
            score_threshold=None,
            iou_threshold=0.25,
            scene_ids=DEV8_SCENE_IDS,
        ),
        "uniform_050": _tiny_small_recall(
            rows,
            ground_truth,
            mode="uniform",
            score_threshold=None,
            iou_threshold=0.50,
            scene_ids=DEV8_SCENE_IDS,
        ),
        "class_050": _tiny_small_recall(
            rows,
            ground_truth,
            mode="class",
            score_threshold=None,
            iou_threshold=0.50,
            scene_ids=DEV8_SCENE_IDS,
        ),
    }
    uniform_tiny_recall = tiny_small["uniform_025"]["scene_equal_recall"]
    class_tiny_recall = tiny_small["class_025"]["scene_equal_recall"]
    tiny_gate = (
        uniform_tiny_recall is not None
        and class_tiny_recall is not None
        and float(class_tiny_recall) >= float(uniform_tiny_recall)
    )
    fp_tp = _fp_tp_gate(uniform_acceptance, class_acceptance)
    gates = {
        "mechanically_effective": bool(mechanical["mechanically_effective"]),
        "scene_equal_ap_025_improvement": delta_ap25 >= AP25_IMPROVEMENT_MIN,
        "scene_equal_ap_050_non_degradation": delta_ap50 >= -AP50_DECLINE_MAX,
        "positive_scene_count": len(positive_scene_ids) >= POSITIVE_SCENE_MIN,
        "tiny_small_recall_025_non_degradation": tiny_gate,
        "fp_tp_non_degradation": bool(fp_tp["passed"]),
    }
    analysis = {
        "schema": "saga-category-candidate-prior-dev8-evaluation-v1",
        "scene_ids": list(DEV8_SCENE_IDS),
        "scene_count": len(DEV8_SCENE_IDS),
        "candidate_count": len(rows),
        "acceptance_threshold": None,
        "threshold_selection_allowed_only_after_this_gate": True,
        "candidate_label_counts": {
            "same_class_iou_025": int(
                sum(row.same_class_iou >= 0.25 for row in rows)
            ),
            "same_class_iou_050": int(
                sum(row.same_class_iou >= 0.50 for row in rows)
            ),
            "positive_scene_count_050": len(
                {
                    row.scene_id
                    for row in rows
                    if row.same_class_iou >= 0.50
                }
            ),
        },
        "candidate_ap": {
            "aggregation": "scene_equal",
            "uniform_025": uniform_ap25,
            "class_025": class_ap25,
            "delta_025": delta_ap25,
            "uniform_050": uniform_ap50,
            "class_050": class_ap50,
            "delta_050": delta_ap50,
        },
        "mechanical_effect": mechanical,
        "positive_scene_ids_ap_025": positive_scene_ids,
        "positive_scene_count_ap_025": len(positive_scene_ids),
        "support_only_acceptance": {
            "uniform_025": uniform_acceptance,
            "class_025": class_acceptance,
        },
        "tiny_small_recall": tiny_small,
        "fp_tp_gate": fp_tp,
        "gates": gates,
        "passed": all(gates.values()),
        "conclusion_boundary": {
            "threshold_selected_before_this_gate": False,
            "threshold_may_be_selected_from": "DEV2_uniform_only_after_pass",
            "DEV8_used_for_threshold_selection": False,
            "candidate_AP_is_threshold_free": True,
            "candidate_rows_are_independent_replicates": False,
            "independent_experimental_unit": "scene",
            "fp_tp_gate_is_auxiliary_not_significance_test": True,
        },
    }
    return CandidatePriorDev8Evaluation(
        per_scene=tuple(per_scene),
        analysis=analysis,
    )


__all__ = [
    "AP25_IMPROVEMENT_MIN",
    "AP50_DECLINE_MAX",
    "CandidatePriorDev8Evaluation",
    "CandidatePriorExample",
    "CandidateThresholdSelection",
    "DEV2_SCENE_IDS",
    "DEV8_SCENE_IDS",
    "FP_TP_WORSENING_MAX",
    "OfficialCandidateGroundTruth",
    "POSITIVE_SCENE_MIN",
    "THRESHOLD_GRID",
    "binary_average_precision",
    "candidate_prior_mechanical_effect",
    "evaluate_candidate_prior_dev8",
    "join_candidate_prior_rows",
    "normalize_official_gt_rows",
    "select_uniform_threshold_dev2",
]

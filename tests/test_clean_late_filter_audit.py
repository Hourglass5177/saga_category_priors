from __future__ import annotations

from dataclasses import asdict
import inspect
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

from category_priors.clean_baseline.consensus import ConsensusConfig
from category_priors.clean_baseline.evidence import build_sparse_frame_evidence
from category_priors.clean_baseline.late_filter_audit import (
    _export_with_reasons,
    _production_class_for_masks,
    replay_late_filter_factorial,
)
from category_priors.clean_baseline.late_filter_experiment import (
    _candidate_metrics,
    _nonempty_geometric_candidates,
)
from category_priors.clean_baseline.metric_reaudit import build_bidirectional_nearest
from category_priors.clean_baseline.evaluation import CleanCandidate, GroundTruthObject
from category_priors.clean_baseline.models import AlphaMaskEvidenceBank
from category_priors.clean_baseline.pipeline import _object_class_distribution
from category_priors.clean_baseline.stage_funnel import FunnelObject, _class_for_masks


CLASSES = (
    "chair",
    "table",
    "plant",
    "flower",
    "foliage",
    "tv",
    "painting",
    "sofa",
    "cabinet",
    "bed",
    "wall",
    "floor",
    "ceiling",
    "person",
    "socket",
    "remote",
    "key",
    "book",
    "lighting",
    "switch",
    "door",
    "window",
    "lamp",
    "speaker",
    "computer",
    "fan",
    "refrigerator",
    "robot",
    "cup",
    "vase",
    "phone",
    "trash can",
)


def _frame(
    frame_id: int,
    *,
    masks: Sequence[tuple[int, Sequence[int]]],
    visible_ids: Sequence[int],
) -> object:
    visible = np.asarray(sorted(set(map(int, visible_ids))), dtype=np.int32)
    rows = [np.asarray(sorted(set(map(int, ids))), dtype=np.int32) for _, ids in masks]
    posterior = np.zeros((len(rows), len(CLASSES)), dtype=np.float32)
    if len(rows):
        posterior[:, 0] = 1.0
    return build_sparse_frame_evidence(
        frame_id=frame_id,
        image_name=f"frame-{frame_id}",
        point_count=13,
        visible_ids=visible,
        visible_mass=np.ones(len(visible), dtype=np.float32),
        mask_gaussian_ids=rows,
        mask_inside_mass=[np.ones(len(row), dtype=np.float32) for row in rows],
        mask_inside_ratio=[np.ones(len(row), dtype=np.float32) for row in rows],
        semantic_posteriors=posterior,
        semantic_abstained=np.zeros(len(rows), dtype=np.bool_),
        global_mask_ids=[mask_id for mask_id, _ in masks],
        valid_pixel_count=max(1, len(visible)),
        class_count=len(CLASSES),
    )


def _bank() -> AlphaMaskEvidenceBank:
    """Small bank containing one survivor and every early drop reason.

    Component ``(0, 1)`` has a five-point full union.  Point four is ambiguous
    in its only visible frame, so its eligible-visible denominator is zero.
    Historical A1 removes it, whereas A0 must retain it with ratio zero.

    Component ``(2,)`` has only one member view, ``(6, 7)`` has an empty full
    union, and ``(3, 5)`` is detected in two of five eligible views (ratio
    0.4).  These components make the three early rejection mechanisms
    independently observable instead of collapsing them into one counter.
    """

    frames = (
        _frame(
            0,
            masks=(
                (0, (0, 1, 2, 3, 4)),
                (2, (5, 6, 7, 8)),
                (3, (9, 10, 11, 12)),
                (4, (4,)),  # rejected helper; makes point 4 ambiguous
                (6, ()),
            ),
            visible_ids=tuple(range(13)),
        ),
        _frame(
            1,
            masks=((1, (0, 1, 2, 3)), (5, (9, 10, 11, 12)), (7, ())),
            visible_ids=(0, 1, 2, 3, 9, 10, 11, 12),
        ),
        _frame(2, masks=(), visible_ids=(9, 10, 11, 12)),
        _frame(3, masks=(), visible_ids=(9, 10, 11, 12)),
        _frame(4, masks=(), visible_ids=(9, 10, 11, 12)),
    )
    xyz = np.zeros((13, 3), dtype=np.float32)
    xyz[:9, 0] = np.arange(9, dtype=np.float32) * 0.01
    xyz[9:, 0] = 1.0 + np.arange(4, dtype=np.float32) * 0.01
    return AlphaMaskEvidenceBank.from_frames(
        scene_id="scene-late-filter",
        point_count=13,
        xyz_m=xyz,
        class_names=CLASSES,
        frames=frames,
        source={"fixture": "late-filter-factorial"},
    )


def _edge(left: int, right: int) -> dict[str, object]:
    return {
        "left_mask_ids": [left],
        "right_mask_ids": [right],
        "observer_count": 2,
        "supporter_count": 2,
        "consensus": 1.0,
        "observer_level": 2,
    }


def _diagnostics() -> dict[str, object]:
    return {
        "scene_id": "scene-late-filter",
        "condition": "C0-no-prior",
        "config": asdict(ConsensusConfig()),
        "rejected_undersegmented_mask_ids": [4],
        "accepted_edges": [_edge(0, 1), _edge(3, 5), _edge(6, 7)],
        "consensus": {"component_count_before_output_filters": 4},
    }


def _run(*, frozen_output: Mapping[str, Any] | None = None) -> object:
    return replay_late_filter_factorial(
        bank=_bank(),
        diagnostics=_diagnostics(),
        allowed_classes=("chair",),
        frozen_output=frozen_output,
    )


def _find_component(rows: Sequence[object], mask_ids: tuple[int, ...]) -> object:
    return next(row for row in rows if tuple(row.mask_ids) == mask_ids)


def _prediction_from_formal_objects(
    rows: Sequence[object], *, point_count: int
) -> dict[str, object]:
    labels = np.full(point_count, -1, dtype=np.int64)
    instances: dict[str, dict[str, object]] = {}
    for instance_id, row in enumerate(rows):
        labels[row.gaussian_ids] = instance_id
        instances[str(instance_id)] = {
            "class": row.class_name,
            "score": float(row.metadata.get("score", 1.0)),
            "point_count": int(len(row.gaussian_ids)),
        }
    return {
        "scene_id": "scene-late-filter",
        "condition": "C0-no-prior",
        "point_labels": labels.tolist(),
        "instances": instances,
    }


def _snapshot(result: object) -> tuple[object, ...]:
    def objects(rows: Sequence[object] | None) -> tuple[object, ...] | None:
        if rows is None:
            return None
        return tuple(
            (
                row.stable_id,
                tuple(map(int, row.mask_ids)),
                tuple(map(int, row.frame_ids)),
                tuple(map(int, row.gaussian_ids.tolist())),
                row.class_name,
                tuple(sorted(dict(row.metadata).items())),
            )
            for row in rows
        )

    evidence = tuple(
        (
            row.stable_id,
            tuple(map(int, row.mask_ids)),
            tuple(map(int, row.frame_ids)),
            tuple(map(int, row.full_ids.tolist())),
            tuple(map(float, row.detection_ratios.tolist())),
            tuple(map(int, row.historical_ids.tolist())),
            row.dropped_reason,
        )
        for row in result.detection_evidence
    )
    arms = tuple(
        (
            code,
            arm.name,
            arm.detection_hard_filter,
            arm.strict_late_export,
            objects(arm.detection_objects),
            objects(arm.physical_objects),
            objects(arm.ownership_objects),
            objects(arm.formal_output),
            arm.formal_output_allowed,
            tuple(sorted(dict(arm.drop_reasons).items())),
        )
        for code, arm in result.arms.items()
    )
    return (
        result.scene_id,
        result.condition,
        result.point_count,
        objects(result.accepted_components),
        evidence,
        arms,
        tuple(sorted(dict(result.shared_identity).items())),
        tuple(result.issues),
    )


def test_historical_arm_reconstructs_frozen_output_exactly() -> None:
    initial = _run()
    historical = initial.arms["A1B1"]
    assert historical.formal_output is not None
    frozen = _prediction_from_formal_objects(
        historical.formal_output, point_count=initial.point_count
    )

    replayed = _run(frozen_output=frozen)
    equivalence = replayed.arms["A1B1"].final_equivalence

    assert equivalence is not None
    assert equivalence.equivalent is True
    assert equivalence.changed_points == 0
    assert equivalence.class_exact is True


def test_formal_export_reuses_production_semantic_reducer_and_score_bits() -> None:
    """Hierarchical same-view masks expose both former replay divergences."""

    def frame(
        frame_id: int,
        global_mask_ids: Sequence[int],
        posteriors: Sequence[tuple[float, float]],
    ) -> object:
        values = np.zeros((len(posteriors), len(CLASSES)), dtype=np.float32)
        values[:, :2] = np.asarray(posteriors, dtype=np.float32)
        ids = np.arange(4, dtype=np.int32)
        return build_sparse_frame_evidence(
            frame_id=frame_id,
            image_name=f"posterior-{frame_id}",
            point_count=4,
            visible_ids=ids,
            visible_mass=np.ones(4, dtype=np.float32),
            mask_gaussian_ids=[ids for _ in posteriors],
            mask_inside_mass=[np.ones(4, dtype=np.float32) for _ in posteriors],
            mask_inside_ratio=[np.ones(4, dtype=np.float32) for _ in posteriors],
            semantic_posteriors=values,
            semantic_abstained=np.zeros(len(posteriors), dtype=np.bool_),
            global_mask_ids=global_mask_ids,
            valid_pixel_count=4,
            class_count=len(CLASSES),
        )

    bank = AlphaMaskEvidenceBank.from_frames(
        scene_id="scene-production-score",
        point_count=4,
        xyz_m=np.zeros((4, 3), dtype=np.float32),
        class_names=CLASSES,
        frames=(
            # Both rows satisfy the persisted-posterior contract, but their
            # float32 sums differ slightly.  Production averages them before
            # its per-view normalization; the former replay normalized each
            # mask first and therefore produced different score bits.
            frame(0, (0, 1), ((0.900004, 0.1), (0.2, 0.799996))),
            frame(1, (2,), ((0.6, 0.4),)),
        ),
        source={"fixture": "production-score-equivalence"},
    )
    gaussian_ids = np.arange(4, dtype=np.int64)
    mean_consensus = 0.37
    mean_detection = 0.61
    ownership = FunnelObject(
        stable_id="object:0",
        gaussian_ids=gaussian_ids,
        mask_ids=(0, 1, 2),
        frame_ids=(0, 1),
        metadata={
            "mean_view_consensus": mean_consensus,
            "mean_detection_ratio": mean_detection,
            "geometric_quality": float(np.sqrt(mean_consensus * mean_detection)),
        },
    )

    rows, reasons = _export_with_reasons(
        (ownership,), bank=bank, allowed_classes=("chair",)
    )
    posterior = _object_class_distribution(ownership.mask_ids, bank)
    winner = int(np.flatnonzero(posterior == posterior.max())[0])
    winner_probability = float(posterior[winner])
    expected = CleanCandidate(
        object_id=ownership.stable_id,
        gaussian_ids=gaussian_ids,
        class_id=bank.class_names[winner],
        winner_probability=winner_probability,
        view_consensus=mean_consensus,
        detection_ratio=mean_detection,
    )
    _, former_replay_probability = _class_for_masks(ownership.mask_ids, bank)

    assert former_replay_probability != winner_probability
    assert _production_class_for_masks(ownership.mask_ids, bank) == (
        "chair",
        winner_probability,
    )
    assert len(rows) == 1
    assert rows[0].metadata["winner_probability"].hex() == winner_probability.hex()
    assert rows[0].metadata["score"].hex() == expected.score.hex()
    assert not any(reasons.values())


def test_four_arms_share_the_same_frozen_components_and_factor_order() -> None:
    result = _run()

    assert tuple(result.arms) == ("A1B1", "A0B1", "A1B0", "A0B0")
    assert [tuple(row.mask_ids) for row in result.accepted_components] == [
        (0, 1),
        (2,),
        (3, 5),
        (6, 7),
    ]
    assert result.shared_identity
    assert all(bool(value) for value in result.shared_identity.values())


def test_a0_retains_full_union_and_records_zero_visible_ratio() -> None:
    result = _run()
    profile = _find_component(result.detection_evidence, (0, 1))

    assert profile.full_ids.tolist() == [0, 1, 2, 3, 4]
    assert profile.detection_ratios.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]
    assert profile.historical_ids.tolist() == [0, 1, 2, 3]

    a0 = _find_component(result.arms["A0B1"].detection_objects, (0, 1))
    a1 = _find_component(result.arms["A1B1"].detection_objects, (0, 1))
    assert a0.gaussian_ids.tolist() == profile.full_ids.tolist()
    assert a1.gaussian_ids.tolist() == profile.historical_ids.tolist()
    assert a0.metadata["mean_detection_ratio"] == pytest.approx(0.8)


def test_b0_is_exactly_the_physical_diagnostic_stage_and_never_formal_output() -> None:
    result = _run()

    for a_code, b0_code in (("A1B1", "A1B0"), ("A0B1", "A0B0")):
        strict = result.arms[a_code]
        diagnostic = result.arms[b0_code]
        assert _snapshot_objects(diagnostic.physical_objects) == _snapshot_objects(
            strict.physical_objects
        )
        assert diagnostic.formal_output_allowed is False
        assert diagnostic.formal_output is None
        assert diagnostic.final_equivalence is None


def _snapshot_objects(rows: Sequence[object]) -> tuple[object, ...]:
    return tuple(
        (
            row.stable_id,
            tuple(map(int, row.mask_ids)),
            tuple(map(int, row.frame_ids)),
            tuple(map(int, row.gaussian_ids.tolist())),
            row.class_name,
            tuple(sorted(dict(row.metadata).items())),
        )
        for row in rows
    )


def test_drop_reasons_keep_min_views_empty_and_detection_separate() -> None:
    result = _run()
    reasons = dict(result.arms["A1B1"].drop_reasons)

    assert reasons["min_views"] == 1
    assert reasons["empty_full"] == 1
    assert reasons["detection_ratio"] == 1
    assert sum(reasons.values()) >= 3


def test_empty_accepted_full_union_is_diagnostic_not_geometric_candidate() -> None:
    """Exercise the real replay/evaluator boundary that failed on cloud DEV2."""

    result = _run()
    candidates, inventory = _nonempty_geometric_candidates(
        result.accepted_components
    )

    assert inventory == {
        "total_component_count": 4,
        "empty_full_component_count": 1,
        "geometric_candidate_count": 3,
    }
    assert any(len(item.gaussian_ids) == 0 for item in result.accepted_components)
    assert all(len(item.gaussian_ids) > 0 for item in candidates)
    assert result.arms["A1B1"].drop_reasons["empty_full"] == 1

    bank = _bank()
    nearest = build_bidirectional_nearest(bank.xyz_m, bank.xyz_m)
    metrics = _candidate_metrics(
        candidates,
        gt_objects=(
            GroundTruthObject(
                object_id=0,
                class_id="chair",
                point_ids=np.arange(bank.point_count, dtype=np.int64),
                official_valid=True,
                is_tiny_small=True,
            ),
        ),
        nearest=nearest,
        min_region_size=1,
    )

    assert metrics["subsets"]["all"]["candidate_count"] == 3


def test_replay_api_has_no_ground_truth_or_category_prior_channel() -> None:
    parameters = inspect.signature(replay_late_filter_factorial).parameters

    assert tuple(parameters) == (
        "bank",
        "diagnostics",
        "allowed_classes",
        "frozen_output",
    )
    assert not any(
        token in name.lower()
        for name in parameters
        for token in ("gt", "ground_truth", "prior", "iou")
    )
    with pytest.raises(TypeError):
        replay_late_filter_factorial(
            bank=_bank(),
            diagnostics=_diagnostics(),
            allowed_classes=("chair",),
            gt_instance=np.zeros(13, dtype=np.int64),
        )


def test_factorial_replay_is_deterministic_and_stably_ordered() -> None:
    first = _run()
    second = _run()

    assert _snapshot(first) == _snapshot(second)

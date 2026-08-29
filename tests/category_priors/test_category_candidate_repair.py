from __future__ import annotations

import inspect
from dataclasses import fields

import numpy as np
import pytest

from category_priors.category_candidate_repair import (
    ASSIGNMENT_THRESHOLD,
    CONSISTENT_ENVELOPE,
    LEGACY,
    RAW_ANCHORED_ENVELOPE,
    CandidateRepairScene,
    CandidateRepairTrace,
    repair_class_candidates,
)


def _one_dimensional_scene(
    positions: list[float],
    *,
    scores: list[float] | None = None,
    features: np.ndarray | None = None,
) -> CandidateRepairScene:
    count = len(positions)
    if features is None:
        features = np.column_stack((np.ones(count), np.zeros(count)))
    if scores is None:
        scores = [0.9] * count
    return CandidateRepairScene(
        instance_features=np.asarray(features, dtype=np.float64),
        xyz_scene=np.column_stack(
            (
                np.asarray(positions, dtype=np.float64),
                np.zeros(count),
                np.zeros(count),
            )
        ),
        semantic_top1_score=np.asarray(scores, dtype=np.float64),
    )


def _two_cluster_trace(selected_count: int) -> CandidateRepairTrace:
    return CandidateRepairTrace(
        selected_global_indices=np.arange(selected_count, dtype=np.int64),
        sampled_local_indices=np.arange(6, dtype=np.int64),
        raw_cluster_labels=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64),
    )


def test_legacy_is_an_exact_selected_axis_passthrough() -> None:
    scene = _one_dimensional_scene(list(range(8)))
    selected = np.asarray([4, 1, 7], dtype=np.int64)
    full = np.asarray([10, -1, 3], dtype=np.int64)
    core = np.asarray([10, 6, -1], dtype=np.int64)
    confidence = np.asarray([0.75, 0.125, 1.0], dtype=np.float64)
    trace = CandidateRepairTrace(
        selected_global_indices=selected,
        sampled_local_indices=np.asarray([2, 0], dtype=np.int64),
        raw_cluster_labels=np.asarray([9, 4], dtype=np.int64),
        legacy_full_labels=full,
        legacy_core_labels=core,
        legacy_assignment_confidence=confidence,
    )

    result = repair_class_candidates(scene, trace, LEGACY)

    np.testing.assert_array_equal(result.selected_global_indices, selected)
    np.testing.assert_array_equal(result.full_candidate_labels, full)
    np.testing.assert_array_equal(result.trusted_core_labels, core)
    np.testing.assert_array_equal(result.assignment_confidence, confidence)
    # Raw labels are compacted only for the diagnostic raw-seed axis.  Legacy
    # output labels themselves are deliberately not compacted or repaired.
    np.testing.assert_array_equal(result.raw_seed_cluster_index, [0, -1, 1])
    np.testing.assert_array_equal(
        result.scatter_labels(8), [-1, -1, -1, -1, 10, -1, -1, 3]
    )
    assert result.diagnostics["legacy_passthrough"] is True
    assert result.diagnostics["core_contract_enforced"] is False
    assert result.diagnostics["trusted_core_outside_full_count"] == 1


@pytest.mark.parametrize(
    "mode", [CONSISTENT_ENVELOPE, RAW_ANCHORED_ENVELOPE]
)
def test_envelope_uses_nearest_distance_and_leaves_exact_ties_background(
    mode: str,
) -> None:
    # The first six points are the raw HDBSCAN members.  Point 6 is a nearby
    # non-core point inside cluster 0's q95 envelope.  Point 7 is exactly
    # halfway between the two medoids and therefore remains background.
    scene = _one_dimensional_scene([0.0, 0.1, 0.2, 9.8, 9.9, 10.0, 0.15, 5.0])
    trace = _two_cluster_trace(8)

    result = repair_class_candidates(scene, trace, mode)

    np.testing.assert_array_equal(
        result.full_candidate_labels, [0, 0, 0, 1, 1, 1, 0, -1]
    )
    np.testing.assert_array_equal(
        result.trusted_core_labels, [0, 0, 0, 1, 1, 1, -1, -1]
    )
    assert result.assignment_confidence[6] >= ASSIGNMENT_THRESHOLD
    assert result.assignment_confidence[7] == 0.0
    assert result.diagnostics["exact_tie_point_count"] == 1
    assert result.diagnostics["trusted_core_outside_full_count"] == 0
    assert np.all(
        (result.trusted_core_labels < 0)
        | (result.trusted_core_labels == result.full_candidate_labels)
    )


def test_raw_anchoring_keeps_every_nonnoise_seed_with_own_probability() -> None:
    # The fourth raw member of cluster 0 is geometrically beside cluster 1.
    # C1 rejects it as a trusted member; C2 pins it to its raw cluster even
    # though the probability of that *own* cluster is below 0.3.
    scene = _one_dimensional_scene(
        [0.0, 0.1, 0.2, 9.6, 9.7, 9.8, 9.9, 10.0]
    )
    trace = CandidateRepairTrace(
        selected_global_indices=np.arange(8, dtype=np.int64),
        sampled_local_indices=np.arange(8, dtype=np.int64),
        raw_cluster_labels=np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
    )

    consistent = repair_class_candidates(scene, trace, CONSISTENT_ENVELOPE)
    anchored = repair_class_candidates(scene, trace, RAW_ANCHORED_ENVELOPE)

    assert consistent.full_candidate_labels[3] == -1
    assert consistent.trusted_core_labels[3] == -1
    np.testing.assert_array_equal(anchored.full_candidate_labels, [0] * 4 + [1] * 4)
    np.testing.assert_array_equal(
        anchored.trusted_core_labels, anchored.full_candidate_labels
    )
    assert anchored.raw_seed_own_probability[3] < ASSIGNMENT_THRESHOLD
    assert anchored.assignment_confidence[3] == pytest.approx(
        anchored.raw_seed_own_probability[3]
    )
    assert anchored.assignment_confidence[3] != pytest.approx(
        float(np.max(anchored.assignment_confidence))
    )


def test_raw_medoid_and_spatial_scale_use_the_original_complete_scene_axis() -> None:
    # Unselected points at -100 and 100 make the original all-scene XYZ span
    # 200.  The sampled selected points span 10, so the persisted spatial max
    # must be 10/200 = 0.05 rather than a selected-only value of 1.0.
    scene = _one_dimensional_scene(
        [-100.0, 0.0, 0.1, 0.2, 9.8, 9.9, 10.0, 100.0]
    )
    trace = CandidateRepairTrace(
        selected_global_indices=np.arange(1, 7, dtype=np.int64),
        sampled_local_indices=np.arange(6, dtype=np.int64),
        raw_cluster_labels=np.asarray([0, 0, 0, 1, 1, 1]),
    )

    result = repair_class_candidates(scene, trace, RAW_ANCHORED_ENVELOPE)

    assert result.diagnostics["spatial_distance_max"] == pytest.approx(0.05)
    assert [row["medoid_global_index"] for row in result.candidates] == [2, 5]
    assert all(row["envelope_radius"] > 0.0 for row in result.candidates)


def test_empty_raw_clusters_are_a_complete_background_result() -> None:
    scene = _one_dimensional_scene([0.0, 1.0, 2.0, 3.0])
    trace = CandidateRepairTrace(
        selected_global_indices=np.asarray([1, 3], dtype=np.int64),
        sampled_local_indices=np.asarray([0, 1], dtype=np.int64),
        raw_cluster_labels=np.asarray([-1, -1], dtype=np.int64),
    )

    result = repair_class_candidates(scene, trace, CONSISTENT_ENVELOPE)

    np.testing.assert_array_equal(result.full_candidate_labels, [-1, -1])
    np.testing.assert_array_equal(result.trusted_core_labels, [-1, -1])
    np.testing.assert_array_equal(result.assignment_confidence, [0.0, 0.0])
    assert result.raw_cluster_ids == ()
    assert result.candidates == ()
    assert result.diagnostics["candidate_count"] == 0


def test_repair_is_deterministic_and_returns_read_only_arrays() -> None:
    scene = _one_dimensional_scene([0.0, 0.1, 0.2, 9.8, 9.9, 10.0, 0.15, 5.0])
    selected_input = np.arange(8, dtype=np.int64)
    trace = CandidateRepairTrace(
        selected_global_indices=selected_input,
        sampled_local_indices=np.arange(6, dtype=np.int64),
        raw_cluster_labels=np.asarray([0, 0, 0, 1, 1, 1]),
    )

    first = repair_class_candidates(scene, trace, CONSISTENT_ENVELOPE)
    second = repair_class_candidates(scene, trace, CONSISTENT_ENVELOPE)

    for name in (
        "selected_global_indices",
        "raw_seed_cluster_index",
        "trusted_core_labels",
        "full_candidate_labels",
        "assignment_confidence",
        "raw_seed_own_probability",
    ):
        left = getattr(first, name)
        right = getattr(second, name)
        np.testing.assert_allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True)
        assert left.flags.writeable is False
    assert first.candidates == second.candidates
    assert first.diagnostics == second.diagnostics
    assert selected_input.flags.writeable is True


def test_runtime_interface_has_no_gt_class_or_prior_input() -> None:
    function_parameters = set(inspect.signature(repair_class_candidates).parameters)
    scene_fields = {field.name for field in fields(CandidateRepairScene)}
    trace_fields = {field.name for field in fields(CandidateRepairTrace)}
    runtime_names = function_parameters | scene_fields | trace_fields

    assert function_parameters == {"scene", "trace", "mode"}
    assert not any(
        forbidden in name.lower()
        for name in runtime_names
        for forbidden in ("ground_truth", "gt_", "prior", "class_name", "class_id")
    )


def test_trace_axis_validation_rejects_duplicate_or_out_of_range_indices() -> None:
    scene = _one_dimensional_scene([0.0, 1.0, 2.0])
    duplicate_selected = CandidateRepairTrace(
        selected_global_indices=np.asarray([0, 0]),
        sampled_local_indices=np.asarray([0]),
        raw_cluster_labels=np.asarray([-1]),
    )
    with pytest.raises(ValueError, match="selected_global_indices must be unique"):
        repair_class_candidates(scene, duplicate_selected, CONSISTENT_ENVELOPE)

    invalid_sample = CandidateRepairTrace(
        selected_global_indices=np.asarray([0, 1]),
        sampled_local_indices=np.asarray([2]),
        raw_cluster_labels=np.asarray([-1]),
    )
    with pytest.raises(ValueError, match="outside the selected axis"):
        repair_class_candidates(scene, invalid_sample, CONSISTENT_ENVELOPE)

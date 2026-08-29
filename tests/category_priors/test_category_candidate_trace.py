from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from category_priors.category_candidate_trace import (
    CandidateFormationClassCapture,
    assert_candidate_bank_identity,
    build_candidate_formation_trace,
    compare_candidate_bank_identity,
    load_candidate_formation_trace,
    save_candidate_formation_trace,
    validate_candidate_formation_trace,
    validate_candidate_formation_trace_views,
)
from category_priors.category_denoise import CandidateBank


def _classes() -> tuple[str, ...]:
    return ("chair",) + tuple(f"class-{index}" for index in range(1, 32))


def _candidate(*, base_score: float = 0.55) -> dict[str, object]:
    return {
        "candidate_id": 0,
        "branch_class": "chair",
        "branch_class_index": 0,
        "hdbscan_cluster_id": 0,
        "semantic_selected_point_count": 8,
        "sampled_point_count": 7,
        "core_point_count": 3,
        "full_point_count": 4,
        "assignment_confidence_mean": 0.825,
        "metric_extents_m": [0.1, 0.2, 0.3],
        "boundary_ratio_5cm": 0.25,
        "vote_winner_index": 0,
        "vote_winner": "chair",
        "vote_winner_unique": True,
        "branch_vote_ratio": 0.70,
        "background_vote_ratio": 0.30,
        "base_score": base_score,
    }


def _bank(
    *,
    candidate: dict[str, object] | None = None,
    full_labels: np.ndarray | None = None,
    assignment_confidence: np.ndarray | None = None,
) -> CandidateBank:
    if full_labels is None:
        full_labels = np.asarray([0, -1, -1, 0, 0, -1, 0, -1], dtype=np.int64)
    if assignment_confidence is None:
        assignment_confidence = np.asarray(
            [0.9, 0.0, 0.0, 0.8, 0.8, 0.0, 0.8, 0.0],
            dtype=np.float64,
        )
    return CandidateBank(
        class_names=_classes(),
        saga20_names=("chair",),
        scene_scale_m_per_unit=1.5,
        seed=42,
        global_pre_knn=np.full(8, -1, dtype=np.int64),
        semantic_top1=np.zeros(8, dtype=np.int64),
        semantic_top1_score=np.full(8, 0.9, dtype=np.float64),
        branch_full_labels=np.asarray(full_labels, dtype=np.int64),
        branch_core_labels=np.asarray(
            [0, 0, 0, -1, -1, -1, -1, -1], dtype=np.int64
        ),
        assignment_confidence=np.asarray(
            assignment_confidence, dtype=np.float64
        ),
        candidates=(dict(candidate or _candidate()),),
        diagnostics={
            "scene_id": "scene0001_00",
            "semantic_threshold": 0.7,
            "sample_cap": 7,
            "min_cluster_size": 3,
            "min_samples": 3,
            "weights": {
                "instance": 0.5,
                "spatial": 0.3,
                "semantic": 0.2,
            },
            "assignment_threshold": 0.3,
        },
    )


def _capture() -> CandidateFormationClassCapture:
    # Raw cluster 0 contains sampled points 0/1/2; point 1 is reassigned to
    # cluster 1 and point 2 misses the confidence threshold.  Raw cluster 1
    # contains 3/5/7; only point 1 is assigned to it above threshold, so its
    # full set has one point and the raw cluster is discarded from bank v1.
    legacy_feature_similarity = np.asarray(
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
        dtype=np.float64,
    )
    legacy_spatial_distance = np.asarray(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        dtype=np.float64,
    )
    legacy_spatial_similarity = np.exp(-legacy_spatial_distance)
    return CandidateFormationClassCapture(
        branch_class="chair",
        branch_class_index=0,
        selected_indices=np.arange(8, dtype=np.int64),
        sampled_local_indices=np.asarray([3, 5, 2, 0, 1, 7, 4], dtype=np.int64),
        hdbscan_labels=np.asarray([1, 1, 0, 0, 0, 1, -1], dtype=np.int64),
        hdbscan_membership=np.asarray(
            [0.95, 0.75, 0.7, 0.9, 0.8, 0.85, 0.0], dtype=np.float64
        ),
        raw_cluster_ids=(0, 1),
        prethreshold_argmax_center=np.asarray(
            [0, 1, 0, 0, 0, 1, 0, 1], dtype=np.int64
        ),
        prethreshold_assignment_confidence=np.asarray(
            [0.9, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8, 0.2],
            dtype=np.float64,
        ),
        legacy_assignment_chosen_center=np.asarray(
            [0, 1, 0, 0, 0, 1, 0, 1], dtype=np.int64
        ),
        legacy_assignment_feature_similarity=legacy_feature_similarity,
        legacy_assignment_feature_center_norm=np.ones(8, dtype=np.float64),
        legacy_assignment_spatial_distance_standardized=(
            legacy_spatial_distance
        ),
        legacy_assignment_spatial_similarity=legacy_spatial_similarity,
        legacy_assignment_hybrid_similarity=(
            0.5 * legacy_feature_similarity
            + 0.5 * legacy_spatial_similarity
        ),
        legacy_assignment_xyz_denominator=np.asarray(
            [1.0, 2.0, 3.0], dtype=np.float64
        ),
        legacy_assignment_softmax_temperature=10.0,
        # sampled order is [3,5,2,0,1,7,4].  Raw cluster 1 occupies sample
        # positions 0/1/5 (medoid 0), raw cluster 0 positions 2/3/4
        # (medoid 2), and position 6 is noise.
        sampled_raw_medoid_local_index=np.asarray(
            [0, 0, 2, 2, 2, 0, -1], dtype=np.int64
        ),
        sampled_medoid_instance_distance=np.asarray(
            [0.0, 0.2, 0.0, 0.2, 0.4, 0.4, 0.0], dtype=np.float64
        ),
        sampled_medoid_spatial_distance=np.asarray(
            [0.0, 0.1, 0.0, 0.1, 0.2, 0.2, 0.0], dtype=np.float64
        ),
        sampled_medoid_semantic_distance=np.asarray(
            [0.0, 0.3, 0.0, 0.3, 0.1, 0.1, 0.0], dtype=np.float64
        ),
        sampled_medoid_hybrid_distance=np.asarray(
            [0.0, 0.19, 0.0, 0.19, 0.28, 0.28, 0.0], dtype=np.float64
        ),
        diagnostics={
            "distance_components": {
                "instance_weight": 0.5,
                "spatial_weight": 0.3,
                "semantic_weight": 0.2,
            },
            "scene_xyz_safe_span": [1.0, 2.0, 3.0],
        },
    )


def _trace(bank: CandidateBank | None = None):
    return build_candidate_formation_trace(
        scene_id="scene0001_00",
        bank=bank or _bank(),
        class_captures=(_capture(),),
        sample_cap=7,
        diagnostics={"purpose": "shadow-only"},
    )


def test_trace_preserves_discarded_raw_cluster_and_sample_order() -> None:
    bank = _bank()
    trace = _trace(bank)
    rows = {
        int(row["hdbscan_cluster_id"]): row
        for row in trace.raw_cluster_rows
    }

    np.testing.assert_array_equal(trace.sample_rank, [3, 4, 2, 0, 6, 1, -1, 5])
    np.testing.assert_array_equal(
        trace.hdbscan_labels, [0, 0, 0, 1, -1, 1, -1, 1]
    )
    np.testing.assert_array_equal(
        trace.raw_medoid_point_index, [2, 2, 2, 3, -1, 3, -1, 3]
    )
    np.testing.assert_allclose(
        trace.raw_medoid_hybrid_distance,
        0.5 * trace.raw_medoid_instance_distance
        + 0.3 * trace.raw_medoid_spatial_distance
        + 0.2 * trace.raw_medoid_semantic_distance,
    )
    assert rows[0]["retained_candidate_id"] == 0
    assert rows[0]["thresholded_full_count"] == 4
    assert rows[1]["retained_candidate_id"] is None
    assert rows[1]["retention_status"] == (
        "discarded_full_below_min_cluster_size"
    )
    assert rows[1]["sampled_member_count"] == 3
    assert rows[1]["thresholded_full_count"] == 1
    assert 1 not in set(np.asarray(bank.branch_core_labels).tolist())


def test_trace_exposes_cross_cluster_and_threshold_reassignment() -> None:
    trace = _trace()
    rows = {
        int(row["hdbscan_cluster_id"]): row
        for row in trace.raw_cluster_rows
    }

    # Sample point 1 belongs to raw cluster 0 but full assignment chooses raw
    # cluster 1.  Sample point 3 moves in the opposite direction.
    assert trace.raw_cluster_membership[1] == 0
    assert trace.prethreshold_argmax_raw_cluster[1] == 1
    assert trace.raw_cluster_membership[3] == 1
    assert trace.prethreshold_argmax_raw_cluster[3] == 0
    assert rows[0]["raw_member_cross_argmax_count"] == 1
    assert rows[1]["raw_member_cross_argmax_count"] == 1
    assert rows[0]["raw_member_threshold_rejected_count"] == 1
    assert rows[1]["raw_member_threshold_rejected_count"] == 2


def test_trace_separates_raw_medoid_distances_from_legacy_mean_assignment() -> None:
    trace = _trace()

    np.testing.assert_array_equal(
        trace.legacy_assignment_chosen_raw_cluster,
        trace.prethreshold_argmax_raw_cluster,
    )
    np.testing.assert_allclose(
        trace.legacy_assignment_spatial_similarity,
        np.exp(-trace.legacy_assignment_spatial_distance_standardized),
    )
    np.testing.assert_allclose(
        trace.legacy_assignment_hybrid_similarity,
        0.5 * trace.legacy_assignment_feature_similarity
        + 0.5 * trace.legacy_assignment_spatial_similarity,
    )
    assert not np.array_equal(
        trace.raw_medoid_hybrid_distance,
        trace.legacy_assignment_hybrid_similarity,
    )
    class_row = trace.class_rows[0]
    assert class_row["legacy_assignment_xyz_denominator"] == [1.0, 2.0, 3.0]
    assert class_row["legacy_assignment_softmax_temperature"] == 10.0


def test_trace_roundtrip_is_pickle_free_read_only_and_bank_validated(
    tmp_path: Path,
) -> None:
    bank = _bank()
    trace = _trace(bank)

    npz_path, json_path = save_candidate_formation_trace(
        trace, tmp_path / "trace"
    )
    loaded = load_candidate_formation_trace(tmp_path / "trace")

    assert npz_path.name == "formation_trace.npz"
    assert json_path is not None and json_path.name == "formation_trace.json"
    assert npz_path.is_file() and json_path.is_file()
    expected_views = {
        "sample_rank.npz",
        "raw_hdbscan_labels.npz",
        "raw_membership.npz",
        "prethreshold_assignment.npz",
        "distance_components.npz",
        "raw_clusters.json",
        "trace_diagnostics.json",
    }
    assert expected_views.issubset(
        {path.name for path in (tmp_path / "trace").iterdir()}
    )
    with np.load(tmp_path / "trace" / "sample_rank.npz") as view:
        np.testing.assert_array_equal(view["sample_rank"], trace.sample_rank)
        np.testing.assert_array_equal(
            view["semantic_selected_class_index"],
            trace.semantic_selected_class_index,
        )
    with np.load(tmp_path / "trace" / "raw_hdbscan_labels.npz") as view:
        np.testing.assert_array_equal(
            view["hdbscan_labels"], trace.hdbscan_labels
        )
        np.testing.assert_allclose(
            view["hdbscan_membership"], trace.hdbscan_membership
        )
    with np.load(tmp_path / "trace" / "raw_membership.npz") as view:
        np.testing.assert_array_equal(
            view["raw_cluster_membership"], trace.raw_cluster_membership
        )
    with np.load(
        tmp_path / "trace" / "prethreshold_assignment.npz"
    ) as view:
        np.testing.assert_array_equal(
            view["prethreshold_argmax_raw_cluster"],
            trace.prethreshold_argmax_raw_cluster,
        )
        np.testing.assert_allclose(
            view["prethreshold_assignment_confidence"],
            trace.prethreshold_assignment_confidence,
        )
        np.testing.assert_array_equal(
            view["legacy_assignment_chosen_raw_cluster"],
            trace.legacy_assignment_chosen_raw_cluster,
        )
    with np.load(tmp_path / "trace" / "distance_components.npz") as view:
        assert "raw_medoid_hybrid_distance" in view.files
        assert "legacy_assignment_hybrid_similarity" in view.files
        np.testing.assert_allclose(
            view["legacy_assignment_feature_similarity"],
            trace.legacy_assignment_feature_similarity,
        )
        np.testing.assert_allclose(
            view["legacy_assignment_spatial_distance_standardized"],
            trace.legacy_assignment_spatial_distance_standardized,
        )
        np.testing.assert_allclose(
            view["legacy_assignment_hybrid_similarity"],
            trace.legacy_assignment_hybrid_similarity,
        )
    raw_clusters = json.loads(
        (tmp_path / "trace" / "raw_clusters.json").read_text(encoding="utf-8")
    )
    diagnostics = json.loads(
        (tmp_path / "trace" / "trace_diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    assert raw_clusters["read_only"] is True
    assert raw_clusters["raw_clusters"] == list(trace.raw_cluster_rows)
    assert diagnostics["read_only"] is True
    assert diagnostics["class_rows"] == list(trace.class_rows)
    np.testing.assert_array_equal(loaded.sample_rank, trace.sample_rank)
    np.testing.assert_array_equal(
        loaded.raw_cluster_membership, trace.raw_cluster_membership
    )
    np.testing.assert_allclose(
        loaded.legacy_assignment_hybrid_similarity,
        trace.legacy_assignment_hybrid_similarity,
    )
    assert loaded.raw_cluster_rows == trace.raw_cluster_rows
    assert loaded.class_rows == trace.class_rows
    assert not loaded.sample_rank.flags.writeable
    assert not loaded.prethreshold_assignment_confidence.flags.writeable
    validate_candidate_formation_trace(loaded, bank=bank)
    validate_candidate_formation_trace_views(loaded, tmp_path / "trace")


def test_canonical_view_validator_covers_legacy_assignment_components(
    tmp_path: Path,
) -> None:
    trace = _trace()
    save_candidate_formation_trace(trace, tmp_path / "trace")
    component_path = tmp_path / "trace" / "distance_components.npz"
    with np.load(component_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["legacy_assignment_hybrid_similarity"][0] += 0.01
    np.savez_compressed(component_path, **arrays)

    with pytest.raises(ValueError, match="legacy_assignment_hybrid_similarity"):
        validate_candidate_formation_trace_views(trace, tmp_path / "trace")


def test_trace_validation_rejects_array_row_and_bank_tampering() -> None:
    bank = _bank()
    trace = _trace(bank)

    bad_membership = np.asarray(trace.raw_cluster_membership).copy()
    bad_membership[6] = 99
    with pytest.raises(ValueError, match="undeclared|membership"):
        validate_candidate_formation_trace(
            replace(trace, raw_cluster_membership=bad_membership)
        )

    bad_rank = np.asarray(trace.sample_rank).copy()
    bad_rank[0], bad_rank[1] = bad_rank[1], bad_rank[0]
    with pytest.raises(ValueError, match="deterministic RNG"):
        validate_candidate_formation_trace(replace(trace, sample_rank=bad_rank))

    bad_rows = [dict(row) for row in trace.raw_cluster_rows]
    bad_rows[1]["thresholded_full_count"] = 3
    with pytest.raises(ValueError, match="thresholded_full_count"):
        validate_candidate_formation_trace(
            replace(trace, raw_cluster_rows=tuple(bad_rows))
        )

    changed_confidence = np.asarray(bank.assignment_confidence).copy()
    changed_confidence[0] += 2e-6
    with pytest.raises(ValueError, match="confidence"):
        validate_candidate_formation_trace(
            trace,
            bank=replace(bank, assignment_confidence=changed_confidence),
        )

    bad_legacy_hybrid = np.asarray(
        trace.legacy_assignment_hybrid_similarity
    ).copy()
    bad_legacy_hybrid[0] += 0.01
    with pytest.raises(ValueError, match="legacy hybrid similarity"):
        validate_candidate_formation_trace(
            replace(
                trace,
                legacy_assignment_hybrid_similarity=bad_legacy_hybrid,
            )
        )


def test_bank_identity_uses_exact_labels_and_one_e_minus_six_numeric_gate() -> None:
    reference = _bank()
    close_candidate = dict(reference.candidates[0])
    close_candidate["base_score"] = float(close_candidate["base_score"]) + 5e-7
    close_confidence = np.asarray(reference.assignment_confidence).copy()
    close_confidence[0] += 5e-7
    close = replace(
        reference,
        candidates=(close_candidate,),
        assignment_confidence=close_confidence,
    )

    result = compare_candidate_bank_identity(reference, close)

    assert result.matches
    assert_candidate_bank_identity(reference, close)

    far_candidate = dict(reference.candidates[0])
    far_candidate["base_score"] = float(far_candidate["base_score"]) + 2e-6
    far = replace(reference, candidates=(far_candidate,))
    result = compare_candidate_bank_identity(reference, far)
    assert not result.matches
    assert "candidates[0].base_score" in result.mismatches
    with pytest.raises(ValueError, match="base_score"):
        assert_candidate_bank_identity(reference, far)

    changed_labels = np.asarray(reference.branch_full_labels).copy()
    changed_labels[0] = -1
    label_result = compare_candidate_bank_identity(
        reference, replace(reference, branch_full_labels=changed_labels)
    )
    assert not label_result.matches
    assert "branch_full_labels" in label_result.mismatches

    changed_candidate = dict(reference.candidates[0])
    changed_candidate["hdbscan_cluster_id"] = 1
    candidate_result = compare_candidate_bank_identity(
        reference, replace(reference, candidates=(changed_candidate,))
    )
    assert not candidate_result.matches
    assert "candidates[0].hdbscan_cluster_id" in candidate_result.mismatches


def test_build_rejects_capture_that_cannot_reproduce_frozen_bank() -> None:
    capture = replace(
        _capture(),
        prethreshold_assignment_confidence=np.asarray(
            [0.9, 0.8, 0.2, 0.8, 0.8, 0.2, 0.2, 0.2],
            dtype=np.float64,
        ),
    )

    with pytest.raises(ValueError, match="full labels"):
        build_candidate_formation_trace(
            scene_id="scene0001_00",
            bank=_bank(),
            class_captures=(capture,),
            sample_cap=7,
        )

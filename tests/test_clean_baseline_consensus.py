from __future__ import annotations

import inspect

import numpy as np
import pytest

import category_priors.clean_baseline.consensus as consensus_module
from category_priors.clean_baseline.consensus import (
    ConsensusConfig,
    ConsensusObject,
    MaskObservation,
    PairConsensus,
    compute_pair_consensus,
    detect_undersegmented_masks,
    remove_contained_objects,
    run_mask_consensus,
    split_disconnected_support,
)
from category_priors.clean_baseline.size_prior import (
    SizePriorTable,
    base_ap_score,
    global_size_compatibility,
    make_size_merge_veto,
    oracle_class_size_compatibility,
    pca_sorted_extents_m,
    predicted_size_compatibility,
)
from category_priors.io import hash_json


def _signed_train_prior(payload: dict[str, object]) -> dict[str, object]:
    value = {
        "kind": "category_priors",
        "schema_version": "1.0",
        "provenance": {"splits": ["train"]},
        **payload,
    }
    value["content_sha256"] = hash_json(value)
    return value


def _line_points(count: int, *, start: float = 0.0) -> np.ndarray:
    return np.column_stack(
        (
            start + np.arange(count, dtype=np.float64) * 0.01,
            np.zeros(count),
            np.zeros(count),
        )
    )


def _observation(mask_id: int, frame_id: int, ids: object) -> MaskObservation:
    return MaskObservation(mask_id, frame_id, np.asarray(ids, dtype=np.int64))


def test_observer_supporter_consensus_counts_physical_views() -> None:
    ids = np.arange(4)
    observations = (
        _observation(10, 0, ids),
        _observation(20, 1, ids),
    )
    visibility = np.ones((2, 4), dtype=bool)

    result = compute_pair_consensus((0,), (1,), observations, visibility)

    assert result.observer_count == 2
    assert result.supporter_count == 2
    assert result.consensus == 1.0


def test_visible_frame_without_supporting_mask_counts_as_observer() -> None:
    ids = np.arange(4)
    observations = (
        _observation(10, 0, ids),
        _observation(20, 1, ids),
    )
    visibility = np.ones((3, 4), dtype=bool)

    result = compute_pair_consensus((0,), (1,), observations, visibility)

    assert result.observer_count == 3
    assert result.supporter_count == 2
    assert result.consensus == pytest.approx(2 / 3)


def test_supporter_requires_bilateral_eighty_percent_containment() -> None:
    left_ids = np.arange(5)
    right_ids = np.arange(5, 10)
    visibility = np.zeros((3, 10), dtype=bool)
    visibility[2, :] = True

    sixty_percent = (
        _observation(0, 0, left_ids),
        _observation(1, 1, right_ids),
        _observation(2, 2, [0, 1, 2, 5, 6, 7]),
    )
    eighty_percent = (
        _observation(0, 0, left_ids),
        _observation(1, 1, right_ids),
        _observation(2, 2, [0, 1, 2, 3, 5, 6, 7, 8]),
    )

    below = compute_pair_consensus((0,), (1,), sixty_percent, visibility)
    at_threshold = compute_pair_consensus(
        (0,), (1,), eighty_percent, visibility
    )

    assert (below.observer_count, below.supporter_count) == (1, 0)
    assert (at_threshold.observer_count, at_threshold.supporter_count) == (1, 1)


def test_ambiguous_hierarchical_points_do_not_supply_association_evidence() -> None:
    left = MaskObservation(0, 0, np.arange(8), np.arange(4))
    right = MaskObservation(1, 1, np.arange(4), np.arange(4))
    visibility = np.ones((2, 8), dtype=bool)

    result = compute_pair_consensus((0,), (1,), (left, right), visibility)

    assert result.observer_count == 0
    assert result.supporter_count == 0
    assert result.consensus == 0.0


def test_undersegmentation_filter_uses_diverse_observable_frame_frequency() -> None:
    one_diverse_frame = (
        _observation(0, 0, np.arange(10)),
        _observation(1, 1, np.arange(5)),
        _observation(2, 1, np.arange(5, 10)),
        _observation(3, 2, np.arange(10)),
        _observation(4, 3, np.arange(10)),
        _observation(5, 4, np.arange(10)),
    )
    two_diverse_frames = one_diverse_frame[:3] + (
        _observation(6, 2, np.arange(5)),
        _observation(7, 2, np.arange(5, 10)),
    ) + one_diverse_frame[4:]
    visibility = np.ones((5, 10), dtype=bool)

    below_threshold = detect_undersegmented_masks(one_diverse_frame, visibility)
    above_threshold = detect_undersegmented_masks(two_diverse_frames, visibility)

    assert 0 not in below_threshold  # 1 / 5 observable frames
    assert 0 in above_threshold  # 2 / 5 observable frames, strictly > .30


def test_undersegmentation_treats_no_mask_evidence_as_nondiverse() -> None:
    observations = (_observation(0, 0, np.arange(10)),)
    visibility = np.ones((4, 10), dtype=bool)

    assert detect_undersegmented_masks(observations, visibility) == ()


def test_undersegmentation_frequency_is_strictly_greater_than_thirty_percent() -> None:
    full = np.arange(10)
    observations: list[MaskObservation] = [_observation(0, 0, full)]
    mask_id = 1
    for frame_id in range(1, 4):
        observations.extend(
            (
                _observation(mask_id, frame_id, np.arange(5)),
                _observation(mask_id + 1, frame_id, np.arange(5, 10)),
            )
        )
        mask_id += 2
    for frame_id in range(4, 10):
        observations.append(_observation(mask_id, frame_id, full))
        mask_id += 1

    rejected = detect_undersegmented_masks(
        observations, np.ones((10, 10), dtype=bool)
    )

    assert 0 not in rejected  # exactly 3 / 10 is not strictly greater than .30


def test_undersegmentation_excludes_same_frame_ambiguous_distribution() -> None:
    full = np.arange(10)
    observations = (
        _observation(0, 0, full),
        MaskObservation(1, 1, np.arange(5), np.arange(5)),
        MaskObservation(2, 1, np.arange(5, 10), np.arange(5, 10)),
    )

    assert detect_undersegmented_masks(
        observations, np.ones((2, 10), dtype=bool)
    ) == ()


def test_iterative_consensus_merges_three_views_deterministically() -> None:
    ids = np.arange(6)
    observations = tuple(
        _observation(100 + frame_id, frame_id, ids) for frame_id in range(3)
    )
    visibility = np.ones((3, 6), dtype=bool)
    xyz = _line_points(6)

    forward = run_mask_consensus(observations, visibility, xyz)
    reverse = run_mask_consensus(tuple(reversed(observations)), visibility, xyz)

    assert len(forward.objects) == 1
    assert forward.objects[0].mask_ids == (100, 101, 102)
    assert forward.objects[0].gaussian_ids.tolist() == ids.tolist()
    assert forward.objects[0].mean_view_consensus == 1.0
    assert len(forward.accepted_edges) == 2
    assert [obj.mask_ids for obj in reverse.objects] == [(100, 101, 102)]
    assert reverse.objects[0].gaussian_ids.tolist() == ids.tolist()


def test_observer_percentiles_select_the_registered_top_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = np.arange(4)
    observations = tuple(_observation(mask_id, mask_id, ids) for mask_id in range(7))
    pair_counts = {
        pair: rank
        for rank, pair in enumerate(
            ((left, right) for left in range(7) for right in range(left + 1, 7)),
            start=1,
        )
    }

    def synthetic_consensus(
        _left_support: np.ndarray,
        _right_support: np.ndarray,
        left_masks: tuple[int, ...],
        right_masks: tuple[int, ...],
        *_args: object,
    ) -> PairConsensus:
        observer_count = pair_counts[(left_masks[0], right_masks[0])]
        return PairConsensus(left_masks, right_masks, observer_count, 0, 0.0)

    monkeypatch.setattr(
        consensus_module, "_pair_consensus_from_support", synthetic_consensus
    )
    result = run_mask_consensus(
        observations, np.ones((7, 4), dtype=bool), _line_points(4)
    )
    schedule = result.diagnostics["observer_schedule"]

    assert schedule[0]["observer_threshold"] == 20  # ceil(5% of 21) = 2 pairs
    assert schedule[1]["observer_threshold"] == 19  # ceil(10% of 21) = 3 pairs
    assert schedule[-1]["observer_threshold"] == 2  # ceil(95% of 21) = 20 pairs


def test_percentile_round_merges_the_whole_eligible_connected_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = np.arange(6)
    observations = tuple(_observation(mask_id, mask_id, ids) for mask_id in range(3))

    def synthetic_consensus(
        _left_support: np.ndarray,
        _right_support: np.ndarray,
        left_masks: tuple[int, ...],
        right_masks: tuple[int, ...],
        *_args: object,
    ) -> PairConsensus:
        endpoints = frozenset(left_masks + right_masks)
        # A-B and B-C qualify as singleton edges.  Once an old one-edge-at-a-
        # time implementation merges A-B, its recomputed (A,B)-C edge fails.
        qualifies = (
            len(left_masks) == len(right_masks) == 1
            and endpoints in (frozenset((0, 1)), frozenset((1, 2)))
        )
        return PairConsensus(
            left_masks,
            right_masks,
            10,
            10 if qualifies else 0,
            1.0 if qualifies else 0.0,
        )

    monkeypatch.setattr(
        consensus_module, "_pair_consensus_from_support", synthetic_consensus
    )
    forward = run_mask_consensus(
        observations, np.ones((3, 6), dtype=bool), _line_points(6)
    )
    reverse = run_mask_consensus(
        tuple(reversed(observations)),
        np.ones((3, 6), dtype=bool),
        _line_points(6),
    )

    assert [item.mask_ids for item in forward.objects] == [(0, 1, 2)]
    assert [item.mask_ids for item in reverse.objects] == [(0, 1, 2)]
    assert len(forward.accepted_edges) == 2
    assert [row["top_percent"] for row in forward.diagnostics["observer_schedule"]] == list(
        range(5, 100, 5)
    )


def test_connected_component_size_veto_is_all_or_nothing_and_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = np.arange(6)
    observations = tuple(_observation(mask_id, mask_id, ids) for mask_id in range(3))

    def synthetic_consensus(
        _left_support: np.ndarray,
        _right_support: np.ndarray,
        left_masks: tuple[int, ...],
        right_masks: tuple[int, ...],
        *_args: object,
    ) -> PairConsensus:
        endpoints = frozenset(left_masks + right_masks)
        qualifies = endpoints in (frozenset((0, 1)), frozenset((1, 2)))
        return PairConsensus(
            left_masks,
            right_masks,
            10,
            10 if qualifies else 0,
            1.0 if qualifies else 0.0,
        )

    monkeypatch.setattr(
        consensus_module, "_pair_consensus_from_support", synthetic_consensus
    )
    calls: list[tuple[int, ...]] = []

    def reject(mask_ids: tuple[int, ...], _ids: np.ndarray) -> bool:
        calls.append(mask_ids)
        return False

    result = run_mask_consensus(
        observations,
        np.ones((3, 6), dtype=bool),
        _line_points(6),
        merge_veto=reject,
    )

    assert calls == [(0, 1, 2)]
    assert result.accepted_edges == ()
    assert result.objects == ()


def test_same_frame_alternatives_never_share_an_object() -> None:
    ids = np.arange(6)
    observations = (
        _observation(0, 0, ids),
        _observation(1, 0, ids),
        _observation(2, 1, ids),
    )
    visibility = np.ones((2, 6), dtype=bool)
    result = run_mask_consensus(observations, visibility, _line_points(6))

    assert all(len(obj.frame_ids) == len(set(obj.frame_ids)) for obj in result.objects)
    assert not any({0, 1}.issubset(obj.mask_ids) for obj in result.objects)


def test_final_detection_ambiguous_frame_abstains_from_denominator() -> None:
    full = np.arange(4)
    observations = (
        MaskObservation(0, 0, full, np.asarray([0])),
        _observation(1, 1, full),
    )
    visibility = np.zeros((2, 4), dtype=bool)
    visibility[0, 0] = True
    visibility[1, :] = True

    result = run_mask_consensus(
        observations,
        visibility,
        _line_points(4),
        config=ConsensusConfig(point_filter_threshold=0.75),
    )

    assert len(result.objects) == 1
    assert result.objects[0].gaussian_ids.tolist() == full.tolist()
    assert result.objects[0].mean_detection_ratio == 1.0


def test_detection_ratio_is_inclusive_and_drops_weakly_observed_gaussian() -> None:
    common = np.arange(6)
    first_ids = np.arange(7)
    observations = (
        _observation(0, 0, first_ids),
        _observation(1, 1, common),
    )
    visibility = np.zeros((3, 7), dtype=bool)
    visibility[0, :] = True
    visibility[1, :] = True
    visibility[2, 6] = True

    result = run_mask_consensus(observations, visibility, _line_points(7))

    assert len(result.objects) == 1
    assert result.objects[0].gaussian_ids.tolist() == common.tolist()


def test_physical_dbscan_splits_disconnected_support_and_drops_noise() -> None:
    left = _line_points(4)
    right = _line_points(4, start=1.0)
    noise = np.asarray([[3.0, 0.0, 0.0]])
    xyz = np.vstack((left, right, noise))

    parts = split_disconnected_support(np.arange(9), xyz)

    assert [part.tolist() for part in parts] == [list(range(4)), list(range(4, 8))]


def test_consensus_outputs_one_object_per_disconnected_physical_part() -> None:
    xyz = np.vstack((_line_points(4), _line_points(4, start=1.0)))
    ids = np.arange(8)
    observations = (
        _observation(0, 0, ids),
        _observation(1, 1, ids),
    )
    result = run_mask_consensus(observations, np.ones((2, 8), dtype=bool), xyz)

    assert [obj.gaussian_ids.tolist() for obj in result.objects] == [
        list(range(4)),
        list(range(4, 8)),
    ]


def test_containment_dedup_keeps_higher_quality_candidate() -> None:
    low = ConsensusObject(0, (0, 1), (0, 1), np.arange(4), 0.8, 0.8, 0.8)
    high = ConsensusObject(1, (2, 3), (2, 3), np.arange(5), 0.9, 0.9, 0.9)

    result = remove_contained_objects((low, high))

    assert len(result) == 1
    assert result[0].mask_ids == (2, 3)


def test_merge_veto_changes_only_acceptance_not_pair_evidence() -> None:
    ids = np.arange(6)
    observations = (
        _observation(0, 0, ids),
        _observation(1, 1, ids),
    )
    visibility = np.ones((2, 6), dtype=bool)
    xyz = _line_points(6)
    calls: list[tuple[tuple[int, ...], list[int]]] = []

    def reject(mask_ids: tuple[int, ...], gaussian_ids: np.ndarray) -> bool:
        calls.append((mask_ids, gaussian_ids.tolist()))
        return False

    evidence = compute_pair_consensus((0,), (1,), observations, visibility)
    result = run_mask_consensus(
        observations, visibility, xyz, merge_veto=reject
    )

    assert evidence.consensus == 1.0
    assert calls == [((0, 1), ids.tolist())]
    assert result.accepted_edges == ()
    assert result.objects == ()


def test_sparse_pair_index_skips_pairs_without_any_possible_supporter() -> None:
    observations = (
        _observation(0, 0, np.arange(4)),
        _observation(1, 1, np.arange(4)),
        _observation(2, 2, np.arange(4, 8)),
        _observation(3, 3, np.arange(8, 12)),
    )
    visibility = np.ones((4, 12), dtype=bool)
    xyz = _line_points(12)

    result = run_mask_consensus(observations, visibility, xyz)

    assert result.diagnostics["total_possible_cross_frame_pair_count"] == 6
    assert result.diagnostics["sparse_candidate_pair_count"] == 1
    assert result.diagnostics["pair_evaluation_count"] == 1


def test_sparse_index_keeps_disjoint_pair_when_a_third_mask_supports_both() -> None:
    left_ids = np.arange(4)
    right_ids = np.arange(4, 8)
    observations = (
        _observation(0, 0, left_ids),
        _observation(1, 1, right_ids),
        _observation(2, 2, np.arange(8)),
    )
    visibility = np.zeros((3, 8), dtype=bool)
    visibility[0, left_ids] = True
    visibility[1, right_ids] = True
    visibility[2, :] = True

    evidence = compute_pair_consensus(
        (0,), (1,), observations, visibility
    )
    result = run_mask_consensus(observations, visibility, _line_points(8))

    assert evidence.observer_count == 1
    assert evidence.supporter_count == 1
    assert evidence.consensus == 1.0
    assert result.diagnostics["sparse_candidate_pair_count"] == 3


def test_pca_extents_are_rotation_translation_invariant_and_sorted() -> None:
    box = np.asarray(
        [[x, y, z] for x in (-2.0, 2.0) for y in (-1.0, 1.0) for z in (-0.5, 0.5)]
    )
    theta = np.deg2rad(37.0)
    rotation = np.asarray(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    original = pca_sorted_extents_m(box)
    transformed = pca_sorted_extents_m(box @ rotation.T + 17.0)

    assert np.allclose(original, [1.0, 2.0, 4.0])
    assert np.allclose(transformed, original)


def test_size_q95_modes_and_missing_class_global_fallback() -> None:
    priors = SizePriorTable(
        global_log_q95=tuple(np.log([2.0, 2.0, 6.0])),
        class_log_q95={
            "cup": tuple(np.log([0.5, 0.5, 1.0])),
            "bed": tuple(np.log([2.0, 2.0, 6.0])),
        },
    )
    extents = np.asarray([1.0, 1.0, 5.0])

    assert global_size_compatibility(extents, priors) == 1.0
    assert predicted_size_compatibility(
        extents, priors, {"cup": 0.8, "bed": 0.2}
    ) == pytest.approx(0.2)
    assert predicted_size_compatibility(extents, priors, {"unknown": 1.0}) == 1.0
    assert oracle_class_size_compatibility(extents, priors, "cup") == 0.0
    assert oracle_class_size_compatibility(extents, priors, "bed") == 1.0


def test_size_merge_veto_supports_c0_u_d_but_not_oracle_runtime() -> None:
    xyz = np.asarray(
        [[x, y, z] for x in (0.0, 5.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
    )
    priors = SizePriorTable(
        global_log_q95=tuple(np.log([2.0, 2.0, 6.0])),
        class_log_q95={
            "cup": tuple(np.log([0.5, 0.5, 1.0])),
            "bed": tuple(np.log([2.0, 2.0, 6.0])),
        },
    )
    ids = np.arange(len(xyz))
    c0 = make_size_merge_veto("none", xyz, priors)
    uniform = make_size_merge_veto("global", xyz, priors)
    predicted = make_size_merge_veto(
        "predicted",
        xyz,
        priors,
        {0: {"cup": 0.8, "bed": 0.2}, 1: {"cup": 0.8, "bed": 0.2}},
    )

    assert c0((0, 1), ids)
    assert uniform((0, 1), ids)
    assert not predicted((0, 1), ids)
    with pytest.raises(ValueError, match="none.*global.*predicted"):
        make_size_merge_veto("oracle", xyz, priors)  # type: ignore[arg-type]


def test_prior_table_reads_train_only_repository_schema() -> None:
    payload = _signed_train_prior({
        "global": {
            "raw": {
                "geometry": {
                    "log_extent_short_m": {"q95": 91.0},
                    "log_extent_mid_m": {"q95": 92.0},
                    "log_extent_long_m": {"q95": 93.0},
                }
            },
            "shrunk": {
                "geometry": {
                    "log_extent_short_m": {"q95": 1.0},
                    "log_extent_mid_m": {"q95": 2.0},
                    "log_extent_long_m": {"q95": 3.0},
                }
            },
        },
        "categories": {
            "chair": {
                "shrunk": {
                    "geometry": {
                        "log_extent_short_m": {"q95": 0.1},
                        "log_extent_mid_m": {"q95": 0.2},
                        "log_extent_long_m": {"q95": 0.3},
                    }
                }
            }
        },
    })

    result = SizePriorTable.from_category_priors(payload)

    assert result.global_log_q95 == (1.0, 2.0, 3.0)
    assert result.class_log_q95["chair"] == (0.1, 0.2, 0.3)


def test_prior_table_refuses_to_silently_mix_global_raw_statistics() -> None:
    payload = _signed_train_prior({
        "global": {
            "raw": {
                "geometry": {
                    "log_extent_short_m": {"q95": 1.0},
                    "log_extent_mid_m": {"q95": 2.0},
                    "log_extent_long_m": {"q95": 3.0},
                }
            }
        },
        "categories": {},
    })

    with pytest.raises(ValueError, match="global prior is missing shrunk"):
        SizePriorTable.from_category_priors(payload)


def test_ap_score_is_prior_independent_and_runtime_signature_has_no_gt() -> None:
    assert base_ap_score(0.8, 0.81, 1.0) == pytest.approx(0.72)
    signature = inspect.signature(run_mask_consensus)
    lowered = " ".join(signature.parameters).lower()
    assert "gt" not in lowered
    assert "oracle" not in lowered


def test_default_config_matches_registered_clean_baseline_values() -> None:
    config = ConsensusConfig()
    assert config.mask_visible_threshold == 0.30
    assert config.undersegment_filter_threshold == 0.30
    assert config.view_consensus_threshold == 0.90
    assert config.contained_threshold == 0.80
    assert config.point_filter_threshold == 0.50
    assert config.dbscan_eps_m == 0.10
    assert config.dbscan_min_samples == 4
    assert config.min_views == 2

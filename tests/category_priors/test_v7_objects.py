from __future__ import annotations

import numpy as np

from category_priors.v7_evaluation import _opacity_precision
from category_priors.v7_objects import (
    CoreAssignment,
    Fragment,
    FrameEvidence,
    Track,
    V7Config,
    associate_fragments,
    attach_local_labels,
    attach_unique_halo,
    build_consensus_core,
    lift_frame,
    materialize_instances,
)
from category_priors.v7_replay import core_compatibility, score_candidate, size_compatibility


def _fragment(
    fragment_id: int,
    frame_id: int,
    core: list[int],
    full: list[int] | None = None,
    class_id: int = 0,
    mask_index: int = 0,
) -> Fragment:
    full_ids = np.asarray(full or core, dtype=np.int32)
    core_ids = np.asarray(core, dtype=np.int32)
    return Fragment(
        fragment_id=fragment_id,
        frame_id=frame_id,
        mask_index=mask_index,
        class_id=class_id,
        full_ids=full_ids,
        core_ids=core_ids,
        full_support=np.ones(len(full_ids), dtype=np.float32),
        core_support=np.ones(len(core_ids), dtype=np.float32),
    )


def test_lift_frame_filters_invalid_pixels_and_accumulates_repeated_ids() -> None:
    ids = np.array([[0, 0, -1, 1], [0, 2, 2, 3]])
    weights = np.array([[0.4, 0.5, 0.0, 0.0], [0.6, 0.3, 0.4, 0.2]])
    masks = np.ones((1, 2, 4), dtype=bool)
    config = V7Config(fragment_min_core=1, fragment_min_full=1)
    frame = lift_frame(ids, weights, masks, [4], 7, 5, config=config)
    assert frame.visible_ids.tolist() == [0, 2, 3]
    assert len(frame.fragments) == 1
    fragment = frame.fragments[0]
    assert fragment.full_ids.tolist() == [0, 2, 3]
    assert np.isclose(fragment.full_support[0], 1.5)
    assert 1 not in fragment.full_ids


def test_zero_mask_frame_is_background_evidence() -> None:
    ids = np.array([[0, 1], [0, -1]])
    weights = np.array([[0.4, 0.2], [0.3, 0.0]])
    frame = lift_frame(ids, weights, np.zeros((0, 2, 2), bool), [], 0, 2)
    assert frame.fragments == ()
    assert frame.visible_ids.tolist() == [0, 1]
    assert frame.background_ids.tolist() == [0, 1]


def test_association_ignores_semantics_and_never_uses_same_frame_twice() -> None:
    first = _fragment(0, 0, list(range(10)), class_id=1, mask_index=0)
    same_frame = _fragment(1, 0, list(range(10, 20)), class_id=1, mask_index=1)
    next_frame = _fragment(2, 1, list(range(10)), class_id=7)
    tracks = associate_fragments([first, same_frame, next_frame])
    assert len(tracks) == 2
    assert tracks[0].fragment_ids == [0, 2]
    assert tracks[1].fragment_ids == [1]


def test_ambiguous_bridge_starts_new_track() -> None:
    left = _fragment(0, 0, list(range(0, 10)), mask_index=0)
    right = _fragment(1, 0, list(range(10, 20)), mask_index=1)
    bridge = _fragment(2, 1, list(range(0, 5)) + list(range(10, 15)))
    tracks = associate_fragments([left, right, bridge])
    assert len(tracks) == 3
    assert tracks[2].fragment_ids == [2]


def test_consensus_core_is_unique_and_requires_multiview_support() -> None:
    fragments = [
        _fragment(0, 0, list(range(12))),
        _fragment(1, 1, list(range(12))),
        _fragment(2, 0, list(range(6, 18)), mask_index=1),
        _fragment(3, 1, list(range(6, 18)), mask_index=1),
    ]
    tracks = [
        Track(0, [0, 1], {0, 1}, set(range(12)), set(range(12)), [1.0]),
        Track(1, [2, 3], {0, 1}, set(range(6, 18)), set(range(6, 18)), [1.0]),
    ]
    frames = [
        FrameEvidence(0, (fragments[0], fragments[2]), np.arange(18), np.array([], int)),
        FrameEvidence(1, (fragments[1], fragments[3]), np.arange(18), np.array([], int)),
    ]
    config = V7Config(core_min_points=3)
    core = build_consensus_core(tracks, fragments, frames, 18, config)
    assert np.all(core.core_track_id[:6] == 0)
    assert np.all(core.core_track_id[6:12] == -1)
    assert np.all(core.core_track_id[12:] == 1)


def test_halo_is_one_shot_and_requires_unique_affinity() -> None:
    xyz = np.array([[i * 0.005, 0.0, 0.0] for i in range(8)], dtype=float)
    affinity = np.tile(np.array([[1.0, 0.0]]), (8, 1))
    track = Track(0, [0, 1], {0, 1}, set(range(6)), set(range(8)), [1.0])
    core = CoreAssignment(
        core_track_id=np.array([0, 0, 0, 0, 0, 0, -1, -1], dtype=np.int32),
        positive_views=np.array([2] * 6 + [0, 0], dtype=np.int16),
        visible_views=np.array([2] * 8, dtype=np.int16),
        background_views=np.zeros(8, dtype=np.int16),
        conflict_views=np.zeros(8, dtype=np.int16),
        assignment_margin=np.ones(8, dtype=np.float32),
        valid_track_ids=(0,),
        track_positive={0: np.array([2] * 6 + [1, 0], dtype=np.int16)},
        track_conflict={0: np.zeros(8, dtype=np.int16)},
    )
    result = attach_unique_halo(xyz, affinity, [track], core)
    assert result[6] == 0
    assert result[7] == -1


def test_legacy_local_attach_is_one_shot_and_does_not_bridge() -> None:
    xyz = np.column_stack((np.arange(7) * 0.01, np.zeros((7, 2))))
    affinity = np.tile(np.array([[1.0, 0.0]]), (7, 1))
    labels = np.array([0, 0, 0, -1, -1, -1, -1], dtype=np.int32)
    result = attach_local_labels(xyz, affinity, labels, radius_m=0.035)
    assert result[3] == 0
    assert result[4] == -1


def test_semantics_are_attached_only_after_track_construction() -> None:
    fragments = [
        _fragment(0, 0, list(range(10)), class_id=0),
        _fragment(1, 1, list(range(10)), class_id=0),
    ]
    track = Track(0, [0, 1], {0, 1}, set(range(10)), set(range(10)), [1.0])
    core = CoreAssignment(
        core_track_id=np.zeros(10, dtype=np.int32),
        positive_views=np.full(10, 2, dtype=np.int16),
        visible_views=np.full(10, 2, dtype=np.int16),
        background_views=np.zeros(10, dtype=np.int16),
        conflict_views=np.zeros(10, dtype=np.int16),
        assignment_margin=np.ones(10, dtype=np.float32),
        valid_track_ids=(0,),
        track_positive={0: np.full(10, 2, dtype=np.int16)},
        track_conflict={0: np.zeros(10, dtype=np.int16)},
    )
    labels, candidates = materialize_instances(
        np.column_stack((np.arange(10) * 0.01, np.zeros((10, 2)))),
        [track], fragments, core, np.zeros(10, dtype=np.int32),
        ["chair", "wall"],
    )
    assert np.all(labels == 0)
    assert candidates[0]["branch_class"] == "chair"


def test_priors_change_only_scores_not_candidate_structure() -> None:
    candidate = {
        "candidate_id": 0,
        "branch_class": "chair",
        "base_score": 0.8,
        "metric_extents_m": [0.5, 0.7, 1.0],
        "local_surface_density": 100.0,
        "core_point_count": 20,
    }
    global_geometry = {
        "log_extent_short_m": {"q50": np.log(0.5), "q75": np.log(0.7)},
        "log_extent_mid_m": {"q50": np.log(0.7), "q75": np.log(0.9)},
        "log_extent_long_m": {"q50": np.log(1.0), "q75": np.log(1.2)},
        "log_surface_area_m2": {"q50": np.log(2.0)},
    }
    class_geometry = {
        "log_extent_short_m": {"q50": np.log(0.1), "q75": np.log(0.2)},
        "log_extent_mid_m": {"q50": np.log(0.2), "q75": np.log(0.3)},
        "log_extent_long_m": {"q50": np.log(0.3), "q75": np.log(0.4)},
        "log_surface_area_m2": {"q50": np.log(0.1)},
    }
    priors = {
        "global": {"shrunk": {"geometry": global_geometry}},
        "categories": {"chair": {"shrunk": {"geometry": class_geometry}}},
    }
    uniform = score_candidate(candidate, priors, "U00-uniform")
    data = score_candidate(candidate, priors, "D11-combined")
    assert uniform["Q"] == data["Q"] == 0.8
    assert data["G"] < uniform["G"]
    assert size_compatibility(candidate, priors["global"]) == uniform["G"]
    assert core_compatibility(candidate, priors["global"]) == uniform["C"]


def test_opacity_precision_keeps_unsupported_weight_in_denominator() -> None:
    audit = {
        "instances": [{"instance_id": 3}],
        "point_categories": {3: np.array([0, 3], dtype=np.int8)},
    }
    labels = np.array([3, 3, -1], dtype=np.int64)
    correct, total, macro = _opacity_precision(
        audit, labels, np.array([0.75, 0.25, 1.0], dtype=np.float64)
    )
    assert correct == 0.75
    assert total == 1.0
    assert macro == 0.75

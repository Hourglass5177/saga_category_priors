from __future__ import annotations

import numpy as np
from dataclasses import replace
import inspect
from pathlib import Path
from scipy.spatial import cKDTree

from category_priors.iterative_refinement.contracts import (
    CandidateSeed,
    GaussianEvidence,
    MaskHypothesis,
    ObjectState,
    RefinementConfig,
    ViewObservation,
)
from category_priors.iterative_refinement.evidence import (
    AlphaMass,
    aggregate_gaussian_evidence,
    hypothesis_gaussian_evidence,
)
from category_priors.iterative_refinement.local_refine import (
    SizePrior,
    binary_graph_cut,
    fuse_objects_with_b0,
    local_roi_point_ids,
    trim_oversize_additions,
)
from category_priors.iterative_refinement.objects import combine_states, merge_objects_once
from category_priors.iterative_refinement.pipeline import _review_round, candidate_export_contract
from category_priors.iterative_refinement.reviewer import (
    dispersed_prompt_points,
    rank_detection_proposals,
)
from category_priors.iterative_refinement.rendering import expected_max_contributor_package
from category_priors.iterative_refinement.alpha_backend import (
    AlphaEvidenceCache,
    expected_fused_package,
    mask_sha256,
    pack_mask_bits,
)
from category_priors.iterative_refinement.runtime_io import save_scene_cache
from category_priors.iterative_refinement.views import (
    CropSpec,
    crop_box_to_image,
    observations_are_independent,
    pack_mask,
    select_diverse_views,
)


def _hypothesis(mask: np.ndarray, camera: int, identifier: str) -> MaskHypothesis:
    packed, shape = pack_mask(mask)
    return MaskHypothesis(
        identifier, 0, 1, camera, f"frame-{camera}", "tight", "chair", .8, .9,
        (0., 0., float(mask.shape[1]), float(mask.shape[0])), .8, .4,
        int(mask.sum()), packed, shape, camera,
    )


def test_binary_graph_cut_respects_hard_foreground_and_background() -> None:
    foreground, _ = binary_graph_cut(
        positive=np.array([0., 0., 0.]),
        negative=np.array([0., 0., 0.]),
        hard_positive=np.array([True, False, False]),
        hard_negative=np.array([False, False, True]),
        edges=np.array([[0, 1], [1, 2]]),
        edge_weights=np.array([2., 2.]),
    )
    assert foreground[0]
    assert not foreground[2]


def test_size_trim_protects_anchor_but_can_remove_polluted_support() -> None:
    xyz = np.array([[0., 0., 0.], [.01, 0., 0.], [2., 0., 0.]])
    selected, removed = trim_oversize_additions(
        np.array([0, 1, 2]), np.array([0]), np.array([1]), np.array([2., 1., -2.]),
        xyz, (.1, .1, .1),
    )
    assert selected.tolist() == [0, 1]
    assert removed == 1


def test_full_image_detection_ranking_penalizes_giant_box() -> None:
    seed = np.zeros((20, 20), bool)
    seed[8:12, 8:12] = True
    spec = CropSpec("tight", 20, 10, 20, 20., False, 0.)
    rows = rank_detection_proposals(
        boxes_xyxy=np.array([[0, 0, 20, 20], [7, 7, 13, 13]]),
        scores=np.array([.9, .7]), class_ids=np.array([0, 0], dtype=object),
        classes=("chair",), seed_mask=seed, crop_spec=spec,
    )
    assert rows[0].box_crop_xyxy == (7., 7., 13., 13.)
    assert rows[0].box_image_xyxy == (17., 27., 23., 33.)


def test_prompt_points_are_deterministic_and_spread() -> None:
    mask = np.ones((10, 10), bool)
    first = dispersed_prompt_points(mask)
    second = dispersed_prompt_points(mask)
    assert np.array_equal(first, second)
    assert len(first) == 4
    assert np.linalg.norm(first[0] - first[1]) > 5


def test_view_selection_records_non_independent_repeats() -> None:
    def row(index: int, center: tuple[float, float, float], quality: float) -> ViewObservation:
        return ViewObservation(0, index, str(index), 10, (0, 0, 2, 2), (1., 1.), quality, 1., (0., 0., 1.), center)
    a, b, c = row(0, (0., 0., 0.), 3.), row(1, (.01, 0., 0.), 2.), row(2, (1., 0., 0.), 1.)
    assert not observations_are_independent(a, b)
    assert observations_are_independent(a, c)
    selected = select_diverse_views((a, b, c))
    assert [item.camera_index for item in selected[:2]] == [0, 2]


def test_hard_and_soft_evidence_roles_are_separate() -> None:
    mask = np.array([[True, False], [False, False]])
    hypothesis = _hypothesis(mask, 0, "h0")
    mass = AlphaMass(np.array([[1., 0., 0.]]), np.array([1., 1., 1.]), 4)
    row = hypothesis_gaussian_evidence(
        hypothesis,
        contributor_ids=np.array([[0, 1], [1, 2]]),
        max_weights=np.ones((2, 2)), opacity=np.ones((2, 2)),
        alpha_mass=mass, alpha_mask_index=0, point_count=3,
    )
    assert row.positive_ids.tolist() == [0]
    assert row.negative_ids.tolist() == [1]
    assert row.soft_ids.tolist() == [0]


def test_single_view_never_becomes_two_view_hard_support() -> None:
    mask = np.array([[True]])
    hypothesis = _hypothesis(mask, 0, "h0")
    row = type("Row", (), {
        "hypothesis_id": "h0", "camera_index": 0,
        "positive_ids": np.array([2]), "negative_ids": np.array([], dtype=int),
        "soft_ids": np.array([2]), "soft_ratios": np.array([1.]),
    })()
    evidence = aggregate_gaussian_evidence(0, (hypothesis,), {"h0": row})
    assert evidence.hard_positive_views.tolist() == [1.]


def test_combine_states_does_not_sum_duplicate_view_evidence() -> None:
    left = ObjectState(0, (0,), np.array([0, 1]), np.array([1]), np.array([0]), np.array([2., 0.]), np.array([2., 0.]), "chair", True, 1, True)
    right = ObjectState(1, (1,), np.array([0, 2]), np.array([2]), np.array([0]), np.array([2., 0.]), np.array([1., 0.]), "chair", True, 1, True)
    merged = combine_states(left, right, round_index=1)
    assert merged.hard_positive_counts.tolist() == [2., 0., 0.]


def test_b0_fusion_needs_two_view_hard_support_to_take_owned_point() -> None:
    b0 = np.array([0, 0, -1, -1])
    xyz = np.array([[0., 0., 0.], [.01, 0., 0.], [.02, 0., 0.], [.03, 0., 0.]])
    weak = ObjectState(4, (4,), np.array([0, 2, 3]), np.array([0]), np.array([], dtype=int), np.array([1., 1., 1.]), np.array([2., 2., 2.]), "chair", True, 1, True)
    result = fuse_objects_with_b0(b0, (weak,), xyz)
    assert result.rejected_object_ids == (4,)
    assert np.array_equal(result.labels, b0)


def test_empty_refinement_is_exact_b0_identity() -> None:
    b0 = np.array([0, -1, 1, 1])
    xyz = np.arange(12, dtype=float).reshape(4, 3) / 100
    result = fuse_objects_with_b0(b0, (), xyz)
    assert np.array_equal(result.labels, b0)
    assert result.accepted_object_ids == ()


def test_shared_scene_tree_preserves_local_roi_exactly() -> None:
    xyz = np.array([[0., 0., 0.], [.01, 0., 0.], [.20, 0., 0.]])
    seed = CandidateSeed(0, (0,), "chair", np.array([0]), np.array([0]), "post_filter", .5)
    evidence = GaussianEvidence(0, np.array([1]), np.array([2.]), np.array([0.]), np.array([1.]), 2, 0, ())
    prior = SizePrior(.2, (1., 1., 1.), "global")
    direct = local_roi_point_ids(xyz, seed, evidence, prior)
    shared = local_roi_point_ids(xyz, seed, evidence, prior, scene_tree=cKDTree(xyz))
    assert np.array_equal(shared, direct)


def test_frozen_membership_can_be_reused_in_next_round() -> None:
    seed = CandidateSeed(0, (0,), "chair", np.array([2, 1]), np.array([1]), "post_filter", .5)
    reused = replace(seed, seed_support=seed.seed_support, seed_anchor=seed.seed_anchor)
    assert reused.seed_support.tolist() == [1, 2]


def test_merge_is_one_pass_and_does_not_close_a_weak_chain() -> None:
    xyz = np.array([[0., 0., 0.], [.01, 0., 0.], [.02, 0., 0.]])
    states = tuple(
        ObjectState(index, (index,), np.array([index]), np.array([index]), np.array([index]),
                    np.array([2.]), np.array([2.]), "chair", True, 1, True)
        for index in range(3)
    )
    masks = {}
    for index in range(3):
        masks[index] = (_hypothesis(np.ones((2, 2), bool), 0, f"{index}a"),
                        _hypothesis(np.ones((2, 2), bool), 1, f"{index}b"))
    evidence = {index: GaussianEvidence(index, np.array([index]), np.array([2.]), np.array([0.]), np.array([1.]), 2, 0, ()) for index in range(3)}
    prior = {index: SizePrior(1., (1., 1., 1.), "global") for index in range(3)}
    merged, _ = merge_objects_once(
        states, hypotheses=masks, evidence=evidence,
        independent_pairs={(0, 1): True}, xyz_m=xyz,
        prior_by_object=prior, round_index=1,
    )
    assert len(merged) == 2


def test_merged_export_has_one_evaluator_source_and_lossless_lineage() -> None:
    state = ObjectState(
        7, (9, 3), np.array([0, 1, 2]), np.array([0]), np.array([0]),
        np.array([2., 2., 2.]), np.array([1., 1., 1.]), "chair", True, 2, True,
    )
    evaluator, lineage = candidate_export_contract({7: 42}, {42: 5}, {7: state})
    assert evaluator == {"3": 5}
    assert lineage == {"5": [3, 9]}


def test_runtime_does_not_import_gt_or_legacy_postprocessing() -> None:
    source = (Path(__file__).parents[2] / "category_priors" / "iterative_refinement" / "pipeline.py").read_text(encoding="utf-8")
    assert "ground_truth" not in source
    assert "legacy_candidate_replay" not in source
    assert "postprocess" not in source
    assert "hdbscan" not in source.lower()


def test_sparse_cache_round_trip_is_json_and_npz_only(tmp_path) -> None:
    mask = _hypothesis(np.array([[True, False]]), 0, "h")
    evidence = GaussianEvidence(0, np.array([1]), np.array([2.]), np.array([0.]), np.array([.7]), 2, 0, ("h",))
    state = ObjectState(0, (0,), np.array([1]), np.array([1]), np.array([1]), np.array([2.]), np.array([1.]), "chair", True, 2, True)
    save_scene_cache(
        tmp_path, hypotheses=(mask,), evidence={0: evidence},
        states_by_profile={"balanced": (state,)}, lineage=(), provenance={"commit": "abc"},
    )
    assert (tmp_path / "refinement_cache.json").is_file()
    with np.load(tmp_path / "refinement_cache.npz", allow_pickle=False) as archive:
        assert archive["mask_0"].tolist() == mask.packed_mask.tolist()
        assert archive["s_balanced_0_anchors"].tolist() == [1]


def test_max_contributor_producer_can_be_pinned_for_read_only_reuse(tmp_path, monkeypatch) -> None:
    producer = tmp_path / "producer" / "diff_gaussian_rasterization_max_contributor"
    monkeypatch.setenv("SAGA_MAX_CONTRIBUTOR_PACKAGE_ROOT", str(producer))
    assert expected_max_contributor_package() == producer.resolve()


def test_review_round_builds_seed_index_before_class_prior_lookup() -> None:
    source = inspect.getsource(_review_round)
    assignment = source.index("seed_by_id =")
    lookup = source.index("seed_by_id[row.candidate_id]")
    assert assignment < lookup


def test_mask_bit_packing_supports_32_and_deterministic_33_chunk_boundary() -> None:
    masks = np.zeros((33, 2, 2), dtype=bool)
    for index in range(33):
        masks[index, index % 2, (index // 2) % 2] = True
    first = pack_mask_bits(masks[:32])
    second = pack_mask_bits(masks[32:])
    assert first.dtype == np.int32
    assert first.view(np.uint32)[0, 0] & np.uint32(1)
    assert second.view(np.uint32)[0, 0] == 1


def test_mask_hash_is_content_and_shape_sensitive() -> None:
    a = np.array([[True, False], [False, True]])
    b = a.reshape(1, 4)
    assert mask_sha256(a) == mask_sha256(a.copy())
    assert mask_sha256(a) != mask_sha256(b)


def test_sparse_alpha_cache_rejects_corruption_and_wrong_identity(tmp_path) -> None:
    path = tmp_path / "mass.npz"
    values = np.array([0.0, 1.25, 0.0, 2.5])
    AlphaEvidenceCache._save_sparse(path, values, "right")
    loaded = AlphaEvidenceCache._load_sparse(path, 4, "right")
    assert loaded is not None and np.allclose(loaded, values)
    assert AlphaEvidenceCache._load_sparse(path, 4, "wrong") is None
    path.write_bytes(b"not an npz")
    assert AlphaEvidenceCache._load_sparse(path, 4, "right") is None


def test_fused_backend_producer_can_be_pinned(tmp_path, monkeypatch) -> None:
    producer = tmp_path / "producer" / "diff_gaussian_rasterization_alpha_mass"
    monkeypatch.setenv("SAGA_ALPHA_MASS_PACKAGE_ROOT", str(producer))
    assert expected_fused_package() == producer.resolve()

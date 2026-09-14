from __future__ import annotations

import inspect
import re
from dataclasses import replace

import numpy as np
import pytest

from category_priors.object_verification.observation import (
    CameraView, MaskObservation, SemanticObservation, cameras_independent,
    fixed_view_panel, identity_consensus, identity_pair, reliable_contributors,
    semantic_decision, verify_projection,
)
from category_priors.object_verification.model_adapter import (
    CropTransform, FrozenLocator, InjectedModelAdapter, class_caption,
    class_token_spans, decode_dino_queries, prior_crop, reliable_prompt_point,
)


CLASSES = ("cup", "table", "lamp", "phone") + tuple(f"class{i}" for i in range(28))


def camera(index):
    return CameraView(str(index), (0., 0., 1.), (index * .1, 0., 0.), 1.)


def observation(uid, index, positives, *, label="cup", score=.5, opacity=1.):
    ids = np.arange(12, dtype=np.int64).reshape(3, 4)
    return MaskObservation(uid, camera(index), np.isin(ids, positives), ids,
                           np.ones((3, 4)), np.full((3, 4), opacity),
                           np.ones((3, 4), dtype=bool), score, label)


def raw_queries(class_scores_per_query):
    caption, _ = class_caption(CLASSES)
    offsets = [(0, 0)] + [m.span() for m in re.finditer(r"\w+|\.", caption)] + [(0, 0)]
    special = [True] + [False] * (len(offsets) - 2) + [True]
    spans = class_token_spans(CLASSES, np.array(offsets), special)
    logits = np.full((len(class_scores_per_query), len(offsets)), -10.)
    for index, scores in enumerate(class_scores_per_query):
        for name, score in scores.items():
            logits[index, list(spans[CLASSES.index(name)])] = np.log(score / (1 - score))
    return dict(raw_logits=logits, boxes_crop_xyxy=np.tile([0., 0., 4., 4.], (len(logits), 1)),
                offset_mapping=np.asarray(offsets), special_tokens_mask=np.asarray(special))


class MockSam:
    def __init__(self, outputs=None):
        self.inputs = []
        self.prompts = []
        self.outputs = list(outputs or [])

    def set_image(self, image):
        self.inputs.append(image.copy())

    def predict(self, **kwargs):
        self.prompts.append(kwargs)
        assert kwargs["multimask_output"] is True
        masks = self.outputs.pop(0) if self.outputs else np.ones((3,) + self.inputs[-1].shape[:2], dtype=bool)
        return masks, np.array([.9, .8, .7]), None


def test_geometry_independent_from_class_and_sam_rank_between_identities():
    a = observation("a", 0, [0, 1, 2, 3], label="cup", score=.01)
    b = observation("b", 1, [0, 1, 2, 3], label="table", score=.99)
    result = identity_consensus([a, b])
    assert result.status == "accepted" and result.groups == (("a", "b"),)
    c = observation("c", 0, [8, 9, 10, 11], label="cup", score=1.)
    d = observation("d", 1, [8, 9, 10, 11], label="cup", score=1.)
    ambiguous = identity_consensus([a, b, c, d])
    assert ambiguous.status == "unknown"
    assert set(ambiguous.groups) == {("a", "b"), ("c", "d")}


def test_three_nested_boundaries_each_camera_are_one_identity_but_all_saved():
    rows = [observation(f"{cam}-{i}", cam, list(range(4 + i)), score=.6 + i * .1)
            for cam in (0, 1) for i in range(3)]
    result = identity_consensus(rows)
    assert result.status == "accepted"
    assert result.groups == (("0-2", "1-2"),)
    assert len(result.alternative_groups) == 9


def test_identity_is_not_transitive_closure():
    a = observation("a", 0, [0, 1, 2, 3, 4, 5])
    b = observation("b", 1, [2, 3, 4, 5, 6, 7])
    c = observation("c", 2, [4, 5, 6, 7, 8, 9])
    assert identity_pair(a, b)["compatible"] and identity_pair(b, c)["compatible"]
    assert not identity_pair(a, c)["compatible"]
    assert identity_consensus([a, b, c]).status == "unknown"


def test_common_visibility_not_full_support_and_no_soft_alpha():
    a = observation("a", 0, range(12))
    b = observation("b", 1, [0, 1, 2])
    valid = np.zeros((3, 4), dtype=bool)
    valid.flat[:3] = True
    b = MaskObservation(b.observation_uid, b.camera, b.mask, b.contributor_ids,
                        b.max_contribution, b.opacity, valid)
    pair = identity_pair(a, b)
    assert pair["compatible"] and pair["common_visible"] == 3 and pair["jaccard"] == 1.
    invisible = observation("invisible", 2, range(12), opacity=.49)
    assert not reliable_contributors(invisible)[0]
    assert identity_pair(a, invisible)["reason"] == "unknown_common_visibility"


def test_threshold_boundaries_and_duplicate_camera():
    first = observation("first", 0, [0, 1, 2])
    second = observation("second", 1, [0, 1, 2, 3, 4, 5])
    assert identity_pair(first, second)["compatible"]  # exactly .50 bidirectional coverage.
    assert not identity_pair(first, observation("too_few", 1, [0, 1]))["compatible"]
    assert not identity_pair(first, observation("same_camera", 0, [0, 1, 2]))["compatible"]
    assert not cameras_independent(camera(0), CameraView("near", (0, 0, 1), (.01, 0, 0), 1.))


@pytest.mark.parametrize("count,roles", [(0, []), (1, ["I1"]), (2, ["I1", "I2"]),
    (3, ["I1", "I2", "Hfinal"]), (4, ["I1", "I2", "I3", "Hfinal"]),
    (5, ["I1", "I2", "Hmid", "I3", "Hfinal"]), (6, ["I1", "I2", "Hmid", "I3", "I4", "Hfinal"])])
def test_fixed_small_view_panels(count, roles):
    panel = fixed_view_panel([camera(i) for i in range(count)])
    assert [r for r, _ in panel.assignments] == roles
    assert panel.feedback_applicable == (count >= 5)
    assert panel.can_verify_new_object == (count >= 3)


def test_panel_retains_all_ancestry_and_never_uses_source_as_holdout():
    panel = fixed_view_panel([camera(i) for i in range(9)], source_camera_ids=["2", "5", "external"])
    assert len(panel.assignments) == 6
    assert all(v.camera_uid not in panel.source_camera_ids for r, v in panel.assignments if r.startswith("H"))
    assert panel.source_camera_ids == ("2", "5", "external")
    all_sources = fixed_view_panel([camera(i) for i in range(9)], source_camera_ids=[str(i) for i in range(9)])
    assert len(all_sources.assignments) == 2 and not all_sources.can_verify_new_object
    with pytest.raises(ValueError, match="duplicate"):
        fixed_view_panel([camera(0), camera(0)])
    with pytest.raises(ValueError, match="source camera"):
        replace(panel, source_camera_ids=(panel.camera_for("Hfinal").camera_uid,))
    with pytest.raises(ValueError, match="role table"):
        replace(panel, feedback_applicable=False)


def test_observation_state_cannot_be_mutated_or_change_source_arrays():
    row = observation("immutable", 0, [0, 1, 2])
    with pytest.raises(ValueError):
        row.mask.setflags(write=True)
    semantic = semantic_row("immutable-sem", 0, {"cup": .8})
    with pytest.raises(TypeError):
        semantic.class_scores["cup"] = .1


def semantic_row(uid, index, scores, role="construction", **kwargs):
    return SemanticObservation(uid, camera(index), role, scores, **kwargs)


def decide(rows):
    return semantic_decision(rows, classes32=CLASSES, saga20=CLASSES[:20])


def test_semantic_denominator_and_excluded_roles_are_explicit():
    rows = [semantic_row("i1", 0, {"cup": .8, "table": .4}), semantic_row("i2", 1, {"cup": .6}),
            semantic_row("i3", 2, {}, unknown_reason="no_detection"),
            semantic_row("h", 3, {"table": .99}, "online_verification"),
            semantic_row("pre", 0, {"table": .99}, "prepass")]
    result = decide(rows)
    assert result["final_class"] == "cup" and result["score"] == .7
    assert result["N"] == 2 and result["scores"]["table"] == .2
    assert result["construction_camera_count"] == 3 and result["unknown_reasons"] == {"no_detection": 1}
    assert result["excluded_observation_uids"] == ["h", "pre"]


def test_semantic_null_tie_out_of_scope_and_single_view():
    result = decide([semantic_row("empty", 0, {})])
    assert result["N"] == 0 and result["score"] is None and all(v is None for v in result["scores"].values())
    assert decide([semantic_row("one", 0, {"cup": .99})])["status"] == "unknown"
    tie = decide([semantic_row(str(i), i, {"cup": .8, "table": .8}) for i in range(2)])
    assert tie["reason"] == "exact_class_score_tie"
    outside = decide([semantic_row(str(i), i, {CLASSES[-1]: .9, "cup": .8}) for i in range(2)])
    assert outside["status"] == "out_of_scope" and outside["final_class"] == CLASSES[-1]


def test_semantic_duplicate_camera_and_nonindependent_do_not_create_two_votes():
    with pytest.raises(ValueError, match="one adopted"):
        decide([semantic_row("a", 0, {"cup": .8}), semantic_row("b", 0, {"cup": .8})])
    with pytest.raises(ValueError, match="independent"):
        decide([semantic_row("a", 0, {"cup": .8}), SemanticObservation("b", CameraView("near", (0, 0, 1), (.01, 0, 0), 1.), "construction", {"cup": .8})])


def test_semantic_unknown_cannot_supply_scores_and_exact_saga20_is_required():
    with pytest.raises(ValueError, match="unknown"):
        semantic_row("bad", 0, {"cup": .8}, unknown_reason="crop_insufficient")
    with pytest.raises(ValueError, match="SAGA20"):
        semantic_decision([], classes32=CLASSES, saga20=CLASSES[:19])


def test_mask_and_point_crop_roundtrips_padding_non_square_and_scaling():
    image = np.arange(6 * 10 * 3, dtype=np.uint8).reshape(6, 10, 3)
    crop = CropTransform((6, 10), -2, 1, 8, 7)
    encoded, valid = crop.extract(image)
    assert encoded.shape == (7, 8, 3) and np.all(encoded[:, :2] == 127)
    assert valid.sum() == 30
    points = np.array([[0., 1.], [5., 5.]])
    assert np.array_equal(crop.crop_to_image_points(crop.image_to_crop_points(points, encoded_shape=(14, 24)), encoded_shape=(14, 24)), points)
    mask = np.zeros((6, 10), dtype=bool)
    mask[1:6, :6] = True
    assert np.array_equal(crop.mask_to_image(crop.mask_to_crop(mask)), mask)
    assert crop.crop_to_image_box(crop.image_to_crop_box((0, 1, 5, 5))) == (0., 1., 5., 5.)


def test_prior_crop_uses_positive_optical_depth_and_registered_center():
    crop = prior_crop(image_shape=(100, 200), bbox_xyxy=(40, 20, 60, 40),
                      focal_geometric_mean=100, prior_diagonal_m=2, positive_optical_z=2)
    assert crop.width == 150 and crop.left == -25 and crop.top == -45
    with pytest.raises(ValueError, match="optical"):
        prior_crop(image_shape=(100, 200), bbox_xyxy=(40, 20, 60, 40),
                   focal_geometric_mean=100, prior_diagonal_m=2, positive_optical_z=-1)


def test_reliable_point_prefers_anchor_then_ratio_and_stable_pixel():
    ids = np.arange(4).reshape(2, 2)
    locator = FrozenLocator((0, 0, 2, 2), (1, 2), (0, 1, 2, 3), "frozen_source", "hash")
    point = reliable_prompt_point(locator=locator, contributor_ids=ids,
        max_contribution=np.array([[1., .7], [.7, 1.]]), opacity=np.ones((2, 2)), valid_pixels=np.ones((2, 2), dtype=bool))
    assert point["gaussian_id"] == 1 and point["source"] == "anchor"
    assert reliable_prompt_point(locator=locator, contributor_ids=ids,
        max_contribution=np.ones((2, 2)) * .49, opacity=np.ones((2, 2)), valid_pixels=np.ones((2, 2), dtype=bool)) is None


def test_actual_crop_sam_with_one_positive_and_three_masks_no_dino():
    image = np.arange(6 * 10 * 3, dtype=np.uint8).reshape(6, 10, 3)
    crop = CropTransform((6, 10), -2, 1, 8, 5)
    ids = np.arange(60).reshape(6, 10)
    locator = FrozenLocator((0, 1, 6, 6), (11,), tuple(range(60)), "frozen_source", "sourcehash")
    sam = MockSam()
    adapter = InjectedModelAdapter(sam_predictor=sam)
    result = adapter.geometry(image_rgb=image, crop=crop, locator=locator, contributor_ids=ids,
                              max_contribution=np.ones((6, 10)), opacity=np.ones((6, 10)), observation_uid="obs")
    assert result.status == "observed" and len(result.masks) == 3
    assert np.array_equal(sam.inputs[0], crop.extract(image)[0])
    assert np.array_equal(sam.prompts[0]["point_coords"], [[3, 0]])
    assert np.array_equal(sam.prompts[0]["point_labels"], [1])
    assert result.trace["calls"] == {"sam": 1, "dino": 0}
    assert not result.masks[0].mask_image.flags.writeable


def test_empty_prompt_is_measured_unknown_without_model_call():
    sam = MockSam()
    result = InjectedModelAdapter(sam_predictor=sam).geometry(image_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        crop=CropTransform((4, 4), 0, 0, 4, 4), locator=FrozenLocator((0, 0, 4, 4), (0,), (0,), "verified_direct", "h"),
        contributor_ids=np.zeros((4, 4), dtype=np.int64), max_contribution=np.zeros((4, 4)),
        opacity=np.zeros((4, 4)), observation_uid="empty")
    assert result.status == "unknown" and not result.masks and not sam.inputs


def test_raw_token_scores_do_not_inherit_first_phrase_class_or_special_max():
    raw = raw_queries([{"cup": .4, "table": .9}, {"cup": .2}])
    raw["raw_logits"][:, 0] = 30.  # Special token must never become a class score.
    decoded = decode_dino_queries(classes32=CLASSES, **raw)
    first, second = decoded["proposals"]
    assert first["class_scores"]["table"] == pytest.approx(.9)
    assert first["class_scores"]["cup"] == pytest.approx(.4)
    assert first["qualified_classes"] == ["cup", "table"]
    assert second["status"] == "rejected"
    assert len(decoded["raw_logits"]) == 2


def test_class_caption_truncation_and_punctuation_fail_closed():
    raw = raw_queries([{"cup": .8}])
    with pytest.raises(ValueError, match="truncated"):
        class_token_spans(CLASSES, raw["offset_mapping"][:-5], raw["special_tokens_mask"][:-5])
    decoded = decode_dino_queries(classes32=CLASSES, **raw)
    caption = decoded["caption"]
    for span in decoded["class_token_spans"]:
        assert all(any(c.isalnum() for c in caption[s:e]) for s, e in raw["offset_mapping"][list(span)])


def test_cup_cannot_inherit_table_detection_box_mask():
    raw = raw_queries([{"table": .95}, {"cup": .7}, {"lamp": .2}])
    cup = np.zeros((4, 4), dtype=bool)
    cup[:2] = True
    table_masks = np.ones((3, 4, 4), dtype=bool)
    cup_masks = np.broadcast_to(cup, (3, 4, 4)).copy()
    sam = MockSam([table_masks, cup_masks])
    adapter = InjectedModelAdapter(sam_predictor=sam, dino_raw=lambda image, caption: raw)
    row, masks, trace = adapter.semantic(image_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        crop=CropTransform((4, 4), 0, 0, 4, 4), object_mask=cup, classes32=CLASSES,
        camera=camera(0), observation_uid="semantic")
    assert row.class_scores == pytest.approx({"cup": .7})
    assert len(trace["dino"]["proposals"]) == 3 and len(masks) == 6
    assert trace["dino"]["proposals"][0]["reason"] == "no_grounded_object_mask"
    assert trace["dino"]["proposals"][2]["reason"] == "below_box_and_text_threshold"
    assert all(p["point_coords"] is None for p in sam.prompts)


def test_legacy_dino_output_is_engineering_error_not_unknown():
    adapter = InjectedModelAdapter(sam_predictor=MockSam(), dino_raw=lambda image, caption: {"scores": [.9], "class_ids": [0]})
    with pytest.raises(ValueError, match="legacy"):
        adapter.semantic(image_rgb=np.zeros((4, 4, 3), dtype=np.uint8), crop=CropTransform((4, 4), 0, 0, 4, 4),
                         object_mask=np.ones((4, 4), dtype=bool), classes32=CLASSES,
                         camera=camera(0), observation_uid="legacy")


def test_h_mask_generation_cannot_read_pending_projection_or_human_assets():
    parameters = inspect.signature(InjectedModelAdapter.independent_verification_masks).parameters
    assert not set(parameters) & {"proposal", "projected_mask", "pending_members", "gt", "human_mask"}
    with pytest.raises(ValueError, match="locator"):
        FrozenLocator((0, 0, 4, 4), (0,), (0,), "pending_proposal", "hash")
    with pytest.raises(TypeError):
        InjectedModelAdapter(sam_predictor=MockSam()).independent_verification_masks(projected_mask=np.ones((4, 4), dtype=bool))
    box = [0, 0, 4, 4]
    locator = FrozenLocator(box, (0,), (0,), "frozen_source", "hash")
    box[0] = 10
    assert locator.bbox_xyxy == (0., 0., 4., 4.)


def test_h_all_boundary_pass_and_mixed_never_best_of():
    projection = np.zeros((4, 4), dtype=bool)
    projection[:2] = True
    exact = projection.copy()
    expanded = np.ones((4, 4), dtype=bool)
    good = verify_projection(projection, [exact, expanded], np.ones_like(exact), identity_status="accepted")
    assert good["status"] == "pass"  # Both .50 recall or better.
    small = np.zeros((4, 4), dtype=bool)
    small[0] = True
    mixed = verify_projection(projection, [exact, small], np.ones_like(exact), identity_status="accepted")
    assert mixed["status"] == "unknown" and mixed["reason"] == "boundary_alternatives_disagree"
    assert [r["status"] for r in mixed["reference_results"]] == ["pass", "reject"]


def test_h_incomplete_unknown_and_full_rgb_denominator():
    projection = np.zeros((4, 4), dtype=bool)
    projection[0] = True
    whole = np.ones((4, 4), dtype=bool)
    result = verify_projection(projection, [whole], whole, identity_status="accepted")
    assert result["status"] == "incomplete"
    assert result["reference_results"][0]["recall"] == .25  # Non-GS-covered object pixels remain.
    empty = verify_projection(np.zeros((4, 4), dtype=bool), [whole], whole, identity_status="accepted")
    assert empty["status"] == "unknown" and empty["reference_results"][0]["precision"] is None
    valid = whole.copy()
    valid[0, 0] = False
    clipped = verify_projection(projection, [whole], valid, identity_status="accepted")
    assert clipped["reason"] == "crop_or_projection_insufficient"
    assert clipped["projection_outside_observed_pixels"] == 1


def test_h_visible_anchor_exclusion_does_not_single_view_refute_identity():
    anchor = np.zeros((4, 4), dtype=bool)
    anchor[0] = True
    other = ~anchor
    result = verify_projection(anchor, [other], np.ones_like(anchor), identity_status="accepted", visible_anchor_mask=anchor)
    assert result["status"] == "reject" and result["visible_anchor_excluded"]
    assert result["identity_refuted"] is False

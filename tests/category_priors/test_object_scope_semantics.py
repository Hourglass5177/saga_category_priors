from dataclasses import replace
import numpy as np
import pytest

from category_priors.object_scope.artifacts import array_digest, digest
from category_priors.object_scope.encoding import encode_region, full_image_plan
from category_priors.object_scope.geometry import CameraView
from category_priors.object_scope.semantics import (
    aggregate_semantics, cosine_scores, observe_region_pair, validate_observation_trace,
    save_observation, replay_observation)

CLASSES = tuple(f"class{i}" for i in range(32))
SAGA = CLASSES[:20]


class FakeEncoder:
    def __init__(self, winner=0, context_winner=None):
        self.winner, self.context_winner = winner, winner if context_winner is None else context_winner
        self.calls = 0

    def encode_pair(self, detail, context, prompts):
        self.calls += 1
        text = np.eye(32)
        return dict(detail_features=text[self.winner], context_features=text[self.context_winner],
                    text_features=text, token_ids=np.tile(np.arange(77, dtype=np.int64), (32, 1)),
                    runtime_record={"injected_cpu_test": True})


def make_row(index=0, *, model=None, empty=False, final=False, role="construction", manual=False):
    mask = np.ones((10, 20), bool)
    if empty:
        mask[:] = False
    encoded = encode_region(np.zeros((10, 20, 3), np.uint8), mask, np.ones_like(mask), full_image_plan(mask.shape))
    camera = CameraView(f"cam{index}", (0, 0, 1), (index * .2, 0, 0), 1)
    binding = dict(source_kind="diagnostic_manual" if manual else "final_members_projection" if final else "rgb_scope",
        region_version="object-v1", mask_sha256=array_digest(mask), member_sha256="a" * 64 if final else None,
        scene_revision="scene-0", condition="U1", role=role, source_sha256="b" * 64, camera_uid=camera.camera_uid)
    return observe_region_pair(model or FakeEncoder(), encoded, encoded, camera=camera, classes32=CLASSES,
                               region_binding=binding, role=role, allow_diagnostic=role == "diagnostic")


def test_fixed_formula_order_and_all32():
    rows = [make_row(1), make_row(0)]
    result = aggregate_semantics(rows, classes32=CLASSES, saga20=SAGA)
    assert result["status"] == "accepted" and result["score"] == 1
    assert len(result["scores"]) == 32 and result["scores"][CLASSES[1]] == .5
    assert result == aggregate_semantics(rows[::-1], classes32=CLASSES, saga20=SAGA)


def test_per_view_disagreement_cannot_be_fixed_by_mean():
    rows = [make_row(0), make_row(1), make_row(2, model=FakeEncoder(1))]
    result = aggregate_semantics(rows, classes32=CLASSES, saga20=SAGA)
    assert result["status"] == "unknown" and result["reason"] == "valid_camera_top1_disagreement"


def test_detail_context_tie_not_resolved_by_higher_quality():
    rows = [make_row(i, model=FakeEncoder(0, 1)) for i in range(2)]
    result = aggregate_semantics(rows, classes32=CLASSES, saga20=SAGA)
    assert result["reason"] == "exact_class_score_tie"


def test_empty_observations_complete_without_model_calls():
    model = FakeEncoder()
    rows = [make_row(i, model=model, empty=True) for i in range(2)]
    assert model.calls == 0
    result = aggregate_semantics(rows, classes32=CLASSES, saga20=SAGA)
    assert result["N"] == 0 and result["scores"][CLASSES[0]] is None
    assert result["unknown_camera_count"] == 2


def test_missing_view_not_a_zero_or_negative_vote():
    rows = [make_row(0), make_row(1), make_row(2, empty=True)]
    result = aggregate_semantics(rows, classes32=CLASSES, saga20=SAGA)
    assert result["N"] == 2 and result["score"] == 1 and result["unknown_camera_count"] == 1


def test_non_saga_highest_never_falls_back():
    result = aggregate_semantics([make_row(i, model=FakeEncoder(31)) for i in range(2)], classes32=CLASSES, saga20=SAGA)
    assert result["status"] == "out_of_scope" and result["final_class"] == CLASSES[31]


def test_repeated_or_non_independent_camera_cannot_double_vote():
    row = make_row(0)
    with pytest.raises(ValueError, match="one adopted"):
        aggregate_semantics([row, row], classes32=CLASSES, saga20=SAGA)
    other = make_row(1)
    camera = CameraView("cam1", (0, 0, 1), (.001, 0, 0), 1)
    from category_priors.object_scope.artifacts import plain
    trace = {**other.trace, "camera": plain(camera)}
    trace.pop("evidence_sha256")
    trace["evidence_sha256"] = digest(trace)
    other = replace(other, camera=camera, trace=trace)
    with pytest.raises(ValueError, match="pairwise independent"):
        aggregate_semantics([row, other], classes32=CLASSES, saga20=SAGA)


def test_saved_features_replay_and_score_tampering():
    row = make_row()
    validate_observation_trace(row)
    tampered = replace(row, detail_cos=np.ones(32))
    with pytest.raises(ValueError, match="saved raw features"):
        validate_observation_trace(tampered)
    with pytest.raises(ValueError):
        aggregate_semantics([row], classes32=CLASSES, saga20=(*SAGA[:-1], SAGA[0]))


def test_formal_manual_and_online_roles_rejected():
    row = make_row(role="diagnostic", manual=True)
    with pytest.raises(ValueError):
        aggregate_semantics([row], classes32=CLASSES, saga20=SAGA)
    online = make_row(role="online_verification")
    with pytest.raises(ValueError, match="not construction"):
        aggregate_semantics([online], classes32=CLASSES, saga20=SAGA)


def test_member_version_change_cannot_reuse_old_evidence():
    a, b = make_row(0, final=True), make_row(1, final=True)
    binding = {**b.region_binding, "member_sha256": "c" * 64}
    trace = {**b.trace, "region_binding": binding}
    trace.pop("evidence_sha256")
    trace["evidence_sha256"] = digest(trace)
    b = replace(b, region_binding=binding, trace=trace)
    with pytest.raises(ValueError, match="cannot inherit"):
        aggregate_semantics([a, b], classes32=CLASSES, saga20=SAGA)


def test_cosine_rejects_zero_and_nonfinite():
    with pytest.raises(ValueError, match="zero feature"):
        cosine_scores(np.zeros(32), np.eye(32))
    with pytest.raises(ValueError, match="nonfinite"):
        cosine_scores(np.full(32, np.nan), np.eye(32))


@pytest.mark.parametrize("precision", ["float16", "float32"])
def test_full_raw_disk_cold_hot_replay_and_role_identity(tmp_path, precision):
    from category_priors.object_scope.artifacts import EvidenceStore
    class ActualSchemaEncoder(FakeEncoder):
        def encode_pair(self, detail, context, prompts):
            raw = super().encode_pair(detail, context, prompts)
            images = np.stack([detail.rgb_tensor, context.rgb_tensor]).astype(precision)
            alphas = np.stack([detail.alpha_tensor, context.alpha_tensor]).astype(precision)
            record = dict(identity={"model_version": "synthetic-native-schema", "precision": precision},
                prompts=list(prompts), exact_forward_token_ids=raw["token_ids"],
                token_ids_sha256=array_digest(raw["token_ids"]), forward_rgb=images, forward_alpha=alphas,
                forward_rgb_sha256=array_digest(images), forward_alpha_sha256=array_digest(alphas),
                detail_encoding_sha256=detail.trace["encoding_sha256"],
                context_encoding_sha256=context.trace["encoding_sha256"], text_cache_hit=False,
                text_cache_key="synthetic", calls={"visual": 1, "text": 1}, seconds=.1)
            record["runtime_record_sha256"] = digest(record)
            raw["runtime_record"] = record
            return raw
    model = ActualSchemaEncoder()
    row = make_row(model=model)
    identity = save_observation(EvidenceStore(tmp_path), row)
    for _ in range(2):
        replayed = replay_observation(EvidenceStore(tmp_path), identity)
        assert replayed.trace["evidence_sha256"] == row.trace["evidence_sha256"]
        assert np.array_equal(replayed.trace["raw"]["runtime_record"]["forward_alpha"],
                              row.trace["raw"]["runtime_record"]["forward_alpha"])
    assert model.calls == 1
    changed = {**identity, "role": "prepass"}
    with pytest.raises(FileNotFoundError):
        replay_observation(EvidenceStore(tmp_path), changed)
    # Updating the actual-input self-hash cannot hide a different RGB or alpha.
    for label in ("rgb", "alpha"):
        raw = dict(row.trace["raw"])
        record = dict(raw["runtime_record"])
        actual = record["forward_" + label].copy()
        actual.flat[0] += 1
        record["forward_" + label] = actual
        record["forward_" + label + "_sha256"] = array_digest(actual)
        record.pop("runtime_record_sha256")
        record["runtime_record_sha256"] = digest(record)
        raw["runtime_record"] = record
        trace = {**row.trace, "raw": raw}
        trace.pop("evidence_sha256")
        trace["evidence_sha256"] = digest(trace)
        with pytest.raises(ValueError, match="after dtype conversion"):
            validate_observation_trace(replace(row, trace=trace))
        class WrongForward(ActualSchemaEncoder):
            def encode_pair(self, *args):
                return raw
        with pytest.raises(ValueError, match="after dtype conversion"):
            make_row(model=WrongForward())
    # Preserve hashing while introducing inconsistent native token provenance.
    raw = dict(row.trace["raw"])
    record = dict(raw["runtime_record"])
    record["exact_forward_token_ids"] = raw["token_ids"] + 1
    record.pop("runtime_record_sha256")
    record["runtime_record_sha256"] = digest(record)
    raw["runtime_record"] = record
    trace = {**row.trace, "raw": raw}
    trace.pop("evidence_sha256")
    trace["evidence_sha256"] = digest(trace)
    with pytest.raises(ValueError, match="input IDs differ"):
        validate_observation_trace(replace(row, trace=trace))


def test_all_unknown_can_be_saved_and_replayed(tmp_path):
    from category_priors.object_scope.artifacts import EvidenceStore
    row = make_row(empty=True)
    identity = save_observation(EvidenceStore(tmp_path), row)
    assert replay_observation(EvidenceStore(tmp_path), identity).trace["status"] == "unknown"


def test_final_members_fewer_than_four_cannot_reuse_rgb_scope_score():
    mask = np.zeros((10, 20), bool)
    mask[0, :3] = True
    encoded = encode_region(np.zeros((10, 20, 3), np.uint8), mask, np.ones_like(mask), full_image_plan(mask.shape))
    model = FakeEncoder()
    camera = CameraView("cam0", (0, 0, 1), (0, 0, 0), 1)
    binding = dict(source_kind="final_members_projection", region_version="v1", mask_sha256=array_digest(mask),
                   member_sha256="a" * 64, scene_revision="s0", condition="U1", role="construction",
                   source_sha256="b" * 64, camera_uid="cam0")
    row = observe_region_pair(model, encoded, encoded, camera=camera, classes32=CLASSES, region_binding=binding)
    assert model.calls == 0 and row.detail_cos is None
    assert "actual_members" in row.trace["unknown_reasons"]


def test_one_missing_branch_does_not_become_a_single_branch_score():
    from category_priors.object_scope.encoding import CropEncodingPlan
    mask = np.zeros((10, 20), bool)
    mask[1:5, 1:5] = True
    rgb, valid = np.zeros((10, 20, 3), np.uint8), np.ones_like(mask)
    detail = encode_region(rgb, mask, valid, full_image_plan(mask.shape))
    context = encode_region(rgb, mask, valid, CropEncodingPlan(mask.shape, (10, 0, 20, 10), "context", 10, {}))
    model = FakeEncoder()
    camera = CameraView("cam0", (0, 0, 1), (0, 0, 0), 1)
    binding = dict(source_kind="rgb_scope", region_version="v1", mask_sha256=array_digest(mask), member_sha256=None,
                   scene_revision="s0", condition="U1", role="construction", source_sha256="b" * 64, camera_uid="cam0")
    row = observe_region_pair(model, detail, context, camera=camera, classes32=CLASSES, region_binding=binding)
    assert model.calls == 0 and row.trace["status"] == "unknown"

import dataclasses
import numpy as np
import pytest

from category_priors.object_scope.artifacts import EvidenceStore, array_digest
from category_priors.object_scope.observations import (
    CaptureFailure, CropSpec, MaskPayload, capture_bank, discovery_crops,
    validate_bank, bank_record, bank_from_record,
)


class FakeSAM:
    mask_threshold = 0.

    def __init__(self, nan=False, fail_batch=None):
        self.inputs, self.batches = [], []
        self.nan, self.fail_batch = nan, fail_batch

    def set_image(self, rgb):
        self.image = rgb.copy()
        self.inputs.append(rgb.copy())
        return {"actual_tensor": rgb, "encoded_tensor_sha256": array_digest(rgb), "encoded_shape": list(rgb.shape), "kind": "CPU_fake"}

    def predict_batch(self, points):
        self.batches.append(points.copy())
        assert points.shape[1] == 2 and len(points) <= 64
        h, w = self.image.shape[:2]
        assert (points[:, 0] > 0).all() and (points[:, 0] < w).all()
        assert (points[:, 1] > 0).all() and (points[:, 1] < h).all()
        if self.fail_batch == len(self.batches):
            raise RuntimeError("injected decoder failure")
        logits = np.full((len(points), 3, h, w), -3., dtype=np.float32)
        logits[:, 0, 0, :] = 3.  # touches crop edge and has very low quality
        logits[:, 1, :, 0] = 3.
        quality = np.full((len(points), 3), .01, np.float32)
        if self.nan:
            logits[0, 0, 0, 0] = np.nan
            self.nan = False
        return logits, quality


def rgb():
    return np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)


def test_full_registered_sampling_before_every_filter_and_nms():
    fake = FakeSAM()
    image = rgb()
    bank = capture_bank(image_rgb=image, camera_uid="s:a", predictor=fake, model_identity="native-frozen-test")
    assert len(fake.inputs) == 5 and len(fake.batches) == 32
    assert sum(len(x) for x in fake.batches) == 2048
    assert len(bank.slots) == 6144
    for supplied, crop in zip(fake.inputs, discovery_crops(image.shape[:2])):
        x0, y0, x1, y1 = crop.xyxy
        np.testing.assert_array_equal(supplied, image[y0:y1, x0:x1])
    assert all(s.quality < .88 for s in bank.slots)
    assert sum(s.status == "empty" for s in bank.slots) == 2048
    assert bank.slots[0].status == "observed"  # boundary mask retained
    assert bank.slots[0].touch_crop_edge and bank.slots[0].touch_image_edge
    assert "unknown" in bank.slots[0].potential_incomplete
    assert len(bank.payloads) < len(bank.slots)
    assert validate_bank(bank)["slot_count"] == 6144
    with pytest.raises(ValueError, match="missing or extra"):
        validate_bank(dataclasses.replace(bank, slots=bank.slots[:-1]))
    with pytest.raises(ValueError, match="source accounting"):
        validate_bank(dataclasses.replace(bank, slots=(bank.slots[1], bank.slots[0]) + bank.slots[2:]))


def test_rle_roundtrip_and_immutable_content():
    for mask in (np.zeros((3, 7), bool), np.ones((3, 7), bool), np.eye(5, dtype=bool)):
        payload = MaskPayload.encode(mask)
        np.testing.assert_array_equal(payload.decode(), mask)
        with pytest.raises(ValueError):
            payload.run_lengths.setflags(write=True)
    with pytest.raises(ValueError, match="content differs"):
        MaskPayload((3, 7), np.array([21], np.uint32), "wrong")


def test_iteration_schedule_is_exact_two_crops_and_96_slots():
    crops = (CropSpec("detail", (2, 1, 12, 8), 4), CropSpec("context", (0, 0, 16, 12), 4))
    fake = FakeSAM()
    bank = capture_bank(image_rgb=rgb(), camera_uid="s:a", predictor=fake, model_identity="model", mode="iteration", iteration_crops=crops)
    assert len(bank.slots) == 96 and len(fake.inputs) == 2 and sum(map(len, fake.batches)) == 32
    assert isinstance(bank.encodings[0].predictor_trace["encoded_shape"], tuple)
    with pytest.raises(TypeError):
        bank.encodings[0].predictor_trace["kind"] = "changed"
    with pytest.raises(ValueError, match="two actual"):
        capture_bank(image_rgb=rgb(), camera_uid="s:a", predictor=FakeSAM(), model_identity="model", mode="iteration", iteration_crops=crops[:1])


def test_nonfinite_output_retained_as_invalid_not_an_empty_observation():
    crops = (CropSpec("detail", (0, 0, 16, 12), 4), CropSpec("context", (0, 0, 16, 12), 4))
    bank = capture_bank(image_rgb=rgb(), camera_uid="s:a", predictor=FakeSAM(nan=True), model_identity="model", mode="iteration", iteration_crops=crops)
    check = validate_bank(bank)
    assert len(check["invalid_slots"]) == 1
    assert bank.slots[0].mask_sha256 is None and bank.slots[0].status == "invalid"
    assert bank.slots[0].raw_logits_sha256


def test_decoder_failure_preserves_actual_partial_evidence_without_filling_slots():
    with pytest.raises(CaptureFailure) as failure:
        capture_bank(image_rgb=rgb(), camera_uid="s:a", predictor=FakeSAM(fail_batch=2), model_identity="model")
    partial = failure.value.partial
    assert len(partial.encodings) == 1 and len(partial.slots) == 64 * 3
    with pytest.raises(ValueError, match="encoding"):
        validate_bank(partial)


def test_actual_array_storage_two_independent_reopens(tmp_path):
    crops = (CropSpec("detail", (0, 0, 16, 12), 4), CropSpec("context", (0, 0, 16, 12), 4))
    bank = capture_bank(image_rgb=rgb(), camera_uid="s:a", predictor=FakeSAM(), model_identity="model", mode="iteration", iteration_crops=crops)
    record = bank_record(bank, EvidenceStore(tmp_path))
    first = bank_from_record(record, EvidenceStore(tmp_path))
    second = bank_from_record(record, EvidenceStore(tmp_path))
    assert first.sha256 == second.sha256 == bank.sha256
    for before, after in zip(bank.payloads, second.payloads):
        np.testing.assert_array_equal(before.decode(), after.decode())
    for before, after in zip(bank.encodings, second.encodings):
        np.testing.assert_array_equal(before.predictor_trace["actual_tensor"], after.predictor_trace["actual_tensor"])


def test_native_adapter_captures_actual_pre_hook_input_not_second_preprocess(monkeypatch):
    import sys
    from types import SimpleNamespace
    from contextlib import nullcontext
    from category_priors.object_scope.observations import NativeSAM1Predictor
    class Tensor:
        def __init__(self, value): self.value = np.asarray(value)
        def permute(self, *axes): return Tensor(self.value.transpose(axes))
        def contiguous(self): return self
        def __getitem__(self, key): return Tensor(self.value[key])
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.value
        @property
        def shape(self): return self.value.shape
    class Encoder:
        def register_forward_pre_hook(self, callback):
            self.callback = callback
            return SimpleNamespace(remove=lambda: setattr(self, "callback", None))
    encoder = Encoder()
    class Predictor:
        device = "cpu"
        transform = SimpleNamespace(apply_image=lambda image: image)
        model = SimpleNamespace(mask_threshold=0., image_encoder=encoder)
        calls = 0
        def set_torch_image(self, tensor, shape):
            self.calls += 1
            self.actual = tensor.value.astype(np.float32) + 7  # actual internal normalization sentinel
            encoder.callback(encoder, (Tensor(self.actual),))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(as_tensor=lambda a, device: Tensor(a), inference_mode=nullcontext))
    predictor = Predictor()
    trace = NativeSAM1Predictor(predictor).set_image(rgb())
    assert predictor.calls == 1 and encoder.callback is None
    np.testing.assert_array_equal(trace["actual_tensor"], predictor.actual)
    assert trace["encoded_tensor_sha256"] == array_digest(predictor.actual)
    with pytest.raises(ValueError):
        trace["actual_tensor"].setflags(write=True)

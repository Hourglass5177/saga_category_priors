"""Source-free SAM1 observations, before filtering and NMS.

The injected predictor receives RGB and regular grids only. Masks are losslessly
RLE stored; per-crop validity and every prompt/alternative remain separate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from collections.abc import Mapping
import numpy as np

from .artifacts import array_digest, digest, plain, readonly


ALGORITHM = "scope-v2-raw-sam1-grid-v1"


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, np.ndarray):
        return readonly(value)
    return value


def _shape(value):
    result = tuple(value)
    if len(result) != 2 or any(type(v) is not int or v <= 0 for v in result):
        raise ValueError("positive integer image H/W required")
    return result


@dataclass(frozen=True)
class CropSpec:
    name: str
    xyxy: tuple[int, int, int, int]
    grid_side: int

    def __post_init__(self):
        if not self.name or len(self.xyxy) != 4 or any(type(v) is not int for v in self.xyxy):
            raise ValueError("named integer crop required")
        x0, y0, x1, y1 = self.xyxy
        if min(x0, y0) < 0 or x1 <= x0 or y1 <= y0 or type(self.grid_side) is not int or self.grid_side <= 0:
            raise ValueError("invalid actual RGB crop")
        object.__setattr__(self, "xyxy", tuple(self.xyxy))


def discovery_crops(image_shape):
    h, w = _shape(image_shape)
    overlap = int((512 / 1500) * min(h, w))
    cw, ch = math.ceil((w + overlap) / 2), math.ceil((h + overlap) / 2)
    result = [CropSpec("full", (0, 0, w, h), 32)]
    for ix, x0 in enumerate((0, cw - overlap)):
        for iy, y0 in enumerate((0, ch - overlap)):
            result.append(CropSpec(f"tile-{ix}-{iy}", (x0, y0, min(x0 + cw, w), min(y0 + ch, h)), 16))
    return tuple(result)


def grid_points(crop):
    n = crop.grid_side
    axis = (np.arange(n, dtype=np.float64) + .5) / n
    xx, yy = np.meshgrid(axis, axis)
    x0, y0, x1, y1 = crop.xyxy
    return readonly(np.stack((xx.ravel() * (x1 - x0), yy.ravel() * (y1 - y0)), axis=1))


@dataclass(frozen=True)
class MaskPayload:
    image_shape: tuple[int, int]
    run_lengths: np.ndarray
    mask_sha256: str

    def __post_init__(self):
        shape = _shape(self.image_shape)
        runs = np.asarray(self.run_lengths)
        if runs.ndim != 1 or runs.dtype.kind not in "iu" or len(runs) == 0 or int(runs.sum()) != math.prod(shape):
            raise ValueError("invalid complete RLE payload")
        if runs.dtype.kind == "i" and np.any(runs < 0):
            raise ValueError("negative RLE run")
        object.__setattr__(self, "image_shape", shape)
        object.__setattr__(self, "run_lengths", readonly(runs, np.uint32))
        if array_digest(self.decode()) != self.mask_sha256:
            raise ValueError("mask payload content differs")

    @classmethod
    def encode(cls, mask):
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.ndim != 2:
            raise ValueError("raw mask must be HxW bool")
        flat = mask.ravel()
        boundaries = np.concatenate(([0], np.flatnonzero(flat[1:] != flat[:-1]) + 1, [flat.size]))
        runs = np.diff(boundaries)
        if bool(flat[0]):
            runs = np.concatenate(([0], runs))
        return cls(tuple(mask.shape), readonly(runs, np.uint32), array_digest(mask))

    def decode(self):
        return readonly(np.repeat(np.arange(len(self.run_lengths)) % 2 == 1, self.run_lengths).reshape(self.image_shape))


@dataclass(frozen=True)
class MaskSlot:
    uid: str
    crop_name: str
    prompt_index: int
    alternative: int
    point_image_xy: tuple[float, float]
    mask_sha256: str | None
    quality: float | None
    stability: float | None
    raw_logits_sha256: str
    raw_quality_sha256: str
    status: str
    reason: str
    touch_crop_edge: bool | None
    touch_image_edge: bool | None
    potential_incomplete: str

    def __post_init__(self):
        if self.touch_crop_edge is not None and type(self.touch_crop_edge) is not bool:
            raise ValueError("crop boundary flag must be measured bool or invalid None")
        if self.touch_image_edge is not None and type(self.touch_image_edge) is not bool:
            raise ValueError("image boundary flag must be measured bool or invalid None")
        if (type(self.prompt_index) is not int or self.prompt_index < 0 or type(self.alternative) is not int
                or not 0 <= self.alternative < 3 or len(self.point_image_xy) != 2
                or not np.isfinite(self.point_image_xy).all()):
            raise ValueError("valid immutable prompt identity required")
        object.__setattr__(self, "point_image_xy", tuple(map(float, self.point_image_xy)))


@dataclass(frozen=True)
class EncodingRecord:
    crop: CropSpec
    input_rgb_sha256: str
    valid_pixels_sha256: str
    predictor_trace: dict

    def __post_init__(self):
        tensor = self.predictor_trace.get("actual_tensor")
        if not isinstance(tensor, np.ndarray) or array_digest(tensor) != self.predictor_trace.get("encoded_tensor_sha256"):
            raise ValueError("actual image encoder tensor must be retained and hashed")
        object.__setattr__(self, "predictor_trace", _freeze(dict(self.predictor_trace)))


@dataclass(frozen=True)
class RGBObservationBank:
    camera_uid: str
    image_shape: tuple[int, int]
    rgb_sha256: str
    model_identity: str
    mode: str
    crops: tuple[CropSpec, ...]
    encodings: tuple[EncodingRecord, ...]
    slots: tuple[MaskSlot, ...]
    payloads: tuple[MaskPayload, ...]
    algorithm: str = ALGORITHM

    def __post_init__(self):
        object.__setattr__(self, "image_shape", _shape(self.image_shape))
        for name in ("crops", "encodings", "slots", "payloads"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (any(not isinstance(v, str) or not v for v in (self.camera_uid, self.rgb_sha256, self.model_identity))
                or self.algorithm != ALGORITHM):
            raise ValueError("actual RGB/model/algorithm identity required")
        if self.mode == "discovery":
            if self.crops != discovery_crops(self.image_shape):
                raise ValueError("discovery sampling schedule changed")
        elif self.mode == "iteration":
            if tuple(c.name for c in self.crops) != ("detail", "context") or any(c.grid_side != 4 for c in self.crops):
                raise ValueError("iteration has two actual 4x4 crops")
        else:
            raise ValueError("unregistered sampling mode")
        h, w = self.image_shape
        if any(c.xyxy[2] > w or c.xyxy[3] > h for c in self.crops):
            raise ValueError("crop outside actual RGB")

    @property
    def sha256(self):
        return digest(self)

    def payload(self, sha):
        return next(p for p in self.payloads if p.mask_sha256 == sha)


def crop_valid(image_shape, crop):
    valid = np.zeros(image_shape, bool)
    x0, y0, x1, y1 = crop.xyxy
    valid[y0:y1, x0:x1] = True
    return readonly(valid)


def _slot_uid(bank_identity, crop_name, prompt_index, alternative):
    return digest((bank_identity, crop_name, prompt_index, alternative))


def validate_bank(bank):
    """Recount actual slots; no caller-provided completeness boolean exists."""
    identity = (bank.algorithm, bank.camera_uid, bank.rgb_sha256, bank.model_identity, bank.mode, bank.crops)
    if tuple(e.crop for e in bank.encodings) != bank.crops:
        raise ValueError("missing/duplicated actual image encoding")
    for encoding in bank.encodings:
        if encoding.valid_pixels_sha256 != array_digest(crop_valid(bank.image_shape, encoding.crop)):
            raise ValueError("crop validity identity differs")
        if not encoding.input_rgb_sha256 or not encoding.predictor_trace.get("encoded_tensor_sha256"):
            raise ValueError("actual encoder tensor trace missing")
    payloads = {p.mask_sha256: p for p in bank.payloads}
    if len(payloads) != len(bank.payloads):
        raise ValueError("exact mask contents should be deduplicated")
    expected = []
    for crop in bank.crops:
        points = grid_points(crop) + np.asarray(crop.xyxy[:2])
        for i, point in enumerate(points):
            expected.extend((crop, i, j, point) for j in range(3))
    if len(bank.slots) != len(expected):
        raise ValueError("raw bank has missing or extra slots")
    seen = set()
    invalid = []
    edge_cache = {}
    for slot, (crop, i, j, point) in zip(bank.slots, expected):
        if (slot.uid in seen or slot.uid != _slot_uid(identity, crop.name, i, j)
                or (slot.crop_name, slot.prompt_index, slot.alternative) != (crop.name, i, j)
                or tuple(slot.point_image_xy) != tuple(point)):
            raise ValueError("raw prompt/alternative source accounting differs")
        seen.add(slot.uid)
        if slot.status not in {"observed", "empty", "invalid"}:
            raise ValueError("unknown raw slot status")
        if not slot.raw_logits_sha256 or not slot.raw_quality_sha256:
            raise ValueError("raw model output identities absent")
        if slot.status == "invalid":
            if slot.mask_sha256 is not None or not slot.reason:
                raise ValueError("invalid output cannot masquerade as measured empty mask")
            invalid.append(slot.uid)
        else:
            if slot.mask_sha256 not in payloads or slot.quality is None or not math.isfinite(slot.quality):
                raise ValueError("valid output payload/quality missing")
            if slot.stability is not None and (not math.isfinite(slot.stability) or not 0 <= slot.stability <= 1):
                raise ValueError("invalid stability statistic")
            edge_key = (slot.mask_sha256, crop.name)
            if edge_key not in edge_cache:
                edge_cache[edge_key] = _edge_record(payloads[slot.mask_sha256].decode(), crop)
            edges = edge_cache[edge_key]
            if (slot.touch_crop_edge, slot.touch_image_edge, slot.potential_incomplete) != edges:
                raise ValueError("boundary incompleteness record differs from raw mask")
    return {"slot_count": len(bank.slots), "encoding_count": len(bank.encodings),
            "invalid_slots": tuple(invalid), "bank_sha256": bank.sha256}


class CaptureFailure(RuntimeError):
    def __init__(self, message, partial):
        super().__init__(message)
        self.partial = partial


def _edge_record(mask, crop):
    x0, y0, x1, y1 = crop.xyxy
    local = mask[y0:y1, x0:x1]
    crop_edge = bool(local[0].any() or local[-1].any() or local[:, 0].any() or local[:, -1].any())
    image_edge = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    description = ("touches_image_boundary_unobserved_extent_unknown" if image_edge else
                   "touches_crop_boundary_unobserved_extent_unknown" if crop_edge else
                   "no_boundary_contact_does_not_establish_physical_completeness")
    return crop_edge, image_edge, description


def capture_bank(*, image_rgb, camera_uid, predictor, model_identity,
                 mode="discovery", iteration_crops=None, on_event=None):
    """Predictor protocol: set_image(rgb)->tensor trace; predict_batch(points)->logits,qualities.

    No candidate, source, mask, class, inherited box or anchor is an input.
    A sink may persist each event while it is produced. Failure exposes partial
    evidence and raises; no missing model output is fabricated.
    """
    image = readonly(image_rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("actual uint8 RGB required")
    shape = tuple(image.shape[:2])
    crops = discovery_crops(shape) if mode == "discovery" else tuple(iteration_crops or ())
    rgb_sha = array_digest(image)
    identity = (ALGORITHM, camera_uid, rgb_sha, model_identity, mode, crops)
    slots, encodings, payloads = [], [], {}
    threshold = float(predictor.mask_threshold)
    if not math.isfinite(threshold):
        raise ValueError("finite frozen SAM mask threshold required")

    def bank():
        return RGBObservationBank(camera_uid, shape, rgb_sha, model_identity, mode, crops,
                                  tuple(encodings), tuple(slots), tuple(payloads.values()))
    bank()  # Validate the sampling contract before the first model call.
    try:
        for crop in crops:
            x0, y0, x1, y1 = crop.xyxy
            actual = image[y0:y1, x0:x1]
            trace = predictor.set_image(actual)
            if not isinstance(trace, dict) or not trace.get("encoded_tensor_sha256"):
                raise ValueError("SAM adapter must record its actual encoded tensor")
            enc = EncodingRecord(crop, array_digest(actual), array_digest(crop_valid(shape, crop)), trace)
            encodings.append(enc)
            if on_event:
                on_event("encoding", enc)
            points = grid_points(crop)
            for start in range(0, len(points), 64):
                batch = points[start:start + 64]
                logits, qualities = map(np.asarray, predictor.predict_batch(batch))
                if logits.shape != (len(batch), 3, y1 - y0, x1 - x0) or qualities.shape != (len(batch), 3):
                    raise ValueError("SAM must preserve all three outputs in actual crop axes")
                for offset in range(len(batch)):
                    for ordinal in range(3):
                        logit = logits[offset, ordinal]
                        quality = qualities[offset, ordinal]
                        finite = np.isfinite(logit).all() and np.isfinite(quality)
                        status, reason, mask_sha, stability = "invalid", "nonfinite_model_output", None, None
                        edges = (None, None, "invalid_raw_output_extent_unknown")
                        if finite:
                            full = np.zeros(shape, bool)
                            full[y0:y1, x0:x1] = logit > threshold
                            mask_sha = array_digest(full)
                            if mask_sha not in payloads:
                                payloads[mask_sha] = MaskPayload.encode(full)
                                if on_event:
                                    on_event("payload", payloads[mask_sha])
                            denominator = int((logit > threshold - 1.).sum())
                            stability = float((logit > threshold + 1.).sum() / denominator) if denominator else None
                            status, reason = ("observed", "all_raw_outputs_retained") if full.any() else ("empty", "measured_empty")
                            edges = _edge_record(full, crop)
                        i = start + offset
                        slot = MaskSlot(_slot_uid(identity, crop.name, i, ordinal), crop.name, i, ordinal,
                            tuple(points[i] + np.asarray((x0, y0))), mask_sha,
                            float(quality) if np.isfinite(quality) else None, stability,
                            array_digest(logit), array_digest(np.asarray(quality)), status, reason, *edges)
                        slots.append(slot)
                        if on_event:
                            on_event("slot", slot)
        result = bank()
        validate_bank(result)
        return result
    except Exception as error:
        raise CaptureFailure(str(error), bank()) from error


class NativeSAM1Predictor:
    """Minimal native SAM1 adapter; imports torch only at actual inference."""
    def __init__(self, sam_predictor):
        self.predictor = sam_predictor
        self.mask_threshold = float(sam_predictor.model.mask_threshold)

    def set_image(self, image_rgb):
        import torch
        p = self.predictor
        transformed = p.transform.apply_image(image_rgb)
        tensor = torch.as_tensor(transformed, device=p.device).permute(2, 0, 1).contiguous()[None]
        actual_inputs = []

        def record_actual(module, args):
            encoded = args[0]
            actual = readonly(encoded.detach().cpu().numpy())
            actual_inputs.append({"actual_tensor": actual, "encoded_tensor_sha256": array_digest(actual),
                                  "encoded_shape": list(encoded.shape)})

        handle = p.model.image_encoder.register_forward_pre_hook(record_actual)
        try:
            with torch.inference_mode():
                p.set_torch_image(tensor, image_rgb.shape[:2])
        finally:
            handle.remove()
        if len(actual_inputs) != 1:
            raise ValueError("exactly one actual image encoder input must be captured")
        trace = {**actual_inputs[0], "resized_shape": list(transformed.shape),
                 "input_shape": list(image_rgb.shape), "preprocessing": "native-SAM1-normalize-pad"}
        return trace

    def predict_batch(self, points_crop_xy):
        import torch
        p = self.predictor
        points = p.transform.apply_coords(points_crop_xy, p.original_size)
        coordinates = torch.as_tensor(points, dtype=torch.float32, device=p.device)
        labels = torch.ones((len(points), 1), dtype=torch.int64, device=p.device)
        with torch.inference_mode():
            logits, quality, _ = p.predict_torch(coordinates[:, None, :], labels,
                boxes=None, mask_input=None, multimask_output=True, return_logits=True)
        return logits.detach().cpu().numpy(), quality.detach().cpu().numpy()


def bank_record(bank, store):
    validate_bank(bank)
    row = plain(bank)
    def save(value):
        if isinstance(value, np.ndarray):
            return {"stored_ndarray": store.put_array(value)}
        if isinstance(value, Mapping):
            return {k: save(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [save(v) for v in value]
        return value
    for entry, encoding in zip(row["encodings"], bank.encodings):
        entry["predictor_trace"] = save(encoding.predictor_trace)
    row["payloads"] = [{"image_shape": list(p.image_shape), "mask_sha256": p.mask_sha256,
                        "run_lengths": store.put_array(p.run_lengths)} for p in bank.payloads]
    return row


def bank_from_record(row, store):
    row = dict(row)
    def restore(value):
        if isinstance(value, dict):
            if set(value) == {"stored_ndarray"}:
                return store.read_array(value["stored_ndarray"])
            return {k: restore(v) for k, v in value.items()}
        if isinstance(value, list):
            return [restore(v) for v in value]
        return value
    row["crops"] = tuple(CropSpec(**c) for c in row["crops"])
    row["encodings"] = tuple(EncodingRecord(CropSpec(**e["crop"]), e["input_rgb_sha256"],
        e["valid_pixels_sha256"], restore(e["predictor_trace"])) for e in row["encodings"])
    row["slots"] = tuple(MaskSlot(**s) for s in row["slots"])
    row["payloads"] = tuple(MaskPayload(tuple(p["image_shape"]), store.read_array(p["run_lengths"]),
        p["mask_sha256"]) for p in row["payloads"])
    result = RGBObservationBank(**row)
    validate_bank(result)
    return result

"""Fixed region-local classification; no detector, SAM or instance controller."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from collections.abc import Mapping
import re

import numpy as np

from .artifacts import array_digest, digest, readonly, plain, file_digest
from .encoding import EncodedRegion, validate_encoding
from .geometry import CameraView, cameras_independent

SEMANTIC_VERSION = "scope-alpha-clip-cosine-pair-mean-v2"
BINDING_KEYS = {"source_kind", "region_version", "mask_sha256", "member_sha256", "scene_revision",
                "condition", "role", "source_sha256", "camera_uid"}
ROLES = {"construction", "prepass", "online_verification", "diagnostic"}


def validate_classes(classes32, saga20=None):
    classes = tuple(classes32)
    if (len(classes) != 32 or any(not isinstance(c, str) or not c.strip() for c in classes)
            or len(set(classes)) != 32):
        raise ValueError("freeze 32 unique nonempty class names in order")
    if saga20 is not None:
        saga = tuple(saga20)
        if len(saga) != 20 or len(set(saga)) != 20 or not set(saga) <= set(classes):
            raise ValueError("freeze 20 unique SAGA20 names within classes32")
    return classes


def class_prompts(classes32):
    return tuple(f"a photo of a {name}." for name in validate_classes(classes32))


def validate_region_binding(binding, *, camera, role, allow_diagnostic=False):
    if not isinstance(binding, Mapping) or set(binding) != BINDING_KEYS:
        raise ValueError("strict region provenance binding required; no GT/manual metadata fields")
    if role not in ROLES or binding["role"] != role or binding["camera_uid"] != camera.camera_uid:
        raise ValueError("camera/role identity mismatch")
    if any(not isinstance(binding[k], str) or not binding[k] for k in ("region_version", "scene_revision", "condition")):
        raise ValueError("nonempty object, scene and condition identity required")
    for key in ("mask_sha256", "source_sha256"):
        if not isinstance(binding[key], str) or not re.fullmatch("[0-9a-f]{64}", binding[key]):
            raise ValueError("invalid region/source content identity")
    kind = binding["source_kind"]
    if kind == "diagnostic_manual":
        if not allow_diagnostic or role != "diagnostic":
            raise ValueError("human masks cannot enter the formal semantic interface")
    elif kind not in {"rgb_scope", "final_members_projection"}:
        raise ValueError("unknown or forbidden region source")
    if role == "diagnostic" and not allow_diagnostic:
        raise ValueError("diagnostic entrypoint must be explicit")
    member = binding["member_sha256"]
    if kind == "final_members_projection":
        if not isinstance(member, str) or not re.fullmatch("[0-9a-f]{64}", member):
            raise ValueError("final semantics require actual member content identity")
    elif member is not None:
        raise ValueError("RGB/manual scope is not an actual-members projection")
    return dict(binding)


def cosine_scores(visual, text):
    visual, text = np.asarray(visual, np.float64), np.asarray(text, np.float64)
    if visual.ndim != 1 or text.ndim != 2 or text.shape != (32, len(visual)):
        raise ValueError("expected one image feature and 32 compatible text vectors")
    if not np.isfinite(visual).all() or not np.isfinite(text).all():
        raise ValueError("nonfinite model features are an engineering failure")
    vn, tn = np.linalg.norm(visual), np.linalg.norm(text, axis=1)
    if vn == 0 or np.any(tn == 0):
        raise ValueError("zero feature norm is not a scientific unknown")
    return readonly((text / tn[:, None]) @ (visual / vn), np.float64)


@dataclass(frozen=True)
class RegionSemanticObservation:
    camera: CameraView
    role: str
    region_binding: dict
    detail_cos: np.ndarray | None
    context_cos: np.ndarray | None
    trace: dict


def observe_region_pair(encoder, detail: EncodedRegion, context: EncodedRegion, *, camera,
                        classes32, region_binding, role="construction", allow_diagnostic=False):
    classes = validate_classes(classes32)
    binding = validate_region_binding(region_binding, camera=camera, role=role, allow_diagnostic=allow_diagnostic)
    validate_encoding(detail)
    validate_encoding(context)
    identity_keys = ("original_rgb_sha256", "original_mask_sha256", "original_valid_sha256", "alpha_mode")
    if any(detail.trace[k] != context.trace[k] for k in identity_keys):
        raise ValueError("two encoding branches changed the original region/RGB/alpha condition")
    if binding["mask_sha256"] != detail.trace["original_mask_sha256"]:
        raise ValueError("semantic region does not match bound mask")
    if detail.trace["alpha_mode"] != "object" and role != "diagnostic":
        raise ValueError("valid-alpha controls cannot enter formal semantics")
    reasons = {k: list(v.trace["unknown_reasons"]) for k, v in (("detail", detail), ("context", context))}
    if binding["source_kind"] == "final_members_projection" and detail.trace["original_target_count"] < 4:
        reasons["actual_members"] = ["fewer_than_four_reliable_projection_pixels"]
    valid = detail.valid and context.valid and "actual_members" not in reasons
    raw = None
    dc = cc = None
    if valid:
        raw = encoder.encode_pair(detail, context, class_prompts(classes))
        required = {"detail_features", "context_features", "text_features", "token_ids", "runtime_record"}
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError("full raw Alpha-CLIP evidence required")
        tokens = np.asarray(raw["token_ids"])
        if tokens.ndim != 2 or tokens.shape[0] != 32 or tokens.dtype.kind not in "iu":
            raise ValueError("actual 32-class input token IDs required")
        raw = {**raw, **{k: readonly(raw[k]) for k in required - {"runtime_record"}}}
        dc = cosine_scores(raw["detail_features"], raw["text_features"])
        cc = cosine_scores(raw["context_features"], raw["text_features"])
    expected_inputs = None
    if raw is not None and "runtime_record_sha256" in raw["runtime_record"]:
        expected_inputs = {label: readonly(np.stack([getattr(detail, label + "_tensor"),
                                                    getattr(context, label + "_tensor")]))
                           for label in ("rgb", "alpha")}
    trace = {"schema": SEMANTIC_VERSION, "semantic_source_sha256": file_digest(__file__),
             "camera": plain(camera), "role": role,
             "region_binding": binding, "classes32": list(classes), "prompts": list(class_prompts(classes)),
             "detail_encoding": detail.trace, "context_encoding": context.trace,
             "raw": raw, "expected_encoding_tensors": expected_inputs,
             "status": "complete" if valid else "unknown", "unknown_reasons": reasons,
             "model_calls": 0 if raw is None else 1, "calibrated_probability": False}
    trace["evidence_sha256"] = digest(trace)
    row = RegionSemanticObservation(camera, role, binding, dc, cc, trace)
    validate_observation_trace(row, allow_diagnostic=allow_diagnostic)
    return row


def validate_observation_trace(row: RegionSemanticObservation, *, allow_diagnostic=False):
    validate_region_binding(row.region_binding, camera=row.camera, role=row.role, allow_diagnostic=allow_diagnostic)
    trace = dict(row.trace)
    claimed = trace.pop("evidence_sha256")
    if digest(trace) != claimed or trace["region_binding"] != row.region_binding or trace["role"] != row.role:
        raise ValueError("semantic trace content/binding mismatch")
    if digest(trace["camera"]) != digest(row.camera):
        raise ValueError("semantic trace camera geometry mismatch")
    raw = trace["raw"]
    for name in ("detail_encoding", "context_encoding"):
        encoding = dict(trace[name])
        encoding_hash = encoding.pop("encoding_sha256")
        if digest(encoding) != encoding_hash or encoding["original_mask_sha256"] != row.region_binding["mask_sha256"]:
            raise ValueError("original region/encoding identity differs")
    if trace["prompts"] != list(class_prompts(trace["classes32"])):
        raise ValueError("semantic prompt template/order changed")
    if raw is None:
        if row.detail_cos is not None or row.context_cos is not None or trace["status"] != "unknown":
            raise ValueError("empty measured observation cannot supply scores")
    else:
        runtime = raw["runtime_record"]
        # Native records are checked down to actual forward arrays and token IDs.
        # Injected CPU encoders remain possible; production factory admission is
        # enforced by the independently reviewed launcher, never a truthy score.
        if "runtime_record_sha256" in runtime:
            record = dict(runtime)
            signature = record.pop("runtime_record_sha256")
            if digest(record) != signature:
                raise ValueError("actual runtime record hash mismatch")
            if not np.array_equal(raw["token_ids"], record["exact_forward_token_ids"]):
                raise ValueError("features/tokenizer input IDs differ")
            if array_digest(raw["token_ids"]) != record["token_ids_sha256"]:
                raise ValueError("tokenizer content identity mismatch")
            for label in ("rgb", "alpha"):
                actual = np.asarray(record["forward_" + label])
                if array_digest(actual) != record["forward_" + label + "_sha256"]:
                    raise ValueError("actual forward tensor identity mismatch")
                precision = record["identity"].get("precision")
                if precision not in {"float16", "float32"} or actual.dtype != np.dtype(precision):
                    raise ValueError("actual forward tensor dtype differs from frozen precision")
                expected = np.asarray(trace["expected_encoding_tensors"][label])
                if expected.dtype != np.dtype("float32") or len(expected) != 2:
                    raise ValueError("original encoding tensors must preserve both float32 branches")
                for index, branch in enumerate(("detail", "context")):
                    if array_digest(expected[index]) != trace[branch + "_encoding"][label + "_tensor_sha256"]:
                        raise ValueError("saved expected tensor differs from registered encoding")
                if actual.shape != expected.shape or not np.array_equal(actual, expected.astype(precision)):
                    raise ValueError("actual forward tensor differs from registered encoding after dtype conversion")
            for branch in ("detail", "context"):
                if record[branch + "_encoding_sha256"] != trace[branch + "_encoding"]["encoding_sha256"]:
                    raise ValueError("forward consumed a different encoded region")
            if record["prompts"] != trace["prompts"]:
                raise ValueError("actual text forward differs from registered prompts")
        for name, stored in (("detail", row.detail_cos), ("context", row.context_cos)):
            computed = cosine_scores(raw[name + "_features"], raw["text_features"])
            if stored is None or not np.array_equal(computed, stored):
                raise ValueError("scores differ from saved raw features")
        if trace["status"] != "complete":
            raise ValueError("unknown observation cannot contribute features")
    return claimed


def observation_cache_identity(row):
    """Complete frozen input identity; elapsed time and cache-hit flags are outputs."""
    raw = row.trace["raw"]
    runtime = None if raw is None else raw["runtime_record"].get("identity", raw["runtime_record"])
    if isinstance(runtime, Mapping):
        runtime = {k: v for k, v in runtime.items() if k != "load_seconds"}
    return {"schema": SEMANTIC_VERSION, "semantic_source_sha256": row.trace["semantic_source_sha256"],
            "region_binding": row.region_binding, "camera": plain(row.camera),
            "role": row.role, "classes32": row.trace["classes32"], "prompts": row.trace["prompts"],
            "detail_encoding_sha256": row.trace["detail_encoding"]["encoding_sha256"],
            "context_encoding_sha256": row.trace["context_encoding"]["encoding_sha256"],
            "model_runtime_identity": runtime}


def save_observation(store, row, *, allow_diagnostic=False):
    """Save lossless large arrays separately; no pixel-list JSON expansion."""
    validate_observation_trace(row, allow_diagnostic=allow_diagnostic)
    def pack(value):
        if isinstance(value, np.ndarray):
            return {"__scope_array__": store.put_array(value)}
        if isinstance(value, Mapping):
            return {k: pack(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [pack(v) for v in value]
        return value
    identity = observation_cache_identity(row)
    data = {"camera": plain(row.camera), "role": row.role, "region_binding": row.region_binding,
            "detail_cos": pack(row.detail_cos), "context_cos": pack(row.context_cos), "trace": pack(row.trace)}
    store.put(identity, data)
    return identity


def replay_observation(store, identity, *, allow_diagnostic=False):
    """Independent disk reopen and full score reconstruction; zero model calls."""
    def unpack(value):
        if isinstance(value, Mapping):
            if set(value) == {"__scope_array__"}:
                return store.read_array(value["__scope_array__"])
            return {k: unpack(v) for k, v in value.items()}
        if isinstance(value, list):
            return [unpack(v) for v in value]
        return value
    data = unpack(store.get(identity))
    if not (digest(data["camera"]) == digest(data["trace"]["camera"]) == digest(identity["camera"])):
        raise ValueError("recorded semantic camera differs from requested evidence identity")
    row = RegionSemanticObservation(CameraView.from_record(data["camera"]), data["role"], data["region_binding"],
                                    data["detail_cos"], data["context_cos"], data["trace"])
    validate_observation_trace(row, allow_diagnostic=allow_diagnostic)
    if digest(observation_cache_identity(row)) != digest(identity):
        raise ValueError("replayed model/role/encoding identity differs from requested cache")
    return row


def aggregate_semantics(observations, *, classes32, saga20, allow_diagnostic=False):
    classes = validate_classes(classes32, saga20)
    rows = sorted(tuple(observations), key=lambda r: r.camera.camera_uid)
    if len({r.camera.camera_uid for r in rows}) != len(rows):
        raise ValueError("one adopted region per camera, not per crop or mask")
    if len({r.role for r in rows}) > 1:
        raise ValueError("do not mix construction/prepass/online/diagnostic evidence")
    if any(r.role == "online_verification" for r in rows):
        raise ValueError("online observations are not construction class votes")
    for row in rows:
        validate_observation_trace(row, allow_diagnostic=allow_diagnostic)
        if row.trace["classes32"] != list(classes):
            raise ValueError("class ordering changed across observations")
    # Pixel masks differ between views, but physical region and scene must agree.
    for key in ("region_version", "member_sha256", "scene_revision", "condition", "source_kind"):
        if len({r.region_binding[key] for r in rows}) > 1:
            raise ValueError("cannot inherit semantic support across object/member/condition versions")
    valid = [r for r in rows if r.detail_cos is not None]
    if any(not cameras_independent(a.camera, b.camera) for a, b in combinations(valid, 2)):
        raise ValueError("semantic support cameras must be pairwise independent")
    count = len(valid)
    vector = None if not count else np.mean(np.stack([(r.detail_cos + r.context_cos) / 2 for r in valid]), axis=0, dtype=np.float64)
    result = {"schema": SEMANTIC_VERSION, "status": "unknown", "reason": "no_valid_observation",
              "N": count, "planned_camera_count": len(rows), "unknown_camera_count": len(rows) - count,
              "final_class": None, "score": None, "scores": {c: None for c in classes},
              "per_camera": [], "calibrated_probability": False,
              "evidence_sha256": [r.trace["evidence_sha256"] for r in rows]}
    winners = []
    for row in valid:
        dv = (row.detail_cos + row.context_cos) / 2
        indices = np.flatnonzero(dv == dv.max())
        winner = classes[int(indices[0])] if len(indices) == 1 else None
        winners.append(winner)
        result["per_camera"].append({"camera_uid": row.camera.camera_uid,
            "d": {c: float(v) for c, v in zip(classes, dv)}, "unique_top1": winner})
    if count:
        scores = (1 + vector) / 2
        result["scores"] = {c: float(v) for c, v in zip(classes, scores)}
        indices = np.flatnonzero(vector == vector.max())
        leader = classes[int(indices[0])] if len(indices) == 1 else None
        if count < 2:
            result["reason"] = "fewer_than_two_independent_valid_cameras"
        elif leader is None or any(w is None for w in winners):
            result["reason"] = "exact_class_score_tie"
        elif any(w != leader for w in winners):
            result["reason"] = "valid_camera_top1_disagreement"
        else:
            result.update(status="accepted" if leader in saga20 else "out_of_scope", reason="unanimous_unique_top1",
                          final_class=leader, score=float(scores[int(indices[0])]))
    result["result_sha256"] = digest(result)
    return result

"""Immutable scene data used by the frozen B0 loader."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from hashlib import sha256
import json
from typing import Any

import numpy as np


def _ids(values) -> tuple[int, ...]:
    raw = tuple(values)
    if any(isinstance(v, (bool, np.bool_)) or int(v) != v or int(v) < 0 for v in raw):
        raise ValueError("members must be nonnegative integer Gaussian IDs")
    return tuple(sorted(set(int(v) for v in raw)))


def _names(values) -> tuple[str, ...]:
    result = tuple(sorted(set(str(v) for v in values)))
    if any(not v for v in result):
        raise ValueError("empty identity")
    return result


def _pairs(values) -> tuple[tuple[str, str], ...]:
    result = set()
    for row in values:
        if len(row) != 2 or not all(row) or str(row[0]) == str(row[1]):
            raise ValueError("independent pairs require two distinct camera IDs")
        result.add(tuple(sorted(map(str, row))))
    return tuple(sorted(result))


def _plain(value):
    if is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, np.ndarray):
        return {"dtype": value.dtype.str, "shape": list(value.shape),
                "sha256": sha256(value.tobytes(order="C")).hexdigest()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _digest(value) -> str:
    return sha256(json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf8")).hexdigest()


def _immutable(value):
    if isinstance(value, dict):
        return tuple((str(k), _immutable(v)) for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_immutable(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_immutable(v) for v in value))
    if isinstance(value, np.ndarray):
        return _immutable(value.tolist())
    return value


def _tuple_fields(instance, *names):
    for name in names:
        object.__setattr__(instance, name, _immutable(getattr(instance, name)))


def member_sha256(members) -> str:
    return _digest({"gaussian_ids": _ids(members)})


@dataclass(frozen=True)
class DirectObservation:
    observation_id: str
    camera_id: str
    hard_ids: tuple[int, ...] = ()
    alpha_ids: tuple[int, ...] = ()
    negative_ids: tuple[int, ...] = ()
    role: str = "construction"
    source_sha256: str = ""

    def __post_init__(self):
        if self.role != "construction" or not self.observation_id or not self.camera_id:
            raise ValueError("only identified construction observations generate members")
        for name in ("hard_ids", "alpha_ids", "negative_ids"):
            object.__setattr__(self, name, _ids(getattr(self, name)))


@dataclass(frozen=True)
class OwnershipGroup:
    previous_owners: tuple[str | None, ...]
    members: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "previous_owners", tuple(self.previous_owners))
        object.__setattr__(self, "members", _ids(self.members))
        if (not self.previous_owners or self.previous_owners[-1] is not None
                or any(v is None or not isinstance(v, str) or not v
                       for v in self.previous_owners[:-1])):
            raise ValueError("ownership history must end with original background")


@dataclass(frozen=True)
class SceneObject:
    uid: str
    identity_version: str
    members: tuple[int, ...]
    class_name: str
    score: float
    observations: tuple[DirectObservation, ...] = ()
    ownership: tuple[OwnershipGroup, ...] = ()
    last_receipt_id: str | None = None
    independent_pairs: tuple[tuple[str, str], ...] = ()
    source_camera_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if any(not isinstance(v, str) or not v for v in (self.uid, self.identity_version, self.class_name)):
            raise ValueError("object identity and class required")
        if not np.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("object score must be finite in [0,1]")
        object.__setattr__(self, "members", _ids(self.members))
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "ownership", tuple(self.ownership))
        object.__setattr__(self, "independent_pairs", _pairs(self.independent_pairs))
        object.__setattr__(self, "source_camera_ids", _names(self.source_camera_ids))
        _validate_observations(self.observations)
        if self.ownership:
            flattened = tuple(v for row in self.ownership for v in row.members)
            if len(set(flattened)) != len(flattened) or _ids(flattened) != self.members:
                raise ValueError("ownership history must partition actual members")

    @property
    def sha256(self):
        return _digest(self)

    @property
    def member_sha256(self):
        return member_sha256(self.members)


def _validate_observations(observations):
    if any(not isinstance(row, DirectObservation) for row in observations):
        raise ValueError("untyped or nonconstruction direct evidence")
    cameras = [row.camera_id for row in observations]
    identities = [row.observation_id for row in observations]
    if len(set(cameras)) != len(cameras) or len(set(identities)) != len(identities):
        raise ValueError("one adopted observation per camera and unique observation IDs")


@dataclass(frozen=True)
class Reason:
    code: str
    uid: str = ""
    point_ids: tuple[int, ...] = ()
    detail: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "point_ids", _ids(self.point_ids))
        object.__setattr__(self, "detail", _immutable(self.detail))


@dataclass(frozen=True)
class OnlineCheck:
    check_id: str
    camera_id: str
    status: str
    precision: float | None
    recall: float | None
    projected_pixels: int
    source_sha256: str
    role: str = "online_verification"

    def __post_init__(self):
        if self.role != "online_verification" or not all((self.check_id, self.camera_id, self.source_sha256)):
            raise ValueError("online check identity and role required")
        if self.status not in {"pass", "reject", "incomplete", "unknown"}:
            raise ValueError("unknown online verification status")
        if self.projected_pixels < 0 or int(self.projected_pixels) != self.projected_pixels:
            raise ValueError("invalid projected pixel count")
        if any(v is not None and (not np.isfinite(v) or not 0 <= v <= 1)
               for v in (self.precision, self.recall)):
            raise ValueError("invalid precision/recall")


@dataclass(frozen=True)
class VerificationRecord:
    uid: str
    identity_version: str
    member_sha256: str
    base_scene_sha256: str
    class_name: str | None
    score: float | None
    semantic_camera_ids: tuple[str, ...]
    semantic_source_sha256: str
    checks: tuple[OnlineCheck, ...]
    source_camera_ids: tuple[str, ...]
    semantic_status: str = "accepted"

    def __post_init__(self):
        for key in ("semantic_camera_ids", "source_camera_ids"):
            object.__setattr__(self, key, _names(getattr(self, key)))
        object.__setattr__(self, "checks", tuple(self.checks))
        if self.semantic_status not in {"accepted", "unknown", "out_of_scope"}:
            raise ValueError("invalid semantic result")
        if self.score is not None and (not np.isfinite(self.score) or not 0 <= self.score <= 1):
            raise ValueError("invalid semantic score")
        if len({v.check_id for v in self.checks}) != len(self.checks):
            raise ValueError("duplicate online check ID")


@dataclass(frozen=True)
class TransactionEvent:
    event_id: str
    kind: str
    group_id: str
    status: str
    reasons: tuple[Reason, ...] = ()
    consumed_checks: tuple[tuple[str, str, str, str], ...] = ()
    receipt_ids: tuple[str, ...] = ()
    consumed_online_views: tuple[tuple[str, str, str], ...] = ()

    def __post_init__(self):
        _tuple_fields(self, "reasons", "consumed_checks", "receipt_ids", "consumed_online_views")


@dataclass(frozen=True)
class CommitReceipt:
    receipt_id: str
    batch_id: str
    group_id: str
    proposal_uids: tuple[str, ...]
    before: tuple[SceneObject, ...]
    after: tuple[SceneObject, ...]
    write_uids: tuple[str, ...]
    read_object_versions: tuple[tuple[str, str], ...]
    depends_on: tuple[str, ...]
    verifications: tuple[VerificationRecord, ...]
    guards: tuple[Reason, ...]
    before_scene_sha256: str
    after_content_sha256: str

    def __post_init__(self):
        _tuple_fields(self, "proposal_uids", "before", "after", "write_uids", "read_object_versions",
                      "depends_on", "verifications", "guards")


@dataclass(frozen=True)
class SceneSnapshot:
    scene_id: str
    xyz_m: np.ndarray
    objects: tuple[SceneObject, ...]
    b0_objects: tuple[SceneObject, ...]
    receipts: tuple[CommitReceipt, ...] = ()
    events: tuple[TransactionEvent, ...] = ()
    counterevidence: tuple[CounterEvidence, ...] = ()

    def __post_init__(self):
        xyz = np.asarray(self.xyz_m, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all() or not self.scene_id:
            raise ValueError("finite metric Nx3 coordinates and scene identity required")
        # An immutable bytes backing store cannot be made writeable by callers.
        backing = xyz
        while isinstance(backing, np.ndarray) and backing.base is not None:
            backing = backing.base
        frozen = xyz if not xyz.flags.writeable and isinstance(backing, bytes) else np.frombuffer(
            xyz.tobytes(order="C"), dtype=np.float64).reshape(xyz.shape)
        object.__setattr__(self, "xyz_m", frozen)
        for key in ("objects", "b0_objects"):
            rows = tuple(sorted(getattr(self, key), key=lambda row: row.uid))
            _validate_objects(rows, len(xyz))
            object.__setattr__(self, key, rows)
        for key in ("receipts", "events", "counterevidence"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        if len({r.receipt_id for r in self.receipts}) != len(self.receipts):
            raise ValueError("duplicate receipt ID")
        if len({e.event_id for e in self.events}) != len(self.events):
            raise ValueError("duplicate event ID")

    @property
    def content_sha256(self):
        return _digest((self.scene_id, self.xyz_m, self.objects, self.b0_objects))

    @property
    def sha256(self):
        return _digest(self)

    def object(self, uid):
        return next((row for row in self.objects if row.uid == uid), None)


def _validate_objects(objects, count):
    if len({row.uid for row in objects}) != len(objects):
        raise ValueError("duplicate persistent object UID")
    seen = set()
    for row in objects:
        if row.members and row.members[-1] >= count:
            raise ValueError("Gaussian member outside frozen scene")
        if seen.intersection(row.members):
            raise ValueError("scene objects have conflicting ownership")
        seen.update(row.members)
        if row.members and not row.ownership:
            raise ValueError("missing acquisition-before ownership history")


def make_scene(scene_id, xyz_m, objects) -> SceneSnapshot:
    originals = tuple(replace(row, ownership=(OwnershipGroup((None,), row.members),),
                              last_receipt_id=None) for row in objects)
    return SceneSnapshot(scene_id, xyz_m, originals, originals)


@dataclass(frozen=True)
class CounterEvidence:
    evidence_id: str
    uid: str
    identity_version: str
    camera_id: str
    source_sha256: str
    visible_anchor_excluded: bool
    role: str = "online_verification"

    def __post_init__(self):
        if self.role not in {"construction", "online_verification"} or not all((self.evidence_id, self.uid,
                self.identity_version, self.camera_id, self.source_sha256)):
            raise ValueError("explicit identified construction/online counterevidence required")

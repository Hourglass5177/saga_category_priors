"""Separate immutable source, raw evidence, proposal and committed-state identities."""
from __future__ import annotations

from dataclasses import dataclass, replace
import numbers

from .artifacts import digest


def ids(values) -> tuple[int, ...]:
    values = tuple(values)
    if any(isinstance(value, bool) or not isinstance(value, numbers.Integral) for value in values):
        raise ValueError("Gaussian IDs must be integers, never bool/float")
    converted = tuple(sorted(set(int(value) for value in values)))
    if any(value < 0 for value in converted):
        raise ValueError("negative Gaussian ID")
    return converted


@dataclass(frozen=True)
class CandidateSource:
    candidate_uid: str
    parent_uids: tuple[str, ...]
    anchor_ids: tuple[int, ...]
    support_ids: tuple[int, ...]
    source_camera_ids: tuple[str, ...]
    input_sha256: str
    bank_version: str = "C0"

    def __post_init__(self):
        object.__setattr__(self, "anchor_ids", ids(self.anchor_ids))
        object.__setattr__(self, "support_ids", ids(self.support_ids))
        object.__setattr__(self, "parent_uids", tuple(sorted(set(self.parent_uids))))
        object.__setattr__(self, "source_camera_ids", tuple(sorted(set(self.source_camera_ids))))
        if not self.candidate_uid or not self.parent_uids or not self.input_sha256:
            raise ValueError("candidate source identity and lineage are required")
        if not set(self.anchor_ids) <= set(self.support_ids):
            raise ValueError("source anchor outside support")


@dataclass(frozen=True)
class RawEvidenceRef:
    observation_uid: str
    camera_uid: str
    role: str
    identity_uid: str
    payload_sha256: str
    round_index: int
    source_kind: str = "model"

    def __post_init__(self):
        if self.role not in {"prepass", "construction", "online_verification", "offline_evaluation"}:
            raise ValueError("unknown observation role")
        if self.source_kind != "model" and self.role != "offline_evaluation":
            raise ValueError("human/GT cannot be formal algorithm evidence")
        if self.round_index not in {0, 1, 2} or not all((self.observation_uid, self.camera_uid,
                                                      self.identity_uid, self.payload_sha256)):
            raise ValueError("incomplete raw observation identity")


@dataclass(frozen=True)
class EvidenceVersion:
    object_uid: str
    identity_uid: str
    version: int
    observations: tuple[RawEvidenceRef, ...] = ()
    revoked_observation_uids: tuple[str, ...] = ()
    revocation_history: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "revoked_observation_uids", tuple(self.revoked_observation_uids))
        object.__setattr__(self, "revocation_history", tuple(tuple(row) for row in self.revocation_history))
        unique = [row.observation_uid for row in self.observations]
        if len(set(unique)) != len(unique):
            raise ValueError("raw observation IDs must be unique")
        if not set(self.revoked_observation_uids) <= set(unique):
            raise ValueError("cannot revoke an absent observation")

    @property
    def sha256(self) -> str:
        return digest(self)

    def active(self, role="construction") -> tuple[RawEvidenceRef, ...]:
        return tuple(row for row in self.observations if row.role == role
                     and row.observation_uid not in self.revoked_observation_uids
                     and row.identity_uid == self.identity_uid)

    def append(self, observations: tuple[RawEvidenceRef, ...]) -> "EvidenceVersion":
        return replace(self, version=self.version + 1, observations=self.observations + tuple(observations))

    def revoke(self, observation_uids, reason: str) -> "EvidenceVersion":
        if not reason:
            raise ValueError("revocation requires evidence-linked reason")
        revoked = tuple(sorted(set(self.revoked_observation_uids) | set(observation_uids)))
        return replace(self, version=self.version + 1, revoked_observation_uids=revoked,
                       revocation_history=self.revocation_history + tuple((uid, reason) for uid in observation_uids))

    def change_identity(self, identity_uid: str) -> "EvidenceVersion":
        """A physical identity switch cannot inherit votes; class is deliberately absent."""
        if not identity_uid:
            raise ValueError("empty identity")
        return replace(self, version=self.version + 1, identity_uid=identity_uid)

"""Class-free observation identity and role-separated decisions; NumPy only.

These are new v1 decision primitives. No legacy reviewer, graph or voter is
imported. A compatible mask clique is evidence of a possible identity, not an
instance probability. Ambiguous cliques are never resolved by semantic class.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


def _readonly(value: Any, dtype=None) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    if array.dtype.hasobject:
        raise ValueError("object arrays are not immutable numeric observations")
    # A flag on an owning ndarray can be flipped back. Immutable bytes cannot.
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _boolean(value: Any, shape=None) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.bool_ or array.ndim != 2 or (shape is not None and array.shape != shape):
        raise ValueError("expected a boolean HxW mask in the registered image geometry")
    return _readonly(array)


@dataclass(frozen=True)
class CameraView:
    camera_uid: str
    view_ray: tuple[float, float, float]
    camera_center: tuple[float, float, float]
    depth: float

    def __post_init__(self):
        ray = np.asarray(self.view_ray, dtype=np.float64)
        center = np.asarray(self.camera_center, dtype=np.float64)
        if (not self.camera_uid or ray.shape != (3,) or center.shape != (3,)
                or not np.isfinite(ray).all() or not np.isfinite(center).all()
                or np.linalg.norm(ray) == 0 or not np.isfinite(self.depth) or self.depth <= 0):
            raise ValueError("camera requires a nonzero ray, finite center and positive depth")
        object.__setattr__(self, "view_ray", tuple(ray / np.linalg.norm(ray)))
        object.__setattr__(self, "camera_center", tuple(center))


def cameras_independent(first: CameraView, second: CameraView) -> bool:
    if first.camera_uid == second.camera_uid:
        return False
    angle = np.degrees(np.arccos(np.clip(np.dot(first.view_ray, second.view_ray), -1., 1.)))
    baseline = np.linalg.norm(np.asarray(first.camera_center) - second.camera_center)
    return bool(angle >= 15. or baseline / min(first.depth, second.depth) >= .05)


@dataclass(frozen=True)
class ViewPanel:
    assignments: tuple[tuple[str, CameraView], ...]
    source_camera_ids: tuple[str, ...]
    omitted_camera_ids: tuple[str, ...]
    feedback_applicable: bool
    can_verify_new_object: bool

    def __post_init__(self):
        assignments = tuple(tuple(row) for row in self.assignments)
        if len(assignments) > 6 or tuple(r for r, _ in assignments) != _PANELS.get(len(assignments)):
            raise ValueError("panel roles differ from the preregistered small-view table")
        views = [view for _, view in assignments]
        if len({v.camera_uid for v in views}) != len(views) or any(not cameras_independent(a, b) for a, b in combinations(views, 2)):
            raise ValueError("panel cameras must be distinct and pairwise independent")
        sources = tuple(sorted(set(str(v) for v in self.source_camera_ids)))
        if any(view.camera_uid in sources for role, view in assignments if role.startswith("H")):
            raise ValueError("source camera cannot become an online verification role")
        if self.feedback_applicable != (len(assignments) >= 5) or self.can_verify_new_object != (len(assignments) >= 3):
            raise ValueError("panel applicability flags disagree with the frozen role table")
        object.__setattr__(self, "assignments", assignments)
        object.__setattr__(self, "source_camera_ids", sources)
        object.__setattr__(self, "omitted_camera_ids", tuple(self.omitted_camera_ids))

    def camera_for(self, role: str) -> CameraView | None:
        return next((camera for name, camera in self.assignments if name == role), None)


_PANELS = {6: ("I1", "I2", "Hmid", "I3", "I4", "Hfinal"),
           5: ("I1", "I2", "Hmid", "I3", "Hfinal"),
           4: ("I1", "I2", "I3", "Hfinal"), 3: ("I1", "I2", "Hfinal"),
           2: ("I1", "I2"), 1: ("I1",), 0: ()}


def fixed_view_panel(ordered_views: Sequence[CameraView], *, source_camera_ids=()) -> ViewPanel:
    """Select the first feasible ordered panel, before any new inference.

    The input order must already be frozen by the caller. All ancestry is kept,
    including cameras outside the six actually used. H can never be a source.
    A longest feasible panel wins; ties follow this input order, not model scores.
    """
    views = tuple(ordered_views)
    if len({v.camera_uid for v in views}) != len(views):
        raise ValueError("duplicate camera in frozen view inventory")
    sources = frozenset(str(v) for v in source_camera_ids)

    def search(roles, selected=(), start=0):
        if len(selected) == len(roles):
            return selected
        remaining_roles = roles[len(selected):]
        if len(views) - start < len(remaining_roles):
            return None
        if sum(v.camera_uid not in sources for v in views[start:]) < sum(r.startswith("H") for r in remaining_roles):
            return None
        role = roles[len(selected)]
        for index in range(start, len(views)):
            candidate = views[index]
            if role.startswith("H") and candidate.camera_uid in sources:
                continue
            if not all(cameras_independent(candidate, old) for old in selected):
                continue
            result = search(roles, selected + (candidate,), index + 1)
            if result is not None:
                return result
        return None

    for count in range(min(6, len(views)), -1, -1):
        selected = search(_PANELS[count])
        if selected is not None:
            selected_ids = {v.camera_uid for v in selected}
            return ViewPanel(tuple(zip(_PANELS[count], selected)), tuple(sorted(sources)),
                             tuple(v.camera_uid for v in views if v.camera_uid not in selected_ids),
                             count >= 5, count >= 3)
    raise AssertionError("empty panel must be feasible")


@dataclass(frozen=True)
class MaskObservation:
    observation_uid: str
    camera: CameraView
    mask: np.ndarray
    contributor_ids: np.ndarray
    max_contribution: np.ndarray
    opacity: np.ndarray
    valid_pixels: np.ndarray
    sam_quality: float = 0.
    proposed_class: str | None = None  # Audit only; never used by identity.

    def __post_init__(self):
        mask = _boolean(self.mask)
        valid = _boolean(self.valid_pixels, mask.shape)
        ids = np.asarray(self.contributor_ids)
        weight = np.asarray(self.max_contribution, dtype=np.float64)
        opacity = np.asarray(self.opacity, dtype=np.float64)
        if (not self.observation_uid or ids.shape != mask.shape or ids.dtype.kind not in "iu"
                or weight.shape != mask.shape or opacity.shape != mask.shape
                or not np.isfinite(weight).all() or not np.isfinite(opacity).all()
                or np.any(weight < 0) or np.any(opacity < 0) or not np.isfinite(self.sam_quality)):
            raise ValueError("invalid contributor observation")
        for key, value in (("mask", mask), ("valid_pixels", valid), ("contributor_ids", ids),
                           ("max_contribution", weight), ("opacity", opacity)):
            object.__setattr__(self, key, _readonly(value))


def reliable_contributors(observation: MaskObservation) -> tuple[frozenset[int], frozenset[int]]:
    ratio = np.divide(observation.max_contribution, observation.opacity,
                      out=np.zeros_like(observation.opacity), where=observation.opacity > 0)
    valid = (observation.valid_pixels & (observation.contributor_ids >= 0)
             & (observation.opacity >= .50) & (ratio >= .50))
    visible = frozenset(int(v) for v in np.unique(observation.contributor_ids[valid]))
    positive = frozenset(int(v) for v in np.unique(observation.contributor_ids[valid & observation.mask]))
    return visible, positive


def _identity_pair(first: MaskObservation, second: MaskObservation, first_support, second_support) -> dict[str, Any]:
    result = dict(first=first.observation_uid, second=second.observation_uid, compatible=False,
                  common_visible=0, intersection=0, union=0, first_positive=0, second_positive=0,
                  jaccard=None, first_coverage=None, second_coverage=None)
    if not cameras_independent(first.camera, second.camera):
        return {**result, "reason": "same_or_nonindependent_camera"}
    visible_a, positive_a = first_support
    visible_b, positive_b = second_support
    common = visible_a & visible_b
    a, b = positive_a & common, positive_b & common
    intersection, union = len(a & b), len(a | b)
    result.update(common_visible=len(common), intersection=intersection, union=union,
                  first_positive=len(a), second_positive=len(b),
                  jaccard=intersection / union if union else None,
                  first_coverage=intersection / len(a) if a else None,
                  second_coverage=intersection / len(b) if b else None)
    if not common or not a or not b:
        return {**result, "reason": "unknown_common_visibility"}
    passed = intersection >= 3 and intersection / union >= .30 and intersection / len(a) >= .50 and intersection / len(b) >= .50
    return {**result, "compatible": passed,
            "reason": "compatible" if passed else "insufficient_pairwise_identity_support"}


def identity_pair(first: MaskObservation, second: MaskObservation) -> dict[str, Any]:
    return _identity_pair(first, second, reliable_contributors(first), reliable_contributors(second))


@dataclass(frozen=True)
class IdentityVerdict:
    status: str
    reason: str
    groups: tuple[tuple[str, ...], ...]
    pair_records: tuple[Mapping[str, Any], ...]
    alternative_groups: tuple[tuple[str, ...], ...] = ()


def identity_consensus(observations: Sequence[MaskObservation]) -> IdentityVerdict:
    """Enumerate maximal all-pair cliques, never graph connected components.

    Multiple cliques may differ only in compatible same-camera boundaries. If
    their entire union is compatible across cameras, this is one identity and
    SAM quality may choose one boundary per camera. Any cross-camera conflict
    remains unknown; quality never chooses between physical interpretations.
    """
    rows = tuple(sorted(observations, key=lambda item: item.observation_uid))
    if len({r.observation_uid for r in rows}) != len(rows):
        raise ValueError("duplicate observation UID")
    adjacent = {i: set() for i in range(len(rows))}
    # Each real image is decoded once, not once per edge of the hypothesis graph.
    supports = [reliable_contributors(row) for row in rows]
    pairs = []
    for a, b in combinations(range(len(rows)), 2):
        record = _identity_pair(rows[a], rows[b], supports[a], supports[b])
        pairs.append(record)
        if record["compatible"]:
            adjacent[a].add(b)
            adjacent[b].add(a)
    groups = []

    def maximal(selected, possible, excluded):
        if not possible and not excluded:
            if len(selected) >= 2:
                groups.append(tuple(sorted(rows[i].observation_uid for i in selected)))
            return
        for vertex in sorted(tuple(possible)):
            maximal(selected | {vertex}, possible & adjacent[vertex], excluded & adjacent[vertex])
            possible.remove(vertex)
            excluded.add(vertex)

    maximal(set(), set(adjacent), set())
    groups = tuple(sorted(set(groups)))
    union_ids = {uid for group in groups for uid in group}
    union_rows = [row for row in rows if row.observation_uid in union_ids]
    index_by_uid = {row.observation_uid: index for index, row in enumerate(rows)}
    same_identity = bool(groups) and all(index_by_uid[b.observation_uid] in adjacent[index_by_uid[a.observation_uid]]
        for a, b in combinations(union_rows, 2) if a.camera.camera_uid != b.camera.camera_uid)
    if same_identity:
        adopted = tuple(sorted(min((row for row in union_rows if row.camera.camera_uid == camera),
                                   key=lambda row: (-row.sam_quality, row.observation_uid)).observation_uid
                               for camera in sorted({row.camera.camera_uid for row in union_rows})))
        return IdentityVerdict("accepted", "unique_pairwise_identity", (adopted,), tuple(pairs), groups)
    return IdentityVerdict("unknown", "multiple_identity_explanations" if groups else "insufficient_identity_observation",
                           groups, tuple(pairs), groups)


@dataclass(frozen=True)
class SemanticObservation:
    observation_uid: str
    camera: CameraView
    role: str
    class_scores: Mapping[str, float] = field(default_factory=dict)
    observable: bool = True
    unknown_reason: str | None = None

    def __post_init__(self):
        if not self.observation_uid or self.role not in {"construction", "prepass", "online_verification", "offline_evaluation", "diagnostic"}:
            raise ValueError("semantic observation requires an explicit input role")
        scores = {str(key): float(value) for key, value in self.class_scores.items()}
        if any(not np.isfinite(v) or not 0 < v <= 1 for v in scores.values()):
            raise ValueError("only positive, matched per-class scores belong in class_scores")
        if not self.observable and scores:
            raise ValueError("unobservable camera cannot supply semantic scores")
        if self.unknown_reason is not None and scores:
            raise ValueError("unknown semantic observation cannot also supply accepted scores")
        object.__setattr__(self, "class_scores", MappingProxyType(scores))


def semantic_decision(observations: Sequence[SemanticObservation], *, classes32: Sequence[str],
                      saga20: Sequence[str]) -> dict[str, Any]:
    classes = tuple(classes32)
    if (len(classes) != 32 or len(set(classes)) != 32 or len(saga20) != 20
            or len(set(saga20)) != 20 or not set(saga20) <= set(classes)):
        raise ValueError("freeze exactly 32 unique classes and the SAGA20 subset")
    active = sorted((r for r in observations if r.role == "construction"), key=lambda r: r.camera.camera_uid)
    if len({r.camera.camera_uid for r in active}) != len(active):
        raise ValueError("at most one adopted construction observation per camera")
    if any(set(r.class_scores) - set(classes) for r in active):
        raise ValueError("semantic class outside the frozen class table")
    if any(not cameras_independent(a.camera, b.camera) for a, b in combinations(active, 2)):
        raise ValueError("semantic construction panel must be pairwise independent")
    valid = [r for r in active if r.class_scores]
    count = len(valid)
    vector = {c: float(np.sum([r.class_scores.get(c, 0.) for r in valid], dtype=np.float64) / count)
              if count else None for c in classes}
    supports = {c: sum(c in r.class_scores for r in valid) for c in classes}
    reasons: dict[str, int] = {}
    for row in active:
        if not row.class_scores:
            reason = row.unknown_reason or "no_qualified_semantic_proposal"
            reasons[reason] = reasons.get(reason, 0) + 1
    result = dict(status="unknown", reason="no_qualified_semantic_proposal", final_class=None,
                  score=None, N=count, scores=vector, supporting_camera_counts=supports,
                  construction_camera_count=len(active), observable_camera_count=sum(r.observable for r in active),
                  unknown_camera_count=len(active) - count, unknown_reasons=reasons,
                  excluded_observation_uids=sorted(r.observation_uid for r in observations if r.role != "construction"),
                  score_definition="mean_matched_class_token_score_over_qualified_construction_cameras",
                  calibrated_probability=False)
    if not count:
        return result
    best = max(vector.values())
    winners = [c for c in classes if vector[c] == best]
    if len(winners) != 1:
        return {**result, "reason": "exact_class_score_tie"}
    winner = winners[0]
    if supports[winner] < 2 or best < .30:
        return {**result, "reason": "insufficient_independent_class_support", "leading_class": winner}
    return {**result, "status": "accepted" if winner in saga20 else "out_of_scope",
            "reason": "unique_supported_class", "final_class": winner, "score": best}


def verify_projection(projected_mask: Any, reference_masks: Sequence[Any], valid_pixels: Any, *,
                      identity_status: str, projection_complete: bool = True,
                      visible_anchor_mask: Any | None = None) -> dict[str, Any]:
    """Evaluate all viable independent RGB boundaries, without selecting a best.

    The generator that supplies reference_masks must not receive projected_mask.
    Low projected precision rejects this *proposal*, not the physical identity.
    Identity refutation still requires two independent anchor-exclusion views.
    """
    projection = _boolean(projected_mask)
    valid = _boolean(valid_pixels, projection.shape)
    references = tuple(_boolean(m, projection.shape) for m in reference_masks)
    anchor = _boolean(visible_anchor_mask, projection.shape) if visible_anchor_mask is not None else None
    records = []
    for index, mask in enumerate(references):
        p, o = projection & valid, mask & valid
        intersection, p_count, o_count = int((p & o).sum()), int(p.sum()), int(o.sum())
        precision = intersection / p_count if p_count else None
        recall = intersection / o_count if o_count else None
        status = ("unknown" if min(p_count, o_count) < 4 else
                  "pass" if precision >= .80 and recall >= .50 else
                  "incomplete" if precision >= .80 else "reject")
        records.append(dict(reference_index=index, intersection=intersection, projected_pixels=p_count,
                            visible_object_pixels=o_count, precision=precision, recall=recall, status=status))
    result = dict(status="unknown", reason="no_independent_identity_mask", reference_results=records,
                  visible_anchor_excluded=False, identity_refuted=False,
                  projection_outside_observed_pixels=int((projection & ~valid).sum()))
    if identity_status != "accepted":
        return {**result, "reason": "identity_unresolved"}
    if not projection_complete or np.any(projection & ~valid):
        return {**result, "reason": "crop_or_projection_insufficient"}
    if not records:
        return result
    statuses = {r["status"] for r in records}
    if anchor is not None:
        visible_anchor = anchor & valid
        result["visible_anchor_excluded"] = bool(visible_anchor.sum() >= 4 and
                                                 all(not np.any(mask & visible_anchor) for mask in references))
    if len(statuses) == 1:
        status = next(iter(statuses))
        return {**result, "status": status, "reason": {"pass": "all_identity_boundaries_pass",
                "incomplete": "all_identity_boundaries_incomplete", "reject": "all_identity_boundaries_low_precision",
                "unknown": "insufficient_observed_pixels"}[status]}
    return {**result, "reason": "boundary_alternatives_disagree"}

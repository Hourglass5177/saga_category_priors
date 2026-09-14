"""Registered camera geometry; no model, class, GT or legacy controller."""
from dataclasses import dataclass
from itertools import combinations
from collections.abc import Mapping
import numbers
import numpy as np
from .artifacts import digest, readonly


def ids(values):
    result = tuple(values)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Integral) or v < 0 for v in result):
        raise ValueError("Gaussian IDs require nonnegative integers")
    return tuple(sorted(set(map(int, result))))


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
            raise ValueError("finite camera geometry and positive depth required")
        object.__setattr__(self, "view_ray", tuple(ray / np.linalg.norm(ray)))
        object.__setattr__(self, "camera_center", tuple(center))
        object.__setattr__(self, "depth", float(self.depth))

    @classmethod
    def from_record(cls, record):
        """Restore a hash-bound computed camera without normalizing it again.

        The caller must verify its enclosing evidence identity. Unit length is
        an input validity check only; every recorded float is retained exactly.
        New camera calculations continue to use the ordinary constructor.
        """
        if (not isinstance(record, Mapping) or set(record) !=
                {"camera_uid", "view_ray", "camera_center", "depth"}
                or not isinstance(record["camera_uid"], str) or not record["camera_uid"]):
            raise ValueError("strict recorded camera fields required")
        vectors = []
        for name in ("view_ray", "camera_center"):
            values = record[name]
            if (not isinstance(values, (tuple, list)) or len(values) != 3
                    or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real)
                           for v in values)):
                raise ValueError("recorded camera vectors require three real values")
            vector = tuple(float(v) for v in values)
            if not np.isfinite(vector).all():
                raise ValueError("finite recorded camera required")
            vectors.append(vector)
        depth = record["depth"]
        if (isinstance(depth, (bool, np.bool_)) or not isinstance(depth, numbers.Real)
                or not np.isfinite(depth) or depth <= 0):
            raise ValueError("positive recorded camera depth required")
        if abs(float(np.linalg.norm(vectors[0])) - 1.) > 4 * np.finfo(np.float64).eps:
            raise ValueError("recorded camera ray must already be normalized")
        obj = object.__new__(cls)
        for name, value in zip(("camera_uid", "view_ray", "camera_center", "depth"),
                               (record["camera_uid"], *vectors, float(depth))):
            object.__setattr__(obj, name, value)
        return obj


def cameras_independent(first, second):
    if first.camera_uid == second.camera_uid:
        return False
    angle = np.degrees(np.arccos(np.clip(np.dot(first.view_ray, second.view_ray), -1., 1.)))
    baseline = np.linalg.norm(np.asarray(first.camera_center) - second.camera_center)
    return bool(angle >= 15. or baseline / min(first.depth, second.depth) >= .05)


PANELS = {6: ("I1", "I2", "Hmid", "I3", "I4", "Hfinal"),
          5: ("I1", "I2", "Hmid", "I3", "Hfinal"),
          4: ("I1", "I2", "I3", "Hfinal"), 3: ("I1", "I2", "Hfinal"),
          2: ("I1", "I2"), 1: ("I1",), 0: ()}


@dataclass(frozen=True)
class ViewPanel:
    assignments: tuple
    source_camera_ids: tuple[str, ...]
    ranking: tuple

    def __post_init__(self):
        object.__setattr__(self, "assignments", tuple(tuple(x) for x in self.assignments))
        object.__setattr__(self, "source_camera_ids", tuple(sorted(set(self.source_camera_ids))))
        object.__setattr__(self, "ranking", tuple(tuple(x) for x in self.ranking))
        views = [v for _, v in self.assignments]
        if tuple(r for r, _ in self.assignments) != PANELS.get(len(views)):
            raise ValueError("unregistered role table")
        if any(not cameras_independent(a, b) for a, b in combinations(views, 2)):
            raise ValueError("panel not pairwise independent")
        if any(r.startswith("H") and v.camera_uid in self.source_camera_ids for r, v in self.assignments):
            raise ValueError("source camera cannot verify its dependent version")

    def camera_for(self, role):
        return next((v for r, v in self.assignments if r == role), None)

    @property
    def sha256(self):
        return digest(self)


def select_panel(views, visible_counts, image_names, source_camera_ids=()):
    views = tuple(views)
    if len({v.camera_uid for v in views}) != len(views):
        raise ValueError("duplicated physical camera")
    if set(visible_counts) != {v.camera_uid for v in views} or set(image_names) != set(visible_counts):
        raise ValueError("complete frozen visibility and name inventory required")
    if any(not isinstance(n, numbers.Integral) or isinstance(n, bool) or n < 0 for n in visible_counts.values()):
        raise ValueError("invalid reliable Gaussian visibility count")
    order = sorted(views, key=lambda v: (-visible_counts[v.camera_uid], image_names[v.camera_uid], v.camera_uid))
    available = [v for v in order if visible_counts[v.camera_uid] > 0]
    sources = set(source_camera_ids)

    def search(roles, selected=(), start=0):
        if len(selected) == len(roles):
            return selected
        if len(available) - start < len(roles) - len(selected):
            return None
        role = roles[len(selected)]
        for j in range(start, len(available)):
            view = available[j]
            if role.startswith("H") and view.camera_uid in sources:
                continue
            if all(cameras_independent(view, old) for old in selected):
                answer = search(roles, selected + (view,), j + 1)
                if answer is not None:
                    return answer
        return None

    for n in range(min(6, len(available)), -1, -1):
        selected = search(PANELS[n])
        if selected is not None:
            return ViewPanel(tuple(zip(PANELS[n], selected)), tuple(sources),
                             tuple((v.camera_uid, int(visible_counts[v.camera_uid]), image_names[v.camera_uid]) for v in order))
    raise AssertionError("empty panel must be feasible")


def reliable_pixels(contributor_ids, max_contribution, opacity, valid):
    idx, maximum, total, valid = map(np.asarray, (contributor_ids, max_contribution, opacity, valid))
    if idx.ndim != 2 or idx.dtype.kind not in "iu" or valid.dtype != np.bool_ or any(x.shape != idx.shape for x in (maximum, total, valid)):
        raise ValueError("registered HxW contributor/valid geometry required")
    if not np.isfinite(maximum).all() or not np.isfinite(total).all() or np.any(maximum < 0) or np.any(total < 0):
        raise ValueError("invalid contribution measurement")
    ratio = np.divide(maximum, total, out=np.zeros_like(total, dtype=float), where=total > 0)
    return readonly(valid & (idx >= 0) & (total >= .5) & (ratio >= .5))


def member_projection(members, contributor_ids, max_contribution, opacity, valid):
    reliable = reliable_pixels(contributor_ids, max_contribution, opacity, valid)
    return readonly(reliable & np.isin(contributor_ids, ids(members)))

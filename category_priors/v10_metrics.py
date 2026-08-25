from __future__ import annotations

"""Offline root-cause metrics for the V10 ObjectBank audit.

The module deliberately has no renderer, training, replay, or official ScanNet
evaluator dependency.  It consumes frozen Gaussian supports and accepted
fragment edges, then projects them into GT-point space for diagnostics only.
Ground truth therefore cannot influence candidate construction.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .evaluator import apply_transform, load_ground_truth_npz, load_ply_xyz
from .io import load_json, read_rows, write_json, write_rows
from .runner import load_scene_runtime_manifest
from .v9_evaluation import score_iou_spearman


V10_SCHEMA = "saga-v10-objectbank-audit-v1"
V10_STAGE_ORDER = (
    "single_full",
    "single_core",
    "component_full_union",
    "component_core_union",
    "pre_conflict",
    "post_conflict",
    "unique_ownership",
    "final_candidate",
)

# Association precision must not hide low-quality accepted edges behind an
# ``unknown`` label.  A fragment is identifiable whenever it intersects any GT
# object; IoU and purity remain continuous diagnostics only.
ASSOCIATION_MIN_INTERSECTION = 1
ASSOCIATION_MIN_IOU = 0.0
ASSOCIATION_MIN_PURITY = 0.0


def _ids(values: np.ndarray | Sequence[int], *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    try:
        result = raw.astype(np.int64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must contain integers") from exc
    if not np.array_equal(raw, result) or np.any(result < 0):
        raise TypeError(f"{name} must contain non-negative integers")
    result = np.unique(result)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class V10GroundTruthObject:
    scene_id: str
    class_name: str
    instance_id: int
    point_ids: np.ndarray
    official_valid: bool = True
    size_bin: str | None = None

    def __post_init__(self) -> None:
        scene_id = str(self.scene_id).strip()
        class_name = str(self.class_name).strip()
        instance_id = int(self.instance_id)
        size_bin = None if self.size_bin is None else str(self.size_bin).strip().lower()
        if not scene_id or not class_name:
            raise ValueError("GT scene_id and class_name must be non-empty")
        if instance_id < 0:
            raise ValueError("GT instance_id must be non-negative")
        if size_bin not in {None, "tiny", "small"}:
            raise ValueError("V10 GT size_bin must be tiny, small, or None")
        point_ids = _ids(self.point_ids, name="GT point_ids")
        if not len(point_ids):
            raise ValueError("GT object must contain at least one point")
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "class_name", class_name)
        object.__setattr__(self, "instance_id", instance_id)
        object.__setattr__(self, "point_ids", point_ids)
        object.__setattr__(self, "official_valid", bool(self.official_valid))
        object.__setattr__(self, "size_bin", size_bin)

    @property
    def identity(self) -> tuple[str, str, int]:
        return self.scene_id, self.class_name, self.instance_id


@dataclass(frozen=True)
class V10FragmentSupport:
    scene_id: str
    fragment_id: int
    gaussian_ids: np.ndarray

    def __post_init__(self) -> None:
        scene_id = str(self.scene_id).strip()
        fragment_id = int(self.fragment_id)
        if not scene_id:
            raise ValueError("fragment scene_id must be non-empty")
        if fragment_id < 0:
            raise ValueError("fragment_id must be non-negative")
        gaussian_ids = _ids(self.gaussian_ids, name="fragment gaussian_ids")
        if not len(gaussian_ids):
            raise ValueError("fragment support must be non-empty")
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "fragment_id", fragment_id)
        object.__setattr__(self, "gaussian_ids", gaussian_ids)


@dataclass(frozen=True)
class V10AcceptedEdge:
    scene_id: str
    left_fragment_id: int
    right_fragment_id: int
    left_frame_id: int | None = None
    right_frame_id: int | None = None
    kind: str | None = None
    score: float | None = None
    shared: int | None = None
    p0_overlap: float | None = None
    left_coverage: float | None = None
    right_coverage: float | None = None
    row_margin: float | None = None
    column_margin: float | None = None
    component_support_ratio: float | None = None
    strong: bool | None = None
    cycle_supported: bool | None = None

    def __post_init__(self) -> None:
        scene_id = str(self.scene_id).strip()
        left = int(self.left_fragment_id)
        right = int(self.right_fragment_id)
        if not scene_id:
            raise ValueError("edge scene_id must be non-empty")
        if left < 0 or right < 0 or left == right:
            raise ValueError("accepted edge requires two distinct non-negative IDs")
        left_frame = None if self.left_frame_id is None else int(self.left_frame_id)
        right_frame = None if self.right_frame_id is None else int(self.right_frame_id)
        if (left_frame is None) != (right_frame is None):
            raise ValueError("accepted edge frame IDs must be both present or both absent")
        if left_frame is not None and (left_frame < 0 or right_frame < 0):
            raise ValueError("accepted edge frame IDs must be non-negative")
        numeric_names = (
            "score",
            "p0_overlap",
            "left_coverage",
            "right_coverage",
            "row_margin",
            "column_margin",
            "component_support_ratio",
        )
        numeric = {
            name: None if getattr(self, name) is None else float(getattr(self, name))
            for name in numeric_names
        }
        if any(value is not None and not np.isfinite(value) for value in numeric.values()):
            raise ValueError("accepted edge evidence must be finite")
        if self.shared is not None and int(self.shared) < 0:
            raise ValueError("accepted edge shared support must be non-negative")
        swapped = left > right
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "left_fragment_id", min(left, right))
        object.__setattr__(self, "right_fragment_id", max(left, right))
        object.__setattr__(self, "left_frame_id", right_frame if swapped else left_frame)
        object.__setattr__(self, "right_frame_id", left_frame if swapped else right_frame)
        object.__setattr__(
            self, "left_coverage", numeric["right_coverage"] if swapped else numeric["left_coverage"]
        )
        object.__setattr__(
            self, "right_coverage", numeric["left_coverage"] if swapped else numeric["right_coverage"]
        )
        object.__setattr__(
            self, "row_margin", numeric["column_margin"] if swapped else numeric["row_margin"]
        )
        object.__setattr__(
            self, "column_margin", numeric["row_margin"] if swapped else numeric["column_margin"]
        )
        for name in ("score", "p0_overlap", "component_support_ratio"):
            object.__setattr__(self, name, numeric[name])
        object.__setattr__(self, "shared", None if self.shared is None else int(self.shared))
        object.__setattr__(self, "kind", None if self.kind is None else str(self.kind))


@dataclass(frozen=True)
class V10StageCandidate:
    scene_id: str
    stage: str
    candidate_id: int
    gaussian_ids: np.ndarray
    class_name: str | None = None
    score: float | None = None

    def __post_init__(self) -> None:
        scene_id = str(self.scene_id).strip()
        stage = str(self.stage).strip()
        candidate_id = int(self.candidate_id)
        class_name = None if self.class_name is None else str(self.class_name).strip()
        score = None if self.score is None else float(self.score)
        if not scene_id:
            raise ValueError("candidate scene_id must be non-empty")
        if stage not in V10_STAGE_ORDER:
            raise ValueError(f"unsupported V10 funnel stage: {stage}")
        if candidate_id < 0:
            raise ValueError("candidate_id must be non-negative")
        if class_name == "":
            raise ValueError("candidate class_name cannot be empty")
        if score is not None and (not np.isfinite(score) or not 0.0 <= score <= 1.0):
            raise ValueError("candidate score must be finite and in [0, 1]")
        gaussian_ids = _ids(self.gaussian_ids, name="candidate gaussian_ids")
        if not len(gaussian_ids):
            raise ValueError("stage candidate support must be non-empty")
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "gaussian_ids", gaussian_ids)
        object.__setattr__(self, "class_name", class_name)
        object.__setattr__(self, "score", score)


@dataclass(frozen=True)
class V10PersistedAuditInputs:
    """Strict metrics-side view of one immutable V10 runner bank."""

    scene_id: str
    fragments: tuple[V10FragmentSupport, ...]
    accepted_edges: tuple[V10AcceptedEdge, ...]
    stage_candidates: tuple[V10StageCandidate, ...]


@dataclass(frozen=True)
class V10ProjectedSupport:
    """A Gaussian support represented in GT-point space plus FP sentinels."""

    support_ids: np.ndarray
    gaussian_count: int
    mapped_gaussian_count: int
    unmapped_gaussian_count: int


@dataclass(frozen=True)
class GaussianGTIndex:
    """Compact GT-point -> nearest-Gaussian inverse mapping.

    A Gaussian with no GT point assigned within the evaluation radius receives
    the globally unique support ID ``gt_point_count + gaussian_id``.  Thus two
    unsupported Gaussians are two false-positive support elements, rather than
    the single shared sentinel used by the old proxy metric.
    """

    gt_point_count: int
    gaussian_count: int
    indptr: np.ndarray
    gt_point_ids: np.ndarray

    def __post_init__(self) -> None:
        gt_count = int(self.gt_point_count)
        gaussian_count = int(self.gaussian_count)
        indptr = np.asarray(self.indptr, dtype=np.int64).copy()
        gt_ids = np.asarray(self.gt_point_ids, dtype=np.int64).copy()
        if gt_count < 0 or gaussian_count < 0:
            raise ValueError("point counts must be non-negative")
        if indptr.shape != (gaussian_count + 1,):
            raise ValueError("indptr length must be gaussian_count + 1")
        if indptr[0] != 0 or np.any(np.diff(indptr) < 0) or indptr[-1] != len(gt_ids):
            raise ValueError("invalid Gaussian-to-GT CSR index")
        if np.any(gt_ids < 0) or np.any(gt_ids >= gt_count):
            raise ValueError("GT point ID outside the declared range")
        indptr.setflags(write=False)
        gt_ids.setflags(write=False)
        object.__setattr__(self, "gt_point_count", gt_count)
        object.__setattr__(self, "gaussian_count", gaussian_count)
        object.__setattr__(self, "indptr", indptr)
        object.__setattr__(self, "gt_point_ids", gt_ids)

    @classmethod
    def from_nearest(
        cls,
        nearest_gaussian: np.ndarray | Sequence[int],
        valid: np.ndarray | Sequence[bool],
        *,
        gaussian_count: int,
    ) -> "GaussianGTIndex":
        nearest = np.asarray(nearest_gaussian, dtype=np.int64)
        valid_mask = np.asarray(valid, dtype=bool)
        if nearest.ndim != 1 or nearest.shape != valid_mask.shape:
            raise ValueError("nearest and valid must be matching one-dimensional arrays")
        count = int(gaussian_count)
        selected = nearest[valid_mask]
        if np.any(selected < 0) or np.any(selected >= count):
            raise ValueError("valid nearest Gaussian ID outside the declared range")
        gt_ids = np.flatnonzero(valid_mask).astype(np.int64)
        counts = np.bincount(selected, minlength=count)
        indptr = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
        order = np.argsort(selected, kind="stable")
        return cls(len(nearest), count, indptr, gt_ids[order])

    def project(self, gaussian_ids: np.ndarray | Sequence[int]) -> V10ProjectedSupport:
        query = _ids(gaussian_ids, name="projected gaussian_ids")
        if np.any(query >= self.gaussian_count):
            raise ValueError("Gaussian ID outside the declared range")
        chunks: list[np.ndarray] = []
        mapped = 0
        for gaussian_id in query:
            start = int(self.indptr[gaussian_id])
            stop = int(self.indptr[gaussian_id + 1])
            if stop > start:
                chunks.append(self.gt_point_ids[start:stop])
                mapped += 1
            else:
                chunks.append(
                    np.asarray(
                        [self.gt_point_count + int(gaussian_id)], dtype=np.int64
                    )
                )
        support = np.sort(np.concatenate(chunks)) if chunks else np.empty(0, np.int64)
        support.setflags(write=False)
        return V10ProjectedSupport(
            support_ids=support,
            gaussian_count=len(query),
            mapped_gaussian_count=mapped,
            unmapped_gaussian_count=len(query) - mapped,
        )


def build_gaussian_gt_index(
    gt_xyz: np.ndarray,
    gaussian_xyz: np.ndarray,
    *,
    radius_m: float = 0.05,
) -> GaussianGTIndex:
    """Build the same GT-nearest-Gaussian projection used by V9 evaluation."""

    gt = np.asarray(gt_xyz, dtype=np.float64)
    gaussians = np.asarray(gaussian_xyz, dtype=np.float64)
    if gt.ndim != 2 or gt.shape[1:] != (3,):
        raise ValueError("gt_xyz must have shape (N, 3)")
    if gaussians.ndim != 2 or gaussians.shape[1:] != (3,):
        raise ValueError("gaussian_xyz must have shape (M, 3)")
    if not len(gaussians):
        raise ValueError("at least one Gaussian is required")
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("radius_m must be positive and finite")
    distances, nearest = cKDTree(gaussians).query(
        gt, k=1, distance_upper_bound=float(radius_m), workers=-1
    )
    valid = np.isfinite(distances) & (nearest < len(gaussians))
    return GaussianGTIndex.from_nearest(
        nearest, valid, gaussian_count=len(gaussians)
    )


def ground_truth_objects_from_arrays(
    scene_id: str,
    semantic: np.ndarray,
    instance: np.ndarray,
    class_names: Sequence[str],
    *,
    min_region_size: int = 100,
    size_bins: Mapping[tuple[str, int], str] | None = None,
) -> list[V10GroundTruthObject]:
    semantic_ids = np.asarray(semantic, dtype=np.int64)
    instance_ids = np.asarray(instance, dtype=np.int64)
    if semantic_ids.ndim != 1 or semantic_ids.shape != instance_ids.shape:
        raise ValueError("semantic and instance must be matching one-dimensional arrays")
    valid = (semantic_ids >= 0) & (instance_ids >= 0)
    objects: list[V10GroundTruthObject] = []
    for class_id, instance_id in sorted(
        set(zip(semantic_ids[valid].tolist(), instance_ids[valid].tolist()))
    ):
        if not 0 <= int(class_id) < len(class_names):
            continue
        point_ids = np.flatnonzero(
            valid & (semantic_ids == class_id) & (instance_ids == instance_id)
        ).astype(np.int64)
        objects.append(
            V10GroundTruthObject(
                scene_id=scene_id,
                class_name=str(class_names[int(class_id)]),
                instance_id=int(instance_id),
                point_ids=point_ids,
                official_valid=len(point_ids) >= int(min_region_size),
                size_bin=(size_bins or {}).get(
                    (str(class_names[int(class_id)]), int(instance_id))
                ),
            )
        )
    return objects


def _match(
    support: np.ndarray,
    ground_truth: Sequence[V10GroundTruthObject],
    *,
    class_name: str | None = None,
    official_only: bool = False,
) -> dict[str, Any]:
    candidates = [
        gt
        for gt in ground_truth
        if (class_name is None or gt.class_name == class_name)
        and (not official_only or gt.official_valid)
    ]
    scored: list[tuple[float, float, int, V10GroundTruthObject]] = []
    for gt in candidates:
        intersection = int(
            len(np.intersect1d(support, gt.point_ids, assume_unique=True))
        )
        union = len(support) + len(gt.point_ids) - intersection
        iou = intersection / union if union else 0.0
        purity = intersection / len(support) if len(support) else 0.0
        scored.append((iou, purity, intersection, gt))
    if not scored:
        return {
            "class_name": None,
            "instance_id": None,
            "iou": 0.0,
            "purity": 0.0,
            "intersection": 0,
            "official_valid": False,
        }
    iou, purity, intersection, gt = min(
        scored,
        key=lambda item: (
            -item[0],
            -item[1],
            -item[2],
            item[3].class_name,
            item[3].instance_id,
        ),
    )
    return {
        "class_name": gt.class_name,
        "instance_id": gt.instance_id,
        "iou": float(iou),
        "purity": float(purity),
        "intersection": int(intersection),
        "official_valid": bool(gt.official_valid),
    }


def _identifiable(match: Mapping[str, Any]) -> bool:
    return bool(int(match["intersection"]) >= ASSOCIATION_MIN_INTERSECTION)


def _support_fields(
    projected: V10ProjectedSupport,
    ground_truth: Sequence[V10GroundTruthObject],
    *,
    class_name: str | None = None,
) -> dict[str, Any]:
    best = _match(projected.support_ids, ground_truth)
    official = _match(projected.support_ids, ground_truth, official_only=True)
    same_class = _match(projected.support_ids, ground_truth, class_name=class_name)
    same_class_official = _match(
        projected.support_ids,
        ground_truth,
        class_name=class_name,
        official_only=True,
    )
    identifiable = _identifiable(best)
    return {
        "gaussian_count": int(projected.gaussian_count),
        "mapped_gaussian_count": int(projected.mapped_gaussian_count),
        "unmapped_gaussian_count": int(projected.unmapped_gaussian_count),
        "unmapped_gaussian_fraction": (
            projected.unmapped_gaussian_count / projected.gaussian_count
            if projected.gaussian_count
            else 0.0
        ),
        "projected_support_count": int(len(projected.support_ids)),
        "identifiable": identifiable,
        "best_gt_class_name": best["class_name"],
        "best_gt_instance_id": best["instance_id"],
        "best_gt_official_valid": bool(best["official_valid"]),
        "best_iou": float(best["iou"]),
        "best_purity": float(best["purity"]),
        "best_intersection": int(best["intersection"]),
        "best_official_gt_class_name": official["class_name"],
        "best_official_gt_instance_id": official["instance_id"],
        "best_official_iou": float(official["iou"]),
        "best_same_class_iou": float(same_class["iou"]),
        "best_same_class_official_iou": float(same_class_official["iou"]),
        "best_same_class_official_gt_class_name": same_class_official["class_name"],
        "best_same_class_official_gt_instance_id": same_class_official["instance_id"],
    }


def accepted_fragment_pair_rows(
    fragments: Sequence[V10FragmentSupport],
    accepted_edges: Sequence[V10AcceptedEdge],
    ground_truth: Sequence[V10GroundTruthObject],
    gaussian_gt_index: GaussianGTIndex,
) -> list[dict[str, Any]]:
    """Classify only actual accepted edges, never transitive component pairs."""

    fragment_by_key = {(row.scene_id, row.fragment_id): row for row in fragments}
    duplicate_count = len(fragments) - len(fragment_by_key)
    if duplicate_count:
        raise ValueError("fragment IDs must be unique within each scene")
    truth_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for fragment in fragments:
        projected = gaussian_gt_index.project(fragment.gaussian_ids)
        scene_gt = [gt for gt in ground_truth if gt.scene_id == fragment.scene_id]
        truth_by_key[(fragment.scene_id, fragment.fragment_id)] = _support_fields(
            projected, scene_gt
        )

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for edge in accepted_edges:
        edge_key = (edge.scene_id, edge.left_fragment_id, edge.right_fragment_id)
        if edge_key in seen:
            raise ValueError(f"duplicate accepted edge: {edge_key}")
        seen.add(edge_key)
        left_key = (edge.scene_id, edge.left_fragment_id)
        right_key = (edge.scene_id, edge.right_fragment_id)
        if left_key not in fragment_by_key or right_key not in fragment_by_key:
            raise KeyError(f"accepted edge references unknown fragment: {edge_key}")
        left = truth_by_key[left_key]
        right = truth_by_key[right_key]
        if not left["identifiable"] or not right["identifiable"]:
            classification = "unknown"
        elif (
            left["best_gt_class_name"],
            left["best_gt_instance_id"],
        ) == (
            right["best_gt_class_name"],
            right["best_gt_instance_id"],
        ):
            classification = "same_gt"
        else:
            classification = "different_gt"
        row: dict[str, Any] = {
            "schema": V10_SCHEMA,
            "row_type": "accepted_fragment_pair",
            "scene_id": edge.scene_id,
            "left_fragment_id": edge.left_fragment_id,
            "right_fragment_id": edge.right_fragment_id,
            "classification": classification,
            "left_frame_id": edge.left_frame_id,
            "right_frame_id": edge.right_frame_id,
            "edge_kind": edge.kind,
            "edge_score": edge.score,
            "edge_shared": edge.shared,
            "p0_overlap": edge.p0_overlap,
            "left_coverage": edge.left_coverage,
            "right_coverage": edge.right_coverage,
            "row_margin": edge.row_margin,
            "column_margin": edge.column_margin,
            "component_support_ratio": edge.component_support_ratio,
            "strong": edge.strong,
            "cycle_supported": edge.cycle_supported,
        }
        for prefix, values in (("left", left), ("right", right)):
            for key in (
                "identifiable",
                "best_gt_class_name",
                "best_gt_instance_id",
                "best_iou",
                "best_purity",
                "best_intersection",
                "gaussian_count",
                "mapped_gaussian_count",
                "unmapped_gaussian_count",
            ):
                row[f"{prefix}_{key}"] = values[key]
        rows.append(row)
    return rows


def stage_funnel_rows(
    candidates: Sequence[V10StageCandidate],
    ground_truth: Sequence[V10GroundTruthObject],
    gaussian_gt_index: GaussianGTIndex,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for candidate in candidates:
        key = (candidate.scene_id, candidate.stage, candidate.candidate_id)
        if key in seen:
            raise ValueError(f"duplicate stage candidate: {key}")
        seen.add(key)
        scene_gt = [gt for gt in ground_truth if gt.scene_id == candidate.scene_id]
        projected = gaussian_gt_index.project(candidate.gaussian_ids)
        fields = _support_fields(projected, scene_gt, class_name=candidate.class_name)
        rows.append(
            {
                "schema": V10_SCHEMA,
                "row_type": "stage_candidate",
                "scene_id": candidate.scene_id,
                "stage": candidate.stage,
                "candidate_id": candidate.candidate_id,
                "class_name": candidate.class_name,
                "score": candidate.score,
                **fields,
            }
        )
        # Recall is GT-centric.  Keeping one row for every non-zero overlap is
        # necessary because a merged candidate can overlap a large object most
        # strongly while still covering a tiny object above an IoU threshold.
        # Looking only at the candidate's dominant GT would undercount recall.
        for gt in scene_gt:
            if not gt.official_valid:
                continue
            intersection = int(
                len(
                    np.intersect1d(
                        projected.support_ids, gt.point_ids, assume_unique=True
                    )
                )
            )
            if not intersection:
                continue
            union = len(projected.support_ids) + len(gt.point_ids) - intersection
            rows.append(
                {
                    "schema": V10_SCHEMA,
                    "row_type": "stage_candidate_gt_overlap",
                    "scene_id": candidate.scene_id,
                    "stage": candidate.stage,
                    "candidate_id": candidate.candidate_id,
                    "candidate_class_name": candidate.class_name,
                    "gt_class_name": gt.class_name,
                    "gt_instance_id": gt.instance_id,
                    "gt_size_bin": gt.size_bin,
                    "intersection": intersection,
                    "iou": float(intersection / union) if union else 0.0,
                }
            )
    return rows


def ground_truth_rows(
    ground_truth: Sequence[V10GroundTruthObject],
) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, int]] = set()
    rows: list[dict[str, Any]] = []
    for gt in ground_truth:
        if gt.identity in seen:
            raise ValueError(f"duplicate GT identity: {gt.identity}")
        seen.add(gt.identity)
        rows.append(
            {
                "schema": V10_SCHEMA,
                "row_type": "ground_truth",
                "scene_id": gt.scene_id,
                "class_name": gt.class_name,
                "instance_id": gt.instance_id,
                "point_count": int(len(gt.point_ids)),
                "official_valid": bool(gt.official_valid),
                "size_bin": gt.size_bin,
            }
        )
    return rows


def _divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _missing(value: Any) -> bool:
    return value is None or (
        isinstance(value, (float, np.floating)) and bool(np.isnan(value))
    )


def _association_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    same = sum(row["classification"] == "same_gt" for row in rows)
    different = sum(row["classification"] == "different_gt" for row in rows)
    unknown = sum(row["classification"] == "unknown" for row in rows)
    if same + different + unknown != total:
        raise ValueError("invalid accepted-edge classification")
    identifiable = same + different
    return {
        "accepted_edge_count": total,
        "same_gt_edge_count": same,
        "different_gt_edge_count": different,
        "unknown_edge_count": unknown,
        "identifiable_edge_count": identifiable,
        "identifiable_precision": _divide(same, identifiable),
        "all_edge_precision": _divide(same, total),
        "unknown_rate": _divide(unknown, total),
    }


def _stage_block(
    rows: Sequence[Mapping[str, Any]],
    overlap_rows: Sequence[Mapping[str, Any]],
    official_gt: set[tuple[str, str, int]],
    tiny_small_gt: set[tuple[str, str, int]],
) -> dict[str, Any]:
    count = len(rows)
    gaussian_count = sum(int(row["gaussian_count"]) for row in rows)
    unmapped_count = sum(int(row["unmapped_gaussian_count"]) for row in rows)
    match_025 = [row for row in rows if float(row["best_official_iou"]) >= 0.25]
    match_050 = [row for row in rows if float(row["best_official_iou"]) >= 0.50]
    classified = [row for row in rows if not _missing(row.get("class_name"))]
    same_class_025 = [
        row
        for row in classified
        if float(row["best_same_class_official_iou"]) >= 0.25
    ]
    same_class_050 = [
        row
        for row in classified
        if float(row["best_same_class_official_iou"]) >= 0.50
    ]
    late_classifier_correct_025 = [
        row
        for row in classified
        if float(row["best_official_iou"]) >= 0.25
        and str(row["class_name"]) == str(row["best_official_gt_class_name"])
    ]

    def covered_identities(threshold: float) -> set[tuple[str, str, int]]:
        identities = {
            (
                str(row["scene_id"]),
                str(row["gt_class_name"]),
                int(row["gt_instance_id"]),
            )
            for row in overlap_rows
            if float(row["iou"]) >= threshold
        }
        return identities & official_gt

    covered_025 = covered_identities(0.25)
    covered_050 = covered_identities(0.50)
    tiny_small_025 = len(covered_025 & tiny_small_gt)
    tiny_small_050 = len(covered_050 & tiny_small_gt)

    def same_class_covered(threshold: float) -> set[tuple[str, str, int]]:
        identities = {
            (
                str(row["scene_id"]),
                str(row["gt_class_name"]),
                int(row["gt_instance_id"]),
            )
            for row in overlap_rows
            if not _missing(row.get("candidate_class_name"))
            and str(row["candidate_class_name"]) == str(row["gt_class_name"])
            and float(row["iou"]) >= threshold
        }
        return identities & official_gt

    same_class_covered_025 = same_class_covered(0.25)
    same_class_covered_050 = same_class_covered(0.50)
    scored = [
        row
        for row in rows
        if not _missing(row.get("score"))
        and np.isfinite(float(row["score"]))
    ]
    score_iou = score_iou_spearman(
        [float(row["score"]) for row in scored],
        [float(row["best_same_class_official_iou"]) for row in scored],
    )
    return {
        "candidate_count": count,
        "identifiable_candidate_count": sum(bool(row["identifiable"]) for row in rows),
        "identifiable_candidate_rate": _divide(
            sum(bool(row["identifiable"]) for row in rows), count
        ),
        "gaussian_count": gaussian_count,
        "unmapped_gaussian_count": unmapped_count,
        "unmapped_gaussian_fraction": _divide(unmapped_count, gaussian_count),
        "candidate_match_025_count": len(match_025),
        "candidate_match_050_count": len(match_050),
        "candidate_match_050_scene_count": len(
            {str(row["scene_id"]) for row in match_050}
        ),
        "candidate_precision_025": _divide(len(match_025), count),
        "candidate_precision_050": _divide(len(match_050), count),
        "same_class_candidate_count": len(classified),
        "same_class_candidate_match_025_count": len(same_class_025),
        "same_class_candidate_match_050_count": len(same_class_050),
        "same_class_candidate_match_050_scene_count": len(
            {str(row["scene_id"]) for row in same_class_050}
        ),
        "late_classifier_correct_025_count": len(late_classifier_correct_025),
        "same_class_candidate_precision_025": _divide(
            len(same_class_025), len(classified)
        ),
        "same_class_candidate_precision_050": _divide(
            len(same_class_050), len(classified)
        ),
        "scored_candidate_count": len(scored),
        "score_iou_spearman": float(score_iou),
        "official_gt_count": len(official_gt),
        "covered_official_gt_025_count": len(covered_025),
        "covered_official_gt_050_count": len(covered_050),
        "official_gt_recall_025": _divide(len(covered_025), len(official_gt)),
        "official_gt_recall_050": _divide(len(covered_050), len(official_gt)),
        "same_class_covered_official_gt_025_count": len(same_class_covered_025),
        "same_class_covered_official_gt_050_count": len(same_class_covered_050),
        "same_class_official_gt_recall_025": _divide(
            len(same_class_covered_025), len(official_gt)
        ),
        "same_class_official_gt_recall_050": _divide(
            len(same_class_covered_050), len(official_gt)
        ),
        "official_tiny_small_gt_count": len(tiny_small_gt),
        "covered_tiny_small_gt_025_count": tiny_small_025,
        "covered_tiny_small_gt_050_count": tiny_small_050,
        "geometric_tiny_small_recall_025": _divide(
            tiny_small_025, len(tiny_small_gt)
        ),
        "geometric_tiny_small_recall_050": _divide(
            tiny_small_050, len(tiny_small_gt)
        ),
        "same_class_tiny_small_recall_025": _divide(
            len(same_class_covered_025 & tiny_small_gt), len(tiny_small_gt)
        ),
        "same_class_tiny_small_recall_050": _divide(
            len(same_class_covered_050 & tiny_small_gt), len(tiny_small_gt)
        ),
        "mean_best_iou": _divide(
            sum(float(row["best_iou"]) for row in rows), count
        ),
        "mean_best_purity": _divide(
            sum(float(row["best_purity"]) for row in rows), count
        ),
    }


def analyse_v10_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = [dict(row) for row in rows]
    if any(row.get("schema") != V10_SCHEMA for row in normalized):
        raise ValueError("row schema does not match V10")
    allowed = {
        "ground_truth",
        "accepted_fragment_pair",
        "stage_candidate",
        "stage_candidate_gt_overlap",
    }
    if any(row.get("row_type") not in allowed for row in normalized):
        raise ValueError("unsupported V10 row_type")

    gt_rows = [row for row in normalized if row["row_type"] == "ground_truth"]
    edge_rows = [
        row for row in normalized if row["row_type"] == "accepted_fragment_pair"
    ]
    candidate_rows = [
        row for row in normalized if row["row_type"] == "stage_candidate"
    ]
    overlap_rows = [
        row
        for row in normalized
        if row["row_type"] == "stage_candidate_gt_overlap"
    ]
    official_gt = {
        (str(row["scene_id"]), str(row["class_name"]), int(row["instance_id"]))
        for row in gt_rows
        if bool(row["official_valid"])
    }
    tiny_small_gt = {
        (str(row["scene_id"]), str(row["class_name"]), int(row["instance_id"]))
        for row in gt_rows
        if bool(row["official_valid"]) and row.get("size_bin") in {"tiny", "small"}
    }
    scene_ids = sorted({str(row["scene_id"]) for row in normalized})
    associations_by_scene = {
        scene_id: _association_block(
            [row for row in edge_rows if str(row["scene_id"]) == scene_id]
        )
        for scene_id in scene_ids
    }
    stages = {
        stage: _stage_block(
            [row for row in candidate_rows if row["stage"] == stage],
            [row for row in overlap_rows if row["stage"] == stage],
            official_gt,
            tiny_small_gt,
        )
        for stage in V10_STAGE_ORDER
    }
    return {
        "schema": V10_SCHEMA,
        "row_count": len(normalized),
        "scene_ids": scene_ids,
        "association_thresholds": {
            "min_intersection": ASSOCIATION_MIN_INTERSECTION,
            "min_iou": ASSOCIATION_MIN_IOU,
            "min_purity": ASSOCIATION_MIN_PURITY,
            "unknown_rule": "best_intersection == 0",
            "iou_and_purity_are_diagnostics_only": True,
        },
        "ground_truth": {
            "object_count": len(gt_rows),
            "official_object_count": len(official_gt),
            "official_tiny_small_object_count": len(tiny_small_gt),
        },
        "accepted_fragment_pairs": _association_block(edge_rows),
        "accepted_fragment_pairs_by_scene": associations_by_scene,
        "stage_order": list(V10_STAGE_ORDER),
        "stages": stages,
    }


def evaluate_v10_audit(
    *,
    ground_truth: Sequence[V10GroundTruthObject],
    gaussian_gt_index: GaussianGTIndex,
    fragments: Sequence[V10FragmentSupport] = (),
    accepted_edges: Sequence[V10AcceptedEdge] = (),
    stage_candidates: Sequence[V10StageCandidate] = (),
) -> dict[str, Any]:
    rows = ground_truth_rows(ground_truth)
    rows.extend(
        accepted_fragment_pair_rows(
            fragments, accepted_edges, ground_truth, gaussian_gt_index
        )
    )
    rows.extend(stage_funnel_rows(stage_candidates, ground_truth, gaussian_gt_index))
    return {"rows": rows, "analysis": analyse_v10_rows(rows)}


def write_v10_results(
    *,
    rows_output: Path,
    analysis_output: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Persist rows and an analysis recomputed solely from those rows."""

    analysis = analyse_v10_rows(rows)
    write_rows(rows_output, rows)
    write_json(analysis_output, analysis)
    return analysis


def recompute_saved_v10_analysis(
    rows_path: Path,
    analysis_path: Path | None = None,
) -> dict[str, Any]:
    """Independently recompute and optionally verify a saved V10 analysis."""

    analysis = analyse_v10_rows(read_rows(rows_path))
    if analysis_path is not None:
        saved = load_json(analysis_path)
        if analysis != saved:
            raise ValueError("saved V10 analysis is inconsistent with its rows")
    return analysis


def adapt_v10_persisted_bank(
    bank_dir: str | Path,
    *,
    classifier: str = "mv-label",
) -> V10PersistedAuditInputs:
    """Adapt one runner bank without reconstructing or inventing audit evidence.

    The runner persists the actual accepted association edges in JSON and all
    eight frozen support levels as ragged NPZ arrays.  This adapter intentionally
    rejects an older or incomplete bank rather than deriving missing edges from
    tracks or deriving missing stages from the final candidate masks.
    """

    # Delayed import keeps the pure array-level metrics usable without loading
    # the V10 filesystem/runtime contract.
    from .v10_runner import (
        V10_CLASSIFIERS,
        V10_FUNNEL_STAGES,
        load_v10_audit_supports,
        load_v10_candidate_bank,
    )

    if tuple(V10_FUNNEL_STAGES) != V10_STAGE_ORDER:
        raise ValueError("metrics and runner V10 funnel stages disagree")
    if classifier not in V10_CLASSIFIERS:
        raise ValueError(f"unknown V10 late classifier: {classifier}")

    metadata, bank = load_v10_candidate_bank(bank_dir)
    audit_metadata, persisted_supports = load_v10_audit_supports(bank_dir)
    if audit_metadata != metadata:
        raise ValueError("V10 bank metadata changed between immutable reads")
    scene_id = str(metadata.get("scene_id", "")).strip()
    if not scene_id:
        raise ValueError("persisted V10 bank has no scene_id")

    fragment_payload = metadata.get("fragments")
    if isinstance(fragment_payload, (str, bytes)) or not isinstance(
        fragment_payload, Sequence
    ):
        raise ValueError("persisted V10 bank has no fragment rows")
    if set(persisted_supports) != set(V10_STAGE_ORDER):
        raise ValueError("persisted V10 support stages are incomplete")
    single_full = persisted_supports["single_full"]
    single_core = persisted_supports["single_core"]
    if len(fragment_payload) != len(single_full) or len(fragment_payload) != len(
        single_core
    ):
        raise ValueError("persisted fragment rows and single-stage supports disagree")

    fragments: list[V10FragmentSupport] = []
    seen_fragment_ids: set[int] = set()
    for index, raw in enumerate(fragment_payload):
        if not isinstance(raw, Mapping):
            raise ValueError(f"fragment row {index} must be a mapping")
        if "fragment_id" not in raw:
            raise ValueError(f"fragment row {index} lacks fragment_id")
        fragment_id = int(raw["fragment_id"])
        if fragment_id in seen_fragment_ids:
            raise ValueError(f"duplicate persisted fragment ID: {fragment_id}")
        seen_fragment_ids.add(fragment_id)
        full_ids = _ids(
            single_full[index], name=f"fragment {fragment_id} persisted full_ids"
        )
        core_ids = _ids(
            single_core[index], name=f"fragment {fragment_id} persisted core_ids"
        )
        if not len(full_ids) or not len(core_ids):
            raise ValueError("persisted fragment full/core support cannot be empty")
        if not np.all(np.isin(core_ids, full_ids, assume_unique=True)):
            raise ValueError(f"fragment {fragment_id} core is not a subset of full")
        fragments.append(V10FragmentSupport(scene_id, fragment_id, full_ids))

    edge_payload = metadata.get("accepted_edges")
    if isinstance(edge_payload, (str, bytes)) or not isinstance(edge_payload, Sequence):
        raise ValueError("persisted V10 bank lacks actual accepted_edges")
    accepted_edges: list[V10AcceptedEdge] = []
    for index, raw in enumerate(edge_payload):
        if not isinstance(raw, Mapping):
            raise ValueError(f"accepted edge row {index} must be a mapping")
        missing = {"left_fragment_id", "right_fragment_id"}.difference(raw)
        if missing:
            raise ValueError(
                f"accepted edge row {index} lacks required fields: {sorted(missing)}"
            )
        accepted_edges.append(
            V10AcceptedEdge(
                scene_id,
                int(raw["left_fragment_id"]),
                int(raw["right_fragment_id"]),
                (
                    None
                    if raw.get("left_frame_id") is None
                    else int(raw["left_frame_id"])
                ),
                (
                    None
                    if raw.get("right_frame_id") is None
                    else int(raw["right_frame_id"])
                ),
                None if raw.get("kind") is None else str(raw["kind"]),
                None if raw.get("score") is None else float(raw["score"]),
                None if raw.get("shared") is None else int(raw["shared"]),
                (
                    None
                    if raw.get("p0_overlap") is None
                    else float(raw["p0_overlap"])
                ),
                (
                    None
                    if raw.get("left_coverage") is None
                    else float(raw["left_coverage"])
                ),
                (
                    None
                    if raw.get("right_coverage") is None
                    else float(raw["right_coverage"])
                ),
                None if raw.get("row_margin") is None else float(raw["row_margin"]),
                (
                    None
                    if raw.get("column_margin") is None
                    else float(raw["column_margin"])
                ),
                (
                    None
                    if raw.get("component_support_ratio") is None
                    else float(raw["component_support_ratio"])
                ),
                None if raw.get("strong") is None else bool(raw["strong"]),
                (
                    None
                    if raw.get("cycle_supported") is None
                    else bool(raw["cycle_supported"])
                ),
            )
        )

    stage_payload = metadata.get("stage_supports")
    if not isinstance(stage_payload, Mapping):
        raise ValueError("persisted V10 bank lacks stage_supports descriptors")
    missing_stages = set(V10_STAGE_ORDER).difference(stage_payload)
    unexpected_stages = set(stage_payload).difference(V10_STAGE_ORDER)
    if missing_stages or unexpected_stages:
        raise ValueError(
            "V10 diagnostics stage set mismatch: "
            f"missing={sorted(missing_stages)}, unexpected={sorted(unexpected_stages)}"
        )
    stage_candidates: list[V10StageCandidate] = []
    supports_by_stage: dict[str, dict[int, np.ndarray]] = {}
    for stage in V10_STAGE_ORDER:
        descriptors = stage_payload[stage]
        supports = persisted_supports[stage]
        if isinstance(descriptors, (str, bytes)) or not isinstance(
            descriptors, Sequence
        ):
            raise ValueError(f"V10 stage {stage} descriptors must be a sequence")
        if len(descriptors) != len(supports):
            raise ValueError(f"V10 stage {stage} descriptors/supports disagree")
        stage_supports: dict[int, np.ndarray] = {}
        for index, (raw, persisted_ids) in enumerate(zip(descriptors, supports)):
            if not isinstance(raw, Mapping):
                raise ValueError(f"V10 stage {stage} row {index} must be a mapping")
            missing = {"candidate_id", "class_name", "support_count"}.difference(raw)
            if missing:
                raise ValueError(
                    f"V10 stage {stage} row {index} lacks fields: {sorted(missing)}"
                )
            candidate_id = int(raw["candidate_id"])
            if candidate_id in stage_supports:
                raise ValueError(f"duplicate {stage} candidate ID: {candidate_id}")
            gaussian_ids = _ids(
                persisted_ids, name=f"{stage} candidate {candidate_id} support"
            )
            if not len(gaussian_ids):
                raise ValueError(f"{stage} candidate support cannot be empty")
            if int(raw["support_count"]) != len(gaussian_ids):
                raise ValueError(
                    f"V10 stage {stage} candidate {candidate_id} support_count mismatch"
                )
            class_name = raw["class_name"]
            score: float | None = None
            if stage == "final_candidate":
                candidate_rows = metadata.get("candidates")
                if not isinstance(candidate_rows, Sequence) or candidate_id >= len(
                    candidate_rows
                ):
                    raise ValueError("V10 final stage lacks candidate metadata")
                classifier_rows = candidate_rows[candidate_id].get("classifiers")
                if not isinstance(classifier_rows, Mapping) or classifier not in classifier_rows:
                    raise ValueError(
                        f"V10 final candidate lacks {classifier} evidence"
                    )
                selected = classifier_rows[classifier]
                if not isinstance(selected, Mapping):
                    raise ValueError(f"invalid {classifier} candidate evidence")
                class_name = selected.get("branch_class")
                if selected.get("base_score") is None:
                    raise ValueError(
                        f"V10 final candidate lacks {classifier} base_score"
                    )
                score = float(selected["base_score"])
            stage_supports[candidate_id] = gaussian_ids
            stage_candidates.append(
                V10StageCandidate(
                    scene_id,
                    stage,
                    candidate_id,
                    gaussian_ids,
                    None if class_name is None else str(class_name),
                    score,
                )
            )
        supports_by_stage[stage] = stage_supports

    final_supports = supports_by_stage["final_candidate"]
    expected_final = {
        index: np.asarray(ids, dtype=np.int64)
        for index, ids in enumerate(bank.full_ids)
    }
    if set(final_supports) != set(expected_final) or any(
        not np.array_equal(final_supports[key], expected_final[key])
        for key in expected_final
    ):
        raise ValueError("V10 final_candidate diagnostics disagrees with bank masks")

    return V10PersistedAuditInputs(
        scene_id=scene_id,
        fragments=tuple(fragments),
        accepted_edges=tuple(accepted_edges),
        stage_candidates=tuple(stage_candidates),
    )


def load_v10_scene_geometry(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    scene_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reuse the established V9 runtime/GT readers without reusing its schema."""

    scenes = load_scene_runtime_manifest(runtime_manifest)
    scene = scenes[str(scene_id)]
    gt_xyz, gt = load_ground_truth_npz(gt_dir / f"{scene_id}.npz", str(scene_id))
    explicit = scene.get("gaussian_ply")
    if explicit:
        gaussian_path = Path(str(explicit))
        if not gaussian_path.is_absolute():
            gaussian_path = Path(str(scene["base_path"])) / gaussian_path
    else:
        root = (
            Path(str(scene["base_path"]))
            / "output_models/point_cloud/iteration_30000"
        )
        registered = root / "scene_point_cloud.ply"
        gaussian_path = registered if registered.is_file() else root / "point_cloud.ply"
    transform = scene.get(
        "gaussian_to_gt_transform",
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    gaussian_xyz = apply_transform(load_ply_xyz(gaussian_path), transform)
    return gt_xyz, gt.semantic, gt.instance, gaussian_xyz

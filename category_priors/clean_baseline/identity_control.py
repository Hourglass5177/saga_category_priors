from __future__ import annotations

"""One-shot learned identity-edge *capacity control* for the clean baseline.

This module is intentionally outside the formal C0/U/D runtime.  Ground truth
is used to label local edges in two registered development scenes and to
evaluate a third, held-out development scene.  The resulting classifier never
changes an evidence bank, a formal prediction, or a category-prior result.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluator import apply_transform, load_ground_truth_npz, load_ply_xyz
from ..io import load_json, write_json
from ..taxonomy import load_taxonomy
from .evaluation import (
    CleanCandidate,
    evaluate_candidates,
    ground_truth_objects_from_arrays,
    gt_point_to_gaussian_mapping,
)
from .evidence import load_evidence_bank
from .models import AlphaMaskEvidenceBank


IDENTITY_CONTROL_SCHEMA = "saga-clean-baseline-identity-edge-control-v1"
IDENTITY_CONTROL_REGISTRATION_SCHEMA = (
    "saga-clean-baseline-identity-edge-control-registration-v1"
)
IDENTITY_CONTROL_RESULT_SCHEMA = (
    "saga-clean-baseline-identity-edge-control-result-v1"
)
IDENTITY_TRAIN_SCENES = ("scene0645_00", "scene0025_01")
IDENTITY_VALIDATION_SCENE = "scene0046_00"
IDENTITY_FEATURE_NAMES = (
    "affinity_cosine",
    "soft_semantic_overlap",
    "abs_dx_m",
    "abs_dy_m",
    "abs_dz_m",
    "distance_m",
    "mean_scale_short_m",
    "mean_scale_mid_m",
    "mean_scale_long_m",
    "abs_scale_short_delta_m",
    "abs_scale_mid_delta_m",
    "abs_scale_long_delta_m",
    "mean_opacity",
    "abs_opacity_delta",
    "log1p_covisible_views",
    "log1p_same_mask_views",
    "same_mask_view_ratio",
)


def _resolved(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


@dataclass(frozen=True)
class IdentityAssetPaths:
    feature_ply: Path
    gaussian_ply: Path


@dataclass(frozen=True)
class IdentityControlConfig:
    """Frozen registration for the conditional three-scene control."""

    assets: Mapping[str, IdentityAssetPaths]
    train_scene_ids: tuple[str, ...] = IDENTITY_TRAIN_SCENES
    validation_scene_id: str = IDENTITY_VALIDATION_SCENE
    seed: int = 42
    physical_neighbors: int = 24
    max_edge_distance_m: float = 0.10
    max_training_edges_per_class: int = 200_000
    l2_c: float = 1.0
    probability_threshold: float = 0.50
    min_component_points: int = 4

    def __post_init__(self) -> None:
        if tuple(self.train_scene_ids) != IDENTITY_TRAIN_SCENES:
            raise ValueError("identity-control training scenes differ from registration")
        if str(self.validation_scene_id) != IDENTITY_VALIDATION_SCENE:
            raise ValueError("identity-control validation scene differs from registration")
        required = set(self.train_scene_ids) | {self.validation_scene_id}
        if set(self.assets) != required:
            raise ValueError("identity-control assets must register exactly three scenes")
        if int(self.seed) != 42:
            raise ValueError("identity-control seed is frozen at 42")
        if int(self.physical_neighbors) != 24:
            raise ValueError("identity-control local graph is frozen at 24-NN")
        if not math.isclose(float(self.max_edge_distance_m), 0.10):
            raise ValueError("identity-control edge radius is frozen at 0.10 m")
        if int(self.max_training_edges_per_class) != 200_000:
            raise ValueError("identity-control edge cap differs from registration")
        if not math.isclose(float(self.l2_c), 1.0):
            raise ValueError("identity-control L2 C is frozen at 1")
        if not math.isclose(float(self.probability_threshold), 0.50):
            raise ValueError("identity-control probability threshold is frozen at 0.50")
        if int(self.min_component_points) != 4:
            raise ValueError("identity-control minimum component is frozen at four points")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, base: str | Path
    ) -> "IdentityControlConfig":
        if value.get("schema") != IDENTITY_CONTROL_SCHEMA:
            raise ValueError("identity_control has an unsupported schema")
        root = Path(base).resolve()
        raw_assets = value.get("assets")
        if not isinstance(raw_assets, Mapping):
            raise TypeError("identity_control.assets must be an object")
        assets: dict[str, IdentityAssetPaths] = {}
        for scene_id, row in raw_assets.items():
            if not isinstance(row, Mapping):
                raise TypeError(f"identity asset {scene_id!r} must be an object")
            assets[str(scene_id)] = IdentityAssetPaths(
                feature_ply=_resolved(root, row["feature_ply"]),
                gaussian_ply=_resolved(root, row["gaussian_ply"]),
            )
        return cls(
            assets=assets,
            train_scene_ids=tuple(map(str, value.get("train_scene_ids", ()) )),
            validation_scene_id=str(value.get("validation_scene_id", "")),
            seed=int(value.get("seed", 42)),
            physical_neighbors=int(value.get("physical_neighbors", 24)),
            max_edge_distance_m=float(value.get("max_edge_distance_m", 0.10)),
            max_training_edges_per_class=int(
                value.get("max_training_edges_per_class", 200_000)
            ),
            l2_c=float(value.get("l2_c", 1.0)),
            probability_threshold=float(value.get("probability_threshold", 0.50)),
            min_component_points=int(value.get("min_component_points", 4)),
        )

    def identity(self) -> dict[str, Any]:
        return {
            "schema": IDENTITY_CONTROL_SCHEMA,
            "train_scene_ids": list(self.train_scene_ids),
            "validation_scene_id": self.validation_scene_id,
            "seed": self.seed,
            "physical_neighbors": self.physical_neighbors,
            "max_edge_distance_m": self.max_edge_distance_m,
            "max_training_edges_per_class": self.max_training_edges_per_class,
            "l2_c": self.l2_c,
            "probability_threshold": self.probability_threshold,
            "min_component_points": self.min_component_points,
            "feature_names": list(IDENTITY_FEATURE_NAMES),
            "assets": {
                scene_id: {
                    "feature_ply": str(self.assets[scene_id].feature_ply),
                    "gaussian_ply": str(self.assets[scene_id].gaussian_ply),
                }
                for scene_id in sorted(self.assets)
            },
        }


@dataclass(frozen=True)
class IdentitySceneInput:
    scene_id: str
    bank_dir: Path
    gt_npz: Path
    gaussian_to_gt_transform: tuple[tuple[float, float, float, float], ...]
    uniform_output_json: Path
    evaluation_class_names: tuple[str, ...]
    tiny_small_instance_ids: tuple[int, ...] = ()
    min_region_size: int = 100
    radius_m: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(self, "bank_dir", Path(self.bank_dir))
        object.__setattr__(self, "gt_npz", Path(self.gt_npz))
        object.__setattr__(
            self, "uniform_output_json", Path(self.uniform_output_json)
        )
        evaluation_classes = tuple(map(str, self.evaluation_class_names))
        if evaluation_classes != load_taxonomy().canonical_classes:
            raise ValueError(
                "identity-control GT classes must use the registered ScanNet20 order"
            )
        object.__setattr__(self, "evaluation_class_names", evaluation_classes)
        transform = np.asarray(
            self.gaussian_to_gt_transform, dtype=np.float64
        )
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("identity-control transform must be finite 4x4")
        object.__setattr__(
            self,
            "gaussian_to_gt_transform",
            tuple(tuple(float(value) for value in row) for row in transform),
        )
        if int(self.min_region_size) != 100:
            raise ValueError("identity-control min_region_size is frozen at 100")
        if not math.isclose(float(self.radius_m), 0.05):
            raise ValueError("identity-control GT radius is frozen at 0.05 m")


@dataclass(frozen=True)
class BalancedLogisticModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    l2_c: float = 1.0

    def decision_function(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.coefficients):
            raise ValueError("identity features have the wrong shape")
        normalized = (matrix - self.mean) / self.scale
        return normalized @ self.coefficients + float(self.intercept)

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        score = self.decision_function(values)
        result = np.empty_like(score)
        positive = score >= 0
        result[positive] = 1.0 / (1.0 + np.exp(-score[positive]))
        exp_score = np.exp(score[~positive])
        result[~positive] = exp_score / (1.0 + exp_score)
        return result


def fit_balanced_l2_logistic(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    l2_c: float = 1.0,
) -> BalancedLogisticModel:
    """Fit one deterministic, balanced L2 logistic regression."""

    from scipy.optimize import minimize

    matrix = np.asarray(values, dtype=np.float64)
    target = np.asarray(labels)
    if matrix.ndim != 2 or not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError("training features must be a non-empty finite matrix")
    if target.shape != (len(matrix),) or not np.isin(target, (0, 1)).all():
        raise ValueError("training labels must be an aligned binary vector")
    target = target.astype(np.float64, copy=False)
    negative = int(np.count_nonzero(target == 0))
    positive = int(np.count_nonzero(target == 1))
    if not negative or not positive:
        raise ValueError("identity control requires both positive and negative edges")
    c_value = float(l2_c)
    if not math.isfinite(c_value) or c_value <= 0:
        raise ValueError("l2_c must be finite and positive")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized = (matrix - mean) / scale
    weights = np.where(
        target > 0.5,
        len(target) / (2.0 * positive),
        len(target) / (2.0 * negative),
    )
    weight_sum = float(weights.sum())

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        coefficient = parameters[:-1]
        intercept = parameters[-1]
        score = normalized @ coefficient + intercept
        loss = np.logaddexp(0.0, score) - target * score
        probability = np.empty_like(score)
        mask = score >= 0
        probability[mask] = 1.0 / (1.0 + np.exp(-score[mask]))
        exp_score = np.exp(score[~mask])
        probability[~mask] = exp_score / (1.0 + exp_score)
        residual = weights * (probability - target) / weight_sum
        penalty = 0.5 * np.dot(coefficient, coefficient) / c_value
        value = float(np.dot(weights, loss) / weight_sum + penalty)
        gradient = np.concatenate(
            (normalized.T @ residual + coefficient / c_value, [residual.sum()])
        )
        return value, gradient

    initial = np.zeros(matrix.shape[1] + 1, dtype=np.float64)
    fitted = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"balanced logistic regression failed: {fitted.message}")
    return BalancedLogisticModel(
        mean=np.asarray(mean, dtype=np.float64),
        scale=np.asarray(scale, dtype=np.float64),
        coefficients=np.asarray(fitted.x[:-1], dtype=np.float64),
        intercept=float(fitted.x[-1]),
        l2_c=c_value,
    )


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Tie-aware binary AUROC without an optional ML dependency."""

    target = np.asarray(labels)
    value = np.asarray(scores, dtype=np.float64)
    if target.shape != value.shape or target.ndim != 1:
        raise ValueError("AUROC labels and scores must be aligned vectors")
    if not np.isin(target, (0, 1)).all() or not np.isfinite(value).all():
        raise ValueError("AUROC inputs must be finite and binary")
    positive = int(np.count_nonzero(target == 1))
    negative = int(np.count_nonzero(target == 0))
    if not positive or not negative:
        raise ValueError("AUROC requires both classes")
    order = np.argsort(value, kind="mergesort")
    ranks = np.empty(len(value), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and value[order[stop]] == value[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    rank_sum = float(ranks[target == 1].sum())
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-12)


def load_affinity_feature_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"].data
    names = set(vertex.dtype.names or ())
    fields = [f"f_{index}" for index in range(32)]
    missing = [name for name in fields if name not in names]
    if missing:
        raise ValueError(f"{path}: missing affinity fields {missing[:3]}")
    xyz = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(
        np.float64
    )
    affinity = np.column_stack([vertex[name] for name in fields]).astype(np.float64)
    if not np.isfinite(xyz).all() or not np.isfinite(affinity).all():
        raise ValueError(f"{path}: non-finite affinity PLY")
    return xyz, _normalize_rows(affinity)


def load_gaussian_attributes_ply(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"].data
    names = set(vertex.dtype.names or ())
    required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2"}
    missing = sorted(required.difference(names))
    if missing:
        raise ValueError(f"{path}: missing Gaussian attributes {missing}")
    xyz = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(
        np.float64
    )
    log_scale = np.column_stack(
        [vertex[f"scale_{index}"] for index in range(3)]
    ).astype(np.float64)
    opacity_logit = np.asarray(vertex["opacity"], dtype=np.float64)
    if not all(np.isfinite(value).all() for value in (xyz, log_scale, opacity_logit)):
        raise ValueError(f"{path}: non-finite Gaussian attributes")
    scale = np.sort(np.exp(np.clip(log_scale, -20.0, 20.0)), axis=1)
    opacity = np.empty_like(opacity_logit)
    positive = opacity_logit >= 0
    opacity[positive] = 1.0 / (1.0 + np.exp(-opacity_logit[positive]))
    exp_value = np.exp(opacity_logit[~positive])
    opacity[~positive] = exp_value / (1.0 + exp_value)
    return xyz, scale, opacity


def local_edge_index(
    xyz_m: np.ndarray,
    *,
    physical_neighbors: int = 24,
    max_distance_m: float = 0.10,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz_m, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("xyz_m must be a finite Nx3 matrix")
    if len(xyz) < 2:
        return np.empty((0, 2), dtype=np.int64)
    k = min(int(physical_neighbors) + 1, len(xyz))
    distance, neighbor = cKDTree(xyz).query(
        xyz, k=k, distance_upper_bound=float(max_distance_m), workers=1
    )
    distance = np.atleast_2d(distance)
    neighbor = np.atleast_2d(neighbor)
    pairs: set[tuple[int, int]] = set()
    for source in range(len(xyz)):
        for target, radius in zip(neighbor[source, 1:], distance[source, 1:]):
            target_id = int(target)
            if target_id >= len(xyz) or not np.isfinite(radius) or target_id == source:
                continue
            pairs.add((min(source, target_id), max(source, target_id)))
    return np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def gaussian_soft_semantics(bank: AlphaMaskEvidenceBank) -> np.ndarray:
    result = np.zeros((bank.point_count, len(bank.class_names)), dtype=np.float64)
    count = np.zeros(bank.point_count, dtype=np.float64)
    for row in range(len(bank.masks)):
        if bool(bank.semantic_abstained[row]):
            continue
        ids, _, _, ambiguous = bank.mask_support.row(row)
        ids = ids[~ambiguous]
        if not len(ids):
            continue
        result[ids] += bank.semantic_posteriors[row]
        count[ids] += 1.0
    valid = count > 0
    result[valid] /= count[valid, None]
    return result


def _edge_view_evidence(
    bank: AlphaMaskEvidenceBank, edge_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    relevant = set(map(int, np.unique(edge_index)))
    visible: dict[int, set[int]] = {point: set() for point in relevant}
    mask_rows: dict[int, set[int]] = {point: set() for point in relevant}
    for position, frame in enumerate(bank.frames):
        ids, _ = bank.frame_visibility.row(position)
        for point in ids:
            value = int(point)
            if value in visible:
                visible[value].add(int(frame.frame_id))
    for row, mask in enumerate(bank.masks):
        ids, _, _, ambiguous = bank.mask_support.row(row)
        for point in ids[~ambiguous]:
            value = int(point)
            if value in mask_rows:
                mask_rows[value].add(int(mask.global_mask_id))
    mask_frame = {int(mask.global_mask_id): int(mask.frame_id) for mask in bank.masks}
    co_visible = np.zeros(len(edge_index), dtype=np.float64)
    same_mask = np.zeros(len(edge_index), dtype=np.float64)
    ratio = np.zeros(len(edge_index), dtype=np.float64)
    for index, (left, right) in enumerate(edge_index):
        shared_visible = visible[int(left)].intersection(visible[int(right)])
        shared_masks = mask_rows[int(left)].intersection(mask_rows[int(right)])
        shared_mask_frames = {mask_frame[value] for value in shared_masks}
        co_visible[index] = len(shared_visible)
        same_mask[index] = len(shared_mask_frames)
        ratio[index] = len(shared_mask_frames) / max(len(shared_visible), 1)
    return co_visible, same_mask, ratio


def edge_feature_matrix(
    *,
    edge_index: np.ndarray,
    xyz_m: np.ndarray,
    affinity: np.ndarray,
    soft_semantic: np.ndarray,
    gaussian_scale_m: np.ndarray,
    opacity: np.ndarray,
    bank: AlphaMaskEvidenceBank,
) -> np.ndarray:
    edges = np.asarray(edge_index, dtype=np.int64)
    xyz = np.asarray(xyz_m, dtype=np.float64)
    affinity_value = _normalize_rows(affinity)
    semantics = _normalize_rows(soft_semantic)
    scale = np.asarray(gaussian_scale_m, dtype=np.float64)
    opacity_value = np.asarray(opacity, dtype=np.float64)
    count = len(xyz)
    if edges.ndim != 2 or edges.shape[1:] != (2,):
        raise ValueError("edge_index must be Ex2")
    if np.any(edges < 0) or np.any(edges >= count):
        raise ValueError("edge_index contains out-of-range Gaussian IDs")
    if (
        affinity_value.shape[0] != count
        or semantics.shape[0] != count
        or scale.shape != (count, 3)
        or opacity_value.shape != (count,)
    ):
        raise ValueError("identity-control inputs do not share the point axis")
    left, right = edges[:, 0], edges[:, 1]
    delta = np.abs(xyz[left] - xyz[right])
    distance = np.linalg.norm(delta, axis=1)
    affinity_cosine = np.sum(affinity_value[left] * affinity_value[right], axis=1)
    semantic_overlap = np.sum(semantics[left] * semantics[right], axis=1)
    mean_scale = 0.5 * (scale[left] + scale[right])
    scale_delta = np.abs(scale[left] - scale[right])
    mean_opacity = 0.5 * (opacity_value[left] + opacity_value[right])
    opacity_delta = np.abs(opacity_value[left] - opacity_value[right])
    co_visible, same_mask, view_ratio = _edge_view_evidence(bank, edges)
    result = np.column_stack(
        (
            affinity_cosine,
            semantic_overlap,
            delta,
            distance,
            mean_scale,
            scale_delta,
            mean_opacity,
            opacity_delta,
            np.log1p(co_visible),
            np.log1p(same_mask),
            view_ratio,
        )
    )
    if result.shape[1] != len(IDENTITY_FEATURE_NAMES) or not np.isfinite(result).all():
        raise AssertionError("identity feature construction violated its contract")
    return result


def gaussian_gt_labels(
    gt_semantic: np.ndarray,
    gt_instance: np.ndarray,
    gt_point_to_gaussian: np.ndarray,
    *,
    gaussian_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    semantic = np.asarray(gt_semantic, dtype=np.int64)
    instance = np.asarray(gt_instance, dtype=np.int64)
    mapping = np.asarray(gt_point_to_gaussian, dtype=np.int64)
    if semantic.shape != instance.shape or mapping.shape != instance.shape:
        raise ValueError("GT arrays and mapping must be aligned")
    valid = (mapping >= 0) & (mapping < int(gaussian_count)) & (instance >= 0)
    votes: dict[int, dict[tuple[int, int], int]] = {}
    for point in np.flatnonzero(valid):
        gaussian = int(mapping[point])
        key = (int(instance[point]), int(semantic[point]))
        bucket = votes.setdefault(gaussian, {})
        bucket[key] = bucket.get(key, 0) + 1
    gaussian_instance = np.full(int(gaussian_count), -1, dtype=np.int64)
    gaussian_semantic = np.full(int(gaussian_count), -1, dtype=np.int64)
    for gaussian, bucket in votes.items():
        winner = min(bucket, key=lambda key: (-bucket[key], key[0], key[1]))
        gaussian_instance[gaussian], gaussian_semantic[gaussian] = winner
    return gaussian_instance, gaussian_semantic


def labelled_hard_edges(
    edge_index: np.ndarray,
    gaussian_instance: np.ndarray,
    gaussian_semantic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.asarray(edge_index, dtype=np.int64)
    left, right = edges[:, 0], edges[:, 1]
    mapped = (gaussian_instance[left] >= 0) & (gaussian_instance[right] >= 0)
    positive = mapped & (gaussian_instance[left] == gaussian_instance[right])
    hard_negative = (
        mapped
        & (gaussian_instance[left] != gaussian_instance[right])
        & (gaussian_semantic[left] >= 0)
        & (gaussian_semantic[left] == gaussian_semantic[right])
    )
    selected = positive | hard_negative
    return np.flatnonzero(selected), positive[selected].astype(np.int8)


def _subsample_balanced(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    per_class_cap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    generator = np.random.default_rng(int(seed))
    chosen: list[np.ndarray] = []
    original: dict[str, int] = {}
    retained: dict[str, int] = {}
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        original[str(label)] = len(indices)
        if len(indices) > int(per_class_cap):
            indices = np.sort(
                generator.choice(indices, int(per_class_cap), replace=False)
            )
        retained[str(label)] = len(indices)
        chosen.append(indices)
    selected = np.concatenate(chosen)
    selected.sort()
    return features[selected], labels[selected], {
        "negative_original": original["0"],
        "positive_original": original["1"],
        "negative_retained": retained["0"],
        "positive_retained": retained["1"],
    }


def edge_components(
    point_count: int,
    edge_index: np.ndarray,
    accepted: np.ndarray,
    *,
    min_component_points: int = 4,
) -> tuple[np.ndarray, ...]:
    edges = np.asarray(edge_index, dtype=np.int64)
    keep = np.asarray(accepted, dtype=bool)
    if keep.shape != (len(edges),):
        raise ValueError("accepted must align with edge_index")
    parent = np.arange(int(point_count), dtype=np.int64)
    size = np.ones(int(point_count), dtype=np.int64)
    touched = np.zeros(int(point_count), dtype=bool)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for left, right in edges[keep]:
        a, b = find(int(left)), find(int(right))
        touched[int(left)] = touched[int(right)] = True
        if a == b:
            continue
        if size[a] < size[b] or (size[a] == size[b] and a > b):
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
    groups: dict[int, list[int]] = {}
    for point in np.flatnonzero(touched):
        groups.setdefault(find(int(point)), []).append(int(point))
    result = [
        np.asarray(points, dtype=np.int64)
        for _, points in sorted(groups.items())
        if len(points) >= int(min_component_points)
    ]
    result.sort(key=lambda values: (int(values[0]), len(values)))
    return tuple(result)


def _uniform_candidates(path: Path) -> tuple[CleanCandidate, ...]:
    payload = load_json(path)
    labels = np.asarray(payload["point_labels"], dtype=np.int64)
    result: list[CleanCandidate] = []
    for raw_id, metadata in sorted(payload["instances"].items(), key=lambda row: int(row[0])):
        points = np.flatnonzero(labels == int(raw_id))
        if not len(points):
            continue
        result.append(
            CleanCandidate(
                object_id=f"uniform-{raw_id}",
                gaussian_ids=points,
                class_id=metadata.get("class"),
                winner_probability=1.0,
                view_consensus=1.0,
                detection_ratio=1.0,
            )
        )
    return tuple(result)


def _matched_gt_050_count(evaluation: Mapping[str, Any]) -> int:
    return sum(float(row["best_geometry_iou"]) >= 0.50 for row in evaluation["gt_rows"])


def _assert_point_order_geometry(
    reference_xyz: np.ndarray, candidate_xyz: np.ndarray, *, label: str
) -> None:
    left = np.asarray(reference_xyz, dtype=np.float64)
    right = np.asarray(candidate_xyz, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 3:
        raise ValueError(f"{label}: point arrays have different shapes")
    if len(left) < 2:
        return
    sample = np.unique(np.linspace(0, len(left) - 1, min(len(left), 64)).astype(int))
    left_distance = np.linalg.norm(left[sample] - left[sample[0]], axis=1)
    right_distance = np.linalg.norm(right[sample] - right[sample[0]], axis=1)
    if not np.allclose(left_distance, right_distance, rtol=1e-5, atol=1e-6):
        raise ValueError(f"{label}: Gaussian point order or metric scale differs")


def _prepare_scene(
    control: IdentityControlConfig,
    scene: IdentitySceneInput,
) -> dict[str, Any]:
    asset = control.assets[scene.scene_id]
    bank = load_evidence_bank(scene.bank_dir, expected_scene_id=scene.scene_id)
    feature_xyz, affinity = load_affinity_feature_ply(asset.feature_ply)
    gaussian_xyz, scale_raw, opacity = load_gaussian_attributes_ply(asset.gaussian_ply)
    if len(feature_xyz) != bank.point_count or len(gaussian_xyz) != bank.point_count:
        raise ValueError(f"{scene.scene_id}: identity assets have the wrong point count")
    transform = np.asarray(scene.gaussian_to_gt_transform, dtype=np.float64)
    xyz_m = apply_transform(gaussian_xyz, transform)
    feature_xyz_m = apply_transform(feature_xyz, transform)
    _assert_point_order_geometry(xyz_m, feature_xyz_m, label=scene.scene_id)
    _assert_point_order_geometry(xyz_m, bank.xyz_m, label=scene.scene_id)
    # PLY scale values are in the same scene unit as PLY XYZ.  Infer the rigid
    # transform's uniform metric factor without allowing a sheared transform.
    singular = np.linalg.svd(transform[:3, :3], compute_uv=False)
    if not np.allclose(singular, singular.mean(), rtol=1e-5, atol=1e-7):
        raise ValueError(f"{scene.scene_id}: transform is not uniformly metric")
    scale_m = scale_raw * float(singular.mean())
    edge_index = local_edge_index(
        xyz_m,
        physical_neighbors=control.physical_neighbors,
        max_distance_m=control.max_edge_distance_m,
    )
    semantic = gaussian_soft_semantics(bank)
    features = edge_feature_matrix(
        edge_index=edge_index,
        xyz_m=xyz_m,
        affinity=affinity,
        soft_semantic=semantic,
        gaussian_scale_m=scale_m,
        opacity=opacity,
        bank=bank,
    )
    gt_coords, gt_scene = load_ground_truth_npz(scene.gt_npz, scene.scene_id)
    mapping, mapping_diagnostics = gt_point_to_gaussian_mapping(
        gt_coords, xyz_m, radius_m=scene.radius_m
    )
    gaussian_instance, gaussian_semantic = gaussian_gt_labels(
        gt_scene.semantic,
        gt_scene.instance,
        mapping,
        gaussian_count=bank.point_count,
    )
    labelled_index, labels = labelled_hard_edges(
        edge_index, gaussian_instance, gaussian_semantic
    )
    gt_objects = ground_truth_objects_from_arrays(
        gt_scene.semantic,
        gt_scene.instance,
        # ScanNet GT integer labels use the registered 20-class evaluation
        # order.  They must never be indexed through the 32-class evidence
        # codebook (whose index 3 is ``flower``, not ``tv``).
        class_names=scene.evaluation_class_names,
        min_region_size=scene.min_region_size,
        tiny_small_instance_ids=set(scene.tiny_small_instance_ids),
    )
    return {
        "scene": scene,
        "bank": bank,
        "edge_index": edge_index,
        "features": features,
        "affinity_cosine": features[:, 0],
        "labelled_index": labelled_index,
        "labels": labels,
        "mapping": mapping,
        "mapping_diagnostics": mapping_diagnostics,
        "gt_objects": gt_objects,
    }


def identity_control_result_is_complete(
    path: str | Path, *, expected_identity: Mapping[str, Any]
) -> bool:
    try:
        payload = load_json(path)
        training = payload.get("training")
        model = payload.get("model")
        validation = payload.get("validation")
        gate = payload.get("gate")
        gate_names = (
            "edge_auroc_at_least_0_75",
            "edge_auroc_delta_at_least_0_05",
            "new_iou050_objects_at_least_2",
        )
        return (
            payload.get("schema") == IDENTITY_CONTROL_RESULT_SCHEMA
            and payload.get("identity") == dict(expected_identity)
            and payload.get("formal_method") is False
            and payload.get("category_prior_tested") is False
            and payload.get("gt_used_for_training_and_evaluation_only") is True
            and isinstance(training, Mapping)
            and int(training.get("edge_count", 0)) > 0
            and isinstance(model, Mapping)
            and model.get("feature_names") == list(IDENTITY_FEATURE_NAMES)
            and isinstance(validation, Mapping)
            and validation.get("scene_id")
            == expected_identity.get("validation_scene_id")
            and isinstance(gate, Mapping)
            and all(isinstance(gate.get(name), bool) for name in gate_names)
            and isinstance(gate.get("passed"), bool)
            and gate.get("passed")
            == all(bool(gate[name]) for name in gate_names)
        )
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return False


def run_identity_edge_control(
    *,
    control: IdentityControlConfig,
    scenes: Mapping[str, IdentitySceneInput],
    output_path: str | Path,
) -> dict[str, Any]:
    """Run the preregistered offline identity capacity control exactly once."""

    identity = control.identity()
    target = Path(output_path)
    if identity_control_result_is_complete(target, expected_identity=identity):
        return load_json(target)
    if set(scenes) != set(control.assets):
        raise ValueError("identity-control scene inputs differ from registered assets")
    prepared = {
        scene_id: _prepare_scene(control, scenes[scene_id])
        for scene_id in (*control.train_scene_ids, control.validation_scene_id)
    }
    train_features: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    train_rows: list[dict[str, Any]] = []
    for offset, scene_id in enumerate(control.train_scene_ids):
        item = prepared[scene_id]
        selected = item["labelled_index"]
        features, labels, counts = _subsample_balanced(
            item["features"][selected],
            item["labels"],
            per_class_cap=control.max_training_edges_per_class,
            seed=control.seed + offset,
        )
        train_features.append(features)
        train_labels.append(labels)
        train_rows.append({"scene_id": scene_id, **counts})
    training_matrix = np.concatenate(train_features, axis=0)
    training_target = np.concatenate(train_labels, axis=0)
    model = fit_balanced_l2_logistic(training_matrix, training_target, l2_c=control.l2_c)

    validation = prepared[control.validation_scene_id]
    labelled = validation["labelled_index"]
    validation_labels = validation["labels"]
    raw_auc = binary_auroc(validation_labels, validation["affinity_cosine"][labelled])
    probabilities = model.predict_proba(validation["features"])
    model_auc = binary_auroc(validation_labels, probabilities[labelled])
    components = edge_components(
        validation["bank"].point_count,
        validation["edge_index"],
        probabilities >= control.probability_threshold,
        min_component_points=control.min_component_points,
    )
    learned_candidates = tuple(
        CleanCandidate(
            object_id=f"identity-{index}",
            gaussian_ids=points,
            class_id=None,
            winner_probability=1.0,
            view_consensus=1.0,
            detection_ratio=1.0,
        )
        for index, points in enumerate(components)
    )
    learned_evaluation = evaluate_candidates(
        learned_candidates,
        validation["gt_objects"],
        validation["mapping"],
        gaussian_count=validation["bank"].point_count,
    )
    baseline_candidates = _uniform_candidates(
        validation["scene"].uniform_output_json
    )
    baseline_evaluation = evaluate_candidates(
        baseline_candidates,
        validation["gt_objects"],
        validation["mapping"],
        gaussian_count=validation["bank"].point_count,
    )
    learned_matches = _matched_gt_050_count(learned_evaluation)
    baseline_matches = _matched_gt_050_count(baseline_evaluation)
    added_matches = learned_matches - baseline_matches
    gate = {
        "edge_auroc_at_least_0_75": model_auc >= 0.75,
        "edge_auroc_delta_at_least_0_05": model_auc - raw_auc >= 0.05,
        "new_iou050_objects_at_least_2": added_matches >= 2,
    }
    gate["passed"] = all(gate.values())
    result = {
        "schema": IDENTITY_CONTROL_RESULT_SCHEMA,
        "identity": identity,
        "formal_method": False,
        "category_prior_tested": False,
        "gt_used_for_training_and_evaluation_only": True,
        "training": {
            "scene_rows": train_rows,
            "edge_count": len(training_target),
            "negative_edge_count": int(np.count_nonzero(training_target == 0)),
            "positive_edge_count": int(np.count_nonzero(training_target == 1)),
        },
        "model": {
            "feature_names": list(IDENTITY_FEATURE_NAMES),
            "mean": model.mean.tolist(),
            "scale": model.scale.tolist(),
            "coefficients": model.coefficients.tolist(),
            "intercept": model.intercept,
            "l2_c": model.l2_c,
        },
        "validation": {
            "scene_id": control.validation_scene_id,
            "local_edge_count": len(validation["edge_index"]),
            "labelled_hard_edge_count": len(labelled),
            "raw_affinity_auroc": raw_auc,
            "learned_edge_auroc": model_auc,
            "edge_auroc_delta": model_auc - raw_auc,
            "learned_component_count": len(components),
            "uniform_matched_gt_iou050_count": baseline_matches,
            "learned_matched_gt_iou050_count": learned_matches,
            "new_matched_gt_iou050_count": added_matches,
            "mapping": validation["mapping_diagnostics"],
            "uniform_candidate_evaluation": baseline_evaluation,
            "learned_candidate_evaluation": learned_evaluation,
        },
        "gate": gate,
        "conclusion": (
            "a dedicated local identity edge has positive capacity; expanding it into a formal baseline requires a separate authorization"
            if gate["passed"]
            else "the fixed local Gaussian attributes and co-view evidence did not establish sufficient held-out identity capacity"
        ),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, result)
    if not identity_control_result_is_complete(target, expected_identity=identity):
        raise RuntimeError("identity-control result failed completion validation")
    return result


__all__ = [
    "IDENTITY_CONTROL_REGISTRATION_SCHEMA",
    "IDENTITY_CONTROL_RESULT_SCHEMA",
    "IDENTITY_CONTROL_SCHEMA",
    "IDENTITY_FEATURE_NAMES",
    "IDENTITY_TRAIN_SCENES",
    "IDENTITY_VALIDATION_SCENE",
    "BalancedLogisticModel",
    "IdentityAssetPaths",
    "IdentityControlConfig",
    "IdentitySceneInput",
    "binary_auroc",
    "edge_components",
    "edge_feature_matrix",
    "fit_balanced_l2_logistic",
    "gaussian_gt_labels",
    "gaussian_soft_semantics",
    "identity_control_result_is_complete",
    "labelled_hard_edges",
    "local_edge_index",
    "run_identity_edge_control",
]

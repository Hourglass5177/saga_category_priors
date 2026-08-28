from __future__ import annotations

"""Pure algorithms for the all-category denoising experiment.

The module deliberately separates three things which the historical
``postprocess.py`` mixed together:

* construction of one immutable, class-exclusive candidate bank;
* offline global-versus-class prior scoring of that same bank; and
* replay of the legacy global KNN/filter while accepted branch points are
  excluded from both operations.

There is no scene I/O, renderer, evaluator or GT access here.  HDBSCAN and
SciPy are imported only by the functions which need them so that lightweight
replay and score-only jobs do not inherit the clustering runtime.
"""

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .prediction_contract import PredictionContractResult, normalize_prediction
from .scannet import pca_obb

SCHEMA = "saga-category-denoise-bank-v1"
EXPECTED_CLASS_COUNT = 32
SEMANTIC_THRESHOLD = 0.7
SAMPLE_CAP = 5_000
MIN_CLUSTER_SIZE = 3
MIN_SAMPLES = 3
CLUSTER_SELECTION_EPSILON = 0.01
INSTANCE_WEIGHT = 0.5
SPATIAL_WEIGHT = 0.3
SEMANTIC_WEIGHT = 0.2
ASSIGNMENT_THRESHOLD = 0.3
ASSIGNMENT_TEMPERATURE = 10.0
GLOBAL_KNN_K = 256
GLOBAL_MIN_COUNT = 10
SCORE_THRESHOLD = 0.20
BOUNDARY_RADIUS_M = 0.05


@dataclass(frozen=True)
class Top1Assignment:
    """Result of normalized competition over the complete 32-class table."""

    top_class_index: np.ndarray
    top_score: np.ndarray
    branch_class_index: np.ndarray
    eligible_mask: np.ndarray
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class CandidateBank:
    """One frozen candidate pool shared byte-for-byte by U and D replay."""

    class_names: tuple[str, ...]
    saga20_names: tuple[str, ...]
    scene_scale_m_per_unit: float
    seed: int
    global_pre_knn: np.ndarray
    semantic_top1: np.ndarray
    semantic_top1_score: np.ndarray
    branch_full_labels: np.ndarray
    branch_core_labels: np.ndarray
    assignment_confidence: np.ndarray
    candidates: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]
    schema: str = SCHEMA

    @property
    def point_count(self) -> int:
        return len(self.global_pre_knn)


def _as_numpy(value: Any, dtype: np.dtype[Any] | type | None = None) -> np.ndarray:
    """Convert NumPy or CPU/GPU tensor-like input without importing torch."""

    converted = value
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    if hasattr(converted, "numpy"):
        converted = converted.numpy()
    return np.asarray(converted, dtype=dtype)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


def _normalize_rows(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return np.divide(array, norms, out=np.zeros_like(array), where=norms > 0)


def _validate_names(
    class_names: Sequence[str], saga20_names: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    classes = tuple(str(name) for name in class_names)
    branches = tuple(str(name) for name in saga20_names)
    if len(classes) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"complete semantic competition requires {EXPECTED_CLASS_COUNT} classes, "
            f"got {len(classes)}"
        )
    if len(set(classes)) != len(classes):
        raise ValueError("class_names must be unique")
    if len(set(branches)) != len(branches):
        raise ValueError("saga20_names must be unique")
    unknown = sorted(set(branches) - set(classes))
    if unknown:
        raise ValueError(f"SAGA20 names are absent from class_names: {unknown}")
    return classes, branches


def normalized_top1_32(
    semantic_features: Any,
    label_features: Any,
    class_names: Sequence[str],
    saga20_names: Sequence[str],
    threshold: float = SEMANTIC_THRESHOLD,
) -> Top1Assignment:
    """Run L2-normalized top-1 competition across all 32 classes.

    A point enters a branch only when the *global* winner belongs to SAGA20
    and clears ``threshold``.  It can therefore never be claimed by two
    classes and changing the order of ``saga20_names`` cannot change output.
    """

    classes, branches = _validate_names(class_names, saga20_names)
    semantic = _as_numpy(semantic_features, np.float64)
    labels = _as_numpy(label_features, np.float64)
    if semantic.ndim != 2 or labels.ndim != 2:
        raise ValueError("semantic_features and label_features must be matrices")
    if labels.shape[0] != len(classes) or semantic.shape[1] != labels.shape[1]:
        raise ValueError("semantic and label feature shapes do not match the class table")
    if not np.isfinite(semantic).all() or not np.isfinite(labels).all():
        raise ValueError("semantic and label features must be finite")

    similarity = _normalize_rows(semantic) @ _normalize_rows(labels).T
    top_class = np.argmax(similarity, axis=1).astype(np.int64, copy=False)
    top_score = similarity[np.arange(len(similarity)), top_class]
    branch_indices = np.asarray([classes.index(name) for name in branches], dtype=np.int64)
    belongs = np.isin(top_class, branch_indices)
    eligible = belongs & (top_score >= float(threshold))
    selected_class = np.where(eligible, top_class, -1).astype(np.int64, copy=False)
    return Top1Assignment(
        top_class_index=_readonly(top_class),
        top_score=_readonly(top_score.astype(np.float64, copy=False)),
        branch_class_index=_readonly(selected_class),
        eligible_mask=_readonly(eligible.astype(bool, copy=False)),
        class_names=classes,
    )


def stable_class_seed(seed: int, class_name: str) -> int:
    """Return a deterministic, non-cryptographic per-class RNG seed."""

    value = int(seed) & ((1 << 63) - 1)
    for index, character in enumerate(str(class_name)):
        value = (value * 1_000_003 + ord(character) + index + 1) & ((1 << 63) - 1)
    return value


def pca_sorted_extents_m(
    points_scene: Any, scene_scale_m_per_unit: float
) -> np.ndarray:
    """Match the train-prior PCA OBB definition and return sorted metric extents."""

    points = _as_numpy(points_scene, np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points_scene must contain at least one 3D point")
    scale = float(scene_scale_m_per_unit)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scene_scale_m_per_unit must be finite and positive")
    extents, _, _ = pca_obb(points * scale)
    return np.sort(np.maximum(np.asarray(extents, dtype=np.float64), 0.0))


def boundary_fixed_ratio_5cm(
    xyz_scene: Any,
    candidate_mask: Any,
    scene_scale_m_per_unit: float,
) -> float:
    """Match ``boundary_fixed:0.05`` used by the train statistics.

    The value is the fraction of candidate points having at least one *other*
    Gaussian within 5 cm which is outside the candidate.  It is not the ratio
    of crossing edges, and points with no non-self neighbor are non-boundary.
    """

    from scipy.spatial import cKDTree

    xyz = _as_numpy(xyz_scene, np.float64)
    mask = _as_numpy(candidate_mask, bool)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or mask.shape != (len(xyz),):
        raise ValueError("xyz_scene and candidate_mask must describe the same 3D points")
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return 0.0
    scale = float(scene_scale_m_per_unit)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scene_scale_m_per_unit must be finite and positive")
    xyz_m = xyz * scale
    tree = cKDTree(xyz_m)
    return _boundary_fixed_ratio_with_tree(xyz_m, mask, tree)


def _boundary_fixed_ratio_with_tree(
    xyz_m: np.ndarray, candidate_mask: np.ndarray, tree: Any
) -> float:
    """Compute the fixed-radius boundary statistic with a shared scene tree."""

    mask = np.asarray(candidate_mask, dtype=bool)
    neighborhoods = tree.query_ball_point(
        xyz_m[np.flatnonzero(mask)], BOUNDARY_RADIUS_M, workers=1
    )
    indices = np.flatnonzero(mask)
    is_boundary = np.zeros(len(indices), dtype=bool)
    for row, (point_index, neighbors) in enumerate(zip(indices, neighborhoods)):
        neighbor_indices = np.asarray(neighbors, dtype=np.int64)
        neighbor_indices = neighbor_indices[neighbor_indices != point_index]
        if len(neighbor_indices):
            is_boundary[row] = bool(np.any(~mask[neighbor_indices]))
    return float(is_boundary.mean())


def _default_hdbscan_factory(**kwargs: Any) -> Any:
    try:
        from hdbscan import HDBSCAN
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError("candidate-bank construction requires hdbscan") from exc
    return HDBSCAN(**kwargs)


def _scaled_distance(distance: np.ndarray) -> np.ndarray:
    maximum = float(np.max(distance)) if distance.size else 0.0
    return distance / (maximum + 1e-8) if maximum > 0 else distance


def _softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=1, keepdims=True)


def build_candidate_bank(
    instance_features: Any,
    semantic_features: Any,
    xyz_scene: Any,
    label_features: Any,
    class_names: Sequence[str],
    saga20_names: Sequence[str],
    global_pre_knn: Any,
    scene_scale_m_per_unit: float,
    seed: int = 42,
    hdbscan_factory: Callable[..., Any] | None = None,
) -> CandidateBank:
    """Build one class-exclusive bank using the teacher branch mechanics.

    ``hdbscan_factory`` is injectable for tests and must accept the same
    keyword arguments as :class:`hdbscan.HDBSCAN`, returning an object with
    ``fit_predict(distance_matrix)``.
    """

    from scipy.spatial.distance import cdist

    classes, branches = _validate_names(class_names, saga20_names)
    instance = _as_numpy(instance_features, np.float64)
    semantic = _as_numpy(semantic_features, np.float64)
    xyz = _as_numpy(xyz_scene, np.float64)
    global_labels = _as_numpy(global_pre_knn, np.int64)
    count = len(xyz)
    if (
        instance.ndim != 2
        or semantic.ndim != 2
        or xyz.shape != (count, 3)
        or global_labels.shape != (count,)
        or len(instance) != count
        or len(semantic) != count
    ):
        raise ValueError("features, xyz_scene and global_pre_knn must share a point axis")
    if not np.isfinite(instance).all() or not np.isfinite(xyz).all():
        raise ValueError("instance features and xyz must be finite")
    if np.any(global_labels < -1):
        raise ValueError("global_pre_knn may only use -1 as its negative label")
    scale = float(scene_scale_m_per_unit)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scene_scale_m_per_unit must be finite and positive")

    top1 = normalized_top1_32(
        semantic, label_features, classes, branches, SEMANTIC_THRESHOLD
    )
    normed_instance = _normalize_rows(instance)
    minimum = xyz.min(axis=0)
    span = xyz.max(axis=0) - minimum
    standardized_xyz = (xyz - minimum) / np.where(span > 0, span, 1.0)
    branch_full = np.full(count, -1, dtype=np.int64)
    branch_core = np.full(count, -1, dtype=np.int64)
    assignment_confidence = np.zeros(count, dtype=np.float64)
    candidate_rows: list[dict[str, Any]] = []
    class_diagnostics: dict[str, dict[str, Any]] = {}
    next_candidate_id = 0
    factory = hdbscan_factory or _default_hdbscan_factory

    # Lexicographic processing plus a per-name seed makes the result invariant
    # to the input ordering of saga20_names.
    for class_name in sorted(branches):
        class_index = classes.index(class_name)
        selected_indices = np.flatnonzero(
            top1.eligible_mask & (top1.branch_class_index == class_index)
        )
        selected_count = len(selected_indices)
        diagnostic: dict[str, Any] = {
            "class_index": class_index,
            "selected_points": selected_count,
            "sampled_points": 0,
            "hdbscan_noise_points": 0,
            "candidate_count": 0,
        }
        class_diagnostics[class_name] = diagnostic
        if selected_count < MIN_CLUSTER_SIZE:
            continue

        selected_features = normed_instance[selected_indices]
        selected_xyz = standardized_xyz[selected_indices]
        selected_scores = top1.top_score[selected_indices]
        sample_count = min(selected_count, SAMPLE_CAP)
        rng = np.random.default_rng(stable_class_seed(seed, class_name))
        sampled_local = rng.permutation(selected_count)[:sample_count]
        sampled_features = selected_features[sampled_local]
        sampled_xyz = selected_xyz[sampled_local]
        sampled_scores = selected_scores[sampled_local]
        diagnostic["sampled_points"] = int(sample_count)

        instance_distance = np.maximum(
            1.0 - sampled_features @ sampled_features.T, 0.0
        )
        spatial_distance = cdist(sampled_xyz, sampled_xyz, metric="euclidean")
        semantic_distance = np.clip(
            1.0 - np.outer(sampled_scores, sampled_scores), 0.0, 1.0
        )
        hybrid_distance = (
            INSTANCE_WEIGHT * _scaled_distance(instance_distance)
            + SPATIAL_WEIGHT * _scaled_distance(spatial_distance)
            + SEMANTIC_WEIGHT * semantic_distance
        )
        clusterer = factory(
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=MIN_SAMPLES,
            cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
            allow_single_cluster=False,
            metric="precomputed",
        )
        cluster_labels = np.asarray(
            clusterer.fit_predict(hybrid_distance.astype(np.float64, copy=False)),
            dtype=np.int64,
        )
        if cluster_labels.shape != (sample_count,):
            raise ValueError("HDBSCAN returned an invalid label vector")
        diagnostic["hdbscan_noise_points"] = int(np.count_nonzero(cluster_labels < 0))
        raw_cluster_ids = [
            int(value) for value in np.unique(cluster_labels) if int(value) >= 0
        ]
        if not raw_cluster_ids:
            continue

        feature_centers = []
        xyz_centers = []
        for raw_cluster_id in raw_cluster_ids:
            mask = cluster_labels == raw_cluster_id
            center = sampled_features[mask].mean(axis=0, keepdims=True)
            feature_centers.append(_normalize_rows(center)[0])
            xyz_centers.append(sampled_xyz[mask].mean(axis=0))
        feature_centers_array = np.asarray(feature_centers, dtype=np.float64)
        xyz_centers_array = np.asarray(xyz_centers, dtype=np.float64)
        feature_similarity = np.clip(
            selected_features @ feature_centers_array.T, -1.0, 1.0
        )
        xyz_similarity = np.exp(-cdist(selected_xyz, xyz_centers_array))
        hybrid_similarity = (
            INSTANCE_WEIGHT * feature_similarity
            + (1.0 - INSTANCE_WEIGHT) * xyz_similarity
        )
        probability = _softmax(hybrid_similarity * ASSIGNMENT_TEMPERATURE)
        assigned_center = np.argmax(probability, axis=1)
        assigned_confidence = probability[
            np.arange(selected_count), assigned_center
        ]
        assigned_center[assigned_confidence < ASSIGNMENT_THRESHOLD] = -1

        for center_index, raw_cluster_id in enumerate(raw_cluster_ids):
            full_local = assigned_center == center_index
            if int(np.count_nonzero(full_local)) < MIN_CLUSTER_SIZE:
                continue
            candidate_id = next_candidate_id
            next_candidate_id += 1
            full_indices = selected_indices[full_local]
            core_local = sampled_local[cluster_labels == raw_cluster_id]
            core_indices = selected_indices[core_local]
            branch_full[full_indices] = candidate_id
            branch_core[core_indices] = candidate_id
            assignment_confidence[full_indices] = assigned_confidence[full_local]
            extents = pca_sorted_extents_m(xyz[full_indices], scale)
            candidate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "branch_class": class_name,
                    "branch_class_index": class_index,
                    "hdbscan_cluster_id": raw_cluster_id,
                    "semantic_selected_point_count": selected_count,
                    "sampled_point_count": sample_count,
                    "core_point_count": len(core_indices),
                    "full_point_count": len(full_indices),
                    "assignment_confidence_mean": float(
                        assigned_confidence[full_local].mean()
                    ),
                    "metric_extents_m": extents.tolist(),
                }
            )
        diagnostic["candidate_count"] = int(
            sum(row["branch_class"] == class_name for row in candidate_rows)
        )

    # Boundary statistics need the final, class-exclusive candidate labels.
    # The all-scene tree is shared across candidates; rebuilding it for each
    # candidate dominates runtime on a 30k-Gaussian scene.
    from scipy.spatial import cKDTree

    xyz_m = xyz * scale
    boundary_tree = cKDTree(xyz_m)
    enriched_rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        candidate_id = int(row["candidate_id"])
        enriched = dict(row)
        enriched["boundary_ratio_5cm"] = _boundary_fixed_ratio_with_tree(
            xyz_m, branch_full == candidate_id, boundary_tree
        )
        enriched_rows.append(enriched)

    bank = CandidateBank(
        class_names=classes,
        saga20_names=tuple(sorted(branches)),
        scene_scale_m_per_unit=scale,
        seed=int(seed),
        global_pre_knn=_readonly(global_labels.astype(np.int64, copy=True)),
        semantic_top1=_readonly(top1.top_class_index.astype(np.int64, copy=True)),
        semantic_top1_score=_readonly(top1.top_score.astype(np.float64, copy=True)),
        branch_full_labels=_readonly(branch_full),
        branch_core_labels=_readonly(branch_core),
        assignment_confidence=_readonly(assignment_confidence),
        candidates=tuple(enriched_rows),
        diagnostics={
            "semantic_threshold": SEMANTIC_THRESHOLD,
            "sample_cap": SAMPLE_CAP,
            "min_cluster_size": MIN_CLUSTER_SIZE,
            "min_samples": MIN_SAMPLES,
            "weights": {
                "instance": INSTANCE_WEIGHT,
                "spatial": SPATIAL_WEIGHT,
                "semantic": SEMANTIC_WEIGHT,
            },
            "assignment_threshold": ASSIGNMENT_THRESHOLD,
            "class_diagnostics": class_diagnostics,
        },
    )
    _validate_bank(bank)
    return bank


def _validate_bank(bank: CandidateBank) -> None:
    if bank.schema != SCHEMA:
        raise ValueError(f"unsupported candidate-bank schema: {bank.schema}")
    _validate_names(bank.class_names, bank.saga20_names)
    point_count = bank.point_count
    arrays = {
        "semantic_top1": bank.semantic_top1,
        "semantic_top1_score": bank.semantic_top1_score,
        "branch_full_labels": bank.branch_full_labels,
        "branch_core_labels": bank.branch_core_labels,
        "assignment_confidence": bank.assignment_confidence,
    }
    for name, value in arrays.items():
        if np.asarray(value).shape != (point_count,):
            raise ValueError(f"candidate-bank array {name} has an invalid shape")
    if np.any(np.asarray(bank.global_pre_knn) < -1):
        raise ValueError("global_pre_knn may only use -1 as its negative label")
    semantic_top1 = np.asarray(bank.semantic_top1, dtype=np.int64)
    if np.any((semantic_top1 < 0) | (semantic_top1 >= len(bank.class_names))):
        raise ValueError("semantic_top1 contains an invalid class index")
    semantic_scores = np.asarray(bank.semantic_top1_score, dtype=np.float64)
    confidence = np.asarray(bank.assignment_confidence, dtype=np.float64)
    if not np.isfinite(semantic_scores).all():
        raise ValueError("semantic_top1_score must be finite")
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("assignment_confidence must be finite and in [0, 1]")
    candidate_ids = [int(row["candidate_id"]) for row in bank.candidates]
    if candidate_ids != list(range(len(candidate_ids))):
        raise ValueError("candidate IDs must be contiguous and follow row order")
    valid_ids = np.asarray(candidate_ids, dtype=np.int64)
    for name, labels in (
        ("branch_full_labels", bank.branch_full_labels),
        ("branch_core_labels", bank.branch_core_labels),
    ):
        observed = np.unique(np.asarray(labels, dtype=np.int64))
        if np.any(observed < -1):
            raise ValueError(f"{name} may only use -1 as its negative label")
        observed = observed[observed >= 0]
        if not np.all(np.isin(observed, valid_ids)):
            raise ValueError(f"{name} contains an undeclared candidate ID")
    for row in bank.candidates:
        candidate_id = int(row["candidate_id"])
        class_name = str(row["branch_class"])
        class_index = int(row["branch_class_index"])
        if (
            class_name not in bank.saga20_names
            or not 0 <= class_index < len(bank.class_names)
            or bank.class_names[class_index] != class_name
        ):
            raise ValueError(f"candidate {candidate_id} has an invalid branch class")
        full_mask = np.asarray(bank.branch_full_labels) == candidate_id
        core_mask = np.asarray(bank.branch_core_labels) == candidate_id
        if int(np.count_nonzero(full_mask)) != int(row["full_point_count"]):
            raise ValueError(f"candidate {candidate_id} full point count is inconsistent")
        if int(np.count_nonzero(core_mask)) != int(row["core_point_count"]):
            raise ValueError(f"candidate {candidate_id} core point count is inconsistent")
        semantic_mask = full_mask | core_mask
        if np.any(semantic_top1[semantic_mask] != class_index):
            raise ValueError(f"candidate {candidate_id} contains another class winner")
        if np.any(semantic_scores[semantic_mask] < SEMANTIC_THRESHOLD):
            raise ValueError(f"candidate {candidate_id} contains sub-threshold semantic points")


def _resolve_bank_paths(
    path: str | Path, json_path: str | Path | None = None
) -> tuple[Path, Path | None]:
    location = Path(path)
    if location.suffix.lower() == ".npz":
        npz_path = location
        sidecar = Path(json_path) if json_path is not None else None
    else:
        npz_path = location / "bank_labels.npz"
        sidecar = Path(json_path) if json_path is not None else location / "candidates.json"
    return npz_path, sidecar


def _bank_metadata(bank: CandidateBank) -> dict[str, Any]:
    return {
        "schema": bank.schema,
        "point_count": bank.point_count,
        "class_names": list(bank.class_names),
        "saga20_names": list(bank.saga20_names),
        "scene_scale_m_per_unit": bank.scene_scale_m_per_unit,
        "seed": bank.seed,
        "candidates": [dict(row) for row in bank.candidates],
        "diagnostics": bank.diagnostics,
    }


def save_candidate_bank(
    bank: CandidateBank,
    npz_path: str | Path,
    json_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """Save a bank to an NPZ file or the canonical scene-bank directory."""

    _validate_bank(bank)
    destination, sidecar = _resolve_bank_paths(npz_path, json_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = _bank_metadata(bank)
    encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with destination.open("wb") as handle:
        np.savez_compressed(
            handle,
            global_pre_knn=bank.global_pre_knn,
            semantic_top1=bank.semantic_top1,
            semantic_top1_score=bank.semantic_top1_score,
            branch_full_labels=bank.branch_full_labels,
            branch_core_labels=bank.branch_core_labels,
            assignment_confidence=bank.assignment_confidence,
            metadata_json=np.asarray(encoded),
        )
    if sidecar is not None:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return destination, sidecar


def load_candidate_bank(npz_path: str | Path) -> CandidateBank:
    """Load and validate a bank from an NPZ file or canonical bank directory."""

    source, _ = _resolve_bank_paths(npz_path)
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "global_pre_knn",
            "semantic_top1",
            "semantic_top1_score",
            "branch_full_labels",
            "branch_core_labels",
            "assignment_confidence",
            "metadata_json",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"candidate bank is missing arrays: {missing}")
        metadata = json.loads(str(archive["metadata_json"].item()))
        bank = CandidateBank(
            class_names=tuple(metadata["class_names"]),
            saga20_names=tuple(metadata["saga20_names"]),
            scene_scale_m_per_unit=float(metadata["scene_scale_m_per_unit"]),
            seed=int(metadata["seed"]),
            global_pre_knn=_readonly(archive["global_pre_knn"].astype(np.int64)),
            semantic_top1=_readonly(archive["semantic_top1"].astype(np.int64)),
            semantic_top1_score=_readonly(
                archive["semantic_top1_score"].astype(np.float64)
            ),
            branch_full_labels=_readonly(
                archive["branch_full_labels"].astype(np.int64)
            ),
            branch_core_labels=_readonly(
                archive["branch_core_labels"].astype(np.int64)
            ),
            assignment_confidence=_readonly(
                archive["assignment_confidence"].astype(np.float64)
            ),
            candidates=tuple(dict(row) for row in metadata["candidates"]),
            diagnostics=dict(metadata.get("diagnostics", {})),
            schema=str(metadata["schema"]),
        )
    _validate_bank(bank)
    if int(metadata.get("point_count", -1)) != bank.point_count:
        raise ValueError("candidate-bank metadata point count does not match arrays")
    return bank


def attach_candidate_votes(
    bank: CandidateBank,
    ratios_by_candidate: Mapping[int | str, Sequence[float] | np.ndarray],
    class_names: Sequence[str],
) -> CandidateBank:
    """Attach fixed 2D vote evidence and Q without changing any bank labels."""

    _validate_bank(bank)
    classes = tuple(str(name) for name in class_names)
    if classes != bank.class_names:
        raise ValueError("vote class_names must exactly match the bank class table")
    enriched: list[dict[str, Any]] = []
    for candidate in bank.candidates:
        candidate_id = int(candidate["candidate_id"])
        raw_ratio = ratios_by_candidate.get(
            candidate_id, ratios_by_candidate.get(str(candidate_id))
        )
        if raw_ratio is None:
            raise ValueError(f"missing vote ratios for candidate {candidate_id}")
        ratio = _as_numpy(raw_ratio, np.float64)
        if ratio.shape != (len(classes),) or not np.isfinite(ratio).all():
            raise ValueError(f"candidate {candidate_id} vote ratios have an invalid shape")
        if np.any(ratio < 0) or float(ratio.sum()) > 1.0 + 1e-6:
            raise ValueError(f"candidate {candidate_id} vote ratios are not probabilities")
        background_ratio = max(0.0, 1.0 - float(ratio.sum()))
        votes_with_background = np.concatenate(
            (ratio, np.asarray([background_ratio], dtype=np.float64))
        )
        maximum = float(np.max(votes_with_background))
        winner_indices = np.flatnonzero(
            np.isclose(votes_with_background, maximum, rtol=0.0, atol=1e-12)
        )
        winner_unique = bool(len(winner_indices) == 1 and maximum > 0.0)
        raw_winner = int(winner_indices[0]) if winner_unique else -1
        winner_is_background = raw_winner == len(classes)
        winner_index = -1 if winner_is_background else raw_winner
        branch_index = int(candidate["branch_class_index"])
        branch_ratio = float(ratio[branch_index])
        row = dict(candidate)
        row.update(
            {
                "vote_winner_index": winner_index,
                "vote_winner": (
                    classes[winner_index]
                    if winner_index >= 0
                    else "background" if winner_is_background else None
                ),
                "vote_winner_unique": winner_unique,
                "branch_vote_ratio": branch_ratio,
                "background_vote_ratio": background_ratio,
                "base_score": float(
                    np.clip(
                        branch_ratio
                        * float(candidate["assignment_confidence_mean"]),
                        0.0,
                        1.0,
                    )
                ),
            }
        )
        enriched.append(row)
    updated = replace(bank, candidates=tuple(enriched))
    _validate_bank(updated)
    return updated


def _global_node(priors: Mapping[str, Any]) -> Mapping[str, Any]:
    node = priors.get("global")
    if not isinstance(node, Mapping) or not isinstance(node.get("shrunk"), Mapping):
        raise TypeError("category priors are missing a global shrunk node")
    return node


def _class_node(
    priors: Mapping[str, Any], class_name: str
) -> Mapping[str, Any]:
    categories = priors.get("categories")
    node = categories.get(class_name) if isinstance(categories, Mapping) else None
    if isinstance(node, Mapping) and isinstance(node.get("shrunk"), Mapping):
        return node
    return _global_node(priors)


def _summary(
    node: Mapping[str, Any], section: str, field: str
) -> Mapping[str, Any]:
    shrunk = node.get("shrunk")
    subsection = shrunk.get(section) if isinstance(shrunk, Mapping) else None
    value = subsection.get(field) if isinstance(subsection, Mapping) else None
    if not isinstance(value, Mapping):
        raise TypeError(f"prior node is missing shrunk.{section}.{field}")
    return value


def size_compatibility(
    candidate: Mapping[str, Any], node: Mapping[str, Any]
) -> float:
    extents = np.sort(
        np.maximum(np.asarray(candidate["metric_extents_m"], dtype=np.float64), 1e-9)
    )
    if extents.shape != (3,):
        raise ValueError("candidate metric_extents_m must contain three values")
    fields = ("log_extent_short_m", "log_extent_mid_m", "log_extent_long_m")
    z_values: list[float] = []
    for extent, field in zip(extents, fields):
        summary = _summary(node, "geometry", field)
        q50 = float(summary["q50"])
        q75 = float(summary["q75"])
        z_values.append(
            max(0.0, math.log(float(extent)) - q50) / max(q75 - q50, 1e-6)
        )
    return float(
        math.exp(-0.5 * float(np.mean(np.minimum(np.square(z_values), 25.0))))
    )


def smoothness_compatibility(
    candidate: Mapping[str, Any], node: Mapping[str, Any]
) -> float:
    boundary = float(candidate["boundary_ratio_5cm"])
    if not np.isfinite(boundary) or not 0.0 <= boundary <= 1.0:
        raise ValueError("candidate boundary_ratio_5cm must be finite and in [0, 1]")
    summary = _summary(node, "neighborhood", "boundary_fixed:0.05")
    q50 = float(summary["q50"])
    q75 = float(summary["q75"])
    z_value = max(0.0, boundary - q50) / max(q75 - q50, 1e-6)
    return float(math.exp(-0.5 * min(z_value * z_value, 25.0)))


def support_threshold(
    priors: Mapping[str, Any], class_name: str, mode: str
) -> int:
    if mode == "uniform":
        return 5
    if mode != "class":
        raise ValueError("mode must be 'uniform' or 'class'")
    global_area = math.exp(
        float(_summary(_global_node(priors), "geometry", "log_surface_area_m2")["q50"])
    )
    class_area = math.exp(
        float(_summary(_class_node(priors, class_name), "geometry", "log_surface_area_m2")["q50"])
    )
    return int(np.clip(round(5.0 * math.sqrt(class_area / global_area)), 3, 10))


def materialize_category_denoise_params(
    priors: Mapping[str, Any], saga20_names: Sequence[str]
) -> dict[str, Any]:
    """Return the exact readable global/class table consumed by replay."""

    def row(class_name: str | None) -> dict[str, Any]:
        node = _global_node(priors) if class_name is None else _class_node(
            priors, class_name
        )
        geometry = {
            field: {
                "q50": float(_summary(node, "geometry", field)["q50"]),
                "q75": float(_summary(node, "geometry", field)["q75"]),
            }
            for field in (
                "log_extent_short_m",
                "log_extent_mid_m",
                "log_extent_long_m",
            )
        }
        area = float(
            _summary(node, "geometry", "log_surface_area_m2")["q50"]
        )
        boundary = _summary(node, "neighborhood", "boundary_fixed:0.05")
        return {
            "extent_log_m": geometry,
            "log_surface_area_m2_q50": area,
            "boundary_fixed_0.05": {
                "q50": float(boundary["q50"]),
                "q75": float(boundary["q75"]),
            },
            "support_threshold": (
                5
                if class_name is None
                else support_threshold(priors, class_name, "class")
            ),
        }

    names = tuple(sorted(map(str, saga20_names)))
    return {
        "schema": "saga-category-denoise-params-v1",
        "uniform": row(None),
        "classes": {name: row(name) for name in names},
        "fixed": {
            "semantic_threshold": SEMANTIC_THRESHOLD,
            "sample_cap": SAMPLE_CAP,
            "min_cluster_size": MIN_CLUSTER_SIZE,
            "min_samples": MIN_SAMPLES,
            "epsilon": CLUSTER_SELECTION_EPSILON,
            "assignment_threshold": ASSIGNMENT_THRESHOLD,
            "vote_threshold": ASSIGNMENT_THRESHOLD,
            "score_threshold": SCORE_THRESHOLD,
            "global_knn_k": GLOBAL_KNN_K,
            "global_min_count": GLOBAL_MIN_COUNT,
        },
    }


def score_bank_candidates(
    bank: CandidateBank,
    priors: Mapping[str, Any],
    mode: str,
    *,
    score_threshold: float = SCORE_THRESHOLD,
    vote_threshold: float = ASSIGNMENT_THRESHOLD,
) -> list[dict[str, Any]]:
    """Score one frozen bank using either global or class-shrunk statistics."""

    _validate_bank(bank)
    if mode not in {"uniform", "class"}:
        raise ValueError("mode must be 'uniform' or 'class'")
    global_node = _global_node(priors)
    decisions: list[dict[str, Any]] = []
    for candidate in bank.candidates:
        if "base_score" not in candidate or "vote_winner_index" not in candidate:
            raise ValueError("candidate votes must be attached before scoring")
        class_name = str(candidate["branch_class"])
        node = global_node if mode == "uniform" else _class_node(priors, class_name)
        q_value = float(candidate["base_score"])
        g_value = size_compatibility(candidate, node)
        b_value = smoothness_compatibility(candidate, node)
        support = support_threshold(priors, class_name, mode)
        score = float(np.clip(q_value * g_value * b_value, 0.0, 1.0))
        vote_matches = bool(candidate.get("vote_winner_unique", False)) and int(
            candidate["vote_winner_index"]
        ) == int(candidate["branch_class_index"])
        accepted = bool(
            vote_matches
            and float(candidate["branch_vote_ratio"]) >= float(vote_threshold)
            and int(candidate["core_point_count"]) >= support
            and score >= float(score_threshold)
        )
        decisions.append(
            {
                "candidate_id": int(candidate["candidate_id"]),
                "branch_class": class_name,
                "mode": mode,
                "Q": q_value,
                "G_size": g_value,
                "B_smooth": b_value,
                "score": score,
                "ap_score": q_value,
                "support_threshold": support,
                "core_point_count": int(candidate["core_point_count"]),
                "vote_matches_branch": bool(vote_matches),
                "accepted": accepted,
            }
        )
    return decisions


def _majority_vote_nearest_tie(
    neighbor_labels: np.ndarray, label_values: np.ndarray
) -> np.ndarray:
    """Vectorized majority vote with the legacy nearest-neighbor tie rule."""

    rows, width = neighbor_labels.shape
    encoded = np.searchsorted(label_values, neighbor_labels)
    row_ids = np.repeat(np.arange(rows, dtype=np.int64), width)
    keys = row_ids * len(label_values) + encoded.reshape(-1)
    unique_keys, first, counts = np.unique(
        keys, return_index=True, return_counts=True
    )
    pair_rows = unique_keys // len(label_values)
    pair_labels = unique_keys % len(label_values)
    maximum = np.zeros(rows, dtype=np.int64)
    np.maximum.at(maximum, pair_rows, counts)
    tied = counts == maximum[pair_rows]
    best_first = np.full(rows, keys.size, dtype=np.int64)
    np.minimum.at(best_first, pair_rows[tied], first[tied])
    chosen = tied & (first == best_first[pair_rows])
    output = np.empty(rows, dtype=label_values.dtype)
    output[pair_rows[chosen]] = label_values[pair_labels[chosen]]
    return output


def replay_protected_denoise(
    xyz_scene: Any,
    bank: CandidateBank,
    accepted_rows: Sequence[Mapping[str, Any]],
    k: int = GLOBAL_KNN_K,
    min_count: int = GLOBAL_MIN_COUNT,
    *,
    chunk_size: int = 8_192,
) -> tuple[np.ndarray, dict[int, str], dict[int, float], dict[str, Any]]:
    """Run global KNN/filter only on unprotected points, then insert branches."""

    # Use the exact same neighbour implementation as the historical B0
    # ``filter3d`` path.  Different KD-tree implementations may order
    # equidistant neighbours differently, which would break the required
    # pointwise equivalence through B0's first-neighbour tie rule.
    from scipy.spatial import KDTree

    _validate_bank(bank)
    xyz = _as_numpy(xyz_scene, np.float64)
    if xyz.shape != (bank.point_count, 3):
        raise ValueError("xyz_scene does not match the candidate bank")
    accepted = {
        int(row["candidate_id"]): row
        for row in accepted_rows
        if bool(row.get("accepted", True))
    }
    if len(accepted) != sum(bool(row.get("accepted", True)) for row in accepted_rows):
        raise ValueError("accepted_rows contains duplicate candidate IDs")
    candidates = {int(row["candidate_id"]): row for row in bank.candidates}
    unknown = sorted(set(accepted) - set(candidates))
    if unknown:
        raise ValueError(f"accepted_rows refers to unknown candidate IDs: {unknown}")

    protected = np.isin(bank.branch_full_labels, np.asarray(sorted(accepted), dtype=np.int64))
    active_indices = np.flatnonzero(~protected)
    source_labels = bank.global_pre_knn[active_indices]
    voted = np.empty(len(active_indices), dtype=np.int64)
    if len(active_indices):
        k_effective = min(max(int(k), 1), len(active_indices))
        tree = KDTree(xyz[active_indices])
        label_values = np.unique(source_labels)
        chunk = max(int(chunk_size), 1)
        for start in range(0, len(active_indices), chunk):
            stop = min(start + chunk, len(active_indices))
            _, neighbor_indices = tree.query(
                xyz[active_indices[start:stop]], k=k_effective
            )
            neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64).reshape(
                stop - start, k_effective
            )
            voted[start:stop] = _majority_vote_nearest_tie(
                source_labels[neighbor_indices], label_values
            )

    # The historical filter_num(10) is applied only to the unprotected global
    # path.  Rejected branch points remain active and therefore fall back to it.
    filtered = voted.copy()
    values, counts = np.unique(voted[voted >= 0], return_counts=True)
    for value, count in zip(values, counts):
        if int(count) < int(min_count):
            filtered[voted == value] = -1
    output = np.full(bank.point_count, -1, dtype=np.int64)
    output[active_indices] = filtered
    next_instance = int(output[output >= 0].max()) + 1 if np.any(output >= 0) else 0
    class_by_id: dict[int, str] = {}
    score_by_id: dict[int, float] = {}
    inserted: dict[str, int] = {}
    for candidate_id in sorted(
        accepted,
        key=lambda item: (str(candidates[item]["branch_class"]), int(item)),
    ):
        mask = bank.branch_full_labels == candidate_id
        output[mask] = next_instance
        class_by_id[next_instance] = str(candidates[candidate_id]["branch_class"])
        decision = accepted[candidate_id]
        if "ap_score" in decision:
            ap_score = decision["ap_score"]
        elif "base_score" in candidates[candidate_id]:
            ap_score = candidates[candidate_id]["base_score"]
        else:
            raise ValueError(
                f"accepted candidate {candidate_id} is missing its AP score"
            )
        score_by_id[next_instance] = float(ap_score)
        inserted[str(candidate_id)] = next_instance
        next_instance += 1

    diagnostics = {
        "accepted_candidate_ids": sorted(accepted),
        "protected_gaussian_count": int(np.count_nonzero(protected)),
        "active_global_gaussian_count": len(active_indices),
        "global_instance_count_before_filter": len(values),
        "global_instance_count_after_filter": len(np.unique(filtered[filtered >= 0])),
        "inserted_candidate_to_instance": inserted,
        "protected_instance_survival_rate": 1.0 if accepted else None,
        "protected_class_rewrite_rate": 0.0 if accepted else None,
        "knn_k_effective": int(min(max(int(k), 1), len(active_indices)))
        if len(active_indices)
        else 0,
        "global_min_count": int(min_count),
    }
    return _readonly(output), class_by_id, score_by_id, diagnostics


def _pca_bbox_corners(points: np.ndarray) -> list[float]:
    extents, center, axes = pca_obb(points)
    half = np.asarray(extents, dtype=np.float64) / 2.0
    signs = np.asarray(
        [
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, -1],
            [1, -1, 1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, -1],
            [-1, -1, 1],
        ],
        dtype=np.float64,
    )
    corners = np.asarray(center) + (signs * half) @ np.asarray(axes).T
    return corners.reshape(-1).tolist()


def build_strict_prediction_metadata(
    point_labels: Any,
    xyz_scene: Any,
    class_by_id: Mapping[int | str, str],
    score_by_id: Mapping[int | str, float],
) -> PredictionContractResult:
    """Build complete class/score/bbox metadata and enforce the sole output truth."""

    labels = _as_numpy(point_labels, np.int64)
    xyz = _as_numpy(xyz_scene, np.float64)
    if labels.ndim != 1 or xyz.shape != (len(labels), 3):
        raise ValueError("point_labels and xyz_scene must describe the same points")
    instances: dict[str, dict[str, Any]] = {}
    for instance_id in np.unique(labels[labels >= 0]):
        raw_id = int(instance_id)
        class_name = class_by_id.get(raw_id, class_by_id.get(str(raw_id)))
        score = score_by_id.get(raw_id, score_by_id.get(str(raw_id)))
        if not isinstance(class_name, str) or not class_name:
            raise ValueError(f"instance {raw_id} is missing its class")
        if score is None:
            raise ValueError(f"instance {raw_id} is missing its score")
        mask = labels == raw_id
        instances[str(raw_id)] = {
            "class": class_name,
            "score": float(score),
            "bbox": _pca_bbox_corners(xyz[mask]),
        }
    result = normalize_prediction(labels, instances)
    if result.audit["orphan_gaussian_count"] != 0:
        raise RuntimeError("strict metadata construction produced orphan labels")
    return result


__all__ = [
    "ASSIGNMENT_THRESHOLD",
    "BOUNDARY_RADIUS_M",
    "GLOBAL_KNN_K",
    "GLOBAL_MIN_COUNT",
    "MIN_CLUSTER_SIZE",
    "MIN_SAMPLES",
    "SAMPLE_CAP",
    "SCORE_THRESHOLD",
    "SEMANTIC_THRESHOLD",
    "CandidateBank",
    "Top1Assignment",
    "attach_candidate_votes",
    "boundary_fixed_ratio_5cm",
    "build_candidate_bank",
    "build_strict_prediction_metadata",
    "load_candidate_bank",
    "materialize_category_denoise_params",
    "normalized_top1_32",
    "pca_sorted_extents_m",
    "replay_protected_denoise",
    "save_candidate_bank",
    "score_bank_candidates",
    "size_compatibility",
    "smoothness_compatibility",
    "stable_class_seed",
    "support_threshold",
]

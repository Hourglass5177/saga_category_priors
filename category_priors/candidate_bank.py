from __future__ import annotations

"""Frozen all-category candidate-bank construction and serialization.

This module contains only the shared candidate representation and its original
32-class construction contract.  It has no evaluator, ground-truth, prior
scoring, or downstream denoising dependency.
"""

import json
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import pca_sorted_extents_m

# The recheck experiment deliberately creates one fresh bank per scene under
# the current contract.  Retired denoising banks are not silently upgraded.
SCHEMA = "saga-instance-recheck-candidate-bank-v1"
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
SAGA20_CLASSES = (
    "chair",
    "table",
    "plant",
    "tv",
    "painting",
    "sofa",
    "cabinet",
    "bed",
    "socket",
    "book",
    "switch",
    "door",
    "window",
    "lamp",
    "speaker",
    "fan",
    "refrigerator",
    "cup",
    "phone",
    "trash can",
)


@dataclass(frozen=True)
class Top1Assignment:
    """Normalized competition over the complete 32-class table."""

    top_class_index: np.ndarray
    top_score: np.ndarray
    branch_class_index: np.ndarray
    eligible_mask: np.ndarray
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class CandidateBank:
    """One immutable candidate pool shared by every downstream condition."""

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
    gaussian_xyz_sha256: str | None = None
    schema: str = SCHEMA

    @property
    def point_count(self) -> int:
        return len(self.global_pre_knn)


def _as_numpy(value: Any, dtype: Any | None = None) -> np.ndarray:
    """Convert NumPy or CPU/GPU tensor-like input without importing torch."""

    converted = value
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    if hasattr(converted, "numpy"):
        converted = converted.numpy()
    return np.asarray(converted, dtype=dtype)


def gaussian_xyz_sha256(value: Any) -> str:
    """Fingerprint Gaussian coordinates including their exact point order."""

    xyz = np.ascontiguousarray(_as_numpy(value, np.float64).astype("<f8", copy=False))
    if xyz.ndim != 2 or xyz.shape[1:] != (3,) or not np.isfinite(xyz).all():
        raise ValueError("Gaussian XYZ must be a finite N x 3 matrix")
    digest = hashlib.sha256()
    digest.update(np.asarray(xyz.shape, dtype="<i8").tobytes())
    digest.update(xyz.tobytes(order="C"))
    return digest.hexdigest()


def _numeric_array_sha256(value: Any, dtype: Any) -> str:
    array = np.ascontiguousarray(_as_numpy(value, dtype))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _registered_generation_contract() -> dict[str, Any]:
    return {
        "semantic_threshold": SEMANTIC_THRESHOLD,
        "sample_cap": SAMPLE_CAP,
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "min_samples": MIN_SAMPLES,
        "cluster_selection_epsilon": CLUSTER_SELECTION_EPSILON,
        "allow_single_cluster": False,
        "cluster_metric": "precomputed",
        "weights": {
            "instance": INSTANCE_WEIGHT,
            "spatial": SPATIAL_WEIGHT,
            "semantic": SEMANTIC_WEIGHT,
        },
        "assignment_threshold": ASSIGNMENT_THRESHOLD,
        "assignment_temperature": ASSIGNMENT_TEMPERATURE,
    }


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
    """Assign a point only when its global 32-class winner enters SAGA20."""

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
    branch_indices = np.asarray(
        [classes.index(name) for name in branches], dtype=np.int64
    )
    eligible = np.isin(top_class, branch_indices) & (top_score >= float(threshold))
    selected_class = np.where(eligible, top_class, -1).astype(np.int64, copy=False)
    return Top1Assignment(
        top_class_index=_readonly(top_class),
        top_score=_readonly(top_score.astype(np.float64, copy=False)),
        branch_class_index=_readonly(selected_class),
        eligible_mask=_readonly(eligible.astype(bool, copy=False)),
        class_names=classes,
    )


def stable_class_seed(seed: int, class_name: str) -> int:
    """Return the historical deterministic per-class sampling seed."""

    value = int(seed) & ((1 << 63) - 1)
    for index, character in enumerate(str(class_name)):
        value = (
            value * 1_000_003 + ord(character) + index + 1
        ) & ((1 << 63) - 1)
    return value


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
    """Build the frozen class-exclusive bank using the registered mechanics."""

    from scipy.spatial.distance import cdist

    classes, branches = _validate_names(class_names, saga20_names)
    instance = _as_numpy(instance_features, np.float64)
    semantic = _as_numpy(semantic_features, np.float64)
    xyz = _as_numpy(xyz_scene, np.float64)
    labels = _as_numpy(label_features, np.float64)
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
        semantic, labels, classes, branches, SEMANTIC_THRESHOLD
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
        sampled_local = np.random.default_rng(
            stable_class_seed(seed, class_name)
        ).permutation(selected_count)[:sample_count]
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
            feature_centers.append(
                _normalize_rows(sampled_features[mask].mean(axis=0, keepdims=True))[0]
            )
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
        candidates=tuple(candidate_rows),
        diagnostics={
            "generation_contract": _registered_generation_contract(),
            "input_fingerprints": {
                "instance_features": _numeric_array_sha256(instance, np.float64),
                "semantic_features": _numeric_array_sha256(semantic, np.float64),
                "label_features": _numeric_array_sha256(labels, np.float64),
                "global_pre_knn": _numeric_array_sha256(global_labels, np.int64),
            },
            "class_diagnostics": class_diagnostics,
        },
        gaussian_xyz_sha256=gaussian_xyz_sha256(xyz),
    )
    validate_candidate_bank(bank)
    return bank


def validate_candidate_bank(bank: CandidateBank) -> None:
    if bank.schema != SCHEMA:
        raise ValueError(f"unsupported candidate-bank schema: {bank.schema}")
    _validate_names(bank.class_names, bank.saga20_names)
    point_count = bank.point_count
    if bank.gaussian_xyz_sha256 is not None:
        digest = str(bank.gaussian_xyz_sha256)
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("candidate-bank Gaussian XYZ fingerprint is invalid")
        if bank.diagnostics.get("generation_contract") != _registered_generation_contract():
            raise ValueError("candidate-bank generation contract is incomplete")
        fingerprints = bank.diagnostics.get("input_fingerprints")
        required_fingerprints = {
            "instance_features",
            "semantic_features",
            "label_features",
            "global_pre_knn",
        }
        if not isinstance(fingerprints, Mapping) or set(fingerprints) != required_fingerprints:
            raise ValueError("candidate-bank input fingerprints are incomplete")
        for fingerprint in fingerprints.values():
            value = str(fingerprint)
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError("candidate-bank input fingerprint is invalid")
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
            raise ValueError(
                f"candidate {candidate_id} contains sub-threshold semantic points"
            )


def assert_candidate_bank_matches_inputs(
    bank: CandidateBank,
    *,
    xyz_scene: Any,
    global_pre_knn: Any,
    instance_features: Any,
    semantic_features: Any,
    label_features: Any,
    class_names: Sequence[str],
    saga20_names: Sequence[str],
    scene_scale_m_per_unit: float,
    seed: int,
) -> None:
    """Reject a bank whose point axis or registered generation inputs differ."""

    validate_candidate_bank(bank)
    if bank.gaussian_xyz_sha256 is None:
        raise ValueError(
            "candidate bank predates the Gaussian XYZ fingerprint and cannot be "
            "silently treated as input-compatible"
        )
    if gaussian_xyz_sha256(xyz_scene) != bank.gaussian_xyz_sha256:
        raise ValueError("candidate-bank Gaussian XYZ coordinates/order do not match")
    labels = _as_numpy(global_pre_knn, np.int64)
    if labels.shape != np.asarray(bank.global_pre_knn).shape or not np.array_equal(
        labels, bank.global_pre_knn
    ):
        raise ValueError("candidate-bank global_pre_knn does not match")
    if tuple(str(value) for value in class_names) != bank.class_names:
        raise ValueError("candidate-bank class table does not match")
    if tuple(sorted(str(value) for value in saga20_names)) != bank.saga20_names:
        raise ValueError("candidate-bank SAGA20 branch table does not match")
    if int(seed) != bank.seed:
        raise ValueError("candidate-bank seed does not match")
    if not np.isclose(
        float(scene_scale_m_per_unit), bank.scene_scale_m_per_unit, rtol=0.0, atol=0.0
    ):
        raise ValueError("candidate-bank scene scale does not match")
    if bank.diagnostics.get("generation_contract") != _registered_generation_contract():
        raise ValueError("candidate-bank generation parameters do not match")
    expected_fingerprints = {
        "instance_features": _numeric_array_sha256(instance_features, np.float64),
        "semantic_features": _numeric_array_sha256(semantic_features, np.float64),
        "label_features": _numeric_array_sha256(label_features, np.float64),
        "global_pre_knn": _numeric_array_sha256(labels, np.int64),
    }
    if bank.diagnostics.get("input_fingerprints") != expected_fingerprints:
        raise ValueError("candidate-bank feature/input fingerprints do not match")


def _resolve_bank_paths(
    path: str | Path, json_path: str | Path | None = None
) -> tuple[Path, Path | None]:
    location = Path(path)
    if location.suffix.lower() == ".npz":
        return location, Path(json_path) if json_path is not None else None
    return (
        location / "bank_labels.npz",
        Path(json_path) if json_path is not None else location / "candidates.json",
    )


def _bank_metadata(bank: CandidateBank) -> dict[str, Any]:
    return {
        "schema": bank.schema,
        "point_count": bank.point_count,
        "class_names": list(bank.class_names),
        "saga20_names": list(bank.saga20_names),
        "scene_scale_m_per_unit": bank.scene_scale_m_per_unit,
        "seed": bank.seed,
        "gaussian_xyz_sha256": bank.gaussian_xyz_sha256,
        "candidates": [dict(row) for row in bank.candidates],
        "diagnostics": bank.diagnostics,
    }


def save_candidate_bank(
    bank: CandidateBank,
    npz_path: str | Path,
    json_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """Save a bank to an NPZ file or the canonical scene-bank directory."""

    validate_candidate_bank(bank)
    if bank.gaussian_xyz_sha256 is None:
        raise ValueError("new candidate banks must record the Gaussian XYZ fingerprint")
    destination, sidecar = _resolve_bank_paths(npz_path, json_path)
    occupied = [path for path in (destination, sidecar) if path is not None and path.exists()]
    if occupied:
        raise FileExistsError(
            "candidate bank is frozen and will not be overwritten: "
            + ", ".join(str(path) for path in occupied)
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = _bank_metadata(bank)
    encoded = json.dumps(
        metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    temporary = destination.with_name(destination.name + ".part")
    with temporary.open("wb") as handle:
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
    temporary.replace(destination)
    if sidecar is not None:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar_temporary = sidecar.with_name(sidecar.name + ".part")
        sidecar_temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        sidecar_temporary.replace(sidecar)
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
            gaussian_xyz_sha256=metadata.get("gaussian_xyz_sha256"),
            schema=str(metadata["schema"]),
        )
    validate_candidate_bank(bank)
    if int(metadata.get("point_count", -1)) != bank.point_count:
        raise ValueError("candidate-bank metadata point count does not match arrays")
    return bank


def attach_candidate_votes(
    bank: CandidateBank,
    ratios_by_candidate: Mapping[int | str, Sequence[float] | np.ndarray],
    class_names: Sequence[str],
) -> CandidateBank:
    """Attach fixed 2D vote evidence and Q without changing bank labels."""

    validate_candidate_bank(bank)
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
    validate_candidate_bank(updated)
    return updated


__all__ = [
    "ASSIGNMENT_THRESHOLD",
    "MIN_CLUSTER_SIZE",
    "MIN_SAMPLES",
    "SAMPLE_CAP",
    "SAGA20_CLASSES",
    "SCHEMA",
    "SEMANTIC_THRESHOLD",
    "CandidateBank",
    "assert_candidate_bank_matches_inputs",
    "Top1Assignment",
    "attach_candidate_votes",
    "build_candidate_bank",
    "gaussian_xyz_sha256",
    "load_candidate_bank",
    "normalized_top1_32",
    "save_candidate_bank",
    "stable_class_seed",
    "validate_candidate_bank",
]

from __future__ import annotations

"""Read-only three-space reassessment for persisted V10 ObjectBanks.

The original V10 diagnostic mixed two different nearest-neighbour directions:
it projected GT points to their nearest Gaussian, inverted that relation, and
then treated every Gaussian absent from the inverse relation as a false-positive
GT support element.  That is neither the ScanNet projection nor a valid
Gaussian-to-GT precision measure.

This module keeps the three questions separate:

* official projection: GT point -> nearest scene Gaussian -> candidate mask;
* Gaussian precision: candidate Gaussian -> nearest GT point;
* GT recall: target GT point -> nearest scene Gaussian -> candidate membership.

It only reads immutable V10 banks and scene geometry.  It does not change bank
ownership, acceptance thresholds, replay output, or the official evaluator.
"""

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .evaluator import apply_transform, load_ground_truth_npz, load_ply_xyz
from .io import load_json, write_json, write_rows
from .runner import load_scene_runtime_manifest
from .taxonomy import load_taxonomy
from .v10_metrics import (
    V10GroundTruthObject,
    adapt_v10_persisted_bank,
    ground_truth_objects_from_arrays,
)


V10_REASSESSMENT_SCHEMA = "saga-v10-bank-three-space-reassessment-v1"
DEFAULT_RADII_M = (0.02, 0.05, 0.10)
OFFICIAL_RADIUS_M = 0.05


def load_persisted_bank_scene_geometry(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    bank_directory: Path,
    scene_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the exact Gaussian PLY recorded by the bank's lifting producer."""

    bank_metadata = load_json(bank_directory / "object_bank.json")
    if str(bank_metadata.get("scene_id")) != str(scene_id):
        raise ValueError("persisted bank metadata scene mismatch")
    lifting_root = Path(str(bank_metadata["source_lifting_bank"]))
    lifting_metadata = load_json(lifting_root / "lifting_bank.json")
    try:
        point_cloud_record = lifting_metadata["identity"][
            "feature_record_identity"
        ]["inputs"]["point_cloud"]
        gaussian_path = Path(str(point_cloud_record["path"])).resolve()
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "lifting identity does not record its source point-cloud PLY"
        ) from exc
    if not gaussian_path.is_file():
        raise FileNotFoundError(gaussian_path)

    scenes = load_scene_runtime_manifest(runtime_manifest)
    scene = scenes[str(scene_id)]
    gt_path = Path(gt_dir) / f"{scene_id}.npz"
    if scene.get("gt_npz"):
        configured = Path(str(scene["gt_npz"]))
        gt_path = (
            configured
            if configured.is_absolute()
            else Path(str(scene["base_path"])) / configured
        )
    gt_xyz, gt = load_ground_truth_npz(gt_path, str(scene_id))
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
    expected_count = int(bank_metadata["point_count"])
    if len(gaussian_xyz) != expected_count:
        raise ValueError(
            f"bank point_count={expected_count} but source PLY has "
            f"{len(gaussian_xyz)} vertices"
        )
    return gt_xyz, gt.semantic, gt.instance, gaussian_xyz


@dataclass(frozen=True)
class BidirectionalNearest:
    """Unthresholded nearest-neighbour results in both directions."""

    gt_to_gaussian_index: np.ndarray
    gt_to_gaussian_distance_m: np.ndarray
    gaussian_to_gt_index: np.ndarray
    gaussian_to_gt_distance_m: np.ndarray

    def __post_init__(self) -> None:
        gt_index = np.asarray(self.gt_to_gaussian_index, dtype=np.int64).copy()
        gt_distance = np.asarray(
            self.gt_to_gaussian_distance_m, dtype=np.float64
        ).copy()
        gaussian_index = np.asarray(
            self.gaussian_to_gt_index, dtype=np.int64
        ).copy()
        gaussian_distance = np.asarray(
            self.gaussian_to_gt_distance_m, dtype=np.float64
        ).copy()
        if gt_index.ndim != 1 or gt_index.shape != gt_distance.shape:
            raise ValueError("GT nearest-neighbour arrays must be one-dimensional")
        if gaussian_index.ndim != 1 or gaussian_index.shape != gaussian_distance.shape:
            raise ValueError(
                "Gaussian nearest-neighbour arrays must be one-dimensional"
            )
        for value in (gt_index, gt_distance, gaussian_index, gaussian_distance):
            value.setflags(write=False)
        object.__setattr__(self, "gt_to_gaussian_index", gt_index)
        object.__setattr__(self, "gt_to_gaussian_distance_m", gt_distance)
        object.__setattr__(self, "gaussian_to_gt_index", gaussian_index)
        object.__setattr__(self, "gaussian_to_gt_distance_m", gaussian_distance)

    @property
    def gt_count(self) -> int:
        return int(len(self.gt_to_gaussian_index))

    @property
    def gaussian_count(self) -> int:
        return int(len(self.gaussian_to_gt_index))


def build_bidirectional_nearest(
    gt_xyz: np.ndarray,
    gaussian_xyz: np.ndarray,
) -> BidirectionalNearest:
    """Build both nearest-neighbour directions once without a radius cutoff."""

    gt = np.asarray(gt_xyz, dtype=np.float64)
    gaussians = np.asarray(gaussian_xyz, dtype=np.float64)
    if gt.ndim != 2 or gt.shape[1:] != (3,):
        raise ValueError("gt_xyz must have shape (N, 3)")
    if gaussians.ndim != 2 or gaussians.shape[1:] != (3,):
        raise ValueError("gaussian_xyz must have shape (M, 3)")
    if not len(gt) or not len(gaussians):
        raise ValueError("GT and Gaussian point sets must both be non-empty")
    gt_to_gaussian_distance, gt_to_gaussian_index = cKDTree(gaussians).query(
        gt, k=1, workers=-1
    )
    gaussian_to_gt_distance, gaussian_to_gt_index = cKDTree(gt).query(
        gaussians, k=1, workers=-1
    )
    return BidirectionalNearest(
        gt_to_gaussian_index,
        gt_to_gaussian_distance,
        gaussian_to_gt_index,
        gaussian_to_gt_distance,
    )


def _normalize_radii(radii_m: Sequence[float]) -> tuple[float, ...]:
    radii = tuple(sorted({float(value) for value in radii_m}))
    if not radii or any(not np.isfinite(value) or value <= 0 for value in radii):
        raise ValueError("radii_m must contain positive finite values")
    return radii


def _candidate_lookup(
    gaussian_ids: Sequence[int] | np.ndarray,
    gaussian_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(gaussian_ids, dtype=np.int64)
    if ids.ndim != 1 or not len(ids):
        raise ValueError("candidate Gaussian IDs must be a non-empty vector")
    ids = np.unique(ids)
    if np.any(ids < 0) or np.any(ids >= int(gaussian_count)):
        raise ValueError("candidate Gaussian ID outside scene point range")
    lookup = np.zeros(int(gaussian_count), dtype=bool)
    lookup[ids] = True
    return ids, lookup


def official_projected_mask(
    candidate_lookup: np.ndarray,
    nearest: BidirectionalNearest,
    radius_m: float,
) -> np.ndarray:
    """Project one candidate into GT-point space exactly as the evaluator does."""

    lookup = np.asarray(candidate_lookup, dtype=bool)
    if lookup.shape != (nearest.gaussian_count,):
        raise ValueError("candidate lookup length differs from Gaussian count")
    valid = nearest.gt_to_gaussian_distance_m <= float(radius_m)
    mask = np.zeros(nearest.gt_count, dtype=bool)
    mask[valid] = lookup[nearest.gt_to_gaussian_index[valid]]
    return mask


def _best_gt_match(
    predicted_mask: np.ndarray,
    ground_truth: Sequence[V10GroundTruthObject],
    *,
    class_name: str | None = None,
) -> dict[str, Any]:
    mask = np.asarray(predicted_mask, dtype=bool)
    predicted_count = int(np.count_nonzero(mask))
    candidates = [
        row
        for row in ground_truth
        if row.official_valid
        and (class_name is None or row.class_name == str(class_name))
    ]
    scored: list[tuple[float, float, int, V10GroundTruthObject]] = []
    for gt in candidates:
        intersection = int(np.count_nonzero(mask[gt.point_ids]))
        union = predicted_count + len(gt.point_ids) - intersection
        iou = intersection / union if union else 0.0
        precision = intersection / predicted_count if predicted_count else 0.0
        scored.append((iou, precision, intersection, gt))
    if not scored:
        return {
            "class_name": None,
            "instance_id": None,
            "intersection": 0,
            "gt_point_count": 0,
            "predicted_gt_point_count": predicted_count,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }
    iou, precision, intersection, gt = min(
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
        "instance_id": int(gt.instance_id),
        "intersection": int(intersection),
        "gt_point_count": int(len(gt.point_ids)),
        "predicted_gt_point_count": predicted_count,
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(intersection / len(gt.point_ids)) if len(gt.point_ids) else 0.0,
    }


def _no_class_match(predicted_mask: np.ndarray) -> dict[str, Any]:
    return {
        "class_name": None,
        "instance_id": None,
        "intersection": 0,
        "gt_point_count": 0,
        "predicted_gt_point_count": int(np.count_nonzero(predicted_mask)),
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }


def _target_identity(
    *,
    official_same_class: Mapping[str, Any],
    candidate_ids: np.ndarray,
    candidate_class_name: str | None,
    nearest: BidirectionalNearest,
    gt_semantic: np.ndarray,
    gt_instance: np.ndarray,
    canonical_classes: Sequence[str],
    official_ground_truth: Mapping[tuple[str, int], V10GroundTruthObject],
) -> tuple[V10GroundTruthObject | None, str | None]:
    if (
        official_same_class.get("class_name") is not None
        and int(official_same_class.get("intersection", 0)) > 0
    ):
        identity = (
            str(official_same_class["class_name"]),
            int(official_same_class["instance_id"]),
        )
        return official_ground_truth.get(identity), "official_same_class_5cm"

    class_to_id = {str(name): index for index, name in enumerate(canonical_classes)}
    class_id = class_to_id.get(str(candidate_class_name))
    valid = nearest.gaussian_to_gt_distance_m[candidate_ids] <= OFFICIAL_RADIUS_M
    gt_ids = nearest.gaussian_to_gt_index[candidate_ids[valid]]
    if class_id is not None and len(gt_ids):
        supported = (
            (gt_semantic[gt_ids] == int(class_id))
            & (gt_instance[gt_ids] >= 0)
        )
        identities = Counter(
            (str(candidate_class_name), int(value))
            for value in gt_instance[gt_ids[supported]]
        )
        if identities:
            identity = min(
                identities,
                key=lambda item: (-identities[item], item[0], item[1]),
            )
            target = official_ground_truth.get(identity)
            if target is not None:
                return target, "gaussian_majority_same_class_5cm"
    return None, None


def evaluate_candidate_three_spaces(
    *,
    scene_id: str,
    candidate_id: int,
    candidate_gaussian_ids: Sequence[int] | np.ndarray,
    candidate_class_name: str | None,
    candidate_score: float | None,
    nearest: BidirectionalNearest,
    gt_semantic: np.ndarray,
    gt_instance: np.ndarray,
    ground_truth: Sequence[V10GroundTruthObject],
    precision_ground_truth: Sequence[V10GroundTruthObject] | None = None,
    canonical_classes: Sequence[str],
    radii_m: Sequence[float] = DEFAULT_RADII_M,
) -> list[dict[str, Any]]:
    """Evaluate one proposal independently in the three coordinate spaces."""

    semantic = np.asarray(gt_semantic, dtype=np.int64)
    instance = np.asarray(gt_instance, dtype=np.int64)
    if semantic.shape != (nearest.gt_count,) or instance.shape != (nearest.gt_count,):
        raise ValueError("GT labels differ from nearest-neighbour GT point count")
    candidate_ids, lookup = _candidate_lookup(
        candidate_gaussian_ids, nearest.gaussian_count
    )
    radii = _normalize_radii(radii_m)
    official_mask = official_projected_mask(
        lookup, nearest, OFFICIAL_RADIUS_M
    )
    official_geometric = _best_gt_match(official_mask, ground_truth)
    official_same_class = (
        _best_gt_match(
            official_mask,
            ground_truth,
            class_name=candidate_class_name,
        )
        if candidate_class_name is not None
        else _no_class_match(official_mask)
    )
    precision_objects = (
        tuple(precision_ground_truth)
        if precision_ground_truth is not None
        else tuple(ground_truth)
    )
    precision_ground_truth_by_identity = {
        (row.class_name, int(row.instance_id)): row
        for row in precision_objects
    }
    target, target_source = _target_identity(
        official_same_class=official_same_class,
        candidate_ids=candidate_ids,
        candidate_class_name=candidate_class_name,
        nearest=nearest,
        gt_semantic=semantic,
        gt_instance=instance,
        canonical_classes=canonical_classes,
        official_ground_truth=precision_ground_truth_by_identity,
    )
    class_to_id = {str(name): index for index, name in enumerate(canonical_classes)}
    candidate_class_id = class_to_id.get(str(candidate_class_name))

    rows: list[dict[str, Any]] = []
    for radius in radii:
        projected_mask = official_projected_mask(lookup, nearest, radius)
        geometric = _best_gt_match(projected_mask, ground_truth)
        same_class = (
            _best_gt_match(
                projected_mask, ground_truth, class_name=candidate_class_name
            )
            if candidate_class_name is not None
            else _no_class_match(projected_mask)
        )

        gaussian_gt_ids = nearest.gaussian_to_gt_index[candidate_ids]
        gaussian_valid = (
            nearest.gaussian_to_gt_distance_m[candidate_ids] <= radius
        )
        evaluable = gaussian_valid.copy()
        evaluable[gaussian_valid] = (
            (semantic[gaussian_gt_ids[gaussian_valid]] >= 0)
            & (instance[gaussian_gt_ids[gaussian_valid]] >= 0)
        )
        same_class_support = np.zeros(len(candidate_ids), dtype=bool)
        if candidate_class_id is not None:
            same_class_support[evaluable] = (
                semantic[gaussian_gt_ids[evaluable]] == int(candidate_class_id)
            )
        correct = np.zeros(len(candidate_ids), dtype=bool)
        if target is not None and candidate_class_id is not None:
            correct[evaluable] = (
                (semantic[gaussian_gt_ids[evaluable]] == int(candidate_class_id))
                & (instance[gaussian_gt_ids[evaluable]] == int(target.instance_id))
            )
        same_class_wrong = same_class_support & ~correct
        wrong_class = evaluable & ~same_class_support
        unsupported = ~evaluable

        target_point_count = int(len(target.point_ids)) if target is not None else 0
        target_recalled = 0
        target_asset_covered = 0
        if target is not None:
            target_distances = nearest.gt_to_gaussian_distance_m[target.point_ids]
            target_valid = target_distances <= radius
            target_asset_covered = int(np.count_nonzero(target_valid))
            target_nearest = nearest.gt_to_gaussian_index[target.point_ids]
            target_recalled = int(
                np.count_nonzero(target_valid & lookup[target_nearest])
            )
        scene_asset_coverage = float(
            np.mean(nearest.gt_to_gaussian_distance_m <= radius)
        )
        rows.append(
            {
                "schema": V10_REASSESSMENT_SCHEMA,
                "scene_id": str(scene_id),
                "candidate_id": int(candidate_id),
                "candidate_class_name": candidate_class_name,
                "candidate_score": candidate_score,
                "radius_m": float(radius),
                "official_primary_radius": bool(
                    np.isclose(radius, OFFICIAL_RADIUS_M)
                ),
                "candidate_gaussian_count": int(len(candidate_ids)),
                "official_geometric_gt_class_name": geometric["class_name"],
                "official_geometric_gt_instance_id": geometric["instance_id"],
                "official_geometric_iou": float(geometric["iou"]),
                "official_geometric_precision": float(geometric["precision"]),
                "official_geometric_recall": float(geometric["recall"]),
                "official_same_class_gt_instance_id": same_class["instance_id"],
                "official_same_class_iou": float(same_class["iou"]),
                "official_same_class_precision": float(same_class["precision"]),
                "official_same_class_recall": float(same_class["recall"]),
                "official_projected_gt_point_count": int(
                    np.count_nonzero(projected_mask)
                ),
                "precision_target_class_name": (
                    None if target is None else target.class_name
                ),
                "precision_target_instance_id": (
                    None if target is None else int(target.instance_id)
                ),
                "precision_target_source": target_source,
                "gaussian_correct_count": int(np.count_nonzero(correct)),
                "gaussian_same_class_wrong_instance_count": int(
                    np.count_nonzero(same_class_wrong)
                ),
                "gaussian_wrong_class_count": int(np.count_nonzero(wrong_class)),
                "gaussian_unsupported_count": int(np.count_nonzero(unsupported)),
                "gaussian_to_gt_precision": float(
                    np.count_nonzero(correct) / len(candidate_ids)
                ),
                "gaussian_semantic_precision": float(
                    np.count_nonzero(same_class_support) / len(candidate_ids)
                ),
                "gaussian_unsupported_fraction": float(
                    np.count_nonzero(unsupported) / len(candidate_ids)
                ),
                "target_gt_point_count": target_point_count,
                "target_gt_recalled_point_count": target_recalled,
                "target_gt_asset_covered_point_count": target_asset_covered,
                "gt_to_candidate_recall": float(
                    target_recalled / target_point_count
                )
                if target_point_count
                else 0.0,
                "target_gt_asset_coverage": float(
                    target_asset_covered / target_point_count
                )
                if target_point_count
                else 0.0,
                "scene_gt_asset_coverage": scene_asset_coverage,
                # The fixed target is chosen once at 5 cm.  These fields make
                # sensitivity rows auditable without silently rematching it.
                "official_5cm_geometric_iou": float(official_geometric["iou"]),
                "official_5cm_same_class_iou": float(official_same_class["iou"]),
            }
        )
    return rows


def summarise_reassessment_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate proposal diagnostics without pretending proposals are unique AP."""

    result: dict[str, Any] = {}
    radii = sorted({float(row["radius_m"]) for row in rows})
    for radius in radii:
        selected = [row for row in rows if float(row["radius_m"]) == radius]
        gaussian_count = sum(int(row["candidate_gaussian_count"]) for row in selected)
        correct = sum(int(row["gaussian_correct_count"]) for row in selected)
        target_points = sum(int(row["target_gt_point_count"]) for row in selected)
        recalled = sum(int(row["target_gt_recalled_point_count"]) for row in selected)
        result[f"{radius:.2f}"] = {
            "radius_m": radius,
            "candidate_count": len(selected),
            "geometric_match_025_count": sum(
                float(row["official_geometric_iou"]) >= 0.25 for row in selected
            ),
            "geometric_match_050_count": sum(
                float(row["official_geometric_iou"]) >= 0.50 for row in selected
            ),
            "same_class_match_025_count": sum(
                float(row["official_same_class_iou"]) >= 0.25 for row in selected
            ),
            "same_class_match_050_count": sum(
                float(row["official_same_class_iou"]) >= 0.50 for row in selected
            ),
            "proposal_macro_gaussian_precision": float(
                np.mean([float(row["gaussian_to_gt_precision"]) for row in selected])
            )
            if selected
            else 0.0,
            "proposal_weighted_gaussian_precision": float(correct / gaussian_count)
            if gaussian_count
            else 0.0,
            "proposal_macro_gt_recall": float(
                np.mean(
                    [
                        float(row["gt_to_candidate_recall"])
                        for row in selected
                        if int(row["target_gt_point_count"]) > 0
                    ]
                )
            )
            if any(int(row["target_gt_point_count"]) > 0 for row in selected)
            else 0.0,
            "proposal_weighted_gt_recall": float(recalled / target_points)
            if target_points
            else 0.0,
            "overlapping_proposals_are_counted_independently": True,
        }
    return {
        "schema": V10_REASSESSMENT_SCHEMA,
        "official_primary_radius_m": OFFICIAL_RADIUS_M,
        "scope": "proposal_diagnostics_not_official_ap",
        "radii": result,
    }


def reassess_v10_banks(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    bank_root: Path,
    scene_ids: Sequence[str],
    condition: str,
    classifier: str,
    class_names: Sequence[str],
    rows_output: Path,
    analysis_output: Path,
    radii_m: Sequence[float] = DEFAULT_RADII_M,
    min_region_size: int = 100,
    legacy_analysis: Path | None = None,
) -> dict[str, Any]:
    """Read existing banks and write an independent three-space reassessment."""

    rows: list[dict[str, Any]] = []
    for scene_id in map(str, scene_ids):
        bank_directory = bank_root / str(condition) / scene_id
        persisted = adapt_v10_persisted_bank(
            bank_directory,
            classifier=str(classifier),
        )
        if persisted.scene_id != scene_id:
            raise ValueError(f"bank scene mismatch: {persisted.scene_id} != {scene_id}")
        gt_xyz, semantic, instance, gaussian_xyz = load_persisted_bank_scene_geometry(
            runtime_manifest=runtime_manifest,
            gt_dir=gt_dir,
            bank_directory=bank_directory,
            scene_id=scene_id,
        )
        nearest = build_bidirectional_nearest(gt_xyz, gaussian_xyz)
        ground_truth = ground_truth_objects_from_arrays(
            scene_id,
            semantic,
            instance,
            class_names,
            min_region_size=min_region_size,
        )
        precision_ground_truth = ground_truth_objects_from_arrays(
            scene_id,
            semantic,
            instance,
            class_names,
            min_region_size=1,
        )
        for candidate in persisted.stage_candidates:
            if candidate.stage != "final_candidate":
                continue
            rows.extend(
                evaluate_candidate_three_spaces(
                    scene_id=scene_id,
                    candidate_id=candidate.candidate_id,
                    candidate_gaussian_ids=candidate.gaussian_ids,
                    candidate_class_name=candidate.class_name,
                    candidate_score=candidate.score,
                    nearest=nearest,
                    gt_semantic=semantic,
                    gt_instance=instance,
                    ground_truth=ground_truth,
                    precision_ground_truth=precision_ground_truth,
                    canonical_classes=class_names,
                    radii_m=radii_m,
                )
            )
    analysis = summarise_reassessment_rows(rows)
    analysis.update(
        {
            "scene_ids": list(map(str, scene_ids)),
            "condition": str(condition),
            "classifier": str(classifier),
        }
    )
    if legacy_analysis is not None:
        legacy_payload = load_json(legacy_analysis)
        legacy_metrics = legacy_payload.get("gate_metrics", {})
        corrected = analysis["radii"].get(f"{OFFICIAL_RADIUS_M:.2f}", {})
        candidate_count = int(corrected.get("candidate_count", 0))
        corrected_precision_025 = (
            float(corrected.get("geometric_match_025_count", 0))
            / candidate_count
            if candidate_count
            else 0.0
        )
        legacy_precision_025 = legacy_metrics.get(
            "geometric_candidate_precision_025"
        )
        legacy_match_050 = legacy_metrics.get("geometric_match_050_count")
        analysis["legacy_unique_fp_sentinel_closeout"] = {
            "source": str(Path(legacy_analysis).resolve()),
            "legacy_geometric_candidate_precision_025": legacy_precision_025,
            "corrected_projected_candidate_precision_025": corrected_precision_025,
            "precision_025_difference_corrected_minus_legacy": (
                corrected_precision_025 - float(legacy_precision_025)
                if legacy_precision_025 is not None
                else None
            ),
            "legacy_geometric_match_050_count": legacy_match_050,
            "corrected_geometric_match_050_count": corrected.get(
                "geometric_match_050_count"
            ),
            "match_050_count_difference_corrected_minus_legacy": (
                int(corrected.get("geometric_match_050_count", 0))
                - int(legacy_match_050)
                if legacy_match_050 is not None
                else None
            ),
            "warning": (
                "the corrected and legacy precision values use different "
                "projection definitions; the difference is a historical "
                "closeout, not a treatment effect"
            ),
        }
    write_rows(rows_output, rows)
    write_json(analysis_output, analysis)
    return analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only three-space reassessment of persisted V10 banks"
    )
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--scene", action="append", required=True, dest="scene_ids")
    parser.add_argument("--condition", default="VC1")
    parser.add_argument("--classifier", default="mv-label")
    parser.add_argument("--taxonomy", type=Path)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--analysis-output", type=Path, required=True)
    parser.add_argument(
        "--radius-m",
        action="append",
        type=float,
        dest="radii_m",
        help="Repeat for sensitivity radii; defaults to 0.02/0.05/0.10",
    )
    parser.add_argument("--min-region-size", type=int, default=100)
    parser.add_argument("--legacy-analysis", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    taxonomy = load_taxonomy(args.taxonomy)
    reassess_v10_banks(
        runtime_manifest=args.runtime_manifest,
        gt_dir=args.gt_dir,
        bank_root=args.bank_root,
        scene_ids=args.scene_ids,
        condition=args.condition,
        classifier=args.classifier,
        class_names=taxonomy.canonical_classes,
        rows_output=args.rows_output,
        analysis_output=args.analysis_output,
        radii_m=args.radii_m or DEFAULT_RADII_M,
        min_region_size=args.min_region_size,
        legacy_analysis=args.legacy_analysis,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by deployment CLI
    raise SystemExit(main())

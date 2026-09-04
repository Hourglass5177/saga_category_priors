from __future__ import annotations

"""Single-scene two-round refinement runtime.

Ground truth is intentionally absent from this module.  Evaluation lives in a
separate command so no oracle information can leak into review or fusion.
"""

import json
import math
import hashlib
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..candidate_bank import gaussian_xyz_sha256, load_candidate_bank
from ..geometry import pca_sorted_extents_m
from ..prediction_finalization import (
    finalize_prediction,
    prediction_output_payload,
    write_prediction_output_atomic,
)
from .contracts import (
    CandidateSeed,
    GaussianEvidence,
    LineageRecord,
    MaskHypothesis,
    ObjectState,
    PROFILES,
    RefinementConfig,
    ViewObservation,
)
from .evidence import (
    HypothesisGaussianEvidence,
    aggregate_gaussian_evidence,
    hypothesis_gaussian_evidence,
    normalized_soft_membership,
)
from .local_refine import fuse_objects_with_b0, refine_candidate_local, size_prior_from_payload
from .objects import merge_objects_once, split_disconnected_objects
from .rendering import render_alpha_mass, render_camera_maps
from .reviewer import GroundedSamFullImageReviewer
from .runtime_io import camera_center, camera_rgb, file_sha256, focal_geometric_mean, json_atomic, load_cameras, npz_atomic, save_scene_cache
from .seeds import load_reservoir
from .views import (
    make_crop_spec,
    observations_are_independent,
    select_consistent_hypotheses,
    select_diverse_views,
)


def _projection_mask(point_ids: np.ndarray, contributor: np.ndarray) -> np.ndarray:
    valid = contributor >= 0
    result = np.zeros(contributor.shape, dtype=bool)
    result[valid] = np.isin(contributor[valid], point_ids)
    return result


def _empty_evidence(candidate_id: int) -> GaussianEvidence:
    return GaussianEvidence(candidate_id, np.empty(0, np.int64), np.empty(0), np.empty(0), np.empty(0), 0, 0, ())


def _round_seed(seed: CandidateSeed, state: ObjectState) -> CandidateSeed:
    anchor = np.union1d(seed.seed_anchor, state.hard_positive_ids)
    anchor = np.intersect1d(anchor, state.point_ids, assume_unique=True)
    return replace(seed, seed_support=state.point_ids, seed_anchor=anchor)


def _review_class(hypotheses: Sequence[MaskHypothesis]) -> tuple[str | None, bool]:
    by_camera: dict[int, str] = {}
    for row in hypotheses:
        by_camera[row.camera_index] = row.detected_class
    if not by_camera:
        return None, False
    values, counts = np.unique(list(by_camera.values()), return_counts=True)
    maximum = int(counts.max())
    winners = sorted(str(value) for value in values[counts == maximum])
    return winners[0], maximum >= 2 and len(winners) == 1


def _candidate_raw_vote_valid(
    raw_row: np.ndarray,
    *,
    classes: Sequence[str],
    selected_classes: Sequence[str],
    threshold: float,
) -> bool:
    votes = np.asarray(raw_row, dtype=np.float64)
    if votes.shape != (len(classes) + 1,) or votes.sum() <= 0:
        return False
    ratios = votes / votes.sum()
    maximum = float(ratios[:-1].max())
    winners = np.flatnonzero(np.isclose(ratios[:-1], maximum, rtol=0.0, atol=1e-12))
    return bool(
        len(winners) == 1
        and maximum >= threshold
        and maximum > float(ratios[-1])
        and classes[int(winners[0])] in set(selected_classes)
    )


def _camera_cache_path(root: Path, camera_index: int) -> Path:
    return root / "camera_maps" / f"{camera_index:05d}.npz"


def _review_cache_path(root: Path, round_index: int, camera_index: int, candidate_id: int) -> Path:
    return root / "review_cache" / f"round{round_index}" / f"camera{camera_index:05d}" / f"candidate{candidate_id:05d}.npz"


def _review_cache_identity(
    *, seed: CandidateSeed, camera: Any, gaussian_sha: str, condition: str,
    round_index: int, classes: Sequence[str], args: Any, config: RefinementConfig,
) -> str:
    digest = hashlib.sha256()
    digest.update(_camera_cache_identity(camera, gaussian_sha).encode("ascii"))
    digest.update(np.ascontiguousarray(seed.seed_support, dtype="<i8").tobytes())
    payload = {
        "condition": condition, "round": round_index,
        "classes": list(classes), "config": asdict(config),
        "dino_config": str(Path(args.groundingdino_config_path).resolve()),
        "dino_checkpoint": str(Path(args.groundingdino_checkpoint_path).resolve()),
        "sam_checkpoint": str(Path(args.sam_checkpoint_path).resolve()),
        "asset_stats": [
            [Path(value).stat().st_size, Path(value).stat().st_mtime_ns]
            for value in (
                args.groundingdino_config_path,
                args.groundingdino_checkpoint_path,
                args.sam_checkpoint_path,
            )
        ],
    }
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _save_review_cache(path: Path, rows: Sequence[MaskHypothesis], identity: str) -> None:
    metadata = []
    arrays = {"identity": np.asarray(identity)}
    for index, row in enumerate(rows):
        key = f"mask_{index}"
        arrays[key] = row.packed_mask
        metadata.append({key: value for key, value in row.__dict__.items() if key != "packed_mask"} | {"packed_key": key})
    arrays["metadata"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    npz_atomic(path, **arrays)


def _load_review_cache(path: Path, identity: str) -> tuple[MaskHypothesis, ...] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["identity"].item()) != identity:
                return None
            metadata = json.loads(str(archive["metadata"].item()))
            rows = []
            for row in metadata:
                values = dict(row)
                packed = np.asarray(archive[values.pop("packed_key")], dtype=np.uint8)
                for key in ("box_xyxy", "mask_shape"):
                    values[key] = tuple(values[key])
                rows.append(MaskHypothesis(packed_mask=packed, **values))
        return tuple(rows)
    except Exception:
        return None


def _camera_cache_identity(camera: Any, gaussian_sha: str) -> str:
    digest = hashlib.sha256()
    digest.update(str(gaussian_sha).encode("ascii"))
    digest.update(str(camera.image_name).encode("utf-8"))
    digest.update(np.asarray([camera.image_height, camera.image_width, camera.FoVx, camera.FoVy], dtype="<f8").tobytes())
    for value in (camera.world_view_transform, camera.full_proj_transform, camera.camera_center):
        array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
    return digest.hexdigest()


def _get_camera_maps(root: Path, camera_index: int, camera: Any, model: Any, args: Any, background: Any, *, gaussian_sha: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = _camera_cache_path(root, camera_index)
    identity = _camera_cache_identity(camera, gaussian_sha)
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as archive:
                stored_identity = str(archive["identity"].item())
                ids = np.asarray(archive["ids"], dtype=np.int32)
                weights = np.asarray(archive["weights"], dtype=np.float32)
                opacity = np.asarray(archive["opacity"], dtype=np.float32)
            if stored_identity == identity and ids.shape == weights.shape == opacity.shape == (camera.image_height, camera.image_width):
                return ids, weights, opacity
        except Exception:
            pass
    ids, weights, opacity = render_camera_maps(camera, model, args, background)
    ids = ids.astype(np.int32, copy=False)
    weights = weights.astype(np.float32, copy=False)
    opacity = opacity.astype(np.float32, copy=False)
    npz_atomic(path, identity=np.asarray(identity), ids=ids, weights=weights, opacity=opacity)
    return ids, weights, opacity


def _candidate_observations(
    *,
    seeds: Sequence[CandidateSeed], cameras: Sequence[Any], camera_maps: Mapping[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    xyz: np.ndarray, scene_scale_m_per_unit: float, config: RefinementConfig,
) -> dict[int, tuple[ViewObservation, ...]]:
    rows_by_candidate: dict[int, list[ViewObservation]] = {seed.candidate_id: [] for seed in seeds}
    point_count = len(xyz)
    primary = np.full(point_count, -1, dtype=np.int64)
    extra: dict[int, set[int]] = {}
    seed_by_id = {seed.candidate_id: seed for seed in seeds}
    for seed in seeds:
        for point_id in seed.seed_support:
            pid = int(point_id)
            if primary[pid] < 0:
                primary[pid] = seed.candidate_id
            elif primary[pid] != seed.candidate_id:
                extra.setdefault(pid, {int(primary[pid])}).add(seed.candidate_id)
    centers = {
        seed.candidate_id: np.median(xyz[seed.seed_support], axis=0)
        for seed in seeds if len(seed.seed_support)
    }
    for camera_index, camera in enumerate(cameras):
        ids, weights, opacity = camera_maps[camera_index]
        valid = (
            (ids >= 0) & (ids < point_count) & np.isfinite(weights)
            & (weights > 0) & np.isfinite(opacity) & (opacity >= config.alpha_opacity_min)
        )
        flat_positions = np.flatnonzero(valid)
        if not len(flat_positions):
            continue
        gaussian_ids = ids.reshape(-1)[flat_positions].astype(np.int64, copy=False)
        candidate_ids = primary[gaussian_ids]
        keep = candidate_ids >= 0
        positions_by_candidate: dict[int, list[np.ndarray]] = {}
        if np.any(keep):
            kept_candidates = candidate_ids[keep]
            kept_positions = flat_positions[keep]
            order = np.argsort(kept_candidates, kind="stable")
            ordered_candidates = kept_candidates[order]
            boundaries = np.flatnonzero(np.r_[True, ordered_candidates[1:] != ordered_candidates[:-1], True])
            for start, stop in zip(boundaries[:-1], boundaries[1:]):
                candidate_id = int(ordered_candidates[start])
                positions_by_candidate.setdefault(candidate_id, []).append(kept_positions[order[start:stop]])
        if extra:
            conflict_points = np.asarray(sorted(extra), dtype=np.int64)
            conflict_mask = np.isin(gaussian_ids, conflict_points)
            for position, point_id in zip(flat_positions[conflict_mask], gaussian_ids[conflict_mask]):
                for candidate_id in extra[int(point_id)]:
                    if candidate_id != int(primary[int(point_id)]):
                        positions_by_candidate.setdefault(candidate_id, []).append(np.asarray([position], dtype=np.int64))
        center_camera = camera_center(camera) * float(scene_scale_m_per_unit)
        width = int(camera.image_width)
        flat_weights = weights.reshape(-1)
        flat_opacity = opacity.reshape(-1)
        for candidate_id, chunks in positions_by_candidate.items():
            positions = np.unique(np.concatenate(chunks))
            if len(positions) < config.min_projected_pixels:
                continue
            ys, xs = np.divmod(positions, width)
            ratios = np.divide(
                flat_weights[positions], flat_opacity[positions],
                out=np.zeros(len(positions), dtype=np.float64),
                where=flat_opacity[positions] > 0,
            )
            center = centers[candidate_id]
            ray = center - center_camera
            ray_norm = float(np.linalg.norm(ray))
            ray = ray / ray_norm if ray_norm > 0 else np.zeros(3)
            seed = seed_by_id[candidate_id]
            depth = float(np.median(np.linalg.norm(xyz[seed.seed_support] - center_camera[None, :], axis=1)))
            edge_touches = int(xs.min() == 0) + int(ys.min() == 0) + int(xs.max() == width - 1) + int(ys.max() == int(camera.image_height) - 1)
            truncation_factor = 1.0 - 0.25 * (edge_touches / 4.0)
            rows_by_candidate[candidate_id].append(ViewObservation(
                candidate_id, camera_index, camera.image_name, int(len(positions)),
                (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                (float(xs.mean()), float(ys.mean())),
                float(math.log1p(len(positions)) * np.clip(np.mean(ratios), 0., 1.) * truncation_factor),
                depth, tuple(float(value) for value in ray),
                tuple(float(value) for value in center_camera), True,
            ))
    return {
        candidate_id: select_diverse_views(rows, config)
        for candidate_id, rows in rows_by_candidate.items()
    }


def _independence_map(observations: Sequence[ViewObservation], config: RefinementConfig) -> dict[tuple[int, int], bool]:
    by_camera = {row.camera_index: row for row in observations}
    result = {}
    cameras = sorted(by_camera)
    for left_index, left in enumerate(cameras):
        for right in cameras[left_index + 1 :]:
            result[(left, right)] = observations_are_independent(by_camera[left], by_camera[right], config)
    return result


def _review_round(
    *,
    round_index: int,
    seeds: Sequence[CandidateSeed],
    observations: Mapping[int, Sequence[ViewObservation]],
    cameras: Sequence[Any],
    camera_maps: Mapping[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    b0_labels: np.ndarray,
    model: Any,
    args: Any,
    background: Any,
    priors: Mapping[str, Any],
    xyz: np.ndarray,
    condition: str,
    reviewer: GroundedSamFullImageReviewer,
    classes: Sequence[str],
    config: RefinementConfig,
    cache_root: Path,
    gaussian_sha: str,
) -> tuple[tuple[MaskHypothesis, ...], dict[int, GaussianEvidence], dict[int, tuple[MaskHypothesis, ...]]]:
    tasks: dict[int, list[tuple[CandidateSeed, ViewObservation]]] = {}
    for seed in seeds:
        for row in observations.get(seed.candidate_id, ())[: config.views_per_round]:
            if row.independent:
                tasks.setdefault(row.camera_index, []).append((seed, row))
    hypotheses: list[MaskHypothesis] = []
    for camera_index in sorted(tasks):
        camera = cameras[camera_index]
        image = camera_rgb(camera)
        contributor, _, _ = camera_maps[camera_index]
        cached_rows: dict[int, tuple[MaskHypothesis, ...]] = {}
        pending: list[tuple[CandidateSeed, ViewObservation, str, Path]] = []
        for seed, observation in sorted(tasks[camera_index], key=lambda pair: pair[0].candidate_id):
            identity = _review_cache_identity(
                seed=seed, camera=camera, gaussian_sha=gaussian_sha,
                condition=condition, round_index=round_index, classes=classes,
                args=args, config=config,
            )
            cache_path = _review_cache_path(cache_root, round_index, camera_index, seed.candidate_id)
            loaded = _load_review_cache(cache_path, identity)
            if loaded is not None:
                cached_rows[seed.candidate_id] = loaded
            else:
                pending.append((seed, observation, identity, cache_path))
        if pending:
            reviewer.begin_camera(camera.image_name, image)
        hypotheses.extend(row for rows in cached_rows.values() for row in rows)
        for seed, observation, identity, cache_path in pending:
            seed_mask = _projection_mask(seed.seed_support, contributor)
            competing = (contributor >= 0) & (b0_labels[np.clip(contributor, 0, len(b0_labels) - 1)] >= 0) & ~seed_mask
            class_name = seed.branch_class if condition == "class" else None
            prior = size_prior_from_payload(priors, class_name)
            candidate_rows: list[MaskHypothesis] = []
            for crop_kind in ("tight", "prior"):
                crop = make_crop_spec(
                    observation, image.shape[:2], kind=crop_kind,
                    focal_geometric_mean=focal_geometric_mean(camera),
                    prior_diagonal_m=prior.diagonal_q50_m, config=config,
                )
                candidate_rows.extend(
                    reviewer.review_crop(
                        candidate_id=seed.candidate_id, round_index=round_index,
                        camera_index=camera_index, crop_spec=crop,
                        seed_mask_full=seed_mask, negative_mask_full=competing,
                        classes=classes, config=config,
                    )
                )
            candidate_rows.sort(key=lambda row: (-row.seed_coverage, -row.sam_score, -row.detection_score, row.mask_area, row.hypothesis_id))
            kept = tuple(candidate_rows[: config.max_masks_per_view])
            _save_review_cache(cache_path, kept, identity)
            hypotheses.extend(kept)

    per_hypothesis: dict[str, HypothesisGaussianEvidence] = {}
    soft_membership: dict[str, Mapping[int, float]] = {}
    size_penalties: dict[str, float] = {}
    by_camera: dict[int, list[MaskHypothesis]] = {}
    for row in hypotheses:
        by_camera.setdefault(row.camera_index, []).append(row)
    for camera_index, rows in sorted(by_camera.items()):
        rows.sort(key=lambda row: row.hypothesis_id)
        alpha_mass = render_alpha_mass(
            cameras[camera_index], model, args, background,
            [row.unpack_mask() for row in rows], config=config,
        )
        ids, weights, opacity = camera_maps[camera_index]
        for mask_index, row in enumerate(rows):
            per_hypothesis[row.hypothesis_id] = hypothesis_gaussian_evidence(
                row, contributor_ids=ids, max_weights=weights, opacity=opacity,
                alpha_mass=alpha_mass, alpha_mask_index=mask_index,
                point_count=len(b0_labels), config=config,
            )
            soft_membership[row.hypothesis_id] = normalized_soft_membership(alpha_mass, mask_index, config)
            support_ids = np.asarray(sorted(soft_membership[row.hypothesis_id]), dtype=np.int64)
            # First-round class sizing comes from the frozen branch label.
            # ``seeds`` is updated before round two only after two independent
            # views agree, so a single DINO box can never resize its own test.
            prior_class = seed_by_id[row.candidate_id].branch_class if condition == "class" else None
            prior = size_prior_from_payload(priors, prior_class)
            if len(support_ids) >= 3:
                extents = pca_sorted_extents_m(xyz[support_ids], 1.0)
                oversize = np.count_nonzero(extents > np.asarray(prior.extents_q95_m)) >= 2 or extents[-1] > 1.25 * prior.extents_q95_m[-1]
                size_penalties[row.hypothesis_id] = 0.5 if oversize else 1.0
            else:
                size_penalties[row.hypothesis_id] = 1.0

    selected_by_candidate: dict[int, tuple[MaskHypothesis, ...]] = {}
    evidence_by_candidate: dict[int, GaussianEvidence] = {}
    for seed in seeds:
        candidate_rows = [row for row in hypotheses if row.candidate_id == seed.candidate_id]
        independence = _independence_map(observations.get(seed.candidate_id, ()), config)
        selected = select_consistent_hypotheses(
            candidate_rows, soft_membership, independence,
            hypothesis_penalties=size_penalties, config=config,
        )
        selected_by_candidate[seed.candidate_id] = selected
        evidence_by_candidate[seed.candidate_id] = (
            aggregate_gaussian_evidence(seed.candidate_id, selected, per_hypothesis)
            if selected else _empty_evidence(seed.candidate_id)
        )
    return tuple(hypotheses), evidence_by_candidate, selected_by_candidate


def _refine_profiles(
    *, seeds: Sequence[CandidateSeed], evidence: Mapping[int, GaussianEvidence],
    selected: Mapping[int, Sequence[MaskHypothesis]], xyz: np.ndarray,
    affinity: np.ndarray, b0: np.ndarray, priors: Mapping[str, Any],
    condition: str, round_index: int, config: RefinementConfig,
) -> tuple[dict[str, tuple[ObjectState, ...]], list[LineageRecord]]:
    states_by_profile: dict[str, tuple[ObjectState, ...]] = {}
    lineage: list[LineageRecord] = []
    for profile_name, profile in PROFILES.items():
        states = []
        for seed in seeds:
            review_class, reliable = _review_class(selected.get(seed.candidate_id, ()))
            class_name = (
                seed.branch_class
                if condition == "class" and round_index == 1
                else (review_class if condition == "class" and reliable else None)
            )
            prior = size_prior_from_payload(priors, class_name)
            result = refine_candidate_local(
                seed=seed, evidence=evidence[seed.candidate_id], xyz_m=xyz,
                affinity=affinity, b0_labels=b0, prior=prior, profile=profile,
                round_index=round_index, review_class=review_class,
                reliable_review_class=reliable, config=config,
            )
            states.append(result.state)
            lineage.append(LineageRecord(
                node_id=f"{profile_name}:r{round_index}:refine:{seed.candidate_id}",
                parent_node_ids=(f"seed:{seed.candidate_id}",),
                candidate_ids=seed.parent_candidate_ids, affected_b0_ids=(),
                round_index=round_index, operation="refine",
                added_point_ids=tuple(np.setdiff1d(result.state.point_ids, seed.seed_support).tolist()),
                removed_point_ids=tuple(np.setdiff1d(seed.seed_support, result.state.point_ids).tolist()),
                hypothesis_ids=evidence[seed.candidate_id].selected_hypothesis_ids,
            ))
        states_by_profile[profile_name] = tuple(states)
    return states_by_profile, lineage


def _refine_second_round_profiles(
    *, original_seeds: Sequence[CandidateSeed], states1: Mapping[str, Sequence[ObjectState]],
    evidence: Mapping[int, GaussianEvidence], selected: Mapping[int, Sequence[MaskHypothesis]],
    xyz: np.ndarray, affinity: np.ndarray, b0: np.ndarray, priors: Mapping[str, Any],
    condition: str, config: RefinementConfig,
) -> tuple[dict[str, tuple[ObjectState, ...]], list[LineageRecord]]:
    output: dict[str, tuple[ObjectState, ...]] = {}
    lineage: list[LineageRecord] = []
    seeds_by_id = {row.candidate_id: row for row in original_seeds}
    for profile_name, profile in PROFILES.items():
        states = []
        for previous in states1[profile_name]:
            original = seeds_by_id[previous.object_id]
            seed = _round_seed(original, previous)
            review_class, reliable = _review_class(selected.get(seed.candidate_id, ()))
            if condition == "class" and reliable and review_class is not None:
                seed = replace(seed, branch_class=review_class)
            prior = size_prior_from_payload(priors, review_class if condition == "class" and reliable else None)
            result = refine_candidate_local(
                seed=seed, evidence=evidence[seed.candidate_id], xyz_m=xyz,
                affinity=affinity, b0_labels=b0, prior=prior, profile=profile,
                round_index=2, review_class=review_class,
                reliable_review_class=reliable, config=config,
            )
            states.append(result.state)
            lineage.append(LineageRecord(
                node_id=f"{profile_name}:r2:refine:{seed.candidate_id}",
                parent_node_ids=(f"{profile_name}:r1:refine:{seed.candidate_id}",),
                candidate_ids=seed.parent_candidate_ids, affected_b0_ids=(),
                round_index=2, operation="refine",
                added_point_ids=tuple(np.setdiff1d(result.state.point_ids, previous.point_ids).tolist()),
                removed_point_ids=tuple(np.setdiff1d(previous.point_ids, result.state.point_ids).tolist()),
                hypothesis_ids=evidence[seed.candidate_id].selected_hypothesis_ids,
            ))
        output[profile_name] = tuple(states)
    return output, lineage


def _load_models(args: Any) -> tuple[Any, Any, np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from scene import FeatureGaussianModel, GaussianModel

    rgb = GaussianModel(args.sh_degree)
    rgb.load_ply(args.point_cloud_path)
    feature = FeatureGaussianModel(args.feature_dim, args.semantic_feature_dim)
    feature.load_ply(args.contrastive_feature_point_cloud_path)
    rgb_xyz = rgb.get_xyz.detach().cpu()
    feature_xyz = feature.get_xyz.detach().cpu()
    if not torch.allclose(rgb_xyz, feature_xyz, rtol=0.0, atol=1e-6):
        raise ValueError("RGB and feature Gaussian coordinates disagree")
    xyz = feature_xyz.numpy().astype(np.float64, copy=False)
    affinity = feature.get_point_features.detach().cpu().numpy().astype(np.float64, copy=False)
    scales = feature.get_scaling.detach().cpu()
    is_big = (scales.max(dim=-1).values > scales.max(dim=-1).values.median() * args.scale_threshold).numpy()
    is_transparent = (feature.get_opacity.detach().cpu().squeeze() < args.opcity_threshold).numpy()
    return rgb, feature, xyz, affinity, np.column_stack((is_big, is_transparent))


def run_scene(args: Any) -> dict[str, Any]:
    import torch
    from ..semantic_voting import compute_instance_vote_evidence, summarize_mask_label_assets

    config = RefinementConfig()
    if args.condition not in {"global", "class"}:
        raise ValueError("condition must be global or class")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seeds, b0, reservoir_metadata = load_reservoir(args.reservoir)
    bank = load_candidate_bank(args.candidate_bank)
    if args.classes is None:
        args.classes = list(bank.class_names)
    if tuple(args.classes) != bank.class_names or len(bank.class_names) != 32:
        raise ValueError("iterative refinement requires the CandidateBank's frozen 32-class order")
    rgb, _, xyz_scene, affinity, diagnostic_masks = _load_models(args)
    if len(xyz_scene) != len(b0) or bank.gaussian_xyz_sha256 != gaussian_xyz_sha256(xyz_scene):
        raise ValueError("reservoir/bank/Gaussian identity mismatch")
    xyz = xyz_scene * float(bank.scene_scale_m_per_unit)
    priors = json.loads(Path(args.priors).read_text(encoding="utf-8"))
    cameras = load_cameras(args)
    semantic_assets = summarize_mask_label_assets(args, cameras)
    background = torch.tensor([1., 1., 1.] if args.white_background else [0., 0., 0.], dtype=torch.float32, device="cuda")
    camera_maps = {
        index: _get_camera_maps(
            output, index, camera, rgb, args, background,
            gaussian_sha=bank.gaussian_xyz_sha256,
        )
        for index, camera in enumerate(cameras)
    }
    observations = _candidate_observations(
        seeds=seeds, cameras=cameras, camera_maps=camera_maps, xyz=xyz,
        scene_scale_m_per_unit=bank.scene_scale_m_per_unit, config=config,
    )
    reviewer = GroundedSamFullImageReviewer(
        dino_config=args.groundingdino_config_path,
        dino_checkpoint=args.groundingdino_checkpoint_path,
        sam_checkpoint=args.sam_checkpoint_path,
    )
    observations_round1 = {
        candidate_id: tuple(rows[: config.views_per_round])
        for candidate_id, rows in observations.items()
    }
    hypotheses1, evidence1, selected1 = _review_round(
        round_index=1, seeds=seeds, observations=observations_round1, cameras=cameras,
        camera_maps=camera_maps, b0_labels=b0, model=rgb, args=args,
        background=background, priors=priors, condition=args.condition,
        reviewer=reviewer, classes=bank.class_names, xyz=xyz, config=config,
        cache_root=output, gaussian_sha=bank.gaussian_xyz_sha256,
    )
    states1, profile_lineage = _refine_profiles(
        seeds=seeds, evidence=evidence1, selected=selected1, xyz=xyz,
        affinity=affinity, b0=b0, priors=priors, condition=args.condition,
        round_index=1, config=config,
    )
    lineage = [
        LineageRecord(
            node_id=f"seed:{seed.candidate_id}", parent_node_ids=(),
            candidate_ids=seed.parent_candidate_ids, affected_b0_ids=(),
            round_index=0, operation="seed", added_point_ids=(),
            removed_point_ids=(), hypothesis_ids=(),
        )
        for seed in seeds
    ]
    lineage.extend(profile_lineage)
    # Round two observes the balanced C1 state.  The three graph profiles still
    # replay identical model evidence, so their only varying factors remain the
    # registered unary/pairwise weights.
    balanced1 = {row.object_id: row for row in states1["balanced"]}
    seeds2_rows = []
    for seed in seeds:
        state = balanced1[seed.candidate_id]
        updated = _round_seed(seed, state)
        if args.condition == "class" and state.reliable_review_class and state.review_class is not None:
            updated = replace(updated, branch_class=state.review_class)
        seeds2_rows.append(updated)
    seeds2 = tuple(seeds2_rows)
    observations2_all = _candidate_observations(
        seeds=seeds2, cameras=cameras, camera_maps=camera_maps, xyz=xyz,
        scene_scale_m_per_unit=bank.scene_scale_m_per_unit, config=config,
    )
    observations2 = {}
    for seed in seeds:
        used_cameras = {row.camera_index for row in observations_round1.get(seed.candidate_id, ())}
        observations2[seed.candidate_id] = tuple(
            row for row in observations2_all.get(seed.candidate_id, ())
            if row.camera_index not in used_cameras
        )[: config.views_per_round]
    changed_threshold = {
        seed.candidate_id: max(3, int(math.ceil(len(seed.seed_support) * config.round_change_fraction)))
        for seed in seeds
    }
    triggered = []
    for seed, updated in zip(seeds, seeds2):
        state = balanced1[seed.candidate_id]
        delta = len(np.setxor1d(seed.seed_support, updated.seed_support))
        class_changed = state.reliable_review_class and state.review_class != seed.branch_class
        ownership_conflict = bool(np.any(b0[state.point_ids] >= 0))
        if delta >= changed_threshold[seed.candidate_id] or class_changed or ownership_conflict:
            triggered.append(updated)
    hypotheses2: tuple[MaskHypothesis, ...] = ()
    evidence2: dict[int, GaussianEvidence] = {}
    selected2: dict[int, tuple[MaskHypothesis, ...]] = {}
    if triggered:
        hypotheses2, evidence2, selected2 = _review_round(
            round_index=2, seeds=triggered, observations=observations2,
            cameras=cameras, camera_maps=camera_maps, b0_labels=b0, model=rgb,
            args=args, background=background, priors=priors,
            condition=args.condition, reviewer=reviewer, classes=bank.class_names, xyz=xyz, config=config,
            cache_root=output, gaussian_sha=bank.gaussian_xyz_sha256,
        )
    combined_evidence: dict[int, GaussianEvidence] = {}
    combined_selected: dict[int, tuple[MaskHypothesis, ...]] = {}
    for seed in seeds:
        rows = tuple(selected1.get(seed.candidate_id, ())) + tuple(selected2.get(seed.candidate_id, ()))
        # If the second round has no fresh independent consensus, retain round 1.
        combined_selected[seed.candidate_id] = rows
        if seed.candidate_id in evidence2:
            # Aggregate raw per-hypothesis evidence is not persisted between
            # calls, so combine aligned per-point summaries conservatively.
            a, b = evidence1[seed.candidate_id], evidence2[seed.candidate_id]
            ids = np.union1d(a.point_ids, b.point_ids)
            def values(source: GaussianEvidence, field: str) -> np.ndarray:
                lookup = {int(pid): float(value) for pid, value in zip(source.point_ids, getattr(source, field))}
                return np.asarray([lookup.get(int(pid), 0.) for pid in ids])
            combined_evidence[seed.candidate_id] = GaussianEvidence(
                seed.candidate_id, ids,
                values(a, "hard_positive_views") + values(b, "hard_positive_views"),
                values(a, "hard_negative_views") + values(b, "hard_negative_views"),
                values(a, "alpha_soft_support") + values(b, "alpha_soft_support"),
                a.independent_positive_views + b.independent_positive_views,
                a.independent_negative_views + b.independent_negative_views,
                tuple(row.hypothesis_id for row in rows),
            )
        else:
            combined_evidence[seed.candidate_id] = evidence1[seed.candidate_id]
    states2, lineage2 = _refine_second_round_profiles(
        original_seeds=seeds, states1=states1,
        evidence=combined_evidence, selected=combined_selected,
        xyz=xyz, affinity=affinity, b0=b0, priors=priors,
        condition=args.condition, config=config,
    )
    lineage.extend(lineage2)
    independent_pairs = {}
    pair_votes: dict[tuple[int, int], list[bool]] = {}
    for candidate_id in observations:
        rows = tuple(observations_round1.get(candidate_id, ())) + tuple(observations2.get(candidate_id, ()))
        for pair, value in _independence_map(rows, config).items():
            pair_votes.setdefault(pair, []).append(value)
    independent_pairs = {pair: all(values) for pair, values in pair_votes.items()}
    final_states: dict[str, tuple[ObjectState, ...]] = {}
    seed_by_parent = {parent: seed for seed in seeds for parent in seed.parent_candidate_ids}
    balanced2 = {row.object_id: row for row in states2["balanced"]}
    prior_by_object = {}
    for seed in seeds2:
        state = balanced2[seed.candidate_id]
        class_name = state.review_class if args.condition == "class" and state.reliable_review_class else None
        prior_by_object[seed.candidate_id] = size_prior_from_payload(priors, class_name)
    for profile_name, profile_states in states2.items():
        merged, merge_lineage = merge_objects_once(
            profile_states, hypotheses=combined_selected, evidence=combined_evidence,
            independent_pairs=independent_pairs, xyz_m=xyz,
            prior_by_object=prior_by_object, round_index=2, config=config,
        )
        split, split_lineage = split_disconnected_objects(
            merged, seed_by_parent=seed_by_parent, xyz_m=xyz,
            round_index=2, config=config,
        )
        final_states[profile_name] = split
        lineage.extend(merge_lineage)
        lineage.extend(split_lineage)

    outputs = {}
    for profile_name, states in final_states.items():
        provisional = fuse_objects_with_b0(b0, states, xyz, config=config)
        vote_ratios, raw_votes = compute_instance_vote_evidence(
            label_sets={profile_name: provisional.labels}, camera_list=cameras,
            gs_model=rgb, args=args, bg_color=background, update_progress=False,
        )
        provisional_final = finalize_prediction(
            point_labels=provisional.labels, xyz_scene=xyz_scene,
            is_big_gaussian=diagnostic_masks[:, 0],
            vote_ratios_by_raw=vote_ratios[profile_name], class_names=args.classes,
            selected_classes=bank.saga20_names, label_threshold=config.final_label_threshold,
        )
        valid_raw = {
            raw_id
            for raw_id in provisional_final.raw_instances
            if _candidate_raw_vote_valid(
                raw_votes[profile_name][raw_id], classes=bank.class_names,
                selected_classes=bank.saga20_names,
                threshold=config.final_label_threshold,
            )
        }
        valid_object_ids = {
            object_id for object_id, raw_id in provisional.object_raw_labels.items() if raw_id in valid_raw
        }
        accepted_states = tuple(row for row in states if row.object_id in valid_object_ids)
        fused = fuse_objects_with_b0(b0, accepted_states, xyz, config=config)
        vote_ratios, raw_votes = compute_instance_vote_evidence(
            label_sets={profile_name: fused.labels}, camera_list=cameras,
            gs_model=rgb, args=args, bg_color=background, update_progress=False,
        )
        finalized = finalize_prediction(
            point_labels=fused.labels, xyz_scene=xyz_scene,
            is_big_gaussian=diagnostic_masks[:, 0],
            vote_ratios_by_raw=vote_ratios[profile_name], class_names=args.classes,
            selected_classes=bank.saga20_names, label_threshold=config.final_label_threshold,
        )
        payload = prediction_output_payload(
            finalized, is_big_gaussian=diagnostic_masks[:, 0],
            is_transparent_gaussian=diagnostic_masks[:, 1],
        )
        state_by_object = {row.object_id: row for row in accepted_states}
        candidate_export_ids = {}
        for object_id, raw_id in fused.object_raw_labels.items():
            export_id = finalized.contracted.export_id_by_raw.get(raw_id)
            if export_id is None:
                continue
            for parent_id in state_by_object[object_id].parent_candidate_ids:
                candidate_export_ids[str(parent_id)] = int(export_id)
        payload["candidate_export_ids"] = candidate_export_ids
        payload.update({"condition": args.condition, "profile": profile_name})
        write_prediction_output_atomic(output / profile_name / "output.json", payload)
        for state in states:
            raw_id = fused.object_raw_labels.get(state.object_id)
            export_id = None if raw_id is None else finalized.contracted.export_id_by_raw.get(raw_id)
            affected = tuple(sorted(int(value) for value in np.unique(b0[state.point_ids]) if int(value) >= 0))
            lineage.append(LineageRecord(
                node_id=f"{profile_name}:final:{state.object_id}",
                parent_node_ids=(f"{profile_name}:r2:refine:{state.object_id}",),
                candidate_ids=state.parent_candidate_ids,
                affected_b0_ids=affected,
                round_index=2,
                operation="export" if export_id is not None else "delete",
                added_point_ids=(), removed_point_ids=(),
                hypothesis_ids=combined_evidence.get(state.object_id, _empty_evidence(state.object_id)).selected_hypothesis_ids,
                export_id=None if export_id is None else int(export_id),
            ))
        outputs[profile_name] = {
            "instance_count": len(finalized.contracted.instances),
            "accepted_object_ids": sorted(valid_object_ids),
            "rolled_back_object_ids": sorted(set(provisional.object_raw_labels) - valid_object_ids),
            "claimed_points": fused.diagnostics["claimed_point_count"],
            "candidate_classes": {
                str(state.object_id): {
                    "review_class": state.review_class,
                    "review_reliable": state.reliable_review_class,
                    "final_class": (
                        finalized.class_by_raw.get(fused.object_raw_labels[state.object_id])
                        if state.object_id in fused.object_raw_labels else None
                    ),
                }
                for state in states
            },
        }
    provenance = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip(),
        "condition": args.condition,
        "candidate_bank": str(Path(args.candidate_bank).resolve()),
        "reservoir": str(Path(args.reservoir).resolve()),
        "candidate_count": len(seeds),
        "camera_count": len(cameras),
        "sam_encoding_count": reviewer.sam_encoding_count,
        "gaussian_xyz_sha256": bank.gaussian_xyz_sha256,
        "assets": {
            "rgb_gaussian": {"path": str(Path(args.point_cloud_path).resolve()), "sha256": file_sha256(args.point_cloud_path)},
            "feature_gaussian": {"path": str(Path(args.contrastive_feature_point_cloud_path).resolve()), "sha256": file_sha256(args.contrastive_feature_point_cloud_path)},
            "priors": {"path": str(Path(args.priors).resolve()), "sha256": file_sha256(args.priors)},
            "groundingdino_checkpoint": {"path": str(Path(args.groundingdino_checkpoint_path).resolve()), "sha256": file_sha256(args.groundingdino_checkpoint_path)},
            "groundingdino_config": {"path": str(Path(args.groundingdino_config_path).resolve()), "sha256": file_sha256(args.groundingdino_config_path)},
            "sam_checkpoint": {"path": str(Path(args.sam_checkpoint_path).resolve()), "sha256": file_sha256(args.sam_checkpoint_path)},
            "semantic_votes": semantic_assets,
        },
        "config": asdict(config),
        "profiles": {name: asdict(value) for name, value in PROFILES.items()},
        "round2_triggered_candidates": [row.candidate_id for row in triggered],
        "reservoir_provenance": reservoir_metadata.get("provenance", {}),
    }
    save_scene_cache(
        output, hypotheses=hypotheses1 + hypotheses2,
        evidence=combined_evidence, states_by_profile=final_states,
        lineage=lineage, provenance=provenance,
    )
    result = {"schema": "saga-iterative-refinement-scene-v1", "provenance": provenance, "outputs": outputs}
    json_atomic(output / "iterative_refinement.json", result)
    return result


__all__ = ["run_scene"]

"""Teacher automatic-instance postprocess with a single audit side path.

The prediction path intentionally stays close to the July teacher handoff:
global HDBSCAN, the optional small-object branch, global KNN, the ten-point
filter, 2D semantic voting, and JSON export. Historical category-prior and
Retired experimental branches do not belong in this runtime. The only side path
captures one immutable all-category CandidateBank without changing prediction.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from hdbscan import HDBSCAN

from arguments import ModelParams, PipelineParams
from category_priors.candidate_bank import SAGA20_CLASSES
from category_priors.geometry import nonnegative_cluster_ids, unit_cube_coordinates
from category_priors.legacy_candidate_replay import (
    legacy_filter_small_clusters,
    legacy_knn_filter,
)
from category_priors.prediction_finalization import (
    finalize_prediction,
    prediction_output_payload,
    write_prediction_output_atomic,
)
from category_priors.semantic_voting import compute_instance_vote_evidence
from scene import FeatureGaussianModel, GaussianModel
from scene.dataset_readers import (
    readColmapCameras,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)
from utils.camera_utils import cameraList_from_camInfos
from utils.resource_exit import resource_error_handler


DEFAULT_CLASSES = (
    "chair", "table", "plant", "flower", "foliage", "tv", "painting",
    "sofa", "cabinet", "bed", "wall", "floor", "ceiling", "person",
    "socket", "remote", "key", "book", "lighting", "switch", "door",
    "window", "lamp", "speaker", "computer", "fan", "refrigerator",
    "robot", "cup", "vase", "phone", "trash can",
)
DEFAULT_SELECTED_CLASSES = SAGA20_CLASSES
DEFAULT_OTHER_CLASSES = (
    "switch", "socket", "book", "remote", "key", "cup", "vase", "phone",
)


def uniform_sample(xyz: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Return the teacher baseline's RNG-driven uniform sample mask."""

    count = int(xyz.shape[0])
    selected = torch.randperm(count, device=xyz.device)[: int(n_samples)]
    mask = torch.zeros(count, dtype=torch.bool, device=xyz.device)
    mask[selected] = True
    return mask


def select_points_by_semantic_similarity(
    point_semantic_features: torch.Tensor,
    label_features: torch.Tensor,
    class_idx: int,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select points by cosine similarity to one class feature."""

    class_feature = label_features[class_idx : class_idx + 1]
    similarity = torch.einsum(
        "nc,mc->nm", point_semantic_features, class_feature
    ).squeeze(-1)
    return similarity >= float(threshold), similarity


def _unit_cube_tensor(points: torch.Tensor) -> torch.Tensor:
    normalized = unit_cube_coordinates(points.detach().cpu().numpy())
    return torch.from_numpy(np.ascontiguousarray(normalized)).to(dtype=points.dtype)


def _cluster_centers(
    features: torch.Tensor,
    xyz: torch.Tensor,
    cluster_labels: np.ndarray,
) -> tuple[tuple[int, ...], torch.Tensor, torch.Tensor]:
    """Build centers without assuming that HDBSCAN emitted noise ``-1``."""

    cluster_ids = nonnegative_cluster_ids(cluster_labels)
    if not cluster_ids:
        return (
            cluster_ids,
            torch.empty((0, features.shape[-1]), dtype=features.dtype),
            torch.empty((0, xyz.shape[-1]), dtype=xyz.dtype),
        )
    feature_centers = torch.stack(
        [
            F.normalize(
                features[cluster_labels == cluster_id].mean(dim=0), dim=-1
            )
            for cluster_id in cluster_ids
        ]
    )
    xyz_centers = torch.stack(
        [xyz[cluster_labels == cluster_id].mean(dim=0) for cluster_id in cluster_ids]
    )
    return cluster_ids, feature_centers, xyz_centers


def _assign_to_centers(
    features: torch.Tensor,
    xyz: torch.Tensor,
    feature_centers: torch.Tensor,
    xyz_centers: torch.Tensor,
    feature_ratio: float,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the historical full-point center assignment."""

    if len(feature_centers) == 0:
        return (
            torch.zeros(len(features), dtype=torch.float32),
            torch.full((len(features),), -1, dtype=torch.long),
        )
    feature_similarity = torch.clamp(
        torch.einsum("ac,bc->ab", features, feature_centers), -1, 1
    )
    xyz_similarity = torch.clamp(
        torch.exp(-torch.norm(xyz[:, None, :] - xyz_centers[None, :, :], dim=-1)),
        0,
        1,
    )
    hybrid_similarity = (
        float(feature_ratio) * feature_similarity
        + (1.0 - float(feature_ratio)) * xyz_similarity
    )
    confidence = torch.softmax(hybrid_similarity * 10, dim=-1)
    maximum, labels = confidence.max(dim=-1)
    labels[maximum < float(threshold)] = -1
    return maximum, labels


def cluster_other_classes(
    point_features: torch.Tensor,
    point_semantic_features: torch.Tensor,
    point_xyz: torch.Tensor,
    label_features: torch.Tensor,
    class_to_idx: Mapping[str, int],
    other_classes: list[str] | tuple[str, ...],
    args,
) -> tuple[torch.Tensor, dict[int, str]]:
    """Run the teacher handoff's semantic-guided small-object branch."""

    point_count = int(point_features.shape[0])
    output = torch.full((point_count,), -1, dtype=torch.long)
    instance_to_class: dict[int, str] = {}
    next_instance_id = 0
    normalized_xyz = _unit_cube_tensor(point_xyz)

    for class_name in other_classes:
        if class_name not in class_to_idx:
            print(f"Warning: class {class_name!r} is absent from label features")
            continue
        class_idx = int(class_to_idx[class_name])
        selected_mask, similarity = select_points_by_semantic_similarity(
            point_semantic_features,
            label_features,
            class_idx,
            args.other_classes_similarity_threshold,
        )
        selected_count = int(selected_mask.sum())
        print(
            f"other class {class_name}: selected {selected_count} points "
            f"at similarity >= {args.other_classes_similarity_threshold}"
        )
        if selected_count < int(args.other_classes_min_cluster_size):
            continue

        selected_features = point_features[selected_mask]
        selected_xyz = normalized_xyz[selected_mask]
        selected_similarity = similarity[selected_mask]
        sample_count = min(selected_count, int(args.other_classes_sample_num))
        sampled_local = torch.randperm(selected_count)[:sample_count]
        sampled_features = selected_features[sampled_local]
        sampled_xyz = selected_xyz[sampled_local]
        sampled_similarity = selected_similarity[sampled_local]

        feature_distance = torch.clamp(
            1 - torch.einsum("ac,bc->ab", sampled_features, sampled_features), 0
        )
        spatial_distance = torch.clamp(
            torch.norm(sampled_xyz[:, None, :] - sampled_xyz[None, :, :], dim=-1),
            0,
        )
        semantic_distance = torch.clamp(
            1 - torch.outer(sampled_similarity, sampled_similarity), 0, 1
        )
        if float(feature_distance.max()) > 0:
            feature_distance = feature_distance / (feature_distance.max() + 1e-8)
        if float(spatial_distance.max()) > 0:
            spatial_distance = spatial_distance / (spatial_distance.max() + 1e-8)
        hybrid_distance = (
            float(args.other_classes_feature_ratio) * feature_distance
            + float(args.other_classes_spatial_ratio) * spatial_distance
            + float(args.other_classes_semantic_ratio) * semantic_distance
        )
        cluster_labels = HDBSCAN(
            min_cluster_size=int(args.other_classes_min_cluster_size),
            min_samples=int(args.other_classes_min_cluster_size),
            cluster_selection_epsilon=0.01,
            allow_single_cluster=False,
            metric="precomputed",
        ).fit_predict(hybrid_distance.numpy().astype(np.float64))
        cluster_ids, feature_centers, xyz_centers = _cluster_centers(
            sampled_features, sampled_xyz, cluster_labels
        )
        print(f"other class {class_name}: found {len(cluster_ids)} clusters")
        if not cluster_ids:
            continue

        _, center_labels = _assign_to_centers(
            selected_features,
            selected_xyz,
            feature_centers,
            xyz_centers,
            args.other_classes_feature_ratio,
            args.instance_threshold,
        )
        selected_indices = torch.where(selected_mask)[0]
        for center_index in range(len(cluster_ids)):
            local_mask = center_labels == center_index
            if int(local_mask.sum()) < int(args.other_classes_min_cluster_size):
                continue
            original_indices = selected_indices[local_mask]
            output[original_indices] = next_instance_id
            instance_to_class[next_instance_id] = class_name
            print(
                f"other instance {next_instance_id}: "
                f"{int(local_mask.sum())} points -> {class_name}"
            )
            next_instance_id += 1

    print(f"semantic-guided branch created {next_instance_id} instances")
    return output, instance_to_class


def _load_cameras(args):
    binary_extrinsics = os.path.join(args.sparse_path, "images.bin")
    binary_intrinsics = os.path.join(args.sparse_path, "cameras.bin")
    text_extrinsics = os.path.join(args.sparse_path, "images.txt")
    text_intrinsics = os.path.join(args.sparse_path, "cameras.txt")
    binary_exists = (os.path.isfile(binary_extrinsics), os.path.isfile(binary_intrinsics))
    text_exists = (os.path.isfile(text_extrinsics), os.path.isfile(text_intrinsics))
    if all(binary_exists):
        cameras = readColmapCameras(
            read_extrinsics_binary(binary_extrinsics),
            read_intrinsics_binary(binary_intrinsics),
            args.images_path,
        )
    elif all(text_exists):
        cameras = readColmapCameras(
            read_extrinsics_text(text_extrinsics),
            read_intrinsics_text(text_intrinsics),
            args.images_path,
        )
    else:
        raise FileNotFoundError(
            "COLMAP cameras require a complete images/cameras pair in binary or text form"
        )
    return cameraList_from_camInfos(cameras, 1, args)


def _write_stage_trace(
    path: str,
    *,
    global_sample_core: torch.Tensor,
    global_full_assignment: torch.Tensor,
    other_class_candidates: torch.Tensor,
    branch_membership: torch.Tensor,
    merged_partition: torch.Tensor,
    post_global_knn: torch.Tensor,
    post_filter: torch.Tensor,
    final_internal_labels: torch.Tensor,
    exported_prediction: np.ndarray,
    branch_instance_classes: Mapping[int, str],
    raw_instances: Mapping[int, dict],
    export_id_by_raw: Mapping[int, int],
    vote_histogram_33: Mapping[int, np.ndarray],
    vote_ratios_32: Mapping[int, np.ndarray],
) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            global_sample_core=global_sample_core.numpy(),
            global_full_assignment=global_full_assignment.numpy(),
            other_class_candidates=other_class_candidates.numpy(),
            branch_class_before_merge=branch_membership.numpy(),
            merged_partition=merged_partition.numpy(),
            post_global_knn=post_global_knn.numpy(),
            post_filter=post_filter.numpy(),
            final_internal_labels=final_internal_labels.numpy(),
            exported_prediction=exported_prediction,
        )
    os.replace(temporary, destination)
    metadata = {
        "schema": "saga-teacher-stage-trace-v3",
        "point_count": int(len(final_internal_labels)),
        "level": "L0",
        "branch_instance_classes": {
            str(key): value for key, value in sorted(branch_instance_classes.items())
        },
        "raw_instances": {str(key): value for key, value in raw_instances.items()},
        "export_id_by_raw": {
            str(key): int(value) for key, value in sorted(export_id_by_raw.items())
        },
        "vote_histogram_33": {
            str(key): np.asarray(value, dtype=np.int64).tolist()
            for key, value in vote_histogram_33.items()
        },
        "vote_ratios_32": {
            str(key): np.asarray(value, dtype=np.float64).tolist()
            for key, value in vote_ratios_32.items()
        },
    }
    metadata_destination = destination.with_suffix(".json")
    metadata_temporary = metadata_destination.with_name(
        metadata_destination.name + ".part"
    )
    metadata_temporary.write_text(json.dumps(metadata), encoding="utf-8")
    os.replace(metadata_temporary, metadata_destination)


@resource_error_handler("语义识别后处理阶段")
def main() -> None:
    parser = ArgumentParser(description="SAGA automatic-instance postprocess")
    ModelParams(parser)
    PipelineParams(parser)
    parser.add_argument("--progress_path", "--progress-path", required=True)
    parser.add_argument(
        "--stage_trace_path", "--stage-trace-path", dest="stage_trace_path"
    )
    parser.add_argument("--candidate-bank-path")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--feature_ratio", type=float, default=0.5)
    parser.add_argument("--instance_threshold", type=float, default=0.3)
    parser.add_argument(
        "--min_cluster_size", "--min-cluster-size",
        dest="min_cluster_size", type=int, default=10,
    )
    parser.add_argument("--label_threshold", type=float, default=0.3)
    parser.add_argument("--scale_threshold", type=float, default=0.8)
    parser.add_argument("--opcity_threshold", type=float, default=0.005)
    parser.add_argument("--sample_num", type=int, default=10000)
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument(
        "--other_classes", nargs="+", default=list(DEFAULT_OTHER_CLASSES)
    )
    parser.add_argument(
        "--disable_other_classes", "--disable-other-classes",
        dest="disable_other_classes", action="store_true",
    )
    parser.add_argument(
        "--other_classes_similarity_threshold", type=float, default=0.7
    )
    parser.add_argument("--other_classes_min_cluster_size", type=int, default=5)
    parser.add_argument("--other_classes_sample_num", type=int, default=5000)
    parser.add_argument("--other_classes_feature_ratio", type=float, default=0.5)
    parser.add_argument("--other_classes_spatial_ratio", type=float, default=0.3)
    parser.add_argument("--other_classes_semantic_ratio", type=float, default=0.2)
    parser.add_argument(
        "--scene_scale_m_per_unit", "--scene-scale-m-per-unit",
        dest="scene_scale_m_per_unit", type=float, default=0.0,
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(sys.argv[1:])

    if args.min_cluster_size <= 0:
        parser.error("--min-cluster-size must be positive")
    if args.other_classes_min_cluster_size <= 0:
        parser.error("--other_classes_min_cluster_size must be positive")
    if args.feature_dim != 32 or args.semantic_feature_dim != 32:
        parser.error("the active feature contract requires 32-D affinity and semantic features")
    if args.candidate_bank_path and args.scene_scale_m_per_unit <= 0:
        parser.error("--candidate-bank-path requires positive --scene-scale-m-per-unit")
    if len(args.classes) != 32:
        parser.error("the active semantic contract requires exactly 32 classes")
    if tuple(args.classes) != DEFAULT_CLASSES:
        parser.error("--classes must preserve the registered 32-class order")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    bg_color = torch.tensor(
        [1, 1, 1] if args.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda",
    )

    gs_model = GaussianModel(args.sh_degree)
    gs_model.load_ply(args.point_cloud_path)
    feature_model = FeatureGaussianModel(args.feature_dim, args.semantic_feature_dim)
    feature_model.load_ply(args.contrastive_feature_point_cloud_path)
    scale_gate = torch.nn.Sequential(
        torch.nn.Linear(1, args.feature_dim, bias=True), torch.nn.Sigmoid()
    ).cuda()
    scale_gate.load_state_dict(torch.load(args.scale_gate_path))
    camera_list = _load_cameras(args)

    point_features = feature_model.get_point_features.detach().cpu()
    semantic_features = feature_model.get_point_semantic_features.detach().cpu()
    point_xyz = feature_model.get_xyz.detach().cpu()
    rgb_xyz = gs_model.get_xyz.detach().cpu()
    if rgb_xyz.shape != point_xyz.shape or not torch.allclose(
        rgb_xyz, point_xyz, rtol=0.0, atol=1e-6
    ):
        raise ValueError("RGB and feature Gaussians must have identical XYZ ordering")
    point_scales = feature_model.get_scaling.detach().cpu()
    max_scale = point_scales.max(dim=-1).values
    is_big_gaussian = max_scale > max_scale.median() * args.scale_threshold
    point_opacities = feature_model.get_opacity.detach().cpu().squeeze()
    is_transparent_gaussian = point_opacities < args.opcity_threshold
    gate = scale_gate(torch.tensor([args.scale]).cuda()).unsqueeze(0).detach().cpu()
    print(f"point_features={point_features.shape}, point_xyz={point_xyz.shape}")

    if os.path.isfile(args.label_features_path):
        label_features = torch.load(args.label_features_path, map_location="cpu")
        class_to_idx = {name: index for index, name in enumerate(args.classes)}
        if label_features.ndim != 2 or len(label_features) != len(args.classes):
            raise ValueError("label features do not match the 32-class table")
    else:
        label_features = None
        class_to_idx = None
        if args.candidate_bank_path:
            raise FileNotFoundError(
                "CandidateBank capture requires the 32-class label feature table"
            )
        print("Warning: label features are absent; small-object branch is disabled")

    sampled_mask = uniform_sample(point_xyz, args.sample_num)
    conditioned = F.normalize(point_features, dim=-1, p=2) * gate
    normalized_features = F.normalize(conditioned, dim=-1, p=2)
    normalized_semantic = F.normalize(semantic_features, dim=-1, p=2)
    normalized_xyz = _unit_cube_tensor(point_xyz)
    sampled_features = normalized_features[sampled_mask]
    sampled_xyz = normalized_xyz[sampled_mask]
    feature_distance = torch.clamp(
        1 - torch.einsum("ac,bc->ab", sampled_features, sampled_features), 0
    )
    spatial_distance = torch.clamp(
        torch.norm(sampled_xyz[:, None, :] - sampled_xyz[None, :, :], dim=-1), 0
    )
    hybrid_distance = (
        args.feature_ratio * feature_distance
        + (1.0 - args.feature_ratio) * spatial_distance
    )

    started = datetime.now()
    if len(sampled_features) < int(args.min_cluster_size):
        cluster_labels = np.full(len(sampled_features), -1, dtype=np.int64)
    else:
        cluster_labels = HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_cluster_size,
            cluster_selection_epsilon=0.01,
            allow_single_cluster=False,
            metric="precomputed",
        ).fit_predict(hybrid_distance.numpy().astype(np.float64))
    cluster_ids, feature_centers, xyz_centers = _cluster_centers(
        sampled_features, sampled_xyz, cluster_labels
    )
    _, point_labels = _assign_to_centers(
        normalized_features, normalized_xyz, feature_centers, xyz_centers,
        args.feature_ratio, args.instance_threshold,
    )
    assigned = point_labels >= 0
    print(
        f"global HDBSCAN: {len(cluster_ids)} clusters, "
        f"assigned={int(assigned.sum())}, background={int((~assigned).sum())}, "
        f"elapsed={datetime.now() - started}"
    )

    sampled_indices = torch.nonzero(sampled_mask, as_tuple=False).flatten()
    sampled_raw = torch.as_tensor(cluster_labels, dtype=torch.long)
    global_sample_core = torch.full((len(point_xyz),), -1, dtype=torch.long)
    sampled_foreground = sampled_raw >= 0
    global_sample_core[sampled_indices[sampled_foreground]] = sampled_raw[
        sampled_foreground
    ]
    global_full_assignment = point_labels.clone()

    candidate_bank = None
    candidate_bank_reused = False
    if args.candidate_bank_path:
        from category_priors.candidate_bank import (
            assert_candidate_bank_matches_inputs,
            build_candidate_bank,
            load_candidate_bank,
        )

        bank_path = Path(args.candidate_bank_path)
        bank_npz = bank_path if bank_path.suffix.lower() == ".npz" else bank_path / "bank_labels.npz"
        if bank_npz.exists():
            candidate_bank = load_candidate_bank(bank_path)
            assert_candidate_bank_matches_inputs(
                candidate_bank,
                xyz_scene=point_xyz,
                global_pre_knn=global_full_assignment,
                instance_features=normalized_features,
                semantic_features=normalized_semantic,
                label_features=label_features,
                class_names=args.classes,
                saga20_names=SAGA20_CLASSES,
                scene_scale_m_per_unit=args.scene_scale_m_per_unit,
                seed=args.seed,
            )
            candidate_bank_reused = True
        else:
            candidate_bank = build_candidate_bank(
                normalized_features, normalized_semantic, point_xyz, label_features,
                args.classes, SAGA20_CLASSES, global_full_assignment,
                args.scene_scale_m_per_unit, seed=args.seed,
            )

    other_class_candidates = torch.full_like(point_labels, -1)
    branch_membership = torch.full_like(point_labels, -1)
    branch_instance_classes: dict[int, str] = {}
    if (
        not args.disable_other_classes and label_features is not None
        and class_to_idx is not None and args.other_classes
    ):
        other_labels, other_classes = cluster_other_classes(
            normalized_features.clone(), normalized_semantic.clone(),
            point_xyz.clone(), label_features, class_to_idx,
            args.other_classes, args,
        )
        other_class_candidates = other_labels.clone()
        maximum_main = int(point_labels.max()) if bool((point_labels >= 0).any()) else -1
        for other_id, class_name in other_classes.items():
            merged_id = maximum_main + 1 + int(other_id)
            mask = other_labels == int(other_id)
            point_labels[mask] = merged_id
            branch_membership[mask] = merged_id
            branch_instance_classes[merged_id] = class_name
        print(f"merged {len(other_classes)} small-object instances")

    merged_partition = point_labels.clone()
    started = datetime.now()
    if args.k > 0:
        replay = legacy_knn_filter(point_xyz, point_labels, k=args.k, min_count=10)
        post_global_knn = torch.from_numpy(replay.after_knn.copy())
        post_filter = torch.from_numpy(replay.after_filter.copy())
    else:
        post_global_knn = point_labels.clone()
        filtered, _ = legacy_filter_small_clusters(point_labels, min_count=10)
        post_filter = torch.from_numpy(filtered.copy())
    point_labels = post_filter.clone()
    print(
        f"legacy KNN/filter: {len(torch.unique(point_labels))} labels, "
        f"elapsed={datetime.now() - started}"
    )

    label_sets: dict[str, torch.Tensor | np.ndarray] = {"prediction": point_labels}
    if candidate_bank is not None:
        label_sets["candidate_bank"] = candidate_bank.branch_full_labels
    ratios, raw_votes = compute_instance_vote_evidence(
        label_sets=label_sets, camera_list=camera_list, gs_model=gs_model,
        args=args, bg_color=bg_color, update_progress=True,
    )
    instance_ratio = ratios["prediction"]
    if candidate_bank is not None:
        from category_priors.candidate_bank import attach_candidate_votes, save_candidate_bank

        voted_candidate_bank = attach_candidate_votes(
            candidate_bank, ratios["candidate_bank"], args.classes
        )
        if candidate_bank_reused:
            if voted_candidate_bank.candidates != candidate_bank.candidates:
                raise ValueError(
                    "frozen CandidateBank vote evidence differs from this replay"
                )
            print(f"CandidateBank reused read-only from {args.candidate_bank_path}")
        else:
            candidate_bank = voted_candidate_bank
            save_candidate_bank(candidate_bank, args.candidate_bank_path)
            print(f"CandidateBank saved to {args.candidate_bank_path}")

    finalized = finalize_prediction(
        point_labels=point_labels,
        xyz_scene=point_xyz,
        is_big_gaussian=is_big_gaussian,
        vote_ratios_by_raw=instance_ratio,
        class_names=args.classes,
        selected_classes=DEFAULT_SELECTED_CLASSES,
        label_threshold=args.label_threshold,
    )
    contracted = finalized.contracted
    raw_instances = dict(finalized.raw_instances)

    if args.stage_trace_path:
        _write_stage_trace(
            args.stage_trace_path,
            global_sample_core=global_sample_core,
            global_full_assignment=global_full_assignment,
            other_class_candidates=other_class_candidates,
            branch_membership=branch_membership,
            merged_partition=merged_partition,
            post_global_knn=post_global_knn,
            post_filter=post_filter,
            final_internal_labels=point_labels,
            exported_prediction=contracted.point_labels,
            branch_instance_classes=branch_instance_classes,
            raw_instances=raw_instances,
            export_id_by_raw=contracted.export_id_by_raw,
            vote_histogram_33=raw_votes["prediction"],
            vote_ratios_32=instance_ratio,
        )

    output = prediction_output_payload(
        finalized,
        is_big_gaussian=is_big_gaussian,
        is_transparent_gaussian=is_transparent_gaussian,
    )
    write_prediction_output_atomic(args.json_path, output)

    if args.clean:
        for directory in (args.masks_path, args.labels_path, args.mask_scales_path):
            if os.path.isdir(directory):
                shutil.rmtree(directory)
        for filename in (
            args.contrastive_feature_point_cloud_path, args.scale_gate_path,
        ):
            if os.path.isfile(filename):
                os.remove(filename)


if __name__ == "__main__":
    main()

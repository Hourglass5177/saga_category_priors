import shutil
import torch
from scene import Scene
import os
import json
import hashlib
import sys
from datetime import datetime
from tqdm import tqdm
from gaussian_renderer import render_with_max_contributor
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
# from gaussian_renderer import GaussianModel
import numpy as np
import cv2
from sklearn.decomposition import PCA
import torch.nn.functional as F

# from scene.gaussian_model import GaussianModel
from scene import Scene, GaussianModel, FeatureGaussianModel
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from scene.dataset_readers import readColmapCameras, read_extrinsics_binary, read_intrinsics_binary, read_extrinsics_text, read_intrinsics_text
from utils.camera_utils import cameraList_from_camInfos
from utils.visualization_utils import save_image
from utils.clip_utils import get_relevancy

from scipy.spatial import KDTree
from hdbscan import HDBSCAN
from utils.resource_exit import resource_error_handler
from category_priors.prediction_contract import normalize_prediction

def uniform_sample(xyz, n_samples):
    device = xyz.device
    N, _ = xyz.shape
    # 生成均匀随机采样的索引
    selected_indices = torch.randperm(N,device=device)[:n_samples]
    # 创建全False的布尔张量
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    # 将选中的位置设置为True
    mask[selected_indices] = True
    return mask

def select_points_by_semantic_similarity(point_semantic_features, label_features, class_idx,
                                         threshold, device='cpu'):
    """
    Select points based on semantic feature similarity to a specific class.

    Args:
        point_semantic_features: [N, feature_dim] - Normalized point features
        label_features: [num_classes, feature_dim] - Normalized class features
        class_idx: int - Index of target class in label_features
        threshold: float - Minimum cosine similarity for selection
        device: torch device

    Returns:
        selection_mask: [N] boolean tensor - True if point matches class
        similarity_scores: [N] tensor - Cosine similarity scores
    """
    # Compute cosine similarity between all points and target class feature
    class_feature = label_features[class_idx:class_idx+1]  # [1, feature_dim]
    similarity = torch.einsum('nc,mc->nm', point_semantic_features, class_feature).squeeze(-1)  # [N]

    # Select points above threshold
    selection_mask = similarity >= threshold

    return selection_mask, similarity

def cluster_other_classes(point_features, point_semantic_features, point_xyz, label_features, class_to_idx,
                          other_classes, args, device='cpu', legacy_prior=None,
                          raw_point_features=None, scale_gate=None,
                          mask_scales=(), surface_density=1.0,
                          semantic_vote_scores=None, teacher_prior=None,
                          exclusive_masks=None, diagnostics_attribute=None,
                          v4_candidate=None, v5_candidate=None):
    """
    Perform semantic-guided clustering for 'other_classes' (small objects).

    Pipeline:
    1. For each class in other_classes:
       a. Select points with high similarity to class semantic feature
       b. Sample selected points for efficiency
       c. Apply HDBSCAN clustering with hybrid distance (instance + spatial + semantic)
       d. Assign class label directly (no voting needed)

    Args:
        point_features: [N, feature_dim] - Normalized point features (instance features)
        point_semantic_features: [N, semantic_feature_dim] - Normalized semantic features
        point_xyz: [N, 3] - Spatial coordinates
        label_features: [num_classes, feature_dim] - Semantic embeddings
        class_to_idx: dict - Class name to index mapping
        other_classes: list[str] - Classes to cluster with this method
        args: ArgumentParser namespace with hyperparameters
        device: torch device

    Returns:
        other_point_labels: [N] tensor - Instance labels (-1 for unassigned, 0+ for assigned)
        other_instance_to_class: dict - Mapping instance_id -> class_name
    """
    N = point_features.shape[0]
    other_point_labels = torch.full((N,), -1, dtype=torch.long, device=device)
    other_instance_to_class = {}
    current_instance_id = 0
    branch_diagnostics = {}
    candidate_records = []
    candidate_memberships = torch.zeros(N, dtype=torch.int32, device=device)
    v5_core_labels = torch.full((N,), -1, dtype=torch.long, device=device)
    v5_assignment_confidence = torch.zeros(N, dtype=torch.float32, device=device)
    preserve_teacher_branch = bool(
        teacher_prior is not None
        and teacher_prior["table"].get("branch_preservation", False)
    )

    # Normalize spatial coordinates
    min_val = torch.min(point_xyz, dim=0).values
    max_val = torch.max(point_xyz, dim=0).values
    std_point_xyz = (point_xyz - min_val) / (max_val - min_val)

    for class_name in other_classes:
        class_started = datetime.now()
        if class_name not in class_to_idx:
            print(f"Warning: Class '{class_name}' not in label_features, skipping")
            continue

        class_idx = class_to_idx[class_name]
        print(f"\nProcessing other class: {class_name} (idx={class_idx})")

        # Step 1: Select points by semantic similarity (using semantic features)
        semantic_threshold = args.other_classes_similarity_threshold
        parameters = None
        teacher_parameters = None
        v4_parameters = None
        v5_parameters = None
        if teacher_prior is not None:
            from category_priors.teacher_prior import resolve_teacher_parameters
            teacher_parameters = resolve_teacher_parameters(
                teacher_prior["table"], class_name, teacher_prior["mode"]
            )
            semantic_threshold = float(teacher_parameters["semantic_threshold"])
        elif legacy_prior is not None:
            from category_priors.legacy_prior import resolve_class_parameters
            if semantic_vote_scores is not None:
                preliminary_scores = semantic_vote_scores[:, class_idx]
                preliminary_mask = (
                    preliminary_scores >= legacy_prior["config"].semantic_threshold
                )
            else:
                preliminary_mask, _ = select_points_by_semantic_similarity(
                    point_semantic_features, label_features, class_idx,
                    legacy_prior["config"].semantic_threshold, device
                )
            parameters = resolve_class_parameters(
                legacy_prior["priors"], legacy_prior["config"], class_name,
                legacy_prior["mode"], int(preliminary_mask.sum()),
                surface_density, mask_scales,
            )
            semantic_threshold = float(parameters["semantic_threshold"])
        elif v4_candidate is not None:
            from category_priors.v4_candidate import resolve_v4_candidate_parameters
            preliminary_mask = torch.as_tensor(
                exclusive_masks[class_idx], dtype=torch.bool, device=device
            )
            v4_parameters = resolve_v4_candidate_parameters(
                v4_candidate["priors"], v4_candidate["mode"], class_name,
                int(preliminary_mask.sum()), surface_density, mask_scales,
                v4_candidate["config"],
            )
            semantic_threshold = float(v4_parameters["semantic_threshold"])
        elif v5_candidate is not None:
            from category_priors.v5_candidate import resolve_v5_candidate_parameters
            preliminary_mask = torch.as_tensor(
                exclusive_masks[class_idx], dtype=torch.bool, device=device
            )
            v5_parameters = resolve_v5_candidate_parameters(
                int(preliminary_mask.sum()), v5_candidate["config"]
            )
            semantic_threshold = float(v5_parameters["semantic_threshold"])
        if exclusive_masks is not None:
            similarity_scores = torch.einsum(
                'nc,c->n', point_semantic_features, label_features[class_idx]
            )
            selection_mask = torch.as_tensor(
                exclusive_masks[class_idx], dtype=torch.bool, device=device
            )
        elif semantic_vote_scores is not None:
            similarity_scores = semantic_vote_scores[:, class_idx]
            selection_mask = similarity_scores >= semantic_threshold
        else:
            selection_mask, similarity_scores = select_points_by_semantic_similarity(
                point_semantic_features, label_features, class_idx,
                semantic_threshold, device
            )
        candidate_memberships += selection_mask.to(torch.int32)

        num_selected = selection_mask.sum().item()
        print(f"  Selected {num_selected} points (similarity >= {semantic_threshold})")

        min_cluster_size = (
            int(teacher_parameters["min_cluster_size"])
            if teacher_parameters is not None else (
                int(parameters["min_cluster_size"])
                if parameters is not None else (
                    int(v4_parameters["min_cluster_size"])
                    if v4_parameters is not None else (
                        int(v5_parameters["min_cluster_size"])
                        if v5_parameters is not None else args.other_classes_min_cluster_size
                    )
                )
            )
        )
        min_samples = (
            int(teacher_parameters["min_samples"])
            if teacher_parameters is not None else (
                int(parameters["min_samples"])
                if parameters is not None else (
                    int(v4_parameters["min_samples"])
                    if v4_parameters is not None else (
                        int(v5_parameters["min_samples"])
                        if v5_parameters is not None else min_cluster_size
                    )
                )
            )
        )
        if num_selected < min_cluster_size:
            print(f"  Skipping: insufficient points")
            branch_diagnostics[class_name] = {
                "candidate_points": int(num_selected), "status": "insufficient_points",
                "parameters": teacher_parameters if teacher_parameters is not None else (
                    parameters if parameters is not None else (
                        v4_parameters if v4_parameters is not None else v5_parameters
                    )
                ),
            }
            continue

        # Step 2: Sample selected points for efficiency
        selected_features = point_features[selection_mask]
        if (
            (parameters is not None or v4_parameters is not None)
            and raw_point_features is not None
            and scale_gate is not None
        ):
            gate_input = float(
                parameters["scale_gate_input"]
                if parameters is not None else v4_parameters["scale_gate_input"]
            )
            with torch.no_grad():
                gate = scale_gate(
                    torch.tensor([gate_input], dtype=torch.float32, device="cuda")
                ).detach().cpu()
            selected_features = F.normalize(
                F.normalize(raw_point_features[selection_mask], dim=-1, p=2) * gate,
                dim=-1,
                p=2,
            )
        selected_xyz_m = None
        if teacher_parameters is not None:
            selected_xyz_m = (
                point_xyz[selection_mask] * float(args.scene_scale_m_per_unit)
            )
            selected_xyz = selected_xyz_m / max(
                float(teacher_parameters["spatial_scale_m"]), 1e-12
            )
        elif parameters is not None and parameters["spatial_scale_m"] is not None:
            selected_xyz = (
                point_xyz[selection_mask]
                * float(args.scene_scale_m_per_unit)
                / max(float(parameters["spatial_scale_m"]), 1e-12)
            )
        else:
            selected_xyz = std_point_xyz[selection_mask]
        selected_similarities = similarity_scores[selection_mask]

        sample_size = (
            min(num_selected, int(teacher_parameters["sample_num"]))
            if teacher_parameters is not None else (
                int(parameters["sample_count"])
                if parameters is not None else (
                    int(v4_parameters["sample_count"])
                    if v4_parameters is not None else (
                        int(v5_parameters["sample_count"])
                        if v5_parameters is not None else min(num_selected, args.other_classes_sample_num)
                    )
                )
            )
        )
        if v4_parameters is not None:
            from category_priors.v4_candidate import nested_permutation
            sampled_indices = torch.as_tensor(
                nested_permutation(num_selected, args.seed, class_name)[:sample_size],
                dtype=torch.long, device=device,
            )
        elif v5_parameters is not None:
            from category_priors.v5_candidate import nested_permutation
            sampled_indices = torch.as_tensor(
                nested_permutation(num_selected, args.seed, class_name)[:sample_size],
                dtype=torch.long, device=device,
            )
        elif teacher_parameters is not None and exclusive_masks is not None:
            class_seed = int(args.seed) + sum(
                (index + 1) * ord(character)
                for index, character in enumerate(class_name)
            )
            generator = torch.Generator(device=device)
            generator.manual_seed(class_seed)
            sampled_indices = torch.randperm(
                num_selected, device=device, generator=generator
            )[:sample_size]
        else:
            sampled_indices = torch.randperm(num_selected, device=device)[:sample_size]
        sampled_features = selected_features[sampled_indices]
        sampled_xyz = selected_xyz[sampled_indices]
        sampled_similarities = selected_similarities[sampled_indices]

        # Step 3: Compute hybrid distance (instance feature + spatial + semantic)
        # Instance feature distance
        instance_feature_dist = torch.clamp(
            1 - torch.einsum('ac,bc->ab', sampled_features, sampled_features), 0
        )

        # Spatial distance
        spatial_dist = torch.clamp(
            torch.norm(sampled_xyz[:, None, :] - sampled_xyz[None, :, :], dim=-1), 0
        )
        if teacher_parameters is not None:
            spatial_dist = torch.clamp(spatial_dist, max=1.0)

        # Semantic distance: 1 - similarity to class feature (lower is better)
        # Both points should be similar to the class semantic feature
        semantic_sim_matrix = torch.outer(sampled_similarities, sampled_similarities)
        # Use negative correlation: if both points have high similarity, semantic distance should be low
        semantic_dist = 1 - semantic_sim_matrix
        semantic_dist = torch.clamp(semantic_dist, 0, 1)

        # Normalize distances to [0, 1] range
        if instance_feature_dist.max() > 0:
            instance_feature_dist = instance_feature_dist / (instance_feature_dist.max() + 1e-8)
        if teacher_parameters is None and spatial_dist.max() > 0:
            spatial_dist = spatial_dist / (spatial_dist.max() + 1e-8)

        # Hybrid distance with three components
        feature_ratio = (
            float(teacher_parameters["feature_ratio"])
            if teacher_parameters is not None else args.other_classes_feature_ratio
        )
        spatial_ratio = (
            float(teacher_parameters["spatial_ratio"])
            if teacher_parameters is not None else args.other_classes_spatial_ratio
        )
        semantic_ratio = (
            float(teacher_parameters["semantic_ratio"])
            if teacher_parameters is not None else args.other_classes_semantic_ratio
        )
        hybrid_distance = (feature_ratio * instance_feature_dist +
                          spatial_ratio * spatial_dist +
                          semantic_ratio * semantic_dist)

        # Step 4: HDBSCAN clustering
        if teacher_parameters is not None:
            from category_priors.teacher_prior import build_teacher_hdbscan
            clusterer = build_teacher_hdbscan(
                HDBSCAN, min_cluster_size=min_cluster_size, min_samples=min_samples
            )
        else:
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=0.01,
                allow_single_cluster=False,
                metric='precomputed'
            )
        cluster_labels = clusterer.fit_predict(hybrid_distance.numpy().astype(np.float64))

        num_clusters = len([l for l in np.unique(cluster_labels) if l >= 0])
        print(f"  Found {num_clusters} clusters")

        if num_clusters == 0:
            print(f"  Skipping: no valid clusters")
            branch_diagnostics[class_name] = {
                "candidate_points": int(num_selected),
                "sampled_points": int(sample_size),
                "hdbscan_noise_points": int((cluster_labels < 0).sum()),
                "status": "no_clusters",
                "parameters": teacher_parameters if teacher_parameters is not None else (
                    parameters if parameters is not None else (
                        v4_parameters if v4_parameters is not None else v5_parameters
                    )
                ),
            }
            continue

        # Step 5: Compute cluster centers for full assignment
        feature_cluster_centers = []
        xyz_cluster_centers = []

        for i in np.unique(cluster_labels):
            if i < 0:
                continue
            feature_cluster_centers.append(
                F.normalize(sampled_features[cluster_labels == i].mean(dim=0), dim=-1)
            )
            xyz_cluster_centers.append(sampled_xyz[cluster_labels == i].mean(dim=0))

        feature_cluster_centers = torch.stack(feature_cluster_centers)  # [num_clusters, feature_dim]
        xyz_cluster_centers = torch.stack(xyz_cluster_centers)  # [num_clusters, 3]

        # Step 6: Assign all selected points to clusters (not just sampled)
        selected_feature_sim = torch.clamp(
            torch.einsum('ac,bc->ab', selected_features, feature_cluster_centers), -1, 1
        )
        selected_xyz_sim = torch.clamp(
            torch.exp(-torch.norm(selected_xyz[:, None, :] - xyz_cluster_centers[None, :, :], dim=-1)), 0, 1
        )
        selected_hybrid_sim = (feature_ratio * selected_feature_sim +
                              (1 - feature_ratio) * selected_xyz_sim)
        selected_confidence = torch.softmax(selected_hybrid_sim * 10, dim=-1)
        selected_mask, selected_cluster_labels = selected_confidence.max(dim=-1)

        # Apply threshold
        assignment_threshold = (
            float(teacher_parameters["assignment_threshold"])
            if teacher_parameters is not None else (
                float(parameters["assignment_threshold"])
                if parameters is not None else (
                    float(v5_parameters["assignment_threshold"])
                    if v5_parameters is not None else args.instance_threshold
                )
            )
        )
        below_threshold = selected_mask < assignment_threshold
        selected_cluster_labels[below_threshold] = -1

        rescued = 0
        if teacher_parameters is not None and preserve_teacher_branch:
            from category_priors.teacher_prior import (
                class_local_knn,
                filter_small_class_clusters,
                rescue_same_class_noise,
            )
            local = selected_cluster_labels.detach().cpu().numpy()
            selected_xyz_m_np = selected_xyz_m.detach().cpu().numpy()
            local = class_local_knn(
                local, selected_xyz_m_np, int(teacher_parameters["knn_k"])
            )
            local = filter_small_class_clusters(local, min_cluster_size)
            local, rescued = rescue_same_class_noise(
                local, selected_xyz_m_np, teacher_parameters["rescue_radius_m"]
            )
            selected_cluster_labels = torch.as_tensor(local, device=device)
        elif parameters is not None:
            from category_priors.legacy_prior import radius_vote_labels, rescue_halo
            selected_xyz_m = (
                point_xyz[selection_mask].detach().cpu().numpy()
                * float(args.scene_scale_m_per_unit)
            )
            local = selected_cluster_labels.detach().cpu().numpy()
            local = radius_vote_labels(
                local, selected_xyz_m, parameters["smoothing_radius_m"],
                int(parameters["knn_max"]),
            )
            if parameters["rescue_enabled"]:
                local, rescued = rescue_halo(
                    local, selected_xyz_m, parameters["rescue_radius_m"],
                    int(parameters["halo_neighbors"]),
                    int(parameters["halo_min_agreement"]),
                )
            selected_cluster_labels = torch.as_tensor(local, device=device)

        # Step 7: Map back to original point indices and assign instance IDs
        selected_indices_original = torch.where(selection_mask)[0]

        for local_cluster_id in range(num_clusters):
            # Find points assigned to this cluster
            points_in_cluster = (selected_cluster_labels == local_cluster_id)

            if points_in_cluster.sum() < min_cluster_size:
                continue

            # Get original point indices
            original_indices = selected_indices_original[points_in_cluster]
            candidate_core_mask = points_in_cluster & (
                selected_mask >= float(v5_parameters["core_assignment_threshold"])
            ) if v5_candidate is not None else None

            candidate_semantic = torch.einsum(
                'nc,mc->nm', point_semantic_features[original_indices], label_features
            )
            semantic_top1 = candidate_semantic.argmax(dim=1)
            semantic_purity = float(
                (semantic_top1 == int(class_idx)).float().mean()
            )
            if candidate_semantic.shape[1] > 1:
                semantic_top2 = torch.topk(candidate_semantic, k=2, dim=1).values
                semantic_margin = float(
                    (semantic_top2[:, 0] - semantic_top2[:, 1]).mean()
                )
            else:
                semantic_margin = 0.0
            candidate_xyz_m = (
                point_xyz[original_indices] * float(args.scene_scale_m_per_unit)
            )
            candidate_extent_m = candidate_xyz_m.max(dim=0).values - candidate_xyz_m.min(dim=0).values
            sample_core_mask = cluster_labels == local_cluster_id
            persistence_values = getattr(clusterer, 'cluster_persistence_', [])
            membership_values = getattr(clusterer, 'probabilities_', None)
            persistence = (
                float(persistence_values[local_cluster_id])
                if local_cluster_id < len(persistence_values) else None
            )
            membership_mean = (
                float(np.asarray(membership_values)[sample_core_mask].mean())
                if membership_values is not None and np.any(sample_core_mask) else None
            )
            metric_extents_m = np.sort(
                candidate_extent_m.detach().cpu().numpy()
            ).tolist()
            local_surface_density = None
            if v5_candidate is not None and int(candidate_core_mask.sum()) >= 17:
                from scipy.spatial import cKDTree
                core_xyz_np = (
                    point_xyz[selected_indices_original[candidate_core_mask]]
                    * float(args.scene_scale_m_per_unit)
                ).detach().cpu().numpy()
                distances, _ = cKDTree(core_xyz_np).query(
                    core_xyz_np, k=17, workers=-1
                )
                radius16 = np.maximum(np.asarray(distances)[:, -1], 1e-12)
                local_surface_density = float(np.median(
                    16.0 / (np.pi * radius16 * radius16)
                ))

            # Assign instance ID (starts from 0)
            instance_id = current_instance_id
            other_point_labels[original_indices] = instance_id
            other_instance_to_class[instance_id] = class_name
            if v5_candidate is not None:
                v5_assignment_confidence[original_indices] = (
                    selected_mask[points_in_cluster].to(torch.float32)
                )
                v5_core_labels[selected_indices_original[candidate_core_mask]] = instance_id

            candidate_records.append({
                "candidate_id": int(instance_id),
                "branch_class": class_name,
                "branch_class_index": int(class_idx),
                "semantic_candidate_points": int(num_selected),
                "sampled_points": int(sample_size),
                "sample_core_points": int(np.sum(sample_core_mask)),
                "full_assignment_points": int(points_in_cluster.sum()),
                "assignment_confidence_mean": float(selected_mask[points_in_cluster].mean()),
                "hdbscan_persistence": persistence,
                "hdbscan_membership_mean": membership_mean,
                "semantic_top1_purity": semantic_purity,
                "semantic_margin_mean": semantic_margin,
                "bbox_diag_m": float(torch.linalg.norm(candidate_extent_m)),
                "metric_extents_m": metric_extents_m,
                "local_surface_density": local_surface_density,
                "core_assignment_points": int((
                    candidate_core_mask.sum()
                )) if v5_candidate is not None else None,
                "class_runtime_seconds": float((datetime.now() - class_started).total_seconds()) if v5_candidate is not None else None,
            })

            current_instance_id += 1
            print(f"  Instance {instance_id}: {points_in_cluster.sum()} points -> {class_name}")

        branch_diagnostics[class_name] = {
            "candidate_points": int(num_selected),
            "sampled_points": int(sample_size),
            "hdbscan_noise_points": int((cluster_labels < 0).sum()),
            "rescued_points": int(rescued),
            "final_instances": int(sum(1 for value in other_instance_to_class.values() if value == class_name)),
            "status": "complete",
            "parameters": teacher_parameters if teacher_parameters is not None else (
                parameters if parameters is not None else (
                    v4_parameters if v4_parameters is not None else v5_parameters
                )
            ),
        }

    print(f"\nSemantic-guided clustering complete: {current_instance_id} instances created")
    branch_diagnostics["__summary__"] = {
        "candidate_points": int((candidate_memberships > 0).sum()),
        "candidate_memberships": int(candidate_memberships.sum()),
        "overlap_points": int((candidate_memberships > 1).sum()),
    }
    branch_diagnostics["__candidates__"] = candidate_records
    if diagnostics_attribute is None:
        diagnostics_attribute = (
            "_teacher_prior_diagnostics"
            if teacher_prior is not None else "_legacy_prior_diagnostics"
        )
    setattr(args, diagnostics_attribute, branch_diagnostics)
    if v5_candidate is not None:
        setattr(args, "_v5_core_labels", v5_core_labels.detach().cpu())
        setattr(args, "_v5_assignment_confidence", v5_assignment_confidence.detach().cpu())
    return other_point_labels, other_instance_to_class


def run_class_first_postprocess(args):
    """Independent class-first path: no RGB model, cameras, masks, or 2-D vote."""
    from category_priors.class_first import (
        build_class_first_metadata,
        load_class_first_config,
        run_class_first,
    )
    from category_priors.io import load_json, write_json

    config = load_class_first_config(args.class_first_config)
    priors = load_json(args.category_priors)
    feature_model = FeatureGaussianModel(args.feature_dim, args.semantic_feature_dim)
    feature_model.load_ply(args.contrastive_feature_point_cloud_path)
    point_features = feature_model.get_point_features.detach().cpu()
    semantic_features = feature_model.get_point_semantic_features.detach().cpu()
    point_xyz = feature_model.get_xyz.detach().cpu()
    point_scales = feature_model.get_scaling.detach().cpu()
    point_opacities = feature_model.get_opacity.detach().cpu().squeeze()

    scale_gate = torch.nn.Sequential(
        torch.nn.Linear(1, args.feature_dim, bias=True), torch.nn.Sigmoid()
    ).cuda()
    scale_gate.load_state_dict(torch.load(args.scale_gate_path))
    gates = scale_gate(torch.tensor([args.scale]).cuda()).unsqueeze(0).detach().cpu()
    point_features = F.normalize(
        F.normalize(point_features, dim=-1, p=2) * gates, dim=-1, p=2
    )
    semantic_features = F.normalize(semantic_features, dim=-1, p=2)
    label_features = torch.load(args.label_features_path, map_location="cpu")
    if not isinstance(label_features, torch.Tensor):
        raise TypeError("label_features must be a tensor")
    label_features = F.normalize(label_features.detach().cpu(), dim=-1, p=2)

    max_scale = point_scales.max(dim=-1).values
    valid_mask = (
        (point_opacities >= config.opacity_threshold)
        & (max_scale <= config.scale_threshold)
    )
    result = run_class_first(
        point_features,
        semantic_features,
        point_xyz,
        label_features,
        args.classes,
        priors,
        config,
        args.class_prior_mode,
        args.scene_scale_m_per_unit,
        seed=args.seed,
        valid_mask=valid_mask,
        selected_classes=args.selected_classes,
    )
    output = {
        "point_labels": result.labels.tolist(),
        "is_big_gaussian": (max_scale > config.scale_threshold).tolist(),
        "is_transparent_gaissian": (
            point_opacities < config.opacity_threshold
        ).tolist(),
        "instances": {
            str(instance_id): dict(values)
            for instance_id, values in sorted(result.instances.items())
        },
    }
    write_json(args.json_path, output)
    if args.prior_metadata_path:
        write_json(
            args.prior_metadata_path,
            build_class_first_metadata(
                result,
                {
                    "clustering_mode": "class-first",
                    "class_prior_mode": args.class_prior_mode,
                    "seed": int(args.seed),
                    "scene_scale_m_per_unit": float(args.scene_scale_m_per_unit),
                },
            ),
        )
    progress_path = os.path.abspath(args.progress_path)
    os.makedirs(os.path.dirname(progress_path), exist_ok=True)
    with open(progress_path, "w", encoding="utf-8") as handle:
        handle.write("1.0\n")
    print(
        "class-first complete: "
        f"{result.diagnostics['totals']['final_instances']} instances, "
        f"coverage={result.diagnostics['totals']['coverage']:.4f}"
    )

@resource_error_handler("语义识别后处理阶段")
def main():
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--progress_path", type=str, required=True)
    parser.add_argument(
        "--stage_trace_path",
        type=str,
        default=None,
        help="Optional V9 forensic NPZ sidecar; never consumed by prediction.",
    )
    parser.add_argument("--clean", action='store_true')
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--feature_ratio", type=float, default=0.5)
    parser.add_argument("--instance_threshold", type=float, default=0.3)
    parser.add_argument("--min_cluster_size", type=int, default=10)
    parser.add_argument("--label_threshold", type=float, default=0.3)
    parser.add_argument("--scale_threshold", type=float, default=0.8)
    parser.add_argument("--opcity_threshold", type=float, default=0.005)
    parser.add_argument("--sample_num", type=int, default=10000)
    parser.add_argument(
        "--v7-causal-ablation", choices=("L0", "L1", "L2", "L3"), default="L0"
    )
    parser.add_argument("--classes", nargs="+", type=str, default=[
        'chair', 'table', 'plant', 'flower', 'foliage', 'tv', 'painting', 'sofa',
        'cabinet', 'bed', 'wall', 'floor', 'ceiling', 'person', 'socket', 'remote',
        'key', 'book', 'lighting', 'switch', 'door', 'window', 'lamp', 'speaker',
        'computer', 'fan', 'refrigerator', 'robot', 'cup', 'vase', 'phone', 'trash can'
    ])
    parser.add_argument("--selected_classes", nargs="+", type=str, default=[
        'chair', 'table', 'plant', 'flower', 'foliage', 'tv', 'painting', 'sofa',
        'cabinet', 'bed', 'socket', 'remote', 'key', 'book', 'lighting', 'switch',
        'door', 'window', 'lamp', 'speaker', 'computer', 'fan', 'refrigerator',
        'robot', 'cup', 'vase', 'phone', 'trash can'
    ])
    parser.add_argument("--other_classes", nargs="+", type=str, default=['switch', 'socket', 'book', 'remote', 'key', 'cup', 'vase', 'phone'])
    parser.add_argument("--disable_other_classes", action='store_true',
                        help="Disable the semantic-guided small-object branch (registered B0 legacy condition)")
    # New arguments for semantic-guided clustering of other_classes
    parser.add_argument("--other_classes_similarity_threshold", type=float, default=0.7,
                        help="Minimum cosine similarity to class semantic feature for point selection")
    parser.add_argument("--other_classes_min_cluster_size", type=int, default=5,
                        help="Minimum cluster size for HDBSCAN on other_classes")
    parser.add_argument("--other_classes_sample_num", type=int, default=5000,
                        help="Number of points to sample for clustering other_classes")
    parser.add_argument("--other_classes_feature_ratio", type=float, default=0.5,
                        help="Weight of instance feature distance in hybrid distance (0-1)")
    parser.add_argument("--other_classes_spatial_ratio", type=float, default=0.3,
                        help="Weight of spatial distance in hybrid distance (0-1)")
    parser.add_argument("--other_classes_semantic_ratio", type=float, default=0.2,
                        help="Weight of semantic feature similarity in hybrid distance (0-1)")
    parser.add_argument("--prior_config", type=str, default=None,
                        help="Train-only category_priors.json (disabled unless --prior_mode is non-off)")
    parser.add_argument("--prior_mapping_config", type=str, default=None,
                        help="Validation-derived prior_mapping_config.json")
    parser.add_argument("--prior_mode", choices=[
        'off', 'global', 'size', 'smooth', 'small',
        'size-smooth', 'size-small', 'smooth-small', 'combined'
    ], default='off')
    parser.add_argument("--prior_gate", choices=['on', 'off'], default='on')
    parser.add_argument("--prior_shrink", choices=['on', 'off'], default='on')
    parser.add_argument("--prior_metadata_path", type=str, default=None,
                        help="Optional sidecar with AP scores, gates, resolved parameters and provenance")
    parser.add_argument("--minimal_metadata", action='store_true',
                        help="Omit per-artifact SHA-256 fields in locked runs")
    parser.add_argument("--max_contributor_cache_path", type=str, default=None,
                        help="Optional shared cache for config-invariant max-contributor renders")
    parser.add_argument("--scene_scale_m_per_unit", type=float, default=0.0,
                        help="Known metric conversion for Gaussian coordinates; required by category priors")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clustering-mode", choices=['legacy', 'class-first', 'legacy-prior'], default='legacy')
    parser.add_argument("--class-prior-mode", choices=[
        'uniform', 'size', 'smooth', 'small', 'combined'
    ], default='uniform')
    parser.add_argument("--category-priors", type=str, default=None)
    parser.add_argument("--class-first-config", type=str, default=None)
    parser.add_argument("--legacy-prior-config", type=str, default=None)
    parser.add_argument("--legacy-prior-mode", choices=[
        'uniform', 'size', 'smooth', 'small', 'combined'
    ], default='uniform')
    parser.add_argument("--legacy-prior-score", choices=[
        'unit', 'vote', 'assignment'
    ], default='unit')
    parser.add_argument("--legacy-prior-semantic-source", choices=[
        'gaussian', 'vote'
    ], default='gaussian')
    parser.add_argument("--teacher-prior-mode", choices=[
        'off', 'original', 'all-uniform', 'size', 'smooth', 'small', 'combined'
    ], default='original')
    parser.add_argument("--teacher-category-params", type=str, default=None)
    parser.add_argument(
        "--teacher-evidence-protection",
        choices=["off", "multi-anchor"],
        default="off",
    )
    parser.add_argument("--v3-shadow-mode", choices=['off', 'exact', 'exclusive', 'both'], default='off')
    parser.add_argument("--v3-shadow-output", type=str, default=None)
    parser.add_argument("--v3-branch-labels-output", type=str, default=None)
    parser.add_argument("--v3-shadow-git-commit", type=str, default=None)
    parser.add_argument("--v3-shadow-scene-id", type=str, default=None)
    parser.add_argument("--v4-candidate-mode", choices=[
        'off', 'uniform', 'class-scale', 'class-core', 'combined'
    ], default='off')
    parser.add_argument("--v4-candidate-output", type=str, default=None)
    parser.add_argument("--v4-candidate-labels-output", type=str, default=None)
    parser.add_argument("--v4-git-commit", type=str, default=None)
    parser.add_argument("--v4-scene-id", type=str, default=None)
    parser.add_argument("--v5-candidate-source", choices=['off', 'codebook', 'multiview'], default='off')
    parser.add_argument("--v5-candidate-output", type=str, default=None)
    parser.add_argument("--v5-candidate-labels-output", type=str, default=None)
    parser.add_argument("--v5-git-commit", type=str, default=None)
    parser.add_argument("--v5-scene-id", type=str, default=None)
    parser.add_argument("--v6-candidate-mode", choices=['off', 'affinity-first'], default='off')
    parser.add_argument("--v6-candidate-output", type=str, default=None)
    parser.add_argument("--v6-candidate-labels-output", type=str, default=None)
    parser.add_argument("--v6-git-commit", type=str, default=None)
    parser.add_argument("--v6-scene-id", type=str, default=None)
    parser.add_argument(
        "--category-denoise-action",
        choices=("off", "bank", "candidate-repair", "replay", "candidate-replay"),
        default="off",
    )
    parser.add_argument("--category-denoise-bank-path", type=str, default=None)
    parser.add_argument("--category-candidate-trace-path", type=str, default=None)
    parser.add_argument(
        "--category-candidate-sample-cap", type=int, default=5000
    )
    parser.add_argument(
        "--category-candidate-score-threshold", type=float, default=0.20
    )
    parser.add_argument(
        "--category-denoise-mode", choices=("uniform", "class"), default="uniform"
    )
    parser.add_argument("--category-denoise-scene-id", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    if args.v3_shadow_mode != 'off':
        if not args.v3_shadow_output or not args.v3_branch_labels_output:
            parser.error("V3 shadow mode requires --v3-shadow-output and --v3-branch-labels-output")
        if not args.v3_shadow_git_commit or not args.v3_shadow_scene_id:
            parser.error("V3 shadow mode requires commit and scene identifiers")
        if args.v3_shadow_mode == 'both' and (
            '{mode}' not in args.v3_shadow_output
            or '{mode}' not in args.v3_branch_labels_output
        ):
            parser.error("V3 shadow mode both requires {mode} in both output paths")
        if args.clustering_mode != 'legacy' or args.prior_mode != 'off':
            parser.error("V3 shadow audit requires the unchanged legacy main path")
        if args.scene_scale_m_per_unit <= 0:
            parser.error("V3 shadow audit requires positive --scene_scale_m_per_unit")
    if args.v4_candidate_mode != 'off':
        if not args.v4_candidate_output or not args.v4_candidate_labels_output:
            parser.error("V4 candidate mode requires candidate JSON and labels outputs")
        if not args.v4_git_commit or not args.v4_scene_id or not args.category_priors:
            parser.error("V4 candidate mode requires priors, commit, and scene identifiers")
        if args.clustering_mode != 'legacy' or args.prior_mode != 'off':
            parser.error("V4 candidate shadow requires the unchanged legacy main path")
        if args.scene_scale_m_per_unit <= 0:
            parser.error("V4 candidate shadow requires positive --scene_scale_m_per_unit")
    if args.v5_candidate_source != 'off':
        if not args.v5_candidate_output or not args.v5_candidate_labels_output:
            parser.error("V5 candidate source requires proposal JSON and labels outputs")
        if not args.v5_git_commit or not args.v5_scene_id:
            parser.error("V5 candidate source requires commit and scene identifiers")
        if args.clustering_mode != 'legacy' or args.prior_mode != 'off':
            parser.error("V5 candidate shadow requires the unchanged legacy main path")
        if args.scene_scale_m_per_unit <= 0:
            parser.error("V5 candidate shadow requires positive --scene_scale_m_per_unit")
    if args.v6_candidate_mode != 'off':
        if not args.v6_candidate_output or not args.v6_candidate_labels_output:
            parser.error("V6 candidate mode requires proposal JSON and labels outputs")
        if not args.v6_git_commit or not args.v6_scene_id:
            parser.error("V6 candidate mode requires commit and scene identifiers")
        if args.clustering_mode != 'legacy' or args.prior_mode != 'off':
            parser.error("V6 candidate bank requires the unchanged legacy main path")
        if args.scene_scale_m_per_unit <= 0:
            parser.error("V6 candidate bank requires positive --scene_scale_m_per_unit")
    if args.category_denoise_action != "off":
        if not args.category_denoise_bank_path or not args.category_denoise_scene_id:
            parser.error(
                "category denoising requires --category-denoise-bank-path and scene ID"
            )
        if not args.category_priors:
            parser.error("category denoising requires --category-priors")
        if not args.disable_other_classes:
            parser.error("category denoising requires --disable-other-classes")
        if args.prior_mode != "off" or args.clustering_mode != "legacy":
            parser.error("category denoising requires the unchanged legacy global path")
        if args.scene_scale_m_per_unit <= 0:
            parser.error("category denoising requires a positive scene scale")
        if args.category_candidate_sample_cap <= 0:
            parser.error("category candidate sample cap must be positive")
        if not 0.0 <= args.category_candidate_score_threshold <= 1.0:
            parser.error("category candidate score threshold must be in [0, 1]")
        if (
            args.category_denoise_action == "candidate-repair"
            and not args.category_candidate_trace_path
        ):
            parser.error(
                "candidate repair requires --category-candidate-trace-path"
            )
    prior_resolver = None
    if args.prior_mode != 'off':
        if not args.prior_config or not args.prior_mapping_config:
            parser.error("non-off --prior_mode requires --prior_config and --prior_mapping_config")
        if args.scene_scale_m_per_unit <= 0:
            parser.error("non-off --prior_mode requires a positive --scene_scale_m_per_unit")
        if not args.prior_metadata_path:
            args.prior_metadata_path = f"{args.json_path}.metadata.json"
        from category_priors.runtime import PriorResolver
        prior_resolver = PriorResolver.from_paths(args.prior_config, args.prior_mapping_config)
        tuned_baseline = prior_resolver.mapping['baseline']
        args.feature_ratio = float(tuned_baseline['feature_ratio'])
        args.instance_threshold = float(tuned_baseline['instance_threshold'])
        args.min_cluster_size = int(tuned_baseline['min_cluster_size'])
        args.k = int(tuned_baseline['knn_k'])
        args.sample_num = int(tuned_baseline['sample_num'])
    if args.clustering_mode == 'class-first':
        if not args.category_priors or not args.class_first_config:
            parser.error(
                "--clustering-mode class-first requires --category-priors and --class-first-config"
            )
        if args.scene_scale_m_per_unit <= 0:
            parser.error(
                "--clustering-mode class-first requires positive --scene_scale_m_per_unit"
            )
        if not args.prior_metadata_path:
            args.prior_metadata_path = f"{args.json_path}.metadata.json"
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        run_class_first_postprocess(args)
        return
    legacy_prior = None
    if args.clustering_mode == 'legacy-prior':
        if not args.category_priors or not args.legacy_prior_config:
            parser.error(
                "--clustering-mode legacy-prior requires --category-priors and --legacy-prior-config"
            )
        if args.scene_scale_m_per_unit <= 0:
            parser.error(
                "--clustering-mode legacy-prior requires positive --scene-scale-m-per-unit"
            )
        from category_priors.io import load_json
        from category_priors.legacy_prior import load_legacy_prior_config
        legacy_prior = {
            "priors": load_json(args.category_priors),
            "config": load_legacy_prior_config(args.legacy_prior_config),
            "mode": args.legacy_prior_mode,
        }
        if not args.prior_metadata_path:
            args.prior_metadata_path = f"{args.json_path}.metadata.json"
    teacher_prior = None
    if args.teacher_prior_mode in {'all-uniform', 'size', 'smooth', 'small', 'combined'}:
        if args.clustering_mode != 'legacy' or args.prior_mode != 'off':
            parser.error(
                "data-driven --teacher-prior-mode requires the unchanged legacy main path"
            )
        if not args.teacher_category_params:
            parser.error(
                "data-driven --teacher-prior-mode requires --teacher-category-params"
            )
        if args.scene_scale_m_per_unit <= 0:
            parser.error(
                "data-driven --teacher-prior-mode requires positive --scene_scale_m_per_unit"
            )
        from category_priors.teacher_prior import load_teacher_category_params
        teacher_prior = {
            "table": load_teacher_category_params(args.teacher_category_params),
            "mode": args.teacher_prior_mode,
        }
        if args.teacher_evidence_protection != 'off' and (
            teacher_prior["table"].get("branch_preservation", False)
            or teacher_prior["table"].get("restore_after_global_filter", False)
        ):
            parser.error(
                "multi-anchor evidence protection requires the original merge table "
                "(branch_preservation=false, restore_after_global_filter=false)"
            )
    elif args.teacher_evidence_protection != 'off':
        parser.error("teacher evidence protection requires a data teacher-prior mode")
    bg_color = torch.tensor([1,1,1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # sam = sam_model_registry['vit_h']('./third_party/segment-anything/weights/sam_vit_h_4b8939.pth').to('cuda')
    # mask_predictor = SamPredictor(sam)
    # clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to('cuda')
    # clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    gs_model = GaussianModel(args.sh_degree)
    gs_model.load_ply(args.point_cloud_path)
    feat_gs_model = FeatureGaussianModel(args.feature_dim, args.semantic_feature_dim)
    feat_gs_model.load_ply(args.contrastive_feature_point_cloud_path)
    scale_gate = torch.nn.Sequential(
            torch.nn.Linear(1, args.feature_dim, bias=True),
            torch.nn.Sigmoid()
        ).cuda()
    scale_gate.load_state_dict(torch.load(args.scale_gate_path))
    try:
        cameras = readColmapCameras(read_extrinsics_binary(os.path.join(args.sparse_path, 'images.bin')), 
                                    read_intrinsics_binary(os.path.join(args.sparse_path, 'cameras.bin')), 
                                    args.images_path)
    except:
        cameras = readColmapCameras(read_extrinsics_text(os.path.join(args.sparse_path, 'images.txt')), 
                                    read_intrinsics_text(os.path.join(args.sparse_path, 'cameras.txt')), 
                                    args.images_path)
    camera_list = cameraList_from_camInfos(cameras, 1, args)

    max_contributor_cache_dir = None
    max_contributor_memory = {}
    max_contributor_cache_hits = 0
    max_contributor_cache_misses = 0
    if args.max_contributor_cache_path:
        point_cloud_stat = os.stat(args.point_cloud_path)
        cache_identity = {
            "format": "saga-max-contributor-v1",
            "point_cloud": {
                "path": os.path.abspath(args.point_cloud_path),
                "size": point_cloud_stat.st_size,
                "mtime_ns": point_cloud_stat.st_mtime_ns,
                "gaussians": int(gs_model.get_xyz.shape[0]),
            },
            "cameras": [
                {
                    "image_name": camera.image_name,
                    "width": int(camera.image_width),
                    "height": int(camera.image_height),
                    "world_view_transform": camera.world_view_transform.detach().cpu().tolist(),
                    "full_proj_transform": camera.full_proj_transform.detach().cpu().tolist(),
                }
                for camera in camera_list
            ],
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        max_contributor_cache_dir = os.path.join(args.max_contributor_cache_path, cache_key)
        os.makedirs(max_contributor_cache_dir, exist_ok=True)
        manifest_path = os.path.join(max_contributor_cache_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            temporary_manifest = f"{manifest_path}.tmp-{os.getpid()}"
            with open(temporary_manifest, "w", encoding="utf-8") as handle:
                json.dump(cache_identity, handle, sort_keys=True)
            os.replace(temporary_manifest, manifest_path)
        print(f"max-contributor cache: {max_contributor_cache_dir}")

    def get_max_contributor(camera, device):
        nonlocal max_contributor_cache_hits, max_contributor_cache_misses
        memory_key = camera.image_name
        if memory_key in max_contributor_memory:
            max_contributor_cache_hits += 1
            return max_contributor_memory[memory_key].to(device)
        cache_file = None
        max_contributor = None
        if max_contributor_cache_dir is not None:
            camera_key = hashlib.sha256(camera.image_name.encode("utf-8")).hexdigest()
            cache_file = os.path.join(max_contributor_cache_dir, f"{camera_key}.pt")
            if os.path.isfile(cache_file):
                try:
                    candidate = torch.load(
                        cache_file, map_location="cpu", weights_only=True
                    )
                    if (
                        isinstance(candidate, torch.Tensor)
                        and candidate.shape == (camera.image_height, camera.image_width)
                        and candidate.numel() > 0
                        and int(candidate.min()) >= -1
                        and int(candidate.max()) < int(gs_model.get_xyz.shape[0])
                    ):
                        max_contributor = candidate.long().contiguous()
                        max_contributor_cache_hits += 1
                except (OSError, RuntimeError, ValueError):
                    max_contributor = None
        if max_contributor is None:
            render_pkg = render_with_max_contributor(camera, gs_model, args, bg_color)
            max_contributor = render_pkg['max_contributor'].detach().cpu().long().contiguous()
            max_contributor_cache_misses += 1
            if cache_file is not None:
                temporary_cache = f"{cache_file}.tmp-{os.getpid()}"
                torch.save(max_contributor, temporary_cache)
                os.replace(temporary_cache, cache_file)
        max_contributor_memory[memory_key] = max_contributor
        return max_contributor.to(device)

    point_features = feat_gs_model.get_point_features.detach().cpu()
    point_semantic_features = feat_gs_model.get_point_semantic_features.detach().cpu()
    point_xyz = feat_gs_model.get_xyz.detach().cpu()
    if args.category_denoise_action != "off":
        rgb_xyz = gs_model.get_xyz.detach().cpu()
        if rgb_xyz.shape != point_xyz.shape or not torch.allclose(
            rgb_xyz, point_xyz, rtol=0.0, atol=1e-6
        ):
            raise ValueError(
                "category denoising requires RGB and feature Gaussians in identical order"
            )
    point_scales = feat_gs_model.get_scaling.detach().cpu()
    is_big_gaussian = point_scales.max(dim=-1).values>point_scales.max(dim=-1).values.median()*args.scale_threshold
    point_opacities = feat_gs_model.get_opacity.detach().cpu().squeeze()
    is_transparent_gaissian = point_opacities<args.opcity_threshold
    gates = scale_gate(torch.tensor([args.scale]).cuda()).unsqueeze(0).detach().cpu()
    print(f'{point_features.shape=}, {point_xyz.shape=}')

    # Load label features for semantic-guided clustering of other_classes
    label_features_path = args.label_features_path
    if os.path.exists(label_features_path):
        label_features = torch.load(label_features_path)
        # Create dictionary mapping class names to feature indices
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(args.classes)}
        print(f"Loaded label features: {label_features.shape}, num_classes: {len(args.classes)}")
    else:
        label_features = None
        class_to_idx = None
        print("Warning: label_features.pt not found, semantic-guided clustering disabled")

    legacy_mask_scales = []
    legacy_surface_density = 1.0
    if legacy_prior is not None or args.v4_candidate_mode != 'off':
        from category_priors.legacy_prior import estimate_surface_density
        if os.path.isdir(args.mask_scales_path):
            for filename in sorted(os.listdir(args.mask_scales_path)):
                if not filename.endswith('.pt'):
                    continue
                values = torch.load(
                    os.path.join(args.mask_scales_path, filename), map_location='cpu'
                )
                if isinstance(values, torch.Tensor):
                    legacy_mask_scales.extend(values.detach().cpu().flatten().tolist())
        legacy_surface_density = estimate_surface_density(
            point_xyz.detach().cpu().numpy() * float(args.scene_scale_m_per_unit)
        )

    sampled_mask = uniform_sample(point_xyz, args.sample_num)
    # sampled_mask = torch.rand(point_features.shape[0]) > 0.99

    scale_conditioned_point_features = F.normalize(point_features, dim = -1, p = 2) * gates
    normed_point_features = F.normalize(scale_conditioned_point_features, dim = -1, p = 2)
    # Normalize semantic features
    normed_point_semantic_features = F.normalize(point_semantic_features, dim = -1, p = 2)
    sampled_normed_point_features = normed_point_features[sampled_mask]

    min_val = torch.min(point_xyz, dim=0).values
    max_val = torch.max(point_xyz, dim=0).values
    new_min = 0.0
    new_max = 1.0
    std_point_xyz = (point_xyz - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
    sampled_std_point_xyz = std_point_xyz[sampled_mask]

    hybird_point_features = torch.cat((normed_point_features, std_point_xyz), dim=1)
    sampled_hybird_point_features = torch.cat((sampled_normed_point_features, sampled_std_point_xyz), dim=1)

    sampled_normed_point_features_distance = torch.clamp(1-torch.einsum('ac,bc -> ab', sampled_normed_point_features, sampled_normed_point_features), 0)
    sampled_std_point_xyz_distance = torch.clamp(torch.norm(sampled_std_point_xyz[:,None,:] - sampled_std_point_xyz[None,:,:], dim=-1), 0)
    hybird_distance = args.feature_ratio*sampled_normed_point_features_distance + (1-args.feature_ratio)*sampled_std_point_xyz_distance

    # def hybird_distance(u, v):
    #     u_feature, u_xyz = u[:args.feature_dim], u[args.feature_dim:]
    #     v_feature, v_xyz = v[:args.feature_dim], v[args.feature_dim:]

    #     feature_distance = 1-np.dot(u_feature, v_feature)
    #     xyz_distance = np.linalg.norm(u_xyz-v_xyz)

    #     return 0.2*feature_distance+0.8*xyz_distance

    clusterer = HDBSCAN(min_cluster_size=args.min_cluster_size, cluster_selection_epsilon=0.01, allow_single_cluster = False, metric='precomputed') # HDBSCAN

    start_time = datetime.now()
    cluster_labels = clusterer.fit_predict(hybird_distance.numpy().astype(np.float64))
    end_time = datetime.now()
    elapsed_time = end_time - start_time

    sampled_indices_for_trace = torch.nonzero(
        sampled_mask, as_tuple=False
    ).flatten()
    global_sample_core_trace = torch.full(
        (len(point_xyz),), -1, dtype=torch.long
    )
    sampled_cluster_labels_for_trace = torch.as_tensor(
        cluster_labels, dtype=torch.long
    )
    sampled_core_for_trace = sampled_cluster_labels_for_trace >= 0
    global_sample_core_trace[
        sampled_indices_for_trace[sampled_core_for_trace]
    ] = sampled_cluster_labels_for_trace[sampled_core_for_trace]

    if args.v7_causal_ablation in {"L2", "L3"}:
        point_labels = torch.full((len(point_xyz),), -1, dtype=torch.long)
        point_assignment_confidence = torch.zeros(len(point_xyz), dtype=torch.float32)
        sampled_indices = torch.nonzero(sampled_mask, as_tuple=False).flatten()
        sampled_cluster_labels = torch.as_tensor(cluster_labels, dtype=torch.long)
        sampled_core = sampled_cluster_labels >= 0
        point_labels[sampled_indices[sampled_core]] = sampled_cluster_labels[sampled_core]
        point_assignment_confidence[sampled_indices[sampled_core]] = 1.0
        assigned_mask = point_labels >= 0
    else:
        feature_cluster_centers = torch.zeros(len(np.unique(cluster_labels)) - 1, point_features.shape[-1])
        xyz_cluster_centers = torch.zeros(len(np.unique(cluster_labels)) - 1, point_xyz.shape[-1])
        for i in np.unique(cluster_labels):
            if i<0:
                continue
            feature_cluster_centers[i] = F.normalize(sampled_normed_point_features[cluster_labels == i].mean(dim = 0), dim = -1)
            xyz_cluster_centers[i] = sampled_std_point_xyz[cluster_labels == i].mean(dim = 0)

        normed_point_features_sim = torch.clamp(torch.einsum('ac,bc->ab', normed_point_features, feature_cluster_centers), -1, 1)
        std_point_xyz_sim = torch.clamp(torch.exp(-torch.norm(std_point_xyz[:,None,:] - xyz_cluster_centers[None,:,:], dim=-1)), 0, 1)
        hybird_sim = args.feature_ratio*normed_point_features_sim + (1-args.feature_ratio)*std_point_xyz_sim
        confidence = torch.softmax(hybird_sim*10, dim=-1)
        point_assignment_confidence, point_labels = confidence.max(dim=-1)
        assigned_mask = point_assignment_confidence>args.instance_threshold
        point_labels[~assigned_mask] = -1
    print(f'{assigned_mask.sum()=}, {(~assigned_mask).sum()=}')
    fallback_point_labels = point_labels.detach().cpu().clone()
    global_full_assignment_trace = fallback_point_labels.clone()
    fallback_assignment_confidence = point_assignment_confidence.detach().cpu().clone()
    category_denoise_bank = None
    category_candidate_family = None
    category_denoise_decisions = []
    category_denoise_classes = {}
    category_denoise_scores = {}
    category_denoise_diagnostics = {}
    if args.category_denoise_action == "bank":
        from category_priors.category_denoise import build_candidate_bank
        from category_priors.teacher_prior import saga20_branch_classes

        category_denoise_bank = build_candidate_bank(
            normed_point_features,
            normed_point_semantic_features,
            point_xyz,
            F.normalize(label_features.detach().cpu(), dim=-1, p=2),
            args.classes,
            saga20_branch_classes(class_to_idx),
            fallback_point_labels,
            args.scene_scale_m_per_unit,
            seed=args.seed,
        )
        category_denoise_bank.diagnostics["scene_id"] = (
            args.category_denoise_scene_id
        )
        print(
            "category-denoise bank captured "
            f"{len(category_denoise_bank.candidates)} frozen candidates"
        )
    elif args.category_denoise_action == "candidate-repair":
        from category_priors.category_denoise import build_candidate_repair_family
        from category_priors.teacher_prior import saga20_branch_classes

        category_candidate_family = build_candidate_repair_family(
            normed_point_features,
            normed_point_semantic_features,
            point_xyz,
            F.normalize(label_features.detach().cpu(), dim=-1, p=2),
            args.classes,
            saga20_branch_classes(class_to_idx),
            fallback_point_labels,
            args.scene_scale_m_per_unit,
            scene_id=args.category_denoise_scene_id,
            seed=args.seed,
            sample_cap=args.category_candidate_sample_cap,
        )
        # The bank side path remains a read-only observer of the B0 output.
        # C0 is exposed here only so the common metadata path can report the
        # candidate count; C1/C2 never alter point_labels in this action.
        category_denoise_bank = category_candidate_family.legacy
        print(
            "category candidate repair family captured "
            + ", ".join(
                f"{name}={len(bank.candidates)}"
                for name, bank in category_candidate_family.banks.items()
            )
        )
    elif args.category_denoise_action in {"replay", "candidate-replay"}:
        from category_priors.category_denoise import load_candidate_bank

        category_denoise_bank = load_candidate_bank(args.category_denoise_bank_path)
        if category_denoise_bank.class_names != tuple(args.classes):
            raise ValueError("category-denoise bank class table does not match runtime")
        if category_denoise_bank.seed != int(args.seed):
            raise ValueError("category-denoise bank seed does not match runtime")
        if not np.array_equal(
            category_denoise_bank.global_pre_knn,
            fallback_point_labels.numpy(),
        ):
            raise ValueError(
                "category-denoise replay does not reproduce the bank global-pre-KNN labels"
            )
    prior_overlay = None
    if prior_resolver is not None:
        if label_features is None:
            raise FileNotFoundError("Category priors require label_features.pt")
        from category_priors.runtime import apply_prior_overlay
        prior_overlay = apply_prior_overlay(
            normed_point_features,
            normed_point_semantic_features,
            point_xyz,
            F.normalize(label_features.detach().cpu(), dim=-1, p=2),
            args.classes,
            fallback_point_labels,
            fallback_assignment_confidence,
            prior_resolver,
            args.prior_mode,
            args.scene_scale_m_per_unit,
            args.seed,
            args.prior_gate == 'on',
            args.prior_shrink == 'on',
        )
        point_labels = prior_overlay.labels
        point_assignment_confidence = prior_overlay.assignment_confidence
    print(f'{elapsed_time=}, {len(torch.unique(point_labels))=}') # 3
    print(f'HDBSCAN finish')

    # ========== SEMANTIC-GUIDED CLUSTERING (for other_classes) ==========
    pending_legacy_branch = None
    pending_legacy_classes = {}
    pending_teacher_branch = None
    pending_teacher_classes = {}
    teacher_preservation = None
    teacher_preserved_classes = {}
    teacher_merged_membership = torch.full(
        (len(point_xyz),), -1, dtype=torch.long
    )
    teacher_merged_classes = {}
    teacher_post_filter = {}
    teacher_after_knn = None
    teacher_restored_after_filter = 0
    teacher_protection_diagnostics = {
        "mode": args.teacher_evidence_protection,
        "candidate_branches": 0,
        "accepted_branches": 0,
        "restored_points": 0,
    }
    v3_shadow_captures = {}
    v3_semantic_top1 = None
    v3_semantic_score = None
    v3_semantic_margin = None
    v3_sam_covered = torch.zeros(len(point_xyz), dtype=torch.bool)
    v4_candidate_capture = None
    other_class_candidates_trace = torch.full(
        (len(point_xyz),), -1, dtype=torch.long
    )
    teacher_branch_preservation = bool(
        teacher_prior is not None
        and teacher_prior["table"].get("branch_preservation", False)
    )
    teacher_restore_after_global_filter = bool(
        teacher_prior is not None
        and teacher_prior["table"].get("restore_after_global_filter", False)
    )
    legacy_semantic_vote_scores = None
    if legacy_prior is not None and args.legacy_prior_semantic_source == 'vote':
        gaussian_votes = torch.zeros(
            (len(point_xyz), len(args.classes)), dtype=torch.float64
        )
        for camera in tqdm(camera_list, desc='Gaussian semantic vote'):
            mask_file = os.path.join(args.masks_path, f'{camera.image_name}.pt')
            label_file = os.path.join(args.labels_path, f'{camera.image_name}.pt')
            if not os.path.isfile(mask_file) or not os.path.isfile(label_file):
                continue
            masks = torch.load(mask_file, map_location='cpu')
            if masks.shape[-2:] != (camera.image_height, camera.image_width):
                masks = torch.nn.functional.interpolate(
                    masks.float().unsqueeze(1), mode='bilinear',
                    size=(camera.image_height, camera.image_width),
                    align_corners=False,
                ).squeeze(1) > 0.5
            else:
                masks = masks.bool()
            labels_2d = torch.load(label_file, map_location='cpu')
            contributors = get_max_contributor(camera, torch.device('cpu'))
            for label_2d, mask_2d in zip(labels_2d, masks):
                class_index = int(label_2d)
                if class_index < 0 or class_index >= len(args.classes):
                    continue
                selected_contributors = contributors[mask_2d].long()
                selected_contributors = selected_contributors[
                    (selected_contributors >= 0)
                    & (selected_contributors < len(point_xyz))
                ]
                if selected_contributors.numel():
                    gaussian_votes[:, class_index] += torch.bincount(
                        selected_contributors, minlength=len(point_xyz)
                    ).double()
        vote_sum = gaussian_votes.sum(dim=1, keepdim=True)
        legacy_semantic_vote_scores = torch.where(
            vote_sum > 0, gaussian_votes / vote_sum.clamp_min(1), gaussian_votes
        ).float()
    if (
        prior_resolver is None
        and args.teacher_prior_mode != 'off'
        and not args.disable_other_classes
        and label_features is not None
        and class_to_idx is not None
        and len(args.other_classes) > 0
    ):
        print(f"\n{'='*60}")
        print(f"Starting semantic-guided clustering for other_classes")
        branch_classes = args.other_classes
        if legacy_prior is not None:
            supported = set(legacy_prior['priors'].get('categories', {}))
            branch_classes = [
                name for name in args.selected_classes if name in supported
            ]
        elif teacher_prior is not None:
            from category_priors.teacher_prior import saga20_branch_classes
            branch_classes = list(saga20_branch_classes(class_to_idx))
        teacher_exclusive_masks = None
        normalized_label_features = label_features
        if teacher_prior is not None:
            normalized_label_features = F.normalize(
                label_features.detach().cpu(), dim=-1, p=2
            )
            if args.teacher_evidence_protection == 'multi-anchor':
                from category_priors.v3_shadow import target_top1_masks
                teacher_exclusive_masks, _, _, _ = target_top1_masks(
                    normed_point_semantic_features.numpy(),
                    normalized_label_features.numpy(),
                    [class_to_idx[name] for name in branch_classes],
                    threshold=0.7,
                )
            elif teacher_branch_preservation:
                from category_priors.teacher_prior import exclusive_top1_masks
                teacher_exclusive_masks = exclusive_top1_masks(
                    normed_point_semantic_features.numpy(),
                    normalized_label_features.numpy(),
                    [class_to_idx[name] for name in branch_classes],
                    threshold=0.7,
                )
        print(f"Classes: {branch_classes}")
        print(f"Feature ratio: {args.other_classes_feature_ratio}, Spatial ratio: {args.other_classes_spatial_ratio}, Semantic ratio: {args.other_classes_semantic_ratio}")
        print(f"{'='*60}")

        other_point_labels, other_instance_to_class = cluster_other_classes(
            normed_point_features.clone(),  # Instance features for clustering
            normed_point_semantic_features.clone(),  # Semantic features for class filtering
            point_xyz.clone(),
            F.normalize(label_features.detach().cpu(), dim=-1, p=2)
            if legacy_prior is not None else normalized_label_features,
            class_to_idx,
            branch_classes,
            args,
            device='cpu',
            legacy_prior=legacy_prior,
            raw_point_features=point_features,
            scale_gate=scale_gate,
            mask_scales=legacy_mask_scales,
            surface_density=legacy_surface_density,
            semantic_vote_scores=legacy_semantic_vote_scores,
            teacher_prior=teacher_prior,
            exclusive_masks=teacher_exclusive_masks,
        )
        other_class_candidates_trace = other_point_labels.detach().cpu().clone()

        # ========== MERGE other_class instances into main labels (BEFORE filters) ==========
        if len(other_instance_to_class) > 0:
            if legacy_prior is not None:
                pending_legacy_branch = other_point_labels
                pending_legacy_classes = other_instance_to_class
                print(
                    "Deferred legacy-prior proposals until after the global KNN; "
                    "small proposals will not be voted away by global smoothing"
                )
            elif teacher_branch_preservation:
                pending_teacher_branch = other_point_labels
                pending_teacher_classes = other_instance_to_class
                print(
                    f"Deferred {len(pending_teacher_classes)} teacher-prior "
                    "instances until after legacy global filtering"
                )
            else:
            # Get max instance ID from main clustering (excluding -1 background)
                max_main_instance_id = point_labels.max().item() if point_labels.max() >= 0 else -1

            # Merge assigned instances (>= 0)
                for other_instance_id in other_instance_to_class.keys():
                    new_instance_id = max_main_instance_id + 1 + other_instance_id
                    mask = (other_point_labels == other_instance_id)
                    point_labels[mask] = new_instance_id
                    if bool(mask.any()):
                        teacher_merged_membership[mask] = new_instance_id
                        teacher_merged_classes[new_instance_id] = (
                            other_instance_to_class[other_instance_id]
                        )

                print(f"Merged {len(other_instance_to_class)} other_class instances into main labels")
                print(f"Total instances before filters: {len(torch.unique(point_labels))}")

        print(f"{'='*60}\n")

    if args.v3_shadow_mode != 'off':
        from category_priors.teacher_prior import saga20_branch_classes
        from category_priors.v3_shadow import target_top1_masks

        shadow_classes = list(saga20_branch_classes(class_to_idx))
        normalized_label_features = F.normalize(
            label_features.detach().cpu(), dim=-1, p=2
        )
        shadow_masks, v3_semantic_top1, v3_semantic_score, v3_semantic_margin = (
            target_top1_masks(
                normed_point_semantic_features.numpy(),
                normalized_label_features.numpy(),
                [class_to_idx[name] for name in shadow_classes],
                threshold=args.other_classes_similarity_threshold,
            )
        )
        shadow_modes = (
            ('exact', 'exclusive')
            if args.v3_shadow_mode == 'both' else (args.v3_shadow_mode,)
        )
        rng_state = torch.random.get_rng_state()
        for shadow_mode in shadow_modes:
            try:
                torch.manual_seed(args.seed)
                shadow_branch, shadow_instance_classes = cluster_other_classes(
                    normed_point_features.clone(),
                    normed_point_semantic_features.clone(),
                    point_xyz.clone(),
                    normalized_label_features,
                    class_to_idx,
                    shadow_classes,
                    args,
                    device='cpu',
                    exclusive_masks=(
                        None if shadow_mode == 'exact' else shadow_masks
                    ),
                    diagnostics_attribute=f'_v3_shadow_diagnostics_{shadow_mode}',
                )
            finally:
                torch.random.set_rng_state(rng_state)
            v3_shadow_captures[shadow_mode] = {
                'branch_labels': shadow_branch,
                'classes': shadow_instance_classes,
            }
            print(
                f"V3 {shadow_mode} shadow captured "
                f"{len(shadow_instance_classes)} candidates without modifying legacy labels"
            )

    if args.v4_candidate_mode != 'off':
        from category_priors.io import load_json
        from category_priors.teacher_prior import saga20_branch_classes
        from category_priors.v3_shadow import target_top1_masks
        from category_priors.v4_candidate import V4CandidateConfig

        v4_classes = list(saga20_branch_classes(class_to_idx))
        v4_label_features = F.normalize(label_features.detach().cpu(), dim=-1, p=2)
        v4_masks, v4_top1, v4_score, v4_margin = target_top1_masks(
            normed_point_semantic_features.numpy(),
            v4_label_features.numpy(),
            [class_to_idx[name] for name in v4_classes],
            threshold=args.other_classes_similarity_threshold,
        )
        v4_branch, v4_instance_classes = cluster_other_classes(
            normed_point_features.clone(), normed_point_semantic_features.clone(),
            point_xyz.clone(), v4_label_features, class_to_idx, v4_classes, args,
            device='cpu', raw_point_features=point_features, scale_gate=scale_gate,
            mask_scales=legacy_mask_scales, surface_density=legacy_surface_density,
            exclusive_masks=v4_masks, diagnostics_attribute='_v4_candidate_diagnostics',
            v4_candidate={
                'priors': load_json(args.category_priors),
                'mode': args.v4_candidate_mode,
                'config': V4CandidateConfig(),
            },
        )
        v4_candidate_capture = {
            'branch_labels': v4_branch,
            'classes': v4_instance_classes,
            'semantic_top1': v4_top1,
            'semantic_score': v4_score,
            'semantic_margin': v4_margin,
        }
        print(
            f"V4 {args.v4_candidate_mode} captured {len(v4_instance_classes)} "
            "shadow candidates without modifying B1 labels"
        )

    def filter3d(pos, label, k):
        print('begin filter3d')
        assert pos.shape[0] == label.shape[0]
        pos=pos.detach().cpu().numpy()
        label=label.detach().cpu().numpy()
        new_label = []
        kdtree = KDTree(pos)
        for i,p in enumerate(pos):
            d, index = kdtree.query(x=p, k=k)
            # assert i == index[0]
            # print(f'query index {index[1:]} for {index[0]}')
            # print(f'query label {label[index[1:]].tolist()} for {label[index[0]].tolist()}')
            # index = index[1:]
            bin = []
            counts = []
            for l in label[index]:
                try:
                    counts[bin.index(l)]+=1
                except:
                    bin.append(l)
                    counts.append(1)
            # print(f'{bin}\n{counts}')
            new_label.append(bin[counts.index(max(counts))])
        print('finish filter3d')
        return torch.tensor(new_label).cuda()
    def filter_num(point_labels, min_num=10):
        print('begin filter_num')
        point_labels = point_labels.detach().cpu().numpy()
        unique, counts = np.unique(point_labels, return_counts=True)
        count_dict = dict(zip(unique, counts))
        new_label = point_labels.copy()
        for instance, count in count_dict.items():
            if instance==-1:
                continue
            if count<min_num:
                new_label[point_labels==instance] = -1
        print('finish filter_num')
        return torch.tensor(new_label)
    def compute_instance_ratios(labels_for_vote, update_progress=True):
        instance_ids = [int(value) for value in torch.unique(labels_for_vote).tolist() if int(value) >= 0]
        if not instance_ids:
            return {}
        maximum_instance_id = max(instance_ids)
        vote = np.zeros(
            (maximum_instance_id + 1, len(args.classes) + 1), dtype=np.int64
        )
        for i, camera in tqdm(list(enumerate(camera_list))):
            if update_progress:
                with open(args.progress_path, 'w') as f:
                    f.write(str(0+(i+1)*100//len(camera_list)))
            if not os.path.exists(os.path.join(args.masks_path, f'{camera.image_name}.pt')):
                continue
            masks = torch.load(os.path.join(args.masks_path, f'{camera.image_name}.pt'))
            if masks.shape[-2:] != (camera.image_height, camera.image_width):
                masks = torch.nn.functional.interpolate(
                    masks.float().unsqueeze(1), mode='bilinear',
                    size=(camera.image_height, camera.image_width),
                    align_corners=False,
                ).squeeze(1) > 0.5
            else:
                masks = masks.bool()
            labels_2d = torch.load(os.path.join(args.labels_path, f'{camera.image_name}.pt'))
            max_contributor = get_max_contributor(camera, labels_for_vote.device)
            valid_contributor = (
                (max_contributor >= 0) & (max_contributor < len(labels_for_vote))
            )
            if (args.v3_shadow_mode != 'off' or args.v4_candidate_mode != 'off') and masks.numel() > 0:
                foreground = masks.any(dim=0)
                covered_indices = max_contributor[foreground & valid_contributor].detach().cpu().long()
                if covered_indices.numel():
                    v3_sam_covered[covered_indices.unique()] = True
            safe_contributor = max_contributor.clamp(min=0)
            max_instance_contributor = labels_for_vote[safe_contributor]
            max_instance_contributor = max_instance_contributor.clone()
            max_instance_contributor[~valid_contributor] = -1
            background_label = len(args.classes)
            background = torch.ones(
                (camera.image_height, camera.image_width),
                dtype=torch.bool,
                device=masks.device,
            )
            for label_2d, mask_2d in zip(labels_2d, masks):
                background &= ~mask_2d
                vote_for_label = max_instance_contributor[mask_2d]
                label_index = int(label_2d)
                if label_index < 0 or label_index >= len(args.classes):
                    continue
                valid_instances = vote_for_label[
                    (vote_for_label >= 0) & (vote_for_label <= maximum_instance_id)
                ].long()
                if valid_instances.numel():
                    vote[:, label_index] += torch.bincount(
                        valid_instances, minlength=maximum_instance_id + 1
                    ).cpu().numpy()
            vote_for_background_label = max_instance_contributor[background]
            valid_background = vote_for_background_label[
                (vote_for_background_label >= 0)
                & (vote_for_background_label <= maximum_instance_id)
            ].long()
            if valid_background.numel():
                vote[:, background_label] += torch.bincount(
                    valid_background, minlength=maximum_instance_id + 1
                ).cpu().numpy()

        ratios = {}
        for instance in instance_ids:
            votes = vote[instance].astype(np.float64, copy=False)
            denominator = votes.sum()
            ratios[instance] = votes[:-1] / denominator if denominator > 0 else np.zeros(len(args.classes), dtype=np.float64)
        return ratios

    merged_partition_trace = point_labels.detach().cpu().clone()
    post_filter_trace = None
    post_attach_trace = None
    start_time = datetime.now()
    v3_shadow_stages = {}
    if prior_overlay is not None:
        from category_priors.runtime import filter_small_clusters, smooth_labels, validate_overlay
        preliminary_ratio = compute_instance_ratios(point_labels, update_progress=False)
        point_labels, point_assignment_confidence, rejected_prior_instances = validate_overlay(
            prior_overlay,
            preliminary_ratio,
            args.classes,
            args.label_threshold,
        )
        print(f'prior vote validation rejected {len(rejected_prior_instances)} instances')
        semantic_similarity_by_instance = {}
        normalized_label_features = F.normalize(label_features.detach().cpu(), dim=-1, p=2)
        for instance_id, ratio in preliminary_ratio.items():
            if instance_id < 0 or not bool((point_labels == instance_id).any()) or (ratio.max() if ratio.size else 0.0) <= 0:
                continue
            class_idx = int(np.argmax(ratio))
            instance_semantic = normed_point_semantic_features[point_labels == instance_id]
            if len(instance_semantic) == 0:
                continue
            mean_semantic = F.normalize(instance_semantic.mean(dim=0), dim=-1, p=2)
            semantic_similarity_by_instance[int(instance_id)] = float(torch.dot(mean_semantic, normalized_label_features[class_idx]))
        point_labels = smooth_labels(
            point_xyz,
            point_labels,
            preliminary_ratio,
            args.classes,
            prior_resolver,
            args.prior_mode,
            args.scene_scale_m_per_unit,
            float(prior_overlay.diagnostics['surface_density_points_per_m2']),
            args.prior_gate == 'on',
            args.prior_shrink == 'on',
            semantic_similarity_by_instance,
        )
        point_labels = filter_small_clusters(point_labels, prior_overlay.branch_instances, default_min=args.min_cluster_size)
        post_filter_trace = point_labels.detach().cpu().clone()
        post_attach_trace = post_filter_trace.clone()
    else:
        for shadow_mode, capture in v3_shadow_captures.items():
            shadow_branch = capture['branch_labels']
            shadow_classes = capture['classes']
            shadow_global_pre = point_labels.detach().cpu().clone()
            shadow_filter_input = point_labels.detach().cpu().clone()
            next_shadow_id = (
                int(shadow_filter_input.max()) + 1
                if bool((shadow_filter_input >= 0).any()) else 0
            )
            merged_ids = {}
            for shadow_id in sorted(shadow_classes):
                mask = shadow_branch == int(shadow_id)
                if not bool(mask.any()):
                    continue
                shadow_filter_input[mask] = next_shadow_id
                merged_ids[int(shadow_id)] = next_shadow_id
                next_shadow_id += 1
            if args.k > 0:
                shadow_after_knn = filter3d(
                    point_xyz, shadow_filter_input, args.k
                ).detach().cpu()
            else:
                shadow_after_knn = shadow_filter_input
            shadow_after_filter = filter_num(
                shadow_after_knn, min_num=10
            ).detach().cpu()
            v3_shadow_stages[shadow_mode] = {
                'global_pre': shadow_global_pre,
                'after_knn': shadow_after_knn,
                'after_filter': shadow_after_filter,
                'merged_ids': merged_ids,
            }
        if args.category_denoise_action == "candidate-replay":
            from category_priors.category_candidate_legacy_replay import (
                LegacyReplayCandidate,
                replay_candidates_through_legacy,
            )
            from category_priors.category_candidate_prior_v2 import (
                score_candidate_prior_v2,
            )
            from category_priors.io import load_json

            category_denoise_decisions = list(
                score_candidate_prior_v2(
                    category_denoise_bank.candidates,
                    load_json(args.category_priors),
                    args.category_denoise_mode,
                )
            )
            threshold = float(args.category_candidate_score_threshold)
            for decision in category_denoise_decisions:
                decision["accepted"] = bool(
                    decision["support_pass"]
                    and float(decision["S"]) >= threshold
                )
                decision["ap_score"] = float(decision["Q"])
            replay_candidates = []
            for candidate in category_denoise_bank.candidates:
                candidate_id = int(candidate["candidate_id"])
                replay_candidates.append(
                    LegacyReplayCandidate(
                        candidate_id=candidate_id,
                        branch_class=str(candidate["branch_class"]),
                        q_score=float(candidate["base_score"]),
                        full_point_indices=np.flatnonzero(
                            category_denoise_bank.branch_full_labels == candidate_id
                        ),
                        trusted_core_indices=np.flatnonzero(
                            category_denoise_bank.branch_core_labels == candidate_id
                        ),
                    )
                )
            accepted_ids = [
                int(row["candidate_id"])
                for row in category_denoise_decisions
                if bool(row["accepted"])
            ]
            replay_result = replay_candidates_through_legacy(
                xyz_scene=point_xyz.numpy(),
                global_pre_knn=category_denoise_bank.global_pre_knn,
                candidates=replay_candidates,
                accepted_candidate_ids=accepted_ids,
                k=args.k,
                min_count=10,
            )
            point_labels = torch.from_numpy(
                np.asarray(replay_result.after_filter, dtype=np.int64)
            )
            teacher_after_knn = torch.from_numpy(
                np.asarray(replay_result.after_knn, dtype=np.int64)
            )
            post_filter_trace = point_labels.detach().cpu().clone()
            category_denoise_classes = dict(
                replay_result.candidate_class_by_raw_label
            )
            category_denoise_scores = dict(
                replay_result.candidate_score_by_raw_label
            )
            category_denoise_diagnostics = {
                **dict(replay_result.diagnostics),
                "mode": args.category_denoise_mode,
                "score_threshold": threshold,
                "candidate_count": len(category_denoise_decisions),
                "accepted_candidate_count": len(accepted_ids),
                "decisions": category_denoise_decisions,
                "candidate_survival": [
                    row.to_dict() for row in replay_result.candidates
                ],
            }
        elif args.category_denoise_action == "replay":
            from category_priors.category_denoise import (
                replay_protected_denoise,
                score_bank_candidates,
            )
            from category_priors.io import load_json

            category_denoise_decisions = score_bank_candidates(
                category_denoise_bank,
                load_json(args.category_priors),
                args.category_denoise_mode,
            )
            replayed, category_denoise_classes, category_denoise_scores, replay_diag = (
                replay_protected_denoise(
                    point_xyz.numpy(),
                    category_denoise_bank,
                    category_denoise_decisions,
                    k=args.k,
                    min_count=10,
                )
            )
            point_labels = torch.from_numpy(np.asarray(replayed, dtype=np.int64))
            teacher_after_knn = point_labels.detach().cpu().clone()
            post_filter_trace = point_labels.detach().cpu().clone()
            category_denoise_diagnostics = {
                **replay_diag,
                "mode": args.category_denoise_mode,
                "candidate_count": len(category_denoise_decisions),
                "accepted_candidate_count": int(
                    sum(bool(row["accepted"]) for row in category_denoise_decisions)
                ),
                "decisions": category_denoise_decisions,
            }
        else:
            if args.k>0 and args.v7_causal_ablation == "L0":
                point_labels = filter3d(point_xyz, point_labels, args.k)
            teacher_after_knn = point_labels.detach().cpu().clone()
            point_labels = filter_num(point_labels, min_num=10)
            # This snapshot belongs exactly to filter_num.  Later local attachment
            # and branch-preservation operations must not overwrite its meaning.
            post_filter_trace = point_labels.detach().cpu().clone()
        if args.v7_causal_ablation == "L3":
            from category_priors.v7_objects import attach_local_labels
            point_labels = torch.from_numpy(attach_local_labels(
                point_xyz.detach().cpu().numpy() * float(args.scene_scale_m_per_unit),
                normed_point_features.detach().cpu().numpy(),
                point_labels.detach().cpu().numpy(),
            ))
        post_attach_trace = point_labels.detach().cpu().clone()
        if (
            args.teacher_evidence_protection == 'multi-anchor'
            and teacher_merged_classes
        ):
            from category_priors.teacher_prior import (
                protect_multi_anchor_halo,
                resolve_teacher_parameters,
            )
            preliminary_teacher_ratio = compute_instance_ratios(
                point_labels, update_progress=False
            )
            branch_parameters = {
                int(branch_id): resolve_teacher_parameters(
                    teacher_prior["table"], branch_class, teacher_prior["mode"]
                )
                for branch_id, branch_class in teacher_merged_classes.items()
            }
            protected_labels, teacher_protection_diagnostics = (
                protect_multi_anchor_halo(
                    point_labels.detach().cpu().numpy(),
                    teacher_merged_membership.numpy(),
                    point_xyz.detach().cpu().numpy()
                    * float(args.scene_scale_m_per_unit),
                    teacher_merged_classes,
                    branch_parameters,
                    preliminary_teacher_ratio,
                    class_to_idx,
                    args.label_threshold,
                )
            )
            point_labels = torch.from_numpy(protected_labels)
        if teacher_restore_after_global_filter and teacher_merged_classes:
            from category_priors.teacher_prior import restore_surviving_branches
            restored_labels, teacher_restored_after_filter = (
                restore_surviving_branches(
                    point_labels.numpy(), teacher_merged_membership.numpy()
                )
            )
            point_labels = torch.from_numpy(restored_labels)
        if pending_legacy_branch is not None:
            max_main_instance_id = (
                point_labels.max().item() if point_labels.max() >= 0 else -1
            )
            for branch_id in pending_legacy_classes:
                mask = (pending_legacy_branch == branch_id) & (point_labels < 0)
                if int(mask.sum()) < 3:
                    continue
                point_labels[mask] = max_main_instance_id + 1 + int(branch_id)
        if pending_teacher_branch is not None:
            next_instance_id = (
                int(point_labels.max()) + 1 if bool((point_labels >= 0).any()) else 0
            )
            ordered_branches = sorted(
                pending_teacher_classes.items(), key=lambda item: (item[1], item[0])
            )
            for branch_id, branch_class in ordered_branches:
                mask = pending_teacher_branch == int(branch_id)
                if not bool(mask.any()):
                    continue
                point_labels[mask] = next_instance_id
                teacher_merged_membership[mask] = next_instance_id
                teacher_merged_classes[next_instance_id] = branch_class
                teacher_preserved_classes[next_instance_id] = branch_class
                next_instance_id += 1
            teacher_preservation = teacher_merged_membership.clone()
            teacher_after_knn = point_labels.detach().cpu().clone()
        def teacher_stage_survival(labels):
            labels = labels.detach().cpu()
            retained_points = 0
            survived_instances = 0
            for merged_id in teacher_merged_classes:
                retained = (
                    (teacher_merged_membership == merged_id)
                    & (labels == merged_id)
                )
                retained_count = int(retained.sum())
                retained_points += retained_count
                survived_instances += int(retained_count > 0)
            return survived_instances, retained_points

        merged_points = int((teacher_merged_membership >= 0).sum())
        after_knn_instances, after_knn_points = teacher_stage_survival(
            teacher_after_knn
        )
        survived_instances, retained_points = teacher_stage_survival(point_labels)
        teacher_post_filter = {
            "merged_instances": len(teacher_merged_classes),
            "merged_points": merged_points,
            "after_knn_survived_instances": after_knn_instances,
            "after_knn_retained_points": after_knn_points,
            "after_knn_point_survival_rate": (
                after_knn_points / merged_points if merged_points else None
            ),
            "restored_surviving_instances": teacher_restored_after_filter,
            "survived_instances": survived_instances,
            "retained_points": retained_points,
            "point_survival_rate": (
                retained_points / merged_points if merged_points else None
            ),
        }
    if post_filter_trace is None:
        post_filter_trace = point_labels.detach().cpu().clone()
    if post_attach_trace is None:
        post_attach_trace = post_filter_trace.clone()
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f'{elapsed_time=}, {len(torch.unique(point_labels))=}') # 89, pytorch3d.ops.knn_points=49
    print(f'knn finish')
    # torch.save(point_labels, os.path.join(args.model_path, 'point_labels.pth'))
    # point_labels = torch.load(os.path.join(args.model_path, 'point_labels.pth'))

    instance_ratio = compute_instance_ratios(point_labels, update_progress=True)
    if args.category_denoise_action == "bank":
        from category_priors.category_denoise import (
            attach_candidate_votes,
            save_candidate_bank,
        )

        branch_vote_ratios = compute_instance_ratios(
            torch.from_numpy(category_denoise_bank.branch_full_labels.copy()),
            update_progress=False,
        )
        category_denoise_bank = attach_candidate_votes(
            category_denoise_bank, branch_vote_ratios, args.classes
        )
        save_candidate_bank(
            category_denoise_bank, args.category_denoise_bank_path
        )
        print(
            "category-denoise bank saved: "
            f"{args.category_denoise_bank_path}"
        )
    elif args.category_denoise_action == "candidate-repair":
        from pathlib import Path

        from category_priors.category_candidate_trace import (
            save_candidate_formation_trace,
        )
        from category_priors.category_denoise import (
            attach_candidate_votes,
            save_candidate_bank,
        )

        family_root = Path(args.category_denoise_bank_path)
        saved_counts = {}
        for condition, family_bank in category_candidate_family.banks.items():
            branch_vote_ratios = compute_instance_ratios(
                torch.from_numpy(family_bank.branch_full_labels.copy()),
                update_progress=False,
            )
            voted_bank = attach_candidate_votes(
                family_bank, branch_vote_ratios, args.classes
            )
            save_candidate_bank(voted_bank, family_root / condition)
            saved_counts[condition] = len(voted_bank.candidates)
        save_candidate_formation_trace(
            category_candidate_family.formation_trace,
            args.category_candidate_trace_path,
        )
        category_denoise_diagnostics = {
            "candidate_family_counts": saved_counts,
            "trace_path": os.path.abspath(args.category_candidate_trace_path),
            "b0_side_path_unchanged": True,
        }
        print(
            "category candidate family and trace saved: "
            f"{family_root}; {args.category_candidate_trace_path}"
        )
    if args.v5_candidate_source != 'off':
        from category_priors.teacher_prior import saga20_branch_classes
        from category_priors.v3_shadow import target_top1_masks, vote_summary
        from category_priors.v5_candidate import (
            V5CandidateConfig, source_masks, write_v5_proposal_capture,
        )

        v5_config = V5CandidateConfig()
        v5_classes = list(saga20_branch_classes(class_to_idx))
        normalized_label_features = F.normalize(
            label_features.detach().cpu(), dim=-1, p=2
        )
        all_class_indices = list(range(len(args.classes)))
        codebook_masks, codebook_top1, codebook_score, codebook_margin = (
            target_top1_masks(
                normed_point_semantic_features.numpy(),
                normalized_label_features.numpy(), all_class_indices,
                threshold=v5_config.semantic_threshold,
            )
        )
        source_views = np.zeros(len(point_xyz), dtype=np.int16)
        source_ratio = np.zeros(len(point_xyz), dtype=np.float32)
        source_margin = np.zeros(len(point_xyz), dtype=np.float32)
        source_top1 = codebook_top1.copy()
        source_score = codebook_score.copy()
        source_semantic_margin = codebook_margin.copy()
        if args.v5_candidate_source == 'multiview':
            pixel_votes = np.zeros((len(point_xyz), len(args.classes)), dtype=np.int32)
            for camera in tqdm(camera_list, desc='V5 multiview semantic source'):
                mask_file = os.path.join(args.masks_path, f'{camera.image_name}.pt')
                label_file = os.path.join(args.labels_path, f'{camera.image_name}.pt')
                if not os.path.isfile(mask_file) or not os.path.isfile(label_file):
                    continue
                masks = torch.load(mask_file, map_location='cpu')
                if masks.shape[-2:] != (camera.image_height, camera.image_width):
                    masks = torch.nn.functional.interpolate(
                        masks.float().unsqueeze(1), mode='bilinear',
                        size=(camera.image_height, camera.image_width),
                        align_corners=False,
                    ).squeeze(1) > 0.5
                else:
                    masks = masks.bool()
                labels_2d = torch.load(label_file, map_location='cpu')
                contributors = get_max_contributor(camera, 'cpu').detach().cpu().numpy()
                valid_seen = []
                for label_2d, mask_2d in zip(labels_2d, masks):
                    class_index = int(label_2d)
                    if class_index < 0 or class_index >= len(args.classes):
                        continue
                    gaussian_ids = contributors[mask_2d.detach().cpu().numpy()]
                    if gaussian_ids.size == 0:
                        continue
                    values, counts = np.unique(gaussian_ids, return_counts=True)
                    pixel_votes[values, class_index] += counts.astype(np.int32)
                    valid_seen.append(values)
                if valid_seen:
                    source_views[np.unique(np.concatenate(valid_seen))] += 1
            totals = pixel_votes.sum(axis=1)
            source_top1 = pixel_votes.argmax(axis=1).astype(np.int16)
            source_score = np.divide(
                pixel_votes[np.arange(len(point_xyz)), source_top1], totals,
                out=np.zeros(len(point_xyz), dtype=np.float32), where=totals > 0,
            ).astype(np.float32)
            if pixel_votes.shape[1] > 1:
                two = np.partition(pixel_votes, -2, axis=1)[:, -2:]
                source_margin = np.divide(
                    two[:, 1] - two[:, 0], totals,
                    out=np.zeros(len(point_xyz), dtype=np.float32), where=totals > 0,
                ).astype(np.float32)
            source_semantic_margin = source_margin.copy()
            v5_masks = source_masks(
                source='multiview', winner=source_top1, score=source_score,
                class_indices=class_to_idx, multiview_views=source_views,
                multiview_ratio=source_score, multiview_margin=source_margin,
                config=v5_config,
            )
        else:
            v5_masks = source_masks(
                source='codebook', winner=codebook_top1, score=codebook_score,
                class_indices=class_to_idx, config=v5_config,
            )
        v5_branch, v5_instance_classes = cluster_other_classes(
            normed_point_features.clone(), normed_point_semantic_features.clone(),
            point_xyz.clone(), normalized_label_features, class_to_idx, v5_classes, args,
            device='cpu', exclusive_masks=v5_masks,
            diagnostics_attribute='_v5_candidate_diagnostics',
            v5_candidate={'source': args.v5_candidate_source, 'config': v5_config},
        )
        v5_ratios = compute_instance_ratios(v5_branch, update_progress=False)
        v5_diagnostics = dict(getattr(args, '_v5_candidate_diagnostics', {}))
        v5_rows = list(v5_diagnostics.pop('__candidates__', []))
        v5_final_labels = point_labels.detach().cpu().numpy()
        for row in v5_rows:
            candidate_id = int(row['candidate_id'])
            candidate_mask = (v5_branch == candidate_id).numpy()
            candidate_count = int(candidate_mask.sum())
            overlaps = []
            for b1_instance_id in np.unique(v5_final_labels[candidate_mask]):
                if int(b1_instance_id) < 0:
                    continue
                b1_mask = v5_final_labels == int(b1_instance_id)
                intersection = int((candidate_mask & b1_mask).sum())
                union = int((candidate_mask | b1_mask).sum())
                ratio = instance_ratio.get(int(b1_instance_id))
                b1_class = 'background'
                if ratio is not None and ratio.size and ratio.max() >= args.label_threshold:
                    b1_class = args.classes[int(np.argmax(ratio))]
                overlaps.append({
                    'instance_id': int(b1_instance_id), 'class': b1_class,
                    'iou': intersection / union if union else 0.0,
                    'intersection_points': intersection,
                })
            overlaps.sort(key=lambda item: (-item['iou'], item['instance_id']))
            row['b1_instance_iou'] = overlaps
            row['background_points_against_b1'] = int(
                candidate_count - int((v5_final_labels[candidate_mask] >= 0).sum())
            )
            row['vote'] = vote_summary(
                v5_ratios.get(candidate_id, np.zeros(len(args.classes), dtype=np.float64)),
                args.classes, row['branch_class'],
            )
        write_v5_proposal_capture(
            json_path=args.v5_candidate_output,
            labels_path=args.v5_candidate_labels_output,
            scene_id=args.v5_scene_id,
            seed=args.seed,
            source=args.v5_candidate_source,
            git_commit=args.v5_git_commit,
            class_names=args.classes,
            branch_labels=v5_branch.numpy(),
            core_labels=getattr(args, '_v5_core_labels').numpy(),
            assignment_confidence=getattr(args, '_v5_assignment_confidence').numpy(),
            semantic_winner=source_top1,
            semantic_score=source_score,
            semantic_margin=source_semantic_margin,
            source_view_count=source_views,
            source_vote_ratio=source_score,
            source_vote_margin=source_margin,
            candidates=v5_rows,
            class_diagnostics=v5_diagnostics,
        )
        print(
            f"V5 {args.v5_candidate_source} captured {len(v5_instance_classes)} "
            "proposal-bank candidates without modifying B1 labels"
        )
    if args.v6_candidate_mode != 'off':
        from category_priors.v6_candidate import (
            V6GraphConfig, build_affinity_components, finalise_multiview_candidates,
            normalized_top1 as v6_normalized_top1, write_v6_proposal_bank,
        )

        v6_config = V6GraphConfig()
        v6_gate = scale_gate(torch.tensor([1.0]).cuda()).unsqueeze(0).detach().cpu()
        v6_affinity = F.normalize(
            F.normalize(point_features, dim=-1, p=2) * v6_gate, dim=-1, p=2
        ).numpy()
        v6_metric_xyz = point_xyz.numpy() * float(args.scene_scale_m_per_unit)
        v6_components = build_affinity_components(v6_metric_xyz, v6_affinity, v6_config)
        v6_full_provisional = np.asarray(v6_components['full_labels'], dtype=np.int32)
        v6_candidate_votes = np.zeros(
            (len(v6_components['candidates']), len(args.classes)), dtype=np.int16
        )
        v6_point_views = np.zeros(len(point_xyz), dtype=np.int16)
        v6_point_votes = np.zeros((len(point_xyz), len(args.classes)), dtype=np.int16)
        for camera in tqdm(camera_list, desc='V6 multiview candidate semantics'):
            mask_file = os.path.join(args.masks_path, f'{camera.image_name}.pt')
            label_file = os.path.join(args.labels_path, f'{camera.image_name}.pt')
            if not os.path.isfile(mask_file) or not os.path.isfile(label_file):
                continue
            masks = torch.load(mask_file, map_location='cpu')
            if masks.shape[-2:] != (camera.image_height, camera.image_width):
                masks = torch.nn.functional.interpolate(
                    masks.float().unsqueeze(1), mode='bilinear',
                    size=(camera.image_height, camera.image_width), align_corners=False,
                ).squeeze(1) > 0.5
            else:
                masks = masks.bool()
            labels_2d = torch.load(label_file, map_location='cpu')
            contributors = get_max_contributor(camera, 'cpu').detach().cpu().numpy()
            point_pixels = np.zeros((len(point_xyz), len(args.classes)), dtype=np.int32)
            candidate_pixels = np.zeros(v6_candidate_votes.shape, dtype=np.int32)
            for label_2d, mask_2d in zip(labels_2d, masks):
                class_index = int(label_2d)
                if class_index < 0 or class_index >= len(args.classes):
                    continue
                gaussian_ids = contributors[mask_2d.detach().cpu().numpy()]
                gaussian_ids = gaussian_ids[(gaussian_ids >= 0) & (gaussian_ids < len(point_xyz))]
                if not gaussian_ids.size:
                    continue
                values, counts = np.unique(gaussian_ids, return_counts=True)
                point_pixels[values, class_index] += counts.astype(np.int32)
                candidate_ids = v6_full_provisional[values]
                valid = candidate_ids >= 0
                if np.any(valid):
                    candidate_pixels[candidate_ids[valid], class_index] += counts[valid].astype(np.int32)
            visible_points = np.flatnonzero(point_pixels.sum(axis=1) > 0)
            if len(visible_points):
                point_winners = point_pixels[visible_points].argmax(axis=1)
                v6_point_views[visible_points] += 1
                v6_point_votes[visible_points, point_winners] += 1
            visible_candidates = np.flatnonzero(candidate_pixels.sum(axis=1) > 0)
            if len(visible_candidates):
                candidate_winners = candidate_pixels[visible_candidates].argmax(axis=1)
                v6_candidate_votes[visible_candidates, candidate_winners] += 1
        v6_finalised = finalise_multiview_candidates(
            v6_components, v6_candidate_votes, args.classes, v6_config
        )
        v6_final_labels = np.asarray(v6_finalised['full_labels'], dtype=np.int32)
        v6_core_labels = np.asarray(v6_finalised['core_labels'], dtype=np.int32)
        v6_b1_labels = point_labels.detach().cpu().numpy()
        v6_candidate_count = len(v6_finalised['candidates'])
        v6_candidate_sizes = np.bincount(
            v6_final_labels[v6_final_labels >= 0], minlength=v6_candidate_count,
        )
        v6_core_background_sizes = np.bincount(
            v6_core_labels[(v6_core_labels >= 0) & (v6_b1_labels < 0)],
            minlength=v6_candidate_count,
        )
        v6_overlaps: list[list[dict]] = [[] for _ in range(v6_candidate_count)]
        v6_pair_mask = (v6_final_labels >= 0) & (v6_b1_labels >= 0)
        if np.any(v6_pair_mask):
            v6_pairs, v6_intersections = np.unique(
                np.column_stack((
                    v6_final_labels[v6_pair_mask], v6_b1_labels[v6_pair_mask],
                )), axis=0, return_counts=True,
            )
            v6_b1_ids, v6_b1_sizes = np.unique(
                v6_b1_labels[v6_b1_labels >= 0], return_counts=True,
            )
            v6_b1_size_by_id = dict(zip(v6_b1_ids.tolist(), v6_b1_sizes.tolist()))
            for (candidate_id, b1_instance_id), intersection in zip(
                v6_pairs.tolist(), v6_intersections.tolist(),
            ):
                candidate_size = int(v6_candidate_sizes[candidate_id])
                b1_size = int(v6_b1_size_by_id[b1_instance_id])
                union = candidate_size + b1_size - int(intersection)
                ratio = instance_ratio.get(int(b1_instance_id))
                b1_class = 'background'
                if ratio is not None and ratio.size and ratio.max() >= args.label_threshold:
                    b1_class = args.classes[int(np.argmax(ratio))]
                v6_overlaps[candidate_id].append({
                    'instance_id': int(b1_instance_id), 'class': b1_class,
                    'iou': int(intersection) / union if union else 0.0,
                    'intersection_points': int(intersection),
                })
        for row in v6_finalised['candidates']:
            candidate_id = int(row['candidate_id'])
            row['b1_instance_iou'] = sorted(
                v6_overlaps[candidate_id], key=lambda item: (-item['iou'], item['instance_id'])
            )
            row['background_points_against_b1'] = int(
                v6_candidate_sizes[candidate_id]
                - sum(item['intersection_points'] for item in v6_overlaps[candidate_id])
            )
            row['core_background_points_against_b1'] = int(v6_core_background_sizes[candidate_id])
        v6_codebook_winner, v6_codebook_score, v6_codebook_margin = v6_normalized_top1(
            normed_point_semantic_features.numpy(),
            F.normalize(label_features.detach().cpu(), dim=-1, p=2).numpy(),
        )
        point_totals = v6_point_votes.sum(axis=1)
        v6_point_winner = v6_point_votes.argmax(axis=1).astype(np.int16)
        v6_point_ratio = np.divide(
            v6_point_votes[np.arange(len(point_xyz)), v6_point_winner], point_totals,
            out=np.zeros(len(point_xyz), dtype=np.float32), where=point_totals > 0,
        ).astype(np.float32)
        v6_two = np.partition(v6_point_votes, -2, axis=1)[:, -2:]
        v6_point_margin = np.divide(
            v6_two[:, 1] - v6_two[:, 0], point_totals,
            out=np.zeros(len(point_xyz), dtype=np.float32), where=point_totals > 0,
        ).astype(np.float32)
        write_v6_proposal_bank(
            json_path=args.v6_candidate_output, labels_path=args.v6_candidate_labels_output,
            scene_id=args.v6_scene_id, seed=args.seed, git_commit=args.v6_git_commit,
            class_names=args.classes, finalised=v6_finalised,
            codebook_winner=v6_codebook_winner, codebook_score=v6_codebook_score,
            codebook_margin=v6_codebook_margin, point_view_count=v6_point_views,
            point_vote_winner=v6_point_winner, point_vote_ratio=v6_point_ratio,
            point_vote_margin=v6_point_margin,
        )
        print(
            f"V6 affinity-first captured {len(v6_finalised['candidates'])} "
            "proposal-bank candidates without modifying B1 labels"
        )
    if v3_shadow_captures:
        from category_priors.v3_shadow import (
            candidate_survival,
            label_overlap,
            vote_summary,
            write_shadow_capture,
        )

        for shadow_mode, capture in v3_shadow_captures.items():
            shadow_branch = capture['branch_labels']
            stages = v3_shadow_stages[shadow_mode]
            shadow_ratios = compute_instance_ratios(
                shadow_branch, update_progress=False
            )
            shadow_diagnostics = dict(
                getattr(args, f'_v3_shadow_diagnostics_{shadow_mode}', {})
            )
            candidate_rows = list(shadow_diagnostics.pop('__candidates__', []))
            enriched_candidates = []
            for candidate in candidate_rows:
                candidate = dict(candidate)
                candidate_id = int(candidate['candidate_id'])
                candidate_mask = (shadow_branch == candidate_id).numpy()
                candidate['active_branch_points'] = int(candidate_mask.sum())
                merged_id = stages['merged_ids'].get(candidate_id)
                candidate['hypothetical_merged_instance_id'] = merged_id
                if merged_id is not None:
                    candidate.update(
                        candidate_survival(
                            candidate_mask,
                            merged_id,
                            stages['after_knn'].numpy(),
                            stages['after_filter'].numpy(),
                        )
                    )
                else:
                    candidate.update({
                        'after_knn_points': 0,
                        'after_knn_survival_rate': 0.0,
                        'after_filter_points': 0,
                        'after_filter_survival_rate': 0.0,
                    })
                candidate['global_pre_overlap'] = label_overlap(
                    stages['global_pre'].numpy(), candidate_mask
                )
                candidate['global_final_overlap'] = label_overlap(
                    point_labels.detach().cpu().numpy(), candidate_mask
                )
                class_ratios = shadow_ratios.get(
                    candidate_id, np.zeros(len(args.classes), dtype=np.float64)
                )
                candidate['vote'] = vote_summary(
                    class_ratios, args.classes, candidate['branch_class']
                )
                enriched_candidates.append(candidate)
            output_json = args.v3_shadow_output.format(mode=shadow_mode)
            output_labels = args.v3_branch_labels_output.format(mode=shadow_mode)
            write_shadow_capture(
                json_path=output_json,
                labels_path=output_labels,
                scene_id=args.v3_shadow_scene_id,
                seed=args.seed,
                mode=shadow_mode,
                git_commit=args.v3_shadow_git_commit,
                class_names=args.classes,
                affinity_gate=gates.numpy(),
                branch_labels=shadow_branch.numpy(),
                semantic_top1=v3_semantic_top1,
                semantic_top1_score=v3_semantic_score,
                semantic_margin=v3_semantic_margin,
                sam_covered=v3_sam_covered.numpy(),
                candidates=enriched_candidates,
                class_diagnostics=shadow_diagnostics,
            )
    if v4_candidate_capture is not None:
        from category_priors.v3_shadow import label_overlap, vote_summary
        from category_priors.v4_candidate import write_v4_candidate_capture

        branch = v4_candidate_capture['branch_labels']
        ratios = compute_instance_ratios(branch, update_progress=False)
        diagnostics = dict(getattr(args, '_v4_candidate_diagnostics', {}))
        candidate_rows = list(diagnostics.pop('__candidates__', []))
        enriched = []
        final_labels = point_labels.detach().cpu().numpy()
        for candidate in candidate_rows:
            row = dict(candidate)
            candidate_id = int(row['candidate_id'])
            mask = (branch == candidate_id).numpy()
            row['active_branch_points'] = int(mask.sum())
            row['global_final_overlap'] = label_overlap(final_labels, mask)
            row['vote'] = vote_summary(
                ratios.get(candidate_id, np.zeros(len(args.classes), dtype=np.float64)),
                args.classes, row['branch_class'],
            )
            enriched.append(row)
        write_v4_candidate_capture(
            json_path=args.v4_candidate_output,
            labels_path=args.v4_candidate_labels_output,
            scene_id=args.v4_scene_id,
            seed=args.seed,
            mode=args.v4_candidate_mode,
            git_commit=args.v4_git_commit,
            class_names=args.classes,
            affinity_gate=gates.numpy(),
            branch_labels=branch.numpy(),
            semantic_top1=v4_candidate_capture['semantic_top1'],
            semantic_top1_score=v4_candidate_capture['semantic_score'],
            semantic_margin=v4_candidate_capture['semantic_margin'],
            sam_covered=v3_sam_covered.numpy(),
            candidates=enriched,
            class_diagnostics=diagnostics,
        )
    print(
        f"max-contributor cache summary: hits={max_contributor_cache_hits}, "
        f"misses={max_contributor_cache_misses}"
    )

    def get_class(classes, ratio:np.ndarray):
        if ratio.max()<args.label_threshold:
            return 'background'
        return classes[ratio.argmax()]
    def get_bbox(point_labels, xyz, is_big_gaussian):
        from trimesh.bounds import oriented_bounds_2D
        dir1 = torch.tensor([0.,1.,0.])
        bbox = {}
        for instance_id in torch.unique(point_labels).tolist():
            if instance_id < 0:
                continue
            instance_xyz = xyz[(point_labels==instance_id)&~is_big_gaussian]
            points_3d = instance_xyz.numpy()
            # --- 1. 投影到X-Z平面 (忽略Y) ---
            N, D = points_3d.shape
            points_2d = points_3d[:, [0, 2]]  # shape: (N, 2), X和Z坐标

            # --- 2. 使用trimesh计算2D定向包围盒 (fallback到简单AABB) ---
            # 注意：oriented_bounds_2D 返回的是一个变换矩阵，将点变换后，其AABB中心在原点

            # Empty bbox when no valid points
            if N == 0:
                bbox[instance_id] = [0.0] * 24  # 8 corners * 3 coordinates = 24
                continue

            # Fallback: 当点数不足时使用简单的轴对齐包围盒(AABB)
            if N < 3:
                # 直接用AABB: xmin, xmax, ymin, ymax, zmin, zmax
                p_min = points_3d.min(axis=0)
                p_max = points_3d.max(axis=0)
                # 构建8个角点
                bbox_corners_world = np.array([
                    [p_max[0], p_max[1], p_max[2]],
                    [p_max[0], p_max[1], p_min[2]],
                    [p_max[0], p_min[1], p_min[2]],
                    [p_max[0], p_min[1], p_max[2]],
                    [p_min[0], p_max[1], p_max[2]],
                    [p_min[0], p_max[1], p_min[2]],
                    [p_min[0], p_min[1], p_min[2]],
                    [p_min[0], p_min[1], p_max[2]]
                ])
                bbox[instance_id] = torch.from_numpy(bbox_corners_world).flatten().tolist()
                continue

            transform_2d, rectangle_extents_2d = oriented_bounds_2D(
                points_2d,
            )
            # transform_2d: (3, 3) 2D齐次变换矩阵
            # rectangle_extents_2d: (2,) [width, height] in the transformed 2D space
        
            # --- 3. 将2D变换扩展为3D变换 ---
            # 我们需要构造一个 (4, 4) 的3D齐次变换矩阵
            transform_3d = np.eye(4)  # 初始化为单位矩阵
        
            # 将2D变换的旋转和平移部分复制到3D变换中
            # transform_2d 是:
            # [ R_xx  R_xz  tx ]
            # [ R_zx  R_zz  tz ]
            # [  0     0    1 ]
            transform_3d[0, 0] = transform_2d[0, 0]  # R_xx
            transform_3d[0, 2] = transform_2d[0, 1]  # R_xz
            transform_3d[0, 3] = transform_2d[0, 2]  # tx
        
            transform_3d[2, 0] = transform_2d[1, 0]  # R_zx
            transform_3d[2, 2] = transform_2d[1, 1]  # R_zz
            transform_3d[2, 3] = transform_2d[1, 2]  # tz
        
            # Y轴保持不变: transform_3d[1,1] = 1, 其他为0 (已由eye(4)设置)
        
            # --- 4. 应用3D变换，将点云"摆正" ---
            # 将3D点云转换为齐次坐标
            points_3d_hom = np.hstack([points_3d, np.ones((N, 1))])  # (N, 4)
            points_3d_transformed = (transform_3d @ points_3d_hom.T).T  # (N, 4)
            points_3d_transformed = points_3d_transformed[:, :3]  # 去掉齐次维度 (N, 3)
        
            # --- 5. 计算变换后点云的AABB ---
            aabb_min = points_3d_transformed.min(axis=0)  # (3,)
            aabb_max = points_3d_transformed.max(axis=0)  # (3,)
        
            # Y方向的尺寸来自原始点云的Y范围
            y_extent = points_3d[:, 1].max() - points_3d[:, 1].min()
            # 注意：transform_3d 不改变Y坐标，所以 aabb_min[1] 和 aabb_max[1] 就是原始Y的平移
        
            # 在变换空间中构建8个角点 (局部坐标)
            # 注意：X和Z来自rectangle_extents_2d，Y来自原始范围
            half_extents_xz = rectangle_extents_2d / 2.0
            # 由于transform_2d已经将AABB中心移到原点，所以角点在±half_extents
            corners_local = np.array([
                [ half_extents_xz[0],  aabb_max[1],  half_extents_xz[1]],
                [ half_extents_xz[0],  aabb_max[1], -half_extents_xz[1]],
                [ half_extents_xz[0],  aabb_min[1], -half_extents_xz[1]],
                [ half_extents_xz[0],  aabb_min[1],  half_extents_xz[1]],
                [-half_extents_xz[0],  aabb_max[1],  half_extents_xz[1]],
                [-half_extents_xz[0],  aabb_max[1], -half_extents_xz[1]],
                [-half_extents_xz[0],  aabb_min[1], -half_extents_xz[1]],
                [-half_extents_xz[0],  aabb_min[1],  half_extents_xz[1]]
            ])  # (8, 3)
        
            # --- 6. 将局部角点转换回世界坐标 ---
            # 需要应用 transform_3d 的逆矩阵
            transform_3d_inv = np.linalg.inv(transform_3d)
            corners_local_hom = np.hstack([corners_local, np.ones((8, 1))])  # (8, 4)
            bbox_corners_world_hom = (transform_3d_inv @ corners_local_hom.T).T  # (8, 4)
            bbox_corners_world = bbox_corners_world_hom[:, :3]  # (8, 3)
            bbox[instance_id] = torch.from_numpy(bbox_corners_world).flatten().tolist()
        return bbox
    def combine_prop(bbox, clazz):
        merged = {}
        for instance in bbox.keys() & clazz.keys():
            ratio = np.asarray(instance_ratio.get(instance, []), dtype=np.float64)
            score = float(ratio.max()) if ratio.size else 0.0
            merged[instance] = {
                "bbox": bbox[instance],
                "class": clazz[instance],
                "score": score,
            }
        return merged
    bbox = get_bbox(point_labels.cpu(), point_xyz, is_big_gaussian)
    clazz = {instance: get_class(args.classes, ratio) for instance, ratio in instance_ratio.items()}
    for instance_id, branch_class in category_denoise_classes.items():
        if instance_id in clazz:
            clazz[instance_id] = branch_class
    teacher_vote_class_mismatches = 0
    teacher_vote_class_matches = 0
    teacher_vote_class_total = 0
    for instance_id, branch_class in teacher_merged_classes.items():
        if instance_id not in clazz:
            continue
        teacher_vote_class_total += 1
        if clazz[instance_id] == branch_class:
            teacher_vote_class_matches += 1
    for instance_id, branch_class in teacher_preserved_classes.items():
        if instance_id not in clazz:
            continue
        if clazz[instance_id] != branch_class:
            teacher_vote_class_mismatches += 1
        clazz[instance_id] = branch_class
    raw_instances = combine_prop(bbox, clazz)
    for instance_id, score in category_denoise_scores.items():
        if instance_id in raw_instances:
            raw_instances[instance_id]["score"] = float(score)
    raw_instances = {
        key: value
        for key, value in raw_instances.items()
        if value.get('class') in args.selected_classes
    }
    contracted = normalize_prediction(point_labels.tolist(), raw_instances)
    export_id_by_raw = {
        int(raw_id): int(new_id)
        for new_id, raw_id in enumerate(sorted(int(value) for value in raw_instances))
    }
    if category_denoise_diagnostics.get("candidate_survival"):
        for row in category_denoise_diagnostics["candidate_survival"]:
            raw_id = row.get("final_id")
            row["final_instance_id"] = (
                export_id_by_raw.get(int(raw_id)) if raw_id is not None else None
            )
    category_denoise_exported_classes = {
        export_id_by_raw[raw_id]: branch_class
        for raw_id, branch_class in category_denoise_classes.items()
        if raw_id in export_id_by_raw
    }
    category_denoise_exported_scores = {
        export_id_by_raw[raw_id]: score
        for raw_id, score in category_denoise_scores.items()
        if raw_id in export_id_by_raw
    }
    if args.stage_trace_path:
        trace_path = os.path.abspath(args.stage_trace_path)
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        post_knn_trace = (
            teacher_after_knn
            if teacher_after_knn is not None
            else post_filter_trace
        )
        np.savez_compressed(
            trace_path,
            global_sample_core=global_sample_core_trace.numpy(),
            global_full_assignment=global_full_assignment_trace.numpy(),
            other_class_candidates=other_class_candidates_trace.numpy(),
            branch_class_before_merge=teacher_merged_membership.numpy(),
            merged_partition=merged_partition_trace.numpy(),
            post_global_knn=post_knn_trace.detach().cpu().numpy(),
            post_filter=post_filter_trace.numpy(),
            post_attach=post_attach_trace.numpy(),
            final_internal_labels=point_labels.detach().cpu().numpy(),
            exported_prediction=contracted.point_labels,
        )
        trace_metadata = {
            "schema": "saga-v9-legacy-stage-trace-v1",
            "point_count": int(len(point_labels)),
            "level": args.v7_causal_ablation,
            "branch_instance_classes": {
                str(key): value
                for key, value in sorted(teacher_merged_classes.items())
            },
            "raw_instances": {str(key): value for key, value in raw_instances.items()},
            "vote_histogram": {
                str(key): np.asarray(value, dtype=np.float64).tolist()
                for key, value in instance_ratio.items()
            },
        }
        with open(os.path.splitext(trace_path)[0] + ".json", "w", encoding="utf-8") as handle:
            json.dump(trace_metadata, handle)
    output = dict()
    output['point_labels'] = contracted.point_labels.tolist()
    output['is_big_gaussian'] = is_big_gaussian.tolist()
    output['is_transparent_gaissian'] = is_transparent_gaissian.tolist()
    output['instances'] = contracted.instances
    output['prediction_contract'] = contracted.audit
    with open(args.json_path,'w') as f:
        json.dump(output,f)
    if args.prior_metadata_path:
        from category_priors.runtime import build_instance_metadata, write_instance_metadata

        run_info = {
            "seed": args.seed,
            "prior_mode": args.prior_mode,
            "clustering_mode": args.clustering_mode,
            "teacher_prior_mode": args.teacher_prior_mode,
            "legacy_prior_mode": args.legacy_prior_mode,
            "prior_gate": args.prior_gate,
            "prior_shrink": args.prior_shrink,
            "scene_scale_m_per_unit": args.scene_scale_m_per_unit,
            "output_json": os.path.abspath(args.json_path),
        }
        if not args.minimal_metadata:
            from category_priors.io import sha256_file
            run_info["output_json_sha256"] = sha256_file(args.json_path)
        if args.prior_config:
            run_info["category_priors"] = os.path.abspath(args.prior_config)
            if not args.minimal_metadata:
                run_info["category_priors_sha256"] = sha256_file(args.prior_config)
        if args.prior_mapping_config:
            run_info["prior_mapping_config"] = os.path.abspath(args.prior_mapping_config)
            if not args.minimal_metadata:
                run_info["prior_mapping_config_sha256"] = sha256_file(args.prior_mapping_config)
        metadata = build_instance_metadata(
            point_labels,
            instance_ratio,
            point_assignment_confidence,
            args.classes,
            prior_overlay,
            run_info,
            include_content_hash=not args.minimal_metadata,
        )
        for instance_id, branch_class in teacher_preserved_classes.items():
            values = metadata["instances"].get(str(instance_id))
            if values is None:
                continue
            class_index = class_to_idx[branch_class]
            ratio = instance_ratio.get(instance_id)
            semantic_confidence = (
                float(ratio[class_index])
                if ratio is not None and len(ratio) > class_index else 0.0
            )
            values["class"] = branch_class
            values["semantic_confidence"] = semantic_confidence
            values["score"] = float(np.clip(
                semantic_confidence * float(values["mean_assignment_confidence"]),
                0.0,
                1.0,
            ))
        if teacher_preserved_classes:
            metadata.pop("content_sha256", None)
        if legacy_prior is not None:
            metadata.pop("content_sha256", None)
            metadata["legacy_prior"] = {
                "mode": args.legacy_prior_mode,
                "surface_density_points_per_m2": legacy_surface_density,
                "classes": getattr(args, '_legacy_prior_diagnostics', {}),
            }
            for values in metadata["instances"].values():
                if args.legacy_prior_score == 'unit':
                    values["score"] = 1.0
                elif args.legacy_prior_score == 'vote':
                    values["score"] = float(values["semantic_confidence"])
                else:
                    values["score"] = float(values["mean_assignment_confidence"])
        # `build_instance_metadata` operates in the internal/raw label space,
        # whereas output.json is already normalized to contiguous export IDs.
        # Remap once here so diagnostics and the exported prediction cannot
        # become two incompatible truths after an inserted/deleted instance.
        from category_priors.prediction_contract import (
            remap_instance_metadata_to_export,
        )
        metadata["instances"] = remap_instance_metadata_to_export(
            metadata.get("instances", {}), export_id_by_raw, contracted
        )
        if args.clustering_mode == 'legacy':
            metadata["teacher_prior"] = {
                "mode": args.teacher_prior_mode,
                "branch_preservation": teacher_branch_preservation,
                "restore_after_global_filter": (
                    teacher_restore_after_global_filter
                ),
                "evidence_protection": teacher_protection_diagnostics,
                "classes": (
                    getattr(args, '_teacher_prior_diagnostics', {})
                    if teacher_prior is not None
                    else getattr(args, '_legacy_prior_diagnostics', {})
                ),
                "instance_id_to_branch_class": {
                    str(key): value for key, value in teacher_merged_classes.items()
                },
                "post_global_filter": teacher_post_filter,
                "vote_class_matches": teacher_vote_class_matches,
                "vote_class_total": teacher_vote_class_total,
                "vote_class_agreement": (
                    teacher_vote_class_matches / teacher_vote_class_total
                    if teacher_vote_class_total else None
                ),
                "preserved_instances": len(teacher_preserved_classes),
                "preserved_points": (
                    int((teacher_preservation >= 0).sum())
                    if teacher_preservation is not None else 0
                ),
                "vote_class_mismatches_overridden": teacher_vote_class_mismatches,
                "final_instances": len(output['instances']),
                "assigned_points": int((point_labels >= 0).sum()),
                "total_points": len(point_labels),
                "coverage": (
                    float((point_labels >= 0).sum()) / len(point_labels)
                    if len(point_labels) else 0.0
                ),
            }
        if args.category_denoise_action != "off":
            metadata.pop("content_sha256", None)
            metadata["category_denoise"] = {
                "action": args.category_denoise_action,
                "mode": args.category_denoise_mode,
                "scene_id": args.category_denoise_scene_id,
                "bank_path": os.path.abspath(args.category_denoise_bank_path),
                **category_denoise_diagnostics,
            }
            for instance_id, branch_class in category_denoise_exported_classes.items():
                values = metadata["instances"].get(str(instance_id))
                if values is None:
                    continue
                values["class"] = branch_class
                values["score"] = float(category_denoise_exported_scores[instance_id])
        write_instance_metadata(args.prior_metadata_path, metadata)
    if(args.clean):
        if os.path.isdir(args.masks_path):
            shutil.rmtree(args.masks_path)
        if os.path.isdir(args.labels_path):
            shutil.rmtree(args.labels_path)
        if os.path.isdir(args.mask_scales_path):
            shutil.rmtree(args.mask_scales_path)
        if os.path.isfile(args.contrastive_feature_point_cloud_path):
            os.remove(args.contrastive_feature_point_cloud_path)
        if os.path.isfile(args.scale_gate_path):
            os.remove(args.scale_gate_path)


if __name__ == "__main__":
    main()

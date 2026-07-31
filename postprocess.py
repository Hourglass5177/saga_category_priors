import shutil
import torch
from scene import Scene
import os
import json
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
                          other_classes, args, device='cpu'):
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

    # Normalize spatial coordinates
    min_val = torch.min(point_xyz, dim=0).values
    max_val = torch.max(point_xyz, dim=0).values
    std_point_xyz = (point_xyz - min_val) / (max_val - min_val)

    for class_name in other_classes:
        if class_name not in class_to_idx:
            print(f"Warning: Class '{class_name}' not in label_features, skipping")
            continue

        class_idx = class_to_idx[class_name]
        print(f"\nProcessing other class: {class_name} (idx={class_idx})")

        # Step 1: Select points by semantic similarity (using semantic features)
        selection_mask, similarity_scores = select_points_by_semantic_similarity(
            point_semantic_features, label_features, class_idx,
            args.other_classes_similarity_threshold, device
        )

        num_selected = selection_mask.sum().item()
        print(f"  Selected {num_selected} points (similarity >= {args.other_classes_similarity_threshold})")

        if num_selected < args.other_classes_min_cluster_size:
            print(f"  Skipping: insufficient points")
            continue

        # Step 2: Sample selected points for efficiency
        selected_features = point_features[selection_mask]
        selected_xyz = std_point_xyz[selection_mask]
        selected_similarities = similarity_scores[selection_mask]

        sample_size = min(num_selected, args.other_classes_sample_num)
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

        # Semantic distance: 1 - similarity to class feature (lower is better)
        # Both points should be similar to the class semantic feature
        semantic_sim_matrix = torch.outer(sampled_similarities, sampled_similarities)
        # Use negative correlation: if both points have high similarity, semantic distance should be low
        semantic_dist = 1 - semantic_sim_matrix
        semantic_dist = torch.clamp(semantic_dist, 0, 1)

        # Normalize distances to [0, 1] range
        if instance_feature_dist.max() > 0:
            instance_feature_dist = instance_feature_dist / (instance_feature_dist.max() + 1e-8)
        if spatial_dist.max() > 0:
            spatial_dist = spatial_dist / (spatial_dist.max() + 1e-8)

        # Hybrid distance with three components
        hybrid_distance = (args.other_classes_feature_ratio * instance_feature_dist +
                          args.other_classes_spatial_ratio * spatial_dist +
                          args.other_classes_semantic_ratio * semantic_dist)

        # Step 4: HDBSCAN clustering
        clusterer = HDBSCAN(
            min_cluster_size=args.other_classes_min_cluster_size,
            cluster_selection_epsilon=0.01,
            allow_single_cluster=False,
            metric='precomputed'
        )
        cluster_labels = clusterer.fit_predict(hybrid_distance.numpy().astype(np.float64))

        num_clusters = len([l for l in np.unique(cluster_labels) if l >= 0])
        print(f"  Found {num_clusters} clusters")

        if num_clusters == 0:
            print(f"  Skipping: no valid clusters")
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
        selected_hybrid_sim = (args.other_classes_feature_ratio * selected_feature_sim +
                              (1 - args.other_classes_feature_ratio) * selected_xyz_sim)
        selected_confidence = torch.softmax(selected_hybrid_sim * 10, dim=-1)
        selected_mask, selected_cluster_labels = selected_confidence.max(dim=-1)

        # Apply threshold
        below_threshold = selected_mask < args.instance_threshold
        selected_cluster_labels[below_threshold] = -1

        # Step 7: Map back to original point indices and assign instance IDs
        selected_indices_original = torch.where(selection_mask)[0]

        for local_cluster_id in range(num_clusters):
            # Find points assigned to this cluster
            points_in_cluster = (selected_cluster_labels == local_cluster_id)

            if points_in_cluster.sum() < args.other_classes_min_cluster_size:
                continue

            # Get original point indices
            original_indices = selected_indices_original[points_in_cluster]

            # Assign instance ID (starts from 0)
            instance_id = current_instance_id
            other_point_labels[original_indices] = instance_id
            other_instance_to_class[instance_id] = class_name

            current_instance_id += 1
            print(f"  Instance {instance_id}: {points_in_cluster.sum()} points -> {class_name}")

    print(f"\nSemantic-guided clustering complete: {current_instance_id} instances created")
    return other_point_labels, other_instance_to_class

@resource_error_handler("语义识别后处理阶段")
def main():
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--progress_path", type=str, required=True)
    parser.add_argument("--clean", action='store_true')
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--feature_ratio", type=float, default=0.5)
    parser.add_argument("--instance_threshold", type=float, default=0.3)
    parser.add_argument("--label_threshold", type=float, default=0.3)
    parser.add_argument("--scale_threshold", type=float, default=0.8)
    parser.add_argument("--opcity_threshold", type=float, default=0.005)
    parser.add_argument("--sample_num", type=int, default=10000)
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
    args = parser.parse_args(sys.argv[1:])
    bg_color = torch.tensor([1,1,1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    torch.manual_seed(42)
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

    point_features = feat_gs_model.get_point_features.detach().cpu()
    point_semantic_features = feat_gs_model.get_point_semantic_features.detach().cpu()
    point_xyz = feat_gs_model.get_xyz.detach().cpu()
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

    clusterer = HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=0.01, allow_single_cluster = False, metric='precomputed') # HDBSCAN

    start_time = datetime.now()
    cluster_labels = clusterer.fit_predict(hybird_distance.numpy().astype(np.float64))
    end_time = datetime.now()
    elapsed_time = end_time - start_time

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
    mask, point_labels = confidence.max(dim=-1)
    mask = mask>args.instance_threshold
    print(f'{mask.sum()=}, {(~mask).sum()=}')
    point_labels[~mask] = -1
    print(f'{elapsed_time=}, {len(torch.unique(point_labels))=}') # 3
    print(f'HDBSCAN finish')

    # ========== SEMANTIC-GUIDED CLUSTERING (for other_classes) ==========
    if label_features is not None and class_to_idx is not None and len(args.other_classes) > 0:
        print(f"\n{'='*60}")
        print(f"Starting semantic-guided clustering for other_classes")
        print(f"Classes: {args.other_classes}")
        print(f"Feature ratio: {args.other_classes_feature_ratio}, Spatial ratio: {args.other_classes_spatial_ratio}, Semantic ratio: {args.other_classes_semantic_ratio}")
        print(f"{'='*60}")

        other_point_labels, other_instance_to_class = cluster_other_classes(
            normed_point_features.clone(),  # Instance features for clustering
            normed_point_semantic_features.clone(),  # Semantic features for class filtering
            point_xyz.clone(),
            label_features,
            class_to_idx,
            args.other_classes,
            args,
            device='cpu'
        )

        # ========== MERGE other_class instances into main labels (BEFORE filters) ==========
        if len(other_instance_to_class) > 0:
            # Get max instance ID from main clustering (excluding -1 background)
            max_main_instance_id = point_labels.max().item() if point_labels.max() >= 0 else -1

            # Merge assigned instances (>= 0)
            for other_instance_id in other_instance_to_class.keys():
                new_instance_id = max_main_instance_id + 1 + other_instance_id
                mask = (other_point_labels == other_instance_id)
                point_labels[mask] = new_instance_id

            print(f"Merged {len(other_instance_to_class)} other_class instances into main labels")
            print(f"Total instances before filters: {len(torch.unique(point_labels))}")

        print(f"{'='*60}\n")

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
    start_time = datetime.now()
    if args.k>0:
        point_labels = filter3d(point_xyz, point_labels, args.k)
    point_labels = filter_num(point_labels, min_num=10)
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f'{elapsed_time=}, {len(torch.unique(point_labels))=}') # 89, pytorch3d.ops.knn_points=49
    print(f'knn finish')
    # torch.save(point_labels, os.path.join(args.model_path, 'point_labels.pth'))
    # point_labels = torch.load(os.path.join(args.model_path, 'point_labels.pth'))

    vote = {instance: [0 for _ in range(len(args.classes)+1)] for instance in torch.unique(point_labels).tolist()} 
    for i, camera in tqdm(list(enumerate(camera_list))):
        with open(args.progress_path, 'w') as f:
            f.write(str(0+(i+1)*100//len(camera_list)))
        if not os.path.exists(os.path.join(args.masks_path, f'{camera.image_name}.pt')):
            continue
        masks = torch.load(os.path.join(args.masks_path, f'{camera.image_name}.pt')).float()
        masks = torch.nn.functional.interpolate(masks.unsqueeze(1), mode = 'bilinear', size = (camera.image_height, camera.image_width), align_corners = False).squeeze(1)
        masks[masks>0.5] = 1
        masks[masks!=1] = 0
        masks = masks.bool()
        labels = torch.load(os.path.join(args.labels_path, f'{camera.image_name}.pt'))
        render_pkg = render_with_max_contributor(camera, gs_model, args, bg_color)
        max_contributor = render_pkg['max_contributor'].to(point_labels.device)
        max_contribute = render_pkg['max_contribute'].to(point_labels.device)
        max_instance_contributor = point_labels[max_contributor]
        background_label = len(args.classes)
        background = torch.ones_like(masks[0])
        for label, mask in zip(labels, masks):
            background &= ~mask
            vote_for_label = max_instance_contributor[mask]
            for instance in torch.unique(point_labels).tolist():
                vote[instance][label]+=(vote_for_label==instance).sum().item()
        vote_for_background_label = max_instance_contributor[background]
        for instance in torch.unique(point_labels).tolist():
            vote[instance][background_label]+=(vote_for_background_label==instance).sum().item()

    instance_ratio = {}
    for instance, votes in vote.items():
        votes = np.array(votes)
        votes = votes[:-1]/votes.sum()
        instance_ratio[instance] = votes

    def get_class(classes, ratio:np.ndarray):
        if ratio.max()<args.label_threshold:
            return 'background'
        return classes[ratio.argmax()]
    def get_bbox(point_labels, xyz, is_big_gaussian):
        from trimesh.bounds import oriented_bounds_2D
        dir1 = torch.tensor([0.,1.,0.])
        bbox = {}
        for instance_id in torch.unique(point_labels).tolist():
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
        merged = {instance: {"bbox": bbox[instance], "class": clazz[instance]}
                for instance in bbox.keys() & clazz.keys()}
        return merged
    bbox = get_bbox(point_labels.cpu(), point_xyz, is_big_gaussian)
    clazz = {instance: get_class(args.classes, ratio) for instance, ratio in instance_ratio.items()}
    output = dict()
    output['point_labels'] = point_labels.tolist()
    output['is_big_gaussian'] = is_big_gaussian.tolist()
    output['is_transparent_gaissian'] = is_transparent_gaissian.tolist()
    # output['instances'] = {instance: {'class': get_class(args.classes, ratio)} for instance, ratio in instance_ratio.items()}
    output['instances'] = combine_prop(bbox, clazz)
    output['instances'] = {k: v for k, v in output['instances'].items() if v.get('class') in args.selected_classes}
    with open(args.json_path,'w') as f:
        json.dump(output,f)
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

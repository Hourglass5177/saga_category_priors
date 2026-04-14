#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from gaussian_renderer import render_contrastive_feature, render_semantic_feature
import sys
from scene import Scene, GaussianModel, FeatureGaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

import numpy as np


import torch
from torch import nn
import pytorch3d.ops


import time

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from sklearn.preprocessing import QuantileTransformer
from utils.resource_exit import resource_error_handler

import torch

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

def farthest_point_sample(xyz, n_samples):
    """
    输入：
        xyz:       点云坐标，形状为 [N, 3] 的PyTorch张量
        n_samples: 需要采样的点数
    输出：
        mask:      布尔掩码，形状为 [N]，True表示被采样的点
    """
    xyz = xyz.detach().cpu()
    device = xyz.device
    N, _ = xyz.shape
    
    # 初始化采样点索引和距离矩阵
    centroids = torch.zeros(n_samples, dtype=torch.long, device=device)
    distance = torch.full((N,), float('inf'), device=device)
    
    # 随机选择第一个点（或按质心优化选择）
    farthest = torch.randint(0, N, (1,), device=device).item()
    
    for i in range(n_samples):
        centroids[i] = farthest
        centroid = xyz[farthest].view(1, 3)
        
        # 计算所有点到当前采样点的欧氏距离
        dist = torch.sum((xyz - centroid) ** 2, dim=1)
        
        # 更新每个点的最小距离（与已选点集的最近距离）
        mask = dist < distance
        distance[mask] = dist[mask]
        
        # 选择距离最大的点作为下一个采样点
        farthest = torch.argmax(distance)
    
    # 生成布尔掩码
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    mask[centroids] = True
    return mask

# Borrowed from GARField but modified
def get_quantile_func(scales: torch.Tensor, distribution="normal"):
    """
    Use 3D scale statistics to normalize scales -- use quantile transformer.
    """
    scales = scales.flatten()

    scales = scales.detach().cpu().numpy()

    # Calculate quantile transformer
    quantile_transformer = QuantileTransformer(output_distribution=distribution)
    quantile_transformer = quantile_transformer.fit(scales.reshape(-1, 1))

    def quantile_transformer_func(scales):
        # This function acts as a wrapper for QuantileTransformer.
        # QuantileTransformer expects a numpy array, while we have a torch tensor.
        scales = scales.reshape(-1,1)
        return torch.Tensor(
            quantile_transformer.transform(scales.detach().cpu().numpy())
        ).to(scales.device)

    return quantile_transformer_func


def _camera_mask_count(camera):
    masks = getattr(camera, "original_masks", None)
    if masks is None:
        return 0
    if masks.ndim == 0:
        return 0
    return int(masks.shape[0])


def _camera_scale_count(camera):
    mask_scales = getattr(camera, "mask_scales", None)
    if mask_scales is None:
        return 0
    if mask_scales.ndim == 0:
        return 1
    return int(mask_scales.numel())


def _collect_valid_train_cameras(scene):
    valid_cameras = []
    invalid_reasons = {
        "missing_masks": [],
        "missing_mask_scales": [],
        "too_few_masks": [],
        "too_few_mask_scales": [],
        "count_mismatch": [],
    }

    for cam in scene.getTrainCameras():
        mask_count = _camera_mask_count(cam)
        scale_count = _camera_scale_count(cam)

        if getattr(cam, "original_masks", None) is None:
            invalid_reasons["missing_masks"].append(cam.image_name)
            continue
        if getattr(cam, "mask_scales", None) is None:
            invalid_reasons["missing_mask_scales"].append(cam.image_name)
            continue
        if mask_count < 2:
            invalid_reasons["too_few_masks"].append(cam.image_name)
            continue
        if scale_count < 2:
            invalid_reasons["too_few_mask_scales"].append(cam.image_name)
            continue
        if mask_count != scale_count:
            invalid_reasons["count_mismatch"].append(cam.image_name)
            continue

        valid_cameras.append(cam)

    return valid_cameras, invalid_reasons


def _format_invalid_camera_summary(invalid_reasons):
    parts = []
    labels = {
        "missing_masks": "missing masks",
        "missing_mask_scales": "missing mask scales",
        "too_few_masks": "fewer than 2 masks",
        "too_few_mask_scales": "fewer than 2 mask scales",
        "count_mismatch": "mask/scale count mismatch",
    }
    for key, names in invalid_reasons.items():
        if names:
            preview = ", ".join(names[:3])
            if len(names) > 3:
                preview += ", ..."
            parts.append(f"{labels[key]}: {len(names)} ({preview})")
    return "; ".join(parts)

def training(args, dataset, opt, pipe, iteration, saving_iterations, checkpoint_iterations, debug_from):
    print("RFN weight:", opt.rfn)
    print("Smooth K:", opt.smooth_K)
    print("Scale aware dim:", opt.scale_aware_dim)
    assert opt.ray_sample_rate > 0 or opt.num_sampled_rays > 0

    dataset.need_features = False
    dataset.need_masks = True
    dataset.allow_principle_point_shift = False

    gaussians = None #GaussianModel(dataset.sh_degree)

    feature_gaussians = FeatureGaussianModel(dataset.feature_dim, dataset.semantic_feature_dim)

    sample_rate = 1.0
    scene = Scene(dataset, gaussians, feature_gaussians, load_iteration=iteration, shuffle=False, target='contrastive_feature', mode='train', sample_rate=sample_rate)

    feature_gaussians.change_to_segmentation_mode(opt, "contrastive_feature", fixed_feature=False)
    feature_gaussians._xyz.requires_grad_(False)
    feature_gaussians._scaling.requires_grad_(False)
    feature_gaussians._rotation.requires_grad_(False)
    feature_gaussians._opacity.requires_grad_(False)
    feature_gaussians._point_features.requires_grad_(True)
    # 30030
    scale_gate = torch.nn.Sequential(
        torch.nn.Linear(1, 32, bias=True),
        torch.nn.Sigmoid()
    )
    scale_gate = scale_gate.cuda()
    scale_gate.train()

    param_group = {'params': scale_gate.parameters(), 'lr': opt.feature_lr, 'name': 'f'}
    feature_gaussians.optimizer.add_param_group(param_group)

    smooth_weights = None

    del gaussians
    torch.cuda.empty_cache()

    background = torch.ones([dataset.feature_dim], dtype=torch.float32, device="cuda") if dataset.white_background else torch.zeros([dataset.feature_dim], dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    first_iter = 0
    viewpoint_stack = None
    valid_train_cameras, invalid_camera_reasons = _collect_valid_train_cameras(scene)
    invalid_camera_summary = _format_invalid_camera_summary(invalid_camera_reasons)
    if invalid_camera_summary:
        print(f"Skipping unusable training cameras: {invalid_camera_summary}")

    if not valid_train_cameras:
        raise RuntimeError(
            "No usable training cameras were found for contrastive feature training. "
            f"Checked {len(scene.getTrainCameras())} cameras under masks_path='{dataset.masks_path}' "
            f"and mask_scales_path='{dataset.mask_scales_path}'. "
            "Run `get_scale.py` for the same dataset paths and ensure each training image has matching "
            "mask and mask-scale `.pt` files with at least two entries."
        )

    if not opt.iterations:
        opt.iterations = min(len(valid_train_cameras)*10, 10000)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    print("Preparing Quantile Transform...")
    # gather scales
    all_scales = torch.cat([cam.mask_scales.reshape(-1) for cam in valid_train_cameras])

    upper_bound_scale = all_scales.max().item()
    # upper_bound_scale = np.percentile(all_scales.detach().cpu().numpy(), 75)

    # all_scales = []
    # for cam in scene.getTrainCameras():
    #     cam.mask_scales = torch.clamp(cam.mask_scales, 0, upper_bound_scale)
    #     all_scales.append(cam.mask_scales)
    # all_scales = torch.cat(all_scales)

    scale_aware_dim = opt.scale_aware_dim

    if scale_aware_dim <= 0 or scale_aware_dim >= 32:
        print("Using adaptive scale gate.")
        q_trans = get_quantile_func(all_scales, "uniform")
    else:
        q_trans = get_quantile_func(all_scales, "uniform")
        fixed_scale_gate = torch.tensor([[1 for j in range(32 - scale_aware_dim + i)] + [0 for k in range(scale_aware_dim - i)] for i in range(scale_aware_dim+1)]).cuda()

    for iteration in range(first_iter, opt.iterations + 1):
        with open(args.progress_path, 'w') as f:
            f.write(str(0+(iteration)*100//opt.iterations))
        iter_start.record()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = valid_train_cameras.copy()
        
        if iteration < -1:
            viewpoint_cam = viewpoint_stack[0]
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        with torch.no_grad():
            # N_mask, H, W
            sam_masks = viewpoint_cam.original_masks.cuda().float() # float[masks, h, w]
            viewpoint_cam.feature_height, viewpoint_cam.feature_width = viewpoint_cam.image_height, viewpoint_cam.image_width

            # N_mask
            mask_scales = viewpoint_cam.mask_scales.cuda()

            mask_scales, sort_indices = torch.sort(mask_scales, descending=True)
            sam_masks = sam_masks[sort_indices, :, :] # scale descending mask list bool[masks, h, w]

            num_sampled_scales = 8 # 8+2=10 scale pivot

            sampled_scale_index = torch.randperm(len(mask_scales))[:num_sampled_scales]

            tmp = torch.zeros(num_sampled_scales+2)

            tmp[1:len(sampled_scale_index)+1] = sampled_scale_index
            tmp[-1] = len(mask_scales) - 1
            tmp[0] = -1 # attach a bigger scale
            sampled_scale_index = tmp.long()
            

            sampled_scales = mask_scales[sampled_scale_index] # 10 scale pivot

            smaller_scales = mask_scales[mask_scales < upper_bound_scale]
            second_big_scale = smaller_scales.max() if smaller_scales.numel() > 0 else upper_bound_scale

            non_mask_region = sam_masks.sum(dim = 0) == 0
            ray_sample_rate = opt.ray_sample_rate if opt.ray_sample_rate > 0 else torch.clamp(opt.num_sampled_rays / (~non_mask_region).sum(), 0, 1)

            sampled_ray = torch.rand(sam_masks.shape[-2], sam_masks.shape[-1]).cuda()
            sampled_ray[non_mask_region] = 1e9
            sampled_ray = sampled_ray < ray_sample_rate

            sampled_ray = torch.logical_and(sampled_ray, ~non_mask_region) # sampled pixel mask

            # H W
            per_pixel_mask_size = sam_masks * sam_masks.sum(-1).sum(-1)[:,None,None] # mask fill with size, [masks, h, w]

            per_pixel_mean_mask_size = per_pixel_mask_size.sum(dim = 0) / (sam_masks.sum(dim = 0) + 1e-9) # float[h, w]

            per_pixel_mean_mask_size = per_pixel_mean_mask_size[sampled_ray] # sampled pixel mean mask size, float[sampled pixels]


            pixel_to_pixel_mask_size = per_pixel_mean_mask_size.unsqueeze(0) * per_pixel_mean_mask_size.unsqueeze(1) # float[1, sampled pixels] * float[sampled pixels, 1] -> float[sampled pixels, sampled pixels]
            ptp_max_size = pixel_to_pixel_mask_size.max()
            pixel_to_pixel_mask_size[pixel_to_pixel_mask_size == 0] = 1e10
            per_pixel_weight = torch.clamp(ptp_max_size / pixel_to_pixel_mask_size, 1.0, None)
            per_pixel_weight = (per_pixel_weight - per_pixel_weight.min()) / (per_pixel_weight.max() - per_pixel_weight.min() + 1e-9) * 9. + 1. # pixel的平均mask size越大其权重越小, float[sampled pixels, sampled pixels]
            
            sam_masks_sampled_ray = sam_masks[:, sampled_ray] # bool[masks, sampled pixels]

            gt_corrs = []

            sampled_scales[0] = upper_bound_scale + upper_bound_scale * torch.rand(1)[0]
            for idx, si in enumerate(sampled_scale_index): # 公式(6)
                upper_bound = sampled_scales[idx] >= upper_bound_scale

                if si != len(mask_scales) - 1 and not upper_bound:
                    sampled_scales[idx] -= (sampled_scales[idx] - mask_scales[si+1]) * torch.rand(1)[0]
                elif upper_bound:
                    sampled_scales[idx] -= (sampled_scales[idx] - second_big_scale) * torch.rand(1)[0]
                else:
                    sampled_scales[idx] -= sampled_scales[idx] * torch.rand(1)[0]

                if not upper_bound:
                    gt_vec = torch.zeros_like(sam_masks_sampled_ray)
                    gt_vec[:si+1,:] = sam_masks_sampled_ray[:si+1,:]
                    for j in range(si, -1, -1):
                        gt_vec[j,:] = torch.logical_and(
                            torch.logical_not(gt_vec[j+1:,:].any(dim = 0)), gt_vec[j,:]
                        )
                    gt_vec[si+1:,:] = sam_masks_sampled_ray[si+1:,:]
                else:
                    gt_vec = sam_masks_sampled_ray # float[masks, sampled pixels], a pixel belong to a mask

                gt_corr = torch.einsum('nh,nj->hj', gt_vec, gt_vec)
                gt_corr[gt_corr != 0] = 1 # float[sampled pixels, sampled pixels], a pixel in the same mask with another pixel
                gt_corrs.append(gt_corr)

            # N_scale S C_clip
            # gt_clip_features = torch.stack(gt_clip_features, dim = 0)
            # N_scale S S
            gt_corrs = torch.stack(gt_corrs, dim = 0) # float[scale pivots, sampled pixels, sampled pixels]
            # sampled_scales normalization to [0, 1]
            sampled_scales = q_trans(sampled_scales).squeeze()
            sampled_scales = sampled_scales.squeeze()

        render_pkg_feat = render_contrastive_feature(viewpoint_cam, feature_gaussians, pipe, background, norm_point_features=True, smooth_type = 'traditional', smooth_weights=torch.softmax(smooth_weights, dim = -1) if smooth_weights is not None else None, smooth_K = opt.smooth_K)
        rendered_features = render_pkg_feat["render"]

        rendered_feature_norm = rendered_features.norm(dim = 0, p=2).mean()
        rendered_feature_norm_reg = (1-rendered_feature_norm)**2 # regularization term, keep aligned on a ray

        rendered_features = torch.nn.functional.interpolate(rendered_features.unsqueeze(0), viewpoint_cam.original_masks.shape[-2:], mode='bilinear').squeeze(0)

        # float[sampled_scales, 32] gates
        if scale_aware_dim <= 0 or scale_aware_dim >= 32:
            gates = scale_gate(sampled_scales.unsqueeze(-1))
        else:
            int_sampled_scales = ((1 - sampled_scales.squeeze()) * scale_aware_dim).long()
            gates = fixed_scale_gate[int_sampled_scales].detach()

        # float[sampled_scales, C, H, W]
        feature_with_scale = rendered_features.unsqueeze(0).repeat([sampled_scales.shape[0],1,1,1])
        feature_with_scale = feature_with_scale * gates.unsqueeze(-1).unsqueeze(-1) # element-wise mul

        sampled_feature_with_scale = feature_with_scale[:,:,sampled_ray] # float[sampled scales, C, sampled pixels]

        scale_conditioned_features_sam = sampled_feature_with_scale.permute([0,2,1]) # float[sampled scales, sampled pixels, C]

        scale_conditioned_features_sam = torch.nn.functional.normalize(scale_conditioned_features_sam, dim=-1, p=2)
        corr = torch.einsum('nhc,njc->nhj', scale_conditioned_features_sam, scale_conditioned_features_sam) # sampled pixel to sampled pixel similarity, float[sampled scales, sampled pixels, sampled pixels]

        diag_mask = torch.eye(corr.shape[1], dtype=bool, device=corr.device)

        sum_0 = gt_corrs.sum(dim = 0)
        consistent_negative = sum_0 == 0 # two sampled pixels belong to different mask in any sampled scales, bool[sampled pixels, sampled pixels]
        consistent_positive = sum_0 == len(gt_corrs) # two sampled pixels belong to same mask in any sampled scales, bool[sampled pixels, sampled pixels]
        inconsistent = torch.logical_not(torch.logical_or(consistent_negative, consistent_positive))
        inconsistent_num = inconsistent.count_nonzero()
        sampled_num = inconsistent_num / 2

        rand_num = torch.rand_like(sum_0)

        sampled_positive = torch.logical_and(consistent_positive, rand_num < sampled_num / consistent_positive.count_nonzero())

        sampled_negative = torch.logical_and(consistent_negative, rand_num < sampled_num / consistent_negative.count_nonzero())

        sampled_mask_positive = torch.logical_or(
            torch.logical_or(
                sampled_positive, torch.any(torch.logical_and(corr < 0.75, gt_corrs == 1), dim = 0)
            ), 
            inconsistent
        )
        sampled_mask_positive = torch.logical_and(sampled_mask_positive, ~diag_mask)
        sampled_mask_positive = torch.triu(sampled_mask_positive, diagonal=0)
        sampled_mask_positive = sampled_mask_positive.bool()

        sampled_mask_negative = torch.logical_or(
            torch.logical_or(
                sampled_negative, torch.any(torch.logical_and(corr > 0.5, gt_corrs == 0), dim = 0)
            ), 
            inconsistent
        )
        sampled_mask_negative = torch.logical_and(sampled_mask_negative, ~diag_mask)
        sampled_mask_negative = torch.triu(sampled_mask_negative, diagonal=0)
        sampled_mask_negative = sampled_mask_negative.bool()

        per_pixel_weight = per_pixel_weight.unsqueeze(0)

        min_val = torch.min(feature_gaussians.get_xyz, dim=0).values
        max_val = torch.max(feature_gaussians.get_xyz, dim=0).values
        new_min = 0.0
        new_max = 1.0
        std_point_xyz = (feature_gaussians.get_xyz - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
        sample_mask = uniform_sample(feature_gaussians.get_xyz, opt.distance_sample_num)
        sample_xyz = std_point_xyz[sample_mask]
        sample_features = feature_gaussians.get_point_features[sample_mask]
        sample_scaled_features = torch.nn.functional.normalize(sample_features[None,...]*gates[:,None,...], dim=-1)
        ptp_xyz_distance = torch.norm(sample_xyz[:,None,:] - sample_xyz[None,:,:], dim=-1) # float[fps,fps]
        ptp_feature_sim = torch.einsum('sac, sbc -> sab', sample_scaled_features, sample_scaled_features) # float[scale,fps,fps]
        distance_loss = (ptp_xyz_distance*torch.clamp(ptp_feature_sim,0)).mean()

        loss = (- per_pixel_weight[:, sampled_mask_positive] * gt_corrs[:, sampled_mask_positive] * corr[:, sampled_mask_positive]).mean().nan_to_num() \
                + (per_pixel_weight[:, sampled_mask_negative] * (1 - gt_corrs[:, sampled_mask_negative]) * torch.relu(corr[:, sampled_mask_negative])).mean().nan_to_num() \
                + opt.rfn * rendered_feature_norm_reg + opt.distance_weight * distance_loss
        
        semantic_pkg = render_semantic_feature(viewpoint_cam, feature_gaussians, pipe, background, norm_point_features=True)
        semantic_map = semantic_pkg["render"]  # [C, H, W]

        # Interpolate semantic_map to match original_masks resolution (same as rendered_features)
        semantic_map = torch.nn.functional.interpolate(
            semantic_map.unsqueeze(0),
            viewpoint_cam.original_masks.shape[-2:],
            mode='bilinear'
        ).squeeze(0)

        # Build semantic_map_gt from labels and label_features
        # sam_masks: [N_masks, H, W] (already sorted by scale)
        # labels: class indices for each mask
        # label_features: [num_classes, C] feature vectors per class
        C, H, W = semantic_map.shape
        semantic_map_gt = torch.zeros(C, H, W, device="cuda")

        # Sort labels to match the sorted sam_masks
        if viewpoint_cam.labels is not None and viewpoint_cam.label_features is not None:
            labels = viewpoint_cam.labels.cuda()
            label_features = viewpoint_cam.label_features.cuda()  # [num_classes, C]
            masks = viewpoint_cam.original_masks.cuda()[sort_indices]
            labels_sorted = labels[sort_indices]  # [N_masks]

            # Build semantic map GT by iterating masks from largest to smallest scale
            # Later masks (smaller scale) override earlier ones for overlapping pixels
            for mask_idx in range(len(masks)):
                mask = masks[mask_idx]  # [H, W]
                label_idx = labels_sorted[mask_idx]
                if label_idx >= 0 and label_idx < len(label_features):
                    semantic_map_gt[:, mask] = label_features[label_idx].unsqueeze(1)  # [C] -> [C, num_pixels]

        # Compute semantic L2 loss
        semantic_loss = torch.tensor(0.0, device="cuda")
        if viewpoint_cam.labels is not None and viewpoint_cam.label_features is not None:
            semantic_loss = ((semantic_map - semantic_map_gt) ** 2).mean()
            loss = loss + opt.semantic_loss_weight * semantic_loss

        with torch.no_grad():
            cosine_pos = corr[gt_corrs == 1].mean().nan_to_num()
            cosine_neg = corr[gt_corrs == 0].mean().nan_to_num()

        loss.backward()

        feature_gaussians.optimizer.step()
        feature_gaussians.optimizer.zero_grad(set_to_none = True)

        iter_end.record()

        if iteration % 10 == 0:
            progress_bar.set_postfix({
                "RFN": f"{rendered_feature_norm.item():.{3}f}",
                "Pos cos": f"{cosine_pos.item():.{3}f}",
                "Neg cos": f"{cosine_neg.item():.{3}f}",
                "Sem Loss": f"{semantic_loss.item():.{4}f}",
                "Loss": f"{loss.item():.{3}f}",
            })
            progress_bar.update(10)

    
    # scene.save_feature(iteration, target = 'contrastive_feature', smooth_weights = torch.softmax(smooth_weights, dim = -1) if smooth_weights is not None else None, smooth_type = 'traditional', smooth_K = opt.smooth_K)
    scene.feature_gaussians.save_ply(args.contrastive_feature_point_cloud_path, 
                                     smooth_weights=torch.softmax(smooth_weights, dim = -1) if smooth_weights is not None else None,
                                     smooth_type = 'traditional', 
                                     smooth_K = opt.smooth_K)
    # torch.save(scale_gate.state_dict(), os.path.join(scene.model_path, "point_cloud/iteration_{}/".format(iteration) + "scale_gate.pt"))
    torch.save(scale_gate.state_dict(), args.scale_gate_path)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@resource_error_handler("语义识别特征训练阶段")
def main():
        # Set up command line argument parser
        parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        parser.add_argument("--progress_path", type=str, required=True)
        parser.add_argument('--ip', type=str, default="127.0.0.1")
        parser.add_argument('--port', type=int, default=np.random.randint(10000, 20000))
        parser.add_argument('--debug_from', type=int, default=-1)
        parser.add_argument('--detect_anomaly', action='store_true', default=False)
        parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
        parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
        parser.add_argument("--quiet", action="store_true")
        parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
        parser.add_argument("--start_checkpoint", type=str, default = None)
        parser.add_argument('--target', default='contrastive_feature', const='contrastive_feature', nargs='?', choices=['scene', 'seg', 'feature', 'coarse_seg_everything', 'contrastive_feature'])
        parser.add_argument("--iteration", default=-1, type=int)
    
        # args = get_combined_args(parser, target_cfg_file = 'cfg_args')
        args = parser.parse_args(sys.argv[1:])
        args.save_iterations.append(args.iterations)
    
        print("Optimizing " + args.model_path)

        # Initialize system state (RNG)
        safe_state(args.quiet)

        torch.autograd.set_detect_anomaly(args.detect_anomaly)
        training(args, lp.extract(args), op.extract(args), pp.extract(args), args.iteration, args.save_iterations, args.checkpoint_iterations, args.debug_from)

        # All done
        print("\nTraining complete.")


if __name__ == "__main__":
    main()

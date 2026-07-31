"""
SAGA 交叉掩码生成（ARKit 深度版）
使用 ARKit 采集的原始深度图进行掩码精炼
输入: <data_path>/data/ 下的 .dmb 深度文件和 .json 相机参数
输出: cross_mask/ 目录，格式与原版完全兼容
"""

import torch
import numpy as np
import os
import struct
import json
import cv2
import glob
import re
from tqdm import tqdm
from PIL import Image
from scipy import ndimage
from skimage.morphology import convex_hull_image
from argparse import ArgumentParser


# ============================================================
# ARKit 深度图读取函数
# ============================================================

def read_arkit_depth(dmb_path):
    """
    读取 ARKit .dmb 深度文件
    格式: 纯二进制 float32, 256×192 分辨率, 单位: 米
    """
    W, H = 256, 192
    expected_bytes = W * H * 4  # float32 = 4 字节

    with open(dmb_path, 'rb') as f:
        data = f.read()

    if len(data) < expected_bytes:
        raise ValueError(f"深度文件太小: {len(data)} < {expected_bytes} 字节")

    depth = np.frombuffer(data[:expected_bytes], dtype=np.float32).reshape(H, W).copy()

    # 清理无效值 (极小值视为 0)
    depth[depth < 1e-6] = 0

    return depth


def align_depth_to_rgb(depth, target_w, target_h):
    """
    将 256×192 的深度图上采样到 RGB 分辨率
    使用最近邻插值保持边缘清晰
    """
    return cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def load_arkit_intrinsics_from_json(json_path, target_w=None, target_h=None):
    """
    从 ARKit JSON 文件读取相机内参
    如果指定了 target_w/h，则按比例缩放内参
    """
    with open(json_path, 'r') as f:
        meta = json.load(f)

    K = np.array(meta['cameraIntrinsics'])

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # 如果深度图分辨率与 RGB 不同，需要缩放内参
    if target_w is not None and target_h is not None:
        # ARKit 深度原始分辨率 256×192
        scale_x = target_w / 256.0
        scale_y = target_h / 192.0
        fx = fx * scale_x
        fy = fy * scale_y
        cx = cx * scale_x
        cy = cy * scale_y

    return torch.Tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1.0]
    ])


def load_arkit_c2w_from_json(json_path):
    """
    从 ARKit JSON 文件读取相机外参 (worldToLocal 求逆)
    """
    with open(json_path, 'r') as f:
        meta = json.load(f)

    w2l = np.array(meta['worldToLocal'])
    c2w = np.linalg.inv(w2l)

    return torch.from_numpy(c2w).float()


# ============================================================
# 辅助函数
# ============================================================

def load_mask_png(mask_path, target_size=None):
    """加载 PNG 格式的掩码"""
    mask = np.array(Image.open(mask_path))

    if len(mask.shape) == 3:
        mask = mask.mean(axis=2)

    mask_binary = (mask > 128).astype(np.float32)

    if target_size is not None:
        mask_pil = Image.fromarray((mask_binary * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(target_size, Image.NEAREST)
        mask_binary = np.array(mask_pil) / 255.0

    return mask_binary


def extract_base_name(view_name):
    """从 SAGA view_name 提取基础名称（去掉扩展名）"""
    return view_name.replace('.jpg', '').replace('.png', '').replace('.JPG', '')


def extract_timestamp_int(view_name):
    """
    从 SAGA view_name 提取整数时间戳（用于匹配深度/JSON文件）
    SAGA: "00000000001-00000000001-456886.359435958"
    返回: "456886"
    """
    base = extract_base_name(view_name)
    parts = base.split('-')
    if len(parts) >= 3:
        full_ts = parts[-1]
        return full_ts.split('.')[0]
    return base.split('.')[0]


# ============================================================
# 点云投影函数
# ============================================================

def depth_to_pointcloud_with_depth_value(depth, mask, intrinsics, c2w):
    """从深度图反投影到 3D 点云，保留深度值作为第 4 维"""
    H, W = depth.shape
    device = depth.device

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    valid_pixels = mask > 0.5
    if valid_pixels.sum() == 0:
        return torch.empty((0, 4), device=device)

    ys_valid = ys[valid_pixels].float()
    xs_valid = xs[valid_pixels].float()
    depths_valid = depth[valid_pixels]

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x_cam = (xs_valid - cx) * depths_valid / fx
    y_cam = (ys_valid - cy) * depths_valid / fy
    z_cam = depths_valid

    points_cam = torch.stack([x_cam, y_cam, z_cam], dim=1)
    points_hom = torch.cat([points_cam, torch.ones_like(points_cam[:, :1])], dim=1)
    points_world = (c2w.to(device) @ points_hom.T).T[:, :3]

    # 保留深度值作为第 4 维
    points_with_depth = torch.cat([points_world, depths_valid.unsqueeze(1)], dim=1)

    return points_with_depth


def project_points_to_camera(points, w2c, intrinsics, H, W):
    """将 3D 点投影到相机视角，返回像素坐标和深度"""
    N = points.shape[0]
    device = points.device

    points_3d = points[:, :3]
    points_hom = torch.cat([points_3d, torch.ones_like(points_3d[:, :1])], dim=1)
    points_cam = (w2c.to(device) @ points_hom.T).T[:, :3]

    depths = points_cam[:, 2]
    valid = depths > 0

    if valid.sum() == 0:
        return torch.empty((0, 3), device=device), valid

    points_cam_valid = points_cam[valid]
    depths_valid = depths[valid]

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u = points_cam_valid[:, 0] * fx / depths_valid + cx
    v = points_cam_valid[:, 1] * fy / depths_valid + cy

    uv = torch.stack([u, v], dim=1)
    projected = torch.cat([uv, depths_valid.unsqueeze(1)], dim=1)

    return projected, valid


# ============================================================
# SAGA 相机参数获取函数（回退用）
# ============================================================

def get_camera_intrinsics(view):
    """从 SAGA Camera 对象获取内参矩阵"""
    from utils.graphics_utils import fov2focal

    f_x = fov2focal(view.FoVx, view.image_width)
    f_y = fov2focal(view.FoVy, view.image_height)
    c_x = view.image_width / 2.0
    c_y = view.image_height / 2.0

    return torch.Tensor([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1.0]
    ])


def get_camera_c2w(view):
    """从 SAGA Camera 对象获取 C2W 矩阵"""
    R = view.R
    T = view.T

    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R).float()
    if isinstance(T, np.ndarray):
        T = torch.from_numpy(T).float()

    c2w = torch.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = T

    return c2w


def get_camera_w2c(view):
    """从 SAGA Camera 对象获取 W2C 矩阵"""
    c2w = get_camera_c2w(view)
    w2c = torch.inverse(c2w)
    return w2c


# ============================================================
# 核心：ARKit 深度版交叉掩码生成
# ============================================================

def find_intersect_mask_arkit(
    instance_mask,              # 当前视角的实例掩码 (H, W)
    view_idx,                   # 当前视角索引
    views,                      # 所有视角列表
    mask_map,                   # 掩码文件映射表 {prefix: full_path}
    depth_map,                  # 深度文件映射表 {ts_int: full_path}
    json_map,                   # JSON 文件映射表 {ts_int: full_path}
    arkit_data_dir,             # ARKit 数据目录（用于回退）
    window_size=100,
):
    """
    使用 ARKit 原始深度图生成交叉掩码（无深度测试版本）
    """
    H, W = instance_mask.shape

    if instance_mask.sum() == 0:
        return np.zeros((H, W), dtype=np.uint8)

    # 初始化掩码（凸包处理，使边界更规整）
    # mask_now = convex_hull_image(instance_mask).astype(np.float32)
    mask_now = instance_mask.astype(np.float32)
    mask_now = torch.Tensor(mask_now).cuda()

    print(f"\n  [调试] 当前视角 {view_idx}: 初始掩码覆盖 {mask_now.sum().item():.0f} 像素")

    processed_views = 0
    removed_pixels_total = 0
    missing_depth = 0
    missing_mask = 0
    empty_background = 0

    for idx in range(len(views)):
        # 只考虑邻近视角
        if not ((idx < int(view_idx + window_size)) and (idx > int(view_idx - window_size))):
            continue
        if idx == view_idx:
            continue

        view = views[idx]
        view_name = view.image_name

        # 使用映射表查找文件
        base_name = extract_base_name(view_name)
        ts_int = extract_timestamp_int(view_name)

        mask_path = mask_map.get(base_name)
        depth_path = depth_map.get(ts_int)
        json_path = json_map.get(ts_int)

        if depth_path is None:
            missing_depth += 1
            continue
        if mask_path is None:
            missing_mask += 1
            continue

        # 读取 ARKit 深度（256×192）
        other_depth_raw = read_arkit_depth(depth_path)

        # 上采样到当前视角的分辨率（如果不同）
        if other_depth_raw.shape[1] != W or other_depth_raw.shape[0] != H:
            other_depth = align_depth_to_rgb(other_depth_raw, W, H)
        else:
            other_depth = other_depth_raw

        other_depth = torch.from_numpy(other_depth).cuda()

        # 读取实例掩码
        other_mask = load_mask_png(mask_path, target_size=(W, H))
        other_mask = torch.from_numpy(other_mask).cuda()

        # 获取背景区域
        if other_mask.sum() == 0:
            continue
        # mask_fg = convex_hull_image(other_mask.cpu().numpy())
        mask_fg = other_mask
        mask_fg = torch.Tensor(mask_fg).cuda()
        mask_others = 1.0 - mask_fg  # 背景 = 1 - 前景

        if mask_others.sum() < 100:
            empty_background += 1
            continue

        # 使用 ARKit 内参和外参
        if json_path is not None:
            try:
                intrinsics = load_arkit_intrinsics_from_json(json_path, W, H).cuda()
                c2w = load_arkit_c2w_from_json(json_path).cuda()
            except Exception:
                intrinsics = get_camera_intrinsics(view).cuda()
                c2w = get_camera_c2w(view).cuda()
        else:
            intrinsics = get_camera_intrinsics(view).cuda()
            c2w = get_camera_c2w(view).cuda()

        # 从背景提取 3D 点云
        depth_points = depth_to_pointcloud_with_depth_value(
            depth=other_depth,
            mask=mask_others,
            intrinsics=intrinsics,
            c2w=c2w
        )

        if depth_points.shape[0] == 0:
            continue

        # 投影到当前视角
        current_view = views[view_idx]
        ts_int_curr = extract_timestamp_int(current_view.image_name)
        json_path_curr = json_map.get(ts_int_curr)

        if json_path_curr is not None:
            try:
                current_intrinsics = load_arkit_intrinsics_from_json(json_path_curr, W, H).cuda()
                current_w2c = torch.inverse(load_arkit_c2w_from_json(json_path_curr)).cuda()
            except Exception:
                current_intrinsics = get_camera_intrinsics(current_view).cuda()
                current_w2c = get_camera_w2c(current_view).cuda()
        else:
            current_intrinsics = get_camera_intrinsics(current_view).cuda()
            current_w2c = get_camera_w2c(current_view).cuda()

        projected, valid = project_points_to_camera(
            points=depth_points,
            w2c=current_w2c,
            intrinsics=current_intrinsics,
            H=H,
            W=W
        )

        if projected.shape[0] == 0:
            continue

        # 筛选有效像素
        b1 = projected[:, 0] <= (W - 1)
        b2 = projected[:, 0] >= 0
        b3 = projected[:, 1] <= (H - 1)
        b4 = projected[:, 1] >= 0
        selected = (b1 & b2 & b3 & b4).nonzero(as_tuple=True)[0]

        if selected.shape[0] == 0:
            continue

        projected_selected = projected[selected]
        swap_index = projected_selected[:, :2].long().T

        # 只保留落在当前掩码内的像素
        valid_mask = mask_now[swap_index[1], swap_index[0]] > 0.5
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]

        if len(valid_indices) == 0:
            continue

        swap_index_valid = (swap_index[1][valid_indices], swap_index[0][valid_indices])

        # 从掩码中移除这些像素
        mask_now[swap_index_valid] = 0
        removed_pixels_total += len(valid_indices)

        processed_views += 1

        if processed_views == 1:
            print(f"    [调试] 第一个相邻视角 {idx}: 移除了 {len(valid_indices)} 个像素")

    print(f"    可用相邻视角: {processed_views}")
    print(f"    缺失深度: {missing_depth}, 缺失掩码: {missing_mask}, 背景不足: {empty_background}")
    print(f"    总共移除了 {removed_pixels_total} 个像素")
    print(f"    最终掩码覆盖: {mask_now.sum().item():.0f} 像素")

    if instance_mask.sum() > 0:
        reduction = (1 - mask_now.sum().item() / instance_mask.sum()) * 100
        print(f"    掩码缩减: {reduction:.1f}%")

    # 形态学后处理
    mask_final = mask_now.cpu().numpy()
    struct = np.ones((3, 3))
    mask_final = ndimage.binary_erosion(mask_final, struct)
    mask_final = ndimage.binary_dilation(mask_final, struct)

    return mask_final.astype(np.float32)


# ============================================================
# 主函数
# ============================================================

def generate_cross_mask_arkit(
    data_path,
    instance_id,
    window_size=100
):
    """
    使用 ARKit 深度生成交叉掩码
    """
    from scene.dataset_readers import readColmapSceneInfo
    from utils.camera_utils import cameraList_from_camInfos
    from argparse import Namespace

    print("=" * 80)
    print("SAGA 交叉掩码生成（ARKit 深度版）")
    print("=" * 80)

    # 设置路径
    arkit_data_dir = os.path.join(data_path, 'data')
    mask_dir = os.path.join(data_path, 'inpaint', str(instance_id), 'mask')
    output_dir = os.path.join(data_path, 'inpaint', str(instance_id), 'cross_mask')

    os.makedirs(output_dir, exist_ok=True)

    print(f" ARKit 数据目录: {arkit_data_dir}")
    print(f" 实例掩码目录: {mask_dir}")
    print(f" 输出目录: {output_dir}")
    print(f" 实例 ID: {instance_id}")
    print(f" 搜索窗口: ±{window_size} 帧")

    # 验证目录存在
    if not os.path.exists(arkit_data_dir):
        raise FileNotFoundError(f"ARKit 数据目录不存在: {arkit_data_dir}")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"实例掩码目录不存在: {mask_dir}")

    # ========== 建立文件映射表 ==========

    # 掩码映射: {完整 base_name: full_path}
    mask_files = glob.glob(os.path.join(mask_dir, '*_mask.png'))
    mask_map = {}
    for f in mask_files:
        basename = os.path.basename(f)
        base_name = basename.replace('_mask.png', '')
        mask_map[base_name] = f

    # 深度映射: {整数时间戳: full_path}
    depth_files = glob.glob(os.path.join(arkit_data_dir, '*_smoothDepth.dmb'))
    depth_map = {}
    for f in depth_files:
        basename = os.path.basename(f)
        ts_float = basename.replace('_smoothDepth.dmb', '')
        ts_int = ts_float.split('.')[0]
        depth_map[ts_int] = f

    # JSON 映射: {整数时间戳: full_path}
    json_files = glob.glob(os.path.join(arkit_data_dir, '*.json'))
    json_map = {}
    for f in json_files:
        basename = os.path.basename(f)
        ts_float = basename.replace('.json', '')
        ts_int = ts_float.split('.')[0]
        json_map[ts_int] = f

    # 加载相机数据
    images_path = os.path.join(data_path, 'fastRecon', 'dense', 'sparse', '0', 'images')
    colmap_path = os.path.join(data_path, 'fastRecon', 'dense', 'sparse', '0')

    print(f"\n 加载相机数据...")

    args = Namespace(
        images_path=images_path,
        features_path=None,
        masks_path=None,
        mask_scales_path=None,
        labels_path=None,
        label_features_path=None
    )

    scene_info = readColmapSceneInfo(
        path=colmap_path,
        images=None,
        eval=False,
        sample_rate=1.0,
        args=args
    )

    camera_args = Namespace(
        resolution=1,
        data_device='cuda'
    )

    cameras = cameraList_from_camInfos(scene_info.train_cameras, 1.0, camera_args)
    print(f" 加载了 {len(cameras)} 个训练视角")

    # 显示映射匹配情况
    print(f"\n 检查映射匹配情况 (前5个视角):")
    matched = 0
    for idx, view in enumerate(cameras[:5]):
        base_name = extract_base_name(view.image_name)
        ts_int = extract_timestamp_int(view.image_name)
        mask_ok = base_name in mask_map
        depth_ok = ts_int in depth_map
        json_ok = ts_int in json_map
        if mask_ok and depth_ok:
            matched += 1
        print(f"   {idx}: base={base_name}, ts={ts_int}, mask={mask_ok}, depth={depth_ok}, json={json_ok}")

    if matched == 0:
        print(f"\n ⚠️ 警告: 前5个视角都没有匹配到文件!")
        print(f"   请检查 base_name 和 ts_int 的提取是否正确")
        return

    # 为每个视角生成交叉掩码
    print(f"\n 开始生成交叉掩码...")

    processed = 0
    for idx, view in enumerate(tqdm(cameras, desc="处理视角")):
        view_name = view.image_name
        base_name = extract_base_name(view_name)      # 完整 base_name
        ts_int = extract_timestamp_int(view_name)     # 整数时间戳

        mask_path = mask_map.get(base_name)
        depth_path = depth_map.get(ts_int)
        json_path = json_map.get(ts_int)

        output_path = os.path.join(output_dir, f'{view_name}.png')


        if mask_path is None or depth_path is None:
            continue

        instance_mask = load_mask_png(mask_path)

        if instance_mask.sum() == 0:
            Image.fromarray(np.zeros_like(instance_mask, dtype=np.uint8)).save(output_path)
            processed += 1
            continue

        cross_mask = find_intersect_mask_arkit(
            instance_mask=instance_mask,
            view_idx=idx,
            views=cameras,
            mask_map=mask_map,
            depth_map=depth_map,
            json_map=json_map,
            arkit_data_dir=arkit_data_dir,
            window_size=window_size
        )

        Image.fromarray((cross_mask * 255).astype(np.uint8)).save(output_path)
        processed += 1

    print(f"\n 完成！处理了 {processed} 个视角")
    print(f" 交叉掩码已保存到: {output_dir}")


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    parser = ArgumentParser(description="SAGA 交叉掩码生成（ARKit 深度版）")
    parser.add_argument('--data_path', type=str, required=True,
                       help='数据根目录')
    parser.add_argument('--instance_id', type=int, required=True,
                       help='要移除的实例 ID')
    parser.add_argument('--window_size', type=int, default=100,
                       help='搜索相邻视角的窗口大小')

    args = parser.parse_args()

    generate_cross_mask_arkit(
        data_path=args.data_path,
        instance_id=args.instance_id,
        window_size=args.window_size
    )
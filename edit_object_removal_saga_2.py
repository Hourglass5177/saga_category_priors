'''
SAGA 项目 - 物体移除实例掩码渲染脚本
为指定实例 ID 渲染掩码，用于后续物体移除和修复
'''
import json
import torch
import argparse
import os
import re
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2

# 导入 SAGA 项目核心模块
from scene import GaussianModel
from scene.dataset_readers import (
    readColmapCameras,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_text
)
from utils.camera_utils import cameraList_from_camInfos
from gaussian_renderer import render_with_depth, render
from arguments import ModelParams, PipelineParams
from torchvision.utils import save_image


def depth_to_colormap(depth, colormap=cv2.COLORMAP_TURBO):
    """将深度图转换为彩色可视化"""
    valid_mask = depth > 0
    if not valid_mask.any():
        return np.zeros_like(depth, dtype=np.uint8)

    depth_min = depth[valid_mask].min()
    depth_max = depth[valid_mask].max()
    depth_normalized = np.zeros_like(depth)
    depth_normalized[valid_mask] = (depth[valid_mask] - depth_min) / (depth_max - depth_min + 1e-7)

    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_uint8, colormap)

    return depth_colored


def create_instance_mask_colors(point_labels, target_instance_id, instances_data):
    """
    为指定实例 ID 创建掩码颜色

    参数:
        point_labels: 每个高斯点的实例 ID 列表
        target_instance_id: 目标实例 ID，如 0
        instances_data: output.json 中的 instances 字典

    返回:
        override_color: [N, 3] 张量，目标实例为白色，其他为黑色
        instance_stats: 包含统计信息的字典
    """
    N = len(point_labels)
    point_labels_tensor = torch.tensor(point_labels)

    # 验证实例 ID
    inst_id_str = str(target_instance_id)
    if inst_id_str not in instances_data:
        print(f"  ️  警告: 实例 ID {target_instance_id} 在 output.json 中未找到")
        return torch.zeros((N, 3), dtype=torch.float32, device='cuda'), {}

    class_name = instances_data[inst_id_str].get('class', 'unknown')
    num_points = (point_labels_tensor == target_instance_id).sum().item()

    instance_stats = {target_instance_id: {'class': class_name, 'points': num_points}}

    print(f"   实例 {target_instance_id} ({class_name}): {num_points} 个点")

    # 创建掩码：目标实例为白色 (1,1,1)，其他为黑色 (0,0,0)
    override_color = torch.zeros((N, 3), dtype=torch.float32, device='cuda')
    mask = (point_labels_tensor == target_instance_id)
    override_color[mask] = 1.0  # 白色

    return override_color, instance_stats


def main():
    parser = argparse.ArgumentParser(description="SAGA 物体移除 - 实例掩码渲染")
    parser.add_argument('--data_path', type=str, required=True, help='数据根路径')
    parser.add_argument('--ply_path', type=str, default=None, help='PLY 文件路径（为空则自动查找最新）')
    parser.add_argument('--outputjson_path', type=str, default=None, help='output.json 路径（可选）')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录（可选）')
    parser.add_argument('--white_background', action='store_true', help='使用白色背景')

    # 实例 ID 指定
    parser.add_argument('--instance_id',  type=int, required=True,
                       help='要渲染的实例 ID，如 --instance_id 0')
    # parser.add_argument('--mask_color', type=str, default='white',
    #                    choices=['white', 'red', 'green', 'blue'],
    #                    help='掩码颜色（默认：白色）')

    args = parser.parse_args()

    # ========== 1. 路径设置 ==========
    data_path = args.data_path
    ply_path = args.ply_path
    outputjson_path = args.outputjson_path

    # 自动推断路径
    if outputjson_path is None:
        outputjson_path = os.path.join(data_path, 'output.json')

    if ply_path is None:
        ply_base = os.path.join(data_path, 'output_models', 'point_cloud')
        if os.path.exists(ply_base):
            files = os.listdir(ply_base)
            sorted_files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()))
            ply_path = os.path.join(ply_base, sorted_files[-1], 'point_cloud.ply')
        else:
            raise FileNotFoundError(f"PLY 目录不存在: {ply_base}")

    colmap_path = os.path.join(data_path, 'fastRecon', 'dense','sparse', '0')

    images_path = os.path.join(data_path, 'fastRecon', 'dense', 'sparse', '0', 'images')

    # 输出目录
    if args.output_dir is None:
        output_dir = os.path.join(data_path, 'inpaint',str(args.instance_id) , 'mask')
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("SAGA 实例掩码渲染工具")
    print("="*60)
    print(f"数据路径: {data_path}")
    print(f"PLY 文件: {ply_path}")
    print(f"COLMAP 路径: {colmap_path}")
    print(f"图像路径: {images_path}")
    print(f"输出目录: {output_dir}")
    print(f" 目标实例 ID: {args.instance_id}")
    # print(f"   掩码颜色: {args.mask_color}")
    print("="*60)

    # ========== 2. 加载 output.json ==========
    print("\n[1/4] 加载实例标签...")
    if not os.path.exists(outputjson_path):
        raise FileNotFoundError(f"output.json 不存在: {outputjson_path}")

    with open(outputjson_path, 'r') as f:
        data = json.load(f)

    point_labels = data['point_labels']
    instances = data['instances']

    print(f"  总点数: {len(point_labels)}")
    print(f"  总实例数: {len(instances)}")

    # 打印所有可用的实例 ID
    available_ids = sorted([int(k) for k in instances.keys()])
    print(f"  可用实例 ID: {available_ids[:20]}{'...' if len(available_ids) > 20 else ''}")

    # ========== 3. 加载 3DGS 模型 ==========
    print("\n[2/4] 加载高斯模型...")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY 文件不存在: {ply_path}")

    # 自动检测 PLY 文件中的 SH 阶数
    from plyfile import PlyData
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    sh_channel_count = 0
    for name in vertex.data.dtype.names:
        if name.startswith('f_rest_'):
            sh_channel_count += 1

    if sh_channel_count > 0:
        sh_degree = int(round(((sh_channel_count + 3) / 3) ** 0.5 - 1))
        print(f"  检测到 SH 阶数: {sh_degree} (球谐系数通道数: {sh_channel_count})")
    else:
        sh_degree = 0
        print(f"  未检测到球谐系数，使用 SH 阶数: 0")

    gs_model = GaussianModel(sh_degree=sh_degree)
    gs_model.load_ply(ply_path)

    print(f"  已加载: {gs_model.get_xyz.shape[0]} 个高斯点")
    print(f"  SH 阶数: {gs_model.active_sh_degree}")
    # ========== 4. 准备实例掩码颜色 ==========
    print(f"\n[2.5/4] 为实例 {args.instance_id} 准备掩码颜色...")
    override_color, instance_stats = create_instance_mask_colors(
        point_labels,
        args.instance_id,
        instances
    )

    # ========== 5. 加载相机视角 ==========
    print("\n[3/4] 加载相机视角...")
    if not os.path.exists(colmap_path):
        raise FileNotFoundError(f"COLMAP 路径不存在: {colmap_path}")

    try:
        cameras = readColmapCameras(
            read_extrinsics_binary(os.path.join(colmap_path, 'images.bin')),
            read_intrinsics_binary(os.path.join(colmap_path, 'cameras.bin')),
            images_path
        )
        print("  使用二进制 COLMAP 格式")
    except Exception as e:
        print(f"  二进制加载失败: {e}，尝试文本格式...")
        cameras = readColmapCameras(
            read_extrinsics_text(os.path.join(colmap_path, 'images.txt')),
            read_intrinsics_text(os.path.join(colmap_path, 'cameras.txt')),
            images_path
        )
        print("  使用文本 COLMAP 格式")

    from argparse import Namespace
    camera_args = Namespace(
        resolution=1,
        data_device='cuda'
    )
    camera_list = cameraList_from_camInfos(cameras, camera_args.resolution, camera_args)

    print(f"  已加载: {len(camera_list)} 个相机视角")

    # ========== 6. 设置渲染参数 ==========
    print("\n[4/4] 开始渲染...")

    bg_color = torch.tensor(
        [1, 1, 1] if args.white_background else [0, 0, 0],
        dtype=torch.float32,
        device='cuda'
    )

    pipeline_parser = argparse.ArgumentParser()
    pipeline_params = PipelineParams(pipeline_parser)
    pipe = pipeline_params.extract(pipeline_parser.parse_args([]))

    print(f"  背景: {'白色' if args.white_background else '黑色'}")
    print(f"  SH 转换: {'Python' if pipe.convert_SHs_python else 'CUDA'}")
    print(f"  协方差计算: {'Python' if pipe.compute_cov3D_python else 'CUDA'}")
    print(f"  调试模式: {pipe.debug}")

    # ========== 7. 渲染所有视角 ==========
    for i, camera in tqdm(enumerate(camera_list), total=len(camera_list), desc="渲染进度"):
        try:
            # 渲染实例掩码
            rendered_pkg = render(
                viewpoint_camera=camera,
                pc=gs_model,
                pipe=pipe,
                bg_color=bg_color,
                override_color=override_color
            )

            rgb = rendered_pkg['render']  # [3, H, W]，目标实例为白色

            # 掩码后处理 - 二值化 + 形态学清理
            # 1. 转换为灰度图（取通道平均值）
            mask_gray = rgb.mean(dim=0)  # [H, W]

            # 2. 二值化：设置阈值（128/255 = 0.5）
            threshold = 0.5
            mask_binary = (mask_gray > threshold).float()

            # 3. 形态学操作：去除噪点和填充空洞
            from scipy import ndimage
            mask_np = mask_binary.cpu().numpy()

            # 移除小的孤立区域（面积 < 100 像素）
            mask_cleaned = ndimage.binary_opening(mask_np, structure=np.ones((3, 3)), iterations=2)

            # 填充小的空洞（面积 < 100 像素）
            mask_cleaned = ndimage.binary_closing(mask_cleaned, structure=np.ones((3, 3)), iterations=2)

            # 转换回 numpy uint8
            mask_uint8 = (mask_cleaned * 255).astype(np.uint8)

            # 保存掩码（PNG，灰度）
            mask_path = os.path.join(output_dir, f'{camera.image_name}_mask.png')
            Image.fromarray(mask_uint8, mode='L').save(mask_path)

            # 彩色版本
            # if args.mask_color != 'white':
            #     mask_colors_map = {
            #         'red': torch.tensor([1.0, 0.0, 0.0], device='cuda'),
            #         'green': torch.tensor([0.0, 1.0, 0.0], device='cuda'),
            #         'blue': torch.tensor([0.0, 0.0, 1.0], device='cuda'),
            #     }
            #     color_tensor = mask_colors_map[args.mask_color]

            #     # 应用颜色到掩码
            #     mask_3ch = rgb * color_tensor.view(3, 1, 1)
            #     color_mask = (mask_3ch.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            #     color_mask_path = os.path.join(output_dir, f'{camera.image_name}_mask_color.jpg')
            #     cv2.imwrite(color_mask_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"\n   渲染视角 {camera.image_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*60)
    print(" 渲染完成！")
    print(f"输出目录: {output_dir}")
    print("="*60)

    # 列出输出文件
    output_files = os.listdir(output_dir)
    print(f"\n生成的文件 (共 {len(output_files)} 个):")
    file_types = {}
    for f in output_files:
        ext = os.path.splitext(f)[1]
        file_types[ext] = file_types.get(ext, 0) + 1

    for ext, count in sorted(file_types.items()):
        print(f"  {ext}: {count} 个文件")

    # 打印实例统计信息
    if instance_stats:
        print(f"\n 实例统计:")
        for inst_id, stats in instance_stats.items():
            print(f"  实例 {inst_id} ({stats['class']}):")
            print(f"    点数: {stats['points']}")


if __name__ == "__main__":
    main()
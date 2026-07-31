'''
SAGA 项目 - 物体移除深度图渲染脚本
用于渲染原始场景的深度图、RGB 图像，为后续的物体移除和修复提供数据
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


from scene import GaussianModel
from scene.dataset_readers import (
    readColmapCameras,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_text
)
from utils.camera_utils import cameraList_from_camInfos
from gaussian_renderer import render_with_depth
from arguments import ModelParams, PipelineParams
from torchvision.utils import save_image


def depth_to_colormap(depth, colormap=cv2.COLORMAP_TURBO):
    """将深度图转换为彩色可视化"""
    # 处理无效深度值
    valid_mask = depth > 0
    if not valid_mask.any():
        return np.zeros_like(depth, dtype=np.uint8)

    # 归一化到 [0, 1]
    depth_min = depth[valid_mask].min()
    depth_max = depth[valid_mask].max()
    depth_normalized = np.zeros_like(depth)
    depth_normalized[valid_mask] = (depth[valid_mask] - depth_min) / (depth_max - depth_min + 1e-7)

    # 转换为 uint8
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)

    # 应用颜色映射
    depth_colored = cv2.applyColorMap(depth_uint8, colormap)

    return depth_colored


def main():
    parser = argparse.ArgumentParser(description="SAGA 物体移除 - 深度图渲染")
    parser.add_argument('--data_path', type=str, required=True, help='数据根路径')
    parser.add_argument('--ply_path', type=str, default=None, help='PLY 文件路径（可选，自动查找最新）')
    parser.add_argument('--outputjson_path', type=str, default=None, help='output.json 路径（可选）')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录(可选，默认为 data_path/inpaint/depth)')
    parser.add_argument('--white_background', action='store_true', help='使用白色背景')
    parser.add_argument('--render_rgb', action='store_true', help='是否渲染 RGB 图像')
    parser.add_argument('--save_depth_vis', action='store_true', help='是否渲染深度可视化（彩色）')
    parser.add_argument('--save_depth_npy', action='store_true', help='是否保存原始深度图为 .npy 格式')

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

    if args.output_dir is None:
        output_dir = os.path.join(data_path, 'inpaint','depth')
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("SAGA 深度图渲染工具")
    print("="*60)
    print(f"数据路径: {data_path}")
    print(f"PLY 文件: {ply_path}")
    print(f"COLMAP 路径: {colmap_path}")
    print(f"图像路径: {images_path}")
    print(f"输出目录: {output_dir}")
    print("="*60)

    # ========== 2. 加载 output.json ==========
    print("\n[1/4] 加载实例标签...")
    if not os.path.exists(outputjson_path):
        raise FileNotFoundError(f"output.json 不存在: {outputjson_path}")
    # outputjson_path = 'output.json'
    with open(outputjson_path, 'r') as f:
        data = json.load(f)

    point_labels = data['point_labels']
    is_big_gaussian = data['is_big_gaussian']
    is_transparent_gaissian = data['is_transparent_gaissian']
    instances = data['instances']

    print(f"  点数: {len(point_labels)}")
    print(f"  实例数: {len(instances)}")
    print(f"  大高斯点: {sum(is_big_gaussian)}")
    print(f"  透明高斯点: {sum(is_transparent_gaissian)}")

        # ========== 3. 加载 3DGS 模型 ==========
    print("\n[2/4] 加载高斯模型...")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY 文件不存在: {ply_path}")

    # 自动检测 PLY 文件中的 SH 阶数
    from plyfile import PlyData
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    # 统计球谐系数通道数量（f_rest_* 字段）
    sh_channel_count = 0
    for name in vertex.data.dtype.names:
        if name.startswith('f_rest_'):
            sh_channel_count += 1

    # 计算 SH 阶数: sh_channels = 3 * (sh_degree + 1)^2 - 3
    # 反推: sh_degree = sqrt((sh_channels + 3) / 3) - 1
    if sh_channel_count > 0:
        sh_degree = int(round(((sh_channel_count + 3) / 3) ** 0.5 - 1))
        print(f"  检测到 SH 阶数: {sh_degree} (球谐系数通道数: {sh_channel_count})")
    else:
        sh_degree = 0
        print(f"  未检测到球谐系数，使用 SH 阶数: 0 (仅 DC 分量)")

    # 初始化高斯模型
    gs_model = GaussianModel(sh_degree=sh_degree)
    gs_model.load_ply(ply_path)

    print(f"  加载完成: {gs_model.get_xyz.shape[0]} 个高斯点")
    print(f"  SH 阶数: {gs_model.active_sh_degree}")

    # ========== 4. 加载相机视角 ==========
    print("\n[3/4] 加载相机视角...")
    if not os.path.exists(colmap_path):
        raise FileNotFoundError(f"COLMAP 路径不存在: {colmap_path}")

    # 尝试加载二进制格式，失败则加载文本格式
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

    # 创建 CameraArgs 对象列表
    from argparse import Namespace
    camera_args = Namespace(
        resolution=1,
        data_device='cuda'
    )
    camera_list = cameraList_from_camInfos(cameras, camera_args.resolution, camera_args)

    print(f"  加载完成: {len(camera_list)} 个相机视角")

    # ========== 5. 设置渲染参数 ==========
    print("\n[4/4] 开始渲染...")

    # 背景颜色
    bg_color = torch.tensor(
        [1, 1, 1] if args.white_background else [0, 0, 0],
        dtype=torch.float32,
        device='cuda'
    )

    # 创建 PipelineParams（使用默认配置）
    pipeline_parser = argparse.ArgumentParser()
    pipeline_params = PipelineParams(pipeline_parser)
    pipe = pipeline_params.extract(pipeline_parser.parse_args([]))

    print(f"  背景颜色: {'白色' if args.white_background else '黑色'}")
    print(f"  SH 转换: {'Python' if pipe.convert_SHs_python else 'CUDA'}")
    print(f"  协方差计算: {'Python' if pipe.compute_cov3D_python else 'CUDA'}")
    print(f"  调试模式: {pipe.debug}")

    # ========== 6. 遍历所有视角进行渲染 ==========
    for i, camera in tqdm(enumerate(camera_list), total=len(camera_list), desc="渲染进度"):
        try:
            # 渲染（包含 RGB、Mask、Depth）
            rendered_pkg = render_with_depth(
                viewpoint_camera=camera,
                pc=gs_model,
                pipe=pipe,
                bg_color=bg_color
            )

            # 提取结果
            rgb = rendered_pkg['render']           # [3, H, W]
            depth = rendered_pkg['depth']          # [1, H, W]
            # mask = rendered_pkg['mask']            # [1, H, W]

            # 保存 RGB 图像
            if args.render_rgb:
                rgb_path = os.path.join(output_dir, f'{camera.image_name}_rgb.png')
                save_image(rgb.detach(), rgb_path)

            # 处理深度图
            depth_map = depth.detach().cpu().squeeze().numpy()  # [H, W]

            # 保存原始深度图（.npy 格式，用于后续处理）
            if args.save_depth_npy:
                depth_npy_path = os.path.join(output_dir, f'{camera.image_name}_depth.npy')
                np.save(depth_npy_path, depth_map)

            # # 保存深度可视化（彩色）
            if args.save_depth_vis:
                # 归一化深度图到 [0, 1]
                valid_mask = depth_map > 0
                if valid_mask.any():
                    depth_min = depth_map[valid_mask].min()
                    depth_max = depth_map[valid_mask].max()
                    depth_normalized = np.zeros_like(depth_map)
                    depth_normalized[valid_mask] = (depth_map[valid_mask] - depth_min) / (depth_max - depth_min + 1e-7)

                    # 转换为 uint8
                    depth_uint8 = (depth_normalized * 255).astype(np.uint8)

                    # 应用颜色映射
                    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)

                    depth_vis_path = os.path.join(output_dir, f'{camera.image_name}_depth_vis.png')
                    cv2.imwrite(depth_vis_path, depth_colored)

            # 打印深度统计信息（仅第一个视角）
            if i == 0:
                valid_depth = depth_map[depth_map > 0]
                if len(valid_depth) > 0:
                    print(f"\n  深度统计 (首个视角):")
                    print(f"    最小值: {valid_depth.min():.4f}")
                    print(f"    最大值: {valid_depth.max():.4f}")
                    print(f"    平均值: {valid_depth.mean():.4f}")
                    print(f"    有效像素: {len(valid_depth)} / {depth_map.size}")

        except Exception as e:
            print(f"\n   渲染视角 {camera.image_name} 失败: {e}")
            continue

    print("\n" + "="*60)
    print(" 渲染完成！")
    print(f"输出目录: {output_dir}")
    print("="*60)

    # 列出输出文件
    output_files = os.listdir(output_dir)
    print(f"\n生成的文件 ({len(output_files)} 个):")
    file_types = {}
    for f in output_files:
        ext = os.path.splitext(f)[1]
        file_types[ext] = file_types.get(ext, 0) + 1

    for ext, count in sorted(file_types.items()):
        print(f"  {ext}: {count} 个文件")


if __name__ == "__main__":
    main()
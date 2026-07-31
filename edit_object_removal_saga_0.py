'''
SAGA 项目 - 删除单个实例的高斯点并保存为新 PLY
用于物体移除前的预处理
'''
import json
import torch
import argparse
import os
import re
import numpy as np

# 导入 SAGA 项目核心模块
from scene import GaussianModel


def main():
    parser = argparse.ArgumentParser(description="SAGA - 删除单个实例的高斯点")
    parser.add_argument('--data_path', type=str, required=True, help='数据根路径')
    parser.add_argument('--ply_path', type=str, default=None, help='输入 PLY 文件路径（可选，自动查找最新）')
    parser.add_argument('--outputjson_path', type=str, default=None, help='output.json 路径（可选）')
    parser.add_argument('--instance_id', type=int, required=True,
                       help='要删除的实例 ID，如 --instance_id 5')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录（可选）')

    args = parser.parse_args()

    # ========== 1. 路径设置 ==========
    data_path = args.data_path
    ply_path = args.ply_path
    outputjson_path = args.outputjson_path
    instance_id = args.instance_id

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

    # 输出目录：inpaint/remove_{id}/output.ply
    if args.output_dir is None:
        output_dir = os.path.join(data_path, 'inpaint', str(instance_id))
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    output_ply_path = os.path.join(output_dir, 'output.ply')

    print("="*60)
    print("SAGA 高斯点删除工具")
    print("="*60)
    print(f"数据路径: {data_path}")
    print(f"输入 PLY: {ply_path}")
    print(f"输出 PLY: {output_ply_path}")
    print(f" 要删除的实例 ID: {instance_id}")
    print("="*60)

    # ========== 2. 加载 output.json ==========
    print("\n[1/3] 加载实例标签...")
    if not os.path.exists(outputjson_path):
        raise FileNotFoundError(f"output.json 不存在: {outputjson_path}")

    with open(outputjson_path, 'r') as f:
        data = json.load(f)

    point_labels = np.array(data['point_labels'])
    instances = data['instances']

    print(f"  总点数: {len(point_labels)}")
    print(f"  总实例数: {len(instances)}")

    # 验证实例 ID
    inst_id_str = str(instance_id)
    if inst_id_str not in instances:
        raise ValueError(f"实例 ID {instance_id} 在 output.json 中未找到")

    class_name = instances[inst_id_str].get('class', 'unknown')
    num_points_to_delete = np.sum(point_labels == instance_id)

    print(f"   实例 {instance_id} ({class_name})")
    print(f"     包含点数: {num_points_to_delete}")

    if num_points_to_delete == 0:
        raise ValueError(f"实例 {instance_id} 不包含任何高斯点")

    # ========== 3. 加载 3DGS 模型 ==========
    print("\n[2/3] 加载高斯模型...")
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

    total_points = gs_model.get_xyz.shape[0]
    print(f"  已加载: {total_points} 个高斯点")
    print(f"  SH 阶数: {gs_model.active_sh_degree}")

    # ========== 4. 创建保留掩码 ==========
    print("\n[3/3] 删除实例的高斯点...")
    point_labels_tensor = torch.tensor(point_labels, device='cuda')

    # 创建删除掩码
    delete_mask = (point_labels_tensor == instance_id)
    keep_mask = ~delete_mask

    # 统计信息
    num_delete = delete_mask.sum().item()
    num_keep = keep_mask.sum().item()

    print(f"  原始点数: {total_points}")
    print(f"  删除点数: {num_delete} ({num_delete/total_points*100:.2f}%)")
    print(f"  保留点数: {num_keep} ({num_keep/total_points*100:.2f}%)")

    if num_keep == 0:
        raise ValueError("所有点都被删除，无法保存空模型")

    # ========== 5. 过滤高斯点 ==========
    print("\n  正在过滤高斯点属性...")

    # 获取所有需要过滤的属性
    filtered_attrs = {}

    # 基本属性
    filtered_attrs['_xyz'] = gs_model._xyz[keep_mask]
    filtered_attrs['_features_dc'] = gs_model._features_dc[keep_mask]
    filtered_attrs['_features_rest'] = gs_model._features_rest[keep_mask]
    filtered_attrs['_opacity'] = gs_model._opacity[keep_mask]
    filtered_attrs['_scaling'] = gs_model._scaling[keep_mask]
    filtered_attrs['_rotation'] = gs_model._rotation[keep_mask]

    # 其他可能的属性
    if hasattr(gs_model, '_mask') and gs_model._mask is not None:
        filtered_attrs['_mask'] = gs_model._mask[keep_mask]

    if hasattr(gs_model, '_objects_dc') and gs_model._objects_dc is not None:
        filtered_attrs['_objects_dc'] = gs_model._objects_dc[keep_mask]

    print(f"   过滤完成")

    # ========== 6. 创建新的高斯模型 ==========
    print("\n  创建新的高斯模型...")
    new_gs_model = GaussianModel(sh_degree=gs_model.active_sh_degree)

    # 设置过滤后的属性
    for attr_name, attr_value in filtered_attrs.items():
        setattr(new_gs_model, attr_name, torch.nn.Parameter(attr_value.detach().requires_grad_(False)))

    # 更新计数
    new_gs_model.num_objects = num_keep

    print(f"   新模型创建完成: {new_gs_model.get_xyz.shape[0]} 个点")

    # ========== 7. 保存新 PLY ==========
    print(f"\n  保存到: {output_ply_path}")
    new_gs_model.save_ply(output_ply_path)
    print(f"   保存成功")

    # ========== 8. 输出统计信息 ==========
    print("\n" + "="*60)
    print(" 处理完成！")
    print("="*60)
    print(f"输出文件: {output_ply_path}")
    print(f"文件大小: {os.path.getsize(output_ply_path) / 1024 / 1024:.2f} MB")
    print(f"\n删除统计:")
    print(f"  实例 ID: {instance_id} ({class_name})")
    print(f"  原始点数: {total_points}")
    print(f"  删除点数: {num_delete}")
    print(f"  保留点数: {num_keep}")
    print(f"  删除比例: {num_delete/total_points*100:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
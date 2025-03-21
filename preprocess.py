import os
import sys
import cv2
from tqdm import tqdm
import shutil
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('preprocess parser')
    parser.add_argument('--source_path', '-s', type=str, required=True)
    parser.add_argument('--output_path', '-o', type=str, default='images')
    args = parser.parse_args(sys.argv[1:])
    source_path = args.source_path
    output_path = args.output_path
    images_path = os.path.join(source_path, 'data')
    output_images_path = os.path.join(output_path, 'images')
    os.makedirs(output_images_path, mode=0o777, exist_ok=True)
    point_cloud_path = os.path.join(source_path, 'output_models/point_cloud/iteration_30000/point_cloud.ply')
    output_point_cloud_path = os.path.join(output_path, 'point_cloud/iteration_30000/scene_point_cloud.ply')
    os.makedirs(os.path.dirname(output_point_cloud_path), mode=0o777, exist_ok=True)
    sparse_path = os.path.join(source_path, 'dense', 'sparse')
    output_sparse_path = os.path.join(output_path, 'sparse')
    cfg_path = os.path.join(source_path, 'output_models/cfg_args')
    output_cfg_path = os.path.join(output_path, 'cfg_args')

    shutil.copy(point_cloud_path, output_point_cloud_path)
    shutil.copytree(sparse_path, output_sparse_path)
    shutil.copy(cfg_path, output_cfg_path)

    for file_name in tqdm(os.listdir(images_path)):
        if not file_name.endswith('.jpg'):
            continue
        shutil.copy(os.path.join(images_path, file_name), os.path.join(output_images_path, file_name))

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
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)
from transformers import CLIPProcessor, CLIPModel

class Config:
    scale = 1.0

    model_path = ''
    source_path = ''

    @property
    def scale_gate_path(self):
        return os.path.join(self.model_path, f'point_cloud/iteration_10000/scale_gate.pt')
    @property
    def feature_pcd_path(self):
        return os.path.join(self.model_path, f'point_cloud/iteration_10000/contrastive_feature_point_cloud.ply')
    @property
    def scene_pcd_path(self):
        return os.path.join(self.model_path, f'point_cloud/iteration_30000/scene_point_cloud.ply')

    data_device = 'cpu'

    sh_degree = 3
    feature_dim = 32
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    resolution = 1
    debug = False
    convert_SHs_python = False
    compute_cov3D_python = False

    classes = ['chair', 'table', 'plant', 'wall', 'floor', 'ceiling', 'person']

cfg = Config()
parser = ArgumentParser(description="Training script parameters")
parser.add_argument("--source_path", '-s', type=str, required=True)
parser.add_argument("--model_path", '-m', type=str, required=True)
parser.add_argument("--sh_degree", type=int, default=3)
parser.add_argument("--k", type=int, default=128)
parser.add_argument("--classes", nargs="+", type=str, default=['chair', 'table', 'plant', 'wall', 'floor', 'ceiling', 'person'])
args = parser.parse_args(sys.argv[1:])
cfg.model_path = args.model_path
cfg.source_path = args.source_path
cfg.sh_degree = args.sh_degree
cfg.classes = args.classes
torch.manual_seed(0)
# sam = sam_model_registry['vit_h']('./third_party/segment-anything/weights/sam_vit_h_4b8939.pth').to('cuda')
# mask_predictor = SamPredictor(sam)
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to('cuda')
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

gs_model = GaussianModel(cfg.sh_degree)
gs_model.load_ply(cfg.scene_pcd_path)
feat_gs_model = FeatureGaussianModel(cfg.feature_dim)
feat_gs_model.load_ply(cfg.feature_pcd_path)
scale_gate = torch.nn.Sequential(
        torch.nn.Linear(1, cfg.feature_dim, bias=True),
        torch.nn.Sigmoid()
    ).cuda()
scale_gate.load_state_dict(torch.load(cfg.scale_gate_path))
try:
    cameras = readColmapCameras(read_extrinsics_binary(os.path.join(cfg.source_path, 'sparse/0/images.bin')), 
                                read_intrinsics_binary(os.path.join(cfg.source_path, 'sparse/0/cameras.bin')), 
                                os.path.join(cfg.source_path, 'images'))
except:
    cameras = readColmapCameras(read_extrinsics_text(os.path.join(cfg.source_path, 'sparse/0/images.txt')), 
                                read_intrinsics_text(os.path.join(cfg.source_path, 'sparse/0/cameras.txt')), 
                                os.path.join(cfg.source_path, 'images'))
camera_list = cameraList_from_camInfos(cameras, 1, cfg)

point_features = feat_gs_model.get_point_features
point_xyz = feat_gs_model.get_xyz
gates = scale_gate(torch.tensor([cfg.scale]).cuda()).unsqueeze(0)
print(f'{point_features.shape=}, {point_xyz.shape=}')

scale_conditioned_point_features = F.normalize(point_features, dim = -1, p = 2) * gates

normed_point_features = F.normalize(scale_conditioned_point_features, dim = -1, p = 2)
sampled_index = torch.rand(normed_point_features.shape[0]) > 0.98
normed_sampled_point_features = normed_point_features[sampled_index]

clusterer = HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=0.01, allow_single_cluster = False) # HDBSCAN

start_time = datetime.now()
cluster_labels = clusterer.fit_predict(normed_sampled_point_features.detach().cpu().numpy())
end_time = datetime.now()
elapsed_time = end_time - start_time
print(f'{elapsed_time=}') # 3

cluster_centers = torch.zeros(len(np.unique(cluster_labels)), normed_sampled_point_features.shape[-1])
for i in range(0, len(np.unique(cluster_labels))):
    cluster_centers[i] = F.normalize(normed_sampled_point_features[cluster_labels == i-1].mean(dim = 0), dim = -1)

seg_score = torch.einsum('nc,bc->bn', cluster_centers.cpu(), normed_point_features.cpu())
point_labels = seg_score.argmax(dim = -1)
print(f'HDBSCAN finish')
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
start_time = datetime.now()
if args.k>0:
    point_labels = filter3d(point_xyz, point_labels, args.k)
end_time = datetime.now()
elapsed_time = end_time - start_time
print(f'{elapsed_time=}') # 89, pytorch3d.ops.knn_points=49
print(f'knn finish')
torch.save(point_labels, os.path.join(cfg.model_path, 'point_labels.pth'))
point_labels = torch.load(os.path.join(cfg.model_path, 'point_labels.pth'))

vote = {instance: [0 for _ in range(len(args.classes)+1)] for instance in torch.unique(point_labels).tolist()} 
for i, camera in enumerate(camera_list):
    if not os.path.exists(os.path.join(cfg.source_path, 'sam_masks', f'{camera.image_name}.pt')):
        continue
    masks = torch.load(os.path.join(cfg.source_path, 'sam_masks', f'{camera.image_name}.pt')).float()
    masks = torch.nn.functional.interpolate(masks.unsqueeze(1), mode = 'bilinear', size = (camera.image_height, camera.image_width), align_corners = False).squeeze(1)
    masks[masks>0.5] = 1
    masks[masks!=1] = 0
    masks = masks.bool()
    labels = torch.load(os.path.join(cfg.source_path, 'labels', f'{camera.image_name}.pt'))
    render_pkg = render_with_max_contributor(camera, gs_model, cfg, cfg.bg_color)
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


output = dict()
output['point_labels'] = point_labels.tolist()
output['instances'] = {instance: {'class': [*cfg.classes, 'background'][vote[instance].index(max(vote[instance]))]} for instance in torch.unique(point_labels).tolist()}
with open(os.path.join(cfg.model_path, 'output.json'),'w') as f:
    json.dump(output,f)
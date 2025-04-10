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
parser.add_argument("--classes", nargs="+", type=str, default=['chair', 'table', 'plant', 'flower', 'foliage', 'tv', 'painting', 'sofa', 'cabinet', 'bed', 'wall', 'floor', 'ceiling', 'person'])
args = parser.parse_args(sys.argv[1:])
bg_color = torch.tensor([1,1,1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
torch.manual_seed(42)
# sam = sam_model_registry['vit_h']('./third_party/segment-anything/weights/sam_vit_h_4b8939.pth').to('cuda')
# mask_predictor = SamPredictor(sam)
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to('cuda')
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

gs_model = GaussianModel(args.sh_degree)
gs_model.load_ply(args.point_cloud_path)
feat_gs_model = FeatureGaussianModel(args.feature_dim)
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
point_xyz = feat_gs_model.get_xyz.detach().cpu()
point_scales = feat_gs_model.get_scaling.detach().cpu()
is_big_gaussian = point_scales.max(dim=-1).values>point_scales.max(dim=-1).values.median()*args.scale_threshold
point_opacities = feat_gs_model.get_opacity.detach().cpu().squeeze()
is_transparent_gaissian = point_opacities<args.opcity_threshold
gates = scale_gate(torch.tensor([args.scale]).cuda()).unsqueeze(0).detach().cpu()
print(f'{point_features.shape=}, {point_xyz.shape=}')

sampled_mask = uniform_sample(point_xyz, args.sample_num)
# sampled_mask = torch.rand(point_features.shape[0]) > 0.99

scale_conditioned_point_features = F.normalize(point_features, dim = -1, p = 2) * gates
normed_point_features = F.normalize(scale_conditioned_point_features, dim = -1, p = 2)
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
print(f'{elapsed_time=}, {len(torch.unique(point_labels))=}') # 89, pytorch3d.ops.knn_points=49
print(f'knn finish')
# torch.save(point_labels, os.path.join(args.model_path, 'point_labels.pth'))
# point_labels = torch.load(os.path.join(args.model_path, 'point_labels.pth'))

vote = {instance: [0 for _ in range(len(args.classes)+1)] for instance in torch.unique(point_labels).tolist()} 
for i, camera in tqdm(list(enumerate(camera_list))):
    with open(args.progress_path, 'w') as f:
        f.write(str(75+(i+1)*25//len(camera_list)))
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

output = dict()
output['point_labels'] = point_labels.tolist()
output['is_big_gaussian'] = is_big_gaussian.tolist()
output['is_transparent_gaissian'] = is_transparent_gaissian.tolist()
output['instances'] = {instance: {'class': get_class(args.classes, ratio)} for instance, ratio in instance_ratio.items()}
output['instances'] = {k: v for k, v in output['instances'].items() if v.get('class') in ['chair', 'table', 'plant', 'flower', 'foliage', 'tv', 'painting', 'sofa', 'cabinet', 'bed']}
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

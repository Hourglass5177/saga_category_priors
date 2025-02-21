import torch
from scene import Scene
import os
from datetime import datetime
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_contrastive_feature
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
# from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import cv2
from sklearn.decomposition import PCA
import torch.nn.functional as F

# from scene.gaussian_model import GaussianModel
from scene import Scene, GaussianModel, FeatureGaussianModel
import dearpygui.dearpygui as dpg
import math
from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from scene.dataset_readers import readColmapCameras, read_extrinsics_binary, read_intrinsics_binary, read_extrinsics_text, read_intrinsics_text
from utils.camera_utils import cameraList_from_camInfos
from utils.visualization_utils import save_image

from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
import pytorch3d.ops
from hdbscan import HDBSCAN
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)
from transformers import CLIPProcessor, CLIPModel

class Config:
    scale = 1.0

    model_path = 'output/temp/tys'
    data_path = 'data/temp/tys'
    scale_gate_path = os.path.join(model_path, f'point_cloud/iteration_10000/scale_gate.pt')
    feature_pcd_path = os.path.join(model_path, f'point_cloud/iteration_10000/contrastive_feature_point_cloud.ply')
    scene_pcd_path = os.path.join(model_path, f'point_cloud/iteration_30000/scene_point_cloud.ply')

    data_device = 'cpu'

    sh_degree = 3
    feature_dim = 32
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    resolution = 1
    debug = False
    convert_SHs_python = False
    compute_cov3D_python = False

    label_to_color = np.random.rand(1000, 3)

cfg = Config()

sam = sam_model_registry['vit_h']('./third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth').to('cuda')
mask_predictor = SamPredictor(sam)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to('cuda')
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

gs_model = GaussianModel(cfg.sh_degree)
gs_model.load_ply(cfg.scene_pcd_path)
feat_gs_model = FeatureGaussianModel(cfg.feature_dim)
feat_gs_model.load_ply(cfg.feature_pcd_path)
scale_gate = torch.nn.Sequential(
        torch.nn.Linear(1, cfg.feature_dim, bias=True),
        torch.nn.Sigmoid()
    ).cuda()
scale_gate.load_state_dict(torch.load(cfg.scale_gate_path))
# cameras = readColmapCameras(read_extrinsics_binary(os.path.join(cfg.data_path, 'sparse/0/images.bin')), 
#                             read_intrinsics_binary(os.path.join(cfg.data_path, 'sparse/0/cameras.bin')), 
#                             os.path.join(cfg.data_path, 'images'))
cameras = readColmapCameras(read_extrinsics_text(os.path.join(cfg.data_path, 'sparse/0/images.txt')), 
                            read_intrinsics_text(os.path.join(cfg.data_path, 'sparse/0/cameras.txt')), 
                            os.path.join(cfg.data_path, 'images'))
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
def filter3d(pos, label):
    print('begin filter3d')
    assert pos.shape[0] == label.shape[0]
    pos=pos.detach().cpu().numpy()
    label=label.detach().cpu().numpy()
    new_label = []
    kdtree = KDTree(pos)
    for i,p in enumerate(pos):
        d, index = kdtree.query(x=p, k=512)
        assert i == index[0]
        # print(f'query index {index[1:]} for {index[0]}')
        # print(f'query label {label[index[1:]].tolist()} for {label[index[0]].tolist()}')
        index = index[1:]
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
point_labels = filter3d(point_xyz, point_labels)
end_time = datetime.now()
elapsed_time = end_time - start_time
print(f'{elapsed_time=}') # 89, pytorch3d.ops.knn_points=49
print(f'knn finish')
torch.save(point_labels, 'temp/tys/point_labels.pth')
point_labels = torch.load('temp/tys/point_labels.pth')
cluster_point_colors = cfg.label_to_color[point_labels.detach().cpu().numpy()]

# extract langauge features
def mask_to_bbox(mask: torch.Tensor) -> torch.Tensor:
    """
    根据二值掩码生成边界框 (XYXY格式)。
    
    :param mask: 输入二值掩码 (torch.Tensor)，形状为 (H, W),值为0或1。
    :return: 边界框 (torch.Tensor)，格式为 [x_min, y_min, x_max, y_max]。
    """
    if mask.dim() != 2:
        raise ValueError("输入掩码必须是二维的 (H, W)")

    # 找到掩码中值为1的位置
    y_indices, x_indices = torch.where(mask > 0)

    if len(y_indices) == 0 or len(x_indices) == 0:
        # 如果没有前景像素，返回一个空的bbox
        return torch.tensor([0, 0, 0, 0], dtype=torch.float32)

    # 计算边界框
    x_min = x_indices.min().item()
    x_max = x_indices.max().item()
    y_min = y_indices.min().item()
    y_max = y_indices.max().item()

    return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

def get_entity_image(image: np.ndarray, mask: np.ndarray)->np.ndarray:
    def get_bbox(mask: np.ndarray):
        # 查找掩码中的 True 元素的索引
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        # 如果没有 True 元素，则返回全零的边界框
        if not np.any(rows) or not np.any(cols):
            return (0, 0, 0, 0)
        
        # 获取边界框的上下左右边界
        x_min, x_max = np.where(rows)[0][[0, -1]] # h
        y_min, y_max = np.where(cols)[0][[0, -1]] # w
        
        # 返回边界框
        return (x_min, y_min, x_max + 1 - x_min, y_max + 1 - y_min) # x, y, h, w
    if mask.sum()==0:
        return np.zeros((224,224,3), dtype=np.uint8)
    image = image.copy()
    # crop by bbox
    x,y,h,w = get_bbox(mask)
    image[~mask] = np.zeros(3, dtype=np.uint8) #分割区域外为白色
    image = image[x:x+h, y:y+w, ...] #将img按分割区域bbox裁剪
    # pad to square
    l = max(h,w)
    paded_img = np.zeros((l, l, 3), dtype=np.uint8)
    if h > w:
        paded_img[:,(h-w)//2:(h-w)//2 + w, :] = image
    else:
        paded_img[(w-h)//2:(w-h)//2 + h, :, :] = image
    paded_img = cv2.resize(paded_img, (224,224))
    return paded_img

start_time = datetime.now()
langauge_features = []
for label in tqdm(list(range(0, len(np.unique(cluster_labels)))), desc='get langauge feature'):
    features = []
    for i, camera in enumerate(camera_list):
        render_pkg = render(camera, gs_model, cfg, cfg.bg_color, filtered_mask=~(point_labels==label))
        if torch.logical_and(render_pkg['visibility_filter'], (point_labels==label)).sum()/(point_labels==label).sum() > 0.9: # valid camera
            render_image = render_pkg['render']
            original_image = camera.original_image.to('cuda')
            prompt_mask = (render_image!=0).any(dim=0)
            prompt_bbox = torch.Tensor(mask_to_bbox(prompt_mask)).to('cuda')
            transformed_bbox = mask_predictor.transform.apply_boxes_torch(prompt_bbox, original_image.shape[-2:])
            batched_image = original_image.unsqueeze(0)
            transformed_image = mask_predictor.transform.apply_image_torch(batched_image)
            mask_predictor.set_torch_image(transformed_image, original_image.shape[-2:])
            masks, scores, _ = mask_predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_bbox, multimask_output=False)

            inputs = clip_processor(images=get_entity_image((original_image*255).permute(1,2,0).to('cpu', torch.uint8).numpy()*masks[0,0,...][None,...].permute(1,2,0).cpu().numpy(), masks[0,0].cpu().numpy()), return_tensors='pt')
            inputs = inputs.to(clip_model.device)
            semantic0 = clip_model.get_image_features(**inputs)
            semantic0 = F.normalize(semantic0,dim=-1).detach().cpu()
        else:
            semantic0 = torch.zeros((1,512), dtype=torch.float32, device='cpu')
        features.append(semantic0)
    langauge_features.append(torch.concat(features, dim=0))
langauge_features = torch.stack(langauge_features, dim=0)
end_time = datetime.now()
elapsed_time = end_time - start_time
print(f'{elapsed_time=}') # 2796
print(f'{langauge_features.shape=}') # float[labels, cameras, clip_dim]
torch.save(langauge_features, 'temp/tys/langauge_features.pth')
langauge_features = torch.load('temp/tys/langauge_features.pth')

def get_relevancy(raw_semantic_map: torch.Tensor, pembed: torch.Tensor, nembed: torch.Tensor):
    s = raw_semantic_map.shape[:-1]
    c = raw_semantic_map.shape[-1]
    raw_semantics = raw_semantic_map.flatten(0, -2)
    psim=pembed@raw_semantics.T # (p, i)
    nsim=nembed@raw_semantics.T # (n, i)
    nsim=nsim.unsqueeze(0).repeat_interleave(pembed.shape[0],dim=0) # (p, n ,i)
    psim=psim.unsqueeze(1).repeat_interleave(nembed.shape[0],dim=1) # (p, n, i)
    sim=torch.stack((psim,nsim), dim=-1) # (p, n, i, 2)
    sim=torch.softmax(10*sim, dim=-1) # (p, n, i, 2)
    sim, indice = sim[...,0].min(dim=1) # (p, i)
    return sim.unflatten(1, s)
ptexts = ['plant', 'sofa', 'pillow', 'chair', 'table']
ntexts = ["object", "things", "stuff", "texture"]
nembed = clip_processor(text=ntexts, return_tensors='pt', padding=True)
nembed = nembed.to(clip_model.device)
nembed = clip_model.get_text_features(**nembed)
nembed = F.normalize(nembed, dim=-1)
nembed = nembed.detach().cpu()
pembed = clip_processor(text=ptexts, return_tensors='pt', padding=True)
pembed = pembed.to(clip_model.device)
pembed = clip_model.get_text_features(**pembed)
pembed = F.normalize(pembed, dim=-1)
pembed = pembed.detach().cpu()
sims = get_relevancy(langauge_features, pembed, nembed) # float[p, e, c]
for ptext, sim in zip(ptexts, sims):
    entity_index,camera_index = torch.unravel_index(sim.argmax(), sim.shape)
    render_pkg = render(camera_list[camera_index], gs_model, cfg, cfg.bg_color, filtered_mask=~(point_labels==entity_index))
    save_image(render_pkg['render'], f'temp/tys/{ptext}.jpg', dataformat='CHW')
import torch
import os
import json
import sys
from tqdm import tqdm
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
# from gaussian_renderer import GaussianModel

# from scene.gaussian_model import GaussianModel
from scene import GaussianModel
from scene.dataset_readers import readColmapCameras, read_extrinsics_binary, read_intrinsics_binary, read_extrinsics_text, read_intrinsics_text
from utils.camera_utils import cameraList_from_camInfos
from utils.visualization_utils import save_image
from torchvision.utils import save_image


parser = ArgumentParser(description="Training script parameters")
lp = ModelParams(parser)
pp = PipelineParams(parser)
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument("--render_path", type=str, required=True)
args = parser.parse_args(sys.argv[1:])
bg_color = torch.tensor([1,1,1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
torch.manual_seed(42)

gs_model = GaussianModel(args.sh_degree)
gs_model.load_ply(args.point_cloud_path)
try:
    cameras = readColmapCameras(read_extrinsics_binary(os.path.join(args.sparse_path, 'images.bin')), 
                                read_intrinsics_binary(os.path.join(args.sparse_path, 'cameras.bin')), 
                                args.images_path)
except:
    cameras = readColmapCameras(read_extrinsics_text(os.path.join(args.sparse_path, 'images.txt')), 
                                read_intrinsics_text(os.path.join(args.sparse_path, 'cameras.txt')), 
                                args.images_path)
camera_list = cameraList_from_camInfos(cameras, 1, args)

with open(args.json_path,'r') as f:
    output = json.load(f)
point_labels = torch.tensor(output['point_labels'], device='cuda')
color_map = torch.rand((1000,3), device='cuda')
precompute_color = color_map[point_labels]
smooth_weights=None

os.makedirs(args.render_path, exist_ok=True)
for i, camera in tqdm(list(enumerate(camera_list))):
    render_pkg = render(camera, gs_model, args, bg_color, override_color = precompute_color)
    image = render_pkg["render"]
    save_image(image, os.path.join(args.render_path, f'{camera.image_name}.jpg'))

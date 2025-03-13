# Borrowed from OmniSeg3D-GS (https://github.com/OceanYing/OmniSeg3D-GS)
import torch
from scene import Scene
import os
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
from utils.clip_utils import get_relevancy

from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree

# from cuml.cluster.hdbscan import HDBSCAN
from hdbscan import HDBSCAN
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)
from transformers import CLIPProcessor, CLIPModel

def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min() + 1e-7)
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)
    return depth_img

class CONFIG:
    r = 2   # scale ratio
    window_width = int(2160/r)
    window_height = int(1200/r)

    width = int(2160/r)
    height = int(1200/r)

    radius = 2

    debug = False
    dt_gamma = 0.2

    resolution = 1
    data_device = 'cpu'

    # gaussian model
    sh_degree = 3

    convert_SHs_python = False
    compute_cov3D_python = False

    white_background = False

    feature_dim = 32
    model_path = None # 30000
    source_path = None

    feature_gaussian_iteration = 10000
    scene_gaussian_iteration = 30000
    @property
    def scale_gate_path(self):
        return os.path.join(self.model_path, f'point_cloud/iteration_{str(self.feature_gaussian_iteration)}/scale_gate.pt')
    @property
    def feature_pcd_path(self):
        return os.path.join(self.model_path, f'point_cloud/iteration_{str(self.feature_gaussian_iteration)}/contrastive_feature_point_cloud.ply')
    @property
    def scene_pcd_path(self):
        return os.path.join(self.model_path, f'point_cloud/iteration_{str(self.scene_gaussian_iteration)}/scene_point_cloud.ply')
    @property
    def point_labels_path(self):
        return os.path.join(self.model_path, 'point_labels.pth')
    @property
    def langauge_features_path(self):
        return os.path.join(self.model_path, 'langauge_features.pth')
    @property
    def json_path(self):
        return os.path.join(self.model_path, 'output.json')
    pos_texts = ['chair', 'table', 'plant', 'wall', 'floor', 'ceiling', 'person']
    neg_texts = ["object", "things", "stuff", "texture"]


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_quat(
            [0, 0, 0, 1]
        )  # init camera matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!
        self.right = np.array([1, 0, 0], dtype=np.float32)  # need to be normalized!
        self.fovy = fovy
        self.translate = np.array([0, 0, self.radius])
        self.scale_f = 1.0


        self.rot_mode = 1   # rotation mode (1: self.pose_movecenter (movable rotation center), 0: self.pose_objcenter (fixed scene center))
        # self.rot_mode = 0


    @property
    def pose_movecenter(self):
        # --- first move camera to radius : in world coordinate--- #
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        
        # --- rotate: Rc --- #
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        # --- translate: tc --- #
        res[:3, 3] -= self.center
        
        # --- Convention Transform --- #
        # now we have got matrix res=c2w=[Rc|tc], but gaussian-splatting requires convention as [Rc|-Rc.T@tc]
        res[:3, 3] = -rot[:3, :3].transpose() @ res[:3, 3]
        
        return res
    
    @property
    def pose_objcenter(self):
        res = np.eye(4, dtype=np.float32)
        
        # --- rotate: Rw --- #
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        # --- translate: tw --- #
        res[2, 3] += self.radius    # camera coordinate z-axis
        res[:3, 3] -= self.center   # camera coordinate x,y-axis
        
        # --- Convention Transform --- #
        # now we have got matrix res=w2c=[Rw|tw], but gaussian-splatting requires convention as [Rc|-Rc.T@tc]=[Rw.T|tw]
        res[:3, :3] = rot[:3, :3].transpose()
        
        return res

    @property
    def opt_pose(self):
        # --- deprecated ! Not intuitive implementation --- #
        res = np.eye(4, dtype=np.float32)

        res[:3, :3] = self.rot.as_matrix()

        scale_mat = np.eye(4)
        scale_mat[0, 0] = self.scale_f      # why apply scale ratio to rotation matrix? It's confusing.
        scale_mat[1, 1] = self.scale_f
        scale_mat[2, 2] = self.scale_f

        transl = self.translate - self.center
        transl_mat = np.eye(4)
        transl_mat[:3, 3] = transl

        return transl_mat @ scale_mat @ res

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])

    def orbit(self, dx, dy):
        if self.rot_mode == 1:    # rotate the camera axis, in world coordinate system
            up = self.rot.as_matrix()[:3, 1]
            side = self.rot.as_matrix()[:3, 0]
        elif self.rot_mode == 0:    # rotate in camera coordinate system
            up = -self.up
            side = -self.right
        rotvec_x = up * np.radians(0.01 * dx)
        rotvec_y = side * np.radians(0.01 * dy)

        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        # self.radius *= 1.1 ** (-delta)    # non-linear version
        self.radius -= 0.1 * delta      # linear version

    def pan(self, dx, dy, dz=0):
        
        if self.rot_mode == 1:
            # pan in camera coordinate system: project from [Coord_c] to [Coord_w]
            self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])
        elif self.rot_mode == 0:
            # pan in world coordinate system: at [Coord_w]
            self.center += 0.0005 * np.array([-dx, dy, dz])


class GaussianSplattingGUI:
    def __init__(self, opt, gaussian_model:GaussianModel, feature_gaussian_model:FeatureGaussianModel, scale_gate: torch.nn.modules.container.Sequential) -> None:
        self.opt = opt

        self.width = opt.width
        self.height = opt.height
        self.window_width = opt.window_width
        self.window_height = opt.window_height
        self.camera = OrbitCamera(opt.width, opt.height, r=opt.radius)

        bg_color = [1, 1, 1] if opt.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        bg_feature = [0 for i in range(opt.feature_dim)]
        bg_feature = torch.tensor(bg_feature, dtype=torch.float32, device="cuda")

        self.bg_color = background
        self.bg_feature = bg_feature
        self.render_buffer = np.zeros((self.width, self.height, 3), dtype=np.float32)
        self.update_camera = True
        self.dynamic_resolution = True
        self.debug = opt.debug
        self.engine = {
            'scene': gaussian_model,
            'feature': feature_gaussian_model,
            'scale_gate': scale_gate
        }
        # try:
        #     self.cameras = readColmapCameras(read_extrinsics_binary(os.path.join(self.opt.DATA_PATH, 'sparse/0/images.bin')), 
        #                                     read_intrinsics_binary(os.path.join(self.opt.DATA_PATH, 'sparse/0/cameras.bin')), 
        #                                     os.path.join(self.opt.DATA_PATH, 'images'))
        # except:
        #     self.cameras = readColmapCameras(read_extrinsics_text(os.path.join(self.opt.DATA_PATH, 'sparse/0/images.txt')), 
        #                                     read_intrinsics_text(os.path.join(self.opt.DATA_PATH, 'sparse/0/cameras.txt')), 
        #                                     os.path.join(self.opt.DATA_PATH, 'images'))
        # self.camera_list = cameraList_from_camInfos(self.cameras, 1, self.opt)

        sam = sam_model_registry['vit_h']('./third_party/segment-anything/weights/sam_vit_h_4b8939.pth').to('cuda')
        # self.mask_generator = SamAutomaticMaskGenerator(
        #     model=sam,
        #     points_per_side=32,
        #     pred_iou_thresh=0.88,
        #     box_nms_thresh=0.7,
        #     stability_score_thresh=0.95,
        #     crop_n_layers=0,
        #     crop_n_points_downscale_factor=1,
        #     min_mask_region_area=100,
        # )
        self.mask_predictor = SamPredictor(sam)
        
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to('cuda')
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.nembed = self.clip_processor(text=self.opt.neg_texts, return_tensors='pt', padding=True)
        self.nembed = self.nembed.to(self.clip_model.device)
        self.nembed = self.clip_model.get_text_features(**self.nembed)
        self.nembed = F.normalize(self.nembed, dim=-1)
        self.nembed = self.nembed.detach().cpu()
        self.pembed = self.clip_processor(text=self.opt.pos_texts, return_tensors='pt', padding=True)
        self.pembed = self.pembed.to(self.clip_model.device)
        self.pembed = self.clip_model.get_text_features(**self.pembed)
        self.pembed = F.normalize(self.pembed, dim=-1)
        self.pembed = self.pembed.detach().cpu()

        self.cluster_point_colors = None
        self.label_to_color = np.random.rand(1000, 3)
        self.seg_score = None

        self.proj_mat = None

        self.load_model = False
        print("loading model file...")
        self.engine['scene'].load_ply(self.opt.scene_pcd_path)
        self.engine['feature'].load_ply(self.opt.feature_pcd_path)
        self.engine['scale_gate'].load_state_dict(torch.load(self.opt.scale_gate_path))
        self.do_pca()   # calculate self.proj_mat
        self.load_model = True

        print("loading model file done.")

        if opt.point_labels_path:
            self.point_labels = torch.load(opt.point_labels_path)
        if opt.langauge_features_path:
            self.langauge_features = torch.load(opt.langauge_features_path)

        self.mode = "image"  # choose from ['image', 'depth']

        dpg.create_context()
        self.register_dpg()

        self.frame_id = 0

        # --- for better operation --- #
        self.moving = False
        self.moving_middle = False
        self.mouse_pos = (0, 0)

        # --- for interactive segmentation --- #
        self.img_mode = 0
        self.clickmode_button = False
        self.clickmode_multi_button = False     # choose multiple object 
        self.new_click = False
        self.prompt_num = 0
        self.new_click_xy = []
        self.clear_edit = False                 # clear all the click prompts
        self.roll_back = False
        self.preview = False    # binary segmentation mode
        self.segment3d_flag = False
        self.reload_flag = False        # reload the whole scene / point cloud
        self.object_seg_id = 0          # to store the segmented object with increasing index order (path at: ./)
        self.cluster_in_3D_flag = False
        self.label = 0

        self.render_mode_rgb = False
        self.render_mode_similarity = False
        self.render_mode_pca = False
        self.render_mode_cluster = False
        self.render_mode_label = False

        self.save_flag = False
    def __del__(self):
        dpg.destroy_context()

    def prepare_buffer(self, outputs):
        if self.model == "images":
            return outputs["render"]
        else:
            return np.expand_dims(outputs["depth"], -1).repeat(3, -1)
    
    def grayscale_to_colormap(self, gray):
        """Convert a grayscale value to Jet colormap RGB values."""
        # Ensure the grayscale values are in the range [0, 1]
        # gray = np.clip(gray, 0, 1)

        # Jet colormap ranges (these are normalized to [0, 1])
        jet_colormap = np.array([
            [0, 0, 0.5],
            [0, 0, 1],
            [0, 0.5, 1],
            [0, 1, 1],
            [0.5, 1, 0.5],
            [1, 1, 0],
            [1, 0.5, 0],
            [1, 0, 0],
            [0.5, 0, 0]
        ])

        # Corresponding positions for the colors in the colormap
        positions = np.linspace(0, 1, jet_colormap.shape[0])

        # Interpolate the RGB values based on the grayscale value
        r = np.interp(gray, positions, jet_colormap[:, 0])
        g = np.interp(gray, positions, jet_colormap[:, 1])
        b = np.interp(gray, positions, jet_colormap[:, 2])

        return np.stack((r, g, b), axis=-1)

    def register_dpg(self):
        
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.width, self.height, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window
        with dpg.window(tag="_primary_window", width=self.window_width+300, height=self.window_height):
            dpg.add_image("_texture")   # add the texture

        dpg.set_primary_window("_primary_window", True)

        # def callback_depth(sender, app_data):
            # self.img_mode = (self.img_mode + 1) % 4
            
        # --- interactive mode switch --- #
        def clickmode_callback(sender):
            self.clickmode_button = 1 - self.clickmode_button
        def clickmode_multi_callback(sender):
            self.clickmode_multi_button = dpg.get_value(sender)
            print("clickmode_multi_button = ", self.clickmode_multi_button)
        def preview_callback(sender):
            self.preview = dpg.get_value(sender)
            # print("binary_threshold_button = ", self.binary_threshold_button)
        def clear_edit():
            self.clear_edit = True
        def roll_back():
            self.roll_back = True
        def callback_segment3d():
            self.segment3d_flag = True
        def callback_save():
            self.save_flag = True
        def callback_reload():
            self.reload_flag = True
        def callback_cluster():
            self.cluster_in_3D_flag =True
        def callback_label(sender, app_data, user_data):
            self.label = (self.label+user_data)%len(torch.unique(self.point_labels))
            langauge_feature = self.langauge_features[self.label, ...]
            sims = get_relevancy(langauge_feature, self.pembed, self.nembed)
            sims, _ = torch.max(sims, dim=1)
            _, index = torch.max(sims, dim=0)
            dpg.set_value('label', f'{self.label}:{self.json['instance'][str(index.item())]}')
        def callback_reshuffle_color():
            self.label_to_color = np.random.rand(1000, 3)
            try:
                self.cluster_point_colors = self.label_to_color[self.seg_score.argmax(dim = -1).cpu().numpy()]
                self.cluster_point_colors[self.seg_score.max(dim = -1)[0].detach().cpu().numpy() < 0.5] = (0,0,0)
            except:
                pass

        def render_mode_rgb_callback(sender):
            self.render_mode_rgb = not self.render_mode_rgb
        def render_mode_similarity_callback(sender):
            self.render_mode_similarity = not self.render_mode_similarity
        def render_mode_pca_callback(sender):
            self.render_mode_pca = not self.render_mode_pca
        def render_mode_cluster_callback(sender):
            self.render_mode_cluster = not self.render_mode_cluster
        def render_mode_label_callback(sender):
            self.render_mode_label = not self.render_mode_label
        # control window
        with dpg.window(label="Control", tag="_control_window", width=300, height=550, pos=[self.window_width+10, 0]):

            dpg.add_text("Mouse position: click anywhere to start. ", tag="pos_item")
            dpg.add_slider_float(label="Scale", default_value=0.5,
                                 min_value=0.0, max_value=1.0, tag="_Scale")
            dpg.add_slider_float(label="ScoreThres", default_value=0.0,
                                 min_value=0.0, max_value=1.0, tag="_ScoreThres")
            # dpg.add_button(label="render_option", tag="_button_depth",
                            # callback=callback_depth)
            dpg.add_text("\nRender option: ", tag="render")
            dpg.add_checkbox(label="RGB", callback=render_mode_rgb_callback, user_data="Some Data")
            dpg.add_checkbox(label="PCA", callback=render_mode_pca_callback, user_data="Some Data")
            dpg.add_checkbox(label="SIMILARITY", callback=render_mode_similarity_callback, user_data="Some Data")
            dpg.add_checkbox(label="3D CLUSTER", callback=render_mode_cluster_callback, user_data="Some Data")
            dpg.add_checkbox(label="Label", callback=render_mode_label_callback, user_data="Some Data")
            

            dpg.add_text("\nSegment option: ", tag="seg")
            dpg.add_checkbox(label="clickmode", callback=clickmode_callback, user_data="Some Data")
            dpg.add_checkbox(label="multi-clickmode", callback=clickmode_multi_callback, user_data="Some Data")
            dpg.add_checkbox(label="preview_segmentation_in_2d", callback=preview_callback, user_data="Some Data")
            
            dpg.add_text("\n")
            dpg.add_button(label="segment3d", callback=callback_segment3d, user_data="Some Data")
            dpg.add_button(label="roll_back", callback=roll_back, user_data="Some Data")
            dpg.add_button(label="clear", callback=clear_edit, user_data="Some Data")
            dpg.add_button(label="save as", callback=callback_save, user_data="Some Data")
            dpg.add_input_text(label="", default_value="precomputed_mask", tag="save_name")
            dpg.add_text("\n")

            dpg.add_button(label="cluster3d", callback=callback_cluster, user_data="Some Data")
            dpg.add_input_text(label='label', tag='label')
            dpg.add_button(label="prev", callback=callback_label, user_data=-1)
            dpg.add_button(label="next", callback=callback_label, user_data=1)
            dpg.add_button(label="reshuffle_cluster_color", callback=callback_reshuffle_color, user_data="Some Data")
            dpg.add_button(label="reload_data", callback=callback_reload, user_data="Some Data")

            # def callback(sender, app_data, user_data):
            #     self.load_model = False
            #     file_data = app_data["selections"]
            #     file_names = []
            #     for key in file_data.keys():
            #         file_names.append(key)

            #     self.opt.ply_file = file_data[file_names[0]]

            #     # if not self.load_model:
            #     print("loading model file...")
            #     self.engine.load_ply(self.opt.ply_file)
            #     self.do_pca()   # calculate new self.proj_mat after loading new .ply file
            #     print("loading model file done.")
            #     self.load_model = True

        if self.debug:
            with dpg.collapsing_header(label="Debug"):
                dpg.add_separator()
                dpg.add_text("Camera Pose:")
                dpg.add_text(str(self.camera.pose), tag="_log_pose")


        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            delta = app_data
            self.camera.scale(delta*5)
            self.update_camera = True
            if self.debug:
                dpg.set_value("_log_pose", str(self.camera.pose))
        

        def toggle_moving_left():
            self.moving = not self.moving


        def toggle_moving_middle():
            self.moving_middle = not self.moving_middle


        def move_handler(sender, pos, user):
            if self.moving and dpg.is_item_focused("_primary_window"):
                dx = self.mouse_pos[0] - pos[0]
                dy = self.mouse_pos[1] - pos[1]
                if dx != 0.0 or dy != 0.0:
                    self.camera.orbit(-dx*10, dy*10)
                    self.update_camera = True

            if self.moving_middle and dpg.is_item_focused("_primary_window"):
                dx = self.mouse_pos[0] - pos[0]
                dy = self.mouse_pos[1] - pos[1]
                if dx != 0.0 or dy != 0.0:
                    self.camera.pan(-dx*5, dy*5)
                    self.update_camera = True
            
            self.mouse_pos = pos


        def change_pos(sender, app_data):
            # if not dpg.is_item_focused("_primary_window"):
            #     return
            xy = dpg.get_mouse_pos(local=False)
            dpg.set_value("pos_item", f"Mouse position = ({xy[0]}, {xy[1]})")
            if self.clickmode_button and app_data == 1:     # in the click mode and right click
                print(xy)
                self.new_click_xy = np.array(xy)
                self.new_click = True


        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Left, callback=lambda:toggle_moving_left())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Left, callback=lambda:toggle_moving_left())
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Middle, callback=lambda:toggle_moving_middle())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Middle, callback=lambda:toggle_moving_middle())
            dpg.add_mouse_move_handler(callback=lambda s, a, u:move_handler(s, a, u))
            
            dpg.add_mouse_click_handler(callback=change_pos)
            
        dpg.create_viewport(title="Gaussian-Splatting-Viewer", width=self.window_width+320, height=self.window_height, resizable=False)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        dpg.show_viewport()


    def render(self):
        while dpg.is_dearpygui_running():
            # update texture every frame
            # TODO : fetch rgb and depth
            if self.load_model:
                cam = self.construct_camera()
                self.fetch_data(cam)
            dpg.render_dearpygui_frame()


    def construct_camera(
        self,
    ) -> Camera:
        if self.camera.rot_mode == 1:
            pose = self.camera.pose_movecenter
        elif self.camera.rot_mode == 0:
            pose = self.camera.pose_objcenter

        R = pose[:3, :3]
        t = pose[:3, 3]

        ss = math.pi / 180.0
        fovy = self.camera.fovy * ss

        fy = fov2focal(fovy, self.height)
        fovx = focal2fov(fy, self.width)

        cam = Camera(
            colmap_id=0,
            R=R,
            T=t,
            FoVx=fovx,
            FoVy=fovy,
            image=torch.zeros([3, self.height, self.width]),
            gt_alpha_mask=None,
            image_name=None,
            uid=0,
        )
        cam.feature_height, cam.feature_width = self.height, self.width
        return cam
    
    def cluster_in_3D(self):
        # try:
        #     self.engine['scene'].roll_back()
        #     self.engine['feature'].roll_back()
        # except:
        #     pass
        # point_features = self.engine['feature'].get_point_features
        # point_xyz = self.engine['feature'].get_xyz
        # print(point_features.shape, point_xyz.shape)

        # scale_conditioned_point_features = torch.nn.functional.normalize(point_features, dim = -1, p = 2) * self.gates.unsqueeze(0)

        # normed_point_features = torch.nn.functional.normalize(scale_conditioned_point_features, dim = -1, p = 2)
        # # normed_point_xyz = point_xyz
        # # normed_point_xyz = torch.nn.functional.normalize(point_xyz, dim = 0, p = 2)
        # # normed_point_xyz = 20*point_xyz / (point_xyz.abs().max())
        # # normed_point_features = torch.cat((normed_point_features, normed_point_xyz), dim=-1)
        # sampled_index = torch.rand(normed_point_features.shape[0]) > 0.98
        # normed_sampled_point_features = normed_point_features[sampled_index]

        # # normed_sampled_point_features = sampled_point_features / torch.norm(sampled_point_features, dim = -1, keepdim = True)

        # clusterer = HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=0.01, allow_single_cluster = False) # HDBSCAN

        # cluster_labels = clusterer.fit_predict(normed_sampled_point_features.detach().cpu().numpy())

        # cluster_centers = torch.zeros(len(np.unique(cluster_labels)), normed_sampled_point_features.shape[-1])
        # for i in range(0, len(np.unique(cluster_labels))):
        #     cluster_centers[i] = torch.nn.functional.normalize(normed_sampled_point_features[cluster_labels == i-1].mean(dim = 0), dim = -1)

        # self.seg_score = torch.einsum('nc,bc->bn', cluster_centers.cpu(), normed_point_features.cpu())
        # self.point_ins_labels = self.seg_score.argmax(dim = -1)
        # def filter3d(pos, label):
        #     print('begin filter3d')
        #     assert pos.shape[0] == label.shape[0]
        #     pos=pos.detach().cpu().numpy()
        #     label=label.detach().cpu().numpy()
        #     new_label = []
        #     kdtree = KDTree(pos)
        #     for i,p in enumerate(pos):
        #         d, index = kdtree.query(x=p, k=512)
        #         assert i == index[0]
        #         # print(f'query index {index[1:]} for {index[0]}')
        #         # print(f'query label {label[index[1:]].tolist()} for {label[index[0]].tolist()}')
        #         index = index[1:]
        #         bin = []
        #         counts = []
        #         for l in label[index]:
        #             try:
        #                 counts[bin.index(l)]+=1
        #             except:
        #                 bin.append(l)
        #                 counts.append(1)
        #         # print(f'{bin}\n{counts}')
        #         new_label.append(bin[counts.index(max(counts))])
        #     print('finish filter3d')
        #     return torch.tensor(new_label).cuda()
        # self.point_ins_labels = filter3d(point_xyz, self.point_ins_labels)
        # self.cluster_point_colors = self.label_to_color[self.point_ins_labels.detach().cpu().numpy()]
        import json
        with open(self.opt.json_path, 'r') as f:
            self.json = json.load(f)
        self.cluster_point_colors = self.label_to_color[np.array(self.json['point_labels'])]
        # self.cluster_point_colors[self.seg_score.max(dim = -1)[0].detach().cpu().numpy() < 0.5] = (0,0,0)

        # extract langauge features
        def mask_to_bbox(mask: torch.Tensor) -> torch.Tensor:
            """
            根据二值掩码生成边界框 (XYXY格式)。
            
            :param mask: 输入二值掩码 (torch.Tensor)，形状为 (H, W)，值为0或1。
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
        
        # self.langauge_features = []
        # for label in range(0, len(np.unique(cluster_labels))):
        #     features = []
        #     for i, camera in enumerate(self.camera_list):
        #         render_pkg = render(camera, self.engine['scene'], self.opt, self.bg_color, filtered_mask=~(self.point_ins_labels==label))
        #         if torch.logical_and(render_pkg['visibility_filter'], (self.point_ins_labels==label)).sum()/(self.point_ins_labels==label).sum() > 0.9: # valid camera
        #             render_image = render_pkg['render']
        #             original_image = camera.original_image.to('cuda')
        #             prompt_mask = (render_image!=0).any(dim=0)
        #             prompt_bbox = torch.Tensor(mask_to_bbox(prompt_mask)).to('cuda')
        #             transformed_bbox = self.mask_predictor.transform.apply_boxes_torch(prompt_bbox, original_image.shape[-2:])
        #             batched_image = original_image.unsqueeze(0)
        #             transformed_image = self.mask_predictor.transform.apply_image_torch(batched_image)
        #             self.mask_predictor.set_torch_image(transformed_image, original_image.shape[-2:])
        #             masks, scores, _ = self.mask_predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_bbox, multimask_output=False)

        #             inputs = self.clip_processor(images=get_entity_image((original_image*255).permute(1,2,0).to('cpu', torch.uint8).numpy()*masks[0,0,...][None,...].permute(1,2,0).cpu().numpy(), masks[0,0].cpu().numpy()), return_tensors='pt')
        #             inputs = inputs.to(self.clip_model.device)
        #             semantic0 = self.clip_model.get_image_features(**inputs)
        #             semantic0 = F.normalize(semantic0,dim=-1).detach().cpu()
        #         else:
        #             semantic0 = torch.zeros((1,512), dtype=torch.float32, device='cpu')
        #         features.append(semantic0)
        #     self.langauge_features.append(torch.concat(features, dim=0))
        # self.langauge_features = torch.stack(self.langauge_features, dim=0)
        # torch.save(self.langauge_features, f'{self.opt.langauge_features_path}')
        self.langauge_features = torch.load(f'{self.opt.langauge_features_path}')


    def pca(self, X, n_components=3):
        n = X.shape[0]
        mean = torch.mean(X, dim=0)
        X = X - mean
        covariance_matrix = (1 / n) * torch.matmul(X.T, X).float()  # An old torch bug: matmul float32->float16, 
        eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)
        eigenvalues = torch.norm(eigenvalues, dim=1)
        idx = torch.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        proj_mat = eigenvectors[:, 0:n_components]
        
        return proj_mat
    

    def do_pca(self):
        sems = self.engine['feature'].get_point_features.clone().squeeze()
        N, C = sems.shape
        torch.manual_seed(0)
        randint = torch.randint(0, N, [200_000])
        sems /= (torch.norm(sems, dim=1, keepdim=True) + 1e-6)
        sem_chosen = sems[randint, :]
        self.proj_mat = PCA(n_components=3)
        self.proj_mat.fit(sem_chosen.detach().cpu())
        print("project mat initialized !")


    @torch.no_grad()
    def fetch_data(self, view_camera):
        
        scene_outputs = render(view_camera, self.engine['scene'], self.opt, self.bg_color)
        feature_outputs = render_contrastive_feature(view_camera, self.engine['feature'], self.opt, self.bg_feature)
        if self.cluster_in_3D_flag:
            self.cluster_in_3D_flag = False
            print("Clustering in 3D...")
            self.cluster_in_3D()
            print("Clustering finished.")
        self.rendered_cluster = None if self.cluster_point_colors is None else render(view_camera, self.engine['scene'], self.opt, self.bg_color, override_color=torch.from_numpy(self.cluster_point_colors).cuda().float())["render"].permute(1, 2, 0)

        # --- RGB image --- #
        img = scene_outputs["render"].permute(1, 2, 0)  #

        rgb_score = img.clone()
        depth_score = rgb_score.cpu().numpy().reshape(-1)

        # --- semantic image --- #
        sems = feature_outputs["render"].permute(1, 2, 0)
        H, W, C = sems.shape
        sems /= (torch.norm(sems, dim=-1, keepdim=True) + 1e-6)
        sem_transed = torch.from_numpy(self.proj_mat.transform(sems.flatten(0,1).detach().cpu())).cuda().reshape(H,W,-1)
        sem_transed_rgb = torch.clip(sem_transed*0.5+0.5, 0, 1).float()

        scale = dpg.get_value('_Scale')
        self.gates = self.engine['scale_gate'](torch.tensor([scale]).cuda())
        scale_gated_feat = sems * self.gates.unsqueeze(0).unsqueeze(0)
        scale_gated_feat = torch.nn.functional.normalize(scale_gated_feat, dim = -1, p = 2)
        
        if self.clear_edit:
            self.new_click_xy = []
            self.clear_edit = False
            self.prompt_num = 0
            try:
                self.engine['scene'].clear_segment()
                self.engine['feature'].clear_segment()
            except:
                pass

        if self.roll_back:
            self.new_click_xy = []
            self.roll_back = False
            self.prompt_num = 0
            # try:
            self.engine['scene'].roll_back()
            self.engine['feature'].roll_back()
            # except:
                # pass
        
        if self.reload_flag:
            self.reload_flag = False
            print("loading model file...")
            self.engine['scene'].load_ply(self.opt.scene_pcd_path)
            self.engine['feature'].load_ply(self.opt.feature_pcd_path)
            self.engine['scale_gate'].load_state_dict(torch.load(self.opt.scale_gate_path))
            self.do_pca()   # calculate self.proj_mat
            self.load_model = True

        score_map = None
        if len(self.new_click_xy) > 0:

            featmap = scale_gated_feat.reshape(H, W, -1)
            
            if self.new_click:
                xy = self.new_click_xy
                new_feat = featmap[int(xy[1])%H, int(xy[0])%W, :].reshape(featmap.shape[-1], -1)
                if (self.prompt_num == 0) or (self.clickmode_multi_button == False):
                    self.chosen_feature = new_feat
                else:
                    self.chosen_feature = torch.cat([self.chosen_feature, new_feat], dim=-1)    # extend to get more prompt features
                self.prompt_num += 1
                self.new_click = False
            
            score_map = featmap @ self.chosen_feature
            # print(score_map.shape, score_map.min(), score_map.max(), "score_map_shape")

            score_map = (score_map + 1.0) / 2
            score_binary = score_map > dpg.get_value('_ScoreThres')
            
            score_map[~score_binary] = 0.0
            score_map = torch.max(score_map, dim=-1).values
            score_norm = (score_map - dpg.get_value('_ScoreThres')) / (1 - dpg.get_value('_ScoreThres'))

            if self.preview:
                rgb_score = img * torch.max(score_binary, dim=-1, keepdim=True).values    # option: binary
            else:
                rgb_score = img
            depth_score = 1 - torch.clip(score_norm, 0, 1)
            depth_score = depth2img(depth_score.cpu().numpy()).astype(np.float32)/255.0

            if self.segment3d_flag:
                """ gaussian point cloud core params
                self.engine._xyz            # (N, 3)
                self.engine._features_dc    # (N, 1, 3)
                self.engine._features_rest  # (N, 15, 3)
                self.engine._opacity        # (N, 1)
                self.engine._scaling        # (N, 3)
                self.engine._rotation       # (N, 4)
                self.engine._objects_dc     # (N, 1, 16)
                """
                self.segment3d_flag = False
                feat_pts = self.engine['feature'].get_point_features.squeeze()
                scale_gated_feat_pts = feat_pts * self.gates.unsqueeze(0)
                scale_gated_feat_pts = torch.nn.functional.normalize(scale_gated_feat_pts, dim = -1, p = 2)

                score_pts = scale_gated_feat_pts @ self.chosen_feature
                score_pts = (score_pts + 1.0) / 2
                self.score_pts_binary = (score_pts > dpg.get_value('_ScoreThres')).sum(1) > 0

                # save_path = "./debug_robot_{:0>3d}.ply".format(self.object_seg_id)
                # try:
                #     self.engine['scene'].roll_back()
                #     self.engine['feature'].roll_back()
                # except:
                #     pass
                self.engine['scene'].segment(self.score_pts_binary)
                self.engine['feature'].segment(self.score_pts_binary)

        if self.save_flag:
            print("Saving ...")
            self.save_flag = False
            try:
                os.makedirs("./segmentation_res", exist_ok=True)
                save_mask = self.engine['scene']._mask == self.engine['scene'].segment_times + 1
                torch.save(save_mask, f"./segmentation_res/{dpg.get_value('save_name')}.pt")
            except:
                with dpg.window(label="Tips"):
                    dpg.add_text('You should segment the 3D object before save it (click segment3d first).')

        self.render_buffer = None
        render_num = 0
        if self.render_mode_rgb or (not self.render_mode_pca and not self.render_mode_cluster and not self.render_mode_similarity):
            self.render_buffer = rgb_score.cpu().numpy().reshape(-1)
            render_num += 1
        
        if self.render_mode_pca:
            self.render_buffer = sem_transed_rgb.cpu().numpy().reshape(-1) if self.render_buffer is None else self.render_buffer + sem_transed_rgb.cpu().numpy().reshape(-1)
            render_num += 1
        if self.render_mode_cluster:
            if self.rendered_cluster is None:
                self.render_buffer = rgb_score.cpu().numpy().reshape(-1) if self.render_buffer is None else self.render_buffer + rgb_score.cpu().numpy().reshape(-1)
            else:
                # print(f'self.rendered_cluster.shape={self.rendered_cluster.shape}') # [600, 1080, 3]
                def filter2d(map: torch.Tensor):
                    res = cv2.blur(map.cpu().numpy(), (5,5))
                    return torch.from_numpy(res).cuda()
                # self.rendered_cluster = filter2d(self.rendered_cluster)
                self.render_buffer = self.rendered_cluster.cpu().numpy().reshape(-1) if self.render_buffer is None else self.render_buffer + self.rendered_cluster.cpu().numpy().reshape(-1)
            render_num += 1
        if self.render_mode_label:
            self.render_buffer = render(view_camera, self.engine['scene'], self.opt, self.bg_color, filtered_mask=~(self.point_labels==self.label))['render'].permute(1,2,0).cpu().numpy().reshape(-1)
            render_num += 1
        if self.render_mode_similarity:
            if score_map is not None:
                self.render_buffer = self.grayscale_to_colormap(score_map.squeeze().cpu().numpy()).reshape(-1).astype(np.float32) if self.render_buffer is None else self.render_buffer + self.grayscale_to_colormap(score_map.squeeze().cpu().numpy()).reshape(-1).astype(np.float32)
            else:
                self.render_buffer = rgb_score.cpu().numpy().reshape(-1) if self.render_buffer is None else self.render_buffer + rgb_score.cpu().numpy().reshape(-1)

            render_num += 1
        self.render_buffer /= render_num

        dpg.set_value("_texture", self.render_buffer)


if __name__ == "__main__":
    parser = ArgumentParser(description="GUI option")

    parser.add_argument('-m', '--model_path', type=str, default="./output/temp/tys")
    parser.add_argument('-s', '--source_path', type=str, default="./data/temp/tys")
    parser.add_argument('--sh_degree', type=int, default=3)
    parser.add_argument('--feature_iteration', type=int, default=10000)
    parser.add_argument('--scene_iteration', type=int, default=30000)

    args = parser.parse_args()

    opt = CONFIG()

    opt.model_path = args.model_path
    opt.source_path = args.source_path
    opt.sh_degree = args.sh_degree
    opt.feature_gaussian_iteration = args.feature_iteration
    opt.scene_gaussian_iteration = args.scene_iteration

    gs_model = GaussianModel(opt.sh_degree)
    feat_gs_model = FeatureGaussianModel(opt.feature_dim)
    scale_gate = torch.nn.Sequential(
        torch.nn.Linear(1, opt.feature_dim, bias=True),
        torch.nn.Sigmoid()
    ).cuda()
    gui = GaussianSplattingGUI(opt, gs_model, feat_gs_model, scale_gate)

    gui.render()
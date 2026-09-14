# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAGA (Segment Any 3D Gaussians) is a 3D scene segmentation framework built on top of 3D Gaussian Splatting. It combines GroundedDINO for object detection, SAM for mask extraction, and contrastive learning for 3D instance segmentation.

## Common Workflow

The full pipeline consists of 4 sequential steps. See [command.txt](command.txt) for a complete bash script example.

### 1. Extract Masks (grounded_SAM_masks.py)

Uses GroundingDINO for text-conditioned object detection and SAM for mask refinement.

```bash
python grounded_SAM_masks.py \
  --images_path <path/to/images> \
  --masks_path <output/masks> \
  --labels_path <output/labels> \
  --progress_path <progress.txt> \
  --sam_checkpoint_path weights/sam_vit_h_4b8939.pth \
  --groundingdino_checkpoint_path weights/groundingdino_swint_ogc.pth \
  --groundingdino_config_path weights/GroundingDINO_SwinT_OGC.py
```

Key parameters:
- `--downsample`: Image resolution downsampling factor (1, 2, 4, 8)
- `--classes`: Classes to detect (default: chair, table, plant, etc.)
- `--box_threshold`, `--text_threshold`, `--nms_threshold`: Detection filtering thresholds

### 2. Extract Mask Scales (get_scale.py)

Computes 3D scale for each mask using depth rendering from the pre-trained 3DGS model.

```bash
python get_scale.py \
  --sh_degree 0 \
  --masks_path <masks/path> \
  --point_cloud_path <pretrained/scene_point_cloud.ply> \
  --sparse_path <colmap/sparse/0> \
  --images_path <images/path> \
  --mask_scales_path <output/mask_scales> \
  --progress_path <progress.txt>
```

**Important**: Images are rotated 90 degrees clockwise in `grounded_SAM_masks.py` (line 120) and masks are stored permuted/flipped. The scale calculation accounts for this.

### 3. Train Contrastive Features (train_contrastive_feature.py)

Trains scale-aware instance features using contrastive learning.

```bash
python train_contrastive_feature.py \
  --sh_degree 0 \
  --feature_dim 32 \
  --images_path <images/path> \
  --sparse_path <colmap/sparse/0> \
  --masks_path <masks/path> \
  --mask_scales_path <mask_scales/path> \
  --point_cloud_path <pretrained/scene_point_cloud.ply> \
  --contrastive_feature_point_cloud_path <output/features.ply> \
  --scale_gate_path <output/scale_gate.pt> \
  --iterations 10000 \
  --num_sampled_rays 1000 \
  --progress_path <progress.txt>
```

Key training parameters (OptimizationParams):
- `--iterations`: Training iterations (default: 30000)
- `--num_sampled_rays` or `--ray_sample_rate`: Pixels sampled per training step
- `--distance_weight`: Distance loss weight (default: 100)
- `--smooth_K`: Smoothing KNN parameter (default: 16)

### 4. Post-process for Instance Labels (postprocess.py)

Performs 3D clustering (HDBSCAN) and assigns semantic class labels.

```bash
python postprocess.py \
  --sh_degree 0 \
  --feature_dim 32 \
  --images_path <images/path> \
  --sparse_path <colmap/sparse/0> \
  --masks_path <masks/path> \
  --labels_path <labels/path> \
  --mask_scales_path <mask_scales/path> \
  --point_cloud_path <pretrained/scene_point_cloud.ply> \
  --contrastive_feature_point_cloud_path <features.ply> \
  --scale_gate_path <scale_gate.pt> \
  --json_path <output.json> \
  --progress_path <progress.txt> \
  --clean  # Optional: removes intermediate files
```

Key parameters:
- `--scale`: Scale value for scale-gate conditioning (default: 1.0)
- `--k`: KNN filter parameter (default: 256)
- `--instance_threshold`: Minimum instance confidence (default: 0.3)
- `--label_threshold`: Minimum class confidence (default: 0.3)
- `--sample_num`: Number of points for HDBSCAN (default: 10000)

## Architecture

### Core Modules

- **[scene/](scene/)**: Scene loading and Gaussian model management
  - `__init__.py`: `Scene` class - manages cameras and Gaussian models
  - `gaussian_model.py`: `GaussianModel` - 3DGS scene representation
  - `gaussian_model_ff.py`: `FeatureGaussianModel` - feature-augmented Gaussians
  - `dataset_readers.py`: COLMAP data loading
  - `cameras.py`: CameraInfo class

- **[arguments/](arguments/)**: Parameter management
  - `ModelParams`: Data paths and model configuration
  - `PipelineParams`: Rendering pipeline options
  - `OptimizationParams`: Training hyperparameters

- **[gaussian_renderer/](gaussian_renderer/)**: Custom rasterization
  - Three specialized rasterizers in `submodules/`:
    - `diff-gaussian-rasterization`: Standard RGB rendering
    - `diff-gaussian-rasterization_contrastive_f`: Feature rendering
    - `diff-gaussian-rasterization-depth`: Depth rendering

- **[utils/](utils/)**: Utilities for cameras, graphics, visualization

### Third-party Dependencies

- `third_party/segment-anything/`: SAM for mask generation
- `third_party/GroundingDINO/`: Text-conditioned object detection
- `third_party/kmeans_pytorch/`: K-means clustering
- `submodules/simple-knn/`: K-nearest neighbors
- `submodules/torch_kdtree/`: KD-tree for fast spatial queries

## Data Structure

Expected directory structure:
```
<scene_base>/
├── images/                    # Input images (jpg)
├── sparse/0/                  # COLMAP sparse reconstruction
│   ├── cameras.bin/txt
│   ├── images.bin/txt
│   └── points3D.bin/txt
├── point_cloud/iteration_30000/
│   └── scene_point_cloud.ply  # Pre-trained 3DGS model
├── masks/                     # Output of step 1
├── labels/                    # Output of step 1
├── mask_scales/               # Output of step 2
├── contrastive_feature_point_cloud.ply  # Output of step 3
├── scale_gate.pt              # Output of step 3
└── output.json                # Final output (step 4)
```

## Key Implementation Details

### Image Rotation Handling
Images are rotated 90° clockwise during mask extraction (`grounded_SAM_masks.py:120`). This affects:
1. Detection (GroundingDINO runs on rotated images)
2. Mask storage (permuted and flipped)

### Scale Gate Mechanism
The scale gate (1→32 linear layer + sigmoid) conditions features on 3D scale, enabling scale-aware segmentation. See `train_contrastive_feature.py:140-144`.

### Feature Dimension
All feature models use `feature_dim=32` by default. Modify both `--feature_dim` and model initialization when changing.

### Progress Tracking
All scripts support `--progress_path` for writing progress (0-100) to a file, useful for GUI monitoring.

## Installation

```bash
conda env create --file environment.yml
conda activate saga
```

Download required weights:
```bash
mkdir weights
cd weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

Build GroundingDINO:
```bash
cd third_party/GroundingDINO
pip install .
```

## Pre-requisites

Before running the SAGA pipeline, you must have a pre-trained 3D Gaussian Splatting model. Use `train_scene.py` (from 3DGS) to train the base scene model first.

# SegAnyGAussians

`SegAnyGAussians` 是一个基于 3D Gaussian Splatting 的三维开放词汇分割与实例后处理项目。本仓库当前更适合作为“工程使用版”来理解：你先准备好 COLMAP 稀疏重建和预训练 3DGS 点云，再依次执行 `grounded_SAM_masks.py`、`get_scale.py`、`train_contrastive_feature.py`、`postprocess.py`，得到最终的三维实例结果与 `output.json`。

公开 SAGA 的论文任务以提示式分割为主；本仓库中的自动实例后处理属于老师交付原型的工程扩展，不能称为公开 SAGA 的官方自动实例基线。当前类别先验研究已经收束为一个小型二维复核实验，基线和评价口径见 [category_priors/INSTANCE_RECHECK_BASELINE_STANDARD.md](category_priors/INSTANCE_RECHECK_BASELINE_STANDARD.md)。历史实验代码不再属于活跃运行路径，由 Git 历史保存。

论文链接：<https://arxiv.org/abs/2312.00860>

![SAGA teaser](./assets/saga-teaser.png)

## 快速开始

如果你已经有一份可用场景数据，并希望尽快跑通当前研究仓库，先取得本仓库的当前工作副本，再在仓库根目录执行：

```bash
bash install_env.sh
conda activate saga

bash run_pipeline.sh --base-path data/temp/your_scene
```

公开官方上游是 [Jumpat/SegAnyGAussians](https://github.com/Jumpat/SegAnyGAussians)。它适合查阅公开 SAGA 的提示式分割实现，但不包含本研究仓库中的老师自动实例扩展、贡献映射修复和类别先验审计代码，不能用它替代当前研究副本。

上面的 `run_pipeline.sh` 会按顺序执行：
- `grounded_SAM_masks.py`
- `get_scale.py`
- `train_contrastive_feature.py`
- `postprocess.py`

如果你更希望明确掌控每一步参数，下面的 README 主体会优先介绍原始 Python 命令用法。

## 环境安装

当前推荐环境来源是 `requirements.txt`，不是 `environment.yml`。

### 推荐方式：使用安装脚本

```bash
bash install_env.sh
conda activate saga
```

`install_env.sh` 会做这些事情：
- 基于 `requirements.txt` 创建或复用 Conda 环境
- 初始化 git submodule
- 安装 `torch==2.4.1+cu118`、`torchvision==0.19.1+cu118`、`torchaudio==2.4.1+cu118`
- 安装仓库依赖与本地 CUDA 扩展
- 安装 `third_party/GroundingDINO`
- 下载运行所需权重到仓库根目录 `weights/`

常用参数：

```bash
bash install_env.sh --env-name saga
bash install_env.sh --skip-weights
bash install_env.sh --cuda-home /usr/local/cuda-11.8
```

### 手动安装说明

如果你不想使用脚本，可以参考 `requirements.txt` 顶部注释中的安装流程手动执行。

### 权重位置

当前仓库主流程默认从根目录 `weights/` 读取以下文件：

- `weights/sam_vit_h_4b8939.pth`
- `weights/groundingdino_swint_ogc.pth`
- `weights/GroundingDINO_SwinT_OGC.py`

这与以下脚本保持一致：
- `grounded_SAM_masks.py`
- `run_pipeline.sh`

## 运行前准备

在执行分割流水线之前，你需要先准备好：

- 一套场景图像目录 `images_path`
- 一套 COLMAP 稀疏重建目录 `sparse_path`，其中至少包含 `cameras.bin` / `images.bin` / `points3D.bin`，或者对应的文本格式文件
- 一份预训练好的 3DGS 点云文件 `point_cloud_path`

也就是说，SegAnyGAussians 不是直接从原始图片开始端到端训练；它默认依赖已经存在的几何重建结果和已有 3D Gaussian 模型。

如果你还没有 3DGS 点云，可以先使用仓库中的场景训练脚本：

```bash
python train_scene.py -s <path_to_colmap_or_nerf_dataset>
```

## 推荐目录结构

当前仓库更适合下面这种目录组织方式：

```text
<base_path>/
├── fastRecon/
│   └── dense/
│       └── sparse/
│           └── 0/
│               ├── images/
│               ├── cameras.bin
│               ├── images.bin
│               └── points3D.bin
├── output_models/
│   └── point_cloud/
│       └── iteration_30000/
│           └── point_cloud.ply
└── saga/
    ├── masks/
    ├── labels/
    │   └── label_features.pt
    ├── mask_scales/
    ├── contrastive_feature_point_cloud.ply
    ├── scale_gate.pt
    ├── output.json
    ├── progress
    └── render/
```

其中：
- 输入目录主要是 `images_path`、`sparse_path`、`point_cloud_path`
- 流水线输出默认写到 `<base_path>/saga/`
- 权重统一放在仓库根目录 `weights/`

## 路径约定

### 通用占位符版

```bash
base_path=<base_path>
images_path=<images_path>
sparse_path=<sparse_path>
point_cloud_path=<point_cloud_path>
```

### 按当前仓库推荐布局推导

```bash
base_path="data/temp/suzongbangongshi"

images_path="${base_path}/fastRecon/dense/sparse/0/images/"
sparse_path="${base_path}/fastRecon/dense/sparse/0/"
point_cloud_path="${base_path}/output_models/point_cloud/iteration_30000/point_cloud.ply"

masks_path="${base_path}/saga/masks"
labels_path="${base_path}/saga/labels"
label_features_path="${base_path}/saga/labels/label_features.pt"
mask_scales_path="${base_path}/saga/mask_scales"
contrastive_feature_point_cloud_path="${base_path}/saga/contrastive_feature_point_cloud.ply"
scale_gate_path="${base_path}/saga/scale_gate.pt"
json_path="${base_path}/saga/output.json"
progress_path="${base_path}/saga/progress"
render_path="${base_path}/saga/render"

sam_checkpoint_path="weights/sam_vit_h_4b8939.pth"
groundingdino_checkpoint_path="weights/groundingdino_swint_ogc.pth"
groundingdino_config_path="weights/GroundingDINO_SwinT_OGC.py"
```

## 完整工作流

完整工作流按下面顺序执行：

1. `grounded_SAM_masks.py`：提取 2D masks、labels 与 label features
2. `get_scale.py`：为 masks 估计三维尺度信息
3. `train_contrastive_feature.py`：训练实例对比特征与 `scale_gate`
4. `postprocess.py`：做聚类和语义后处理，输出 `output.json`

### 1. 提取 Grounded SAM masks

用途：
- 从图像中提取类别相关的 2D mask
- 保存每张图的 mask 与 label
- 生成后续后处理需要的 `label_features.pt`

最小示例：

```bash
python grounded_SAM_masks.py \
  --progress_path "$progress_path" \
  --images_path "$images_path" \
  --masks_path "$masks_path" \
  --labels_path "$labels_path" \
  --label_features_path "$label_features_path" \
  --sam_checkpoint_path "$sam_checkpoint_path" \
  --groundingdino_checkpoint_path "$groundingdino_checkpoint_path" \
  --groundingdino_config_path "$groundingdino_config_path" \
  --downsample 1
```

常用输入：
- `--images_path`
- `--sam_checkpoint_path`
- `--groundingdino_checkpoint_path`
- `--groundingdino_config_path`

主要输出：
- `--masks_path`
- `--labels_path`
- `--label_features_path`
- `--progress_path`

常用可调参数：
- `--downsample`
- `--downsample_type`
- `--classes`
- `--box_threshold`
- `--text_threshold`
- `--nms_threshold`

说明：
- 该脚本默认权重路径就是根目录 `weights/`
- 如果显存不够，优先尝试增大 `--downsample`

如果想一键跑这一步：

```bash
bash run_pipeline.sh --base-path "$base_path" --stage masks
```

### 2. 计算 mask scales

用途：
- 结合已有 3DGS 点云和相机参数，为每个 2D mask 估计尺度信息
- 生成训练阶段需要的 `mask_scales`

最小示例：

```bash
python get_scale.py \
  --progress_path "$progress_path" \
  --sh_degree 0 \
  --masks_path "$masks_path" \
  --point_cloud_path "$point_cloud_path" \
  --sparse_path "$sparse_path" \
  --images_path "$images_path" \
  --mask_scales_path "$mask_scales_path"
```

主要输入：
- `--masks_path`
- `--point_cloud_path`
- `--sparse_path`
- `--images_path`

主要输出：
- `--mask_scales_path`
- `--progress_path`

说明：
- `--sh_degree` 需要和你的场景点云设置一致；你当前工作流里常用 `0`
- 运行这一步前应确保 masks 已经提取完成

如果想一键跑这一步：

```bash
bash run_pipeline.sh --base-path "$base_path" --stage scale
```

### 3. 训练 contrastive feature

用途：
- 在已有场景高斯上训练实例特征
- 输出带特征的点云和 `scale_gate.pt`

最小示例：

```bash
python train_contrastive_feature.py \
  --progress_path "$progress_path" \
  --sh_degree 0 \
  --feature_dim 32 \
  --images_path "$images_path" \
  --sparse_path "$sparse_path" \
  --masks_path "$masks_path" \
  --mask_scales_path "$mask_scales_path" \
  --point_cloud_path "$point_cloud_path" \
  --labels_path "$labels_path" \
  --label_features_path "$label_features_path" \
  --contrastive_feature_point_cloud_path "$contrastive_feature_point_cloud_path" \
  --scale_gate_path "$scale_gate_path" \
  --num_sampled_rays 1000
```

主要输入：
- `--images_path`
- `--sparse_path`
- `--masks_path`
- `--mask_scales_path`
- `--point_cloud_path`
- `--labels_path`
- `--label_features_path`

主要输出：
- `--contrastive_feature_point_cloud_path`
- `--scale_gate_path`
- `--progress_path`

常用可调参数：
- `--feature_dim`
- `--num_sampled_rays`
- `--sh_degree`
- 以及脚本内部已有的训练参数，如 `--iterations`

特征训练未显式指定 `--iterations` 时，会使用
`min(10 * 训练相机数, 10000)` 的自适应预算；场景 3DGS 训练仍默认使用 30,000 轮。

说明：
- 当前仓库工作流里，`feature_dim=32` 是最常见设置
- `label_features_path` 会在训练与后处理阶段重复使用

如果想一键跑这一步：

```bash
bash run_pipeline.sh --base-path "$base_path" --stage train
```

### 4. 后处理并导出结果

用途：
- 对训练后的高斯特征做聚类与语义筛选
- 结合 `label_features.pt` 输出实例化结果 `output.json`

最小示例：

```bash
python postprocess.py \
  --progress_path "$progress_path" \
  --sh_degree 0 \
  --feature_dim 32 \
  --images_path "$images_path" \
  --sparse_path "$sparse_path" \
  --masks_path "$masks_path" \
  --labels_path "$labels_path" \
  --label_features_path "$label_features_path" \
  --mask_scales_path "$mask_scales_path" \
  --point_cloud_path "$point_cloud_path" \
  --contrastive_feature_point_cloud_path "$contrastive_feature_point_cloud_path" \
  --scale_gate_path "$scale_gate_path" \
  --json_path "$json_path"
```

主要输入：
- `--images_path`
- `--sparse_path`
- `--masks_path`
- `--labels_path`
- `--label_features_path`
- `--mask_scales_path`
- `--point_cloud_path`
- `--contrastive_feature_point_cloud_path`
- `--scale_gate_path`

主要输出：
- `--json_path`
- `--progress_path`

常用可调参数：
- `--scale`
- `--k`
- `--feature_ratio`
- `--instance_threshold`
- `--label_threshold`
- `--scale_threshold`
- `--sample_num`
- `--clean`

说明：
- `postprocess.py` 会读取 `label_features_path`
- 如果使用 `--clean`，它会清理部分中间产物，建议在确认结果可复现后再开启

如果想一键跑这一步：

```bash
bash run_pipeline.sh --base-path "$base_path" --stage postprocess
```

## 一条命令跑完整流水线

如果你的路径布局符合默认约定，可以直接使用：

```bash
bash run_pipeline.sh --base-path data/temp/suzongbangongshi
```

它默认会推导这些路径：

- `images_path=${base_path}/fastRecon/dense/sparse/0/images/`
- `sparse_path=${base_path}/fastRecon/dense/sparse/0/`
- `point_cloud_path=${base_path}/output_models/point_cloud/iteration_30000/point_cloud.ply`
- `masks_path=${base_path}/saga/masks`
- `labels_path=${base_path}/saga/labels`
- `label_features_path=${base_path}/saga/labels/label_features.pt`
- `mask_scales_path=${base_path}/saga/mask_scales`
- `contrastive_feature_point_cloud_path=${base_path}/saga/contrastive_feature_point_cloud.ply`
- `scale_gate_path=${base_path}/saga/scale_gate.pt`
- `json_path=${base_path}/saga/output.json`
- `progress_path=${base_path}/saga/progress`
- `render_path=${base_path}/saga/render`

也支持单阶段运行：

```bash
bash run_pipeline.sh --base-path "$base_path" --stage masks
bash run_pipeline.sh --base-path "$base_path" --stage scale
bash run_pipeline.sh --base-path "$base_path" --stage train
bash run_pipeline.sh --base-path "$base_path" --stage postprocess
bash run_pipeline.sh --base-path "$base_path" --stage render
bash run_pipeline.sh --base-path "$base_path" --stage gui
```

如果你的目录结构不同，可以直接覆盖对应路径参数，例如：

```bash
bash run_pipeline.sh \
  --base-path "$base_path" \
  --images-path "$images_path" \
  --sparse-path "$sparse_path" \
  --point-cloud-path "$point_cloud_path"
```

## 渲染与 GUI

### 渲染实例结果

如果已经得到 `output.json`，可以把实例标签渲染回训练视角：

```bash
python render_instance.py \
  --sh_degree 0 \
  --feature_dim 32 \
  --images_path "$images_path" \
  --sparse_path "$sparse_path" \
  --masks_path "$masks_path" \
  --labels_path "$labels_path" \
  --mask_scales_path "$mask_scales_path" \
  --point_cloud_path "$point_cloud_path" \
  --contrastive_feature_point_cloud_path "$contrastive_feature_point_cloud_path" \
  --scale_gate_path "$scale_gate_path" \
  --json_path "$json_path" \
  --render_path "$render_path"
```

也可以使用脚本方式：

```bash
bash run_pipeline.sh --base-path "$base_path" --stage render
```

### 启动 GUI

`saga_gui.py` 适合在已经完成特征训练和后处理后做可视化与交互查看。

```bash
python saga_gui.py \
  --sh_degree 0 \
  --scale_gate_path "$scale_gate_path" \
  --feature_pcd_path "$contrastive_feature_point_cloud_path" \
  --scene_pcd_path "$point_cloud_path" \
  --json_path "$json_path"
```

也可以使用：

```bash
bash run_pipeline.sh --base-path "$base_path" --stage gui
```

GUI 交互保留简要说明：
- 左键拖动：旋转
- 中键拖动：平移
- 右键：点选提示
- 可结合当前实例结果做可视化、聚类和交互分析

## 辅助脚本说明

### `install_env.sh`

适合这些场景：
- 第一次配置环境
- 想要自动初始化 submodule 和下载权重
- 希望安装过程尽量标准化

主命令：

```bash
bash install_env.sh
```

### `run_pipeline.sh`

适合这些场景：
- 你的数据目录符合推荐布局
- 你希望把四阶段流程串起来执行
- 你只想单独运行某一阶段或调用 `render` / `gui`

主命令：

```bash
bash run_pipeline.sh --base-path <base_path>
```

## 常见问题与注意事项

### 1. `requirements.txt` 和 `environment.yml` 应该用哪个？

当前推荐使用 `requirements.txt` 对应的安装方案，也就是 `bash install_env.sh`。`environment.yml` 属于旧环境定义，不再作为主推荐路径。

### 2. 权重应该放在哪里？

当前主流程默认使用仓库根目录：

```text
weights/sam_vit_h_4b8939.pth
weights/groundingdino_swint_ogc.pth
weights/GroundingDINO_SwinT_OGC.py
```

不要默认依赖 `third_party/segment-anything/weights/`。

### 3. 运行前必须有哪些输入？

至少需要：
- 图像目录
- COLMAP 稀疏重建目录
- 预训练 3DGS 点云

没有这些输入时，`run_pipeline.sh` 的阶段检查会直接报错退出。

### 4. 显存不够怎么办？

优先尝试：
- 在 `grounded_SAM_masks.py` 中调大 `--downsample`
- 降低训练阶段的 `--num_sampled_rays`
- 确认没有额外占用 GPU 的进程

### 5. GroundingDINO 安装失败怎么办？

优先检查：
- `CUDA_HOME` 是否指向正确 CUDA 路径
- 当前 Conda 环境是否已经激活
- `third_party/GroundingDINO` submodule 是否完整初始化

安装脚本支持：

```bash
bash install_env.sh --cuda-home /usr/local/cuda-11.8
bash install_env.sh --force-groundingdino-reinstall
```

### 6. 为什么 README 不再主推旧的 `model_path` 风格命令？

因为当前仓库你实际在用的是显式路径风格：
- `images_path`
- `sparse_path`
- `point_cloud_path`
- `masks_path`
- `labels_path`
- `label_features_path`
- `mask_scales_path`

这套接口与 `run_pipeline.sh` 和你现在的工作流更一致，也更容易定位每一步输入输出。

## Citation

如果这个项目对你的研究或工程有帮助，可以引用原论文：

```bibtex
@article{cen2023saga,
  title={Segment Any 3D Gaussians},
  author={Jiazhong Cen and Jiemin Fang and Chen Yang and Lingxi Xie and Xiaopeng Zhang and Wei Shen and Qi Tian},
  year={2023},
  journal={arXiv preprint arXiv:2312.00860},
}
```

## Acknowledgement

本仓库实现参考了以下项目：
- [GARField](https://github.com/chungmin99/garfield.git)
- [OmniSeg3D-GS](https://github.com/OceanYing/OmniSeg3D-GS)
- [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)

感谢这些工作的开源贡献。

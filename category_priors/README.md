# 类别先验研究代码

这个目录只保留当前研究需要的公共数据合同、候选库、旧后处理重放和评价工具。过去的 V3–V10、ObjectBank、提示尺度、HDBSCAN 修复和三维尺寸恢复实验已经退出活跃代码；如需追溯，请查看 Git 历史，而不是把旧状态机重新接回运行路径。

最近完成的 DEV2 结果表明，这套两轮闭环没有救回任何 B0 漏检对象，反而降低了 AP 并把 B0 前景切成大量碎片，因此已在扩展 DEV8 前停止。完整结论、数值和失败链见 [ITERATIVE_REFINEMENT_DEV2_CLOSEOUT.md](ITERATIVE_REFINEMENT_DEV2_CLOSEOUT.md)。该结果只否定当前闭环组合，没有检验 `D-class - U-global`，不能写成类别尺寸先验无效。

在开始实现或运行前，必须完整阅读 [ITERATIVE_REFINEMENT_EXPERIMENT_STANDARD.md](ITERATIVE_REFINEMENT_EXPERIMENT_STANDARD.md)。旧的 [INSTANCE_RECHECK_BASELINE_STANDARD.md](INSTANCE_RECHECK_BASELINE_STANDARD.md) 只记录已经结案的一次布尔复核。

## 保留的公共能力

- `candidate_bank.py`：冻结的全类别候选成员与 `global_pre_knn` 数据合同；
- `candidate_replay.py`：把外部确认的候选编号接回同一套旧 KNN/过滤，并追踪最终存活；
- `legacy_candidate_replay.py`：同一起点上的旧 KNN、10 点过滤和候选存活追踪；
- `prediction_contract.py`：最终实例编号、类别、分数和包围盒的统一输出合同；
- `prediction_finalization.py`：基线和实验条件共享的最终二维类别、包围盒、分数及导出路径；
- `semantic_voting.py`：基线和实验条件共享的修正 contributor 二维投票；
- `evaluation_strata.json`：冻结的实例级小物体阈值和训练频率级尾部类别名单；
- `evaluator.py`：ScanNet 实例评价，官方主协议为 IoU 0.50–0.90 九档，AP25 单列；
- `instance_projection.py`、`gaussian_object_audit.py`：高斯与真实点之间的诊断映射，不替代官方评价；
- `geometry.py`：候选米制主成分三轴；
- `priors.py`、`taxonomy.py`、`scannet.py`：训练集统计、类别表和 ScanNet 数据读取；
- `alignment.py`：相机、三维高斯与 ScanNet 坐标对齐审计。

## 当前阶段

`iterative_refinement/` 已完成 DEV2 结案，保留用于复核，不再扩场景。旧 `instance_recheck.py` 及其 `B0/raw/global/class` 结果只作静态对照。

## 基础命令

训练集类别统计和通用评价仍通过简化后的命令入口运行：

```bash
python -m category_priors fit --stats train_instances.parquet --output category_priors.json
python -m category_priors evaluate --manifest evaluation_manifest.json --output metrics.json
```

新的唯一入口是 `run_iterative_refinement.py`。它不导入旧后处理器，也不会重跑 HDBSCAN：

```bash
python run_iterative_refinement.py prepare \
  --candidate-bank <candidate-bank> --stage-trace <stage_trace.npz> \
  --b0-output <B0/output.json> --scene-id <scene> --output-dir <reservoir>

python run_iterative_refinement.py refine \
  --candidate-bank <candidate-bank> --reservoir <reservoir> \
  --priors <category_priors.json> --condition class --output-dir <scene-output> \
  --point_cloud_path <30k-point-cloud.ply> \
  --contrastive_feature_point_cloud_path <2k-feature.ply> \
  --images_path <images> --sparse_path <COLMAP-sparse> \
  --masks_path <semantic-masks> --labels_path <semantic-labels> \
  --groundingdino-config-path <config.py> \
  --groundingdino-checkpoint-path <checkpoint.pth> \
  --sam-checkpoint-path <sam_vit_h.pth>
```

`refine` 一次生成稳健、平衡、覆盖三种局部图重放结果。`replay` 只核验并复用这三份结果；`evaluate` 在独立进程读取 GT，运行 B0 漏检救回和官方指标评价。正式云端命令必须记录全部绝对资产路径和当前提交。

# 类别先验研究代码

这个目录只保留当前研究需要的公共数据合同、候选库、旧后处理重放和评价工具。过去的 V3–V10、ObjectBank、提示尺度、HDBSCAN 修复和三维尺寸恢复实验已经退出活跃代码；如需追溯，请查看 Git 历史，而不是把旧状态机重新接回运行路径。

当前研究问题是：老师兼容自动流程产生全 SAGA20 分支候选后，把每个具有可用投影视角的候选投回最清楚的二维图像，用 GroundingDINO 与 SAM 做一次复核，是否能减少假实例；按预测类别的典型物理尺寸决定裁图范围，是否比全类别共用一个尺寸更好。

在开始实现或运行前，必须完整阅读 [INSTANCE_RECHECK_BASELINE_STANDARD.md](INSTANCE_RECHECK_BASELINE_STANDARD.md)。其中固定了候选同源关系、投影和裁图规则、二维复核逻辑、B0 漏检救回指标、ScanNet 官方评价口径，以及真实标注不得进入运行时的边界。

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

本次仓库重整只建立干净基线和技术标准，尚未实现或运行二维复核实验。下一阶段最多新增两个小模块：

```text
instance_recheck.py
recheck_evaluation.py
```

正式比较固定为 `B0/raw/global/class`，只运行 DEV8。每个场景只在当前基线下生成一次带完整指纹的候选库，随后三条件只读共享；不得下载或训练，不得为不同条件重跑候选生成，也不得通过保护式 KNN、事后插回或新对象主干改变候选几何。

## 基础命令

训练集类别统计和通用评价仍通过简化后的命令入口运行：

```bash
python -m category_priors fit --stats train_instances.parquet --output category_priors.json
python -m category_priors evaluate --manifest evaluation_manifest.json --output metrics.json
```

二维复核命令会在下一阶段实现完成后再写入这里。在此之前，README 不提供占位命令，避免把尚未验证的接口误当成可运行功能。

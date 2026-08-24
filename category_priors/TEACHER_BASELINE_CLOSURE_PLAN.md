# SAGA 原始交付基线闭环

## 目标与证据边界

本阶段不调类别先验，也不继续 V8。它只回答：老师交付的自动实例流水线本身是什么、能否按原资产协议复现、低分主要在哪一层产生。

Git 与本机 reflog 能证明的三个锚点必须分开命名：

- `official-upstream-96e5021`：公开 SAGA；只有提示式分割，不是自动 ScanNet 实例 AP 基线。
- `teacher-handoff-bfc2192`：本克隆首次切到 a800 时最早可证的交付快照；18 类词表、4 类 `other_classes`。
- `teacher-source-tip-8c5e167`：后来扩到 28 类、7 类 `other_classes` 的演化版本；不是最初交付快照。

聊天中的 `saga.zip` 无法与某个 Git 对象做字节级核对，所以不得宣称 `bfc2192` 就是压缩包逐字节内容；它只是当前证据链中最早可复现的 a800 锚点。

## 研究问题

1. `teacher-handoff-bfc2192` 的 B0/B1 在隔离重建的原始 18 类资产上表现如何？
2. 随后 `95073c6` 修复的 feature-channel 归一化、mask/label 排序和训练参数作用域，是否解释主要差距？
3. 历史 contributor、全点中心吸附、全局 256-NN 和最终类别 vote 分别造成何种精度/召回变化？
4. 自适应 feature 预算与显式 10k 是否改变上述归因？

三场结束即停止。本阶段不得据三场宣布类别先验有效或无效，也不扩展到 tune24/final48。

## 冻结场景与实验单位

固定三个既有开发物理场景：

- `scene0064_01`：21 个 book 与 1 个 socket，直接覆盖 handoff 中可评估的两个分支类；
- `scene0025_01`：book/cup/phone 等小物体支持丰富；
- `scene0231_00`：多个 book，且已有历史 B0/B1 viewer；

物理场景是独立单位；同一场景内的软件版本、预算和后处理快照都是配对技术测量。

## 精确词表

`teacher-handoff-bfc2192` 的 18 类顺序：

```text
chair table plant flower foliage tv painting sofa cabinet bed wall floor ceiling
person socket book remote key
```

输出类：

```text
chair table plant flower foliage tv painting sofa cabinet bed socket book remote key
```

老师的小物体分支：

```text
socket book remote key
```

指标同时报告完整 SAGA20 和 handoff 可预测类交集。词表中不存在的类别只能记为词表上限，不能混作实例主干的同类预测失败。

## 资产隔离与训练预算

复用：图像、COLMAP、30k RGB 3DGS、相机、GT、SAM/GroundingDINO 权重。

独立重建且不得覆盖现有 32 类资产：18 类 masks、labels、label features、mask scales、affinity/semantic feature PLY 和 scale gate。

同一套 handoff masks/scales 分别训练：

- `adaptive`：源码原有的 `min(10 * camera_count, 10000)`；
- `10k-control`：显式 10,000 轮，只作 feature-budget 正控。`bfc2192` 与
  `95073c6` 把 `iterations=None` 注册成了不可解析的 `NoneType` CLI 参数，故该正控只能在
  明确命名的 `full950-iterations-cli` 机械变体上运行；唯一修复是把 falsey 默认值从
  `None` 改为整数 `0`。同一变体同时跑 adaptive 与 10k，避免把 CLI 修复混入预算差异。

不重新训练 3DGS，不下载数据或权重。

## 最小条件矩阵

### A. 精确交付与训练修复

- `H-literal/B0,B1`：在 `scene0064_01` 原样运行 `bfc2192` 的 adaptive CLI。该脚本的
  `args` 位于模块全局作用域，`training()` 可以合法读取它；先前预注册的 NameError 判断错误。
- `H-args-only/B0,B1`：仅把模块全局 CLI `args` 显式传入训练函数，作为预期逐点等价的
  plumbing 负控；保留错误的
  feature 归一化维度和未同步的 mask/label 排序。
- `H-args-norm/B0,B1`：在 args-only 上再加入 feature-channel `dim=-1` 修复。
- `R-full950/B0,B1`：再加入 `masks[sort_indices]`，即 `95073c6` 的完整三项训练修复。

三个部分修复臂只跑 `scene0064_01`，用于定位修复影响；`R-full950` 再扩到全部三场。
各训练臂使用自己的 feature/scale gate。Grounded-SAM 与 scale 资产共享，因为这两个锚点的
相关变化只有程序封装，不改变冻结的输出语义。若 literal Grounded-SAM 遇到 `class_id=None`
而崩溃，只允许启用单一的 `None-filter` 机械修复，并在产物名中明确标记，不能静默算作 exact handoff。

### B. contributor 与后处理因果拆解

在 `R` 的同一 feature 上比较：

- `R-historical/B0,B1`：历史 `alpha*T_new` contributor 与空像素 ID 0；
- `R-fixed/B0,B1`：只修为 `alpha*T_prev`、空像素 `-1/0`、Python 过滤非法/零权重；
- `L1`：B1-fixed 去掉全局 256-NN；
- `L2`：只保留 sampled HDBSCAN core，关闭全点中心吸附和全局 KNN；
- `L3`：L2 加一次性局部 attach。

当前诊断 harness 必须先在同一 18 类资产上与 `R-fixed` 的 L0 逐点等价；实例 ID 只允许规范化重编号。机械等价失败时停止，不解释任何 AP。

最终类别 vote 在冻结几何上离线对比 historical vote、fixed vote 和 semantic-head class。GT oracle class/ranking 仅作离线诊断，不进入运行时。

### C. 10k 正控

10k 只在相机数最少、adaptive 约 410 轮且 branch GT 最丰富的 `scene0064_01`，使用同一
`full950-iterations-cli` 源分别训练 adaptive 与 10k，再重跑 `R-fixed/B0,B1`，
不重复整个结构消融网格。

`8c5e167` 的 28 类/7 类版本本阶段不重新训练；它只作为“交付后扩类”的代码历史背景。若原始闭环后仍需评估扩类贡献，再另立计划。

## 评价协议

同时报告：

- ScanNet 官方九阈值 mAP（0.50–0.90）和 AP25；
- 历史项目十阈值 mAP（0.50–0.95）；
- handoff 可预测类交集与完整 SAGA20；
- Gaussian→GT precision、unsupported fraction、GT→Gaussian recall（2/5/10 cm）；
- class-agnostic/same-class candidate precision/recall@0.25/0.50；
- merge、split、duplicate、实例数和覆盖率；
- 分支实例在 merge、KNN/filter、vote 前后的存活率。

老师原输出没有实例 confidence。AP 分别用：

- `unit-score`：所有实例同分，作为主适配；
- `final-vote-ratio`：只读工程适配；
- `gt-oracle-ranking`：只作排序上限。

不得把后两者称为原仓库原生 confidence。

## 预注册归因

- harness 与 `R-fixed/L0` 不等价：机械复现失败，停止。
- literal 与 args-only 必须先作为隐式/显式参数传递的机械等价检查；args-norm、full950 的
  逐级变化再分别归因 distance regularizer 归一化与 semantic supervision 对齐。不得把
  它们合称为类别先验收益。
- 某后处理阶段使 Gaussian precision 下降至少 5 个百分点或 unsupported 增加至少 5 个百分点，且 GT recall 增益不足 10 个百分点：确认该阶段造成污染。
- 几何 IoU≥0.50 明显多于 same-class IoU≥0.50，或晚分类准确率低于 70%：确认语义/vote 是主要瓶颈。
- B1 在 pre-KNN 有收益，但至少 50% 分支点或实例在 KNN/filter/vote 后丢失：确认老师分支被后续主干吞掉。
- 10k 相对 adaptive 的 precision 提高至少 5 个百分点，或新增至少 2 个 same-class IoU≥0.50 匹配且 GT recall 下降不超过 5 个百分点：才称 feature budget 是重要瓶颈。

## 验收产物

```text
teacher_handoff_provenance.json
teacher_handoff_asset_audit.json
teacher_handoff_metrics.parquet
teacher_handoff_analysis.json
viewer/
```

分析必须明确区分：交付快照、训练修复、contributor 修复、后处理阶段和 score adapter。

## 工程约束

- 历史与 fixed contributor 扩展隔离编译，不覆盖现有 Python 环境；
- 单 GPU、单训练或单 postprocess；
- 可解析完整产物直接复用，损坏/缺失项才重跑；
- 不覆盖现有 32 类/V8/locked 结果；
- 不生成 SHA 文件、lock、schedule hash 或 contributor cache；
- 磁盘至少保留 80GB；内存只按 90GiB cgroup 读取。

## 实施检查点（2026-08-24）

冻结为一条可恢复的顺序 runner：隔离导出 `bfc2192/95073c6`，物化
`args-only/args-norm`，在各 source workspace 内独立编译 max-contributor 扩展，重建
bfc18 资产，运行 historical/fixed B0/B1 与 10k 正控，再做 fixed-950 ↔ current-L0
机械等价检查。只有规范化实例 ID 后逐点等价，才运行 L1/L2/L3。最后只读生成双 AP
协议、Gaussian→GT 精度和 3D viewer。

实现文件为：

- `category_priors/baseline_closure.py`
- `category_priors/baseline_closure_variants.py`
- `category_priors/baseline_closure_budget.py`
- `category_priors/baseline_closure_contributor.py`
- `category_priors/baseline_closure_runner.py`
- `category_priors/baseline_closure_ablation.py`
- `category_priors/baseline_closure_evaluation.py`
- `category_priors/baseline_closure_analysis.py`
- `category_priors/baseline_closure_precision.py`
- `continue_teacher_baseline_closure.sh`

扩展通过各 source 的局部 `PYTHONPATH` 加载，不安装或覆盖共享 Python 环境。机械等价失败
时 runner 以预注册复现失败状态停止，不解释 L1–L3 的 AP。

### 2026-08-24 运行时纠错

首次 runner 人为加入 `--iterations 1`，在进入训练前触发了历史 argparse 的
`invalid NoneType value`。该失败不属于老师基线；原失败记录保留为 harness 审计，但不进入
条件矩阵。闭环已改为 literal adaptive 真运行；10k 只允许由上述独立 CLI 机械变体启动。

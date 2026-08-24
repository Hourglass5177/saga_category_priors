# SAGA 类别先验与小物体保护实验计划（当前权威：V8）

> **权威状态：V8 已按第 20 节门槛停止；第 21 节原始交付基线闭环是当前唯一执行路径。** 旧 B2、class-first、
> prior-v2、teacher-preservation 和 selective-restore 文档与代码只作为失败审计记录，
> 不得覆盖本文件的研究问题、条件定义、阶段门槛或结论边界。

- 版本：V8.0
- 冻结日期：2026-08-23
- V3 起点代码检查点：`f1367fa58c8f50df75f80b86f67bab469af06531`
- 当前状态：**V8 已在自动 bank 健康门槛停止；正在闭合最早可证的 teacher handoff 基线**
- 适用范围：现有 24 个 tune 场景、原 48 个内部评估场景及现有训练资产
- 独立实验单位：physical scene
- 技术重复：seed `42`、`3407`、`20260804`

## 0. 每次恢复任务时的强制回读协议

无论是新对话、上下文压缩、自动心跳还是人工继续，开始任何代码修改或云端运行前，
必须按顺序执行：

1. **完整读取本文件，不得只读摘要或最后一节。**
2. 读取本文“当前执行检查点”，确认正在进行的阶段和最后一个已验收产物。
3. 检查本地 `git status`、HEAD 和未跟踪文件；不得覆盖用户改动。
4. 若涉及云端，先只读检查进程、日志、磁盘、GPU、
   `/sys/fs/cgroup/memory.current`、`memory.max` 和 `memory.events`；不使用 `free`。
5. 对照该阶段的输入、输出、门槛和禁止事项；只补缺失项，不重跑完整项。
6. 完成一个阶段后，在本文“当前执行检查点”和“阶段记录”中更新事实，再进入下一阶段。
7. 若代码事实与本文冲突，先停下并报告；不得凭记忆修改实验语义。

本文件只允许两类修改：

- 记录已经发生且可核验的进度、产物和故障；
- 在尚未收集对应阶段结果前修正明确的代码事实错误，并在变更记录中说明。

不得在看到某阶段 AP 后反向更改该阶段的成功门槛、因素定义或主指标。

## 1. 执行判断

上一轮 `f1367fa` 的“只恢复全局 KNN 后仍有点幸存的类别分支”修复过小，
而且作用在错误层级。它没有回答：

- 类别分支是否产生了真实、完整的 GT 候选；
- 全局 KNN 删除的是真阳性还是误候选；
- 2D vote 是否把正确候选改成错误类别；
- 数据先验是否真正改变候选形成，而不是被后续统一过滤抵消；
- 小物体在现有 SAM mask 和 2k affinity feature 中是否已经缺失。

因此：

1. **不宣布类别先验或小物体保护失败。**
2. **不继续扩大 selective restore、全量 branch preservation 或 B2 overlay。**
3. 下一阶段先测输入和候选的可达上限，再根据候选死亡类型只实现一种最小保护。
4. 最终必须把“全类别结构”“逐类数据参数”“小物体保护”分开归因。

## 2. 已有证据与结论边界

| 实验 | 范围 | mAP 基线 → 方法 | 差值 | 已证明的事实 |
|---|---:|---:|---:|---|
| B1 人工小类分支 vs B0 | 48 scenes × 3 seeds | 0.055338 → 0.056164 | +0.000825 | AP50/AP25 有弱正信号，但 mAP 只在一个 seed 为正，不稳定 |
| P111 vs P000-B2 | 48 × 3 | 0.019761 → 0.020532 | +0.000770 | 坏主干上的小补偿，不能代表类别先验有效 |
| class-first D-small vs U0 | 48 × 3 | 0.000208 → 0.001617 | +0.001409 | 主干覆盖崩塌，相对提升没有方法意义 |
| prior-v2 D-size vs L1 | 48 × 3 | 0.053637 → 0.053193 | −0.000444 | 强主干正常，但 tune 小增益在三 seed 和 final 中消失 |
| teacher 原结构 D-size vs U0 | 8 scenes | 0.016242 → 0.016369 | +0.000128 | 数据参数已改变输出，但未形成有效实例收益 |
| teacher 全量保护 vs 原结构 | 8 scenes | 0.016242 → 0.004211 | −0.012030 | 被 KNN 删除的候选中含大量 FP/错类，不能全保留 |
| selective restore | scene0231_00 | 0.039120 → 0.016667 | −0.022454 | “有一个点幸存就恢复整簇”仍然过宽 |

现有结果只能支持以下判断：

- 老师原 8 类分支确实运行并改变了小类相关点，但平均收益弱且不稳定；
- 全类别候选会被全局 KNN 大量删除；
- 无条件或弱条件保护会造成 FP、错类和实例结构破坏；
- 目前没有候选级 GT best-IoU 漏斗，无法知道被删候选中真阳性的比例；
- 以前没有一个实验同时满足“老师原结构严格对照、参数真正生效、候选质量一致、
  小物体有足够 GT 支持”。

## 3. 三个研究问题与可证伪假设

### Q1：老师的全类别结构是否有价值？

- `B0`：关闭老师类别分支，旧结果直接复用。
- `B1-original`：当前 32 类资产兼容的 a800 原 8 类分支，旧结果直接复用。
- `T-all-exact`：只把 a800 的分支类别扩展到全部 SAGA20；其余阈值、采样、
  空间归一化、HDBSCAN、KNN、`filter_num`、2D vote 和类别输出保持原样。

`T-all-exact` 必须保持：

- semantic threshold `0.7`；
- 每类 sample cap `5000`；
- instance/spatial/semantic 权重 `0.5/0.3/0.2`；
- HDBSCAN epsilon `0.01`；
- assignment threshold `0.3`；
- a800 原空间归一化；
- a800 原 `min_samples` 行为；
- 原全局 KNN、10 点过滤和 2D vote。

不得把数据先验或新保护混入 `T-all-exact`。它只回答“扩全类别”这一件事。

### Q2：逐类数据参数是否优于共享参数？

在后续选定的统一候选质量与融合结构下比较：

- `U000`：所有类别使用相同全局数值；
- 数据条件：只切换本文注册的 size、smooth、small 三个因素。

除因素值外，语义候选、采样、HDBSCAN实现、候选质量、融合、评分、输出和随机种子
必须完全一致。

### Q3：小物体保护能否提高真实小实例召回且不制造 FP？

小物体是实例属性，不等同于某几个类别。除官方 ScanNet 指标外，必须按 GT 实例
物理尺寸和映射点数分层报告 Recall/AP。`<100` 点实例只作诊断，不混入官方主指标。

## 4. 资产与范围

必须复用：

- 73 套现有场景资产；
- 24 个 tune 场景与 GT；
- 原 48 个不同 physical scene 与 GT；
- 30k 3DGS；
- instance/semantic feature、scale gate；
- masks、labels、label features、mask scales；
- 已有 B0/B1 和失败实验 metrics、analysis、logs、viewer。

默认禁止：

- 下载 `.sens` 或重建已有 GT；
- 重训 3DGS；
- 重训全部 SAGA feature；
- 新网络、LLM、LangSplat、ScoreNet 或学习式 NMS；
- B2 overlay、class-first、prior-v2 或 selective-restore 作为正式主干；
- 无必要 SHA、lock、多层 schedule/manifest 或 contributor cache；
- 把 seed 当作独立场景样本。

唯一可能的训练例外：Stage 1 证明现有 SAM mask 覆盖目标、但 2k affinity feature 明显
不可分时，允许只选 2 个诊断场景做官方 10k affinity feature 正控。该正控不得自动扩展
到全部场景，且不重训 3DGS。

## 5. Stage 0：零成本历史锚点与诊断场景选择

### 输入

- 已有 B0/B1 三 seed 结果；
- 24 tune GT；
- train-only `category_priors.json`。

### 操作

1. 用同一官方 evaluator 复评已有 B0/B1，不重跑 postprocess。
2. 冻结 train-only 的 GT 物理尺寸分层边界；另登记 `<100` 映射点实例。
3. 只按 GT 的小实例支持、类别覆盖和 physical scene 多样性，从 24 tune 选 8 个诊断场景。
   不得按任何方法 AP 选场景。
4. 记录剩余 16 场景，供 Stage 4 独立复核。

### 输出

- `v3_history_anchor.parquet`
- `v3_gt_size_bins.json`
- `v3_diagnostic8_scenes.json`

## 6. Stage 1：输入上限与 shadow/oracle 候选漏斗

这是下一阶段的首要工作。在此阶段，类别分支只旁路生成候选，**不得改变 strong legacy
最终输出**。

### 6.1 一次场景加载，共享计算

每个场景只加载一次 3DGS、features、mask scales 和 2D vote 数据。共享：

- global legacy 初始/最终标签；
- 32 类 semantic scores；
- 2D vote 累积；
- GT→Gaussian 映射。

不同候选参数只重复必要的逐类 HDBSCAN，不重复场景下载、训练或完整渲染。

### 6.2 语义竞争

- 所有 32 个现有 codebook 类共同参与 top-1；
- 只有赢家属于 SAGA20 时才进入正式类别分支；
- 同时保留 a800 独立阈值模式用于 `T-all-exact` 对照；
- 报告每类候选数、GT语义召回和背景污染率，禁止只看最终实例。

### 6.3 候选漏斗

逐候选追踪：

```text
semantic candidate
→ sampled HDBSCAN/core
→ full assignment
→ global KNN
→ filter_num
→ 2D vote
→ final instance
```

每个候选保存 compact 记录：

- scene、seed、branch class、candidate ID；
- candidate/sample/core/full point count；
- HDBSCAN noise、membership、persistence；
- semantic margin/purity；
- 2D branch-class vote、winner、background ratio；
- global KNN/filter 后保留点数和比例；
- 与 global pre/post 实例的交叠；
- 同类 GT best IoU、任意类 GT best IoU；
- GT bbox diagonal、映射点数、尺寸分层；
- 是否为 global backbone 新增的 oracle match。

每个条件最多额外保存一个按 Gaussian 编号的压缩 `int32` branch-label 数组；不得保存重复
dense bool mask 集合。

### 6.4 输入上限

按 GT 实例检查：

1. 多视角 SAM mask 是否覆盖该实例；
2. semantic candidate 是否覆盖该实例；
3. scale-gated affinity feature 是否将其与邻近实例分开；
4. HDBSCAN 候选是否达到 IoU 0.25/0.50；
5. 候选在哪个后处理阶段死亡或错类。

### 6.5 判定树

- **SAM mask 无覆盖**：停止针对该类的 postprocess 调参；记录为输入监督缺失。
- **mask 有覆盖、feature 不可分**：记录为 affinity/scale-gate 瓶颈；必要时触发 2 场景
  10k feature 正控。
- **候选阶段无增量 oracle recall**：当前类别分支构造无价值，不实现融合保护。
- **高 IoU 候选被 KNN/filter 删除**：进入保护结构选择。
- **候选纯度低、数量爆炸**：先修候选质量，不保护。
- **KNN 后仍正确、vote 后错类**：只修类别仲裁。

### Stage 1 最低信息门槛

至少满足一项才进入保护实现：

- 相对 B1，small-size oracle Recall@0.25 提升至少 0.02；
- 或跨 8 场景至少新增 5 个 IoU≥0.25 的小实例候选，覆盖至少 2 类和 4 个场景。

否则扩展 shadow 审计到剩余 16 tune 场景；全 24 仍未满足则停止 postprocess 扩展。

### 输出

- `v3_input_ceiling.json`
- `v3_proposal_funnel.parquet`
- `v3_proposal_oracle_analysis.json`
- compact `branch_labels.npz`

## 7. Stage 2：只实现一种证据驱动保护

不再并行维护多个融合框架。根据 Stage 1 中 oracle-positive 候选的主要死亡类型二选一：

### A. 多-anchor局部保护

当多数正确候选在 global KNN 后仍保留足够 anchor 时使用：

- 必须满足同候选、同类别的多个 survivor anchors；
- anchor 数达到该条件使用的 core 支持阈值；
- 2D vote winner 必须等于 branch class；
- 只恢复同类别、物理半径内的 halo 点；
- 禁止“一点幸存恢复整簇”。

### B. proposal级准入

当多数正确候选被完全删除、但 candidate purity 和 vote 一致率高时使用：

- candidate 通过统一质量条件后才能成为实例；
- 与 global 实例做简单同类合并或 IoU NMS；
- 第一版不得无条件覆盖现有 global 实例；
- 只有 shadow 显示主要问题是 same-class merge 时，才允许对被吞小实例做 carve-out。

两种方案不得同时进入正式实验。选择规则基于候选死亡类型，而不是哪个方案在8场景AP更高。

### 2×2 结构实验

在8个诊断场景、seed42运行：

| 因素 | 低水平 | 高水平 |
|---|---|---|
| 参数 | shared uniform | data-driven combined |
| 融合 | a800原合并 | Stage 1 选定的证据保护 |

共 `8 scenes × 4 conditions = 32` 次。

结构推进门槛：

- 证据保护的 uniform 主干不得比同场景 B1 低超过 0.002 mAP；
- 预测实例数不得超过其 2 倍；
- 覆盖率不得下降超过 10 个百分点；
- data vs uniform 必须在目标点或候选结构上产生真实变化；
- 无条件/弱条件恢复造成的实例爆炸不得重现。

## 8. Stage 3：size × smooth × small 完整 2³ 因子实验

只有 Stage 2 结构通过后执行。三个因素均使用 train-only 收缩统计；所有实验臂共享同一个
候选生成、质量控制、融合、评分和输出流程。

### A：类别尺寸/粒度先验

定义：

```text
d_c = exp(shrunk.log_bbox_diag_m.q50)
```

- 低水平：所有类别使用 global scale；
- 高水平：将 `d_c` 通过当前场景 mask-scale 经验 CDF 映射到 SAGA 原生 scale gate，
  再计算该类 gated affinity feature；
- 第一版不同时修改 HDBSCAN 坐标归一化，避免一个因素混入两个机制。

### B：类别边界/平滑先验

定义：

```text
b_c = shrunk.boundary_fixed:0.05.q50
```

- 低水平：global physical smoothing radius；
- 高水平：由 `b_c` 映射为同类别物理半径；边界比例高的类别使用更小半径；
- 使用物理半径投票，不使用密度敏感的固定 `K_c`；
- 不跨类别传播，不让 `-1` 作为普通实例参与多数票。

### C：类别 core–halo 小物体保护

定义：

```text
A_c = exp(shrunk.log_surface_area_m2.q50)
```

- 低水平：global core support，halo off；
- 高水平：使用场景 Gaussian 密度、类别典型面积和实际采样率估计 sampled support，
  得到类别 core 阈值 `m_c`；
- HDBSCAN `min_samples` 在所有条件显式固定，不随 `m_c` 改变；
- 只有已通过质量控制的同类 core 能接收 halo；
- noise/fragment 不能自行生成实例；
- halo 半径由类别物理尺度给出，并受局部 anchor 支持限制。

### 因子布局

8个诊断场景、seed42，完整8条件：

```text
U000
D100 size
D010 smooth
D001 small
D110 size+smooth
D101 size+small
D011 smooth+small
D111 combined
```

共 `8 × 8 = 64` 次。physical scene 是随机完整区组；每个场景内八个条件顺序用固定seed
随机化。相同 scene/seed 使用 common random numbers 和相同采样索引。

因子阶段报告主效应和二阶交互，但属于探索性机制估计，不把每个效应都包装成正式显著性
检验。最终确认只比较预先选定的 `U000` 与一个 best-D。

若 small 主效应为正，再增加：

- core-threshold only；
- halo-rescue only。

共 `8 × 2 = 16` 次，用于拆分 small 收益来源。

### 8场景候选门槛

best-D 相对 U000 满足一项：

- `ΔmAP ≥ 0.002`；
- 或物理小实例 Recall@0.5 提升至少 0.01，且总体 mAP 不下降。

同时要求：

- 正向场景多于负向场景；
- 预测实例数不超过2倍；
- 覆盖率下降不超过10个百分点；
- oracle-positive候选接受率提高，而 FP/TP 比未明显恶化。

只选择一个 best-D，依次按 mAP、小实例 Recall@0.5、AP50、简单性破并列。

## 9. Stage 4：剩余16场景复核与三seed稳定性

### 4A：剩余16场景

只运行 `U000` 和 best-D、seed42，共 `16 × 2 = 32` 次。

通过条件：

- 剩余16场景 `ΔmAP > 0`；
- 正向场景多于负向场景；
- 全24场景 `ΔmAP ≥ 0.002`；
- 若 best-D 包含 small，small-size Recall@0.5 也必须为正。

### 4B：补技术seed

仅对 U000/best-D 补 seed `3407` 和 `20260804`：

```text
24 scenes × 2 conditions × 2 added seeds = 96 runs
```

进入48内部评估的条件：

- 三seed场景内均值的 `ΔmAP ≥ 0.002`；
- 至少2/3 seed方向为正；
- 正向 physical scene 多于负向；
- small机制被选择时，小实例指标也为正。

seed是技术重复。分析时必须先在每个 physical scene 内对seed取平均，不能把72个scene-seed
当作72个独立样本。

## 10. Stage 5：原48场景内部验证

不下载、不重训、不重选场景。旧方法结果已被查看，因此本阶段称为“独立场景内部验证”，
不包装成从未接触的盲测。

必跑：

```text
48 scenes × {U000, best-D} × 3 seeds = 288 runs
```

若 `T-all-exact` 在24 tune 上达到 `ΔmAP ≥ 0.002` 或小实例提升门槛，再追加：

```text
48 scenes × T-all-exact × 3 seeds = 144 runs
```

最大新 final 运行数为432；若早期门槛失败则不扩展。

48场景结果不得用于修改参数、融合规则或质量阈值。

## 11. 指标与统计

### 主指标

- 官方 ScanNet `mAP@[0.50:0.95]`
- 官方 `min_region_size=100`

### 次指标

- AP50、AP25；
- 全部 per-class AP；
- 每类 GT 实例数和场景支持数；
- small-size/tiny-size 实例 Recall@0.25/0.50；
- `<100` 点 GT 的诊断召回；
- 预测实例数、有效覆盖率；
- candidate oracle recall、candidate purity、FP/TP；
- KNN/filter/vote 各阶段真候选存活率；
- HDBSCAN noise、halo/rescue比例；
- 运行时间、GPU峰值、cgroup峰值。

### 最终推断

- physical scene 是独立单位；
- 每个场景先平均技术seed；
- 对场景级配对差做10,000次 paired bootstrap；
- 报告平均差和95% CI；
- 最终只有一个预先选定的 U000 vs best-D 主比较；
- 2³ 阶段的主效应/交互只作机制解释，不反向改写最终主比较。

### 结论规则

- `T-all-exact > B1` 且24/48场景方向稳定：支持全类别分支结构有价值。
- best-D `ΔmAP ≥ 0.002` 且 final 95% CI 下界 `>0`：支持数据驱动类别参数稳定改进。
- 总体 mAP 无提升、但 small-size Recall 的95% CI下界 `>0`：只支持小物体保护有效。
- `ΔmAP >0` 但CI跨0：正向趋势，证据不足。
- `ΔmAP ≤0`：当前注册的数据映射无改进。
- 输入阶段无目标信号：当前mask/feature资产不足，不能外推为类别先验概念失败。
- proposal oracle有信号但最终失败：融合/仲裁实现失败。
- proposal和融合均正常但D不优于U：当前类别统计映射失败。

## 12. 最小实现边界

V3 正式代码只允许新增：

1. 一个 shadow/oracle 候选审计出口；
2. 一个由 Stage 1 死亡类型确定的证据保护路径；
3. size/smooth/small 三个正交开关；
4. 一个轻量 runner/evaluator 输出既定产物。

不得新增：

- 第二套并行融合框架；
- 逐类 val 调参；
- reliability gate、复杂mapping或复合打分搜索；
- 新模型、学习式score或候选网络；
- 无必要的安全合同、hash、lock和缓存层。

候选质量规则必须对 U000 和所有D条件相同。不得为了某个先验条件单独放宽准入。

## 13. 预期产物

```text
v3_history_anchor.parquet
v3_gt_size_bins.json
v3_diagnostic8_scenes.json
v3_input_ceiling.json
v3_proposal_funnel.parquet
v3_proposal_oracle_analysis.json
v3_structure_2x2_metrics.parquet
v3_factorial_tune8_metrics.parquet
v3_factorial_analysis.json
v3_tune24_metrics.parquet
v3_seed_stability.json
v3_internal48_metrics.parquet
v3_final_analysis.json
viewer/best
viewer/median
viewer/worst
```

所有结果文件记录 Git commit、命令、参数、scene、seed和运行时间；不增加无必要SHA。

## 14. 当前执行检查点

### 已完成

- [x] 汇总 B0/B1、B2、class-first、prior-v2、teacher 和 selective-restore 数值证据。
- [x] 审计 source/a800、source/refactor 和当前 teacher 路径。
- [x] 确认旧结果不能证明类别先验整体失败。
- [x] 识别候选级 GT 漏斗为最大证据缺口。
- [x] 结合 SAGA、SoftGroup、HAIS、PBNet 和 HDBSCAN 机制重构实验设计。
- [x] 将V3计划冻结到本文件。
- [x] 实现并测试 Stage 0 轻量导出命令（commit `83be51254eff18debe0b1d52e6ca0011e7c449ac`）。
- [x] 验收 B0/B1 历史锚点、train-only 尺寸边界和 GT-only 诊断8场景。
- [x] commit `a11a138b5f4c494fa25a5c649e9cdb423c622c17` 实现 Stage 1 shadow/oracle；legacy最终输出保持旁路不变。
- [x] 冻结8场景审计：1916个候选；small GT=90；global/candidate Recall@0.25为0.1556/0.1111；新增small match=2，未过8场景门槛。
- [x] 自动扩展剩余16场景并冻结24场景审计：6391个候选；small GT=195；global/candidate Recall@0.25为0.1846/0.1897；新增small match=14，覆盖6个场景、4类，通过预注册第二门槛。
- [x] 核验所有20个全尺寸bin的oracle-positive新增候选：均保留多个global KNN/filter survivor anchors且2D vote winner等于branch class；据第7节唯一选择A“多-anchor局部保护”，不实现proposal级准入B。

### 下一步

- [x] 完成Stage 2多-anchor局部保护实现：同候选/同类、anchor数达到core、2D vote确认、物理半径内才恢复halo；禁止整簇恢复。
- [x] 全套测试、commit/push并部署同一commit（`7bd8be8`）。
- [x] 在冻结8场景完成结构门槛复测。
- [x] 生成并审计Stage 2指标；多-anchor结构未过门槛，停止进入Stage 3。
- [x] 追加一次不占GPU的保守proposal replay诊断（`aefe150`）；未改善严格mAP，停止扩展。

当前不得直接进入2³、24场景或48场景。V3 Stage 2已按预注册门槛停止；若继续，必须先形成新的候选生成实验，而不是继续放宽保护阈值。

### Stage 1 冻结产物

- 8场景：`v3_proposal_funnel.parquet`、`v3_input_ceiling.json`、`v3_proposal_oracle_analysis.json`
- 24场景：`v3_proposal_funnel_24.parquet`、`v3_input_ceiling_24.json`、`v3_proposal_oracle_analysis_24.json`
- 云端目录：`/root/autodl-tmp/saga/artifacts/teacher-prior-v3-a11a138`

### Stage 0 已验收事实

- 云端代码：`/root/autodl-tmp/saga/workspace/teacher-prior-v3-83be512`
- 云端产物：`/root/autodl-tmp/saga/artifacts/teacher-prior-v3-83be512`
- Windows副本：`F:\\3DGS_Research\\saga\\artifacts\\teacher-prior-v3-83be512`
- `v3_history_anchor.parquet`：6行，严格为B0/B1 × seeds
  `42/3407/20260804` × 48 scenes；复用并校验已有官方 locked evaluator 指标。
- B0 三seed平均 mAP：`0.0553383453`；B1：`0.0561636026`；
  B1−B0：`+0.0008252573`。
- train-only有效实例：10,841；物理尺寸边界：
  `tiny ≤ 0.871638m`、`small ≤ 1.306831m`、`medium ≤ 1.938700m`、其余为large。
- 24 tune GT：361个实例；tiny/small/medium/large分别为140/55/70/96；
  `<100`映射点实例26个。
- 冻结诊断8场景：
  `scene0645_00`、`scene0025_01`、`scene0046_00`、`scene0474_01`、
  `scene0591_02`、`scene0329_02`、`scene0164_03`、`scene0064_01`。
- 8个场景对应8个不同physical scenes；覆盖全部17个在tune中实际有tiny/small实例的类别。
- 剩余16场景已在 `v3_diagnostic8_scenes.json` 中冻结，留作Stage 4复核。
- Stage 0 未运行postprocess、未下载、未训练、未读取任何方法AP来选择场景。

## 15. 参考依据

- SAGA：promptable、多粒度scale-gated affinity feature；官方示例训练10k affinity迭代：
  <https://arxiv.org/abs/2312.00860>，
  <https://github.com/Jumpat/SegAnyGAussians>
- SoftGroup：hard semantic grouping 会传播语义错误并制造低IoU/FP，需候选质量控制：
  <https://arxiv.org/abs/2203.01509>
- HAIS：同类fragment按实例/类别尺度合并，并进行proposal质量处理：
  <https://arxiv.org/abs/2108.02350>
- PBNet：可靠高密度core先聚类，低密度/noise点再按邻域投票补回：
  <https://arxiv.org/abs/2207.11209>
- HDBSCAN：`min_samples`与`min_cluster_size`必须显式区分：
  <https://hdbscan.readthedocs.io/en/latest/parameter_selection.html>

## 16. 阶段记录

### 2026-08-15 — V3.0 冻结

- 放弃将 `f1367fa` selective restore 扩展为正式方法。
- 将问题拆为输入上限、候选质量、死亡阶段、保护结构和三类数据先验。
- 采用 physical-scene block、2×2结构实验和2³完整因子设计。
- 明确 final48 仍复用原场景，称为内部验证，不重新下载或训练。

### 2026-08-15 — Stage 0 完成

- 实现 `prepare-v3-stage0`，新增4项定向测试；全套
  `tests/category_priors` 为120 passed。
- 代码经commit `83be51254eff18debe0b1d52e6ca0011e7c449ac`部署并运行。
- 第一次云端启动因PowerShell提前展开远端变量而在程序启动前退出；未生成或覆盖产物。
  修正命令引用后只重试一次并成功。
- 三项注册产物均可解析；历史锚点条件/seed/scene count、尺寸桶、physical-scene唯一性、
  tiny/small类别覆盖均通过验收。
- 完成后磁盘空闲165GB，GPU空闲，cgroup约37.1GiB/90GiB，`oom_kill=0`。
- 下一允许动作是实现Stage 1 shadow/oracle；禁止跳到保护融合或效果筛选。

### 2026-08-15 — Stage 2 停止

- 24场景shadow/oracle共3,202个exclusive候选，只有78个候选的同类GT best IoU达到0.25，
  15个达到0.50；候选质量分布本身高度偏向低IoU。
- 多-anchor保护8场景结构门槛失败：B1 exact mAP为`0.05795314`，
  U0 multi-anchor为`0.03038257`，差值`-0.02757058`；覆盖率相近，
  但实例数从每场15.125增至36.5，属于FP/过分割而非单纯漏召回修复。
- 同一失败保护结构内，data相对uniform虽有`+0.00267887` mAP，
  但不能抵消结构主干约`-0.02757`的损失，因此不得进入2³或扩大场景数。
- 额外执行一次固定严格规则的离线proposal replay：仅接纳2个proposal，只填B1背景点，
  不改已有B1实例。AP25由`0.21676888`升至`0.22073713`，但AP50由
  `0.18867544`降至`0.18702201`，mAP由`0.05795314`降至`0.05762245`；
  差值`-0.00033069`，95% CI为`[-0.00109036,-0.00000422]`。
- 输入上限审计显示tiny/small实例的SAM覆盖仅`17.97%/27.08%`，语义top-1召回仅
  `13.96%/21.27%`。exclusive候选Recall@0.25对tiny仅由global的`15.00%`升至
  `16.43%`，对small反而由`27.27%`降至`25.45%`。当前瓶颈首先是候选输入和候选质量，
  不是再放宽保护规则即可解决。
- 结论边界：本阶段否定的是当前all-class候选生成与两种保护/融合实现；没有否定类别先验
  或小物体保护的研究假设。下一轮若继续，应只在既有资产上先改候选生成（类别scale gate、
  core支持与core-halo），并先用oracle recall验证，不再直接修改最终B1输出。

后续每次阶段完成均在此处追加日期、commit、产物和门槛判定。

## 17. V4：输入优先的最小重验证（当前权威计划）

### 17.1 状态与研究边界

V3已经按预注册门槛停止。V4不再放宽保护阈值，也不直接进入旧2³实验；顺序固定为：

> 两场景10k特征正控 → 8场景类别候选2×2 → 单一保守融合 → 24场景复核 → 48场景内部验证。

全程复用已有3DGS、GT、相机、masks、labels和scale gate，不下载数据、不训练3DGS。
10k affinity/semantic feature训练只限 `scene0011_00` 与 `scene0608_00`，输出到独立目录，
不得覆盖2k资产，也不得把10k正控结果混入正式uniform/data比较。

### 17.2 Stage A：2k与10k正控

- 相同30k 3DGS、masks、labels、相机和seed，从头训练10k轮feature。
- 比较SAM覆盖、semantic top-1 recall、同/异实例affinity margin、tiny/small候选
  Recall@0.25/0.50、IoU≥0.25匹配数和候选精度。
- 明显改善门槛：平均semantic recall提高≥0.05，或tiny/small Recall@0.25提高≥0.10；
  同时新增≥2个IoU≥0.25实例匹配，任一场景不损失超过1个原匹配，候选数≤2k的1.5倍。
- 无论是否通过，未经额外授权不得扩展10k训练。

### 17.3 Stage B：8场景候选2×2

候选模式固定为 `uniform`、`class-scale`、`class-core`、`combined`。四臂共享32类top-1
竞争、SAGA20赢家过滤、semantic阈值、距离权重、assignment阈值、sample cap和显式
`min_samples=3`；同scene/seed使用同一随机排列，采样数变化只取嵌套前缀。shadow只生成
候选，不改变B1输出。

- `class-scale`：以train-only `log_bbox_diag_m.q50`得到 `d_c`，用场景mask-scale经验CDF
  得到 `g_c`，仅通过 `scale_gate(g_c)`重算该类affinity feature，不改XYZ归一化。
- `class-core`：由当前场景Gaussian表面密度、train-only典型表面积和实际采样率估计
  `m_c`，限制到[3,20]；只改变`min_cluster_size`。
- `combined`：同时启用以上两项。

进入融合的候选臂必须同时满足：tiny/small Recall@0.25相对uniform提高≥0.02；新增≥5个
IoU≥0.25匹配且覆盖≥2类、≥4场景；候选精度≥uniform的80%；候选数≤uniform的1.5倍；
正向场景多于负向。只选一个best，依次按tiny/small Recall@0.25、Recall@0.50、候选精度、
更简单单因素破并列。无臂通过则停止V4。

### 17.4 Stage C：唯一允许的融合

只有Stage B通过后才实现并运行pointwise-evidence融合。B1始终为基线；候选须满足2D vote
赢家为branch class、比例≥0.60且高于背景、至少100个pointwise同意的core点、HDBSCAN
persistence≥0.05。与同类B1实例IoU≥0.25时仅合并core；与全部B1实例IoU≤0.25时core
可新建实例；与异类实例IoU>0.25时拒绝。halo只能恢复B1背景点，且须同候选、同类别、
半径≤0.1d_c、至少3个同候选core anchor，不得覆盖任何B1实例。

8场景比较B1、uniform+融合、best+融合。uniform结构门槛：相对B1 mAP下降≤0.002、
实例数≤1.5倍、覆盖不下降、AP50下降≤0.005。best进入24场景门槛：相对uniform
ΔmAP≥0.002，或tiny/small Recall@0.50提高≥0.01且总体mAP不下降；正向场景更多，
FP/TP恶化≤20%。失败后不得放宽融合阈值。

### 17.5 Stage D/E：扩展与停止

Stage D先在剩余16个tune场景运行uniform/best seed42；剩余16 ΔmAP>0、全24
ΔmAP≥0.002、正向场景更多且tiny/small Recall@0.50为正后，才补seed 3407与20260804。
进入48场景要求三seed平均ΔmAP≥0.002、至少2/3 seed为正、小物指标为正且结构指标仍合格。

Stage E只运行48×{uniform,best}×3 seeds。每个physical scene先平均seed，再做10,000次
paired bootstrap。结果不得再用于修改候选、阈值或融合；48场景仍称内部验证。

### 17.6 V4验收产物

`v4_feature_10k_control.json`、`v4_candidate_factorial8.parquet`、
`v4_candidate_analysis8.json`、`v4_fusion8_metrics.parquet`、
`v4_tune24_metrics.parquet`、`v4_final_metrics.parquet`、`v4_analysis.json`和`viewer/`。

### 17.7 当前执行检查点

- [x] V3 Stage 2按门槛停止，未进入旧2³。
- [x] V4实验顺序、因素、融合规则和停止门槛冻结到本文件。
- [x] 实现并测试Stage A正控和Stage B shadow候选接口。
- [x] commit/push并部署相同commit（Stage A/B实现`c62554e`；10k资产路由与resume身份修复`93ab3ee`）。
- [x] 运行两场景10k正控并冻结诊断结论；未过改善门槛，不扩展10k训练。
- [x] 运行8场景候选2×2并按门槛决定是否停止：Stage B未通过，V4按预注册规则停止。
- [x] 未实现Stage C融合，也未运行24或48场景。

### 2026-08-15 — V4.0 冻结

- 将输入/候选质量置于最终实例融合之前。
- 10k只作两场景诊断正控，不改变正式比较资产。
- 将类别先验收缩为两个可归因因素：SAGA原生class-scale gate和density-calibrated core支持。
- 明确只有候选门槛通过后才允许实现唯一融合结构。

### 2026-08-15 — V4 Stage A/B 实现完成

- 新增隔离的两场景10k训练入口；输出feature PLY、scale gate、日志和记录均位于
  `feature-10k-control/<scene>/`，不写入原2k资产目录。
- 新增四臂V4 shadow入口。四臂共用32类top-1/SAGA20过滤、距离、阈值和按类确定性
  随机排列；`class-scale`只改变SAGA scale gate，`class-core`只改变显式
  `min_cluster_size`，`min_samples`固定为3。
- 新增2k/10k正控比较器与8场景候选门槛评估器；shadow路径继续输出原始B1结果，候选
  仅写独立JSON/NPZ。
- 定向及全部`tests/category_priors`共141项通过；`compileall`与`bash -n`通过。
- Stage C融合尚未实现，只有Stage B通过才允许落盘。

### 2026-08-15 — V4 Stage A 完成

- `scene0011_00`与`scene0608_00`的10k feature和scale gate均写入独立目录，原2k资产未覆盖。
- 10k相对2k：semantic top-1 recall平均`+0.00939`，tiny/small候选Recall@0.25
  `-0.15`，新增IoU≥0.25匹配`0`，`scene0608_00`损失3个原匹配；候选数
  `174→190`，候选precision@0.25由`0.0977→0.0737`。
- 未达到预注册明显改善门槛。结论：简单把feature训练从2k增加到10k没有改善这两个场景的
  候选质量；不得将10k训练扩展到其他场景。
- 首次10k shadow启动暴露CLI漏传feature-control路径；错误产物已整体移至带说明的诊断目录，
  未混入正式比较。修复后新增运行命令身份校验，全部143项测试通过（`93ab3ee`）。
- Stage B的32次旁路候选实验已按冻结8场景启动。

### 2026-08-16 — V4 Stage B 完成并停止

- 冻结8场景、4条件、seed42的32次shadow均完成且可解析；B1最终输出始终保持旁路不变。
- `uniform`：79个官方有效tiny/small GT，Recall@0.25=`0.15190`、Recall@0.50=`0.05063`，
  599个候选，precision@0.25=`0.03339`。
- `class-scale`与uniform的两项Recall完全相同；新增IoU≥0.25匹配为0，正向场景为0。
- `class-core`：Recall@0.25=`0.12658`、Recall@0.50=`0.03797`，448个候选，
  precision@0.25=`0.04018`；相对uniform无新增匹配，2个场景退化。
- `combined`：Recall@0.25=`0.12658`、Recall@0.50=`0.03797`，450个候选，
  precision@0.25=`0.04000`；相对uniform无新增匹配，2个场景退化。
- 三个数据臂均未达到预注册联合门槛，`best_candidate=null`、`stage_b_passed=false`。
  因此不实现pointwise-evidence融合，不运行24/48场景，也不放宽阈值。
- 结论边界：两场景10k正控未改善，8场景class-scale/class-core映射也未改善pre-fusion候选；
  这否定当前输入表示下这两种具体映射的继续计算价值，不等价于证伪所有类别先验或老师未留存的人工配置。
- 验收产物：`v4_candidate_factorial8.parquet`（2638行）与
  `v4_candidate_analysis8.json`；运行结束后GPU空闲，磁盘空闲158GB，cgroup无OOM事件。

## 18. V6 执行检查点：Affinity-first 候选质量重验证

> 2026-08-16 用户以 V6 计划取代 V4/V5 的后续执行路径；完整的冻结设计见
> [`AFFINITY_FIRST_V6_EXPERIMENT_PLAN.md`](./AFFINITY_FIRST_V6_EXPERIMENT_PLAN.md)。

- [x] V5 两场景旁路证明没有意外修改 B1 输出；V5 八场来源候选绝对门槛失败，因此
  未进入其融合、校准、24 或 48 场景阶段。
- [x] V5 失败定位在候选供给：codebook 只有 5 个同类 IoU≥.50 候选，multiview 为 0；
  不得把它解释为类别先验失败。
- [ ] V6 Stage 0：固定公开上游与本仓历史差异，复用 B0/B1，并在 V3 诊断八场生成
  SAM/语义/affinity 输入漏斗与 affinity-first graph candidate bank。
- [ ] 仅由 Stage 0 判定是否进入八场 SAM 输入修复或三场 10k feature 正控；两者都不
  自动扩展。
- [ ] V6 候选绝对门槛通过后，才实现唯一保守 B1 replay；融合安全后，才比较 U00/D10/D01/D11。

V6 的候选形成不再预先按语义类路由：全体 Gaussian 先建立物理 24-NN 中 mutual-top4
affinity 图，再由多视角 32 类投票确认 SAGA20 类别。B1 输出始终旁路保留；GT 只用于离线
审计和评估。不得恢复 V5 或继续放宽已停止方案的阈值。

## 19. V7 当前权威计划：正确 lifting、跨视角 track 与终局先验检验

### 19.1 已确认的 P0 根因

- max-contributor CUDA 过去用 `alpha*T_new` 选赢家，但颜色实际使用 `alpha*T_prev`；
  无贡献像素又默认返回 Gaussian 0。V5/V6 未同时过滤零贡献和非法 ID。
- 该错误至少从 `source/v2` 已存在，不是 a800 类别分支或近期实验新增。
- 修复后统一使用 `weight=alpha*T_prev`，空像素返回 `id=-1, weight=0`，Python 只接受
  `id>=0 && weight>0`。旧结果只作 `B1-historical`，不得冒充修复后基线。
- 旧自动主干缺少跨视角 object ID，且单帧语义路由、中心吸附、全局 256-NN 和最终 vote
  会连续改写实例。旧实验因此不能证伪类别先验。

### 19.2 唯一 V7 对象主干

每帧用修正后的 contributor ID/weight 将已有 mask 提升为 Gaussian fragment。每个
Gaussian 至少有 2 个 mask 像素且其可见像素至少 50% 在 mask 内才属于 core；fragment
至少 5 core、10 full。语义只随 fragment 保存，不参与 track 关联。

fragment 按固定帧序关联：core overlap coefficient≥0.25、共享 core≥3、最佳相对第二名
margin≥0.10；同一 track 每帧最多一个 fragment，模糊桥接必须新建 track。有效 track
至少 2 帧；最终唯一 core 要求 positive views≥2、positive/visible≥0.60、
conflict/visible≤0.25 且至少 10 点。

是否启用一次性局部 halo 只由两场因果消融决定。halo 限定 fragment union、5 cm、3 个
同 track anchor、affinity cosine≥0.95、最佳 track margin≥0.02；不迭代、不跨 track。

track 完成后才进行 32 类逐帧投票：SAGA20 winner、有效语义视角≥2、winner ratio≥0.60、
margin≥0.10。候选基础分数 Q 按用户冻结公式计算。

### 19.3 类别先验的唯一作用位置

fragment、track、core、halo 和类别全部先冻结。CPU replay 只比较：

- `U00-uniform`：global size × global support；
- `D10-size`：class size × global support；
- `D01-core`：global size × class support；
- `D11-combined`：class size × class support。

四臂均使用 `S=QGC`、接受阈值 0.20；类别统计只读 ScanNet-train
`category_priors.json`，缺失类别回退 global。prior 不得改变候选构造。

### 19.4 冻结阶段与停止门槛

1. **Stage 0（scene0645_00、scene0025_01）**：P0历史 B1、L0 contributor-fixed、
   L1去全局KNN、L2只留HDBSCAN core、L3加一次性局部attach；同时计算单mask、完美关联、
   score oracle。association oracle须有≥6个同类IoU≥.50匹配，且有效tiny/small
   Recall@.25≥.20，否则停止。
2. **Stage 1（冻结8个physical scenes）**：确定性uniform bank须有≥12个同类IoU≥.50
   候选、覆盖≥4场景、precision@.25≥10%、tiny/small Recall@.25≥.20；Gaussian micro
   precision相对B1-fixed提高≥5个百分点或unsupported实例比例下降≥10个百分点；GT recall
   下降≤5个百分点；U00的mAP/AP50/实例数和score-IoU相关性须满足用户冻结门槛。
3. **Stage 2**：同一bank replay四臂；数据臂相对U须ΔmAP≥.002（或冻结的小物条件）、
   正向场景更多且FP/TP恶化≤20%。机械上不改变分数/接受集合只能判为未生效。
4. **Stage 3**：先在5个未开发physical scenes验证 U/best-D，要求mean ΔmAP>0且至少3/5
   为正；再把其余11个重复扫描按physical scene内平均，13个physical scenes宏平均
   ΔmAP≥.002才进入final。
5. **Stage 4**：48个不同physical scenes各建一个确定性bank，只replay U/best-D；
   10,000次paired bootstrap，ΔmAP≥.002且95% CI下界>0才支持稳定有效。final不得调参。

主比较严格拆为 `V7-U − B1-fixed`（对象主干）和 `V7-D − V7-U`（类别先验）。

### 19.5 工程边界和验收产物

- 独立 `v7_objects.py/v7_worker.py/v7_runner.py/v7_replay.py/v7_evaluation.py`；不向巨型
  postprocess继续添加正式 V7 路径。B1只用于只读历史/因果对照。
- 不下载、不训练、不覆盖旧结果；单GPU单进程；磁盘≥80GB；内存只看90GiB cgroup。
- 不生成SHA文件、lock、schedule hash或contributor cache；完整bank复用，损坏/缺失才重跑。
- 产物：`v7_contributor_audit2.json`、`v7_causal_ablation2.parquet`、`v7_oracle2.json`、
  `v7_bank8.parquet`、`v7_prior_replay8.parquet`、`v7_tune24_metrics.parquet`、
  `v7_final_metrics.parquet`、`v7_analysis.json`和viewer。

### 19.6 当前执行检查点（2026-08-22）

- [x] P0 CUDA根因静态确认并修正源码；Python legacy vote过滤非法 contributor。
- [x] V7 fragment/track/core/halo与prior replay纯算法实现并建立定向测试。
- [x] V7 worker、顺序runner、bank/replay评估和阶段controller初版完成。
- [ ] 本地/云端编译修正后的CUDA扩展并跑全部定向测试。
- [ ] commit/push `origin/a800`，部署同commit到固定云端目录。
- [ ] 云端启动Stage 0并验证runner健康。
- [ ] runner健康后创建并核验绑定当前主任务ID的每小时Codex自动化。

### 19.7 V7 实际闭环（2026-08-23）

- V7 代码以 commit `8820c0a5dc27125e36d1fef94421541e505c7990` 部署到
  `/root/autodl-tmp/saga/workspace/v7-object-tracks`，本地/云端定向测试和 CUDA 扩展验收完成。
- Stage 0 两场景已完成；`v7_status.json` 为 `stopped`。P0 historical 与 L0
  contributor-fixed 最终点标签逐点一致。L0 两场 mAP=`0.0544444`、AP50=`0.1833333`、
  Gaussian micro precision=`0.238188`、matched-GT recall=`0.621284`。
- L1 去 global KNN、L2 core-only、L3 local attach 均未满足注册的 precision/recall 因果门槛；
  因此 global KNN 污染和中心吸附污染未被确认，halo 判定为关闭。
- V7 bank 共 56 个候选，same-class IoU≥.25 仅 1 个、IoU≥.50 为 0；tiny/small
  Recall@.25=`0.047619`。原 V7 association oracle 的 IoU≥.50 为 0，tiny/small
  Recall@.25=`0.142857`，未过门槛，未进入 8/24/48 场景。
- 后续只读复核发现原 oracle 同时混入类别，并无条件 union fragment，不是纯几何且不是单调上界；
  但改成 class-agnostic、只接受 IoU 改善的组合后，IoU≥.50 仍为 0。5 cm GT→Gaussian
  覆盖为 `98.56%`，10 cm 为 `99.94%`，排除坐标和映射半径为主因。
- 对所有现有 fragment 做 perfect-trim 支持上界时，忽略类别也只有 2/38 个 GT 具备
  IoU≥.50 的覆盖；最佳候选表现为局部 precision 高而 recall 很低。V7 停止证明当前
  Grounded-SAM + max-one lifting 的候选支持不足，**没有检验、更没有证伪类别先验**。

## 20. V8 当前权威计划：Mask × Alpha 因果审计与对象 bank

### 20.1 唯一研究顺序

V8 先区分 Grounded-SAM mask 覆盖与 max-one lifting 两个根因，再建立不依赖语义路由的
对象 bank，最后只在冻结 bank 上比较类别先验。公开 LBG 的 max contributor 与 Trace3D
式全 alpha attribution 都作为有效假设，不预判哪一种正确。

Stage 0/1 固定 `scene0645_00`、`scene0025_01`，运行：

```text
G-M1  G-AM
S-M1  S-AM
```

- `G` 为现有 Grounded-SAM mask；`S` 使用现有 checkpoint 按固定官方参数生成
  segment-everything mask，写独立目录。
- `M1` 使用修正后的最大 `alpha*T_prev` contributor，并把该像素归一化为赢家的
  单位 one-hot mass；`AM` 使用每像素对全部实际
  contributor 归一化的 alpha mass，不保存逐像素 contributor cache。
- 两种 lifting 共用：full `inside_mass>=0.5`；core `inside_mass>=2` 且
  `inside_mass/visible_mass>=0.50`；fragment 至少 5 core、10 full。
- oracle 拆为 geometric/semantic single、单调 greedy upper bound 与 perfect-trim
  support ceiling。GT 只进入离线 evaluator。

进入 8 场景的组合须有 geometric greedy IoU≥.50 匹配至少 6 个，且 official-valid
tiny/small Recall@.25≥.20。因素实质作用定义为相对对照新增至少 2 个 IoU≥.50 匹配，
或 tiny/small Recall@.25 提高至少 .05。四臂均失败则停止纯后处理。

### 20.2 V8 确定性对象 bank

- fragment 关联不读取类别；weighted core overlap≥.25、共享 core≥3、best-second
  margin≥.10。同帧不合并，模糊 fragment 新建 track，不允许桥接已有 track。
- consensus 只累计 `core_ids`；positive/conflict 每物理视角最多一次。core 要求≥2个
  positive views、positive/visible≥.60、conflict/visible≤.25、至少10点；full只来自
  成员 fragment union 且 positive mass ratio≥.40。不做全局 KNN、中心吸附或迭代 halo。
- 类别在 track 冻结后确定。开发 8 场景比较 MV-label 与完整32类 codebook，按几何
  IoU≥.25 候选的类别准确率选一个；差≤2个百分点时选 MV-label。非 SAGA20 不输出。
- 8 场景固定为 V7 `DEV8`。geometry IoU≥.50 候选须≥16/4场景；选定晚分类器的
  same-class IoU≥.50 须≥12/4场景；precision@.25≥10%、tiny/small Recall@.25≥.20，
  并通过相对 B1-fixed 的 Gaussian precision、recall、mAP/AP50、实例数和 score-IoU 门槛。
- 若 oracle 通过但 bank 失败，则停止在对象关联主干，不把失败归因到类别先验。
  affinity edge AUROC 仅作为输入表示诊断记录，不触发训练。

### 20.3 冻结 bank 上的类别先验

四臂固定为 `U00/D10/D01/D11`，候选、track、core、full、类别和输出规则完全相同；
prior 只替换 global/class size 与 support 统计，统一 `S=QGC`、阈值 .20。size 只惩罚
异常过大的 sorted extent；support 使用16-NN局部密度与 train-only典型表面积。缺失类回退
global。同类 core IoU≥.50 时按分数NMS；重叠Gaussian归最高分候选；最终少于10点的实例删除。

机械生效要求至少10%候选的D−U分数绝对差≥.01，或接受/所有权实际改变。best-D 进入
holdout须 ΔmAP≥.002，或 tiny/small Recall@.50≥.01 且 ΔmAP≥−.0005；正向场景更多且
FP/TP恶化≤20%。不通过时停止，不增加学习式校准或阈值搜索。

### 20.4 独立复核与 final

先运行 `scene0231_00/scene0608_00/scene0356_00/scene0011_00/scene0593_00`；要求平均
ΔmAP>0、至少3/5为正、tiny/small Recall@.50为正。之后才运行 tune24 其余重复扫描，
先在同一 physical scene 内平均，再对13个physical scenes等权；宏平均 ΔmAP≥.002 才进入
final48。

final48 每场只生成一次确定性 bank，再 CPU replay U/best-D；不制造多 seed。主检验为
10,000次 physical-scene paired bootstrap，ΔmAP≥.002 且95% CI下界>0。final不得调参。

### 20.5 工程边界、产物和当前检查点

- 新增独立 V8 lifting/object/worker/runner/replay/evaluation，不恢复 V3–V7 旧实验运行时，
  不写兼容 adapter。
- 不下载新权重、不重训3DGS；SAM-everything只用已有checkpoint。单GPU单进程，磁盘≥80GB，
  内存只按90GiB cgroup。
- 不生成SHA文件、lock、schedule hash或 contributor cache；只保存 compact fragment/bank。
- 产物：`v8_provenance_and_v7_closeout.json`、`v8_lifting_factorial2.parquet`、
  `v8_lifting_analysis2.json`、`v8_bank8.parquet`、`v8_bank8_analysis.json`、
  `v8_prior_replay8.parquet`、`v8_tune24_metrics.parquet`、
  `v8_final_metrics.parquet`、`v8_analysis.json`和viewer。

### 20.6 结果采集前的静态因果纠错（2026-08-23）

实现审查发现，20.2 冻结的 V8 bank 只读取 mask lifting、跨视角 weighted overlap 和
late semantics，**完全不读取 affinity feature**。因此原草案中“bank 失败后把2k feature
换成10k，并要求自动几何候选新增”的正控在数学上不可能生效；若通过，只能来自同时改变
semantic/classifier 的混杂。该分支在任何 V8 结果产生前删除，不属于结果驱动调参。

V8 仍记录 selected-mask 局部 affinity edge AUROC，但它只解释输入表示，不作为 bank 失败的
升级门。若 Stage 1 oracle 通过而 Stage 2 bank 失败，结论固定为 mask-overlap 跨视角对象主干
未达到健康门槛；停止，不训练10k，不把失败外推为类别先验无效。

当前检查点：

- [x] V7 云端停止结果和只读根因复核完成。
- [x] V8 条件、公式、场景、门槛和升级边界在收集 V8 结果前冻结。
- [x] 实现并测试 V8 oracle、M1/AM lifting、对象 bank、晚分类和 replay（本地 `157 passed, 1 skipped`；skip 为本机无 Torch，待云端补测）。
- [x] commit/push 并部署；两场 Mask×Alpha 因果实验完成，`S-AM` 通过几何上限门槛。
- [x] 扩展到固定 8 场景；V8 自动 bank 未通过健康门槛，按预注册规则停止，未进入 prior replay/24/48。

V8 的实际停止结果：`S-AM` 两场 geometric greedy IoU≥.50 为 16 个，证明 SAM-everything
和 alpha-mass lifting 能提供几何支持；但 8 场自动 bank 只有 2 个 geometric IoU≥.50、
0 个 same-class IoU≥.50，U00 mAP 为 0、tiny/small Recall@.25 为 0，且 precision/unsupported
均劣于 B1-fixed。因此停止原因为跨视角 mask-overlap bank 不健康，不是类别先验失败。

## 21. 原始交付基线闭环（当前执行）

只读 Git/reflog 和老师聊天重新核对后，最早可证的 a800 交付锚点是 `bfc2192`，而非
`8c5e167`。前者是 18 类/4 个 `other_classes`；后者是交付后扩到 28 类/7 个小类的演化版本。
公开 SAGA `96e5021` 仍只作提示式分割上游，不是自动 ScanNet AP 基线。

当前实验严格按
[`TEACHER_BASELINE_CLOSURE_PLAN.md`](./TEACHER_BASELINE_CLOSURE_PLAN.md) 执行：

- 三个开发物理场景：`scene0064_01/scene0025_01/scene0231_00`；
- 隔离重建 handoff 的 18 类 masks/scales/features，不复用 32 类语义资产；
- 比较 `bfc2192` literal handoff、`95073c6` 训练修复、historical/fixed contributor、
  L1/L2/L3 后处理快照和 adaptive/10k 预算；
- 同时报官方 9 阈值、历史 10 阈值、Gaussian 精度和 handoff 可预测类交集；
- 三场结束即停止，不测试类别先验，不扩展 24/48。

当前检查点：

- [x] Git provenance、老师聊天、README、评估协议和三场资产范围完成只读审计。
- [x] 冻结原始基线闭环设计与结论边界。
- [x] 实现隔离 runner/evaluator、fixed contributor、L0 等价门槛、Gaussian 精度和 viewer。
- [ ] commit/push，云端部署同 commit，启动三场闭环。
- [ ] runner 健康后创建并核验绑定当前任务的每小时自动检查。

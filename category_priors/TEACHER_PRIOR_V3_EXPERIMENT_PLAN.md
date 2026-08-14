# SAGA 类别先验与小物体保护 V3 实验计划

> **权威状态：本文件是下一阶段唯一实验事实源。** 旧 B2、class-first、
> prior-v2、teacher-preservation 和 selective-restore 文档与代码只作为失败审计记录，
> 不得覆盖本文件的研究问题、条件定义、阶段门槛或结论边界。

- 版本：V3.0
- 冻结日期：2026-08-15
- V3 起点代码检查点：`f1367fa58c8f50df75f80b86f67bab469af06531`
- 当前状态：**只完成了审计和计划，V3 尚未实现或运行**
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

### 下一步

- [ ] 实现 Stage 0 的历史锚点/GT尺寸分层导出。
- [ ] 实现 Stage 1 shadow/oracle 候选漏斗，不改变legacy最终输出。
- [ ] 本地测试并回读本文件核对每个字段与门槛。
- [ ] commit/push后部署同一commit到云端。
- [ ] 运行8个GT支持充分的诊断场景。
- [ ] 根据漏斗事实决定 A 或 B 保护，禁止凭直觉同时实现两套。

当前不得直接进入2×2、2³、24场景或48场景。

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

后续每次阶段完成均在此处追加日期、commit、产物和门槛判定。

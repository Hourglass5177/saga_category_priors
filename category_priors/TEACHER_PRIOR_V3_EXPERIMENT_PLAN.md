# SAGA 类别先验与小物体保护实验计划（当前权威：清洁基线两步闭环）

> **权威状态：第34节的完整掩码几何覆盖上限健康，但自动共识输出不健康。复核又发现旧
> 候选几何评价混入密度相关的伪假阳性，且分层 SAM 掩码的大量重叠支持被作为歧义弃权。
> 当前执行第35节：先用冻结 DEV8 纠正评价并逐级定位损失，再用同一次 SAM 输出派生
> 分层 H′ 与平面化 P 两个条件，检验掩码观察合同是否是主结构故障。类别先验本轮不测试。**
> 旧B2、class-first、prior-v2、teacher-preservation和selective-restore文档与代码只作为
> 失败审计记录，不得覆盖本文件的研究问题、条件定义、阶段门槛或结论边界。

- 版本：CMC-1.0
- 冻结日期：2026-09-01
- V3 起点代码检查点：`f1367fa58c8f50df75f80b86f67bab469af06531`
- 当前状态：**清洁基线评价纠正与 H′/P 同源掩码合同正控正在实施；类别先验尚未进入本轮**
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

后续只读 Git/reflog、老师聊天与 2026-07-31 AutoDL 完整复现报告交叉核对后确认：
`bfc2192` 只能称为最早可见的 18 类/4 `other_classes` prototype ancestor，不能称为老师
2026-07 实际交付快照。实际交付候选更接近名义 `8c5e167` 加当时 dirty working tree；
办公室四阶段复现使用的也是该工作树。V9 法证检查已从不可达 `git stash` 恢复
`5804fcb2`：第一父提交为 `8c5e167`，tracked dirty tree 仅修改
`train_contrastive_feature.py`。该 tree 可字节恢复，但因原 `saga.zip` 没有历史校验值，仍以
办公室历史输出作行为 oracle，不宣称两者逐字节相同。公开 SAGA `96e5021`
仍只作提示式分割上游，不是自动 ScanNet AP 基线。

当前实验严格按
[`TEACHER_BASELINE_CLOSURE_PLAN.md`](./TEACHER_BASELINE_CLOSURE_PLAN.md) 执行：

- 三个开发物理场景：`scene0064_01/scene0025_01/scene0231_00`；
- 隔离重建 handoff 的 18 类 masks/scales/features，不复用 32 类语义资产；
- 比较 `bfc2192` literal handoff、`95073c6` 训练修复、historical/fixed contributor、
  L1/L2/L3 后处理快照和 adaptive/10k 预算；
- 同时报官方 9 阈值、历史 10 阈值、Gaussian 精度和 handoff 可预测类交集；
- 三场结束即停止，不测试类别先验，不扩展 24/48。

当前检查点：

- [x] Git provenance、老师聊天、README、评估协议和三场资产范围完成只读审计；后续已纠正
  `bfc2192` 的命名边界。
- [x] 冻结原始基线闭环设计与结论边界。
- [x] 实现隔离 runner/evaluator、fixed contributor、L0 等价门槛、Gaussian 精度和 viewer。
- [x] commit/push，云端部署同 commit，启动三场闭环。
- [x] runner 健康后创建并核验绑定当前任务的每小时自动检查。
- [x] 运行时纠正错误预注册：`bfc2192` 的模块全局 `args` 合法，literal adaptive 应真运行；
  首次 `--iterations 1` 失败是 harness 触发历史 NoneType CLI 缺陷，不属于老师基线。
- [ ] 修正 partition comparator 的全局 rank 假阳性后恢复闭环；prototype ancestor 的
  literal/args-plumbing/args-norm/full950 只作法证，实际方法结论以 8c reconstructed
  candidate 与办公室行为 oracle 为准。

## 22. V9 当前权威计划：法证闭环、10k Clean ObjectBank 与类别先验终局验证

### 22.1 已冻结的纠正

- `scene0025_01/B1` 的两份 raw `point_labels` 在 1,459,291 点上逐点相同。旧 comparator
  因为用全局实例 rank 作 canonical ID，把一个新增的 31-Gaussian metadata 实例放大成
  370,226 点差异。真实 exported 差异为 31 点（0.002124%）。撤销
  `stopped-current-harness-parity-failed` 的方法结论，修 comparator 后恢复 L1–L3。
- 已确认且仍需修复的 P0/P1：negative metadata、undeclared orphan、raw labels 与 metadata
  双真相、错误 contributor、类别顺序覆盖、全点中心吸附、全场 256-NN、分支类别被最终 vote
  改写、缺少原生 score。
- 四条比较必须分离：`T1-B1−T1-B0`、`F10k−T1`、`N0−F10k-legacy`、
  `N-data−N0-uniform`。后一级结果不得反向包装成前一级结论。

### 22.2 法证与唯一输出合同

provenance 固定为 official SAGA / bfc prototype / 950 training fix / 8c reconstructed
candidate 四锚点。使用已恢复的 `5804fcb2` dirty tree，并以办公室历史输出作行为 oracle。

partition 比较分为 internal raw partition、declared exported partition 与 metadata 三层；
实例用成员集合的最大重叠一一匹配，不使用全局 rank。最终输出规定：`-1` 只表示背景；所有
非负标签必须有 class、score 与非空 mask；未声明内部标签投影为背景；最终 ID 连续；内部
cluster/track ID 只写 diagnostics。官方 AP、Gaussian precision 与 viewer 共用该投影。

Legacy 固定 L0/L1/L2/L3：L0 为修 contributor/输出合同后的交付候选结构；L1 去全场 KNN；
L2 再去全点中心吸附；L3 加一次性局部同候选 attach。每一级保存 global core/assignment、
other candidate/class、merge、post-KNN、post-filter、vote 与 export sidecar。归因门槛沿用：
precision +5pp 或 unsupported -5pp 且 recall 损失不超过10pp；L3须恢复至少一半 recall 缺口并
保留80% precision增益。

### 22.3 直接 10k 双源特征

对每个实际进入当前阶段的场景固定现有30k 3DGS、相机、图像、checkpoint和seed42：
SAM-everything mask只监督affinity；Grounded-SAM 32类mask/label只监督semantic；训练10k，
输出到独立 `feature-10k-objectbank`，不覆盖历史feature/scale gate，不重训3DGS。相同10k
feature同时供 `F10k-B0/F10k-B1/N0` 使用，以分离输入和主干效应。

### 22.4 Clean ObjectBank

新主干独立于巨型 `postprocess.py`。每帧同时流式计算正确最大 `alpha*T_prev` contributor
与归一化 alpha mass；空像素为 `-1/0`。fragment规则固定：full inside mass≥0.5；core
inside mass≥2且inside/visible≥0.50；至少3 core、10 full；类别不参与生成或关联。

两场景固定比较：A0 V8顺序overlap负控、A1全局受约束overlap、A2 A1+10k affinity
bridge；A0–A2失败才运行A3物理24-NN/mutual-top4 affinity core graph。direct edge要求
跨帧weighted core overlap≥0.25且共享≥3 core；bridge只允许mutual-best、5cm、cosine≥0.95、
conflict≤0.25，并且只能把singleton附着component，不能桥接两个既有component。同帧为
cannot-link，合并确定性。

最终core要求positive views≥2、positive/visible≥0.60、conflict/visible≤0.25；一次性局部
attach只写未分配点，要求5cm、3个同object anchor、affinity≥0.95、best-second margin≥0.02，
禁止迭代和跨object覆盖。对象冻结后才比较MV-label与32类codebook晚分类；差≤2pp选MV。
基础score为语义、多视角支持、内部affinity、重叠、视角数与低冲突的固定几何平均。

### 22.5 同bank 2^3类别先验

bank保留core≥3候选及一次性halo证据。U/D不得重建fragment、graph或object ID。所有条件
同时包含size `G`、support `C`、smoothness `B`，只切换global/class-shrunk统计：

```text
U000 D100 D010 D001 D110 D101 D011 D111
S = Q * G * C * B
```

size仅惩罚过大sorted extent；support用局部Gaussian密度和train-only典型表面积；smoothness
用5cm图边界率相对q50/q75的单边破碎惩罚。阈值只在两个开发场景U000上从
`.05/.10/.15/.20/.25`选择，按官方mAP、结构门槛、并列更高阈值冻结；D不得单独调参。

### 22.6 阶段门槛

1. Stage 0：修provenance/comparator/output contract并恢复Legacy闭环。
2. Stage 1：完成L0–L3、historical/fixed contributor、B0/B1，输出AP与precision漏斗。
3. Stage 2（scene0645_00/scene0025_01）：10k geometric oracle须≥6个IoU≥.50且tiny/small
   Recall@.25≥.20；从A0–A3按几何匹配、association F1、merge/split、candidate precision和
   简单性选一个。A3仍失败则停止类别先验，归因为identity/tracker不足。
4. Stage 3（冻结8 physical scenes）：geometric≥16/4场景、same-class≥12/4场景、
   precision@.25≥10%、tiny/small Recall@.25≥.20；Gaussian precision相对T1 +5pp或
   unsupported -10pp；GT recall损失≤5pp；N0相对F10k-B0 mAP≥-.001、AP50≥-.002、
   instances≤1.25x、score-IoU Spearman≥.20、orphan/negative metadata为0。
5. Stage 4：先验机械生效后，best-D相对U须ΔmAP≥.002，或tiny/small Recall@.50 +.01且
   ΔmAP≥-.0005；正向场景更多、FP/TP恶化≤20%。只保留一个D。
6. Stage 5：五个未开发canonical scenes要求mean ΔmAP>0、至少3/5为正、小物指标为正；
   再按physical scene聚合tune24，宏平均ΔmAP≥.002才进入final。
7. Stage 6：48 physical scenes每场一个10k bank，只CPU replay U/best-D；10,000次paired
   bootstrap，ΔmAP≥.002且95% CI下界>0才支持稳定有效，final禁止调参。

### 22.7 工程与当前检查点

- 不下载、不重训3DGS；单GPU单训练/后处理；磁盘≥80GB；只读90GiB cgroup；不生成SHA、
  lock、schedule hash或contributor cache。
- Stage 3健康通过后，从active runtime删除B2/class-first/prior-v2/V3–V8入口；Git与只读
  audit runner保留历史，不写兼容adapter。
- [x] V9计划经用户确认并冻结到本节。
- [x] 修正 comparator 与输出合同，更新原始闭环状态；加入31-Gaussian插入回归、
  internal/exported/metadata三层比较和strict prediction contract。
- [x] 实现隔离10k双源训练、原生alpha lifting、Clean ObjectBank、2^3 replay、
  Stage 2--6 controller、runner/evaluator与本地测试（`320 passed, 1 skipped`；
  skip为本机缺少云端CUDA/Torch运行环境的条件测试）。
- [ ] commit/push、云端部署并恢复Stage 0–1。
- [ ] runner健康后创建并核验绑定当前任务ID的每小时自动化。

### 22.8 V9 实际停止与结论纠正（2026-08-25）

- T1 16/16、两场10k双源feature和两场S-AM lifting均完整；A0–A3两场全部完成。
- geometric oracle为`16/38`个IoU≥.50，tiny/small Recall@.25=`0.954545`，证明输入
  fragment-full支持充足；但A0–A3最终bank的geometric IoU≥.50均为0，类别先验未运行。
- A0/A1/A2/A3记录的association pair precision约为`0.0506/0.0637/0.0636/0.0615`，
  但后续静态复核发现A0/A2保存的是component代理边，不能把该数值直接解释为真实identity精度。
- 进一步确认V9最终candidate完全丢弃成员fragment的`full_ids`，只保留consensus core与
  全场halo；current overlap又是对小mask有利的containment，并会经全局连通传递放大假边。
- 因此撤销“10k identity/tracker表示已被充分证实不足”的过早归因。V9只证明其具体
  association/reconstruction实现失败；没有检验、更没有证伪类别先验。

## 23. V10 当前权威计划：证据保真的多视角共识 ObjectBank

### 23.1 唯一研究顺序

V10先修正V9的真实edge评价与fragment-full证据重建，再以训练免费的多视角共识替换
全场贪心连通；只有uniform bank健康后才运行类别先验。主比较固定为：

1. `V10-U − B1-fixed`：对象主干是否健康；
2. `V10-D − V10-U`：train-only类别统计是否带来额外收益。

首轮只读复用`scene0645_00/scene0025_01`的完整S-AM lifting，不下载、不训练、不重做lifting。

### 23.2 Stage 0：评价修正与完整漏斗

评价必须分别保存single fragment full/core、component full/core union、冲突前后consensus、
唯一归属和最终candidate。accepted edge必须是真实支持fragment pair；same-GT、different-GT、
GT-unknown分开统计，并报告identifiable precision、all-edge precision和unknown rate。
无GT映射的预测Gaussian以每点唯一sentinel计入诊断IoU的FP；官方ScanNet evaluator不变。

归因门槛：full oracle为16而core oracle<6时归因core；component-full≥6而final<6时归因
重建；component-full<6且identifiable edge precision<50%时归因关联；同帧多重声明>30%且
冲突删除>50%正支持时归因层级mask替代假设处理。

### 23.3 Stage 1A：P×R 两因素闭环

两场运行`P0R0/P1R0/P0R1/P1R1`：P0为V9 containment；P1以逐Gaussian
`p=clip(inside/visible,0,1)`计算双向coverage及其几何平均，要求共享≥3且两个方向≥.25。
R0为V9 core+halo；R1只从成员fragment-full union重建，membership≥.40进入full、≥.60且
至少两视角进入core；跨track最佳与第二名差<.10时保留背景，不向union外扩张。

### 23.4 Stage 1B：最终 view-consensus

- frame visible alpha mass的weighted Jaccard构造共视图；每帧top-8，取对称并集；
- 每个共视frame pair对全部fragment运行Hungarian；仅保留mutual-best、双向coverage≥.25、
  行列margin≥.10的匹配；
- 双向coverage均≥.80的强边可形成两视角track，其他边必须属于三视图一致cycle；
- component合并不得含同帧两个fragment，可比较跨component证据支持比例须≥.80；每次合并
  后重算，不允许单条弱边桥接两个既有component；
- 同帧层级mask是替代假设，不互记负冲突；最终full/core固定使用R1；禁止全场KNN、中心吸附、
  迭代halo和affinity graph；类别在track冻结后才确定。

进入8场景须同时满足：geometric IoU≥.50累计≥6、candidate precision@.25≥10%、
official-valid tiny/small Recall@.25≥.20、identifiable association precision≥50%，且candidate
数量≤P0R0的1.5倍。失败时必须生成`V10B_IDENTITY_TRAINING_PROPOSAL.md`并明确请求批准
新的跨视角identity训练；停止当前计算不等于放弃后续。

### 23.5 Stage 2：8场景uniform健康门槛

沿用冻结DEV8。完整且身份一致的S-AM lifting直接复用，缺失时只补lifting。几何IoU≥.50
须≥16/4场景、same-class≥12/4场景、precision@.25≥10%、tiny/small Recall@.25≥.20；
Gaussian micro precision相对B1-fixed提高≥5pp或unsupported下降≥10pp；GT recall下降≤5pp；
U000相对B1-fixed mAP≥-.001、AP50≥-.002、实例数≤1.25x；score-IoU Spearman≥.20，
orphan与negative metadata均为0。失败则转入V10B审批，不运行prior。

### 23.6 Stage 3–4：同bank先验与内部验证

U000阈值只在两开发场景从`.05/.10/.15/.20/.25`选择并冻结。运行
`U000/D100/D010/D001/D110/D101/D011/D111`；所有条件共享candidate/track/full/core/类别和
`S=QGCB`结构，只切换global/class-shrunk统计。机械生效要求≥10%候选`|D-U|≥.01`或接受/
所有权变化。best-D须ΔmAP≥.002，或tiny/small Recall@.50 +.01且ΔmAP≥-.0005；正向场景
更多，FP/TP恶化≤20%。

随后先验证五个canonical holdout，要求mean ΔmAP>0、至少3/5为正、小物指标为正；再按
13个physical scenes等权汇总tune24，宏平均ΔmAP≥.002才进入final48。final每场一个确定性
bank，只CPU replay U/best-D；10,000次physical-scene paired bootstrap，ΔmAP≥.002且95%CI
下界>0才支持稳定有效。final不得调参。

### 23.7 V10 当前执行检查点

- [x] V9停止产物和代码路径完成只读复核；确认full证据丢失、containment假边、代理edge
  diagnostics和A3无绝对affinity门槛。
- [x] 用户确认V10计划和失败后的V10B审批边界。
- [x] 实现独立V10 objectbank、真实accepted-edge/unknown-edge/八级漏斗评价、P×R因果臂、
  VC1多视角共识、runner/replay、缺失场景lifting-only worker与四个CLI；DEV2的V9 lifting
  只读复用，后续场景只允许用既有3DGS/feature/SAM资产补lifting，不调用训练或下载。
- [x] 实现可恢复生产orchestrator、V9结论纠正产物、V10B失败升级文档和固定验收产物；
  补齐注册回归测试并通过全部`tests/category_priors`（`398 passed, 1 skipped`；skip仍为
  本地无云端CUDA/Torch条件）。
- [ ] commit/push、云端部署、只读复用两场lifting并启动Stage 0–1。
- [ ] runner健康后创建并回读核验绑定当前任务的每小时自动化；停止或完成后删除。

## 24. 当前权威：提示式最小机理重置（PMR-1）

### 24.1 为什么停止继续扩建自动 ObjectBank

V10 Stage 1 的实际停止值为：

- geometric IoU≥.50 匹配数：`0`（门槛 `6`）；
- geometric candidate precision@.25：`0.0005730659`（约 `0.0573%`，门槛 `10%`）；
- official-valid tiny/small Recall@.25：`0.0454545`（门槛 `20%`）；
- identifiable association precision：`0.9285317`（该项通过）；
- candidate 数：`1745`，而 P0R0 为 `107`。

V10 因此按预注册门槛停止，V10B 没有启动，类别先验没有运行。候选数量爆炸是真实结构
故障；但后续静态审计又确认 V8/V9/V10 使用了彼此不一致的诊断投影，V10 的极低精度
和零 IoU 还受到评价定义错误的严重影响，不能作为训练新身份头的充分依据。

V10 的 `GaussianGTIndex` 先让每个 GT 点只选择一个最近 Gaussian，再把所有未被任何 GT
点选中的预测 Gaussian 各自变成唯一 FP sentinel。当 Gaussian 数远多于 GT 点时，多个真实
贴在同一物体表面的 Gaussian 也会被结构性地判成 FP。该定义不是老师要求的
“预测 Gaussian→最近 GT 点精度”。历史诊断现在必须分成三个互不替代的空间：

1. **官方点空间 AP/IoU**：把预测投影到官方 GT 点上，保持现有官方 evaluator；
2. **Gaussian 精度/纯度**：每个预测 Gaussian 查询 2/5/10 cm 内最近 GT 点，未映射者才计 FP；
3. **GT 覆盖/召回**：每个 GT 点查询最近预测 Gaussian。

V10 raw bank 保持只读；只允许按上述三种定义零 GPU 重评。旧 V8/V9/V10 的 candidate
precision/IoU 不再跨版本直接比较。

更根本的研究纠正是：老师的问题是“已知物体大致位置和类别后，类别的一般大小、平滑程度
和小物体属性能否改善分割”，而 V5–V10 实际在同时解决全场自动提案、跨视角身份、对象补全、
分类、打分和去重。后者远大于原问题，并使类别先验多次在介入前就被上游门槛阻断。因此当前
禁止继续 V10B、V11 或新的自动对象跟踪主干，先做下述最小、可判别的机理实验。

### 24.2 唯一研究问题、处理和统计单位

研究问题固定为：

> 对同一个已知对象，使用完全相同的正提示和类别，在同一份 SAGA feature 上，只把共享全局
> 参数替换成 ScanNet-train 得到的逐类参数，分割质量是否稳定提高？

首轮条件只保留：

- `U-global`：把 train-only global 典型包围盒对角线映射为该场景的原生 scale-gate 输入；
- `D-class`：只把 global 典型对角线替换成该类别的 train-only shrunk 典型对角线；无类别
  统计时回退 global。

两组必须共享场景、对象、正提示、类别、feature、Gaussian、原始相似度定义、阈值候选、连通
结构和评价点。D 不得新增候选来源、身份跟踪、学习器、融合器或另一个后处理框架。

实验处理在对象内配对；对象嵌套于 physical scene。对象级差异用于机制描述，physical scene
才是独立统计单位：先在每个场景内平均对象差异，再跨场景汇总；不得把同一场景的多个对象
或重复扫描伪装成独立样本。

GT 的使用边界固定：只允许离线选择预先定义的对象、生成一个确定性的正提示、提供已知类别
并评价输出。GT 不得参与候选生长、参数拟合、阈值搜索、连通性或 U/D 所有权判断。

首轮只检验最贴近 SAGA 原生接口的 **size gate（类别典型尺寸）**。不得同时加入 smooth、
support、HDBSCAN、最小簇过滤或 rescue；否则又无法知道收益来自哪一层。平滑程度和小物最小
支持只有在 size gate 的最小机理结果形成后，才能作为后续单独假设提出。

### 24.3 提示式最小主干

主干直接复用公开 SAGA 的提示式分割机制，不建立 ObjectBank：

1. 对 official-valid SAGA20 GT 实例离线生成一个确定性像素正提示：先用修正后的最大贡献
   Gaussian 找到该实例可见 footprint 最大的已有训练视角，再取 footprint 中离边界最远的
   内部像素。运行 JSON 只保存 scene、camera、`(x,y)` 和 class；GT instance ID 单独放入
   evaluation-only 文件。U/D 共用逐字相同的运行提示。
2. 以公开 SAGA `prompt_segmenting.ipynb` 为权威核：

   ```text
   gate = scale_gate(s)
   query = normalize(rendered_affinity[y,x] * gate)
   points = normalize(gaussian_affinity * gate)
   similarity = points @ query
   mask = similarity > 0.75
   ```

   查询特征固定 `render_contrastive_feature(..., norm_point_features=True)`；不得混用 GUI 的
   隐式默认。输出就是 Boolean Gaussian mask，不再做连通组件、HDBSCAN 或 KNN。
3. 训练时 mask 的物理尺度经过场景分位数变换后才送入 gate。运行时必须复用同一语义：

   ```text
   d_global = exp(global.shrunk.geometry.log_bbox_diag_m.q50)
   d_class  = exp(class.shrunk.geometry.log_bbox_diag_m.q50)
   s_U = scene_mask_scale_CDF(d_global)
   s_D = scene_mask_scale_CDF(d_class)
   ```

   场景 CDF 只读现有 `saga/mask_scales/*.pt`；U/D 共享同一 feature PLY、scale gate 和提示。
4. D 唯一允许改变的是标量 `s` 及其产生的32维 gate 向量。类别不能参与 query、阈值或 mask
   的其他部分。小物体在首轮只通过更小的典型尺寸改变原生 scale gate，不加强制保留旁路。

正式实现前必须做机械干预审计：记录物理尺度、场景分位数输入、32维 gate、相似度分布、
预测 Gaussian 数和最终 mask。若 D 参数数值发生变化但候选与输出没有变化，只能判为“参数没有
有效介入”，不能判类别先验无效。

### 24.4 分阶段实验与停止规则

**Stage A：V10 历史口径闭环（零 GPU）**

- 只读复用两场 V10 bank；并列输出官方点空间 IoU、Gaussian→GT 2/5/10 cm 精度/纯度、
  GT→Gaussian 2/5/10 cm 召回。
- 保存旧定义与纠正定义的差值；撤销由旧 unique-FP-sentinel 直接推出的结论。
- 本阶段只纠正历史归因，不选择 PMR 参数。

**Stage B：两场景机械验证**

固定 `scene0645_00`、`scene0025_01`，只复用现有 3DGS、feature 和 GT，不训练、不下载。

必须同时满足：

- 同一对象的 U/D 正提示、类别、feature、gate 权重和相似度阈值完全一致；
- global/class 参数表均可解析，缺失类严格回退 global；
- 两场各选择至少2个大物体和2个小物体，总计≥8个目标；
- `scale=0.5` 时新 worker 与 Notebook 参考核逐点等价；worker 的分割调用链没有 GT 路径；
- RGB/feature PLY 的 Gaussian 数与 XYZ 顺序一致；查询像素自身相似度约为1且必被选中；
- 至少4个目标 `|s_D-s_U|≥0.05` 且 `max|gate_D-gate_U|>1e-6`，至少2个目标的
  U/D Gaussian mask 存在至少1个逐点差异；
- 输出非空、长度正确，重复运行逐点一致。

若提示无法形成有效候选，停止并归因当前 SAGA feature/提示接口不足；若参数变化不进入输出，
只修机械实现。不得在这两场景上按结果搜索阈值或逐类公式。

> **2026-08-27 机械门槛纠正（在查看其余6个场景前登记）：** 原门槛错误地要求单个提示对象
> 改变“全场1% Gaussian”。这把实现生效检查混成了效果大小检查，而且对本身远小于全场1%的
> cup、phone 等小物体形成不可能门槛。两场已运行结果显示尺度标量和逐点输出均实际改变，故将
> 机械门槛纠正为“至少两个对象存在至少一个确定性逐点差异，且每个登记场景至少有一个对象
> 发生变化”。这次修正不改变提示、参数、
> `0.75` 阈值、对象选择、8场景集合或 Stage C 的任何效果门槛；旧1%计数继续作为诊断报告，
> 不再用于早停。

**Stage C：8 个 physical scenes 的配对机理检验**

固定：

```text
scene0645_00  scene0025_01  scene0046_00  scene0474_01
scene0591_02  scene0329_02  scene0164_03  scene0064_01
```

每个 official-valid SAGA20 对象只运行一次确定性提示，并配对比较 U/D。主响应为对象 mask 的
官方 GT 点空间 IoU；诊断响应为 Gaussian precision、GT recall、组件数、是否被过滤和
tiny/small 分层结果。physical scene 内先平均，再报告8场景的均值、正向场景数和配对区间。

支持当前类别先验映射继续集成的预注册条件为：

- 跨场景平均 `ΔIoU ≥ 0.02`；
- 至少 5/8 个 physical scenes 方向为正；
- Gaussian precision 不下降超过 1 个百分点；
- tiny/small 平均 IoU 或召回至少一项提高，且另一项不下降超过 2 个百分点。

若 D 不通过：结论是“在对象身份和类别已知、提示完全相同的条件下，当前 train-derived
典型尺寸经 SAGA 原生 scale gate 没有稳定改善分割”，这才是对**当前类别尺寸机理**的直接
负证据；不得外推为平滑/支持等所有类别知识都无效，也不得再用
自动 ObjectBank 解释或继续扩建身份模型。

若 D 通过：老师的类别先验机理获得直接支持；下一阶段只把同一组逐类参数替换进最简单的
B1 路径，比较共享参数 B1 与逐类参数 B1，不再建立新主干。

### 24.5 实现、产物和当前检查点

实现必须是独立小模块，禁止向巨型 `postprocess.py` 继续加实验分支。最多保留一个纯算法模块、
一个场景运行/评价模块和一个轻量 CLI。必须测试：U/D 同参逐点恒等；提示逐点一致；GT 不进入
分割函数；类别参数只改变已登记参数；三种评价空间方向正确；对象嵌套场景的统计不发生伪重复；
重复运行确定；完整结果跳过、损坏结果只重跑当前对象。

验收产物固定为：

```text
v10_metric_closeout.json
prompt_prior_params.json
prompt_prior_mechanical2.parquet
prompt_prior_paired8.parquet
prompt_prior_analysis.json
viewer/
```

资源约束：不下载、不训练、不重训 3DGS；单 GPU、单进程；磁盘≥80GB；内存只看90GiB
cgroup；不生成 SHA 文件、lock、schedule hash 或 contributor cache。预计本地实现与测试
3–5小时，两场景1–2小时，8场景3–8小时。云端 runner 健康启动后才创建绑定当前任务
`01a0018b-d00d-7bb2-a64d-bcaa3cbd3bbe`的每小时自动化，并回读核对；停止或完成后删除。

- [x] V10 按预注册门槛停止；V10B 未启动，旧自动化已删除。
- [x] 发现并静态确认 V10 unique-FP-sentinel 评价定义错误。
- [x] 用户批准停止扩建自动系统，转为提示式最小机理实验。
- [x] 实现三空间 V10 只读重评。
- [x] 审计并冻结提示接口、正提示生成规则和 U/D 物化参数表。
- [x] 本地测试、commit/push、云端两场机械验证。
- [x] 运行8场配对机理检验并形成结论（commit `da22c5f`）。

## 25. 当前权威：原生尺度门控容量与映射归因（PMR-2）

### 25.1 PMR-1 已完成事实与结论边界

固定8个physical scenes、124个对象、相同正提示、原生2k feature、scale gate和相似度阈值
`0.75`的U/D配对已完成。D唯一把global train-only典型bbox对角线替换为class-shrunk典型
bbox对角线。机械干预确实生效：122/124个gate向量改变、119/124个Boolean Gaussian mask
改变；U/D并非同一输出。

以physical scene等权的主结果为：

- mean `D-class − U-global` IoU = `-0.0000474341`；
- 8场景paired bootstrap 95%区间=`[-0.00106548,+0.00063856]`；
- Gaussian precision差=`-0.0003428140`；
- tiny/small IoU差=`+0.0000973715`，recall差=`-0.0000728759`；
- 5/8场景为正，但实际幅度距离登记的`+0.02`门槛超过两个数量级。

因此已经可靠否定的是：

> 完整3D GT物体的逐类典型bbox对角线，经场景mask-scale分位数映射后直接控制当前SAGA
> scale gate，不能在当前原生2k feature和固定提示下形成稳定、实用的分割收益。

该结论不能外推为所有类别先验、平滑/支持先验或小物体保护均无效。PMR-1还留下一个必须
闭环的歧义：是“类别物理量选错了”，还是“当前2k feature的scale gate本身几乎没有可用
控制容量”。PMR-2只回答这个歧义，不再建立自动候选、聚类或对象跟踪主干。

### 25.2 已确认的尺度定义错位

`get_scale.py`训练gate时的原生尺度不是完整物体bbox。它对某一训练视角的某一2D mask：

1. 渲染历史定义的累计depth；
2. mask双线性放大后做3×3邻域和`>=5`的多数滤波；
3. 将保留像素反投影为三维点；
4. 计算样本标准差：

```text
native_scale = ||2 * std(x,y,z)||_2
```

5. 再由场景全部mask scales拟合的`QuantileTransformer`映射到`[0,1]`后输入gate。

它描述的是“特定视角、遮挡和SAM层级下的可见mask三维扩散”，而PMR-1的bbox对角线描述完整
物体。二者虽然都以米为单位，却不是同一个随机变量。另有公开上游已存在的历史定义瑕疵：
像素行列与`cx/cy`配对互换、depth未除总alpha、所有视角沿用首相机FoV、主点固定中心。
当前gate正是按这套历史定义训练，因此主正控必须按历史数值定义等价复现；几何修正版只能作旁路
静态诊断，不得直接喂给旧gate并冒充同接口实验。

### 25.3 诊断A：既有U/D变化方向审计

只读复用124个U/D masks。每个`U XOR D` Gaussian在5cm最近GT映射下固定分为：

- D新增且属于目标GT实例：有益；
- D删除且不属于目标实例（同类其他实例、异类、无GT支持）：有益；
- D删除且属于目标实例：有害；
- D新增且不属于目标实例：有害。

每对象计算：

```text
help_ratio = helpful / (helpful + harmful)
direction = (helpful - harmful) / (helpful + harmful)
```

无变化对象的`direction=0`、`help_ratio=null`。主统计先在场景内让对象等权，再让8个physical
scenes等权；点数加权只作诊断。2cm/10cm是敏感性分析，5cm是唯一主口径。

冻结解释：场景等权`help_ratio>=0.60`且至少5/8场景`direction>0`为方向一致有益；
`help_ratio<=0.40`且至少5/8场景`direction<0`为方向一致有害；其余为混合/近似随机边界扰动。
诊断A不得作为是否运行容量正控的早停门槛。

### 25.4 诊断B：原生scale-gate容量正控

继续使用完全相同的124个对象、提示、原生2k feature、gate和阈值。固定运行归一化gate输入：

```text
0, .25, .50, .75, 1.0
```

若五点网格未显示容量，预先登记且只允许补`.125/.375/.625/.875`，形成九点网格；不再依据
结果增加其他点。`GridOracle`用GT在`{s_U, 固定网格}`内为每对象选最高IoU；精确并列优先
离`s_U`最近，再选更小scale。它只表示gate能力上界，不是可部署方法。

广泛且实用的容量必须同时满足：

- 8场景等权`GridOracle−U`平均`>=0.02`；
- 至少5/8场景的平均增益`>=0.01`；
- 至少25%对象自身增益`>=0.02`。

同时计算`O-instance`原生尺度正控。它固定使用对象已登记的提示相机和该视角目标GT footprint，
按历史数值定义等价复现`get_scale.py`的depth、像素坐标、多数滤波和样本标准差，再经同场景
QuantileTransformer得到gate输入。footprint不足或scale非有限时标记ineligible，不得换视角。
`O-instance`是GT-derived oracle，不可部署；成功门槛沿用PMR-1 Stage C：scene-equal
`DeltaIoU>=0.02`、至少5/8为正、precision下降不超过1个百分点，且tiny/small IoU或recall
至少一项提高、另一项下降不超过2个百分点。为避免只剩少量“容易计算”的对象造成选择偏差，
还必须有总体至少80%的对象可计算，且每个场景至少50%的对象可计算；否则只能判为
`O-instance`覆盖不足，不能判物理尺度映射有效或无效。

### 25.5 冻结决策树与下一步边界

1. 九点`GridOracle`无广泛容量：停止“类别尺寸→当前2k scale gate”路线。结论是该接口无
   实用尺度控制能力，不是否定所有类别知识。
2. Grid有容量、`O-instance`不通过：gate可改变结果，但有益位置与原生物理尺度无稳定关系；
   类别平均尺寸不适合控制当前gate，停止继续校准bbox公式。
3. Grid和`O-instance`都通过：确认PMR-1失败根因是尺度统计定义错位。下一步只允许从
   ScanNet-train按相同原生visible-mask公式统计global/class-shrunk中位数，并在未用于构造
   映射的holdout scenes做一次`U-global-native`与`D-class-native`配对。
4. 只有第3步的holdout比较稳定通过，才支持老师的“简单类别尺寸先验”有效；若它仍失败，
   才有力说明实例/视角差异压过类别均值，逐类平均原生尺度没有额外预测价值。

现有8场已被反复查看，只作机理开发。不得用容量oracle选择新阈值、逐类公式或对象。GT不得
进入分割核心；`prepare-capacity`只物化明确标注的oracle计划，`segment-capacity`不接收GT路径。

### 25.6 工程、产物与当前检查点

新增独立小模块和runner，不修改PMR-1 U/D产物、不修改`postprocess.py`、不启用ObjectBank。
运行前逐项验证旧124结果可解析、mask为二值、metadata gate与重算gate一致。容量按scene加载
feature/gate一次、camera query一次；固定scale按scene共享point feature gate，避免重复归一化。

验收产物：

```text
prompt_prior_direction_audit.parquet
prompt_prior_direction_audit.json
prompt_prior_scale_capacity_plan.json
prompt_prior_scale_capacity.parquet
prompt_prior_scale_capacity.json
```

- [x] PMR-1八场124对象配对实验完成，当前bbox-diagonal映射未通过。
- [x] 静态确认训练mask-scale与完整3D bbox的定义错位。
- [x] 冻结变化方向、固定网格、O-instance和决策门槛。
- [x] 实现独立诊断模块、测试、commit/push（commit `1f23085`）。
- [x] 云端复用既有资产运行方向审计与容量正控。
- [x] 按冻结决策树停止；不进入train-only native class-scale holdout检验。

### 25.7 PMR-2 实际结果与最终归因

PMR-2 在固定8个physical scenes、124个对象上完成。没有下载、训练、聚类、ObjectBank或自动
实例候选；所有比较只改变原生scale gate接收的一个归一化标量。

既有D-class变化方向审计的5cm主结果为：

- 场景等权`help_ratio=0.485216`；
- 正向场景`2/8`、负向场景`5/8`；
- 冻结结论为`mixed-or-boundary-perturbation`。

这说明PMR-1的类别bbox尺度确实改变了Gaussian mask，但新增和删除的点没有稳定地朝正确方向
移动，表现为近似混合的边界扰动。

固定九点`GridOracle`已经给每个对象使用GT，从`{原U, 0,.125,.25,.375,.5,.625,.75,
.875,1}`中选择最高IoU，因而是同一2k feature/gate/阈值接口的强能力上界。结果为：

- 8场景等权平均`GridOracle-U DeltaIoU=+0.00367325`，远低于`+0.02`门槛；
- `0/8`场景达到平均`+0.01`；
- 仅`4/124=3.23%`对象达到自身`+0.02`，远低于25%门槛。

对象级`O-instance`正控的124个对象全部可计算，排除了可计算覆盖选择偏差。它使用GT对象在
冻结提示视角的真实footprint，并按历史原生`get_scale.py`数值定义计算gate输入，结果为：

- 场景等权`DeltaIoU=-0.00015527`；
- 正向场景`4/8`；
- Gaussian precision差`-0.00024960`；
- tiny/small `DeltaIoU=+0.00049286`、`DeltaRecall=+0.00034221`，仍为近零量级。

因此按照25.5的第1条停止：

> 当前原生2k feature的scale gate虽然能改变少量边界，但没有足以支持显著类别尺寸收益的广泛
> 控制容量。即使逐对象使用GT从九个尺度中择优，平均收益也只有0.00367；继续拟合逐类物理
> 尺度无法越过这个接口上限，不再运行train-only逐类原生尺度holdout。

该结论有力否定的是“类别尺寸通过当前冻结2k scale-gate带来显著提升”这条具体机制，不是否定
类别语义、类别形状、类别支持度或在重新训练的尺度条件表示中使用类别知识。若继续研究，必须
更换能力不足的表示/门控接口并单独获得授权，不能继续微调bbox公式或门控标量。

## 26. 当前权威：10k 尺度门控容量正控（PMR-3）

### 26.1 唯一问题与授权边界

PMR-2 已证明冻结的原生约2k feature/scale-gate缺少广泛、实用的尺度控制容量，但没有回答
公开SAGA与老师交付命令所用的10k训练预算是否能学出这种容量。用户于2026-08-28明确授权
启动10k正控。本阶段只回答：

> 在训练数据、30k 3DGS、提示、损失、随机种子和代码均相同的同一训练轨迹中，把训练从
> 原生自适应预算延长到10,000轮，是否使scale gate获得足以支持类别尺度研究的控制容量？

本阶段不测试类别先验本身，不改变类别映射，不引入ObjectBank、聚类、跟踪、新mask来源或
额外网络，也不下载数据或重训3DGS。通过只能证明“2k训练不足且10k恢复了尺度容量”；失败
只能否定“按当前原生训练方法增加到10k即可恢复容量”，不得外推为所有类别知识无效。

### 26.2 冻结场景、训练和快照

只使用两个已经参与机理开发的physical scenes：

```text
scene0591_02  # PMR-2中容量相对较高，登记对象15个
scene0645_00  # PMR-2中容量相对较低，登记对象19个，类别更丰富
```

两场均固定复用原始图像、相机、原生SAM masks、原生mask scales、30k 3DGS和PMR-1登记的
提示/对象；不切换到SAM-everything或外部语义监督。旧流水线外层虽登记seed 42，但没有把
该参数传入特征训练器；原生训练器实际使用默认seed 0。因此正控固定seed 0，显式
`--iterations 10000 --num_sampled_rays 1000`，其余优化参数保持当前老师交付兼容路径。

原生所谓“2k”并非固定2000轮，而是训练器在未显式指定预算时使用：

```text
native_iteration = min(10 * number_of_train_cameras, 10000)
```

为排除“两次独立训练的随机差异”，每场只从头训练一条10k轨迹，并在同一轨迹保存：

```text
iteration native_iteration: affinity/semantic feature PLY + scale_gate.pt
iteration 10000: affinity/semantic feature PLY + scale_gate.pt
```

快照写入独立`pmr3-scale-capacity-10k`目录，不覆盖历史2k feature、gate或PMR-1/2产物。
保存原生预算快照不得重置优化器、随机数、相机栈或训练状态；训练必须连续到10k。历史原生
结果只作上下文，不作为主配对。若同轨迹原生快照与历史结果的scene-equal容量差超过0.02，
必须标注“当前轨迹不能复现历史原生预算”，但不据此改变10k结果或门槛。

### 26.3 冻结评价与统计单位

对原生预算和10k快照分别复用PMR-2完全相同的34个对象、正提示、相似度阈值0.75、U输入
和九点网格：

```text
0, .125, .25, .375, .50, .625, .75, .875, 1.0
```

每个快照都必须用自己的feature/gate重新计算`U-global`和九点mask，随后计算
`GridOracle-U`、对象IoU、Gaussian precision/recall和tiny/small指标。不得拿历史U充当新
检查点的U。GT只允许进入离线评价和GridOracle选优；训练、特征提取、提示分割均不得读取GT，
也不得把混有GT-derived `O-instance`字段的旧PMR-2 plan交给分割worker。

物理场景是独立实验单位：先让同一场景内登记对象等权，再让两个场景等权。对象是场景内的
嵌套观测，不能把34个对象伪装成34个独立样本。本阶段是容量正控和因果诊断，不以两场景
置信区间声称泛化。

### 26.4 预注册通过与停止门槛

10k被判定为恢复了广泛且实用的尺度容量，必须同时满足：

1. 10k的两场scene-equal `GridOracle-U >= 0.02`；
2. 两个场景各自的`GridOracle-U >= 0.01`；
3. 至少25%的登记对象自身`GridOracle-U >= 0.02`；
4. 同轨迹scene-equal `(10k容量 - 原生预算容量) >= 0.01`；
5. 10k GridOracle相对10k U的Gaussian precision下降不超过1个百分点。

冻结决策：

- 全部通过：确认当前原生自适应预算训练不足是尺度容量瓶颈。下一步只允许用同一原生visible-mask定义
  生成train-only global/class-shrunk尺度，再在未参与映射构造的holdout场景比较U/D；本阶段
  本身不能声称类别先验有效。
- 任一不通过：停止“仅把当前原生训练延长到10k即可修复类别尺寸→scale-gate”路线；不得
  继续试20k、改网格或按结果挑场景。结论应是当前损失/表示没有学出足以承载显著类别尺度
  收益的可控接口，而不是类别先验整体无效。

### 26.5 工程、环境和验收产物

训练脚本只允许增加不改变训练动力学的中途快照能力。新增轻量runner负责：检查输入、启动
单场10k训练、验证2k/10k双快照、复用PMR-2计划生成固定提示结果并聚合门槛。完整结果跳过，
损坏或缺失项只重跑当前场景；不生成SHA文件、lock、schedule hash或contributor cache。

新5090云实例必须在运行前完成环境审计：PyTorch/CUDA必须实际支持`sm_120`，自定义CUDA
扩展必须用CUDA 12.8重编译并通过最小前向/反向测试；不得把旧cu118环境显示的
`cuda_available=True`误当作可运行。资源继续按实际cgroup读取，当前实例`memory.max=92GiB`；
不得使用宿主机`free`。磁盘全过程至少保留80GB，单GPU、单训练或单评估。

验收产物：

```text
pmr3_training_manifest.json
pmr3/<scene>/iteration_native_<N>/contrastive_feature_point_cloud.ply
pmr3/<scene>/iteration_native_<N>/scale_gate.pt
pmr3/<scene>/iteration_10000/contrastive_feature_point_cloud.ply
pmr3/<scene>/iteration_10000/scale_gate.pt
pmr3_scale_capacity.parquet
pmr3_scale_capacity.json
pmr3_analysis.json
```

- [x] 用户明确授权10k尺度容量正控。
- [x] 冻结两场景、同轨迹原生预算/10k快照、九点评价与五条门槛。
- [x] 实现快照、runner、聚合器和测试（commit `db872b7`）。
- [x] 在5090/CUDA12.8环境重编译扩展并通过GPU冒烟。
- [x] 完成两场同轨迹10k训练与固定评价。
- [x] 按冻结决策树停止“延长训练即可恢复尺度容量”路线。

### 26.6 PMR-3 实际结果

两场景34个固定对象的同轨迹原生预算与10k快照均已完成。10k快照的场景等权
`GridOracle-U`只有`+0.00071346`，原生预算快照为`+0.00241653`，延长训练后的容量反而
减少`0.00170307`。34个对象中没有一个达到`+0.02`收益；五条预注册门槛只有precision
保护项通过。因此结论为：按当前损失和表示把训练延长到10k，不能恢复实用的尺度控制容量。
尺度门控路线到此停止，不再试20k、改网格或继续拟合物理尺度映射。

## 27. 当前权威：固定候选的小簇删除最终验证

### 27.1 唯一问题与为什么还能离线回答

老师原始小类分支允许5个Gaussian形成实例，但后面的全场`filter_num(10)`又会把少于10点的
实例整簇删除。已有8场T1-B1阶段快照保存了全场256-NN之后、`filter_num`之前的完整partition，
所以可以冻结feature、HDBSCAN、候选ID、KNN结果和类别，只重放最后一次整簇接受/删除。

8场共有258个原分支候选、184,051个分支点；HDBSCAN噪声率为4.23%，但经过全局KNN和
过滤后仅剩35/258个分支实例与38,108/184,051个分支点。最后一次验证只问：

> 被统一最小簇门槛删除的候选中，是否有足够真实小物体；由训练集类别大小决定的门槛，
> 是否优于一个对所有类别统一降低的门槛？

不再修改HDBSCAN `min_cluster_size/min_samples`、语义阈值、采样、scale gate、距离、全点
分配、KNN、类别投票或score公式。这些因素会重造候选池，不能混入本次因果比较。

### 27.2 冻结输入、条件和唯一数据公式

冻结输入为T1-B1、原生约2k feature、fixed contributor、seed42的阶段快照：

```text
/root/autodl-tmp/saga/runs/v9-objectbank/t1-legacy/T1-B1/
  <scene>/seed-42/stage_trace.npz
```

主数组为`post_global_knn`，历史统一阈值10的参照为`post_filter`；分支实例类别来自
`stage_trace.json/branch_instance_classes`。非分支实例在所有条件中始终使用门槛10。

条件固定为：

- `U10`：老师原链的统一门槛10；
- `S3/S5`：所有分支类别统一使用门槛3或5，只作“普遍放宽”对照；
- `D-class`：只对分支实例使用train-only逐类门槛。

逐类门槛只读`category_priors.json`的收缩后典型表面积：

```text
A_c = exp(shrunk.geometry.log_surface_area_m2.q50)
m_c = clip(round(10 * sqrt(A_c / A_global)), 3, 10)
```

缺失类别和非SAGA20类别回退10。指数、上下界和锚点不搜索；逐类门槛永远不比历史门槛10
更严格。`U*`只从`S3/S5/U10`中在两个开发场景按官方mAP选择一次，精确并列选更高门槛，
随后冻结。

### 27.3 重放合同和评价

`U10`从`post_global_knn`重放后必须逐点等于已有`post_filter`，否则只修replayer，不解释结果。
低门槛新增候选只能写入原B1输出中的背景点，不能覆盖任何已有前景；候选类别沿用冻结分支类别。
新增候选的score固定为assignment confidence、HDBSCAN membership和semantic margin的几何平均，
所有条件完全相同，不使用GT打分。原B1已有实例及score原样保留。

同时报告两套互补结果：

1. 不受score影响的候选级same-class IoU、precision和tiny/small Recall；
2. 保守背景补入后的官方ScanNet mAP、AP50、AP25和Gaussian precision。

GT只进入离线评价。未映射Gaussian计FP；官方`min_region_size=100`固定不变。physical scene
是独立单位，候选和对象不是独立样本。

### 27.4 两场机械与能力门槛

开发场景固定为`scene0591_02`和`scene0645_00`。必须依次满足：

1. `U10`逐点复现`post_filter`；
2. `S3`相对`U10`两场各改变至少一个分支候选，累计至少3个；
3. 被U10删除而低门槛可保留的候选中，累计至少2个达到same-class IoU≥0.25，覆盖至少2类；
4. 仅作能力上界的`O-class-threshold`从`{3,5,10}`为每类选择GT最佳门槛后，相对U10
   满足官方mAP至少`+0.002`，或official-valid tiny/small Recall@0.25至少`+0.05`且
   mAP不低于`-0.001`。

GT最佳门槛绝不部署。任一门槛失败即停止，结论为固定候选上的最后整簇删除没有足够可恢复
真阳性，类别相关门槛没有可利用的作用空间。

若能力门槛通过，`D-class`还必须相对U10和U*都改变至少一个接受候选；实例数不超过U10的
1.5倍、Gaussian precision下降不超过2个百分点、mAP下降不超过0.002，才进入确认场景。

### 27.5 六个确认场景与最终判断

确认场景固定为现有阶段快照中未用于选择U*的六个不同physical scenes：

```text
scene0025_01  scene0046_00  scene0474_01
scene0329_02  scene0164_03  scene0064_01
```

这六场属于内部机理确认，不包装为全新外部盲测。每场只CPU重放`U10/U*/D-class`，不生成
伪seed。对场景级差异做10,000次配对bootstrap。

最终判断分开回答两件事：

- 统一小簇保护有效：`U*−U10`的mAP至少`+0.002`且95%区间下界大于0；或tiny/small
  Recall@0.25至少`+0.05`、区间下界大于0且mAP不低于`-0.0005`；至少4/6场景非负。
- 类别先验有额外价值：`D-class−U*`的mAP至少`+0.002`、95%区间下界大于0、至少4/6
  场景为正、tiny/small Recall不下降、Gaussian precision下降不超过1个百分点、实例数不超过
  U*的1.25倍。

若U*通过而D不通过，只能说明统一降低门槛有用，不需要类别先验。D只优于U10但不优于U*，
也不能声称类别先验有效。若能力上界通过而U*/D都失败，说明候选大小无法区分真阳性和假阳性。
若机械、能力和确认均不通过，则当前仓库中两条最直接的类别机制，即原生尺度门控和类别相关
小簇保护，都没有稳定实用收益；该结论不外推为所有可能的类别知识在理论上无效。

### 27.6 产物、资源和当前检查点

验收产物：

```text
noise_threshold_plan.json
noise_threshold_candidates.parquet
noise_threshold_dev2.json
noise_threshold_confirm6.parquet
noise_threshold_analysis.json
viewer/
```

不下载、不训练、不重聚类、不改巨型`postprocess.py`；只新增纯CPU replayer、评价器和测试。
完整trace只读复用；不生成SHA、lock、schedule hash或新cache。预计实现与测试1至2小时，两场
重放与评价约30分钟；通过后六场约1小时。

- [x] 完整回读权威计划、老师原始聊天和README。
- [x] 静态确认a800没有SOR；真实删除链为HDBSCAN、全局KNN、`filter_num(10)`和最终vote。
- [x] 核验8场T1-B1 pre-filter阶段快照、GT和train-only prior均完整。
- [x] 冻结本节问题、条件、公式、场景、门槛和结论边界。
- [x] 实现CPU replayer、评价器和回归测试；本地与云端相关测试均为22项通过。
- [x] 运行两场机械/能力检验，并按门槛停止或进入六场确认。
- [x] 形成面向老师的失败结论与因果证据文档。

### 27.7 实际结果与停止结论（2026-08-28）

`U10`从`post_global_knn`重放后，在8/8场景逐点等于原`post_filter`，机械复现通过。
开发场景`scene0591_02`和`scene0645_00`中均没有任何3至9点的分支候选，因此`S3`
相对`U10`改变候选数分别为0和0，累计为0，未通过第27.4节的第二道能力门槛。
按预注册规则未运行六场方法确认，也未用确认场景结果修改门槛。

为确认该现象不是开发场景偶然，对8场冻结快照只做了不参与选择的描述性候选审计：258个
分支候选中仅3个在全局KNN后落入3至9点区间，分布于2个场景，分别为3点socket、6点phone
和3点socket。三者的same-class最佳IoU均为0；训练集逐类门槛只会额外保留其中6点phone，
但该候选没有映射到任何GT评价点。因而最后`filter_num(10)`不是当前有用小物体被删除的主要
位置，逐类最小簇门槛在冻结候选上没有可利用的真阳性作用空间。

最终决定为`stop-no-recoverable-final-filter-intervention`。该结果与第24至26节共同表明：
在当前SAGA自动实例分割主干上，原生尺度门控和类别相关最终小簇保护均未产生稳定实用收益。
这否定的是当前主干和当前train-derived映射，不外推为任何类别知识在其他健康实例主干上都无效。

## 28. 当前权威：全类别类别化降噪的结构修正

### 28.1 结论边界更正

截至第27节，已经失败的是若干具体接口：把尺寸映射到原生scale gate、在全局256-NN和
统一过滤之后恢复小簇，以及若干重造候选的复杂ObjectBank路线。它们不能回答老师最初的
简单问题，因为老师分支产生的候选在进入最终输出以前仍会参加全场256-NN和统一小簇过滤，
很多候选在类别参数真正起保护作用以前就已被背景或大实例吞掉。

因此，当前结论更正为：**类别先验尚未完成一次忠实的全类别类别化降噪检验。** 本节只修正
降噪发生的位置，不再扩展尺度门控、跨视角跟踪、候选融合或训练。

### 28.2 唯一主比较

每个场景只生成一次冻结候选池。完整32类归一化top-1先竞争，阈值固定0.7；只有赢家属于
SAGA20的Gaussian进入分支，每点最多属于一个类别。候选仍使用老师原分支的距离权重
`0.5/0.3/0.2`、每类最多5000点、HDBSCAN `epsilon=0.01`和全点分配阈值0.3；为保留
小簇，`min_cluster_size=min_samples=3`。seed42按类别产生稳定且互相独立的排列。

冻结bank后只比较：

- `U-all-uniform`：所有SAGA20类别进入同一保护结构，尺寸、边界和平滑支持量读取global统计；
- `D-all-class`：候选、类别、采样、置信度和输出结构完全不变，只改为train-only逐类shrunk统计。

主效应是`D-U`。`U-B0`只用于确认保护结构没有破坏原主干；B1只作老师早期小物体试验的历史背景。

### 28.3 真正的类别化降噪

候选公共证据分数为二维branch类别票比例乘平均分配置信度。尺寸只惩罚相对典型类别异常过大，
平滑项只惩罚5cm边界率异常高。候选还必须类别票赢家一致、票比例至少0.3、core点数达到支持
门槛且总接受分数至少0.20。U统一使用支持门槛5；D使用：

```text
m_c = clip(round(5 * sqrt(A_c / A_global)), 3, 10)
```

`A_c`、三轴extent和5cm边界统计只来自冻结的ScanNet-train先验；缺失类回退global，禁止用
验证GT逐类调参。

关键结构修正发生在全局KNN之前：被U或D接纳的branch点从全局256-NN的查询点和投票邻居中
同时排除，也不参加统一10点过滤。未接纳点严格回退原global标签。全局降噪结束后再插回受保护
实例，此后不再KNN、过滤或改类。branch类别直接继承；最终二维vote只记录证据和共同AP score
`Q`，不再覆盖类别。这样才是在降噪过程中保护类别对象，而不是在对象已经被吞掉之后补救。

### 28.4 冻结阶段和停止规则

两场机械验证固定`scene0645_00/scene0025_01`：关闭新结构时必须逐点等于B0；U/D必须读取
同一bank；3至9点受保护实例必须100%穿过输出且类别改写为0；输出必须无orphan和负ID metadata；
D若只改数值而不改接受结果，只能判为干预未生效。

开发8场固定为：

```text
scene0645_00 scene0025_01 scene0046_00 scene0474_01
scene0591_02 scene0329_02 scene0164_03 scene0064_01
```

必须先有至少12个same-class IoU≥0.50候选、覆盖4场；U相对B0的mAP/AP50降幅分别不超过
0.001/0.002，实例数不超过1.25倍且覆盖下降不超过1个百分点；D至少使10%候选分数变化0.01，
或改变5个接受候选并覆盖2类2场。D进入独立复核还需mAP至少提高0.002，或tiny/small
Recall@0.50提高0.01且mAP不低于-0.0005，并满足场景方向和FP/TP门槛。

独立复核先使用`scene0231_00/scene0608_00/scene0356_00/scene0011_00/scene0593_00`，通过后才
补齐tune24；同一物理环境先平均扫描。只有13个物理环境宏平均`D-U≥0.002`才进入原48个不同
物理场景。最终使用10,000次场景级配对bootstrap；`D-U≥0.002`且95%区间下界大于0才支持
全类别数据先验稳定有效。任何阶段失败均按候选空间、结构安全、干预生效或数据映射分别归因，
不得笼统写成“类别先验无效”。

最终辅助退化界限在查看结果前固定为：AP50下降不超过0.002，Gaussian micro precision下降
不超过1个百分点，FP/TP比不恶化超过20%。它们只防止主mAP改善由明显的精度崩塌换来，
不参与方法选择或逐类调参。

### 28.5 工程边界

新实现使用独立的`category_denoise`模块和三个入口：`run-category-denoise-bank`、
`replay-category-denoise`、`evaluate-category-denoise`。旧实验分支不接入新runner。复用现有2k
feature、30k 3DGS、mask、label、GT和train-only先验；不下载、不训练、不运行ObjectBank，
不生成SHA、lock或contributor cache。physical scene是统计独立单位，bank是U/D的配对区组。

## 29. 当前权威：候选形成、KNN吞噬与先验域三分诊断

### 29.1 为什么第28节不能直接写成“类别先验无效”

第28节在冻结的8个开发场景上共得到1344个候选，但same-class IoU达到0.25和0.50的候选
分别只有15个和4个；candidate precision@0.25为1.116%。79个tiny/small GT中，Recall@0.25
为10.13%，Recall@0.50为3.80%。这是候选池质量不足的直接证据。

U和D都没有接纳任何候选，最终输出逐点相同。这个现象不只是“参数改得不够多”：808个候选
的U/D分数发生了非零变化，1258个候选的支持门槛不同，但最大U分数约为3.51e-6，最大D分数
约为0.002253，远低于固定接纳阈值0.20。U的平滑因子在全部候选上精确落到
`exp(-12.5)`下限，D也大面积落底。更重要的是，D的尺寸项单独使用就会淘汰全部15个
IoU达到0.25的正候选；原始证据分数Q与IoU仍有弱正相关，加入当前尺寸和平滑项后相关方向
转为负值。因此，当前尺寸/平滑评分映射已有明确的负面机械证据，但它仍不能回答两件事：

1. 正确候选究竟在语义筛选、HDBSCAN core还是full assignment阶段丢失；
2. 已经正确形成的少数候选是否又被全场256-NN或统一10点过滤吞掉。

第29节只拆开这三个问题，不再修改0.7语义阈值、HDBSCAN参数、0.3全点分配阈值或0.20
接纳阈值，也不重新生成1344个候选。

### 29.2 E1：候选形成漏斗

E1读取冻结bank和GT，在同一5cm评价映射下分别检查：完整32类top-1语义可达集合、保留下来的
HDBSCAN raw core和full assignment候选。它同时报告same-class与class-agnostic最佳IoU、
Gaussian纯度、GT覆盖、unsupported比例、core是否真为full子集，以及Q、尺寸、平滑各层的
反事实筛选结果。

E1只回答“候选在哪一层失去完整性或纯度”。S0是全场同类语义可达集合，不是实例候选；现有
bank也无法恢复因full少于3点而已经丢弃的原始HDBSCAN core。因此漏斗标签只用于分层计数，
不能写成严格因果证明。输出为：

```text
candidate_funnel_candidates.parquet
candidate_funnel_gt.parquet
candidate_funnel_analysis.json
```

### 29.3 E2：正确候选的KNN生存实验

E2先在评价阶段用GT选出same-class IoU至少0.50的冻结full候选，并把候选ID、类别和目标实例
写入只读计划；不足0.50时不得降到0.25。随后进行的KNN重放不接受GT路径，只从冻结bank的
`global_pre_knn`开始比较：

- `O1-unprotected`：插入正确候选后照常执行历史256-NN和10点过滤；
- `O2-protected`：候选从KNN查询点和投票邻居中同时排除，结束后原样插回。

完整场景预测只有在raw global partition与现有B0声明实例能逐点精确映射时才生成；若出现
一对多、多对一或部分重叠，仍保存候选生存统计，但不近似匹配、不伪造AP。E2回答的是
“全局降噪会不会吞掉已经正确形成的候选”。O2是机制上界，不是可部署方法，也不能用少量
oracle候选作显著性声明。输出为：

```text
knn_oracle_plan.json
knn_oracle_candidates.parquet
knn_oracle_metrics.parquet
knn_oracle_analysis.json
```

### 29.4 E3：健康对象上的先验域能力

E3用GT在Gaussian空间构造完整对象，并为每个对象确定性生成PCA半对象碎片和最近对象合并
两个负例。所有对象的公共证据Q固定为1，只比较global与class-shrunk尺寸、平滑和支持统计。
主半径固定5cm，2cm和10cm只作敏感性报告，不参与参数选择。

这个实验回答“在对象完整、类别正确时，当前train-derived统计能否把完整对象排在碎片和误合并
之前”。它不生成正式预测。由于尺寸公式只惩罚过大的对象，完整对象与半对象碎片在size-only
条件下并列是公式能力边界，不是实现错误。输出为：

```text
prior_oracle_objects.parquet
prior_oracle_pairs.parquet
prior_oracle_analysis.json
```

### 29.5 GT边界和解释规则

- E1与E3全程属于离线评价，GT不得进入原bank、正式预测或类别参数拟合。
- E2只有`prepare`阶段使用GT选定候选；后续KNN/filter重放的公共接口不接受GT、IoU或半径。
- 三项实验都复用同一冻结DEV8 bank，不重跑HDBSCAN，不训练、不下载，也不改变B0/U/D产物。
- 所有统计以scene为单位汇总；同场景多个对象不能伪装成独立实验样本。
- 仓库的`map_50_95`明确标为0.50至0.95十阈值历史口径，不直接称为外部可比的ScanNet官方mAP。

结果解释固定如下：

- E1在core以前已无正确候选：主要限制位于语义或候选形成，不能用后置先验补救；
- E1已有完整候选、E2的O1明显损坏而O2保留：全局KNN/过滤确实吞噬正确候选；
- E3完整对象仍大面积落底：当前统计域与Gaussian对象表示不匹配；
- E3的D稳定优于U：类别统计仍有潜力，自动候选形成是主要瓶颈；
- E3的D不优于U：只否定当前尺寸、平滑和支持映射，不外推为所有类别知识无效。

### 29.6 当前执行检查点

- [x] 冻结DEV8事实和三项诊断的数据合同；
- [x] 实现E1、E2、E3纯CPU算法及五个独立CLI；
- [x] 本地核心回归测试和CLI静态检查通过；
- [x] 提交并部署独立云端工作区（commit `ffe37bf99893162ce6164158acb57dc7f27db7f9`）；
- [x] 在冻结8场运行E1、E2、E3并核对1344/15/4等输入身份；
- [x] 根据三项结果更新根因结论；未用诊断结果修改原bank、HDBSCAN或KNN。

### 29.7 E1/E2/E3实际结果（2026-08-28）

- E1核验冻结DEV8共有1344个候选、124个official-valid GT、79个tiny/small GT；same-class
  IoU达到0.25/0.50的候选为15/4，candidate precision@0.25为1.116%。完整语义可达集合
  覆盖28/124个GT，但采样HDBSCAN直接达到IoU0.25的只有2个，full assignment扩张后为15个。
- 515/1344个候选出现记录的HDBSCAN成员不属于最终full（38.3%）。现有bank只保存最终
  core/full，无法恢复原始采样顺序、原始簇、membership和阈值前分配，因此必须重新做shadow
  trace，不能继续从冻结bank推断严格因果。
- 原证据Q与真实IoU的Spearman为`+0.225`；旧combined先验与IoU为`-0.125`。旧class-size
  会淘汰全部15个IoU≥0.25候选，5cm smoothness在完整对象和候选上大面积饱和。
- E2只有4个oracle same-class IoU≥0.50候选。它们经过原256-NN/filter后全部存活，平均IoU
  从`0.5222`升至`0.6340`；保护式旁路保持`0.5222`。因此少量证据不支持绕过KNN，正式
  方法继续使用U/D共享的legacy KNN/filter。
- E3显示class-size对完整对象和错误合并对象有明显区分能力，但旧单边尺寸不能识别偏小碎片，
  smoothness对完整对象大量落底。结论是“旧映射失败但类别统计仍有能力信号”，不是一般类别
  先验已经被否定。

## 30. 当前权威：候选形成追踪、完整分配修复与同池类别先验复测

### 30.1 唯一研究顺序

本轮不增加ObjectBank、尺度门控、保护式KNN旁路、rescue或学习式评分器。执行顺序固定为：

1. shadow重跑HDBSCAN并保存原始采样簇到full assignment的完整转移；
2. 区分采样不足、原始聚类失败和完整分配污染；
3. 只比较legacy、consistent-envelope和raw-anchored-envelope三个候选形成条件；
4. 唯一修复在DEV2和DEV8均健康后，才测试双侧尺寸和类别支持量；
5. 候选级先验通过后，才接回U/D完全相同的legacy KNN/filter并扩展场景。

E2已经表明现有四个真候选没有被KNN吞掉，因此本轮禁止保护式旁路和事后插回。若先验差异
被公共KNN/filter完全抹掉，只能报告“先验未穿过legacy接口”，不能写成“类别先验无效”。

### 30.2 追踪合同与根因判定

新增独立trace，不修改或升级冻结bank v1。每类记录全部语义入选点、确定性sample rank、
HDBSCAN原始标签/noise/membership、阈值前argmax簇和置信度、原始簇到最终candidate ID映射、
三个距离分量及空间尺度。两场`scene0645_00/scene0025_01`的trace版必须先逐点复现冻结bank；
候选/full/core/Q或置信度误差超过`1e-6`时只修trace。

GT只在离线诊断中把可达对象分为：采样点少于3、充分采样但最佳raw簇F1<0.50、raw簇健康但
扩张IoU下降至少0.10、扩张precision下降至少20个百分点、以及后处理损失。两场可诊断对象
少于8时只扩展trace到DEV8。DEV8是追踪范围上限；若扩展后仍少于8个，不再因样本数单独停止，
而是用DEV8中全部可诊断对象按采样不足、raw聚类失败、完整分配污染的既定多数规则继续分流。

若失败对象中过半为采样不足，只比较嵌套5000/10000采样；10000须新增至少2个IoU≥0.25
raw簇或raw recall提高0.10，且候选数不超过1.5倍。若充分采样对象中过半仍raw F1<0.50，
计算局部affinity AUROC并做GT评价专用oracle-seed正控；AUROC<0.60且oracle Recall@0.25<0.20
时只允许两场10k feature正控，不得自动扩展训练。

### 30.3 唯一候选修复选择

DEV2比较：

- `C0-legacy`：当前完整分配；
- `C1-consistent-envelope`：可信核心只含被分回原始簇且置信度至少0.3的raw成员；
- `C2-raw-anchored-envelope`：所有HDBSCAN非噪声raw成员锚定在所属簇。

C1/C2共用老师原`0.5/0.3/0.2`混合距离、原空间尺度、raw成员medoid和成员到medoid距离q95
包络；非核心点还须置信度至少0.3且位于最近簇包络。多包络取最小距离，精确并列保持背景；
不搜索半径倍率。`trusted_core`必须是`full`子集，类别、GT和prior不得参与候选构造。

修复臂必须保持IoU0.25/0.50匹配数不低于C0，并使candidate precision相对提高至少25%或
unsupported下降10个百分点；tiny/small recall不降、候选数不超过1.25倍、core合同零违规。
按IoU0.50、IoU0.25、precision、unsupported、候选数排序，精确并列选C1。无臂通过即停止。

冻结修复扩展到DEV8后，必须达到：same-class IoU0.50至少12个/4场景、precision@0.25至少5%、
official-valid tiny/small Recall@0.25至少0.20、正向场景多于负向、候选数不超过C0的1.25倍。
未通过时停止类别先验，结论为候选空间仍不健康。

### 30.4 同一bank上的先验

5cm smoothness正式停用。尺寸使用sorted PCA log extent的双侧IQR平台：q25到q75内不惩罚，
低于q25和高于q75分别以到q50的半IQR归一化，`G=exp(-0.5*mean(min(z^2,25)))`。支持量只
统计trusted core：U固定5，D使用`clip(round(5*sqrt(A_c/A_global)),3,10)`。缺失类回退global。
候选分数为`S=QG`；U/D必须共享candidate ID、full/core、类别、采样和Q。

先在E3完整/碎片/合并对象上重做能力正控。完整对象落底比例须不超过10%；D相对U至少改善
一种负例判别0.02，另一种不退化超过0.02；完整对象的中位G必须同时高于碎片和错误合并
对象，且完整对象支持通过率下降不超过5个百分点。这里“落底”按预注册业务阈值`G<=0.001`
计算，不等同于公式裁剪下限`exp(-12.5)`。失败即停止当前统计映射。

DEV8先以无硬阈值candidate AP比较U/D。机械生效要求至少10%候选`|D-U|>=0.01`或支持集合
改变至少5个候选/2类/2场景；D须scene-equal AP@0.25提高0.002、AP@0.50下降不超过0.002、
至少5/8场景为正、tiny/small recall不降。通过后仅在DEV2的U上从
`.05/.10/.15/.20/.25`按candidate F1@0.25选择一次阈值，精确并列选更高值，D共享该阈值。

### 30.5 完整后处理和扩展

U/D都从同一`global_pre_knn`开始，先写入各自被接纳候选，拒绝点回退global，再运行完全相同
的legacy 256-NN和filter10；禁止保护、事后插回、重救援和二次改类。主官方比较输出共同Q，
S排序只作校准诊断。关闭全部候选必须逐点等于B0，并逐候选保存pre-KNN、post-KNN、
post-filter和final ID生存信息。

DEV8的U结构门槛为相对B0 mAP/AP50下降不超过0.001/0.002、实例数不超过1.25倍、覆盖下降
不超过1个百分点。D相对U须mAP提高0.002，或tiny/small Recall@0.50提高0.01且mAP不低于
-0.0005；至少5/8场景为正、FP/TP恶化不超过20%、Gaussian precision下降不超过1个百分点。

通过后先跑五个canonical holdout，要求mean差大于0、至少3/5为正、小物指标为正；再按13个
physical scenes等权汇总tune24，宏平均差至少0.002才进入final48。最终以48个physical scenes
做10000次配对bootstrap；mAP差至少0.002且95%区间下界大于0才支持稳定有效。final禁止调参。

### 30.6 当前执行检查点

- [x] 第29节E1/E2/E3结果闭环并冻结解释边界；
- [x] 用户批准第30节计划；
- [x] 实现trace数据合同、身份门槛和DEV2根因评价；
- [x] 实现C0/C1/C2纯候选修复与测试；
- [x] 实现双侧IQR size、trusted-core support和无smooth同池评价；
- [x] 实现条件触发的两场同源10k表示正控；无论通过或失败都按授权边界停止，禁止偷换回2k继续；
- [x] 本地625项类别先验回归测试通过；DEV8不足8个对象的分流语义及旧停止状态恢复已覆盖测试；
- [x] 云端已部署固定代码`0a2a418`并启动DEV2，工作区为`/root/autodl-tmp/saga/workspace/category-candidate-0a2a418`；
- [x] 已创建并回读核验“SAGA 候选形成修复每小时检查”，目标任务为`01a0018b-d00d-7bb2-a64d-bcaa3cbd3bbe`。

### 30.7 第30节实际停止结果（2026-08-29）

第30节已完成并按预注册门槛停止，不能继续解释为“类别先验失败”。可复核事实如下：

- DEV2的B0机械等价逐点通过，两场点数分别为`1,533,098`和`1,459,291`；
- DEV8 trace全部完成，但124个official-valid GT中仅7个满足既定可诊断条件；其中6个归为
  `raw_clustering_failed`，1个健康，主要故障已经位于完整分配之前；
- 充分采样对象的局部affinity边AUROC均值为`0.84515`，oracle-seed Recall@0.25为
  `0.57143`，没有触发“两场10k表示正控”的双重门槛；现有2k特征至少具备局部区分信号；
- `C1-consistent-envelope`虽使candidate precision相对提高`43.92%`，但IoU≥0.50匹配数
  下降，且某GT最佳IoU最大下降`0.06893`，超过`0.05`安全界；
- `C2-raw-anchored-envelope`使precision相对下降`60.46%`，候选数超过C0的1.25倍，亦未通过；
- 因此停止原因是“在错误原始聚类之后修补完整分配不能安全恢复实例”，不是“类别尺寸、
  支持量或一般类别知识已经无效”。第30节没有运行正式U/D类别先验比较。

静态溯源同时确认：最早可见的老师原型`bfc2192`已经包含
`1-outer(sampled_scores,sampled_scores)`、按场景样本最大值归一化空间距离、HDBSCAN后均值中心
和全点重新吸附。该组合不是最近实验新增，但它仍只是早期试验代码；本节证据只说明原始实例
形成存在结构风险，不能把责任扩大到老师的类别先验想法本身。

## 31. 当前权威：修正HDBSCAN输入距离并重建原始实例

### 31.1 研究问题与冻结边界

本节只回答一个更靠前的问题：在现有2k affinity、30k Gaussian和相同语义候选下，修正
HDBSCAN的距离定义和簇扩张语义，能否形成足够健康的原始实例候选。只有候选健康后，才恢复
第30.4至30.5节已经冻结的U/D类别先验比较。

禁止下载、训练、重训3DGS、启用ObjectBank、尺度门控或V3–V10复杂主干；GT只进入离线评价。
不搜索HDBSCAN参数，不依据DEV2/DEV8结果修改本节公式或门槛。

### 31.2 已确认的旧距离问题

旧语义距离为`1-outer(score,score)`。它只比较“各点有多自信”，没有比较两点的affinity
方向；同一物体的两个低置信点会被人为推远，不同物体的两个高置信点反而会被拉近，而且对角
通常不为0。旧instance和XYZ距离又分别除以当前样本中的最大距离，使同一物理物体在不同场景、
不同采样组成下得到不同尺度。HDBSCAN随后聚出的raw簇还会被均值中心和全点softmax重新定义；
在冻结样例中，4个采样点的raw簇可扩张为约13,027点，raw成员也可能被分给别的中心。

### 31.3 修正距离与固定聚类

对语义top-1阈值`0.7`后同一branch类别内的采样点，定义：

\[
D_{aff}=\arccos(\operatorname{clip}(\cos(f_i,f_j),-1,1))/\pi
\]

\[
D_{xyz}=\min(\lVert x_i-x_j\rVert/d_g,1)
\]

其中`d_g`只读取冻结ScanNet-train priors的global `log_bbox_diag_m.q50`，不得按场景样本最大值
重新归一化。最终距离固定为：

\[
D=0.625D_{aff}+0.375D_{xyz}
\]

输入矩阵必须有限、对称、非负且对角严格为0。语义置信度只负责top-1和`0.7`准入，不再作为
两点identity距离。HDBSCAN固定`metric=precomputed`、`min_cluster_size=3`、
`min_samples=3`、`cluster_selection_epsilon=0.01`；不调参。

### 31.4 冻结的四个候选条件

- `R0-legacy`：第30节冻结C0，作为旧主干负控；
- `R1-corrected-distance-legacy-expand`：只替换HDBSCAN输入距离，保留旧均值中心全点分配；
- `R2-corrected-distance-anchored-expand`：修正距离，并保持所有raw非噪声采样成员属于原簇；
- `G1-mutual-local-graph`：只有R1/R2均未通过DEV2时才允许运行的预注册图后备，不是临时调参。

R2中，未采样点在相同修正距离下查询最近raw成员；每个簇的扩张半径固定为raw成员的
leave-one-out最近邻距离q95。仅当最近距离不超过该簇半径时附着，多簇精确并列保持背景。
raw成员为可信core，`core`必须是`full`子集；扩张置信度固定为
`exp(-d/max(radius,1e-8))`。最终full少于3点的候选删除。实现必须分块，禁止构造全场
`N×K`稠密矩阵。

G1只在R1/R2失败时运行：先建物理24-NN，每点仅保留affinity cosine最高的4个物理邻居，
边须互选，连通分量至少3点才成候选；不读取类别统计或GT，不增加半径和阈值搜索。

### 31.5 Stage 0：实现与R0身份门槛

新增独立入口：

```text
audit-category-cluster-distance
run-category-cluster-bank
evaluate-category-cluster-bank
```

首先用冻结trace重算旧距离，R0的raw labels、full/core、candidate ID、Q必须与第30节产物一致；
浮点误差不超过`1e-6`。同时验证新距离的对称、有限、零对角、排列等变性和全局尺度来源。
R0不等价时只修审计/接线，不进入DEV2。

### 31.6 Stage 1：DEV2结构选择

固定`scene0645_00`、`scene0025_01`。R1/R2必须同时满足：

- same-class IoU≥0.25与≥0.50匹配数均不低于R0；
- candidate precision@0.25相对提高至少25%，或unsupported比例下降至少10个百分点；
- official-valid tiny/small Recall@0.25不下降；
- 候选数不超过R0的1.25倍；
- 至少一个场景改善，另一个场景任一GT最佳IoU下降不超过0.05；
- raw成员保留率100%、`core⊆full`、orphan和负ID metadata违规均为0；
- 重复运行逐点确定。

通过者按IoU≥0.50、IoU≥0.25、precision、unsupported、候选数和结构简单度排序。若R1/R2
均失败，按同一门槛运行一次G1。三者均失败即停止，结论为“当前2k表示加固定无监督聚类仍不足
以形成原始实例”；不得进入类别先验。

### 31.7 Stage 2：DEV8候选健康门槛

冻结唯一DEV2胜者后运行八个不同物理场景。必须同时满足：

- same-class IoU≥0.50候选不少于12个，覆盖至少4场景；
- candidate precision@0.25至少5%；
- official-valid tiny/small Recall@0.25至少0.20；
- 相对R0正向场景多于负向场景；
- 候选数不超过R0的1.25倍；
- raw成员保留、core/full、orphan、metadata和确定性合同全部通过。

未通过即停止，并按语义准入、采样、raw聚类、扩张四级漏斗报告。通过后才恢复第30.4节的
完美对象能力正控和同bank U/D比较；其门槛、后续legacy接回、holdout/tune/final流程均保持
第30.4至30.5节不变，不得因本节结果重新设计prior。

### 31.8 测试、产物与资源

必须测试：旧距离R0回归；修正距离零对角/对称/有限/排列等变；全局米制尺度而非样本最大值；
raw成员不可跨簇；q95半径、单点退化和并列背景；chunked与dense小样例等价；R1仅改变聚类
输入；R2/G1不读取GT或类别先验；重复运行确定；完整产物跳过、损坏项仅重跑当前场景；官方
evaluator parity继续通过。

DEV2逐点重复测量的bank固定保存在独立的`dev2_measured_runs`目录；DEV8只能引用其已验证的
算法合同，不能覆盖这两场直接测量证据。距离审计或DEV2分析损坏时必须从该不可变bank恢复。

验收产物：

```text
cluster_distance_audit.json
cluster_repair_dev2.parquet
cluster_repair_dev2_analysis.json
cluster_repair_dev8.parquet
cluster_repair_dev8_analysis.json
category_denoise_v3_dev8.parquet
category_denoise_v3_analysis.json
```

单GPU单进程；数据盘至少80GB；内存只读90GiB cgroup的`memory.current/max/events`；不生成
SHA文件、lock、schedule hash或contributor cache。

### 31.9 当前执行检查点

- [x] 第30节真实停止结果与结论边界写入权威文档；
- [x] 用户批准修正HDBSCAN结构，并预注册R1/R2及条件性G1；
- [x] 实现修正距离、R1、R2、G1及独立离线评价；
- [x] 完成新增回归和全量本地测试（707项通过、2项跳过）；
- [x] commit/push并部署commit绑定的新云端工作区（`ca0e069`；评价接口修复`ccb5677`）；
- [x] R0身份通过并完成DEV2；R1、R2和条件性G1均未通过冻结门槛；
- [ ] 仅DEV2通过后运行DEV8；仅DEV8健康后恢复类别先验复测。

### 31.10 第31节实际停止结果（2026-08-29）

两场景中，R0共有374个候选、7个same-class IoU≥0.25匹配和1个IoU≥0.50匹配。
R1修正距离但保留旧扩张后只剩3个IoU≥0.25匹配；R2锚定raw成员后产生5016个碎片候选，
仍只有3个IoU≥0.25匹配；条件性G1生成58473个局部连通分量，却没有任何IoU≥0.25匹配。
三种结构均未通过，故没有进入DEV8，也没有运行类别先验。

这一结果把结论进一步收窄：旧raw成员重分配确实是错误，但不是低候选质量的唯一主因；
修正距离、强制锚定或绕开HDBSCAN都不能把当前语义路由后的特征稳定组织成完整对象。下一步必须
把“语义路由错误”和“实例关联特征不足”分开，不能再继续修改扩张或降噪。

## 32. 当前权威：特征表示版本 × 语义路由的两场景2×2根因诊断

### 32.1 研究问题与边界

本节只回答：第31节raw实例形成失败，主要来自语义点被分错类别，还是现有实例关联特征不能把
同一物体聚在一起。固定`scene0645_00`和`scene0025_01`，运行：

```text
native-2k-grounded × predicted-32-top1
native-2k-grounded × gt-class-oracle
v9-10k-dual-source × predicted-32-top1
v9-10k-dual-source × gt-class-oracle
```

这里的10k资产同时改变了训练预算和affinity掩码来源，因此只能称“表示版本”，不能把差异
单独归因于训练轮数。GT正控只提供类别路由，不提供GT实例ID；GT实例只用于离线算IoU。

### 32.2 冻结机械条件

- 只评价采样后的raw HDBSCAN簇，不做全点扩张、KNN、filter或类别先验；
- 两个表示版本共享32类表、SAGA20范围、seed42、每类确定性采样前缀和5000点上限；
- predicted路由使用完整32类归一化top-1和0.7阈值；
- GT路由只把5cm内映射到official-valid GT的Gaussian送入对应类别；
- corrected metric、`min_cluster_size=3`、`min_samples=3`和epsilon 0.01保持第31节不变；
- 两个feature PLY必须与同一30k Gaussian的点数、XYZ顺序逐点一致，否则停止。

### 32.3 预注册解释门槛

相对对应对照，某因素只有在候选数不超过1.5倍且满足以下任一项时才算有实质作用：

- 新增至少2个same-class IoU≥0.25 raw匹配；
- official-valid tiny/small raw-cluster Recall@0.25提高至少0.10。

解释固定为：GT路由有增益而表示版本无增益，语义路由为主因；表示版本有增益而GT路由无增益，
当前表示版本为主因；两者均有增益，两个问题共同存在；四臂都没有增益，则当前affinity训练目标
或raw无监督聚类本身不适合自动实例身份形成。只有真实predicted路由臂累计至少6个IoU≥0.50
候选且tiny/small Recall@0.25至少0.20，才允许讨论恢复类别先验。GT臂永远不能进入正式方法。

本节不评价类别先验；无论结果如何，都不得把它写成类别先验有效或无效。

### 32.4 实际结果与结论（2026-08-30）

四臂均完成，进程正常退出，输出合同无违规：

| 表示与路由 | 候选数 | IoU≥0.25 | IoU≥0.50 | tiny/small Recall@0.25 |
|---|---:|---:|---:|---:|
| 原生2k + 预测32类top-1 | 5033 | 0 | 0 | 0 |
| 原生2k + GT类别路由 | 5889 | 2 | 1 | 0.0476 |
| V9双源10k + 预测32类top-1 | 4557 | 1 | 0 | 0.0476 |
| V9双源10k + GT类别路由 | 5764 | 1 | 0 | 0.0476 |

GT类别路由在原生2k表示上新增2个IoU≥0.25匹配，达到“语义路由有实质影响”的机械门槛；
V9双源10k表示没有达到表示改善门槛。但即使提供正确类别，5889个raw候选也只有2个达到
IoU≥0.25、1个达到IoU≥0.50，远未达到健康标准。故准确结论不是“语义路由是唯一根因”，
而是：**错误语义路由已被确认是一个前置损失，但修好类别仍不足以形成对象；当前affinity
身份表示与raw无监督聚类的组合仍是主要容量瓶颈。**

本节仍未测试类别先验。它说明继续修改KNN、filter或类别降噪没有意义；若要继续，应先用不依赖
硬32类路由的实例身份形成方式建立健康候选，再在同一候选池比较global/class参数。

## 33. 当前权威：类别先验参与raw碎片组装的最小验证

### 33.1 研究问题与结论边界

第24至32节多数实验把类别知识作用在完整候选形成之后，或者用类别值替换HDBSCAN参数；它们
没有直接回答：**当一个对象被raw HDBSCAN切成多个局部碎片时，逐类典型尺寸和支持量能否比
全局共享统计更准确地决定哪些碎片应合并、何时停止。**

类别先验可以参与实例形成，但它只能约束公共邻接图中已经存在的连接，不能创造图中不存在的
身份边。本节因此先评价公共碎片图是否包含正确拼装路径，再比较U/D。运行时只使用预测类别；
错误语义路由已由第32节单独量化，本节不能声称类别先验能够修复错误类别。

本节的结构依据与现有证据一致：PointGroup和HAIS都把局部点/集合聚合与完整实例形成分层；
SoftGroup说明硬语义决策会把分类错误传播到实例分组。本节不移植这些方法的训练网络，只采用
“先冻结局部碎片，再做受约束集合聚合”的最小结构。

### 33.2 冻结原子碎片与公共邻接图

正式运行只使用`native-2k-grounded + predicted-32-top1`：完整32类归一化top-1、阈值0.7、
SAGA20范围、seed42、每类最多5000点、以及第31节修正距离的raw HDBSCAN。每个raw非噪声簇
直接成为一个原子碎片；不做full assignment、中心吸附、KNN、filter、halo或类别先验。

每个场景只生成一次`FragmentGraph`，U/D逐字节共用：

- 节点保存raw fragment ID、成员Gaussian ID、预测类别、membership、语义分数和稳定来源ID；
- 只在同一预测类别的raw成员上建物理24近邻，每点保留affinity最高的4个邻居，只留互选边；
- 两个碎片间至少有3条跨碎片互选点边，且跨边物理距离不超过
  `0.1 * exp(global.log_bbox_diag_m.q50)`，才形成碎片边；
- 边证据只保存跨边数、affinity余弦中位数和最小物理距离，不读取类别统计；
- 边证据只按跨边数降序、余弦中位数降序比较；两项完全相同时视为精确并列，不用fragment
  ID强行破并列；稳定来源ID只用于确定性排序、序列化和lineage；
- GT、GT类别、GT实例和IoU不得进入graph worker或其函数签名。

第32节只保存了汇总，未保存raw bank。本节允许用完全冻结的第32节真实运行臂机械重建并持久化
raw bank；重建后的候选数和same-class IoU计数必须复现第32.4节，否则只修接线。

### 33.3 U/D唯一差异与唯一合并算法

两臂从同一批单碎片组件和同一张边表开始。尺寸统一使用第30.4节已登记的sorted PCA三轴
log extent双侧IQR平台：

\[
G_\theta(X)=\exp[-0.5\,mean(\min(z_\theta^2,25))]
\]

支持量为：

\[
C_\theta(X)=\min(1,n_{raw}(X)/m_\theta),\qquad P_\theta(X)=G_\theta(X)C_\theta(X)
\]

- `U-global`：三轴q25/q50/q75读取global，`m=5`；
- `D-class`：读取预测类别的train-only shrunk统计，
  `m_c=clip(round(5*sqrt(A_c/A_global)),3,10)`；缺失类别回退global。

除统计表外公式完全相同。5cm smoothness、类别scale gate和新训练不进入本轮。

合并固定为确定性的逐轮互为最佳：

1. 每轮先对所有公共邻接边计算合并后的先验兼容度，只保留
   `P(A union B) > max(P(A),P(B)) + 1e-6`的合格邻居；
2. 当前每个组件只在这些合格邻居中按公共边证据选择一个最佳相邻组件；
3. 只有双方互为最佳、类别相同且本轮均未被其他合并占用时才接受合并；
4. 每轮结束后用原始公共边重新聚合组件间证据，并重新计算extent和support；组件间跨边数取
   fragment边跨边数之和，余弦证据取以跨边数为权重的fragment-level余弦中位数的加权中位数
   （weighted median-of-medians），最小物理距离取所有fragment边的最小值；该聚合在U/D间共用；
5. 精确并列不合并；无可接受合并时结束；禁止一条非互选弱边桥接两个已有组件；
6. 最终mask是成员raw碎片点的并集；不做扩张、KNN、救援或事后保护；
7. 最终对象少于对应支持门槛时拒绝；主排序分数使用两臂共同的平均语义分数与membership
   几何平均，不使用`P`，避免把prior排序变化冒充mask收益。

内部对象身份固定为排序后的`source_fragment_ids`元组；紧凑输出ID只用于导出，比较器不得因
插入或删除一个对象把后续ID重排解释成大面积变化。

### 33.4 Stage A：DEV2图能力与机械验证

固定`scene0645_00`和`scene0025_01`。先完成：

- raw bank复现第32.4节真实预测路由臂；
- U/D节点、成员、类别、边端点、边证据和边顺序逐字节一致；
- 相同参数表时U/D逐点恒等；缺类时D严格等于U；
- 输入fragment顺序和类别处理顺序不改变最终partition；重复运行逐点确定；
- 最终fragment lineage完整，重叠所有权、orphan、negative metadata和`core/full`合同违规为0。

GT只在独立评价器计算公共图上限：对每个GT，只允许沿公共图中的same-GT边合并其dominant
fragment，未映射Gaussian计FP。进入实际U/D比较必须同时满足：

- 两场累计至少6个same-class IoU≥0.50图上限对象；
- official-valid tiny/small GT Recall@0.25至少0.20。

图上限失败立即停止，结论为“固定raw碎片图没有足够正确连接路径”，不是类别先验失败。

机械生效要求至少满足一项：

- 至少10%的共同首轮合并提案`|P_D-P_U|>=0.01`；
- 或U/D最终至少5个合并/保留决定不同，覆盖至少2类和2场景。

若先验未机械介入，只能判当前统计映射在该合并接口无作用。机械生效后，D进入DEV8还必须：

- IoU≥0.25和0.50匹配数均不低于U；
- candidate precision@0.25相对U提高至少25%，或unsupported比例下降至少10个百分点；
- tiny/small Recall@0.25不下降；候选数不超过U的1.25倍；
- 至少一个场景改善，另一个场景任何GT最佳IoU下降不超过0.05；
- D累计same-class IoU≥0.50至少4个、candidate precision@0.25至少10%。

未通过时停止；不得现场修改24NN、top4、3条边、0.1全局尺度、合并公式或再加第二种聚合器。

### 33.5 Stage B：DEV8配对验证与后续边界

只有Stage A通过才运行冻结DEV8：

```text
scene0645_00 scene0025_01 scene0046_00 scene0474_01
scene0591_02 scene0329_02 scene0164_03 scene0064_01
```

physical scene是独立实验单位；U/D在场景内配对，不把碎片、边或对象当独立样本，不制造多个
随机seed。D必须先达到绝对健康门槛：

- same-class IoU≥0.50候选至少12个，覆盖至少4场景；
- candidate precision@0.25至少10%；
- official-valid tiny/small Recall@0.25至少0.20；
- 候选数不超过U的1.25倍；输出合同零违规。

类别先验的相对收益门槛为：

- 场景等权candidate AP@0.25的`D-U>=0.002`；
- candidate AP@0.50下降不超过0.002；
- 至少5/8场景方向为正；
- tiny/small Recall不下降；FP/TP不恶化超过20%。

候选级通过后，才允许U/D使用同一个无类别扩张方法补齐未采样点并进入未参与设计的五场
canonical holdout；扩张方法必须在看DEV8结果前另行登记，不得把legacy KNN、保护旁路或
ObjectBank临时接回。本节不直接授权tune24/final48。

结果解释固定为：

- 图上限失败：公共实例证据不足，不能评价先验；
- 图上限通过、U/D未机械分开：该统计映射没有进入实例形成；
- D真实改变合并但不优于U：当前train-derived尺寸/支持先验对碎片组装无额外价值；
- D只改善小物：只支持小物体类别化组装；
- D在DEV8健康且优于U：支持类别先验能帮助实例形成，随后才做独立holdout确认。

### 33.6 实现、测试、产物与当前检查点

新增独立小模块与三个入口：

```text
build-category-fragment-graph
merge-category-fragments --mode global|class
evaluate-category-fragment-merge
```

不向`postprocess.py`增加路径。必须测试：100点对象被切碎后可恢复；相邻同类对象在尺寸恶化时
停止；U/D共用graph；不跨类别；互为最佳阻止错误链传递；global回退；GT不进入worker；
lineage稳定；无ID级联假差异；确定性、断点复用和损坏项单场重跑；官方evaluator保持不变。

验收产物：

```text
category_fragment_graph_dev2.parquet
category_fragment_graph_oracle_dev2.json
category_fragment_merge_dev2.parquet
category_fragment_merge_dev2_analysis.json
category_fragment_merge_dev8.parquet
category_fragment_merge_dev8_analysis.json
viewer/
```

只复用现有2k feature、30k 3DGS、GT和train-only priors；不下载、不训练、不重训3DGS、
不运行ObjectBank/尺度门控/V3–V10。单GPU单进程；磁盘至少80GB；只读cgroup memory；
不生成SHA文件、lock、schedule hash或contributor cache。

- [x] 用户确认类别先验可能参与第一步，并授权先调查、预注册后实施；
- [x] 完整回读权威文档、审计raw fragment、mutual graph与prior复用路径；
- [x] 冻结本节研究问题、公共graph、唯一合并算法和DEV2/DEV8门槛；
- [x] 实现独立模块、runner、评价与测试；
- [x] 本地全量类别先验测试通过（758项通过、2项跳过）；
- [ ] commit/push后才开启云端；
- [ ] DEV2通过才运行DEV8，失败立即形成结论并关闭云电脑。

## 34. 当前权威：高斯—掩码共识干净基线

第24至33节证明了若干旧接口失败，但没有建立一条健康、独立、完整掩码优先的
自动实例基线。老师原型确实使用二维掩码；结构缺口是掩码身份在特征训练后被
丢弃，自动实例仍由HDBSCAN、中心分配和全场近邻重新形成。不能再把“旧基线
用了掩码”和“旧基线直接以完整掩码形成跨视角对象”混为一谈。

本节由`CLEAN_ALPHA_MASK_BASELINE_PLAN.md`定义：新主干以SAM-everything完整掩码
和正确alpha贡献建立类别无关多视角共识对象，只在合并时比较global/class的
train-only米制三轴q95尺寸上界。支持量、平滑度、HDBSCAN、全场KNN、ObjectBank
和新训练均不进入首轮正式比较。

当前检查点：

- [x] 完成老师意图、交付代码、公开SAGA、MaskClustering、SAI3D、SoftGroup和
  Gaussian Grouping的一手资料审计；
- [x] 用户选择首轮只测尺寸先验，并批准独立重构；
- [x] 冻结DEV2/DEV8门槛、U/D/oracle边界和不复制无许可证源码的实现纪律；
- [x] 完成独立证据库与`alpha × T_prev`贡献归一化；完整SAM掩码ID贯穿lifting、共识、
  geometry oracle和最终对象，歧义Gaussian仅在跨视角关联与最终检测频率中弃权；
- [x] 完成并用反例测试MaskClustering行为：supporter双侧包含率均须达到0.80，欠分割按
  observable frame中diverse mask-ID distribution的频率严格大于0.30过滤，无有效mask证据
  不误计为diverse；observer按top 5%、10%至95%逐轮推进，每轮合格图按connected component
  合并并在轮末统一重算；
- [x] 完成同帧替代假设、确定性排序和整component尺寸veto的稳定处理；veto拒绝时不静默接受
  任何子集，也不在不变的后续百分位轮次重复提交同一component；
- [x] 完成C0/U/D同证据、同原始几何共识和同基础Q的尺寸重放，类别后验仅在对象形成后聚合；
  GT只进入独立oracle/evaluator，正式主干不依赖HDBSCAN、旧`postprocess.py`、ObjectBank或
  全场KNN；
- [x] 完成DEV2、DEV8、holdout5、tune24和final48的冻结门槛、物理场景配对汇总、恢复状态与
  10,000次final paired bootstrap实现及针对性测试；
- [x] 云端运行前第二轮静态审计完成：修复输入身份、无掩码帧误作负证据、拆分后沿用旧统计、恢复状态与U/D同源性，以及主要内存放大点；详细记录见`CLEAN_ALPHA_MASK_BASELINE_PLAN.md`；
- [x] 全量干净基线专项验证通过（154项通过，零失败）；仓库首方测试为913项通过、2项跳过；
- [x] 云端DEV2两场完整证据库与几何上限完成：累计36个IoU≥0.50对象，21/21个
  official-valid tiny/small对象达到Recall@0.25，输入几何上限通过；
- [x] 首个自动共识条件单核运行超过4小时40分钟仍无结果；审计确认是逐候选对、逐视角重复集合
  运算的实现复杂度问题，而非死锁、OOM或方法门槛失败；
- [x] 严格等价的逐组件/逐帧证据缓存与增量pair重算已通过150个随机完整场景的新旧输出差分、
  40组状态路径差分、241项干净基线测试及仓库1,002项首方测试；
- [x] 新consumer commit已只读导入并校验旧producer完成的两套证据库，没有重做lifting；
- [x] DEV2几何上限与机械门槛通过后进入DEV8；DEV8八场几何上限为100个IoU≥0.50对象，tiny/small Recall@0.25为75/79；
- [x] DEV8自动共识健康门槛失败：C0的554个候选和U的134个候选均为0个几何IoU≥0.25匹配，严格mAP与AP50均为0；主要断点锁定为自动跨视角关联/对象重建，而不是完整掩码输入覆盖；
- [ ] 条件性身份边正控须在固定验证难边仅含一种标签时记录AUROC不可定义和`inconclusive`终态；修复后从DEV8完整C0/U产物恢复，不重跑已完成阶段。

## 35. 当前权威：清洁基线评价纠正与平面化掩码正控

### 35.1 为什么必须先纠正评价

第34节正式 ScanNet GT 点域的 AP25 很低、AP50 为零，这一事实保留；但旧候选几何诊断还把
“没有成为任何 GT 点唯一最近邻”的大量 Gaussian 统一追加成假阳性。Gaussian 数量越密，
这个惩罚越重，它不等于 ScanNet 在 GT 点域计算的实例 IoU。因此旧报告中“所有候选几何
IoU 都为零”和极高 unsupported 污染率不能直接用于结构归因。

本节先将三种空间彻底分开：正式 IoU/AP 只在 GT 点域计算；Gaussian→GT 只报告预测点精度
和无映射比例；GT→Gaussian 只报告对象覆盖率。不得再加 synthetic FP sentinel，也不得把
后二者拼成伪 IoU 或 F1。候选在 IoU 0.25/0.50 分别进行确定性一对一最大匹配，重复预测只能
有一个真阳性，其余计假阳性。主 AP 协议为官方 0.50 至 0.90 九档与单列 AP25；0.50 至 0.95
十档仅作历史对照。tiny/small 分母使用全部原始 GT 点数不少于100的正式有效实例，不能先按
映射成功数筛选。

历史 `perfect_trim` 更名为 `support_coverage_ceiling`。它允许 GT 完美删除假阳性并跨 GT
复用证据，只能说明逐物体覆盖上限，不再充当自动实例可行性门槛。

### 35.2 第一步：冻结 DEV8 的阶段漏斗

只读复用第34节 DEV8 evidence bank、C0/U 输出和 diagnostics，按以下固定阶段重建：

```text
完整掩码支持
→ 去除同帧歧义后的关联支持
→ 删除欠分割掩码
→ accepted edges 形成的合并组件
→ 检测比例过滤
→ 物理 DBSCAN 拆分与包含去重
→ 唯一 Gaussian 所有权
→ 最终 export
```

生产转换抽成纯函数供审计和正式运行共同调用。每一级保存对象数、Gaussian 点数、一对一
IoU≥0.25/0.50匹配、GT召回及相对上一级损失。重建出的最后分区在实例重编号后必须与冻结
`output.json`逐点一致，类别和实例数完全相同。任何冻结 bank 或历史输出在运行前后内容身份
必须不变。

第一步只有技术门槛，没有科学早停：GT-as-prediction在官方九档和AP25均为1；新正式域
AP25/AP50逐值复现现有正式输出；历史十档复评一致；每场GT→Gaussian 5cm映射率不少于0.90；
阶段重建最终changed points为0、类别和实例数一致。任一失败只修评价实现并阻止第二步，不能
输出科学归因。科学得分高低本身不阻止第二步。

### 35.3 第二步：同源 H′ 与 P 的两场正控

固定`scene0645_00`和`scene0025_01`。使用同一图像、同一现有 SAM checkpoint、相同配置和
同一次生成结果，每帧保存压缩原始栈以及`predicted_iou/stability_score/area`。从它同时派生：

- `H-hierarchy`：保留全部原始分层重叠掩码；
- `P-flat`：把相同像素并集确定性转换为对象级唯一观察。

历史 H 只作漂移审计，主比较只能是同源 H′ 对 P。P 对每个被覆盖像素按 predicted-IoU、
stability score、原始 mask ID 的顺序选唯一归属；只删除平面化后为空的 mask，不增加400像素
过滤或任何新阈值。P 与 H′ 的像素并集必须逐点相同，P 内像素重叠率必须为0。

由于一个 Gaussian 的 alpha footprint 仍可能跨越两个不重叠像素掩码，P 在提升后还要执行
一次帧内唯一归属：对满足原 inside/visible 门槛的多个 mask，按 inside/visible 比例最高者，
再按上述 mask 优先级选一个。由此每个 frame×Gaussian 最多属于一个 P mask。H′/P 除
`mask_observation_mode`外必须共享30k Gaussian、相机、`alpha×T_prev`、inside/visible阈值、
多视角共识、欠分割、DBSCAN、包含去重、Grounded-SAM后置分类和纠正后的评价。

本阶段只运行无先验C0；不得运行U/D、身份头、下载或训练。P的机械门槛为像素重叠率0、像素
并集完全一致、frame×Gaussian唯一、重复运行逐字节确定，以及无orphan、负metadata和重复
所有权。

### 35.4 两场科学门槛和固定解释

P通过须同时满足：相对H′新增至少2个一对一几何IoU≥0.50对象；两场累计至少6个；正式可评
候选precision@0.25不少于10%；全部official-valid tiny/small Recall@0.25不少于0.20；候选数
不超过H′的1.5倍；至少一场改善，另一场几何Recall@0.25下降不超过0.05。歧义率下降只作机制
检查，不是效果门槛。

归因固定为：完整支持到活跃支持大幅下降且P改善，说明分层掩码合同错配是主因；P像素唯一后
Gaussian冲突仍高，说明alpha footprint或遮挡提升是主因；歧义下降但几何不升，说明当前SAM
掩码完整性或污染仍不足；几何改善但same-class/AP50不升，说明晚语义分类是主因；组件健康而
最终export坍塌，说明检测过滤、DBSCAN或唯一所有权是主因。P全部通过后才可扩DEV8无先验
基线；P失败就停止调共识阈值，下一授权只能比较真正对象级掩码来源或SAI3D式几何超点，不能
外推为所有对象级掩码无效。

### 35.5 入口、产物和执行检查点

新增入口：`audit-clean-baseline`、`prepare-flat-mask-control`、
`run-clean-baseline-two-step`。产物固定为：

```text
clean_metric_reaudit_dev8.parquet
clean_metric_reaudit_dev8.json
clean_stage_funnel_dev8.parquet
clean_stage_funnel_dev8.json
sam_metadata_regeneration_dev2.json
flat_mask_input_audit_dev2.json
mask_contract_ablation_dev2.parquet
mask_contract_ablation_dev2.json
clean_two_step_analysis.json
```

输出使用commit绑定的新`artifacts/clean-mask-contract-*`和`runs/clean-mask-contract-*`，旧产物
只读。第一步零GPU；第二步单GPU单进程、按帧恢复。不得下载、训练、重训3DGS、生成独立SHA
文件、lock、schedule hash或逐像素贡献缓存；磁盘至少80GB，内存只读90GiB cgroup。完成或
停止后删除当前任务的每小时检查自动化，不自动关闭用户的AutoDL云电脑。

- [x] 用户批准两步一次性实施，科学得分不在两步之间触发询问；
- [x] 主代理完整回读权威文档并冻结本节边界；
- [x] 实现三空间评价、一对一匹配、双AP协议和阶段重建；
- [x] 实现同源SAM元数据、H′/P派生和Gaussian唯一观察合同；
- [x] 本地专项305项、仓库首方1068项通过（另2项按既有条件跳过）；
- [ ] commit/push并部署新工作区；
- [ ] 通过第一步技术门槛后自动完成两场H′/P并形成固定归因。

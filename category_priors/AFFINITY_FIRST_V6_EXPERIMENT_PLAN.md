# SAGA V6：Affinity-first 候选质量修复与类别先验重验证

## 状态与研究问题

V6 把 B0/B1 冻结为历史工程锚点，不把它们当作健康的自动 ScanNet 实例分割主干。现有
`source/a800` 的小物体分支只是共享参数的早期启发式；它有弱 AP25/AP50 信号，却没有
稳定的严格 mAP 收益。V5 旁路已确认没有意外改坏 B1，但也证明原 codebook/multiview
候选供给不足，故 V5 不检验类别先验本身。

本阶段唯一目标是先证明现有资产能形成足以测试类别先验的实例候选；只有候选和保守融合
均通过，才把 ScanNet-train 的类别统计用于离线候选接受分数。

## 冻结的 V6 候选结构

1. 对全部 30k Gaussian 的 `scale=1` affinity feature 建物理 24-NN 图。
2. 每点只保留 affinity cosine 最高的 4 个物理近邻；边必须双向入选。
3. 互选边的连通分量是候选。候选 core 是内部度数至少 3 的点，且 core 至少 10 点。
4. 候选形成后才以多视角最大贡献 Gaussian 投票定类：每帧每 Gaussian/候选最多记一次；
   32 类共同竞争，只有 SAGA20 胜者、至少 3 个有效视角、投票比例至少 .60、领先第二类
   至少 .10 的候选进入 bank。
5. 候选阶段只写 `v6_proposals.json` 与 `v6_proposal_labels.npz`，B1 `output.json` 不改写。

禁止语义预路由 HDBSCAN、全局中心分配、全局 KNN、`filter_num(10)`、B2、class-first、
prior-v2、halo/carve-out、持久 contributor cache 和任何 GT 运行时输入。

## 分阶段和门槛

### Stage 0：溯源与输入漏斗

复用 B0/B1 和固定的 V3 八个 tune 物理场景；生成 raw SAM 覆盖、正确类别 mask 覆盖、
raw/阈值 codebook 语义召回、affinity 局部边 AUROC 以及 V6 候选 oracle Recall。GT 仅离线
诊断。

- tiny/small GT 的中位原始 SAM 覆盖低于 .35：只进入 8 场景 SAM 输入修复。
- SAM 覆盖足够、但 affinity edge AUROC 低于 .60：只进入三场 10k feature 正控。
- 都未触发且 V6 候选绝对门槛失败：停止，结论为当前 graph 构造无效。

### Stage 1/2：条件性输入修复

SAM 修复仅把 GroundingDINO box/text threshold 从 .35 改为 .25，独立输出；tiny/small 覆盖
至少 +10pp、medium/large 不低于 -2pp、mask 数不超过 1.5 倍才接受。10k 正控仅运行
`scene0011_00`、`scene0608_00`、`scene0645_00`，不覆盖 2k 资产；edge AUROC 平均至少
提高 `.05`、同类 IoU≥.50 候选新增至少 2、无场景损失超过一个既有匹配、候选数不超过 1.5 倍
才说明 2k feature 是限制因素。两者都不自动扩展到 24/48。

### Stage 3：V6 候选门槛

固定八场景、seed42。候选须同时达到：同类 IoU≥.50 至少 12 个且覆盖 4 场景、
precision@.25 至少 5%、tiny/small Recall@.25 相对 V5 codebook +.02 或新增至少 5 个
IoU≥.50 匹配、候选数不超过 V5 codebook 的 1.5 倍。失败即停止，不实现融合。

### Stage 4：唯一保守融合

B1 永远为默认。异类 B1 IoU>.25 的候选拒绝；同类 IoU≥.25 时只补 B1 背景 core 点；所有
B1 IoU≤.25 时 core 才可新建实例。不覆盖 B1 前景、不 carve-out、不 halo、不做后续 KNN
或二次 vote。uniform 融合相对 B1 必须满足 mAP 不低于 -.001、AP50 不低于 -.002、实例数
不超过 1.25 倍且覆盖不低于 -1pp。

### Stage 5：同 bank 的类别先验

候选 ID/core/类别/融合固定；只离线改变 `U00-uniform`、`D10-size`、`D01-core`、
`D11-combined` 的接受分数。size 是单边异常大惩罚；core 是局部密度校准支持，统计只来自
冻结 ScanNet-train priors。若任何 D 对 U 达到 mAP +.002，或 tiny/small Recall@.50 +.01
且 mAP 不低于 -.0005，并且正向场景更多、FP/TP 不恶化超过 20%，才进入独立复核。

仅在候选至少有 12 个 same-class .50 正例、precision@.25 至少 5%、uniform 融合安全但
所有 D 未通过时，允许固定 L2 logistic 后备校准器：仅用八个开发场景拟合，固定阈值 .50，
再在剩余16场景独立评估；不得搜索特征、阈值或正则。

### Stage 6：独立复核与内部验证

通过八场门槛的唯一方法才可跑余下 16 个 tune 场景，要求余下16 ΔmAP>0、全24 +.002、
正向场景更多且 tiny/small 指标为正。图构造无随机采样，不伪造 seed 重复。通过后才运行
48 个内部验证场景的 U 与 best-D，并做 10,000 次 physical-scene paired bootstrap；只有
ΔmAP≥.002 且 95% CI 下界>0 才支持稳定改进。48 结果不得调参，只称内部验证。

## 产物与结论边界

`v6_provenance_audit.json`、`v6_input_funnel8.parquet`、`v6_candidate_bank8.parquet`、
`v6_replay8_metrics.parquet`、`v6_tune24_metrics.parquet`、`v6_final_metrics.parquet`、
`v6_analysis.json` 和 viewer。无需下载、3DGS 重训、GT 重建、SHA/lock/schedule hash。

- 输入不足：类别先验没有可稳定作用的候选空间。
- 候选失败：当前 affinity graph 构造无效。
- uniform 融合失败：实例仲裁不安全。
- 候选/融合均健康而 D 不优于 U：当前 train-derived 类别统计无额外价值。
- 48 场景显著优于 U：支持类别先验在当前自动化改造中的稳定收益。

# 类别先验与实例修复

当前状态（整理于2026-09-22，最近实验为2026-09-19）：`object-explanation-v1`
已完成类别／通用两组各十个局部诊断与五个训练频次长尾对象的缓存回放，未达验收目标。
本轮未启动374全量，未完成38 GT场景评价，也未执行新查询反馈。

| 十诊断平均留出视角二维投影 IoU | 旧选择 | 新纯选择 | 组合后实际写回 | 固定库事后上界 |
| --- | ---: | ---: | ---: | ---: |
| 类别组 | 52.35% | 57.06% | 57.09% | 76.75% |
| 通用组 | 50.94% | 57.13% | 57.16% | 76.61% |

- 选择差距仅缩小约19%／24%，未达到减半目标，不能宣称稳定选中最佳候选。
- 类别组办公室电话的固定源—GT三维选择IoU从38.73%降至9.84%；五长尾仍无IoU 0.5救回。
- 共同原始掩码去重327张：复用29张，补算298张，全部可读；但本批语义身份约束触发仍为0。
  缓存补齐不等于身份机制有效，下一步优先查触发条件及电话错误替换。
- 尚未证明类别尺寸相对通用尺寸有额外收益。此前五长尾的类别一致平均3D IoU为
  旧修复11.47%、类别19.94%、通用20.07%；整物体组装后类别19.55%、通用18.96%。
  这与上表局部二维投影指标不同，不能混用；两个场景、五个长尾不能证明泛化。

原始照片、标注、B0/C0、评价分母、历史结果与GPU账本均保留。
语义补算占用40.027秒，累计共享验证约127.098／1080秒；执行前仍须读取原账本各分项余额，
不能把算术余额视为新的授权额度。没有实验进程或定时监控需要恢复。

最新产物目录：`artifacts/multiview-repair-20260918/object-explanation-semantic-v2-20260919/`，
含`summary.json`、`local-paired.csv`、`tail-paired.csv`、`declines-over-10pp.csv`、`results.html`和配对图。
精确回放脚本、环境及云端路径见同级`HANDOFF.md`；运行产物不进入Git。

保留类别统计、候选读取、SAM/Alpha-CLIP 接口、成员修复和 ScanNet 实例评价。
老师原有的 SAGA 流程见仓库根目录 `README.md`。

当前实验入口：

- `run_effective_repair.py`：引导 SAM 分割、G0/G2 成员规则、语义分类和反馈。
- `run_effective_repair_evaluation.py`：完整场景导出与原始 B0 的配对评价。
- `effective_repair_core.py`：CPU 成员规则、场景合并和局部像素指标。
- `category_scale_experiment.py`：类别尺寸观察、15 个候选的固定评分、等结构全局对照。
- `object_scope/`、`object_verification/`：入口使用的模型、相机、渲染和预算工具。

```bash
python -m category_priors fit --stats train_instances.parquet --output category_priors.json
python -m category_priors evaluate --manifest evaluation_manifest.json --output metrics.json
python run_effective_repair_evaluation.py --manifest inputs.json --output-dir results
python -m pytest -q
```

GPU 入口依赖已有实验计划、模型和场景资产，不能仅凭代码仓库重跑。
`SAGA_ROOT` 可指定资产根目录，默认是仓库目录；计划内的资产路径仍需有效。
使用已有 `plan.json`，并保留同一份累计 GPU 预算记录。

早期完整场景实验结果保存在本地
`artifacts/dev2-object-scope-v2-20260909/effective-repair-01/`：
`metrics.csv`、`gt_outcomes.csv`、`prediction_outcomes.csv`、`evaluation.json`
和 `scenes/`。这些文件不进入 Git。该轮覆盖 374 个输入和 18 份场景导出；
局部有收益，但九个完整场景条件均未达到预设目标。

旧实验入口和生成报告已清理。历史上的负结果不等于错误基线，保留的数值结果
仍用于比较；已提交过的旧代码可从 Git 历史查阅。

类别尺寸实验仍通过原入口执行，旧命令行为不变：

```bash
python run_effective_repair.py --experiment category-scale --output CATEGORY_OUTPUT --stage all
python run_effective_repair.py --experiment category-scale --output GLOBAL_OUTPUT --prior-mode global --reuse-output CATEGORY_OUTPUT --stage all
python run_effective_repair.py --experiment category-scale --output CATEGORY_OUTPUT --stage ranked
```

先完成类别组，根据阳性证据再跑全局对照、整物体组装；不是默认把所有组全跑。
`--reference-output` 指向旧 `effective-repair-01`；可用 `--priors` 指定训练集先验。
整个阶段最多新增 18 GPU 小时，同时受原累计 24 小时上限约束。
输出包括 `bank.json`、实际输入图、`object_iou.csv`、`paired-object-effects.csv`
和 `index.html`。候选上界只用于事后诊断；自动选择不读取人工标注。
只有绑定先验确实不足时，才用 `python -m category_priors.fit_category_sizes --help`
中的尺寸统计入口补齐；它排除 DEV2 的所有同物理场景扫描。

`--projection-domain observed` 在候选评分时仅比较实际 SAM 裁剪内的像素，
避免把未观察区域当作反证。默认 `full` 保留已完成 E1/E2 的评分；切换须用新输出目录。

新批准的多视角实验使用 `--experiment multiview-repair`，不改变上述旧默认行为。
场景竞争实例在固定冲突对象集合内共用观察池，并排除集合中任一对象的评价视角。
反馈定位只读初始两视角的实际写回；开放观察候选保留，反馈写回以开放观察场景作回退基线。
`--stage writeback` 只重算已保存的十对象候选写回与评价，使用 E3 写回预算，不重新调用 SAM。
已提交实例的纯子集遇到输出词表不支持的重分类时，保留已有有效类别并显式记录分类冲突；
新实例和扩张不适用这一回退。已确认重复实例去重时先保留无可靠反证的独立核心，
再统一分配和重评分，避免去重本身制造“核心丢失”。
同一 `--shared-output` 固定两组共同的初始观察和后续照片；每组的 `scene` 阶段
同时保存 `initial`、`matched-open`、`feedback`，以及同一候选的旧组装输出。
先对类别、通用两组分别运行 `local` 和 `diagnostic`，检查十个原诊断对象与五个
长尾对象，再对两组分别运行 `scene`。通用组增加 `--prior-mode global`。

```bash
python run_effective_repair.py --experiment multiview-repair --output STUDY/category --shared-output STUDY/shared --stage local
python run_effective_repair.py --experiment multiview-repair --output STUDY/category --shared-output STUDY/shared --stage diagnostic
python run_effective_repair.py --experiment multiview-repair --output STUDY/global --shared-output STUDY/shared --prior-mode global --stage local
python run_effective_repair.py --experiment multiview-repair --output STUDY/global --shared-output STUDY/shared --prior-mode global --stage diagnostic
python run_effective_repair.py --experiment multiview-repair --output STUDY/category --shared-output STUDY/shared --stage scene
python run_effective_repair.py --experiment multiview-repair --output STUDY/global --shared-output STUDY/shared --prior-mode global --stage scene
python -m category_priors.multiview_repair_report --output STUDY
```

尺寸和 SAM 质量不参与新模式排名；三张原始掩码均作为跨视角关联起点。
照片边缘和近似深度遮挡仅提供观察线索，未知不记反证，不补造几何。
一轮反馈读取实际写回成员，候选上界和 GT 只在预测保存后评价。
新增计算最多 11.2 GPU 小时，仍由原账本同时约束分项、18 小时阶段和累计
24 小时上限；中断或具体错误重跑不清除已花预算。所有 374 输入与 38 GT
完整保留，五个训练频次长尾对象独立报告。

`--selector common-geometry-v1` 在独立输出目录启用共同通用观察的几何选择。
类别、尺寸和 SAM 质量不参与几何排序；候选差异、逐视角 TP/FP/FN、
删一视角敏感性和备选接替保存在原候选目录。旧默认行为不变。
内部 `scene.json` 保留全部类别及 unknown；`scene-evaluation.json` 使用原评价词表，
并引用完整几何，类别无关评价仍包含词表外实例。缓存缺少分类计算时明确记录
`pending_model`，不生成正式类别导出；现有 `--stage writeback` 可在 E3 预算内补算，
不会重新调用 SAM。反馈候选必须基于新选择器实际初始写回重新生成。

`--selector regional-evidence-v2` 保留共同观察的多个解释，前景/背景分歧对称记未知，
按同一解释比较候选差异，独立支持与反证决定替换资格。未知不加分也不作隐性否决。
原库纯选择和 `regional_repair` 分开保存；`*-pure` 场景用于单列纯选择写回，
局部组合不进入原库上界。共同互相提示验证仅更新证据，不增加候选或反馈轮数。

旧`regional_evidence_revision=known-domain-v1`仍可用于历史复现。
批次回放脚本与资产不在Git中，不能仅凭入口默认配置复现具体历史成绩。

`--selector object-explanation-v1` 比较共同原始观察的竞争解释，采用固定三项等权能量。
当前修正版对齐裁剪已知域并报告覆盖缺口；已有完整区域语义仅用于相对身份对应，
不加入边界能量。两组使用相同 `neutral_semantic_roots`，缺失编码记录 pending_model。
原库选择、区域组合、双身份及争抢区域组合分别保存。新增验证必须保存真实
observation/alpha 文件，经原 G0/G2 组成成员；查询预演与实际回答都新增原始解释，
使用同一选择函数及写回函数。诊断目标未通过前不执行374全量。

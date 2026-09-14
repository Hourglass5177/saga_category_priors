# 类别先验与实例修复

保留类别统计、候选读取、SAM/Alpha-CLIP 接口、成员修复和 ScanNet 实例评价。
老师原有的 SAGA 流程见仓库根目录 `README.md`。

当前实验入口：

- `run_effective_repair.py`：引导 SAM 分割、G0/G2 成员规则、语义分类和反馈。
- `run_effective_repair_evaluation.py`：完整场景导出与原始 B0 的配对评价。
- `effective_repair_core.py`：CPU 成员规则、场景合并和局部像素指标。
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

最近一轮结果保存在本地
`artifacts/dev2-object-scope-v2-20260909/effective-repair-01/`：
`metrics.csv`、`gt_outcomes.csv`、`prediction_outcomes.csv`、`evaluation.json`
和 `scenes/`。这些文件不进入 Git。该轮覆盖 374 个输入和 18 份场景导出；
局部有收益，但九个完整场景条件均未达到预设目标。

旧实验入口和生成报告已清理。历史上的负结果不等于错误基线，保留的数值结果
仍用于比较；已提交过的旧代码可从 Git 历史查阅。

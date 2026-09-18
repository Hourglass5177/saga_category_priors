# Repository instructions

- Preserve the teacher's original files from `source/a800` (`8c5e167`), including
  `README.md`, `CLAUDE.md`, `command.txt`, third-party documentation and licenses.
- Current research entry points and data locations are in `category_priors/README.md`.
  Keep documentation short. Do not recreate audit reports, review ladders or
  abandoned experiment controllers. Git history contains earlier tracked work.
- Preserve original annotations, frozen B0/C0 inputs, evaluation denominators,
  negative results and the cumulative GPU ledger. Ground truth and human answers
  belong in evaluation, never in automatic model decisions.
- Use focused tests for the changed behavior. GPU experiments require the existing
  budget accounting; the authorized DEV2 budget is cumulative, not reset per run.
  DEV8 execution remains outside the authorized experiment scope.
- Do not commit datasets, model weights, generated reports, deployment bundles,
  source checkpoints or runtime artifacts.

## 工程与实验原则

- 以老师的原始问题为主线：让物体类别的尺寸、形状或平滑性先验实际参与
  3DGS 实例分割，并在公共数据上验证；通用修复的收益不能直接算作类别先验收益。
- 优先复用已经有效的完整流程。新方案必须做到同条件的最终 3D 写回比较，
  不因理论上更整洁、约束更严格或局部指标更好就替换旧方案。
- 先做有望产生明显收益的完整实验，再围绕阳性结果做必要消融。
  不把反复审计、哈希闭环、合同、审批梯级或通用框架建设作为探索前提。
- 只做与当前问题有关的实现和检查；遇到具体错误就修复并继续实验。
  科研进展用假设得到的证据衡量，不用代码量、测试数和报告数量衡量。
- 先看原图、投影点与框、模型实际输入、输出掩码和最终 3D 成员的配对图。
  图片突出证据，解释放在正文；失败应定位到具体环节，不能凭猜测归因。
- 初始投影和正确锚点可以定位目标，但旧簇不是目标真值。允许补全物体和剔除
  错误成员，不能要求新结果先符合旧错误簇，才能接受新证据。
- 类别先验应改变实际观测或成员决策；检查它是否被旧包围框、裁剪上限等抵消。
  先验是软约束，不能仅因尺寸偏离就删除真实物体或强迫部分可见物体达到典型大小。
- 自动类别判断允许不确定和回退；训练集类别统计可作先验，评价集真值和人工
  指定类别不得进入自动决策。人工辅助诊断要明确标注。
- 每轮用简短说明固定假设、对照、输入范围、评价结果和计算预算，完成整个
  约定样本集，保留失败与退化；看到收益后再细分机制，不挑成功样本冒充整体。
- 同时看物体修复和完整场景结果：局部阳性值得继续研究，全局未达标不抹杀它；
  同时记录救回物体、损伤原有正确物体和新增假阳性，局部成功不等于全局提升。
- 按老师近期反馈，低信息量的边缘碎片可统一过滤，暂不作为主要攻关对象。
  规则须基于推理时可用信息，对各对照一致；不能排除整个薄物体类别，
  也不能追溯改写已有结果、标注或评价分母。
- 当前优先在有效的单轮修复上检验类别条件先验；复杂多轮反馈和更大范围实验，
  应由明确的失败机制与新证据驱动，不继续扩建控制流程。

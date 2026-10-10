# Stages127–129：单独修补任务协方差仍不足

Stage127完成原27列内的稀疏来源任务模板。原法则复现、PSD／支持／频率边际、独立弱尾、稠密协方差及导数检查均通过；固定q=.01的来源对数似然平均变化为−.00020746／−.00067595，12折全下降。按登记条件停止该方向，未拟合它的q。

Stage128将来源导数拆开：新增任务组边际及水平—形状相关项均受惩罚；布局6删除task1时的水平方差边界惩罚尤大。排除原共同／任务局部自由方向后，模板信息平均只剩6.15%／22.42%；这说明模板方向与原参数变化高度重叠，不能据此声称联合重估必然改善。保持原任务形状边际、仅加入形状相关的固定探针有微小平均收益，分别4/6、3/6折改善。

Stage129因此完成12个单参数来源拟合和12个重新构建的条件独立截距方差积分。原字典、组预算／频率、独立弱尾及无下限截距方差均保持；原优化合同、导数、稠密条件化和全尾积分检查通过。六个重叠折的总来源对数似然增益仅.1070／.2914，不能当作独立证据累加。q在布局5为[1,0,1,1,1,0]，布局6为[1,0,0,1,1,0]；两个task1折都选择零，积分后法则仍复原旧方差混合，关键水平误差没有修复。

后续[Stages130–134](orthogonal_meta_prior_stage130_134_20261006.md)已完成联合来源估计、重新积分和缓存8次前缀预测：来源似然改善，但任务模板q全为零；预测仍弱于原方差混合及来源GP，候选停止。下一项已登记任务形状组方差的来源不确定性诊断。

限制：全部仍是训练来源0–5的开发计算；未产生付费预测或独立部署证据，完整模型仍HOLD。约1.51本地单线程CPU秒；12个一维来源拟合、12个来源方差积分，新增仿真／验证／终端调用、服务器任务及拉取均0。

结果：[模板方向](../paper_artifacts/legacy_joint_population_f_sparse_task_template_v1/results.json)、[失败归因](../paper_artifacts/legacy_joint_population_f_task_template_attribution_v1/results.json)、[相关拟合与积分](../paper_artifacts/legacy_joint_population_f_task_shape_correlation_reference_v1/results.json)、[零相关法则复原](../paper_artifacts/legacy_joint_population_f_task_shape_correlation_reference_v1/native_equivalence.json)。

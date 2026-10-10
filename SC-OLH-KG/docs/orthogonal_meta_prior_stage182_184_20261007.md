# Stages182–184：联合谱法则改善外层误差，但未通过完整门

两个五来源拟合及结构／条件化检查通过。固定四次观测、十二评分点和相同风险先验，相对保留混合，布局5／6 MSE下降10.17%／31.67%；布局6三个指标均改善。布局5批密度下降0.06658，相对GP的MSE上升2.85%，因此按原门停止扩大该候选。[外层结果](../paper_artifacts/legacy_joint_population_f_outer_joint_spectral_geometry_v1/results.json)。

缓存归因显示布局5损失来自条件均值密度−0.06841，方差标签密度+0.00184。水平误差缩小，但形状MSE相对混合增加32.1%，形状方差只剩29.5%；最大密度损失在策略72，未删除该点。候选90%覆盖为7/12、4/12。布局6水平误差仍占MSE的72%。[归因结果](../paper_artifacts/legacy_joint_population_f_outer_joint_spectral_transfer_v1/results.json)。

12个来源法则的条件Fisher检查通过，任务谱均位于上界4，局部无约束标准误约1.47–2.92；有限来源对谱的信息偏弱。该量不是边界后验区间，也未计模板／支持估计不确定性。[来源信息](../paper_artifacts/legacy_joint_population_f_joint_spectral_information_v1/results.json)。

后续有界谱及幅度不确定性试点、来源稳定性归因和噪声口径核对已完成，见[Stages185–191](orthogonal_meta_prior_stage185_191_20261007.md)。谱积分影响小；幅度混合未通过内部筛选，未追加task1预测。下一步转向来源潜在方差似然诊断。

限制：全部为已见开发数据。该法则有预测价值，但没有形成两布局一致优势或稳定独立决策证据；点谱候选仍停止，保留混合仍作参考，完整模型HOLD，论文及旧60/60未变。

成功计算约0.64本地单核CPU秒，两次来源拟合、两份新预测，NPZ仅4.8KB、三包共约399KB；新增仿真、验证、终端及服务器工作均0。归因结果导出时修复一次NumPy计数类型，重放24项诊断积分；失败记录保留、未改参数或容差。入口为`check_legacy_f_outer_joint_spectral_geometry.py`、`diagnose_legacy_f_outer_joint_spectral_transfer.py`、`diagnose_legacy_f_joint_spectral_information.py`。

# Stages211–219：来源水平／风险学习推进，完整模型仍HOLD

完成九个阶段，保留原正交、稀疏、低阶／加性、低频结构及正弱尾。来源后验矩与同一原始似然的Stein导数检查均通过；风险任务／独立扰动的巨大负矩梯度属于估计不稳定，导数估计不支持增加这些方差。[来源方向](../paper_artifacts/legacy_joint_population_f_raw_source_level_risk_score_v1/results.json)、[导数诊断](../paper_artifacts/legacy_joint_population_f_raw_source_risk_stein_score_v1/results.json)。

一次来源联合M步四组收敛，全量和两半段的函数、风险Q均改善；按新函数和风险法则重新条件化，四组后验及512状态保留检查全部通过。公共函数／风险截距协方差保持固定，任务截距及风险强形状幅度从来源学习。[M步](../paper_artifacts/legacy_joint_population_f_raw_level_risk_mstep_v1/results.json)、[新后验](../paper_artifacts/legacy_joint_population_f_level_risk_source_update_v1/results.json)。

这次M步后的task0开发预测：结构MSE相对同样学习的GP在布局5下降24.18%，布局6上升19.27%；四组密度分段精度均失败。缓存诊断的均值功能量精度通过，布局6MSE差的条件MC区间为[4.550e−5,4.584e−5]，约70%为整体水平、30%为政策形状。[预测及保留失败](../paper_artifacts/legacy_joint_population_f_level_risk_predictive_mixture_v1/results.json)、[误差／来源梯度](../paper_artifacts/legacy_joint_population_f_level_risk_learning_attribution_v1/results.json)。

随后沿同一来源目标额外接受两轮M步及新后验。期间一次GP优化失败由同目标的Cholesky白化求值修复，参数、数据、收敛门槛均未改变，旧失败保留。之后风险均值的半段梯度符号分歧触发停机；函数中心已接近稳定，风险形状幅度仍有同向下降梯度。[原失败](../paper_artifacts/legacy_joint_population_f_bounded_level_risk_learning_v1/results.json)、[数值修复](../paper_artifacts/legacy_joint_population_f_level_risk_whitened_Q_v1/results.json)、[最终来源状态](../paper_artifacts/legacy_joint_population_f_bounded_level_risk_learning_v2/results.json)。

原登记的[来源风险幅度似然比诊断](../performance/manifests/legacy_joint_population_f_risk_amplitude_importance_plan_v1.json)现已在历史复用阶段完成，复用了旧混合权重引擎，见[历史核查与适配记录](historical_component_reuse_20261007.md)。后续接续历史模块；本阶段旧结果、失败和完整模型HOLD保持原有结论。

限制：当前只学习既有支持上的四参数子问题，尚未收敛完整群体模型；后两轮来源更新没有再次评分task0。task0是反复开发数据，MC区间只描述计算误差；密度点优势、因子恢复、KG及独立部署收益均未建立，原生混合和旧负结果继续保留。

九阶段记录25.77本地单核CPU秒、约56.24MB JSON，17次优化（含数值重放）、48条新条件化链及一次原状态重放；零新增数据、坐标拟合、仿真、验证器、终端积分及服务器工作。开发预测首次因任务名／整数索引不匹配在预测前停止，修正同一任务键后完成，[事件已留存](../paper_artifacts/legacy_joint_population_f_level_risk_predictive_mixture_v1/execution_incidents.json)，该失败尝试CPU未记录。

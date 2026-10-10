# Stages114–115：完整门仍缺一个案例，剩余差距在f均值

已合并冻结的六来源f方差混合与Stage104 g法则，保留七个风险身份、共享权重及原付费更新。四份预测先于评分保存；原生接口、加密协方差／证据检查均通过，三个评分点的密度加密差异最多3.44e-8。批量密度积分与4788次原分量积分差异最多4.44e-15；这项密度计算的实测加速约15.14倍。

同216门**4/4通过**，原144门仍**3/4**。分配6／task7的联合密度差由-.031591缩至-.010498；f密度差-.125692，g+.114664。相对Stage104，四个联合密度均提高，但合并候选仍停止，完整模型HOLD。

缓存归因显示，该案例f均值MSE比旧144高68.11%；差额.000121877中，整体水平占39.47%、中心化形状占60.53%。固定预测下的近似95%测量区间[.000106960,.000136793]保持正值。其他三个案例的f均值MSE分别下降24.58%、20.08%、35.54%。这些分解不是因果证明；测量区间不表示任务总体确认，source6/7也是已打开的开发记录。

后续来源增量、信息上限及4／8次观测对照已完成，见[Stages116–122](orthogonal_meta_prior_stage116_122_20261006.md)。新增来源观测有效，固定旧参数仍失败；8次训练观测降低原生f误差约34%／36%，但同预算来源GP仍更强。下一步已登记Stage113 f方差混合的4／8次相同预算对照。保留Stage113开发通过结果，不增加方差下限、不重调失败候选、不进入目标／KG。

本次完整检验约127.480本地单线程CPU秒，缓存归因不到.1秒；新拟合、仿真／验证／终端调用、服务器任务及拉取均0。入口：[完整门](../performance/check_legacy_f_task_level_reference_complete.py)、[批量密度](../performance/vectorized_joint_batch_density.py)、[缓存归因](../performance/diagnose_legacy_f_reference_complete_attribution.py)；结果：[Stage114](../paper_artifacts/legacy_joint_population_f_task_level_reference_complete_v1/results.json)、[Stage115](../paper_artifacts/legacy_joint_population_f_reference_complete_attribution_v1/results.json)。

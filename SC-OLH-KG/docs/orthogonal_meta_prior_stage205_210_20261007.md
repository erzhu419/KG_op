# Stages205–210：来源联合混合打通，布局6预测劣势成立

一次MCEM协方差M步四组均收敛，全量／分段Q改善；结构每批改善仅1.82e−5／7.40e−6。新协方差下原48条来源后验四组通过精度门，最小ESS291–396；结构公共均值移动仅0.00467／0.00212个旧SD。[来源学习](../paper_artifacts/legacy_joint_population_f_raw_covariance_mstep_v1/results.json)、[重新条件化](../paper_artifacts/legacy_joint_population_f_learned_covariance_source_update_v1/results.json)。

共同函数／风险的正权混合四组重建通过，保留混合间跨头协方差及未来任务残差；原正交、稀疏、低阶、低频和正尾约束保留。[联合混合](../paper_artifacts/legacy_joint_population_f_raw_source_shared_mixture_v1/results.json)。

在同12条task0开发批上冻结零观测预测后评分。512状态试点三组密度分段精度失败；原生混合精细积分通过。按原seed／原长度重放全部4096状态，复现原后验及512子集；没有追加独立样本。布局6两臂随后通过全部未来密度门，布局5仍超过0.02界。[保留的512失败](../paper_artifacts/legacy_joint_population_f_raw_source_predictive_mixture_v1/results.json)、[完整状态结果](../paper_artifacts/legacy_joint_population_f_raw_source_full_state_prediction_v1/results.json)。

精度合格的布局6结构MSE高于GP13.68%，批密度差−0.02967。MSE增量约63%为整体水平、37%为政策形状；平均密度差的条件Monte Carlo区间为[−0.04210,−0.01724]。增加抽样不能解决这项稳定均值差距。[缓存归因](../paper_artifacts/legacy_joint_population_f_raw_source_predictive_attribution_v1/results.json)。

后续[来源水平／风险学习方向](../performance/manifests/legacy_joint_population_f_raw_source_level_risk_score_plan_v1.json)及联合M步现已执行，见[Stages211–219](orthogonal_meta_prior_stage211_219_20261007.md)。此前固定的函数水平参数及风险群体法则确有来源学习方向；保持原结构并重新条件化。布局5采样救援及付费预测扩张仍停止。

限制：一次M步尚非收敛联合训练，风险群体参数仍是旧经验估计。task0为反复开发数据；MC区间不覆盖任务／参数不确定性，布局5门未放宽。物理因子恢复、KG及独立决策收益仍未完成，完整模型HOLD。

六步成功记录33.99本地单核CPU秒、33.58MB JSON，四次M步优化；零新增数据、NPZ、仿真、验证、终端及服务器工作。首次重放误读旧协方差在预测前被复现检查截停，已修正并[记录](../paper_artifacts/legacy_joint_population_f_raw_source_full_state_prediction_v1/execution_incidents.json)，该尝试CPU未记录。

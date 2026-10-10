# Stages171–174：校准反转保留，高斯过滤误差不足以解释差距

14份外层缓存状态及12份参考通过检查。内部形状混合支持收窄，外层四个设计却均支持放宽，90%覆盖仅50%–83%；GP也覆盖不足。新预算相对保留混合的密度损失主要来自条件均值，四个外层均值偏差都向下，不能用统一比例修复。

固定五个内部任务、四次观测、八评分点及四来源风险法则，只换已有五来源均值训练包。十份新预测及十份参考通过：布局5平均MSE下降33.64%，布局6上升23.83%；两布局平均方差变化不足2%，仍支持收窄。来源训练包有影响，但不足以单独解释外层task1的反转；来源数量、task1加入及参数／映射变化仍未分开。

进一步对两布局×两设计×原生点／GP先验做四观测联合积分，避免中途高斯投影。八个单观测校验及后验矩检查通过；密度初始精度不足的两轮记录保留，加密最终8/8通过，最大逐点密度差9.61e-7。精确条件化MSE变化最大0.38%，协方差变化最大0.61%，不足以解释这些高斯对照的主要预测差距。

结果：[外层校准](../paper_artifacts/legacy_joint_population_f_outer_conditional_calibration_v1/results.json)、[匹配来源包](../paper_artifacts/legacy_joint_population_f_source_package_transfer_v1/results.json)、[精确过滤及加密](../paper_artifacts/legacy_joint_population_f_exact_raw_density_refinement_v2/results.json)。合计56.43单核CPU秒、2.93MB新分量，新增仿真、验证、终端及服务器任务均0。[完整截距混合的精确矩试点](../performance/manifests/legacy_joint_population_f_exact_mixture_moments_plan_v1.json)及后续联合来源法则诊断已完成，见[Stages175–178](orthogonal_meta_prior_stage175_178_20261007.md)。

限制：均为已见开发数据。未来方差条件化仅诊断；高斯对照的精确积分未覆盖完整来源方差混合，后者也不能仅凭矩结果主张密度或部署收益。候选停止状态、完整模型HOLD、论文及旧60/60证据保持。

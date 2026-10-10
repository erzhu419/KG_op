# Stages166–170：校准缺口定位，预算收缩未迁移

30份缓存预测的密度拆分通过：GP优势的95.5%–98.3%来自给定样本方差后的均值密度。结构混合的条件批均值方差约为GP的2.1／2.5倍，90%覆盖92.5%／100%，局部密度导数指向收窄。因子化、导数及混合加密检查通过；十个结构拟合的白噪声残差项均为0。

来源端±1%探针支持收缩。固定q=1，以五个内部留一任务、固定四次原始观测后的联合密度学习预算比例0.43364／0.57794；保留支持、低频谱及弱尾，重建共同均值／协方差与完整尾部截距混合。内部密度提高.00875／.00518，MSE却升0.59%／11.53%。优化、32／64节点检查及两份五来源法则均完成，未用task1值。

两布局×两冻结设计的四份外层task1预测及四份数值参考通过，复用20份对照。对保留混合，固定前缀MSE升0.82%／0.77%，来源选点MSE降1.15%／1.69%；四份transcript密度均下降，未来联合密度仅布局6来源选点改善。四个保留混合门及GP门均失败，停止该候选。全局预算收缩改善训练校准，未修复一致的外层预测收益。

结果：[归因](../paper_artifacts/legacy_joint_population_f_inner_raw_density_attribution_v1/results.json)、[校准](../paper_artifacts/legacy_joint_population_f_inner_conditional_calibration_v1/results.json)、[探针](../paper_artifacts/legacy_joint_population_f_source_shape_scale_direction_v1/results.json)、[学习](../paper_artifacts/legacy_joint_population_f_predictive_shape_scale_v1/results.json)、[试点](../paper_artifacts/legacy_joint_population_f_predictive_shape_scale_pilot_v1/results.json)。入口／参数在execution字段。合计96.88单核CPU秒、10.33MB新分量，新增仿真、验证、终端及服务器任务均0。后续[Stages171–174](orthogonal_meta_prior_stage171_174_20261007.md)已完成外层校准、匹配来源包及高斯先验的精确过滤对照。

限制：内部任务用于选择，task1是已见开发任务，均非独立确认。未来方差条件化及局部导数只作诊断。未评估来源6/7、目标或KG／部署；保留原生截距混合，完整模型HOLD，论文及旧60/60证据未改。

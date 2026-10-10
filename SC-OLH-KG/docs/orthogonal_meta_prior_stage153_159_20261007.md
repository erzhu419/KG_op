# Stages153–159：来源预测收益未通过原始观测试点

联合水平／形状方向未获两布局一致支持，停止且未拟合。归因显示布局6新增水平方差的密度惩罚超过残差改善；形状组方差分配在两布局均获支持。保留原生截距、低频谱、选中支持和1%弱尾，两个严格内部留一预测目标拟合均收敛到q=1，平均密度改善.02535／.05184。它是GP辅助的结构学习候选；完整来源训练似然略降，来源选择证据未被当作独立验证。

重新积分两个五来源截距方差分布，176／96分量规则通过加密。两套布局、固定四点和已冻结来源四点的四份原始观测预测，均先于评分保存；16份同设计对照复用。候选MSE比保留混合分别增加4.55%、4.71%、4.78%、.87%，四个门均失败，仍弱于同设计GP，候选停止。预测及原始密度的积分加密误差通过；失败是科学门失败。

同一批10个内部任务的已知噪声高斯对照进一步完成：适应四个已有三重复批次后，候选平均MSE下降3.37e-5／4.26e-5，未来八点密度提高.02921／.05980。因此不能把失败归因于适应本身；任务差异、单次原始观测／未知风险处理及方差混合仍未分开。[下一项严格四来源风险／原始观测对照](../performance/manifests/legacy_joint_population_f_inner_raw_contract_plan_v1.json)已登记，尚未实施；先重拟合排除查询任务的风险表示及η_f，再比较相同任务、四点、风险法则下的点先验和方差混合。四来源方差后验二阶矩发散，只验证实际预测所需的有限矩，不添加截断。

结果入口：[联合方向](../paper_artifacts/legacy_joint_population_f_inner_task_level_shape_v1/results.json)、[归因](../paper_artifacts/legacy_joint_population_f_inner_joint_attribution_v1/results.json)、[形状方向](../paper_artifacts/legacy_joint_population_f_inner_shape_allocation_v1/results.json)、[来源拟合](../paper_artifacts/legacy_joint_population_f_source_predictive_shape_v1/results.json)、[来源积分](../paper_artifacts/legacy_joint_population_f_predictive_shape_reference_v1/results.json)、[四份试点](../paper_artifacts/legacy_joint_population_f_predictive_shape_pilot_v1/results.json)、[同任务适应](../paper_artifacts/legacy_joint_population_f_inner_batch_adaptation_v1/results.json)。配置和入口见各结果execution字段；单线程串行运行。记录CPU合计4.18秒，预测分量约2.33MB，新增仿真、验证、终端及服务器任务均0。

后续[Stages160–165](orthogonal_meta_prior_stage160_165_20261007.md)已完成四来源风险重拟合、十折原始观测和固定查询／GP对照；内部平均收益保留，但查询位置敏感且联合原始密度仍弱于GP。

限制：内部任务参与q选择，适应对照是训练诊断；四个高斯批次对应12次已有调用，不是四次单回合观测。任务1试点仅重用已经打开的开发缓存，未评估来源6/7、目标、KG／部署或独立确认。完整模型HOLD，保留原生截距混合，论文和旧60/60证据未改。

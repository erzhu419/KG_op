# Stages147–152：选点有效，来源水平协方差缺口仍在

来源水平信息与10个严格内部留一检验已完成，60个拟合均收敛。共享均值不确定性已计入；内部水平误差标准化平方平均1.168／.724，不支持统一放大截距方差。GP整任务预测密度10/10折更好。保持原生形状边际的相关探针未获两布局一致支持，停止，未拟合q。

来源先验选出的四点设计预期残余方差降低7.94%／9.52%。16份新缓存预测、4份数值参考及8份旧预测复用完成；来源法则和选点先冻结，预测先于评分冻结，积分加密通过。结构混合MSE比原固定前缀下降15.75%／32.90%，未来原始密度增加.22155／.64798；但同设计、各自设计都弱于GP，来源先验优势门未通过。剩余同设计MSE差距83.3%／84.1%来自水平项。

后续[Stages153–159](orthogonal_meta_prior_stage153_159_20261007.md)已完成：联合方向停止；形状预算通过来源预测检验，但四个原始观测试点均未通过保留混合门。同任务高斯适应的收益仍存在，下一项转为严格四来源风险／原始观测合同对照。

记录CPU合计6.47秒，六阶段新增文件约5.43MB，其中预测分量2.36MB；新增仿真、验证、终端及服务器任务均0。入口和配置见结果execution字段，统一单线程运行。结果：[水平信息](../paper_artifacts/legacy_joint_population_f_level_information_v1/results.json)、[内部预测](../paper_artifacts/legacy_joint_population_f_inner_task_learning_v1/results.json)、[停止的形状方向](../paper_artifacts/legacy_joint_population_f_inner_shape_geometry_v1/results.json)、[选点信息](../paper_artifacts/legacy_joint_population_f_source_information_design_v1/results.json)、[预测对照](../paper_artifacts/legacy_joint_population_f_source_information_pilot_v1/results.json)、[误差归因](../paper_artifacts/legacy_joint_population_f_source_design_error_v1/results.json)。

限制：仍是开发任务1及重叠来源训练队列，未评估来源6/7、目标或独立确认，未运行KG／部署。不同付费前缀的transcript分数不用于跨设计排名；噪声标签归因不是oracle恢复。初始化收益未补成稳定结构先验优势，完整模型HOLD，论文及旧60/60主张未改。

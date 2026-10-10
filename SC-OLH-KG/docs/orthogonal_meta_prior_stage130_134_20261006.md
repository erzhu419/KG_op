# Stages130–134：联合来源似然改善，预测候选仍失败

Stage130完成12个共同／任务联合拟合和12个新条件截距方差积分，原优化合同全部收敛；原法则复现、梯度、稠密／分量条件化及全尾积分通过。来源总对数似然每折平均增加6.014／5.690，但共同相关强度p全为1、任务模板q全为0。Stage131检验原白色差异项，固定增量12折全降低来源似然，未增加拟合。

Stage132完成12个来源正权重积分规则；Stage133冻结12份同一8次前缀预测后评分，积分加密检查通过。候选虽优于原生／平谱，却比已有方差混合的MSE高.986%／1.449%，原始密度低.00657／.00604；比来源GP的MSE高32.09%／39.73%。task1两折密度差为−.22845／−.23319，候选停止。评分汇总中的对照字段不一致已修复，只重评冻结预测，未重跑预测或改变法则。

Stage134只用训练来源检验目标差异：共同联合法则的条件整任务密度诊断也优于原生，但q=.01探针仍在两布局平均上下降。仅布局5的无穷小导数反向，不足以宣称换学习目标即可修复。该诊断的超参数／经验中心使用全部五个来源，不是诚实的内部验证。

后续[任务形状组方差的来源不确定性诊断](../performance/manifests/legacy_joint_population_f_task_shape_uncertainty_plan_v1.json)已在[Stages135–137](orthogonal_meta_prior_stage135_137_20261006.md)完成：回到保留的原生来源法则，发现可观的形状方差不确定性，并建立了截距／形状联合来源积分。原字典、支持和频率保留；未继续采用失败的共同相关候选，未检验新的预测收益。

限制：全部为重复使用来源0–5的开发证据，完整模型仍HOLD，无独立部署优势。全部本地单线程，新增仿真／验证／终端调用及服务器任务／拉取均0；预测分量约2.61MB。预测写入过程约86秒墙钟，首次评分失败前CPU未保存；后续评分修复约3.94CPU秒，未把它作为整阶段CPU成本。

结果：[联合拟合](../paper_artifacts/legacy_joint_population_f_joint_common_task_geometry_v1/results.json)、[差异项诊断](../paper_artifacts/legacy_joint_population_f_joint_geometry_discrepancy_v1/results.json)、[来源积分规则](../paper_artifacts/legacy_joint_population_f_joint_geometry_quadrature_v1/results.json)、[匹配预测](../paper_artifacts/legacy_joint_population_f_joint_geometry_development_v1/results.json)、[来源任务目标诊断](../paper_artifacts/legacy_joint_population_f_joint_geometry_task_score_v1/results.json)。

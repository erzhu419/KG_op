# Stages144–146：批量更新完成，形状不确定性候选未过预测门

Stage144批量原始观测过滤与现有逐分量过滤一致；两个task1来源先验、更新诱导的f／eta相关及完整条件场检查最大差异约7.1e-15。16策略／4096分量的一次更新为.055–.076单核CPU秒。单独新进程内存测量峰值约214MiB；预测与加密阶段峰值约545MiB。

Stage145完成两task1折、原四次观测前缀的紧凑／完整规则比较，原生对照重现；预测均值、协方差、证据及原始密度加密均通过。联合截距／形状方差候选优于原生，但相对保留的截距混合MSE增加1.12%／.13%，原始密度仅增加.00219／.00474；相对sourceGP的MSE高15.8%／35.2%，原始密度增加.129／.277，原定义transcript score下降。候选未过预定门，停止扩展矩阵，原截距混合仍为参考。

Stage146用已冻结预测分解误差：观测均值误差的水平项占联合候选MSE的54.8%／79.8%。sourceGP在布局5的中心化形状误差反而更大，优势来自水平项；布局6优势也主要来自水平项。当前证据支持优先检查均值水平的信息传递，不继续扩充形状方差参数。

记录CPU合计约64.15秒，另有不足1秒的小内存测量；预测分量文件约47.9MB，仅保留12评分点。新增拟合、仿真、验证、终端及服务器工作均0。结果：[批量过滤](../paper_artifacts/legacy_joint_population_f_level_shape_vectorized_filter_v1/results.json)、[冻结预测比较](../paper_artifacts/legacy_joint_population_f_level_shape_development_v1/results.json)、[均值误差归因](../paper_artifacts/legacy_joint_population_f_level_shape_mean_error_v1/results.json)。下一项[来源均值水平信息诊断](../performance/manifests/legacy_joint_population_f_level_information_plan_v1.json)已登记，先查来源中心、水平／形状协方差及四点观测的信息覆盖。

后续[Stages147–152](orthogonal_meta_prior_stage147_152_20261007.md)已完成来源水平信息、严格内部来源预测、停止的形状相关探针和来源先验选点对照。选点降低MSE15.75%／32.90%，但同设计及各自设计均未优于GP；来源协方差学习仍有缺口。

限制：两布局属于同一已反复使用的开发任务；观测标签有原噪声，水平误差分解不是oracle恢复或独立显著性证据。来源6/7、目标和确认块未评估，完整模型仍HOLD。

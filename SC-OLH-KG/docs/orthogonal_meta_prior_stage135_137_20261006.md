# Stages135–137：任务方差不确定性与联合来源积分

Stage135完成12个原生f任务形状组方差的条件来源积分；新任务中心化形状方差在两布局平均增加19.50%／18.88%。task1的组方差后验均值为原生点估计的2.33／2.25倍，条件95%区间跨度约134／119倍。原生NLL、共同β/H、原频率和1%弱尾复现，稠密检查及全尾积分通过。这表明点估计遗漏了可观的条件来源不确定性。

Stage136建立独立截距v与选定形状组a的联合来源法则和Jeffreys测度sqrt(det I)dv da，12折可识别；固定来源分位网格显示似然不能完全分离，未拼接两个条件后验。Stage137完成12个正权重全尾联合积分；64／128节点每轴加密的最大归一化相对差、缩放矩差约1.05e-9／1.04e-9，均通过预定1e-8要求。联合积分后的形状方差仍平均增加19.47%／18.86%。加密是数值证据，不是解析误差界。

三步合计25.65单核CPU秒，新增拟合、仿真、验证、终端及服务器工作均0；来源规则约2.04MB。每折参考规则有16384分量；后续[Stages138–143](orthogonal_meta_prior_stage138_143_20261007.md)已完成正权重紧凑积分，12折均在2048–4096节点内达到原精度要求，下一项为批量过滤验证及小预算预测检验。

限制：当前只确认来源不确定性和联合积分的构造，尚未检验新的预测／决策收益；原截距混合仍是保留参考，完整模型HOLD。未读取查询结果、来源6/7或目标／确认块。

结果：[条件形状不确定性](../paper_artifacts/legacy_joint_population_f_task_shape_uncertainty_v1/results.json)、[联合来源几何](../paper_artifacts/legacy_joint_population_f_level_shape_joint_reference_v1/results.json)、[联合积分规则](../paper_artifacts/legacy_joint_population_f_level_shape_joint_quadrature_v1/results.json)。

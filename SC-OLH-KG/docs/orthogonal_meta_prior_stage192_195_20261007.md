# Stages192–195：来源覆盖试点停止，区分新任务与来源训练的噪声影响

两例来源留出的潜在方差积分通过偏差修正、度量及独立实现核对。总预测方差平均改变约1%–3%，结构／GP密度排名和方差调整方向不变；这不足以认定固定噪声近似导致失败。[来源积分](../paper_artifacts/legacy_joint_population_f_source_latent_noise_feasibility_v1/results.json)。

当前相关协方差法则的信息界核对通过。每任务12点的所选方向秩为12，24点来源并集可覆盖全部16个所选系数方向。固定法则下，理想来源系数观测最多降低约17%–23%的形状不确定性，水平不确定性仅0.33%–0.65%；这是方差界，不是预期MSE收益。[信息界](../paper_artifacts/legacy_joint_population_f_joint_source_information_ceiling_v1/results.json)。

据此完成两例等信息覆盖试点：结构及GP均从144增至288历史均值调用，风险法则保持原96调用，固定协方差／支持／谱，仅重新条件化缓存来源标签。原GP复现、旧标签保留及协方差信息顺序检查通过。结构来源留出MSE增加6.47%/21.98%，GP增加1.31%/13.89%；结构相对GP的差距均变差。按登记门停止，不扩大十折、原始观测或task1测试。[覆盖结果](../paper_artifacts/legacy_joint_population_f_crossed_source_reconditioning_v1/results.json)。

新增48来源均值的条件Gaussian密度与整批密度差一致。结构联合标准化二次误差约0.52/0.67；有效观测噪声信息占比为23.3%/30.0%，GP接近。训练条件下噪声影响明显高于新来源预测，因此新任务边际诊断不能证明来源训练稳健。[训练归因](../paper_artifacts/legacy_joint_population_f_source_incremental_calibration_v1/results.json)。

已知来源潜在方差诊断及联合更新试点已完成，结果见[Stages196–198](orthogonal_meta_prior_stage196_198_20261007.md)。复用现有记录，不增加任务／标签，不做外层预测；联合GP精度缺口仍保留。

限制：均为已见开发数据；来源Gaussian训练仍固定样本方差/3，风险来源后验仍有log-variance近似。信息界和条件误差不是独立优势或显著性证据。固定法则覆盖失败不能证明重新学习或增加独立任务无效；当前候选停止、原生截距混合作参考、完整模型HOLD，四项原始先验约束与论文主张未变。

四步共约0.66本地单核CPU秒、250KB JSON，零新优化、四份新增来源留出预测，无NPZ；新增仿真、验证、终端及服务器工作均0。各结果的`execution`记录入口与执行规则，下一步尚未执行。

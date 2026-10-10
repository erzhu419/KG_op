# Stages185–191：条件不确定性试点停止，下一步核对来源似然

完成七步，沿用原正交、稀疏、低阶、低频表示及正弱尾。来源谱积分16/32节点收敛；任务谱后验SD约0.66/0.56，但形状方差仅增加0.11%/0.38%，按登记规则未扩展原始观测试验。[谱结果](../paper_artifacts/legacy_joint_population_f_source_spectral_uncertainty_v1/results.json)。

新预测矩导数核对通过，任务形状幅度最影响协方差。固定来源模板单位尺度，正规半Cauchy幅度后验在全正区间收敛，来源尾秩48/47保证条件二阶矩有限；公共形状方差增加11.8%/12.6%。保留64/128全部分量、同四次观测及八评分点，两例预测／密度精度均通过。相对点谱法则，MSE增加2.15%/2.20%，批密度下降0.00314/0.00665，真实混合transcript也下降；按原筛选门停止，不做十折或task1扩展。[幅度积分](../paper_artifacts/legacy_joint_population_f_source_task_amplitude_uncertainty_v1/results.json)、[原始观测结果](../paper_artifacts/legacy_joint_population_f_inner_task_amplitude_raw_v1/results.json)。

复用十二份来源法则，删一来源jackknife形状散布约为五来源预测形状方差的31%/19%。但将十份四来源法则固定，再条件化同五来源数据后，形状散布仅剩原来的8.9%/1.6%；等观测法则散布只相当于完整预测形状方差的0.69%/0.075%。独立系数条件化及分解核对通过。大部分删一变化来自少一批观测，不能直接归因于模板估计不确定性。[等观测归因](../paper_artifacts/legacy_joint_population_f_source_learning_attribution_v1/results.json)。

本轮更正一个真实口径：`replicated_source_labels`的均值噪声为三次重复的样本方差/3，属于固定plug-in估计，非已知真实噪声。在Gaussian重复假设下，其样本方差相对SD为100%，逆方差期望不有限。两个来源布局的噪声最大／最小比为2141/285，最大单点占对角估计精度的69%/29%。当前Gaussian NLL仅在固定这些估计值后精确。[噪声口径结果](../paper_artifacts/legacy_joint_population_f_source_noise_contract_v1/results.json)。

后续来源潜在方差诊断、当前法则信息界、等信息覆盖试点及训练归因已完成，见[Stages192–195](orthogonal_meta_prior_stage192_195_20261007.md)。新来源边际积分影响小，固定法则覆盖试点停止；下一步检查已知来源训练条件下的方差不确定性。

限制：以上均为已见开发／来源数据；局部信息、jackknife及等观测散布不是后验方差修正，也尚未证明plug-in近似导致失败。共同／任务协方差会限制实际精度，对角权重不等于完整模型杠杆。幅度后验仍条件于固定其余参数／模板；点谱和幅度候选停止，保留原生截距混合作参考，完整模型HOLD，未更新论文或独立决策主张。

本轮约1.54本地单核CPU秒，零新优化、两份候选及两份数值参考预测；四个NPZ共249KB，七包共约1.34MB。新增仿真、验证、终端和服务器任务均0。七个入口及已执行规则见各结果的`execution`；下一步未执行。

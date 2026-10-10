# Stages199–204：原48条来源的原始观测后验打通，协方差学习仍需修正

函数头解析积分及Rao–Blackwell矩估计关闭新增来源对照的精度缺口，四组均通过原门槛，分段方差差降至2.34%–6.07%。[新增来源结果](../paper_artifacts/legacy_joint_population_f_collapsed_source_raw_update_v1/results.json)。

随后恢复原48条来源的未条件化群体先验。重放旧Gaussian条件更新，复现函数头与风险头的均值、协方差、NLL；GP的旧NLL存于归一化单位，按log scale Jacobian核对物理单位。新原始观测似然每条均值／样本方差只计一次，不额外套用旧32条Gaussian风险标签后验。[原来源合同](../paper_artifacts/legacy_joint_population_f_original_source_raw_contract_v1/results.json)。

原来源采样首次有一组GP风险ESS=78未达100，保留失败。用Gaussian方差参考分布并除去其近似似然，逐点密度等式误差小于3e−12；参考分布仅改善计算，未替换科学先验。[参考核对](../paper_artifacts/legacy_joint_population_f_variance_reference_v1/results.json)。四组随后均通过相同精度门，最小ESS199–446，分段方差差0.93%–8.62%。[完整原来源更新](../paper_artifacts/legacy_joint_population_f_corrected_reference_source_update_v1/results.json)。

结构来源点后验方差平均为固定噪声近似的2.02/1.73倍，但公共函数系数后验方差平均仅增加0.10%/1.88%，均值位移约原后验SD的0.063/0.071。仅修正后验计算对迁移公共函数的影响有限。

原始观测协方差梯度恒等式通过旧Gaussian梯度核对。结构旧投影梯度约1e−6，原始似然梯度达1e−3量级，两半段方向一致；GP也出现稳定的新学习方向。旧固定噪声下学习的参数仍需重新评估，不能把新后验接到旧协方差就视为训练修复完成。[来源学习方向](../paper_artifacts/legacy_joint_population_f_raw_source_covariance_score_v1/results.json)。

后续一次来源协方差M步、重新条件化及联合混合重建已完成，来源门均通过；完整状态下布局6精度门通过但预测劣势成立，见[Stages205–210](orthogonal_meta_prior_stage205_210_20261007.md)。原始来源学习打通不等于迁移优势成立。

限制：坐标、模板、风险群体参数仍是旧来源经验估计，未处理其参数不确定性；公共函数后验矩不等于可直接部署的Gaussian先验。来源梯度和数值PASS不能证明留出预测或独立决策收益；完整模型HOLD，旧候选失败保留。

六步记录的计算CPU合计约15.37本地单核秒、6.91MB JSON；零新增数据／协方差优化、NPZ、仿真、验证、终端及服务器工作。

# Stages109–110：联合共同尺度估计与开发检验

Stage109完成12个原稀疏支持内的共同组尺度／频率／相关强度拟合，全部按原停止规则收敛，p均为1，频率指数约1.29–1.63。来源总对数似然平均增加5.99／5.68。起点复原旧法则，梯度有限差分最大误差约1e-9，稠密条件化及支持、弱尾、截距、任务模型核对通过；全部法则先于开发观测保存。

Stage110沿用原付费0/1前缀、12评分点和eta_f先验。分配5的付费MSE降低.746%、原始批次密度增加.001134、序列密度增加.001521，三项通过；分配6分别降低.0269%、增加.002714、下降.000433，未通过序列门。因此联合共同尺度候选停止，未采用。分配6的两次付费总证据变化为-.030511，后续12点总Gaussian均值密度变化为+.024448，合计仍负；未放宽门槛。

持续问题仍是task1：两分配的付费前缀证据提高，但未观测点MSE变差，原始批次密度变化为-.13414／-.12410；其他5任务平均密度提高.02819／.02808。共同尺度重估未修复这一差异。后续[Stages111–113](orthogonal_meta_prior_stage111_113_20261006.md)已完成原生任务方差的条件后验积分及混合开发检验，两套原开发门均通过；本阶段失败的共同尺度法则仍未采用。下一步检查通过的f混合与现有g修复的完整来源合并门。

限制：这些重叠来源折是开发证据；未检验source6/7、目标、KG或独立确认。下一项固定经验均值、支持及其他参数，不等于完整总体不确定性。完整模型仍HOLD。两阶段共1.615本地单线程CPU秒、12个新f来源拟合；新g／风险／表示拟合、仿真／验证／终端调用、服务器派单及拉取均0。

入口：[来源拟合](../performance/check_legacy_f_common_scale_correlation.py)、[开发检验](../performance/check_legacy_f_common_scale_correlation_development.py)；结果：[Stage109](../paper_artifacts/legacy_joint_population_f_common_scale_correlation_v1/results.json)、[Stage110及失败分解](../paper_artifacts/legacy_joint_population_f_common_scale_correlation_development_v1/failure_attribution.json)。

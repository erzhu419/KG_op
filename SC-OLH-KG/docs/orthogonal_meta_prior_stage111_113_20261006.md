# Stages111–113：任务方差积分与混合预测通过开发门

Stage111完成原生f法则的14份条件来源后验积分（两个完整六来源、12个删除折）。固定原正交稀疏支持、共同先验、任务形状及弱尾，只积分任务截距方差v及其对应的共同beta(v)/H(v)。低秩似然、Fisher信息和条件化与稠密计算一致，全部积分达到原精度。删除task1时，分配6的点估计虽为0，后验均值仍为3.2884e-5，约47.14%的后验质量超过Stage105半对数似然区间上端；分配5后验均值为原点估计的2.555倍。这表明点估计遗漏了条件不确定性，未证明真实总体方差已恢复。

Stage112将连续后验转换为正权重来源积分规则。初始q端点数值错误改用等价u坐标解决；两种全区间Gaussian压缩未达到协方差精度，结果保留。最终使用原自适应子区间的正权重规则，14份全部通过，88–176个节点，无方差下限或尾部截断。共同协方差最大相对误差1.23e-8。规则先于付费观测保存。

Stage113继续执行完整混合预测，每个分量保留beta(v)/H(v)和新任务截距v；原eta_f先验不变，两次原有付费观测更新分量权重。12份预测及加密参考先于评分标签保存，均通过数值检查；付费证据最大加密差异3.52e-9，task1两折原始密度加密差异最多5.08e-9。沿用原开发门，两分配的付费MSE分别降低2.115%／4.125%，原始三重复批次密度增加.003178／.041696，序列密度增加.001418／.084155，**两套门均通过**。

收益集中在task1：其付费后原始密度提高.019132／.283862；其他五任务平均变化为-.000013／-.006738。不是每任务普遍改善。旧门使用Gaussian矩近似的均值／序列密度；另报真实混合均值密度得到的序列变化为+.008614／+.112964，未替换原评分定义。两个密度都保留，原始批次密度直接评分混合分布。

限制：来源0–5反复用于开发；Jeffreys测度及区间条件于固定经验参数和支持，付费更新仍采用原逐批Gaussian矩投影。完整六来源f／g合并门、目标KG和独立部署收益尚未验证，完整模型仍HOLD，旧60/60与论文主张不变。本次正式运行与保留的两次压缩尝试共156.130本地单线程CPU秒；初始失败启动另用约.444墙钟秒。新来源／风险／表示拟合、仿真／验证／终端调用、服务器任务及拉取均0。

后续[Stages114–115](orthogonal_meta_prior_stage114_115_20261006.md)已执行完整来源合并门：同216合同4/4通过，原144合同仍3/4，合并候选停止；本阶段两套开发通过结果保留。剩余分配6／task7的f均值误差包含整体水平与形状两部分，下一步只登记来源新增观测与法则重估的条件化归因。source6/7是已打开的开发记录。

入口：[条件来源积分](../performance/diagnose_legacy_f_task_level_reference_integration.py)、[正权重规则](../performance/prepare_legacy_f_task_level_reference_quadrature.py)、[混合开发检验](../performance/check_legacy_f_task_level_reference_development.py)。结果：[Stage111](../paper_artifacts/legacy_joint_population_f_task_level_reference_integration_v1/results.json)、[Stage112](../paper_artifacts/legacy_joint_population_f_task_level_reference_quadrature_v1/results.json)、[Stage113](../paper_artifacts/legacy_joint_population_f_task_level_reference_development_v1/results.json)。

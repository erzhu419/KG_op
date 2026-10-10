# Stages116–122：新任务观测有效，原生结构头仍未超过来源GP

完成嵌套来源归因、追加记录条件残差、训练折预算对照及信息上限计算。旧48条记录与新72条逐项匹配；原生共同均值、协方差和似然复原通过。共同曲线变化主要来自追加观测，而不是参数重估；追加24条记录的条件残差能量为7.35／10.36，最大边际标准化残差1.15／1.93，没有显示明显异常。

Stage118只用训练0–5的删除折，固定旧参数后把来源120调用增至180调用，付费后f MSE下降23.68%／9.80%。相对当前native180仍高26.56%／1.28%，原始批次密度低.11442／.01841，序列密度也下降；两套开发门均失败，固定旧参数候选停止。去掉task1后原始密度差仍为−.10030／−.01633。Stage113通过的开发证据保留。

当前冻结模型下，来源共同函数占新任务f总方差约22.6%／22.8%，任务自身差异占约77%。即便精确知道六个来源函数，总方差最多再降10.25%／10.46%；完全消除来源共同函数的不确定性也不能消除任务残差。两条无噪声付费观测的信息上限如下，实际带噪更新不能直接当作达到该上限：

| 当前native216 | 布局5 | 布局6 |
|---|---:|---:|
| 固定0/1消除形状方差 | 13.55% | 15.25% |
| 全2628对中最优两点消除形状方差 | 24.55% | 26.15% |
| 最优两点后仍留形状方差 | 75.45% | 73.85% |

Stage121已完成2／4／8次原始观测的训练曲线：公共几何固定顺序，评分点全部排除，24份新预测先保存再评分。4次未同时改善两布局的MSE；8次MSE下降34.10%／36.30%，原始密度增加.22774／.26661，训练信息信号通过。

Stage122同时补齐同来源180调用、同eta_f、同付费点及预算的平谱／来源RBF GP；24份先验和72份预测先冻结再评分。8次时原生结构f MSE为.00025946／.00032618，GP为.00019414／.00018358，前者高33.65%／77.67%；结构原始密度低.18137／.36508。去掉task1后密度差仍为−.15810／−.23623。结构8次优于平谱，但三个预算均未超过GP，不能由自身改善主张meta-prior优势。

后续来源方差混合、先验移植及协方差几何已完成，见[Stages123–126](orthogonal_meta_prior_stage123_126_20261006.md)。混合在4／8次均优于原生和平谱，但仍弱于GP；移植协方差的两个布局门通过，移植均值未通过。来源分析指向原任务支持遗漏及水平／形状相关结构，下一项已登记原字典内的稀疏任务协方差模板诊断。多观测的决策价值以后仍须计入完整验证预算。

限制：信息上限条件于经验参数及同一73点几何，不是预测误差下界或数据质量检验；来源折已反复用于开发。原两观测完整门仍HOLD，不能用新预算替换旧门，也未建立独立决策优势。

七项计算合计约6.832本地单线程CPU秒；新增拟合、仿真／验证／终端调用、服务器任务及拉取均0，未索引source6/7结果。结果：[来源归因](../paper_artifacts/legacy_joint_population_f_nested_source_conditioning_v1/results.json)、[条件残差](../paper_artifacts/legacy_joint_population_f_extra_source_innovations_v1/results.json)、[训练预算对照](../paper_artifacts/legacy_joint_population_f_nested_budget_development_v1/results.json)、[信息上限](../paper_artifacts/legacy_joint_population_f_information_ceiling_v1/results.json)、[两点几何](../paper_artifacts/legacy_joint_population_f_pair_geometry_v1/results.json)、[信息曲线](../paper_artifacts/legacy_joint_population_f_training_information_budget_v1/results.json)、[同预算强控制](../paper_artifacts/legacy_joint_population_f_training_budget_controls_v1/results.json)。

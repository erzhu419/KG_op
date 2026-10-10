# Stages102–104：共同先验相关性与完整来源门

Stage102投影的是来源GP的共同**先验**核，不是后验H；24个原来源法则和两种解析导数／有限差分均核对通过。原27列正交字典、稀疏支持、低频边际预算、共同截距和独立弱尾保持不变，24个固定1%相关探针均改善来源似然，方向信息量5.09–7.09。它来自同一来源数据，不能当作总体相关性检验。

Stage103据此先登记再拟合每折单个相关强度p，24次求解均收敛，完整来源法则先冻结再读付费观测。g相对Stage88中心化法则的付费MSE再降4.32%／.91%，原始／顺序密度均改善，两套开发门通过；去掉task1后收益仍保留。f两套均失败，task1原始密度变化−.12645／−.11783，不采用该f候选。

Stage104只接续通过开发门的g，在原完整六来源法则上拟合两次p，均收敛；f和风险身份先验保留。已有source6/7完整模型的同216调用门从Stage89的2/4变为**4/4**，g平均MSE相对Stage89再降5.13%／2.06%。原144调用门仍为**3/4**：分配6／task7的f原始密度−.146786，g+.114642，联合−.031591。因此完整修复仍HOLD，不进入目标／独立确认。

后续f任务水平方差与共享频率诊断已完成，见[Stages105–106](orthogonal_meta_prior_stage105_106_20261006.md)。原完整f截距方差为.000120421／.000103567，删除task1后为.0000378095／0；来源诊断未发现严重干扰方向混淆，未增加方差下限或分组频率拟合。下一步只登记先验压缩与来源条件化的顺序诊断，不继续扩展已失败的f共同相关候选。

限制：所有结果仍是同一73坐标银行及已打开来源检查上的开发证据，未建立跨控制几何泛化或稳定独立决策收益，历史60/60合同及论文主张不变。三阶段共26次单参数拟合（f12、g14）、约14.19本地单线程CPU秒；新增组／谱、风险／表示拟合、仿真／验证调用、终端积分、服务器派单与拉取均0。

入口：[来源诊断](../performance/diagnose_legacy_common_prior_geometry.py)、[开发对照](../performance/check_legacy_common_correlation.py)、[完整来源门](../performance/check_legacy_common_correlation_holdout.py)；结果分别位于`paper_artifacts/legacy_joint_population_common_prior_geometry_v1`、`legacy_joint_population_common_correlation_v1`、`legacy_joint_population_common_correlation_holdout_v1`。

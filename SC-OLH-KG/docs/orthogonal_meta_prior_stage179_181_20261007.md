# Stages179–181：正弱尾和低频谱恢复后，联合先验保留内部收益

40项来源协方差检查通过。原弱尾相对平滑任务形状方差约2.7–2.8倍，共同函数仅约1.7%；同一预算不匹配两层尺度。原稀疏支持上，即使保持组预算和相关系数，施加单一低频谱仍使中心化协方差改变27%–30%，因此联合学习尺度和谱。[Stage179结果](../paper_artifacts/legacy_joint_population_f_inner_joint_spectral_compatibility_v1/results.json)。

使用来源学到的相关模板和原组选组，固定每层正弱尾为其来源形状系数预算的1%，联合拟合共同／任务幅度及原谱指数。十个source4拟合收敛，梯度最大误差2.93e-9，PSD、稀疏支持、谱边际和独立条件化全部通过。共同谱指数0.87–1.52，任务谱指数均为4；来源NLL相对GP均降低，但没有据此判定收益。[Stage180结果](../paper_artifacts/legacy_joint_population_f_inner_joint_spectral_geometry_v1/results.json)。

在相同风险先验、固定四次观测和八评分点上，十份预测先冻结再评分。相对保留原生混合，布局5／6平均MSE下降38.43%／20.92%，批密度增加0.20082／0.25301，序列密度增加0.21892／0.25809。相对同源GP，平均MSE下降11.42%／22.45%，批密度增加0.08632／0.07821，序列密度增加0.04264／0.05345。[Stage181全部配对结果](../paper_artifacts/legacy_joint_population_f_inner_joint_spectral_geometry_raw_v1/results.json)。

[五来源、task1外层试点](../performance/manifests/legacy_joint_population_f_outer_joint_spectral_geometry_plan_v1.json)及损失归因已完成，见[Stages182–184](orthogonal_meta_prior_stage182_184_20261007.md)。外层未通过完整门，候选停止，未开启独立确认。

限制：仍为反复开发的有限字典／公共银行。相对GP逐任务MSE仅2/5、3/5改善，不能称普适优势；相关模板来自来源GP压缩，尚不支持物理因子恢复、银行外泛化或独立决策收益。新候选进入下一项开发试点，保留原生混合仍作参考，完整模型HOLD，论文与旧60/60未变。

三阶段合计1.21本地单核CPU秒，10次新来源拟合、10份新预测，NPZ仅18KB，结果包共约2.54MB；新增仿真、验证、终端及服务器工作均0。入口为`diagnose_legacy_f_inner_joint_spectral_compatibility.py`、`fit_legacy_f_inner_joint_spectral_geometry.py`、`check_legacy_f_inner_joint_spectral_geometry.py`及对应执行清单。

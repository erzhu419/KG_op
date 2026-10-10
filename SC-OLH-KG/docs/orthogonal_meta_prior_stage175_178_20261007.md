# Stages175–178：过滤误差很小，联合来源法则的弱尾尺度需修正

完整截距混合的四观测精确矩试点通过4/4积分检查。布局6、task1固定前缀的MSE仅变化0.0005%，协方差变化0.020%；均值仍低估0.03294，水平误差占MSE的79.8%。这一个案例的主要差距不来自过滤投影。见[Stage175结果](../paper_artifacts/legacy_joint_population_f_exact_mixture_moments_v1/results.json)。

十个诚实source4内部折的均值／协方差交换均未同时改善两布局的三个指标，不能孤立修复均值或协方差。随后把共同与任务GP先验同时压缩到原字典／原选组，再来源条件化；这与旧的后验投影不同。保留原弱尾时，稀疏联合方案MSE下降38.93%／3.84%，但布局6批密度下降0.04088、序列密度下降0.05047。[Stage176](../paper_artifacts/legacy_joint_population_f_inner_mean_covariance_v1/results.json)、[Stage177](../paper_artifacts/legacy_joint_population_f_inner_joint_compression_v1/results.json)保留全部任务与负结果。

只去掉额外弱尾，其他来源法则、风险、四次观测及八评分点均相同。联合稀疏方案的布局5／6 MSE相对原生点下降35.90%／23.87%，批密度提高0.15690／0.29249，序列密度提高0.20796／0.32685；每布局批密度5/5改善、MSE4/5改善。相对GP的平均MSE下降4.00%／24.61%，但逐任务MSE只有2/5、3/5改善。弱尾对这套联合法则的密度损失有明确影响；去尾并非所有任务／指标的普遍修复。见[Stage178配对结果](../paper_artifacts/legacy_joint_population_f_inner_joint_compression_tail_attribution_v1/results.json)。

[来源谱／弱尾兼容性检查](../performance/manifests/legacy_joint_population_f_inner_joint_spectral_compatibility_plan_v1.json)、正弱尾／低频谱联合来源拟合及匹配内部预测已完成，见[Stages179–181](orthogonal_meta_prior_stage179_181_20261007.md)。

限制：均为反复开发的数据；去尾方案未满足原正弱尾合同，也未施加原低频谱学习，不能替换保留混合或进入论文优势主张。完整模型仍HOLD，未评估source6/7、目标、KG或独立决策。Stage175未计算完整混合的未来批密度。

成功计算合计14.81本地单核CPU秒，60份新单分量NPZ共108KB，四包总计3.63MB；新增来源优化、仿真、验证、终端及服务器工作均0。Stage177在任何来源条件化／预测前有一次坐标名册检查失败，已按原名册修正、未改容差，记录在其`execution_incidents.json`。重放入口：`check_legacy_f_exact_mixture_moments.py`、`diagnose_legacy_f_inner_mean_covariance.py`、`diagnose_legacy_f_inner_joint_compression.py`；Stage178执行清单含独立配对汇总入口。

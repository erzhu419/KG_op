# f 的主要缺口指向来源估计，两个固定修复均未通过

Stage91的24份来源法则核对通过，系数相关方向信息较弱，未扩展模型。Stage92发现固定下限将f信号尺度抬高约37%，g未受影响；这不是已证实的实现错误。

Stage93缩小f弱尾后MSE下降1.19%/.71%，原始密度变化+.030173/−.002452，未通过训练门。损失集中在任务1，其他五折密度均改善；未删除任务1或重调比例。

Stage94分解确认划分6任务1的损失几乎全来自条件均值密度，方差边际基本未变。原f截距方差为零，水平残差z=5.668，缩尾后6.132，形状能量/维仅.580/.629。Stage95将已有水平/形状分离用于f，仍未通过密度/顺序密度门。

Stage96的36份投影/GP预测先冻结再评分。原27列字典保留GP收益；原选中组投影的f MSE比结构估计低21.36%/24.02%，密度高+.153930/+.167564。这些组能表示有用场，优先查来源估计；投影场未成为最终方法。

随后[登记的f来源初始化对照](../performance/manifests/legacy_joint_population_f_gp_profile_start_plan_v1.json)已在Stage97完成：用已有来源GP方差核生成初值，拟合原正交、稀疏、低频似然，以来源NLL选择，不移植GP后验。未取得实质改善，训练门未通过；后续归因与计划见[Stage97–101](orthogonal_meta_prior_stage97_101_20261006.md)。此前g任务6缺口仍在。

六份计划/执行清单位于`performance/manifests/legacy_joint_population_{coefficient_correlation_score,function_signal_floor_audit,data_scaled_f_tail,data_scaled_f_tail_endpoint,f_task_bank_centering,f_gp_orthogonal_projection}_*_v1.json`，其中记录入口和结果路径。成功计算合计8.03本地CPU秒、24份f拟合；Stage94提高独立积分核对的节点数后通过，失败时间另计、无重复拟合，见其`execution_incidents.json`。新仿真/验证调用、终端积分、服务器任务/拉取均0。

限制：仅使用反复开发的来源0–5，未评估来源6/7、目标或confirmation。未恢复稳定独立决策优势，论文主张与旧60/60合同未改。来源估计的具体病因尚待检验。

# 历史来源采样复用

九月九日前已有HVD形状混合、幅度校准和精确χ²结构更新。此次直接复用后续Stage203/206的方差参考、精确密度补偿、collapsed采样、共享系数投影和正混合保留。[历史出处及登记](../performance/manifests/historical_source_reconditioning_v1.json)。

16个冻结f/ηf法则全部重新条件化，10个通过原新链精度门槛，6个保留HOLD。16组新法则的密度／共享投影核对、固定512分量保留检查全部通过。六组均未通过半程方差稳定性检查，其中布局5/additive/joint/observed还出现Rhat=1.0605、ESS=91.56。[结果与失败条件](../paper_artifacts/historical_source_reconditioning_v1/summary.json)。旧重要性权重门槛仍失败，新链使用原Stage203/206门槛，两类精度分别记录。

保存样本诊断中，六组波动最大位置的方差约76%–100%来自给定η后的条件方差。[定位](../paper_artifacts/historical_source_reconditioning_v1/failed_precision_diagnostic.json)。本轮计算约22.5单核CPU秒，无新仿真、验证器、目标查询或服务器任务；固定η状态已保存，已通过案例无需重跑。

## 限制与下一步

来源计算达标尚不提供模型证据常数、物理因子恢复或部署收益；保存的抽取样本不能撤销完整链的失败门槛。经验地图／正则保持原样，g/ηg仍待完成。

下一步先适配旧`diagnose_legacy_f_inner_conditional_calibration.integrate`，检查六组方差方向的条件一维积分能否减少采样波动，再决定受限续链。随后补齐g/ηg，接回同一开发状态下的单新点／balanced4比较，继续复用旧KG和验证器。

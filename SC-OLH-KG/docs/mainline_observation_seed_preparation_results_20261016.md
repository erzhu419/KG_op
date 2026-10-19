# 观测种子复验接线完成

[冻结协议](../performance/manifests/mainline_observation_seed_replication_v1.json)的执行准备已通过，[验证记录](../paper_artifacts/mainline_observation_seed_preparation_v1/validation.json)。模型与统计规则保持冻结；已生成两组复验的6组搜索／终端子配置，覆盖24个任务—观测流组合、168条历史、216个终端格。

独立observation_seed只进入模拟观测及其日志；优化器种子保留原值。初始化复用10个物理策略，重新生成本组raw并跨臂共享；64邻居原样复制。新终端直接复用既有短名单变换与验证器，无需运行原终验。两组全部查询和终端输入冻结后才终验，全部终验后才读真值评分。

36项替身／保存数据测试通过：种子与初始化索引24项、固定策略库2项、新终端4项、整批顺序与失败保留4项、统计屏障与缺失处理2项。统计出口先按几何平均两组种子，再给12几何／3家族等权结果；结构失败保留未知及已付费用。

本次真实仿真、终验、真值调用、来源拟合、模型恢复和下载均0。当前验证的是执行接线，尚无新的部署收益结果。

下一步直接按登记方案启动两组复验：`python3 -m performance.run_mainline_observation_seed_replication run --workers 12`。保持online365／STOP362及物理上限76704，分别分析获证率与超最优可行初始化的质量增量是否跨观测流稳定；不追加模型或精度变体。

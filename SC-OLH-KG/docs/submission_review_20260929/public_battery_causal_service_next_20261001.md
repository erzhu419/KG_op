# 连续服务小试验与下一步

**两条保守接纳运行均完成 168 小时物理执行，但每条有 95 个容量短缺小时、58 次请求未完全交付；四条试验均未通过整窗服务。** 两条完整请求运行分别在第 430、445 分钟耗尽单元库存。首次服务短缺都在第 60 分钟、无激活时发生：补能 PN 下，满额双向请求的总绝对功率为 107–151 MW，超过固定 98 MW 上限。

7 项集成测试通过。已修复小时包络与逐分钟 SOC 累加顺序产生的约 1e-13 MWh 假中断，库存界、1e-8 容差及协议参数均未改变；首次输出另存 `numerical_repair_initial_run/`，固定四条复算后累计 8 条运行。无新数据请求、LP、优化器或认证调用，2025 未读取，source 比较 **HOLD**。

下一步冻结 `performance/manifests/public_battery_full_service_energy_bound_v1_20261001.json`：沿用现有 168 个请求时钟，推导满额双向义务对净 PN 的限制，再计算含转换损耗的总库存上界。先判断固定 49 MW 产品是否自身存在能量冲突；不缩减义务、不调目标、不扩大策略库。

## 局限

这是公开指令方向驱动的模拟产品，49 MW 并非历史业务合同。物理安全改善与服务成功不同；中断运行的交付量只记实际执行前缀。当前结果不能替代真实 BM 业务验证或原 95% 开发门槛。

结果：`paper_artifacts/public_battery_causal_service_pilot_v1_20261001/summary.json`、`hourly_service.json`、`nominations.json`、`first_failure_diagnosis.json`。复算：`python3 performance/inspect_public_battery_causal_service.py`；仅复算首次功率原因：`python3 performance/inspect_public_battery_service_failure.py`。

后续：总能量上界在第 500 分钟为负，已判定该满额模拟产品自身不可行。见 [条件性证明与下一步](public_battery_full_service_bound_next_20261001.md)。

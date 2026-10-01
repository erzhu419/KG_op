# 可用性与服务义务组件

已实现零基线组件：在请求出现前，按自身库存申报可持续功率，并约束两个单元总绝对功率。服务义务独立固定；申报不足、请求未满足分别保留，进出口缺额不能相抵。脉冲为一分钟爬坡、30 分钟保持、一分钟返回，每 MW 的能量为 31/60 MWh。

四条冻结轨迹均物理可行；半满库存两条履约成功，不均衡库存两条仍为服务失败，出口、进口缺额分别为 **24.3967、24.2297 MWh**。7 项针对性测试通过，包括零请求下的申报不足。历史评估、业务数据请求、LP、优化器调用均为零，2025 未读，source 比较 **HOLD**。

下一步已冻结 `performance/manifests/public_battery_availability_pending_preflight_v1_20260930.json`：六个非零基线/已锁定计划场景，明确绝对功率与相对基线的服务容量，区分没有指令与 0 MW 指令。先处理这些接口，再注册因果试跑。

## 局限

本组件仅覆盖零基线、单次脉冲，没有既有 BOA 承诺；四条轨迹不代表整周可靠性或真实 BM 业务效果。

结果：`paper_artifacts/public_battery_availability_contract_preflight_v1_20260930/summary.json`、`traces.json`。复算：`python3 performance/inspect_public_battery_availability.py`。

后续：六个非零基线场景已完成，两个履约成功，四个失败均保留；见 [锁定计划结果与下一步](public_battery_pending_availability_next_20261001.md)。

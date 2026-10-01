# 单元库存控制与下一步

已实现两个独立库存状态和各自提前一小时提交的补能计划，利用已收到指令投影，并共享整站功率余量。保持原来的 64 条候选、三个开发时段、共同半满初始状态和 95% 标准。四月/七月/十月可行候选为 **0/3/22**，三段共同可行候选为零，source 比较继续暂缓。

四月全部 4,672 个候选库窗口失败，首个失败区间的原因计数为：指令下库存越界 3,381、计划下库存越界 628、功率越界 663；首小时失败为零。原因标记可重叠。共 14,673 次开发窗口评估，七项针对性测试通过，最大单元能量账误差 7.94e-11 MWh。零新增数据请求、零算法优化器/认证调用，2025 未读取。

连续强制指令的必要容量检查覆盖全部 219 个窗口，未发现要求超过 98 MWh 的单元区间；四月两个单元的最大跨度为 48.42/31.96 MWh。这排除了该项简单不可行原因，尚未证明在固定初始状态、半小时计划及共享功率下存在可行解。

局限：每单元 98 MWh、效率、半满启动及绝对功率合计上限都是既定仿真假设，未观测实际单元 SOC；历史指令固定，开发窗口重叠。失败窗口没有整周经济结果，已完成窗口的条件平均成本不作为优化目标。

下一步已固定于 `performance/manifests/public_battery_unit_physical_oracle_v1_20260930.json`：对同一 219 个窗口求有完全未来信息的联合补能可行解，保留两个独立库存、首小时零计划和共享功率约束；每个解必须逐分钟物理回放，求解未决与回放失败分别保留。如果物理上界仍不足 95%，暂缓这个固定应用模型；否则继续处理因果控制和候选覆盖。仍只用缓存。

结果：`paper_artifacts/public_battery_unit_controller_development_v3_20260930/summary.json`、`failure_diagnostic.json`、`forced_run_summary.json`。复算：`python3 performance/inspect_public_battery_units.py` 和 `python3 performance/inspect_public_battery_unit_forced_runs.py`；测试：`python3 -m pytest -q tests/test_public_battery_units.py`。

联合物理上界已完成，结论与首次失效信息追溯计划见 [联合物理开发记录](public_battery_unit_feasibility_next_20260930.md)。

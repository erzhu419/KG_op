# 整站开发结果与下一步

已完成双 BM 单元指令重建、已收到未来指令的因果投影及配对开发实验。使用指令投影时，四月/七月/十月可行候选为 23/22/42；仅看待执行计划时为 17/17/38。三段共同可行候选分别为 10 和 6，均来自原来的 64 条候选库。每段保留全部 73 个窗口。

分拆检查发现共享库存掩盖了单元越界：上述 10 条共同可行策略，在每单元 98 MWh、分别半满启动的假设下，均不满足三段共同 95% 可行标准，四月全部 0/73。主结果 `summary.json` 已据此改为 `hold_source_comparison_for_unit_controller`，共享库存通过只保留为诊断。

共 29,346 次配对窗口评估、2,190 次分拆回放，七项测试通过，最大整站能量账误差 2.14e-10 MWh。新增第二单元 18 个小请求、880,509 字节；其他数据复用缓存。零算法优化器/认证调用，2025 未读取。

局限：公开资料给出整站 [98 MW/196 MWh](https://harmonyenergy.co.uk/projects/)，其中 [98 MW 是并网容量](https://www.harmonyenergy.co.uk/wp-content/uploads/2023/06/Harmony-Energy-final.pdf)。每单元 98 MWh、效率、初始库存和绝对功率合计上限均属仿真假设；未测得单元容量或 SOC。历史指令固定、开发窗口重叠。失败窗口没有整周成本；已完成窗口的平均成本不能直接充当带失败策略的优化目标。这些结果不证明 source 收益或实际电站可行性。

下一步协议已固定为 `performance/manifests/public_battery_unit_controller_development_v3_20260930.json`：分别维护两个单元库存，按各自已收到指令与库存提交补能计划，共享整站功率约束；保持候选库、样本、延迟、初始比例和服务标准，用缓存重跑。通过后再启动 source 档案试验。

结果：`paper_artifacts/public_battery_site_development_v2_20260930/summary.json`、`partition_summary.json`。离线分拆复算：`python3 performance/inspect_public_battery_partition.py`；测试：`python3 -m pytest -q tests/test_public_battery_site.py`。

单元控制的后续开发已完成，结果与联合物理上界计划见 [单元库存开发记录](public_battery_units_next_20260930.md)。

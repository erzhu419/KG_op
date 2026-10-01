# 联合物理上界与下一步

同一批 219 个开发窗口全部得到通过回放的联合补能计划：四月/七月/十月均为 **73/73**，没有求解未决或回放不符。两个独立库存、共同半满启动、首小时零计划及共享功率约束全部保留。已有 V3 成功窗口全部包含在物理可行集合内，四月 73 个物理可行窗口仍被整个候选库遗漏。

完成 219 次诊断 LP 求解和 219 次分钟轨迹回放，七项解析测试通过。最大库存边界偏差 7.68e-13 MWh、单元能量账误差 6.08e-12 MWh、执行功率偏差 1.43e-14 MW，均在既定 1e-8 容差内。零新增数据请求，零算法优化器/认证调用，2025 未读取。

判定：这批固定样本在当前仿真假设下存在物理可行解，后续应处理因果控制与候选覆盖。source 比较继续暂缓。

局限：每个窗口可以使用不同计划且预知全部指令，因此这些解没有证明存在一个共享的因果策略。容量分配、效率、库存初始化、固定历史指令及保守功率边界仍是原协议的仿真假设；没有实际电站性能或 source 收益结论。

下一步已固定为 `performance/manifests/public_battery_unit_failure_visibility_v1_20260930.json`：追溯四月全部 64 条候选及原定 50% 参考策略、全部 73 个窗口的首次失效；重建提交时仅凭已收到指令可预测的库存/功率路径，区分已知计划越界与后来指令造成的偏差。仅添加记录，控制规则不变；据此选择修正规则并先固定新协议。

该追溯现已完成：4,720 次对应提交路径安全、后来指令改变路径，25 次待执行小时已知越界；全部 4,745 次精确复现 V3。已固定 V4 因果备用规则，见 `public_battery_visibility_next_20260930.md`。

结果：`paper_artifacts/public_battery_unit_physical_oracle_v1_20260930/summary.json`、`windows.json`、`witnesses.npz`。复算：`python3 performance/inspect_public_battery_unit_feasibility.py`；测试：`python3 -m pytest -q tests/test_public_battery_unit_feasibility.py`。

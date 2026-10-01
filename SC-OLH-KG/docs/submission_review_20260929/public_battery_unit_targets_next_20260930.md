# V5 整周评估与下一步

V5 完成 **31,536 次新评估，复用 2,847 次 V3 结果**。154 条联合候选中，四月、七月、十月分别有 **0、6、26** 条达到原 95% 门槛，没有共同可行候选；四月全部 11,242 个候选窗口失败。七月的可行候选全为常数组合，十月为 25 个常数组合和 1 个函数配对。开发门槛及 source 比较均为 **HOLD**。

27 项针对性测试通过，最大新轨迹能量账误差 5.97e-11 MWh。所有新失败状态已压缩保留约 2.2 MiB，无需为后续诊断重跑矩阵。新增数据请求、算法优化器、认证调用均为零，2025 未读取。

下一步已冻结 `performance/manifests/public_battery_unit_targets_failure_visibility_v1_20260930.json`：检查原常数对 (0.35,0.75)、(0.45,0.75) 的全部 146 个四月失败上下文。两对均在 67 个窗口因第一单元库存失败、另 6 个因共享功率失败；提交时已知违约与后来指令改变仍待区分。直接使用保存的实际库存及计划，不新增整周评估或 LP。先定位因果投影或信息/承诺模型问题，再决定是否修控制器。

局限：局部动作可表达尚未带来整周可行性；候选数已从 64 增至 154，三季改善不能单独归因于目标解耦。重叠窗口非独立样本，历史 BOA 仍外生固定；失败窗口没有完整成本，条件成本不能用于方法排名。

结果：`paper_artifacts/public_battery_unit_controller_development_v5_20260930/summary.json`、`profiles.csv`、`windows.csv`、`failure_contexts.jsonl.gz`。复算：`python3 performance/inspect_public_battery_unit_targets.py`。

后续：上述可见性诊断已完成，全部 146 个上下文归为后续接收指令改变路径；结果与新的业务口径核查计划见 [诊断记录](public_battery_unit_targets_visibility_next_20260930.md)。

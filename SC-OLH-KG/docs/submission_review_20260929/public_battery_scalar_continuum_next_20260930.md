# 连续共用目标诊断与下一步

23 个参考状态的完整 θ∈[0,1] 动作族共分成 **161 条解析线段，全部不可行**，没有未决项。符号、功率截断和共享功率缩放的断点均纳入，未用目标网格近似。原控制函数与线段构造最大偏差 2.85e-14 MW，50% 参考动作精确复现；15 项针对性测试通过。复用上一轮的 50 个覆盖状态和 73 个负对照，没有重复其 LP。

因此，扩大候选库不能修复这些实际状态下的缺口；缺口属于当前共用目标比例的动作表达。下一步已冻结 `performance/manifests/public_battery_unit_target_decoupling_v1_20260930.json`：用已保存的 23 个物理可行后缀，反解两台电池各自的目标比例，再用原提交函数验证能否复现可行动作。只做一次确定性的反解，不增加数据、LP 或回放，也不搜索候选组合。全部可复现后再注册因果双单元控制器及候选映射，开始整周开发评估；此次尚未运行反解。

局限：23 个重叠窗口仍来自同一次提交事件；结论只针对这些参考状态，其他早期决策可能改变后来状态。物理可行后缀预知未来，不能证明在线策略成功或 source 有收益。source 比较保持 HOLD，2025 未读取，算法优化器及认证调用为零。

结果：`paper_artifacts/public_battery_unit_scalar_action_continuum_v1_20260930/summary.json`、`windows.csv`、`segments.csv`、`probes.json`。复算：`python3 performance/inspect_public_battery_scalar_continuum.py`。

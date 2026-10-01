# 候选动作覆盖结果与下一步

原 64 条候选在 73 个可恢复参考状态下完成 4,672 次动作评估：912 次允许可行后缀，3,760 次不可行。50 个状态有覆盖，23 个当前快照没有覆盖；73 个参考动作负对照全部不可行，没有未决项。精确重复动作复用 850 次，实际完成 3,895 个 LP、739 次独立见证回放。11 项针对性测试通过，包括禁止求解器改掉第三段待评估计划的测试。

这 23 个重叠窗口均指向四月十五日 23:00 UTC 的同一次提交时点，不能视作 23 次独立业务事件。它们有物理可行后缀，但当前候选没有可行新动作；因此现在应先区分有限候选采样缺口与共用目标比例的动作表达缺口。source 比较保持 HOLD，2025 未读取，新增数据请求及算法优化器、认证调用均为零。

下一步已冻结 `performance/manifests/public_battery_unit_scalar_action_continuum_v1_20260930.json`：在同样 23 个实际状态下，检查原 V3 控制规则的完整连续目标区间 θ∈[0,1]。按符号、功率截断和共享功率归一化的解析断点分成可达线段，每段用 LP 限制新动作并回放，最多 322 个 LP。连续区间可行则修候选覆盖；仍不可行则考虑分单元目标或更早的因果规划。此次只冻结，尚未运行。

局限：结果属于参考轨迹上的局部动作及预知未来的后缀；没有证明完整候选策略成功、在线问题无解或 source 有收益。

结果：`paper_artifacts/public_battery_unit_action_coverage_v1_20260930/summary.json`、`windows.csv`、`actions.csv`、`negative_controls.csv`、`probes.json`、`witnesses.npz`。复算：`python3 performance/inspect_public_battery_action_coverage.py`。见证文件约 534 KiB，仅保存半小时计划。

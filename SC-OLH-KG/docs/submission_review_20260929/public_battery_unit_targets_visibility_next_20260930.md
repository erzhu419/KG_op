# V5 指令可见性诊断与下一步

两对固定常数目标的 **146 个四月失败上下文全部复现保存的首次失效分段**，全部归为后续接收指令改变路径。提交时可见指令下，到首次失效时刻的计划前缀可行；没有已知前缀违约或未决记录。两个候选各涉及 9、7 个绝对失效分钟，合计去重为 10 个。

5 项针对性测试通过，覆盖已有分类以及失效分钟、零交叉分段错配。新增整周评估、LP、数据请求、算法优化器及认证调用均为零，2025 未读取。开发门槛和 source 比较继续 **HOLD**。

下一步已固定为 `performance/manifests/public_battery_information_contract_audit_v1_20260930.json`：用 Elexon/BSC/NESO 一手资料核查一小时 Gate Closure 与本模型固定补能计划的关系，以及历史 BOA 绝对功率和替换基线之间的依赖。先判断任务口径是否成立，再决定修正仿真假设或增加有业务依据的预测输入；尚不运行 V6 或扩大策略库。

## 局限

这只是两个开发阶段选定候选的失败解释，重叠窗口不是 146 次独立事件，也不证明所有因果策略不可行。历史 BOA 外生固定、均分容量及统一初始库存仍为仿真假设。

结果：`paper_artifacts/public_battery_unit_targets_failure_visibility_v1_20260930/summary.json`、`failures.csv`、`probes.json`、`examples.json`。离线复算：`python3 performance/inspect_public_battery_unit_targets_visibility.py`。

后续：业务口径核查已完成；一小时基线承诺有依据，历史 BOA 与模拟可用性之间的反馈缺失限制了业务解释。见 [核查结论与下一步](public_battery_information_contract_next_20260930.md)。

# 非零基线与锁定计划

六个冻结场景已完成：**5 个物理可行、2 个履约成功、4 个服务失败**，其中 1 个锁定计划本身不安全。三个功率受限场景只能保证每单元 36 MW 服务容量，保持原 49 MW 义务，未交付量各为 13.4333 MWh。没有指令和 0 MW 指令的进口能量分别为每单元 13、6.2833 MWh，正确区分。

8 项针对性测试通过，覆盖相对服务容量与绝对功率、库存极值及锁定尾段。160 次已知路径计算只用于求可申报范围；没有新增历史整周评估、数据请求、LP、优化器调用，2025 未读取，source 比较 **HOLD**。

下一步冻结 `performance/manifests/public_battery_received_availability_preflight_v1_20261001.json`：先处理已有 BOA 承诺的四个组件场景，再对先前保存的全部 146 个四月上下文做双向可用性诊断。使用原库存、锁定计划和已收到指令；不重跑控制器，不扩大候选库。

## 局限

当前六例尚无既有 BOA。共同缩放是保守的模拟接纳规则，不能据其申报不足断言所有在线策略不可行；这些结果不是整周可靠性或真实 BM 业务效果。

结果：`paper_artifacts/public_battery_availability_pending_preflight_v1_20260930/summary.json`、`traces.json`。复算：`python3 performance/inspect_public_battery_pending_availability.py`。

后续：四个已接收 BOA 场景及 146 个保存状态的双向探测已完成；292 次固定容量探测全部通过，但不代表整周可靠性。见 [接收承诺结果与下一步](public_battery_received_availability_next_20261001.md)。

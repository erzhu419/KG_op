# 已接收 BOA 可用性诊断与下一步

**4 个组件场景完成，146 个保存状态的 292 次双向探测均能交付每单元 49 MW 的固定脉冲。** 已接收 BOA 覆盖 PN，明确的 0 MW 指令只替换自身 32 分钟区间，到期恢复已有 BOA；未来接收指令不改变当前申报。三个组件场景物理及服务通过，原库存不足场景保留承诺失败。

14 项针对性测试通过，包括原 PN 回归和接收承诺的新测试。146 条参考轨迹与既有记录一致且安全；292 次探测无容量或交付短缺。886 次包络路径计算、296 次短轨迹复算；新增历史整周评估、数据请求、LP、优化器及认证调用为零，2025 未读取，source 比较 **HOLD**。

下一步已冻结 `performance/manifests/public_battery_causal_service_pilot_v1_20261001.json`，尚未执行：两对原固定目标、四月首个原始 168 小时窗口，比较保守接纳与完整请求执行，共四条连续运行。用缓存 BOA 的接收时间和方向构造每小时模拟请求，保持 49 MW 服务义务及一小时 PN 延迟；记录无激活时的可用性短缺、拒绝及未交付量。先检验重复请求下的连续运行，再决定扩大开发样本。

## 局限

保存状态只涉及 8 个绝对申报时刻、10 个原失效时刻，锁定 PN 接近零。这是单次脉冲承受能力，不是整周可靠性；此前的后续指令失效仍成立。下一试验的小时方向映射与请求形状属于模拟设定，不能识别真实 BM 反事实或据此判断论文已可投 OR。

结果：`paper_artifacts/public_battery_received_availability_preflight_v1_20261001/summary.json`、`probes.json`、`component_traces.json`。离线复算：`python3 performance/inspect_public_battery_received_availability.py`。

后续：四条连续试验已完成；两条保守接纳运行物理完成，但四条均未通过服务义务。见 [连续运行结果与下一步](public_battery_causal_service_next_20261001.md)。

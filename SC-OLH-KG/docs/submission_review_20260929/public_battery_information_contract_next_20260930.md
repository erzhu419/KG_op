# 储能信息与业务合同核查

**一小时延迟有依据，BOA 绝对功率解释正确；V1–V5 的固定指令压力测试成立，但不足以验证真实 BM 反事实运行。** 不取消延迟，不把 BOA 改成可直接叠加到新基线的增量，也不改写已有负结果。

2024-04-02 的 Grid Code BC2（Issue 6 Revision 22）§5.3 禁止 Gate Closure 后改报当期 PN/报价；§5.1 要求执行 FPN 及已确认指令；§4.1、§5.3.2、§7.2 同时规定调度使用基线、报价和更新的进出口可用限额。一小时定义另见 2023 年 CP1581 报告。这支持固定基线，也揭示了当前模拟可用性不会改变后续 BOA 的缺口。[BC2](https://www.neso.energy/document/33881/download)、[CP1581，2023-07-25，p.4](https://www.elexon.co.uk/bsc/documents/groups/isg/2023-meeting/268-august/isg268-04-cp1581-assessment-report/)

Elexon 的 Acceptance Data 定义明确给出从零点计量的 MW 水平。2024 年 OTF 材料说明储能 30 分钟规则在 3 月 11–25 日过渡，要求可支撑 30 分钟加两分钟爬坡；当前 MEL/MIL 指南还要求按能量状态重新申报。当前指南仅作机制补充，不冒充历史文件。[BOA 定义](https://www.elexon.co.uk/bsc/glossary/acceptance-data/)、[2024 年 OTF 材料](https://www.neso.energy/document/304631/download)、[当前申报指南](https://www.neso.energy/document/300231/download)

本地已缓存的 2024-04-15 单元资料含 PN 245、MEL 119、MIL 119 条，缺少申报时间和实测 SOC。它能支持回顾性描述，不能当作新策略的实时输入。判断：换预测信号不能自行修复上述调度依赖；未建立可用性与服务义务模型前，真实业务验证和 source 比较保持 **HOLD**。本次没有新业务数据请求、控制器评估、LP 或优化器调用，2025 确认样本未读。

下一步已冻结 `performance/manifests/public_battery_availability_contract_preflight_v1_20260930.json`：先定义因果申报、已承诺服务和请求未满足的记账，再验证两个 32 分钟小轨迹。缩小申报或拒绝指令不得自动算服务成功；验证完成前不运行 V6。

## 局限

公开单元 BOA 是历史基线与可用性下的调度结果；其他策略下会收到何种指令无法由这些记录单独识别。后续请求生成必须明确属于模拟假设。固定半小时基线、瞬时收到指令、均分容量及共享总绝对功率约束仍为模型设定。

结构化结果：`paper_artifacts/public_battery_information_contract_audit_v1_20260930/summary.json`。

后续：零基线可用性组件及四条脉冲复算已完成；两个库存不足场景仍为服务失败。见 [组件结果与下一步](public_battery_availability_next_20260930.md)。

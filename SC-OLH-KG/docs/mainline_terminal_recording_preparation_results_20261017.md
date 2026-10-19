# 最终观测池保存接口与复验登记完成

原任务`t169515`已在node002完成，未重复提交。12项替身检查全部通过，确认记录开启／no-op时真实终端路径的选择、随机状态、评分调用次数与预算一致；旧／新输入传递、native隔离、名单变换及异常记录均通过。接口只复制现成分数、排序和名单过程，没有补预测。

随后[正式登记](../performance/manifests/mainline_terminal_recording_seed_replication_v1.json)完成：固定7001–7024的24个d=1000几何、两组新观测／终验流、全部9参照与每套352上限，生成两子流及6组家族计划；96个新seed无冲突，81项冻结资产存在。主比较仍为旧端AFTER−POINT，同几何内先平均两流。状态为`registered_not_started`，模型、搜索与终端冻结，新增仿真、拟合、真值及后验恢复均0；无失败或重跑。

检查与登记子进程CPU分别4.463336／1.353728秒，整job含开销5.954213秒，两者不重复相加；内部登记0.058892秒也已包含。历史费用及本轮明细见[执行账](../paper_artifacts/mainline_terminal_recording_preparation_v1/scheduler_execution.json)。大表、raw和模型仍留服务器。

下一步按[冻结协议](mainline_terminal_recording_seed_replication_protocol_20261017.md)接通两流共同封存后的完整池评分及旧端AFTER−POINT主汇总，再执行登记复验。当前只完成接线验证和登记，没有新增收益证据；新记录不能补回旧流缺失的分数，原接口未提供的方差分解仍不额外预测。

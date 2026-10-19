# 原始 d=1000 两轮闭环（2026-10-15）

已完成原始 nominal InventorySupplyChain 的四个独立观测种子、四臂比较。每臂362次调用：两次初始化，online两轮各一次观测，剩余358次终验；stop使用360次终验。全部73点可查询和推荐，固定order3近似，未做单query精算。

跨维度接线正确：直接恢复原始73条d1000策略，其降维索引与d50库一致；反向插值只能还原54/73条，未采用。真实d1000库的63/73条表示有变化，已重算mean4/risk10；响应单位映射为1，log-variance偏移为0。风险encoder从原32条来源记录确定性重建，旧73点坐标误差为0；来源law和GP未重新优化。

| 方法 | 平均实际安全收益 | 通过终验 |
|---|---:|---:|
| 结构prior，停止 | 0.807974 | 4/4 |
| 结构prior，两轮 | 0.808345 | 4/4 |
| 同源GP，停止 | 0.603160 | 3/4 |
| 同源GP，两轮 | 0.804786 | 4/4 |

收益为 `certified × truly_feasible × clip(1−f/1.5,0,1)`，误认证0。结构online相对自身停止+0.000371（1胜2负1平），相对GP online +0.003559。

**预测传播尚未变成结构模型的正向决策证据。** 所有更新都影响剩余未采样策略（69或70点）；结构模型8次更新中，未采样f/g均值MSE各改善6次。每轮付费观测后、终验与真值评分前，另冻结“未采样边际保持观测前值”的同预算终端诊断：结构模型仅1/8轮改变推荐（71→4），条件收益−0.000493；其余推荐相同。GP有1/8轮正向转换（12→未采样39），条件收益+0.117392，实际该seed online/stop收益差为+0.000381。

**局限。** 四个seed是同一个nominal问题的重复，结构收益差的bootstrap区间均含0。GP停止臂一次终验拒绝影响实际均值，其条件期望收益为0.626873，online为0.746055。结构模型log-variance误差平均增加，GP均值预测绝对误差仍较小。边际干预只是本轮终端诊断，不是第五个优化器；逐轮收益不能相加。当前是原73点库上的结构化高维迁移，不是V51动态搜索全协议复现。

**后续动态扩展已完成：** [73→89→104共享候选结果](mainline_d1000_dynamic_results_20261015.md)。本库仅9个真实可行策略，最优71的质量为0.808592；结构停止臂距上限平均仅0.000617。这是当时扩展共同候选集、保持来源模型和预算冻结的依据。

证据：[协议](../performance/manifests/mainline_d1000_two_round_v1.json)、[维度审计](../paper_artifacts/mainline_d1000_two_round_v1/dimension_audit.json)、[完整结果](../paper_artifacts/mainline_d1000_two_round_v1/results.json)、[库内上限](../paper_artifacts/mainline_d1000_two_round_v1/post_result_bank_ceiling.json)。复用[原runner](../performance/run_mainline_two_round73.py)，指定新 `--plan`，依次 `prepare`、8个 `case --case-index 0..7`、`assemble`、`verify`。

成本：5768次物理调用（初始化8、搜索16、终验5744），5792次逻辑调用；主实验CPU52.11秒，11个阶段/分片无失败。16推荐先全部冻结，再终验，最后评分。维度审计曾误填layout导致KeyError，已修复并保留失败账（wall0.496秒、CPU未知、0仿真）；成功审计CPU0.362秒。4项测试与2个subtest通过，旧账/HOLD保留，无服务器下载。

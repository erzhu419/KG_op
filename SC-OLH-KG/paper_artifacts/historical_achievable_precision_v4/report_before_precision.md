# 完整正混合批次KG接续

已把历史批次均值／样本方差似然接入正混合原始后验、幻想缓存和分头终端缓存。来源身份由两头共同更新；所有正来源分量保留；新批次只扣其自身次数。73策略、365动作的原生source／GP四次批次验收通过，付费数2→6、剩余验证预算360→356。source缓存与实际更新的身份权重、投影终端值差均为0；GP终端值差2.22e−16。

主要效率瓶颈是`raw_eta_proposal`先按积分节点复制整张146×146协方差，再截取2×2块：小动作集剖析中占约98%时间。改为先取局部块再选来源分量，GP完整批次验收188.84→4.53 CPU秒，约41.7倍；两个积分阶数下的全部动作、奖励和成本完全一致。source同阶验收74.21 CPU秒完成，旧实现三次限时记录仍保留。高斯条件计算也复用了每来源分量一次分解的历史实现，并经独立完整原始后验采样核对。

| 数值验收 | 最大动作差 | 结果 |
|---|---:|---|
| source任务05／布局5停止规则，12／14阶 | 0.00009013 | PASS |
| source任务02／布局6停止规则，14／16阶 | 0.00009381 | PASS |
| GP任务05／布局6停止规则，14／16阶 | 0.00003757 | PASS |
| source任务02／布局6原始批次终端，14／16阶 | 0.00015039 | HOLD |
| GP任务02／布局6原始批次终端，14／16阶 | 0.00016241 | HOLD |

source原始批次的似然归一化诊断误差0.00015825，GP为0.00000767；动作差／归一化门槛均为0.0001。48项不同的相关测试通过；全部数值验收与小动作集剖析累计约37.7单核CPU分钟，包含三次限时；新增仿真、来源拟合和实际验证调用均为0。仅下载约2.9KB压缩汇总。

补充验收已完成：分头归因定位到风险积分，旧全动作值复现误差1.11e−16。复用16阶缓存、按策略分六片后，source／GP原始终端最大动作差5.05e−5／2.85e−5，归一化误差2.11e−5／4.44e−7；实际推荐最大动作差2.42e−5／2.69e−5，全部365动作通过原门。两者均与原始后验选择相同，分别直接验证71／22，选择损失为0；另补GP任务02停止规则，差1.93e−5。20项相关测试通过，无新增仿真、来源拟合或验证，仅下载4.5KB。限时、断言失败及重复运行均留档。

已接上实际推荐的可实现KG：先用原生付费矩滤波选择动作，再按下一步原始后验计值，保留负切换收益；固定动作控制变量使用扣费后的验证预算。任务02／布局6对两个学习器冻结同一查询63，比较STOP、单次和四次采样；所有365动作及联合身份均保留。96个分头状态覆盖384个联合未来，四个扣费基线全部通过。source单次目标头进一步复用旧8点缓存扩到16／32点，净值0.01779363→0.01777939→0.01777376，最后差5.64e−6；风险预测4／8点原先差4.35e−5。目标头预测积分的坐标门通过，无需继续增加这一轴的采样点。

| 同预算开发预测比较 | 相对STOP净值 | 完整KG验收 |
|---|---:|---|
| source，单次，目标32／风险8点 | +0.01777376 | HOLD |
| source，四次，双头8点 | −0.00005397 | HOLD |
| same-source GP，单次，双头8点 | −0.00004191 | HOLD |
| same-source GP，四次，双头8点 | −0.00016874 | HOLD |

分头归因进一步锁定终端风险积分：source单次原始／原生最大动作差约0.00619／0.01197，目标头贡献仅1.43e−5／6.22e−7。目标头归一化最坏1.21e−4，风险头5.20e−4，仍超过原1e−4门。22个调度任务均完成，累计29.1单核CPU分钟；27项相关测试通过。新增仿真、来源拟合及部署验证为0；只下载约7.4KB压缩结果和小型归因表。

限制：当前覆盖六个已开结果开发任务中的四个停止精度案例、两个固定四次批次锚点，以及一个目标状态／冻结查询的预测先导。上述净值是模型内开发估计；四项完整KG验收仍为HOLD，不能冻结策略或据此主张独立部署优势。相邻积分规则通过是数值诊断，不是严格误差证明。矩投影仍改变动作估值；完整查询和批次数选择、其余目标状态以及独立部署收益仍开放。

下一步复用现有分头缓存，只提升风险头原始／原生终端积分；目标头仅补归一化未过门的记录。完整门通过后，再扩展同合同的冻结查询比较，冻结算法并进入新任务确认。当前不扩展物理实验。

[可实现KG先导](../paper_artifacts/historical_achievable_batch_kg_v1/phase_results.json)、[目标头扩展](../paper_artifacts/historical_achievable_objective_v3/phase_results.json)、[终端误差归因](../paper_artifacts/historical_achievable_objective_v3/terminal_attribution.json)、[阶段运行账](../paper_artifacts/historical_achievable_batch_kg_v1/phase_account.json)。服务器入口`python -m performance.historical_achievable_batch_kg collect`及`python -m performance.refine_historical_achievable_objective --collect`；逐任务命令保存在本阶段各目录的`*_specs.jsonl`。

[本轮汇总](../paper_artifacts/historical_batch_head_precision_v1/phase_results.json)、[本轮测试](../paper_artifacts/historical_batch_head_precision_v1/test_results.json)、[运行账](../paper_artifacts/historical_batch_head_precision_v1/execution_account.json)。全量动作表留在服务器，逐片重放命令保存在各阶段`scheduler_specs.jsonl`。

[汇总和历史限时](../paper_artifacts/historical_full_bank_batch_kg_v1/outcome_summary.json)、[测试](../paper_artifacts/historical_full_bank_batch_kg_v1/test_results.json)、[运行记录](../paper_artifacts/historical_full_bank_batch_kg_v1/execution_account.json)。服务器隔离目录中`python -m performance.historical_full_bank_batch_kg collect`重放汇总；逐案例命令保存在各阶段的`scheduler_specs.jsonl`，全量动作表保留在服务器。

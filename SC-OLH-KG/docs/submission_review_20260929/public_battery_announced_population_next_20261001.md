# 早停验证与人口筛查进度

8 条保存案例的早停等价验证和 4 项接口测试已通过；首批续跑新增控制器调用为 0。后续沿用已保存的验证结果。

每个候选覆盖三季各 73 个窗口及两种规则。表中仅计新单元，经济重跑另记。

| 原索引 | 新增单元 | 新单元成功 | 新单元物理分钟调用 | 状态 |
| --- | ---: | ---: | ---: | --- |
| [0,4) | 1,752 | 0 | 995,053 | complete |
| [4,8) | 1,752 | 146 | 7,306,763 | complete |
| [8,12) | 1,752 | 292 | 6,679,450 | complete |
| [12,16) | 1,752 | 0 | 2,796,679 | complete |
| [16,20) | 1,752 | 438 | 11,131,977 | complete |
| [20,24) | 1,752 | 0 | 927,901 | complete |
| [24,28) | 1,752 | 146 | 7,249,163 | complete |
| [28,32) | 1,752 | 292 | 6,771,533 | complete |
| [32,36) | 1,752 | 0 | 3,057,369 | complete |
| [36,40) | 1,314 | 438 | 9,290,943 | complete |
| [40,44) | 1,752 | 0 | 2,023,032 | complete |
| [44,157) | 49,032 | 19,974 | 324,621,349 | complete |

累计覆盖 **68,766/68,766** 个唯一单元，已完整记账的控制器执行 **68,454 次**：67,866 条新路径、8 条等价重跑、580 条经济重跑。完整成本已保存 **22,310 条**，旧成功成本缺口 **0 条**。新失败上下文 **46,140 条**。

已完整评估候选中，有 **28 个联合候选**在三季、两种规则下均达到每季至少 70/73。达标集合见 [可靠性汇总](../../paper_artifacts/public_battery_announced_population_screen_v1_20261001/population_reliability.json)。

完整人口与旧成功成本已全部评估，下一步依据保存的达标候选集合评估 source-archive 研究。 进度与调用见 [人口汇总](../../paper_artifacts/public_battery_announced_population_screen_v1_20261001/summary.json)和 [产物记账](../../paper_artifacts/public_battery_announced_population_screen_v1_20261001/artifact_accounting.json)。

剩余 46,731 条单元通过 **4,096 个逻辑 shard / 144 个四核 scheduler 任务**完成，整批从首任务启动到 scheduler 确认结束约 **9.8 分钟**。实际任务分布在 node001、node002、node004、node005、node006；node003 当时被 scheduler 判为无可用容量。5 项并行执行测试通过。达标集合包含 **26 个常数联合候选、2 个函数联合候选**。调度记录见 [dispatch_layout.json](../../paper_artifacts/public_battery_announced_population_screen_v1_20261001/parallel/dispatch_layout.json)。

## 局限

开发窗口重叠，服务通知、重置库存及 MID 代理均为研究设定；失败路径的完整成本未定义。调度迁移时撤下的旧进程可能有 0–2 条未完成路径，其实际调用数未知，未混入已完成记录；见 execution_interruptions.json。source 比较仍为 **HOLD**，2025 未读取；本阶段不构成 OR 投稿就绪结论。

完整人口 JSONL 记录与 baseline/shard 日志保存在服务器；本地已同步小汇总，本地大 JSONL 文件仍是迁移时的快照。数据位置见 [scheduler_dispatch.json](../../paper_artifacts/public_battery_announced_population_screen_v1_20261001/scheduler_dispatch.json)。

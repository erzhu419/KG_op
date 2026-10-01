# 三季预告任务筛查与下一步

**冻结的开发门槛通过：0.45/0.75 组合在三个样本、两种交付规则下均为 73/73 成功。** 所有成功窗口均安全执行 170 小时，并满额履行该窗口的全部请求。

| 源样本 | 0.35 / 0.75 | 0.45 / 0.75 |
| --- | --- | --- |
| April 2024 | 73/73 | 73/73 |
| July 2024 | 0/73 | 73/73 |
| October 2024 | 0/73 | 73/73 |

表中两种交付规则的成功窗口数相同。本轮新执行 872 条、复用 4 条，无实现重跑；五项新接口测试通过，没有重跑原控制器测试。876 个冻结单元及全部 292 个失败上下文已完整落盘，产物约 3.3 MB。新执行调用数见 [汇总](../../paper_artifacts/public_battery_announced_development_screen_v1_20261001/summary.json)。

July 首个失败请求前，两单元库存仅为 0.786/3.406 MWh，零 PN 下满额出口分别需要 27.518 MWh；前一提名的最低共享功率需求达 116.178 MW，超过 98 MW。保守规则保住物理边界但未交付完整请求，满额规则在第 841 分钟中断。诊断只使用保存状态与标量算术。

下一阶段已冻结 [库与目标函数预检](../../performance/manifests/public_battery_announced_library_objective_preflight_v1_20261001.json)：保留原 154 个联合候选编码，核对公告任务下的预测发布时间；先查原始缓存，必要时小规模补齐各样本末尾两小时的价格与预测记录；从已保存的四条 April 试验重建经济记账。该预检不新增物理试验，完成后再登记动态目标接入组件。

该预检随后完成：边界和历史预测可见性通过，四条保存试验经济记账一致；见 [完成结果与动态目标下一阶段](public_battery_announced_library_next_20261001.md)。

## 局限

这通过的是三个固定、重叠窗口样本的有限开发标准，仍是分析者定义的预告仿真。筛查时发现的价格/预测边界缺口已在后续预检中单独补齐。完整函数库、source 收益、独立确认和 OR 投稿就绪性均未建立；source 比较仍为 **HOLD**，2025 未读取。

复算入口：`python3 performance/inspect_public_battery_announced_development.py --output paper_artifacts/announced_development_reproduction`。旧结果及全部负结果保留。

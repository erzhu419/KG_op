# 完整预告控制器结果与下一步

**四条 170 小时试验全部安全完成，各自满额交付 95/95 次请求。** 两组原目标库存下，保守接纳与直接满额执行结果相同；请求误差、规划失败、已知承诺失败及物理中断均为零。每条试验有 75 个空闲/启动小时，它们没有计入交付成功数。

| 原目标库存比例 | 两种交付规则 | 全程最低单元库存 | 最小容量余量 |
| --- | --- | --- | --- |
| 0.35 / 0.75 | 均 95/95、10,200 分钟安全 | 2.531918 MWh | 1.388871 MWh |
| 0.45 / 0.75 | 均 95/95、10,200 分钟安全 | 6.950394 MWh | 1.388871 MWh |

控制器补齐整点零 PN、空公告、已接纳脉冲剩余段及小时起点安全累计。22 个组件通过，36 个满额脉冲可交付；库存投影最大误差 3.55e-14 MWh。组件另计 22 条执行轨迹、66 条安全轨迹、4,784 次标量投影。46 项相关测试通过，单独计 4,537 条物理轨迹、3,554 次标量投影及 21 次短仿真。

四条正式试验共计 22,416 条声明轨迹、2,028 条提名安全轨迹、142,496 次标量投影和 40,800 次分钟物理检查，保存 680 条小时记录及 1,352 条提名记录。实现重跑为零。全部输入来自本地缓存，2025 未读取。

下一阶段已冻结 [三季完整窗口开发筛查协议](../../performance/manifests/public_battery_announced_development_screen_v1_20261001.json)：保持这两组目标与控制规则不变，覆盖原 April/July/October 2024 样本各 73 个完整源窗口。报告 876 条试验，其中复用本轮 4 条、新执行 872 条；同一目标组合须在三个样本、两种交付规则下分别达到 70/73。当前尚未执行扩展筛查。

## 局限

四条试验只有一个源窗口和两组目标，不能当作四个独立可靠性样本。履行义务针对已公布方向的请求；其他方向的声明容量受限不等于当前请求失败。预告任务仍是分析者定义的公开数据仿真，改变了原任务的信息条件；它没有建立历史 BM 反事实、95% 可靠性、source 收益或 OR 投稿就绪性。Source 比较仍为 **HOLD**，原任务与全部负结果保留。

结果：[汇总](../../paper_artifacts/public_battery_announced_controller_pilot_v1_20261001/summary.json)。复算应使用单独输出目录，避免覆盖本轮记录：

```bash
python3 performance/test_public_battery_announced_controller.py --output paper_artifacts/announced_controller_reproduction
python3 performance/inspect_public_battery_announced_controller.py --output paper_artifacts/announced_controller_reproduction
```

后续完成：[三季完整窗口筛查结果](public_battery_announced_development_next_20261001.md)。

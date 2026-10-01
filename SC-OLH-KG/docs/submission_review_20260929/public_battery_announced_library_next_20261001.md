# 库与经济目标预检完成

原 154 个联合候选保持原编码、顺序和生成参数；0.35/0.75、0.45/0.75 在全部 1000 个节点上与筛查目标完全一致。三季 219 个窗口的 74,022 次有效提名均通过历史预测发布时间检查，沿用 2022 年拟合，未读取末端无效提名的目标。

原始缓存各自包含末尾第一个半小时。通过 6 次小请求补齐其余 9 个时段的预测、历史发布时间和 APXMIDP，响应合计 3,198 字节；边界单独保存，原 CSV 和全部窗口保留。

四条 April 起点试验的 680 条小时记录完成精确脉冲、恢复 PN 和零点交叉记账。流量与库存最大复算误差 **8.46×10⁻¹¹ MWh**，低于 10⁻⁸ 容差。同一目标组合在两种规则下路径和成本完全相同：

| 目标组合 | 计量现金 GBP | 终端库存调整 GBP | 总成本 GBP |
| --- | ---: | ---: | ---: |
| 0.35 / 0.75 | 27,857.21 | 1,018.41 | 28,875.61 |
| 0.45 / 0.75 | 27,775.42 | 463.06 | 28,238.48 |

八项接口与记账测试通过。首次测试发现读取 V5 时未解析其 V3 基础参数，修正读取入口后通过；追加测试确认独立复算目录复用边界缓存、无需再次公共请求。测试记录均保留。本轮新增控制器、提名、声明和轨迹调用均为零。

下一阶段已冻结 [动态目标接入与有限试验](../../performance/manifests/public_battery_announced_dynamic_targets_pilot_v1_20261001.json)：先做 6 个短组件，验证常量等价、双通道和目标时钟；再用原首个函数组合及其交换组合，在三季首尾窗口、两种规则下执行 24 条试验。保留失败，不改种子、目标、物理参数或请求分母。结果见 [本轮汇总](../../paper_artifacts/public_battery_announced_library_objective_preflight_v1_20261001/summary.json)。

动态目标阶段随后完成，保留 24 条负结果，并修复实际小时起点导致的数值判定差异；见 [动态结果与人口筛查下一阶段](public_battery_announced_dynamic_next_20261001.md)。

## 局限

经济结果只覆盖同一个 April 窗口，表中差额不能推广为策略收益。函数库的动态执行、完整人口可靠性、source 收益与 OR 投稿就绪性仍未建立；source 比较保持 **HOLD**，2025 未读取。

复算：`python3 -m unittest tests.test_public_battery_announced_library -v`；随后运行 `python3 performance/inspect_public_battery_announced_library.py --output paper_artifacts/announced_library_reproduction`。该入口自动复用本轮边界缓存。

# 公开储能应用：业务口径与指令预检

已确定参考资产和服务输入，并完成离线指令重建；完整方法比较尚未启动，当前仍不建议直接投 OR。

采用 49 MW/98 MWh、两小时参考电池，有 Jamesfield 项目依据；工作负荷来自 `E_PILLB-1` 历史调度，其样本 MEL 达到 49 MW。[业主资料](https://harmonyenergy.co.uk/projects/)

重建 BOA 绝对功率，按接收时间处理修订，段内线性插值。全国负荷预测仅作信号，旧 11% 阈值不作合同要求。[Elexon 定义](https://www.elexon.co.uk/bsc/glossary/acceptance-data/)、[历史结算说明 §3.2、§3.5](https://assets.elexon.co.uk/wp-content/uploads/sites/11/2018/11/28160209/SAA_SD_v26.2_P369.pdf)

2024-04-15 有 22 条接受、59 段指令。正确峰值 49 MW，错误叠加 111 MW；出口 61.27、进口 7.60 MWh。单程效率 0.92、间隙空闲的诊断允许初始库存 [59.60,98.00] MWh，统一 49 MWh 起点需要补能。

2023 年七、十月来源样本分别有 12、6 条接受；仅检查这两天。后续候选十天档案未拉取，2025 确认数据未读取；零优化器/认证调用。

重建程序 `performance/inspect_public_battery_dispatch.py` 可离线重算；三项测试通过，覆盖真实修订与未来信息隔离、重复功率及空白时段、双向能量与段内库存极值。

## 局限

98 MWh 并非该单元实测容量；实际 SOC 与其他交易未观测。无指令时功率保留为空值，空闲只用于能量诊断。固定 BOA 是外生工作负荷假设；该协议冻结资产和输入接口，尚无策略认证、市场利润或部署证据。

下一步：建立因果补能控制，明确购电时点与期末库存估值，再检查既定 64 个库内策略的可行性分布；保留已有负结果。

后续：[补能控制预检与下一步](public_battery_controller_next_20260930.md) 已完成上述开发计算；完整比较仍暂缓，先定位四月不可行性。

协议：`performance/manifests/public_battery_dispatch_preflight_20260930.json`；结果：`paper_artifacts/public_battery_application_spec_20260930/dispatch_preflight_summary.json`。Contego 空样本与别名检查保留。

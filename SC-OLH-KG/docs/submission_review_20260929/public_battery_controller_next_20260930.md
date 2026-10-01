# 补能控制预检与下一步

已完成每半小时决策、提前一小时提交的因果补能控制；收到 BOA 后优先执行指令。成本为指数价格下的净购电，加上按末期价格估值的初末库存差。[交付时序依据](https://www.elexon.co.uk/bsc/glossary/gate-closure/)、[价格指数说明](https://www.elexon.co.uk/bsc/article/market-index-definition-statement-review-2023/)

固定库内 64 个策略：四月 0、七月 12、十月 33 个达到 95% 窗口成功率；每段均为 73 个完整七天窗口。四月所有库内策略在所有窗口均有指令供能缺口；18 个窗口另有 50 MW 请求，使 49 MW 资产成功率上限仅 75.34%。完整比较暂缓。

七项测试通过；开发计算的最大能量账误差 8.94e-11 MWh。额外首小时诊断在三段各 73 个窗口上均无失败，保留现有启动设定。2023 两个固定来源日的最高 MEL 为 37、49 MW，属于动态可用功率，不据此更改额定容量。累计 14,892 次诊断窗口计算；新增有效 API 响应 1.224 MB，零优化器/认证调用，未读取 2025 确认数据。

局限：参考资产可用容量、固定外生 BOA 和其他交易缺席是仿真假设；MID 能量成本尚未包含实际 BM 结算、费用和退化。重叠开发窗口的可行率不代表独立认证或 source 优势。

下一步：在既定参数下计算离线物理可行性上界，区分四月的任务不可行与控制规则/候选库覆盖不足，再确定资产与工作负荷的对应关系。完整三季结果和 V1 协议保留；不删四月、不根据可行率调容量。

后续：[离线物理上界](public_battery_feasibility_next_20260930.md) 已完成；四月 55/73 个、七月与十月各 73/73 个窗口有经过回放的可行解，下一步统一整站资产与双单元输入。

协议：`performance/manifests/public_battery_controller_development_v1_20260930.json`；结果：`paper_artifacts/public_battery_controller_development_v1_20260930/summary.json`、`windows.csv`、`startup_diagnostic.json`。重算：`python3 performance/inspect_public_battery_controller.py`；测试：`python3 -m pytest -q tests/test_public_battery_dispatch.py tests/test_public_battery_controller.py`。

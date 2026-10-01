# 独立公开数据预检与下一步

结论：Elexon 数据可用，但当前候选库在开发样本上缺乏认证率鉴别力，暂缓完整比较。该预检没有建立 source 收益，当前仍不建议直接投 OR。

三个年份的一天样本成功对齐 144 个半小时。继而固定检查 2024 年一、四、七、十月各十天，复用 2022 年校准样本与原 V5 物理参数：每季 64/64 库内策略在 73/73 七天窗口上安全。四、十月的失败仅来自额外的零响应常数参照。共 19,564 次诊断窗口计算，零优化器/终端认证调用；季节预检的有效 API 响应合计 1.36 MB，三天接口样本另为 0.08 MB，未读取 2025 年。

已修复预检程序的英国结算日/UTC 日期映射错误；夏令时样本完整对齐，无插值。价格请求按接口七天上限拆分。程序继承现有共享功率引擎，原论文模型与实验参数保持原样。[官方 API](https://data.elexon.co.uk/swagger/v1/swagger.json)

PERFORM 的预测由研究模型生成，每个地区只有第二年提供预测，不适合直接补齐三段历史预测期。[数据方说明](https://github.com/PERFORM-Forecasts/documentation)

局限：这是单一国家市场、短期源校准和四段重叠窗口的仿真预检。保留旧参数时，对应假设资产约 14.03 GWh/14.03 GW；尚无该规模的业务依据，也没有全年覆盖或实际部署证据。

下一步：先从公开业务资料确定资产规模、调度任务与服务约束，再冻结独立确认协议。保留本次结果；新的应用定义依据业务资料确定。已有负结果和稿件的结论边界继续保留。

后续进展：[公开储能业务口径与指令预检](public_battery_dispatch_20260930.md) 已确定参考资产并完成时序重建；下一步接入补能控制与成本模型。

数据与判定：`paper_artifacts/independent_data_feasibility_20260930/feasibility_decision.json`。离线重算：`python3 performance/analyze_independent_energy_feasibility.py`；三个预检协议均位于 `performance/manifests/independent_energy_*preflight_20260930.json`。

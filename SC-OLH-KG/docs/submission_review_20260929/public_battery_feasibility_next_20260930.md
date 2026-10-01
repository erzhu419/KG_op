# 离线物理上界与下一步

已完成全部 219 个开发窗口：四月 55/73 个物理可行，18 个功率超限；七月、十月各 73/73 个可行。没有额外的库存/半小时计划不可行窗口，也没有求解未决项。四月 55 个物理可行窗口均被现有候选库遗漏。

采用每半小时一个有符号库存变化率的线性规划，保留共同 49 MWh 初始库存、首小时零计划、49 MW/98 MWh 边界及逐分钟双向能量。201 次 LP 求解均得到通过物理回放的补能计划，最大未供能量 4.22e-12 MWh，最大能量账误差 9.69e-11 MWh。此前成功的控制窗口全部包含在上界可行集合中；新增五项测试通过。

判定：当前参考资产与单元请求存在功率匹配问题；其余窗口在当前库存模型下存在离线可行解，现有补能规则/候选库和信息限制仍需区分。完整比较继续暂缓。零新增数据请求、零算法优化器/认证调用，2025 确认数据未读取。

局限：上界预先知道全部指令，可以为每个窗口选择不同计划；它没有证明存在一个共享的因果策略，也不构成 source 收益或独立认证证据。

下一步：将资产与请求统一到公开的 Pillswood 整站 98 MW/196 MWh 和两个 BM 单元，核实整站功率及双向流量的对应关系；再把已收到的未来指令纳入因果库存投影。[资产依据](https://harmonyenergy.co.uk/projects/)

协议：`performance/manifests/public_battery_physical_oracle_v1_20260930.json`；结果与可行计划：`paper_artifacts/public_battery_physical_oracle_v1_20260930/summary.json`、`windows.json`、`witnesses.npz`。离线重算：`python3 performance/inspect_public_battery_feasibility.py`；测试：`python3 -m pytest -q tests/test_public_battery_feasibility.py`。

后续整站开发与单元库存诊断已完成，结论和下一步见 [整站开发记录](public_battery_site_next_20260930.md)。

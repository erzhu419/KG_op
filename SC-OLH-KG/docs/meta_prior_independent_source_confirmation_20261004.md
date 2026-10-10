# 独立来源与目标的早期结构确认

协议在新观测前冻结为4个来源规律（93008–93011）和4个目标规律（95021–95024）。每个来源规律在公开四场景的16个站点上各取12次重复，共768次来源调用；每个目标规律只取共同8个初始策略，各12次重复，共384次目标调用。三种均值先验仍是来源 billing、同来源全字典和 target-only 全字典；风险后验固定为来源 factor-family 分布或同来源 pooled scale。没有运行搜索或验证器，oracle 只在推荐之后评分。

独立批次重复了来源结构的预测优势。以来源学习风险上下文为例，八个“任务×预算”重复记录的平均 f RMSE 为 source billing 0.0577、同来源全字典 0.5850、target-only 0.4557；g RMSE 为 0.0860、0.2681、0.2264。风险上下文只改变风险预测误差，没有改变三种均值先验的主要推荐模式。

决策证据仍然为空：四个目标规律在三个已冻结预算 1.0、1.17547896、1.18858716 下，最佳真实预计验证收益最高也只有约 `6.4e-11`，所有推荐收益均接近零。因此这批实验确认的是来源结构的早期预测可重复性，不能把预算1附近的无可行任务写成结构先验没有作用，也不能把预测优势写成稳定部署优势。

逐条记录和新观测保存在 `paper_artifacts/inventory_physical_risk_20261004/independent_source_confirmation.json`，冻结协议为 `performance/inventory_independent_source_confirmation_protocol.json`，实现为 `performance/run_inventory_independent_source_confirmation.py`。下一步应停止继续扩大这个 Inventory 预算族，转向成熟方法的信息合同与跨应用确认；若没有能产生可区分可行集的任务，投稿中只能主张“早期结构预测收益”，不能主张风险 meta-prior 的部署收益。

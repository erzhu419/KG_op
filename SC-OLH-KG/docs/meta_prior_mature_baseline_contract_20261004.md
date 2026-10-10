# 成熟方法信息合同审计

本阶段只审计 HyperBO 成熟对照的可执行合同，复用已经保存的 Inventory 来源记录和8个目标初始观测，没有新的模拟调用，也没有使用目标真值。

固定合同包括：三条可观测标量头（目标均值、约束均值、log 样本方差）、来源任务的公开轨迹坐标、共同8点目标适配、来源先验冻结、目标后验条件化、潜在函数协方差用于 KG，批次标签噪声只在观测协方差中加入一次。审计同时运行了本地 paper-core HyperBO 重实现，三个头都返回有限均值和 `8×8` 潜在协方差，目标适配点数均为8。

官方仓库存在，commit 为 `e720fc1`，与适配器固定版本一致。依赖已安装到隔离的 `/tmp/kgop-hyperbo-venv`，没有改项目 Python。官方入口完成了1步来源训练、8点目标条件化和完整协方差预测，均值与 `8×8` 潜在协方差有限；适配器记录了 `tensorflow_io_gfile_only` 兼容 shim、源端标量噪声和在线只更新 target posterior。

因此成熟对照的科学状态分成两层：**信息合同和官方单任务接线已经闭合；Inventory 新闭环因目标没有可行候选，尚未形成部署排名证据**。smoke 不能列入 OR 论文的性能表，也不能替代同预算、同初始设计的完整配对；paper-core 仍只作为接口和数值控制。当前主方法的来源结构预测证据、旧 native-transfer 矩阵与新物理风险闭环分开保留。

逐项审计结果在 `paper_artifacts/inventory_meta_prior_interface_20261004/mature_baseline_contract_audit.json`，实现为 `performance/audit_mature_baseline_contract.py`。同预算、同8点初始设计的单任务配对协议已经冻结并执行；在新目标合同具备非空可行集之前，不扩展旧的118个实验单元。

按该合同复用独立来源/目标批次又完成了 HyperBO 的初始后验对照：4个目标规律、3个已冻结预算，共12个回放单元，来源训练步数固定为200，新增来源和目标调用均为0。HyperBO 的三头平均 RMSE 为 `f=0.7706, g=0.3649, eta=1.1251`；同批 source billing 结构的对应值为 `0.0577, 0.0860, 0.2572`。这个结果说明当前公开 Inventory 坐标上的物理 billing 结构比通用 RBF 来源先验更贴合，但风险头的模型形式不同，不能把 RMSE 差异直接写成最终部署优势。四个目标在三个预算下的真实收益仍接近零，故没有形成决策胜负证据。

逐目标记录在 `paper_artifacts/inventory_physical_risk_20261004/official_hyperbo_initial.json`，协议为 `performance/inventory_official_hyperbo_initial_protocol.json`，实现为 `performance/run_inventory_official_hyperbo_initial.py`。该回放仍只是预测比较；当前批次不支持扩展矩阵。

该完整闭环已按新的预冻结目标95025完成，结果见 `docs/meta_prior_hyperbo_full_loop_20261004.md`。官方 HyperBO 的端到端接线通过，但目标96候选全部不可行，因而没有成熟方法的部署排名证据。

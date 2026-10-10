# 可比较独立部署合同 V1：开发阶段结果

## 合同目的

原 Inventory 合同的约束价格范围为 `[0.6,1.4]`。在该范围和 `cost_budget=1.0` 下，未使用的 100 个 law seed 中有 90 个在 96 个公开候选内完全不可行，不能用来比较表示或部署决策。V1 保持需求、库存动力学、96 个公开策略和预算 `1.0`，把公开约束价格生成范围固定为 `[0.4,0.9]`，并保留原合同作为负对照。

开发块使用 seed `95425,95426,95427,95428,95430,95431,95432,95434`。确认块 `95433--95440` 在本阶段没有读取真值，后续只有开发门通过后才可使用。边界 seed `95429` 保留为开发负例，不因只有三个可行策略而放宽门槛。

合同和入口：

- [合同 JSON](../performance/meta_prior_feasible_deployment_contract.json)
- [真值预检](../performance/preflight_meta_prior_feasible_deployment.py)
- [来源档案生成器](../performance/generate_feasible_inventory_source_archive.py)
- [来源档案](../paper_artifacts/meta_prior_feasible_deployment_v1/source_archive.json)

## 已完成的本地证据

来源档案使用 8 个来源任务、16 个公开策略点、每点 12 次重复，共 1536 次来源调用。没有目标观测和验证调用。开发块的真值预检通过：

| 指标 | 结果 |
|---|---:|
| 可行候选数 | 19--75 |
| 不可行候选数 | 21--77 |
| 可行候选内目标跨度 | 0.053--0.130 |
| 开发门失败数 | 0 |
| 确认块读取 | 否 |

随后在开发块上只使用共同的 8 个初始策略点，每点 12 次重复，共 768 次目标观测，比较固定 policy 坐标和 Inventory observable exposure。没有进行 24 点搜索、KG 或独立验证。

| 初始后验指标 | 固定 policy 坐标 | observable exposure |
|---|---:|---:|
| `f` 留出 RMSE | 0.594 | 0.270 |
| `g` 留出 RMSE | 0.204 | 0.141 |
| `g` 排序 Spearman | 0.119 | 0.624 |
| 初始推荐真实可行任务数 | 7/8 | 7/8 |
| 初始推荐平均真实奖励 | 0.762 | 0.761 |

结果说明 observable exposure 已经改善目标均值预测和约束排序，但尚未产生可见的初始部署收益；它只能进入下一阶段的候选表示，不应直接写成已完成的轨迹表示主张。完整逐任务结果保存在 [representation_initial.json](../paper_artifacts/meta_prior_feasible_deployment_v1/representation_initial.json)。

## 24 点开发闭环

在同一开发块上运行 24 个单元：8 个任务、3 个 arm，所有 arm 共享初始八点、目标噪声流和独立验证器。三臂分别是固定 policy 坐标+pooled risk、observable exposure+pooled risk、observable exposure+learned factor-family risk。确认 seed 块仍未读取。

| arm | 平均初始奖励 | 平均最终真实奖励 | 平均已验证奖励 | 认证 | 错误认证 |
|---|---:|---:|---:|---:|---:|
| policy + pooled risk | 0.8455 | 0.8538 | 0.8603 | 8/8 | 0 |
| observable + pooled risk | 0.8455 | 0.8613 | 0.8615 | 8/8 | 0 |
| observable + learned risk | 0.8455 | 0.8608 | 0.8611 | 8/8 | 0 |

Observable exposure 相对固定 policy arm 在 3/8 任务上提高最终真实奖励，0/8 任务退步，平均差 `+0.0075`；但只有 3/8 任务出现正的搜索增量，未达到预先固定的“至少一半任务正增量”晋级门。学习风险相对 pooled risk 的平均差为 `−0.00045`，仅 seed 95434 退步 `−0.0036`，没有形成独立风险部署收益。

因此本开发合同完成了“可行任务和可比较验证器”的修复，却没有完成轨迹表示或风险因子的投稿级主张。完整闭环记录在 [closed_loop.json](../paper_artifacts/meta_prior_feasible_deployment_v1/closed_loop.json)。按停止规则，确认块 `95433--95440` 暂不读取；下一步只做失败归因和模型合同修订，不扩大任务矩阵。

## 来源结构归因对照

为补齐方法合同中尚未单独完成的均值结构对照，另行冻结了
[source_structure_arm_v1](../performance/meta_prior_feasible_source_arm_v1.json)。它复用同一来源档案、observable exposure、8 个初始点、24 点搜索、keyed 目标噪声流和 928 次验证，只比较来源选择结构与同来源全字典。结果见
[source_structure_arm.json](../paper_artifacts/meta_prior_feasible_deployment_v1/source_structure_arm.json)：来源结构平均最终真实奖励 `0.86127`，全字典 `0.85582`；8 个任务中 3 个提高、1 个下降、4 个相同，均 8/8 认证且无错误认证。该结果支持“来源结构有可测均值归因”，但没有通过至少一半任务正差的稳定部署门，因此仍属于开发证据，不读取确认块，也不升级为普遍部署主张。

## 下一道门

在读取确认块前，原计划需要完成两件事；当前第一项已经完成但未通过晋级门，因此不启动确认：

1. 固定 observable exposure 作为候选表示，比较 24 点搜索与初始推荐的增量，并使用独立验证器记录真实奖励、认证和 adaptive loss；已完成，晋级门失败。
2. 在相同目标观测上加入 pooled-risk 对照，检查风险结构是否改善 `eta` 校准和最终排序，而不是只报告风险 RMSE；已完成，未观察到独立收益。

因此当前动作是停止确认块，转向表示/风险失败归因；若不改变可识别性合同，不继续换 encoder 或扩大矩阵。

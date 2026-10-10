# 来源结构与同来源全字典对照

## 目的

可行部署 V1 原有三臂闭环比较了固定 policy 坐标和 observable exposure，尚未把 M1 的来源选择结构与同一来源档案的全字典基线放在同一合同中。本阶段冻结一个独立对照，只改变均值先验：

- `source_selected_structure`：来源任务交叉验证选出的函数支持；
- `same_source_full_dictionary`：相同来源观测、相同 observable exposure、同一 68 函数字典和 pooled factor-family risk。

两臂共享 8 个初始点、24 点目标搜索、928 次独立验证、目标 law seed、keyed 观测噪声流和验证器。确认块没有读取，也没有新增来源调用；原 V1 24-cell 文件保持不变。

合同和入口：

- [冻结合同](../performance/meta_prior_feasible_source_arm_v1.json)
- [运行入口](../performance/run_meta_prior_feasible_source_arm_v1.py)
- [逐任务结果](../paper_artifacts/meta_prior_feasible_deployment_v1/source_structure_arm.json)

## 结果

| arm | 任务数 | 平均最终真实奖励 | 认证 | 错误认证 |
|---|---:|---:|---:|---:|
| source-selected structure | 8 | 0.86127 | 8/8 | 0 |
| same-source full dictionary | 8 | 0.85582 | 8/8 | 0 |

逐任务比较中，来源结构相对全字典在 3/8 个任务提高最终真实奖励，在 1/8 个任务下降，在 4/8 个任务选择相同。最大正差为 seed 95431 的 `+0.04332`，最大负差为 seed 95428 的 `-0.00125`；平均差为 `+0.00545`。认证结果相同，因此差异来自推荐而不是验证通过率。

## 解释与限制

这项对照补上了“来源选择结构是否比同来源泛化字典更有用”的归因缺口，并支持在可行开发任务上保留结构先验。它没有通过稳定部署门：正差任务少于一半，且只覆盖同一开发块的 8 个目标。结果不能外推到确认 seed、其他应用或普遍风险因子恢复；`decision_performance_established` 保持为 `false`。

风险 arm 仍固定为 pooled factor-family，因而该对照只识别均值结构，不证明风险结构的独立价值。原确认区间 `95433--95440` 没有作为整块运行，但其中 `95434` 与开发块重叠，因此不再把它当作干净确认。独立 v2 固定块另行保留全部八个新 seed，并报告了开发块均值排序未复现的结果。

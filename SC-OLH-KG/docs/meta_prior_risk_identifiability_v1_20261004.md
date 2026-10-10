# 风险因子可识别性合同 V1：几何与初始校准结果

## 为什么重做设计

原 Inventory 风险设计只有 16 个点，虽然数值秩为 9，但条件数约为 `19,658`；这些点主要覆盖默认 context，难以把 local、shared 和 volume 风险方向分开。新合同保持公开 96 策略和同一物理风险模型，只把四个公开 context 的风险暴露都纳入来源设计。

每个 context 使用九列公开风险特征做确定性 QR 选点，每个选 16 个，共 64 个来源点。选点不读取来源响应、目标响应或 oracle 风险。

合同与入口：

- [风险合同](../performance/meta_prior_risk_identifiability_contract_v1.json)
- [几何预检](../performance/preflight_meta_prior_risk_identifiability_v1.py)
- [几何结果](../paper_artifacts/meta_prior_risk_identifiability_v1/geometry_preflight.json)
- [来源档案生成器](../performance/generate_meta_prior_risk_identifiability_source.py)
- [来源档案](../paper_artifacts/meta_prior_risk_identifiability_v1/source_archive.json)

## 几何门

| 指标 | 原 16 点设计 | 新 64 点多 context 设计 |
|---|---:|---:|
| 风险特征秩 | 9 | 9 |
| 条件数 | 约 19,658 | 264.2 |
| 每个 context 点数 | 不均衡 | 16/16/16/16 |
| 目标/来源 oracle 用于选点 | 否 | 否 |

几何门通过。新来源档案含 8 个 source tasks、64 个点、每点 12 次重复，共 6,144 次来源调用；没有目标观测和验证调用。

## 来源与目标初始校准

来源留一中，learned factor-family prior 相对同一粒子库 pooled-risk 的 log predictive density 平均提升 `+0.692`；8 个留出任务中 7 个为正，1 个为负，负迁移保留。

在 8 个开发目标上，每个只使用共同的 8 个初始策略点，共 768 次目标观测：

| 指标 | learned factor-family | pooled-risk |
|---|---:|---:|
| `eta` 留出 RMSE | 0.301 | 0.363 |
| 90% 近似区间覆盖 | 99.6% | 100.0% |
| 因子系数 RMSE | 0.255 | 0.251 |
| 平均有效粒子数 | 71.1 | 236.4 |

learned risk 在 8/8 目标上降低了 `eta` RMSE，但系数 RMSE 没有降低。这说明当前方法更像是在改善总风险预测，而不是可靠地恢复每个真实物理因子。该结果还没有搜索、推荐或独立验证，不能写成风险结构的部署收益。

## 下一道门

在读取确认块前，先做一个小规模风险闭环：固定 observable exposure 的均值模型，比较 learned factor-family 与 pooled-risk 的 24 点搜索、最终推荐和独立验证；同时保留已知几何的 oracle 诊断，但不把它列为可部署 arm。若风险 RMSE 的改善不能转化为推荐/验证收益，风险主张应限定为预测校准，而不是因子级部署贡献。

## 24 点风险闭环结果

该闭环在 8 个开发目标上比较 observable mean + pooled-risk 与 observable mean + learned factor-family risk，共 16 个单元。两臂共享初始点、目标观测流、24 点搜索预算和独立验证器。

| arm | 平均初始奖励 | 平均最终真实奖励 | 平均已验证奖励 | 认证 | 错误认证 |
|---|---:|---:|---:|---:|---:|
| observable + pooled risk | 0.8461 | 0.8605 | 0.8608 | 8/8 | 0 |
| observable + learned risk | 0.8461 | 0.8599 | 0.8602 | 8/8 | 0 |

learned-risk 相对 pooled-risk 在 1/8 任务上略有改善、1/8 任务退步，其余任务推荐相同；平均最终真实奖励差为 `-0.00059`。因此多 context 风险设计解决了明显的几何病态，并把预测 RMSE 的改善稳定地带到初始校准，但没有产生独立部署收益。闭环记录见 [closed_loop.json](../paper_artifacts/meta_prior_risk_identifiability_v1/closed_loop.json)。确认 seed 块没有读取，风险因子恢复不晋级为投稿主张。

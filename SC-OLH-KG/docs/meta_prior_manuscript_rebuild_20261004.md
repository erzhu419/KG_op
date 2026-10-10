# Meta-prior 主文重建主张矩阵

## 目的

当前 `manuscript/main.tex` 是已经完成的 atlas/初始化论文。它的证据和叙事仍然有效，但不能把它直接改名为完整 meta-prior 论文：旧主文只证明历史结果可以帮助选择初始点，而 M1 研究的是来源函数结构如何进入目标后验、在线采样和独立验证。两条证据链的对象、预算和结果身份不同。

因此保留旧稿作为 legacy atlas 版本，同时按本文件重建一份 meta-prior 主文。重建稿只使用已经冻结的 M1 结果；未完成的轨迹表示、因子级风险恢复和普适部署收益不写成贡献。

## 主线结论

当前可承担的主结论是：

> 在声明的公共策略表示和有限函数字典内，来源任务可以学习响应函数的结构先验；少量目标观测可以更新结构混合和函数后验；在匹配机制上，这一结构先验改善早期目标预测和样本效率。来源失配时，目标证据可以撤回错误结构，但风险头和候选排序仍可能限制独立部署收益。

这句话包含三个彼此可核对的层次：来源结构学习、目标后验纠偏、早期决策收益。它不把风险先验或 KG 的小增量扩大为普适优势。

## 主张—证据—写法

| 主张 | 主要证据 | 当前状态 | 主文允许的写法 |
|---|---|---|---|
| 来源数据能学习目标响应的公共函数结构 | `meta_prior_mechanism_v3_20261002.md`；`meta_prior_source_contribution_replay_20261004.md`；`meta_prior_independent_source_confirmation_20261004.md` | 已支持，但限定在公开八维表示和 68 函数字典 | “source-learned structural prior improves early target prediction under matched mechanisms” |
| 目标付费观测会纠偏来源结构 | V3 的模型混合权重；`meta_prior_method_contract_20261004.md` | 已实现并观察到失配时权重塌缩；决策收益不总是同步 | “target evidence can downweight a mismatched source structure” |
| 风险头能被来源数据学习 | V3 协方差似然；`meta_prior_physical_risk_20261004.md` | 预测误差小幅改善，支持恢复和部署贡献不稳定 | “risk learning is measurable but not yet a stable deployment advantage” |
| 联合 KG 能使用均值—风险协方差 | `meta_prior_joint_kg_20261004.md`；`core/joint_verification_kg.py` | 工程闭环和单元测试完成；增量收益很小且受积分近似影响 | “KG is a decision rule in the contract; its incremental value is evaluated and found small in current pilots” |
| 结构先验在早期预算下有决策价值 | V3 24 批和 V6 总预算桥接 | 匹配机制上支持，80 批时通用先验可追平；失配时可退步 | “early-budget gain under matched mechanisms, with negative transfer retained” |
| 方法有独立运营部署优势 | `meta_prior_operational_closed_loop_20261004.md`；`meta_prior_hyperbo_full_loop_20261004.md` | 尚未支持；多个新任务无可行候选或未通过认证 | 不作正面主张；作为边界和负结果报告 |
| 完整四性质先验、轨迹表示和高维自由优化已实现 | `meta_prior_method_contract_20261004.md` | 未完成 | 不写入贡献或摘要 |

## 证据层级和身份

1. **机制主结果：** V3/V6 使用固定来源档案、公共字典、相同初始设计和目标预算，回答来源结构是否改善早期目标学习。V3 的 1,920 格和 V6 的 1,920 格是主要定量证据。
2. **归因和复核：** 来源贡献重放、独立来源确认、风险结构和联合 KG 分别回答信息来自哪里、预测是否可复现、风险是否有独立价值、KG 是否增加收益。这些结果不能合并成一个“所有模块都显著有效”的数字。
3. **边界证据：** 物理风险闭环、官方 HyperBO 全闭环和运营目标的独立验证失败保留在结果与讨论中，说明任务可行性、均值错配、候选覆盖和有限样本风险校准是当前限制。
4. **legacy 对照：** 原 atlas 主文及 Energy V5 结果继续作为初始化方法和外部边界，不作为 M1 结构后验优势的证明。

## 建议的重建结构

1. **Introduction：** 从“历史如何进入目标后验”开始，明确与 outcome-ranked warm-start 的区别；给出三项受限贡献和负迁移原则。
2. **Related work：** transfer Bayesian optimization、multi-task/structured priors、chance-constrained simulation optimization、knowledge-gradient verification；只写与当前信息合同一致的比较。
3. **Model and prior：** `u=psi(x)`、公共字典、`f/g/eta` 三头、来源任务分布、目标后验、模型混合和风险测量噪声。
4. **Decision rule：** 预计认证部署效用、相关 Gaussian KG、验证预算和停止规则；明确它不是原数学机会约束最优的替代定理。
5. **Experiments：** 来源档案、目标机制、公共初始设计、总预算、配对种子、独立验证器和成本核算；把 V3/V6、来源复核、风险和 KG 分开。
6. **Results：** 先报告均值预测和早期样本效率，再报告目标纠偏、风险校准、KG 增量和失配退步，最后报告运营/HyperBO 负结果。
7. **Discussion：** 解释风险排序、候选池覆盖、任务可行性、测量噪声和 KG 计算成本；把轨迹表示、因子级恢复和部署价值列为后续工作。
8. **Conclusion：** 只保留“公共表示内的来源结构先验改善早期学习”这一可重复结论。

## 不能混写的对象

- `source-scored initial design` 是旧 atlas 贡献，`source-learned structural prior` 是 M1 贡献。
- “预测误差下降”不等于“认证部署效用上升”；后者必须使用独立验证器。
- “KG 已实现”不等于“KG 已证明有稳定增益”。当前联合 KG 试点的增量接近零，且高预算下可能因验证机会成本变差。
- “风险 RMSE 改善”不等于“风险结构带来独立决策价值”。当前风险结构改善小且不稳定，不能写成主贡献。
- “候选银行扩展后有收益”不等于“高维自由优化已解决”。当前表示仍是公开八维坐标，候选池覆盖和表示获取成本需单列。

## 投稿前的停止判据

在不新增大规模实验的前提下，主文可进入写作收口的条件是：

- 所有主表注明实验身份、预算、种子和独立验证状态；
- 摘要和结论只使用上表中“已支持”或明确限定的“部分支持”措辞；
- V3/V6 的正收益、失配退步、风险不稳定和 HyperBO/运营负结果同时出现；
- 旧 atlas 稿件保留，不与新主文共享会改变含义的标题、摘要或图表；
- 新主文的代码入口能从固定配置重放表格，不把开发试点 A45 当作确认实验。

据此，当前项目不是“meta-prior 没做成”，而是已经完成最小闭环，尚未完成更强的表示、风险和运营部署主张。重建稿应围绕这个边界写，而不是继续用旧 atlas 的标题承载新问题。

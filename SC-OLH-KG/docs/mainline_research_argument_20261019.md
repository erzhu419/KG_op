# Stage101：prior跨策略传播的研究论证稿

**核心论点：在公开结构表示和动态候选前端支持的实际d=1000仿真任务中，来源学习的prior通过初始化阶段的跨策略条件化，提高固定预算内的获证与部署净质量。** 本稿依据[原始目标](original_goal_and_mainline_contract_20261015.md)和[封存证据链](mainline_evidence_chain_20261018.md)收束研究身份；现有[前端初始化论文](../manuscript/README.md)保留独立身份。

## 论证段落草稿

We study whether a source-learned prior can turn observations at sampled policies into better deployment decisions in structured simulation optimization at d=1000. With source assets, the dynamic candidate-generation algorithm and the terminal protocol fixed, joint conditioning of the role backend during initialization, followed by pointwise updating, improved true certification and signed certified-deployment quality over pointwise conditioning from the outset in two independently registered geometry cohorts. Whole-policy source-response permutation controls further support a benefit of correct source association on complete paired subsets and a positive source-by-propagation interaction in certification. The corresponding net-quality interaction remains unresolved. Together, these results support the decision value of initialization propagation under the tested representation, task distribution and budget.

## 术语、合同与主证据

| 沿用术语 | 本稿定义 |
|---|---|
| meta-prior／来源先验 | 来源数据学习函数系数分布；公开表示和角色基提供结构，观测通过共享系数关系影响未采样策略。 |
| AFTER | 角色后端initial10联合条件化，随后仅更新实际观测点；原前端/native提议后验仍正常学习。 |
| POINT | 角色后端从source0起逐点条件化；仍保留相同来源prior、前端和来源相关矩阵，不是无来源或协方差清零。 |
| C | 获证且真实可行的指标；表中差值单位为百分点。 |
| G | 相对初始10点中真实可行最佳点的带符号获证质量增量，正质量减损失；无基线或搜索失败保持null。完成但未真实获证且有基线时按登记为0。 |

现行合同为initial10＋3次在线查询、每格旧终验上限352，在线/STOP总上限365/362。主比较识别角色后端初始化联合条件化的总体效果；各臂沿同一生成算法根据自己的历史产生候选和名单。两流先在同几何内平均，统计单位为几何；以下转录已封存的描述性95%几何配对t区间，两个独立批次不合并。

| 比较与来源 | C：几何数；差值〔区间〕 | G：几何数；差值〔区间〕 | 支持的判断 |
|---|---|---|---|
| [Stage85 AFTER−POINT](mainline_precision_confirmation_results_20261018.md) | 144；+5.903〔2.460,9.346〕 | 134；+0.000521671〔0.000256753,0.000786590〕 | 初始化传播有决策收益。 |
| [Stage93 AFTER−POINT](mainline_source_alignment_target_results_20261018.md) | 144；+9.028〔5.563,12.492〕 | 136；+0.000497317〔0.000242134,0.000752499〕 | 在另一批新几何中复现。 |
| Stage93真实AFTER−置换AFTER | 105；+10.476〔5.332,15.620〕 | 97；+0.000278300〔0.000046477,0.000510124〕 | 固定原学习前端时，正确来源关联具有条件部署价值。 |
| Stage93来源×传播交互 | 105；+9.048〔4.250,13.845〕 | 97；+0.000164725〔−0.000087476,0.000416926〕 | 获证交互为正，净质量交互未确认。 |

来源、查询和部署分别提供三段证据。[来源留出](mainline_role_source_holdout_results_20261016.md)中首次传播使未采样目标均值/目标差MSE下降17.10%/17.78%，同时风险均值MSE上升5.66%。[Stage86](mainline_precision_propagation_paths_results_20261018.md)与[Stage94](mainline_source_alignment_paths_results_20261019.md)的共同prefix10下，真实AFTER/POINT首查询分别有252/288和249/288流改变，保存路径可追到名单、实际验证及部署。两批传播收益的最大正几何分别占10.45%与11.62%，去掉后净差均值仍为正；这支持整体均值并非由单例维持。获证增加伴随新质量与新增损失，Stage93交互中新策略正质量分量为+0.000233836，部分被损失变化−0.000069111抵消。

## 局限及保留证据

- 来源留出为d50、熟悉来源家族和两条锚点输入；目标部署为d1000和initial10。它们相互支持，但未识别“预测误差下降导致质量收益”的因果中介，初始化干预也未分离纯协方差贡献。四类prior的共同必要性、任意高维黑箱优势以及相对外部文献的新颖性定位仍未由本稿建立。
- Stage93全部11臂和67个原失败保留。39个FactorShock几何缺完整来源对，另8个几何无质量基线；来源C105/G97不是全144的无偏替代，全144来源效果保持null。FactorShock仅9个完整对，0099占该家族正贡献91.17%；Queue低ESS和六套结构来源精度未认证状态不变。[Stage100](mainline_source_finite_probe_results_20261019.md)的12项/6144次来源调用未复现异常，不能解释旧失败，有限探针已关闭。
- 保留旧60/60及GP的正反证据。V51含在线后验/HVD/KG，19次原改善与26次终端闭合改善分开；另一[前端矩阵](../manuscript/meta_prior_legacy_frontend.tex)中proposal-only、Stacked GP和SAASBO均60/60，支持前端互补价值。早期[16任务结构系统落后GP](mainline_d1000_new_bank_results_20261015.md)、后续[12任务AFTER−GP为6胜5平1负](mainline_same_policy_confirmation_results_20261016.md)及两批大实验的GP描述结果均保留；不同模型、终端及合同的结果不拼成普遍优于GP的结论。
- 全部正例、平例、负例、9/11臂参照和历史费用沿[完整证据链](mainline_evidence_chain_20261018.md)保留。FULL在Stage85未显示额外净收益，在Stage93有小幅正面辅助结果；不能据冻结AFTER推断持续传播普遍无益。来源、目标、失败、评分和控制费用分别记账，未测量项保留unknown。

## 本轮决定与下一步

本轮只整理已有证据，没有新增仿真、评分、拟合、积分、后验恢复、统计重算或scheduler计算任务，模型、前端、搜索、旧终端和门槛冻结。[Stage101账](../paper_artifacts/mainline_evidence_chain_update_v1/cluster_control/stage101_accounting.json)只记录文档同步；Stage100已知377.955910416 CPU秒及更早费用仅作历史引用。

下一步以这条论证写独立的主线方法与实验正文工作稿：先解释公共表示、来源系数prior和AFTER/POINT实际更新，再呈现两批独立主比较及来源四臂交互，最后连接查询—名单—验证—部署的正反路径。沿封存结果引用表格，不改原前端论文、不重新统计、不把解决全144归因或单query精算设为写作前提。需要扩大的科学主张保持待验证，任何新验证另立具体协议。

Stage102已展开为[主线方法工作稿](mainline_methods_working_draft_20261019.md)与[实验正文工作稿](mainline_experiments_working_draft_20261019.md)，直接依据项目代码和封存记录。原论证和前端论文保留；下一步从方法与结果提炼摘要、引言和讨论，科学结论仍以本稿的支持范围为界。

# KG_op 重构诊断与 Operations Research 研究计划

审阅基线：[d1a13a40a38d396ef97edb9c2fc83d5adf532d4c](https://github.com/erzhu419/KG_op/commit/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c)，直接父提交 1cc72923。审阅对象包括 Gemini 重构报告、新代码、论文源码及编译稿、Lean 声明、实验汇总，以及旧主线的关键历史证据。

**总判断：保留这次重构的主体，当前版本先不投 Operations Research。** 新版确实有可用的函数元先验、持续目标更新、动态运筹仿真器和独立认证流程；但论文最强的几个结论——真实部署成本优势、消除方差爆炸、原版 SOTA 优势、逐点残差安全上界、实际退火算法消除负迁移、成功微观安全部署——还没有得到当前代码和交付证据的共同支持。

问题不在于采用了低维表示、固定函数库、近似 KG 或轻量实现。合理简化可以成为优点。真正需要处理的是：简化之后算法究竟是什么、指标究竟测什么、定理前提是否由实现满足，以及收益能否归因于所声称的机制。

## 1. 审阅范围与结论置信边界

完整 Git 树含 17,609 个跟踪文件，大部分是历史实验输出。本次提交新增 68 个文件，全部为 A，零修改、零删除；所以“新路线隔离开发，旧主线保留”得到 Git 直接支持。

本次审阅完成了全树盘点、本次新增科学路径的逐模块审核、报告—代码—论文—摘要结果交叉核验，以及旧主线关键证据链回溯。并未声称逐行重读全部 17,609 个历史文件，也没有独立重跑服务器科学实验。主要结果判断依赖已提交的 compact summaries；逐运行证书与完整冻结凭据未随这次提交提供。

本地只执行了源码语法检查、紧凑结果统计及小型确定性反例计算：25 个新增 Python 文件通过 AST 解析；精确二项阈值、跨维插值、残差平方界等进行了直接核验。语法通过不代表科学正确。遵守仓库 AGENTS.md 的集群限制，没有在本机运行仿真、源先验拟合或后验恢复，也没有修改仓库。

| 层面 | 当前判断 | 对投稿的含义 |
| - | - | - |
| 工程架构 | 显著变清晰，值得保留 | 有可持续研究底座 |
| 原始函数元先验目标 | 已有真实实现，贡献尚未分离 | 应继续作为主问题 |
| 动态应用真实性 | 比旧闭式基准前进，但若干模型描述失真 | 修正文稿，明确抽象层次 |
| 主结果计量 | 存在阻断结论的问题 | 先重建指标与逐运行账本 |
| 外部 SOTA 比较 | 数值优势存在，基线身份与适配不充分 | 不能按当前名称宣布全面胜出 |
| 理论与 Lean | 支撑性条件引理有效，算法级连接不足 | 收缩主张，选择一个实质理论问题 |
| LiveSUMO | 不可行压力场景，0 次认证成功 | 尚不是成功部署应用 |
| OR 就绪度 | 尚未达到我建议提交的状态 | 先过下文 P0、P1、P2 决策门槛 |

这是审阅判断，不是对编辑决定的预测。OR 的 Simulation 领域欢迎显著原创的方法、建模与应用，也接受分析困难但影响力强、设计充分的实证工作；并没有“必须堆大定理”的统一要求。它要求贡献必要性、适用范围以及实验可重复性说清楚。[官方领域声明](https://pubsonline.informs.org/page/opre/editorial-statement/area-editors-statements)

## 2. 按 Git 历史重建新旧关系

### 2.1 这次是平行新实现，不是逐步修改旧算法

68 个新增文件分为：v2 代码与配置 28，论文与样式资源 22，结果及执行计划 13，重构报告 1，Lean 4。新目录只出现在这一条发布提交中。Git 能证明最终交付内容和旧文件保留，不能独立证明报告所述 Pilot 修复、冻结、确认集首次打开的时间顺序。

此前主线经过 10 月 13 日诊断、10 月 19 日 Stage102/103 整理、10 月 26 日发布门槛核验。Gemini 报告中若干“旧版问题”更接近较早支线，不能代表 d1a13a40 的直接父版本。

### 2.2 六项历史判断应改写

| Gemini 报告的判断 | Git 与旧代码支持的判断 | 对新版的正确定位 |
| - | - | - |
| 旧版“从经典求解器优化轨迹学习先验”尚未落地 | 来源策略确为固定设计，因此日志来源批评成立；但真实响应已用于学习 encoder、函数基、系数分布和噪声结构，不能进一步推成没有函数先验学习 | 新增了自适应求解器日志，简化了函数学习实现；并非首次实现函数先验 |
| AFTER 后整个算法不再在线学习 | 冻结的是 role 后端的跨策略联合更新；native 前端仍吸收目标反馈 | 新版恢复更统一的持续跨策略条件更新 |
| Stage85/93 只有内部 STOP 比较 | 有 AFTER、POINT、FULL、STOP、同源 GP、native 等，主比较是 AFTER−POINT | 现代外部方法与新动态任务比较是拓展，不能抹掉旧对照 |
| 公开 exposure 已证实偷看真值 | 旧表示与闭式问题高度匹配，但所查接口只读公开 x 统计量，没有读私有最优位置或真值函数 | 应批评基准与表示同构及信息公平性，不能直接定性私有真值泄漏 |
| 59× 是所有旧实验共同病因 | 59× 来自一个开发案例的一个策略点；后续分析还发现 source0 阶段的方差问题 | 组内方差是合理修复，但未证明根除旧系统全部病因 |
| 旧版没有核算来源成本 | 已记录 source calls、搜索、终验与失败成本；缺的是明确摊销曲线 | 新增摊销视角有用，成本记账本身不是首次出现 |

旧 Stage85/93 两批各 144 个几何实例，AFTER−POINT 的真实获证率差分别为 +5.903、+9.028 个百分点；带符号净质量差也为正。同源 GP 补充比较有正面结果，对 native 则结果混合。它们不能替新版作证，但也不应被描述为“旧版完全没有函数学习增益”。[旧研究论证](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/docs/mainline_research_argument_20261019.md)；[旧统一方法说明](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/docs/mainline_unified_method_and_plan_20261019.md)

失败计数也要写完整：Stage93 permuted AFTER 全体为 66/288，FactorShock 子集为 66/96，因此 66/96 不是必然算错，问题是未说明分母范围。旧 Stage97 的日志诊断结论是 cause_unidentified，存下的是积分误差/normalizer 门槛失败；不能把原因直接写死为“移除残差核造成协方差奇异和 LinAlgError”。[旧数值可执行性审查](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/docs/mainline_source_numerical_executability_results_20261019.md)

最后，旧最新主线已明确披露 constrained TS，不宜再用“偷偷覆盖 KG”概括它。新旧的任务、表示、源数据、预算、模型及认证规则都变了；不能直接把新旧两张表的差当作受控修复效应，或者据此宣称严格的 20,000 倍算法加速。

## 3. Gemini 做了什么、为何合理、目前得到什么

| 模块 | 改动及动机 | 已有价值 | 仍需补的证据 |
| - | - | - | - |
| contracts 与 simulators | 分离公开名义规格和私有随机环境；从闭式几何函数改为动态系统 | 更贴近库存、排队、储能、交通决策 | 描述必须与实际物理方程一致；公开表示可能损失信息 |
| legacy_solvers | 实际运行自适应搜索，保留重复观测和优化轨迹 | 来源不再只是固定策略库 | 简化求解器需如实命名；日志覆盖和偏倚假设需量化 |
| debiased_spectral_prior | 占据平衡、白化、低通/稀疏选择、源系数分布 | 确实学习跨策略响应函数及不确定性 | 占据平衡不等于已证明消除偏倚；R_perp 不是已证上界 |
| tempered_hvd_posterior | 组内方差与均值模型分离；源先验加局部 GP；目标持续更新 | 结构更清楚，避免一种均值偏差污染噪声的通道 | 实际有限样本校准、数据驱动退火风险尚无保证 |
| verification_kg | 秩一 fantasy、25 点求积、候选近似 | 真正执行一步前瞻，计算可控 | 使用的是认证近似代理；未证明优于 TS |
| gated_verifier_v2 | 冻结三槽名单，独立二项验证，认证候选中择优 | 错误认证控制有坚实基础 | 弃权/部署语义错位，目标择优不保证真目标改善 |
| baselines 与 dispatch | 同一框架下组织外部方法、消融及成本账 | 比零散实验更易审计 | 命名忠实度、通道适配、评价指标需修复 |
| 论文与 Lean | 尝试把算法、证明、结果集中交付 | 提高可读性、可检查性 | 多处从条件引理跳到实际算法保证 |

### 3.1 用户关心的 meta prior 内核是否还在

**在，而且不能只叫 warm start。** 源任务实际用于拟合响应系数，选择基，估计系数均值和协方差；目标观测进入后验，能够影响未采样策略的预测。一个示意性的函数模型是：

```math
f_t(x)=\phi(\psi_t(x))^\top\theta_t+r_t(\psi_t(x)),\qquad \theta_t\sim (\mu_0,\Sigma_0).
```

其跨策略相关性由源学到的系数协方差和局部核共同提供。这就是有实质内容的函数元先验。[先验实现](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/v2_or/priors/debiased_spectral_prior.py#L321-L436)；[目标更新实现](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/v2_or/models/tempered_hvd_posterior.py#L250-L317)

但“学到高维决策耦合”有两层不同含义：响应函数空间中，模型学习策略如何影响成本/约束；策略函数空间中，算法要生成跨时间或跨路网的决策函数。前者已实现，后者主要仍依赖预设 DCT、源模板和局部变异。固定公共表示与 DCT 完全可以是合理设计，研究需要证明源学习在其基础上的增量。

目前所有家族 psi 实为 12 维。局部 Matérn 核也在同一 psi 上。如果两个高维策略具有相同摘要，但真实动态成本或风险不同，局部核不能找回被摘要丢弃的信息。应进行“表示碰撞”诊断，而不仅检查是否读取私有参数。v2 主要验证同家族跨视距、跨市场、跨网络规模，目标维数约 44–336；不能直接继承旧版 d=1000 或跨家族迁移主张。

## 4. P0：主指标、弃权和部署必须先统一

### 4.1 mean_deployed_cost 不是实际部署成本

campaign_runner.py 第 593–605 行，在每个任务和预算下收集所有算法、所有种子审计后“可行”的成本，取其中最坏值与 anchor 成本的最大值，再乘 1.05；若没有可行候选，则用 anchor×1.25×1.05。第 627–630 行把所有未认证运行的成本替换成这一事后构造值。[源码](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/v2_or/dispatch/campaign_runner.py#L593-L630)

因此，该指标实际是依赖比较阵容的事后惩罚分数。增加/删除一个比较方法，就可能改变已有算法的分数。它既不是被返回策略的期望成本，也不是事先固定的安全回退策略成本。

现有 compact summary 给出一个无需重跑的直接反例：

| 同一批 Inventory STOP 运行 | N=15 | N=30 | N=60 |
| - | -: | -: | -: |
| 审计目标均值 | 1.884147 | 1.884147 | 1.884147 |
| 所谓 mean_deployed_cost | 1.916187 | 1.899911 | 1.917705 |

STOP 没有随预算新增搜索，认证数也相同，其分数仍发生变化。这会影响成本曲线、失败惩罚、效应量和相关显著性。它不意味着所有现有结论必然反转：若两个方法在某任务均全部认证，二者该部分成本不受这一替换影响；但全局部署成本主张必须重算。

### 4.2 代码没有执行注释声称的安全回退

gated_verifier_v2.py 第 318–324 行的注释说回到 Operational Anchor，实际却从三个未通过候选中按观测违约率、目标均值选一个，放进 deployed_policy，另标 certified=False。[源码](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/v2_or/verification/gated_verifier_v2.py#L318-L324)

如果下游遵守 certified=False，这可以代表弃权，但没有真实部署成本；如果下游真的部署该向量，现有证书定理不保护它。正确的基本保证是：

```math
\Pr\{\text{certified}\ \land\ p(x_{\mathrm{deploy}})>\alpha\}\leq\delta.
```

它不是所有返回策略不安全概率的无条件保证，也不是已认证条件下的不安全概率保证。alpha=0.05 是政策违约概率容忍线，delta=0.05 是错误认证控制水平，两者含义不同。

### 4.3 Raw True Obj 也不是 primary challenger 的成本

保存和审计的是 verifier 最终返回的 deployed_policy，而非固定 primary x0。返回者可能是 primary、经验 incumbent、posterior interior，或全部失败后选出的候选。现有 per-run 输出没有完整保存 primary 与所有槽位政策。因此报告与 LiveSUMO 表将 Raw True Obj 标成“42 次搜索后主候选真值”并不成立；最终选择还可能使用了验证样本。[保存与审计逻辑](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/v2_or/dispatch/campaign_runner.py#L344-L404)

### 4.4 应怎样修

先确定真实决策合同，二选一：

1. 真弃权：deployed_policy=None，另存 selected_candidate；报告获证率、获证条件下的成本、未部署率。若需要综合分数，拒绝损失必须事先确定，并明确称为损失函数。

2. 真回退：事先固定 operational fallback，单独说明其同分布安全依据与成本；真正返回该策略。若也要认证，预算与失败处理必须计入。

逐运行账本应保存 primary、全部 shortlist、每槽统计量与 certificate、最终动作、是否回退、search/verify/audit seeds、样本数量、失败与恢复状态。已有服务器逐运行数据若仍在，优先重汇总，不重复旧 cohort；无法从旧日志恢复的 primary 字段如实标缺失。

验收标准：相同运行在不同预算展示或比较阵容中，成本必须相同；删除一个对照方法不得改变其他方法的原始结果；任何名为 deployment 的记录都能追溯到实际被部署策略。

### 4.5 三槽参数与 initial guard 还需勘误

报告写的验证样本数为 (130,130,120)，当前代码与论文实际为 (112,128,140)，对应 delta=(.025,.015,.010)，总样本上限 380。三个槽的精确临界违约次数均为 1。报告声称 K=130、alpha=.05、delta=.025 时允许 2 次违约也不对：二项下尾概率约 .039482，大于 .025，最多允许 1 次。这里是报告例子错误，当前精确 CDF 判定逻辑正确。

代码名单为 primary → empirical incumbent → posterior interior，并非报告所写 primary → interior → initial guard。所谓 initial guard 使用全部已评估记录，而不局限最初 n0 个点。它可以是合理 incumbent，但不应再声称专门保护初始候选。认证候选按经验目标择优不会增加错误认证预算，却不能保证真实目标不差于 STOP 或旧 incumbent。

## 5. P0：方差爆炸图测量的是人为注入倍率

tempered_hvd_posterior.py 第 176–188 行对 Untempered-FULL 显式设置：

```python
log_shift = min(0.45 * max(mean_norm_sq_err - 1.0, 0.0), 4.2)
untempered_inflation_factor = exp(log_shift)
```

其他模式直接赋值 1.0。这个字段随后被汇总为 Infl.，用于“消除方差爆炸”的图。[模型源码](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/v2_or/models/tempered_hvd_posterior.py#L147-L203)

因此 27.16×、14.20×、8.44×、2.15× 是人为构造消融的注入倍率；TLS 的 1.00× 是赋值结果，不是预测噪声方差与独立审计方差之比。这可以叫均值失配污染压力试验，不能作为旧系统真实方差病理已被根除的独立实证。Untempered 臂还改变局部核及超参数选择，不是单开关消融。

组内中心化方差对共同均值平移不变，是正确且值得保留的结构。它排除了一个污染路径，但不保证少量重复样本的方差估计准确，不保证跨策略外推准确，更不保证所有目标 shift 下校准。

下一步在固定政策诊断集上，用独立高重复仿真得到方差审计估计，比较组内 SSQ 与自然残差估计器；保持均值模型、重复数、残差核和候选点一致。报告方差误差、log score、预测区间覆盖及其不确定性。不要继续用被算法直接设为 1 的字段证明算法校准。

## 6. P0/P1：外部方法身份与比较公平性

| 当前名称 | 当前代码实际做法 | 处理建议 |
| - | - | - |
| SAAS-CBO | 用边际相关性阈值生成 ARD 长度尺度；没有 SAAS 层级稀疏先验推断，也没有完整 MAP 优化 | 改为 correlation-screened ARD GP，或接入忠实 SAAS 实现 |
| HyperBO | pooled ridge 均值、残差总体方差、固定长度尺度 1.8 | 改为 pooled-prior GP，或实现共享 GP 预训练 |
| RGPE | target 使用训练内预测加独立扰动；源用均值排序 | 补 LOO target ranking 与联合后验采样，或明确为近似 ranking ensemble |
| SCBO | 候选处独立正态抽样，未采样联合 GP 函数 | 接官方/忠实实现，或明确 SCBO-inspired |
| HetGP-KG | 两阶段方差 GP，加作者自己的 V-KG | 可作自定义异方差对照，不能直接代表原版 hetGP |

对应代码在 sota_baselines.py：45–96、217–223、419–449、542–564、664–676。[基线文件](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/v2_or/baselines/sota_baselines.py)

这里有可外部核验的定义差异：SAAS 原论文以 HMC 推断层级稀疏先验；HMC 并非唯一可接受近似，明确标注且真正优化相应后验的 MAP-SAAS 也可以合理，但当前相关性阈值没有实现这一模型与推断。HyperBO 的核心是预训练 GP 分布；BoTorch 的 RGPE 明确对 target 使用 LOOCV 排序损失。这些核心机制不能任意删除后仍按原算法身份宣布胜负。[SAAS 原文](https://proceedings.mlr.press/v161/eriksson21a.html)；[HyperBO 原文](https://jmlr.org/papers/v25/23-0269.html)；[RGPE 官方实现](https://botorch.org/docs/v0.17.0/tutorials/meta_learning_with_rgpe)；[SCBO 官方实现](https://botorch.org/docs/next/tutorials/scalable_constrained_bo)

### 6.1 跨维 adapter 有具体语义错误

\_warp_policy_dim 对整条 flattened 向量插值。Inventory/Queueing 向量是两个控制通道交错排列；这样会把 regular 与 expedite、staff 与 diversion 互相混合。

例如常量双通道 [0.2,0.8,0.2,0.8,0.2,0.8] 从长度 6 映到 8，输出约为 [0.2,0.629,0.543,0.286,0.714,0.457,0.371,0.8]，不是语义正确的 [0.2,0.8] 重复四次。TLS 则有分通道 DCT。这个错误先修，不能把损坏 adapter 的弱表现解释成 raw-coordinate transfer 理论上失败。

### 6.2 当前 TargetOnly 不是纯粹移除源先验

TLS 同时获得源精英初始点、source-template proposals、源学习字典、系数均值/协方差和噪声先验；TargetOnly 的初始点、候选模板、基函数及噪声正则化也不同。因此差异混合了多个来源信息通道。[候选与初始化](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/v2_or/acquisition/verification_kg.py#L203-L316)

需要两组互补比较：实际系统比较保留各方法自然表示，回答谁整体更好；机制比较使用相同公开信息、初始化、候选产生机制和认证器，仅改变被检验组件，回答为什么好。不能只做其中一组。

### 6.3 旧求解器日志仍值得保留

NSGA-II 实现缺少 nondominated sorting/crowding，CMA-ES 是简化 diagonal evolution，Heuristic-Powell 更接近低频方向 pattern search。这不否定它们产生了真实自适应日志。作为来源生成器，诚实命名即可；若把它们作为原版经典求解器竞争力证据，再补忠实实现。不要因为命名问题把整个来源档案推倒。

## 7. 重读确认集：哪些正面结果成立，哪些尚不成立

Confirm summary 包含 4,128 个搜索/认证单元、3,392 个认证运行、2 个按审计阈值标记的假证书。每家族是 8 个任务×3 个算法种子。TLS 在四家族三预算合计 288 个运行中未观察到假证书。[确认集摘要](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/paper_artifacts/v2_or_confirm_v1/compact_summary.json)

以下全部是当前保存指标，越低越好；P0 修复后应重算，不能直接称为真实部署成本。

| N=60 家族 | TLS | STOP | Target-Only-Psi | 同表示 Constrained-TS | TLS/TS 获证率 |
| - | -: | -: | -: | -: | -: |
| Inventory | 1.844959 | 1.917705 | 1.874823 | 1.812800 | 100% / 100% |
| Queueing | 2.475087 | 2.626517 | 2.519949 | 2.476180 | 100% / 100% |
| OPSD | 0.057102 | 0.058101 | 0.059234 | 0.056683 | 100% / 100% |
| Traffic DES | 1.152622 | 1.170384 | 1.097857 | 1.078599 | 62.5% / 83.3% |

### 7.1 真实值得保留的信号

- Queueing N=60 相对 STOP 改善约 5.67%，保存的 Holm p≈8.34×10^-7，支持该环境下持续搜索的增量价值。

- Inventory 相对 STOP 改善约 3.64%，Holm p≈0.001016，但 STOP 有未认证运行，效应须在修正惩罚后复核。

- 早期迁移有正面信号：Queueing N=15 相对 TargetOnly 改善约 6.21%，Holm p≈0.00267；OPSD N=15 改善约 9.63%，Holm p≈0.00150。当前混杂条件下，它支持组合系统有早期优势，尚不能全归因于函数先验。

- N=60 相对七个外部实现的 28 个家族×方法比较，在现有计量下都显著。单独用保存的原始 p 对这 28 项做全局 Holm，最大约 0.00931。主要问题不是“多重比较一改就全没了”，而是指标和对照身份。

### 7.2 报告应收缩的结论

- N=60 TLS 相对 TargetOnly 的四个保存 Holm p 为约 0.8447、0.2813、1、1，未显示家族层面的显著优势。

- 同表示 TS 在 Inventory、OPSD、Traffic 均值更低，Queueing 基本持平。这里不能反向宣称 TS 已统计显著击败 TLS；能确定的是 V-KG 独立必要性未建立。

- OPSD 的 TLS 随 N=15/30/60 为 0.056568→0.056721→0.057102，没有持续下降。STOP 强不能证明已接近 Pareto 前沿或最优，只说明初始化/源信息可能贡献较大。

- Traffic 不只是目标均值问题，TLS 获证率也低于 TS 和 TargetOnly。不能未经诊断直接归因为“拓扑相变下随机探索更好”。

- Undebiased、Dense 消融有时更好；两个消融各出现一次假证书，不能证明去偏/正交/残差项对安全保证必要。独立门控对任意固定候选名单都成立。

### 7.3 统计推断对象要说清楚

8 个任务×3 种子不是 24 个独立环境。现有配对分析可描述固定 benchmark 的种子平均表现；若推广到同家族新环境，应报告 per-task 效应、按任务聚类的区间或层级分析，优先增加独立任务数。

常规 audit 为 4,000 次 Monte Carlo；“true violation probability”仍是估计值。阈值 0.05 附近标准误约 0.00345，应将 near-boundary 结果分为有把握安全、有把握不安全、尚不确定，或追加独立审计。不能把全局两次经验 FC 与单次 5% 错误认证界直接当成理论紧度或校准证据。

论文保存 Holm p<0.01 的计数写 25/28，摘要实为 24/28；这是应修正的小错误，不应喧宾夺主。

## 8. 理论与 Lean：条件引理和实际算法之间的缺口

本次提交有 4 个 Lean 文件及一份验证 receipt，报告列出 19 个已检查声明。声明通过检查的含义是：在其明示假设下，结论能由形式系统推出。它不自动证明 Python 计算满足假设，也不代表 19 项新理论贡献。[证明目录](https://github.com/erzhu419/KG_op/tree/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/proof/SCOLHKG/V2OR)；[论文理论节](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/manuscript_v2_or/sections/05_theory.tex)

| 主张 | 数学上成立的部分 | 当前未闭合的部分 | 建议表述 |
| - | - | - | - |
| 历史日志去偏 | 已知密度比下的重要性加权恒等式/截断偏差 | 代码 affinity 倒数不等于已知 dν/dq；自适应依赖、支持覆盖、归一化误差未处理 | 经验占据平衡 |
| 白化与稀疏低维学习 | 源样本上的 Gram 白化；按系数稳定性筛选 | 裁剪后、目标分布上的正交不自动成立；有限样本率假设不足 | 学习结构的候选机制，需样本外诊断 |
| R_perp 安全包络 | 给定足够 coefficient caps 的条件平方界 | 代码经验分位数不满足目标上界条件，高频项与交叉项未覆盖 | 经验残差惩罚，暂不作为安全保证 |
| 组内方差解耦 | 对共同均值平移不变 | 不等于有限样本方差预测校准 | 正确的结构性设计依据 |
| 退火消除负迁移 | 已知真实 shift 的标量 oracle 收缩界 | 实际 MLL 网格不是 oracle；多维自适应设计未连接 | Oracle 动机，不是实际算法保证 |
| V-KG 非负及精确认证感知 | 指定 tower 条件下的最大值期望不等式 | 当前效用、近似概率、求积、重估参数不自动满足该条件 | 近似认证导向前瞻采集 |
| 分布无关安全 | 冻结名单、独立同分布终验下的错误认证 FWER | 非认证 fallback、分布漂移、认证条件风险不在此结论中 | 精确写出事件与概率所对应的分布 |
| 摊销上界 | 给定正成本节省时的代数公式 | 平均样本成本差不是节省下界，未匹配质量 | 指定比较条件下的摊销点估计 |

### 8.1 R_perp 的具体反例

代码对源系数平方取 80%/85% 分位数，并加 0.25 倍源拟合残差 MSE 中位数。Lean 的剪枝平方界要求对每个目标剪枝系数都有 energyCap_k ≥ |P| θ_k²。[先验第 380–389 行](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/v2_or/priors/debiased_spectral_prior.py#L380-L389)；[Lean 条件](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/proof/SCOLHKG/V2OR/DebiasedSubspace.lean#L216-L222)

取两个剪枝系数都为 1，考察点上两基函数值也都为 1。经验 cap 各为 1，则代码主要给出 1+1=2，但被剪去函数的平方为 (1+1)²=4。全局正交不意味着某一点交叉项消失。完整残差还含高频部分，与剪枝和的交叉项也需控制。

即使机械补上 |P|，目标系数可超出源 85% 分位数、高频残差未受界等问题仍在。最经济的处理是先将它降级为经验不确定性惩罚；门控安全定理可以独立保留。若未来把它作为主理论，必须明确源—目标任务类、覆盖、shift 和上界置信度。

### 8.2 Oracle 退火界没有证明实际 MLL 选择器

标量理论的核心是：

```math
\mathrm{MSE}(\tau;\Delta)=\frac{\tau^2\Delta^2+\rho}{(\tau+\rho)^2}.
```

若知道真实 shift Δ，选 tau\*=Δ^-2，确可达到优于两个端点的 oracle 风险。但代码选择的是有限 beta/shift/local-kernel 网格，由正则化边际似然打分；没有证明它逼近该 oracle。[实际网格](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/v2_or/models/tempered_hvd_posterior.py#L229-L243)

例如 tau=.2、rho=1、Δ=10，公式风险约为 3.472，高于 target-only 的 1。这个例子并非说当前算法一定取该值，而是说明“存在理想温度”不能推出“实际估计温度统一无负迁移”。多维时源协方差与目标信息矩阵也不一定同时对角化，单一温度不能自动逐方向实现 oracle。

可保留此界作为理论动机。若要声称实际保证，需要证明真实数据驱动选择器的 oracle inequality，加入选择误差、源估计误差、表示残差与失配项，并说明自适应目标采样条件。

### 8.3 维度无关有限样本率尚未得到完整证明

论文给出约 r log M+ /(M_src n_eff) 加截断残差的表达，但 companion 的证明概述未处理：估计对象是经验源均值还是总体元均值、跨任务变异、自适应轨迹依赖、估计权重、数据驱动基选择、ridge 偏差及目标 shift。

若估计总体元均值，即使每个源任务的系数已无限精确，有限源任务数仍留下约 Var(θ)/M_src 的任务间不确定性；它不能仅随每任务重复样本增加而消失。若估计对象只是固定源任务均值，则需明确限定，不能直接解释为新任务保证。

### 8.4 V-KG 用的不是精确门控通过概率

代码通过正态 margin 构造 p_pass，未显式使用各槽 delta；实际门控却用精确二项检验。以正态约束、均值方差已知且其他残差/认知不确定性为零的边界策略为例：p=0.05 时 margin=mu+z_0.95 sigma=0，代理给 0.5。对 K=112、delta=.025，真实门控最多允许 1 次违约，边界通过概率仅约 0.0220578，相差约 23 倍。一般非高斯约束下 p=.05 与该正态 margin=0 并不等价，这反而需要另查代理校准。[采集实现](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/v2_or/acquisition/verification_kg.py#L59-L120)

实际代码还包含正部效用、固定超参数 fantasy、25 点求积、非负截断与额外探索奖励；Lean 的 tower equality 前提并未因此自动满足。这不否定近似采集可用，但“精确认证概率”和“已证实现 KG 性质”需要降级。

一个值得先小范围比较的改进是让采集真正使用终验功效：

```math
k^*(K,\alpha,\delta)=\max\{v:F_{\mathrm{Bin}(K,\alpha)}(v)\leq\delta\},\qquad C_K(p)=F_{\mathrm{Bin}(K,p)}(k^*).
```

再对 p(x) 的后验积分，并在统一的部署/弃权效用中考虑搜索、重复样本和验证预算。这是研究候选，不是当前已完成贡献。先检查它是否改变候选排序及认证收益，再投入复杂算法。

## 9. 四类仿真器：进步成立，报告存在具体错写

以下主要是报告与固定代码的核对，不能把文案失配全归为模型不可用。多数简化模型仍可作为机制基准；应写成实际实现的样子。

| 家族/接口 | 报告所写 | 当前实现 | 含义 |
| - | - | - | - |
| 公共 psi | 18 维 | 各家族 12 维 | 统一参数和信息接口说明 |
| Inventory 提前期 | 常规 3/4、加急 1 | 常规 1 期、加急当期 | 任务动力学不同 |
| Inventory 需求 | 对数正态与促销脉冲 | Gaussian macro/idiosyncratic 乘性扰动并截下界 | 不应按另一种尾部机制解释结果 |
| Inventory 缺货 | backorder 跨期积压 | 缺货损失，库存截为非负，没有积压状态 | 属于 lost-sales 风格系统 |
| Queueing | 完整多服务台 DES 的措辞 | 时段聚合/流体式队列，Gamma–Poisson 到达 | 动态性真实，但抽象层次需准确 |
| OPSD 控制 | 充放电与备用双通道，H=168 得 d=336 | 单通道 hourly reserve/SOC target，d=hours=336，即 14 天 | 控制对象和维数解释要重写 |
| OPSD 随机数据 | 负荷、风光净误差，多处所列市场 | 主要为 load_actual−load_forecast；实际市场清单不同 | 不能外推为完整风光净负荷风险结论 |
| Traffic DES 路网 | 真路网拓扑、相位+offset | 真实相位边界，加人工 i+2/i+3 环形 routing | 应称有现实参数的风格化交通模型 |
| Traffic 控制维数 | 21×2+2 协调参数 | 44 个有效相位时长参数 | 不能解释为两个全局 offset 参数 |
| Traffic 来源规模 | 22/30 维、10/14 路口 | 23/31 维、11/15 路口 | 从 manifest/var_map 自动生成 |
| Traffic 1.30 shift | 公共需求放大 1.3 倍洪峰 | 主要改变潜在漂移与随机尺度 | “洪峰拓扑相变”只是未验证解释 |
| Traffic 指标 | HBEFA3、Atkinson 机制 | DES 为简化线性排放项和 CV 风格公平项 | 与真正 LiveSUMO 指标分开 |

[库存实现](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/v2_or/simulators/inventory_des.py)；[排队实现](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/v2_or/simulators/queueing_des.py)；[OPSD 实现](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/v2_or/simulators/opsd_storage.py)；[交通实现](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/v2_or/simulators/traffic_signal_des.py)

### 9.1 OPSD 的数据划分与认证分布需要专门处理

代码以非重叠 48 小时块按模 3 分到 search、verify、audit 池；没有论文所说的 d 小时隔离缓冲，相邻时块仍可能属于不同池。每个池内的 bootstrap replication 可以按其定义成为 IID，但三个经验池对应的分布不同。

独立性与同分布是不同要求。二项门控控制验证池下的 p_ver，不自动控制审计池或未来电网环境下的 p_deploy。可选择：明确只认证冻结经验分布，并把跨池表现当外部分布验证；或提出可核验漂移界，在 p_deploy≤p_ver+epsilon 下对 alpha−epsilon 认证。不能在无额外条件下把跨池 held-out 成功等同于未来分布安全保证。

实际 source zones 为 DK1、NO1、NO2、SE1、IT_CNOR、IT_CSUD；target zones 为 DK2、GB_GBN、IT_NORD、IT_SUD、NO3、NO4、SE3、SE4。公开名义信息与目标私有残差的时间使用也应在任务合同中明确。误差截在 [-0.20,0.20] 会改变风险尾部，应给出截断频率与保留原始尾部的敏感性；“异常值一定是仪表 dropout”需要数据证据。

### 9.2 公开信息隔离测试不等于表示充分性

已有小型 wiring tests 有用，不能证明所有 leakage、参数配置、概率性质或物理动态均正确。尤其 OPSD 测试主要检查调用/流程状态，不能替代对私有数组扰动后的不变性检查。

最重要的新科学诊断是：在 psi 相近的策略中，是否仍有大幅成本/风险差；时间置换、路网节点重标号、局部高频控制是否使摘要不变而真值显著改变。对跨路网问题，简单按数组次序做 DCT 没有天然拓扑语义；应使用固定可解释排序或图结构表示，并以实验决定是否必要。

## 10. LiveSUMO：应保留失败结果，另建可达的验证场景

[LiveSUMO compact summary](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/paper_artifacts/v2_or_live_sumo_v1/compact_summary.json) 显示：5 个方法×4 个任务配置×2 个种子，共 40 次运行；获证 0/40；各方法最终返回候选的审计经验违约率均为 1.0。

| 方法 | 保存的 Raw True Obj | 获证率 |
| - | -: | -: |
| TLS | 0.964230345 | 0% |
| Target-SPSA | 0.964532168 | 0% |
| SCBO | 0.982645138 | 0% |
| STOP | 0.987568323 | 0% |
| RGPE | 0.988812553 | 0% |

所有方法的惩罚后分数都为 1.296183424。TLS 与 SPSA 的 raw 差只有约 0.000302，约 0.031%，缺少对应配对区间；且 raw 不是固定 primary 的成本。无法据此主张强微观优化优势。

每候选 audit 仅 40 次。40/40 违约是很强的不安全信号，但不是数学上已知 p=1；作为例子，其单侧 95% 二项下界约为 0.928。它离 0.05 边界很远，所以“没有发出假证书”主要说明认证器会拒绝明显不安全方案，不能检验近边界功效，更不是安全部署成功。

四个 target 配置主要改变权重/阈值，共用路网与 route 文件；不是四个独立道路网络或交通需求日。Live 实现确实调用真实 libsumo，但仍复用 Final_Submission 的一份仿真 wrapper；打包环境和数据路径应明确。16,640 对应整套 search+verification 仿真调用所代表的模拟小时，不能当作已用 CPU 小时；额外 audit 调用要另外记账。

### 10.1 “Windows/Linux 排放偏移导致不可行”目前是待验证假设

不能只凭两个平台基准值不同就锁定病因。应固定网络、route、车辆 emission class、SUMO/libsumo 版本、随机种子、排放参数、启动选项与 horizon，逐项比较固定配时的输出。保留旧失败结果。

若业务约束定义为“比当前平台固定配时基准低某比例”，独立校准 E0_linux 是合理的；如果约束是绝对排放上限，不能为获得认证而放宽 E0。新的校准 cohort 和确认 cohort 必须分开，新的运行不能继续称为原先盲测。

### 10.2 下一组微观实验应回答什么

先独立确认存在可行区域和至少一个有解释的基准策略，再在有余量、近边界及不可行场景中检查目标成本、获证功效、全额预算和简单对照。最少保留固定配时、SPSA/低频 pattern search、同表示 GP+TS、学习先验+TS，以及只有在有额外价值时才保留 V-KG。

真实应用不一定必须选 SUMO。若修好可达场景代价过大，OPSD 可以成为主要数据应用，但必须先解决任务分布、时间划分、尾部处理与物理控制解释。一个扎实的外部应用优于多个无法支持决策结论的标签。

## 11. 摊销、图表和可复核性

### 11.1 摊销数字是指定比较器下的点估计

图表生成脚本固定取 TLS N=15 与 SCBO N=60，先四舍五入均值调用数，再用 max(B_cold−B_tls,1)，最后计算来源成本除以节省量。得到约 9/14/14/15 次部署的家族点估计。[脚本第 178–204 行](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/v2_or/dispatch/generate_or_tables_and_plots.py#L178-L204)

它不是“保守上界”：尚未证明 comparator 必须 N=60 才达到相同质量，样本均值差不是总体节省下界，且 max(...,1) 会把无节省也处理成正节省。正确的经济比较应固定部署质量与认证功效，寻找各方法成本前沿，再计算摊销。节省的置信下界不为正时，不报告有限的保守 break-even。

每家族 source calls=1,428；468 是 logged evaluations，不是 468 条求解轨迹，实际为 6 个任务×4 个求解器=24 条轨迹。来源日志既有情况下，可分别报告 sunk-cost 和 full-cost 场景；即使日志免费，模型拟合与集成计算也不是零成本。DES 与微观仿真单次耗时不同，跨保真度时不能只按 call 数直接换算业务费用。

### 11.2 Figure 2 需要重画，但不要夸大为所有结果都被隐藏

Figure 2(a) 使用人为 inflation multiplier。Figure 2(b) 挑选 Frozen、Untempered、STOP、Undebiased、TargetOnly，遗漏内部强对照 TS 与 Dense；TargetOnly 取 N=15，其余取 N=60，图内有标签但正文/图注口径不一致。Table 2 实际列出了所有主要消融并标出最小值，因此不能说负面结果全被删掉。

建议统一预算，展示所有机制臂与负增益，分家族给效应与区间。Figure 1 主要随在线预算 N 变化，不能仅凭文件名把它解释为 horizon scaling；验证维度增长需在控制任务结构与预算后专门设计。

### 11.3 发布凭据需补全

这次提交仅有各 campaign 的 compact summary、三个 jobs plan，以及 Lean receipt；未提供完整 per-cell outcomes、逐槽 certificates、source archive/prior bundle、运行期源码/数据 hash 绑定。三个 committed manifests 仍是 sealed_complete\:false。

这说明独立复核冻结顺序的证据不足，不说明发生了确认集调参或造假。服务器副本可能有更完整记录；若当时未保存执行 hash，应像旧版发布说明那样明确“历史执行版本不可追认”，不要用最终发布 hash 追认。

下一版最小可复核包应包含：环境锁、代码与数据/manifest hash、源档案摘要及 hash、每运行结果表、每槽证书摘要、完整成本与失败 ledger、固定生成表格的入口。大型轨迹可以留在集群，用稳定路径与校验和绑定。主 README 仍指向 10 月旧主线，需新增简洁 v2 入口而保留旧材料。

## 12. OR 主线应如何聚焦

我建议把研究问题固定为：

> 能否从以往求解器留下的有偏优化日志中，学习可迁移的响应函数结构，使新的动态运筹任务在更少的全额仿真成本下达到给定部署质量与认证成功率，并识别何时不值得迁移？

这保留了高维耦合与函数元先验的原始内核，同时把价值落到运筹决策：哪些历史记录值得保留、需要多少来源任务、何时迁移、何时回到目标学习、多少重复样本用于搜索或认证、需要多少重复部署才能收回离线成本。

标题和贡献不必绑定 KG。建议暂以“从历史优化日志学习函数先验的认证仿真优化”作为工作题目。若同表示 TS 更稳定，函数先验+TS 仍可承载核心研究；不能为了保住名称继续叠加未经证实的采集机制。

### 12.1 推荐一条主贡献，其他作为支撑

首选主线是有偏日志中可迁移函数结构的学习与适用边界。实证要证明：源表示/系数分布在公共名义动力学、初始点及候选模板之外仍有决策增益；效应随日志覆盖、任务差异和目标预算如何变化；负迁移如何被诊断或抑制。

若机制诊断显示收益主要来自认证功效与预算分配，则可以改以“在搜索、重复观测和终验之间分配全额预算”为主贡献，函数先验成为支撑。不要在证据不充分时把两个方向都写成已经完成的主创新。

理论只选一个与最终算法直接对应的问题：带覆盖/shift 条件的函数先验误差分解，或实际温度选择器的风险界，或真实门控功效下的预算决策性质。组内 SSQ 平移不变、Cauchy–Schwarz、union bound、摊销代数作为支撑性结果即可。继续增加这些基础引理的形式化数量，不能代替核心贡献。

### 12.2 本项目建议采用的投稿门槛

这些是针对当前主张的研究判断，不是 OR 官方硬性清单：

1. 部署/弃权和成本定义一致，结论不依赖比较阵容；所有数字可从逐运行账本生成。

2. 在相同表示、初始化、候选机制和预算下，函数元先验对简单且强的对照仍有实用增益，或得到明确、可解释的适用边界。

3. 至少一个有充分现实依据的应用提供可认证成功及决策价值；若只保留压力测试，论文必须相应缩小应用主张。

4. 至少一项新机制的理论/实证贡献清晰成立，其余模块不冒充必要条件；数学假设与实际实现逐项对应。

5. 负迁移、失败、成本、来源和目标数据划分均完整公开，第三方可在抽样误差范围内复现核心结论。

满足这些后再评估投稿，而不是用“4,410 次运行、19 个 Lean 声明、7 个外部名字”当作就绪程度的替代指标。

## 13. 下一步执行方案：先修账，再识别贡献，再扩大验证

### P0：修复当前证据与文稿，不扩大科学实验规模

| 工作包 | 具体产出 | 验收条件 |
| - | - | - |
| P0-1 决策合同 | 明确 certified deployment、abstention、fallback 三种状态及损失函数 | 每个 deployment 对应一个实际策略；未认证候选不隐式变成安全回退 |
| P0-2 指标与账本 | 从服务器保留的 per-run 文件生成统一结果表；保留旧表供追踪 | 比较阵容变化不改变已有运行结果；STOP 跨展示预算一致 |
| P0-3 统计重汇总 | 任务层效应、区间、获证率与条件成本、边界 audit 不确定性 | 能分清固定任务种子表现与新任务泛化 |
| P0-4 基线与 adapter | 通道分别插值；自写近似方法诚实重命名 | 常量双通道策略跨维保持通道语义；各名称匹配实现 |
| P0-5 理论收缩 | claim→assumption→Lean→Python 对照表 | 无“经验分位数自动成安全上界”“MLL 自动继承 oracle”等跳步 |
| P0-6 文稿自动化 | 从 manifest/代码导出任务表、K、delta、维数、样本与来源数量 | MD、Tex、表格和代码参数一致 |

这一阶段保留所有既有结果，不把旧确认数据重新命名为新盲测，不用最终源码 hash 追认历史执行版本。若完整结果已不可恢复，应明确哪些结论只能视为初步，并在下一次预先冻结的新 cohort 中补足。

### P1：最小机制分解，优先 Inventory 与 Queueing

这两类先用于开发诊断：动态耦合明确、运行成本相对可控、当前有正面信号。把已经看过的配置当开发数据；最后才用新的独立任务确认。

推荐以下顺序化比较，不需要完整 2^k 全因子：

| 臂 | 源信息及模型 | 主要回答的问题 |
| - | - | - |
| A 公共表示基线 | 公共 psi + target GP + TS，公共初始化/候选机制 | 名义动力学表示本身有多强 |
| B 仅源起点 | A 加来源 warm start，其他不变 | 收益有多少只是起点 |
| C 源表示 | B 使用源学习字典，但采用弱/目标估计系数先验 | 学到的函数表示有无额外价值 |
| D 完整函数先验 | C 加源系数均值/协方差，保持初始化/候选机制 | 核心函数元先验的独立价值 |
| E 模板候选 | D 加源 template proposals | 搜索空间引导的独立价值 |
| F 认证前瞻 | E 将 TS 换成当前 V-KG；另小样比较真实 gate-power 版本 | 前瞻是否优于简单采集 |
| G 简单强对照 | 同样源 warm start/策略参数化 + pattern search 或 SPSA | 收益是否需要复杂后验与采集 |

C 不能叫“完全 target-only”，因为字典仍使用了来源信息。每个关键比较保持其余通道一致；实际完整系统另与官方/忠实外部基线比较。先看 D−C、C−B、B−A 和 F−E，决定是否值得追加去偏、白化、稀疏、噪声模块的细消融。

首轮 A–F 固定同一套仅使用目标组内重复观测的噪声估计器及正则化；源 HVD 方差先验另作一个单独开关。否则 D−C 仍可能夹带来源噪声信息，不能解释成系数函数先验的独立价值。完成分解后再恢复默认完整系统作系统层比较。

预算按真实仿真调用和耗时控制，至少同时报告 search calls 与 search+verification calls；N=15/30/60 只是算法步骤，不是完整成本。验证器对全部方法一致。源拟合成本与在线成本分别记账，避免 TLS 预训练被移出计时、RGPE 源训练仍计入在线时间。

P1 同时做三项小而有决定性的诊断：表示碰撞/时序信息损失、独立重复样本方差校准、预测认证概率与实际二项功效校准。这些都在仓库规定的 CPU 集群执行，形成轻量摘要后再分析。

**决策门槛：** 若 D 相对 C/B 没有稳定实用增益，就不能继续把学习系数分布写成主贡献；若 C 有增益，可聚焦表示学习。若 TS 持续与 V-KG 相当或更好，保留 TS，停止为 KG 名称维护复杂度。若所有优势都由 B 解释，就围绕来源起点/策略复用重新评估研究新颖性，不能依靠增加实验数量掩盖机制结果。

### P2：修复外部比较，做一个有成功部署的应用

选择 2–3 个最相关的忠实方法，而不是继续增加七八个低保真名称。对高维受约束优化选 SCBO/SAAS 中适合计算预算的一项或两项；对迁移函数先验选 RGPE 或 HyperBO；同时保留同表示 target GP+TS/cEI 和简单运筹启发式。

将同信息的机制比较与各方法自然设置的系统比较分别报告。对来源数据、维度适配、调参预算、初始化、候选数、重复样本与终验预算作清楚合同。不能给自己的方法额外结构信息，却把对照的高维困难全部解读为函数学习优势。

外部应用按第 10 节修复可行性、固定平台、定义参考值与目标分布，先独立校准，再冻结新确认集。优先增加不同需求日、市场、任务几何或源/目标偏移，避免只在同一个环境上增加算法种子。

预先定义主要比较、最低有意义效应、统计单位、预算与终止规则。阈值应由实际模拟成本和业务价值决定，不应事后选取恰好显著的家族或预算。修复后出现不显著或负迁移，应保留并给出适用边界。

### P3：选择一个理论任务，并与实现同时完成

建议在 P1 看清机制后选择：

- 如果函数先验有独立增益：定义公共策略表示的近似误差、日志覆盖条件及源—目标 shift；证明源估计与目标条件更新的误差分解，至少避免把总体元均值不确定性漏掉。

- 如果温度选择是关键：对实际可实现选择器给风险比较，不用不可观测真实 Δ 的 oracle 替代算法；允许模型选择和失配余项。

- 如果认证功效驱动优势：用真实二项功效定义终端效用，研究重复采样/验证预算决策，并显式保留弃权损失。

R_perp 在真正满足覆盖之前保持经验项身份。终端错误认证保证单独证明，清楚限定 IID、名单冻结、目标分布与多槽错误预算。Lean 用于检查最终定理，不反过来围绕容易形式化的引理拼论文。

### P4：重写论文与最终投稿核查

建议结构：运筹问题和日志来源 → 学习对象及信息合同 → 可实现算法 → 一项核心理论与适用条件 → 同条件机制实验 → 真实应用与失效边界 → 全额成本与管理启示。

管理启示必须来自可核验实验：哪些日志覆盖值得保留、何种偏移下应弱化先验、搜索与认证怎么分配、何时经济上值得迁移。不要把“9 次摊销”“任意 shift 无负迁移”“0 假证书”当成已证普遍政策结论。

最终只保留能由代码、结果和论证共同支持的贡献。若下一轮证据支持函数先验，OR 方向可以继续投入；若只剩工程组合且机制增益无法识别，应及时缩小主张和投稿定位，而不是再整体重构一遍。

## 14. 对 Gemini 报告六个问题的直接回答

| 报告问题 | 回答 |
| - | - |
| 1. 旧硬伤是否彻底解决、现在是否够 OR | 没有。工程方向前进，但旧历史描述部分不准确；当前指标、基线、理论连接与应用证据不足，不建议立即投稿 |
| 2. 定理和 Lean 是否存在漏洞或过头 | 有。最关键是 R_perp 条件、实际 MLL 与 oracle 的断点、密度比与经验权重差异、有限样本率缺项、V-KG tower 条件及安全事件范围 |
| 3. Traffic 上 TS 更好，是否加自适应混合 | 先不要。TS 均值更好不只发生在 Traffic；先控制表示/候选/初始化，检查代理概率与覆盖，再决定是否需要混合 |
| 4. LiveSUMO 零伪证书和 raw 改善是否足够 | 不够。0/40 获证且明显不可行，raw 字段不是固定 primary，与 SPSA 差很小；可补独立平台校准后的可达场景，但不能随意放宽绝对约束 |
| 5. 基线和摊销是否完备公平 | 数量多，但身份与 adapter 不够；缺少简单强对照、同表示机制比较、质量匹配的成本前沿。摊销是点估计，不是现成保守上界 |
| 6. 投稿前文字/图表怎么改 | 先做 P0 与贡献诊断，再润色；最高优先级是指标、名称、理论范围、真实数据/动力学描述和统一预算图表 |

## 15. 文件组覆盖与关键证据索引

下表覆盖本次全部 68 个新增文件组，区分科学逻辑和生成产物；旧目录主要沿方法身份、原始目标、源学习、失败诊断及发布账本追踪。

| 新增组 | 文件/数量 | 审阅重点与结论 |
| - | - | - |
| 契约 | contracts.py，1 | 充分统计合并、隐私接口、审计定义；true 字段仍为 MC 估计 |
| 仿真器 | inventory_des.py、queueing_des.py、opsd_storage.py、traffic_signal_des.py，4 | 动态机制、公共表示、数据池与 Live wrapper；见第 9–10 节 |
| 来源日志 | legacy_solvers/solvers.py，1 | 真实自适应日志成立，求解器名称需收缩 |
| 先验 | priors/debiased_spectral_prior.py，1 | 真函数学习、占据平衡、白化/剪枝、经验残差项 |
| 目标模型 | models/tempered_hvd_posterior.py，1 | 组内方差、持续更新、MLL 网格、人工 inflation |
| 采集 | acquisition/verification_kg.py，1 | rank-one fantasy、近似 p_pass、候选和初始化混杂 |
| 验证 | verification/gated_verifier_v2.py，1 | 精确二项门控、三槽顺序、失败返回语义 |
| 比较方法 | baselines/sota_baselines.py，1 | 名称忠实度、通道插值、共同信息合同 |
| 运行及报告 | dispatch 下 campaign_runner.py、generate_or_tables_and_plots.py、run_v2_lean_cluster.py、sync_v2_cluster.py，4 | 执行路径、账本、统计口径、图表及集群工序 |
| 小测试 | tests/test_v2_or_wiring.py，1 | smoke/wiring 价值有限，不能替代科学复核 |
| Python 包入口 | 各 **init**.py，9 | 包组织，共计 Python 25 个 |
| 协议 | pilot、confirm、live_sumo manifests，3 | 任务、预算、种子、数据分工；冻结状态证据不足 |
| 结果 | 三个 compact summaries、九个 stage jobs plans、一个 Lean receipt，13 | 汇总数字、执行计划、形式化回执，缺逐运行证明链 |
| 论文 | main_or.tex/pdf、sections 01–09、四张表、三张图、bib、cls、bst、logo，22 | 叙事、数值、定理范围与生成脚本交叉核对 |
| 重构报告 | GPT6_PathB_OR_Reconstruction_Report_20261116.md，1 | 历史、参数、算法身份及结果解释的交叉核验 |
| Lean | DebiasedSubspace、TemperedHVD、VerificationKGAmortization、V2ORProofInterface，4 | 假设、声明结论与 Python 实现的连接 |

历史最重要的保留材料：

- [原始目标与主线合同](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/docs/original_goal_and_mainline_contract_20261015.md)：少量目标观测改变未采样策略判断，并改善预算内决策。

- [旧来源函数模型](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/performance/mainline_legacy_source_models.py#L178-L256)：来源响应学习已存在。

- [旧公开 exposure](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/representation/observable_exposure.py#L83-L155)：公开策略统计与私有真值访问应区分。

- [旧 59 倍诊断](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/docs/mainline_factorshock_diagnosis_results_20261015.md)：不能将单开发案例推广成全部实验机制。

- [旧源留出诊断](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/docs/mainline_role_source_holdout_results_20261016.md)：source0 与方差先验问题。

- [旧成本账本](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/reproducibility/mainline_release/COSTS.md)：来源和失败成本并非过去完全未记。

- [旧发布说明](https://github.com/erzhu419/KG_op/blob/d1a13a40a38d396ef97edb9c2fc83d5adf532d4c/SC-OLH-KG/docs/mainline_submission_release_20261026.md)：未知历史执行版本应明确标记，不追认。

建议交给下一轮实现者的第一条任务是：先完成 P0-1 至 P0-3，给出修正后逐任务结果，再开始 P1。现阶段最值得保护的是已经做实的函数元先验和动态仿真底座，最需要放弃的是超出证据的结论。

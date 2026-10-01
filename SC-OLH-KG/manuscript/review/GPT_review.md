# GPT：
我审查了 `SC-OLH-KG/manuscript` 的全部正文、附录、生成表格、final-method manifest、配对统计工件，以及合成问题的主要代码定义。以下不讨论 OR 模板、排版和图形美工；图只评估“缺什么、哪些重复”。

## 总体判断

这篇论文已经**不是 SC-OLH-KG/KG/HVD 方法论文**。当前真正的论文是：

[
\boxed{
\text{source-scored structural initial design}
+
\text{replaceable optimizer}
+
\text{independent terminal verification}
}
]

README 也明确规定，canonical SAASBO 是主后端，KG、SC-V69 和 cumulative HVD 只是比较或诊断。

这个收敛是正确的。论文比早期版本干净、诚实、审计性极强，尤其是：

* 主动承认 HVD 改善方差拟合但不改善优化；
* 保留失败的 traffic external gate；
* 保留 energy 中不胜 natural low-frequency grid 的负结果；
* 分离 source、search、verification 成本；
* 不把标准 SAASBO 包装成创新；
* 有 immutable replay、hash、Lean proof map 和完整负结果记录。

但以 **Operations Research** 标准判断，目前仍然是：

[
\boxed{\text{接近“Reject，鼓励重构后重投”，而不是 Major Revision。}}
]

我的主观评分：

* 研究工程与审计完整度：**9/10**
* 论文叙事清晰度：**7.5/10**
* 算法新颖性：**6/10**
* 当前理论贡献：**4.5/10**
* 当前实验外部有效性：**4.5/10**
* OR 投稿成熟度：**约 5/10**

问题不是代码不够复杂，而是论文最强的结果主要来自一个和 benchmark 生成机制高度同构的前端假设；理论尚未真正解释该前端如何从有限 source data 泛化。

---

# 一、最严重的实质问题

## 1. 合成 benchmark 与方法假设高度同构

这是最可能导致 OR 拒稿的问题。

论文把三个合成域描述成不同 operations mechanisms，但它们本质上都被人工构造为少数平滑摘要：

* FactorShock 主要依赖 head level、tail mean 和 spread；
* Inventory 依赖 stock、reorder、safety 三个 block mean 和 dispersion；
* Queue 依赖 capacity、priority、smoothing 三个 block mean 和 imbalance。

代码中更加明显。HighDimStatePolicy 的全部目标与约束只依赖：

[
(u,q,\mathrm{spread}),
]

而推荐和 anchor 又直接生成 constant-tail policies。

Inventory 和 Queue 也是将 (d) 个变量分成三个块，只使用三个块均值与一个 dispersion；其候选和 oracle 又主要遍历 constant three-block policies。

与此同时，方法前端正好使用：

* constant profiles；
* monotone profiles；
* low-frequency profiles；
* DCT 低频系数；
* source 中共享的64个 normalized profiles。

所以 (d=1{,}000\rightarrow10{,}000) 基本没有增加统计难度，只是增加同一块内重复坐标数量。有效维度一直约为 3–4。

因此下面这些表达会受到攻击：

> high-dimensional optimization
> (d/N=1000)
> succeeds at dimension 10,000

它们在字面上正确，但会被认为利用 nominal dimension 制造震撼感。

### 必须修改

标题和摘要应限定为：

> high-dimensional **ordered policy-profile** optimization

或者：

> functional/profile decisions represented on an increasingly fine grid

更准确的标题例如：

> **Transferable Structural Initial Designs for Chance-Constrained Optimization of High-Dimensional Policy Profiles**

同时增加真正改变有效复杂度的测试：

1. block 数量增加，而不只是每个 block 长度增加；
2. safe basin 依赖随机低频组合，而不是固定 block mean；
3. source 与 target 的频率支持不同；
4. 坐标 permutation；
5. irregular/nonuniform grids；
6. piecewise-smooth profiles；
7. 少量高频 active coordinates；
8. deliberately misspecified target。

否则 (d=10{,}000) 结果最好降为“离散化分辨率不变性测试”，不能叫高维搜索突破。

---

## 2. 外部实验只支持“低频结构有效”，不支持“source learning 有效”

energy holdout 是论文最重要的外部证据，但结果是：

[
\text{atlas wins}=9/20,\qquad
\text{natural low-frequency grid wins}=11/20.
]

差异不显著，atlas 的中位目标还略差。

manifest 也明确禁止主张 source atlas 优于 natural structured control。

因此，当前真实证据链是：

[
\boxed{
\text{low-frequency structural compression有效}
}
]

而不是：

[
\boxed{
\text{source outcomes 提供了超越一般结构先验的价值}
}
]

但标题、主贡献和摘要仍然把 **source-learned transferable atlas** 放在中心。这会形成核心主张与外部证据不匹配。

### 必须补的证据

需要至少一个未参与方法开发的真实系统，其中：

[
\text{source-trained atlas}

>

\text{generic structured design}
]

而不仅是：

[
\text{structured design}

>

\text{Sobol}.
]

或者主动收窄论文：

> 本文贡献不是 source learning，而是把历史任务转成一种 structured initial-design protocol；source ranking 在合成域中有效，但外部证据仅支持结构压缩。

后一种更诚实，但 OR 创新性会降低。

---

## 3. LODO 不是严格意义的 domain holdout

final replay 的 target outcomes 确实没有进入 source atlas，审计合同对此非常严谨。

但是方法本身经历了大量版本迭代，而这三个 synthetic domain 长期参与了：

* representation 设计；
* low-frequency prior 选择；
* profile library 设计；
* source/target schema 设计；
* (K\le8) 的选择；
* universal support 选择；
* maximin penalty 和 scalarization 选择；
* verification budget 选择。

所以这只是：

[
\text{outcome-level LODO}
]

而不是：

[
\text{method-development-level domain holdout}.
]

审稿人会说：

> You did not use the held-out outcomes in the final run, but the benchmark family shaped the algorithm during development.

这在 meta-learning 和 transfer learning 论文中是一个严重问题。

### 需要做什么

在方法、阈值和代码完全冻结后，新增：

* 至少3个全新 synthetic generators；
* 最好由另一位合作者设计；
* 或由预注册随机 generator 生成；
* 不允许根据结果修复 atlas。

当前 immutable replay 只能证明“最后一次没有数据泄漏”，不能证明“方法没有 benchmark overfitting”。

---

## 4. 论文没有精确定义其核心算法

Section 4 目前更多是叙述，而不是可复现的数学方法。

缺少明确内容包括：

1. “frequency-penalized selection”具体优化什么；
2. 哪些 DCT coefficients 被选中；
3. diagonal quadratic expansion 的完整形式；
4. universal support 的完整生成规则与规模；
5. “robustly source feasible”的精确定义；
6. objective/risk percentile 如何计算；
7. 五个 scalarization 的准确权重；
8. source risk 和 objective rank 的 maximin penalty；
9. tie-breaking；
10. endpoint replacement 替换哪一个 atlas member；
11. target dimension inverse map 的误差；
12. target schema 到 profile channel 的映射。

当前 Atlas Algorithm A1–A6 只是高层流程，不足以让读者重新实现。

### 必须增加

给出正式 Algorithm 1，完整定义：

[
s_q(p),\quad r_q(p),\quad
S_\lambda(p),\quad
d_\eta(p,p'),\quad
\operatorname{score}_{\mathrm{maximin}}(p).
]

并把所有固定常数放进一张 method specification 表，而不是散落于代码和 manifest。

---

## 5. 核心理论现在过于接近“将成功写进假设”

coverage theorem 假设：

1. atlas 覆盖 source library；
2. learned coordinate 接近 ideal coordinate；
3. source support 中存在靠近 target safe center 的点；
4. target margin 在该坐标中 Lipschitz；
5. safe center 足够深。

然后推出 atlas 中有可行点。

逻辑正确，但数学本质是：

> 如果 atlas 足够接近一个足够深的安全点，那么 atlas 有安全点。

这并没有解释：

* 为什么 source ranking 能找到这样的 support；
* 为什么 DCT coordinate 逼近 ideal coordinate；
* 为什么 (r_{\mathrm{cover}}) 随 source 样本数减小；
* 为什么 maximin atlas 是好的；
* 为什么 source-target shift 应该小；
* 需要多少 source tasks、profiles 和 replications。

post-run coverage audit 又使用 target truth 计算 safe radius 和 support shift，因此只能作为解释，不能作为可操作保证。

Lean 证明提高了实现可信度，但这些定理本身主要是：

* 集合补集；
* 三角不等式；
* Lipschitz 推导；
* union bound；
* binomial all-success。

形式化验证不能替代理论新颖性。

### 最值得补的四个理论结果

**第一，DCT 跨维一致性。**
若 raw policy 是连续 profile (h:[0,1]\to[0,1]) 在不同 (d) 下的离散化，证明：

[
\left|\widehat c_{k,d}-c_k(h)\right|
\le \frac{C_k}{d}
]

或相应 quadrature bound。这才真正支撑“dimension-equivariant”。

**第二，maximin/farthest-first 的 covering guarantee。**
在有限 library 上，farthest-first 可给 k-center 的2-近似：

[
r_{\text{cover}}(\atlas)
\le 2r^\star_{n_0}.
]

这能直接连接实际算法与 theorem 中的 (r_{\text{cover}})。

**第三，source ranking 的有限样本误差。**
由64 profiles × 3 replications 推出 source percentile/rank 的误差或错误排序概率，而不是将 source score 当已知。

**第四，task-distribution coverage。**
在 source/target 来自 task meta-distribution 时，给出：

[
\Pr_q{\atlas\cap\mathcal F_q\neq\varnothing}
\ge 1-\delta
]

或 PAC-Bayes/coverage bound。

目前的 no-free-lunch theorem 太显然，建议缩成一个 remark；不要把它列为核心理论贡献。

---

## 6. verifier 的真正统计保证没有在论文中完成

ordered-shortlist theorem 证明的是：

[
\Pr(E_j)\le\delta_j;\forall j
\quad\Longrightarrow\quad
\Pr(\text{unsafe deployment})
\le\sum_j\delta_j.
]

这只是 union bound。真正困难的问题是：

[
\Pr(E_j)\le\delta_j
]

为什么成立？

正文和附录只说 synthetic verifier 使用“一侧 mean/scale tolerance rule”，但没有给：

* 检验统计量；
* 临界值；
* 正态性条件；
* 未知方差处理；
* confidence/tolerance interval 公式；
* sequential stopping 对显著性水平的影响；
* objective guard 的精确公式；
* common random numbers 或 pairing 假设。

这部分必须与 constrained ranking-and-selection 文献直接对接。现有文献已经研究 chance-constrained selection、受约束系统的统计有效选择，以及 simulation optimization 后用 ranking-and-selection “clean up”。([PubsOnline][1])

### 必须增加

一个 candidate-wise theorem：

[
\Pr_x{\text{candidate }x\text{ is certified while unsafe}}
\le\delta_j
]

并完整写出该 test。

还应报告 verifier 的 **power/nonvacuity**：

[
\Pr(\text{certificate}\mid\text{safety depth } \gamma)
]

随 replication budget 变化的曲线。

否则“independent certification”是论文最显眼的主张，却只证明了外围的 union bound。

---

## 7. energy 的 binomial independence 需要更严谨

energy 证明把每个 fresh calendar window 当成独立 Bernoulli trial，然后使用 all-success binomial certificate。

manifest 说明采用年度 chronological split，并使用80个 independent windows。

但论文没有说明：

* windows 是否重叠；
* 如何抽样；
* 是否跨季节同分布；
* 时间序列自相关如何处理；
* “independent”是模拟随机独立，还是统计上的独立；
* 20 seeds 是否复用了同一年度数据。

如果 windows 来自同一历史时间序列，Clopper–Pearson 的 iid binomial 前提并不自动成立。必须使用：

* nonoverlapping blocks；
* block bootstrap；
* mixing assumptions；
* effective sample size；
* 或将证书明确限定为 empirical-window distribution。

否则 external certificate 的统计含义可能无效。

---

## 8. “equal-total-cost”命名目前不准确

main method 的平均 verification cost 为约174次，有的方法甚至超过210次。

但 synthetic “equal-total-cost”表只匹配：

[
384\text{ source} +13\text{ search}=397
]

与：

[
397\text{ target search}.
]

表中没有 verification cost。

所以这不是 equal total simulation cost，而是：

[
\boxed{\text{equal pre-verification simulation cost}}
]

或：

[
\boxed{\text{equal source-plus-search cost}}
]

论文有时加了说明，但 Results 和摘要仍容易让人误解。

### 更合理的比较

必须增加 all-in cost：

[
C_{\mathrm{all-in}}
===================

S+N+V_{\mathrm{safety}}+V_{\mathrm{objective}}.
]

然后比较：

1. 固定 all-in budget 下的 certified success；
2. 获得第一个 certificate 所需总调用；
3. 获得 certificate 且 regret (\le\epsilon) 的总成本；
4. archive 可被 (M) 个 targets 复用时的 amortized cost。

这是非常有 OR 意义的结果，甚至可以成为论文最强的 managerial/operational insight。

---

## 9. transfer baseline 比较没有真正比较 end-to-end transfer methods

所有 transfer methods 都接收了你构造的强 atlas。

因此表格主要回答：

> 给定我们的 initial design，哪个 backend 更好？

它没有完整回答：

> 我们的 transfer method 是否优于其他 transfer method？

FSBO、HyperBO、MetaBO、MALIBO 等方法的一个关键价值可能正是它们自己的：

* prior；
* representation；
* acquisition；
* warm-start；
* initial-design mechanism。

将所有方法强制用你的 atlas，会剥离其原生 pipeline 中的重要部分，同时给它们一个本身已经60/60覆盖的前端。

建议把当前表改名为：

> Common-atlas backend comparison

然后新增：

> Native end-to-end transfer comparison

并让每个方法在相同 source information 和总预算下产生自己的 initial design/queries。

---

## 10. 缺少最关键的 structured baseline

external energy 中，natural low-frequency grid 与 atlas 打平，说明这是必须有的 baseline。

但 primary synthetic matrix 没有充分比较：

* generic DCT maximin design；
* natural blockwise constant grid；
* random low-frequency coefficient design；
* low-frequency grid with 397 target calls；
* learned-search-space baseline；
* prior over optimum；
* functional BO。

相近研究已经包括从历史任务学习搜索空间、直接对 optimum 放 prior，以及在函数空间中进行 Bayesian optimization。([arXiv][2])

目前 universal-support-only 27/60 是有价值的，但它仍只是一个10点 ablation，不能排除：

[
\text{简单、通用的低频结构设计}
]

已经足以在相同 all-in cost 下取得接近结果。

---

## 11. “causal evaluation”措辞过强

论文称其为：

> causal frontend/backend evaluation

但这里并不是一般意义的 causal identification。它是一个受控、配对的 factorial intervention：

* 替换 frontend；
* 固定 backend、seeds、budget、verifier。

这足以支持：

[
\text{within these benchmark instances, the frontend intervention causes the observed difference}
]

但不能支持跨任务的一般因果结论。

建议改成：

> controlled factorial attribution experiment

或者：

> matched frontend–backend intervention.

---

## 12. 统计推断存在 pseudo-replication 风险

60个 observations 是：

[
3\text{ domains}\times20\text{ simulation seeds}.
]

对于算法随机性，20 seeds 是合理重复；但对于“transfer across domains”的主张，真正独立的 domain 数量只有3。

将60个 seed-domain pairs 全部作为 McNemar/Wilcoxon 的独立样本，会产生极小的 (p)-value：

[
6.94\times10^{-18},
]

但这并不能说明方法能泛化到 domain population。它主要说明在这3个固定 benchmark 上，结果对随机 seed 稳定。

建议明确分离：

* **algorithmic repeatability inference**：以 seed 为单位；
* **task generalization inference**：以 domain 为单位。

后者只有3个 synthetic domains和1个 external target，不能做强 population claim。

另外，不同方法 feasible count 不同，而 regret 只在 feasible 条件下计算，所以 conditional median regret 不应直接横向比较。需要增加：

* lexicographic outcome；
* penalized regret；
* feasible-regret profile；
* probability of feasible-and-(\epsilon)-optimal；
* all-in loss。

---

## 13. target schema 是一个非常强的先验，正文强调不足

information contract 允许：

* held-out task-family identifier；
* policy/state exposure schema；
* dimension；
* bounds。

在 synthetic domains 中，schema 实际上告诉了算法：

* 哪些坐标属于 stock/reorder/safety；
* 哪些属于 capacity/priority/smoothing；
* 哪个是 head、哪些是 tail；
* 坐标顺序具有 profile 意义。

这几乎已经给出了 low-dimensional sufficient statistic 的结构。

这不是作弊，但不能叫弱 prior。它应被定位为：

[
\boxed{
\text{descriptor-conditional transfer under known policy-channel semantics}
}
]

必须把 domain-blind、schema-blind、permuted-channel stress tests 放进正文，而不是只在 manifest 中登记。

---

## 14. scalarization 和阈值敏感性不足

论文固定两个目标权重：

[
(1/2,1/2)
]

而 source atlas 又使用5组 objective-risk scalarizations。

需要说明：

* 为什么选 (1/2,1/2)；
* 换权重后 safe basin 是否移动；
* atlas 是否仍覆盖；
* chance level (\alpha) 从0.05变化时是否稳定；
* (n_0=10)、(K=8)、64 profiles、3 replications、(\kappa=0.25) 的敏感性。

当前这些值可能被审稿人视为长时间开发中逐步调出的配置。

---

# 二、逐节修改意见

## 摘要

需要修改四处。

1. “statistically impossible”过强，应改为：

> Without assumptions linking source tasks to target feasibility, no proper finite target-label-free design can guarantee coverage.

2. “three held-out synthetic domains”改为：

> three leave-one-domain-out synthetic families in a frozen final replay.

3. “dimension 10,000”明确是：

> 10,000-point discretization of an ordered low-frequency profile.

4. 外部实验结论应在同一句中说：

> source learning did not outperform a natural structured grid.

当前摘要已经披露负结果，但读者仍可能先被“external holdout confirms advantage”误导。

---

## Introduction

优点是贡献边界很诚实。但 contribution (ii) 把 machine checking 与理论贡献绑定过紧。Lean 应作为可信度增强，不是理论新颖性本身。

Contribution (iv) 的“causal frontend/backend evaluation”应改名。

Contribution (v) 的 HVD 可以移至 appendix；它现在会让读者误以为论文仍有第二条方法主线。

---

## Related Literature

当前文献表明显过窄。

至少需要增加四类最接近文献：

1. **历史任务学习 search space / warm start / optimum prior**。([arXiv][2])
2. **functional optimization / profile decision spaces**。([AAAI][3])
3. **chance-constrained ranking-and-selection 与 constrained sample allocation**。([PubsOnline][1])
4. **simulation optimization 后的 independent cleanup/selection**。([PubsOnline][4])

否则“proposal measure + independent verification”的新颖性无法被准确判断。

---

## Problem Formulation

应把“ordered profile”结构写进 problem definition，而不是直到 Section 4 才出现。

目前 (\mathcal X={0,\ldots,L}^d) 看起来是普通高维 cube，但算法依赖坐标顺序和 profile semantics。这是隐藏假设。

建议定义：

[
x_i=h(t_i),\qquad t_i=(i+1/2)/d,
]

并说明哪些问题属于该类、哪些不属于。

---

## Atlas Section

这是核心方法，当前需要重写为数学定义 + pseudocode。

“source templates observed in every source task”也意味着 source library 本身是共享、预先设计的 profile collection，不是完全从数据中发现。论文应区分：

[
\text{library construction}
\quad\text{与}\quad
\text{source-data ranking}.
]

“source-learned atlas”更准确地说是：

[
\text{source-scored subset of a preconstructed structural library}.
]

---

## Theory

建议主文保留：

1. DCT/discretization consistency；
2. maximin k-center approximation；
3. probabilistic source-ranking/coverage；
4. verifier candidate-wise validity；
5. union-bound composition。

将 elementary finite-atlas no-free-lunch、fail-closed identity 等移至 appendix。

coverage audit 表也应移至 appendix，因为它是 post-run oracle diagnostic，不是 theorem evidence。

---

## Experiments

需要新增：

* domain/schema blind；
* coordinate permutation；
* new unseen families；
* structured low-frequency baselines；
* native end-to-end transfer methods；
* source budget learning curve；
* all-in cost frontier；
* certificate power curve；
* objective-weight and (\alpha) sensitivity；
* wall-clock/computational cost。

尤其是 canonical SAASBO 每轮 NUTS 的时间成本与 proposal-only 的差异，目前没有报告。

---

## Results

建议降低极小 p-values 的突出程度。20/20 versus 0/20 在每个 domain 已经足够清楚，(10^{-18}) 并不会增加可信度，反而暴露 pseudo-replication。

更应报告：

* domain-wise effect；
* exact confidence intervals；
* all-in cost；
* certificate power；
* feasible-and-(\epsilon)-optimal probability；
* source-only versus structured-only effect。

---

## Discussion

这是目前写得最好的一节，尤其诚实承认 external structured control 的负结果和 traffic failure。

但 “useful for managers” 仍然比较泛。OR 不一定要求 managerial contribution，但若保留这段，应给出明确的 operational policy：

> 当预计未来 target 数量超过多少时，384次 source archive 成本值得支付？

可以推导 break-even target count：

[
M_{\mathrm{break-even}}
=======================

\frac{S}
{C_{\mathrm{target-only}}-
C_{\mathrm{atlas,target}}}.
]

这会比泛泛讨论 amortization 更有 OR 价值。

---

# 三、图表建议

我没有评价视觉样式，只根据当前 caption 和论证功能判断。

## 必须新增

### 1. 方法机制图

需要一张图明确展示：

[
\text{source profiles}
\to
\text{DCT coordinate}
\to
\text{source scoring}
\to
\text{maximin atlas}
\to
\text{backend}
\to
\text{fresh verifier}.
]

这是目前最缺的图。

### 2. Atlas 与 feasible basin 的几何图

每个 synthetic domain 至少画一个低维 summary/coordinate projection，显示：

* universal candidates；
* source templates；
* selected atlas；
* target feasible region；
* true optimum；
* Sobol points。

这张图会直接揭示方法究竟是发现结构，还是 benchmark 本来就把安全区放在 atlas library 中。

### 3. 全成本分解图

按方法显示：

[
S,\quad N,\quad V_{\mathrm{safety}},\quad
V_{\mathrm{objective}},\quad C_{\mathrm{total}}.
]

当前 verification 成本比 search 大一个数量级，必须可视化。

### 4. Certificate power / nonvacuity 图

横轴 safety depth 或 replication budget，纵轴：

* certificate probability；
* false certificate；
* mean verifier calls。

### 5. Source budget learning curve

source task 数、profiles 数、replication 数对 coverage 的影响。

---

## 建议保留

* frontend/backend factorial figure；
* target convergence figure；
* external energy paired-difference figure。

---

## 建议移到 appendix

* frontend components figure：与表格高度重复；
* HVD diagnostic figure及表格：与主贡献无关；
* coverage audit table：oracle post-run diagnostic；
* dimension/budget figure需要重做，否则只是展示 nominal dimension。

---

# 四、优先修改顺序

## 第一优先级：决定论文究竟主张什么

二选一：

### 路线 A：坚持 source-learning 主张

则必须增加新的、真正未见 domain，并证明：

[
\text{source atlas}

>

\text{generic low-frequency structured design}.
]

### 路线 B：收窄为 structured initial design

论文主张改为：

> Historical simulations can score and select a reusable structural library, but the principal gain comes from the low-frequency profile hypothesis.

这更符合现有证据，但理论和定位要转向 functional experimental design。

---

## 第二优先级：修复理论

不要再增加 Lean 文件数量。重点证明：

* DCT dimension consistency；
* maximin coverage approximation；
* source score finite-sample error；
* candidate-level verification validity；
* probabilistic target coverage。

---

## 第三优先级：修复实验公平性

* structured controls；
* native transfer pipelines；
* all-in total cost；
* unseen domains；
* schema/domain-blind；
* source budget curve。

---

## 第四优先级：删掉旁枝

HVD、KG历史、transformer、orthogonal sparse development history都不要回主文。当前收敛方向是对的。

---

# 五、模拟的三位 OR 审稿意见

## Reviewer 1 — Theory and Methodology

**Recommendation: Reject, with encouragement to resubmit after substantial theoretical redevelopment.**

The manuscript studies a source-informed structural initial design for chance-constrained simulation optimization. I appreciate the unusually transparent information contract, the separation between search and verification, and the retention of negative results. However, I do not believe the present theoretical contribution meets the standard of Operations Research.

The central coverage theorem assumes that the learned coordinate approximates an ideal coordinate, that the source library contains a policy close to a target-safe center, that the target margin is Lipschitz in the learned coordinate, and that the safe center is sufficiently deep. Under these assumptions, coverage follows by the triangle inequality. The theorem is correct but largely restates the desired conclusion in geometric form. It does not analyze how the proposed source-ranking and maximin algorithm obtains these quantities from finite source samples.

The no-free-lunch theorem is elementary, and the terminal safety theorem is a union bound conditional on candidate-wise tests already having the declared error probability. Machine verification in Lean increases confidence that the displayed implications are correct, but it does not make the implications theoretically novel.

To support publication, the authors should establish results directly tied to the implemented algorithm: a cross-dimension consistency result for the DCT representation, an approximation guarantee for maximin atlas construction, finite-sample control of source-profile ranking errors, and a probabilistic source-to-target coverage guarantee. The candidate-level verification rule and its assumptions must also be formally derived rather than assumed.

A second concern is that the term “dimension-equivariant” is not yet mathematically justified. The coordinate has fixed dimension, but invariance or approximation across discretization dimensions is not proved. I recommend recasting the decision variable as an ordered functional profile and proving consistency under refinement of its discretization.

In its current form, the theory is best viewed as an audit framework around an empirical method, not a substantive theory of transferable simulation optimization.

## Reviewer 2 — Experiments and Statistical Validity

**Recommendation: Reject.**

The empirical results are striking, but the benchmark construction makes them difficult to interpret. All three main synthetic domains depend on a handful of block means and dispersion measures, whereas the proposed initial design consists precisely of constant, monotone, and low-frequency policy profiles. Increasing the nominal dimension from 1,000 to 10,000 does not increase the effective dimension of these problems. I therefore do not view the reported (d/N) ratios as evidence of high-dimensional optimization performance.

The leave-one-domain-out protocol also does not constitute an independent domain holdout. The same three benchmark families appear to have influenced representation, library, threshold, and algorithm development. Freezing the final implementation before rerunning target seeds prevents direct label leakage, but it does not prevent benchmark-level overfitting. New task families designed after method freeze are necessary.

The external energy experiment is commendably reported, including the negative structured control. However, the atlas does not outperform a natural low-frequency grid. Thus, the external result validates the low-frequency structural assumption but not the value of source learning. This is inconsistent with the current source-learning emphasis.

The baseline design also needs revision. Transfer methods are all given the proposed atlas, so the comparison largely evaluates backends conditional on the authors’ frontend. Native end-to-end transfer methods should be allowed to construct their own initial designs. Conversely, the manuscript lacks the most important controls: random or maximin designs directly in DCT space, a natural blockwise structured grid on the synthetic domains, and an equal-cost structured target-only search.

The statistical analysis treats 60 domain-seed pairs as the inferential sample, although there are only three task domains. The extremely small McNemar (p)-values show repeatability over simulation noise, not generalization over tasks. Conditional regret is also not directly comparable when feasibility rates differ.

Finally, “equal-total-cost” excludes verification, even though mean verification cost is approximately 90–215 calls. All-in simulation cost should include source, search, objective comparison, and safety verification. The appropriate endpoint is the cost required to obtain a certified and sufficiently good solution.

I consider the audit trail excellent, but the current evidence does not establish the broad empirical claim.

## Reviewer 3 — OR Positioning and Practical Significance

**Recommendation: Major conceptual revision; not suitable for publication in the present form.**

The paper contains an interesting operational decomposition: reusable source experimentation, a small target search stage, and a separate deployment-verification stage. This decomposition may be valuable, especially when source data are amortized over many target deployments. However, the manuscript does not yet convert this observation into a sufficiently general OR contribution.

First, the method applies to ordered policy profiles for which low-frequency shape has operational meaning. This is substantially narrower than generic high-dimensional integer simulation optimization. The title, abstract, and problem formulation should say so explicitly.

Second, the paper needs a decision-theoretic analysis of when the source archive is worth purchasing. The source archive costs 384 simulations and verification frequently costs more than target search. A manager would need a break-even analysis in the number of future target tasks, verification cost, cost of unsafe deployment, and expected benefit of improved coverage.

Third, the terminal verifier is presented as optimizer independent, but its candidate-level statistical procedure is insufficiently specified. The familywise union bound is not the difficult part. The paper must establish validity and power of each candidate test under the actual stochastic assumptions, including the temporal dependence of the energy-window experiment.

Fourth, the closest OR literature on chance-constrained ranking and selection, optimal sampling under stochastic constraints, and post-optimization cleanup is largely absent. The proposed verifier should be positioned against and compared with these procedures, not only against Bayesian-optimization literature.

The negative external result against a natural low-frequency grid is scientifically valuable. In my view, it suggests a more defensible paper: historical simulations are one way to construct a structured initial design, but the deeper contribution is identifying and pricing structural coverage before online optimization begins. Such a paper would need new theory and a broader empirical evaluation, but it could ultimately be compelling.

## 最终结论

这篇稿件最大的优点是**诚实和可审计**，最大的缺点是**主方法与合成任务共享同一低频生成假设，而外部实验没有证明 source learning 超过通用结构设计**。

当前直接投 OR，我预计：

* desk reject：有可能；
* 送审后 reject：概率较高；
* 原因不会是代码或格式，而是 theory–evidence–claim 三者尚未完全对齐。

最值得做的不是继续扩展算法，而是围绕下面一句重新闭环：

[
\boxed{
\text{历史任务何时能够改善一个结构化初始设计，且这种改善超越一个不使用历史结果的通用结构先验？}
}
]

这才是当前论文真正尚未回答、但一旦回答就最接近 OR 的问题。

[1]: https://pubsonline.informs.org/doi/10.1287/ijoc.2014.0628?utm_source=chatgpt.com "Chance Constrained Selection of the Best | INFORMS Journal on Computing"
[2]: https://arxiv.org/abs/1909.12552?utm_source=chatgpt.com "Learning search spaces for Bayesian optimization: Another view of hyperparameter transfer learning"
[3]: https://ojs.aaai.org/index.php/AAAI/article/view/11830?utm_source=chatgpt.com "Bayesian Functional Optimization | Proceedings of the AAAI Conference on Artificial Intelligence"
[4]: https://pubsonline.informs.org/doi/pdf/10.1287/opre.51.5.814.16751?utm_source=chatgpt.com "Using Ranking and Selection to “Clean Up” after Simulation Optimization | Operations Research"

# 我：
所以客观的说，目前效果如何？尤其在超高维度上，比起之前的SOTA呢？以及这个结果没作弊吧（我感觉inventory/queue/factor shock问题，也没办法作弊，没有oracle信息）

# GPT：
客观地说，当前结果可以分成两种完全不同的评价。

[
\boxed{
\text{在“共享低频策略结构”的任务族内，效果非常强；}
}
]

但：

[
\boxed{
\text{作为一般意义上的超高维黑箱优化 SOTA，证据还不成立。}
}
]

## 1. 在当前注册实验中的效果有多强？

非常强，甚至接近“前端把问题解决了”。

在 (d=1000)、目标搜索预算 (N=13) 下，冻结 atlas 配合 proposal-only、stacked GP 和 canonical SAASBO，三者都是：

[
60/60\ \text{true feasible},\qquad
60/60\ \text{independently certified}.
]

换成 common Sobol 前端后，分别只有：

[
0/60,\qquad1/60,\qquad0/60.
]

这说明主要性能来源不是 SAASBO 或 KG 后端，而是：

[
\boxed{\text{source-informed structural atlas 成功覆盖了 feasible basin。}}
]

在线优化只负责覆盖之后的小幅目标改善。SAASBO 的 aggregate median feasible regret 是 (0.0038)，proposal-only 是 (0.0082)；也就是说，自适应后端有作用，但远小于前端从“几乎永远不可行”到“全部可行”的提升。

在 (d=10000) 时，proposal-only 用10个 target evaluations 就达到 (30/30) feasible/certified；不同预算下仍全部成功。

所以，**在当前结构化任务族内部，结果可以评为 9–9.5/10。**

---

## 2. 相比 repo 中此前的 SOTA 呢？

对 repo 中直接比较的 target-only 高维 BO 方法，优势是压倒性的：

* TuRBO，397次 target-search calls：(0/60) feasible；
* SCBO，397次：(0/60)；
* periodic-capped SAASBO，397次：(2/60)；
* atlas + canonical SAASBO，384 source + 13 target search：(60/60)。

所以在这个 benchmark 协议下，你的方法确实显著超过了这些经典高维 BO 基线。

但需要强调，它们解决的实际上不是完全相同的问题：

* target-only 方法从 raw (1000)-维空间开始；
* 你的方法从 source tasks 获得一个低频结构化 initial design；
* 因此比较证明的是：

[
\boxed{
\text{可迁移结构先验远比无结构 target-only 搜索重要。}
}
]

而不是证明：

[
\boxed{
\text{你的在线 optimizer 比 TuRBO/SCBO/SAASBO 更强。}
}
]

事实上，你自己已经很好地证明了后一命题不成立：给不同后端同一个 atlas，绝大多数 transfer/BO 后端都能达到56–60个 feasible trials。

---

## 3. 能不能说超越“目前高维 BO SOTA”？

目前还不能。

2025年的高维 BO 研究仍将方法主要归纳为有效低维、加性和局部搜索三类；最新研究也指出，高维 BO 的成功往往来自局部搜索行为、初始化和隐含有效维度，而不能只依据 nominal dimension 判断难度。([中国开放网络服务联盟][1])

当前 baseline 集合虽然覆盖了 TuRBO、SCBO、SAASBO 和一批 transfer BO 方法，但截至2026年，至少还缺：

* ICML 2025 的 MSR high-dimensional BO；
* AAAI 2026 的 MetaGameBO；
* 新的 transfer-BO/generalization 方法；
* 更原生的 profile/function-space optimization baselines。([Proceedings of Machine Learning Research][2])

而且这些论文的 benchmark、预算、约束和 source-information contract 与你的不一致，所以即使补了，也不能仅看一张表宣布 universal SOTA。

更准确的表述是：

> **在具有跨任务共享低频 policy-profile structure、极小 target budget 和 chance constraints 的设定下，本方法显著优于所测试的 target-only 与 transfer BO baselines。**

这个主张站得住。

“在任意 (d=10000) 黑箱优化上达到 SOTA”则站不住。

---

## 4. (d=10000) 到底有多“高维”？

raw decision vector 确实有10000维，但其有效机制维度很低。

三个 synthetic domains 的目标和约束实际上依赖：

* 2–3个 block means；
* dispersion/spread；
* 少量低频 profile summaries。

而 atlas 正好由：

* constant profiles；
* monotone profiles；
* low-frequency DCT profiles；

组成。

因此从 (d=1000) 增加到 (d=10000)，更像是：

[
\text{同一个低维函数 profile 被更细密地离散化}
]

而不是：

[
\text{新增9000个独立、有效、未知的决策自由度}.
]

所以这个结果依然很有价值，但价值应描述为：

[
\boxed{
\text{跨离散维度迁移和 profile-space compression}
}
]

而不是普通意义上的10000个独立 active variables 优化。

我会给：

* nominal high-dimensional performance：**9/10**
* effective-dimensional difficulty：**约4–5/10**
* 对 ordered functional/profile optimization 的价值：**8/10**
* 对一般 high-dimensional black-box optimization 的证据：**约4/10**

---

# 5. 有没有直接作弊？

根据最新 frozen manifest、审计合同和代码路径，我没有发现直接作弊。

明确没有进入 final proposal/search 的包括：

* target objective labels；
* target feasibility labels；
* target true optimum；
* target true constraint；
* target true sigma；
* target verification outcomes；
* source analytic oracle；
* target oracle。

atlas 在第一次 target simulation 前冻结，verification samples 也不会反馈给 optimizer。

synthetic true functions 只在最后用于计算：

[
\text{true feasibility}
\quad\text{和}\quad
\text{regret}.
]

这是正常 benchmark evaluation，不是作弊。

所以对“是否存在 target truth 泄漏”，我的判断是：

[
\boxed{\text{目前证据显示没有。}}
]

---

# 6. 那为什么结果会这么好？

因为使用了**极强但合法的结构信息**。

算法允许知道：

* held-out task-family identifier；
* policy dimension 与 bounds；
* policy/state exposure schema；
* 哪些坐标构成有序 profile 或功能块；
* source tasks 上相同结构 profiles 的观测结果。

例如 Inventory 中知道坐标可解释为 stock、reorder、safety blocks；Queue 中知道是 capacity、priority、smoothing blocks。这不是告诉算法：

[
x^\star=(\cdots)
]

也不是告诉它 feasible boundary，但它确实已经告诉算法：

[
\boxed{
\text{应当在什么低维结构空间里寻找答案。}
}
]

这类似于在 GP 中给了一个高度匹配真实函数的 kernel，或者在压缩感知中提前告诉模型信号在 Fourier basis 中稀疏。

它是：

[
\textbf{strong admissible prior}
]

而不是：

[
\textbf{oracle cheating}.
]

---

## 7. 还存在一种“非作弊但会被质疑”的问题

虽然 final run 没有使用 target oracle，但方法在开发过程中长期看过 Inventory、Queue、FactorShock 这些 benchmark。

因此可能存在：

[
\boxed{
\text{benchmark-family overfitting}
}
]

也就是：

* 低频 library；
* DCT coordinate；
* constant/monotone profiles；
* block summaries；
* (K\le8)；
* atlas size；
* scalarization；
* verification budgets；

都可能在长期实验中逐渐适配了这三个问题。

这不是伪造结果，也不是运行时泄漏，但 LODO 只能证明：

> 最终 target outcomes 没有进入当前 fold 的训练。

它不能证明：

> 方法设计从未受 held-out benchmark family 影响。

真正关闭这个质疑的方式是：现在冻结全部代码，再由他人或随机生成器构造新的 unseen domains，一次性运行，不允许根据结果修复。

---

# 8. 外部能源结果告诉了什么？

energy holdout 很关键：

* atlas 显著胜过397次 common Sobol；
* 但没有胜过一个自然的 low-frequency grid；
* low-frequency grid 的中位 objective 甚至略好。

所以外部结果证明：

[
\boxed{
\text{低频结构先验是真实有效的。}
}
]

但没有证明：

[
\boxed{
\text{source-data learning 比一个合理的通用低频设计更强。}
}
]

这不是坏结果。它意味着你的真正核心发现可能是：

> 在超高维 ordered policy profile 问题中，结构化 initial design 比 acquisition function 更重要；source data 可以帮助选择该设计，但其额外价值依赖于 source–target alignment。

这其实是一个很好的 OR 结论。

---

# 9. 成本上也要客观看

“13次 target evaluations”非常亮眼，但不是完整成本。

主方法平均 verification cost 约为174次，另有384次 source calls。

所以单个 target 的平均 all-in simulator cost 大约是：

[
384+13+174\approx571.
]

当然：

* source archive 可以在多个 target tasks 间复用；
* verification 是部署安全的额外成本；
* target-only 方法同样需要 verifier 才能给同等级 safety claim。

但论文不应把“13次搜索”表达成“总共13次仿真”。

---

# 最终客观结论

## 效果

[
\boxed{
\text{在注册的结构化 profile 任务族内，效果极其优秀，明显超过所测试基线。}
}
]

## 超高维意义

[
\boxed{
\text{它成功处理的是 nominally ultra-high-dimensional、intrinsically low-dimensional 的 profile optimization。}
}
]

不是任意10000维黑箱函数。

## 相比 SOTA

[
\boxed{
\text{超过了 repo 中的经典 target-only SOTA baselines；尚未证明超过2026年所有高维/transfer BO SOTA。}
}
]

## 是否作弊

[
\boxed{
\text{没有发现 target-oracle 或 truth leakage；结果不是作弊。}
}
]

但：

[
\boxed{
\text{方法使用了非常强的、与 benchmark 结构高度匹配的合法先验，并可能存在 benchmark-family development overfitting。}
}
]

最准确的一句话是：

> **这不是靠作弊击败 SOTA，而是靠一个非常强且基本正确的结构假设，把原本不可解的 (10^3)–(10^4) 维问题压缩成了低维 profile coverage 问题；当前最强证据证明了结构化前端的重要性，还没有证明其对一般高维黑箱或相对于所有合理结构化先验的普遍优势。**

[1]: https://jos.org.cn/josen/article/abstract/7304?st=article_issue&utm_source=chatgpt.com "Survey on High-dimensional Bayesian Optimization"
[2]: https://proceedings.mlr.press/v267/papenmeier25a.html?utm_source=chatgpt.com "Understanding High-Dimensional Bayesian Optimization"

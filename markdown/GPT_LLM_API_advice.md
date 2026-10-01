# 我：
我有个方案，就是让优化器接入AI的API，这可能听起来有点作弊，但AI大模型根据在开始根据问题定义，在后续阶段根据不断累积的数据，和优化器（代理模型）一起一步步去构建出一个前面我们视为“作弊”的强先验。
我不知道这样是否可行，如果可行，去查目前最新文献，看看那些结合了LLM+coupling或者solver的论文都是如何结合的，再根据目前项目代码，找出一个最合理的方案（客观评估，前提是可以这么做）

# GPT；
我认为这个方向**可行，但必须非常克制地做**。最合理的定位不是“LLM 直接替代优化器”，而是：

[
\boxed{
\textbf{LLM 作为可审计、可证伪、可门控的 structural-prior generator；KG/GP/HVD 仍然是正式优化器。}
}
]

这和你现在遇到的问题正好对上：目前被视为“作弊”的部分其实是强结构先验。LLM 可以把这些先验从“手写 benchmark-specific oracle”变成“由问题文本、轨迹摘要、历史数据逐步生成的 admissible learned prior”。但要投 OR，必须把 LLM 严格关在信息边界内，并且所有 LLM 输出都要经过 evidence gate，不能无条件相信。

---

# 1. 先说结论：可以做，但不要让 LLM 直接当 optimizer

我建议的新版本叫：

[
\boxed{
\textbf{EG-LLM-SC-OLH-KG}}
]

即：

[
\boxed{
\textbf{Evidence-Gated LLM Structural Prior for State-Coupled Orthogonal Latent Heteroscedastic KG}}
]

中文可以叫：

[
\boxed{
\textbf{证据门控的大模型结构先验 SC-OLH-KG}}
]

核心思想：

[
\text{LLM proposes structure}
\quad
\rightarrow
\quad
\text{validator checks admissibility}
\quad
\rightarrow
\quad
\text{HVD/GP tests it on observed data}
\quad
\rightarrow
\quad
\text{gate decides how much to trust}
\quad
\rightarrow
\quad
\text{KG selects next sample}
]

而不是：

[
\text{LLM says sample }x
\quad
\rightarrow
\quad
\text{直接执行}.
]

原因很简单：LLM 可能 hallucinate，可能给出看似合理但错误的结构；FunSearch 的成功经验也不是“相信 LLM”，而是把 frozen LLM 生成的程序放进系统评价器里筛选和进化，评价器负责挡住 hallucination 和错误程序。FunSearch 论文明确把 LLM 与 systematic evaluator 配对，并强调 evaluator 防止错误想法进入结果；它搜索的是可解释的程序而不是直接答案。([Nature][1])

---

# 2. 文献里 LLM + BO / solver 通常怎么结合？

目前最新方向大致有五类。

---

## 2.1 LLM 做 warm start、surrogate、candidate sampling：LLAMBO

LLAMBO 把 BO 问题转成自然语言，让 LLM 基于历史 evaluations 迭代提出和评估候选。它的贡献不是完全抛弃 BO，而是把 LLM 的上下文理解、few-shot 能力和领域知识接进 model-based BO；论文摘要明确说 LLAMBO 对 zero-shot warmstarting、surrogate modeling、candidate sampling 有帮助，尤其在早期观测稀疏时。([arXiv][2])

这对你的启发是：

[
\boxed{
\text{LLM 可以参与 early-stage prior/candidate generation，但不应拥有最终决策权。}
}
]

你现在 (N=80) 有效的关键也在 early-stage：如果前 10 到 20 个样本已经进到正确 (\psi=(A,N)) 区域，后面 KG/HVD 才有东西可学。

---

## 2.2 LLM 做 context-aware BO advisor：BORA

BORA 明确把 LLM 用作 BO 的 contextualizer：它把 stochastic inference 和 LLM domain-knowledge insight 混合，LLM 用来建议 search space 里更值得探索的区域，并提供优化过程解释。它也指出 domain knowledge 可以帮助高维 needle-in-a-haystack 搜索，但有 confirmation bias 风险。([arXiv][3])

这非常接近你的问题：当前手写 anchors/refinement 正是人工 domain knowledge；LLM 版本可以把它改成：

[
\text{natural-language problem definition + observed data summary}
\rightarrow
\text{LLM proposes structural search regions}
]

但必须保留防 confirmation bias 的机制：

[
\epsilon\text{-random exploration}
+
\text{evidence gate}
+
\text{negative-control ablations}.
]

---

## 2.3 LLM 做 acquisition strategist：LMABO

LMABO 不让 LLM 直接提点，而是让 LLM 根据完整 BO 状态选择 acquisition function。论文说没有单一 acquisition function 对所有问题最优；LMABO 把 acquisition selection 变成 in-context decision-making，用结构化 BO state prompt 让 LLM 动态选择 acquisition。([arXiv][4])

这对 SC-OLH-KG 很有用。你可以让 LLM 在下面几种采样模式中选择或加权：

[
\mathrm{KG}^{J}
\quad
\mathrm{KG}^{F}
\quad
\mathrm{KG}^{V}
\quad
\mathrm{KG}^{\rho}
]

即：

[
\text{目标均值改善}
]

[
\text{chance boundary resolution}
]

[
\text{variance decomposition learning}
]

[
\text{state-coupling propagation}
]

但最终选点仍然由数值 acquisition score 决定，而不是 LLM 直接拍脑袋。

---

## 2.4 LLM 生成 expert prior，但要 evidence-gated：CALIPER

2026 年的 CALIPER / Evidence-Gated LLM Priors 方向对你尤其关键。它指出 LLM 建议和自报 confidence 不一定与下游 objective 校准，因此把 LLM expert prior 当成可证伪的 prior source，用 observed feedback 在线更新 expert weight，并且允许系统 abstain from LLM prior。论文摘要明确说：不要盲目信 LLM priors；要把 expert-objective pair 当作 falsifiable prior source，并用 evidence gate 动态决定是否使用。([arXiv][5])

这应该直接成为你的主机制：

[
\boxed{
\text{LLM prior 不是默认正确，而是一个可被数据打败的 prior layer。}
}
]

---

## 2.5 LLM + solver：OR-LLM-Agent / solver-in-the-loop

OR-LLM-Agent 的方向是让 LLM 把自然语言 OR 问题转成数学模型和 Gurobi 代码，并通过 sandbox 执行、修复、验证。它不是黑箱仿真优化，而是 optimization modeling automation；但它给了一个重要工程原则：LLM 输出必须经过 solver / sandbox / verification，而不是直接作为可信结果。([arXiv][6])

这对你的代码实现很重要：LLM 可以写 feature DSL、candidate proposal、HVD prior，但必须经过：

[
\text{schema validation}
]

[
\text{admissibility audit}
]

[
\text{cross-validation on observed samples}
]

[
\text{posterior KG scoring}
]

---

# 3. 你的项目最合理的 LLM 接入点

当前 repo 的主代码是 GPR-KG：`ParametricGPR` 用 (2d+1) 的二次 basis，VEPM 用 partition-based variance sharing，Pareto-KG 负责采样；代码注释明确写了这三块结构。([GitHub][7]) 其中 `ParametricGPR` 现在的 basis 是无交叉项二次基，`VEPM` 是基于特征分区共享方差，主循环则先 solve posterior、generate candidate set、compute KG factors、再 Pareto-KG selection。([GitHub][7])

所以 LLM 不应该接在“直接选 (x)”的位置，而应该接在下面四个位置。

---

## 3.1 LLM-Structural-Prior Agent：生成 (\psi=(A,N)) 候选结构

输入给 LLM 的不是 true objective / true sigma / true optimum，而是 admissible prompt：

[
P_n
===

\mathrm{Serialize}
\left(
\mathcal I_n
\right)
]

其中：

[
\mathcal I_n
============

{
\text{problem schema},
\text{variable bounds/types},
\text{trajectory field names},
\text{current observed }(\theta_i,Y_i,C_i,\mathcal T_i),
\text{posterior diagnostics},
\text{HVD residual diagnostics}
}.
]

LLM 输出一个严格 JSON / DSL：

```json
{
  "basis_proposals": [
    {
      "name": "risk_factor_1",
      "type": "A",
      "expression": "sum_t standardized(queue_length_t * phase_switch_t)",
      "rationale": "captures local unstable congestion exposure"
    },
    {
      "name": "shock_regime_1",
      "type": "N",
      "expression": "soft_cluster(high_density_and_low_speed)",
      "rationale": "shared congestion shock regime"
    }
  ],
  "candidate_region_priors": [
    {
      "psi_center": [ ... ],
      "radius": 0.2,
      "reason": "near posterior feasible boundary with high variance"
    }
  ],
  "acquisition_weights": {
    "kg_objective": 0.35,
    "kg_feasibility": 0.30,
    "kg_variance": 0.25,
    "kg_coupling": 0.10
  },
  "confidence": 0.62
}
```

注意：它输出的是**结构建议**，不是最终答案。

然后你把这些建议变成候选 basis：

[
\psi^{\mathrm{LLM}}_n(\theta,\mathcal T)
========================================

(A^{\mathrm{LLM}}_n,N^{\mathrm{LLM}}_n).
]

再让 HVD 在观测数据上验证：

[
\mathrm{Var}(C(\theta)\mid\mathcal T)
=====================================

A^\top\Lambda A
+
N^\top B N
+
N^\top\omega
+
\sigma_0^2.
]

---

## 3.2 LLM 生成 state-coupled GPR basis，而不是继续用 ([1,x,x^2])

当前代码里 GPR basis 是：

[
\phi_{\mathrm{old}}(x)
======================

[1,x,x^2].
]

LLM 可以建议从 observed trajectories 中构造：

[
\rho(x)
=======

\sum_t \varphi(S_t,A_t).
]

然后新 GPR basis 是：

[
\phi_{\mathrm{SC}}(x)
=====================

[
1,\ x,\ x^2,\ \rho(x),\ \rho(x)^2,\ x\otimes \rho(x),\ \psi(x)
].
]

LLM 的作用是建议：

[
\varphi(S_t,A_t)
]

里哪些 trajectory statistics 有意义。例如交通问题里可能是 queue quantile、spillback indicator、phase switching、stop-and-go exposure；库存问题里可能是 stockout duration、demand burst、replenishment lag；queue 问题里可能是 utilization、blocking、service instability。

但这仍然要经过 validator：

[
\text{LLM proposed feature}
\rightarrow
\text{standardize}
\rightarrow
\text{orthogonalize}
\rightarrow
\text{cross-validated likelihood}
\rightarrow
\text{gate}.
]

---

## 3.3 LLM 生成 candidate proposal distribution，而不是 direct recommendation

你现在最容易被质疑的是 problem-specific anchors/refinement。LLM 版本应该改成：

[
q_{\mathrm{cand},n}(x)
======================

\epsilon q_{\mathrm{space}}(x)
+
\eta_n q_{\mathrm{LLM},n}(x)
+
\gamma_n q_{\mathrm{posterior},n}(x)
+
\zeta_n q_{\mathrm{boundary},n}(x).
]

其中：

[
\epsilon>0
]

必须固定存在，保证非 LLM fallback。

LLM 只提出：

[
\psi\text{-space regions}
]

例如：

[
|\psi(x)-c_j|\le r_j.
]

真正的 candidate 由 optimizer / sampler 从这些 region 里采样，再由 SC-OLH-KG acquisition 打分：

[
x_{n+1}
=======

\arg\max_{x\in\mathcal A_n}
\mathrm{SC\text{-}OLHKG}_n(x).
]

所以 LLM 不是直接选点，而是改变候选池分布。

---

## 3.4 LLM 作为 acquisition strategist

定义四个 acquisition components：

[
\mathrm{KG}^{J}_n(x)
====================

\text{objective information value},
]

[
\mathrm{KG}^{F}_n(x)
====================

\text{feasibility-boundary information value},
]

[
\mathrm{KG}^{V}_n(x)
====================

\text{HVD variance-parameter information value},
]

[
\mathrm{KG}^{\rho}_n(x)
=======================

\text{state-coupling propagation value}.
]

普通 SC-OLH-KG：

[
\mathrm{SC\text{-}OLHKG}_n(x)
=============================

w_J\mathrm{KG}^{J}_n(x)
+
w_F\mathrm{KG}^{F}_n(x)
+
w_V\mathrm{KG}^{V}*n(x)
+
w*\rho\mathrm{KG}^{\rho}_n(x).
]

LLM 可以根据当前 state summary 输出：

[
w_n=(w_{J,n},w_{F,n},w_{V,n},w_{\rho,n}).
]

但权重也要门控：

[
w_n^{\mathrm{final}}
====================

(1-\lambda_n)w^{\mathrm{default}}
+
\lambda_n w_n^{\mathrm{LLM}},
]

其中 (\lambda_n) 来自 evidence gate。

这和 LMABO 的思想一致：LLM 不直接做函数评估，而是根据完整 BO state 选择或调整 acquisition strategy。([arXiv][4])

---

# 4. 关键：LLM 先验如何不算作弊？

要把它写成正式定义。

令 target problem 为 (p^\star)。第 (n) 步允许的信息集：

[
\mathcal I_n^{p^\star}
======================

\left{
\mathcal D_{\mathrm{source}},
\mathcal M_{\mathrm{obs}}^{p^\star},
(\theta_i,Y_i,C_i,\mathcal T_i)_{i=1}^{n}
\right}.
]

LLM prompt 必须是：

[
P_n=S(\mathcal I_n^{p^\star}).
]

LLM 输出：

[
Z_n=\mathrm{LLM}(P_n;\xi_n).
]

算法选择：

[
\theta_{n+1}
============

\Pi(D_n,Z_n)
]

必须是：

[
\sigma(\mathcal I_n^{p^\star},\xi_n)\text{-measurable}.
]

也就是说，只要 LLM 没看到 target true objective、true constraint、true sigma、true optimum、hidden active axis、true boundary，原则上就不是作弊。

但是有一个很现实的风险：如果 prompt 里写了 “RZDT1 / FactorShock / Inventory benchmark” 这种名字，LLM 可能凭训练语料或推理猜出 benchmark 结构。为了避免这个问题，主实验必须做：

[
\boxed{
\textbf{anonymized benchmark protocol}}
]

即 prompt 里不能出现 benchmark 名称、不能粘贴 true function 代码、不能出现“active dimension is axis 0”这类信息。只给：

1. variable bounds；
2. variable type；
3. observed samples；
4. observed trajectories；
5. posterior diagnostics；
6. feature field descriptions；
7. current budget and risk level。

如果 LLM 从这些信息里推断出某些结构，那是合法学习；如果它从 benchmark 名称或代码公式里“记忆答案”，那就是 leakage。

---

# 5. Evidence gate：LLM 先验必须可被数据打败

这是最关键的数学设计。

令 LLM 生成一个 prior feature map：

[
\psi^{\mathrm{LLM}}_n(x)
]

或 prior score：

[
p^{\mathrm{LLM}}_n(x).
]

不要直接相信它，而是定义一个 gate weight：

[
\lambda_n\in[0,1].
]

它由 observed data 的 predictive evidence 决定。

---

## 5.1 HVD evidence gate

对 LLM-HVD basis，计算 residual-square likelihood：

[
r_i=C_i-m^g_{-i}(x_i).
]

LLM-HVD 模型：

[
r_i^2
\approx
\psi^{\mathrm{LLM}}(x_i,\mathcal T_i)^\top\eta.
]

baseline HVD 模型：

[
r_i^2
\approx
\psi^{0}(x_i,\mathcal T_i)^\top\eta_0.
]

定义 cross-validated log score：

[
S^{\mathrm{LLM}}_n
==================

\sum_{i\le n}
\log p(r_i^2\mid \psi^{\mathrm{LLM}}(x_i,\mathcal T_i),D_{-i}),
]

[
S^{0}_n
=======

\sum_{i\le n}
\log p(r_i^2\mid \psi^{0}(x_i,\mathcal T_i),D_{-i}).
]

gate：

[
\lambda_n
=========

\sigma\left(
\kappa(S^{\mathrm{LLM}}*n-S^0_n-\tau*{\mathrm{gate}})
\right).
]

如果 LLM-HVD basis 对 residual variance 没帮助：

[
S^{\mathrm{LLM}}_n\le S^0_n,
]

则：

[
\lambda_n\approx 0.
]

---

## 5.2 Candidate gate

LLM candidate proposal 也要有 evidence：

[
E^{\mathrm{cand}}_n
===================

\frac{
\text{posterior acquisition mass captured by LLM candidates}
}{
\text{posterior acquisition mass captured by random/posterior candidates}
}.
]

例如：

[
E^{\mathrm{cand}}_n
===================

\frac{
\max_{x\in\mathcal A^{\mathrm{LLM}}_n}\mathrm{SC\text{-}OLHKG}*n(x)
}{
\max*{x\in\mathcal A^{0}_n}\mathrm{SC\text{-}OLHKG}_n(x)+\epsilon
}.
]

如果 LLM candidates 的 posterior acquisition 分数低，就降低它们在候选池里的比例。

---

## 5.3 Prior mean gate

如果 LLM 给了 objective / constraint prior mean：

[
\mu^{\mathrm{LLM}}(x),
]

不要直接把它当真。用 residual GP：

[
J(x)
====

\lambda_n\mu^{\mathrm{LLM}}_J(x)+r_J(x),
\quad
r_J\sim GP(0,k_J),
]

[
g(x)
====

\lambda_n\mu^{\mathrm{LLM}}_g(x)+r_g(x),
\quad
r_g\sim GP(0,k_g).
]

[
\lambda_n
]

由 observed predictive likelihood 更新。

这和 CALIPER 的精神一致：LLM prior 是可证伪的 prior source，权重必须由 observed objective feedback 更新，而且系统要能 abstain。([arXiv][5])

---

# 6. 数学上如何保证不因为 LLM 错了而崩？

把 candidate proposal 写成 mixture：

[
q_n(x)
======

\epsilon q_{\mathrm{space}}(x)
+
(1-\epsilon)
\left[
\lambda_n q_{\mathrm{LLM},n}(x)
+
(1-\lambda_n)q_{\mathrm{posterior},n}(x)
\right].
]

只要：

[
\epsilon>0,
]

就有基本探索覆盖。LLM 最坏情况下只是浪费一部分 candidate mass，不会完全控制算法。

定义好区域：

[
\mathcal G_\delta
=================

\left{
x:
J(x)-J(x^\star)\le \delta,\
x\text{ safely feasible}
\right}.
]

如果：

[
q_n(\mathcal G_\delta)\ge p_\delta,
]

那么 (M) 个 candidate 至少命中一次好区域的概率：

[
1-(1-p_\delta)^M.
]

LLM 的作用是把：

[
p_\delta
]

从纯 random 的极小值提高到更大。
但如果 LLM 错了，仍有：

[
p_\delta\ge \epsilon q_{\mathrm{space}}(\mathcal G_\delta).
]

所以理论 bound 可以写成：

[
r_N
\le
C_1\sqrt{\frac{\gamma_N(k_\psi)}{N}}
+
C_2\sqrt{\frac{d_{\mathrm{HVD}}\log N}{n_{\mathrm{eff}}}}
+
C_3\epsilon_\psi
+
C_4\sqrt{\frac{\log(1/\delta)}{M p_\delta}}.
]

其中：

[
d_{\mathrm{HVD}}
================

r+\frac{K(K+1)}{2}+K+1.
]

LLM 有效不是因为它“知道答案”，而是因为它提高了：

[
p_\delta
]

并降低了 representation error：

[
\epsilon_\psi.
]

这可以解释为什么 (N=80) 有效：不是在 (d=1000) 原空间盲搜，而是在 LLM + SC/HVD 构造的低维 cumulative-risk coordinate 上做 KG。

---

# 7. 放进你当前代码的最合理方案

基于现在 `gpr_kg.py` 的结构，建议不要大改主循环，而是增加一个 prior layer。

当前结构大致是：

[
\text{pre-sampling}
\rightarrow
\text{GPR initialize}
\rightarrow
\text{VEPM initialize}
\rightarrow
\text{posterior solve}
\rightarrow
\text{candidate generation}
\rightarrow
\text{KG factor}
\rightarrow
\text{selection}
\rightarrow
\text{simulate/update}.
]

你可以改成：

[
\text{pre-sampling}
\rightarrow
\text{LLM structural prior}
\rightarrow
\text{SC-GPR + HVD}
\rightarrow
\text{LLM/posterior candidate mixture}
\rightarrow
\text{SC-OLHKG scoring}
\rightarrow
\text{evidence-gated update}.
]

---

## 7.1 新增模块

建议新增：

```text
sc_olhkg/
  llm_prior/
    prompt_builder.py
    llm_client.py
    schema.py
    validator.py
    cache.py
    audit.py
    evidence_gate.py
  features/
    feature_dsl.py
    state_coupled_basis.py
    hvd_basis.py
  variance/
    orthogonal_hvd.py
  candidates/
    llm_region_sampler.py
    posterior_candidate_generator.py
    mixture_candidate_generator.py
  acquisition/
    sc_olhkg.py
```

---

## 7.2 `prompt_builder.py`

只允许序列化 admissible information：

```python
class PromptBuilder:
    def build(self, problem_schema, observed_data, posterior_diagnostics, hvd_diagnostics, budget_state):
        """
        Must not include:
        - true objective function
        - true constraint function
        - true sigma
        - true optimum
        - benchmark name
        - active axis oracle
        - hidden formula
        """
        ...
```

prompt 中要写明：

```text
You are not allowed to infer or use hidden benchmark formulas.
You may only propose structural features using observed variable names,
trajectory fields, and current posterior diagnostics.
Return valid JSON only.
```

---

## 7.3 `schema.py`

强制 LLM 输出结构化 JSON：

```python
class LLMStructuralPrior(BaseModel):
    basis_proposals: list[BasisProposal]
    candidate_region_priors: list[PsiRegion]
    acquisition_weights: AcquisitionWeights
    uncertainty_flags: list[str]
    confidence: float
```

---

## 7.4 `validator.py`

验证 LLM 有没有用非法字段：

```python
class PriorValidator:
    def validate(self, prior, allowed_symbols):
        # reject expressions using unknown variables
        # reject benchmark names
        # reject direct coordinate oracle phrases
        # reject true_sigma / true_objective / optimum references
        ...
```

输出：

[
\text{valid prior}
\quad\text{or}\quad
\text{abstain}.
]

---

## 7.5 `feature_dsl.py`

不要让 LLM 直接写任意 Python。让它只能从 DSL 里组合：

```text
mean(field)
std(field)
quantile(field, q)
sum(field)
max(field)
lag_diff(field)
cross(field1, field2)
indicator(field > threshold)
soft_cluster(features, K)
random_projection(features, r)
orthogonalize(features)
```

这样可以防 prompt/code injection，也可以审计 prior。

---

## 7.6 `evidence_gate.py`

实现三种 gate：

```python
class EvidenceGate:
    def update_hvd_gate(self, cv_loglik_llm, cv_loglik_baseline):
        ...

    def update_candidate_gate(self, acq_mass_llm, acq_mass_baseline):
        ...

    def update_prior_mean_gate(self, pred_loglik_llm, pred_loglik_baseline):
        ...
```

门控权重：

[
\lambda_n
=========

\sigma(\kappa\Delta S_n).
]

---

## 7.7 `mixture_candidate_generator.py`

替换原先 problem-specific anchors/refinement：

```python
candidate_set = (
    space_filling_candidates
    + posterior_elite_candidates
    + boundary_candidates
    + variance_informative_candidates
    + gated_llm_region_candidates
)
```

注意：LLM candidates 只进候选池，不直接选中。

最终仍然：

```python
scores = sc_olhkg(candidate_set)
x_next = candidate_set[argmax(scores)]
```

---

# 8. LLM API 调用频率：不要每一步都调用

直接每轮调用 LLM 会有三个问题：

1. 不稳定；
2. 成本高；
3. 复现性差。

建议分三种模式。

---

## Mode A：initial structural prior

在 (n=0) 或 initial samples 后调用一次：

[
\mathrm{LLM}(P_0)
\rightarrow
\psi_0,\ q_{\mathrm{LLM},0}.
]

这是最安全、最容易写论文的版本。

---

## Mode B：milestone update

只在固定 milestones 调用：

[
n\in{0,10,20,40}.
]

每次输入的是 compressed diagnostics，不是所有 raw data。

---

## Mode C：offline/frozen transcript

为了论文复现，主实验可以使用 cached LLM outputs：

```text
llm_cache/
  problem_hash_seed_model_promptid.json
```

论文里声明：

1. main results use cached transcripts；
2. API model/version/temperature/prompt hash 全部记录；
3. ablation 使用 local open-source LLM 或 no-LLM；
4. replay mode 可完全复现。

OR 审稿人会非常关心复现性。如果实验依赖一个不可控线上 API 且没有缓存 transcript，风险很大。

---

# 9. 实验设计：证明不是 LLM 作弊

你需要做一套“LLM leakage audit”。

---

## 9.1 Prompt leakage audit

四种 prompt：

### Prompt 0：illegal full prompt

包含 benchmark name / true function description。
只作为 sanity check，不进主结果。

### Prompt 1：problem-name prompt

包含 “RZDT / FactorShock / Inventory / Queue” 名称，但不含 true function。
这要小心，很可能有 memorization risk。

### Prompt 2：anonymized schema prompt

不含 benchmark 名称，只含变量类型、bounds、trajectory field descriptions、observed data summary。
这是主实验。

### Prompt 3：data-only prompt

连领域语义都尽量去掉，只给 normalized observed summaries。
这是最严格版本。

主结果应使用 Prompt 2；Prompt 3 作为 robustness check；Prompt 0/1 只能放 leakage sensitivity。

---

## 9.2 Module ablations

必须有：

1. no LLM；
2. LLM candidates only；
3. LLM basis only；
4. LLM acquisition weights only；
5. LLM basis + HVD gate；
6. LLM candidates + candidate gate；
7. full EG-LLM-SC-OLH-KG；
8. LLM without evidence gate；
9. LLM with shuffled outputs；
10. LLM oracle prompt upper bound。

如果 “LLM without evidence gate” 有时更差，而 “evidence-gated LLM” 稳定更好，论文可信度会大幅提高。

---

## 9.3 Wrong-prior stress test

让 LLM 拿到错误 domain description 或随机打乱 feature names：

[
\text{Inventory prompt} \rightarrow \text{Queue problem}
]

如果算法没有 gate，应该崩；有 gate 应该自动降权回 baseline。

这能证明：

[
\boxed{
\text{系统不是盲信 LLM。}
}
]

---

## 9.4 Leave-one-domain-out + LLM

你前面已经想做 LODO meta-prior。LLM 版本可以做：

[
\text{source-domain records}
\rightarrow
\text{LLM summarizes transferable structural priors}
\rightarrow
\text{freeze}
\rightarrow
\text{held-out domain test}.
]

但是要注意：LLM 不能看到 held-out target true formula。
最好每个 held-out domain 都匿名化。

---

# 10. 理论主线怎么写

可以形成一个很强的 OR 理论框架。

---

## Definition 1：LLM-admissible prior

LLM prior (Z_n) 是 admissible，如果：

[
Z_n
===

\mathrm{LLM}(S(\mathcal I_n),\xi_n)
]

且 (S(\mathcal I_n)) 不包含 target oracle information。

算法选择：

[
x_{n+1}
=======

\Pi(D_n,Z_n)
]

是：

[
\sigma(\mathcal I_n,\xi_n)\text{-measurable}.
]

这正式回答“是否作弊”。

---

## Theorem 1：LLM-HVD decomposition remains valid after validation

如果 validator 输出的 LLM basis 满足：

[
\psi_n(x,\mathcal T)
====================

(A_n,N_n)
]

且 residual process 投影误差为 (\epsilon_{\psi,n})，则：

[
\mathrm{Var}(C(x)\mid\mathcal T)
================================

A_n^\top\Lambda A_n
+
N_n^\top B N_n
+
N_n^\top\omega
+
\sigma_0^2
+
\Delta_{\psi,n},
]

其中：

[
|\Delta_{\psi,n}|\le \epsilon_{\psi,n}.
]

---

## Theorem 2：Evidence gate consistency

如果 LLM prior 的 expected log score 比 baseline 差：

[
\mathbb E[S^{\mathrm{LLM}}-S^0]<-\delta,
]

则：

[
\lambda_n\rightarrow 0.
]

如果 LLM prior 比 baseline 好：

[
\mathbb E[S^{\mathrm{LLM}}-S^0]>\delta,
]

则：

[
\lambda_n\rightarrow 1.
]

这条定理说明 LLM prior 是可被数据打败的。

---

## Theorem 3：Safe certification with gated LLM-HVD

定义：

[
v^+_{C,n}(x)
============

v^{\mathrm{HVD}}_{n}(x;\lambda_n\psi^{\mathrm{LLM}}+(1-\lambda_n)\psi^0)
+
\Gamma_n.
]

若：

[
g(x)\le m^g_n(x)+\sqrt{\beta_n}s^g_n(x),
]

[
v_C(x)\le v^+_{C,n}(x),
]

则：

[
m^g_n(x)
+
\sqrt{\beta_n}s^g_n(x)
+
z_{1-\alpha}\sqrt{v^+_{C,n}(x)}
\le b
]

推出：

[
\mathbb P(C(x)>b)\le \alpha+\delta_{\mathrm{approx}}.
]

---

## Theorem 4：KG one-step optimality under LLM-admissible information

定义 terminal value：

[
V_n^\star
=========

\min_{x\in\mathcal F_n}m^J_n(x).
]

LLM-conditioned KG：

[
\mathrm{EG\text{-}LLM\text{-}SC\text{-}OLHKG}_n(x)
==================================================

## V_n^\star

\mathbb E[V_{n+1}^\star\mid \mathcal I_n,Z_n,x].
]

则：

[
x_n^\star
=========

\arg\max_x
\mathrm{EG\text{-}LLM\text{-}SC\text{-}OLHKG}_n(x)
]

在所有基于 ((\mathcal I_n,Z_n)) 的 one-step lookahead policies 中 Bayes optimal。

---

## Theorem 5：finite-budget regret with LLM coverage and fallback

令 candidate proposal：

[
q_n
===

\epsilon q_{\mathrm{space}}
+
(1-\epsilon)
[
\lambda_n q_{\mathrm{LLM},n}
+
(1-\lambda_n)q_{\mathrm{posterior},n}
].
]

如果：

[
q_n(\mathcal G_\delta)\ge p_\delta,
]

则：

[
r_N
\le
C_1\sqrt{\frac{\gamma_N(k_\psi)}{N}}
+
C_2\sqrt{\frac{d_{\mathrm{HVD}}\log N}{n_{\mathrm{eff}}}}
+
C_3\epsilon_\psi
+
C_4\sqrt{\frac{\log(1/\delta)}{M p_\delta}}.
]

LLM 的理论作用是提高 (p_\delta) 和降低 (\epsilon_\psi)，但 (\epsilon q_{\mathrm{space}}) 保证 LLM 错了也不至于让算法完全失效。

---

# 11. 这个方案是否比 LODO meta-prior 更适合？

我的判断：

[
\boxed{
\textbf{最稳的 OR 主线仍然是 LODO learned prior；LLM 是 prior-construction layer，不是替代 LODO。}
}
]

也就是说，最好做成：

[
\boxed{
\textbf{LODO + LLM-generated structural prior + evidence gate}}
]

而不是：

[
\text{single target problem + online LLM prompting}.
]

原因：

1. 单 target online LLM 容易被审稿人质疑 prompt engineering / hidden leakage；
2. LODO 能证明先验可迁移；
3. evidence gate 能证明不是盲信 LLM；
4. cached transcripts 能解决复现问题；
5. SC/HVD 仍然是数学核心。

最终叙事：

[
\boxed{
\text{LLM does not solve the optimization problem; it proposes a transferable structural coordinate system } \psi=(A,N).
}
]

[
\boxed{
\text{SC-OLH-KG then performs certified Bayesian optimization in this coordinate system.}
}
]

---

# 12. 推荐的最终算法

## Algorithm: EG-LLM-SC-OLH-KG

**Input**：

[
N,\ n_0,\ \alpha,\ b,\ \Theta,\ \text{problem schema},\ \text{trajectory fields}.
]

**Step 0. Anonymize problem description**

删除 benchmark 名称、true formulas、hidden active axes。

---

**Step 1. Initial design**

[
x_1,\ldots,x_{n_0}
\sim
\epsilon q_{\mathrm{space}}
+
(1-\epsilon)q_{\mathrm{generic}}.
]

所有 initial samples 计入预算。

---

**Step 2. Run simulations**

收集：

[
D_{n_0}
=======

{(x_i,Y_i,C_i,\mathcal T_i)}_{i=1}^{n_0}.
]

---

**Step 3. Build LLM prompt**

包含：

1. allowed problem schema；
2. observed sample summary；
3. trajectory field descriptions；
4. posterior diagnostics；
5. residual/HVD diagnostics；
6. budget state；
7. output JSON schema。

不包含：

1. true objective；
2. true constraint；
3. true sigma；
4. true optimum；
5. benchmark name；
6. hidden axis；
7. feasibility boundary oracle。

---

**Step 4. LLM proposes structural prior**

[
Z_n
===

\mathrm{LLM}(P_n).
]

---

**Step 5. Validate and compile prior**

[
Z_n
\rightarrow
\psi^{\mathrm{LLM}}_n=(A^{\mathrm{LLM}}_n,N^{\mathrm{LLM}}_n)
]

via DSL validator.

Invalid output：

[
\lambda_n=0.
]

---

**Step 6. Fit SC-GPR**

[
J(x),g(x)
\sim GP
]

with basis:

[
\phi(x)
=======

[
1,x,x^2,\rho(x),\psi(x)].
]

---

**Step 7. Fit HVD**

[
\mathrm{Var}(C(x)\mid\mathcal T)
================================

A^\top\Lambda A
+
N^\top B N
+
N^\top\omega
+
\sigma_0^2.
]

---

**Step 8. Evidence gate**

Compute:

[
\lambda_n
=========

\mathrm{Gate}
(
\text{LLM-HVD CV likelihood},
\text{baseline-HVD CV likelihood},
\text{candidate acquisition mass},
\text{feasibility calibration}
).
]

---

**Step 9. Generate candidates**

[
\mathcal A_n
============

\mathcal A_{\mathrm{space}}
\cup
\mathcal A_{\mathrm{posterior}}
\cup
\mathcal A_{\mathrm{boundary}}
\cup
\mathcal A_{\mathrm{variance}}
\cup
\mathcal A_{\mathrm{LLM}}.
]

---

**Step 10. Select by SC-OLHKG**

[
x_{n+1}
=======

\arg\max_{x\in\mathcal A_n}
\left[
w_{J,n}\mathrm{KG}^{J}*n(x)
+
w*{F,n}\mathrm{KG}^{F}*n(x)
+
w*{V,n}\mathrm{KG}^{V}*n(x)
+
w*{\rho,n}\mathrm{KG}^{\rho}_n(x)
\right].
]

---

**Step 11. Update and repeat**

At milestones (n\in{0,10,20,40}), refresh LLM prior; otherwise only update GP/HVD/gates.

---

# 13. 论文里要怎么客观评价“是否可行”

我会非常直接地写：

[
\boxed{
\textbf{LLM API 接入可行，但它不能作为无约束 oracle；必须是 admissible, logged, gated, and replaceable.}
}
]

具体要求：

1. **Admissible**：prompt 不包含 target truth；
2. **Logged**：prompt、response、model version、temperature、seed、time 全部记录；
3. **Gated**：LLM prior 必须通过 observed evidence 获得权重；
4. **Replaceable**：有 no-LLM、local-LLM、cached-LLM、wrong-LLM ablation；
5. **Fallback**：(\epsilon q_{\mathrm{space}}) 永远存在；
6. **Auditable**：每个候选点要记录来自 random / posterior / boundary / variance / LLM 哪个 channel；
7. **Reproducible**：主实验用 cached transcript replay。

如果做不到这些，LLM API 接入确实容易被认为是 prompt hacking 或 benchmark-specific tuning。

---

# 14. 最适合当前项目的路线

我建议你不要立刻把所有东西都改成 LLM。按三阶段来。

---

## Phase 1：LLM-HVD basis proposal only

只让 LLM 生成：

[
\psi=(A,N)
]

的候选 feature DSL。

不让它生成 candidate anchors，不让它调 acquisition weights。

目标是证明：

[
\text{LLM-generated } \psi
]

能提高：

1. residual variance likelihood；
2. chance feasibility calibration；
3. true feasible rate；
4. N=80 regret。

这是最干净、最容易解释的一版。

---

## Phase 2：LLM candidate region proposal

加入：

[
q_{\mathrm{LLM}}(x)
]

但只进候选池，不直接采样。

做 ablation：

[
\text{LLM basis only}
]

[
\text{LLM candidates only}
]

[
\text{LLM basis + candidates}
]

---

## Phase 3：LLM acquisition strategist

让 LLM 调：

[
(w_J,w_F,w_V,w_\rho)
]

但必须经过 source validation 或 online gate。

这部分可以放 appendix 或 extended version。主论文未必需要。

---

# 15. 你现在最应该改的代码点

基于当前 repo，我建议先实现四个最小改动。

---

## 15.1 把 problem-specific prior 全部标记为 oracle

例如：

```python
prior_type in {
    "strict_universal",
    "domain_specific",
    "llm_admissible",
    "llm_oracle_prompt",
    "oracle_upper_bound"
}
```

主结果只允许：

```python
prior_type="llm_admissible"
```

---

## 15.2 增加 `LLMStructuralPrior`

```python
class LLMStructuralPrior:
    def propose(self, audit_safe_summary):
        prompt = self.prompt_builder.build(audit_safe_summary)
        response = self.llm_client.call(prompt)
        prior = self.schema.parse(response)
        return self.validator.validate(prior)
```

---

## 15.3 增加 `HVDGate`

```python
class HVDGate:
    def compute_weight(self, llm_basis, baseline_basis, observed_data):
        s_llm = cross_validated_hvd_loglik(llm_basis, observed_data)
        s_base = cross_validated_hvd_loglik(baseline_basis, observed_data)
        return sigmoid(kappa * (s_llm - s_base - threshold))
```

---

## 15.4 改 `_generate_candidate_set`

当前 main loop 里已经有 candidate generation、KG compute、boundary acquisition、Pareto-KG selection 的流程。([GitHub][7]) 你只需要在 `_generate_candidate_set` 后面追加：

```python
if llm_gate_weight > 0:
    candidate_set += llm_region_sampler.sample(
        llm_regions,
        weight=llm_gate_weight,
        n=n_llm_candidates
    )
```

然后仍然让 SC-OLHKG 选点。

---

# 16. 最后给一个明确判断

这条路值得做，但要把论文主张写得非常精确：

不要写：

[
\text{We use an LLM to optimize simulation problems.}
]

要写：

[
\boxed{
\text{We use an LLM to propose admissible, evidence-gated structural priors for state-coupled heteroscedastic risk decomposition.}
}
]

不要写：

[
\text{LLM gives good anchors.}
]

要写：

[
\boxed{
\text{LLM proposes candidate regions in } \psi=(A,N)\text{ space; KG still performs certified value-of-information selection.}
}
]

不要写：

[
\text{LLM confidence is reliable.}
]

要写：

[
\boxed{
\text{LLM confidence is ignored unless validated by predictive evidence.}
}
]

最终主贡献可以是：

[
\boxed{
\textbf{A certified KG framework in which language-model priors are admissible, falsifiable, and only useful insofar as they improve state-coupled HVD evidence.}
}
]

这既能保住你最核心的：

[
\mathrm{Var}(C(x)\mid\mathcal T)
================================

A^\top\Lambda A
+
N^\top B N
+
N^\top\omega
+
\text{floor}
]

也能把“强先验”从人工手写的 benchmark-specific trick，变成一个可审计、可复现、可门控、可迁移的 AI-assisted structural prior。

[1]: https://www.nature.com/articles/s41586-023-06924-6 "Mathematical discoveries from program search with large language models | Nature"
[2]: https://arxiv.org/abs/2402.03921 "[2402.03921] Large Language Models to Enhance Bayesian Optimization"
[3]: https://arxiv.org/abs/2501.16224 "[2501.16224] Language-Based Bayesian Optimization Research Assistant (BORA)"
[4]: https://arxiv.org/html/2602.07904v1 "Adaptive Acquisition Selection for Bayesian Optimization with Large Language Models"
[5]: https://arxiv.org/abs/2606.01730 "[2606.01730] Evidence-Gated LLM Priors for Multi-Objective Bayesian Optimization"
[6]: https://arxiv.org/html/2503.10009v1 "OR-LLM-Agent: Automating Modeling and Solving of Operations Research Optimization Problem with Reasoning Large Language Model"
[7]: https://raw.githubusercontent.com/erzhu419/KG_op/main/Final_Submission/GPR_KG_Code/gpr_kg.py "raw.githubusercontent.com"

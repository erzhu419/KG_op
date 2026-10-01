# 我：
我在做一个从被 desk reject 的 OR 论文上重建的新项目：SC-OLH-KG。核心想法是把 state-coupled structure 和 heteroscedastic variance decomposition 合成一个理论对象：

ψ(x) = (A(x), N(x))
Var(C(x) | T) = A(x)^T Λ A(x) + N(x)^T B N(x) + N(x)^T ω + floor

其中 x 是 policy/design，A 是 local/idiosyncratic exposure，N 是 shared-shock exposure。HVD、chance certification、candidate generation、GPR state basis、KG acquisition 都围绕这个 ψ 坐标工作。

目前发现一个重要问题：

1. 算法没有直接作弊：
   acquisition / candidate selection 没有调用 true objective / true constraint / true sigma / true optimum。true functions 只用于最终 evaluation/regret。

2. 但当前好结果强依赖结构先验：
   - problem-specific initial_samples
   - state_anchor_points / inverse_state_anchor
   - recommendation_refinement_candidates
   - problem-specific risk_exposures / policy_summary

3. 小规模 audit 结果：
   HighDimStatePolicyRZDT1, d=1000, N=12, seeds=3:
   - current: true feasible rate 1.0, median feasible regret 0.0029
   - no calibration + no axis oracle: true feasible rate 1.0, regret 0.0039
   - no refinement + no calibration + no axis oracle: true feasible rate 0.667, regret 0.020
   - full no-cheat: disable problem initial, boundary initial, refinement, calibration, axis oracle: true feasible rate 0.0

4. 跨问题迁移 quick check：
   strict universal prior = 关 problem initial / boundary initial / state candidate anchors / refinement / calibration / axis，只保留同一套 raw+state / factor-HVD 机制。
   domain prior = 关 calibration / axis，但保留每个问题自己的 initial/state/refinement。

   d=50, N=10, seeds=3, additive smoke:
   - strict universal:
     - FactorShock feasible 0.0, violation mean 2.499
     - Inventory feasible 0.0, violation mean 0.062
     - Queue feasible 0.0, violation mean 0.070
   - domain prior:
     - FactorShock feasible 1.0, regret 0.0
     - Inventory feasible 0.0, violation mean 0.123
     - Queue feasible 0.0, violation mean 0.119

所以问题是：

当前项目的“结构先验”还不是像 L2 regularization 那样的通用 prior，而是“通用 SC/HVD 框架 + problem-specific exposure/anchor/refinement”。这可能会被审稿人认为是工程堆砌或 domain-specific tuning。

我想保住核心贡献：state-coupled cumulative heteroscedastic risk decomposition + certified KG，但要避免看起来像每个 benchmark 手写一套先验。

请你给一个下一步方案，重点回答：

1. 这个项目应该如何重新定义“可接受的先验”？
   - 什么样的 ψ(x) / A,N / state encoder 可以算通用？
   - 什么样的 anchor/refinement 算作弊或 unfair？
   - 是否应该把 problem-specific refinement 完全移出主线，只放 oracle/upper-bound ablation？

2. 如何做成一个 leave-one-domain-out meta-prior？
   例如从 FactorShock / Inventory / Queue / Traffic 的轨迹或 policy records 中学习一个统一 representation / anchor proposal distribution，然后冻结，在 held-out domain 测试。
   这样是否能把先验从 problem-specific 变成 learned transferable prior？
请给一个能投 OR 的方案，要求：
- 不像工程堆砌；
- 能解释为什么 N=80 有效而不是作弊；
- 保留 SC 和 HVD 的高依赖关系；
- 有明确 ablation 证明不是手写答案；
- 数学深度足够强。


# GPT：
可以把项目重建成一个非常清楚的 **“admissible learned prior + certified KG”** 框架。现在的结果说明的不是“SC-OLH-KG 不成立”，而是：

[
\boxed{
\text{当前实现的强项来自结构先验；下一步必须证明这些先验不是 target-specific oracle，而是可迁移、可审计、可冻结的 meta-prior。}
}
]

我建议把主论文从“我们提出了很多工程模块”改成：

[
\boxed{
\textbf{在一个可接受信息集下，学习一个跨域 transferable SC/HVD prior，并证明 KG 在该 prior 下以低有效维度工作。}
}
]

下面是完整方案。

---

# 0. 先定性：strict universal prior 全失败并不致命

你的 audit 结果其实很有价值：

[
\text{strict universal prior} \Rightarrow \text{几乎不可行}
]

[
\text{domain prior} \Rightarrow \text{部分问题立刻好很多}
]

这说明 SC-OLH-KG 的能力来自结构信息，而不是普通 BO 的“魔法”。这在高维小预算优化里是合理的。No Free Lunch theorem 的核心正是：如果平均到所有可能问题，任何算法都没有普遍优势；能成功的算法一定利用了某类问题的结构。Wolpert 和 Macready 的原论文明确把优化算法性能与问题结构之间的匹配作为核心问题。([不列颠哥伦比亚大学计算机科学][1])

所以论文不应该假装“没有 prior 也能在 (d=1000,N=80) 成功”。那会不可信。应该正面说：

[
\boxed{
\textbf{SC-OLH-KG 的样本效率来自 learned structural prior；贡献是把这个 prior 做成可迁移、可认证、非 oracle 的先验。}
}
]

---

# 1. 重新定义“可接受的先验”

我建议在论文里专门放一个小节：

> **Admissible Structural Priors and Oracle Information**

这是防 desk reject 的关键。

## 1.1 信息集定义

令目标 domain 为 (d^\star)。在第 (n) 轮，算法允许使用的信息集定义为：

[
\mathcal I_n^{d^\star}
======================

\left{
\mathcal D_{\mathrm{source}},
\mathcal M_{\mathrm{obs}}^{d^\star},
(\theta_i,Y_i,C_i,\mathcal T_i)_{i=1}^{n}
\right}.
]

其中：

[
\mathcal D_{\mathrm{source}}
]

是其他 source domains 的历史 policy records / trajectories / outputs；

[
\mathcal M_{\mathrm{obs}}^{d^\star}
]

是目标问题在真实使用时也能看到的 metadata，例如维度、变量上下界、可执行 policy format、仿真轨迹字段、网络拓扑、库存容量、队列数量等；

[
(\theta_i,Y_i,C_i,\mathcal T_i)_{i=1}^{n}
]

是目标 domain 已经花预算得到的仿真样本。

算法在第 (n+1) 轮选择：

[
\theta_{n+1}
]

必须满足：

[
\theta_{n+1}
\quad \text{is } \mathcal I_n^{d^\star}\text{-measurable}.
]

也就是说，candidate generation、KG acquisition、refinement、calibration 都只能依赖当前可观测信息。只要用到了 target true objective、true constraint、true sigma、true optimum、true feasible boundary、hidden coordinate oracle，就违反 admissibility。

这条定义非常重要。它把“是否作弊”从口头争论变成数学条件。

---

# 2. 先验分层：哪些可以进主线，哪些只能做 oracle ablation

我建议把所有 prior 分成四层。

---

## Tier A：通用算法先验，主线允许

这类 prior 不依赖具体 benchmark 的隐藏结构，只依赖算法设计。比如：

[
\psi(\theta,\mathcal T)
=======================

(A(\theta,\mathcal T),N(\theta,\mathcal T))
]

的通用形式：

[
\mathrm{Var}(C(\theta)\mid \mathcal T)
======================================

A^\top \Lambda A
+
N^\top B N
+
N^\top \omega
+
\sigma_0^2.
]

这里 (A,N) 的**形式**是通用的，但它们的数值必须由统一 encoder 从可观测轨迹中算出。

允许的通用机制包括：

[
\text{trajectory moments},
\quad
\text{occupancy statistics},
\quad
\text{random projections},
\quad
\text{orthogonalized basis},
\quad
\text{soft clustering},
\quad
\text{state-coupled kernel},
\quad
\text{Bayesian shrinkage},
\quad
\text{mixture with space-filling exploration}.
]

这些类似 L2 regularization：它们偏好某种低复杂度结构，但不包含目标问题的答案。

---

## Tier B：learned transferable meta-prior，主线允许

这类 prior 是从 source domains 学出来的，然后在 held-out target domain 上冻结使用。

例如：

[
e_\phi(\mathcal T)\mapsto \psi_\phi(\theta,\mathcal T)
======================================================

(A_\phi,N_\phi),
]

[
q_\phi(\theta\mid \text{target metadata / pilot records})
=========================================================

\text{anchor proposal distribution}.
]

只要满足三点，就可以放主线：

1. source domains 不包含 held-out target domain；
2. encoder / proposal / hyperparameters 在 target evaluation 前冻结；
3. target domain 只通过已花预算获得的 samples 更新 posterior，不允许用 hidden true functions。

这类方法有明确文献位置。MetaBO 用相关任务 meta-train acquisition function，让 BO 自动利用任务族结构；PFNs4BO 用 prior-data fitted networks 通过 in-context learning 近似 BO surrogate 的 posterior predictive distribution，并能把 prior 信息纳入 BO；pre-trained GP / transfer BO 也都在做“从一族任务学 prior，然后迁移到新任务”的事情。([arXiv][2])

所以你可以把 SC-OLH-KG 定位为：

[
\boxed{
\textbf{不是 universal optimizer，而是 source-trained structural optimizer for a family of stochastic simulation problems.}
}
]

---

## Tier C：真实应用中可声明的 expert prior，主线慎用

比如交通信号问题中，专家知道：

[
\text{queue length, spillback, phase switch, density}
]

是重要状态特征。库存问题中，专家知道：

[
\text{inventory level, demand backlog, replenishment delay}
]

重要。

这类 prior 可以使用，但必须在论文里声明：

[
\text{This is expert-observable domain metadata, not oracle target performance information.}
]

不过为了避免审稿人说 domain-specific tuning，我建议主论文不要把这层作为主结果，而是作为应用扩展。

---

## Tier O：oracle / unfair prior，只能做 upper-bound ablation

下面这些应该从主线移出：

[
\text{problem-specific initial_samples}
]

[
\text{state_anchor_points / inverse_state_anchor}
]

[
\text{recommendation_refinement_candidates}
]

[
\text{problem-specific risk_exposures / policy_summary}
]

[
\text{axis oracle}
]

[
\text{calibration using true sigma / true boundary}
]

除非它们完全由 source-trained frozen module 生成，否则都应该进入：

[
\boxed{
\textbf{Oracle / Domain-Tuned Upper Bound}}
]

而不是主方法。

你的 audit 已经说明：full no-cheat 下 feasible rate 0，而有 refinement / anchor / domain prior 时性能大幅提高。这意味着这些模块对结果贡献很大。如果主文继续把它们放在主方法里，审稿人很容易质疑是 benchmark-specific tuning。

---

# 3. 什么样的 (\psi(x)=(A,N)) 算通用？

核心标准是：

[
\boxed{
\psi \text{ 必须由一个跨问题共享的 mapping 从可观测 trajectory / policy records 中生成。}
}
]

不能是：

[
\psi_{\mathrm{FactorShock}}(x),
\quad
\psi_{\mathrm{Inventory}}(x),
\quad
\psi_{\mathrm{Queue}}(x)
]

分别手写。

应该是：

[
\psi_\phi(x,\mathcal T)
=======================

(A_\phi(x,\mathcal T),N_\phi(x,\mathcal T)),
]

其中同一个 (\phi) 用于所有 domains。

---

## 3.1 (A)：local / idiosyncratic exposure

(A) 不应该是“手写哪个坐标重要”，而应该是从轨迹中学习或计算出的连续正交风险因子：

[
A_\phi(\theta,\mathcal T)
=========================

\sum_{t=0}^{T}
a_\phi(S_t,A_t,\theta).
]

约束：

[
\mathbb E[A]=0,
]

[
\mathbb E[AA^\top]=I.
]

它表示局部 exposure，比如：

[
\text{local congestion},
\quad
\text{inventory stress},
\quad
\text{queue overload},
\quad
\text{policy instability},
\quad
\text{rare-state exposure}.
]

但这些语义不需要手写；它们可以通过轨迹 encoder 学出来。

---

## 3.2 (N)：shared-shock exposure

(N) 应该是 soft regime occupancy：

[
N_\phi(\theta,\mathcal T)
=========================

\sum_{t=0}^{T}
\pi_\phi(S_t,A_t,\theta)
\in\mathbb R_+^K,
]

其中：

[
\pi_\phi(S_t,A_t,\theta)\in\Delta^{K-1}.
]

它表示第 (k) 个 shared-shock regime 的累计暴露量。

例如同一个 (N_k) 可以在不同问题里对应不同语义：

[
\text{high demand regime},
\quad
\text{congested regime},
\quad
\text{near-capacity regime},
\quad
\text{stockout-risk regime}.
]

但是算法不需要知道这些名字。只要它能跨 source domains 学到“某些轨迹模式导致 correlated residual shock”，就可以。

---

## 3.3 统一 HVD 形式

主模型固定为：

[
\boxed{
\mathrm{Var}(C(\theta)\mid \mathcal T)
======================================

A_\phi^\top \Lambda A_\phi
+
N_\phi^\top B N_\phi
+
N_\phi^\top \omega
+
\sigma_0^2.
}
]

参数约束：

[
\Lambda\succeq 0,\quad \Lambda\text{ diagonal},
]

[
B\succeq 0,
]

[
\omega\ge 0,
]

[
\sigma_0^2\ge 0.
]

这不是工程堆砌，而是一个统一的 variance geometry：

[
A^\top\Lambda A
===============

\text{orthogonal local factor risk},
]

[
N^\top B N
==========

\text{shared shock / systematic covariance risk},
]

[
N^\top\omega
============

\text{idiosyncratic regime-level noise},
]

[
\sigma_0^2
==========

\text{irreducible floor}.
]

---

# 4. 什么样的 anchor / refinement 是公平的？

## 4.1 公平 anchor

公平 anchor 必须来自下面三类之一：

### 第一类：space-filling anchor

例如 Latin hypercube、Sobol、random projection、coordinate-free trust region。
它不使用问题答案，所以公平。

### 第二类：posterior anchor

例如当前 posterior 下的：

[
\arg\min m_n^J(\theta),
]

[
\arg\min |M_n(\theta)|,
]

[
\arg\max \mathrm{Var}_n(M_n(\theta)),
]

其中：

[
M_n(\theta)
===========

b-
m_n^g(\theta)
-------------

z_{1-\alpha}\sqrt{v_{C,n}^+(\theta)}.
]

这些只依赖已观测样本和 posterior，所以公平。

### 第三类：frozen meta-anchor

从 source domains 学到：

[
q_\phi(\theta\mid \mathcal M_{\mathrm{obs}}^{d^\star},D_n^{d^\star})
]

然后在 target 上冻结或只通过 target observed samples 更新 posterior。这个也公平。

---

## 4.2 不公平 anchor

不公平的是：

[
\text{state_anchor_points}
]

如果它们是根据 target hidden state geometry 手写的；

[
\text{inverse_state_anchor}
]

如果它相当于知道“哪个 policy 可以命中哪个好状态”；

[
\text{boundary initial}
]

如果它直接利用 target true feasible boundary；

[
\text{axis oracle}
]

如果它知道高维里哪个坐标是有效维；

[
\text{calibration}
]

如果它用 true sigma / true constraint distribution；

[
\text{recommendation_refinement_candidates}
]

如果它用 target true optimum 周围的人工候选。

这些都要移到 oracle upper-bound ablation。

---

# 5. 主线应移除 problem-specific refinement

我的建议非常明确：

[
\boxed{
\textbf{problem-specific refinement 不要放主方法。}
}
]

它可以保留，但只能作为：

[
\textbf{Domain-tuned / Oracle upper bound}
]

或者：

[
\textbf{Expert-assisted variant}
]

主方法应该只有：

[
\text{generic candidate generator}
+
\text{frozen meta proposal}
+
\text{posterior KG refinement}.
]

即 refinement 应该优化：

[
\mathrm{SC\text{-}OLHKG}_n(\theta)
]

而不是优化 true objective / true boundary / target-specific inverse map。

具体地，recommendation refinement 改成：

[
\theta' =
\arg\max_{\theta\in \mathcal N_\psi(\theta_0)}
\mathrm{SC\text{-}OLHKG}_n(\theta),
]

其中邻域在 learned (\psi)-space 中定义：

[
\mathcal N_\psi(\theta_0)
=========================

\left{
\theta:
|\psi_\phi(\theta)-\psi_\phi(\theta_0)|\le r_n
\right}.
]

这就不再是 problem-specific refinement，而是 posterior acquisition refinement。

---

# 6. Leave-one-domain-out meta-prior：主方案

你的主线应该变成：

[
\boxed{
\textbf{LODO-SC-OLH-KG: leave-one-domain-out learned prior + certified KG.}
}
]

Domains：

[
\mathcal D
==========

{
\text{FactorShock},
\text{Inventory},
\text{Queue},
\text{Traffic}
}.
]

每次选择一个 held-out domain：

[
d^\star\in \mathcal D.
]

训练集：

[
\mathcal D_{\mathrm{source}}
============================

\mathcal D\setminus{d^\star}.
]

测试只在 (d^\star) 上进行。

---

## 6.1 训练数据

每个 source domain 提供历史 records：

[
\left{
(\theta_i^d,Y_i^d,C_i^d,\mathcal T_i^d)
\right}_{i=1}^{M_d}.
]

这些可以来自：

1. random policies；
2. historical runs；
3. old BO runs；
4. domain simulators；
5. deliberately broad exploration policies。

关键是：held-out target 的 true functions 不得进入 training。

---

## 6.2 学一个统一 encoder

训练：

[
e_\phi:
(\theta,\mathcal T,\mathcal M_{\mathrm{obs}})
\mapsto
z.
]

然后：

[
A_\phi
======

W_A z,
]

[
N_\phi
======

\sum_t \mathrm{softmax}(W_N h_\phi(S_t,A_t,\theta)).
]

为了避免 (A,N) 任意混在一起，加入约束：

[
\widehat{\mathbb E}[A]=0,
]

[
\widehat{\mathbb E}[AA^\top]=I,
]

[
\widehat{\mathrm{Cov}}(A,N)=0.
]

也就是让 (A) 承担正交连续 exposure，让 (N) 承担 regime/shared-shock exposure。

---

## 6.3 学 HVD meta-prior

对 source domains，残差：

[
r_i^d
=====

C_i^d-\hat g^d(\theta_i^d).
]

拟合：

[
(r_i^d)^2
\approx
(A_i^d)^\top \Lambda^d A_i^d
+
(N_i^d)^\top B^d N_i^d
+
(N_i^d)^\top\omega^d
+
\sigma_{0,d}^2.
]

但不要把 (\Lambda,B,\omega) 固定死。更好的方式是学习一个 hierarchical prior：

[
\eta^d
======

(\lambda^d,\mathrm{vec}(B^d),\omega^d,\sigma_{0,d}^2)
\sim
P_\Theta(\eta\mid z_d),
]

其中 (z_d) 是 domain embedding。

在 held-out target 上，先验为：

[
\eta^{d^\star}\sim P_\Theta(\eta\mid z_{d^\star}),
]

然后用 target samples 做 Bayesian update。

这一步能保留“各 domain 方差结构不同”的现实性，同时避免为每个 benchmark 手写 exposure。

---

## 6.4 学 candidate proposal distribution

学一个 proposal：

[
q_\phi(\theta\mid \mathcal C_n),
]

其中 context：

[
\mathcal C_n
============

\left(
\mathcal M_{\mathrm{obs}}^{d^\star},
(\theta_i,Y_i,C_i,\mathcal T_i)_{i=1}^n
\right).
]

训练目标不是“直接输出 true optimum”，而是学会提出高信息候选：

[
\theta
\sim q_\phi
]

应该覆盖：

1. low predicted objective region；
2. chance boundary region；
3. high variance-learning region；
4. unexplored (\psi)-space region。

实际 candidate pool 用 mixture：

[
\boxed{
q_{\mathrm{cand}}(\theta)
=========================

\epsilon q_{\mathrm{space}}(\theta)
+
(1-\epsilon)
q_\phi(\theta\mid \mathcal C_n).
}
]

其中：

[
\epsilon>0.
]

这很重要。它保证即使 meta-prior 错了，算法仍有基础探索覆盖，不会完全被错误 prior 锁死。Prior-guided BO 文献里已经有“把用户或 learned prior 注入 BO 但仍保留恢复能力”的思想；BOPrO / PrBO 这类方法也专门讨论了 prior 不完全准确时的鲁棒恢复。([开放审查][3])

---

# 7. 训练 objective：不要只学 reconstruction，要学 KG-relevant representation

我建议 meta-prior 训练损失包含五项。

## 7.1 Mean prediction loss

[
\mathcal L_{\mathrm{mean}}
==========================

\sum_{d,i}
\left[
(Y_i^d-\hat J_\phi(\theta_i^d))^2
+
(C_i^d-\hat g_\phi(\theta_i^d))^2
\right].
]

---

## 7.2 Variance decomposition loss

[
\mathcal L_{\mathrm{hvd}}
=========================

\sum_{d,i}
\left[
(r_i^d)^2
---------

## (A_i^d)^\top \Lambda^d A_i^d

## (N_i^d)^\top B^d N_i^d

## (N_i^d)^\top\omega^d

\sigma_{0,d}^2
\right]^2.
]

更好的是用 Gaussian negative log likelihood：

[
\mathcal L_{\mathrm{nll}}
=========================

\sum_{d,i}
\left[
\frac{(C_i^d-\hat g_\phi(\theta_i^d))^2}{\hat v_\phi(\theta_i^d)}
+
\log \hat v_\phi(\theta_i^d)
\right].
]

---

## 7.3 Covariance matching loss

如果有重复 runs 或 shared random seeds，可以估计 residual covariance：

[
\widehat{\mathrm{Cov}}(r_i,r_j).
]

加入：

[
\mathcal L_{\mathrm{cov}}
=========================

\sum_{i,j}
\left[
\widehat{\mathrm{Cov}}(r_i,r_j)
-------------------------------

## N_i^\top B N_j

A_i^\top\Lambda A_j
\right]^2.
]

这会让 (N^\top B N) 真正学习 shared shock，而不是只是拟合 marginal variance。

---

## 7.4 Chance calibration loss

对 source domains 的 held-out records，检查：

[
\hat g_\phi(\theta)
+
z_{1-\alpha}\sqrt{\hat v_\phi(\theta)}
\le b
]

是否对应真实 violation rate：

[
\mathbb P(C(\theta)>b).
]

加入 calibration penalty：

[
\mathcal L_{\mathrm{cal}}
=========================

\left|
\widehat{\mathrm{ViolationRate}}
--------------------------------

\alpha
\right|.
]

注意：这只在 source validation 上调，不在 target test 上调。

---

## 7.5 Representation regularization

[
\mathcal L_{\mathrm{orth}}
==========================

|\widehat{\mathbb E}[AA^\top]-I|_F^2
+
|\widehat{\mathrm{Cov}}(A,N)|_F^2.
]

[
\mathcal L_{\mathrm{entropy}}
=============================

-\sum_{t,k}
\pi_k(S_t,A_t)\log \pi_k(S_t,A_t).
]

entropy 可以防止所有 states collapse 到一个 regime，也可以控制 (N) 的稀疏性。

总损失：

[
\mathcal L
==========

\mathcal L_{\mathrm{mean}}
+
\lambda_v\mathcal L_{\mathrm{hvd}}
+
\lambda_c\mathcal L_{\mathrm{cov}}
+
\lambda_{\mathrm{cal}}\mathcal L_{\mathrm{cal}}
+
\lambda_o\mathcal L_{\mathrm{orth}}.
]

---

# 8. Held-out target 上怎么运行

在 held-out domain (d^\star) 上：

## Step 1：冻结 meta-prior

冻结：

[
e_\phi,\quad
\psi_\phi,\quad
q_\phi,\quad
P_\Theta(\eta\mid z).
]

不允许在 target test 后再调。

---

## Step 2：初始化

用 mixture initial design：

[
q_{\mathrm{init}}(\theta)
=========================

\epsilon q_{\mathrm{space}}(\theta)
+
(1-\epsilon)q_\phi(\theta\mid \mathcal M_{\mathrm{obs}}^{d^\star}).
]

所有 initial samples 都计入 (N=80)。

这点很重要。不能把 meta-proposed initial 当免费信息。

---

## Step 3：posterior update

用 target samples 更新：

[
J(\theta)\mid D_n,
\quad
g(\theta)\mid D_n,
\quad
\eta\mid D_n.
]

encoder 不动。HVD 参数可以 Bayesian update，因为这是 target 已观测数据的一部分。

---

## Step 4：certified feasible set

[
\mathcal F_n
============

\left{
\theta:
m_n^g(\theta)
+
\sqrt{\beta_n}s_n^g(\theta)
+
z_{1-\alpha}\sqrt{v_{C,n}^+(\theta)}
\le b
\right}.
]

---

## Step 5：SC-OLH-KG candidate selection

candidate pool：

[
\mathcal A_n
============

\mathcal A_{\mathrm{space}}
\cup
\mathcal A_{\mathrm{meta}}
\cup
\mathcal A_{\mathrm{boundary}}
\cup
\mathcal A_{\mathrm{variance}}
\cup
\mathcal A_{\mathrm{novelty}}.
]

其中所有集合都由：

[
q_{\mathrm{space}},
\quad
q_\phi,
\quad
m_n,
\quad
s_n,
\quad
v_{C,n}^+,
\quad
\psi_\phi
]

生成，不用 target truth。

选择：

[
\theta_{n+1}
============

\arg\max_{\theta\in\mathcal A_n}
\mathrm{SC\text{-}OLHKG}_n(\theta).
]

---

# 9. 解释为什么 (N=80) 有效，而不是作弊

论文里要给出一个理论解释：

[
\boxed{
N=80 \text{ 有效，不是因为知道答案，而是因为搜索维度从 } d \text{ 降到 } d_\psi.
}
]

原始空间：

[
\theta\in\mathbb R^{1000}.
]

但是 SC-OLH-KG 实际搜索的是：

[
\psi_\phi(\theta)
=================

(A_\phi(\theta),N_\phi(\theta))
\in\mathbb R^{r+K}.
]

如果：

[
J(\theta)\approx \tilde J(\psi_\phi(\theta)),
]

[
g(\theta)\approx \tilde g(\psi_\phi(\theta)),
]

[
v_C(\theta)\approx \tilde v_C(\psi_\phi(\theta)),
]

那么 sample complexity 应该依赖：

[
d_\psi=r+K
]

而不是 (d=1000)。

可以写一个 formal assumption：

[
|J(\theta)-\tilde J(\psi(\theta))|\le \epsilon_J,
]

[
|g(\theta)-\tilde g(\psi(\theta))|\le \epsilon_g,
]

[
|v_C(\theta)-\tilde v_C(\psi(\theta))|\le \epsilon_v.
]

再写 regret bound：

[
r_N
\le
C_1
\sqrt{
\frac{\gamma_N(k_\psi)+\log(1/\delta)}{N}
}
+
C_2
\sqrt{
\frac{d_v\log N}{n_{\mathrm{eff}}}
}
+
C_3(\epsilon_J+\epsilon_g+\epsilon_v)
+
C_4\epsilon_{\mathrm{cand}}.
]

其中：

[
d_v
===

r+\frac{K(K+1)}{2}+K+1.
]

[
\gamma_N(k_\psi)
]

是 learned (\psi)-kernel 的 information gain。

这条 bound 表达：

[
\boxed{
\text{性能来自低维 transferable representation + HVD 参数估计，而不是 hidden oracle。}
}
]

---

# 10. Candidate coverage theorem：防止 meta-prior 被质疑

因为 candidate generation 是最容易被审稿人怀疑的地方，所以建议加一个 coverage theorem。

令目标最优安全解附近的好区域为：

[
\mathcal G_\epsilon
===================

\left{
\theta:
J(\theta)-J(\theta^\star)\le \epsilon,\
\theta \text{ safely feasible}
\right}.
]

candidate proposal 是 mixture：

[
q_{\mathrm{cand}}
=================

\epsilon_0 q_{\mathrm{space}}
+
(1-\epsilon_0)q_\phi.
]

只要：

[
q_{\mathrm{cand}}(\mathcal G_\epsilon)\ge p_\epsilon>0,
]

那么 (M) 个 candidates 中至少一个落入好区域的概率：

[
1-(1-p_\epsilon)^M.
]

所以 candidate error 满足：

[
\epsilon_{\mathrm{cand}}(M)
===========================

O\left(
\sqrt{
\frac{\log(1/\delta)}{M p_\epsilon}
}
\right)
]

或用 covering number 写成类似 bound。

这说明 meta-prior 的作用是增大：

[
p_\epsilon.
]

space-filling 保证：

[
p_\epsilon>0
]

而 meta-prior 让：

[
p_\epsilon
]

在结构相似的 held-out domains 上大很多。

这能解释为什么 (N=80) 有效：不是因为直接给了最优点，而是因为 learned proposal 把有效 candidate mass 放在低维 risk-relevant region。

---

# 11. 主 ablation 设计：必须能证明不是手写答案

我建议实验分成五组，不要只报一张总表。

---

## Group 1：Admissibility audit

每个模块标注是否使用 target oracle。

| Variant              | problem initial | state anchor |          refinement | calibration | axis oracle | learned frozen meta-prior | admissible |
| -------------------- | --------------: | -----------: | ------------------: | ----------: | ----------: | ------------------------: | ---------: |
| Full no-cheat strict |              no |           no |                  no |          no |          no |                        no |        yes |
| LODO-SC-OLH-KG       |              no |           no | posterior/meta only | source-only |          no |                       yes |        yes |
| Domain-tuned         |             yes |          yes |                 yes |      no/yes |      no/yes |                        no | no/main no |
| Oracle upper bound   |             yes |          yes |                 yes |         yes |         yes |                        no |         no |

这样审稿人会看到你没有隐藏问题。

---

## Group 2：Leave-one-domain-out main table

四个 held-out：

1. hold out FactorShock；
2. hold out Inventory；
3. hold out Queue；
4. hold out Traffic。

每个 held-out 都训练在其他三个 domains。

报告：

[
\text{true feasible rate},
]

[
\text{mean violation},
]

[
\text{certified feasible rate},
]

[
\text{feasible regret},
]

[
\text{best feasible objective},
]

[
\text{calibration gap}.
]

主表应该比较：

1. Random / space-filling;
2. standard KG pooled variance;
3. state-coupled KG without HVD;
4. HVD-KG without state coupling;
5. strict universal SC/HVD;
6. LODO-SC-OLH-KG;
7. domain-tuned upper bound;
8. oracle upper bound.

---

## Group 3：SC 和 HVD 的依赖关系

你要证明 SC 和 HVD 必须合起来：

[
\text{SC only} \neq \text{full}
]

[
\text{HVD only} \neq \text{full}
]

[
\text{SC + HVD} = \text{best admissible}.
]

Ablations：

[
\psi=(A,N) \text{ learned}
]

[
A \text{ only}
]

[
N \text{ only}
]

[
A,N \text{ shuffled}
]

[
B=0
]

[
\Lambda=0
]

[
\omega \text{ pooled}
]

[
\text{no state-coupled kernel}
]

[
\text{no variance KG term}
]

如果 shuffle (A,N) 之后性能掉，说明不是维度或参数数量带来的假提升，而是 exposure structure 真的有用。

---

## Group 4：candidate generation ablation

比较：

1. uniform candidates only；
2. meta candidates only；
3. posterior boundary candidates only；
4. meta + boundary；
5. meta + boundary + variance VOI；
6. full mixture。

重点报告：

[
q_{\mathrm{cand}}(\mathcal B_n),
]

即 candidate pool 中落在 predicted chance boundary / elite region 的比例。

这能说明 candidate generation 不是手写答案，而是在 learned (\psi)-space 里提高了有效候选覆盖。

---

## Group 5：target leakage tests

必须做这些 sanity checks：

### 5.1 Target label leakage test

训练 meta-prior 时强制 held-out domain 完全不可见，包括 seeds、problem variants、目标函数公式生成的 records。

### 5.2 Hyperparameter leakage test

所有 hyperparameters 只在 source validation folds 上选择。
held-out target 只评估一次或预注册多 seed protocol。

机器学习数据泄漏的一般定义就是模型训练时使用了预测时不可用的信息，这会导致离线表现虚高、部署失败。([IBM][4]) 你的审稿风险本质上就是 optimization benchmark leakage，所以必须用信息集和 LODO protocol 明确规避。

### 5.3 Permutation test

随机打乱：

[
C_i
]

或打乱：

[
(A_i,N_i)\leftrightarrow r_i^2
]

如果性能仍然好，说明模型可能靠别的泄漏；如果显著下降，说明 HVD 学到的结构是真实的。

### 5.4 Wrong-domain prior test

用最不相似 source domains 训练 prior，或随机 domain embedding。
如果效果下降但不崩溃，说明 meta-prior 有效且 mixture exploration 有恢复能力。

---

# 12. 数学主线：OR 论文的理论深度

我建议主论文理论写成六个核心结果。

---

## Definition 1：Admissible prior

一个 prior / candidate rule (\Pi) 是 admissible 的，当且仅当：

[
\theta_{n+1}
============

\Pi(\mathcal I_n^{d^\star})
]

是 (\mathcal I_n^{d^\star})-measurable。

任何依赖：

[
J^{d^\star}*{\mathrm{true}},
\quad
g^{d^\star}*{\mathrm{true}},
\quad
v^{d^\star}_{\mathrm{true}},
\quad
\theta^{\star,d^\star},
\quad
\text{hidden active axes},
\quad
\text{true feasible boundary}
]

的 rule 都不是 admissible。

---

## Theorem 1：Cumulative HVD decomposition

给定 trajectory (\mathcal T)，若 residual noise 可写为：

[
\delta_t
========

a_\phi(X_t)^\top \Lambda^{1/2} Z
+
n_\phi(X_t)^\top b
+
\eta_t,
]

其中：

[
Z\sim N(0,I),
\quad
b\sim N(0,B),
\quad
\eta_t\mid n_\phi(X_t)\sim N(0,n_\phi(X_t)^\top \omega+\sigma_0^2),
]

则：

[
\boxed{
\mathrm{Var}(C(\theta)\mid \mathcal T)
======================================

A^\top\Lambda A
+
N^\top B N
+
N^\top\omega
+
\sigma_0^2.
}
]

这保住你的核心理论对象。

---

## Theorem 2：State-coupled sufficiency bound

若存在 (\tilde J,\tilde g,\tilde v) 使：

[
|J(\theta)-\tilde J(\psi_\phi(\theta))|\le \epsilon_J,
]

[
|g(\theta)-\tilde g(\psi_\phi(\theta))|\le \epsilon_g,
]

[
|v_C(\theta)-\tilde v(\psi_\phi(\theta))|\le \epsilon_v,
]

则优化 regret 可以分解成：

[
\text{statistical error in }\psi\text{-space}
+
\text{representation error}.
]

这条定理把 “state-coupled structure” 正式变成低维 sufficient representation 假设。

---

## Theorem 3：LODO meta-prior generalization

假设 domains：

[
d\sim \mathcal P_{\mathrm{domain}}
]

i.i.d. 来自同一 meta-distribution。学习到的 encoder (\hat\phi) 满足经验风险最小化，则对 held-out domain：

[
\mathbb E_{d^\star}
[
\mathcal R_{d^\star}(\hat\phi)
]
\le
\inf_{\phi\in\Phi}
\mathbb E_d[\mathcal R_d(\phi)]
+
O\left(
\sqrt{\frac{\mathrm{complexity}(\Phi)}{D_{\mathrm{source}}}}
\right).
]

这里 (\mathcal R_d) 可以是 HVD NLL + chance calibration loss。

这证明 learned prior 不是 hand-tuned，而是从 source domains 泛化。

---

## Theorem 4：Chance certification with HVD uncertainty

若：

[
g(\theta)
\le
m_n^g(\theta)+\sqrt{\beta_n}s_n^g(\theta),
]

且：

[
v_C(\theta)
\le
v_{C,n}^+(\theta),
]

则：

[
m_n^g(\theta)
+
\sqrt{\beta_n}s_n^g(\theta)
+
z_{1-\alpha}\sqrt{v_{C,n}^+(\theta)}
\le b
]

推出：

[
\mathbb P(C(\theta)>b)
\le
\alpha+\delta_{\mathrm{approx}}.
]

这让 certified KG 有理论合法性。

---

## Theorem 5：Admissible SC-OLH-KG one-step optimality

定义 terminal value：

[
V_n^\star
=========

\min_{\theta\in\mathcal F_n}m_n^J(\theta).
]

Exact KG：

[
\mathrm{SC\text{-}OLHKG}_n(x)
=============================

## V_n^\star

\mathbb E[V_{n+1}^\star\mid \mathcal I_n,x].
]

则：

[
x_n^\star
=========

\arg\max_x \mathrm{SC\text{-}OLHKG}_n(x)
]

在 admissible one-step lookahead policies 中 Bayes optimal。

---

## Theorem 6：Finite-budget regret with prior coverage

若 candidate proposal 满足：

[
q_{\mathrm{cand}}(\mathcal G_\epsilon)\ge p_\epsilon,
]

且 (\psi)-space GP information gain 为 (\gamma_N(k_\psi))，HVD 参数维度为：

[
d_v=r+\frac{K(K+1)}{2}+K+1,
]

则 with high probability：

[
\boxed{
r_N
\le
C_1
\sqrt{
\frac{\gamma_N(k_\psi)+\log(1/\delta)}{N}
}
+
C_2
\sqrt{
\frac{d_v\log N}{n_{\mathrm{eff}}}
}
+
C_3\epsilon_\psi
+
C_4
\sqrt{
\frac{\log(1/\delta)}{M p_\epsilon}
}.
}
]

这条 theorem 是解释 (N=80) 的关键：
不是 (d=1000) 决定难度，而是：

[
d_\psi,\quad d_v,\quad p_\epsilon.
]

meta-prior 的作用是增大 (p_\epsilon)，HVD 的作用是降低 (d_v)，state coupling 的作用是降低 (\gamma_N(k_\psi))。

---

# 13. 主实验 protocol：三条线并行

最终论文最好明确区分三条线。

## Line 1：Strict universal prior

不使用任何 source training，不使用 domain-specific hand-coded anchors。

这是 negative control，不要求它成功。它回答：

[
\text{没有结构先验时，高维小预算确实很难。}
]

你的结果已经支持这一点。

---

## Line 2：LODO learned meta-prior

这是主方法。

[
\text{train on source domains}
\rightarrow
\text{freeze}
\rightarrow
\text{test on held-out domain}.
]

它回答：

[
\text{结构先验是否可以迁移，而不是手写？}
]

---

## Line 3：Domain / oracle upper bound

包含 problem-specific initial、state anchors、inverse anchors、refinement、axis oracle、calibration。

这不是主方法，只是 upper bound。它回答：

[
\text{如果有专家或 oracle 结构，性能上限在哪里？}
]

这三条线分开后，审稿人很难说你把 oracle 当算法。

---

# 14. 你当前四个结果应该怎么解释

你现在有：

[
\text{strict universal: mostly infeasible}
]

[
\text{domain prior: FactorShock succeeds, Inventory/Queue not}
]

这说明现有 domain prior 还不是稳定的 transferable prior，尤其 Inventory/Queue 的 risk exposure 或 boundary certification 没学好。

下一步实验不要继续在 strict universal 上硬调，而应该做：

[
\boxed{
\text{LODO meta-prior 是否能把 Inventory/Queue 的 feasible rate 从 0 拉起来？}
}
]

如果 LODO 仍然失败，说明：

1. source domains 不够；
2. encoder 没有学到跨域 risk regimes；
3. HVD 对 Inventory/Queue 的 shared-shock 结构不适配；
4. chance threshold 太紧，初始 exploration 覆盖不到 boundary；
5. (N=10) 太低，不足以做 certification。

这不是坏事。OR 论文可以诚实地把结果分成：

[
N=10:\ \text{prior stress test}
]

[
N=40:\ \text{early certification}
]

[
N=80:\ \text{main budget}
]

关键是证明 (N=80) 下 LODO meta-prior 稳定有效。

---

# 15. 具体下一步执行清单

## 第一步：写 admissibility audit layer

给每个函数加 flag：

```text
uses_true_objective
uses_true_constraint
uses_true_sigma
uses_true_optimum
uses_true_boundary
uses_hidden_axis
uses_problem_specific_formula
uses_target_eval_data
uses_source_data
```

每次 run 自动输出 audit report。

主实验只允许：

```text
uses_source_data = yes
uses_target_eval_data = yes
uses_problem_specific_formula = no
uses_true_* = no
uses_hidden_axis = no
```

---

## 第二步：把所有 problem-specific 模块移到 `oracle_variants/`

例如：

```text
oracle_variants/
    problem_initial_samples.py
    inverse_state_anchor.py
    oracle_axis.py
    true_sigma_calibration.py
    problem_refinement.py
```

主方法目录只保留：

```text
core/
    state_encoder.py
    hvd.py
    certified_kg.py
    meta_prior.py
    candidate_generator.py
```

---

## 第三步：实现统一 (\psi_\phi)

先不用 Transformer，第一版可以是 robust 的：

[
h_\phi(\mathcal T)
==================

[
\text{trajectory moments},
\text{quantiles},
\text{occupancy histogram},
\text{transition statistics},
\text{policy summary},
\text{random Fourier features}
].
]

然后：

[
A = \mathrm{PCA/whiten}(h),
]

[
N = \mathrm{softKMeans}(h).
]

这已经是统一 learned representation，不是 benchmark-specific hand-coding。

第二版再加 self-supervised trajectory encoder。轨迹表示学习文献里，自监督和对比学习已经被广泛用于把轨迹序列嵌入固定维表示；例如 TrajRCL 用 reconstruction + contrastive learning 学 trajectory representation。([科学直通车][5])

---

## 第四步：训练 LODO meta-prior

对每个 held-out domain：

```text
for heldout in [FactorShock, Inventory, Queue, Traffic]:
    train encoder/HVD/proposal on other domains
    freeze
    run target BO with N = 20, 40, 80
    report no-cheat metrics
```

---

## 第五步：加 mixture exploration

所有 init / candidates 都用：

[
q=\epsilon q_{\mathrm{space}}+(1-\epsilon)q_{\mathrm{meta}}.
]

推荐先设：

[
\epsilon=0.2
]

然后在 source validation 上选：

[
\epsilon\in{0.1,0.2,0.3,0.5}.
]

不能在 target 上选。

---

## 第六步：重跑主 ablations

最低限度必须有：

1. strict universal；
2. LODO meta-prior；
3. LODO no HVD；
4. LODO no state coupling；
5. LODO no (N^\top BN)；
6. LODO no (A^\top\Lambda A)；
7. LODO shuffled (\psi)；
8. domain-tuned upper bound；
9. oracle upper bound。

---

# 16. 论文最终叙事

最终论文不要说：

> We design problem-specific anchors and risk exposures for each benchmark.

要说：

> We introduce an admissible structural prior framework for stochastic simulation optimization. The prior is learned across source domains, frozen before testing on a held-out domain, and used only through a state-coupled heteroscedastic decomposition. Its role is not to reveal optima, but to concentrate posterior learning and candidate generation in a low-dimensional cumulative-risk coordinate system.

再把核心公式放出来：

[
\boxed{
\psi_\phi(\theta,\mathcal T)=(A_\phi,N_\phi)
}
]

[
\boxed{
\mathrm{Var}(C(\theta)\mid\mathcal T)
=====================================

A_\phi^\top\Lambda A_\phi
+
N_\phi^\top B N_\phi
+
N_\phi^\top\omega
+
\sigma_0^2.
}
]

[
\boxed{
\theta_{n+1}
============

\arg\max_{\theta\in\mathcal A_n}
\mathrm{SC\text{-}OLHKG}*n(\theta),
\quad
\mathcal A_n\sim
\epsilon q*{\mathrm{space}}+(1-\epsilon)q_\phi.
}
]

这三行就是新项目的主干。

---

# 17. 最终回答你的两个核心问题

## 1. 如何重新定义“可接受的先验”？

可接受的先验必须满足：

[
\boxed{
\text{available before target evaluation or learned only from source domains / observed target samples}
}
]

并且所有候选选择必须是：

[
\mathcal I_n^{d^\star}\text{-measurable}.
]

通用 (\psi/A,N) 必须由统一 encoder 从可观测 trajectory/policy records 产生；不能为每个 benchmark 手写。Anchor/refinement 只有在来自 space-filling、posterior acquisition、或 frozen meta-proposal 时才公平。Problem-specific refinement、inverse anchors、axis oracle、true sigma calibration 都应该移出主线，只作为 oracle upper-bound ablation。

## 2. 如何做成 leave-one-domain-out meta-prior？

训练一个统一的：

[
e_\phi,\quad \psi_\phi=(A_\phi,N_\phi),\quad P_\Theta(\eta),\quad q_\phi(\theta\mid context)
]

在 source domains 上学习 HVD、chance calibration 和 candidate proposal，然后冻结，在 held-out domain 上只用 target 已花预算样本做 Bayesian update。主结果用 LODO，而不是 domain-specific tuning。这样 prior 从“手写 benchmark knowledge”变成“learned transferable structural prior”。

---

# 18. 最凝练建议

你现在应该把项目从：

[
\text{SC/HVD framework + problem-specific anchors}
]

改成：

[
\boxed{
\textbf{LODO-learned admissible SC/HVD prior + certified KG}
}
]

然后把当前那些强 problem-specific 模块全部降级为 upper-bound ablation。

这样，(N=80) 的有效性就可以解释为：

[
\boxed{
\text{source-trained prior 把搜索从 }d\text{-维 policy space 压到 }(A,N)\text{ 的低维 cumulative-risk space，HVD 让 chance certification 只需估计低维风险参数，KG 再把预算集中在 objective elite、feasibility boundary 和 variance-informative 区域。}
}
]

这条线既保住了 SC 和 HVD 的强耦合，又避免了“每个 benchmark 手写答案”的风险，也更像一篇能投 OR 的理论型论文。

[1]: https://www.cs.ubc.ca/~hutter/earg/papers07/00585893.pdf?utm_source=chatgpt.com "No Free Lunch Theorems For Optimization"
[2]: https://arxiv.org/abs/1904.02642?utm_source=chatgpt.com "Meta-Learning Acquisition Functions for Transfer Learning in Bayesian Optimization"
[3]: https://openreview.net/pdf?id=4SZ9Ft--pDl&utm_source=chatgpt.com "PRIOR-GUIDED BAYESIAN OPTIMIZATION"
[4]: https://www.ibm.com/think/topics/data-leakage-machine-learning?utm_source=chatgpt.com "What is Data Leakage in Machine Learning?"
[5]: https://www.sciencedirect.com/science/article/abs/pii/S0167739X23002376?utm_source=chatgpt.com "Self-supervised contrastive representation learning for ..."

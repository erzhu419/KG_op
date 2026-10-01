下面这个方案我建议作为**单目标主论文版本**。核心不是“先做一个异方差模型，然后套普通 KG”，而是：

[
\boxed{
\textbf{异方差分解}
+
\textbf{状态—策略耦合 belief model}
+
\textbf{decomposition-aware KG 搜索}
}
]

我建议命名为：

[
\boxed{
\textbf{SC-OLH-KG: State-Coupled Orthogonal Latent Heteroscedastic Knowledge Gradient}}
]

中文可以叫：

[
\boxed{
\textbf{状态耦合正交潜在异方差知识梯度}}
]

论文标题可以更 OR 一点：

> **A State-Coupled Orthogonal Heteroscedastic Knowledge-Gradient Method for Chance-Constrained Simulation Optimization**

---

# 1. 单目标主问题：不要先做双目标

主问题设为：

[
\min_{\theta\in\Theta} J(\theta)
]

subject to：

[
\mathbb P\left(C(\theta,\xi)\le b\right)\ge 1-\alpha.
]

其中 (\theta) 是一个 policy / strategy / operating plan / simulation design。比如在交通信号优化里，(\theta) 是信号配时方案；在库存系统里，(\theta) 是订货策略参数；在排队系统里，(\theta) 是服务规则参数。

一次仿真运行返回：

[
\left(Y_m(\theta), C_m(\theta), \mathcal T_m(\theta)\right),
]

其中：

[
Y_m(\theta)=\text{第 }m\text{ 次仿真的目标输出},
]

[
C_m(\theta)=\text{第 }m\text{ 次仿真的约束输出},
]

[
\mathcal T_m(\theta)
====================

(S_0,A_0,S_1,A_1,\ldots,S_T,A_T)
]

是仿真轨迹。

目标是：

[
J(\theta)=\mathbb E[Y(\theta)].
]

概率约束是：

[
\mathbb P(C(\theta)\le b)\ge 1-\alpha.
]

如果用 Gaussian / sub-Gaussian approximation，则等价或近似为：

[
g(\theta)+z_{1-\alpha}\sqrt{v(\theta)}\le b,
]

其中：

[
g(\theta)=\mathbb E[C(\theta)],
]

[
v(\theta)=\mathrm{Var}(C(\theta)).
]

这正好把问题的核心推到：

[
\boxed{
\textbf{如何学习 }J(\theta),\ g(\theta),\ v(\theta)，\textbf{并用它们决定下一次采样。}
}
]

现有 repo 里已经有 “GPR-KG + probabilistic constraint + unknown variance” 的基础：当前 manuscript 的摘要明确写的是 bi-objective simulation optimization with probabilistic constraints and unknown variances，并且已有 GPR-KG、VEPM、KG sampling policy、finite-budget regret、VEPM consistency、conditional feasibility bound 等组件。([GitHub][1]) 代码中现有 `ParametricGPR` 使用 (2d+1) 维二次 basis，`VEPM` 用 partition-based variance sharing 估计异方差，这说明 repo 已经有可改造的底座，而不是从零开始。([GitHub][2])

---

# 2. 核心建模：策略不是孤立点，而是诱导状态分布

传统 KG/BO 通常把一个 design (\theta) 当成一个点：

[
\theta \mapsto Y(\theta).
]

但你要的“状态和策略之间的耦合关系”应该写成：

[
\theta
\longrightarrow
\mathcal T(\theta)
\longrightarrow
\rho(\theta)
\longrightarrow
\left(J(\theta), g(\theta), v(\theta)\right).
]

其中 (\rho(\theta)) 是 policy-induced state-action occupancy feature：

[
\rho(\theta)
============

\mathbb E_\theta
\left[
\sum_{t=0}^{T}\varphi(S_t,A_t)
\right]
\in\mathbb R^{d_\rho}.
]

这里 (\varphi(S_t,A_t)) 可以是人工特征，也可以是 self-supervised encoder 产生的 latent feature。

这一步非常关键。它把策略之间的相似性从：

[
|\theta-\theta'|
]

升级成：

[
|\rho(\theta)-\rho(\theta')|.
]

两个策略参数可能差很多，但如果诱导出来的状态—动作 occupancy 很像，它们的性能和风险应该相关；反过来，两个策略参数很近，但如果诱导出了完全不同的拥堵 regime / 风险 regime，它们不应该被强行看成相似。

因此定义 state-coupled kernel：

[
k_J(\theta,\theta')
===================

k_{\theta}(\theta,\theta')
+
\kappa_\rho k_{\rho}(\rho(\theta),\rho(\theta')).
]

约束均值也类似：

[
k_g(\theta,\theta')
===================

k_{\theta}^{g}(\theta,\theta')
+
\kappa_g k_{\rho}^{g}(\rho(\theta),\rho(\theta')).
]

这和 BO for policy search 里的 trajectory kernel 思想是同一条线：Wilson、Fern、Tadepalli 的 JMLR 论文明确提出用 RL 轨迹信息构造 policy similarity kernel，以改善 BO 对 expected return 的后验估计和探索质量。([机器学习研究杂志][3]) Contextual BO / safe contextual BO 也在利用 context/state 来建立不同任务或环境之间的耦合关系；例如 offline contextual BO 把不同 plasma state 看成不同 task，safe contextual BO 用 contextual variables 做安全控制器调参。([NeurIPS Papers][4])

---

# 3. 均值 belief model：保留 GPR，但换成状态耦合 basis/kernel

对于目标均值：

[
J(\theta)\sim GP(m^J_0,k_J).
]

对于约束均值：

[
g(\theta)\sim GP(m^g_0,k_g).
]

观测模型：

[
Y_n = J(\theta_n)+\varepsilon^J_n,
]

[
C_n = g(\theta_n)+\varepsilon^C_n.
]

其中：

[
\mathrm{Var}(\varepsilon^J_n\mid \theta_n)=v_J(\theta_n),
]

[
\mathrm{Var}(\varepsilon^C_n\mid \theta_n)=v_C(\theta_n).
]

在最小可行版本里，可以先只对 chance constraint 的 (v_C(\theta)) 做完整异方差分解，目标 (Y) 的噪声先用同方差或 class variance。这样主线更集中。

如果继续沿用当前 repo 的 parametric GPR 写法，可以把原 basis：

[
\phi_{\mathrm{old}}(\theta)
===========================

(1,\theta_1,\ldots,\theta_d,\theta_1^2,\ldots,\theta_d^2)
]

改成：

[
\phi_{\mathrm{SC}}(\theta)
==========================

\left[
1,\theta,\theta^2,\rho(\theta),\rho(\theta)^2,\theta\otimes \rho(\theta)
\right].
]

然后再做正交化：

[
\tilde \phi_{\mathrm{SC}}(\theta)
=================================

Q^\top \phi_{\mathrm{SC}}(\theta),
]

其中 (Q) 来自 QR / PCA / whitening。
这样做的意义是：**均值模型也建立了策略—状态耦合，而不是只在原始决策空间里拟合二次曲面。**

现有代码里 `ParametricGPR` 的 basis 是 (2d+1) 维二次特征，且没有交叉项；这很适合作为第一版可替换接口。([GitHub][2])

---

# 4. 异方差分解：论文最核心的数学对象

你要的核心不是普通异方差：

[
v_C(\theta)\ \text{随 }\theta\text{ 变化}.
]

普通 heteroscedastic GP 已经有人做很多了。stochastic kriging 把 deterministic kriging 推广到 stochastic simulation setting；hetGP 系列则专门处理 input-dependent noise，并强调 replication 对同时学习 mean field 和 variance field 很重要。([PubsOnLine][5]) RAHBO 也已经把 BO 推广到同时学习 mean 和 input-dependent variance，并做 risk-aware acquisition。([arXiv][6])

你需要的更强结构是：

[
\boxed{
\textbf{累计约束输出的方差可以被低维风险 regime 和正交风险因子分解。}
}
]

---

## 4.1 固定轨迹下的最一般表达

令：

[
X_t=(S_t,A_t).
]

累计约束输出：

[
C(\theta,\xi)
=============

\sum_{t=0}^{T} q(X_t).
]

把 instantaneous contribution 写成：

[
q(X_t)=\bar q(X_t)+\delta_t.
]

那么给定一条轨迹 (\mathcal T=(X_0,\ldots,X_T))，累计噪声为：

[
\Delta(\mathcal T)
==================

\sum_{t=0}^{T}\delta_t.
]

因此：

[
\mathrm{Var}(C\mid \mathcal T)
==============================

\mathrm{Var}
\left(
\sum_{t=0}^{T}\delta_t
\mid \mathcal T
\right)
=======

\mathbf 1^\top \Omega(\mathcal T)\mathbf 1,
]

其中：

[
\Omega_{tu}(\mathcal T)
=======================

\mathrm{Cov}(\delta_t,\delta_u\mid \mathcal T).
]

这就是异方差下 (\sum S) 方差的根式。

如果只假设独立异方差：

[
\Omega_{tu}=0,\quad t\ne u,
]

[
\Omega_{tt}=\sigma^2(X_t),
]

则：

[
\mathrm{Var}(C\mid \mathcal T)
==============================

\sum_{t=0}^{T}\sigma^2(X_t).
]

但这太悲观、太高维，因为它要求每个状态都有自己的方差。

---

## 4.2 分类型异方差：商品标准化直觉的数学形式

设状态—动作 pair 属于 (K) 个风险类别之一：

[
c(X_t)\in{1,\ldots,K}.
]

并假设：

[
\sigma^2(X_t)=\omega^2_{c(X_t)}.
]

定义路径中第 (k) 类出现次数：

[
N_k(\mathcal T)
===============

\sum_{t=0}^{T}
\mathbb I{c(X_t)=k}.
]

则：

[
\boxed{
\mathrm{Var}(C\mid \mathcal T)
==============================

\sum_{k=1}^{K}N_k(\mathcal T)\omega_k^2.
}
]

这就是你说的“瓶装饮料 vs 鲜肉”：

[
\omega^2_{\mathrm{standardized}}\approx 0,
]

[
\omega^2_{\mathrm{fresh}}\gg 0.
]

于是风险不是按状态数 (T) 平均增长，而是按：

[
N_{\mathrm{fresh}}(\mathcal T)
]

这类高风险状态的出现次数增长。

这已经是一个很强的降维：

[
\text{完全异方差维度}=|\mathcal S\times\mathcal A|,
]

[
\text{分类型异方差维度}=K.
]

---

## 4.3 共享冲击：风险可以按 (N_k^2) 增长

只做 class variance 还不够。很多仿真环境存在共享冲击，例如交通 demand shock、供应链 disruption、同一批次质量问题。

设：

[
\delta_t=b_{c(X_t)}+\eta_t,
]

其中：

[
b=(b_1,\ldots,b_K)\sim N(0,B),
]

[
\eta_t\mid c(X_t)=k\sim N(0,\omega_k^2),
]

且 (\eta_t) 条件独立。

则：

[
\sum_{t=0}^{T}\delta_t
======================

\sum_{k=1}^{K}N_k(\mathcal T)b_k
+
\sum_{t=0}^{T}\eta_t.
]

所以：

[
\boxed{
\mathrm{Var}(C\mid \mathcal T)
==============================

N(\mathcal T)^\top B N(\mathcal T)
+
\sum_{k=1}^{K}N_k(\mathcal T)\omega_k^2.
}
]

其中：

[
N(\mathcal T)=
(N_1(\mathcal T),\ldots,N_K(\mathcal T))^\top.
]

第一项是系统性风险，可能按 (N_k^2) 放大；第二项是个体级噪声，按 (N_k) 累加。

这在 OR 里很有价值，因为它解释了为什么同样是“经过很多状态”，有些路径风险只是线性增加，有些路径风险会平方级增加。

---

## 4.4 正交潜在风险因子：真正的 10/10 数学核心

更一般地，设噪声过程有低秩正交展开：

[
\delta_t
========

\sum_{r=1}^{R}
\sqrt{\lambda_r}
\psi_r(X_t)Z_r
+
b_{c(X_t)}
+
\eta_t,
]

其中：

[
Z_r\sim N(0,1),
]

[
Z_r\perp Z_\ell,\quad r\ne \ell,
]

[
\mathbb E_{\nu}[\psi_r(X)\psi_\ell(X)]=\delta_{r\ell}.
]

这里 (\nu) 是 reference occupancy measure，可以由初始样本轨迹估计。

定义：

[
A_r(\mathcal T)=
\sum_{t=0}^{T}\psi_r(X_t),
]

[
A(\mathcal T)=
(A_1(\mathcal T),\ldots,A_R(\mathcal T))^\top.
]

那么：

[
\sum_{t=0}^{T}\delta_t
======================

A(\mathcal T)^\top \Lambda^{1/2}Z
+
N(\mathcal T)^\top b
+
\sum_{t=0}^{T}\eta_t,
]

其中：

[
\Lambda=\mathrm{diag}(\lambda_1,\ldots,\lambda_R).
]

于是得到主定理级公式：

[
\boxed{
\mathrm{Var}(C\mid \mathcal T)
==============================

A(\mathcal T)^\top \Lambda A(\mathcal T)
+
N(\mathcal T)^\top B N(\mathcal T)
+
N(\mathcal T)^\top \omega.
}
]

其中：

[
\omega=(\omega_1^2,\ldots,\omega_K^2)^\top.
]

这一个公式同时包含：

同方差：

[
R=0,\quad K=1,\quad B=0.
]

分类型异方差：

[
R=0,\quad K>1,\quad B=0.
]

共享冲击：

[
R=0,\quad K>1,\quad B\ne 0.
]

正交低秩风险：

[
R>0.
]

所以论文不需要在“分解、流形、分类方差”之间二选一。可以提出一个统一模型：

[
\boxed{
\textbf{class regimes + orthogonal latent factors}
}
]

分类方差负责粗粒度风险分层；正交因子负责低维相关噪声；流形/representation learning 负责学习 (c(X)) 和 (\psi(X))。

---

# 5. 随机轨迹下的完整 (\mathrm{Var}(C(\theta)))

上面是固定轨迹 (\mathcal T) 的方差。真实策略 (\theta) 会诱导随机轨迹：

[
\mathcal T\sim P_\theta.
]

所以用 total variance：

[
\mathrm{Var}(C(\theta))
=======================

\mathbb E_{\mathcal T\sim P_\theta}
[
\mathrm{Var}(C\mid \mathcal T)
]
+
\mathrm{Var}*{\mathcal T\sim P*\theta}
(
\mathbb E[C\mid \mathcal T]
).
]

代入正交异方差分解：

[
\boxed{
v_C(\theta)
===========

\mathbb E_\theta
[
A^\top\Lambda A
+
N^\top B N
+
N^\top\omega
]
+
\mathrm{Var}*\theta
\left(
\sum*{t=0}^{T}\bar q(X_t)
\right).
}
]

进一步展开：

[
\mathbb E[A^\top\Lambda A]
==========================

\mathbb E[A]^\top\Lambda \mathbb E[A]
+
\mathrm{tr}
\left(
\Lambda\mathrm{Var}(A)
\right),
]

[
\mathbb E[N^\top B N]
=====================

\mathbb E[N]^\top B\mathbb E[N]
+
\mathrm{tr}
\left(
B\mathrm{Var}(N)
\right).
]

所以策略风险由四类东西决定：

[
\mathbb E[A(\mathcal T)\mid \theta],
\quad
\mathrm{Var}(A(\mathcal T)\mid \theta),
\quad
\mathbb E[N(\mathcal T)\mid \theta],
\quad
\mathrm{Var}(N(\mathcal T)\mid \theta).
]

这就把“状态—策略耦合”也放进了方差里：

[
\boxed{
\theta
\longrightarrow
P_\theta(\mathcal T)
\longrightarrow
(A,N)
\longrightarrow
v_C(\theta).
}
]

这比普通 BO/KG 强很多，因为普通 BO 只学：

[
\theta\mapsto C(\theta),
]

而这里学的是：

[
\theta\mapsto \text{risk occupancy}.
]

---

# 6. “aleatoric 转 epistemic”的严谨表述

你说“增加特征可以把部分 aleatoric 转化为 epistemic”，这个观点可以成为论文里的一个理论亮点，但要写得严谨。

更准确地说：

[
\boxed{
\textbf{aleatoric / epistemic 是相对于当前信息集的分解，不是绝对固定标签。}
}
]

设当前可用信息是 (\mathcal F)，加入新特征、轨迹、状态表示后的信息是 (\mathcal G)，且：

[
\mathcal F\subseteq \mathcal G.
]

则有 identity：

[
\boxed{
\mathbb E[\mathrm{Var}(Y\mid\mathcal F)]
----------------------------------------

# \mathbb E[\mathrm{Var}(Y\mid\mathcal G)]

\mathbb E\left[
\mathrm{Var}
\left(
\mathbb E[Y\mid\mathcal G]\mid\mathcal F
\right)
\right]
\ge 0.
}
]

这条式子说明：增加信息后，平均条件残差方差不会增加。减少掉的那部分，正是新特征解释出来的结构。

所以你可以在论文里说：

> What appears aleatoric under a coarse representation may become epistemically learnable under a refined state-policy representation.

Kendall 和 Gal 区分了 aleatoric uncertainty 与 epistemic uncertainty，Depeweg 等进一步把这两者的分解用于 active learning 和 risk-sensitive RL；近年的 multi-modal uncertainty acquisition 工作也明确挑战“aleatoric 永远不可降低”的粗糙说法，指出增加模态/信息可以降低 apparent aleatoric uncertainty。([arXiv][7])

这条 identity 可以成为你论文里的 **Proposition 1**。它把你“通过增加特征、组合旧特征构造新特征来提升准确率”的经验，变成一个数学结论。

---

# 7. Chance constraint：必须区分真实风险和模型不确定性

真实 chance constraint 是：

[
g(\theta)+z_{1-\alpha}\sqrt{v_C(\theta)}\le b.
]

但是算法不知道 (g(\theta)) 和 (v_C(\theta))。第 (n) 步有后验：

[
g(\theta)\mid D_n
\sim N(m^g_n(\theta),s^g_n(\theta)^2).
]

异方差分解参数记为：

[
\eta=
(\lambda_1,\ldots,\lambda_R,\mathrm{vec}(B),\omega_1,\ldots,\omega_K).
]

估计后验：

[
\eta\mid D_n
\approx N(\hat\eta_n,\Sigma^\eta_n).
]

令路径风险特征：

[
\zeta(\theta)
=============

\left[
\mathbb E[A\otimes A\mid\theta],
\mathbb E[N\otimes N\mid\theta],
\mathbb E[N\mid\theta]
\right].
]

则：

[
\hat v_{C,n}(\theta)
====================

\zeta(\theta)^\top \hat\eta_n
+
\widehat{\mathrm{Var}}_\theta
\left(
\sum_t \bar q(X_t)
\right).
]

为了保守认证，定义上置信方差：

[
v^+_{C,n}(\theta)
=================

\hat v_{C,n}(\theta)
+
\beta^v_n
\sqrt{
\zeta(\theta)^\top\Sigma^\eta_n\zeta(\theta)
}
+
\epsilon_R(\theta)
+
\epsilon_c(\theta)
+
\epsilon_\rho(\theta).
]

其中：

[
\epsilon_R=\text{正交风险因子截断误差},
]

[
\epsilon_c=\text{类内异质性误差},
]

[
\epsilon_\rho=\text{状态—策略 representation 误差}.
]

然后定义 posterior-certified feasible set：

[
\boxed{
\mathcal F_n
============

\left{
\theta:
m^g_n(\theta)
+
\sqrt{\beta^g_n}s^g_n(\theta)
+
z_{1-\alpha}\sqrt{v^+_{C,n}(\theta)}
\le b
\right}.
}
]

这一步非常重要。它把三个不确定性分开：

[
\text{constraint mean 的 epistemic uncertainty}
]

[
\text{environmental aleatoric risk}
]

[
\text{variance decomposition model 的 epistemic uncertainty}.
]

普通 heteroscedastic BO 往往只是学习 input-dependent variance；你这里把 variance 的估计误差也纳入 chance-feasibility certification。RAHBO 已经说明 high-stakes BO 不能只优化期望，必须同时学习并控制 input-dependent variance；你的版本进一步处理的是 chance constraint 和累计路径方差的结构化分解。([arXiv][6])

---

# 8. 搜索端：Decomposition-aware KG

这里是你刚刚特别强调的“不要只剩 decomposition”。
搜索端必须写成：

[
\boxed{
\textbf{KG 的价值来自目标均值改善、可行性边界识别、风险分解学习、状态耦合传播。}
}
]

---

## 8.1 Exact SC-OLH-KG 定义

定义第 (n) 步的 posterior terminal value：

[
V_n^\star
=========

\min_{\theta\in \mathcal F_n}
m^J_n(\theta).
]

如果 (\mathcal F_n=\emptyset)，用 penalty formulation：

[
V_n^\star
=========

\min_{\theta\in\Theta}
\left[
m^J_n(\theta)
+
M\left(
m^g_n(\theta)
+
\sqrt{\beta^g_n}s^g_n(\theta)
+
z_{1-\alpha}\sqrt{v^+_{C,n}(\theta)}
------------------------------------

b
\right)_+
\right].
]

Exact KG：

[
\boxed{
\mathrm{SC\text{-}OLHKG}_n(x)
=============================

## V_n^\star

\mathbb E_n
\left[
V_{n+1}^\star
\mid
\text{sample }x
\right].
}
]

因为是 minimization，所以采样 (x) 的价值是：它预计能让 posterior-certified best feasible value 降低多少。

KG 的基本思想来自 sequential value of information：correlated KG 在相关正态信念下使用一次采样对多个 alternatives 的更新来选点，continuous KG 又把它推广到 GP regression 和连续决策变量。([PubsOnLine][8]) 你这里的创新是把 terminal value 从“最优均值”换成“decomposed chance-certified feasible optimum”。

---

## 8.2 实现上用四项近似

Exact KG 可能计算重，所以实现时可以写成：

[
\boxed{
\mathrm{SC\text{-}OLHKG}_n(x)
=============================

\mathrm{KG}^{J}_n(x)
+
\lambda_F \mathrm{KG}^{F}_n(x)
+
\lambda_V \mathrm{KG}^{V}*n(x)
+
\lambda*\rho \mathrm{KG}^{\rho}_n(x).
}
]

### 第一项：目标均值 KG

[
\mathrm{KG}^{J}_n(x)
]

衡量采样 (x) 对最优 feasible objective 的改善。

这是传统 KG 部分。

---

### 第二项：Feasibility-boundary KG

定义 posterior safety margin：

[
M_n(\theta)
===========

## b

## m^g_n(\theta)

z_{1-\alpha}\sqrt{v^+_{C,n}(\theta)}.
]

如果：

[
M_n(\theta)\approx 0,
]

说明 (\theta) 在 chance constraint 边界附近。
这些点最重要，因为一次采样可能改变它是否可行。

定义 boundary set：

[
\mathcal B_n
============

\left{
\theta:
|M_n(\theta)|\le \Delta_n
\right}.
]

Feasibility KG 可以近似为：

[
\mathrm{KG}^{F}_n(x)
====================

\sum_{\theta\in\mathcal B_n}
w_\theta
\Delta \mathrm{Var}_n
\left(
M_n(\theta)
\mid x
\right).
]

也就是采样 (x) 能减少多少边界点的 margin uncertainty。

---

### 第三项：Variance-decomposition KG

这是新方法最核心的采样项。

因为：

[
M_n(\theta)
===========

## b

## m^g_n(\theta)

z_{1-\alpha}\sqrt{v^+_{C,n}(\theta)},
]

所以对 variance 参数 (\eta) 的梯度为：

[
\nabla_\eta M_n(\theta)
=======================

*

\frac{z_{1-\alpha}}{2\sqrt{v^+_{C,n}(\theta)}}
\zeta(\theta).
]

采样 (x) 会更新 (\Sigma^\eta_n)。若 residual-square observation 的特征为 (\zeta(x))，线性-Gaussian 近似下：

[
(\Sigma^\eta_{n+1})^{-1}
========================

(\Sigma^\eta_n)^{-1}
+
\frac{1}{\sigma_e^2}
\zeta(x)\zeta(x)^\top.
]

于是 variance VOI 可以写成：

[
\boxed{
\mathrm{KG}^{V}_n(x)
====================

\sum_{\theta\in\mathcal B_n\cup\mathcal E_n}
w_\theta
\nabla_\eta M_n(\theta)^\top
\left(
\Sigma^\eta_n
-------------

\mathbb E_n[\Sigma^\eta_{n+1}(x)]
\right)
\nabla_\eta M_n(\theta).
}
]

其中 (\mathcal E_n) 是 objective elite set。

直觉非常清楚：
采样 (x) 有价值，不是因为它本身均值一定好，而是因为它能帮助识别那些会影响当前候选最优解可行性的风险因子。

这正是“瓶装饮料不需要注意力，鲜肉需要注意力”的数学形式。

---

### 第四项：State-coupling propagation KG

采样 (x) 对其他 (\theta) 的影响大小由 posterior covariance 决定：

[
\Delta s^2_n(\theta\mid x)
==========================

\frac{
k_n(\theta,x)^2
}{
k_n(x,x)+\hat v_n(x)
}.
]

因为：

[
k_n(\theta,x)
]

包含 state-policy kernel，所以如果 (x) 与很多 boundary / elite policies 在 occupancy space 上相似，采样 (x) 就能更新一大片相关策略。

定义：

[
\mathrm{KG}^{\rho}_n(x)
=======================

\sum_{\theta\in\mathcal B_n\cup\mathcal E_n}
w_\theta
\frac{
k_n(\theta,x)^2
}{
k_n(x,x)+\hat v_n(x)
}.
]

这就是“搜索端的耦合关系”。不是每个策略独立搜索，而是一次采样通过状态—策略相似性传播信息。

---

# 9. SC-OLH-KG 算法

算法可以写成：

[
\boxed{
\textbf{Algorithm: State-Coupled Orthogonal Latent Heteroscedastic KG}
}
]

**Input**：budget (N)，initial design (n_0)，risk level (\alpha)，threshold (b)，candidate generator，encoder class。

**Step 1. Initial simulation**

采样：

[
\theta_1,\ldots,\theta_{n_0}.
]

对每个 (\theta_i) 运行仿真，记录：

[
(Y_i,C_i,\mathcal T_i).
]

---

**Step 2. Learn / construct state-policy encoder**

从轨迹中构造：

[
\rho(\theta_i)
==============

\sum_t \varphi(S_t,A_t).
]

第一版可以直接用人工统计特征：

[
\varphi(S_t,A_t)
================

[
\text{queue length},
\text{delay},
\text{speed},
\text{phase},
\text{density},
\text{emission regime}
].
]

第二版再用 self-supervised encoder：

[
\varphi_\eta(S_t,A_t,H_t)
]

通过 masked prediction / contrastive learning / next-state prediction 学出来。PFNs4BO、MetaBO 一类方法说明 BO 里使用 learned surrogate / learned acquisition / in-context structure 已经是活跃方向，但主论文最好把 self-supervised encoder 作为实现选择，而不是主理论依赖。([Proceedings of Machine Learning Research][9])

---

**Step 3. Learn risk regimes**

聚类或分类：

[
c(X_t)\in{1,\ldots,K}.
]

可以用：

[
c(X_t)=\mathrm{kmeans}(\varphi(X_t)),
]

或者：

[
c(X_t)=\arg\max_k \pi_k(\varphi(X_t)).
]

选择 (K) 用 BIC、cross-validation、stability selection，或者固定小 (K)。

---

**Step 4. Orthogonalize risk basis**

构造候选风险基：

[
\tilde\psi_1,\ldots,\tilde\psi_R.
]

例如：

[
\tilde\psi(X)=
[
\varphi(X),
\varphi(X)^2,
\text{regime indicators},
\text{graph bottleneck features}
].
]

在 empirical occupancy measure (\hat\nu) 下计算 Gram matrix：

[
G_{r\ell}
=========

\mathbb E_{\hat\nu}
[
\tilde\psi_r(X)\tilde\psi_\ell(X)
].
]

做 whitening：

[
\psi(X)=G^{-1/2}\tilde\psi(X).
]

得到：

[
\mathbb E_{\hat\nu}[\psi_r(X)\psi_\ell(X)]
==========================================

\delta_{r\ell}.
]

---

**Step 5. Fit mean GPs**

拟合：

[
J(\theta)\sim GP(m^J,k_J),
]

[
g(\theta)\sim GP(m^g,k_g).
]

kernel 用：

[
k(\theta,\theta')
=================

k_\theta(\theta,\theta')
+
\kappa_\rho k_\rho(\rho(\theta),\rho(\theta'))
+
\kappa_A k_A(\bar A(\theta),\bar A(\theta')).
]

---

**Step 6. Fit heteroscedastic decomposition**

对每次仿真的 residual：

[
e_i
===

C_i-m^g_n(\theta_i).
]

计算路径风险特征：

[
A_i=A(\mathcal T_i),
]

[
N_i=N(\mathcal T_i).
]

构造：

[
\zeta_i=
[
A_i\otimes A_i,,
N_i\otimes N_i,,
N_i
].
]

拟合：

[
e_i^2
=====

\zeta_i^\top \eta
+
\text{noise}.
]

约束：

[
\eta\ge 0,
]

其中 (\eta) 对应：

[
(\lambda,\mathrm{vec}(B),\omega).
]

可以用 nonnegative ridge / Bayesian ridge / horseshoe shrinkage / spike-and-slab。高维 BO 里 SAASBO、additive BO 都体现了一个共同思想：高维问题必须引入稀疏、低维或加性结构，否则 GP/BO 很难 sample-efficient；你这里的低维结构不是 objective subspace，而是 variance-risk subspace。([Proceedings of Machine Learning Research][10])

---

**Step 7. Construct certified feasible set**

计算：

[
v^+_{C,n}(\theta)
]

和：

[
\mathcal F_n
============

\left{
\theta:
m^g_n(\theta)
+
\sqrt{\beta^g_n}s^g_n(\theta)
+
z_{1-\alpha}\sqrt{v^+_{C,n}(\theta)}
\le b
\right}.
]

---

**Step 8. Candidate generation**

candidate pool 应该由四类点组成：

1. posterior objective elite points；
2. chance-boundary points；
3. variance-informative points；
4. state-policy novelty points。

也就是：

[
\mathcal A_n
============

\mathcal A^{elite}_n
\cup
\mathcal A^{boundary}_n
\cup
\mathcal A^{variance}_n
\cup
\mathcal A^{novelty}_n.
]

这比普通 candidate generation 强，因为它保证搜索不会只盯着均值，也不会平均探索所有状态。

---

**Step 9. Select next simulation**

[
\theta_{n+1}
============

\arg\max_{\theta\in\mathcal A_n}
\mathrm{SC\text{-}OLHKG}_n(\theta).
]

运行仿真，更新：

[
D_{n+1}=D_n\cup{Y_{n+1},C_{n+1},\mathcal T_{n+1}}.
]

---

# 10. 理论包：这是 OR 论文的 10/10 深度

我建议主文放 6 到 8 个核心结果。不是每个定理都要特别长，但它们要形成一条链：

[
\text{方差分解}
\rightarrow
\text{维度下降}
\rightarrow
\text{可行性认证}
\rightarrow
\text{KG 搜索价值}
\rightarrow
\text{有限预算保证}.
]

---

## Theorem 1：固定轨迹的正交异方差分解

在模型：

[
\delta_t
========

\sum_{r=1}^{R}
\sqrt{\lambda_r}
\psi_r(X_t)Z_r
+
b_{c(X_t)}
+
\eta_t
]

下，有：

[
\boxed{
\mathrm{Var}(C\mid\mathcal T)
=============================

A(\mathcal T)^\top \Lambda A(\mathcal T)
+
N(\mathcal T)^\top B N(\mathcal T)
+
N(\mathcal T)^\top\omega.
}
]

这条是主定理。

---

## Theorem 2：随机策略下的 occupancy-risk decomposition

对策略 (\theta)：

[
v_C(\theta)
===========

\mathbb E_\theta[A^\top\Lambda A]
+
\mathbb E_\theta[N^\top B N]
+
\mathbb E_\theta[N^\top\omega]
+
\mathrm{Var}_\theta
\left(
\sum_t\bar q(X_t)
\right).
]

进一步：

[
\mathbb E[A^\top\Lambda A]
==========================

\bar A(\theta)^\top\Lambda\bar A(\theta)
+
\mathrm{tr}(\Lambda\Sigma_A(\theta)).
]

这说明策略风险只需要：

[
\bar A,\Sigma_A,\bar N,\Sigma_N
]

这些低维 occupancy-risk moments，而不是整条高维轨迹。

---

## Theorem 3：信息细化降低 apparent aleatoric uncertainty

若：

[
\mathcal F\subseteq\mathcal G,
]

则：

[
\mathbb E[\mathrm{Var}(Y\mid\mathcal F)]
----------------------------------------

# \mathbb E[\mathrm{Var}(Y\mid\mathcal G)]

\mathbb E
[
\mathrm{Var}
(
\mathbb E[Y\mid\mathcal G]
\mid\mathcal F
)
]
\ge 0.
]

这条定理给你的 self-supervised / feature construction 提供理论正当性：更丰富的状态—策略表示可以把部分 apparent aleatoric uncertainty 变成可学习结构。

---

## Theorem 4：截断误差与有效风险维度

若真实噪声协方差有 Mercer / KL 展开：

[
\Omega(X,X')
============

\sum_{r=1}^{\infty}
\lambda_r\psi_r(X)\psi_r(X'),
]

用前 (R) 个因子近似，则固定路径下：

[
\left|
V_{\mathrm{true}}(\mathcal T)
-----------------------------

V_R(\mathcal T)
\right|
\le
\left(\sum_{t=0}^{T}1\right)^2
\sum_{r>R}\lambda_r
===================

(T+1)^2
\sum_{r>R}\lambda_r,
]

在 (|\psi_r(X)|\le 1) 下成立。

如果 eigenvalue decay 快，比如：

[
\lambda_r\le Cr^{-p},\quad p>1,
]

则：

[
\sum_{r>R}\lambda_r
===================

O(R^{1-p}).
]

这说明只要环境风险协方差低秩或谱衰减快，robust/chance 风险维度就是 (R+K)，而不是 (T) 或 (|\mathcal S|)。

---

## Theorem 5：异方差分解参数的 oracle inequality

令：

[
d_v=R+\frac{K(K+1)}{2}+K.
]

假设 residual-square regression 满足 restricted eigenvalue condition，且 residual noise sub-exponential，则 nonnegative ridge / sparse Bayesian estimator 满足：

[
\boxed{
|\hat\eta_n-\eta^\star|*{\Gamma}^2
\le
C
\frac{d_v\log N}{n*{\mathrm{eff}}}
+
C(\epsilon_R^2+\epsilon_c^2+\epsilon_\rho^2)
}
]

with high probability。

其中：

[
\epsilon_R=\text{低秩截断误差},
]

[
\epsilon_c=\text{class 内异质性误差},
]

[
\epsilon_\rho=\text{state-policy representation 误差}.
]

这个 theorem 是对当前 VEPM cell-average consistency 的升级。当前 VEPM 的问题是 partition 对齐时有效，不对齐时会退化；新理论直接把这种误差写成 (\epsilon_c)，并且再加入低秩正交结构。当前代码也已经暴露出 partition 特征、medoid_K、ridge log-variance surrogate 等组件，所以升级路径自然。([GitHub][2])

---

## Theorem 6：chance-feasibility certification bound

若对所有 (\theta) 有：

[
|m^g_n(\theta)-g(\theta)|
\le
\sqrt{\beta^g_n}s^g_n(\theta),
]

且：

[
v_C(\theta)
\le
v^+_{C,n}(\theta),
]

则：

[
m^g_n(\theta)
+
\sqrt{\beta^g_n}s^g_n(\theta)
+
z_{1-\alpha}\sqrt{v^+_{C,n}(\theta)}
\le b
]

推出：

[
\mathbb P(C(\theta)>b)\le \alpha+\delta_{\mathrm{approx}},
]

其中 (\delta_{\mathrm{approx}}) 来自 Gaussian approximation / Berry-Esseen / sub-Gaussian tail approximation。

这条 theorem 是 OR 审稿人会关心的：你的方法不是随便把 (\hat\sigma) 代进去，而是有 conservative certification。

---

## Theorem 7：Exact SC-OLH-KG 的 one-step Bayes optimality

对于 terminal value：

[
V_n^\star
=========

\min_{\theta\in\mathcal F_n}m^J_n(\theta),
]

定义：

[
\mathrm{SC\text{-}OLHKG}_n(x)
=============================

## V_n^\star

\mathbb E_n[V_{n+1}^\star\mid x].
]

则：

[
x_n^\star
=========

\arg\max_x \mathrm{SC\text{-}OLHKG}_n(x)
]

在所有 one-step lookahead policies 中 Bayes optimal。

这条定理是 KG 的标准正当性，但 terminal value 换成了 decomposed chance-certified feasible optimum。

---

## Theorem 8：有限预算 simple regret bound

令：

[
\theta^\star
============

\arg\min_{\theta:\mathbb P(C(\theta)\le b)\ge 1-\alpha}J(\theta).
]

算法输出：

[
\hat\theta_N.
]

定义 safe simple regret：

[
r_N
===

J(\hat\theta_N)-J(\theta^\star).
]

目标 bound：

[
\boxed{
r_N
\le
C_1
\sqrt{
\frac{\beta_N\gamma_N(k_\rho)}{N}
}
+
C_2
\sqrt{
\frac{d_v\log N}{n_{\mathrm{eff}}}
}
+
C_3
(\epsilon_R+\epsilon_c+\epsilon_\rho)
+
C_4\epsilon_{\mathrm{cand}}.
}
]

其中：

[
\gamma_N(k_\rho)=\text{state-coupled kernel 的 information gain},
]

[
d_v=R+\frac{K(K+1)}{2}+K,
]

[
\epsilon_{\mathrm{cand}}=\text{candidate optimization error}.
]

这条定理的核心信息是：

[
\boxed{
\textbf{复杂度依赖有效风险维度 }d_v\textbf{，而不是状态数或轨迹长度。}
}
]

这就是你的数学深度主张。

---

# 11. 实验设计：必须证明三件事

单目标主论文实验必须证明：

[
\text{异方差分解有用}
]

[
\text{状态—策略耦合有用}
]

[
\text{KG 搜索因为前两者而更高效}
]

---

## 实验 1：Class-Hetero Synthetic

构造：

[
v_C(\theta)
===========

\sum_{k=1}^{K}N_k(\theta)\omega_k^2.
]

设置低方差类、中方差类、高方差类。
目标：证明 class-HVD 比 pooled variance、pointwise variance、当前 VEPM 更稳。

指标：

[
|\hat\omega-\omega^\star|,
]

false feasible rate，

best feasible objective，

sample efficiency。

---

## 实验 2：Shared-Shock Synthetic

构造：

[
v_C(\theta)
===========

N(\theta)^\top B N(\theta)
+
N(\theta)^\top\omega.
]

比较忽略 (B) 与估计 (B) 的方法。

预期：忽略共享冲击会系统性低估方差，导致 chance constraint violation。

---

## 实验 3：Orthogonal Factor Synthetic

构造：

[
v_C(\theta)
===========

A(\theta)^\top\Lambda A(\theta)
+
N(\theta)^\top\omega.
]

设置真实 (R=2) 或 (R=3)，并让 (\lambda_r) 快速衰减。

比较：

1. pooled variance；
2. VEPM；
3. class-HVD；
4. orthogonal-HVD；
5. oracle-HVD；
6. SC-OLH-KG。

预期：orthogonal-HVD 接近 oracle，并且样本数越少优势越明显。

---

## 实验 4：Apparent Aleatoric to Epistemic

构造：

[
C(\theta)
=========

g(\theta)
+
h(\rho(\theta))
+
\epsilon.
]

如果模型没有 (\rho)，则 (h(\rho(\theta))) 会表现为 residual variance。
加入 (\rho) 后，这部分变成均值结构。

验证：

[
\widehat v_{\mathrm{old}}(\theta)

>

\widehat v_{\mathrm{new}}(\theta),
]

且 RMSE 下降。

这直接对应你的特征构造经验。

---

## 实验 5：State-Policy Coupling Synthetic

构造两个策略：

[
\theta,\theta'
]

在参数空间距离很远：

[
|\theta-\theta'|\text{ 大},
]

但 occupancy 很近：

[
|\rho(\theta)-\rho(\theta')|\text{ 小}.
]

再构造另一对参数很近但 occupancy 很远的策略。

比较 raw-kernel KG 与 state-coupled KG。

预期：state-coupled KG 后验传播更准确，采样效率更高。

---

## 实验 6：交通信号单目标版本

把当前双目标交通问题改成：

[
\min_\theta \text{Expected network delay}
]

subject to：

[
\mathbb P(\mathrm{CO}_2(\theta)\le b)\ge 1-\alpha.
]

轨迹特征：

[
\varphi(S_t,A_t)
================

[
\text{queue length},
\text{speed},
\text{density},
\text{phase},
\text{spillback},
\text{emission rate},
\text{intersection ID},
\text{time-of-day}
].
]

风险类别：

1. free-flow；
2. near-capacity；
3. congested；
4. spillback；
5. high-emission stop-and-go。

必须用 fresh seeds 做 out-of-sample certification：

[
\widehat{\mathbb P}(\mathrm{CO}_2(\hat\theta_N)>b)
]

而不是只看保存日志。当前 manuscript 摘要里也承认 fresh-seed out-of-sample certification 是 field deployment 前必要步骤，这一版要把它补上。([GitHub][1])

---

# 12. Baselines 和 ablations

必须包括：

1. Random search；
2. Standard KG with pooled variance；
3. 当前 GPR-KG + VEPM；
4. heteroscedastic GP / hetGP-style surrogate；
5. RAHBO-style mean-variance BO；
6. SafeOpt / safe BO variant；
7. SC-KG without heteroscedastic decomposition；
8. OLH-KG without state-policy coupling；
9. class-only HVD；
10. factor-only HVD；
11. full SC-OLH-KG；
12. oracle variance / oracle risk regime。

最关键的 ablation 是：

[
\text{no coupling}
]

和：

[
\text{no decomposition}.
]

你要证明不是“一个更复杂的 KG 恰好好用”，而是：

[
\boxed{
\textbf{耦合关系提升均值/约束传播，异方差分解提升风险认证，二者共同提升 KG 搜索。}
}
]

---

# 13. repo 改造路径

基于当前 repo，建议新增四个模块，而不是推倒重写。

---

## 13.1 `encoders/policy_state_encoder.py`

```python
class PolicyStateEncoder:
    def fit(self, trajectory_logs):
        ...

    def encode_trajectory(self, traj):
        # returns rho, A, N
        ...

    def encode_policy(self, theta, trajectories=None):
        # returns E[rho], E[A], E[N], covariance estimates
        ...
```

第一版可以不用深度学习，直接用手工 trajectory summaries。

第二版再加入：

```python
class SelfSupervisedTrajectoryEncoder:
    def fit_masked_prediction(self, trajectories):
        ...

    def fit_contrastive(self, trajectories, policy_ids, risk_labels):
        ...
```

理论里只把它视为：

[
\rho_\eta(\theta)
]

并引入：

[
\epsilon_\rho
]

即可。

---

## 13.2 `variance/orthogonal_hvd.py`

```python
class OrthogonalHeteroscedasticDecomposer:
    def fit_basis(self, trajectory_logs):
        # build risk classes c and orthogonal factors psi
        ...

    def update(self, theta, C_obs, g_pred, trajectory):
        # update residual-square regression
        ...

    def predict_variance(self, theta, trajectory_summary):
        # returns v_hat, v_upper, decomposition components
        ...

    def diagnostics(self):
        # lambda, B, omega, truncation energy, class variance
        ...
```

---

## 13.3 `models/state_coupled_gpr.py`

替换当前：

[
\phi(\theta)=(1,\theta,\theta^2)
]

为：

[
\phi_{\mathrm{SC}}(\theta)
==========================

[
1,\theta,\theta^2,\rho,\rho^2,\theta\otimes\rho
].
]

或者改成 kernel-GP：

[
k(\theta,\theta')
=================

k_\theta+k_\rho.
]

当前 `ParametricGPR` 的 rank-one Kalman update 可以保留，只需要改 basis 接口。([GitHub][2])

---

## 13.4 `acquisition/sc_olhkg.py`

```python
class SCOLHKG:
    def score(self, theta):
        kg_mean = self.mean_kg(theta)
        kg_feas = self.feasibility_kg(theta)
        kg_var = self.variance_decomposition_kg(theta)
        kg_couple = self.coupling_propagation_kg(theta)
        return kg_mean + lam_f*kg_feas + lam_v*kg_var + lam_rho*kg_couple
```

---

## 13.5 `run_single_scolhkg.py`

单目标入口：

```python
python run_single_scolhkg.py \
  --objective delay \
  --constraint emission \
  --alpha 0.05 \
  --budget 150 \
  --encoder handcrafted \
  --variance_model orthogonal_hvd \
  --kernel state_coupled
```

当前 repo 已有 logging 和 candidate generation 的结构，包括 pre-sampling、KG compute、VEPM update、hypervolume logging 等；单目标版可以把 hypervolume logging 换成 best feasible objective、false-feasible rate、constraint violation rate。([GitHub][2])

---

# 14. 论文结构建议

## Section 1. Introduction

核心叙事：

[
\text{expensive stochastic simulation}
+
\text{chance constraint}
+
\text{heteroscedasticity}
+
\text{policy-induced states}
]

问题：普通 KG 只在 design space 建模，忽略策略诱导的状态结构；普通异方差模型把风险看成 pointwise variance，无法解释累计风险。

---

## Section 2. Single-Objective Chance-Constrained Simulation Optimization

定义：

[
\min J(\theta)
]

subject to：

[
\mathbb P(C(\theta)\le b)\ge 1-\alpha.
]

定义仿真轨迹、状态—策略 occupancy、KG belief。

---

## Section 3. Orthogonal Latent Heteroscedastic Decomposition

放主公式：

[
\mathrm{Var}(C\mid\mathcal T)
=============================

A^\top\Lambda A
+
N^\top BN
+
N^\top\omega.
]

放信息细化 theorem。

---

## Section 4. State-Coupled Bayesian Belief Model

定义：

[
\rho(\theta),
\quad
k_\rho,
\quad
J\sim GP,
\quad
g\sim GP.
]

说明 self-supervised/meta/hierarchical encoder 只是产生 (\rho,c,\psi) 的方法；理论只需要 approximation error。

---

## Section 5. SC-OLH-KG Acquisition

定义 exact KG：

[
V_n^\star-\mathbb E[V_{n+1}^\star\mid x].
]

再给可计算分解：

[
KG^J+KG^F+KG^V+KG^\rho.
]

---

## Section 6. Theory

放 6 到 8 个 theorem。

---

## Section 7. Experiments

synthetic + traffic single-objective case。

---

## Section 8. Conclusion

强调：

[
\boxed{
\textbf{SC-OLH-KG 不只是异方差 BO，而是利用策略诱导状态结构来学习、分解并搜索风险。}
}
]

---

# 15. 最终论文贡献可以这样写

我建议贡献写成四条：

**第一，累计风险分解。**
提出固定轨迹与随机策略下的正交潜在异方差分解，把 (\sum S) 的方差表示为 class risk、shared shock、orthogonal latent factor 和 path randomness 的组合。

**第二，状态—策略耦合 belief model。**
把策略 (\theta) 映射到 occupancy representation (\rho(\theta))，并用 state-coupled kernel 让一次采样更新轨迹相似的策略。

**第三，decomposition-aware KG 搜索。**
KG 不只追求目标均值改善，也追求 chance-boundary resolution、variance-regime learning 和 coupling propagation。

**第四，有限预算理论。**
证明 regret / feasibility certification 依赖有效维度：

[
d_v=R+\frac{K(K+1)}{2}+K
]

和 state-coupled information gain：

[
\gamma_N(k_\rho),
]

而不是状态空间大小或轨迹长度。

---

# 16. 最后给一个最凝练的主张

这篇单目标论文的核心应该是：

[
\boxed{
\begin{aligned}
&\text{策略 } \theta \text{ 不是孤立设计点，而是诱导状态—动作分布；}\
&\text{异方差不是每个状态一个独立方差，而是由少数风险类别和正交风险因子生成；}\
&\text{KG 的采样价值不只来自目标均值改善，而来自对目标、可行边界、风险因子和状态耦合结构的联合信息增益。}
\end{aligned}
}
]

如果这条线做成，数学上有：

[
\mathrm{Var}\left(\sum S\right)
]

的完整分解，有 chance constraint 的认证，有 KG 的 value-of-information 搜索，还有状态—策略耦合的降维结构。
这比“单纯 VEPM + KG”强，也比“泛泛地做 self-supervised/meta learning”更容易投 OR。

[1]: https://raw.githubusercontent.com/erzhu419/KG_op/main/Final_Submission/Final_Revised_Manuscript_OR.tex "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/erzhu419/KG_op/main/Final_Submission/GPR_KG_Code/gpr_kg.py "raw.githubusercontent.com"
[3]: https://jmlr.org/papers/v15/wilson14a.html?utm_source=chatgpt.com "Using Trajectory Data to Improve Bayesian Optimization for ..."
[4]: https://papers.nips.cc/paper/2019/file/7876acb66640bad41f1e1371ef30c180-Paper.pdf?utm_source=chatgpt.com "Offline Contextual Bayesian Optimization"
[5]: https://pubsonline.informs.org/doi/10.1287/opre.1090.0754?utm_source=chatgpt.com "Stochastic Kriging for Simulation Metamodeling - PubsOnLine"
[6]: https://arxiv.org/abs/2111.03637?utm_source=chatgpt.com "Risk-averse Heteroscedastic Bayesian Optimization"
[7]: https://arxiv.org/abs/1703.04977?utm_source=chatgpt.com "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"
[8]: https://pubsonline.informs.org/doi/10.1287/ijoc.1080.0314?utm_source=chatgpt.com "The Knowledge-Gradient Policy for Correlated Normal Beliefs"
[9]: https://proceedings.mlr.press/v202/muller23a/muller23a.pdf?utm_source=chatgpt.com "PFNs4BO: In-Context Learning for Bayesian Optimization"
[10]: https://proceedings.mlr.press/v161/eriksson21a.html?utm_source=chatgpt.com "High-dimensional Bayesian optimization with sparse axis-aligned ..."

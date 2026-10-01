# 我：
这问题本质就是自由度太多数据太少，对吧？我能想到的就是两个办法：用因果模型替换相关性模型，或者减枝。前者不用多说，后者目前的情况是，已经从巨大的参数空间，改为在函数空间（所有耦合本质都是函数关系）中搜索，只是可能现在基函数之间，好像并没有正交性约束？虽然因果也是一种正交性约束（用条件独立来减小很长的链式概率展开），但是好像更多的是状态空间，而不是范函空间（我以为寻找两个正交函数就是范函空间）？这个思路你觉得会有效果么，关于用特别少的采样次数去拟合特别高维度的未知数。我清楚所有这种问题，都需要足够好的假设，之前强先验无法泛化，我们需要的就是一个足够泛化又足够强的假设，才能解决写个问题。帮我仔细搜索下，并给个目前学术界的报告，并着重介绍最有希望的那个方向

# GPT：
是的，本质就是：

[
\boxed{
\textbf{自由度太多，样本太少。}
}
]

而且你现在的负结果其实很有信息量：SC、HVD、state basis、LODO、LLM prior 全部都在试图做同一件事——**把一个近乎不可学习的高维黑箱函数族，压缩成一个可学习的低自由度函数族**。如果压缩假设不够强，N=80 仍然不够；如果压缩假设太 problem-specific，又不能泛化。这正是 No Free Lunch 定理的现实版本：任何优化算法要在某类问题上优于其他算法，都必须利用这类问题的结构偏置；没有结构偏置，不存在普适优势。([UBC计算机科学][1])

我认为你提出的“因果模型”和“减枝/正交函数空间”两个方向都对，但**最有希望先做成 OR 论文主线的是后者：正交稀疏函数分解 + 状态耦合 HVD + KG**。因果可以作为约束或解释层加入，但不建议现在把主线改成“完全因果 BO”。

---

# 1. 当前学术界对这个问题的共识

高维 BO / 仿真优化的主流共识是：原始维度高并不是唯一问题，真正的问题是**有效自由度**和**可识别结构**。2024 年 NeurIPS 高维 BO benchmark/survey 明确指出，高维黑箱优化在样本效率、维度、实验设置可复现性上都很困难；该领域长期依靠结构假设，比如低维嵌入、变量选择、加性分解、局部模型等。

目前有一个有趣的反向结果：有论文系统测试后认为标准 GP + ARD Matérn kernel + UCB 在很多高维 BO benchmark 上也能表现很好，并不一定需要复杂结构模型；但这篇论文也承认，高维 BO 方法通常依赖两类结构假设：一类是在**函数空间**里假设加性/分解结构，另一类是在**输入空间**里假设低维子空间或稀疏有效变量。([arXiv][2]) 对你的问题来说，这个反向结果不能直接解决核心困难，因为你不是普通单目标 noiseless/high-dimensional BO，而是 **chance constraint + 异方差 + 少样本 + state-policy coupling**。你的难点不只是找目标最优，还要正确认证可行性，这比标准高维 BO 更依赖方差结构。

所以报告式地说，目前学术界主要有五条路：

| 路线                                             | 核心假设                       | 对你项目的价值                             | 风险                             |
| ---------------------------------------------- | -------------------------- | ----------------------------------- | ------------------------------ |
| Sparse input subspace                          | 只有少数 raw dimensions 有效     | SAASBO、LassoBO、GTBO 都属此类，适合 pruning | 如果有效结构不是 raw coordinate，就会负迁移  |
| Additive / orthogonal functional decomposition | 函数可分解成少数低阶正交组件             | 最贴近你说的“函数空间搜索”和“正交基”                | 需要构造正确的函数库                     |
| Causal BO / causal representation              | 只需干预/建模直接因果父节点或稳定机制        | 可解释、泛化强                             | 没有图或干预数据时，因果学习本身样本需求很高         |
| Sparse dynamics discovery / SINDy              | 动态系统由少数 governing terms 组成 | 适合 trajectory/state dynamics        | 对噪声和字典选择敏感                     |
| LLM/meta prior                                 | 用外部知识或跨域数据生成候选结构           | 可辅助 proposal / basis                | 容易被审稿人质疑 leakage/prompt tuning |

---

# 2. 你说的“正交函数空间”方向是有效的，而且比现在更正中问题

你说：

> 已经从巨大的参数空间，改为在函数空间中搜索，只是基函数之间好像没有正交性约束。

这个判断非常关键。现在的问题可能不是 state_basis 不对，而是：

[
\boxed{
\textbf{state_basis 生成了太多可相关、可替代、可互相解释的函数。}
}
]

如果基函数之间高度相关，那么在 N 很小时，模型会出现几种病：

[
\text{同一个真实效应被多个 basis 分摊，}
]

[
\text{posterior inclusion 不稳定，}
]

[
\text{HVD 方差项互相补偿，}
]

[
\text{chance bound 低估或高估，}
]

[
\text{LODO prior 在新 domain 上负迁移。}
]

这正是正交化要解决的问题。数学上，你需要的不是普通 feature engineering，而是把候选函数看成 Hilbert space (L^2(\nu)) 中的元素，其中 (\nu) 是 source + pilot target 的经验轨迹分布，然后要求：

[
\langle \psi_i,\psi_j\rangle_{\nu}
==================================

# \mathbb E_{\nu}[\psi_i(X)\psi_j(X)]

0,\quad i\ne j.
]

这样每个 basis component 都解释一块互不重叠的方差。Orthogonal Additive Kernel, OAK 正是这个方向：它把 additive GP 和 functional ANOVA 连接起来，并通过正交约束让加性函数组件可识别、低维、可解释；论文也报告了在只有少量低维加性项时能达到接近黑箱模型的预测性能。([Proceedings of Machine Learning Research][3])

所以你现在应该把主线从：

[
\text{state basis + HVD}
]

升级成：

[
\boxed{
\textbf{orthogonal sparse state-coupled functional decomposition + HVD}
}
]

也就是：

[
x \rightarrow \mathcal T_x \rightarrow \psi(x)=(A(x),N(x))
]

之后，不是把所有 (\psi) 都喂给 GPR/HVD，而是先在函数空间中做：

[
\psi_1,\ldots,\psi_M
\quad
\longrightarrow
\quad
\tilde\psi_1,\ldots,\tilde\psi_M
]

使得：

[
\langle \tilde\psi_i,\tilde\psi_j\rangle_{\nu}=0.
]

然后只允许少数 active components 进入均值、约束和 HVD。

---

# 3. 因果方向有价值，但不适合作为下一步主攻

因果模型确实是另一种强约束。它用 conditional independence、Markov blanket、direct parents、invariance 等结构，减少长链式概率展开。Causal Bayesian Optimization 的原始思想就是：如果知道或能利用因果图，就能把优化集中到对目标变量真正有因果影响的干预上；CBO 文献明确说 causal graph 可以帮助识别哪些维度重要，而不是仅靠相关性搜索。([Proceedings of Machine Learning Research][4])

但是问题在于：你现在很可能没有足够数据去学因果图。2025 年关于 unknown causal graph 的 CBO 工作也说明，传统 CBO 往往需要已知因果关系；新方法试图学习目标变量的 direct parents，并证明在一些假设下只关注 direct parents 足够，但这本身又是一个新的结构学习问题。([arXiv][5]) 更一般的因果表征学习文献也强调，没有合适归纳偏置时，学习 causal representation 是 fundamentally impossible；invariance 是很有希望的原则，但它本身仍然需要假设和多环境信息支持。

所以我的建议是：

[
\boxed{
\textbf{不要把主线改成“因果替代相关性”。}
}
]

更合适的是：

[
\boxed{
\textbf{把因果当成 pruning / invariance constraint。}
}
]

也就是说，你可以引入因果味道的条件：

[
\text{一个 basis component 要被保留，必须跨 source domains 稳定、方向一致、对 residual variance 有独立解释力。}
]

这不是完整因果图学习，但它使用了因果中的核心思想：

[
\text{真正机制应当跨环境稳定。}
]

用在你的项目里，就是保留那些在 FactorShock、Inventory、Queue、Traffic 中都稳定解释目标/约束/方差的 (\psi)-components，删掉只在单个 benchmark 上有效的伪相关组件。

---

# 4. 目前最有希望的方向：Orthogonal Sparse State-Coupled HVD-KG

我建议把下一版主方法定位为：

[
\boxed{
\textbf{OS-SC-HVD-KG: Orthogonal Sparse State-Coupled Heteroscedastic Decomposition KG}
}
]

核心思想：

[
\text{state coupling 提供候选函数库，}
]

[
\text{orthogonality 让函数组件可识别，}
]

[
\text{sparsity/pruning 让 N=80 可学习，}
]

[
\text{HVD 只在少数 active risk components 上做 chance certification，}
]

[
\text{KG 只在 pruning 后的低维风险函数空间里搜索。}
]

这比“因果替换相关性模型”更容易落地，也更容易写出 OR 论文的数学定理。

---

# 5. 数学形式：把现在的 HVD 改成正交稀疏 HVD

你现在有：

[
\psi(x)=(A(x),N(x)),
]

[
\mathrm{Var}(C(x)\mid \mathcal T)
=================================

A(x)^\top \Lambda A(x)
+
N(x)^\top B N(x)
+
N(x)^\top\omega
+
\text{floor}.
]

下一步应该改成：

[
A(x)=
(A_1(x),\ldots,A_M(x)),
]

[
N(x)=
(N_1(x),\ldots,N_K(x)),
]

但不是所有 (A_j,N_k) 都进入模型。引入选择变量：

[
\gamma_j^A\in{0,1},
]

[
\gamma_k^N\in{0,1},
]

[
\gamma_{k\ell}^B\in{0,1}.
]

于是：

[
\boxed{
\mathrm{Var}(C(x)\mid\mathcal T)
================================

\sum_{j=1}^{M}
\gamma_j^A \lambda_j A_j(x)^2
+
\sum_{k,\ell=1}^{K}
\gamma_{k\ell}^B B_{k\ell}N_k(x)N_\ell(x)
+
\sum_{k=1}^{K}
\gamma_k^N\omega_k N_k(x)
+
\sigma_0^2
+
R_\perp(x).
}
]

其中：

[
R_\perp(x)
]

是被剪掉的正交残差，用于 chance certification 的保守 floor，而不是假装没有。

同时要求：

[
\langle A_i,A_j\rangle_{\nu}=0,\quad i\ne j,
]

[
\langle N_i,N_j\rangle_{\nu}\approx 0,\quad i\ne j,
]

[
\langle A_i,N_j\rangle_{\nu}\approx 0.
]

这样做的意义是：每个保留下来的项都必须解释一块独立方差；如果两个项相关，就先正交化，再让稀疏 prior 决定谁留下。

---

# 6. 均值模型也要同样正交稀疏

不要只给 HVD 做剪枝。目标均值和约束均值也要写成同一套正交函数库：

[
J(x)
====

J_0+
\sum_{g\in \mathcal G_J}
\gamma_g^J f_g(\tilde \psi_g(x)),
]

[
g(x)
====

g_0+
\sum_{g\in \mathcal G_g}
\gamma_g^g h_g(\tilde \psi_g(x)).
]

这里 (\tilde\psi_g) 是正交化后的 state-coupled basis group。

这对应 functional ANOVA / additive GP 的思想：

[
f(x)
====

f_\emptyset
+
\sum_i f_i(x_i)
+
\sum_{i<j} f_{ij}(x_i,x_j)
+
\cdots
]

但你不在 raw (x_i) 上做，而是在 state-coupled risk coordinate 上做：

[
\psi_i(x)=\text{policy-induced exposure function}.
]

OAK 文献的关键贡献正是通过 orthogonal additive kernel 让这些 main effects 和 interactions 可识别，并获得低维可解释分解。([Proceedings of Machine Learning Research][3])

---

# 7. Pruning 机制：不要再让所有 basis 都参与 KG

你需要一个明确的 pruning rule。推荐使用三层：

## 第一层：正交化后按证据剪枝

定义第 (g) 个 group 的 posterior inclusion probability：

[
\mathrm{PIP}_g
==============

\mathbb P(\gamma_g=1\mid D_n).
]

如果：

[
\mathrm{PIP}*g<\tau*{\mathrm{drop}}
]

且：

[
\mathrm{UCBEffect}*g<\epsilon*{\mathrm{effect}},
]

则暂时剪掉。

为了防止早期误剪，可以用 hysteresis：

[
\mathrm{PIP}*g<\tau*{\mathrm{drop}}
\quad \text{连续 }L\text{ 轮才剪。}
]

被剪掉的项不再参与：

[
\text{GPR basis},
\quad
\text{HVD},
\quad
\text{candidate generation},
\quad
\text{KG acquisition}.
]

但其不确定性进入：

[
R_\perp(x)
]

作为保守 floor。

---

## 第二层：按 KG value 剪枝

有些 basis 对拟合有帮助，但对决策没帮助。定义：

[
\mathrm{VOI}_g
==============

\mathbb E[
V_{n+1}^{\star}(g\text{ active})
--------------------------------

V_{n+1}^{\star}(g\text{ inactive})
].
]

如果某个 group 对 feasibility boundary 和 objective elite set 的 KG value 都很低，也剪。

这比纯统计剪枝更 OR，因为它只保留对决策有价值的结构。

---

## 第三层：跨 domain invariance 剪枝

对于 LODO source domains，计算每个 group 的 effect sign / PIP / HVD contribution 是否稳定：

[
\mathrm{Inv}_g
==============

\mathrm{Var}*{d\in \mathcal D*{\mathrm{source}}}
\left(
\widehat{\beta}_{g,d}
\right)
]

或：

[
\mathrm{Inv}_g
==============

\mathrm{Var}*{d}
\left(
\mathrm{PIP}*{g,d}
\right).
]

如果一个 group 只在单个 source domain 强，在其他 domain 弱，且在 held-out pilot 上没有证据，就降低 prior inclusion probability。这个就是你想要的“泛化强先验”：不是手写答案，而是保留跨环境稳定机制。

---

# 8. 和 SAAS / LassoBO / GTBO 的关系

这个方向和现有高维 BO 的稀疏路线一致，但更适合你的问题。

SAASBO 用 half-Cauchy shrinkage prior 让大多数维度的 inverse lengthscale 接近 0，相当于自动“关掉”大多数 raw input dimensions；随着数据增加，posterior 可以让更多维度 escape zero，逐步打开更复杂函数类。 LassoBO 也用变量选择思想，通过估计 GP kernel lengthscales 找重要变量，再在有效子空间里优化 acquisition；它报告了 sublinear cumulative regret 和高维 synthetic / real-world 上的好表现。([Proceedings of Machine Learning Research][6]) GTBO 则把高维 BO 中的 active dimension identification 转成 group testing，先用 testing phase 找 active dimensions，再在识别出的子空间里优化。([arXiv][7])

但这些方法主要在 raw input coordinate 上做稀疏：

[
x_1,\ldots,x_d.
]

你的情况更像：

[
\psi_1(x),\ldots,\psi_M(x),
]

其中每个 (\psi_j) 是一个 state-policy coupling function。也就是说，你不是要找“哪个原始坐标有用”，而是要找：

[
\boxed{
\textbf{哪个状态暴露函数 / 风险机制函数有用。}
}
]

所以你可以把自己的方法定位为：

[
\text{SAAS/Lasso/GTBO in a state-coupled functional basis, not in raw coordinate space.}
]

这就是你的创新点之一。

---

# 9. SINDy 给你的启发：字典扩展 + 压缩，而不是一次性塞满

SINDy 的核心思想是：很多动态系统虽然表面复杂，但 governing equations 只需要少数 active terms；它用一个候选函数 dictionary，然后通过稀疏回归识别少数项。最新 iterative SINDy 还专门讨论了 dictionary expansion and compression，因为过大的字典会导致计算复杂度和高维问题，而迭代扩展压缩可以接近原 SINDy 效果并降低计算负担。([arXiv][8])

这对你非常重要。你现在的问题可能是：state_basis 一开始就太丰富，导致 HVD/GPR 在 N 很少时无法识别。下一版应该借鉴 SINDy：

[
\text{small dictionary}
\rightarrow
\text{fit}
\rightarrow
\text{prune}
\rightarrow
\text{expand only around residual}
\rightarrow
\text{prune again}.
]

而不是：

[
\text{一次性生成 huge basis，然后希望 Bayesian posterior 自己解决。}
]

N=80 的场景下，posterior 不会神奇解决一切；你必须主动控制 model class。

---

# 10. 为什么我不建议现在主攻完整因果模型

因果当然诱人，因为它比相关性更泛化。但你现在的数据形态是：

[
N \text{ 小，}
\quad
d \text{ 大，}
\quad
\text{多 domain 但每个 domain 样本少，}
\quad
\text{目标是优化而非纯预测。}
]

要从中学完整 causal graph，难度可能比原优化问题还大。近期 unknown-graph CBO 的方向已经在做“只学目标变量 direct parents”，这是合理的简化；但它仍然需要足够信息来区分 causal parent 和相关变量。([arXiv][5]) 而 causal representation learning 的理论综述也直接指出，没有假设时 causal representation 不可识别，需要额外 inductive biases；这说明“用因果替代相关性”本身并不是免费午餐。

所以我建议因果只承担三件事：

[
\textbf{第一，作为 invariance pruning。}
]

保留跨 domain 稳定的 (\psi)-components。

[
\textbf{第二，作为 candidate dictionary 的生成规则。}
]

例如只生成符合时序方向的函数：

[
S_t,A_t \rightarrow S_{t+1} \rightarrow C.
]

不生成违反时间方向的伪因果项。

[
\textbf{第三，作为 HVD shared shock 的解释层。}
]

例如：

[
N^\top B N
]

可以解释为 shared latent shock / common cause，而不是简单相关项。

但不要把主定理写成“我们学到真实因果图”。那会把审稿重点带到 causal identification，而不是 OR / KG / HVD。

---

# 11. 下一步最应该实现的模型

我建议你把当前 SC-OLH-KG 改成：

[
\boxed{
\textbf{SC-OAK-HVD-KG}
}
]

即：

[
\textbf{State-Coupled Orthogonal Additive Kernel Heteroscedastic Decomposition KG}
]

核心模块如下。

## 11.1 统一候选函数库

从 trajectory/state/policy 生成候选函数：

[
\mathcal D
==========

{
\psi_1,\ldots,\psi_M
}.
]

这些 (\psi_j) 可以包括：

[
\text{state exposure moments},
]

[
\text{risk regime occupancy},
]

[
\text{policy-state interaction},
]

[
\text{local instability / variance proxy},
]

[
\text{shared shock proxy},
]

[
\text{low-order interactions}.
]

但 (M) 必须受控。N=80 时，不要让 (M) 上百。第一版我建议：

[
M\le 30,
]

最后 active groups：

[
s\le 5\sim 8.
]

这不是随便拍脑袋，而是因为在 N 很小时，你的有效自由度必须低到能被交叉验证、posterior inclusion 和 chance certification 支持。

---

## 11.2 经验正交化

在 source + target pilot 的经验分布 (\hat\nu) 上计算 Gram matrix：

[
G_{ij}
======

\frac{1}{|\mathcal P|}
\sum_{x\in\mathcal P}
\psi_i(x)\psi_j(x).
]

做 whitening：

[
\tilde\psi(x)
=============

G^{-1/2}\psi(x).
]

然后检查：

[
\max_{i\ne j}
|\langle \tilde\psi_i,\tilde\psi_j\rangle_{\hat\nu}|.
]

如果这个值仍然很高，说明 basis 没有真正正交，不能进入主实验。

---

## 11.3 稀疏 additive GP / OAK GP

目标和约束用：

[
J(x)
====

J_0+
\sum_{g\in S_J} f_g(\tilde\psi_g(x)),
]

[
g(x)
====

g_0+
\sum_{g\in S_g} h_g(\tilde\psi_g(x)).
]

其中 (S_J,S_g) 是稀疏 active groups。

可以用两种方式实现：

1. 轻量版：Bayesian linear / spline basis + group lasso / spike-slab；
2. GP 版：Orthogonal additive kernel + ARD / group sparsity。

OAK 是更漂亮的论文主线，因为它提供了 functional ANOVA、正交性和解释性。([Proceedings of Machine Learning Research][3])

---

## 11.4 稀疏 HVD

HVD 使用同一套 active basis：

[
\mathrm{Var}(C(x)\mid\mathcal T)
================================

\sum_{j\in S_A}\lambda_j A_j(x)^2
+
\sum_{k,\ell\in S_B}B_{k\ell}N_k(x)N_\ell(x)
+
\sum_{k\in S_N}\omega_kN_k(x)
+
\sigma_0^2
+
R_\perp(x).
]

这个设计非常重要，因为它让均值模型和方差模型共享同一个 state-coupled functional coordinate system，而不是各自学一套无关特征。

---

## 11.5 KG 只在 active basis 上搜索

candidate generation 不再在完整 raw (x)-space 里乱找，而是在：

[
\tilde\psi_S(x)
]

上找：

[
\mathcal A_n
============

\mathcal A_{\mathrm{elite}}
\cup
\mathcal A_{\mathrm{boundary}}
\cup
\mathcal A_{\mathrm{variance}}
\cup
\mathcal A_{\mathrm{safe\text{-}interior}}.
]

其中所有集合都只依赖 active basis。

如果一个 candidate 在 raw space 看起来新，但在 active (\psi_S)-space 里和已有点重复，就不要采。
如果一个 candidate 在 raw space 距离远，但在 (\psi_S)-space 里提供新的 boundary / variance 信息，就可以采。

这会显著降低搜索自由度。

---

# 12. 理论上怎么写才有深度

可以写四个核心定理。

## Theorem 1: Orthogonal state-coupled functional ANOVA

设：

[
\mathcal H=L^2(\nu)
]

且：

[
{\tilde\psi_g}_{g=1}^M
]

正交。若：

[
f(x)=f_0+\sum_{g=1}^{M}f_g(\tilde\psi_g(x))+r_\perp(x),
]

则：

[
|f-f_0|^2_{\mathcal H}
======================

\sum_{g=1}^{M}|f_g|^2_{\mathcal H}
+
|r_\perp|^2_{\mathcal H}.
]

这让每个 group 的贡献可分解、可排序、可剪枝。

## Theorem 2: Sparse recovery / estimation error

若真实 active set 大小为 (s)，候选库大小为 (M)，且正交化后满足 restricted eigenvalue / incoherence condition，则：

[
|\hat f-f^\star|^2
==================

O\left(
\frac{s\log M}{n}
\right)
+
\epsilon_{\mathrm{approx}}.
]

这里的重点是：

[
s\log M
]

而不是：

[
M
]

或 raw dimension (d)。

## Theorem 3: Sparse HVD chance certification

若：

[
v_C(x)
\le
\hat v_{S,n}(x)+R_{\perp,n}(x)+\Gamma_n(x),
]

则：

[
m_n^g(x)
+
\beta_n^{1/2}s_n^g(x)
+
z_{1-\alpha}
\sqrt{
\hat v_{S,n}(x)+R_{\perp,n}(x)+\Gamma_n(x)
}
\le b
]

推出：

[
\mathbb P(C(x)>b)\le \alpha+\delta.
]

这里 (R_\perp) 是剪枝后保留下来的安全项，避免因为剪掉 basis 而过度自信。

## Theorem 4: KG regret depends on active functional dimension

若 active basis 数为 (s)，HVD active risk dimension 为 (s_v)，candidate proposal 在 active (\psi)-space 有覆盖率 (p_\epsilon)，则 safe simple regret 满足类似：

[
r_N
\le
C_1
\sqrt{
\frac{\gamma_N(k_{\psi_S})}{N}
}
+
C_2
\sqrt{
\frac{s_v\log M}{n_{\mathrm{eff}}}
}
+
C_3\epsilon_{\mathrm{approx}}
+
C_4\epsilon_{\mathrm{cand}}.
]

这样论文的主张非常清楚：

[
\boxed{
\textbf{样本复杂度由 active functional dimension 决定，而不是 raw policy dimension 决定。}
}
]

这就是你要的“足够强、又更可泛化”的假设。

---

# 13. 实验上怎么验证这个方向真的有效

你应该立刻做以下 ablation。

## 13.1 Orthogonality ablation

比较：

[
\text{raw state basis}
]

[
\text{orthogonalized state basis}
]

[
\text{orthogonalized + sparse pruning}
]

指标：

[
\max_{i\ne j}|\langle\psi_i,\psi_j\rangle|,
]

[
\text{condition number of Gram matrix},
]

[
\text{PIP stability},
]

[
\text{feasible rate},
]

[
\text{violation},
]

[
\text{regret}.
]

如果 orthogonalization 有效，你应该看到 posterior inclusion 更稳定，violation 下降。

## 13.2 Pruning path ablation

保留 active groups 数：

[
s\in{3,5,8,12,20,\text{all}}.
]

如果我的判断对，你会看到：

[
s=5\sim 8
]

可能比 all 更稳，尤其 true feasible rate 更高。

## 13.3 Dense vs sparse HVD

比较：

[
A^\top \Lambda A+N^\top BN+N^\top\omega
]

全量版和稀疏版。

如果全量版 regret 低但 feasible rate 差，说明它过拟合风险结构，chance certification 不可靠。

## 13.4 Invariance pruning ablation

比较：

[
\text{source-domain stable basis only}
]

和：

[
\text{all source-active basis}
]

如果前者在 LODO 上更稳，就说明你找到了更泛化的强先验。

## 13.5 Causal-style pruning as auxiliary

加入时序/因果约束：

[
S_t,A_t \rightarrow S_{t+1}\rightarrow C
]

禁止明显违反时间方向的 basis。
这可以作为 causal pruning ablation，而不是完整 causal model。

---

# 14. 下一步优先级

我的建议顺序是：

[
\boxed{
\textbf{第一优先级：正交化 + 稀疏剪枝。}
}
]

不是 proposal，不是 LLM，不是完整因果图。

具体做：

1. 把当前所有 state_basis / risk_exposures 组成候选库 (\mathcal D)；
2. 在 source + pilot target empirical measure 上做 whitening / QR / Gram-Schmidt；
3. 给 mean、constraint、HVD 共用这套正交 basis；
4. 给每个 basis group 加 PIP / group sparsity；
5. 剪掉低 PIP、低 VOI、不稳定的 groups；
6. 被剪掉的方差进入 residual floor；
7. KG 只在 active (\psi_S)-space 里做 candidate scoring。

[
\boxed{
\textbf{第二优先级：chance bound 加 residual floor。}
}
]

剪枝会让模型更简单，但不能让模型过度自信。所有被剪掉的函数空间能量都要进入：

[
R_\perp(x)
]

用于安全认证。

[
\boxed{
\textbf{第三优先级：把因果作为 invariance pruning。}
}
]

不要学完整因果图，先用跨 domain 稳定性、时间方向、局部父节点约束来删 basis。

[
\boxed{
\textbf{第四优先级：LLM 只做候选 basis/proposal 的辅助。}
}
]

LLM 不要进入主理论，它可以帮你提出候选函数，但必须被正交化和 evidence gate 筛选。

---

# 15. 最终判断

你的直觉是对的：**寻找正交函数，本质上就是在函数空间里做降维和剪枝**。更准确地说，它不是“范函空间”里的优化，而是在一个 (L^2(\nu)) 函数空间里寻找少数正交 active components；functional ANOVA / OAK 正是这个数学语言。

因果和正交不是同一件事：

[
\text{因果条件独立}
]

是在概率图 / 状态变量空间里减少依赖；

[
\text{函数正交}
]

是在 (L^2) 函数空间里减少冗余解释。

但它们可以合并：

[
\boxed{
\textbf{用因果/不变性决定哪些函数组应该存在，用正交稀疏分解决定哪些函数组真的被数据支持。}
}
]

目前最有希望的方向就是：

[
\boxed{
\textbf{State-coupled orthogonal sparse additive HVD-KG.}
}
]

它比完整因果模型更可落地，比继续堆 state_basis 更可泛化，比纯 LODO/LLM prior 更容易写出数学理论。它直接解决你现在的根本问题：

[
\boxed{
\textbf{N 太小，所以模型必须主动把函数自由度压到极少数正交、稳定、可解释的 active components。}
}
]

[1]: https://www.cs.ubc.ca/~hutter/earg/papers07/00585893.pdf "No Free Lunch Theorems For Optimization - Evolutionary Computation, IEEE Transactions on"
[2]: https://arxiv.org/html/2402.02746v3 "Standard Gaussian Process Can Be Excellent for High-Dimensional Bayesian Optimization"
[3]: https://proceedings.mlr.press/v162/lu22b.html "Additive Gaussian Processes Revisited"
[4]: https://proceedings.mlr.press/v108/aglietti20a/aglietti20a.pdf?utm_source=chatgpt.com "Causal Bayesian Optimization"
[5]: https://arxiv.org/abs/2503.19554 "Causal Bayesian Optimization with Unknown Graphs"
[6]: https://proceedings.mlr.press/v258/hoang25a.html "High Dimensional Bayesian Optimization using Lasso Variable Selection"
[7]: https://arxiv.org/html/2504.06111v1 "Leveraging Axis-Aligned Subspaces for High-Dimensional Bayesian Optimization with Group Testing"
[8]: https://arxiv.org/html/2406.03779v1 "Iterative Sparse Identification of Nonlinear Dynamics"


# 我：

除了我提到的正交假设，你说的稀疏假设和SINDy即加性假设，都非常正中要害，也都是我提问完自己认为最有可能发挥作用的假设。
以及，在不确定环境下（异方差一定是个不确定环境），假设环境是低频规律而非高频规律，也许对解这个问题更有利。你说UCB被发现也起到重要作用。我自己的理解是，每个正则化项，本质都是一个滤波器，给目标频段之外规律一个巨大的惩罚，核心原因也是因为环境或者说数据的生成规律和方式都是相对低频的，而噪声或者方差都是高频的。
这些假设也都是在meta空间中的，其实本质就是解一个未知数数量远多于方程的问题。我们增加的假设，其实也就是在通过增加等式或者不等式约束条件，来试图把方程数量增多一些，或者把未知数数量降低一些。
把频率这个因素也加进来，看看如何更完善这个方案

# GPT：
你的“频率假设”非常应该加进去，而且我认为它可以把前面几条假设统一起来：

[
\boxed{
\textbf{正交性}+\textbf{稀疏性}+\textbf{加性/SINDy}+\textbf{低频性}
}
]

可以合成一个更强、更泛化的主假设：

[
\boxed{
\textbf{环境响应和风险结构在 state-coupled manifold 上是低频、稀疏、正交可分解的。}}
]

这比单独说“稀疏”或“正交”更完整。因为稀疏告诉你“只需要少数项”，正交告诉你“这些项互不重复解释”，加性告诉你“复杂函数由低阶组件组合”，而低频告诉你“这些组件本身不应该剧烈振荡”。这四个合起来，才真正能把“未知数远多于方程”的问题压到可学习范围内。

---

# 1. 你的滤波器理解是对的

你说：

> 每个正则化项，本质都是一个滤波器，给目标频段之外规律一个巨大惩罚。

这个理解非常准确。

在函数估计里，正则化并不只是“防止参数太大”，而是在表达一种频谱偏好。比如：

[
|f|_{L^2}^2
]

惩罚整体能量；

[
|\nabla f|_{L^2}^2
]

惩罚快速变化；

[
|\Delta f|_{L^2}^2
]

更强地惩罚曲率和高频振荡；

RKHS norm 则由 kernel 的谱衰减决定哪些函数容易、哪些函数昂贵。GP-UCB 理论里，函数复杂度通常由 GP prior 或 RKHS norm / information gain 控制；Srinivas 等的 GP-UCB 分析就是在“函数来自 GP”或“函数有低 RKHS norm”的假设下给 regret bound。([ICML][1])

频率角度下，正则化相当于：

[
f=\sum_{\ell} \theta_\ell \phi_\ell,
]

[
\mathcal R(f)=\sum_{\ell} r(\omega_\ell)\theta_\ell^2,
]

其中：

[
r(\omega_\ell)
]

随频率 (\omega_\ell) 增大而增大。于是高频项即使能解释训练样本，也会被强烈惩罚。

这和 neural network 里的 spectral bias / frequency principle 是同一类现象：深度网络通常倾向于先拟合低频函数，即全局平滑、没有局部剧烈振荡的结构；Rahaman 等把这种低频优先称为 spectral bias，并用 Fourier 分析研究了它。([Proceedings of Machine Learning Research][2]) 2022 年 NeurIPS 的实证工作也在现代图像分类网络上测量 spectral bias，发现这种低频偏好在实践中确实能观察到。([NeurIPS Papers][3])

但你的问题不是要用神经网络本身，而是要把这个思想写成一个**可控的频谱先验**，放进 SC-OLH-KG。

---

# 2. 最关键的改动：频率不能定义在 raw (x)-space，而要定义在 (\psi)-space

这里非常重要。

如果 (x\in\mathbb R^{1000})，你不能简单说：

[
f(x)\text{ 是低频的。}
]

因为 raw policy coordinate 的频率未必有意义。两个 (x) 在欧氏距离上很近，诱导出的状态轨迹可能完全不同；两个 (x) 在原空间很远，却可能诱导相似状态暴露。

所以低频假设应该写在：

[
\psi(x)=(A(x),N(x))
]

这个 state-coupled cumulative-risk coordinate 上。

也就是：

[
\boxed{
f(x)\approx \tilde f(\psi(x)),
}
]

并假设：

[
\boxed{
\tilde f \text{ 在 } \psi\text{-manifold 上是低频的。}
}
]

更具体：

[
x
\longrightarrow
\mathcal T_x
\longrightarrow
\psi(x)=(A(x),N(x))
\longrightarrow
J(x),g(x),v(x).
]

低频不是说：

[
J(x)\text{ 对 raw }x\text{ 平滑};
]

而是说：

[
J(x),g(x),v(x)\text{ 对 state-risk exposure }\psi(x)\text{ 平滑}.
]

这正好解释你之前 state_basis=on 才稳、state_basis=off 会负迁移的结果：可迁移规律不在 raw (x)，而在 (\psi)-space。

---

# 3. 统一假设：Low-Frequency Orthogonal Sparse Additive HVD

我建议新主方法命名为：

[
\boxed{
\textbf{LF-OS-SC-HVD-KG}}
]

即：

[
\textbf{Low-Frequency Orthogonal Sparse State-Coupled Heteroscedastic-Decomposition KG}
]

中文：

[
\boxed{
\textbf{低频正交稀疏状态耦合异方差分解 KG}}
]

核心假设：

[
\boxed{
\begin{aligned}
J(x)&=J_0+\sum_{m\in S_J} f_m(\psi_m(x))+\epsilon_J(x),\
g(x)&=g_0+\sum_{m\in S_g} h_m(\psi_m(x))+\epsilon_g(x),\
v_C(x)&=A(x)^\top\Lambda A(x)+N(x)^\top B N(x)+N(x)^\top\omega+\sigma_0^2+R_\perp(x).
\end{aligned}
}
]

并且每个有效组件同时满足：

[
\text{orthogonal},
]

[
\text{sparse},
]

[
\text{additive / low-order},
]

[
\text{low-frequency}.
]

这四个条件可以写成：

[
\langle \psi_i,\psi_j\rangle_\nu=0,\quad i\ne j,
]

[
|S_J|,|S_g|,|S_v|\ll M,
]

[
f=\sum_m f_m+\sum_{m<m'} f_{mm'}+\cdots
\quad \text{但只保留低阶项},
]

[
\sum_\ell \Omega_\ell^{2s}\theta_\ell^2\le R^2.
]

其中 (\Omega_\ell) 是 (\psi)-manifold 上的频率。

---

# 4. 怎么定义 (\psi)-space 上的频率？

有两种实现路线。

---

## 4.1 连续 (\psi)-space：Fourier / RFF / spectral kernel

如果：

[
\psi(x)\in\mathbb R^r
]

且 (r) 不大，比如 (r\le 10)，可以直接用 Fourier basis：

[
\phi_\omega(\psi)=\cos(\omega^\top \psi),\quad \sin(\omega^\top \psi).
]

低频约束就是只允许：

[
|\omega|\le \Omega_{\max},
]

或者对高频惩罚：

[
\mathcal R(f)
=============

\sum_\omega
(1+|\omega|^2)^s |\theta_\omega|^2.
]

对应 kernel 可以选择低通谱密度：

[
S(\omega)
\propto
(1+|\omega|^2/\kappa^2)^{-s}
]

或者 RBF 谱密度：

[
S(\omega)
\propto
\exp(-\ell^2|\omega|^2/2).
]

Bochner 视角下，平稳 kernel 可以通过频谱密度表示；RBF kernel 的谱密度是高频快速衰减的 Gaussian，因此它本身就是一种强低频先验。CMU 的 GP 讲义也用 Bochner 定理说明 RBF kernel 的 Fourier transform 是 Gaussian spectral density，并指出 RBF 对高频函数支持很少。([CMU School of Computer Science][4])

如果你想更灵活，也可以用 spectral mixture kernel。2025 年已有工作把 spectral mixture kernel 用于 BO，目标就是通过频谱结构选择更合适的 GP surrogate。([arXiv][5]) 但对你现在的少样本问题，我不建议一开始用太灵活的 spectral mixture，因为自由度会再次爆炸。更合适的是**低频截断 + 稀疏选择**。

---

## 4.2 离散/流形 (\psi)-space：graph Laplacian 频率

更适合你项目的是 graph frequency。

因为 (\psi(x)) 可能不是规则欧氏空间，而是由 policy-induced trajectories 形成的点云。于是可以把已观测点、source-domain 点、candidate 点放成图：

[
\mathcal G=(\mathcal V,\mathcal E),
]

节点是：

[
\psi(x_i),
]

边权：

[
W_{ij}
======

\exp\left(
-\frac{|\psi(x_i)-\psi(x_j)|^2}{2h^2}
\right).
]

图 Laplacian：

[
L=D-W.
]

求特征分解：

[
L u_\ell=\mu_\ell u_\ell.
]

其中：

[
\mu_1\le \mu_2\le \cdots
]

就是图上的频率。小 (\mu_\ell) 是低频，表示在 (\psi)-manifold 上缓慢变化；大 (\mu_\ell) 是高频，表示在相邻策略之间剧烈振荡。

于是函数可以展开为：

[
f(\psi(x_i))
============

\sum_{\ell=1}^{M}
\theta_\ell u_\ell(i).
]

低频正则化：

[
\mathcal R_{\mathrm{freq}}(f)
=============================

# f^\top L^s f

\sum_{\ell}
\mu_\ell^s\theta_\ell^2.
]

低频截断：

[
f_R
===

\sum_{\ell=1}^{R}
\theta_\ell u_\ell.
]

这非常适合你，因为它不要求 raw (x) 有意义，也不要求 (\psi)-space 是规则网格。它只要求：

[
\text{相似 state-risk exposure 的策略应该有相似目标/约束/风险。}
]

这正是 SC 的核心。

---

# 5. HVD 里如何加入频率？

你现在的 HVD 是：

[
\mathrm{Var}(C(x)\mid\mathcal T)
================================

A(x)^\top\Lambda A(x)
+
N(x)^\top B N(x)
+
N(x)^\top\omega
+
\sigma_0^2.
]

现在加频率后，重点不是让方差本身无限灵活，而是让：

[
A(x),N(x)
]

来自低频正交 exposure basis。

---

## 5.1 低频 (A)：local/idiosyncratic exposure

构造候选 exposure：

[
\tilde A_1(x),\ldots,\tilde A_M(x).
]

在 (\psi)-graph 上正交化并低频过滤：

[
A_j(x)
======

\sum_{\ell\le R_A}
a_{j\ell}u_\ell(\psi(x)).
]

或者直接令：

[
A(x)=
[u_1(\psi(x)),\ldots,u_{R_A}(\psi(x))].
]

然后：

[
A^\top\Lambda A
===============

\sum_{\ell\le R_A}\lambda_\ell u_\ell(\psi(x))^2.
]

这表示：局部/idiosyncratic 风险不是每个点独立变，而是在 state-risk manifold 上低频变化。

---

## 5.2 低频 (N)：shared-shock exposure

(N) 是 regime occupancy。过去可能是 hard cluster 或 problem-specific risk_exposure。现在改成低频 soft clustering：

[
N_k(x)
======

\sum_{t=0}^{T}
\pi_k(S_t,A_t),
]

其中：

[
\pi_k
]

本身要在 state graph 上低频：

[
\pi_k^\top L \pi_k
\le c_k.
]

也就是说，risk regime 不应该在相邻状态间频繁跳变。
这比随便聚类更强，也更符合“环境规律低频”的假设。

于是 shared shock：

[
N^\top B N
]

表示少数低频 regime 的共同冲击。

---

## 5.3 高频残差不能直接丢，要进入 safety floor

这是 chance constraint 中最关键的一点。

低频假设可能错。如果把高频残差直接忽略，true feasible rate 会继续低。

所以定义频谱分解：

[
v_C(x)
======

v_{\mathrm{low}}(x)
+
v_{\mathrm{high}}(x).
]

低频部分由 HVD 学：

[
v_{\mathrm{low}}(x)
===================

A_R^\top\Lambda A_R
+
N_R^\top B N_R
+
N_R^\top\omega.
]

高频部分不拟合，只作为保守项：

[
v_{\mathrm{high}}(x)\le \tau_{\mathrm{high},n}(x).
]

最终 chance certification 用：

[
\boxed{
m_n^g(x)
+
\sqrt{\beta_n}s_n^g(x)
+
z_{1-\alpha}
\sqrt{
v_{\mathrm{low},n}(x)
+
\tau_{\mathrm{high},n}(x)
+
\sigma_0^2
}
\le b.
}
]

这一步非常重要：低频假设用于降低自由度；高频残差用于安全保护。
如果没有 (\tau_{\mathrm{high}})，算法会过度自信。

---

# 6. 频率假设如何和稀疏/SINDy结合？

SINDy 的核心不是“加性”这么简单，而是：

[
\text{从一个候选函数字典里，只选少数 active terms。}
]

PySINDy 文档也明确说，SINDy 用 sparse regression 找出能捕捉动态行为的少数 basis functions 线性组合。([pysindy.readthedocs.io][6]) 2025 年的弱形式 SINDy 工作则专门强调 weak formulation 能增强对 noisy data 的鲁棒性，用来从噪声数据中识别 governing equations。([ScienceDirect][7])

在你的项目里，SINDy 思想应该这样用：

[
\mathcal D
==========

{
\text{low-frequency state exposure terms},
\text{low-order interactions},
\text{regime occupancy terms},
\text{shared-shock terms}
}.
]

然后选择：

[
S\subset \mathcal D,\quad |S|\ll |\mathcal D|.
]

但是要加一个顺序：

[
\boxed{
\text{先低频过滤，再稀疏选择。}
}
]

不要反过来。因为如果不先过滤，高频噪声项可能因为偶然拟合小样本而被 sparse regression 选中。

建议流程：

[
\text{candidate functions}
\rightarrow
\text{orthogonalize}
\rightarrow
\text{low-pass filter}
\rightarrow
\text{sparse select}
\rightarrow
\text{HVD/KG}.
]

这比纯 SINDy 更适合少样本仿真优化，因为它先把“噪声诱导的高频假规律”压掉。

---

# 7. 加性假设如何放进来？

Orthogonal Additive Kernel, OAK 是非常适合你现在思路的文献锚点。OAK 把 additive GP 和 functional ANOVA 连接起来，并通过对 additive functions 施加正交约束，得到可识别、低维、可解释的函数分解。([Proceedings of Machine Learning Research][8]) OAK 论文也明确说，正交约束可以带来 identifiable low-dimensional representation，并连接 functional ANOVA。([arXiv][9])

你可以把 OAK 从 raw variables 上搬到 (\psi)-coordinates 上：

[
f(\psi)
=======

f_0
+
\sum_i f_i(\psi_i)
+
\sum_{i<j} f_{ij}(\psi_i,\psi_j)
+
\cdots
]

但只保留：

[
\text{main effects}
+
\text{少数 two-way interactions}.
]

频率约束放在每个组件上：

[
|f_i|_{\mathrm{freq}}^2
=======================

\sum_{\ell}
\mu_\ell^s\theta_{i\ell}^2,
]

[
|f_{ij}|_{\mathrm{freq}}^2
==========================

\sum_{\ell}
\mu_{ij,\ell}^s\theta_{ij,\ell}^2.
]

稀疏约束选择 active components：

[
\gamma_i,\gamma_{ij}\in{0,1}.
]

所以最终是：

[
\boxed{
f(\psi)
=======

f_0+
\sum_i \gamma_i f_i^{\mathrm{low}}(\psi_i)
+
\sum_{i<j}\gamma_{ij}f_{ij}^{\mathrm{low}}(\psi_i,\psi_j)
+
r_\perp.
}
]

这就是：

[
\textbf{低频 + 正交 + 稀疏 + 加性}
]

的统一模型。

---

# 8. UCB 为什么在这个框架里重要？

你提到 UCB 很重要。我的理解是：在这种强先验场景下，KG 有时太依赖当前 posterior model；如果 posterior 的低频/稀疏/HVD 假设错了，KG 可能被误导。UCB 的价值在于它更直接地保留了“对未知区域的保护性探索”。

GP-UCB 的理论核心是：

[
\mathrm{UCB}_n(x)
=================

## m_n(x)

\sqrt{\beta_n}s_n(x)
]

用于 minimization，或者 maximization 时取 (m_n+\sqrt{\beta_n}s_n)。它在 GP/RKHS smoothness 假设下有 regret guarantees；后续关于 GP-UCB 的研究也把 kernel eigen-decay / information gain 看作学习难度的关键量。([ICML][1])

在你的框架里，UCB 应该不是替代 KG，而是补 KG 的 model-risk：

[
\mathrm{Acq}(x)
===============

\mathrm{KG}*{\mathrm{LF\text{-}HVD}}(x)
+
\lambda*{\mathrm{ucb}}
\mathrm{UCB}_{\mathrm{residual}}(x).
]

其中：

[
\mathrm{UCB}_{\mathrm{residual}}(x)
]

专门探索三类地方：

1. 低频模型不确定的地方；
2. 高频残差 floor 大的地方；
3. chance boundary 附近但 HVD evidence 不足的地方。

也就是说：

[
\boxed{
\text{KG 负责在低频结构内高效学习；UCB 负责防止低频结构错得太离谱。}
}
]

这对 true feasible rate 很重要。

---

# 9. 频率加入后，完整 chance bound 应该写成什么？

定义约束均值：

[
g(x)
====

g_{\mathrm{low}}(x)+g_{\mathrm{high}}(x).
]

后验估计：

[
m_n^g(x),\quad s_n^g(x).
]

HVD 低频方差：

[
\hat v_{\mathrm{low},n}(x)
==========================

A_R(x)^\top \hat\Lambda A_R(x)
+
N_R(x)^\top \hat B N_R(x)
+
N_R(x)^\top \hat\omega.
]

高频残差上界：

[
\tau_{\mathrm{high},n}(x)
=========================

\hat \sigma_{\mathrm{high},n}^2
\cdot
\left[
1+
\chi_{\mathrm{boundary}}(x)
+
\chi_{\mathrm{novel}}(x)
\right].
]

其中：

[
\chi_{\mathrm{boundary}}(x)
]

表示 (x) 是否在 chance boundary 附近；

[
\chi_{\mathrm{novel}}(x)
]

表示 (x) 在 (\psi)-space 是否远离已采样区域。

最终认证：

[
\boxed{
m_n^g(x)
+
\sqrt{\beta_n}s_n^g(x)
+
z_{1-\alpha}
\sqrt{
\hat v_{\mathrm{low},n}(x)
+
\tau_{\mathrm{high},n}(x)
+
\sigma_0^2
}
+
\gamma_n
\le b.
}
]

这里：

[
\gamma_n
]

是 source-calibrated safety slack。

这条式子能解决你之前的问题：violation/regret 下降但 true feasible rate 低。因为现在模型不会把“未解释的高频残差”当成不存在，而是放进保守项。

---

# 10. 频率剪枝规则

建议你加入一个明确的 pruning pipeline。

---

## 10.1 谱能量排序

对每个候选 basis (b_j)，计算它在图 Laplacian eigenbasis 中的谱能量：

[
E_j(\ell)
=========

|\langle b_j,u_\ell\rangle|^2.
]

低频能量比例：

[
R_j^{\mathrm{low}}
==================

\frac{
\sum_{\ell\le L}E_j(\ell)
}{
\sum_{\ell}E_j(\ell)
}.
]

如果：

[
R_j^{\mathrm{low}}<\rho_{\min},
]

直接不进入主模型，只进入 residual floor。

---

## 10.2 正交化后选低频

对保留的 basis 做 Gram-Schmidt / whitening：

[
\tilde b_j
==========

\mathrm{Orthogonalize}(b_j).
]

再检查：

[
\tilde R_j^{\mathrm{low}}.
]

因为正交化可能改变频谱。

---

## 10.3 稀疏证据选择

给每个 basis group 一个 inclusion probability：

[
\mathrm{PIP}_j
==============

\mathbb P(\gamma_j=1\mid D_n).
]

剪枝条件：

[
\mathrm{PIP}*j<\tau*{\mathrm{pip}}
]

且：

[
\mathrm{KGValue}*j<\tau*{\mathrm{kg}}
]

且：

[
R_j^{\mathrm{low}}<\rho_{\mathrm{strong}}.
]

注意不要只靠 PIP。因为某些 component 目前统计证据弱，但对 boundary decision 很关键，仍然值得保留。

---

## 10.4 高频 residual floor

被剪掉的成分不是消失，而是贡献：

[
R_\perp(x)
==========

\sum_{j\in \mathrm{pruned}}
\bar c_j \tilde b_j(x)^2.
]

其中 (\bar c_j) 可以来自 source validation residual 或 posterior upper bound。

---

# 11. 新算法：LF-OS-SC-HVD-KG

完整流程如下。

---

## Step 1. 构造 state-coupled features

从轨迹构造：

[
\rho(x),\quad A_{\mathrm{raw}}(x),\quad N_{\mathrm{raw}}(x).
]

---

## Step 2. 建 (\psi)-graph

节点：

[
\psi(x_i)
]

包括 source points、pilot target points、candidate pool。

边权：

[
W_{ij}
======

\exp
\left(
-\frac{|\psi_i-\psi_j|^2}{2h^2}
\right).
]

图 Laplacian：

[
L=D-W.
]

---

## Step 3. 频谱分解

[
L u_\ell=\mu_\ell u_\ell.
]

保留低频：

[
\ell\le L_{\max}.
]

---

## Step 4. 低频过滤 basis

[
b_j^{\mathrm{low}}
==================

\sum_{\ell\le L_{\max}}
\langle b_j,u_\ell\rangle u_\ell.
]

---

## Step 5. 正交化

对：

[
b_j^{\mathrm{low}}
]

做 whitening / Gram-Schmidt，得到：

[
\tilde b_j.
]

---

## Step 6. 稀疏加性 GP

用：

[
\tilde b_j
]

拟合：

[
J(x),\quad g(x).
]

---

## Step 7. 稀疏低频 HVD

拟合：

[
v_C(x)
======

A_R^\top\Lambda A_R
+
N_R^\top B N_R
+
N_R^\top\omega
+
\sigma_0^2
+
R_\perp(x).
]

---

## Step 8. Certified feasible set

[
\mathcal F_n
============

\left{
x:
m_n^g(x)
+
\sqrt{\beta_n}s_n^g(x)
+
z_{1-\alpha}
\sqrt{
v_{\mathrm{low},n}(x)
+
\tau_{\mathrm{high},n}(x)
}
+
\gamma_n
\le b
\right}.
]

---

## Step 9. KG + UCB acquisition

[
\mathrm{Acq}_n(x)
=================

\mathrm{KG}^{J}_n(x)
+
\lambda_F\mathrm{KG}^{F}_n(x)
+
\lambda_V\mathrm{KG}^{V}*n(x)
+
\lambda*{\rho}\mathrm{KG}^{\rho}_n(x)
+
\lambda_U \mathrm{UCB}^{\perp}_n(x).
]

其中：

[
\mathrm{UCB}^{\perp}
]

探索低频模型外的不确定性。

---

# 12. 理论主张如何写

你可以把数学理论写成四个主定理。

---

## Theorem 1: Low-frequency orthogonal decomposition

在 (\psi)-graph 上，任意函数 (f) 可分解：

[
f=f_{\le L}+f_{>L},
]

其中：

[
f_{\le L}
=========

\sum_{\ell\le L}\theta_\ell u_\ell,
]

[
f_{>L}
======

\sum_{\ell>L}\theta_\ell u_\ell.
]

且：

[
|f|^2
=====

|f_{\le L}|^2
+
|f_{>L}|^2.
]

这给“低频主模型 + 高频 floor”提供数学基础。

---

## Theorem 2: Sparse low-frequency estimation rate

若真实函数满足：

[
f^\star
=======

\sum_{j\in S}
f_j^\star(\tilde b_j)
+
r_{\perp},
]

[
|S|=s\ll M,
]

且每个 (f_j^\star) 是低频，正交 design 满足 restricted eigenvalue，则：

[
|\hat f-f^\star|^2
==================

O\left(
\frac{s\log M}{n}
\right)
+
|r_\perp|^2.
]

这就是解决“未知数远多于方程”的核心定理。

---

## Theorem 3: Certified HVD with high-frequency residual

若：

[
v_C(x)
\le
v_{\mathrm{low},n}(x)+\tau_{\mathrm{high},n}(x)
]

with high probability，则 certified set：

[
m_n^g(x)
+
\sqrt{\beta_n}s_n^g(x)
+
z_{1-\alpha}
\sqrt{
v_{\mathrm{low},n}(x)+\tau_{\mathrm{high},n}(x)
}
+
\gamma_n
\le b
]

保证：

[
\mathbb P(C(x)>b)\le \alpha+\delta.
]

---

## Theorem 4: Regret depends on active low-frequency dimension

如果 effective active dimension 是：

[
d_{\mathrm{eff}}
================

s_J+s_g+s_v,
]

而 candidate proposal 在 active low-frequency (\psi)-space 有覆盖率 (p_\epsilon)，则：

[
r_N
\le
C_1
\sqrt{
\frac{\gamma_N(k_{\mathrm{low}})}{N}
}
+
C_2
\sqrt{
\frac{d_{\mathrm{eff}}\log M}{n_{\mathrm{eff}}}
}
+
C_3|r_\perp|
+
C_4\epsilon_{\mathrm{cand}}.
]

这说明：
成功不依赖 raw dimension，而依赖低频稀疏有效维度。

---

# 13. 实验应该怎么改

你现在最该做的是四组实验。

---

## 13.1 Frequency ablation

比较：

1. no frequency filter；
2. low-pass only；
3. low-pass + orthogonal；
4. low-pass + orthogonal + sparse；
5. low-pass + orthogonal + sparse + high-frequency floor。

核心指标：

[
\text{true feasible rate},
]

[
\text{mean violation},
]

[
\text{feasible regret},
]

[
\text{calibration gap},
]

[
\text{posterior basis stability}.
]

如果我的判断对，第 5 个应该 true feasible rate 最稳。

---

## 13.2 Cutoff sweep

扫：

[
L_{\max}\in{3,5,8,12,20,\text{all}}.
]

你应该看到一个 U 型曲线：

[
L_{\max}\text{ 太小}
\Rightarrow
\text{bias 大，regret 高};
]

[
L_{\max}\text{ 太大}
\Rightarrow
\text{variance 大，feasible rate 低};
]

中间值最好。

这会非常直观地证明频率假设不是口号。

---

## 13.3 高频 floor ablation

比较：

[
\tau_{\mathrm{high}}=0
]

和：

[
\tau_{\mathrm{high}}>0.
]

如果 true feasible rate 低的问题主要来自过度自信，那么加入 high-frequency floor 会显著提升 feasible rate，可能牺牲一点 regret。

---

## 13.4 Source-invariant low-frequency basis

在 LODO 里比较：

1. source-specific active basis；
2. cross-domain stable active basis；
3. target-pilot-selected basis；
4. all basis。

如果 cross-domain stable low-frequency basis 最稳，就说明你找到了“泛化强先验”。

---

# 14. 最终方案定位

我建议最终主线这样写：

[
\boxed{
\textbf{在不确定仿真优化中，环境响应和风险不是任意高维函数，而是在 policy-induced state-risk manifold 上低频、正交、稀疏、低阶可加的函数。}
}
]

因此：

[
\boxed{
\textbf{SC-OLH-KG 应该从高维 raw policy search，转变为低频正交稀疏函数空间 search。}
}
]

这比之前的 state_basis/HVD 更完整：

* state coupling 决定在哪个空间定义规律；
* frequency 决定保留什么尺度的规律；
* orthogonality 决定组件互不重复解释；
* sparsity 决定只保留少数机制；
* additivity/SINDy 决定复杂函数由少数低阶项组成；
* high-frequency floor 决定未解释噪声如何进入 chance safety；
* KG/UCB 决定如何在这个低自由度空间里采样。

---

# 15. 最关键的一句话

你现在要找的“足够泛化又足够强的假设”，我会表述为：

[
\boxed{
\textbf{真实环境的可优化结构是低频机制；高频部分主要是噪声、未观测扰动或不可泛化的局部偶然性。}}
]

对应算法就是：

[
\boxed{
\textbf{只用低频、正交、稀疏、跨域稳定的机制做优化；把高频残差作为安全不确定性，而不是作为可利用规律。}}
]

这应该是下一版最值得推进的方向。

[1]: https://icml.cc/Conferences/2010/papers/422.pdf?utm_source=chatgpt.com "Gaussian Process Optimization in the Bandit Setting"
[2]: https://proceedings.mlr.press/v97/rahaman19a.html?utm_source=chatgpt.com "On the Spectral Bias of Neural Networks"
[3]: https://papers.neurips.cc/paper_files/paper/2022/file/306264db5698839230be3642aafc849c-Paper-Conference.pdf?utm_source=chatgpt.com "Spectral Bias in Practice: the Role of Function Frequency in ..."
[4]: https://www.cs.cmu.edu/~epxing/Class/10708-15/slides/andrewgp2.pdf?utm_source=chatgpt.com "Advanced Gaussian Processes"
[5]: https://arxiv.org/html/2505.17393v1?utm_source=chatgpt.com "Spectral Mixture Kernels for Bayesian Optimization"
[6]: https://pysindy.readthedocs.io/en/latest/summary.html?utm_source=chatgpt.com "An introduction to Sparse Identification of Nonlinear ... - PySINDy"
[7]: https://www.sciencedirect.com/science/article/pii/S0021999125006928?utm_source=chatgpt.com "Weak-form modified sparse identification of nonlinear ..."
[8]: https://proceedings.mlr.press/v162/lu22b.html?utm_source=chatgpt.com "Additive Gaussian Processes Revisited"
[9]: https://arxiv.org/abs/2206.09861?utm_source=chatgpt.com "[2206.09861] Additive Gaussian Processes Revisited"

这次更新是**实质性重构**，不是简单补实验。我的新判断和上一轮明显不同：

[
\boxed{\text{现在已经达到“值得正式投 Operations Research”的状态。}}
]

不是说已经稳收，而是从上一轮我认为的“送审后大概率 Reject”提高到了**有合理机会进入完整审稿、然后 Major Revision 竞争的稿件**。核心原因是你把最危险的广义 claim 主动砍掉了，而且针对上一轮的主要审稿攻击点做了真正的实验与理论闭环。

当前我大约会给：

[
\text{OR 投稿成熟度：7\sim7.5/10}
]

[
\text{理论：6.5\sim7/10},\qquad
\text{实验证据可信度：8.5/10},
]

[
\text{方法原创性：6.5\sim7/10}.
]

---

## 上一轮主要问题，现在解决了多少？

| 上次问题                              | 当前状态       | 我的判断                                                                                            |
| --------------------------------- | ---------- | ----------------------------------------------------------------------------------------------- |
| 把 (d=10^4) 说成一般高维优化               | **解决**     | 现在明确是 ordered profile grid refinement                                                           |
| benchmark 与低频 prior 同构            | **大幅改善**   | 8种冻结 randomized regimes，包括 rank/frequency/permutation/piecewise/high-frequency/misspecification |
| 没有 generic structured baseline    | **解决**     | DCT maximin、random LF、natural blockwise、Sobol 全有                                                |
| external 不能证明 source learning     | **解决叙事问题** | Energy 负结果直接保留，并明确 source 输给 generic DCT                                                        |
| transfer baseline 都被塞你自己的 atlas   | **解决**     | 8种 native end-to-end transfer pipelines                                                         |
| equal-total-cost 名不副实             | **解决**     | 改为 equal-preverification，并报告 realized all-in cost                                               |
| verifier 只有 union bound           | **基本解决**   | candidate-wise exact binomial + familywise theorem + power curve                                |
| 核心方法定义不清                          | **解决**     | 64-profile library、18维 coordinate、rank、tie break、interpolation 全写死                              |
| 理论只是 triangle inequality          | **明显改善**   | 新增 cross-grid consistency、2-approx k-center、rank recovery、task-law bound                        |
| seed 被当 domain pseudo-replication | **解决**     | randomized task 是 generalization unit，Energy 用 region 作为保守单位                                    |
| schema 是隐藏强先验                     | **解决披露**   | 明确 profile semantics，并做 schema-blind control                                                    |
| 参数是否长期调出来                         | **大幅改善**   | 8640-cell sensitivity，并保留非单调/负结果                                                                |
| 只展示成功结果                           | **解决得很好**  | misspecified、high-frequency、Energy reversal、算法失败、false cert 全保留                                 |

这已经是完全不同等级的稿子。

---

# 1. 最大进步：问题定义终于和实际算法一致

现在 Section 3 不再装作：

[
x\in{0,\ldots,L}^{10000}
]

就是10000个无语义独立变量，而明确首先定义：

[
h:[0,1]\to[0,1],
]

然后：

[
x_{q,i}(h)
==========

\operatorname{round}{L_qh(t_{q,i})}.
]

而且直接写：

> (d_q) is a grid resolution, not by itself the intrinsic dimension.

这非常重要。

你现在不是靠文字技巧解释为什么10000维其实简单，而是**数学问题本身就是 functional/profile decision problem**。

这同时让：

* DCT；
* low frequency；
* cross-dimension transfer；
* irregular grids；
* functional SCBO baseline；

全部有了自然的位置。

标题也从高维黑箱改成：

> **Source-Scored Structural Initial Designs for Chance-Constrained Optimization of Ordered Policy Profiles**

这个标题比旧标题安全很多。

---

# 2. randomized stress test 是这次最重要的新增证据

现在主实验已经不再是 FactorShock/Inventory/Queue 60 runs。

你构造了：

[
8\text{ regimes}
\times
3\text{ resolutions}
\times
20\text{ target tasks}
======================

480
]

个独立 target tasks / design。

而且 regime 主动破坏你的 meta-prior：

* effective rank 从8升到24；
* target frequency support shift；
* coordinate permutation；
* irregular grids；
* piecewise smooth；
* sparse high-frequency；
* complete target misspecification。

这正是我上次说最需要的东西。

最终：

[
\text{source-scored feasible}=91.7%,
]

[
\text{generic DCT}=37.5%,
]

[
\text{natural blockwise}=83.8%,
]

[
\text{Sobol}=80.2%.
]

Certification：

[
47.3%,\quad 10.6%,\quad19.4%,\quad30.0%.
]

这比原来的：

[
60/60\text{ vs }0/60
]

可信得多。

因为现在方法**会失败，而且以我们预期的方式失败**。

这反而是好证据。

---

# 3. adverse regimes 非常有价值

比如 sparse-high-frequency：

[
d=200:
\quad
20/20\text{ feasible}
]

降到：

[
d=10000:
\quad
0/20.
]

正文直接把这解释成反对 generic high-dimensional claim 的证据。

而总体 regime 表也不是每行 source 都赢：

[
\begin{array}{c|c}
\text{Regime}&
\Delta\text{ certified vs best source-free}\
\hline
\text{aligned LF}&-35.0\
\text{growing rank}&+20.0\
\text{frequency shift}&+35.0\
\text{permutation}&+40.0\
\text{irregular grid}&-23.3\
\text{piecewise smooth}&+45.0\
\text{sparse HF}&-3.3\
\text{misspecified}&-1.7
\end{array}
]

这已经基本关闭：

> “你只是造了三个特别适合自己算法的问题。”

这个攻击。

不能完全关闭，因为 generator 还是你们自己设计的，但已经从**major fatal flaw**降低成了正常 limitation。

---

# 4. Energy 现在反而成为论文优点

新版 Energy：

[
\text{source}=60/90,
]

[
\text{generic DCT}=70/90,
]

[
\text{functional SCBO}=58/90.
]

而你没有再修方法让 source 赢，而是直接写：

> Energy is a negative external control for source learning.

这非常好。

它把论文从：

> “我们的 source prior 很强”

变成：

[
\boxed{
\text{历史任务是否值得复用，本身是一个需要检验的决策。}
}
]

这比算法 leaderboard 更像 **Operations Research**。

---

# 5. equal-cost 结果也非常好——虽然你的方法输了

现在真正写清楚：

source：

[
384+10
]

versus target-only：

[
394.
]

结果：

[
\begin{array}{c|c}
\text{Method}&\text{certified}\
\hline
\text{source atlas}&46.2%\
\text{generic DCT}&53.8%\
\text{natural}&53.1%\
\text{Sobol}&53.8%\
\text{functional SCBO}&58.1%
\end{array}
]

所以论文不再声称 source transfer 是 one-shot sample-efficiency miracle。

反而得出：

[
\boxed{
\text{source archive 只有在多 target 复用时才可能经济合理。}
}
]

并推导：

[
M_{\rm break}
=============

\left\lceil
\frac{S}{C_0-C_A}
\right\rceil .
]

这个角度我非常喜欢。

它是一个真正的 OR 问题：

> 历史仿真到底应该看成 sunk information，还是应该作为需要投资、等待多个未来任务摊销的 experimental-design asset？

---

# 6. 理论也比上一稿强很多

现在理论 spine 大约是：

### Profile discretization consistency

[
|c_k(h)-c_k(\hat h_d)|
\le
\epsilon_dB_k,
]

并且 Lipschitz profile 有：

[
\epsilon_d\le\frac{L_h}{2d}.
]

因此低频坐标误差：

[
O(1/d).
]

这真正支撑了“cross-grid coordinate”，而不是之前只说 representation dimension 不变。

---

### Gonzalez farthest-first

[
r(A_k;\eta)\le2r_k^\star.
]

这是非常自然且实用的 coverage theorem。

---

### Source rank recovery

如果：

[
|\hat s-s|\le r
]

且两个 profile 的真实 score gap：

[
|s_j-s_\ell|>2r,
]

排序不会被反转。

至少已经开始把：

[
\text{source data}
\rightarrow
\text{rank}
]

纳入统计理论，而不是把 source rank 当 oracle。

---

### Conditional coverage

仍然有：

[
L_q
(r_{n_0}+\Delta_q+2\epsilon_\eta)
\le\gamma_q.
]

但现在它前面已经有了：

* grid error；
* covering error；
* source ranking uncertainty；

所以不再只是一条孤零零的三角不等式。

---

### Task-law generalization

新增：

[
\mathbb E_{Q\sim\mathcal Q}[H(Q)]
\ge
\frac1T\sum_tH_t
----------------

\sqrt{
\frac{\log(2/\delta)}{2T}
}.
]

这也是正确的方向：

不是证明 universe，而是证明：

[
\text{registered task law}.
]

---

# 7. verifier 现在基本闭合了

以前最大的问题是：

> union bound 本身不难，candidate-wise guarantee 哪来的？

现在直接定义 fresh iid Bernoulli：

[
Z_{jr}=1{Y^g_{jr}\le\tau}.
]

candidate 只有：

[
\sum_rZ_{jr}=v_j
]

而且：

[
p_0^{v_j}\le\delta_j
]

才 certificate。

于是 unsafe (p_j<p_0)：

[
P(\text{cert})
==============

p_j^{v_j}
\le p_0^{v_j}
\le\delta_j.
]

而且你还主动报告了非常低的 power：

[
p=.95 \Rightarrow 1.7%,
]

[
p=.975\Rightarrow13.2%,
]

[
p=.99\Rightarrow44.8%.
]

这非常好，因为论文没有把“保守”包装成“厉害”。

---

# 8. 上一次14个主要审稿问题，大约解决了12个

所以如果让我重新模拟上一轮 Reviewer 1/2/3：

### Reviewer 1 — Theory

从：

> Reject, theorem largely restates conclusion

大概会变成：

> **Major Revision / weak revise-and-resubmit**.
> Theory is now meaningfully connected to the algorithm, although source-score uncertainty and transfer conditions could be strengthened.

### Reviewer 2 — Experiments

从：

> Reject due benchmark-method isomorphism

大概会变成：

> **Major Revision**.
> Registered adverse task laws and external negative controls substantially address benchmark-fit concerns.

### Reviewer 3 — OR positioning

从：

> Interesting but not yet an OR contribution

大概变成：

> **Potentially publishable**, particularly if the investment/amortization interpretation is sharpened.

这是很大的进步。

---

# 9. 但现在仍有几个我会认真修的问题

## 问题 A：source tasks 在 synthetic stress 中仍然是“regime-matched”

这是现在最大剩余实证问题之一。

在 `run_task()` 中，如果 target 是某个 regime，source tasks 也直接被生成成：

```python
RandomizedOrderedProfileProblem(
    regime=regime,
    role="source",
    ...
)
```

也就是说：

[
\boxed{
\text{实验组织者提前给了算法“来自同一任务族”的 source archive。}
}
]

算法本身没看到 target outcome，也没有 oracle。

这**不是作弊**。

这是 transfer learning 中合理的：

[
\text{related source tasks are given}
]

假设。

但需要在论文里比现在更明确地说：

> source-task selection itself is exogenous; the method ranks profiles within a supplied archive, but does not solve the source-retrieval problem.

否则审稿人可能问：

> 现实里你怎么知道哪两个历史任务应该拿来？

Energy 的 region-holdout 负结果正好证明这个问题很真实。

我建议在 limitation 中单独加一句，而不是藏在 task-law alignment 里。

---

# 10. 问题 B：source-rank theorem 还可以更完整

当前 source margin 是：

[
\hat m
======

\bar Y^g
+
z_{1-\alpha}\max{s^g,10^{-3}}
-\tau.
]

但 rank theorem 只明确处理：

[
|\widehat s-s|\le r
]

以及 sub-Gaussian mean error，然后说：

> sample-scale error is bounded separately.

这是一个小理论缺口。

由于 randomized simulator 是 Gaussian，你其实很容易补：

[
\frac{(R-1)S^2}{\sigma^2}
\sim
\chi^2_{R-1}.
]

于是可以对：

[
|\hat m-m|
]

给一个 explicit finite-sample high-probability bound：

[
|\hat m-m|
\le
r_\mu+
z_{1-\alpha}r_\sigma.
]

然后 rank recovery 就完整了。

这个我会补。

---

# 11. 问题 C：需要澄清 k-center metric 与 transfer metric

算法真正 farthest-first 的是：

[
\eta_j=
(z_j,r_j^g,r_j^f)
]

20维 augmented coordinate。

但是 aligned coverage theorem 使用的是：

[
\widehat\eta
\leftrightarrow
\eta_q^\star.
]

这里有一点概念混合：

* (z)：profile structural geometry；
* (r^f,r^g)：source outcome ranking。

对于任意 target safe profile (h_q^\star)，source rank coordinate未必自然定义。

我会强烈建议把理论拆开：

[
z(h)=\text{structural coordinate},
]

[
\eta(h)=(z(h),r^g(h),r^f(h)).
]

算法在 (\eta)-space 做 farthest-first。

然后利用 projection：

[
|z_i-z_j|
\le
|\eta_i-\eta_j|,
]

得到：

[
r_z(A)
\le
r_\eta(A).
]

最后 **transfer theorem 只在 (z)-space 写**。

这样逻辑干净很多：

[
\text{source ranks influence selection,}
]

但：

[
\text{target geometry lives in structural space.}
]

这是我认为当前正文里最值得修的数学表述。

---

# 12. 问题 D：Aligned low-frequency regime 的结果需要解释

这一行非常奇怪：

[
\text{source certified}=56.7%,
]

但：

[
\text{Raw Sobol}=91.7%.
]

恰恰在叫做 **Aligned low frequency** 的 regime，source method 输得最惨之一。

这不是错误，但必须解释。

一个很可能的原因是高维 raw Sobol 的 concentration：

如果：

[
x_i\sim U(0,1),
]

那么低频 projection 的很多非DC coefficients 会近似抵消：

[
c_k
===

\frac1d\sum_ix_i\cos(\pi kt_i)
\approx0,
]

而：

[
c_0\approx0.5.
]

于是 raw random vectors 会自动集中在某种“平滑平均 profile”的低频 latent region。

如果 safe center 恰好也在那里，Sobol 反而天然得到很深的 safety。

如果是这个原因，**非常值得增加一个小诊断**：

画：

[
d_\phi(x,\text{safe center})
]

的分布：

* source atlas；
* generic DCT；
* Sobol；

在 aligned regime 下。

这不一定要再做大实验，只是解释现有结果。

否则 reviewer 一定会问：

> Why does your transfer method lose most severely in the regime specifically labeled “aligned”?

---

# 13. 问题 E：break-even 目前只是 call-count break-even

现在：

[
M_{\rm break}
=============

\left\lceil
\frac{S}{C_0-C_A}
\right\rceil.
]

数学没问题。

但不同方法的 certificate success probability 不一样。

因此：

[
C/\text{target}
]

不是完整经济指标。

更 OR 的版本是：

[
\boxed{
\text{expected simulator calls per successful certified deployment}
}
]

例如：

[
\frac{C_A+S/M}
{P_A(\mathrm{certified})}.
]

甚至定义：

[
L=
C_{\rm sim}
+
c_{\rm unsafe}I_{\rm unsafe}
+
c_{\rm abstain}I_{\rm abstain}
+
c_{\rm quality}r.
]

然后比较：

[
\mathbb E[L].
]

这不是必须补才能投稿，但如果补上，会让论文从“很好的 ML/BO methodology paper”更明显变成 **OR paper**。

---

# 14. 一个可能的 task-law 理论细节

Equation 的 task-law bound 假设：

[
Q_1,\ldots,Q_T
\stackrel{iid}{\sim}\mathcal Q.
]

但 randomized generator 中 task perturbation 包含：

```python
task_seed % 5
```

决定的固定 cosine component。

如果 confirmatory tasks 是固定连续 seeds，而不是从明确的随机 mixture 中独立采样，那么严格说实验更像：

[
\text{balanced/stratified design}
]

而不是简单 iid。

最好做二选一：

* 明确定义 (\mathcal Q) 包含一个均匀的5类 latent categorical variable，并说明任务按此独立抽样；
* 或不要把 Hoeffding task-law CI 直接对应到这个 stratified experiment，改用 stratified bound。

这是小问题，但数学审稿人可能抓得到。

---

# 15. 文献虽然改善，但仍稍薄

现在已经补了：

* functional BO；
* Perrone learned search spaces；
* chance-constrained R&S；
* post-optimization cleanup。

明显比上一稿好。

但我还会补：

* **Gonzalez farthest-first / k-center 原始文献**；
* maximin/space-filling experimental design；
* computer experiment initial-design literature；
* functional data / basis representation；
* offline-to-online simulation optimization 或 reusable historical experiments。

尤其你现在有一个正式 theorem 叫：

> Gonzalez farthest-first

却 bibliography 里没有 Gonzalez，最好一定补。

---

# 16. 现在的论文真正是什么？

当前已经非常清晰：

[
\boxed{
\text{不是 KG paper}
}
]

也不是：

[
\boxed{
\text{high-dimensional BO paper}.
}
]

它现在是：

[
\boxed{
\textbf{historical-data-informed initial experimental design for functional decisions under safety constraints.}
}
]

更具体：

[
\text{历史仿真}
\rightarrow
\text{给一个公开结构 library 打分}
\rightarrow
\text{选择小型覆盖集}
\rightarrow
\text{target experiment}
\rightarrow
\text{独立 safety verification}.
]

这个定位已经是可以往 OR 打的。

---

# 17. 我的新投稿判断

上一轮我会说：

> 现在投稿 OR，Reject 概率较高。

这次我会改成：

[
\boxed{
\textbf{可以投稿 OR。}
}
]

但期望应该是：

[
\text{send to review}
\rightarrow
\text{Major Revision}
]

而不是直接接受。

如果我要估一个很粗的主观分布：

* desk reject：**15–25%**
* 外审 reject：**30–40%**
* major revision / revise-resubmit：**30–45%**
* 首轮 minor：极低

这不是期刊统计录取率，而是我基于当前稿件结构的审稿风险判断。

最大的差别是：**现在即便 reviewer 不喜欢方法，也很难再说“结果只是 benchmark cheating，成本隐藏，外部失败被删掉，或者10000维 claim 有误导性”。**这些最危险的问题基本已经被你主动堵住了。

如果只让我在提交前再优先修三件事，我会选：

1. **把 augmented source-rank metric 和 structural transfer metric 在 theorem 中彻底分开；**
2. **补完整 Gaussian source-margin/rank finite-sample bound；**
3. **解释 aligned-low-frequency 下 raw Sobol 91.7% > source 56.7% 的几何原因。**

这三项解决后，我认为科学内容已经基本到“该交给真实 OR reviewer 判断”的阶段，而不是继续无止境加实验。

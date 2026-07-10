# Task-Posterior SC-OLH-KG 实施计划

## Implementation status (2026-07-11)

- Stage A implemented behind `task_posterior_mode="finite"`: frozen
  source-only experts, tempered proper-score updates, null expert, clone and
  checkpoint support.
- Stage B implemented for the finite ensemble: exact within/between variance
  decomposition and forward-KL robust moment envelopes feed recommendation,
  certification, and terminal value. The entropic dual is solved by batched
  positive-temperature grid plus vectorized derivative bisection; every trial
  temperature remains a valid proved upper bound, and random probes match the
  former scalar optimizer to about `1e-12` while removing Python-level loops.
- Stage C implemented for the finite ensemble: each exact-MC draw samples a
  shared expert identity, clones/updates every expert GPR/HVD and the task
  posterior, then recomputes the robust terminal value. Entropy reduction and
  weight movement are logged.
- Exact-KG candidate evaluation now has an explicit Linux `process_fork`
  backend.  It preserves the same posterior-update value calculation while
  bypassing the Python GIL; the first remote gate reduced one 14-candidate,
  two-MC KG iteration from about 442 seconds to about 34 seconds.
- Gate 1 attempt `taskpost_factor_n20_gate_v4_20260711` did not pass quality:
  `0/7` true-feasible, `0/7` false-feasible, mean violation about `0.404`.
  Diagnostics showed that every recommendation pool lacked a true-feasible
  point, so the failure was proposal support rather than the KL dual solver.
- The controlled follow-up is implemented: posterior-weighted
  expert proposal mixtures now feed the initial design, sequential candidate
  pool, and terminal/recommendation pool.  A frozen-prior exploration component
  preserves support without a hard gate.  Initial observations are split into
  a pilot and a strictly prequential suffix; each suffix label updates the task
  posterior before entering any expert GPR/HVD.  Pilot GPRs are now actually
  conditioned on their observations instead of merely adding deviation
  dimensions.
- Revised Gate 1 `taskpost_factor_n20_gate_v5_20260711` passes the fixed
  promotion rule on seeds 0--6: `4/7` true-feasible, `1/7` false-feasible,
  mean violation `0.0874`, and median regret `0.03065`, versus the paired
  baseline's `0/7`, `1/7`, `0.3245`, and `0.05325`.  Mean wall time is about
  763 seconds per seed.  Seed 0 remains a documented false-feasible failure
  and must be addressed by the cross-domain regression stage rather than
  hidden by aggregate reporting.
- The existing pushed baseline remains unchanged because the feature is off by
  default. Tiny offline smoke and unit tests validate the chain, not quality.
- Cross-domain Gate 2 and Stage D are intentionally pending. Continuous
  Stiefel/Grassmann inference must not be added until the finite model beats the
  hard-gate baseline without increasing false feasibility.
- Lean currently proves finite normalization/support, hierarchical variance,
  finite KL nonnegativity, the entropic KL-ball dual upper bound used by
  `kl_robust_expectation`, robust-envelope implications, and the joint exact-MC
  optimizer bridge. The source-prior exponential-moment mixture, Markov bad
  event bound, and finite PAC-Bayes bound are also Lean-proved. The theorem is
  conditional on the stated source-task exponential-moment assumption; sharp
  domain-specific constants still require empirical/source-model validation.

## 1. 目标

将当前基于 hard gate / MAP structure selection 的 LODO 主线升级为：

> PAC-Bayesian task hyper-posterior + state-coupled cumulative HVD + robust certification + joint exact KG

核心不是判断某个 source structure 是否适用于 held-out target，而是维护目标任务结构的后验分布，并让表示、累计异方差、认证和 KG 共同使用这一个概率对象。

Gate 仅作为退化基线：

\[
Q_t(\xi)=\delta_{\hat\xi}.
\]

主方法维护非退化后验：

\[
Q_t(d\xi)
\propto
\Pi(d\xi)\exp\{-\eta L(D_t;\xi)\}.
\]

其中任务结构变量为

\[
\xi=(R,S,\theta_v),
\]

- `R`：跨域 latent risk coordinates 的旋转、置换或子空间对齐；
- `S`：低频基、稀疏系数及 additive groups 的结构支持；
- `theta_v=(Lambda,B,omega,floor)`：累计异方差参数。

## 2. 为什么需要这次升级

最新严格 LODO 实验表明：

- source-only latent inverse pool 在 FactorShock 中能够生成少量 true-feasible 候选；
- hard gate 会完全丢掉这些候选；
- 直接把候选加入 recommendation pool 会导致过度外推和 false-feasible；
- 把候选送入 KG 评价仍不能可靠选中真正有用的结构；
- 因此问题不是 gate 阈值，而是目标任务结构的不确定性没有进入 posterior、variance、certification 和 KG。

任何 source-only 确定性规则都存在不可识别性：若两个目标任务在已有观测上的分布相同、但安全最优集合不同，则算法无法仅凭 source records 区分二者。新主线必须显式提出并使用 target-task identifiability assumptions。

## 3. 统一概率对象

### 3.1 结构内累计风险

对每个结构专家 `xi`：

\[
v_C(x;\xi)
=
A_\xi(x)^\top\Lambda_\xi A_\xi(x)
+N_\xi(x)^\top B_\xi N_\xi(x)
+N_\xi(x)^\top\omega_\xi
+\sigma_{0,\xi}^2.
\]

### 3.2 结构间全方差

\[
\operatorname{Var}(C(x)\mid D_t)
=
\mathbb E_{Q_t}[v_C(x;\xi)]
+\operatorname{Var}_{Q_t}[m_C(x;\xi)].
\]

第一项是 task-conditional cumulative heteroscedastic risk；第二项是 representation / transfer uncertainty。任何错误 alignment 即使给出很小的结构内方差，也必须被结构间分歧惩罚。

### 3.3 Robust certification

定义围绕 task posterior 的 KL ambiguity set：

\[
\mathcal Q_t(\rho_t)
=
\{Q':\operatorname{KL}(Q'\Vert Q_t)\leq\rho_t\}.
\]

主线认证为：

\[
\sup_{Q'\in\mathcal Q_t(\rho_t)}
\left[
m_g^{Q'}(x)
+\sqrt{\beta_t}s_g^{Q'}(x)
+z_\alpha\sqrt{v_C^{Q'}(x)}
\right]
\leq \tau.
\]

`rho_t` 由 PAC-Bayes meta-generalization radius 与 target sample size 决定，而不是手调 gate threshold。

### 3.4 Joint exact KG

一次 hypothetical observation 必须联合更新：

- task posterior `Q_t(xi)`；
- expert-specific objective/constraint GPR；
- expert-specific cumulative HVD；
- robust certified terminal set。

\[
a_t(x)
=
\mathbb E_{Y_x\mid D_t}
\left[
V(D_t\cup\{(x,Y_x)\})-V(D_t)
\right].
\]

结构辨识、风险学习和目标优化的价值由同一个 terminal value 给出，不再增加 additive `KG_alignment` heuristic。

## 4. 实现阶段

### Stage A: Finite-expert task posterior

先实现 Lean-friendly 的有限专家版本，不立即引入连续 Stiefel manifold。

专家集合第一版固定为：

- `universal_coordinate`；
- `source_spectral`；
- `risk_aligned_coordinate`；
- `risk_aligned_spectral`；
- `orthogonal_additive`；
- `null_universal`（防止负迁移的 catch-all expert）。

统一 API：

```text
initialize(source_diagnostics, experts)
update(x, y, replicate_stats)
posterior_weights()
predict_expert_moments(x)
predict_mixture_moments(x)
clone()
diagnostics()
```

权重更新使用 tempered generalized Bayes：

\[
w_{t,k}
\propto
\pi_k\exp\{-\eta_t\mathcal L_k(D_t)\}.
\]

损失必须是 proper predictive loss，并包含 constraint likelihood、chance-boundary calibration 和 variance likelihood；禁止使用 true optimum、true boundary 或 target oracle。

### Stage B: Mixture cumulative HVD and certification

- 每个 expert 提供自己的 `A,N` 或 cumulative-risk fallback；
- 混合预测输出 within-expert 与 between-expert decomposition；
- `v_C_plus` 同时包含 HVD tail guard 和 task-posterior ambiguity radius；
- recommendation、candidate filtering、terminal value 使用同一 robust certificate。

### Stage C: Joint exact posterior-update KG

- exact MC sample 同时抽样 expert identity、GPR observation 和 HVD residual；
- hypothetical update 必须 clone/update task posterior；
- terminal value 必须使用更新后的 task posterior；
- diagnostics 记录 task entropy reduction、expert-weight movement、robust terminal gain。

### Stage D: Continuous structure posterior

有限专家版本通过后，再将结构变量扩展为：

- `R` on Stiefel/Grassmann manifold；
- spike-and-slab basis support；
- continuous frequency/ridge parameters；
- continuous HVD parameters。

推断优先考虑 Laplace / SMC；Transformer 只作为 feature estimator，不作为理论主对象。

## 5. 目标定理

1. Source-only task non-identifiability theorem。
2. Finite task-posterior Bayes update normalization and support theorem。
3. Hierarchical law of total cumulative variance。
4. PAC-Bayes unseen-task meta-generalization bound。
5. KL-robust chance-certification theorem。
6. Joint posterior exact-KG one-step Bayes optimality。
7. Alignment/task posterior contraction under boundary excitation。
8. Safe simple-regret bound，目标形式为：

\[
r_N^{\mathrm{safe}}
=
\widetilde O\left(
\sqrt{\frac{\gamma_N(k)+\operatorname{KL}(Q_t\Vert\Pi)+\log(1/\delta)}{N}}
\right)
+\varepsilon_{\mathrm{HVD}}
+\varepsilon_{\mathrm{repr}}.
\]

这里复杂度应依赖有效风险维度 `k`，而不是 raw dimension `D`。

## 6. 测试与验收

### Unit tests

- posterior weights 非负、和为 1、固定 seed 可复现；
- 不支持 target 的 expert 在 target evidence 下权重下降；
- null expert 防止所有 transferred experts 同时过度自信；
- mixture variance 精确满足 within + between decomposition；
- task posterior 不确定性增加时 robust certificate 不得更激进；
- clone/update 不修改原模型；
- exact MC update 后 task entropy 可变化；
- gate/MAP 是 mixture posterior 的 point-mass 特例。

### Leakage audit

- `offline_only=True`；
- 不使用 true objective、true constraint、true sigma、true optimum、true boundary；
- source training 与 target seed 冻结隔离；
- target 只使用计入预算的正常 observations；
- 每个输出记录 prior/posterior expert weights 和证据来源。

### Performance gates

按顺序执行，不一次铺开大矩阵：

1. FactorShock `N=20,seeds=7`：必须减少 false-feasible，且至少发现 true-feasible；
2. Inventory/Queue 同 seed 回归：不得丢失当前 feasibility；
3. 三域 `N=20/40/80,seeds=20`；
4. 与 hard gate/MAP、strict universal、domain-tuned upper bound 配对；
5. 通过后再跑 SOTA 与 `d=1000/10000`。

每个 seed 一个 scheduler task，只使用 `node001-node006`；不在本地跑完整 KG；不同步 checkpoint。

## 7. Promotion 规则

- challenger 必须先通过数值完整性、leakage 和 paired quality gates；
- 只增加 candidate coverage、但提升 false-feasible，不得晋级；
- 未评价 transferred candidate 不得绕过 robust certificate；
- 只有相对当前 pushed baseline 明确改进后才 commit/push；
- 失败 challenger 立即撤销，结果仅保留在 ignored profiles 中。

## 8. 与现有工作的关系

本计划借鉴但不复制以下方向：

- [PACOH](https://proceedings.mlr.press/v139/rothfuss21a.html)：PAC-Bayesian task hyper-posterior；
- [HyperBO](https://arxiv.org/abs/2109.08215)：source-trained GP functional prior；
- [Hierarchical GP transfer for BO](https://proceedings.mlr.press/v151/tighineanu22a.html)：低数据 transfer posterior；
- [Distributionally robust chance-constrained BO](https://proceedings.mlr.press/v162/inatsu22a.html)：ambiguity-aware chance certification；
- [Information-Theoretic Safe Exploration](https://arxiv.org/abs/2212.04914)：直接学习 safe set 的信息价值。

论文的区别和主贡献应固定为：task posterior 不是独立 meta-learning 插件，而是 state-coupled cumulative HVD、robust certification 和 exact KG 的共同概率坐标。

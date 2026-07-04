# SC-OLH-KG Theory Skeleton

## Setup

Let `x in X` be a feasible integer design, `s(x)` a policy-state or occupancy
summary, and `C(x)` the random cumulative constraint cost.  The optimizer
seeks small scalar objective `f(x)` subject to

```text
P(C(x) <= tau) >= 1 - alpha.
```

For Gaussian or sub-Gaussian certification, this is enforced through a
conservative upper confidence condition of the form

```text
m_C(x) + sqrt(beta_C) s_C(x) + z_{1-alpha} sqrt(v_C^+(x)) <= tau.
```

Here `m_C,s_C` describe epistemic uncertainty in the constraint mean and
`v_C^+` is a conservative estimator of aleatoric cumulative variance.

## Assumption A1: Fixed-Trajectory Noise Model

For a fixed trajectory `T`, write the cumulative noise as

```text
epsilon_C(T) = sum_k a_k(T)^T xi_k + n(T)^T eta + r(T),
```

where `xi_k` are independent local shocks with diagonal covariance, `eta` is a
shared low-rank shock with covariance `B`, and `r(T)` is independent residual
noise with diagonal rate `omega`.

Define aggregated exposures

```text
A(T) = sum_k a_k(T),     N(T) = n(T).
```

## Theorem 1: Fixed-Trajectory Cumulative Variance Decomposition

Under A1 and mutual independence of local, shared, and residual shocks,

```text
Var(C | T) = A(T)^T Lambda A(T) + N(T)^T B N(T) + N(T)^T omega.
```

Proof sketch: expand the variance of the sum, cancel cross terms by
independence and zero mean, collect local diagonal terms into `Lambda`, shared
terms into `B`, and residual rates into `omega`.

## Assumption A2: Policy-State Occupancy Representation

Each design `x` induces a distribution over trajectories.  Its occupancy
summary `s(x)` determines expected exposures

```text
A(x) = E[A(T) | x],     N(x) = E[N(T) | x],
```

up to approximation error `delta_occ(x)`.

## Theorem 2: Random-Policy Occupancy-Risk Decomposition

Under A1-A2,

```text
Var(C | x)
  = A(x)^T Lambda A(x) + N(x)^T B N(x) + N(x)^T omega
    + R_occ(x),
```

where `|R_occ(x)|` is bounded by a Lipschitz constant times
`delta_occ(x)` plus within-policy exposure variance.

Proof sketch: apply the law of total variance over trajectories and use the
occupancy approximation to replace random exposures by their policy-state
means.

## Proposition 3: Information Refinement Reduces Apparent Aleatoric Variance

Let `F_0 subset F_1` be two information sigma-fields for the same simulation
output.  Then

```text
E[Var(Y | F_1)] <= E[Var(Y | F_0)].
```

Proof sketch: this is the law of total variance:
`Var(Y | F_0) = E[Var(Y | F_1) | F_0] + Var(E[Y | F_1] | F_0)`.

Interpretation: moving from raw design space to state/occupancy-conditioned
features can turn apparent noise into explainable structure.

## Theorem 4: Low-Rank Shared-Shock Truncation

Let `B = U diag(lambda_1,...,lambda_r) U^T` with descending eigenvalues.  If
`B_K` keeps the top `K` eigenpairs, then

```text
0 <= N^T(B - B_K)N <= ||N||_2^2 sum_{j>K} lambda_j.
```

Proof sketch: diagonalize `B - B_K` and bound the quadratic form by its trace
or spectral tail.

## Theorem 5: HVD Estimation Oracle Inequality

Let `phi(x)` be the cumulative-risk feature vector and suppose

```text
v_C(x) = phi(x)^T theta_* + e(x),     theta_* >= 0.
```

For ridge estimator `hat theta` fitted to residual-square observations with
bounded noise, with probability at least `1-delta`,

```text
E_n[(phi^T hat theta - v_C)^2]
  <= c1 inf_theta E_n[(phi^T theta - v_C)^2]
     + c2 ridge ||theta_*||_2^2
     + c3 complexity(phi,delta)/n.
```

Proof sketch: standard ridge basic inequality plus concentration for
sub-exponential residual-square errors.  The nonnegative projection can only
reduce squared error against a nonnegative feasible comparator in the projected
cone.

## Theorem 6: Conservative Chance Certification

Assume the posterior mean confidence event

```text
|m_C(x) - E[C(x)]| <= sqrt(beta_C) s_C(x)
```

holds uniformly over `X`, and `v_C^+(x) >= Var(C(x))`.  For Gaussian cumulative
noise, any design satisfying

```text
m_C(x) + sqrt(beta_C) s_C(x) + z_{1-alpha} sqrt(v_C^+(x)) <= tau
```

is truly chance feasible.

Proof sketch: upper-bound the true mean using the confidence event, upper-bound
the standard deviation using `v_C^+`, then apply the Gaussian quantile bound.

## Theorem 7: One-Step SC-OLH-KG Value Characterization

Define the terminal certified value after one additional simulation at `x` as

```text
V_{n+1}(x) = min_{u in C_{n+1}} m_f^{n+1}(u),
```

where `C_{n+1}` is the certified feasible set using the updated mean and HVD
state.  The exact SC-OLH-KG acquisition is

```text
KG_SC(x) = E_n[V_n - V_{n+1}(x)].
```

The current implementation uses an additive approximation to this expression.
The proof target is to show when the approximation is a lower-order surrogate
or to replace it with a sampled exact estimator.

## Theorem 8: Finite-Budget Safe Simple-Regret Bound

Let `x_N` be the final certified recommendation and `x_*` the best feasible
design.  On the joint event of mean confidence, variance over-certification,
and sufficient state-space coverage,

```text
f(x_N) - f(x_*) <= optimization_error_N + certification_slack_N.
```

The optimization term is controlled by the cumulative information gain over
the state/meta feature space; the certification term is controlled by the HVD
estimation error and chance-bound slack.

Proof sketch: decompose regret into posterior mean error, acquisition/search
error, and feasibility certification error.  Use information-gain bounds for
the surrogate and Theorem 5-6 for variance/certification.

## Current Gaps

1. The code now has a factor-shock synthetic and factor-HVD cumulative feature
   path, but the exact terminal-value KG estimator is still not implemented.
2. The traffic trajectory encoder is not yet connected to real trajectory logs.
3. The proof of Theorem 5 needs the exact residual-square noise assumptions
   chosen for the manuscript.
4. The finite-budget theorem needs a final decision on whether the paper claims
   exact KG or an additive OLH-KG surrogate.

## Lean4 Status

The proof workspace now has two layers.

The Lean-core layer proves bookkeeping/algebra over `Nat`, mostly to keep a
minimal always-fast proof backbone.  The mathlib layer proves the real-valued
versions needed by the manuscript:

| Manuscript item | Lean file | Status |
| --- | --- | --- |
| Fixed trajectory cumulative variance algebra | `SCOLHKG/Real/CumulativeRisk.lean` | Lean-proved over `R` |
| Low-rank truncation bookkeeping | `SCOLHKG/Real/CumulativeRisk.lean` | Lean-proved over `R` |
| Information refinement / law of total variance | `SCOLHKG/Real/ConditionalVariance.lean` | Lean-proved for two-cell and arbitrary finite partitions |
| Chance certification | `SCOLHKG/Real/Certification.lean` | Lean-proved from GP-confidence and variance-upper events |
| HVD oracle inequality | `SCOLHKG/Real/HVD.lean` | Lean-proved from residual-square concentration event |
| Ridge-HVD oracle inequality | `SCOLHKG/Real/RidgeHVD.lean` | Lean-proved from ridge minimizer and uniform residual-square concentration |
| Exact KG maximizer | `SCOLHKG/Real/KG.lean` | Lean-proved for expected terminal gain |
| Additive KG equivalence condition | `SCOLHKG/Real/KG.lean` | Lean-proved when additive score equals exact gain |
| Additive-to-exact KG approximation | `SCOLHKG/Real/AdditiveApproxKG.lean` | Lean-proved with `2 eta` exact-KG gap |
| Information-gain regret accounting | `SCOLHKG/Real/InformationGainRegret.lean` | Lean-proved from an information-gain radius budget |
| Safe simple regret | `SCOLHKG/Real/SafeRegret.lean` | Lean-proved from certification and optimization-error events |
| General conditional variance | `SCOLHKG/Measure/ProbabilityEvents.lean` | Lean-proved by invoking mathlib `condVar` law of total variance |
| GP confidence event | `SCOLHKG/Measure/ProbabilityEvents.lean` | Lean-proved via Chebyshev and finite union bound |
| Sub-Gaussian GP confidence event | `SCOLHKG/Measure/SubGaussianConfidence.lean` | Lean-proved for one-sided and centered finite/adaptive candidate sets |
| Residual-square concentration event | `SCOLHKG/Measure/ProbabilityEvents.lean` | Lean-proved via Chebyshev for an abstract centered estimator |
| Posterior exact KG expectation | `SCOLHKG/Measure/PosteriorKG.lean` | Lean-defined as an integral and linked to exact KG maximization |
| High-probability safe regret | `SCOLHKG/Measure/SafeRegretEvent.lean` | Lean-proved by bad-event containment |

The next mathematical frontier is to derive the events used above from the
probability model:

1. random-policy occupancy decomposition from trajectory exposure maps;
2. kernel-specific proof that the implemented GP posterior error satisfies the
   sub-Gaussian parameters assumed by `SubGaussianConfidence.lean`;
3. distribution-specific residual-square concentration constants for the HVD
   ridge estimator;
4. kernel/feature-specific information-gain upper bounds;
5. exact SC-OLH-KG posterior-update value theorem tied to the implementation.

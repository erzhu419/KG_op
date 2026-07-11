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

Each design `x` induces a distribution over trajectories.  Its state-coupled
cumulative risk coordinate is

```text
psi(x) = (A(x), N(x)).
```

The same `psi` is used for HVD features, SC candidate anchors, GPR state basis,
certification diagnostics, and exact KG terminal values.  Its occupancy summary
`s(x)` determines expected exposures

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

## Proposition 2A: Rotation-Invariant Semantic Group Prior

Partition the ordered mean features into an always-active linear exposure
block and optional semantic blocks such as local curvature `A^2` and shared
exposure `N`.  Within optional block `G`, use one inclusion probability `q_G`
and one isotropic spike/slab precision.  For every orthogonal change of basis
`R_G`,

```text
penalty(R_G beta_G) = penalty(beta_G),
dim_eff(G) = |G| q_G.
```

Thus source domains transfer whether a semantic block is useful and its scale,
but do not transfer a coordinate direction that may rotate or permute across
domains.  The held-out target learns `beta_G` from its charged observations,
and the effective-dimension budget still counts every coordinate in the block.
The real-valued invariance and dimension identities are proved in
`SCOLHKG/Real/GroupSharedShrinkage.lean`.

## Proposition 2B: Task-Adaptive Group-Ridge Complexity

A universal hard effective-rank cap can exclude a necessary risk block.  V27
instead selects one nonnegative ridge precision for each semantic block from a
fixed finite grid using full nested refits on charged target observations.
For information eigenvalue `s_j` and ridge precision `lambda_j`, define

```text
df_eff = sum_j s_j / (s_j + lambda_j).
```

Every summand lies in `[0,1]`; hence `0 <= df_eff <= feature_dim`.  On an event
where nested-refit empirical risk is uniformly within `epsilon` of target risk
over the finite penalty class, the selected model obeys

```text
R(selected) <= min_h R(h) + 2 epsilon.
```

This learns whether a held-out task needs curvature or shared-shock mean
features without a target-specific task label or oracle query.  Exact-KG
fantasy clones execute the same posterior refit, so complexity-parameter VOI
is included rather than frozen outside acquisition.

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

The high-dependence main implementation uses `exact_mc` as the default
posterior-update estimator of this expression.  The implementation exposes
three acquisition variants:

```text
additive, exact_mc, blend.
```

`exact_mc` estimates the posterior-update expectation by cloning and updating
both GPR and factor-HVD states, then recomputing the theory-certified terminal
value with the same provider `v_C^+`.  `additive` is now only an ablation/proxy
and is justified by a uniform approximation gap.  `blend` is the controlled
interpolation used for robustness checks.

For each history `D_t`, the implementation now constructs one measurable
terminal action pool `C_t`.  The current value, every hypothetical update, and
the realized post-update recommendation all use this identical `C_t`; only the
next history may construct `C_{t+1}`.  Posterior risk-frontier actions are
closed into the experiment set before KG maximization.  The shared-pool gain
identity, one-step maximizer statement, and both finite-set inclusion claims
are Lean-proved in `SCOLHKG/Measure/SharedTerminalPoolKG.lean`.

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

## Assumption A3: Transferable Task-Structure Family

The held-out task has a latent structure

```text
xi = (R, S, theta_v)
```

in the support of a source-trained hyper-prior `Pi`. The target algorithm may
update `Q_t(xi)` only with target observations charged to its evaluation
budget. It may not use target truth, optimum, boundary, or hidden simulator
parameters.

## Theorem 9: Hierarchical Task-Posterior Cumulative Variance

For a finite task posterior `Q_t`,

```text
Var(C(x) | D_t)
  = E_Q[v_C(x; xi)] + E_Q[s_C(x; xi)^2]
    + Var_Q[m_C(x; xi)].
```

The final term is representation/transfer uncertainty. Therefore an alignment
that is individually overconfident cannot make the mixture overconfident while
other supported experts disagree. The finite algebra and implementation bridge
are Lean-proved in `SCOLHKG/Real/TaskPosterior.lean`.

### Proposition 9a: Prior-Supported Expert Proposal Mixture

The initial design, sequential candidate pool, and terminal pool use

```text
q_t^prop(xi) = (1 - epsilon) Q_t(xi) + epsilon Pi(xi).
```

For `0 < epsilon <= 1`, this mixture is normalized and satisfies
`q_t^prop(xi) >= epsilon Pi(xi) > 0` for every prior-supported expert. Thus a
small target sample cannot irreversibly remove a transferable proposal family;
the task posterior controls exploitation while the frozen source prior retains
identification support. Both claims are Lean-proved in
`SCOLHKG/Real/TaskPosterior.lean`.

## Theorem 10: Ambiguity-Robust Task Certification

If robust upper moments dominate every normalized task posterior in an
admissible ambiguity set and

```text
mu_upper + sqrt(beta) sqrt(epistemic_upper)
         + z_alpha sqrt(aleatoric_upper) <= tau,
```

then the same upper-moment certificate holds for every posterior in that set.
Finite KL nonnegativity, the change-of-measure inequality, the entropic dual
upper bound, source-prior exponential-moment aggregation, the Markov bad-event
bound, and the robust-envelope implication are Lean-proved. The resulting
finite radius is `(source_slack + KL(Q_t||Pi) + log(1/delta))/n_evidence`.
Its sharpness remains conditional on the source-task exponential-moment model;
domain-specific slack must be validated empirically.

### Proposition 10a: Separate Safe Generalized Posterior

V28 maintains independently normalized predictive and safe-decision masses
from the same frozen source prior. Probability clipping at
`epsilon in (0, 1/2]` bounds each threshold or pairwise log loss in
`[0, -log(epsilon)]`; nonnegative weighted sums preserve the corresponding
component bounds. The Gaussian constraint score remains governed by the
exponential-moment premise above rather than being incorrectly declared
globally bounded. Candidate allocation, robust certification, and exact KG are
centred on `Q_safe`, while objective aggregation may retain `Q_pred`. These
claims are Lean-proved in
`SCOLHKG/Real/SafeGeneralizedTaskPosterior.lean`.

### Proposition 10b: Budgeted Replicated-Finalist Safety

Let the final `R` evaluations be reserved inside the original budget `N`, and
freeze a finite finalist set before observing those new labels. For finalist
`x`, define

```text
U_x = ybar_g(x) + z_alpha sigma_plus(x)
      + z_delta sigma_plus(x) / sqrt(r_x) - tau.
```

On the joint event that the replicated mean upper bound dominates the latent
constraint mean and `sigma_plus` dominates the true aleatoric standard
deviation, `U_x <= 0` implies the original chance constraint. If no finalist
passes this empirical upper bound, the fallback minimizes `U_x` before the
objective and does not claim theory certification. Lean proves this event-wise
soundness, target-freezing contract, strict replicate-deficit decrease, and
that every reserved update remains inside `N` in
`SCOLHKG/Real/FinalistReplication.lean`.

### Proposition 10c: Expert-Stratified Nomination Support

Let every finite structural expert nominate one action from the frozen
terminal pool. The union of expert nominations contains each expert's action
independently of its current task-posterior mass. Consequently, ranking the
finite nominations by their expert-specific safety score cannot erase an
entire supported structural family merely because generalized Bayes assigned
it small mixture mass. The finite-support statement is Lean-proved in
`SCOLHKG/Real/FinalistReplication.lean`; empirical usefulness of each
nomination remains a benchmark question resolved by charged replication.

### Proposition 10d: Bounded Adaptive Expert Race

Let each reserved-stage nomination be a function of the history immediately
before the next paid observation. Insert every nominated action into an
archive, but admit an action to the final empirical race only after it reaches
the declared minimum replication count. After `R` refreshes, the archive has
cardinality at most its initial cardinality plus `R`; it contains every action
actually nominated, while an incomplete action cannot enter the completed
race. Splitting the final error budget across that deterministic finite upper
bound and applying a finite union bound controls whichever completed action is
selected. Event-wise chance-bound soundness then follows from Proposition
10b. These archive, filtering, cardinality, union-bound, and selected-action
claims are Lean-proved in `SCOLHKG/Real/FinalistReplication.lean`.

For V32, if the initial archive and every later nomination belong to one
candidate universe fixed before the reserved observations, the entire archive
remains a subset of that universe and its cardinality is at most the fixed
universe cardinality. Lean proves both the subset and cardinality forms, so the
confidence allocation can be chosen from the pre-observation universe rather
than inferred after the adaptive race.

This result does not certify V31's empirical fallback as the main theory
certificate. It proves the adaptive finite-selection contract under the stated
per-candidate confidence events; the main GP/HVD theory certificate retains
precedence in the implementation.

## Theorem 11: Joint Task-Posterior Exact KG

The exact hypothetical update state contains `Q_t`, every expert objective and
constraint GPR, and every expert cumulative HVD. Each predictive draw samples
one shared expert identity, updates the whole joint state, and recomputes the
robust terminal certified value. A zero-error MC maximizer is therefore a
one-step maximizer for this joint terminal gain; finite-MC error continues to
use the existing `2 eta` and concentration bridges.

## Current Empirical Closure Items

1. Finite task-posterior Stage A-C passed the paired FactorShock N=20 Gate 1:
   `4/7` true-feasible and `1/7` false-feasible versus `0/7` and `1/7` for the
   baseline, with lower mean violation and median regret. Inventory/Queue Gate
   2 and repair of the seed-0 false-feasible case remain mandatory before
   continuous Stiefel/Grassmann inference.
2. The code now has a factor-shock synthetic and factor-HVD cumulative feature
   path that feeds `v_C^+` in theory certification.  Exact-MC/blend are
   implemented and concentration-bridged, including an MC-schedule variance
   theorem. Batched KL-dual evaluation plus process-parallel candidate updates
   reduced the exact path to about 763 seconds per N=20 seed in Gate 1; the
   large matrix must still decide between `exact_mc`, `blend`, and additive
   plus the `2 eta` approximation theorem.
3. The traffic trajectory encoder/log schema and SUMO trajectory logger are
   implemented.  The remaining task is to generate the fresh-seed trajectory
   CSV on the server and include its encoded table.
4. The manuscript still needs a final choice between bounded,
   sub-exponential, or Gaussian-derived residual-square tails.
5. The full recursive `compute_h` sorted-stack fold/output theorem is now
   Lean-proved for the sorted/collapsed active-line loop.

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
| Policy/trajectory occupancy decomposition | `SCOLHKG/Real/OccupancyDecomposition.lean` | Lean-proved as occupancy cumulative risk plus remainder plus explained trajectory variance |
| GPR rank-one update / KG slope / replication VOI | `SCOLHKG/Real/GPRUpdate.lean` | Lean-proved; matches `ParametricGPR.update`, `compute_kg_vectorized`, and proves the observed-point variance reduction `q^2/(q+r)` is in `[0,q]` |
| Chance certification | `SCOLHKG/Real/Certification.lean` | Lean-proved from GP-confidence and variance-upper events |
| Theory certification implementation | `SCOLHKG/Real/CertificationImplementation.lean` | Lean-proved for `mu + sqrt(beta)s + z sqrt(v_C^+) <= tau`, with legacy mode dominated by theory mode |
| HVD oracle inequality | `SCOLHKG/Real/HVD.lean` | Lean-proved from residual-square concentration event |
| HVD implementation guards | `SCOLHKG/Real/HVDImplementation.lean` | Lean-proved for residual squares, nonnegative linear variance, clipping, and certification variance |
| Factor cumulative block implementation | `SCOLHKG/Real/CumulativeRiskImplementation.lean` | Lean-proved for `floor/independent/shared/linear/total` aggregation and shared-shock omission underestimation |
| Ridge-HVD oracle inequality | `SCOLHKG/Real/RidgeHVD.lean` | Lean-proved from ridge minimizer and uniform residual-square concentration |
| Source-prior HVD calibration | `SCOLHKG/Real/RidgeHVD.lean`, `SCOLHKG/Real/HVDImplementation.lean` | Lean-proved nonnegativity for prior-centered penalty, hierarchical variance scale, and within-policy sample variance |
| Posterior recommendation | `SCOLHKG/Real/PosteriorRecommendation.lean` | Lean-proved for robust-feasible posterior certification and objective argmin |
| Exact KG maximizer | `SCOLHKG/Real/KG.lean` | Lean-proved for expected terminal gain |
| Line-envelope KG | `SCOLHKG/Real/LineEnvelopeKG.lean` | Lean-proved at certificate level for active hull regions and `compute_h` sum formula |
| Stack-hull validator bridge | `SCOLHKG/Real/LineEnvelopeStack.lean` | Lean-proved endpoint/tail-slope checks imply active-line certificates |
| Stack-loop step preservation | `SCOLHKG/Real/LineEnvelopeAlgorithm.lean` | Lean-proved pop/push preserve slope/cut order under Python branch conditions |
| Final stack global dominance | `SCOLHKG/Real/LineEnvelopeGlobal.lean` | Lean-proved final global dominance invariant implies atom certificates and exact line-envelope KG without runtime validation |
| Concrete line-envelope branch certificates | `SCOLHKG/Real/LineEnvelopeIntersection.lean` | Lean-proved intersection arithmetic, popped finite-cell takeover, and right-tail finite/tail split certificates |
| Full line-envelope fold/output correctness | `SCOLHKG/Real/LineEnvelopeFold.lean` | Lean-proved recursive insert-loop/fold correctness: every original line is pointwise dominated by final output active lines; output endpoint dominance lifts to original-input `FinalEnvelopeStackInvariant` and exact KG |
| Additive KG equivalence condition | `SCOLHKG/Real/KG.lean` | Lean-proved when additive score equals exact gain |
| Additive-to-exact KG approximation | `SCOLHKG/Real/AdditiveApproxKG.lean` | Lean-proved with `2 eta` exact-KG gap |
| Exact-MC estimator bridge | `SCOLHKG/Real/ExactKGImplementation.lean` | Lean-proved: uniformly accurate exact-MC estimator inherits the same exact-KG gap |
| Information-gain regret accounting | `SCOLHKG/Real/InformationGainRegret.lean` | Lean-proved from an information-gain radius budget |
| Finite-kernel information-gain cap | `SCOLHKG/Real/FiniteKernelInformationGain.lean` | Lean-proved for scalar per-step finite kernel information gain and finite determinant/log-product cap |
| Kernel determinant bridge | `SCOLHKG/Real/KernelDeterminantBridge.lean` | Lean-proved: determinant-ratio cap feeds finite safe-regret accounting |
| Feature/kernel ratio cap | `SCOLHKG/Real/FeatureKernelDeterminantCap.lean` | Lean-proved: finite feature variance/noise ratio caps imply scalar and determinant information-gain caps |
| Feature-map norm cap | `SCOLHKG/Real/FeatureKernelDeterminantCap.lean` | Lean-proved: feature norm, coefficient variance, and noise floor imply the finite information-gain cap |
| ingolstadt21 numeric feature cap | `SCOLHKG/Real/FeatureKernelDeterminantCap.lean` | Lean-proved conservative cap with feature norm `10`, coefficient variance `10`, and noise floor `1e-8` |
| Safe simple regret | `SCOLHKG/Real/SafeRegret.lean` | Lean-proved from certification and optimization-error events |
| General conditional variance | `SCOLHKG/Measure/ProbabilityEvents.lean` | Lean-proved by invoking mathlib `condVar` law of total variance |
| GP confidence event | `SCOLHKG/Measure/ProbabilityEvents.lean` | Lean-proved via Chebyshev and finite union bound |
| Sub-Gaussian GP confidence event | `SCOLHKG/Measure/SubGaussianConfidence.lean` | Lean-proved for one-sided and centered finite/adaptive candidate sets |
| Finite-kernel GP posterior confidence | `SCOLHKG/Measure/GPKernelConfidence.lean` | Lean-proved with explicit `sum_i w_i^2 c_i` parameter |
| Posterior-sampled random candidates | `SCOLHKG/Measure/PosteriorSamplingCandidates.lean` | Lean-proved by posterior-score selector containment and deterministic adaptive envelope pools |
| Posterior coefficient sampler bridge | `SCOLHKG/Measure/PosteriorCoefficientSampler.lean` | Lean-proved for sampled coefficient score selectors staying inside finite pools |
| Multivariate-normal posterior coefficients | `SCOLHKG/Measure/PosteriorMultivariateGaussian.lean` | Lean-proved using mathlib `multivariateGaussian`: law, mean, covariance, and Gaussian linear scores |
| Residual-square concentration event | `SCOLHKG/Measure/ProbabilityEvents.lean` | Lean-proved via Chebyshev for an abstract centered estimator |
| Bounded residual-square constants | `SCOLHKG/Measure/ResidualSquareConcentration.lean` | Lean-proved via Hoeffding's lemma and finite union concentration |
| Sharper residual-square tail interface | `SCOLHKG/Measure/ResidualSquareTail.lean` | Lean-proved finite concentration from generic/sub-exponential residual-square tails and closed-form default radius inversion |
| Posterior exact KG expectation | `SCOLHKG/Measure/PosteriorKG.lean` | Lean-defined as an integral and linked to exact KG maximization |
| Exact posterior-update SC-OLH-KG | `SCOLHKG/Measure/PosteriorUpdateKG.lean` | Lean-proved as expected terminal certified value improvement under explicit update map |
| Exact-MC estimator concentration | `SCOLHKG/Measure/ExactMCConcentration.lean` | Lean-proved finite candidate-pool uniform-error probability bound from centered sub-Gaussian MC errors, plus MC-schedule variance scaling |
| High-probability safe regret | `SCOLHKG/Measure/SafeRegretEvent.lean` | Lean-proved by bad-event containment |
| Traffic finite stochastic model | `SCOLHKG/Real/TrafficTrajectoryModel.lean` | Lean-proved finite state-action occupancy plus shared demand-shock decomposition, schema-row field semantics, and SUMO snapshot row bridge |
| Boundary-aligned risk representation | `SCOLHKG/Real/RiskAlignedRepresentation.lean` | Lean-proved projector rotation invariance, retained-rank whitening, simplex expert bounds, nested-LOO noninterference, strong heredity, and exact weak-support fallback |
| Target boundary-evidence gate | `SCOLHKG/Real/RiskAlignedRepresentation.lean` | Lean-proved exact Stage-1 fallback when the target pilot has no observed feasible or no observed infeasible chance margin |
| Frozen source-boundary episode admission | `SCOLHKG/Real/RiskAlignedRepresentation.lean` | Lean-proved that source support may replace target gain evidence only; two-sided target support and target safety remain necessary, and source-only proposals are target-label invariant |
| Transactional representation switching | `SCOLHKG/Real/RiskAlignedRepresentation.lean` | Lean-proved exact rejection fallback and ordered posterior replay/commit semantics for admitted feature changes |
| Finite task posterior and hierarchical variance | `SCOLHKG/Real/TaskPosterior.lean` | Lean-proved normalization, positive support, prior-supported proposal-mixture lower bound, within/between/aleatoric variance, robust-envelope implication, and joint exact-MC optimizer bridge |
| Stratified finite-expert exact KG | `SCOLHKG/Real/StratifiedExpertKG.lean` | Lean-proved exact categorical posterior summation and posterior-weighted inheritance of the maximum within-expert Gaussian approximation error |
| Ordered cumulative-risk coordinate | `SCOLHKG/Real/OrderedCumulativeExposure.lean` | Lean-proved aggregate zero-frequency special case, finite positional linearity/selection, and reuse of the unchanged cumulative-risk decomposition |
| Orthogonal ordered/local semiparametric mean | `SCOLHKG/Real/OrthogonalSemiparametric.lean` | Lean-proved coefficient-nullspace projection gives zero finite-design ordered/local cross inner product; bounded kernel and bounded projection coefficients also give a global finite feature-amplitude bound |
| Total adaptive coefficient budget | `SCOLHKG/Real/AdaptiveCoefficientSparsity.lean` | Lean-proved that budgeting optional inclusion mass by `rho N - fixed` bounds the complete effective dimension, including the fixed prefix, by `rho N` |
| Finite unseen-task PAC-Bayes concentration | `SCOLHKG/Measure/TaskPACBayes.lean` | Lean-proved source-prior exponential-moment aggregation and bad-event probability `<= delta`; pointwise radius bound is in `Real/TaskPosterior.lean` |

The aligned representation remains experimental.  A five-seed, three-domain
source-only sequential replay activated the frozen coordinate five times and
won all five activations without increasing false feasibility; unsupported
Inventory episodes fell back to Stage 1.  This is offline admission evidence,
not a KG result.  No representation gain is promoted until the paired N=40 KG
control/admission matrix shows no domain-level negative transfer.  The next
frontier is no longer
missing formal structure.  It is to keep the
manuscript honest about the final empirical choice: if exact-MC is not the main
runner, the main text should state additive is an approximation and cite the
`2 eta` theorem; if exact-MC/blend wins after the large benchmark, cite the
posterior-update and MC-schedule concentration theorems.

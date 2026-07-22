# SC-OLH-KG Theory Skeleton

## Statistical Closure V2 Status

The promoted V51 behavior now has a separate finite-sample analysis contract.
`FiniteSampleHVD.lean` proves active-subspace identification, ridge estimation,
and coordinate-misspecification propagation;
`FiniteSampleHVDConcentration.lean` turns a declared replication schedule into
the uniform variance-target event used by that theorem;
`CertificateNonvacuity.lean`
turns positive safety depth into explicit mean/replication thresholds;
`EndToEndSafeRegret.lean` compares the observed-terminal recommendation with a
true safe optimum while retaining every representation, HVD, transfer, pool,
shortlist, MC, and sequential error; and `StatisticalClosure.lean` supplies the
joint high-probability wrapper. `TransferGeneralization.lean` instantiates the
transfer term by a source-task PAC-Bayes radius plus explicit domain shift.

This is a behavior-preserving theorem upgrade, not a retrospective relabeling
of experiments. The exact version policy and unresolved empirical obligations
are recorded in `SC-OLH-KG/docs/math_v1_gap_audit.md`.

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

Each design `x` induces a distribution over trajectories and a common
observable state/trajectory exposure `e(x)`. Two independently fitted heads
construct

```text
phi_mu(x) = h_mu(e(x)),
psi_v(x)  = h_v(e(x)) = (A(x), N(x)).
```

`phi_mu` parameterizes the GPR constraint mean and epistemic variance;
`psi_v` supplies the local and shared cumulative-risk blocks used by HVD,
certification variance, and variance-aware decisions. The heads share no
parameters and are not assumed to be interchangeable. They meet only in the
single certified chance margin

```text
rho(x) = m_g(phi_mu(x)) + sqrt(beta_g) s_g(phi_mu(x))
         + z_alpha sqrt(v_C^+(psi_v(x))) - tau.
```

This separation corrects the empirically false assumption that one coordinate
must simultaneously explain conditional mean and cumulative heteroscedastic
risk. Both maps are frozen from source observations before the held-out run;
only their posterior coefficients are updated by charged target evaluations.
The source regression target for `phi_mu` is the replicated constraint mean;
source chance-margin strata define only its representation. The independent
`psi_v` head is aligned using domain-standardized log variance estimated from
ordinary source replications, after which factor-HVD learns its cumulative
coefficients. Thus aleatoric scale cannot be silently absorbed into the mean
coordinate; it enters the deployed certificate exclusively through
`v_C^+(psi_v)`.
The occupancy summary `s(x)` determines expected exposures

```text
A(x) = E[A(T) | x],     N(x) = E[N(T) | x],
```

up to approximation error `delta_occ(x)`.

### Assumption A2B: Source Boundary Excitation

Let `S_source` be the frozen source policy design and `rho_d(x)` the signed
chance margin in source domain `d`. Learning a signed transferable coordinate
requires strict two-sided support,

```text
exists x_minus in S_source: rho_d(x_minus) < 0,
exists x_plus  in S_source: rho_d(x_plus)  > 0,
```

for enough source domains to identify the selected finite feature family. A
uniformly random design is not assumed to provide this support when feasibility
is rare. The implementation therefore compares random source sampling with a
single formula-free low-frequency source design, charges every replicate, and
freezes that design before the held-out target run.

Two-sided support is necessary here, not by itself sufficient. If two possible
margin functions agree on all of `S_source` but differ by `Delta` at an unseen
policy, every deterministic source-only estimator makes error at least
`|Delta|/2` on one of the two functions. This finite identifiability lower bound
and the failure of one-sided designs are proved in
`SCOLHKG/Real/BoundaryExcitation.lean`.

### Assumption A2C: Source-Consensus Proposal and Committed Suffix

Every source domain evaluates the same frozen universal profile library. Its
observed replicated chance margins are converted to within-domain percentile
ranks before aggregation. Positive affine rescaling of a source constraint
therefore leaves its profile order unchanged. The source-consensus shortlist is
frozen before the held-out target is named; a target-name change cannot alter
the shortlist.

The terminal replication suffix pre-registers the lowest `k` empirical chance
margins among already charged members of the frozen source design. It preserves
the safety role when a challenger is also the minimum-Bayes-risk arm and
commits every reserved challenger to `m` observations before posterior-only
experts can consume the suffix. If two true margins are separated by more than
twice their simultaneous estimation error, their empirical order is preserved.
The ordering and finite-budget contracts are proved in
`SCOLHKG/Real/SourceConsensusCommit.lean`.

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
v_C(x) = phi(x)^T theta_* + e(x),     theta_* in K_C,
```

where `K_C` is the product cone with nonnegative `floor/Lambda/omega` and a
PSD shared-shock matrix `B`. A sample variance based on `r_i` simulator
replications contributes `r_i-1` chi-square degrees of freedom. For the final
replication-weighted ridge objective, if the projected IRLS iterate is an
`epsilon_opt` approximate minimizer, then with probability at least
`1-delta`,

```text
E_n[(phi^T hat theta - v_C)^2]
  <= c1 inf_theta E_n[(phi^T theta - v_C)^2]
     + c2 ridge ||theta_*||_2^2
     + c3 complexity(phi,delta)/sum_i(r_i-1)
     + epsilon_opt.
```

Proof sketch: condition on each history-measurable IRLS weight matrix, apply
the approximate ridge basic inequality and sub-exponential residual-square
concentration, then add the optimization slack. Every accepted backtracking
step weakly decreases its fixed weighted objective, while projection preserves
membership in `K_C`. No claim is made that a one-shot coefficient projection
must improve prediction error.

### Hierarchical source-shape transfer

For source-domain PSD variance shapes `h_k(x) >= 0`, the transferable target
model is the nonnegative low-rank mixture

```text
v_C(x) = sum_k theta_k h_k(x),      theta_k >= 0.
```

Nonnegative mixing preserves the cumulative-risk cone. Replicated target
sample variances contribute scaled-chi-square Fisher information to the
posterior over `theta`; the certification variance adds the pointwise radius
`z_v sqrt(h(x)^T Sigma_theta h(x))`. In each scalar posterior direction,
adding nonnegative target information weakly decreases both posterior variance
and this upper radius. The legacy frozen 95% source multiplier remains an
ablation, not the hierarchical-transfer theorem.

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

## Theorem 7: Stage-I One-Step SC-OLH-KG Value Characterization

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

This theorem applies to adaptive acquisition decisions after the `n0` initial
design and before the `R`-call verification suffix. It characterizes neither
the initial design nor the suffix as exact KG. The complete algorithm is
governed by the two-stage theorem below.

## Theorem 7A: Hierarchical Boundary Certificate And Lexicographic Terminal Value

TCB-V2 writes the held-out domain chance margin as

```text
rho_d(x) = a_d + exp(s_d) h_theta(Q_d z_d(x))
           + c_d^T r_perp(z_d(x)).
```

The source domains identify the shared shape `h_theta`, a source distribution
over planar rotations `Q_d`, an orthogonal low-rank residual map `r_perp`, and
a prior on the task effects. The held-out target updates only location,
positive scale, optional planar angle, and the selected low-rank residual
coefficients from paid replicates. If `L_d` is a Cholesky factor of their
posterior covariance and `r_d^2` is residual variance, then

```text
U_d(x) = rho_d(x)
       + q * sqrt(||L_d^T ell_d(x)||^2 + r_d^2)
```

is never smaller than `rho_d(x)` for `q >= 0`. Posterior covariance therefore
cannot relax the certificate. The same `U_d` is used for frontier nomination,
fantasy terminal evaluation, and final recommendation.

On a frozen terminal set the V33 repair uses

```text
min_lex,x (1{U_d(x)>0}, max(U_d(x),0), m_f(x)).
```

Consequently, a certified action dominates every uncertified action regardless
of objective; among uncertified actions, reduction of positive upper margin
dominates objective improvement. Every fantasy observation refits the same
low-dimensional target posterior before evaluating this tuple. The finalist
frontier reserves Bayes-risk, certificate-margin, robust-violation, and
nominal-violation directions before expert nominations. Positive scale,
planar-rotation norm preservation, nonnegative uncertainty, upper-margin
non-relaxation, reserved-frontier order, three-layer certificate coherence,
and all three lexicographic dominance cases are Lean-proved in
`SCOLHKG/Real/HierarchicalBoundaryCertificate.lean`.

## Theorem 8: Two-Stage Finite-Budget Safe Simple-Regret Bound

Split the charged target budget into `N-R` state-coupled KG search calls and
`R` confirmatory ranking-and-selection calls. Let `x_S` be a search-quality
comparator in the frozen universe, `x_F` the best retained strictly safe
finalist, `x_N` the terminal certified recommendation, and `x_*` the best
feasible design. If

```text
f(x_S) - f(x_*) <= epsilon_search,
f(x_F) - f(x_S) <= epsilon_proposal,
f(x_N) - f(x_F) <= epsilon_verify,
```

then, on the simultaneous terminal upper-coverage event,

```text
true_margin(x_N) <= 0,
f(x_N) - f(x_*)
  <= epsilon_search + epsilon_proposal + epsilon_verify.
```

Search error is controlled by the Stage-I information-gain and exact-MC
results. Proposal error isolates source-prior/candidate-coverage mismatch.
Verification error is at most twice the uniform finalist objective error when
a comparator has a safety buffer larger than the uniform margin error. The
deterministic decomposition and terminal status semantics are Lean-proved in
`SCOLHKG/Real/TwoStageDecision.lean`; the finite-universe concentration and
three-event probability union are proved in
`SCOLHKG/Measure/TwoStageDecision.lean`. A complete derivation is in
`proof/two_stage_theory.md`.

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

### Proposition 9b: Joint Structure-Sensitivity Task State

The finite implementation refines `xi` to the joint state

```text
z = (xi, c),  xi = (R, S, theta_v),
```

where `c=(b,s,ell)` is a task-level signed mean-bias, predictive-error scale,
and decision-loss class. A normalized
positive product source prior followed by a joint generalized-Bayes score
produces a normalized positive posterior on `(xi,c)`, and both finite
marginals remain normalized and nonnegative. The likelihood may therefore
learn dependence between structural support and error sensitivity instead of
combining two independently selected gates. Sensitivity enters posterior Bayes
decision loss. In authoritative inference it may also conservatively inflate
constraint epistemic variance, so the implemented theory margin is

```text
mu_g + sqrt(beta_g) sqrt(s_g^2 max(1,c_scale)^2)
  + z_alpha sqrt(v_C_plus) - tau.
```

V5 preserves useful channel roles without assuming that source and target
enumerate those roles identically. Source domains learn canonical role
prototypes from observable channel distributions. The held-out assignment is
computed from an unlabeled target policy/exposure pool. Simultaneously
permuting channels and their assignment leaves every aligned role value, and
therefore every downstream mean descriptor, unchanged. This equivariance is
Lean-proved in `SCOLHKG/Real/BoundaryCoordinateSufficiency.lean`; whether the
learned roles are semantically sufficient remains an empirical transfer gate.

The same file formalizes V5 source-mean misspecification calibration. A
residual-driven scale is lower-bounded by one, so scaling a PSD source
coefficient law and nonnegative residual floor cannot reduce uncertainty. An
optional PSD directional term can only increase it further. Thus target data
may weaken or redirect a wrong source expert, but the calibration mechanism
cannot make a theory certificate less conservative merely by claiming lower
source uncertainty.

V6 corrects the temporal limitation of that construction. Each source expert
retains its unconditioned frozen law, and every charged target observation adds
a nonnegative standardized innovation square to a full-history sufficient
statistic. The implementation recomputes the source scale and mixture evidence,
then refits from the frozen law. Lean proves accumulation of the statistic,
the unit lower bound of every recomputed scale, and non-relaxation relative to
the frozen coefficient and residual variances. It intentionally does not claim
that successive learned scales are monotone: additional compatible target data
may legitimately reduce a previous empirical scale while it remains at least
one. The nontransfer `target:null` law is exactly unchanged.

V7 separates two failures that a scalar residual scale cannot identify.
Semantic support is estimated from source-standardized channel-role signatures
and an unlabeled target exposure pool. The resulting exponential trust lies in
`(0,1]`, so it may move prior mass from source experts to `target:null` but can
never increase transferred source mass. Conditional on support, frozen source
coefficient means define a PSD low-rank contrast covariance. Its projected
variance is nonnegative and can only enlarge epistemic uncertainty. The first
mechanism represents uncertainty about coordinate semantics; the second
represents directional disagreement among source mean laws. Neither reads
target outcomes or oracle feasibility. These are finite implementation
bridges, not a theorem that held-out semantic alignment must succeed.

V8 addresses the remaining representation failure instead of attempting to
calibrate coefficients inside a semantically unsupported coordinate. The
role-aligned coordinate and a fallback coordinate are fitted independently on
the same frozen source archive. A selector that reads only source and unlabeled
target channel cardinalities uses the role coordinate when that structural
support was observed in source domains and otherwise uses the fallback. Lean
proves the finite closure statement: if both branch-specific certification
bridges are sound, selecting either branch preserves soundness. It does not
assert that cardinality support is sufficient for semantic transfer; that is a
paired empirical gate.

V12 addresses target-candidate extrapolation without changing the HVD head.
The aligned latent coordinate is passed through a source-LODO-selected tanh
map, so every feature is strictly bounded independently of the selected
temperature. The nontransfer `target:null` component may replace its
coordinate-dependent isotropic covariance by an inverse-Gram geometry learned
from the deterministic unlabeled target role-matching pool. Its scale preserves
the previous average prior predictive variance on that pool. Lean proves both
boundedness and this exact scale-preservation identity; whether the resulting
coordinate is sufficient remains a held-out empirical gate.

V13 replaces global compression by a source-support projection. The selected
coordinate is exactly linear within the source-supported interval and bounded
outside it. An optional scalar overflow feature is zero on support and bounded
by one off support. Its coefficient is part of the same conjugate target mean
posterior, so ordinary charged observations can learn a held-out mean
discrepancy without restoring unbounded extrapolation. Lean proves the finite
support, identity, and residual-channel bounds; transfer sufficiency remains a
paired empirical claim.

V14 closes the remaining initialization leak between these heads. A singleton
target response is not treated as variance evidence because its squared
residual mixes mean misspecification and aleatoric noise. The HVD head instead
retains the frozen replicated-source variance prediction until a target policy
is evaluated repeatedly; a within-policy sample variance then supplies target
variance evidence. Lean proves singleton evidence is invariant to the chosen
mean head and replicated evidence remains nonnegative.

V15 also closes the ensemble-level mixture path. The task law is factorized as
`Q_t^mu x Q_t^v`: target response and boundary scores update `Q_t^mu`, while
`Q_t^v` remains at its frozen source prior until a within-policy replication
provides sample-variance evidence. Mean and epistemic bounds are taken under
the first posterior, cumulative aleatoric bounds under the second, and the
separable certified margin combines the two valid upper bounds. Exact-KG uses
the same Cartesian-product posterior in every fantasy clone. Lean proves that
the variance mixture and replication score are independent of all mean-head
weights and predictions.

V16 addresses variable observable-channel semantics without recoupling the
heads. A source-only partial transport maps channels to canonical roles and
retains missing-role mass. Its target matching residual can only multiply the
source mean coefficient covariance by a factor at least one. The Lean bridge
proves nonnegative role mass, a unit lower bound on the mismatch scale, and
non-decreasing scalar epistemic variance. Empirical sufficiency of the
transported coordinate remains a preregistered held-out-domain gate.

V17 changes the role-identification statistic rather than relaxing the
certificate. Source and target channels are probed by the same normalized
low-frequency policy interventions; a variable-cardinality target response is
represented by a nonnegative unit-mass barycenter of source-role responses.
Consequently each scalar aligned response stays in the source-role convex
hull. Target residuals may separately inflate the source mean-coefficient law
through the hierarchical misspecification posterior, but cannot reduce source
epistemic variance and cannot alter `target:null`. These statements do not
assert held-out response sufficiency; that remains the V17 empirical gate.

V18 keeps the V15 source mean coordinate and adds only a target-specific
orthogonal residual dictionary. A deterministic unlabeled target policy pool
defines raw observable exposure features. Their projection onto the source
mean span is removed before a rank-one or rank-two SVD basis is retained. The
new coefficient prior has mean zero and an independent PSD covariance block,
so it cannot shift the frozen source prior mean or reduce predictive
uncertainty. Charged target observations perform the only outcome-dependent
update. Lean proves the corresponding mean preservation, non-decreasing
variance, and exact source-plus-residual energy decomposition. Whether this
small residual span is sufficient on a held-out domain remains an empirical
claim rather than a theorem.

V19 does not select that rank by target name. It embeds ranks zero through two
in one maximum-rank feature map, expands both source and target-null laws over
the same nested structure variable, and updates their joint mass with ordinary
charged target likelihoods. The moment projection includes between-rank
disagreement, so uncertainty about structure can only add epistemic variance.
Lean proves nonnegative active/inactive rank variance, probability
normalization of the target-evidence update, and the variance non-reduction
property. Empirical promotion still requires nonvacuous true certificates.

V20 instead treats every admissible channel-to-role injection as one finite
structure atom. V21 changes only the evidence used to weight those atoms. For
assignment `pi`, let `s_pi` be the sum of Gaussian log predictive densities in
which each charged target margin is predicted while omitted from that atom's
conditioning set. The generalized-Bayes structure posterior is

```text
q(pi | D_t) proportional to q_0(pi) exp(s_pi / T).
```

The source laws, assignment orbit, and temperature are frozen before target
outcomes. The score uses ordinary charged target responses but no target
oracle. After scoring, each atom is conditioned on the complete target history
and moment matching retains between-assignment disagreement. Lean proves
strict positivity of every exponentiated finite score, positive evidence mass
for any nontrivial nonnegative prior, normalized nonnegative posterior weights,
and exact commutation of the likelihood with assignment relabeling. It does not
claim that ten observations identify the oracle-best assignment; that is the
V21 empirical gate.

V22 does not erase the source-learned geometric match before that update. Let
`L_t(pi)` be the matching cost between observable target channel signatures and
the frozen source role atlas. The signatures use a deterministic unlabeled
target policy pool. A source-domain best/second-best cost gap calibrates
`T_src > 0`, giving

```text
q_0(pi | E_t) proportional to exp(-L_t(pi) / (s T_src)).
```

The scale `s` is preregistered, and no target response or oracle enters this
prior. Simultaneous channel/assignment relabeling only permutes its finite
atoms. Lean proves positive geometry likelihoods, monotonic preference for
lower costs at positive temperature, and a normalized nonnegative posterior.
Whether source geometry remains semantically useful is still an empirical
held-out-domain claim.

V23 prevents charged target outcomes from rewriting that semantic match merely
to compensate for a misspecified source mean. It introduces a separate expert
index `e` and uses the hierarchical law

```text
q(pi, e | D_t, E_t) = q_0(pi | E_t) q(e | pi, D_t).
```

Target outcomes update the source/null conditional law within each assignment,
while the geometry-derived assignment marginal remains fixed. If every
conditional expert law sums to one, summing the joint law over `e` recovers
exactly `q_0(pi | E_t)` before and after any target-evidence update. Lean proves
this marginal invariance, joint nonnegativity, and joint normalization. The
implementation additionally retains between-assignment and between-expert
disagreement in epistemic covariance and keeps the cumulative HVD posterior in
an independent product factor.

V24 tests whether a noncontractive full-history scale can calibrate the
conditional source expert while preserving the V23 assignment marginal. The
formal marginal statement is exact, but the empirical gate showed that global
scale inflation can make a diffuse wrong source expert dominate and can erase
all remaining certification depth. V25 therefore moves semantic adaptation to
the finite role law itself. Source domains define a Gaussian law for each
canonical role's Fisher-transformed association with the chance margin; the
charged target pilot supplies the analogous noisy channel statistics. For an
injection `pi`, the code uses

```text
ell(pi) = -1/2 sum_j [
  (z_t,j-z_s,pi(j))^2 / (u_t,j^2+u_s,pi(j)^2)
  + log(u_t,j^2+u_s,pi(j)^2)
].
```

Multiplying `exp(ell(pi)/T)` by the source-geometry prior and normalizing gives
a finite posterior. Lean proves positive evidence, normalized nonnegative
mass, and exact equivariance when target channels and the injection are
relabelled together. The target truth pool is excluded: only charged pilot
responses enter this update. Afterward the assignment law is frozen and the
conditional expert posterior adapts online. Source coefficient disagreement
is also estimated separately inside each assignment block; Lean proves zero
contrast gain outside the active block, while the existing source-contrast
theorems prove nonnegative epistemic gain inside it.

V26 removes the assumption that a source channel has a transferable discrete
role at all. Each source domain first fits the same fixed-dimensional linear
mean coordinate. Its channel coefficient blocks are then reduced to one
exchangeable block law and copied to every target channel. Thus the frozen
source prior carries a distribution over channel effects, but no source channel
identity. For target channel permutation `pi`, the scalar mean satisfies

```text
m(b_(pi(j)), phi_(pi(j))) = m(b_j, phi_j),
```

when coefficients and observable channel features are relabelled together, and
the product source score is unchanged because every channel uses the same block
law. These two invariances are Lean-proved. Ordinary charged target observations
condition the individual blocks, so their posterior means may acquire different
signs and magnitudes. The existing hierarchical predictive-scale posterior is
applied independently to each frozen source expert and can only increase its
epistemic uncertainty; `target:null` remains unscaled. The cumulative HVD
posterior remains the independent `Q_t^v` factor from V15. The theorem does not
assert that ten target observations identify a sufficient held-out boundary
coordinate. Recovery of true feasibility, oracle certifiability, and rank
quality is the paired V26 empirical gate.

The signed bias `b(psi)` (including a source-frozen low-rank functional bias)
is used by posterior Bayes ranking but is absent from this
certificate. This margin is never smaller than the sensitivity-free theory margin. Product-
prior normalization, positive posterior support, marginal normalization and
nonnegativity, signed-bias separation, shadow-mode sensitivity independence,
and authoritative non-relaxation are Lean-proved in
`SCOLHKG/Real/JointTaskLatentPosterior.lean`.

V4 refines the signed-bias coordinate to a Gaussian coefficient posterior for
each structural expert. A charged standardized residual performs a conjugate
precision update `P <- P + w phi phi^T`. Its posterior mean is still absent
from the theory margin. Its coefficient covariance contributes only the
nonnegative epistemic term `predictive_sd^2 phi^T P^{-1} phi`; therefore the
adaptive margin dominates the scale-only authoritative margin. Precision
growth, mean/certificate separation, and covariance non-relaxation are
Lean-proved in the same file.

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

### Theorem 10e: Two-Stage Search-and-Verification Guarantee

The propositions above are the Stage-II implementation lemmas. Combined with
the Stage-I search theorem, they yield the main deployed guarantee. Let the
search, proposal-retention, and verification errors be
`epsilon_S`, `epsilon_P`, and `epsilon_V`. On their joint good event, a
certified terminal report is truly chance feasible and

```text
f(x_N) - f(x_star) <= epsilon_S + epsilon_P + epsilon_V.
```

If no candidate is certified, the algorithm emits an explicitly uncertified
least-upper-risk fallback. Under uniform margin error `epsilon_g`, its true
margin is at most the true margin of any completed finalist plus
`2 epsilon_g`; it does not inherit the safety conclusion above. If search,
proposal, and verification bad-event probabilities are bounded by
`delta_S`, `delta_P`, and `delta_V`, the certified safe-regret failure
probability is at most their sum, without an independence assumption. These
claims are Lean-proved in `SCOLHKG/Real/TwoStageDecision.lean` and
`SCOLHKG/Measure/TwoStageDecision.lean`.

## Theorem 11: Stage-I Joint Task-Posterior Exact KG

The exact hypothetical update state contains `Q_t`, every expert objective and
constraint GPR, and every expert cumulative HVD. Each predictive draw samples
one shared expert identity, updates the whole joint state, and recomputes the
robust terminal certified value. A zero-error MC maximizer is therefore a
one-step maximizer for this joint terminal gain; finite-MC error continues to
use the existing `2 eta` and concentration bridges.

This theorem characterizes the search acquisition only. It is not used to
rename committed terminal verification as exact KG.

## Theorem 12: Promoted Evaluate-or-Replicate Bayes-Risk Closure

Let `O_t` be the set of policies charged to the target budget and let

```text
R_t(x) = E[f(x) | D_t] + rho E[(G(x))_+ | D_t],   x in O_t,
V_t(a) = min_{x in O_t} R_t(x)
         - E[min_{x in O_{t+1}(a)} R_{t+1}(x) | D_t, a].
```

An evaluation action at an unobserved `x` changes the terminal universe to
`O_t union {x}`. An admissible replication leaves `O_t` unchanged. Both have
unit target-simulator cost, update the same objective, constraint, task, and
cumulative-HVD posterior, and are compared by the same `V_t`. The final
recommendation minimizes the same `R_N` on `O_N`; no empirical finalist or
unobserved posterior action may override it.

V51 evaluates exact-MC VOI only on a finite posterior shortlist. Suppose that
the shortlist covers the full finite action pool within `epsilon_S` in exact
VOI and that the MC estimate is uniformly within `eta_MC` on the shortlist.
Then its selected action satisfies

```text
max_{a in A_t} V_t(a) - V_t(a_t) <= epsilon_S + 2 eta_MC.
```

Summing the same Bellman reductions over a target budget telescopes; the only
decision approximation terms are the per-step shortlist and MC errors. This is
not a module-weight argument: mean, cumulative HVD, replication, and source
discrepancy are posterior state components inside one terminal Bayes risk.
Separately, any final action whose conservative theory margin is nonpositive
inherits the chance-feasibility theorem. The optimizer may still return an
explicitly uncertified Bayes action when the certified set is empty.

`SCOLHKG/Real/PromotedV51Closure.lean` proves the action semantics, the
`epsilon_S + 2 eta_MC` bound, finite-budget telescoping, and the joint one-step
optimization/certification statement. The Python bridge is
`SingleOLHKGAlgorithm._terminal_action_pool`: current value, every fantasy, and
the final recommendation now use the same observed terminal universe and the
same decision-risk penalty.

V52 retains V51 as an explicit action fallback. It may append a
certificate-depth policy, an observable cumulative-risk coverage policy, and
a finite-horizon rollout challenger. A challenger is executed only when its
estimated lower-is-better terminal value beats the fallback by at least
`2 eta`. `SCOLHKG/Real/SafeguardedPolicyImprovement.lean` proves that, under a
uniform `eta` numerical-error event, every accepted switch is noninferior in
the exact posterior value. This removes a numerical route to regression but
does not remove representation, transfer, pool-coverage, or sequential-model
error from `StatisticalClosureErrors`.

V53 narrows that policy to one constrained posterior improvement problem. It
removes rollout and defines the second terminal functional
`C(D)=max(min_{x in O_t} M_D(x),0)` on the same observed action universe as
the promoted Bayes-risk decision. A supplemental evaluate/replicate action can
replace the V51 fallback only after separate Bayes-risk and certificate-score
gaps exceed `2 eta_risk` and `2 eta_certificate`.
`SCOLHKG/Real/ConstrainedCertificateDeficit.lean` proves strict exact
improvement of both scores on their two uniform numerical-error events and
joint noninferiority under fallback-or-switch. Full
`stratified_expert_nested` summation remains the exact finite-expert reference,
but the mainline numerical contract uses `factorized_rqmc_nested`: one nested
scrambled Sobol net samples Gaussian innovation and the separate mean/HVD
expert factors. The implementation records its empirical selector law
`qHat`. `SCOLHKG/Real/StratifiedExpertKG.lean` proves
`|E_qHat fHat - E_q f| <= epsilon + ||qHat-q||_1 B` whenever the conditional
quadrature error is at most `epsilon` and the exact finite-expert terminal
value is bounded by `B`. Nested MC8/MC32 calibration and the reported selector
L1 discrepancy remain empirical implementation obligations; neither is
silently promoted to an exact-integral theorem.

The completed V53-v1 RQMC audit showed that one raw-unit error radius is not
portable across domains. V53-v2 normalized each posterior score by
`max(1, ||current terminal value||_infinity)`. The scale is positive,
pre-update, target-oracle free, and common to every action;
`positive_score_normalize_lt_iff`,
`uniform_score_approximation_normalize`,
`two_eta_score_guard_normalize_iff`, and
`constrained_guard_normalize_iff` prove exact scale equivalence. The completed
MC32/MC128 gate passed ranking stability, but every scale was one and the
resulting `eta_risk=343.7621` admitted no supplemental action in 30 cells.
Thus V53-v2 is a negative numerical result, not a promoted policy.

V53-v3 keeps the raw-score V51 fallback and both terminal functionals, but
changes the posterior numerical utility integrated by a supplemental policy.
For each fantasy it uses
`clip((L_current-L_after)/max(1,||L_current||_infinity),-1,1)` before posterior
integration. `boundedCurrentGain_mem_Icc`,
`boundedCurrentGain_abs_le_one`, and
`boundedCurrentGain_pair_difference_le_two` make the finite score range
explicit. The existing two-error theorems then apply to exact and estimated
bounded expected utilities. The RQMC uniform-error event remains an empirical
implementation obligation; boundedness does not silently turn dependent
scrambled-net samples into IID observations.

V54 addresses the resulting empty global guard without weakening the
posterior objective. For each challenger `a` and literal V51 fallback `b`, it
uses the paired difference `Delta_m(a,b)=S_m(a)-S_m(b)` computed from nested
common-random-number prefixes and defines
`r_a=kappa |Delta_high(a,b)-Delta_prefix(a,b)|`. Risk and certificate heads
have separate radii and both must pass. `PairDifferenceApproximation` and
`pair_difference_guard_implies_exact_improvement` show why this is sufficient:
only the selected pair difference, not the worst action in the pool, must be
controlled. `nested_pair_guard_implies_exact_improvement` and the paired
fallback-or-switch theorem connect the recorded radius to joint posterior
noninferiority.

The mathematics deliberately does not claim that one observed prefix
difference always bounds deterministic scrambled-net integration error.
That coverage statement is a frozen numerical-fidelity condition. It must be
audited with a third nested fidelity level before a V54 sentinel is authorized.
The first MC128 action-support run is diagnostic only because it was launched
from the V53-v3 core before the V54 selector was deployed.

The first 15-cell MC128 action-support diagnostic failed before any selector
gate: only Queue seed 4 contained a supplemental action that improved both
posterior terminal heads. The replacement finite support uses oracle-free
extreme points of one posterior action vector: risk EI, constrained EI,
chance-boundary location/uncertainty, certificate depth, constraint-mean and
cumulative-HVD information, their joint margin reduction, and observable
`psi=(A,N)` coverage. `literal_action_superset_preserves_fallback` and
`paired_action_superset_policy_joint_noninferiority` make the relevant
monotonicity explicit: support expansion cannot remove the V51 fallback, and
an accepted pair-guard switch retains the same conditional joint guarantee.
The theorem does not assert that these finite directions contain a useful
challenger; that remains the MC128 support obligation.

The V54 implementation reuses each posterior clone/update across the two
terminal heads. This is an execution-schedule change, not a third objective:
`joint_terminal_head_reuse_eq_separate_passes` proves that accumulating both
finite weighted gains after the same fantasy update equals two separate
passes. A fixed-CRN Python regression requires all raw, transformed, expected
terminal, and selected-action outputs to agree within `1e-12`. The small
warm local probe improved median runtime by `1.25x`; a task-ensemble server
profile remains the relevant performance obligation.


The replacement V54 Pareto-support diagnostic completed all 15 cells. It
found fallback-relative joint dominators in Inventory and Queue but none in
FactorShock; even the convex hull of the finite action scores had negative
max-min fallback-relative gain in all five FactorShock seeds. Thus V54's
premise is infeasible on that posterior family and V54 is closed without
weakening its theorem or launching MC512.

V55 changes the comparison object, not either terminal functional. For an
action `a`, let `S_R(a)` and `S_C(a)` be exact expected reductions from the
unchanged current Bayes-risk and certificate-deficit terminal states. Nested
prefix/high estimates give radii `r_R(a)` and `r_C(a)`. The action is admitted
only when `S_hat_R(a)-r_R(a)>0` and `S_hat_C(a)-r_C(a)>0`.
`AbsoluteScoreApproximation` and `score_lower_confidence_bound_le_exact` lift
each lower bound to the exact score;
`current_relative_joint_guard_improves_both_terminal_scores` and
`current_relative_joint_guard_decreases_both_terminal_costs` then prove exact
current-state Pareto improvement. The theorem does not assert that the action
beats the distinct V51 risk maximizer in both heads. Empty admission retains
that literal fallback. Radius coverage, nonvacuous activation, and final V51
performance noninferiority remain separate preregistered empirical gates.

## Current Empirical Closure Items

1. The observed-terminal repair passed its paired 60-run closure gate at
   `d=1000`, `N=20`, `n0=10`: `60/60` true-feasible recommendations, 26
   adaptive improvements, zero adaptive losses, and zero false certificates.
   Relative to the pre-repair V51 control it won 17 runs, lost 7, and tied 36.
   The repaired decision contract is now the promoted baseline.
2. Certification is sound but currently vacuous at this extreme budget. Both
   60-run V51 matrices had empty certified sets; in the pre-repair control,
   zero of 129 truly feasible evaluated points were certified. Therefore zero
   false certificates is not evidence of useful coverage. Nonvacuity must be
   reported and gated at larger target/replication budgets before the paper
   may claim empirical certified optimization.
3. Exact-MC currently uses two antithetic Gaussian samples. The theorem exposes
   this honestly as `eta_MC`; it is not called an exact numerical integral.
   Paper experiments must include MC-sample and shortlist-size sensitivity, or
   estimate the resulting rank-stability/coverage error.
4. The source archive and frozen proposal are target-oracle free, but the
   finite PAC-Bayes certificate still assumes a source-task exponential-moment
   bound. Its slack must be upper-bounded from source-only held-out episodes and
   frozen before each target run.
5. The first 30-pair V53 numerical gate rejected sampled-expert MC8: risk and
   certificate pairwise agreement were `0.870` and `0.882`, while certificate
   top-1 agreement was `0.567`. V53 is not promoted. Exact enumeration then
   exposed 49 mean/HVD product experts and was cancelled as a runtime reference,
   not reported as evidence. The next gate uses nested factorized RQMC and
   reports both score fidelity and selector L1 discrepancy.
6. The traffic trajectory encoder and SUMO logger exist, but the final table
   still requires real fresh-seed trajectory CSV and out-of-sample replication
   certification. Missing logs must remain `missing_data`, never synthetic
   evidence.
7. The legacy TCB-V2--V5 gates are historical negative results: their formal
   envelopes were sound but empirically vacuous or inaccurate. They are not
   part of promoted V51 and should move to an appendix failure analysis rather
   than remain in the main algorithm narrative.

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
| Theory certification implementation | `SCOLHKG/Real/CertificationImplementation.lean` | Lean-proved for `mu + sqrt(beta)s + z sqrt(v_C^+) <= tau`, with legacy mode dominated by theory mode and the necessary nonvacuity condition `beta * v_epi <= safety_depth^2` under oracle mean |
| Separated mean/risk coordinate bridge | `SCOLHKG/Real/MeanRiskCoordinateSeparation.lean` | Lean-proved that `eta` alone determines constraint mean/epistemic variance, `psi` alone determines certification variance, joint coordinate equivalence preserves the chance margin, and the separated implementation inherits certificate soundness |
| Source-aligned chance-boundary coordinate | `SCOLHKG/Real/BoundaryCoordinateSufficiency.lean` | Lean-proved exact mean/variance oracle-substitution identities; factorization of distinct `phi_mu` and `psi_v` heads through one observable state/trajectory exposure; invariance of the complete separated margin under equal exposures; certificate soundness, head noninterference, candidate-support necessity/restoration, and non-relaxation by a nonnegative transfer guard. V4 proves exact preservation of source discrepancy at reference feature energy. V5 proves conservative static source-law inflation and role-permutation equivariance. V6 proves nonnegative online sufficient-statistic accumulation, a unit lower bound for every recomputed scale, conservative refitting relative to the same frozen source law, and exact invariance of the target-null component. V7 proves role-support trust is in `(0,1]`, cannot increase source mass, and that source-contrast uncertainty is nonnegative and can only increase epistemic variance. V8 proves reduction and sound-property closure for the source-support adaptive role/fallback selector. V12 proves strict source-tanh boundedness and nonnegative average-variance-preserving target-feature rescaling. V13 proves on-support identity, off-support clipping, and a zero-on-support bounded discrepancy feature. It deliberately does not assert temporal monotonicity or universal semantic alignment. Held-out coordinate sufficiency remains an explicit empirical gate rather than an assumed theorem. |
| Variable-cardinality role transport | `SCOLHKG/Real/BoundaryCoordinateSufficiency.lean` | V16 Lean-proves nonnegative partial-role mass, a unit lower bound for the source/target matching epistemic scale, and that multiplying a nonnegative source covariance by that scale cannot decrease predictive variance. Target-free transport fitting and held-out sufficiency remain implementation and empirical contracts. |
| Intervention-response barycentric role transport | `SCOLHKG/Real/BoundaryCoordinateSufficiency.lean` | V17 Lean-proves convex-hull containment for nonnegative unit-mass role responses, exact non-inflation of the target-null mean law, and non-decrease of source epistemic variance under hierarchical misspecification scaling. Source-only role learning and held-out sufficiency remain audited implementation and empirical contracts. |
| Target-orthogonal residual mean coordinate | `SCOLHKG/Real/BoundaryCoordinateSufficiency.lean` | V18 Lean-proves that a zero-mean residual law preserves the source prior mean, an independent nonnegative residual variance cannot reduce predictive uncertainty, and orthogonality gives an exact additive energy decomposition. Outcome-free construction and held-out sufficiency remain implementation and empirical contracts. |
| Bayesian residual-rank structure posterior | `SCOLHKG/Real/BoundaryCoordinateSufficiency.lean` | V19 Lean-proves nonnegative nested active/inactive coefficient variance, normalized nonnegative target-evidence rank weights, and non-reduction of predictive variance when between-rank disagreement is retained. Oracle-free construction and held-out usefulness remain implementation and empirical contracts. |
| Finite channel-role assignment posterior | `SCOLHKG/Real/BoundaryCoordinateSufficiency.lean` | V20 Lean-proves unit mass of the uniform finite assignment prior, invariance of the mixture mean under atom relabeling induced by channel permutations, non-reduction of predictive variance when between-assignment disagreement is retained, and normalized nonnegative target-evidence assignment weights. The source-only hypothesis construction, one-block-per-atom implementation, and post-run-only oracle expressivity audit are implementation contracts checked by the V20 gate. |
| Cross-fitted assignment structure posterior | `SCOLHKG/Real/BoundaryCoordinateSufficiency.lean` | V21 Lean-proves positivity of exponentiated finite LOO scores, positive evidence mass under any nontrivial nonnegative assignment prior, normalization/nonnegativity of the resulting generalized-Bayes posterior, and score-likelihood equivariance under assignment relabeling. Exact Gaussian LOO arithmetic, refitting from frozen component laws, oracle exclusion, and independent HVD state are executable contracts checked by the V21 gate. |
| Source-geometry assignment prior | `SCOLHKG/Real/BoundaryCoordinateSufficiency.lean` | V22 Lean-proves positivity of exponentiated negative matching costs, monotone preference for lower-cost assignments at positive source-calibrated temperature, and normalization/nonnegativity of the finite prior. Source-only temperature calibration, unlabeled target cost construction, source/null orbit equality, and HVD isolation are executable contracts checked by the V22 gate. |
| Factorized assignment/misspecification posterior | `SCOLHKG/Real/BoundaryCoordinateSufficiency.lean` | V23 Lean-proves that a normalized frozen assignment marginal times normalized assignment-conditional expert laws is a nonnegative unit-mass joint posterior, that marginalizing experts exactly recovers the frozen assignment law, and that changing the normalized conditional expert posterior cannot alter that marginal. Geometry-only assignment construction, target-only conditional expert updating, moment-matched disagreement, and HVD isolation are executable contracts checked by the V23 gate. |
| Factorized hierarchical source-mean calibration | `SCOLHKG/Real/BoundaryCoordinateSufficiency.lean` | V24 combines the V6 noncontractive full-history source-scale refit with the V23 hierarchical law. Lean proves that any normalized before/after hierarchical expert conditionals have the same frozen assignment marginal. Refitting from frozen component laws, target-only scale statistics, ten sequential updates, and independent HVD state are executable contracts checked by the V24 gate. |
| Charged-pilot boundary-role posterior | `SCOLHKG/Real/BoundaryCoordinateSufficiency.lean` | V25 Lean-proves positivity and normalization of Gaussian boundary-role assignment evidence, exact channel/assignment relabel equivariance, and zero source-contrast predictive gain outside an assignment's active block. Source-only role statistics, charged-pilot-only target statistics, post-pilot freezing, assignment-conditional covariance construction, target-oracle exclusion, and HVD isolation are executable contracts checked by the V25 gate. |
| Exchangeable target-linear mean coordinate | `SCOLHKG/Real/BoundaryCoordinateSufficiency.lean` | V26 Lean-proves exact scalar-mean equivariance under simultaneous channel feature/coefficient relabeling and exact invariance of a shared exchangeable channel-block prior score. Source-domain block reduction, copied source laws, charged-target-only posterior differentiation, hierarchical misspecification noncontraction, target-null invariance, target-oracle exclusion, and HVD isolation are executable contracts. Held-out boundary sufficiency and certification nonvacuity remain empirical gate criteria. |
| Single exchangeable empirical-Bayes hyperlaw | `SCOLHKG/Real/BoundaryCoordinateSufficiency.lean` | V27 Lean-proves nonnegativity of every projected moment-matched covariance, exact reduction of a one-atom aggregate law to that atom's mean and covariance, and exact one-step growth of the charged target history. Full-history refitting from the frozen law, absence of source identity/null atoms, exact-KG clone equivalence, target-oracle exclusion, and HVD isolation are executable contracts. Held-out mean adequacy and certification nonvacuity remain empirical gate criteria. |
| Constraint-head authority separation | `SCOLHKG/Real/BoundaryCoordinateSufficiency.lean` | V28 Lean-proves that legacy task-joint mean and epistemic moments cannot affect the separated theory certificate or Bayes-risk chance-margin mean, and that task-HVD and direct cumulative-HVD modes reduce exactly to their selected aleatoric variance. Python enforces the same sole-authority routing in candidate filtering, final recommendation, post-run audits, and every cloned posterior terminal update. Safety and certifiability improvement remain paired empirical gate criteria. |
| Posterior-dominance incumbent preservation | `SCOLHKG/Real/BoundaryCoordinateSufficiency.lean` | V29 Lean-proves the covariance-free difference-variance upper bound used by the runtime and the algebraic implication from a `1-delta` Cantelli acceptance threshold to false-switch probability at most `delta`, conditional on the one-sided Cantelli probability inequality. Python maintains one incumbent across all paid updates and uses it as the terminal recommendation without target-oracle access. Posterior calibration and empirical preservation remain gate conditions, not theorem conclusions. |
| Finite-sample robust source-mean posterior | `SCOLHKG/Real/BoundaryCoordinateSufficiency.lean` | V30--V31 Lean-prove that a scale clipped below at one and a finite nonnegative HC3 projected correction cannot reduce posterior uncertainty, and that the post-conditioning covariance correction leaves the conditioned mean unchanged. V34 proves the composed two-layer projected variance is nonshrinking. Python binds these scalar facts to PSD coefficient covariance updates, full charged-history refits from one frozen source hyperlaw, finite-target/source multipliers, and exact-KG clone state. Held-out calibration and certificate coverage remain empirical criteria. |
| Canonical-certificate incumbent initialization | `SCOLHKG/Real/BoundaryCoordinateSufficiency.lean` | V32 Lean-proves the exact limited claim: selecting an index from the canonical certified set yields nonpositive upper margin, while an empty certified set admits no certified initializer. V48 adds an option-valued runtime contract: `some x` must carry certified-set membership, while `none` is valid exactly for an empty set; no protected incumbent may be fabricated in that case. The ordinary posterior Bayes fallback remains an optimization action, not a safety claim. |
| Central-HVD Bayes action and binary chance-failure loss | `SCOLHKG/Real/BoundaryCoordinateSufficiency.lean` | V49 formally separates the posterior-central aleatoric margin used for Bayes ranking from the upper aleatoric margin used for certification. Changing the decision variance cannot alter the certificate, and an upper variance plus nonnegative epistemic radius dominates the central margin. The binary chance-failure Bayes risk is monotone in posterior failure probability and adds a nonnegative penalty. Empirical calibration and the evaluate-or-replicate action mix remain gate conditions. |
| Posterior-nominal versus KL-robust decision risk | `SCOLHKG/Real/BoundaryCoordinateSufficiency.lean` | V50 writes robust decision risk as posterior-nominal risk plus a nonnegative ambiguity premium. With nonnegative violation penalty and ambiguity premium, nominal Bayes risk cannot exceed the KL-robust risk. The certification upper margin is definitionally independent of this decision-only choice. Empirical rank validity remains a paired gate. |
| Nested finite evaluate-or-replicate actions | `SCOLHKG/Real/BoundaryCoordinateSufficiency.lean` | V51 replaces the arbitrary one-new-point discretization by a nested posterior-only finite shortlist. If the smaller action set is a subset of the larger one and the reported large-set action maximizes VOI, `finite_action_set_expansion_cannot_reduce_best_voi` proves its value is at least that of the small-set maximizer. This is an action-approximation theorem, not an empirical claim that more new points improve regret. |
| Predictive/confidence covariance separation | `SCOLHKG/Real/BoundaryCoordinateSufficiency.lean` | V35 Lean-proves that the robust confidence covariance dominates the central scaled-hyperlaw covariance in every scalar projection, while a confidence-only correction leaves the Bayes-ranking variance unchanged. Python routes central uncertainty only to posterior expected-violation ranking and routes robust uncertainty to the canonical theory certificate and Cantelli switch variance. This is an authority split inside one charged-data estimator, not two independently fitted mean heads. Empirical safety and nonvacuity remain paired gate criteria. |
| Shared-mean plus finite-source predictive low-rank hyperlaw | `SCOLHKG/Real/SharedLowRankSourceHyperlaw.lean` | V42 Lean-proves nonnegative weighted shared-estimation variance, nonnegative between-domain/factor-projected discrepancy, exact separation of source-fit uncertainty from target variation, and contraction relative to the legacy within-source transfer. V43 proves that `(1+c)/(1-c)` is nonnegative and at least one for `0 <= c < 1`, that the corrected projected discrepancy dominates its population counterpart, and that the same `Fin (S-1)` factorization and one-source zero are preserved. Python binds this to normalized reliability concentration, PSD covariance projection, exchangeable channel-role covariance, target-only posterior conditioning, and executable multiplier/rank/oracle audits. Held-out calibration and certificate coverage remain empirical gate criteria. |
| Fixed-budget source-task episode capacity | `SCOLHKG/Real/SharedLowRankSourceHyperlaw.lean` | V45 Lean-proves exact simulator-call equality when a fixed per-base-domain record budget is equally partitioned over task episodes, monotonicity of the maximum `S-1` discrepancy-factor capacity in episode count, the two-source one-episode rank-one limit, and nonnegativity of the resulting finite factor projection. Python audits deterministic source-only context draws, exact call equality, frozen episode fingerprints, target-oracle exclusion, and paired target actions. Actual rank gain and held-out calibration remain empirical gate criteria. |
| Grouped within-base source-task hyperlaw | `SCOLHKG/Real/SharedLowRankSourceHyperlaw.lean` | V46 Lean-proves nonnegative centered within-base task variance, invariance of episode contrasts to common base offsets, exact separation of shared-estimation from role/between/within projected variance, and grouped discrepancy capacity `(B-1)+B(E-1)`. Python builds each block from frozen source episodes, applies separate finite-task predictive corrections, projects the final covariance to PSD, and audits grouping, ranks, source cost, target-oracle exclusion, and paired target traces. Held-out calibration, certificate nonvacuity, and performance remain strict empirical criteria. |
| Random-effects fit-noise deconvolution | `SCOLHKG/Real/SharedLowRankSourceHyperlaw.lean` | V47 Lean-models every fixed-direction PSD correction as `max(observed-fitNoise,0)` and proves nonnegativity, contraction relative to a nonnegative observed variance, exact recovery when `observed=latent+fitNoise`, and removal of noise-dominated directions. Python analytically propagates episode-fit covariance into channel-role, between-base, and within-base noise blocks, subtracts before separate PSD projections, keeps shared-mean estimation uncertainty positive, and audits every observed/noise/corrected trace. Matrix-valued latent recovery beyond these projected contracts and held-out performance remain empirical gate criteria. |
| Source-informed constraint-mean posterior | `SCOLHKG/Real/SourceConstraintMeanPosterior.lean` | Lean-proved nonnegativity of hierarchical source directional variance, exact scalar conjugate updating, target-information variance contraction, nonnegative finite-mixture moment variance, preservation of between-source disagreement, nonnegative normalized sequential evidence weights, and inheritance of chance-certificate soundness |
| TCB-V2 hierarchical three-layer certificate | `SCOLHKG/Real/HierarchicalBoundaryCertificate.lean` | Lean-proved positive target scale, planar-rotation norm preservation, nonnegative Cholesky/rotation/orthogonal-residual uncertainty, upper-margin non-relaxation, coverage-reserved frontier order, one shared frontier/terminal/recommendation upper margin, recommendation safety under coverage, and lexicographic terminal dominance |
| TCB-V3 finite boundary-family certificate | `SCOLHKG/Real/BoundaryFamilyMixtureCertificate.lean` | Lean-proved posterior credible-mass contract, pointwise family-envelope coverage, nonnegative family-guard monotonicity, safe recommendation, target-name noninterference, and the combined `delta_family + alpha` failure bound |
| TCB-V4 continuous boundary-family synthesis | `SCOLHKG/Real/BoundaryFamilySynthesisCertificate.lean` | Lean-proved monotonicity of nonnegative source-atom synthesis, nonnegative coefficient/residual predictive variance, upper-margin non-relaxation, recommendation safety on the coverage event, and target-name noninterference |
| TCB-V5 orthogonal semiparametric boundary | `SCOLHKG/Real/BoundaryFamilySemiparametricCertificate.lean`, `SCOLHKG/Real/OrthogonalSemiparametric.lean` | Lean-proved source-design nullspace orthogonality, direct-sum predictive mean bookkeeping, nonnegative synthesis/residual/noise variance, upper-margin non-relaxation, and recommendation safety on the coverage event |
| Source-affine boundary transfer | `SCOLHKG/Real/SourceAffineBoundaryTransfer.lean` | Lean-proved exact error decomposition for one frozen source boundary atom, positive-scale safe-side preservation, a uniform offset/scale transfer radius, and sound target certification when that radius is included in the upper margin |
| Source-rank observable coordinate | `SCOLHKG/Real/SourceConsensusCommit.lean` | Lean-proved invariance of domainwise ranks under strictly increasing margin transformations, unit-interval closure of normalized nonnegative rank interpolation, and a finite `3/2` upper bound for the implementation's consensus score |
| Noise-limited oracle certifiability | `SCOLHKG/Real/OracleCertifiability.lean` | Lean-proved nonnegative and replication-monotone oracle radius, squared-budget sufficiency, and certificate persistence under added replications |
| Source-consensus proposal and suffix commitment | `SCOLHKG/Real/SourceConsensusCommit.lean` | Lean-proved source-rank invariance under strictly increasing domainwise rescaling, weak and strict Pareto monotonicity of the safety-objective source score, domain noninterference and membership preservation of the shared source design, target-name noninterference of a source-frozen selector, bounded-error preservation of two-arm order, and exact shortlist completion within a sufficient reserved replication budget |
| HVD oracle inequality | `SCOLHKG/Real/HVD.lean` | Lean-proved from residual-square concentration event |
| HVD implementation guards | `SCOLHKG/Real/HVDImplementation.lean` | Lean-proved for residual squares, nonnegative linear variance, clipping, and certification variance |
| Factor cumulative block implementation | `SCOLHKG/Real/CumulativeRiskImplementation.lean` | Lean-proved for `floor/independent/shared/linear/total` aggregation and shared-shock omission underestimation |
| Ridge-HVD oracle inequality | `SCOLHKG/Real/RidgeHVD.lean` | Lean-proved from ridge minimizer and uniform residual-square concentration |
| Source-prior HVD calibration | `SCOLHKG/Real/RidgeHVD.lean`, `SCOLHKG/Real/HVDImplementation.lean` | Lean-proved nonnegativity for prior-centered penalty, hierarchical variance scale, and within-policy sample variance |
| Hierarchical source-shape HVD | `SCOLHKG/Real/SourceShapeMixtureHVD.lean` | Lean-proved nonnegative source-shape mixing, a nonnegative target-pooled null component, preservation of nonnegative HVD shape under a shared latent-task posterior, monotone shrinkage of scalar posterior shape variance/radius under added target information, conservative-in-expectation prequential squared-innovation evidence via the exact variance-plus-squared-bias identity, and nonnegative square-root radius reduction for a unit-coherent joint VOI |
| Joint KL task certificate | `SCOLHKG/Real/JointKLChanceCertificate.lean` | Lean-proved centered-second-moment control of mixture epistemic variance, the positive square-root tangent bound, one-common-task-law robustification, and finite-grid minimum closure |
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

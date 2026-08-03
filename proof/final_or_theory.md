# Final Operations Research Theory Contract

## 1. Decision Problem And Information Contract

For a held-out task, let `x` be an integer policy, `F(x)` its random operating
cost, and `G(x)` its random constraint response.  The deployment problem is

```text
minimize E[F(x)]  subject to  P(G(x) <= 0) >= 1 - alpha.
```

Before any target response is observed, the method may use a frozen source
archive and target descriptors such as dimension, bounds, and an unlabeled
policy/state schema.  It may not use target objective values, target
constraint values, a target optimizer, target feasibility labels, or terminal
verification responses.  Source, target-search, and verification calls are
separate resources.

The final method has three mathematical objects:

1. a source-frozen deterministic proposal atlas `A_n0` of at most `n0`
   policies;
2. a replaceable target optimizer operating only after those `n0` calls;
3. an independent ordered verifier applied to a shortlist frozen after search.

The online optimizer is not part of the novelty theorem.  This separation is
intentional: experiments show that the atlas contributes most of the observed
gain, while canonical SAASBO provides a smaller target-stage improvement.

## 2. Why A Transfer Assumption Is Necessary

**Theorem 1 (finite-atlas no-free-lunch).**  For every proper finite proposal
atlas that is frozen before target labels, there exists a nonempty held-out
feasible set that the atlas misses.

**Proof.**  Choose any policy outside the finite atlas and define the held-out
feasible set to contain only that policy.  The feasible set is nonempty and
has empty intersection with the atlas.  Therefore no finite target-label-free
proposal can guarantee arbitrary target coverage without an explicit
structural or source-to-target condition.  The Lean proof is
`SCOLHKG.Real.finite_budget_no_unconditional_target_coverage`.

This theorem is the reason the paper states a bounded transfer condition
instead of presenting ten proposals in ten thousand dimensions as an
unconditional feat.

## 3. Deterministic Atlas Coverage

Let `eta_star(x)` be an ideal transferable coordinate and `eta_hat(x)` its
source-learned approximation.  Let `S` be the frozen source support and `A_n0`
the maximin atlas selected from that support.  Assume:

1. **finite coverage:** every `s in S` is within `r_cover` of an atlas point in
   `eta_hat`;
2. **coordinate approximation:**
   `dist(eta_hat(x), eta_star(x)) <= epsilon_eta`;
3. **source-target support:** a target-safe center `x_c` is within
   `Delta_task` of some source-support point in the ideal coordinate;
4. **one-sided margin regularity:** the target chance margin `m(x)` obeys
   `m(x) <= m(y) + L dist(eta_hat(x), eta_hat(y))`;
5. **safe depth:** `m(x_c) + gamma <= 0`;
6. **radius condition:**

```text
L (r_cover + Delta_task + 2 epsilon_eta) <= gamma.
```

**Theorem 2 (aligned finite-atlas coverage).**  Under assumptions 1--6 and
`|A_n0| <= n0`, at least one atlas policy is target feasible.

**Proof.**  Assumptions 2 and 3 plus two triangle inequalities put a learned
source proxy within `Delta_task + 2 epsilon_eta` of the target-safe center.
Assumption 1 supplies an atlas member at an additional distance at most
`r_cover`.  The one-sided Lipschitz condition then raises its chance margin by
at most `L(r_cover + Delta_task + 2 epsilon_eta)`, which is no larger than the
safe depth.  Its chance margin is therefore nonpositive.  The Lean proof is
`SCOLHKG.Real.finite_aligned_geometric_lipschitz_atlas_coverage`; the composed
implementation theorem is
`SCOLHKG.Real.paper_frontend_aligned_geometric_atlas_and_certificate`.

**Corollary 2.1 (nominal-dimension independence).**  The sufficient condition
depends on the covering geometry of `eta`, not the raw policy dimension `d`.
Increasing `d` does not change the theorem when `r_cover`, `Delta_task`,
`epsilon_eta`, `L`, and `gamma` remain controlled.

The global Lipschitz condition is a declared model assumption.  The current
finite synthetic audit verifies atlas coverage on the registered libraries but
does not estimate a globally valid `L`; no unconditional global-coverage claim
is permitted.

## 4. V3 Source-Monotone Envelope

The V3 challenger uses the normalized zero-frequency policy coefficient.  For
each source task, it computes the rank correlation between that coefficient
and observed chance margin.  It admits the upper endpoint only when at least
two sources all have correlation at most `-kappa`; it admits the lower endpoint
only when they all have correlation at least `kappa`.  Otherwise it returns no
endpoint.

**Theorem 3 (fail-closed identity).**  If source directions do not satisfy an
admission rule, V3 returns the V1 atlas exactly and preserves its cardinality.

**Proof.**  The proposal update is a Boolean branch.  Its false branch is the
identity, and the true branch replaces one existing slot.  The Lean theorems
are `rejected_envelope_preserves_baseline` and
`fail_closed_envelope_preserves_budget`.  The end-to-end paper bridge is
`paper_final_v3_fail_closed_contract`.

**Theorem 4 (transferred endpoint safety).**  Suppose the source tasks agree on
the negative direction and the held-out chance margin is nonincreasing in the
same coordinate.  If a feasible target policy exists and the admitted upper
endpoint is coordinate-wise maximal, then the upper endpoint is feasible.
The symmetric statement holds for positive agreement and the lower endpoint.

**Proof.**  Let `x_s` be a feasible witness.  Maximality gives
`eta(x_s) <= eta(x_upper)`.  Transferred nonincreasing monotonicity gives
`m(x_upper) <= m(x_s) <= 0`.  The lower-endpoint proof is identical with the
order reversed.  The Lean theorem is
`paper_final_v3_admitted_endpoint_contract`.

Source rank agreement is not silently equated with target monotonicity.  It is
the source-only admission statistic; transferred monotonicity is the explicit
identifiability condition.  In the original three domains V3 rejected and was
exactly V1 in all 60 registered cells.  In the energy family it admitted before
the confirmatory market was opened, after which the untouched market supplied
empirical evidence for the transfer condition.

## 5. Independent Terminal Verification

The target optimizer freezes an ordered shortlist containing an objective
challenger, its primary recommendation, and a safe-interior support policy.
Independent samples are never returned to the optimizer.

### 5.1 Synthetic Gaussian verifier

Each candidate receives a one-sided mean/scale certificate with error
allocation `delta_j`.  If the candidate-wise false-certificate probabilities
are bounded by their allocations, then the probability that the first
certified policy in the ordered shortlist is unsafe is at most
`sum_j delta_j`.  This is proved in
`SCOLHKG.Measure.optimizer_agnostic_three_policy_false_deployment_probability_le`.

The objective challenger replaces an independently safe incumbent only when a
one-sided upper confidence bound on the paired objective difference is below
zero.  On upper-bound coverage the switch is correct.  If coverage fails with
probability at most `delta_obj`, the wrong-switch probability is at most
`delta_obj`.  The Lean theorems are
`objective_guard_switch_is_correct_on_upper_coverage` and
`false_objective_switch_probability_le`.

**Theorem 5 (joint terminal contract).**  Unsafe deployment or an incorrect
objective switch occurs with probability at most
`delta_safe + delta_obj`.  The implementation-matched Lean theorem is
`optimizer_agnostic_three_policy_and_objective_guard_failure_le`.

### 5.2 External exact-binomial verifier

For a frozen policy with true feasible-window probability `p`, `n`
independent verification windows produce `K ~ Binomial(n,p)`.  The all-success
event has exact probability `p^n`; if `p <= p_required`, its probability is at
most `p_required^n`.  Finite error spending over a frozen shortlist gives the
familywise bound.  These facts are machine checked in
`SCOLHKG.Measure.ExactBinomialCertificate` using mathlib's binomial law and
`HasLaw` bridge.

The registered 80-replication first-stage rule at required probability 0.95
certifies only an all-success count.  The implementation computes the
one-sided Clopper--Pearson lower bound; the threshold equivalence is regression
tested numerically, while the probability statement is proved from the exact
binomial mass.

## 6. Budget Identity

For source calls `S`, target search calls `N`, and independent verification
calls `V`, total cost is exactly

```text
S + N + V.
```

The theorem `paper_grade_budget_exact_decomposition` formalizes this identity.
No result may label `N=13` as thirteen total evaluations: it means thirteen
target search calls after a 384-call reusable source archive, with verification
reported separately.

## 7. Optional Cumulative Heteroscedastic Calibration

The cumulative decomposition

```text
Var(C | T) = A(T)^T Lambda A(T) + N(T)^T B N(T) + N(T)^T omega
```

and its certification bridges remain Lean-proved.  However, the paired
20-seed causal experiment improved variance RMSE and variance-shape
correlation in all three domains without improving feasibility or regret, and
increased verification cost in Inventory and Queue.  HVD is therefore an
optional calibration diagnostic, not a primary contribution.  This empirical
demotion does not invalidate the decomposition theorem; it limits the paper's
optimization claim.

## 8. Machine-Checked Scope

The final proof build contains no `sorry`, `admit`, or project-defined `axiom`.
The main files are:

| Claim | Lean file |
|---|---|
| No unconditional finite coverage | `Real/ProposalNoFreeLunch.lean` |
| Deterministic aligned atlas coverage | `Real/GeometricAtlasCoverage.lean` |
| Source-label noninterference | `Real/RiskAlignedRepresentation.lean` |
| V3 fail-closed and endpoint transfer | `Real/SourceMonotoneEnvelope.lean` |
| Three-policy safety and objective guard | `Real/MethodIndependentTerminalVerification.lean` |
| Exact binomial all-success certificate | `Measure/ExactBinomialCertificate.lean` |
| Final composed interfaces | `Real/PaperMainline.lean` |

Lean proves the stated implications.  Empirical assumptions such as bounded
source-target discrepancy and target monotonicity remain scientific
obligations; they are not converted into axioms or claimed to hold for every
possible domain.

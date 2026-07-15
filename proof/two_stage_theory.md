# Two-Stage SC-OLH-KG Decision Theory

## 1. Scope and claim

The deployed optimizer is not an exact-KG policy for all `N` evaluations.  It
is a predeclared two-stage simulation-optimization policy:

```text
Stage I:  source-informed design plus state-coupled KG, t < N - R;
Stage II: finite heteroscedastic ranking-and-selection, N - R <= t < N.
```

The main theoretical claim is therefore:

> State-coupled cumulative-risk coordinates support a safe two-stage policy in
> which KG constructs a finite candidate universe and charged terminal
> replications verify a small finalist set.  Certified outputs inherit a
> high-probability chance-feasibility guarantee; uncertified outputs are
> explicitly reported as least-risk fallbacks and receive a relative-risk, not
> a safety, guarantee.

This statement describes the oracle-free source-consensus baseline.  The old
V32 teacher line is a privileged upper bound and is not the main theorem's
information regime.

## 2. Decision problem

For integer policy `x in X`, one simulator call returns

```text
Y_f(x) = f(x) + epsilon_f(x),
Y_g(x) = m_g(x) + epsilon_g(x).
```

The constraint noise is heteroscedastic.  Its cumulative variance is modeled
in the shared risk coordinate `psi(x) = (A(x), N(x))`:

```text
v_C(x) = A(x)^T Lambda A(x) + N(x)^T B N(x) + N(x)^T omega + floor.
```

The chance constraint is

```text
m_g(x) + z_alpha sqrt(v_C(x)) <= tau.
```

The implementation uses the conservative posterior margin

```text
U_post,t(x)
  = m_hat_g,t(x) + sqrt(beta_t) s_g,t(x)
    + z_alpha sqrt(v_C,t^+(x)) - tau.
```

The mean coordinate and the cumulative-risk coordinate need not coincide.
They meet only in `U_post,t`.

## 3. Information contract

The frozen source archive may determine representation hyperparameters,
source-consensus proposal ranks, and coefficient/HVD priors.  It may not query
the held-out target truth, optimum, analytic noise, or problem hooks.  Target
posteriors are updated only by the `N` charged target calls.

The source archive affects candidate coverage and search quality.  It does not
make a target candidate safe.  Safety always requires a target posterior or
replication coverage event.  Consequently source mismatch appears as a
proposal error in the regret theorem, not as an assumed certificate.

## 4. Budget split

Let `R <= N` be fixed before target outcomes are observed.  Define

```text
T_search = N - R,
T_verify = R.
```

Every stage `t < N` belongs to exactly one interval. Within Stage I, the first
`min(n0,N-R)` calls are initial design and the remainder are adaptive search.
Lean proves the three-way identity

```text
min(n0,N-R) + (N-R-min(n0,N-R)) + R = N,
```

all stage partitions, and their disjointness. The final `R` calls remain inside
the original target budget; no verification call is free.

## 5. Stage I: initial design and state-coupled KG search

The first `n0` charged target calls form the frozen source-informed initial
design. For adaptive histories `n0 <= t < N-R`, slots assigned to the main
acquisition use exact-MC SC-OLH-KG to evaluate the expected change in a
posterior certified terminal value over one history-measurable candidate pool.
Every fantasy clones and updates the objective GPR, constraint GPR, cumulative
HVD, and finite task posterior. Any separately predeclared recheck action must
be reported as such and is not covered by the one-step exact-KG claim.

The existing posterior-update theorem establishes one-step optimality for each
adaptive KG decision when the expectation is exact. Uniform MC error `eta`
gives the existing `2 eta` maximizer gap. It does not characterize the initial
design or the verification suffix as exact KG.

Let `x_S` denote the best truly feasible design represented by the search
universe at the switch.  Write

```text
f(x_S) - f(x_star) <= epsilon_search.
```

Information-gain, finite candidate coverage, and exact-MC approximation terms
may be used to upper-bound `epsilon_search`.

## 6. Stage II: confirmatory ranking-and-selection

Before observing suffix labels, freeze a finite universe `U`.  Later expert
nominations and committed source-consensus challengers must remain in `U`.
Only candidates satisfying the declared replication count enter the completed
finalist set `F`.

For finalist `x` with `r_x` replicates, define

```text
U_rep(x)
  = ybar_g(x) + z_alpha sigma_plus(x)
    + z_delta sigma_plus(x) / sqrt(r_x) - tau.
```

This is a certificate only on the joint event

```text
m_g(x) <= ybar_g(x) + z_delta sigma_plus(x) / sqrt(r_x),
sigma_g(x) <= sigma_plus(x).
```

Adaptive finalist selection is handled by simultaneous coverage over the
pre-observation finite universe, not by pretending the chosen arm was fixed.
The union allocation therefore remains valid after history-measurable
nomination and commitment.

## 7. Terminal output semantics

The terminal report has exactly two theorem-level statuses.

### 7.1 Certified

A certified report selects `x_hat in F` with upper margin at most zero and
minimum estimated objective among all finalists satisfying that condition.
On the simultaneous coverage event,

```text
true_margin(x_hat) <= upper_margin(x_hat) <= 0.
```

Thus `x_hat` is truly chance feasible.

### 7.2 Uncertified fallback

If no finalist has a nonpositive upper margin, the fallback minimizes upper
margin before objective.  It does not claim chance feasibility.  If upper
margins have uniform error at most `epsilon_g`, then for every comparator
`x in F`,

```text
true_margin(x_fallback)
  <= true_margin(x) + 2 epsilon_g.
```

This is a useful least-risk guarantee but is deliberately not rewritten as a
certificate.  The Python result field `fallback_claims_certification=false`
records this distinction.

## 8. Verification selection error

Let `x_F` be the best strictly safe comparator retained in `F`.  Suppose

```text
true_margin(x_F) <= -epsilon_g,
|U(x_F) - true_margin(x_F)| <= epsilon_g,
|f_hat(x) - f(x)| <= epsilon_f, x in {x_hat, x_F}.
```

Then `x_F` remains empirically certified, and estimated-objective minimization
implies

```text
f(x_hat) <= f(x_F) + 2 epsilon_f.
```

Lean proves both the strict-safety preservation lemma and the `2 epsilon_f`
selection lemma without asymptotic assumptions.

## 9. Deterministic two-stage regret theorem

Introduce three separately auditable errors:

```text
f(x_S)   - f(x_star) <= epsilon_search,
f(x_F)   - f(x_S)    <= epsilon_proposal,
f(x_hat) - f(x_F)    <= epsilon_verify.
```

The middle term measures source-prior/candidate-retention mismatch.  It is zero
or negative when a search-quality comparator is retained and the finalist is
no worse.  It can be positive when the transferable proposal misses the useful
target structure.

Telescoping gives

```text
f(x_hat) - f(x_star)
  <= epsilon_search + epsilon_proposal + epsilon_verify.
```

On the terminal upper-coverage event, a certified report additionally satisfies

```text
true_margin(x_hat) <= 0.
```

This combined statement is Lean theorem
`two_stage_certified_safe_regret`.

## 10. High-probability theorem

Let the three bad events be

```text
B_search:       Stage-I surrogate/KG guarantee fails,
B_proposal:     required safe comparator is not retained,
B_verification: a finalist margin or objective interval fails.
```

No independence is required.  If their probabilities are bounded by
`delta_S`, `delta_P`, and `delta_V`, then

```text
P(B_search union B_proposal union B_verification)
  <= delta_S + delta_P + delta_V.
```

For a frozen finalist universe, centered sub-Gaussian margin and objective
errors give

```text
delta_V
  <= sum_{x in U} delta_margin(x)
     + sum_{x in U} delta_objective(x).
```

Outside this union, the certified output is safe and satisfies the deterministic
regret bound.  Lean theorem
`two_stage_safe_regret_failure_probability_le` proves the event transfer.

## 11. Relation to the four meta-priors

The four learned structural assumptions enter identifiable theorem terms:

| Meta-prior | Primary theorem term |
|---|---|
| Low frequency | `epsilon_search`, through surrogate sample efficiency |
| Orthogonality | `epsilon_search` and interval radius through conditioning |
| Coefficient sparsity | `epsilon_search`, through effective dimension |
| Additivity | `epsilon_proposal` and approximation bias |

They do not directly reduce `delta_V` unless they produce a valid conservative
variance bound.  This separation is why their empirical contributions require
the pre-registered drop-one-prior ablation rather than interpretation of model
weights.

## 12. What is and is not closed

The former `mathematically_closed` diagnostic asks whether every stage uses one
certified lexicographic exact-KG terminal value.  The promoted oracle-free
baseline correctly fails that narrower test.

The new two-stage implementation contract asks instead whether:

1. `N-R` search calls and `R` verification calls partition the charged budget;
2. the finalist universe is frozen before verification labels;
3. every verification target belongs to that universe;
4. terminal selection is either certified or explicitly uncertified;
5. no target oracle enters either stage.

These conditions match the two-stage theorem and can hold even though the
global exact-KG claim is false.  Statistical coverage remains an explicit
assumption/event; implementation closure must never be reported as empirical
coverage.

## 13. Formal proof inventory

- `SCOLHKG/Real/TwoStageDecision.lean`: budget partition, terminal report
  semantics, certified safety, fallback non-certification, strict-margin
  preservation, objective `2 epsilon` selection, fallback relative-risk bound,
  and the deterministic three-term regret theorem.
- `SCOLHKG/Measure/TwoStageDecision.lean`: finite-universe margin/objective
  concentration, uniform-error extraction, three-event union bound, and
  high-probability safe-regret transfer.
- `SCOLHKG/Real/FinalistReplication.lean`: replicated-margin implementation
  bridge, fixed/adaptive archive invariants, completed-arm filtering, and
  familywise finite selection.
- `SCOLHKG/Real/SourceConsensusCommit.lean`: source-rank invariance and exact
  two-challenger completion within the reserved budget.
- `SCOLHKG/Measure/PosteriorUpdateKG.lean` and
  `SCOLHKG/Measure/ExactMCConcentration.lean`: Stage-I exact target and finite-MC
  approximation.

All formal statements are theorem-level consequences with no `sorry`, `admit`,
or project-local `axiom`.

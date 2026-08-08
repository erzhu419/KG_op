# Final Operations Research Theory Contract

This file is the human-readable theory contract for the final manuscript. The
paper's method is the V2 source-scored structural initial design, not historical
V3 endpoint replacement, V69, KG, SAASBO, or HVD.

## 1. Decision and information contract

The target decision is an ordered bounded profile `h:[0,1]->[0,1]` observed on
a declared grid and deterministically mapped to an integer policy. The target
problem is

```text
minimize E[F_q(h)] subject to P(G_q(h) <= tau_q) >= 1-alpha_q.
```

Before target search, the frontend may use the ordered target grid, integer
bounds, nominal dimension, channel order in declared-schema experiments, a
public outcome-free profile library, and ordinary replicated source outcomes.
It may not use target outcomes, target feasibility labels, target oracle
parameters, target optimum, safe-basin geometry, or terminal verification
samples.

The final algorithm has three interfaces:

1. a source-scored initial design frozen before target outcomes;
2. a replaceable target backend;
3. a frozen-shortlist verifier using independent replications.

Source, search, and verification calls are separate resources.

## 2. Exact profile coordinate and cross-grid consistency

The implemented linear coordinate is the exact cosine integral of the
Voronoi piecewise-constant reconstruction:

```text
c_0 = sum_i h_i (e_i-e_{i-1})
c_k = sqrt(2) sum_i h_i [sin(pi k e_i)-sin(pi k e_{i-1})]/(pi k).
```

Each `c_k` is divided by `1+0.25k`; all nine diagonal squares are appended and
no cross-products are used.

`Real/ProfileCoordinateConsistency.lean` proves:

- a coefficient error at most `epsilon*basisBound` under sup-norm profile
  reconstruction error `epsilon`;
- the `constant*basisBound/d` inverse-grid rate;
- the regular midpoint/Voronoi `L/(2d)` profile rate for Lipschitz profiles;
- the adjacent-grid convex-interpolation radius needed for the implemented
  linear inverse map.

These results justify refinement of one ordered profile grid. They do not
justify treating arbitrary unordered coordinates as a profile.

## 3. Farthest-first finite-library coverage

For a finite library with coordinate `eta`, let

```text
r(A)=max_{h in L} min_{a in A} dist(eta(h),eta(a)).
```

`Real/FarthestFirstKCenter.lean` proves the standard factor-two guarantee from
the checkable Gonzalez witness emitted by Python: the selected `k` centers plus
one farthest witness are pairwise separated by the achieved radius; among any
`k` optimal clusters, two of the `k+1` witnesses share one cluster, and the
triangle inequality gives `r_greedy <= 2 r_optimal`. A separate zero-radius
result covers an exact finite cover.

## 4. Replicated source-score recovery

`Measure/SourceRankRecovery.lean` proves a finite-profile sub-Gaussian union
bound for replicated source means. `Real/SourceRankRecovery.lean` proves the
exact `ddof=1` finite-sum variance identity, propagates declared mean and scale
error radii through the floored Gaussian chance-margin statistic, and proves
that pairs separated by more than twice the uniform score error retain their
order.

The theorem does not assert that three replications recover near ties. Numeric
noise proxies and residual-square radii are experiment assumptions and are
tested by the source-replication sensitivity matrix.

## 5. Conditional source-to-target coverage

Let `eta_star` be an ideal target-relevant coordinate and `eta_hat` the
implemented coordinate. Assume:

1. the selected atlas covers the finite library within `r_cover` in `eta_hat`;
2. implemented and ideal coordinates differ by at most `epsilon_eta` on the
   profiles used in the proof;
3. a library member is within `Delta_task` of a target safe center in the ideal
   coordinate;
4. the target chance margin is `L`-Lipschitz in the ideal coordinate;
5. the center has safety depth `gamma`;
6. `L*(r_cover+Delta_task+2*epsilon_eta) <= gamma`.

Then a selected atlas member is target feasible. The finite geometric and
Lipschitz implications are proved in `Real/GeometricAtlasCoverage.lean`; the
composed frontend interface is exposed by `Real/PaperMainline.lean`.

This is a conditional theorem. It does not explain away adverse regimes:
frequency shift, irregular grids, sparse high-frequency activity, and full
misspecification can enlarge discrepancy or coordinate error beyond safe
depth.

## 6. Task-law calibration

For a fixed atlas and independent held-out tasks from one declared task law,
let each hit indicator lie in `[0,1]`. `Measure/TaskAtlasCoverage.lean` proves a
sub-Gaussian mean-error bound with proxy `1/(4m)` for `m` independent tasks, and
also an exact all-success false-coverage-claim bound. The manuscript uses
task-level bootstrap intervals and scopes every conclusion to the registered
randomized law. Repeated simulation seeds within one external market are not
treated as independent domains.

## 7. Exact terminal verification

For a frozen candidate with true feasibility probability `p`, the all-success
event in `v` iid Bernoulli replications has probability `p^v`. If the required
probability is `p0`, an unsafe candidate satisfies

```text
P(certify unsafe candidate) = p^v <= p0^v <= delta_j.
```

Bonferroni spending over a frozen shortlist gives familywise false deployment
at most `sum_j delta_j`. `Measure/ExactBinomialCertificate.lean` proves the
binomial-law bridge, candidate bound, and three-candidate composition. The
primary rule uses `p0=.95`, three candidates, familywise `.05`, and 80
all-success trials per candidate. Its power is `p^80`, so validity and
nonvacuity are reported separately.

The Energy statement is conditional on the declared empirical-window
replication distribution. Postdecision nonoverlap and block audits are
descriptive and do not create an iid future-calendar theorem.

## 8. Budget identity

The finite accounting identity is

```text
C_all = S + N + V,
C_amort(M) = S/M + N + V.
```

Equal preverification cost matches `S+N`, not total cost. Every displayed
result reports realized verification calls separately.

## 9. Negative and optional results

`Real/ProposalNoFreeLunch.lean` proves that every proper finite target-label-
free atlas misses some nonempty feasible set. This is a scope remark, not the
main novelty theorem.

The cumulative-HVD proofs remain valid mechanistic results, but the matched
experiment improved variance calibration without improving feasible recovery,
regret, false certification, or verification cost. HVD is therefore optional
appendix material and not part of the final optimization claim.

## 10. Machine-checked scope

The final `lake build` contains no `sorry`, `admit`, or project-defined
`axiom`. Lean verifies finite implications from declared premises. It does not
prove that a future system belongs to the registered task law, that source and
target are aligned, or that time-series windows are iid. Those remain empirical
and operational obligations.

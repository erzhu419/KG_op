# V23 Orthogonal Semiparametric Ordered Expert

## Empirical Motivation

V22 proves that the ordered diagonal coordinate has useful target signal: it
reduces Inventory mean violation to `0.00585` and keeps every failed seed close
to the true boundary.  It also proves that replacing the local residual model
is unsafe: FactorShock falls from 7/7 to 5/7 feasible.  Candidate coverage,
rank-cap enforcement, and finite-expert identity noise have already been
excluded as primary explanations.

## Single-Model Hypothesis

Use one expert with the direct-sum mean model

`g(x) = theta^T phi_ordered(psi(x)) + gamma^T k_perp(psi(x)) + delta_x`,

where `phi_ordered` is the frozen source-learned ordered diagonal coordinate,
`k_perp` is a local RBF dictionary residualized against `[1, phi_ordered]` on
a fixed unlabeled target-policy pool, and `delta_x` is the existing sampled
solution deviation.  The empirical projection identity is

`Phi^T K_perp = 0`.

This is one identifiable semiparametric model, not an additional expert.  The
same ordered `psi=(A,N)` continues to define factor-HVD, certification,
candidate proposals, and exact KG updates.

## Implementation

- Add `ordered_semiparametric` to `FixedTaskExpertBasis`.
- Build the existing six-center local RBF dictionary from source-only and
  unlabeled universal policy shapes.
- Fit a fixed least-squares projection of the RBF dictionary onto an intercept
  plus ordered features; expose only the residual features.
- Concatenate ordered and residual features inside one expert.
- Extend source PIP/slab vectors with generic local-residual priors and retain
  the total effective-dimension cap at `0.35 N`.
- Replace `local_risk_kernel` with `ordered_semiparametric`; do not keep a
  separate `ordered_cumulative` expert in the same gate cell.
- Keep IID MC2, the V18b manifest, all recommendation/certification settings,
  and the six-expert count fixed.

## Diagnostics

- finite-pool orthogonality error `||Phi^T K_perp||_F`;
- ordered and local residual feature dimensions;
- posterior effective dimension and local residual inclusion mass;
- expert posterior mass, true margin, and pool-safe ranking by seed;
- factor-HVD coordinate/provider identity remains ordered cumulative risk.

## Falsifiable Gate

- FactorShock must return to 7/7 feasible with zero mean violation;
- Inventory must reach at least 4/7 feasible, at most one false-feasible, and
  retain mean violation below the V18b value `0.01877`;
- effective dimension must not exceed its reported cap;
- every decision remains offline and target-oracle-free;
- Queue remains unopened until both held-out domains pass.

If V23 preserves FactorShock but Inventory stays below 4/7, the next audit is
source-to-target group/subspace inclusion alignment.  An ordered HVD prior is
allowed only after mean/ranking diagnostics improve while variance remains the
limiting error.

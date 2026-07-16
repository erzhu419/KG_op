# Structural Meta-Prior, Cumulative HVD, and Backend Plan

## Objective

The paper mainline is an acquisition-agnostic transfer method:

`source archive -> structural meta-prior -> psi=(A,N) -> cumulative HVD posterior -> decision backend`.

The frozen source-informed proposal is a first-class learned output of the
meta-prior. It is admissible when it uses source observations only. Its value
must nevertheless be separated from target-online adaptation.

## Unified structural prior

The four assumptions are properties of one hierarchical prior,

`g(x) = sum_{j in S} w_j phi_j(psi(x)) + epsilon(x)`,

not independent engineering modules:

1. Low frequency: prior energy decays with graph/Fourier frequency.
2. Orthogonality: the active dictionary is identifiable in a whitened basis.
3. Sparse coefficients: only a small posterior subset of coefficients is active.
4. Additivity: the active dictionary factors into a small number of blocks and
   low-order interactions.

The source archive learns frequency scales, shrinkage, block probabilities,
and proposal mass. Charged target observations update the corresponding
posterior without target oracle labels.

## Cumulative heteroscedasticity

The same risk coordinate defines the cumulative variance model

`v_C(x) = floor + A(x)^T Lambda A(x) + N(x)^T B N(x) + N(x)^T omega`.

Pointwise variance is an ablation. The main model uses cumulative HVD, a PSD
shared-shock block, residual uncertainty, and an upper certification radius.

## Decision backends

All backends consume the same posterior and candidate set.

- `random`: random continuation after the frozen initial design.
- `sobol`: deterministic low-discrepancy continuation.
- `risk_ts`: posterior sampling of objective and cumulative chance margin.
- `bayes_risk_ei`: expected improvement in posterior Bayes risk
  `E[f] + rho E[(G)_+]`.
- `constrained_ei`: objective EI weighted by posterior chance feasibility.
- `utility_head`: source-trained Bayesian query-utility head in psi space,
  inspired by MALIBO but trained on the disclosed source archive.
- `exact_mc`: inherited exact posterior-update KG.

KG is retained as a backend ablation, not assumed to be the paper contribution.

## Source discrepancy

Source experts receive a target-updated discrepancy posterior. Source means
affect ranking, while source uncertainty can only increase certification
uncertainty. The posterior must expose source weights, effective source count,
and target residual evidence. No target truth may enter this update.

## Causal experiment design

### Model-only ablation

Use one identical frozen full-prior initial design for every row. Compare:

- no structural prior;
- each prior alone;
- leave-one-prior-out for each of the four priors;
- all four priors;
- all four plus pointwise heteroscedasticity;
- all four plus cumulative HVD.

This estimates posterior/model value without changing initial target points.

### End-to-end ablation

Each prior variant generates its own source-only proposal. This estimates the
total value of representation, proposal, posterior, and adaptation. Every row
records its source archive and initial-design fingerprints.

### Backend ablation

Fix the full structural prior, cumulative HVD, source archive, candidate pool,
and initial design. Change only the decision backend. Report `n0-best` as the
zero-online baseline from the same run.

### Source-discrepancy ablation

Compare frozen equal/source weights with target-updated discrepancy weights
under the same prior, HVD, backend, and initial design.

## Metrics

Every result must report:

- initial true-feasible count and initial best feasible regret, computed only
  after the run for audit;
- final true feasibility and feasible regret;
- adaptive rescue, adaptive loss, and improvement over initial best;
- posterior certificate coverage and false certification;
- source calls, target calls, `D/N_target`, and `D/(N_source+N_target)`;
- backend selection counts and candidate origins;
- wall time and failures.

Conditional regret is never reported without its feasibility denominator.

## Experiment gates

1. Local unit tests and two-seed smoke only.
2. Three domains, `d=50, N=20, n0=10, seeds=5` for all variants.
3. Promote non-dominated variants to 20 seeds.
4. Test `d=200,N=20` and `d=1000,N=20` under common-Sobol and frozen
   source-informed designs.
5. If adaptive gain survives, test `d=1000,N=40` and `d=10000,N=40`.

All scheduler experiments run on `node001-node006`, use one seed per task, and
declare 12 CPU cores. Source archives and initial proposals are immutable and
shared wherever the comparison contract requires them to be shared.

## Promotion criteria

A mainline variant must:

- improve final feasibility or regret beyond `n0-best` rather than merely
  preserve a strong proposal;
- avoid higher false-certification than the current oracle-free baseline;
- retain value on all three held-out domains;
- show increasing or stable relative advantage as `D/N_target` grows;
- preserve source- and target-oracle-free information contracts.

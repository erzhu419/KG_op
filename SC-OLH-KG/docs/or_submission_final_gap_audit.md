# Operations Research Submission Gap Audit

Audit date: 2026-08-03

## Verdict

The evidence and formal-theory packages are complete enough to lock a first
Operations Research manuscript.  There is no remaining experiment or Lean
theorem that is a hard prerequisite for drafting.  This verdict is conditional
on preserving the claim boundaries below; broadening those claims would reopen
the corresponding evidence obligations.

The final method is not SC-OLH-KG as a monolithic algorithm.  It is a three-part
decision protocol:

1. a source-learned, target-label-free, dimension-equivariant structural
   proposal atlas frozen before target outcomes;
2. a replaceable online optimizer, with canonical SAASBO as the strongest
   audited backend;
3. an optimizer-independent terminal verifier that uses fresh samples and does
   not update search.

The paper's primary contribution is the transferable structural frontend and
its coverage/verification contract.  KG, SC-V69, stacked GP, and SAASBO are
backend comparisons.  Cumulative HVD is an optional calibration diagnostic.

## Evidence Closure

### Primary synthetic study

- Three held-out domains: FactorShock, Inventory, and Queue.
- Primary dimension: `d=1000`, 20 seeds per domain.
- Source archive: two source domains, 64 shared profiles per domain, three
  replications per profile, for 384 reusable source calls.
- Target protocol: `n0=10` frozen proposals and three adaptive target calls,
  hence 13 target search calls.
- Frozen atlas with proposal-only, stacked GP, and canonical SAASBO: 60/60
  true-feasible and independently certified recommendations, with zero false
  certificates.
- Common-Sobol frontend with the same three backends: 0/60, 1/60, and 0/60,
  respectively.
- Component ablation: universal support alone 27/60 true-feasible, source
  templates alone 40/60, and their combined maximin atlas 60/60.

These results identify the frontend as the dominant empirical contribution.
Canonical SAASBO improves conditional objective regret in Inventory and Queue,
but is neither necessary for feasible-basin coverage nor claimed as novel.

### Cost-fair and transfer comparisons

- Archive-fair transfer baselines use the same 384-call source archive, frozen
  `n0=10`, target seeds, 13 target search calls, and independent verifier.
- Target-only controls receive 397 target search calls, equal to the final
  method's 384 source plus 13 target-search calls.
- Target-only TuRBO and SCBO find 0/60 feasible recommendations; the audited
  periodic-capped SAASBO control finds 2/60 true-feasible and two independently
  certified recommendations.
- Transfer baselines range from 56/60 to 60/60 feasible and certified; the final
  atlas plus canonical SAASBO reaches 60/60 and has the strongest audited regret
  profile in the preregistered paired comparisons.

No result may relabel 13 target search calls as 13 total simulator calls.
Source, target-search, verification, and total calls must be displayed
separately in every main table.

### Dimension and budget stress test

- `d=1000`: 60/60 certified at 10 and 13 target search calls, with median
  feasible regret decreasing from 0.00825 to 0.00383 at 13 calls.
- `d=10000`: 30/30 certified with zero false certificates at target search
  budgets 10, 13, 20, and 40.
- Aggregate median regret at `d=10000` is 0.00825, 0.00825, 0.00451, and 0.00579
  for those four budgets.  The nonmonotonic change from 20 to 40 must be shown,
  not smoothed away.
- The dimension result is conditional on a stable low-dimensional structural
  coordinate.  It is not an unconditional claim about arbitrary 10,000-
  dimensional functions.

### External energy holdout

- OPSD GB_GBN is an untouched 20-seed holdout after development on DK_2.
- Frozen atlas, natural constant-policy low-frequency grid, and equal-total-cost
  common Sobol all achieve 20/20 independent certificates and zero false
  certificates.
- Frozen atlas beats equal-total-cost unstructured Sobol in 20/20 paired seeds;
  median objective difference is -0.05331 with bootstrap 95% interval
  [-0.05408, -0.05058].
- Frozen atlas does not beat the natural low-frequency grid: 9 wins and 11
  losses; median difference 0.000945 with bootstrap 95% interval
  [-0.00108, 0.00180].

The external result supports dimension-equivariant low-frequency structural
compression and a total-cost advantage over unstructured search.  It does not
support a claim that source learning is superior to an obvious domain-specific
low-frequency control.

### Cumulative heteroscedastic decomposition

The corrected 20-seed paired study improves variance calibration in every
domain.  Relative to pooled variance, cumulative factor-HVD reduces mean
log-variance RMSE from 1.158 to 0.219 in FactorShock, 0.577 to 0.349 in
Inventory, and 0.501 to 0.254 in Queue; variance-shape correlation rises from
zero to 0.919, 0.953, and 0.995.  It does not improve feasible recovery,
false-certificate count, or regret, and it increases verification cost in
Inventory and Queue.  HVD therefore remains a secondary risk-calibration
result rather than a coequal optimization contribution.

## Lean Closure

- Full `lake build` passes.
- The project contains 102 Lean source files.
- `sorry`, `admit`, and project-defined `axiom` occur zero times.
- Machine-checked results include finite-atlas no-free-lunch, conditional
  geometric atlas coverage, fail-closed source-monotone admission, transferred
  endpoint safety under an explicit held-out monotonicity condition,
  optimizer-independent shortlist verification, objective-switch control,
  exact binomial all-success certification, and budget decomposition.

The largest theoretical limitation is deliberate: the global atlas theorem is
conditional on source-target support shift, coordinate error, chance-margin
regularity, and safe depth.  The finite synthetic audit verifies the registered
library condition in all three domains but does not certify a global Lipschitz
constant.  The manuscript must not convert this conditional theorem into an
universal transfer guarantee.

## Remaining Work

The remaining tasks are manuscript-production tasks, not missing science:

1. render final tables and figures only from compact audited artifacts;
2. freeze terminology and a claim-evidence map;
3. draft a new manuscript without modifying the rejected legacy paper;
4. report bootstrap intervals, Holm-adjusted paired tests, failures, timeouts,
   and all budget components;
5. compile and visually audit the PDF;
6. release code, compact data, a README, and the immutable experiment manifest
   in accordance with the journal's code and data policy.

## Claim Changes That Would Reopen Experiments

New experiments would be required before claiming any of the following:

- source learning beats a natural structured control in the energy domain;
- cumulative HVD improves optimization rather than variance calibration;
- KG or SC-V69 is the essential online decision rule;
- unconditional coverage of arbitrary high-dimensional target functions;
- positive traffic/SUMO generalization;
- a globally certified source-to-target Lipschitz bound.

# Paper-Grade Experiment Protocol

## Claims to test

1. A source-only universal low-frequency support prior and source-ranked
   low-frequency proposal compress high-dimensional policy search into
   dimension-equivariant observable coordinates. Completed causal gates do
   not support describing all four structural switches as coequal drivers.
2. State-coupled cumulative HVD is a mechanistic risk model whose calibration
   value is tested causally against pooled variance under an otherwise
   identical proposal, backend, and verifier. It is not a headline
   optimization claim unless that registered comparison passes.
3. Online acquisition is a replaceable backend. The headline implementation
   uses canonical every-iteration SAASBO; KG, Thompson sampling, and
   evaluate-or-replicate VOI remain backend ablations.
4. Performance depends on effective risk dimension rather than raw dimension.
5. Conservative certification becomes nonvacuous when the charged evidence
   budget is statistically sufficient.

Manifold and Transformer encoders are representation-background ablations, not
main contributions.

## Immutable final replay

The manuscript-lock synthetic tables do not reuse V51/V64/V69 result rows.
Both `Frozen Source-Informed` and `Common Sobol` are rerun from the tracked-only
Git snapshot `27d55e0d5f265034f91ee7b3f7988dd8233881e5`. Each contains 20 paired
seeds, three held-out domains, and proposal-only, Stacked GP, and canonical
SAASBO backends. The execution result records the exact source/proof/scripts
tree hashes and a backend-specific method contract. Later analysis commits may
change audit or rendering code, but cannot silently relabel the executed
algorithm.

The proposal-only backend is charged `N=10`; Stacked GP and canonical SAASBO
are charged `N=13`. Source-informed rows are charged 384 source calls.
Common-Sobol proposal-only and SAASBO rows use zero source calls, whereas its
Stacked GP row still receives the registered 384-call source archive. These
budgets are checked per method rather than hidden behind one table-level
number.

Orthogonality, adaptive sparsity, and additive groups remain members of the
unified structural hypothesis class and must appear in causal ablations. They
are promoted to headline claims only if independently retrained source
proposals/posteriors show incremental held-out value over the supported
low-frequency front end.

## Freeze gates

### Gate A: implementation closure

- Domains: FactorShock, Inventory, Queue.
- `d=1000`, `N=20`, `n0=10`, seeds `0..19`.
- Compare the pre-repair promoted V51 aggregate with the observed-terminal
  challenger.
- Require 60/60 completed, no target-oracle use, no forced override, no
  adaptive-loss increase, no feasibility loss, and paired regret not worse.
- Require every iteration to log one terminal contract identifier and the same
  observed-action semantics for current/fantasy/final values.
- **Passed:** 60/60 complete, 17/7/36 paired wins/losses/ties versus the
  pre-repair V51 control, 60/60 true feasible, 26 adaptive improvements, zero
  adaptive losses, and zero false certificates.

### Gate B: certification nonvacuity

- Budgets `N in {20,40,80}` with `n0=10`.
- Replication caps `{5,10,20}` and exact evaluate-or-replicate actions.
- Five-seed screen, then 20 seeds for surviving settings.
- Report certificate coverage, recall on evaluated feasible points, false
  certification, minimum upper margin, variance calibration, and target calls.
- Promotion requires positive coverage in every primary domain and zero or
  statistically controlled false certification.
- The five-seed screen contains one matched new-point-only control and exact
  joint-VOI variants with replication caps `5`, `10`, and `20` at each budget.
  All variants reuse the identical frozen source archive and `n0` design.

### Gate C: numerical VOI fidelity

- Use the identical posterior after `n0=10` and charge exactly one online
  action (`N=11`), so later trajectory divergence cannot masquerade as a
  numerical error.
- MC samples `{2,8,32}` use pair-indexed nested antithetic draws: every smaller
  sample plan is an exact prefix of every larger plan.
- New-action shortlist sizes `{1,4,8,32}` are nested under the same
  posterior-risk ordering. `mc32/k32` is the declared finite audit universe.
- Report selected-arm agreement, rank correlation, regret, runtime, and
  normalized empirical `epsilon_shortlist`/`eta_MC` proxies using anonymous
  action fingerprints. The finite audit does not claim exhaustive coverage of
  the original continuous policy space.
- **Passed:** 180/180 runs complete with all source, pairing, and closure
  contracts satisfied. Only `mc32/k32` survived every preregistered threshold:
  selected-arm agreement and median rank correlation were both `1.0`, with
  zero normalized proxy error relative to the declared finite audit universe.
  `mc8/k32` retained `0.8` arm agreement but its median normalized MC-error
  proxy was `0.567`, so the paper configuration is frozen at 32 nested
  antithetic samples and a 32-action shortlist.

### Gate D: independently certified deployment

- Freeze the optimization recommendation and one posterior-safe,
  cumulative-risk-diversified member of the charged `n0` atlas before reading
  any verification response.
- Search uses `d=1000`, `n0=10`, and `N_search=13`.
- Rank 1 receives 80 independent replications. Rank 2 is tested only after
  rank 1 fails and receives 96 replications.
- Split one family-wise error budget as `0.025+0.025`; use the exact
  noncentral-t Gaussian quantile-tolerance upper bound.
- Report `source_calls`, `search_calls`, realized `verification_calls`,
  `target_total_calls`, and `source+target_total_calls` separately.
- **Passed on fresh seeds 60..79:** 60/60 deployments certified, 46 at rank 1
  and 14 at rank 2, with zero false certificates. Relative to the frozen V51
  optimization output, deployment produced 14 wins, zero losses, 46 ties,
  three feasibility rescues, and zero feasibility losses. Mean verification
  cost was 102.4 calls and the precommitted maximum was 176.

### Gate E: cumulative-HVD causal role

- Freeze the source archive, target initial design, canonical SAASBO backend,
  search budget, and independent verifier.
- Compare pooled variance with cumulative factor-HVD on all three synthetic
  domains using 20 paired seeds per domain.
- **Five-seed pilot:** all 15 pairs were feasible and independently certified
  with zero false certificates, but cumulative HVD did not change regret or
  feasibility. It improved log-variance scale only on FactorShock, worsened
  scale RMSE on Inventory and Queue, and did not recover FactorShock variance
  shape.
- **Final replay registered:** 120 fresh immutable tasks rerun both variance
  heads on all 60 paired cells. The five-seed pilot is not pooled into this
  final result. HVD remains a mechanistic/conditional component unless the
  final preregistered gate supports universal calibration without harming
  feasibility, regret, or false certification.

## Main matrix

The certified-deployment matrix uses V64's frozen `80/96` verification suffix.
Search-only results and certified-deployment results are both reported; the
verification suffix is never folded silently into `N_search`. Gate C retains
the `32/32` numerical audit schedule for experiments that exercise exact KG.

### Domains

- FactorShock state-policy control: explicit shared-shock stress test.
- Inventory/supply-chain control: stockout-sensitive chance boundary.
- Queue/resource control: burst/common-load risk.
- SUMO ingolstadt21 strict no-history traffic with fresh trajectory CSV and
  out-of-sample seed certification. The main external-validity track may use
  a frozen observable task-family descriptor but no target outcomes, oracle,
  or historical traffic anchor. A five-seed domain-blind split that excludes
  the nearest source analogue is retained as a preregistered negative control.
- Legacy RZDT1/RZDT2/RZDT5_RR bridge, using the original definitions and a
  scalarized single objective while retaining the original heteroscedastic
  profiles.

### Dimension and target-budget frontier

- Certified core matrix: `d=1000`, `N_search=13`, `n0=10`, 20 fresh seeds.
- `d={200,1000}`, `N={20,40,80}`, at least 20 seeds.
- `d=10000`, `N={20,40}`, at least 10 seeds after `d=1000` passes.
- Report `d/N_search`, `d/(N_search+N_verify)`, and
  `d/(N_source+N_search+N_verify)`.
- Source archive is fixed at 384 simulator calls unless a source-cost ablation
  explicitly changes it.

### Main methods

- Frozen source proposal only (`n0-best`).
- Frozen proposal plus canonical every-iteration SAASBO and independent
  deployment verification.
- Frozen proposal plus Stacked GP as the strongest audited transfer backend.
- Common-Sobol versions of the same backends as front-end negative controls.
- SC-V69, legacy exact KG, Thompson sampling, neutral Sobol continuation, and
  evaluate-or-replicate VOI as backend ablations.
- Pooled variance versus cumulative HVD as a registered mechanistic ablation.

### Fair baselines

- Target-only Sobol/random, official TuRBO, official SCBO, official SAASBO.
- Safe F-PACOH, RGPE-CBO, hierarchical/transfer GP-CBO, FSBO/HyperBO-CBO,
  MetaBO/MALIBO-CBO.
- Archive-fair transfer methods receive the identical 384-call frozen archive,
  byte-identical frozen `n0`, target seeds, bounds, and `N_search`. The main
  archive-fair stratum uses `d_source=d_target=1000` and freezes the
  `low_frequency_only + risk_objective_atlas` proposal before any target
  response.
- Every method freezes its shortlist using its own posterior, then receives
  the same independent `80/96`, family-wise `0.05` terminal protocol. No
  comparator is filtered through the SC posterior.
- A total-cost table gives target-only SOTA `384+N_search` search calls as a separate
  comparison; it is not mixed with the equal-target-budget transfer table.
- The separate `d_source=50 -> d_target=1000` SC experiment is a
  cross-dimension transfer stress test. It is never labelled as
  byte-identical archive fairness against methods whose input layer requires
  equal source and target dimensions.
- Timeouts and failures remain in denominators.

## Causal ablations

Retrain the source model and regenerate the proposal for every ablation.

- No structural prior.
- Low-frequency only, orthogonality only, adaptive sparsity only, additive
  groups only.
- Leave-one-prior-out for each of the four principles.
- All four priors.
- Proposal-only, posterior-only, and proposal-plus-posterior roles.
- Pooled, class, orthogonal pointwise, pointwise factor, and cumulative factor
  HVD.
- Raw, state, raw+state, and learned representation backgrounds.

## Metrics

- True-feasible recommendation rate and false-feasible rate.
- Feasible simple regret with failures retained, not conditionalized away.
- `n0-best -> final` improvement, adaptive rescue, and adaptive loss.
- Certificate coverage, precision, recall, false certification, and vacuity.
- Objective and chance-margin convergence versus charged target calls.
- HVD log-variance RMSE and independent/shared/linear/floor decomposition.
- New versus replicate action count and selected VOI difference.
- Wall time split across candidate generation, fantasy update, HVD, simulation,
  and checkpointing.

Use paired seeds, median/IQR curves, paired bootstrap 95% confidence
intervals, exact McNemar tests for binary outcomes, and Wilcoxon signed-rank
tests with matched-pairs rank-biserial effect sizes for continuous outcomes.
Holm correction is applied within preregistered confirmatory families on the
global paired stratum only; domain strata are unadjusted heterogeneity
analyses. Primary conclusions require at least 20 seeds; legacy-style
10-seed plots are visual supplements only.

## Figures

The legacy manuscript's visual grammar is retained while changing the claims:

1. Feasible-regret convergence with median and bootstrap band.
2. True chance margin and posterior upper-margin traces.
3. Certificate coverage/false-certification curves versus budget.
4. HVD calibration and cumulative-risk component trajectories.
5. Dimension/evaluation frontier (`d/N` versus regret and feasibility).
6. Source proposal, online evaluated actions, and final Bayes action in the
   learned observable coordinate.
7. New/replicate action mix and VOI distribution.
8. SUMO trajectory exposure and out-of-sample certification panel.

Every table states target calls, source calls, `n0`, replications, failures,
and whether oracle information was used. In certified tables, "target calls"
is further split into search and verification calls. Raw checkpoints, PKL files, and
profiles remain server-side; only compact aggregate JSON/CSV and publication
figures are synchronized.

# Paper-Grade Experiment Protocol

## Claims to test

1. A transferable structural prior compresses high-dimensional policy search
   into low-dimensional observable mean and cumulative-risk coordinates.
2. State-coupled cumulative HVD improves variance calibration and prevents
   false feasibility under shared shocks.
3. Unified evaluate-or-replicate VOI improves the source-informed initial
   design without sacrificing its incumbent.
4. Performance depends on effective risk dimension rather than raw dimension.
5. Conservative certification becomes nonvacuous when the charged evidence
   budget is statistically sufficient.

Manifold and Transformer encoders are representation-background ablations, not
main contributions.

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

## Main matrix

### Domains

- FactorShock state-policy control: explicit shared-shock stress test.
- Inventory/supply-chain control: stockout-sensitive chance boundary.
- Queue/resource control: burst/common-load risk.
- SUMO ingolstadt21 strict no-history traffic with fresh trajectory CSV and
  out-of-sample seed certification.
- Legacy RZDT1/RZDT2/RZDT5_RR bridge, using the original definitions and a
  scalarized single objective while retaining the original heteroscedastic
  profiles.

### Dimension and target-budget frontier

- `d={200,1000}`, `N={20,40,80}`, at least 20 seeds.
- `d=10000`, `N={20,40}`, at least 10 seeds after `d=1000` passes.
- Report both target-only `d/N` and total-cost `d/(source calls + N)`.
- Source archive is fixed at 384 simulator calls unless a source-cost ablation
  explicitly changes it.

### Main methods

- Frozen source proposal only (`n0-best`).
- Frozen proposal plus neutral Sobol continuation.
- Promoted SC-OLH V51 closure method.
- Same model with new-point-only actions.
- Same model with pooled variance instead of cumulative HVD.
- Same model without source-discrepancy adaptation.

### Fair baselines

- Target-only Sobol/random, official TuRBO, official SCBO, official SAASBO.
- Safe F-PACOH, RGPE-CBO, hierarchical/transfer GP-CBO, FSBO/HyperBO-CBO,
  MetaBO/MALIBO-CBO.
- Archive-fair transfer methods receive the identical 384-call frozen archive,
  identical target `n0`, target seeds, bounds, and total `N`.
- A total-cost table gives target-only SOTA `384+N` target calls as a separate
  comparison; it is not mixed with the equal-target-budget transfer table.
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

Use paired seeds, median/IQR curves, bootstrap 95% confidence intervals,
Wilcoxon signed-rank tests with Holm correction, and effect sizes. Primary
conclusions require at least 20 seeds; legacy-style 10-seed plots are visual
supplements only.

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
and whether oracle information was used. Raw checkpoints, PKL files, and
profiles remain server-side; only compact aggregate JSON/CSV and publication
figures are synchronized.

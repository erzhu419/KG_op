# Quality-First Validation

This folder holds the tracked development gate for `SC-OLH-KG`.

The original research task is an optimization problem, so promotion is
quality-first:

- the primary single-objective candidate must return a truly feasible solution;
- `simple_regret` and `true_objective` must be non-worse than the tracked
  baseline when those metrics exist;
- the state-coupled runner and bi-objective smoke runner must still execute
  cleanly;
- wall time is only a secondary engineering constraint, controlled by
  `--max_wall_slowdown`.

Faster wall time alone is not a valid algorithmic improvement.
SC feasibility is recorded by default, but is only a hard gate when
`--require-sc-feasible` is passed because SC coupling is a separate improvement
stage from the primary single-objective OLH-KG baseline.
The default SC runner uses the state encoder for the coupling acquisition
score while keeping the stable raw quadratic GPR mean basis.  Pass
`--use_state_basis` only when explicitly testing occupancy features inside the
mean model; this is more fragile in very small-budget runs.
The default `--lambda_coupling` is intentionally light (`0.05`) so coverage
guidance cannot swamp objective KG and feasibility learning.
Coverage coupling is also gated by a conservative posterior chance margin:
`--coupling_safety_z 0.5` and `--coupling_gate_temperature 0.25` keep state
coverage rewards concentrated in regions that the constraint model considers
confidently feasible.  This matters on RZDT2-like problems where global novelty
can otherwise pull the search into the high-risk middle of the domain.
`--encoder_kind` controls the SC state encoder used by candidate generation,
coupling scores, optional GPR state basis, and optional manifold HVD features.
The current family is `synthetic`, `self_supervised`, `transformer`,
`pca_manifold`, `kernel_manifold`, `graph_laplacian`, `ssl_masked`,
`ssl_contrastive`, `ssl_next_risk`, `ssl_transformer`, and `ssl_hybrid`.  The
learned encoders are ablations for state-policy representation experiments
rather than hidden oracles.
`--state_basis_mode raw+manifold` tests the learned latent basis inside the
mean model; `raw+state` keeps the older deterministic occupancy basis.

Workflow:

1. Run the validator against the tracked baseline.
2. If there is no tracked baseline yet, the validator uses the legacy
   `Final_Submission/GPR_KG_Code` implementation as the initial baseline.
3. Only promote a candidate when all gates pass.
4. The promoted candidate metrics become `baselines/current.json`, which is the
   next validation baseline.

Default quick gate:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 SC-OLH-KG/performance/validate_performance.py \
  --variance_mode orthogonal \
  --repeats 2 \
  --promote-if-pass
```

The tracked baseline is intentionally small-budget and machine-local.  It is a
development gate, not a final paper experiment.

Quality benchmark:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 SC-OLH-KG/performance/benchmark_quality.py \
  --N 20 \
  --n0 5 \
  --K1 15 \
  --K2 1 \
  --problem RegimeRZDT1 \
  --certification_mode theory \
  --beta_g 2.0 \
  --n_seeds 5
```

This benchmark writes JSON plus row/summary CSV files under
`SC-OLH-KG/profiles/`.  It compares optimization quality across variance modes
and optional SC coupling; wall time is included only as a diagnostic column.
Set `--acquisition_modes additive,exact_mc,blend` to compare the additive proxy,
sampled posterior-update KG, and the blended bridge in the same table.  In
`exact_mc`/`blend` modes, the runner uses cloned GPR and HVD updates before
recomputing the theory-certified terminal value.

Paper-grade HVD/acquisition ablation:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 SC-OLH-KG/performance/benchmark_hvd_suite.py \
  --problems RegimeRZDT1,RZDT2,StatePolicyRZDT1,FactorShockStatePolicyRZDT1 \
  --modes pooled,class,orthogonal,factor \
  --sc_modes factor \
  --acquisition_modes additive,exact_mc,blend \
  --certification_mode theory \
  --beta_g 2.0 \
  --N 80 --n0 10 \
  --K1 25 --K2 1 \
  --n_seeds 20 \
  --jobs 10 \
  --out_prefix hvd_theory_n80_s20
```

Completed additive paper-grade HVD run on 2026-07-04:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 SC-OLH-KG/performance/benchmark_hvd_suite.py \
  --problems RegimeRZDT1,RZDT2,StatePolicyRZDT1,FactorShockStatePolicyRZDT1 \
  --modes pooled,class,orthogonal,factor \
  --sc_modes factor \
  --acquisition_modes additive \
  --certification_mode theory \
  --beta_g 2.0 \
  --N 80 --n0 10 \
  --K1 25 --K2 1 \
  --posterior_pool_size 300 \
  --posterior_keep 15 \
  --eval_pool_size 500 \
  --n_seeds 20 \
  --jobs 4 \
  --out_prefix paper_hvd_additive_n80_s20_20260704
```

The output lives under `SC-OLH-KG/profiles/` with that prefix.  Pooled across
the four problems, `factor+sc` achieved true-feasible rate `1.0`, false-feasible
rate `0.0`, and median feasible simple regret `0.0013932`.  The non-SC
`factor` run achieved true-feasible rate `0.825`, false-feasible rate `0.0`,
and median feasible simple regret `0.00018`.  On the factor-shock synthetic,
`factor+sc` raised true-feasible rate from `0.30` (`factor`) to `1.0` while
keeping false-feasible rate at `0.0`, which is the main evidence that the
cumulative shared-shock block is doing useful work.
For SC coupling evidence, run both a monotone/regime problem and a concave
boundary problem:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 SC-OLH-KG/performance/benchmark_quality.py \
  --problem RegimeRZDT1 \
  --N 20 --n0 5 --K1 15 --K2 1 \
  --n_seeds 20 \
  --modes orthogonal \
  --sc_modes orthogonal

PYTHONDONTWRITEBYTECODE=1 python3 SC-OLH-KG/performance/benchmark_quality.py \
  --problem RZDT2 \
  --N 20 --n0 5 --K1 15 --K2 1 \
  --n_seeds 20 \
  --modes orthogonal \
  --sc_modes orthogonal
```

`StatePolicyRZDT1` is a harder synthetic for policy-state occupancy structure.
It uses problem-provided structured initial samples/candidates and a nominal
residual-variance cap so HVD does not confuse early mean-model bias with
simulation noise.  At `N=20,n0=5`, orthogonal HVD should now recover feasible
recommendations reliably; SC coupling remains a separate improvement target
because the current gated coupling is usually neutral on this benchmark.
When `use_state_coupling=True`, SC now also adds candidates generated in a
state/meta space: anchors are proposed in state space, then inverted back into
raw parameter vectors.  The selected candidate source is logged as
`candidate_source_selected`.  `--structured_candidate_count` defaults to `0`
so these problem-specific anchors do not leak into the raw OLH-KG baseline;
SC uses `--state_candidate_count` for meta-space search.

SOTA/strong-baseline benchmark:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 SC-OLH-KG/performance/benchmark_sota.py \
  --problem StatePolicyRZDT1 \
  --N 20 --n0 5 \
  --n_seeds 5 \
  --variance_mode factor \
  --acquisition_mode exact_mc \
  --eval_pool_size 200 \
  --certification_mode theory \
  --botorch_timeout_sec 30 \
  --baselines sobol,random,turbo_lite,scbo_lite,botorch_turbo,botorch_scbo,botorch_saasbo
```

`turbo_lite` and `scbo_lite` are dependency-light trust-region baselines, not
exact BoTorch TuRBO/SCBO.  `botorch_turbo`, `botorch_scbo`, and
`botorch_saasbo` use the real BoTorch/GPyTorch stack: BoTorch GP models,
BoTorch acquisition functions, and `optimize_acqf`.  The aliases `turbo`,
`scbo`, and `saasbo` map to those BoTorch variants.  If BoTorch is not
installed, the benchmark can either fall back to the lite baselines
(`--botorch_fallback lite`) or fail loudly (`--botorch_fallback error`).
Use small `--botorch_raw_samples`, `--botorch_num_restarts`, and
`--saas_warmup_steps` values for smoke tests; increase them for paper runs.
`--botorch_timeout_sec` is required for SAASBO accounting in formal runs, so
timeouts are counted instead of silently stalling the benchmark.
`--botorch_max_candidate_failures` bounds repeated candidate-generation
failures in one BoTorch run; SAASBO can then switch to a recorded cheap
fallback instead of spending every remaining budget step in NUTS/optimization.
The full lightweight baseline family also includes `hetgp_lite`,
`rahbo_lite`, `safeopt_lite`, and `legacy_vepm_lite`.  These are reproducible
style baselines for the paper matrix, not substitutes for dedicated external
packages.

Encoder ablation suite:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 SC-OLH-KG/performance/benchmark_encoder_suite.py \
  --problem StatePolicyRZDT1 \
  --encoder_kinds synthetic,pca_manifold,kernel_manifold,graph_laplacian,ssl_masked,ssl_contrastive,ssl_next_risk,ssl_transformer,ssl_hybrid \
  --use_state_basis --state_basis_mode raw+manifold \
  --N 30 --n0 8 \
  --n_seeds 10 \
  --out_prefix encoder_suite_statepolicy_n30_s10
```

Smoke result on 2026-07-05 with `N=7,n0=5,n_seeds=1` confirmed that all seven
representation paths run inside SC-OLH-KG.  The tiny probe is a functionality
check only; paper-grade evidence still needs the high-dimensional and traffic
scheduler suites.

Exact-MC decision probe:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 SC-OLH-KG/performance/benchmark_quality.py \
  --problem StatePolicyRZDT1 \
  --modes factor \
  --sc_modes factor \
  --acquisition_modes additive,exact_mc,blend \
  --N 16 --n0 8 \
  --K1 6 --K2 0 \
  --eval_pool_size 80 \
  --exact_kg_mc_samples 1 \
  --n_seeds 2 \
  --out_prefix exact_decision_statepolicy_n16_s2_20260704
```

This probe showed that exact-MC can improve quality but is still much slower:
`factor:olhkg_sc_exact` reached median regret `0.00018` with median wall time
`36.06s`, while `factor:olhkg_sc_additive` reached `0.00181` with `2.30s`.
Thus exact-MC is implemented and theoretically bridged, but should remain an
ablation until candidate/HVD clone updates are vectorized enough for N=80,
20-seed paper runs.

Multi-problem SOTA suite:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 SC-OLH-KG/performance/benchmark_sota_suite.py \
  --problems RegimeRZDT1,RZDT2,StatePolicyRZDT1 \
  --N 20 --n0 5 \
  --n_seeds 20 \
  --baselines sobol,random,botorch_turbo,botorch_scbo,botorch_saasbo \
  --variance_mode factor \
  --acquisition_mode exact_mc \
  --eval_pool_size 200 \
  --certification_mode theory \
  --botorch_timeout_sec 60 \
  --jobs 10 \
  --worker_torch_threads 1 \
  --out_prefix sota_real_n20_s20
```

Completed guarded BoTorch SOTA suite on 2026-07-04:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 SC-OLH-KG/performance/benchmark_sota_suite.py \
  --problems RegimeRZDT1,RZDT2,StatePolicyRZDT1 \
  --variance_mode factor \
  --acquisition_mode additive \
  --certification_mode theory \
  --beta_g 2.0 \
  --N 80 --n0 10 \
  --K1 25 --K2 1 \
  --posterior_pool_size 300 \
  --posterior_keep 15 \
  --eval_pool_size 500 \
  --baselines sobol,random,botorch_turbo,botorch_scbo,botorch_saasbo \
  --botorch_fallback error \
  --botorch_raw_samples 64 \
  --botorch_num_restarts 4 \
  --botorch_maxiter 50 \
  --botorch_timeout_sec 10 \
  --botorch_max_candidate_failures 2 \
  --saas_warmup_steps 4 \
  --saas_num_samples 4 \
  --saas_max_tree_depth 3 \
  --saas_mc_samples 32 \
  --jobs 10 \
  --worker_torch_threads 1 \
  --n_seeds 20 \
  --out_prefix sota_real_additive_n80_s20_20260704_guarded
```

This is a 420-run matrix: three problems, seven variants, and 20 seeds.  The
real BoTorch backends were available and no fit/candidate failures or SAASBO
fallbacks were recorded.  Pooled across the three problems, `olhkg_additive`
kept true-feasible rate `1.0`, false-feasible rate `0.0`, median feasible
simple regret `0.0`, and median wall time `12.69s`.  `olhkg_sc_additive` also
kept true-feasible rate `1.0` and false-feasible rate `0.0`, with median regret
`0.000186` and median wall time `15.47s`.  The real BoTorch baselines had
lower feasibility and higher median regret: TuRBO `0.9667/0.0333/0.00669`,
SCBO `0.9833/0.0167/0.00871`, and SAASBO `0.9667/0.0333/0.00193` for
true-feasible rate / false-feasible rate / median feasible simple regret.

Paper bridge on the original RZDT families:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 SC-OLH-KG/performance/benchmark_paper_bridge.py \
  --modes orthogonal,factor \
  --sc_modes factor \
  --acquisition_modes additive,exact_mc,blend \
  --certification_mode theory \
  --N 80 --n0 10 \
  --n_seeds 20 \
  --out_prefix paper_bridge_theory_n80_s20
```

Fresh traffic trajectory logs:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 SC-OLH-KG/performance/benchmark_traffic_fresh.py \
  --trajectory_log /path/to/fresh_traffic_trajectories.csv \
  --out_prefix traffic_fresh
```

If the CSV is absent, this runner writes a `missing_data` record.  That is the
only valid traffic status until real fresh-seed logs are supplied.
Learned traffic encoders (`ssl_*`) additionally require the trajectory CSV to
include a raw policy column `x`; `generate_traffic_trajectory_logs.py` now
adds that column when it creates fresh logs from candidate summaries.

HVD calibration diagnostic:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 SC-OLH-KG/performance/diagnose_hvd_calibration.py \
  --variance_mode orthogonal \
  --seed 4
```

The diagnostic evaluates the final recommendation and the axis grid against
true chance feasibility.  It reports variance calibration ratios, posterior
constraint-mean error, false-feasible points, and missed-feasible points.

The first promoted baseline used `class` HVD, and guarded `orthogonal` HVD was
temporarily promoted by the earlier speed-first gate.  After switching to
quality-first validation, `pooled` became the first quality baseline.  After
normalizing the objective-KG component when auxiliary scores are enabled,
`orthogonal` is the current tracked primary baseline.  `factor` and SC coupling
variants are promoted only after passing the quality gates and remaining within
the configured wall-time budget.

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

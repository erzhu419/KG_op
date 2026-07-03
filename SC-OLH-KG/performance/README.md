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
  --n_seeds 5
```

This benchmark writes JSON plus row/summary CSV files under
`SC-OLH-KG/profiles/`.  It compares optimization quality across variance modes
and optional SC coupling; wall time is included only as a diagnostic column.

The first promoted baseline used `class` HVD, and guarded `orthogonal` HVD was
temporarily promoted by the earlier speed-first gate.  After switching to
quality-first validation, `pooled` is the current tracked baseline.  Later
candidates, such as fixed `class`/`orthogonal`/`factor` HVD or SC coupling
variants, are promoted only after passing the quality gates and remaining
within the configured wall-time budget.

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

The first promoted baseline used `class` HVD.  Guarded `orthogonal` HVD is the
current tracked baseline.  Later candidates, such as `factor` HVD or SC
coupling variants, are promoted only after passing the quality gates and
remaining within the configured wall-time budget.

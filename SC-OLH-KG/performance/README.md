# Performance Validation

This folder holds the tracked performance gate for `SC-OLH-KG`.

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
  --repeats 2 \
  --promote-if-pass
```

The tracked baseline is intentionally small-budget and machine-local.  It is a
development gate, not a final paper experiment.

The default candidate uses `class` HVD because it is the first stable OLH
milestone.  `orthogonal` and `factor` HVD should be validated as later
candidates against the promoted class-HVD baseline.

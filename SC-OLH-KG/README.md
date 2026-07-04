# SC-OLH-KG

State-Coupled Orthogonal Latent Heteroscedastic Knowledge Gradient prototype.

This directory is intentionally separate from the original `KG_op` code.  It
contains a minimal, testable implementation path:

1. Profile the current GPR-KG baseline.
2. Smoke-test OLH variance decomposition in the existing bi-objective shape.
3. Run single-objective chance-constrained OLH-KG.
4. Add deterministic state-policy coupling for SC-OLH-KG.

Generated outputs should go under `results/`, `profiles/`, or `checkpoints/`;
those directories are ignored by git.

## Quick Commands

```bash
python3 SC-OLH-KG/runners/profile_current.py --N 12 --n0 5 --K1 10 --K2 1
python3 SC-OLH-KG/runners/run_single_olhkg.py --N 30 --n0 8 --K1 25 --K2 0
python3 SC-OLH-KG/runners/run_sc_olhkg.py --N 30 --n0 8 --K1 25 --K2 0
python3 SC-OLH-KG/runners/run_sc_olhkg.py --N 30 --n0 8 --K1 25 --K2 0 --use_state_basis
python3 SC-OLH-KG/runners/run_olh_biobj_smoke.py --N 20 --n0 6 --K1 20
python3 SC-OLH-KG/performance/benchmark_quality.py --N 20 --n0 5 --K1 15 --K2 1 --n_seeds 5
python3 SC-OLH-KG/performance/benchmark_exact_kg.py --N 18 --n0 6 --K1 10 --exact_mc_samples 2 --n_seeds 3
python3 SC-OLH-KG/performance/benchmark_sota.py --problem StatePolicyRZDT1 --N 20 --n0 5 --baselines botorch_turbo,botorch_scbo,botorch_saasbo
python3 SC-OLH-KG/performance/diagnose_hvd_calibration.py --variance_mode orthogonal --seed 4
python3 -m unittest discover -s SC-OLH-KG/tests
```

## Optional Exact KG

`SingleOLHKGConfig(exact_kg_mc_samples>0)` enables a sampled exact
posterior-update KG estimator.  It is off by default; the stable baseline still
uses the additive OLH-KG proxy.  Set `exact_kg_use_score=True` for smoke tests
that select directly by the sampled exact score.  Use
`performance/benchmark_exact_kg.py` to compare additive, blended, and exact
selection before promoting it.

## Compute-H Certificates

`core.kg.compute_h_certificate` returns the active line-envelope hull used by
`compute_h`; `validate_h_certificate` checks all original lines against the
finite-interval endpoint and Gaussian-tail slope conditions.  These checks are
bridged to Lean in `proof/SCOLHKG/Real/LineEnvelopeStack.lean`.

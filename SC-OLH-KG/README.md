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
python3 SC-OLH-KG/runners/run_olh_biobj_smoke.py --N 20 --n0 6 --K1 20
python3 SC-OLH-KG/performance/benchmark_quality.py --N 20 --n0 5 --K1 15 --K2 1 --n_seeds 5
python3 -m unittest discover -s SC-OLH-KG/tests
```

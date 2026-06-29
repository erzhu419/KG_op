"""Quick validation: run each method once on RZDT4 with small budget.

Tests that all 7 methods run without errors.  Uses a reduced budget
(N=60, n0=15) for speed; results are not meaningful numerically.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpr_kg import RZDT4

# Reduced budget for fast validation (not for results)
N_TEST = 60
N0_TEST = 15
SEED = 42

def test_method(method_cls, *args, **kwargs):
    name = method_cls.name if hasattr(method_cls, 'name') else str(method_cls)
    print(f"\n{'='*60}")
    print(f"  Testing: {name}")
    print(f"{'='*60}")

    prob = RZDT4(d=5, L=20, sigma=0.1, heteroscedastic=True)
    prob.calibrate_constraint(0.5)

    method = method_cls(*args, **kwargs)
    t0 = time.time()
    try:
        result = method.run(prob, N=N_TEST, n0=N0_TEST, seed=SEED)
        elapsed = time.time() - t0
        print(f"  OK! Time={elapsed:.1f}s, HV={result['hv_final']:.4f}, "
              f"IGD={result['igd_final']:.4f}, CVR={result['cvr_final']:.4f}, "
              f"|PF|={result['n_pareto_solutions']}, "
              f"n_sims={result['n_simulations']}")
        print(f"  HV history points: {len(result['hv_history'])}")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAILED after {elapsed:.1f}s: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    from methods.random_search import RandomSearch
    from methods.gpr_kg_method import GPRKGMethod
    from methods.gpr_kg_nv import GPRKGnVMethod
    from methods.cehvi_method import cEHVIMethod
    from methods.cparego_method import cParEGOMethod
    from methods.nsga2_kriging import NSGA2Kriging
    from methods.nsga2_direct import NSGA2Direct

    results = {}
    methods = [
        (RandomSearch, [], {}),
        (GPRKGMethod, [], {}),
        (GPRKGnVMethod, [], {}),
        (cEHVIMethod, [], {}),
        (cParEGOMethod, [], {}),
        (NSGA2Kriging, [], {}),
        (NSGA2Direct, [], {}),
    ]

    for cls, args, kwargs in methods:
        # Instantiate to get name
        instance = cls(*args, **kwargs)
        ok = test_method(cls, *args, **kwargs)
        results[instance.name] = ok

    print(f"\n{'='*60}")
    print("  VALIDATION SUMMARY")
    print(f"{'='*60}")
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name:15s}: {status}")
    print(f"{'='*60}")

    n_pass = sum(v for v in results.values())
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} methods passed validation.")

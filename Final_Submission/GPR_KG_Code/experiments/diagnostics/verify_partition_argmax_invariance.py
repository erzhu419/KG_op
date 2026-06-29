"""Verify that binary_bin and aggregate partition schemes yield DIFFERENT
sigma^2 estimates for unvisited candidates but happen to produce the SAME
KG argmax (and hence identical sampling trajectories).

Procedure:
  1. Fix seed; instantiate RZDT1 (d=5).
  2. For each scheme in {binary_bin, aggregate}:
     a. Run pre-sampling (30 LHS pre-samples, identical across schemes
        because seed is fixed).
     b. Initialize VEPM (different partition logic per scheme).
     c. Generate the 50 LHD candidates for iter=1 (same RNG state -> same
        candidates).
     d. Query sigma^2(x_i) for each candidate i under each scheme.
  3. Compare the two sigma^2 sequences (should differ for unvisited x)
     and the candidate ranking under whatever proxy of acquisition we can
     compute.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(r'E:/从Dell移动到X1/科研同行/谭真/OR投稿/Final_Submission/GPR_KG_Code')))
import numpy as np
np.random.seed(1000)

from gpr_kg import RZDT1, VEPM, ParametricGPR

SEED = 1000
D = 5
N0 = 30

def setup(partition_method):
    np.random.seed(SEED)
    prob = RZDT1(d=D, L=100, sigma=0.04, heteroscedastic=True)
    prob.tau = 0.0
    # Build the same pre-samples (deterministic with seed)
    pre = []
    for _ in range(N0):
        x = prob.sample_random()
        pre.append(x)
    # Simulate at each pre-sample (deterministic with seed -> same Y)
    observations = {}
    for x in pre:
        y = prob.simulate(x)
        observations.setdefault(x, []).append(y)
    # Build 3 GPRs
    gprs = [ParametricGPR(d=D) for _ in range(3)]
    for x in pre:
        for i in range(3):
            gprs[i].update(np.array(x), observations[x][0][i], 0.04**2)
    # VEPM
    norm_func = lambda v: prob.normalize(np.asarray(v, dtype=float))
    vepm = VEPM(d=D, L=100, w=1.0, normalize_func=norm_func,
                partition_method=partition_method)
    vepm.initialize(pre, observations, gprs)
    return prob, pre, observations, gprs, vepm

# Run for both schemes
res = {}
for sch in ('binary_bin', 'aggregate'):
    prob, pre, obs, gprs, vepm = setup(sch)
    res[sch] = dict(prob=prob, pre=pre, obs=obs, gprs=gprs, vepm=vepm)

# Sanity: identical pre-samples
print('Pre-samples identical:',
      res['binary_bin']['pre'] == res['aggregate']['pre'])

# Generate 50 LHD candidates with a fresh deterministic RNG state.
# (Same RNG -> same candidates regardless of scheme.)
def lhd_candidates(prob, K1=50, seed=42):
    rng = np.random.default_rng(seed)
    lo, hi = prob.int_bounds()
    d = prob.d
    perms = [rng.permutation(K1) for _ in range(d)]
    cands = []
    for i in range(K1):
        x = []
        for k in range(d):
            u = (perms[k][i] + rng.random()) / K1
            v = int(round(lo[k] + u * (hi[k] - lo[k])))
            v = max(int(lo[k]), min(int(hi[k]), v))
            x.append(v)
        cands.append(tuple(x))
    return cands

candidates = lhd_candidates(res['binary_bin']['prob'], K1=50, seed=42)

# Query sigma^2 from each scheme's VEPM for each candidate, objective f^3
sigma2 = {sch: np.array([res[sch]['vepm'].get_variance(2, np.array(x))
                          for x in candidates])
          for sch in ('binary_bin', 'aggregate')}

# Also query the GPR posterior mean (identical across schemes since GPRs
# use the same data)
mu3 = np.array([res['binary_bin']['gprs'][2].posterior_mean(np.array(x))
                for x in candidates])

# Build a simple "feasibility-aware acquisition" proxy: high if mu3 low
# AND sigma small (so chance constraint pr(f3 <= 0) >= 0.95 has slack).
# acquisition = -(mu3 + 1.645*sqrt(sigma2)) (larger is more feasible)
acq = {sch: -(mu3 + 1.645 * np.sqrt(sigma2[sch])) for sch in sigma2}

print()
print(f'{"i":>3} | {"sigma2_bb":>10s} | {"sigma2_agg":>10s} | {"diff":>10s} | '
      f'{"mu3":>8s} | {"acq_bb":>9s} | {"acq_agg":>9s}')
print('-' * 80)
for i in range(50):
    sbb = sigma2['binary_bin'][i]
    sagg = sigma2['aggregate'][i]
    diff = sagg - sbb
    print(f'{i:>3} | {sbb:>10.6f} | {sagg:>10.6f} | {diff:>+10.6f} | '
          f'{mu3[i]:>+.4f} | {acq["binary_bin"][i]:>+8.4f} | {acq["aggregate"][i]:>+8.4f}')

print()
print('=== Summary ===')
print(f'sigma2 binary_bin: min={sigma2["binary_bin"].min():.6f}  max={sigma2["binary_bin"].max():.6f}')
print(f'sigma2 aggregate:  min={sigma2["aggregate"].min():.6f}  max={sigma2["aggregate"].max():.6f}')
n_same = int(np.sum(np.isclose(sigma2['binary_bin'], sigma2['aggregate'], atol=1e-9)))
print(f'sigma2 identical across schemes: {n_same}/50 candidates')
amax_bb  = int(np.argmax(acq['binary_bin']))
amax_agg = int(np.argmax(acq['aggregate']))
print(f'argmax(acq_bb)  = candidate index {amax_bb}')
print(f'argmax(acq_agg) = candidate index {amax_agg}')
if amax_bb == amax_agg:
    print('-> argmax IDENTICAL despite sigma^2 differing -> same x sampled, same trajectory, same HV')
else:
    print('-> argmax DIFFERS')

# Also show top-5 ranking under each scheme
rank_bb  = np.argsort(-acq['binary_bin'])[:5]
rank_agg = np.argsort(-acq['aggregate'])[:5]
print(f'top-5 candidates (by feasibility-acq) binary_bin: {rank_bb.tolist()}')
print(f'top-5 candidates (by feasibility-acq) aggregate:  {rank_agg.tolist()}')

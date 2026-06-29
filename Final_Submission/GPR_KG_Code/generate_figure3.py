"""Generate Figure 3: VEPM variance convergence at unsampled solution."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

from gpr_kg import RZDT1, ParametricGPR, VEPM, compute_kg_factor, compute_h
from gpr_kg import pareto_filter, crowding_distance_select, compute_hypervolume_2d
from scipy.stats import norm

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14

def run_tracking(use_vepm=True, seed=42):
    """Run GPR-KG tracking variance estimate at a fixed unsampled solution."""
    np.random.seed(seed)
    d, L, sigma = 5, 20, 0.1
    prob = RZDT1(d=d, L=L, sigma=sigma)
    prob.calibrate_constraint(0.5)

    N, n0 = 150, 30
    lambda_i, prior_var = 0.1, 100.0

    gpr = [ParametricGPR(d, lambda_i, prior_var) for _ in range(3)]
    vepm = [VEPM(d, L) for _ in range(3)] if use_vepm else None

    # Pre-sampling
    pre_samples = set()
    while len(pre_samples) < n0:
        pre_samples.add(tuple(np.random.randint(1, L + 1, size=d)))
    pre_samples = list(pre_samples)

    for x_tuple in pre_samples:
        x_arr = np.array(x_tuple)
        for i in range(3):
            gpr[i].dimension_augment(x_arr)

    for x_tuple in pre_samples:
        x_arr = np.array(x_tuple)
        Y = prob.simulate(x_arr)
        for i in range(3):
            sigma2_init = 0.01
            e_tilde = gpr[i].augmented_feature(x_arr)
            Ce = gpr[i].C @ e_tilde
            denom = sigma2_init + e_tilde @ Ce
            if denom > 1e-15:
                innovation = Y[i] - e_tilde @ gpr[i].a
                gain = Ce / denom
                gpr[i].a = gpr[i].a + gain * innovation
                gpr[i].C = gpr[i].C - np.outer(gain, Ce)
                gpr[i].C = 0.5 * (gpr[i].C + gpr[i].C.T)
            if vepm:
                mu_i = gpr[i].posterior_mean(x_arr)
                vepm[i].update(i, x_tuple, Y[i], mu_i, gpr[i])

    # Pick a fixed unsampled solution to track
    target = tuple(np.random.randint(1, L + 1, size=d))
    while target in set(pre_samples):
        target = tuple(np.random.randint(1, L + 1, size=d))
    target_arr = np.array(target)

    # True variance at target (from problem definition)
    true_var = sigma ** 2  # Homoscedastic sigma=0.1, so var = 0.01

    variance_history = []  # (budget, estimated_var)
    # Without VEPM, the algorithm has no way to estimate variance at unsampled
    # solutions. The prior/initial estimate is the global_var from VEPM init,
    # which starts at 0.01 (a rough default). But for solutions in unvisited
    # partitions, the actual prior used would be the global prior_var parameter.
    # We use 1.0 to represent a typical uninformed prior.
    default_var = 1.0  # uninformed prior when VEPM is not available

    for n in range(n0, N):
        # Get current variance estimate
        if vepm:
            est_var = vepm[0].get_variance(0, target)
            if est_var is None:
                est_var = default_var
        else:
            est_var = default_var
        variance_history.append((n, est_var))

        # Simple iteration: pick random candidate, simulate, update
        x_next = tuple(np.random.randint(1, L + 1, size=d))
        x_arr = np.array(x_next)
        Y = prob.simulate(x_arr)
        for i in range(3):
            sigma2_hat = vepm[i].get_variance(i, x_next) if vepm else default_var
            if sigma2_hat is None:
                sigma2_hat = default_var
            gpr[i].update(x_arr, Y[i], max(sigma2_hat, 1e-8))
            if vepm:
                mu_i = gpr[i].posterior_mean(x_arr)
                vepm[i].update(i, x_next, Y[i], mu_i, gpr[i])

    return variance_history, true_var


def main():
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(save_dir, exist_ok=True)

    print("Running with VEPM...")
    hist_vepm, true_var = run_tracking(use_vepm=True, seed=42)
    print("Running without VEPM...")
    hist_novepm, _ = run_tracking(use_vepm=False, seed=42)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    stages_v = [h[0] for h in hist_vepm]
    vars_v = [h[1] for h in hist_vepm]
    stages_nv = [h[0] for h in hist_novepm]
    vars_nv = [h[1] for h in hist_novepm]

    ax.plot(stages_v, vars_v, color='#1f77b4', linewidth=2, label='With VEPM')
    ax.plot(stages_nv, vars_nv, color='#ff7f0e', linewidth=2, linestyle=':', label='Without VEPM')
    ax.axhline(y=true_var, color='#2ca02c', linestyle='--', linewidth=1.5,
               label=f'True variance ($\\sigma^2 = {true_var:.2f}$)', alpha=0.8)

    ax.set_xlabel('Number of Simulation Evaluations')
    ax.set_ylabel('Estimated Variance at Unsampled Solution')
    ax.set_title('VEPM Variance Convergence on RZDT1')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(save_dir, f'fig3_vepm_effect.{ext}'),
                    dpi=300 if ext == 'pdf' else 200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/fig3_vepm_effect.pdf")


if __name__ == '__main__':
    main()

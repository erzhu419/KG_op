"""
Performance metrics for multi-objective constrained simulation optimization.

Implements the four metrics from Section 6.1 of the paper:
- HV: Hypervolume Indicator (reuses compute_hypervolume_2d from gpr_kg.py)
- IGD: Inverted Generational Distance
- CVR: Constraint Violation Rate
- Wilcoxon signed-rank test for statistical comparison
"""

import numpy as np
from scipy.stats import wilcoxon
from scipy.spatial.distance import cdist


def compute_igd(estimated_pf, true_pf):
    """Inverted Generational Distance (IGD).

    IGD = (1/|PF*|) * sum_{p in PF*} min_{q in PF_est} ||p - q||_2

    Measures how well the estimated Pareto front covers the true front.
    Lower is better; 0 means perfect coverage.

    Args:
        estimated_pf: np.array shape (n, 2) - algorithm output (TRUE objective values).
        true_pf: np.array shape (m, 2) - true Pareto front points.

    Returns:
        float: IGD value (lower is better). Returns inf if estimated_pf is empty.
    """
    if len(true_pf) == 0:
        return 0.0
    if len(estimated_pf) == 0:
        return float('inf')

    true_pf = np.atleast_2d(true_pf)
    estimated_pf = np.atleast_2d(estimated_pf)

    # Distance from each true PF point to nearest estimated point
    dist_matrix = cdist(true_pf, estimated_pf, metric='euclidean')
    min_distances = np.min(dist_matrix, axis=1)
    return float(np.mean(min_distances))


def compute_cvr(pareto_solutions, problem):
    """Constraint Violation Rate (CVR).

    CVR = fraction of solutions in the output Pareto front that violate
    the true probabilistic constraint P(f3 + eps3 <= tau) >= 1 - alpha.

    Args:
        pareto_solutions: list of solution tuples (decision vectors).
        problem: TestProblem instance with is_truly_feasible() method.

    Returns:
        float: CVR in [0, 1] (lower is better). Returns 0 if no solutions.
    """
    if len(pareto_solutions) == 0:
        return 0.0

    n_violations = sum(1 for x in pareto_solutions if not problem.is_truly_feasible(x))
    return float(n_violations) / len(pareto_solutions)


def wilcoxon_test(values_a, values_b, alpha=0.05):
    """Wilcoxon signed-rank test for paired samples.

    Tests H0: distributions of a and b are the same.

    Args:
        values_a: array-like of metric values for method A (e.g., GPR-KG).
        values_b: array-like of metric values for method B (e.g., competitor).
        alpha: significance level.

    Returns:
        dict with 'statistic', 'p_value', 'significant' (bool).
    """
    a = np.array(values_a)
    b = np.array(values_b)

    # If all differences are zero, no significant difference
    if np.allclose(a, b):
        return {'statistic': 0.0, 'p_value': 1.0, 'significant': False}

    try:
        stat, p = wilcoxon(a, b, alternative='two-sided')
        return {'statistic': float(stat), 'p_value': float(p),
                'significant': p < alpha}
    except ValueError:
        # Fewer than 10 samples or all zeros
        return {'statistic': 0.0, 'p_value': 1.0, 'significant': False}

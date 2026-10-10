"""Source-task cross-validation selects jointly useful response functions.

The candidate dictionary is public. No target response enters selection or
coefficient moments. The soft arm keeps every dictionary function available;
the adaptive arm reuses the existing target-refitted spike-and-slab posterior.
"""

import numpy as np

from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior


def conditional_selection(batches, observation_variances, basis, active_dim=4):
    """Greedy added held-out value, with a one-source-task-SE stopping rule."""
    records = []
    for task, (batch, noise) in enumerate(zip(batches, observation_variances)):
        X = np.column_stack([np.ones(len(batch.psi)), basis.transform(batch.psi)])
        y = np.asarray(batch.signals).reshape(-1)
        noise = np.asarray(noise)
        order = np.random.default_rng(92000 + task).permutation(len(y))
        halves = np.array_split(order, 2)
        scale = max(float(np.var(y) - np.mean(noise)), float(np.mean(noise)), 1e-8)
        records.append((X, y, noise, halves, scale))

    def losses(indices):
        columns = [0] + [i + 1 for i in indices]
        task_losses = []
        for X, y, noise, halves, scale in records:
            fold_losses = []
            for fold in range(2):
                train, test = halves[fold], halves[1 - fold]
                T = X[np.ix_(train, columns)]
                weights = 1.0 / np.maximum(noise[train], 1e-8)
                penalty = np.eye(len(columns)) * 1e-3
                penalty[0, 0] = 1e-8
                beta = np.linalg.solve(T.T @ (weights[:, None] * T) + penalty,
                                       T.T @ (weights * y[train]))
                residual = y[test] - X[np.ix_(test, columns)] @ beta
                fold_losses.append(float(np.mean(residual ** 2) / scale))
            task_losses.append(np.mean(fold_losses))
        return np.asarray(task_losses)

    selected, history = [], []
    current = losses(selected)
    for _ in range(active_dim):
        remaining = [j for j in range(basis.feature_dim) if j not in selected]
        alternatives = [(float(value.mean()), j, value) for j in remaining
                        for value in [losses(selected + [j])]]
        _, chosen, proposed = min(alternatives, key=lambda row: (row[0], row[1]))
        delta = current - proposed
        se = float(np.std(delta, ddof=1) / np.sqrt(len(delta)))
        accepted = bool(delta.mean() > se)
        history.append({"index": chosen, "name": basis.names[chosen],
                        "added_cv_improvement": float(delta.mean()),
                        "source_task_standard_error": se, "accepted": accepted})
        if not accepted:
            break
        selected.append(chosen)
        current = proposed
    return selected, history


def fit_conditional_priors(batches, noise, public_candidates, active_dim=4):
    full = FixedFunctionBasis(public_candidates)
    selected, history = conditional_selection(batches, noise, full, active_dim)
    hard_basis = FixedFunctionBasis(public_candidates, selected)
    hard = SourceFunctionPrior.from_basis(batches, noise, hard_basis, "conditional_hard")
    hard.diagnostics.update({"selected_indices": selected, "selection_cv": history,
                             "selection": "conditional_two_fold_source_task_cv"})

    # Estimate moments in the selected function coordinates, then retain a
    # weaker total variance budget for functions not selected on the sources.
    m = full.feature_dim
    active = np.array(selected, dtype=int) + 1
    total_signal = float(np.mean([max(float(np.var(b.signals) - np.mean(v)), 0.02)
                                 for b, v in zip(batches, noise)]))
    tail_variance = 0.10 * total_signal / max(m - len(selected), 1)
    covariance = np.diag([hard.covariance[0, 0]] + [tail_variance] * m)
    covariance[np.ix_(active, active)] = hard.covariance[1:, 1:]
    mean = np.zeros(m + 1)
    mean[0] = hard.mean[0]
    info = {**full.diagnostics(), "mode": "conditional_soft",
            "source_selected_names": hard_basis.names,
            "source_selected_indices": selected, "tail_variance_budget": 0.10 * total_signal,
            "coefficient_mean": mean.tolist(), "selection_cv": history}
    soft = SourceFunctionPrior(full, mean, covariance, hard.discrepancy_variance, info)

    # Leave-one-source-task-out selection frequency measures source stability,
    # rather than claiming the empirical frequency is a calibrated probability.
    frequency = np.zeros(m)
    for left_out in range(len(batches)):
        keep = [i for i in range(len(batches)) if i != left_out]
        support, _ = conditional_selection([batches[i] for i in keep],
                                            [noise[i] for i in keep], full, active_dim)
        frequency[support] += 1.0 / len(batches)
    pip = 0.05 + 0.90 * frequency
    selected_scales = np.sqrt(np.diag(hard.covariance)[1:] / total_signal)
    slab = np.full(m, float(np.median(selected_scales)) if len(selected_scales) else 0.5)
    slab[selected] = selected_scales
    adaptive = SourceFunctionPrior(full, mean.copy(), covariance.copy(),
                                   hard.discrepancy_variance,
                                   {**info, "mode": "conditional_adaptive",
                                    "source_selection_frequency": frequency.tolist(),
                                    "source_pip": pip.tolist(), "source_slab_scale": slab.tolist(),
                                    "scale_units": "coefficient_sd_over_source_response_sd"})
    return {"conditional_hard": hard, "conditional_soft": soft,
            "conditional_adaptive": adaptive}


def enable_target_adaptation(model, prior, records):
    spec = {"source_pip": prior.diagnostics["source_pip"],
            "source_slab_scale": prior.diagnostics["source_slab_scale"],
            "spike_ratio": 0.08, "multiplicity_correction": 1.0,
            "max_iter": 40, "residual_floor_scale": 0.05}
    model.enable_adaptive_sparsity(spec, [r[0] for r in records],
                                   [r[1] for r in records], [r[2] for r in records],
                                   deviation_variance=prior.discrepancy_variance)

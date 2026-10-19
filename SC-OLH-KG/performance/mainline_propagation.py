"""Small diagnostics for propagation through an aggregate joint function field.

States contain ``mean[N,4]`` and ``cov[N,4,N,4]`` in f, eta_f, g, eta_g
order. Truth enters only the separate, post-recommendation scoring function.
"""

import numpy as np

HEADS = ("f", "eta_f", "g", "eta_g")


def point_covariance(state):
    covariance = np.asarray(state["cov"])
    index = np.arange(len(state["mean"]))
    return covariance[index, :, index, :]


def snapshot(state, weights):
    """Store marginal moments and model weights without the full joint array."""
    return {
        "head_order": list(HEADS),
        "mean": np.asarray(state["mean"]).tolist(),
        "point_cov": point_covariance(state).tolist(),
        "identity_weights": np.asarray(weights).tolist(),
    }


def _delta_summary(values, tolerance):
    values = np.asarray(values)
    return {
        "signed_mean": float(np.mean(values)) if values.size else 0.0,
        "max_absolute": float(np.max(abs(values))) if values.size else 0.0,
        "median_absolute": float(np.median(abs(values))) if values.size else 0.0,
        "changed_count": int(np.sum(abs(values) > tolerance)),
        "increased_count": int(np.sum(values > tolerance)),
        "decreased_count": int(np.sum(values < -tolerance)),
    }


def propagation(before, after, query, previously_observed, tolerance=1e-12):
    """Measure changes at policies never observed, excluding this query.

    Variance differences retain their sign: mixture reweighting and empirical
    updates can increase uncertainty. The saved query slice is the complete
    before-update Cov[field(policy), field(query)] for every policy and head.
    """
    prior_observed = sorted(set(int(i) for i in previously_observed))
    observed = sorted(set(prior_observed) | {int(query)})
    unobserved = [i for i in range(len(before["mean"])) if i not in observed]
    delta_mean = (np.asarray(after["mean"])-np.asarray(before["mean"]))[unobserved]
    before_variance = np.diagonal(point_covariance(before), axis1=1, axis2=2)
    after_variance = np.diagonal(point_covariance(after), axis1=1, axis2=2)
    delta_variance = (after_variance-before_variance)[unobserved]
    return {
        "head_order": list(HEADS), "query": int(query),
        "previously_observed_indices": prior_observed, "observed_indices": observed,
        "unsampled_indices": unobserved, "unsampled_count": len(unobserved),
        "change_tolerance": tolerance,
        "mean_changed_policy_count": int(np.sum(np.any(abs(delta_mean) > tolerance, axis=1))),
        "variance_changed_policy_count": int(np.sum(np.any(abs(delta_variance) > tolerance, axis=1))),
        "by_head": {
            head: {"mean_delta": _delta_summary(delta_mean[:, j], tolerance),
                   "variance_delta": _delta_summary(delta_variance[:, j], tolerance)}
            for j, head in enumerate(HEADS)
        },
        "before_query_cross_covariance": np.asarray(before["cov"])[:, :, query, :].tolist(),
    }


def score_prediction(saved_snapshot, truth, indices=None):
    """Score frozen marginal predictions after recommendations are fixed."""
    mean = np.asarray(saved_snapshot["mean"])
    indices = list(range(len(mean))) if indices is None else list(indices)
    if not indices:
        return {"point_count": 0, "indices": [], "by_head": {}}
    error = mean[indices]-np.asarray(truth)[indices]
    variance = np.diagonal(np.asarray(saved_snapshot["point_cov"]), axis1=1, axis2=2)[indices]
    return {
        "point_count": len(indices), "indices": indices,
        "by_head": {
            head: {"bias": float(np.mean(error[:, j])),
                   "mse": float(np.mean(error[:, j]**2)),
                   "rmse": float(np.sqrt(np.mean(error[:, j]**2))),
                   "mean_predictive_variance": float(np.mean(variance[:, j]))}
            for j, head in enumerate(HEADS)
        },
    }

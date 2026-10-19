"""Fixed-order primary-certificate KG for the full public policy bank.

Only selection uses Gaussian moments. The caller retains the native mixture
for actual raw observations. This approximation drops cross-objective/risk
identity dependence and uses the historical 9-eta/12-SD terminal quadrature;
it is not a certified raw-KG ranking or the guarded-shortlist action class.
"""

import numpy as np

from core.legacy_joint_kg import legacy_joint_utility, legacy_raw_joint_kg


def moment_state(model):
    """Return latent four-head moments and exact mixture episode variances."""
    mean, cov = model.predict_joint()
    mean, cov = np.asarray(mean), np.asarray(cov)
    count = len(mean)
    noise = np.zeros((count, 2))
    eta_indices = 2 * np.arange(count) + 1
    for identity_weight, pair in zip(model.weights, model.components):
        for head, field in enumerate(pair):
            eta_mean = field.means[:, eta_indices]
            eta_variance = field.covariances[:, eta_indices, eta_indices]
            variances = np.exp(eta_mean + eta_variance / 2)
            noise[:, head] += identity_weight * (np.exp(field.log_weights) @ variances)
    if not (np.isfinite(mean).all() and np.isfinite(cov).all()
            and np.isfinite(noise).all() and (noise > 0).all()):
        raise ValueError("Moment selection requires finite moments and positive episode noise")
    return {"mean": mean, "cov": cov, "noise": noise}


def _protocol(protocol, remaining):
    if remaining < 2 or int(remaining) != remaining:
        raise ValueError("Direct verification needs at least two physical episodes")
    return dict(protocol, candidate_delta=protocol["familywise_delta"],
                verification_replicates=int(remaining))


def available_state(state, count):
    """Expose the released prefix; retain global policy indices and covariance."""
    return {"mean": state["mean"][:count], "cov": state["cov"][:count, :, :count, :],
            "noise": state["noise"][:count]}


def _utilities(state, remaining, protocol):
    mean, cov = state["mean"], state["cov"]
    indices = np.arange(len(mean))
    risk_cov = cov[:, 2:4, :, 2:4]
    utility, power = legacy_joint_utility(
        mean[:, 0], cov[indices, 0, indices, 0], mean[:, 2:4],
        risk_cov[indices, :, indices, :], _protocol(protocol, remaining))
    if not (np.isfinite(utility).all() and np.isfinite(power).all()):
        raise ValueError("The fixed-order terminal rule returned a nonfinite value")
    return utility, power


def terminal_action(state, remaining, protocol):
    """Choose among every public primary using all remaining calls directly."""
    utilities, powers = _utilities(state, remaining, protocol)
    chosen = int(np.argmax(utilities))
    return {
        "action": {"kind": "direct", "primary": chosen, "shortlist": [chosen],
                   "incumbent_position": 0, "budgets": [int(remaining)]},
        "predicted_utility": float(utilities[chosen]),
        "certificate_power": float(powers[chosen]),
        "utilities": utilities.tolist(),
        "certificate_powers": powers.tolist(),
    }


def query_action(state, remaining, protocol):
    """Select one raw episode from the entire bank, with fixed order three.

    Net value is a diagnostic proxy: adaptation minus the terminal budget
    opportunity cost. It does not stop the caller's registered two-round run.
    """
    before, _ = _utilities(state, remaining, protocol)
    after, _ = _utilities(state, remaining - 1, protocol)
    mean, cov = state["mean"], state["cov"]
    scores = legacy_raw_joint_kg(
        mean[:, 0], cov[:, 0, :, 0], mean[:, 2:4], cov[:, 2:4, :, 2:4],
        state["noise"], _protocol(protocol, remaining - 1), order=3)
    if not np.isfinite(scores).all():
        raise ValueError("The fixed-order KG rule returned a nonfinite query score")
    query = int(np.argmax(scores))
    budget_cost = float(before.max() - after.max())
    return {"query": query, "scores73": scores.tolist(),
            "adaptation_value": float(scores[query]),
            "verification_budget_cost": budget_cost,
            "estimated_net_proxy": float(scores[query] - budget_cost), "order": 3}

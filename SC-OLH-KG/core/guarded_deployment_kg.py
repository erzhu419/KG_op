"""Terminal reward and raw-observation KG for the actual guarded shortlist.

Posterior heads are independent joint (f,eta_f) and (g,eta_g) Gaussian moment
fields. Safety replications are independent between shortlist positions. In
the three legacy families, scalarized objective noise has a common direction
across policies, so paired CRNs give SD |sigma_f(c)-sigma_f(i)|. No true sigma
is consulted by acquisition. Conditional certification/comparison power is
integrated numerically; source/target parameter uncertainty is integrated by
scrambled Sobol draws. A failed deployment has reward zero.
"""

from functools import lru_cache

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import ndtr, ndtri
from scipy.stats import nct, norm, qmc, t

from representation.joint_function_risk import local_joint_moments


@lru_cache(maxsize=8)
def _sd_quadrature(reps, order=16):
    df = reps - 1
    j = np.arange(order, dtype=float)
    nodes, vectors = eigh_tridiagonal(2 * j + df / 2,
                                      np.sqrt(j[1:] * (j[1:] + df / 2 - 1)))
    return np.sqrt(2 * nodes / df), vectors[0] ** 2


@lru_cache(maxsize=8)
def _tolerance_factor(reps, alpha, delta):
    return float(nct.ppf(1 - delta, reps - 1, norm.ppf(1 - alpha) * np.sqrt(reps)) / np.sqrt(reps))


def conditional_safety_power(g, sigma, reps, alpha, delta):
    """P[mean + noncentral-t tolerance factor * sample SD <= 0]."""
    sd, weights = _sd_quadrature(reps)
    argument = np.sqrt(reps) * (-np.asarray(g)[..., None] / np.asarray(sigma)[..., None]
                               - _tolerance_factor(reps, alpha, delta) * sd)
    return ndtr(argument) @ weights


def conditional_comparison_power(difference, difference_sd, reps, delta):
    """Power of the actual one-sided paired t test, including equal noise."""
    sd, weights = _sd_quadrature(reps)
    difference, difference_sd = np.broadcast_arrays(difference, difference_sd)
    answer = np.asarray(difference < 0, dtype=float)
    varying = difference_sd > 0
    z = -difference[varying] * np.sqrt(reps) / difference_sd[varying]
    answer[varying] = ndtr(z[:, None] - t.ppf(1 - delta, reps - 1) * sd) @ weights
    return answer


def shortlist_outcomes(quality, powers, comparison_power, budgets, incumbent, comparison_budget):
    """Conditional reward/calls for the exact verify_frozen_shortlist branches.

    Last axis is shortlist order. If incumbent is primary, verification is
    sequential without a comparison. Otherwise primary and incumbent are
    always verified, then either compared, deployed singly or both rejected.
    """
    q, p = np.asarray(quality), np.asarray(powers)
    if incumbent == 0:
        reward = p[..., 0] * q[..., 0]
        cost = np.full_like(reward, budgets[0], dtype=float)
        fail = 1 - p[..., 0]
        for j in range(1, len(budgets)):
            reward += fail * p[..., j] * q[..., j]
            cost += fail * budgets[j]
            fail *= 1 - p[..., j]
        return reward, cost
    i = incumbent
    both = p[..., 0] * p[..., i]
    reward = (p[..., 0] * (1 - p[..., i]) * q[..., 0]
              + (1 - p[..., 0]) * p[..., i] * q[..., i]
              + both * (comparison_power * q[..., 0] + (1 - comparison_power) * q[..., i]))
    cost = np.full_like(reward, budgets[0] + budgets[i], dtype=float) + both * 2 * comparison_budget
    fail = (1 - p[..., 0]) * (1 - p[..., i])
    for j in range(1, len(budgets)):
        if j == i:
            continue
        reward += fail * p[..., j] * q[..., j]
        cost += fail * budgets[j]
        fail *= 1 - p[..., j]
    return reward, cost


def deployment_values(draws, shortlists, incumbent_positions, protocol):
    """Expected reward and calls for every candidate's frozen shortlist.

    draws has shape (S,K,4), columns f,eta_f,g,eta_g, with g centered at tau.
    Full cross-policy posterior covariance is retained in these joint draws.
    """
    quality = np.clip(1 - draws[..., 0] / protocol["objective_reference"], 0, 1)
    sigma_f, sigma_g = np.exp(.5 * draws[..., 1]), np.exp(.5 * draws[..., 3])
    budgets = protocol["candidate_budgets"]
    delta = protocol["familywise_delta"] / len(budgets)
    powers = {n: conditional_safety_power(draws[..., 2], sigma_g, n,
                                         protocol["chance_alpha"], delta) for n in set(budgets)}
    reward, cost = np.empty(len(shortlists)), np.empty(len(shortlists))
    for a, (indices, incumbent) in enumerate(zip(shortlists, incumbent_positions)):
        pair = [indices[0], indices[incumbent]]
        comparison = conditional_comparison_power(draws[:, pair[0], 0] - draws[:, pair[1], 0],
            np.abs(sigma_f[:, pair[0]] - sigma_f[:, pair[1]]),
            protocol["comparison_budget"], protocol["comparison_delta"])
        p = np.column_stack([powers[n][:, i] for i, n in zip(indices, budgets)])
        r, c = shortlist_outcomes(quality[:, indices], p, comparison, budgets,
                                 int(incumbent), protocol["comparison_budget"])
        reward[a], cost[a] = r.mean(), c.mean()
    return reward, cost


def _root(covariance):
    covariance = .5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    if values[0] < -1e-8 * max(float(values[-1]), 1.):
        raise ValueError("posterior moment covariance is not positive semidefinite")
    return vectors * np.sqrt(np.maximum(values, 0.))


def posterior_draws(f_model, g_model, power, seed):
    """One full posterior draw bank, shared by all competing terminal actions."""
    fm, fc = f_model.predict(full_cov=True)
    gm, gc = g_model.predict(full_cov=True)
    n = len(fm)
    u = qmc.Sobol(4 * n, scramble=True, seed=seed).random_base2(power)
    z = ndtri(u)
    f = fm.reshape(-1) + z[:, :2*n] @ _root(fc.reshape(2*n, 2*n)).T
    g = gm.reshape(-1) + z[:, 2*n:] @ _root(gc.reshape(2*n, 2*n)).T
    return np.concatenate([f.reshape(-1, n, 2), g.reshape(-1, n, 2)], axis=-1)


def _completion(model, draws, action, observed):
    """Same raw-Normal local moments and Gaussian completion as paid filtering."""
    mean, covariance = model.predict(full_cov=True)
    n = len(mean)
    C = covariance.reshape(2*n, 2*n)
    local = covariance[action, :, action, :]
    result = local_joint_moments(mean[action], local, float(observed), None, 1)
    gain = np.linalg.solve(local, C[2*action:2*action+2]).T
    before = draws[:, action, :]
    # Preserve the independent Gaussian conditional residual of the whole
    # field and replace its local coordinates by the updated projected pair.
    whitened = np.linalg.solve(_root(local), (before - mean[action]).T).T
    after = result["mean"] + whitened @ _root(result["covariance"]).T
    completed = draws.reshape(-1, 2*n) + (after - before) @ gain.T
    return completed.reshape(-1, n, 2)


def guarded_raw_kg(f_model, g_model, shortlists, incumbent_positions, protocol,
                   *, draw_power=7, fantasy_power=3, seed=80508):
    """KG adaptation value for complete deployment, under moment projection.

    Fantasies come from the raw Normal/lognormal-variance predictive model.
    Their posterior update is identical to a paid single-episode update. The
    score compares the future best shortlist with today's selected shortlist
    in that same fantasy, removing quadrature drift in the baseline. Finite
    Sobol integration and sequential moment projection remain approximations.
    """
    draws = posterior_draws(f_model, g_model, draw_power, seed)
    values, costs = deployment_values(draws, shortlists, incumbent_positions, protocol)
    incumbent = int(values.argmax())
    z = ndtri(qmc.Sobol(6, scramble=True, seed=seed + 1).random_base2(fantasy_power))
    scores = np.empty(len(shortlists))
    fm, fc = f_model.predict(full_cov=True)
    gm, gc = g_model.predict(full_cov=True)
    for action in range(len(shortlists)):
        f_local = fm[action] + z[:, :2] @ _root(fc[action, :, action, :]).T
        g_local = gm[action] + z[:, 3:5] @ _root(gc[action, :, action, :]).T
        f_obs = f_local[:, 0] + np.exp(.5 * f_local[:, 1]) * z[:, 2]
        g_obs = g_local[:, 0] + np.exp(.5 * g_local[:, 1]) * z[:, 5]
        improvements = []
        for f, g in zip(f_obs, g_obs):
            fd = _completion(f_model, draws[..., :2], action, f)
            gd = _completion(g_model, draws[..., 2:], action, g)
            future, _ = deployment_values(np.concatenate([fd, gd], axis=-1),
                                         shortlists, incumbent_positions, protocol)
            improvements.append(future.max() - future[incumbent])
        scores[action] = np.mean(improvements)
    return {"reward": values, "expected_verification_calls": costs, "kg": scores,
            "recommended": incumbent, "draw_count": len(draws), "fantasy_count": len(z)}

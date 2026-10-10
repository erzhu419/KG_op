"""Accounting for a frozen four-branch risk predictor, with no prior fitting."""
import numpy as np
from scipy.special import logsumexp

from performance.function_covariance_source import predictive_scores


def posterior_mass(evidence, prior):
    with np.errstate(divide="ignore"):
        logs = evidence + np.log(prior)
    return np.exp(logs - logsumexp(logs, axis=1)[:, None])


def decompose(evidence, predictive, prior, same_pool_ir_scores):
    prior = np.asarray(prior)
    no_f, no_g, no_fg = prior.copy(), prior.copy(), prior.copy()
    no_f[0] += no_f[1]
    no_f[1] = 0.
    no_g[0] += no_g[2]
    no_g[2] = 0.
    no_fg[0] += no_fg[1] + no_fg[2]
    no_fg[1:3] = 0.
    score, removed_f, removed_g, removed_fg = [predictive_scores(evidence, predictive, masses)
                                             for masses in (prior, no_f, no_g, no_fg)]
    fg = score - removed_fg
    q_effect = removed_fg - same_pool_ir_scores
    f_effect = .5 * ((score - removed_f) + (removed_g - removed_fg))
    g_effect = .5 * ((score - removed_g) + (removed_f - removed_fg))
    return {"log_score": score, "fg_effect": fg, "q_effect": q_effect,
            "f_accounting_effect": f_effect, "g_accounting_effect": g_effect,
            "pure_f_gain": predictive[:, 1] - predictive[:, 0],
            "pure_g_gain": predictive[:, 2] - predictive[:, 0],
            "pure_r_gain": predictive[:, 3] - predictive[:, 0],
            "same_q_ir_gain": removed_fg - predictive[:, 0],
            "posterior_macro_mass": posterior_mass(evidence, prior),
            "same_q_ir_posterior_macro_mass": posterior_mass(evidence, no_fg)}

"""Eight joint structural/full configurations over six existing Gaussian states."""
from itertools import product

import numpy as np
from scipy.special import logsumexp, roots_legendre
from scipy.stats import norm

from representation.conditional_function_prior import conditional_selection
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior, head_batches
from representation.source_risk_ard import fit_source_risk_ard
from representation.transfer_mixture import full_dictionary_prior

# Component index 0 is structural, 1 is full, in the original model ordering.
CHOICES = np.array(list(product((0, 1), repeat=3)))


def joint_log_weights(head_evidence, alpha):
    prior = np.full(8, (1. - alpha) / 8.)
    prior[[0, 7]] += alpha / 2.
    with np.errstate(divide="ignore"):
        logs = np.log(prior) + sum(np.asarray(head_evidence)[h, CHOICES[:, h]] for h in range(3))
    return logs - logsumexp(logs)


def structural_weights(head_evidence, alpha):
    logs = joint_log_weights(head_evidence, alpha)
    return np.array([np.exp(logsumexp(logs[CHOICES[:, h] == 0])) for h in range(3)])


def branch_scores(forecasts):
    evidence = np.array([f["head_log_evidence"] for f in forecasts])
    eta_scores = np.array([f["eta_component_log_density"] for f in forecasts])
    logs = sum(evidence[:, h, CHOICES[:, h]] for h in range(3))
    numerator = logs + eta_scores[:, CHOICES[:, 2]]
    return (logsumexp(logs, axis=1) - np.log(8.),
            logsumexp(logs[:, [0, 7]], axis=1) - np.log(2.),
            logsumexp(numerator, axis=1) - np.log(8.),
            logsumexp(numerator[:, [0, 7]], axis=1) - np.log(2.))


def predictive_scores(forecasts, alpha):
    """Scalar alpha returns N scores; a node vector returns an N-by-K array."""
    independent, coordinated, numerator_i, numerator_c = branch_scores(forecasts)
    alpha = np.asarray(alpha)
    if alpha.ndim:
        independent, coordinated, numerator_i, numerator_c = (
            v[:, None] for v in (independent, coordinated, numerator_i, numerator_c))
    with np.errstate(divide="ignore"):
        a, b = np.log1p(-alpha), np.log(alpha)
    return (np.logaddexp(a + numerator_i, b + numerator_c)
            - np.logaddexp(a + independent, b + coordinated))


def fit_source_alpha(forecasts, nodes=64):
    points, weights = roots_legendre(nodes)
    alpha = (points + 1.) / 2.
    logs = predictive_scores(forecasts, alpha).sum(axis=0) + np.log(weights / 2.)
    masses = np.exp(logs - logsumexp(logs))
    mean = float(masses @ alpha)
    return {"alpha_mean": mean, "alpha_sd": float(np.sqrt(masses @ ((alpha - mean) ** 2))),
            "source_forecasts": len(forecasts), "quadrature_nodes": nodes,
            "source_cv_log_score_at_zero": float(predictive_scores(forecasts, 0.).sum()),
            "source_cv_log_score_at_one": float(predictive_scores(forecasts, 1.).sum())}


def calibrate_source_archives(rows, nodes=64):
    groups = {}
    for row in rows:
        groups.setdefault((row["seed"], row["archive_group"]), []).extend(row["forecasts"])
    return [{"seed": seed, "archive_group": archive, **fit_source_alpha(forecasts, nodes)}
            for (seed, archive), forecasts in groups.items()]


class CrossHeadStructureMixture:
    def __init__(self, structural, full, psi, reference):
        self.models = [[full_dictionary_prior(p, reference).target_model(psi)
                        for p in (structural[h], full[h])] for h in range(3)]
        self.log_evidence = np.zeros((3, 2))
        self.observations = 0

    def predict_eta(self, x, observation_variance):
        return {"head_log_evidence": self.log_evidence.tolist(),
                "eta_component_mean": [m.posterior_mean(x) for m in self.models[2]],
                "eta_component_variance": [m.posterior_var(x) + observation_variance for m in self.models[2]]}

    def update(self, x, response, noise):
        for h, models in enumerate(self.models):
            for c, model in enumerate(models):
                self.log_evidence[h, c] += norm.logpdf(response[h], model.posterior_mean(x),
                    np.sqrt(model.posterior_var(x) + noise[h]))
                model.update(x, response[h], noise[h])
        self.observations += 1

    def marginal_moments(self, x, alpha=0.):
        weights = structural_weights(self.log_evidence, alpha)
        means, variances = [], []
        for h, models in enumerate(self.models):
            w = np.array([weights[h], 1. - weights[h]])
            mu = np.array([m.posterior_mean(x) for m in models])
            var = np.array([m.posterior_var(x) for m in models])
            mean = float(w @ mu)
            means.append(mean)
            variances.append(float(w @ (var + (mu - mean) ** 2)))
        return means, variances, weights.tolist()


def score_prediction(prediction, label):
    scores = norm.logpdf(label, prediction["eta_component_mean"],
                        np.sqrt(prediction["eta_component_variance"]))
    forecast = {**prediction, "eta_component_log_density": scores.tolist()}
    # This is also an independent calculation of the alpha=0 control.
    logs = np.array(prediction["head_log_evidence"])[2]
    forecast["independent_log_score"] = float(logsumexp(logs + scores) - logsumexp(logs))
    return forecast


def forecast_sequence(model, indices, observation, eta_noise, initial_points=8):
    forecasts = []
    for step, index in enumerate(indices, 1):
        x = np.array([index])
        prediction = model.predict_eta(x, eta_noise) if step > initial_points else None
        # Obtain the current response only after issuing the forecast.
        response, noise = observation(step - 1, index)
        if prediction is not None:
            forecasts.append({"step": step, "candidate": int(index),
                              **score_prediction(prediction, response[2])})
        model.update(x, response, noise)
    return forecasts


def fit_source_components(tasks, held_task, reference, active_dim=4):
    ids = [i for i in range(len(tasks)) if i != held_task]
    training = [tasks[i] for i in ids]
    full, structural = [], []
    for h in range(3):
        batches = head_batches(training, h)
        noise = [t["noise"][:, h] for t in training]
        full.append(SourceFunctionPrior.fit(batches, noise, reference, "fixed_source"))
        if h == 2:
            structural.append(fit_source_risk_ard(batches, noise, reference))
        else:
            selected, history = conditional_selection(batches, noise, FixedFunctionBasis(reference), active_dim)
            prior = SourceFunctionPrior.from_basis(batches, noise, FixedFunctionBasis(reference, selected),
                                                  "conditional_hard")
            prior.diagnostics.update(selected_indices=selected, selection_cv=history)
            structural.append(prior)
    info = {"held_task": held_task, "training_tasks": ids,
            "source_label_count_per_head": sum(len(t["psi"]) for t in training),
            "fg_selected_indices": [p.diagnostics["selected_indices"] for p in structural[:2]],
            "optimizer_converged": structural[2].diagnostics["optimizer_converged"],
            "optimizer_iterations": structural[2].diagnostics["optimizer_iterations"]}
    return full, structural, info

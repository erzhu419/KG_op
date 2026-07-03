"""Decomposition-aware OLH-KG acquisition."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from core.kg import compute_kg_vectorized


def safe_normalize(values):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo <= 1e-14:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


class OLHKGAcquisition:
    """Approximate OLH-KG score.

    score = KG_obj + lambda_f KG_feas + lambda_v KG_var + lambda_rho KG_coupling

    Setting all auxiliary lambdas to zero returns the raw objective KG exactly.
    """

    def __init__(
        self,
        lambda_feas=0.25,
        lambda_var=0.25,
        lambda_coupling=0.0,
        boundary_scale=1.0,
        encoder=None,
    ):
        self.lambda_feas = float(lambda_feas)
        self.lambda_var = float(lambda_var)
        self.lambda_coupling = float(lambda_coupling)
        self.boundary_scale = max(float(boundary_scale), 1e-8)
        self.encoder = encoder

    @staticmethod
    def chance_margin(mu_g, variance_g, tau, alpha):
        return float(mu_g + norm.ppf(1 - alpha) * np.sqrt(max(variance_g, 1e-12)) - tau)

    def feasibility_scores(self, candidates, con_gpr, variance_model, problem):
        if len(candidates) == 0:
            return np.zeros(0, dtype=float), []
        mu_g = con_gpr.posterior_mean_many(candidates)
        if hasattr(variance_model, "predict_certification_variance_many"):
            v_g = variance_model.predict_certification_variance_many(
                1, candidates, problem)
        else:
            var_fn = getattr(
                variance_model,
                "predict_certification_variance",
                variance_model.predict_variance,
            )
            v_g = np.array([var_fn(1, x, problem) for x in candidates], dtype=float)
        sig_g = np.sqrt(np.maximum(v_g, 1e-12))
        z = norm.ppf(1 - problem.alpha)
        margins = mu_g + z * sig_g - problem.tau
        scale = np.maximum(self.boundary_scale * sig_g, 1e-8)
        raw = np.exp(-0.5 * (margins / scale) ** 2) * sig_g
        details = [
            {
                "x": list(map(int, candidates[j])),
                "mu_g": float(mu_g[j]),
                "variance_g": float(v_g[j]),
                "chance_margin": float(margins[j]),
            }
            for j in range(len(candidates))
        ]
        return safe_normalize(raw), details

    def variance_scores(self, candidates, variance_model, problem, output_index=1):
        if hasattr(variance_model, "variance_information_many"):
            raw = variance_model.variance_information_many(
                output_index, candidates, problem)
        else:
            raw = np.array([
                variance_model.variance_information(output_index, x, problem)
                for x in candidates
            ], dtype=float)
        return safe_normalize(raw)

    def coupling_scores(self, candidates, observed):
        if self.encoder is None:
            return np.zeros(len(candidates), dtype=float)
        return self.encoder.propagation_scores(candidates, observed)

    def score(
        self,
        candidates,
        obj_gpr,
        con_gpr,
        variance_model,
        problem,
        observed=None,
    ):
        if len(candidates) == 0:
            return {
                "total": np.zeros(0, dtype=float),
                "kg_obj": np.zeros(0, dtype=float),
                "kg_feas": np.zeros(0, dtype=float),
                "kg_var": np.zeros(0, dtype=float),
                "kg_coupling": np.zeros(0, dtype=float),
                "feasibility_details": [],
            }
        if hasattr(variance_model, "predict_variance_many"):
            sigma2_obj = variance_model.predict_variance_many(0, candidates, problem)
        else:
            sigma2_obj = np.array([
                variance_model.predict_variance(0, x, problem)
                for x in candidates
            ], dtype=float)
        kg_obj = compute_kg_vectorized(obj_gpr, candidates, sigma2_obj)
        kg_feas, feas_details = self.feasibility_scores(
            candidates, con_gpr, variance_model, problem)
        kg_var = self.variance_scores(candidates, variance_model, problem, 1)
        kg_coupling = self.coupling_scores(candidates, observed or [])
        total = (
            kg_obj
            + self.lambda_feas * kg_feas
            + self.lambda_var * kg_var
            + self.lambda_coupling * kg_coupling
        )
        return {
            "total": total,
            "kg_obj": kg_obj,
            "kg_feas": kg_feas,
            "kg_var": kg_var,
            "kg_coupling": kg_coupling,
            "feasibility_details": feas_details,
        }

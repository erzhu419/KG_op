"""Decomposition-aware OLH-KG acquisition."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from core.certification import conservative_chance_margin
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

    score = KG_obj + lambda_f KG_feas + lambda_v KG_var + lambda_m KG_mean
        + lambda_e KG_constraint_epistemic + lambda_rho KG_coupling

    Setting all auxiliary lambdas to zero returns the raw objective KG exactly.
    """

    def __init__(
        self,
        lambda_feas=0.25,
        lambda_var=0.25,
        lambda_mean=0.0,
        lambda_constraint_epistemic=0.0,
        lambda_coupling=0.0,
        boundary_scale=1.0,
        constraint_epistemic_margin_softening=3.0,
        coupling_safety_z=0.5,
        coupling_gate_temperature=0.25,
        beta_g=0.0,
        certification_mode="legacy",
        encoder=None,
    ):
        self.lambda_feas = float(lambda_feas)
        self.lambda_var = float(lambda_var)
        self.lambda_mean = float(lambda_mean)
        self.lambda_constraint_epistemic = float(lambda_constraint_epistemic)
        self.lambda_coupling = float(lambda_coupling)
        self.boundary_scale = max(float(boundary_scale), 1e-8)
        self.constraint_epistemic_margin_softening = max(
            float(constraint_epistemic_margin_softening), 1e-8)
        self.coupling_safety_z = float(coupling_safety_z)
        self.coupling_gate_temperature = max(float(coupling_gate_temperature), 1e-8)
        self.beta_g = float(beta_g)
        self.certification_mode = str(certification_mode or "legacy")
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
        if hasattr(con_gpr, "posterior_var_many"):
            epistemic = con_gpr.posterior_var_many(candidates)
        else:
            epistemic = np.array([
                con_gpr.posterior_var(x) for x in candidates
            ], dtype=float)
        cert = conservative_chance_margin(
            mu_g,
            epistemic,
            v_g,
            tau=problem.tau,
            alpha=problem.alpha,
            beta_g=self.beta_g,
            mode=self.certification_mode,
        )
        total_var = (
            np.maximum(cert.aleatoric_var, 1e-12)
            + max(float(cert.beta_g), 0.0) * np.maximum(cert.epistemic_var, 0.0)
        )
        sig_g = np.sqrt(np.maximum(total_var, 1e-12))
        margins = cert.margin
        scale = np.maximum(self.boundary_scale * sig_g, 1e-8)
        raw = np.exp(-0.5 * (margins / scale) ** 2) * sig_g
        details = [
            {
                "x": list(map(int, candidates[j])),
                "mu_g": float(mu_g[j]),
                "epistemic_variance_g": float(cert.epistemic_var[j]),
                "variance_g": float(cert.aleatoric_var[j]),
                "total_certification_variance_g": float(total_var[j]),
                "beta_g": float(cert.beta_g),
                "certification_mode": cert.mode,
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

    def mean_scores(self, candidates, obj_gpr, feasibility_details):
        if len(candidates) == 0:
            return np.zeros(0, dtype=float)
        mu_obj = obj_gpr.posterior_mean_many(candidates)
        exploit = safe_normalize(-mu_obj)
        if not feasibility_details:
            return exploit
        margins = np.array([
            float(item["chance_margin"]) for item in feasibility_details
        ], dtype=float)
        variance = np.array([
            float(item["variance_g"]) for item in feasibility_details
        ], dtype=float)
        total_variance = np.array([
            float(item.get("total_certification_variance_g", item["variance_g"]))
            for item in feasibility_details
        ], dtype=float)
        sig = np.sqrt(np.maximum(total_variance, np.maximum(variance, 1e-12)))
        scaled = -margins / (1.5 * sig)
        scaled = np.clip(scaled, -60.0, 60.0)
        feasible_gate = 1.0 / (1.0 + np.exp(-scaled))
        return exploit * feasible_gate

    def constraint_epistemic_scores(self, feasibility_details):
        """Reward constraint-mean learning near plausible safety boundaries.

        This is deliberately based only on posterior quantities.  It targets
        the failure mode where the candidate pool contains truly safe points
        but the constraint mean model is still too uncertain or misranked to
        certify them.
        """
        if not feasibility_details:
            return np.zeros(0, dtype=float)
        margins = np.array([
            float(item["chance_margin"]) for item in feasibility_details
        ], dtype=float)
        epistemic = np.array([
            float(item["epistemic_variance_g"]) for item in feasibility_details
        ], dtype=float)
        total_variance = np.array([
            float(item.get("total_certification_variance_g", item["variance_g"]))
            for item in feasibility_details
        ], dtype=float)
        sig = np.sqrt(np.maximum(total_variance, 1e-12))
        softened = self.constraint_epistemic_margin_softening * sig
        boundary_weight = np.exp(-0.5 * (margins / np.maximum(softened, 1e-8)) ** 2)
        raw = np.sqrt(np.maximum(epistemic, 0.0)) * (0.25 + boundary_weight)
        return safe_normalize(raw)

    def coupling_scores(self, candidates, observed):
        if self.encoder is None:
            return np.zeros(len(candidates), dtype=float)
        if hasattr(self.encoder, "coupling_scores"):
            return self.encoder.coupling_scores(candidates, observed)
        return self.encoder.propagation_scores(candidates, observed)

    def coupling_feasibility_gate(self, feasibility_details):
        if not feasibility_details:
            return np.zeros(0, dtype=float)
        margins = np.array([
            float(item["chance_margin"]) for item in feasibility_details
        ], dtype=float)
        variance = np.array([
            float(item["variance_g"]) for item in feasibility_details
        ], dtype=float)
        total_variance = np.array([
            float(item.get("total_certification_variance_g", item["variance_g"]))
            for item in feasibility_details
        ], dtype=float)
        sig = np.sqrt(np.maximum(total_variance, np.maximum(variance, 1e-12)))
        guarded_margin = margins + self.coupling_safety_z * sig
        scaled = -guarded_margin / (self.coupling_gate_temperature * sig)
        scaled = np.clip(scaled, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-scaled))

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
                "kg_obj_scaled": np.zeros(0, dtype=float),
                "kg_feas": np.zeros(0, dtype=float),
                "kg_var": np.zeros(0, dtype=float),
                "kg_mean": np.zeros(0, dtype=float),
                "kg_constraint_epistemic": np.zeros(0, dtype=float),
                "kg_coupling": np.zeros(0, dtype=float),
                "kg_coupling_raw": np.zeros(0, dtype=float),
                "kg_coupling_gate": np.zeros(0, dtype=float),
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
        kg_mean = self.mean_scores(candidates, obj_gpr, feas_details)
        kg_coupling_raw = self.coupling_scores(candidates, observed or [])
        kg_coupling_gate = self.coupling_feasibility_gate(feas_details)
        aux_active = (
            abs(self.lambda_feas) > 0.0
            or abs(self.lambda_var) > 0.0
            or abs(self.lambda_mean) > 0.0
            or abs(self.lambda_constraint_epistemic) > 0.0
            or abs(self.lambda_coupling) > 0.0
        )
        kg_obj_scaled = safe_normalize(kg_obj) if aux_active else kg_obj
        kg_constraint_epistemic = self.constraint_epistemic_scores(feas_details)
        if abs(self.lambda_coupling) > 0.0:
            relevance = safe_normalize(
                kg_obj_scaled
                + kg_feas
                + kg_mean
                + kg_constraint_epistemic
                + 0.5 * kg_var)
            kg_coupling = kg_coupling_raw * relevance * kg_coupling_gate
        else:
            kg_coupling = kg_coupling_raw
        total = (
            kg_obj_scaled
            + self.lambda_feas * kg_feas
            + self.lambda_var * kg_var
            + self.lambda_mean * kg_mean
            + self.lambda_constraint_epistemic * kg_constraint_epistemic
            + self.lambda_coupling * kg_coupling
        )
        return {
            "total": total,
            "kg_obj": kg_obj,
            "kg_obj_scaled": kg_obj_scaled,
            "kg_feas": kg_feas,
            "kg_var": kg_var,
            "kg_mean": kg_mean,
            "kg_constraint_epistemic": kg_constraint_epistemic,
            "kg_coupling": kg_coupling,
            "kg_coupling_raw": kg_coupling_raw,
            "kg_coupling_gate": kg_coupling_gate,
            "feasibility_details": feas_details,
        }

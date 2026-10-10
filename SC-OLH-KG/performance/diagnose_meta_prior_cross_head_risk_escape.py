"""Decompose frozen A15 scores; diagnostic counterfactuals are not new methods."""
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.cross_head_risk_escape import macro_prior, mixture_scores, posterior_macro_masses
from performance.run_meta_prior_cross_head_risk_escape import ARTIFACTS, contrast, read, save


def main():
    start = time.process_time()
    target = read(ARTIFACTS / "target_validation.json")
    spec = read(ARTIFACTS / "execution_plan.json")
    if not target["complete"]:
        raise ValueError("A15 decomposition requires completed validation")
    diagnostic_scope = {
        "scope": "Explanatory score decomposition only, not calibration or an additional competing method.",
        "fixed_before_decomposition_results": True,
        "identity": "A15-A12 = coordination_effect + source_q_effect. Remove C but keep A15 q to isolate coordination, then compare that eta-only mixture with A12 to isolate source-q recalibration.",
        "mechanism": "Compare actual R posterior mass with its eta-only counterpart at the same q, and retain saved R/I past-evidence ratios. No labels, priors, covariance, samples or source calibration are changed.",
        "new_component_fits": 0, "server_jobs": 0,
    }
    save("decomposition_plan.json", diagnostic_scope)
    rows, max_error = [], 0.
    for row in target["rows"]:
        forecasts = row["forecasts"]
        evidence = np.array([f["relative_past_log_evidence"] for f in forecasts])
        predictive = np.array([f["component_log_density"] for f in forecasts])
        actual = np.array([f["log_score"] for f in forecasts])
        old12 = np.array([f["a12_log_score"] for f in forecasts])
        q = row["source_macro_prior_mass"][2]
        eta_only = mixture_scores(evidence, predictive, macro_prior(0., q))
        coordination, source_q = actual - eta_only, eta_only - old12
        max_error = max(max_error, float(np.max(np.abs((actual - old12) - coordination - source_q))))
        eta_mass = posterior_macro_masses(evidence, macro_prior(0., q))[:, 2]
        actual_mass = np.array([f["posterior_macro_mass"][2] for f in forecasts])
        rows.append({**{k: row[k] for k in ("seed", "regime", "arm")},
            "coordination_effect": float(coordination.mean()), "source_q_effect": float(source_q.mean()),
            "eta_only_gain": float(np.mean(eta_only - predictive[:, 0])),
            "pure_c_gain": float(np.mean(predictive[:, 1] - predictive[:, 0])),
            "pure_r_gain": float(np.mean(predictive[:, 2] - predictive[:, 0])),
            "mean_r_mass_change_due_to_coordination": float(np.mean(actual_mass - eta_mass)),
            "eta_only_r_mass_step9": float(eta_mass[0]), "eta_only_r_mass_step24": float(eta_mass[-1]),
            "r_over_i_log_evidence_step9": float(evidence[0, 2]),
            "r_over_i_log_evidence_step24": float(evidence[-1, 2])})
    indices = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(16, size=(20000, 16))
    groups = []
    contrasts = ("coordination_effect", "source_q_effect", "eta_only_gain", "pure_c_gain", "pure_r_gain")
    for regime in spec["cohort"]["target_regimes"]:
        for arm in spec["cohort"]["priors"]:
            selected = [r for r in rows if (r["regime"], r["arm"]) == (regime, arm)]
            groups.append({"regime": regime, "arm": arm,
                **{k: contrast([r[k] for r in selected], indices) for k in contrasts},
                **{k: float(np.mean([r[k] for r in selected])) for k in selected[0]
                   if k not in ("seed", "regime", "arm", *contrasts)}})
    if max_error > 1e-10:
        raise ValueError("score decomposition identity failed")
    result = {"complete": True, **diagnostic_scope, "rows": rows, "groups": groups,
              "max_decomposition_identity_error": max_error, "cpu_seconds": time.process_time() - start}
    save("decomposition.json", result)
    print(json.dumps({k: result[k] for k in ("complete", "max_decomposition_identity_error", "cpu_seconds")}))


if __name__ == "__main__":
    main()

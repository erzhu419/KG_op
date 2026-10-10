"""Replay paid paths to attribute the operational pilot; no new observations."""

import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.cumulative_risk import cumulative_feature_vector
from core.verification_kg import verification_utility
from problems.inventory_cost_plan import InventoryPlan, InventoryCostSimulator, sample_cost_law
from representation.physical_risk_prior import PhysicalRiskPrior, RiskBatchData, fit_source_risk_families
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior, head_batches
from representation.transfer_mixture import TransferMixtureGPR


def run():
    start = time.process_time()
    folder = ROOT / "paper_artifacts/inventory_physical_risk_20261004"
    saved = json.loads((folder / "closed_loop.json").read_text())
    assert saved["complete"]
    p = saved["protocol"]
    source = json.loads((ROOT / p["source_records"]).read_text())
    s = source["protocol"]
    old = json.loads((ROOT / s["existing_records"]).read_text())
    design = json.loads((ROOT / s["source_design"]).read_text())
    plans = [InventoryPlan(**context) for context in design["contexts"]]
    policies = np.asarray(old["public_policies"])
    psi = plans[0].coordinates_many(policies)
    X = np.array([cumulative_feature_vector(plans[0].trajectory(x).risk_exposures()) for x in policies])
    Xs = np.asarray(design["risk_design"])
    source_psi = np.array([plans[r["context"]].trajectory(policies[r["policy_index"]]).coordinates(plans[r["context"]].capacity)
                           for r in design["assignments"]])
    data = [{"psi": source_psi, "response": np.asarray(t["response"]), "noise": np.asarray(t["noise"])}
            for t in source["source_records"]]
    full, structural = [], []
    for h in range(2):
        batches, noises = head_batches(data, h), [t["noise"][:, h] for t in data]
        full.append(SourceFunctionPrior.fit(batches, noises, psi, "fixed_source"))
        support = saved["source_mean_info"][h]["selected_indices"]
        structural.append(SourceFunctionPrior.from_basis(batches, noises, FixedFunctionBasis(psi, support), "conditional_hard"))
    risk = fit_source_risk_families([RiskBatchData.from_log_labels(Xs, t["response"][:, 2], p["replicates"]) for t in data],
                                   np.vstack([Xs, X]), **s["source_risk_distribution"])
    pooled = PhysicalRiskPrior(risk.coefficients, np.full(len(risk.weights), 1. / len(risk.weights)),
                               risk.families, risk.family_names, {})
    counts = {r["law_seed"]: r["true_feasible_candidates"] for r in saved["rows"]}
    rows, observation_rows, replay_error = [], [], 0.
    # Feasibility selects failure cases for attribution only, never a method.
    for cell in saved["rows"]:
        if counts[cell["law_seed"]] == 0:
            continue
        means = ([prior.target_model(psi) for prior in full] if cell["arm"] == "same_source_general" else
                 [TransferMixtureGPR([structural[h], full[h]], psi, psi) for h in range(2)])
        rm = (risk if cell["arm"] == "structural_complete" else pooled).target_model()
        world = InventoryCostSimulator(plans[0], sample_cost_law(cell["law_seed"]))
        for step, observation in enumerate(cell["search_path"], 1):
            index, y, noise = observation["index"], observation["response"], observation["noise"]
            for h, model in enumerate(means):
                model.update([index], y[h], noise[h])
            rm.update_batch(RiskBatchData.from_log_labels(X[index:index+1], np.array(y[2:3]), p["replicates"]))
            if step not in (p["initial_points"], p["target_points"]):
                continue
            choice = cell["initial_choice"] if step == p["initial_points"] else cell["choice"]
            eta_m, eta_v = rm.predict(X[choice:choice+1])
            mean = np.array([[*[m.posterior_mean_many([[choice]])[0] for m in means], eta_m[0]]])
            variance = np.array([[*[m.posterior_var_many([[choice]])[0] for m in means], eta_v[0]]])
            truth = world.oracle_moments(policies[choice])[None, :]
            _, power = verification_utility(mean, variance, p)
            alternatives = {"posterior": float(power[0])}
            for name, heads, remove_uncertainty in (
                    ("true_g_center_preserved_uncertainty", [1], False),
                    ("true_mean_predicted_risk", [1], True),
                    ("predicted_mean_true_risk", [2], True),
                    ("truth", [0, 1, 2], True)):
                m, v = mean.copy(), variance.copy()
                m[:, heads] = truth[:, heads]
                if remove_uncertainty:
                    v[:, heads] = 0.
                _, counterfactual = verification_utility(m, v, p)
                alternatives[name] = float(counterfactual[0])
            rows.append({"seed": cell["law_seed"], "arm": cell["arm"], "step": step, "choice": choice,
                         "predicted_mean": mean[0].tolist(), "predicted_variance": variance[0].tolist(),
                         "true_mean": truth[0].tolist(), "verifier_power_counterfactuals": alternatives})
            if step == p["target_points"]:
                replay_error = max(replay_error, abs(float(power[0]) - cell["predicted_verifier_power"]))
                queried = [r for r in cell["search_path"] if r["index"] == choice]
                for query in queried:
                    observation_rows.append({"seed": cell["law_seed"], "arm": cell["arm"], "choice": choice,
                        "visit": query["visit"], "sample_g_mean": query["response"][1], "true_g_mean": float(truth[0, 1]),
                        "posterior_g_mean": float(mean[0, 1]),
                        "batch_error_in_true_standard_errors": float((query["response"][1] - truth[0, 1]) / np.sqrt(np.exp(truth[0, 2]) / p["replicates"])),
                        "sample_to_true_variance_ratio": float(query["noise"][1] * p["replicates"] / np.exp(truth[0, 2])),
                        "posterior_mean_to_sample_mean_difference": float(mean[0, 1] - query["response"][1])})
    assert replay_error < 1e-10
    anchors = np.repeat(np.array([0., .25, .5, .75, 1.])[:, None], 4, axis=1)
    anchor_rows = []
    for seed, count in counts.items():
        world = InventoryCostSimulator(plans[0], sample_cost_law(seed))
        truth = np.array([world.oracle_moments(x) for x in anchors])
        margins = truth[:, 1] + norm.ppf(1 - p["chance_alpha"]) * np.exp(.5 * truth[:, 2])
        anchor_rows.append({"law_seed": seed, "fixed_bank_feasible_candidates": count,
            "constant_level_anchor_margins": margins.tolist(), "constant_level_anchor_feasible_count": int(np.sum(margins <= 0))})
    return {"purpose": "Post-pilot oracle attribution on cached paid paths, not a new method or independent confirmation",
        "feasible_candidate_counts": counts, "rows": rows, "mean_observation_attribution": observation_rows,
        "source_mean_discrepancy_variances": {"structural": [m.discrepancy_variance for m in structural],
                                            "general": [m.discrepancy_variance for m in full]},
        "replayed_verifier_power_max_absolute_error": replay_error,
        "public_anchor_diagnostic": {"purpose": "Five fixed public constant-base-stock anchors, evaluated by private truth only after the pilot; no global feasibility claim",
            "fractions": [0., .25, .5, .75, 1.], "rows": anchor_rows, "new_episode_calls": 0},
        "new_episode_calls": 0, "local_cpu_seconds": time.process_time() - start}


if __name__ == "__main__":
    result = run()
    (ROOT / "paper_artifacts/inventory_physical_risk_20261004/feasible_task_attribution.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"rows": len(result["rows"]), "replay_error": result["replayed_verifier_power_max_absolute_error"],
                      "local_cpu_seconds": result["local_cpu_seconds"]}))

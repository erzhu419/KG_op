"""Attribute the first adjusted-budget regression from the completed pilot."""

import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.joint_verification_kg import budgeted_joint_kg, joint_verification_kg
from core.verification_kg import objective_quality, particle_pass_probability, verification_utility
from performance.run_inventory_joint_kg import prepare, models, update, moments, measurement_noise, recommend
from problems.inventory_cost_plan import InventoryCostSimulator, sample_cost_law


if __name__ == "__main__":
    start = time.process_time()
    folder = ROOT / "paper_artifacts/inventory_physical_risk_20261004"
    saved = json.loads((folder / "joint_kg.json").read_text())
    assert saved["complete"]
    p = saved["protocol"]
    budget = p["cost_budgets"][1]
    cell = next(r for r in saved["rows"] if r["law_seed"] == p["target_law_seeds"][0]
                and r["cost_budget"] == budget and r["arm"] == "learned_joint_kg")
    protocol = {**p, "verification_replicates": p["arms"][cell["arm"]]["verification_replicates"]}
    prepared = prepare()
    f, joint = models(prepared, "risk")
    world = InventoryCostSimulator(prepared["plan"], sample_cost_law(cell["law_seed"]))
    truth = np.array([world.oracle_moments(x) for x in prepared["policies"]])
    truth[:, 1] += prepared["plan"].cost_budget - budget
    true_reward, true_power = verification_utility(truth, np.zeros_like(truth), protocol)
    rows = []
    for step, record in enumerate(cell["search_path"], 1):
        update(prepared, f, joint, record["index"], np.asarray(record["source_budget_response"]), record["noise"], p["replicates"])
        if step not in (8, 12, 24):
            continue
        state = moments(prepared, f, joint, budget)
        choice, _, _ = recommend(state, protocol)
        mean_g, variance_g = joint.conditional_moments(prepared["Z"])
        mean_g += prepared["plan"].cost_budget - budget
        weights = np.exp(joint.log_weights)
        quality = objective_quality(state[0][:, None], np.diag(state[1])[:, None])
        current_power = particle_pass_probability(mean_g, variance_g, prepared["risk_variances"], weights, protocol)
        # Oracle substitutions are attribution only, not posterior conditioning.
        true_g_power = particle_pass_probability(np.broadcast_to(truth[:, 1, None], mean_g.shape), np.zeros_like(variance_g),
                                                  prepared["risk_variances"], weights, protocol)
        true_risk_power = particle_pass_probability(mean_g, variance_g,
            np.broadcast_to(np.exp(truth[:, 2, None]), mean_g.shape), weights, protocol)
        counterfactuals = {"posterior": quality * current_power,
            "reveal_objective": np.clip((5-truth[:, 0])/10, 0, 1) * current_power,
            "reveal_constraint_mean": quality * true_g_power, "reveal_risk": quality * true_risk_power}
        details = []
        for index in sorted({cell["initial_choice"], cell["choice"], int(np.argmax(true_reward))}):
            details.append({"index": index, "predicted_f_mean": float(state[0][index]),
                "predicted_g_eta_mean": state[2][index].tolist(), "true_moments": truth[index].tolist(),
                "predicted_power": float(current_power[index]), "true_power": float(true_power[index]),
                "power_if_constraint_mean_revealed": float(true_g_power[index]),
                "power_if_risk_revealed": float(true_risk_power[index]),
                "true_reward": float(true_reward[index])})
        remaining = p["total_target_calls_per_arm"] - step * p["replicates"]
        now_protocol = {**p, "verification_replicates": remaining}
        next_protocol = {**p, "verification_replicates": remaining - p["replicates"]}
        now = float(recommend(state, now_protocol)[1].max())
        after_cost = float(recommend(state, next_protocol)[1].max())
        scores = joint_verification_kg(*state, measurement_noise(prepared, joint, p["replicates"]), next_protocol,
                                      order=p["quadrature_order"])
        rows.append({"step": step, "choice": choice, "candidate_details": details,
            "counterfactual_choices": {name: int(np.argmax(values)) for name, values in counterfactuals.items()},
            "one_step_budget_aware_predicted_gain": float(scores.max() + after_cost - now),
            "one_step_budget_aware_predicted_action": int(np.argmax(scores)),
            "five_node_budget_aware_predicted_gain": float(budgeted_joint_kg(*state,
                measurement_noise(prepared, joint, p["replicates"]), p, remaining, order=5).max())})
        if step == 12:
            values = {str(order): joint_verification_kg(*state,
                measurement_noise(prepared, joint, p["replicates"]), protocol, order=order) for order in (3, 5, 7)}
            rows[-1]["quadrature_zero_check"] = {order: {"maximum_kg": float(value.max()),
                "positive_actions": int(np.sum(value > 0)), "action": int(np.argmax(value))} for order, value in values.items()}
    result = {"purpose": "Post-pilot attribution of the first law's adjusted-budget loss; oracle values never refit a model or determine a new experiment",
        "law_seed": cell["law_seed"], "cost_budget": budget, "rows": rows, "new_episode_calls": 0,
        "local_cpu_seconds": time.process_time()-start}
    (folder / "joint_kg_attribution.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))

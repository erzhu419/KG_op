"""Independent source/target eight-point confirmation for the mean prior.

The protocol is frozen in the companion JSON.  Four new source laws are used
to fit the billing and full-dictionary priors; four new target laws receive the
same eight public initial policies.  The risk context is held to the declared
source-learned or pooled physical prior so this remains a mean-structure
confirmation, rather than an unplanned full-method search.
"""

import json
from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.cumulative_risk import cumulative_feature_vector
from core.joint_verification_kg import joint_verification_utility
from core.verification_kg import posterior_covariances, verification_utility
from problems.inventory_cost_plan import InventoryCostSimulator, InventoryPlan, sample_cost_law
from representation.inventory_mean_prior import InventoryFunctionBasis
from representation.joint_mean_risk_prior import JointMeanRiskPosterior
from representation.physical_risk_prior import PhysicalRiskPrior, RiskBatchData, fit_source_risk_families
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior, head_batches


PROTOCOL_PATH = ROOT / "performance/inventory_independent_source_confirmation_protocol.json"
BASE_RECORDS = ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/check.json"
DESIGN_PATH = ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/public_source_design.json"


def prepare(p):
    base = json.loads(BASE_RECORDS.read_text())
    design = json.loads(DESIGN_PATH.read_text())
    policies = np.asarray(base["public_policies"], dtype=float)
    plans = [InventoryPlan(**context) for context in design["contexts"]]
    assignments = design["assignments"]
    source_records = []
    source_calls = 0
    for law_seed, noise_seed in zip(p["source_law_seeds"], p["source_noise_seeds"]):
        response, noise, source_psi = [], [], []
        rng = np.random.default_rng(noise_seed)
        for assignment in assignments:
            plan = plans[assignment["context"]]
            world = InventoryCostSimulator(plan, sample_cost_law(law_seed))
            y, v = world.observe(policies[assignment["policy_index"]], p["replicates"], rng)
            response.append(y)
            noise.append(v)
            source_psi.append(plan.trajectory(policies[assignment["policy_index"]]).coordinates(plan.capacity))
            source_calls += p["replicates"]
        source_records.append({"psi": np.asarray(source_psi), "response": np.asarray(response),
                               "noise": np.asarray(noise)})

    target_indices = np.asarray(base["target_policy_indices"], dtype=int)
    target_records = []
    target_calls = 0
    for law_seed, noise_seed in zip(p["target_law_seeds"], p["target_noise_seeds"]):
        world = InventoryCostSimulator(plans[0], sample_cost_law(law_seed))
        rng = np.random.default_rng(noise_seed)
        response, noise = [], []
        for index in target_indices:
            y, v = world.observe(policies[index], p["replicates"], rng)
            response.append(y)
            noise.append(v)
            target_calls += p["replicates"]
        target_records.append({"law_seed": law_seed, "noise_seed": noise_seed,
                              "initial_policy_indices": target_indices.tolist(),
                              "response": np.asarray(response), "noise": np.asarray(noise)})

    source_psi = source_records[0]["psi"]
    target_psi = plans[0].coordinates_many(policies)
    X = np.array([cumulative_feature_vector(plans[0].trajectory(x).risk_exposures()) for x in policies])
    Xs = np.asarray(design["risk_design"])
    batches = [head_batches(source_records, h) for h in range(2)]
    noises = [[task["noise"][:, h] for task in source_records] for h in range(2)]
    billing_means = []
    full_means = []
    target_only_means = []
    billing_basis = InventoryFunctionBasis(source_psi, InventoryFunctionBasis.billing_indices)
    for h in range(2):
        billing_means.append(replace(SourceFunctionPrior.from_basis(
            batches[h], noises[h], billing_basis, "source_billing"), discrepancy_variance=0.0))
        full_means.append(SourceFunctionPrior.fit(batches[h], noises[h], source_psi, "fixed_source"))
        target_only_means.append(SourceFunctionPrior.fit([], [], source_psi, "target_only"))
    risk_tasks = [RiskBatchData.from_log_labels(Xs, task["response"][:, 2], p["replicates"])
                  for task in source_records]
    learned_risk = fit_source_risk_families(
        risk_tasks, np.vstack([Xs, X]), **p["source_risk_distribution"])
    pooled_risk = PhysicalRiskPrior(
        learned_risk.coefficients,
        np.full(len(learned_risk.weights), 1.0 / len(learned_risk.weights)),
        learned_risk.families, learned_risk.family_names,
        {**learned_risk.diagnostics, "mode": "same_source_pooled_scale"},
    )
    return {
        "policies": policies, "plans": plans, "source_records": source_records,
        "target_records": target_records, "target_psi": target_psi, "X": X,
        "means": {
            "source_billing": billing_means,
            "same_source_full_dictionary": full_means,
            "target_only_full_dictionary": target_only_means,
        },
        "risks": {
            "source_learned_factor_families": learned_risk,
            "same_source_pooled_scale": pooled_risk,
        },
        "source_calls": source_calls, "target_calls": target_calls,
    }


def run(p):
    started = time.process_time()
    prepared = prepare(p)
    candidates = np.arange(len(prepared["policies"]))[:, None]
    protocol = json.loads((ROOT / "performance/inventory_meta_prior_closed_loop_protocol.json").read_text())
    protocol = {**protocol, "verification_replicates": p["verification_replicates"]}
    rows = []
    for target in prepared["target_records"]:
        world = InventoryCostSimulator(prepared["plans"][0], sample_cost_law(target["law_seed"]))
        truth = np.array([world.oracle_moments(x) for x in prepared["policies"]])
        indices = np.asarray(target["initial_policy_indices"], dtype=int)
        for mean_name, mean_priors in prepared["means"].items():
            for risk_name, risk_prior in prepared["risks"].items():
                f_model = mean_priors[0].target_model(prepared["target_psi"])
                g_prior = mean_priors[1]
                joint = JointMeanRiskPosterior(g_prior.mean, g_prior.covariance, risk_prior)
                Z = np.column_stack([np.ones(len(prepared["target_psi"])),
                                      g_prior.basis.transform(prepared["target_psi"])])
                for index, response, noise in zip(indices, target["response"], target["noise"]):
                    f_model.update([int(index)], response[0], noise[0])
                    joint.update_batch(
                        Z[int(index):int(index) + 1], response[1:2],
                        RiskBatchData.from_log_labels(prepared["X"][int(index):int(index) + 1],
                                                      response[2:3], p["replicates"]),
                    )
                f_mean = f_model.posterior_mean_many(candidates)
                f_cov = posterior_covariances([f_model], candidates)[..., 0]
                pair_mean, pair_cov = joint.predict(Z, prepared["X"], full_cov=True)
                diagonal = pair_cov[np.arange(len(candidates)), :, np.arange(len(candidates)), :]
                errors = np.column_stack([
                    f_mean - truth[:, 0], pair_mean[:, 0] - truth[:, 1], pair_mean[:, 1] - truth[:, 2]
                ])
                for budget in p["evaluation_budgets"]:
                    shifted_mean = pair_mean.copy()
                    shifted_truth = truth.copy()
                    shifted_mean[:, 0] += prepared["plans"][0].cost_budget - budget
                    shifted_truth[:, 1] += prepared["plans"][0].cost_budget - budget
                    utility, power = joint_verification_utility(
                        f_mean, np.diag(f_cov), shifted_mean, diagonal, protocol
                    )
                    true_utility, true_power = verification_utility(
                        shifted_truth, np.zeros_like(shifted_truth), protocol
                    )
                    choice = int(np.argmax(utility))
                    rows.append({
                        "law_seed": int(target["law_seed"]), "evaluation_budget": float(budget),
                        "mean_arm": mean_name, "risk_context": risk_name,
                        "choice": choice, "predicted_utility": float(utility[choice]),
                        "predicted_verifier_power": float(power[choice]),
                        "true_expected_verified_reward": float(true_utility[choice]),
                        "best_true_expected_verified_reward": float(true_utility.max()),
                        "true_shortfall": float(true_utility.max() - true_utility[choice]),
                        "true_verifier_power": float(true_power[choice]),
                        "rmse_f": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
                        "rmse_g": float(np.sqrt(np.mean(errors[:, 1] ** 2))),
                        "rmse_eta": float(np.sqrt(np.mean(errors[:, 2] ** 2))),
                    })
    return {
        "protocol": p, "rows": rows, "source_records": [
            {"response": r["response"].tolist(), "noise": r["noise"].tolist(), "psi": r["psi"].tolist()}
            for r in prepared["source_records"]],
        "target_records": [{**r, "response": r["response"].tolist(), "noise": r["noise"].tolist()}
                           for r in prepared["target_records"]],
        "source_episode_calls": prepared["source_calls"], "target_episode_calls": prepared["target_calls"],
        "new_source_episode_calls": prepared["source_calls"], "new_target_episode_calls": prepared["target_calls"],
        "target_records_used": True, "oracle_used_only_after_recommendation": True,
        "source_risk_diagnostics": {name: prior.diagnostics for name, prior in prepared["risks"].items()},
        "local_cpu_seconds": time.process_time() - started,
    }


if __name__ == "__main__":
    p = json.loads(PROTOCOL_PATH.read_text())
    output = ROOT / "paper_artifacts/inventory_physical_risk_20261004/independent_source_confirmation.json"
    result = run(p)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"rows": len(result["rows"]), "source_episode_calls": result["source_episode_calls"],
                      "target_episode_calls": result["target_episode_calls"],
                      "local_cpu_seconds": result["local_cpu_seconds"]}, indent=2))

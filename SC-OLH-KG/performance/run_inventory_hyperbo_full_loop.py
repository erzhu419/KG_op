"""One frozen development closed loop: source billing versus official HyperBO."""

import copy
import json
from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import polygamma
from scipy.stats import chi2, norm, t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.transfer_external_models import OfficialHyperBOSurrogate
from baselines.transfer_models import ScalarTaskData
from core.cumulative_risk import cumulative_feature_vector
from core.joint_verification_kg import joint_verification_kg, joint_verification_utility
from core.verification_kg import posterior_covariances, verification_kg_from_moments, verification_utility
from performance.check_inventory_mean_interface import source_data
from problems.inventory_cost_plan import InventoryCostSimulator, sample_cost_law
from representation.inventory_mean_prior import InventoryFunctionBasis
from representation.joint_mean_risk_prior import JointMeanRiskPosterior
from representation.physical_risk_prior import PhysicalRiskPrior, RiskBatchData, fit_source_risk_families
from representation.source_function_prior import SourceFunctionPrior, head_batches


ROOT_PROTOCOL = ROOT / "performance/inventory_hyperbo_full_loop_protocol.json"
INDEPENDENT = ROOT / "paper_artifacts/inventory_physical_risk_20261004/independent_source_confirmation.json"


def prepare(p):
    archive = json.loads(INDEPENDENT.read_text())
    base_source = json.loads((ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/check.json").read_text())
    design = json.loads((ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/public_source_design.json").read_text())
    source, _, policies, plans, _ = source_data("paper_artifacts/inventory_physical_risk_20261004/check.json")
    policies = np.asarray(base_source["public_policies"], dtype=float)
    source_records = [{"psi": np.asarray(r["psi"], dtype=float),
                       "response": np.asarray(r["response"], dtype=float),
                       "noise": np.asarray(r["noise"], dtype=float)}
                      for r in archive["source_records"]]
    target_psi = plans[0].coordinates_many(policies)
    X = np.array([cumulative_feature_vector(plans[0].trajectory(x).risk_exposures()) for x in policies])
    Xs = np.asarray(design["risk_design"])
    billing = []
    billing_basis = InventoryFunctionBasis(source_records[0]["psi"], InventoryFunctionBasis.billing_indices)
    for head in range(2):
        batches = head_batches(source_records, head)
        noises = [r["noise"][:, head] for r in source_records]
        billing.append(replace(SourceFunctionPrior.from_basis(
            batches, noises, billing_basis, "source_billing"), discrepancy_variance=0.0))
    risk_tasks = [RiskBatchData.from_log_labels(Xs, r["response"][:, 2], p["replicates"])
                  for r in source_records]
    learned = fit_source_risk_families(risk_tasks, np.vstack([Xs, X]), **source["protocol"]["source_risk_distribution"])
    pooled = PhysicalRiskPrior(learned.coefficients,
        np.full(len(learned.weights), 1.0 / len(learned.weights)), learned.families,
        learned.family_names, {**learned.diagnostics, "mode": "same_source_pooled_scale"})
    hyperbo = []
    for head in range(3):
        tasks = [ScalarTaskData(f"source_{i}", r["psi"], r["response"][:, head], r["noise"][:, head])
                 for i, r in enumerate(source_records)]
        model = OfficialHyperBOSurrogate(seed=100 + head, source_steps=p["hyperbo_source_training_steps"])
        model.meta_fit(tasks)
        hyperbo.append(model)
    return {"policies": policies, "plans": plans, "source_records": source_records,
            "target_psi": target_psi, "X": X, "risk": learned, "pooled": pooled,
            "billing": billing, "hyperbo": hyperbo,
            "initial": np.asarray(base_source["target_policy_indices"], dtype=int)}


def source_state(prepared, f_model, joint, p):
    candidates = np.arange(len(prepared["policies"]))[:, None]
    f_mean = f_model.posterior_mean_many(candidates)
    f_cov = posterior_covariances([f_model], candidates)[..., 0]
    features = np.column_stack([
        np.ones(len(prepared["target_psi"])),
        prepared["billing"][1].basis.transform(prepared["target_psi"]),
    ])
    pair_mean, pair_cov = joint.predict(features, prepared["X"], full_cov=True)
    pair_mean[:, 0] += prepared["plans"][0].cost_budget - p["cost_budget"]
    return f_mean, f_cov, pair_mean, pair_cov


def source_scores(prepared, f_model, joint, p):
    f_mean, f_cov, pair_mean, pair_cov = source_state(prepared, f_model, joint, p)
    point_cov = pair_cov[np.arange(len(f_mean)), :, np.arange(len(f_mean)), :]
    protocol = {**p, "verification_replicates": p["verification_replicates"]}
    utility, _ = joint_verification_utility(f_mean, np.diag(f_cov), pair_mean, point_cov, protocol)
    noise = np.column_stack([
        np.full(len(f_mean), .12 ** 2 / p["replicates"]),
        (prepared["X"] @ joint.risk_prior.coefficients.T @ np.exp(joint.log_weights)) / p["replicates"],
        np.full(len(f_mean), polygamma(1, (p["replicates"] - 1) / 2)),
    ])
    scores = joint_verification_kg(f_mean, f_cov, pair_mean, pair_cov, noise,
                                   protocol, order=p["quadrature_order"])
    return int(np.argmax(scores)), (f_mean, f_cov, pair_mean, pair_cov), utility


def hyperbo_state(prepared, models, p, obs_X, obs_y, obs_noise):
    for head, model in enumerate(models):
        model.adapt(obs_X, obs_y[:, head], obs_noise[:, head])
    predictions = [model.predict(prepared["target_psi"], full_cov=True) for model in models]
    mean = np.column_stack([pred[0] for pred in predictions])
    covariance = np.stack([pred[1] for pred in predictions], axis=-1)
    mean[:, 1] += prepared["plans"][0].cost_budget - p["cost_budget"]
    return mean, covariance


def hyperbo_scores(prepared, models, p, obs_X, obs_y, obs_noise):
    mean, covariance = hyperbo_state(prepared, models, p, obs_X, obs_y, obs_noise)
    variance = np.maximum(np.einsum("iih->ih", covariance), 0.0)
    noise = np.column_stack([
        np.full(len(mean), .12 ** 2 / p["replicates"]),
        np.exp(mean[:, 2] + .5 * variance[:, 2]) / p["replicates"],
        np.full(len(mean), polygamma(1, (p["replicates"] - 1) / 2)),
    ])
    protocol = {**p, "verification_replicates": p["verification_replicates"]}
    scores = verification_kg_from_moments(mean, covariance, noise, protocol)
    utility, _ = verification_utility(mean, variance, protocol)
    return int(np.argmax(scores)), (mean, covariance), utility


def run(p):
    started = time.process_time()
    prepared = prepare(p)
    world = InventoryCostSimulator(prepared["plans"][0], sample_cost_law(p["target_law_seed"]))
    observation_cache, verification_cache = {}, {}
    rows = []
    for arm in p["arms"]:
        if arm == "source_billing_joint":
            f_model = prepared["billing"][0].target_model(prepared["target_psi"])
            g_prior = prepared["billing"][1]
            joint = JointMeanRiskPosterior(g_prior.mean, g_prior.covariance, prepared["risk"])
            Z = np.column_stack([np.ones(len(prepared["target_psi"])), g_prior.basis.transform(prepared["target_psi"])])
        else:
            models = copy.deepcopy(prepared["hyperbo"])
            obs_X, obs_y, obs_noise = [], [], []
        visits = np.zeros(len(prepared["policies"]), dtype=int)
        path = []
        initial_choice = None
        for step in range(p["target_points"]):
            if step < p["initial_points"]:
                index = int(prepared["initial"][step])
            elif arm == "source_billing_joint":
                index, state, utility = source_scores(prepared, f_model, joint, p)
            else:
                index, state, utility = hyperbo_scores(
                    prepared, models, p,
                    np.asarray(obs_X), np.asarray(obs_y), np.asarray(obs_noise))
            key = (index, int(visits[index]))
            visits[index] += 1
            if key not in observation_cache:
                rng = np.random.default_rng(np.random.SeedSequence([p["target_noise_seed"], 0, *key]))
                observation_cache[key] = world.observe(prepared["policies"][index], p["replicates"], rng)
            response, noise = observation_cache[key]
            if arm == "source_billing_joint":
                f_model.update([index], response[0], noise[0])
                joint.update_batch(Z[index:index + 1], response[1:2],
                    RiskBatchData.from_log_labels(prepared["X"][index:index + 1], response[2:3], p["replicates"]))
            else:
                obs_X.append(prepared["target_psi"][index])
                obs_y.append(response)
                obs_noise.append(noise)
            path.append({"index": index, "visit": key[1], "response": response.tolist(), "noise": noise.tolist()})
            if step + 1 == p["initial_points"]:
                if arm == "source_billing_joint":
                    _, _, utility = source_scores(prepared, f_model, joint, p)
                else:
                    _, _, utility = hyperbo_scores(prepared, models, p,
                        np.asarray(obs_X), np.asarray(obs_y), np.asarray(obs_noise))
                initial_choice = int(np.argmax(utility))
        if arm == "source_billing_joint":
            _, state, utility = source_scores(prepared, f_model, joint, p)
            choice = int(np.argmax(utility))
            predicted_power = float(joint_verification_utility(
                state[0], np.diag(state[1]), state[2],
                state[3][np.arange(len(state[0])), :, np.arange(len(state[0])), :],
                {**p, "verification_replicates": p["verification_replicates"]})[1][choice])
        else:
            _, state, utility = hyperbo_scores(prepared, models, p,
                np.asarray(obs_X), np.asarray(obs_y), np.asarray(obs_noise))
            choice = int(np.argmax(utility))
            predicted_power = float(verification_utility(
                state[0], np.maximum(np.einsum("iih->ih", state[1]), 0.0),
                {**p, "verification_replicates": p["verification_replicates"]})[1][choice])
        n = p["total_target_calls_per_arm"] - len(path) * p["replicates"]
        if (choice, n) not in verification_cache:
            rng = np.random.default_rng(np.random.SeedSequence([p["target_noise_seed"], 1, choice, n]))
            verification_cache[choice, n] = world.episodes(prepared["policies"][choice], n, rng)[:, 1]
        validation = verification_cache[choice, n]
        df = n - 1
        upper = float(validation.mean() + validation.std(ddof=1) * (
            t.ppf(.975, df) / np.sqrt(n) + norm.ppf(1 - p["chance_alpha"]) * np.sqrt(df / chi2.ppf(.025, df))))
        rows.append({"arm": arm, "initial_choice": initial_choice, "choice": choice,
                     "predicted_verifier_power": predicted_power, "validation_upper_margin": upper,
                     "verified": bool(upper <= 0), "verification_replicates": n,
                     "search_path": path})
    truth = np.array([world.oracle_moments(x) for x in prepared["policies"]])
    reward, power = verification_utility(truth, np.zeros_like(truth), p)
    margin = truth[:, 1] + norm.ppf(1 - p["chance_alpha"]) * np.exp(.5 * truth[:, 2])
    for row in rows:
        row.update(true_expected_verified_reward=float(reward[row["choice"]]),
                   initial_true_expected_reward=float(reward[row["initial_choice"]]),
                   best_available_reward=float(reward.max()),
                   true_verifier_power=float(power[row["choice"]]),
                   true_final_margin=float(margin[row["choice"]]),
                   true_feasible_candidates=int(np.sum(margin <= 0)),
                   false_certificate=bool(row["verified"] and margin[row["choice"]] > 0),
                   realized_verified_reward=float(row["verified"] * np.clip((5 - truth[row["choice"], 0]) / 10, 0, 1)),
                   target_calls=p["total_target_calls_per_arm"])
    return {"protocol": p, "rows": rows, "target_law_seed": p["target_law_seed"],
            "new_source_episode_calls": 0, "new_target_episode_calls": world.episode_calls,
            "allocated_target_calls": sum(row["target_calls"] for row in rows),
            "actually_simulated_episode_calls_with_paired_reuse": world.episode_calls,
            "local_cpu_seconds": time.process_time() - started}


if __name__ == "__main__":
    p = json.loads(ROOT_PROTOCOL.read_text())
    output = ROOT / "paper_artifacts/inventory_physical_risk_20261004/hyperbo_full_loop_95025.json"
    result = run(p)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"rows": len(result["rows"]), "summary": [
        {k: row[k] for k in ("arm", "initial_choice", "choice", "verified", "true_expected_verified_reward", "true_feasible_candidates")}
        for row in result["rows"]], "local_cpu_seconds": result["local_cpu_seconds"]}, indent=2))

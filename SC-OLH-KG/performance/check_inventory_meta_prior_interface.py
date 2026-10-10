"""Frozen local integration check, not an optimization-performance experiment."""

import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import polygamma

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.transfer_models import HyperBOSurrogate, ScalarTaskData
from core.cumulative_risk import cumulative_feature_vector
from core.verification_kg import verification_kg_scores, verification_kg_surrogates
from performance.run_meta_prior_mechanism import moments
from problems.inventory_cost_plan import InventoryCostSimulator, InventoryPlan, sample_cost_law
from representation.conditional_function_prior import conditional_selection
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior, head_batches
from representation.source_risk_ard import fit_source_risk_ard
from representation.transfer_mixture import TransferMixtureGPR


def run(protocol):
    started = time.process_time()
    p = protocol
    plan = InventoryPlan(**p["context"])
    rng = np.random.default_rng(p["candidate_seed"])
    policies = rng.uniform(size=(p["candidates"], 4))
    psi = plan.coordinates_many(policies)
    source_indices = rng.choice(len(policies), p["source_points"], replace=False)
    initial = np.random.default_rng(p["initial_design_seed"]).choice(
        len(policies), p["initial_points"], replace=False)
    source, source_calls = [], 0
    for t in range(p["source_tasks"]):
        world = InventoryCostSimulator(plan, sample_cost_law(p["source_law_seed"] + t))
        noise_rng = np.random.default_rng(p["source_noise_seed"] + t)
        labels = [world.observe(policies[i], p["replicates"], noise_rng) for i in source_indices]
        source.append({"psi": psi[source_indices], "response": np.array([y for y, _ in labels]),
                       "noise": np.array([v for _, v in labels])})
        source_calls += world.episode_calls
    batches = [head_batches(source, h) for h in range(3)]
    noises = [[task["noise"][:, h] for task in source] for h in range(3)]
    full = [SourceFunctionPrior.fit(batches[h], noises[h], psi, "fixed_source") for h in range(3)]
    structural = []
    for h in range(2):
        selected, history = conditional_selection(batches[h], noises[h], FixedFunctionBasis(psi),
                                                   p["learned_active_dim_per_head"])
        prior = SourceFunctionPrior.from_basis(batches[h], noises[h], FixedFunctionBasis(psi, selected),
                                               "conditional_hard")
        prior.diagnostics.update(selected_indices=selected, selection_cv=history)
        structural.append(prior)
    structural.append(fit_source_risk_ard(batches[2], noises[2], psi))
    worlds = InventoryCostSimulator(plan, sample_cost_law(p["target_law_seed"]))
    target_rng = np.random.default_rng(p["target_noise_seed"])
    observed = [worlds.observe(policies[i], p["replicates"], target_rng) for i in initial]
    y, noise = np.array([a for a, _ in observed]), np.array([v for _, v in observed])
    candidates = np.arange(len(psi))[:, None]
    results = {}
    for name, priors in (("M1", structural), ("same_source_full_dictionary", full)):
        models = ([TransferMixtureGPR([priors[h], full[h]], psi, psi) for h in range(3)]
                  if name == "M1" else [prior.target_model(psi) for prior in priors])
        for index, label, variance in zip(initial, y, noise):
            for h, model in enumerate(models):
                model.update([int(index)], label[h], variance[h])
        scores = verification_kg_scores(models, candidates, p)
        m, v = moments(models, candidates)
        results[name] = {"finite_kg_scores": bool(np.isfinite(scores).all()),
                         "kg_min": float(scores.min()), "kg_max": float(scores.max()),
                         "finite_posterior_moments": bool(np.isfinite(m).all() and np.isfinite(v).all()),
                         "paid_labels_per_head": len(initial)}
    kernel = []
    for h in range(3):
        # This grid-based reimplementation tests the kernel interface. It is
        # explicitly not an executed official HyperBO comparator.
        model = HyperBOSurrogate().meta_fit([
            ScalarTaskData(str(t), task["psi"], task["response"][:, h], task["noise"][:, h])
            for t, task in enumerate(source)])
        model.adapt(psi[initial], y[:, h], noise[:, h])
        kernel.append(model)
    mean = np.column_stack([model.predict(psi)[0] for model in kernel])
    variance = np.column_stack([model.predict(psi)[1] for model in kernel])
    fantasy_noise = np.column_stack([np.full(len(psi), .12 ** 2 / p["replicates"]),
        np.exp(mean[:, 2] + .5 * variance[:, 2]) / p["replicates"],
        np.full(len(psi), polygamma(1, (p["replicates"] - 1) / 2))])
    scores = verification_kg_surrogates(kernel, psi, fantasy_noise, p)
    risk_design = np.array([cumulative_feature_vector(plan.trajectory(x).risk_exposures()) for x in policies])
    singular = np.linalg.svd(risk_design, compute_uv=False)
    missing = [name for name in ("jax", "flax", "optax") if importlib.util.find_spec(name) is None]
    assert source_calls == p["source_tasks"] * p["source_points"] * p["replicates"]
    assert worlds.episode_calls == p["initial_points"] * p["replicates"]
    return {"protocol": p, "source_episode_calls": source_calls,
        "target_episode_calls": worlds.episode_calls, "verification_calls": 0,
        "public_trajectory_evaluations": 2 * len(policies) + p["source_tasks"] * p["source_points"] + len(initial),
        "public_trajectory_evaluation_breakdown": {
            "candidate_coordinates": len(policies),
            "paid_batch_preparation": p["source_tasks"] * p["source_points"] + len(initial),
            "risk_design_diagnostic": len(policies)},
        "public_coordinate_dimension": psi.shape[1], "risk_design_rank": int(np.linalg.matrix_rank(risk_design)),
        "risk_design_columns": risk_design.shape[1], "risk_design_singular_values": singular.tolist(),
        "source_risk_optimizer_converged": structural[2].diagnostics["optimizer_converged"],
        "source_selected_mean_functions": [prior.basis.names for prior in structural[:2]],
        "source_nonintercept_means_zero": all(not np.any(prior.mean[1:]) for prior in structural),
        "arms": results, "kernel_interface_control": {
            "implementation": "paper_core_grid_reimplementation_not_official_HyperBO",
            "finite_kg_scores": bool(np.isfinite(scores).all()),
            "covariance_noise": "latent_covariance_plus_explicit_batch_noise_once"},
        "official_hyperbo": {"executed": False, "missing_local_runtime_modules": missing},
        "source_records": [{"response": t["response"].tolist(), "noise": t["noise"].tolist()} for t in source],
        "source_policy_indices": source_indices.tolist(), "target_policy_indices": initial.tolist(),
        "target_response": y.tolist(), "target_noise": noise.tolist(), "public_policies": policies.tolist(),
        "decision_performance_established": False, "local_cpu_seconds": time.process_time() - started}


if __name__ == "__main__":
    p = json.loads((ROOT / "performance/inventory_meta_prior_interface_protocol.json").read_text())
    result = run(p)
    output = ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {
        "protocol", "source_records", "target_response", "target_noise", "public_policies",
        "source_policy_indices", "target_policy_indices"}}, allow_nan=False))

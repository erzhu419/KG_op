"""Compare function priors on identical, already paid inventory response paths.

Only public dynamics and saved labels/noise are used. Neither simulator laws,
target truth, fresh episodes nor the unopened confirmation cohort are read.
Recommendations and KG probes are diagnostic: the recorded policy path was
chosen by an older method and is not executed again under either new prior.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.verification_kg import verification_kg_scores, verification_utility
from problems.inventory_cost_plan import InventoryPlan
from representation.orthogonal_function_prior import (
    OrthogonalLowOrderBasis, fit_orthogonal_function_prior,
    same_source_orthogonal_full_prior,
)
from representation.transfer_mixture import TransferMixtureGPR
from representation.transferable_spectral import SourceDomainBatch


def prepare_replay(protocol):
    """Fit the same source-only U1 priors for both recorded-path comparisons."""
    archive = json.loads((ROOT / protocol["source_archive"]).read_text())
    recorded = json.loads((ROOT / protocol["target_response_archive"]).read_text())
    plans = [InventoryPlan(**context) for context in archive["contexts"]]
    policies = np.asarray(archive["public_policies"])
    source_psi = np.array([plans[item["context"]].trajectory(policies[item["policy_index"]])
        .coordinates(plans[item["context"]].capacity) for item in archive["assignments"]])
    target_psi = plans[0].coordinates_many(policies)
    reference = np.vstack([source_psi, target_psi])
    basis = OrthogonalLowOrderBasis(reference)
    priors = []
    for head in range(3):
        batches = [SourceDomainBatch(str(task), source_psi, np.asarray(row["response"])[:, head])
                   for task, row in enumerate(archive["source_records"])]
        noise = [np.asarray(row["noise"])[:, head] for row in archive["source_records"]]
        priors.append(fit_orthogonal_function_prior(batches, noise, basis))
    full = [same_source_orthogonal_full_prior(prior) for prior in priors]
    return archive, recorded, target_psi, reference, basis, priors, full


def run():
    started = time.process_time()
    protocol = json.loads((ROOT / "performance/orthogonal_meta_prior_replay_v1.json").read_text())
    archive, recorded, target_psi, reference, basis, priors, full = prepare_replay(protocol)
    candidates = np.arange(len(target_psi))[:, None]
    rows = []
    for old in recorded["rows"]:
        if old["arm"] != protocol["recorded_path_arm"]:
            continue
        for arm in protocol["arms"]:
            models = ([TransferMixtureGPR([prior, control], target_psi, reference)
                       for prior, control in zip(priors, full)]
                      if arm == "source_group_spectral_mixture"
                      else [prior.target_model(target_psi) for prior in full])
            errors, densities, stages = [], [], []
            for step, observation in enumerate(old["search_path"]):
                point = [int(observation["index"])]
                response, noise = np.asarray(observation["response"]), np.asarray(observation["noise"])
                if step >= protocol["initial_points"]:
                    before = np.array([model.posterior_mean_many([point])[0] for model in models])
                    # Evaluate the actual two-component predictive density,
                    # not the Gaussian moment used by acquisition.
                    log_density = []
                    for h, model in enumerate(models):
                        if isinstance(model, TransferMixtureGPR):
                            log_density.append(float(logsumexp(model.log_weights + np.array([
                                norm.logpdf(response[h], component.posterior_mean(point),
                                    np.sqrt(component.posterior_var(point) + noise[h]))
                                for component in model.models]))))
                        else:
                            log_density.append(float(norm.logpdf(response[h], before[h],
                                np.sqrt(model.posterior_var(point) + noise[h]))))
                    errors.append((before - response) ** 2)
                    densities.append(log_density)
                for h, model in enumerate(models):
                    model.update(point, float(response[h]), float(noise[h]))
                if step + 1 in protocol["stages"]:
                    mean = np.column_stack([m.posterior_mean_many(candidates) for m in models])
                    variance = np.column_stack([m.posterior_var_many(candidates) for m in models])
                    utility, power = verification_utility(mean, variance, protocol)
                    scores = verification_kg_scores(models, candidates, protocol)
                    choice = int(np.argmax(utility))
                    stages.append({"paid_points": step + 1, "recommendation": choice,
                        "predicted_utility": float(utility[choice]),
                        "predicted_verifier_power": float(power[choice]),
                        "next_kg_probe": int(np.argmax(scores)), "max_kg": float(scores.max()),
                        "structural_weights": [float(np.exp(m.log_weights[0])) for m in models]
                            if arm == "source_group_spectral_mixture" else None})
            rows.append({"law_seed": old["law_seed"], "arm": arm,
                "recorded_policy_indices": [item["index"] for item in old["search_path"]],
                "label_rmse_after_initial": np.sqrt(np.mean(errors, axis=0)).tolist(),
                "mean_log_predictive_density_after_initial": np.mean(densities, axis=0).tolist(),
                "stages": stages})
    summary = []
    for arm in protocol["arms"]:
        selected = [row for row in rows if row["arm"] == arm]
        summary.append({"arm": arm, "task_blocks": len(selected),
            "mean_label_rmse_after_initial": np.mean([r["label_rmse_after_initial"]
                for r in selected], axis=0).tolist(),
            "mean_log_predictive_density_after_initial": np.mean([
                r["mean_log_predictive_density_after_initial"] for r in selected], axis=0).tolist()})
    result = {"contract": protocol, "basis": basis.diagnostics(),
        "source_priors": [prior.diagnostics for prior in priors], "rows": rows,
        "summary": summary, "head_order": ["objective_mean", "constraint_mean", "log_variance"],
        "source_calls_in_reused_archive": archive["source_episode_calls"],
        "target_calls_in_reused_paths": sum(len(row["search_path"]) * protocol["replicates"]
            for row in recorded["rows"] if row["arm"] == protocol["recorded_path_arm"]),
        "new_source_calls": 0, "new_target_calls": 0, "new_verification_calls": 0,
        "target_truth_used": False, "independent_confirmation": False,
        "local_cpu_seconds": time.process_time() - started}
    output = ROOT / protocol["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"], "local_cpu_seconds": result["local_cpu_seconds"],
        "product_gram_max_error": result["basis"]["product_gram_max_error"],
        "source_optimizer_converged": [p["optimizer_converged"] for p in result["source_priors"]]}, indent=2))

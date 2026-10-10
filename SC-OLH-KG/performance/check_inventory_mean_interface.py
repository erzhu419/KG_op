"""Source-only mean-interface comparison, reusing every paid observation."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from problems.inventory_cost_plan import InventoryPlan
from representation.conditional_function_prior import conditional_selection
from representation.inventory_mean_prior import InventoryFunctionBasis
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior, head_batches
from representation.transfer_mixture import TransferMixtureGPR


def source_data(path):
    source = json.loads((ROOT / path).read_text())
    design = json.loads((ROOT / source["protocol"]["source_design"]).read_text())
    old = json.loads((ROOT / source["protocol"]["existing_records"]).read_text())
    plans = [InventoryPlan(**p) for p in design["contexts"]]
    policies = np.asarray(old["public_policies"])
    psi = np.array([plans[r["context"]].trajectory(policies[r["policy_index"]]).coordinates(plans[r["context"]].capacity)
                    for r in design["assignments"]])
    data = [{"psi": psi, "response": np.asarray(t["response"]), "noise": np.asarray(t["noise"])}
            for t in source["source_records"]]
    return source, design, policies, plans, data


def fit_priors(data, reference, active_dim):
    output, selection = [], []
    for h in range(2):
        batches, noises = head_batches(data, h), [t["noise"][:, h] for t in data]
        priors, head_info = {}, {}
        for name, cls in (("legacy", FixedFunctionBasis), ("augmented", InventoryFunctionBasis)):
            dictionary = cls(reference)
            support, history = conditional_selection(batches, noises, dictionary, active_dim)
            priors[name + "_full"] = SourceFunctionPrior.from_basis(batches, noises, dictionary, "fixed_source")
            priors[name + "_hard"] = SourceFunctionPrior.from_basis(batches, noises, cls(reference, support), "conditional_hard")
            head_info[name] = {"indices": support, "names": priors[name + "_hard"].basis.names, "cv": history}
        priors["billing"] = replace(SourceFunctionPrior.from_basis(batches, noises,
            InventoryFunctionBasis(reference, InventoryFunctionBasis.billing_indices), "billing_control"), discrepancy_variance=0.)
        output.append(priors)
        selection.append(head_info)
    return output, selection


def run(p):
    start = time.process_time()
    _, _, _, _, data = source_data(p["source_records"])
    reference = data[0]["psi"]
    train_positions, test_positions = p["held_task_adaptation_positions"], p["held_task_test_positions"]
    rows = []
    for left_out, held in enumerate(data):
        if time.process_time() - start > p["local_cpu_limit_seconds"]:
            raise RuntimeError("Source-only check exceeded its local CPU budget")
        priors, selection = fit_priors([t for i, t in enumerate(data) if i != left_out], reference, p["selected_functions_max"])
        candidates = np.asarray(test_positions)[:, None]
        for h in range(2):
            fitted = priors[h]
            for arm in p["arms"]:
                if arm.endswith("mixture"):
                    name = arm.removesuffix("_mixture")
                    model = TransferMixtureGPR([fitted[name + "_hard"], fitted[name + "_full"]], reference, reference)
                else:
                    prior = fitted["billing"] if arm == "known_billing_control" else fitted["augmented_hard"]
                    if arm == "augmented_structural_no_discrepancy":
                        prior = replace(prior, discrepancy_variance=0.)
                    model = prior.target_model(reference)
                for index in train_positions:
                    model.update([index], held["response"][index, h], held["noise"][index, h])
                mean = model.posterior_mean_many(candidates)
                total_variance = model.posterior_var_many(candidates) + held["noise"][test_positions, h]
                error = held["response"][test_positions, h] - mean
                rows.append({"left_out_task": left_out, "head": h, "arm": arm,
                    "label_rmse": float(np.sqrt(np.mean(error ** 2))),
                    "mean_working_conditional_log_density": float(np.mean(norm.logpdf(error, scale=np.sqrt(total_variance)))),
                    "standardized_residual_rms": float(np.sqrt(np.mean(error ** 2 / total_variance))),
                    "source_selection": selection[h]})
    fitted, selection = fit_priors(data, reference, p["selected_functions_max"])
    summary = {str(h): {arm: {metric: float(np.mean([r[metric] for r in rows if r["head"] == h and r["arm"] == arm]))
        for metric in ("label_rmse", "mean_working_conditional_log_density", "standardized_residual_rms")}
        for arm in p["arms"]} for h in range(2)}
    return {"protocol": p, "rows": rows, "summary": summary, "all_source_selection": selection,
        "new_episode_calls": 0, "target_records_used": False, "local_cpu_seconds": time.process_time() - start}


if __name__ == "__main__":
    output = ROOT / "paper_artifacts/inventory_physical_risk_20261004/mean_interface.json"
    if output.exists():
        print("Existing source-only mean comparison retained.")
        sys.exit(0)
    p = json.loads((ROOT / "performance/inventory_mean_interface_protocol.json").read_text())
    result = run(p)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": result["summary"], "all_source_selection": result["all_source_selection"],
                      "local_cpu_seconds": result["local_cpu_seconds"]}))

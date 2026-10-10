"""Outer source-law holdout of the source-learned Inventory billing block."""

import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_inventory_mean_interface import source_data
from representation.inventory_mean_prior import fit_inventory_group_prior
from representation.source_function_prior import head_batches


def run(p, fit_prior=fit_inventory_group_prior):
    start = time.process_time()
    _, _, _, _, data = source_data(p["source_records"])
    reference = data[0]["psi"]
    rows, all_source = [], []
    for left_out, held in enumerate(data):
        for head in range(2):
            if time.process_time() - start > p["local_cpu_limit_seconds"]:
                raise RuntimeError("Source-only group comparison exceeded local CPU budget")
            keep = [t for i, t in enumerate(data) if i != left_out]
            prior = fit_prior(head_batches(keep, head), [t["noise"][:, head] for t in keep], reference,
                                               p["held_task_adaptation_positions"], p["held_task_test_positions"])
            model = prior.target_model(reference)
            for index in p["held_task_adaptation_positions"]:
                model.update([index], held["response"][index, head], held["noise"][index, head])
            test = p["held_task_test_positions"]
            mean = model.posterior_mean_many(np.array(test)[:, None])
            variance = model.posterior_var_many(np.array(test)[:, None]) + held["noise"][test, head]
            error = held["response"][test, head] - mean
            rows.append({"left_out_task": left_out, "head": head, "label_rmse": float(np.sqrt(np.mean(error ** 2))),
                "mean_working_conditional_log_density": float(np.mean(norm.logpdf(error, scale=np.sqrt(variance)))),
                "standardized_residual_rms": float(np.sqrt(np.mean(error ** 2 / variance))), "selection": prior.diagnostics})
    for head in range(2):
        prior = fit_prior(head_batches(data, head), [t["noise"][:, head] for t in data], reference,
                                           p["held_task_adaptation_positions"], p["held_task_test_positions"])
        all_source.append({"mean": prior.mean.tolist(), "covariance": prior.covariance.tolist(),
                           "discrepancy_variance": prior.discrepancy_variance, "diagnostics": prior.diagnostics})
    summary = {str(h): {metric: float(np.mean([r[metric] for r in rows if r["head"] == h]))
        for metric in ("label_rmse", "mean_working_conditional_log_density", "standardized_residual_rms")} for h in range(2)}
    return {"protocol": p, "rows": rows, "summary": summary, "all_source_priors": all_source,
            "new_episode_calls": 0, "target_records_used": False, "local_cpu_seconds": time.process_time() - start}


if __name__ == "__main__":
    output = ROOT / "paper_artifacts/inventory_physical_risk_20261004/mean_group.json"
    if output.exists():
        print("Existing source-only group comparison retained.")
        sys.exit(0)
    p = json.loads((ROOT / "performance/inventory_mean_group_protocol.json").read_text())
    result = run(p)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": result["summary"], "all_source_selection": [r["diagnostics"] for r in result["all_source_priors"]],
                      "local_cpu_seconds": result["local_cpu_seconds"]}))

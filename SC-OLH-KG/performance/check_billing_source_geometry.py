"""Within-context source checks using the four already-paid sites per context."""

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
from representation.inventory_mean_prior import InventoryFunctionBasis, fit_inventory_posterior_group_prior
from representation.source_function_prior import head_batches


def run():
    start = time.process_time()
    p = json.loads((ROOT / "performance/orthogonal_billing_replay_v1.json").read_text())
    archive = json.loads((ROOT / p["source_archive"]).read_text())
    plans = [InventoryPlan(**c) for c in archive["contexts"]]
    policies = np.asarray(archive["public_policies"])
    psi = np.array([plans[a["context"]].trajectory(policies[a["policy_index"]])
        .coordinates(plans[a["context"]].capacity) for a in archive["assignments"]])
    target_public = plans[0].coordinates_many(policies)
    reference = np.vstack([psi, target_public])
    data = [{"psi": psi, "response": np.asarray(r["response"]), "noise": np.asarray(r["noise"])}
            for r in archive["source_records"]]
    rows = []
    for head in range(2):
        for held_index, held in enumerate(data):
            keep = [r for i, r in enumerate(data) if i != held_index]
            prior = fit_inventory_posterior_group_prior(head_batches(keep, head),
                [r["noise"][:, head] for r in keep], reference,
                p["source_adaptation_positions"], p["source_test_positions"])
            for context in range(len(plans)):
                sites = [i for i, a in enumerate(archive["assignments"]) if a["context"] == context]
                for fold in (0, 1):
                    adapt, test = sites[fold::2], sites[1 - fold::2]
                    for arm, candidate in [("source_selected_residual", prior),
                                           ("fixed_residual_002", replace(prior, discrepancy_variance=.02))]:
                        model = candidate.target_model(psi)
                        for index in adapt:
                            model.update([index], held["response"][index, head], held["noise"][index, head])
                        mean = model.posterior_mean_many(np.asarray(test)[:, None])
                        variance = model.posterior_var_many(np.asarray(test)[:, None]) + held["noise"][test, head]
                        error = held["response"][test, head] - mean
                        rows.append({"head": head, "held_source": held_index, "context": context,
                            "fold": fold, "arm": arm, "rmse": float(np.sqrt(np.mean(error ** 2))),
                            "mean_log_density": float(np.mean(norm.logpdf(error, scale=np.sqrt(variance)))),
                            "standardized_residual_rms": float(np.sqrt(np.mean(error ** 2 / variance)))})
    summary = [{"head": head, "arm": arm, **{metric: float(np.mean([r[metric]
        for r in rows if r["head"] == head and r["arm"] == arm]))
        for metric in ("rmse", "mean_log_density", "standardized_residual_rms")}}
        for head in range(2) for arm in ("source_selected_residual", "fixed_residual_002")]
    basis = InventoryFunctionBasis(reference, InventoryFunctionBasis.billing_indices)
    # These are the registered common initial indices, not responses or optima.
    target_initial = [42, 14, 18, 26, 73, 33, 54, 76]
    geometry = {}
    for name, X in [("source_four_context_eight", basis.transform(psi)[p["source_adaptation_positions"]]),
                    ("target_one_context_eight", basis.transform(target_public)[target_initial])]:
        singular = np.linalg.svd(np.column_stack([np.ones(len(X)), X]), compute_uv=False)
        geometry[name] = {"singular_values": singular.tolist(), "condition_number": float(singular[0] / singular[-1])}
    result = {"study_role": "source-only within-context sensitivity diagnostic",
        "selection_rule": "unchanged source selector; outer held law excluded from all learned parameters",
        "adaptation_and_test": "two paid sites adapt, two other paid sites test, two folds in each of four contexts",
        "target_eight_point_match": False,
        "limitation": "only four source sites exist per context; this does not reproduce the eight-site target design",
        "target_response_labels_used": False, "target_truth_used": False, "new_episode_calls": 0,
        "rows": rows, "summary": summary, "public_geometry": geometry,
        "local_cpu_seconds": time.process_time() - start}
    (ROOT / p["output"] / "source_geometry.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    r = run()
    print(json.dumps({"summary": r["summary"], "public_geometry": r["public_geometry"],
                      "local_cpu_seconds": r["local_cpu_seconds"]}, indent=2))

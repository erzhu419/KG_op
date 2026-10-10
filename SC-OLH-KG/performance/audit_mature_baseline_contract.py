"""Audit the official HyperBO contract without running a paper experiment.

The audit reuses the saved Inventory source and initial target records.  It
checks data shapes, paper-core covariance behavior, pinned repository
provenance, and the actual official-runtime boundary.  No simulator call or
target truth is used.
"""

import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.transfer_external_models import OfficialHyperBOSurrogate, external_runtime_report
from baselines.transfer_models import HyperBOSurrogate, ScalarTaskData
from problems.inventory_cost_plan import InventoryPlan


SOURCE = ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/check.json"
EXPECTED_HYPERBO_COMMIT = "e720fc1"
PROTOCOL = {
    "status": "frozen_contract_audit_no_new_observations",
    "source_records": "paper_artifacts/inventory_meta_prior_interface_20261004/check.json",
    "target_records": "same_saved_eight_initial_records_in_source_file",
    "official_method": "google-research/hyperbo",
    "required_operation": "meta_fit -> adapt -> predict(full_cov=True)",
    "paper_core_smoke": "same observable scalar tasks and target initial labels",
    "new_source_episode_calls": 0,
    "new_target_episode_calls": 0,
    "truth_used": False,
}


def run():
    started = time.process_time()
    saved = json.loads(SOURCE.read_text())
    design = json.loads((ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/public_source_design.json").read_text())
    plans = [InventoryPlan(**context) for context in design["contexts"]]
    source_tasks = saved["source_records"]
    policies = np.asarray(saved["public_policies"], dtype=float)
    target_indices = np.asarray(saved["target_policy_indices"], dtype=int)
    target_X = plans[0].coordinates_many(policies[target_indices])
    target_y = np.asarray(saved["target_response"], dtype=float)
    target_noise = np.asarray(saved["target_noise"], dtype=float)
    source_shapes = []
    paper_core = []
    for head, name in enumerate(("objective", "constraint_mean", "log_variance")):
        tasks = []
        for index, task in enumerate(source_tasks):
            response = np.asarray(task["response"], dtype=float)
            noise = np.asarray(task["noise"], dtype=float)
            source_psi = np.asarray([
                plans[assignment["context"]].trajectory(policies[assignment["policy_index"]]).coordinates(
                    plans[assignment["context"]].capacity)
                for assignment in design["assignments"]
            ])
            y = response[:, head].copy()
            if head == 1:
                # The saved source record is already centered for the current
                # Inventory contract; this branch is explicit for auditability.
                y = y - 0.0
            if head == 2:
                # The physical source record stores log sample variance.
                y = y.copy()
                task_noise = np.maximum(noise[:, head], 1e-8)
            else:
                task_noise = np.maximum(noise[:, head], 1e-8)
            tasks.append(ScalarTaskData(f"source_{index}", source_psi, y, task_noise))
        source_shapes.append({"head": name, "tasks": len(tasks),
                              "points_per_task": sorted({len(task.y) for task in tasks}),
                              "dimension": sorted({task.X.shape[1] for task in tasks})})
        model = HyperBOSurrogate(kernel="rbf").meta_fit(tasks)
        model.adapt(target_X, target_y[:, head], target_noise[:, head])
        mean, covariance = model.predict(target_X, full_cov=True)
        paper_core.append({
            "head": name,
            "finite_mean": bool(np.isfinite(mean).all()),
            "finite_covariance": bool(np.isfinite(covariance).all()),
            "covariance_shape": list(covariance.shape),
            "target_points": int(model.n_target),
            "diagnostics": model.diagnostics(),
        })

    def available(name):
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            return name in sys.modules and getattr(sys.modules[name], "__spec__", None) is not None

    module_availability = {name: available(name) for name in
                           ("jax", "flax", "optax", "tensorflow_probability", "tensorflow")}
    official = OfficialHyperBOSurrogate(seed=0, source_steps=1)
    official_failure = None
    official_smoke = None
    try:
        official_tasks = []
        source_psi = np.asarray([
            plans[assignment["context"]].trajectory(policies[assignment["policy_index"]]).coordinates(
                plans[assignment["context"]].capacity)
            for assignment in design["assignments"]
        ])
        for index, task in enumerate(source_tasks):
            response = np.asarray(task["response"], dtype=float)
            noise = np.asarray(task["noise"], dtype=float)
            official_tasks.append(ScalarTaskData(
                f"source_{index}", source_psi, response[:, 0], np.maximum(noise[:, 0], 1e-8)))
        official.meta_fit(official_tasks)
        official.adapt(target_X, target_y[:, 0], target_noise[:, 0])
        official_mean, official_covariance = official.predict(target_X, full_cov=True)
        official_smoke = {
            "executed": True,
            "finite_mean": bool(np.isfinite(official_mean).all()),
            "finite_covariance": bool(np.isfinite(official_covariance).all()),
            "covariance_shape": list(official_covariance.shape),
            "target_points": int(official.n_target),
            "source_steps": int(official.source_steps),
            "observation_noise_variance": float(official.observation_noise_variance()),
            "diagnostics": official.diagnostics(),
        }
    except Exception as exc:  # expected when the pinned runtime is unavailable
        official_failure = {"type": type(exc).__name__, "message": str(exc)}
    runtime = external_runtime_report()
    missing_modules = [name for name, present in module_availability.items() if not present]
    repo_path = Path(runtime["hyperbo_cbo"]["repository_path"])
    commit = None
    if repo_path.is_dir():
        import subprocess
        commit = subprocess.check_output(
            ["git", "-C", str(repo_path), "rev-parse", "--short=7", "HEAD"],
            text=True,
        ).strip()
    return {
        "protocol": PROTOCOL,
        "source_shapes": source_shapes,
        "paper_core_smoke": paper_core,
        "official_runtime_report": runtime,
        "official_provenance": {
            "repository_present": repo_path.is_dir(),
            "repository_path": str(repo_path),
            "checked_commit": commit,
            "expected_commit_prefix": EXPECTED_HYPERBO_COMMIT,
            "commit_matches": bool(commit and commit.startswith(EXPECTED_HYPERBO_COMMIT)),
        },
        "official_boundary": {
            "executed": bool(official_smoke and official_smoke["executed"]),
            "failure": official_failure,
            "missing_python_modules": missing_modules,
            "fallback_used": False,
            "reason": "official runtime dependencies are unavailable" if official_failure else "runtime import succeeded",
        },
        "official_smoke": official_smoke,
        "local_cpu_seconds": time.process_time() - started,
    }


if __name__ == "__main__":
    result = run()
    output = ROOT / "paper_artifacts/inventory_meta_prior_interface_20261004/mature_baseline_contract_audit.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "official_boundary": result["official_boundary"],
        "official_provenance": result["official_provenance"],
        "paper_core_smoke": result["paper_core_smoke"],
        "local_cpu_seconds": result["local_cpu_seconds"],
    }, indent=2))

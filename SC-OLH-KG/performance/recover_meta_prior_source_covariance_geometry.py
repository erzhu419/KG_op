"""A19 recover only source eta covariance metadata; no prediction replay."""
import gzip
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.run_meta_prior_mechanism_v2 import prepare_sources
from performance.run_meta_prior_cross_head_risk_escape import contrast
from performance.source_covariance_geometry import direction_geometry
from performance.source_head_association import allocation_alignment
from representation.source_function_prior import head_batches
from representation.source_risk_ard import fit_source_risk_ard

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_source_covariance_geometry_a19_20261003"


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    start, wall = time.process_time(), time.monotonic()
    spec = read("performance/meta_prior_source_covariance_geometry_plan.json")
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def stop(signum, frame):
        raise TimeoutError("A19 source-metadata recovery CPU cap reached")
    signal.signal(signal.SIGXCPU, stop)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    result = {"stage": spec["stage"], "complete": False, "rows": [],
        "scope": spec["scope"], "new_fg_selection_fits": 0, "new_full_prior_fits": 0,
        "new_simulation_samples": 0, "new_predictive_forecasts": 0, "new_source_calls": 0,
        "server_jobs": 0, "server_pulls": 0, "kg_calls": 0, "new_searches": 0}
    contexts = []
    try:
        old = read(spec["inputs"][1])
        selections = read(spec["inputs"][2])
        frozen = read(spec["inputs"][3])["base_protocol"]
        attribution = read(spec["inputs"][4])
        families = read(spec["inputs"][5])["public_families"]
        cohort = spec["cohort"]
        expected = {(s, g, h) for s in cohort["seeds"] for g in cohort["archive_groups"] for h in range(8)}
        source_folds = {(r["seed"], r["archive_group"], r["held_task"]): r for r in old["folds"]}
        fg_folds = {(r["seed"], r["archive_group"], r["held_task"]): r for r in selections["folds"]}
        gains = {(r["seed"], r["archive_group"], r["held_task"]): r for r in attribution["rows"] if r["arm"] == "risk_ard_mixture"}
        if not old["complete"] or not attribution["complete"] or any(set(indexed) != expected for indexed in (source_folds, fg_folds, gains)):
            raise ValueError("A19 source folds do not match the completed source evidence")
        for seed in cohort["seeds"]:
            for group in cohort["archive_groups"]:
                generation_start = time.process_time()
                tasks, _ = prepare_sources(seed, "shared_structure" if group == "shared" else "separate_head_structure", frozen)
                reference = np.random.default_rng(51000 + seed).uniform(-2., 2., (frozen["reference_bank_size"], frozen["psi_dim"]))
                generation_seconds = time.process_time() - generation_start
                for held in range(8):
                    fit_start = time.process_time()
                    key = seed, group, held
                    training_ids = source_folds[key]["training_tasks"]
                    if held in training_ids or training_ids != fg_folds[key]["training_tasks"]:
                        raise ValueError("held source function enters recovered eta/fg fitting metadata")
                    training = [tasks[i] for i in training_ids]
                    prior = fit_source_risk_ard(head_batches(training, 2), [t["noise"][:, 2] for t in training], reference)
                    fit_seconds = time.process_time() - fit_start
                    diagnostics = prior.diagnostics
                    if diagnostics["optimizer_converged"] != source_folds[key]["optimizer_converged"] or diagnostics["optimizer_iterations"] != source_folds[key]["optimizer_iterations"]:
                        raise ValueError(f"recovered original eta optimizer metadata changed: {key}")
                    tau = diagnostics["source_function_variance"]
                    variances = np.diag(prior.covariance)[1:]
                    selected = dict(zip(("f", "g"), fg_folds[key]["fg_selected_indices"]))
                    geometry = {head: direction_geometry(variances, ids) for head, ids in selected.items()}
                    alignment = {head: allocation_alignment(tau, ids, families) for head, ids in selected.items()}
                    result["rows"].append({"seed": seed, "archive_group": group, "held_task": held,
                        "geometry": geometry, "alignment": alignment,
                        "saved_pure_branch_gain": {head: gains[key]["pure_" + head + "_gain"] for head in selected},
                        "optimizer_converged": diagnostics["optimizer_converged"],
                        "optimizer_iterations": diagnostics["optimizer_iterations"], "eta_fit_cpu_seconds": fit_seconds})
                    contexts.append({"seed": seed, "archive_group": group, "held_task": held,
                        "training_tasks": training_ids, "source_label_count": diagnostics["source_label_count"],
                        "eta_learned_function_variance": tau, "eta_prior_function_variance": variances.tolist(),
                        "eta_intercept_mean": float(prior.mean[0]), "eta_discrepancy_variance": prior.discrepancy_variance,
                        "fg_selected_indices": selected, "optimizer_converged": diagnostics["optimizer_converged"],
                        "optimizer_iterations": diagnostics["optimizer_iterations"]})
                    if len(contexts) == 1:
                        projected = time.process_time() - start + 1.25 * (generation_seconds * 31 + fit_seconds * 255) + 5.
                        result["admission"] = {"first_generation_cpu_seconds": generation_seconds,
                            "retained_first_eta_fit_cpu_seconds": fit_seconds, "projected_stage_cpu_seconds": projected,
                            "safety_multiplier": 1.25, "summary_cpu_reserve": 5., "cap_cpu_seconds": cap,
                            "admitted": projected < cap}
                        print(json.dumps({"admission": result["admission"]}), flush=True)
                        if projected >= cap:
                            raise TimeoutError("A19 complete-cohort recovery projection exceeds local CPU cap")
                print(json.dumps({"recovered_source_folds": len(contexts)}), flush=True)
        metrics = ("selected_covariance_mass", "outside_covariance_mass", "flat_reallocation_total_variation",
                   "extra_within_selection_redistribution")
        indices = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(16, size=(20000, 16))
        summaries = []
        for group in cohort["archive_groups"]:
            for head in ("f", "g"):
                archives = []
                for seed in cohort["seeds"]:
                    rows = [r for r in result["rows"] if (r["seed"], r["archive_group"]) == (seed, group)]
                    archives.append({"seed": seed, **{name: float(np.mean([r["geometry"][head][name] for r in rows])) for name in metrics},
                        "family_matched_excess": float(np.mean([r["alignment"][head]["excess_over_family_matched"] for r in rows])),
                        "pure_branch_gain": float(np.mean([r["saved_pure_branch_gain"][head] for r in rows]))})
                summaries.append({"archive_group": group, "head": head,
                    **{name: contrast([r[name] for r in archives], indices) for name in (*metrics, "family_matched_excess")},
                    "positive_association_archives": sum(r["family_matched_excess"] > 0. for r in archives),
                    "positive_association_but_negative_forecast_archives": sum(r["family_matched_excess"] > 0. and r["pure_branch_gain"] < 0. for r in archives),
                    "archive_values": archives})
        error = max(r["geometry"][h]["decomposition_error"] for r in result["rows"] for h in ("f", "g"))
        if error > 1e-12:
            raise ValueError("covariance geometry decomposition identity failed")
        result.update(complete=True, groups=summaries, eta_ard_refits=len(contexts),
            matched_original_optimizer_folds=len(contexts), optimizer_converged_folds=sum(r["optimizer_converged"] for r in contexts),
            max_geometry_decomposition_error=error, saved_context_file="source_fold_contexts.json.gz")
    except Exception as exc:
        result["failure"] = str(exc)
        raise
    finally:
        result.update(cpu_seconds=time.process_time() - start, wall_seconds=time.monotonic() - wall,
            current_process_vm_hwm_mb=float(next(line.split()[1] for line in
                Path("/proc/self/status").read_text().splitlines() if line.startswith("VmHWM:"))) / 1024)
        (ARTIFACTS / "source_fold_contexts.json.gz").write_bytes(gzip.compress(json.dumps({"rows": contexts}, allow_nan=False).encode()))
        (ARTIFACTS / "geometry.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: result[k] for k in ("complete", "cpu_seconds", "current_process_vm_hwm_mb")}), flush=True)


if __name__ == "__main__":
    main()

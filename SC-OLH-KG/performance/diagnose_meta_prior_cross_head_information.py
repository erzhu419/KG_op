"""A13 direction-information diagnosis; no new model, acquisition or simulation."""
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
from performance.cross_head_information import reconstruct_signals, score_support
from performance.replay_meta_prior_calibration import restore_context
from performance.run_meta_prior_mechanism import shared_support, target_supports

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_cross_head_information_a13_20261003"


def diagnostic_risk_support(seed, regime, dim):
    support = shared_support(seed, dim)
    if regime == "separate_head_structure":
        return target_supports(support, seed, "support_mismatch", dim)[0]
    return target_supports(support, seed, regime, dim)[2]


def summarize(rows, spec):
    cohort = spec["cohort"]
    seeds = cohort["seeds"]
    indices = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(
        len(seeds), size=(20000, len(seeds)))
    groups = []
    for regime in cohort["regimes"]:
        for arm in cohort["priors"]:
            for prefix in cohort["prefix_batches"]:
                rr = sorted((r for r in rows if (r["regime"], r["arm"], r["prefix_batches"]) ==
                             (regime, arm, prefix)), key=lambda r: r["seed"])
                if [r["seed"] for r in rr] != seeds:
                    raise ValueError("incomplete A13 paired-seed group")
                delta = np.array([r["fg_minus_eta"] for r in rr])
                groups.append({"regime": regime, "arm": arm, "prefix_batches": prefix,
                    "paths": len(rr), "mean_fg_minus_eta": float(delta.mean()),
                    "paired_seed_bootstrap95": np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist(),
                    "better": int(np.sum(delta > 1e-12)), "worse": int(np.sum(delta < -1e-12)),
                    "mean_support_mass": {h: float(np.mean([r["support_mass"][h] for r in rr]))
                                          for h in ("f", "g", "fg", "eta")},
                    "seed_deltas": delta.tolist()})
    return groups


def reproduction_error(models, control, row):
    x = np.array([[row["recommendation"]]])
    # Check a recorded candidate, without making any new recommendation.
    actual = [m.posterior_mean_many(x)[0] for m in models] + [m.posterior_var_many(x)[0] for m in models]
    wanted = control["posterior_mean"] + control["posterior_variance"]
    if row["arm"] == "risk_ard_mixture":
        actual += [m.diagnostics()["structural_weight"] for m in models]
        wanted += [m["structural_weight"] for m in row["final_model_state"]]
    error = float(np.max(np.abs(np.array(actual) - wanted)))
    if error > 1e-9:
        raise ValueError(f"A13 old posterior reconstruction changed: {error}")
    return error


def main():
    start, cpu_start = time.monotonic(), time.process_time()
    spec = json.loads((ROOT / "performance/meta_prior_cross_head_information_diagnostic_plan.json").read_text())
    inputs = ROOT / spec["input_artifacts"]
    frozen = json.loads((inputs / "execution_plan.json").read_text())["base_protocol"]
    protocol = {**frozen, **frozen["allocations"]["n24_v928"]}
    records = json.loads(gzip.decompress((inputs / "inputs.json.gz").read_bytes()))["records"]
    cohort = spec["cohort"]
    expected = {(s, r, a) for s in cohort["seeds"] for r in cohort["regimes"] for a in cohort["priors"]}
    key = lambda r: (r["seed"], r["regime"], r["arm"])
    actual = [key(r["row"]) for r in records]
    if (len(actual) != cohort["paths"] or len(actual) != len(expected) or set(actual) != expected or
            any(r["row"]["difficulty"] != "binding" or r["row"]["allocation"] != "n24_v928" for r in records)):
        raise ValueError("A13 archived cohort changed")
    old = json.loads((inputs / "summary.json").read_text())
    if not old["complete"] or not old["old_reproduction_passed"]:
        raise ValueError("A13 requires the completed original replay")
    controls = {key(r): r for r in old["rows"] if r["update"] == "sample_variance_plugin"}
    if set(controls) != expected:
        raise ValueError("A13 original posterior controls are incomplete")
    records.sort(key=lambda r: (r["row"]["seed"], cohort["regimes"].index(r["row"]["regime"]),
                                r["row"]["arm"] != "risk_ard_mixture"))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    cap = int(spec["limits"]["local_cpu_hours_cap"] * 3600)
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def cpu_limit(signum, frame):
        raise TimeoutError("A13 reached its local CPU budget")
    signal.signal(signal.SIGXCPU, cpu_limit)
    result = {"stage": spec["stage"], "complete": False, "paths_completed": 0,
        "rows": [], "max_old_posterior_reproduction_error": 0., "old_reproduction_passed": False,
        "server_experiment_jobs": 0, "new_server_pulls": 0, "new_source_model_selection": 0,
        "new_source_calls": 0, "new_target_calls": 0, "kg_calls": 0, "new_searches": 0,
        "new_random_verifications": 0, "new_joint_prior_fits": 0, "memory_checkpoints": [],
        "interpretation": "Retrospective information diagnosis, not support probability, prediction or decision evidence."}
    contexts = {}
    try:
        for count, record in enumerate(records, 1):
            path_start = time.process_time()
            row = record["row"]
            context_key = row["seed"], row["regime"]
            if context_key not in contexts:
                # Sorted paired paths only need the current seed/regime context.
                contexts.clear()
                contexts[context_key] = restore_context(record, protocol)
            snapshots, models = reconstruct_signals(record, contexts[context_key], protocol,
                                                   cohort["prefix_batches"])
            error = reproduction_error(models, controls[key(row)], row)
            result["max_old_posterior_reproduction_error"] = max(result["max_old_posterior_reproduction_error"], error)
            # Scoring support is obtained only after this path's signals are fixed.
            support = diagnostic_risk_support(row["seed"], row["regime"], protocol["psi_dim"])
            for snapshot in snapshots:
                result["rows"].append({**{k: row[k] for k in ("seed", "regime", "arm")},
                    **snapshot, **score_support(snapshot["signals"], support), "risk_support_for_scoring": support})
            result["paths_completed"] = count
            if count == 1:
                seconds = time.process_time() - path_start
                result["admission"] = {"first_complete_learned_path_cpu_seconds": seconds,
                    "forecast_all_paths_cpu_seconds": seconds * len(records), "cpu_cap_seconds": cap,
                    "admitted": seconds * len(records) + time.process_time() - cpu_start <= cap}
                print(json.dumps(result["admission"]), flush=True)
                if not result["admission"]["admitted"]:
                    result["status"] = "partial_resource_admission_stopped"
                    return
            if count % 20 == 0:
                result["memory_checkpoints"].append({"paths_completed": count,
                    "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024})
                print(f"DIAGNOSED {count}/{len(records)}", flush=True)
        result["groups"] = summarize(result["rows"], spec)
        if len(result["rows"]) != cohort["posterior_snapshots"]:
            raise ValueError("A13 posterior snapshot cohort changed")
        result.update(complete=True, old_reproduction_passed=True, status="information_diagnosis_complete")
    finally:
        result.update(cpu_seconds=time.process_time() - cpu_start, wall_seconds=time.monotonic() - start,
                      peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        (ARTIFACTS / "diagnosis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("complete", "paths_completed", "old_reproduction_passed",
        "max_old_posterior_reproduction_error", "cpu_seconds", "wall_seconds", "peak_rss_mb")}), flush=True)


if __name__ == "__main__":
    main()

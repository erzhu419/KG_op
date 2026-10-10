"""A3 grouped predictive validation on existing V6 paths only."""
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
from performance.replay_meta_prior_calibration import restore_context
from performance.run_meta_prior_mechanism import observe
from performance.target_risk_residual import grouped_validation

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_target_risk_residual_a3_20261003"


def summarize(rows, spec, seeds):
    indices = np.random.default_rng(20261006).integers(len(seeds), size=(20000, len(seeds)))
    groups = []
    for regime in spec["regimes"]:
        for arm in spec["priors"]:
            for prefix in spec["prefix_batches"]:
                rr = sorted((r for r in rows if (r["regime"], r["arm"], r["prefix"]) ==
                            (regime, arm, prefix)), key=lambda r: r["seed"])
                changes = np.array([r["residual_log_score_per_batch"] - r["old_log_score_per_batch"] for r in rr])
                deltas = [f["delta"] for r in rr for f in r["folds"]]
                groups.append({"regime": regime, "arm": arm, "prefix": prefix,
                    "paths": len(rr), "mean_log_score_gain": float(changes.mean()),
                    "paired_seed_bootstrap95": np.quantile(changes[indices].mean(axis=1), [.025, .975]).tolist(),
                    "better": int(np.sum(changes > 1e-12)), "worse": int(np.sum(changes < -1e-12)),
                    "mean_old_score": float(np.mean([r["old_log_score_per_batch"] for r in rr])),
                    "mean_residual_score": float(np.mean([r["residual_log_score_per_batch"] for r in rr])),
                    "median_training_delta": float(np.median(deltas)), "max_training_delta": float(np.max(deltas)),
                    "seed_log_score_gains": changes.tolist()})
    return groups


def main():
    start, cpu_start = time.monotonic(), time.process_time()
    spec = json.loads((ROOT / "performance/meta_prior_target_risk_residual_plan.json").read_text())
    inputs = ROOT / spec["input_artifacts"]
    execution = json.loads((inputs / "execution_plan.json").read_text())
    frozen = execution["base_protocol"]
    protocol = {**frozen, **frozen["allocations"]["n24_v928"]}
    records = json.loads(gzip.decompress((inputs / "inputs.json.gz").read_bytes()))["records"]
    expected = {(s, r, a) for s in execution["stage_a"]["seeds"] for r in spec["regimes"] for a in spec["priors"]}
    actual = [(r["row"]["seed"], r["row"]["regime"], r["row"]["arm"]) for r in records]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("input paths do not match the planned A3 cohort")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    cap = int(spec["limits"]["local_cpu_hours_cap"] * 3600)
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def cpu_limit(signum, frame):
        raise TimeoutError("A3 reached its local CPU budget")
    signal.signal(signal.SIGXCPU, cpu_limit)
    result = {"stage": spec["stage"], "complete": False, "phase": "grouped_predictive_validation",
        "paths": len(records), "rows": [], "server_experiment_jobs": 0, "new_server_pulls": 0,
        "kg_calls": 0, "new_searches": 0, "new_random_verifications": 0,
        "scope": "Retrospective grouped predictive validation; delta refit separately within every outer candidate holdout."}
    contexts = {}
    try:
        for count, record in enumerate(records, 1):
            row = record["row"]
            key = row["seed"], row["regime"]
            if key not in contexts:
                learned = next(r for r in records if (r["row"]["seed"], r["row"]["regime"], r["row"]["arm"]) ==
                               (*key, "risk_ard_mixture"))
                contexts[key] = restore_context(learned, protocol)
            psi, reference, full, structural, truth, target_seed = contexts[key]
            priors = [structural[2], full[2]] if row["arm"] == "risk_ard_mixture" else [full[2]]
            ids = np.asarray(row["queries"])
            visits, labels, noise = np.zeros(len(psi), dtype=int), [], []
            for index in ids:
                response, variances = observe(truth[index], target_seed, int(index), int(visits[index]),
                                               protocol["replicates"])
                visits[index] += 1
                labels.append(response[2])
                noise.append(variances[2])
            labels, noise = np.asarray(labels), np.asarray(noise)
            for prefix in spec["prefix_batches"]:
                # The fitting/CV function receives only features and paid eta labels.
                validated = grouped_validation(priors, psi, ids[:prefix], labels[:prefix], noise[:prefix])
                for fold in validated["folds"]:
                    if fold["held_candidate"] in fold["training_candidates"]:
                        raise ValueError("held candidate leaked into delta fitting")
                result["rows"].append({**{k: row[k] for k in ("seed", "regime", "arm")},
                                       "prefix": prefix, **validated})
            if count % 20 == 0:
                print(f"VALIDATED {count}/{len(records)} paths", flush=True)
        result.update(groups=summarize(result["rows"], spec, execution["stage_a"]["seeds"]), complete=True)
    finally:
        result.update(cpu_seconds=time.process_time() - cpu_start, wall_seconds=time.monotonic() - start,
                      peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        (ARTIFACTS / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("complete", "paths", "cpu_seconds", "wall_seconds", "peak_rss_mb")}), flush=True)


if __name__ == "__main__":
    main()

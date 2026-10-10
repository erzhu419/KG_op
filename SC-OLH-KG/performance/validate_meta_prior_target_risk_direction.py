"""A5 paid-label direction selection on frozen existing query paths."""
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
from performance.target_risk_direction import grouped_direction_validation

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_target_risk_direction_a5_20261003"


def summarize(rows, spec):
    indices = np.random.default_rng(20261009).integers(len(spec["seeds"]), size=(20000, len(spec["seeds"])))
    groups = []
    for regime in spec["regimes"]:
        for arm in spec["priors"]:
            for prefix in spec["prefix_batches"]:
                rr = sorted((r for r in rows if (r["regime"], r["arm"], r["prefix"]) ==
                             (regime, arm, prefix)), key=lambda r: r["seed"])
                gains = np.array([r["direction_log_score_per_batch"] - r["old_log_score_per_batch"] for r in rr])
                folds = [f for r in rr for f in r["folds"]]
                groups.append({"regime": regime, "arm": arm, "prefix": prefix, "paths": len(rr),
                    "mean_log_score_gain": float(gains.mean()),
                    "paired_seed_bootstrap95": np.quantile(gains[indices].mean(axis=1), [.025, .975]).tolist(),
                    "better": int(np.sum(gains > 1e-12)), "worse": int(np.sum(gains < -1e-12)),
                    "zero_correction_folds": sum(f["direction"] is None for f in folds),
                    "total_folds": len(folds), "seed_log_score_gains": gains.tolist()})
    return groups


def main():
    start, cpu_start = time.monotonic(), time.process_time()
    spec = json.loads((ROOT / "performance/meta_prior_target_risk_direction_plan.json").read_text())
    controls = json.loads((ROOT / spec["control_scores"]).read_text())
    if not controls["complete"]:
        raise ValueError("A3 control scores are incomplete")
    old_rows = {(r["seed"], r["regime"], r["arm"], r["prefix"]): r for r in controls["rows"]}
    inputs = ROOT / spec["input_artifacts"]
    frozen = json.loads((inputs / "execution_plan.json").read_text())["base_protocol"]
    protocol = {**frozen, **frozen["allocations"]["n24_v928"]}
    records = json.loads(gzip.decompress((inputs / "inputs.json.gz").read_bytes()))["records"]
    expected = {(s, r, a) for s in spec["seeds"] for r in spec["regimes"] for a in spec["priors"]}
    actual = [(r["row"]["seed"], r["row"]["regime"], r["row"]["arm"]) for r in records]
    if len(actual) != spec["paths"] or len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("A5 input paths changed the planned cohort")
    if set(old_rows) != {(s, r, a, n) for s, r, a in expected for n in spec["prefix_batches"]}:
        raise ValueError("A5 control scores changed the planned cohort")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    cap = int(spec["limits"]["local_cpu_hours_cap"] * 3600)
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def cpu_limit(signum, frame):
        raise TimeoutError("A5 reached its local CPU budget")
    signal.signal(signal.SIGXCPU, cpu_limit)
    result = {"stage": spec["stage"], "complete": False, "phase": "grouped_predictive_validation",
        "paths": len(records), "rows": [], "server_experiment_jobs": 0, "new_server_pulls": 0,
        "kg_calls": 0, "new_searches": 0, "new_random_verifications": 0,
        "scope": "Retrospective candidate-group validation; direction and scale refit using paid training labels only."}
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
            if any(p.basis.feature_dim != 68 or p.basis.names != full[2].basis.names for p in priors):
                raise ValueError("A5 requires the existing common 68-function eta dictionary")
            ids = np.asarray(row["queries"])
            visits, labels, noise = np.zeros(len(psi), dtype=int), [], []
            for index in ids:
                y, n = observe(truth[index], target_seed, int(index), int(visits[index]), protocol["replicates"])
                visits[index] += 1
                labels.append(y[2])
                noise.append(n[2])
            labels, noise = np.asarray(labels), np.asarray(noise)
            for prefix in spec["prefix_batches"]:
                old = old_rows[(*key, row["arm"], prefix)]
                # Selection receives only source priors, public features and paid labels.
                validated = grouped_direction_validation(priors, psi, ids[:prefix], labels[:prefix],
                                                         noise[:prefix], old["folds"])
                result["rows"].append({**{k: row[k] for k in ("seed", "regime", "arm")},
                                      "prefix": prefix, **validated})
            if count % 20 == 0:
                print(f"DIRECTION VALIDATED {count}/{len(records)} paths", flush=True)
        result.update(groups=summarize(result["rows"], spec), complete=True)
    finally:
        result.update(cpu_seconds=time.process_time() - cpu_start, wall_seconds=time.monotonic() - start,
                      peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        (ARTIFACTS / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("complete", "paths", "cpu_seconds", "wall_seconds", "peak_rss_mb")}), flush=True)


if __name__ == "__main__":
    main()

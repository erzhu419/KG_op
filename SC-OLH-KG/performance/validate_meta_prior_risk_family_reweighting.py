"""A9 public-family risk covariance adaptation on existing query paths."""
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
from performance.risk_family_reweighting import grouped_family_validation

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_risk_family_reweighting_a9_20261003"


def summarize(rows, spec):
    indices = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(len(spec["seeds"]),
                                                                          size=(20000, len(spec["seeds"])))
    groups = []
    for regime in spec["regimes"]:
        for arm in spec["priors"]:
            for prefix in spec["prefix_batches"]:
                rr = sorted((r for r in rows if (r["regime"], r["arm"], r["prefix"]) ==
                             (regime, arm, prefix)), key=lambda r: r["seed"])
                gains = np.array([r["family_log_score_per_batch"] - r["old_log_score_per_batch"] for r in rr])
                folds = [f for r in rr for f in r["folds"]]
                groups.append({"regime": regime, "arm": arm, "prefix": prefix, "paths": len(rr),
                    "mean_log_score_gain": float(gains.mean()),
                    "paired_seed_bootstrap95": np.quantile(gains[indices].mean(axis=1), [.025, .975]).tolist(),
                    "better": int(np.sum(gains > 1e-12)), "worse": int(np.sum(gains < -1e-12)),
                    "mean_null_posterior": float(np.mean([f["null_posterior"] for f in folds])),
                    "mean_rho_posterior": float(np.mean([f["mean_rho_posterior"] for f in folds])),
                    "mean_family_posterior": {name: float(np.mean([f["family_posterior"][name] for f in folds]))
                                              for name in spec["public_families"]},
                    "seed_log_score_gains": gains.tolist()})
    return groups


def main():
    start, cpu_start = time.monotonic(), time.process_time()
    spec = json.loads((ROOT / "performance/meta_prior_risk_family_reweighting_plan.json").read_text())
    controls = json.loads((ROOT / spec["control_scores"]).read_text())
    if not controls["complete"]:
        raise ValueError("A3 control scores are incomplete")
    old_rows = {(r["seed"], r["regime"], r["arm"], r["prefix"]): r for r in controls["rows"]}
    families = spec["public_families"]
    if sorted(j for indices in families.values() for j in indices) != list(range(68)):
        raise ValueError("A9 public families do not partition the original dictionary")
    inputs = ROOT / spec["input_artifacts"]
    frozen = json.loads((inputs / "execution_plan.json").read_text())["base_protocol"]
    protocol = {**frozen, **frozen["allocations"]["n24_v928"]}
    records = json.loads(gzip.decompress((inputs / "inputs.json.gz").read_bytes()))["records"]
    expected = {(s, r, a) for s in spec["seeds"] for r in spec["regimes"] for a in spec["priors"]}
    actual = [(r["row"]["seed"], r["row"]["regime"], r["row"]["arm"]) for r in records]
    if len(actual) != spec["paths"] or len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("A9 input paths changed the planned cohort")
    if set(old_rows) != {(s, r, a, n) for s, r, a in expected for n in spec["prefix_batches"]}:
        raise ValueError("A9 control scores changed the planned cohort")
    first = next(r for r in records if (r["row"]["seed"], r["row"]["regime"], r["row"]["arm"]) ==
                 (spec["seeds"][0], "shared_structure", "risk_ard_mixture"))
    records = [first] + [r for r in records if r is not first]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    cap = int(spec["limits"]["local_cpu_hours_cap"] * 3600)
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def cpu_limit(signum, frame):
        raise TimeoutError("A9 reached its local CPU budget")
    signal.signal(signal.SIGXCPU, cpu_limit)
    result = {"stage": spec["stage"], "complete": False, "phase": "grouped_predictive_validation",
        "paths": len(records), "rows": [], "server_experiment_jobs": 0, "new_server_pulls": 0,
        "kg_calls": 0, "new_searches": 0, "new_random_verifications": 0,
        "scope": "Paid-label public-family covariance adaptation; fixed source variance budgets; existing holdouts only."}
    contexts = {}
    try:
        for count, record in enumerate(records, 1):
            path_start = time.process_time()
            row = record["row"]
            key = row["seed"], row["regime"]
            if key not in contexts:
                learned = next(r for r in records if (r["row"]["seed"], r["row"]["regime"], r["row"]["arm"]) ==
                               (*key, "risk_ard_mixture"))
                contexts[key] = restore_context(learned, protocol)
            psi, reference, full, structural, truth, target_seed = contexts[key]
            priors = [structural[2], full[2]] if row["arm"] == "risk_ard_mixture" else [full[2]]
            if any(p.basis.feature_dim != 68 or p.basis.names != full[2].basis.names for p in priors):
                raise ValueError("A9 requires the existing common 68-function eta dictionary")
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
                validated = grouped_family_validation(priors, psi, ids[:prefix], labels[:prefix], noise[:prefix],
                                                       old["folds"], families, spec["quadrature_nodes"])
                result["rows"].append({**{k: row[k] for k in ("seed", "regime", "arm")}, "prefix": prefix, **validated})
            if count == 1:
                seconds = time.process_time() - path_start
                forecast = time.process_time() - cpu_start + seconds * (len(records) - 1)
                result["resource_admission"] = {"seed": row["seed"], "regime": row["regime"], "arm": row["arm"],
                    "first_path_cpu_seconds": seconds, "forecast_cpu_seconds": forecast,
                    "cap_cpu_seconds": cap, "admitted": forecast <= cap}
                (ARTIFACTS / "resource_admission.json").write_text(json.dumps(result["resource_admission"], indent=2) + "\n")
                print("ADMISSION " + json.dumps(result["resource_admission"]), flush=True)
                if forecast > cap:
                    result["stop_reason"] = "First-path forecast exceeds local CPU cap; preserve partial results."
                    break
            if count % 20 == 0:
                print(f"FAMILY VALIDATED {count}/{len(records)} paths", flush=True)
        if len(result["rows"]) == spec["paths"] * len(spec["prefix_batches"]):
            result.update(groups=summarize(result["rows"], spec), complete=True)
    finally:
        result.update(cpu_seconds=time.process_time() - cpu_start, wall_seconds=time.monotonic() - start,
                      peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        (ARTIFACTS / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("complete", "paths", "cpu_seconds", "wall_seconds", "peak_rss_mb")}), flush=True)


if __name__ == "__main__":
    main()

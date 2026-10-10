"""A10 chronological assessment of paid-label predictive calibration."""
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
from performance.risk_adaptation_calibration import prequential_forecasts

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_risk_adaptation_calibration_a10_20261003"


def summarize(rows, spec):
    cohort = spec["cohort"]
    count = len(cohort["seeds"])
    indices = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(count, size=(20000, count))
    groups = []
    for regime in cohort["regimes"]:
        for arm in cohort["priors"]:
            rr = sorted((r for r in rows if (r["regime"], r["arm"]) == (regime, arm)), key=lambda r: r["seed"])
            forecasts = [f for r in rr for f in r["forecasts"]]
            gains = np.array([r["mean_calibrated_minus_source"] for r in rr])
            vs_a9 = np.array([r["mean_calibrated_minus_a9"] for r in rr])
            groups.append({"regime": regime, "arm": arm, "paths": len(rr),
                "mean_gain": float(gains.mean()),
                "paired_seed_bootstrap95": np.quantile(gains[indices].mean(axis=1), [.025, .975]).tolist(),
                "better": int(np.sum(gains > 1e-12)), "worse": int(np.sum(gains < -1e-12)),
                "mean_gain_vs_a9": float(vs_a9.mean()),
                "vs_a9_paired_seed_bootstrap95": np.quantile(vs_a9[indices].mean(axis=1), [.025, .975]).tolist(),
                "mean_pi": float(np.mean([f["pi"] for f in forecasts])),
                "mean_a9_adaptation_mass": float(np.mean([1. - f["a9_null_posterior"] for f in forecasts])),
                "pi_zero_forecasts": sum(f["pi"] == 0. for f in forecasts),
                "pi_one_forecasts": sum(f["pi"] == 1. for f in forecasts),
                "new_candidate_forecasts": sum(f["new_candidate"] for f in forecasts),
                "seed_gains": gains.tolist(), "seed_gains_vs_a9": vs_a9.tolist()})
    return groups


def main():
    start, cpu_start = time.monotonic(), time.process_time()
    spec = json.loads((ROOT / "performance/meta_prior_risk_adaptation_calibration_plan.json").read_text())
    cohort = spec["cohort"]
    a9 = json.loads((ROOT / spec["a9_execution_plan"]).read_text())
    baseline = json.loads((ROOT / a9["artifacts"] / "prequential.json").read_text())
    if not baseline["complete"]:
        raise ValueError("A9 chronological controls are incomplete")
    old_rows = {(r["seed"], r["regime"], r["arm"]): r for r in baseline["rows"]}
    inputs = ROOT / spec["input_artifacts"]
    frozen = json.loads((inputs / "execution_plan.json").read_text())["base_protocol"]
    protocol = {**frozen, **frozen["allocations"]["n24_v928"]}
    records = json.loads(gzip.decompress((inputs / "inputs.json.gz").read_bytes()))["records"]
    expected = {(s, r, a) for s in cohort["seeds"] for r in cohort["regimes"] for a in cohort["priors"]}
    actual = [(r["row"]["seed"], r["row"]["regime"], r["row"]["arm"]) for r in records]
    if len(actual) != cohort["paths"] or len(actual) != len(expected) or set(actual) != expected or set(old_rows) != expected:
        raise ValueError("A10 cohort differs from its fixed plan or A9 controls")
    if any(r["row"]["difficulty"] != "binding" or r["row"]["allocation"] != "n24_v928" for r in records):
        raise ValueError("A10 input difficulty or allocation changed")
    first = next(r for r in records if (r["row"]["seed"], r["row"]["regime"], r["row"]["arm"]) ==
                 (cohort["seeds"][0], "shared_structure", "risk_ard_mixture"))
    records = [first] + [r for r in records if r is not first]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    cap = int(spec["limits"]["local_cpu_hours_cap"] * 3600)
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def cpu_limit(signum, frame):
        raise TimeoutError("A10 reached its local CPU budget")
    signal.signal(signal.SIGXCPU, cpu_limit)
    result = {"stage": spec["stage"], "complete": False, "paths": len(records), "rows": [],
        "max_a9_control_log_score_error": 0., "server_experiment_jobs": 0, "new_server_pulls": 0,
        "kg_calls": 0, "new_searches": 0, "new_random_verifications": 0,
        "scope": "Strictly earlier-label calibration and prediction on unchanged historical queries."}
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
                raise ValueError("A10 changed the common risk dictionary")
            ids = np.asarray(row["queries"])
            if len(ids) != protocol["target_points"] or ids[:8].tolist() != row["initial_design"]:
                raise ValueError("A10 historical query budget or initial design changed")
            visits, labels, noise = np.zeros(len(psi), dtype=int), [], []
            for index in ids:
                y, n = observe(truth[index], target_seed, int(index), int(visits[index]), protocol["replicates"])
                visits[index] += 1
                labels.append(y[2])
                noise.append(n[2])
            forecasts = prequential_forecasts(priors, psi, ids, np.asarray(labels), np.asarray(noise),
                a9["public_families"], a9["quadrature_nodes"], cohort["forecast_steps"])
            old = old_rows[(*key, row["arm"])]["forecasts"]
            for new, previous in zip(forecasts, old):
                if (new["step"], new["candidate"]) != (previous["step"], previous["candidate"]):
                    raise ValueError("A10 forecast sequence changed")
                differences = [abs(new[k] - previous[k]) for k in ("source", "adapted", "a9")]
                result["max_a9_control_log_score_error"] = max(result["max_a9_control_log_score_error"], max(differences))
                if max(differences) > 1e-9:
                    raise ValueError("A10 changed a source or A9 control predictive density")
            result["rows"].append({**{k: row[k] for k in ("seed", "regime", "arm")}, "forecasts": forecasts,
                "mean_calibrated_minus_source": float(np.mean([f["calibrated_log_score"] - f["source"] for f in forecasts])),
                "mean_calibrated_minus_a9": float(np.mean([f["calibrated_log_score"] - f["a9"] for f in forecasts]))})
            if count == 1:
                seconds = time.process_time() - path_start
                forecast = time.process_time() - cpu_start + seconds * (len(records) - 1)
                result["resource_admission"] = {"seed": row["seed"], "regime": row["regime"], "arm": row["arm"],
                    "first_path_cpu_seconds": seconds, "forecast_cpu_seconds": forecast,
                    "cap_cpu_seconds": cap, "admitted": forecast <= cap}
                (ARTIFACTS / "resource_admission.json").write_text(json.dumps(result["resource_admission"], indent=2) + "\n")
                print("ADMISSION " + json.dumps(result["resource_admission"]), flush=True)
                if forecast > cap:
                    result["stop_reason"] = "First-path forecast exceeds local CPU cap; preserve partial evidence."
                    break
            if count % 20 == 0:
                print(f"CALIBRATED {count}/{len(records)} paths", flush=True)
        if len(result["rows"]) == cohort["paths"]:
            result.update(complete=True, groups=summarize(result["rows"], spec))
    finally:
        result.update(cpu_seconds=time.process_time() - cpu_start, wall_seconds=time.monotonic() - start,
                      peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        (ARTIFACTS / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("complete", "cpu_seconds", "wall_seconds", "peak_rss_mb")}), flush=True)


if __name__ == "__main__":
    main()

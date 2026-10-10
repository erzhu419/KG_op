"""Forecast each old risk label from its strictly earlier paid observations."""
import gzip
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.replay_meta_prior_calibration import restore_context
from performance.run_meta_prior_mechanism import observe
from performance.risk_family_reweighting import family_predictive_branches
from performance.target_risk_direction_mixture import joint_label_log_scores

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_risk_family_reweighting_a9_20261003"


def model_log_scores(weights, label_scores, components, families, nodes):
    null = np.zeros(len(weights), dtype=bool)
    null[np.arange(components) * (1 + families * nodes)] = True
    return {"source": float(logsumexp(weights[null] + label_scores[null]) - logsumexp(weights[null])),
            "adapted": float(logsumexp(weights[~null] + label_scores[~null]) - logsumexp(weights[~null])),
            "a9": float(logsumexp(weights + label_scores))}


def main():
    start, cpu_start = time.monotonic(), time.process_time()
    spec = json.loads((ARTIFACTS / "execution_plan.json").read_text())
    audit = {"question": "Does A9 improve the next old observation when only earlier labels are used?",
        "protocol_fixed_before_audit_results": True, "forecast_steps": list(range(9, 25)),
        "paths": spec["paths"], "cohort": "Same 160 A9 paths and source priors; all five strata retained.",
        "primary": "Mean per-batch A9-minus-source log score over steps 9-24; proper full predictive mixture.",
        "interpretation": "Historical-policy prequential development evidence, not independent confirmation or a new-policy outcome.",
        "limits": spec["limits"], "bootstrap_seed": 20261018}
    (ARTIFACTS / "prequential_plan.json").write_text(json.dumps(audit, indent=2) + "\n")
    cap = int(spec["limits"]["local_cpu_hours_cap"] * 3600)
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def cpu_limit(signum, frame):
        raise TimeoutError("A9 prequential audit reached local CPU budget")
    signal.signal(signal.SIGXCPU, cpu_limit)
    inputs = ROOT / spec["input_artifacts"]
    frozen = json.loads((inputs / "execution_plan.json").read_text())["base_protocol"]
    protocol = {**frozen, **frozen["allocations"]["n24_v928"]}
    records = json.loads(gzip.decompress((inputs / "inputs.json.gz").read_bytes()))["records"]
    expected = {(s, r, a) for s in spec["seeds"] for r in spec["regimes"] for a in spec["priors"]}
    actual = [(r["row"]["seed"], r["row"]["regime"], r["row"]["arm"]) for r in records]
    if len(actual) != spec["paths"] or set(actual) != expected:
        raise ValueError("A9 prequential cohort changed")
    result = {"complete": False, "paths": len(records), "rows": [], "server_experiment_jobs": 0,
              "new_server_pulls": 0, "kg_calls": 0, "new_searches": 0, "new_random_verifications": 0}
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
                y, n = observe(truth[index], target_seed, int(index), int(visits[index]), protocol["replicates"])
                visits[index] += 1
                labels.append(y[2])
                noise.append(n[2])
            labels, noise = np.asarray(labels), np.asarray(noise)
            forecasts = []
            for step in range(8, 24):
                # The predictor receives no current/future label or noise realization.
                weights, means, variances, summary = family_predictive_branches(priors, psi,
                    ids[:step], labels[:step], noise[:step], ids[step], spec["public_families"], spec["quadrature_nodes"])
                scores = joint_label_log_scores(means, variances, labels[step:step + 1], noise[step:step + 1])
                forecasts.append({"step": step + 1, "candidate": int(ids[step]),
                    "new_candidate": bool(ids[step] not in ids[:step]),
                    **model_log_scores(weights, scores, len(priors), len(spec["public_families"]), spec["quadrature_nodes"])})
            result["rows"].append({**{k: row[k] for k in ("seed", "regime", "arm")}, "forecasts": forecasts,
                "mean_a9_minus_source": float(np.mean([f["a9"] - f["source"] for f in forecasts]))})
            if count % 20 == 0:
                print(f"PREQUENTIAL AUDITED {count}/{len(records)} paths", flush=True)
        indices = np.random.default_rng(audit["bootstrap_seed"]).integers(16, size=(20000, 16))
        groups = []
        for regime in spec["regimes"]:
            for arm in spec["priors"]:
                rr = sorted((r for r in result["rows"] if (r["regime"], r["arm"]) == (regime, arm)), key=lambda r: r["seed"])
                gains = np.array([r["mean_a9_minus_source"] for r in rr])
                groups.append({"regime": regime, "arm": arm, "paths": len(rr), "mean_gain": float(gains.mean()),
                    "paired_seed_bootstrap95": np.quantile(gains[indices].mean(axis=1), [.025, .975]).tolist(),
                    "better": int(np.sum(gains > 1e-12)), "worse": int(np.sum(gains < -1e-12)),
                    "new_candidate_forecasts": sum(f["new_candidate"] for r in rr for f in r["forecasts"]),
                    "seed_gains": gains.tolist()})
        result.update(complete=True, groups=groups)
    finally:
        result.update(cpu_seconds=time.process_time() - cpu_start, wall_seconds=time.monotonic() - start,
                      peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        (ARTIFACTS / "prequential.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("complete", "cpu_seconds", "wall_seconds", "peak_rss_mb")}), flush=True)


if __name__ == "__main__":
    main()

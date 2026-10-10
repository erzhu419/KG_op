"""A4 nondeployable support reference using A3's exact held-out groups."""
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
from performance.run_meta_prior_total_budget import simulator_supports
from performance.target_risk_residual import predictive_group_score
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_target_risk_support_a4_20261003"


def main():
    start, cpu_start = time.monotonic(), time.process_time()
    spec = json.loads((ROOT / "performance/meta_prior_target_risk_support_diagnostic_protocol.json").read_text())
    controls = json.loads((ROOT / spec["control_scores"]).read_text())
    if not controls["complete"]:
        raise ValueError("A3 control cohort is incomplete")
    control_rows = {(r["seed"], r["regime"], r["arm"], r["prefix"]): r for r in controls["rows"]}
    inputs = ROOT / spec["inputs"]
    frozen = json.loads((inputs / "execution_plan.json").read_text())["base_protocol"]
    protocol = {**frozen, **frozen["allocations"]["n24_v928"]}
    records = json.loads(gzip.decompress((inputs / "inputs.json.gz").read_bytes()))["records"]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    cap = int(spec["limits"]["local_cpu_hours_cap"] * 3600)
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def cpu_limit(signum, frame):
        raise TimeoutError("A4 reached its local CPU budget")
    signal.signal(signal.SIGXCPU, cpu_limit)
    result = {"stage": spec["stage"], "complete": False, "scope": spec["interpretation"],
        "oracle_information": spec["oracle_information"], "rows": [], "server_experiment_jobs": 0,
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
            support = simulator_supports(row["seed"], row["regime"], protocol["psi_dim"])[2]
            # This explicitly oracle-only branch provides directions, not coefficients.
            original = full[2]
            budget = float(np.trace(original.covariance[1:, 1:]))
            basis = FixedFunctionBasis(reference, support)
            oracle = SourceFunctionPrior(basis, np.r_[original.mean[0], np.zeros(len(support))],
                np.diag(np.r_[original.covariance[0, 0], np.full(len(support), budget / len(support))]),
                original.discrepancy_variance, {"mode": "oracle_target_support_diagnostic"})
            np.testing.assert_allclose(np.trace(oracle.covariance[1:, 1:]), budget, atol=1e-12)
            if np.any(oracle.mean[1:]):
                raise ValueError("target coefficients leaked into the oracle direction prior")
            ids = np.asarray(row["queries"])
            visits, labels, noise = np.zeros(len(psi), dtype=int), [], []
            for index in ids:
                y, n = observe(truth[index], target_seed, int(index), int(visits[index]), protocol["replicates"])
                visits[index] += 1
                labels.append(y[2])
                noise.append(n[2])
            labels, noise = np.asarray(labels), np.asarray(noise)
            for prefix in spec["prefix_batches"]:
                control = control_rows[(*key, row["arm"], prefix)]
                paid_ids, paid_y, paid_noise = ids[:prefix], labels[:prefix], noise[:prefix]
                folds = []
                for old in control["folds"]:
                    candidate = old["held_candidate"]
                    train = paid_ids != candidate
                    if np.unique(paid_ids[train]).tolist() != old["training_candidates"]:
                        raise ValueError("A4 changed the A3 candidate holdout")
                    score = predictive_group_score([oracle], psi, paid_ids[train], paid_y[train], paid_noise[train],
                        0., candidate, paid_y[~train], paid_noise[~train])
                    folds.append({"held_candidate": candidate, "held_batches": int((~train).sum()),
                        "old_joint_log_score": old["old_joint_log_score"], "oracle_joint_log_score": score})
                result["rows"].append({**{k: row[k] for k in ("seed", "regime", "arm")}, "prefix": prefix,
                    "oracle_support": support, "source_variance_budget": budget,
                    "old_log_score_per_batch": control["old_log_score_per_batch"],
                    "oracle_log_score_per_batch": sum(f["oracle_joint_log_score"] for f in folds) / prefix,
                    "folds": folds})
            if count % 20 == 0:
                print(f"ORACLE DIAGNOSED {count}/{len(records)} paths", flush=True)
        indices = np.random.default_rng(20261007).integers(16, size=(20000, 16))
        groups = []
        for regime in frozen["regimes"]:
            for arm in ("fixed_source", "risk_ard_mixture"):
                for prefix in spec["prefix_batches"]:
                    rr = sorted((r for r in result["rows"] if (r["regime"], r["arm"], r["prefix"]) ==
                                 (regime, arm, prefix)), key=lambda r: r["seed"])
                    gain = np.array([r["oracle_log_score_per_batch"] - r["old_log_score_per_batch"] for r in rr])
                    groups.append({"regime": regime, "arm": arm, "prefix": prefix,
                        "mean_log_score_gain": float(gain.mean()),
                        "paired_seed_bootstrap95": np.quantile(gain[indices].mean(axis=1), [.025, .975]).tolist(),
                        "better": int(np.sum(gain > 1e-12)), "worse": int(np.sum(gain < -1e-12)),
                        "seed_log_score_gains": gain.tolist()})
        result.update(complete=True, groups=groups)
    finally:
        result.update(cpu_seconds=time.process_time() - cpu_start, wall_seconds=time.monotonic() - start,
                      peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        (ARTIFACTS / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("complete", "cpu_seconds", "wall_seconds", "peak_rss_mb")}), flush=True)


if __name__ == "__main__":
    main()

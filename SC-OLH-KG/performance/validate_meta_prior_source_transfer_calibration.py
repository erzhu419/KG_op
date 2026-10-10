"""A24: pair-held calibration from cached matched and transfer forecasts."""
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
from performance.diagnose_meta_prior_source_cross_archive import block_contrast
from performance.function_covariance_attribution import posterior_mass
from performance.function_covariance_source import node_grid, outer_priors, predictive_scores
from performance.validate_meta_prior_function_covariance_source import profile_arrays
from performance.validate_meta_prior_scale_preserving_source_covariance import read

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_source_transfer_calibration_a24_20261003"
RULES = ("matched_only", "transfer_only", "balanced")


def calibrations(sums, grid):
    result = {}
    for arm in ("fixed_source", "risk_ard_mixture"):
        for rule in RULES:
            scores = {i: sums["matched", arm, i] if rule == "matched_only" else
                      sums["transfer", arm, i] if rule == "transfer_only" else
                      .5 * (sums["matched", arm, i] + sums["transfer", arm, i])
                      for i in range(8)}
            result[arm, rule] = outer_priors(scores, grid)
    return result


def main():
    start, wall = time.process_time(), time.monotonic()
    spec = read("performance/meta_prior_source_transfer_calibration_plan.json")
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def stop(signum, frame):
        raise TimeoutError("A24 local CPU cap reached")
    signal.signal(signal.SIGXCPU, stop)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    result = {"stage": spec["stage"], "complete": False, "rows": [],
              "new_samples": 0, "new_branch_forecasts": 0, "server_jobs": 0,
              "server_pulls": 0, "target_paths_scored": 0, "kg_calls": 0}
    try:
        pairs = spec["cohort"]["seed_pairs"]
        pair_of = {s: i for i, pair in enumerate(pairs) for s in pair}
        expected = {(s, g, h, a) for s in pair_of for g in ("shared", "separate_heads")
                    for h in range(8) for a in ("fixed_source", "risk_ard_mixture")}
        profiles = {c: read(spec["inputs"][i])["rows"] for i, c in enumerate(("matched", "transfer"))}
        previous = read(spec["inputs"][2])
        if not previous["complete"]:
            raise ValueError("A24 requires completed A23 evidence")
        old = {(r["scored_seed"], r["archive_group"], r["held_task"], r["arm"]): r for r in previous["rows"]}
        for condition, rows in profiles.items():
            keys = {(r["seed"], r["archive_group"], r["held_task"], r["arm"]) for r in rows}
            if len(rows) != 512 or keys != expected:
                raise ValueError("A24 cached cohort changed")
            for r in rows:
                if condition == "transfer" and (r["pair_id"] != pair_of[r["seed"]] or
                        r["applied_source_seed"] != next(s for s in pairs[r["pair_id"]] if s != r["seed"])):
                    raise ValueError("A24 transfer partner changed")
                e, p = profile_arrays(r, 32)
                if e.shape != (16, 4) or p.shape != (16, 4):
                    raise ValueError("A24 forecast chronology changed")
        grouped = {(c, i): [r for r in profiles[c] if pair_of[r["seed"]] == i]
                   for c in profiles for i in range(8)}
        fits, checks, scores_primary = {}, [], {}
        controls = {"max_generic_flat_theta_error": 0., "max_generic_1d_q_error": 0.,
                    "max_held_exclusion_error": 0., "max_pure_branch_error": 0.}
        for nodes in (16, 24, 32):
            grid = node_grid(nodes)
            sums = {}
            for i in range(8):
                block_start = time.process_time()
                for condition in profiles:
                    for arm in ("fixed_source", "risk_ard_mixture"):
                        selected = [r for r in grouped[condition, i] if r["arm"] == arm]
                        e, p = [np.concatenate([profile_arrays(r, 32)[k] for r in selected]) for k in (0, 1)]
                        sums[condition, arm, i] = predictive_scores(e, p, grid[0]).sum(axis=0)
                if nodes == 16 and i == 0:
                    first = time.process_time() - block_start
                    projection = time.process_time() - start + first * (8 * (1 + (24/16)**3 + (32/16)**3) - 1) * 1.25 + 20
                    result["admission"] = {"first_pair_cpu_seconds": first,
                        "projected_stage_cpu_seconds": projection, "cap_cpu_seconds": cap,
                        "admitted": projection < cap}
                    print(json.dumps({"admission": result["admission"]}), flush=True)
                    if projection >= cap:
                        raise TimeoutError("A24 projection exceeds frozen CPU cap")
            current = calibrations(sums, grid)
            if nodes == 16:
                fits = current
                # Replacing held data must not change its issued prior.
                changed = {k: v.copy() for k, v in sums.items()}
                for key in changed:
                    if key[2] == 0:
                        changed[key] += np.linspace(-100., 100., len(grid[1]))
                mutated = calibrations(changed, grid)
                controls["max_held_exclusion_error"] = max(float(np.max(np.abs(
                    np.array(fits[k][0]["macro_prior_mass"]) - mutated[k][0]["macro_prior_mass"]))) for k in fits)
                # The generic arm has three identical branches: integrate only q.
                q_nodes = grid[3][:nodes]
                for rule in RULES:
                    for i in range(8):
                        raw = sum(sums["matched", "fixed_source", j] if rule == "matched_only" else
                                  sums["transfer", "fixed_source", j] if rule == "transfer_only" else
                                  .5 * (sums["matched", "fixed_source", j] + sums["transfer", "fixed_source", j])
                                  for j in range(8) if j != i).reshape(nodes, nodes, nodes)
                        qlog = raw[0, 0] + logsumexp(grid[1].reshape(nodes, nodes, nodes), axis=(0, 1))
                        qmean = float(np.exp(qlog - logsumexp(qlog)) @ q_nodes)
                        fit = fits["fixed_source", rule][i]
                        controls["max_generic_1d_q_error"] = max(controls["max_generic_1d_q_error"], abs(qmean - fit["q_mean"]))
                        controls["max_generic_flat_theta_error"] = max(controls["max_generic_flat_theta_error"],
                            float(np.max(np.abs(np.array(fit["theta_mean"]) - 1/3))),
                            float(np.max(np.abs(np.array(fit["theta_sd"]) - np.sqrt(1/18)))))
            error = 0.
            for condition, rows in profiles.items():
                for r in rows:
                    key = (condition, r["seed"], r["archive_group"], r["held_task"], r["arm"])
                    e, p = profile_arrays(r, 32)
                    i = pair_of[r["seed"]]
                    current_scores = {rule: predictive_scores(e, p, current[r["arm"], rule][i]["macro_prior_mass"]) for rule in RULES}
                    if nodes == 16:
                        scores_primary[key] = current_scores
                        previous_r = old[key[1:]]
                        previous_scores = np.array(previous_r["matched_log_scores"] if condition == "matched" else previous_r["log_scores"])
                        for branch in range(4):
                            controls["max_pure_branch_error"] = max(controls["max_pure_branch_error"],
                                float(np.max(np.abs(predictive_scores(e, p, np.eye(4)[branch]) - p[:, branch]))))
                        metrics = {"balanced_minus_matched": float(np.mean(current_scores["balanced"] - current_scores["matched_only"])),
                                   "transfer_only_minus_matched": float(np.mean(current_scores["transfer_only"] - current_scores["matched_only"]))}
                        for rule in RULES:
                            metrics[rule + "_minus_i"] = float(np.mean(current_scores[rule] - p[:, 0]))
                            metrics[rule + "_minus_a20"] = float(np.mean(current_scores[rule] - previous_scores))
                        result["rows"].append({"condition": condition, "seed": r["seed"], "pair_id": i,
                            "archive_group": r["archive_group"], "held_task": r["held_task"], "arm": r["arm"], **metrics,
                            "posteriors": {rule: {"step9": posterior_mass(e, fits[r["arm"], rule][i]["macro_prior_mass"])[0].tolist(),
                                                  "step24": posterior_mass(e, fits[r["arm"], rule][i]["macro_prior_mass"])[-1].tolist()} for rule in RULES}})
                    else:
                        error = max(error, *(float(np.max(np.abs(current_scores[rule] - scores_primary[key][rule]))) for rule in RULES))
            if nodes != 16:
                checks.append({"macro_nodes_per_axis": nodes, "max_held_log_score_change": error})
            print(json.dumps({"macro_order_completed": nodes, "max_held_log_score_change": error}), flush=True)
        indices = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(8, size=(20000, 8))
        metrics = list(metrics)
        groups = []
        for condition in profiles:
            for group in ("shared", "separate_heads"):
                for arm in ("fixed_source", "risk_ard_mixture"):
                    selected = [r for r in result["rows"] if (r["condition"], r["archive_group"], r["arm"]) == (condition, group, arm)]
                    groups.append({"condition": condition, "archive_group": group, "arm": arm,
                        **{k: block_contrast([np.mean([r[k] for r in selected if r["pair_id"] == i]) for i in range(8)], indices) for k in metrics}})
        if max(controls.values()) > 1e-8 or max(c["max_held_log_score_change"] for c in checks) > 1e-4:
            raise ValueError("A24 exclusion, generic reduction or quadrature check failed")
        result.update(complete=True, groups=groups, controls=controls,
            numerical_check={"checks": checks, "passed": True, "primary_parameters_changed": False},
            calibrations=[{"arm": arm, "rule": rule, "held_pair_id": i, "held_seeds": pairs[i],
                "training_pair_ids": f["training_seeds"], **{k: v for k, v in f.items() if k not in ("held_seed", "training_seeds")}}
                for (arm, rule), values in fits.items() for i, f in values.items()],
            primary_macro_fits=48, saved_forecasts_scored=16384)
    except Exception as exc:
        result["failure"] = str(exc)
        raise
    finally:
        result.update(cpu_seconds=time.process_time() - start, wall_seconds=time.monotonic() - wall)
        (ARTIFACTS / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: result[k] for k in ("complete", "cpu_seconds")}), flush=True)


if __name__ == "__main__":
    main()

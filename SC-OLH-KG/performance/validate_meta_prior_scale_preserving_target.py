"""A21: frozen A20 prior, complete old target paths, no new decisions or searches."""
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
from performance.function_covariance_attribution import posterior_mass
from performance.function_covariance_source import predictive_scores
from performance.run_meta_prior_cross_head_risk_escape import contrast
from performance.run_meta_prior_mechanism import observe, task_truth
from performance.run_meta_prior_mechanism_v2 import prepare_sources
from performance.scale_preserving_source_covariance import selected_path_forecasts
from performance.source_calibrated_adaptation import fit_source_calibration, prior_adjusted_log_score
from performance.target_risk_residual import predictive_group_score
from performance.validate_meta_prior_function_covariance_source import calibrate, profile_arrays
from performance.validate_meta_prior_scale_preserving_source_covariance import covariance_controls, read
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior, head_batches

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_scale_preserving_target_validation_a21_20261003"


def key(row):
    return row["seed"], row["regime"], row["arm"]


def restore_risk_context(record, protocol):
    row, info = record["row"], record["source_info"]
    seed, regime = row["seed"], row["regime"]
    psi = np.random.default_rng(51000 + seed).uniform(-2., 2., (protocol["candidates"], protocol["psi_dim"]))
    reference = psi[:protocol["reference_bank_size"]]
    tasks, supports = prepare_sources(seed, regime, protocol)
    batches, noise = head_batches(tasks, 2), [t["noise"][:, 2] for t in tasks]
    full = SourceFunctionPrior.fit(batches, noise, reference, "fixed_source")
    tau = np.asarray(info[2]["source_function_variance"])
    signal_variance = np.mean([max(float(np.var(b.signals) - np.mean(n)), .02) for b, n in zip(batches, noise)])
    basis = FixedFunctionBasis(reference)
    structural = SourceFunctionPrior(basis, np.r_[np.mean([np.mean(b.signals) for b in batches]), np.zeros(len(tau))],
        np.diag(np.r_[.25, tau + .01 * signal_variance / len(tau)]), .02, {})
    selected = {h: info[i]["selected_indices"] for i, h in enumerate(("f", "g"))}
    target_seed = 61000 + 100 * seed + protocol["regimes"].index(regime)
    truth = task_truth(psi, supports, target_seed)
    truth[:, 1] += protocol["difficulty_offsets"][row["difficulty"]]
    return psi, full, structural, selected, truth, target_seed


def old_observations(row, truth, target_seed, protocol):
    ids = np.asarray(row["queries"])
    if len(ids) != 24 or row["queries"][:8] != row["initial_design"]:
        raise ValueError("A21 old path budget or initial design changed")
    visits = np.zeros(len(truth), dtype=int)
    labels, noises, prior_visits = [], [], []
    for candidate in ids:
        count = int(visits[candidate])
        y, n = observe(truth[candidate], target_seed, int(candidate), count, protocol["replicates"])
        labels.append(y[2])
        noises.append(n[2])
        prior_visits.append(count)
        visits[candidate] += 1
    return ids, np.asarray(labels), np.asarray(noises), prior_visits


def main():
    start, wall = time.process_time(), time.monotonic()
    spec = read("performance/meta_prior_scale_preserving_target_validation_plan.json")
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def stop(signum, frame):
        raise TimeoutError("A21 whole-stage local CPU cap reached")
    signal.signal(signal.SIGXCPU, stop)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    result = {"stage": spec["stage"], "complete": False, "rows": [], "controls": {},
        "new_eta_ard_refits": 0, "new_fg_selection_fits": 0, "server_jobs": 0, "server_pulls": 0,
        "new_simulation_samples": 0, "new_source_calls": 0, "new_target_calls": 0,
        "new_searches": 0, "kg_calls": 0, "new_random_verifications": 0, "candidate_recommendations": 0}
    profiles, contexts = [], []
    try:
        cohort = spec["cohort"]
        seeds, regimes, arms = cohort["seeds"], cohort["regimes"], cohort["priors"]
        source = read(spec["inputs"][1])
        source_gate = read("paper_artifacts/meta_prior_scale_preserving_source_covariance_a20_20261003/decision.json")
        execution = read(spec["inputs"][3])
        frozen = execution["base_protocol"]
        protocol = {**frozen, **frozen["allocations"][cohort["allocation"]]}
        records = read(spec["inputs"][4])["records"]
        old10, old11, old12 = [read(p) for p in spec["inputs"][5:8]]
        if not all(d["complete"] for d in (source, old10, old11, old12)) or not source_gate["proceed_to_fixed_target_validation"]:
            raise ValueError("A21 requires the passed complete source gate and saved target controls")
        expected = {(s, r, a) for s in seeds for r in regimes for a in arms}
        indexed = {key(r["row"]): r for r in records}
        original, a12 = [{key(r): r for r in d["rows"]} for d in (old10, old12)]
        if any(len(d) != 160 or set(d) != expected for d in (indexed, original, a12)) or len(records) != 160 or any(
                r["row"]["difficulty"] != cohort["difficulty"] or r["row"]["allocation"] != cohort["allocation"] for r in records):
            raise ValueError("A21 target cohort changed")
        fits = {(f["arm"], f["held_seed"]): f for f in source["calibrations"]}
        if set(fits) != {(a, s) for a in arms for s in seeds} or any(
                f["training_seeds"] != [s for s in seeds if s != f["held_seed"]] for f in fits.values()):
            raise ValueError("A21 source prior includes the held target seed")
        baseline = {(a, held): fit_source_calibration([f for row in old11["rows"] if row["arm"] == a and row["seed"] != held
                    for f in row["forecasts"]], 64) for a in arms for held in seeds}
        result["same_pool_ir_calibrations"] = [{"arm": a, "held_seed": s, **b} for (a, s), b in baseline.items()]
        controls = {"max_original_i_error": 0., "max_saved_r_i_evidence_error": 0.,
                    "max_generic_same_pool_ir_error": 0., "max_generic_q_error": 0.}
        for seed in seeds:
            for regime in regimes:
                path_start = time.process_time()
                learned_record = indexed[seed, regime, "risk_ard_mixture"]
                psi, full, structural, selected, truth, target_seed = restore_risk_context(learned_record, protocol)
                for name, value in covariance_controls([structural, full], selected).items():
                    controls[name] = max(controls.get(name, 0.), value)
                contexts.append({"seed": seed, "regime": regime, "fg_selected_indices": selected,
                    "eta_prior_components": [{"intercept_mean": float(p.mean[0]), "coefficient_variance": np.diag(p.covariance).tolist(),
                        "discrepancy_variance": p.discrepancy_variance} for p in (structural, full)]})
                for arm in ("risk_ard_mixture", "fixed_source"):
                    row = indexed[seed, regime, arm]["row"]
                    ids, labels, noise, visits = old_observations(row, truth, target_seed, protocol)
                    old = original[seed, regime, arm]["forecasts"]
                    identity = [(t + 1, int(ids[t])) for t in range(8, 24)]
                    if any([(f["step"], f["candidate"]) for f in fs] != identity for fs in (old, a12[seed, regime, arm]["forecasts"])) or any(
                            f["new_candidate"] != (visits[t] == 0) for t, f in zip(range(8, 24), old)):
                        raise ValueError("A21 saved candidate/visit chronology changed")
                    ratios = [float(np.log1p(-f["a9_null_posterior"]) - np.log(f["a9_null_posterior"])) for f in old]
                    by_rho = {}
                    if arm == "risk_ard_mixture":
                        for rho in (16, 32, 48):
                            fs = selected_path_forecasts([structural, full], psi, ids, labels, noise, selected["f"], rho)
                            gs = selected_path_forecasts([structural, full], psi, ids, labels, noise, selected["g"], rho)
                            error = max(abs(f["source"] - prev["source"]) for ff in (fs, gs) for f, prev in zip(ff, old))
                            controls["max_original_i_error"] = max(controls["max_original_i_error"], error)
                            if error > 1e-8:
                                raise ValueError("A21 reconstructed learned I forecast changed")
                            by_rho[str(rho)] = {
                                "relative_log_evidence": [[0., f["relative_log_evidence"], g["relative_log_evidence"], ratio] for f, g, ratio in zip(fs, gs, ratios)],
                                "component_log_density": [[o["source"], f["adapted"], g["adapted"], o["adapted"]] for f, g, o in zip(fs, gs, old)]}
                    else:
                        for t, previous in zip(range(8, 24), old):
                            score = predictive_group_score([full], psi, ids[:t], labels[:t], noise[:t], 0., int(ids[t]), labels[t:t + 1], noise[t:t + 1])
                            controls["max_original_i_error"] = max(controls["max_original_i_error"], abs(score - previous["source"]))
                            if abs(score - previous["source"]) > 1e-8:
                                raise ValueError("A21 reconstructed generic I forecast changed")
                        data = {"relative_log_evidence": [[0., 0., 0., ratio] for ratio in ratios],
                                "component_log_density": [[o["source"]] * 3 + [o["adapted"]] for o in old]}
                        by_rho = {str(rho): data for rho in (16, 32, 48)}
                    profile = {"seed": seed, "regime": regime, "arm": arm, "scores_by_rho": by_rho}
                    profiles.append(profile)
                    e, p = profile_arrays(profile, 32)
                    for predictive, o in zip(p, old):
                        reconstructed = float(prior_adjusted_log_score(predictive[0], predictive[3], o["a9_null_posterior"], .5))
                        controls["max_saved_r_i_evidence_error"] = max(controls["max_saved_r_i_evidence_error"], abs(reconstructed - o["a9"]))
                    if controls["max_saved_r_i_evidence_error"] > 1e-8:
                        raise ValueError("A21 retained R/I evidence does not reconstruct A9")
                    prior = fits[arm, seed]["macro_prior_mass"]
                    scores = predictive_scores(e, p, prior)
                    ir_scores = np.array([prior_adjusted_log_score(o["source"], o["adapted"], o["a9_null_posterior"], baseline[arm, seed]["q_mean"]) for o in old])
                    a12_scores = np.array([f["calibrated_log_score"] for f in a12[seed, regime, arm]["forecasts"]])
                    if arm == "fixed_source":
                        controls["max_generic_q_error"] = max(controls["max_generic_q_error"], abs(fits[arm, seed]["q_mean"] - baseline[arm, seed]["q_mean"]))
                        controls["max_generic_same_pool_ir_error"] = max(controls["max_generic_same_pool_ir_error"], float(np.max(np.abs(scores - ir_scores))))
                    posterior = posterior_mass(e, prior)
                    result["rows"].append({"seed": seed, "regime": regime, "arm": arm,
                        "source_macro_prior_mass": prior,
                        "forecasts": [{"step": step, "candidate": candidate, "prior_visits": visits[step - 1],
                            "relative_past_log_evidence": e[i].tolist(), "component_log_density": p[i].tolist(),
                            "posterior_macro_mass": posterior[i].tolist(), "log_score": float(scores[i]),
                            "same_pool_ir_score": float(ir_scores[i]), "a12_score": float(a12_scores[i])}
                            for i, (step, candidate) in enumerate(identity)],
                        "mean_gain": float(np.mean(scores - p[:, 0])), "mean_gain_vs_same_pool_ir": float(np.mean(scores - ir_scores)),
                        "mean_gain_vs_a12": float(np.mean(scores - a12_scores)),
                        "pure_f_gain": float(np.mean(p[:, 1] - p[:, 0])), "pure_g_gain": float(np.mean(p[:, 2] - p[:, 0]))})
                    if len(profiles) == 1:
                        first = time.process_time() - path_start
                        projection = time.process_time() - start + first * 79 * 1.25 + 40.
                        result["admission"] = {"retained_first_complete_learned_path_cpu_seconds": first,
                            "projected_stage_cpu_seconds": projection, "calibration_and_summary_cpu_reserve": 40.,
                            "safety_multiplier": 1.25, "cap_cpu_seconds": cap, "admitted": projection < cap}
                        print(json.dumps({"admission": result["admission"]}), flush=True)
                        if projection >= cap:
                            raise TimeoutError("A21 complete-cohort projection exceeds CPU cap")
            print(json.dumps({"target_paths_scored": len(profiles)}), flush=True)
        result["controls"] = controls
        if max(controls.values()) > 1e-8:
            raise ValueError("A21 original model or generic calibration control failed")
        primary = {key(r): np.array([f["log_score"] for f in r["forecasts"]]) for r in result["rows"]}
        checks = []
        for rho in (16, 48):
            error = max(float(np.max(np.abs(predictive_scores(*profile_arrays(p, rho), fits[p["arm"], p["seed"]]["macro_prior_mass"]) - primary[key(p)]))) for p in profiles)
            checks.append({"macro_nodes_per_axis": 16, "rho_nodes": rho, "max_target_log_score_change": error})
        source_profiles = read(spec["inputs"][2])["rows"]
        result["alternative_source_calibrations"] = {}
        for nodes in (24, 32):
            alternate = calibrate(source_profiles, nodes, 32)
            result["alternative_source_calibrations"][str(nodes)] = [{"arm": arm, **fit} for arm, ff in alternate.items() for fit in ff.values()]
            error = max(float(np.max(np.abs(predictive_scores(*profile_arrays(p, 32), alternate[p["arm"]][p["seed"]]["macro_prior_mass"]) - primary[key(p)]))) for p in profiles)
            checks.append({"macro_nodes_per_axis": nodes, "rho_nodes": 32, "max_target_log_score_change": error})
            print(json.dumps(checks[-1]), flush=True)
        passed = max(c["max_target_log_score_change"] for c in checks) <= 1e-4
        result["numerical_check"] = {"primary_macro_order": 16, "primary_rho_order": 32, "checks": checks,
            "tolerance": 1e-4, "passed": passed, "primary_parameters_changed": False}
        if not passed:
            raise ValueError("A21 target quadrature confirmation failed")
        indices = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(16, size=(20000, 16))
        metrics = ("mean_gain", "mean_gain_vs_same_pool_ir", "mean_gain_vs_a12", "pure_f_gain", "pure_g_gain")
        groups = []
        for regime in regimes:
            for arm in arms:
                rows = sorted((r for r in result["rows"] if (r["regime"], r["arm"]) == (regime, arm)), key=lambda r: r["seed"])
                groups.append({"regime": regime, "arm": arm,
                    **{m: contrast([r[m] for r in rows], indices) for m in metrics},
                    "mean_step9_macro_mass": np.mean([r["forecasts"][0]["posterior_macro_mass"] for r in rows], axis=0).tolist(),
                    "mean_step24_macro_mass": np.mean([r["forecasts"][-1]["posterior_macro_mass"] for r in rows], axis=0).tolist()})
        result.update(complete=True, groups=groups, target_paths_scored=len(profiles), target_forecasts=2560,
            full_eta_prior_refits=len(contexts), historical_source_information=source["historical_source_information"],
            interpretation="Complete retrospective development test on original query paths; no candidate decision or new-policy outcome.")
    except Exception as exc:
        result["failure"] = str(exc)
        raise
    finally:
        result.update(cpu_seconds=time.process_time() - start, wall_seconds=time.monotonic() - wall,
            current_process_vm_hwm_mb=float(next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("VmHWM:"))) / 1024)
        (ARTIFACTS / "target_forecasts.json.gz").write_bytes(gzip.compress(json.dumps({"rows": profiles}, allow_nan=False).encode()))
        (ARTIFACTS / "source_contexts.json.gz").write_bytes(gzip.compress(json.dumps({"rows": contexts}, allow_nan=False).encode()))
        (ARTIFACTS / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: result[k] for k in ("complete", "cpu_seconds", "current_process_vm_hwm_mb")}), flush=True)


if __name__ == "__main__":
    main()

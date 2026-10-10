"""A23: fixed neighboring source pairs, seven-task priors and source-only transfer."""
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
from performance.run_meta_prior_mechanism_v2 import prepare_sources
from performance.source_cross_archive_transfer import fold_forecasts
from performance.validate_meta_prior_function_covariance_source import profile_arrays
from performance.validate_meta_prior_scale_preserving_source_covariance import covariance_controls, read
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior, head_batches

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_source_cross_archive_transfer_a23_20261003"


def block_contrast(values, indices):
    result = contrast(values, indices)
    result["pair_block_values"] = result.pop("seed_gains")
    return result


def main():
    start, wall = time.process_time(), time.monotonic()
    spec = read("performance/meta_prior_source_cross_archive_transfer_plan.json")
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def stop(signum, frame):
        raise TimeoutError("A23 whole-stage local CPU cap reached")
    signal.signal(signal.SIGXCPU, stop)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    result = {"stage": spec["stage"], "complete": False, "rows": [], "folds": [],
        "controls": {"max_matched_reproduction_error": 0., "max_matched_a9_evidence_error": 0.,
                     "max_pure_branch_error": 0., "max_generic_ir_error": 0.},
        "new_eta_ard_refits": 0, "new_fg_selection_fits": 0, "new_calibration_fits": 0,
        "new_simulation_samples": 0, "new_source_calls": 0, "server_jobs": 0, "server_pulls": 0,
        "target_paths_scored": 0, "candidate_recommendations": 0, "kg_calls": 0,
        "new_searches": 0, "new_random_verifications": 0}
    profiles, full_contexts = [], []
    numerical = {str(n): {"max_mixture_score_change": 0., "max_pure_branch_score_change": 0.} for n in (16, 48)}
    try:
        cohort = spec["cohort"]
        seeds, groups, pairs = cohort["seeds"], cohort["archive_groups"], cohort["seed_pairs"]
        contexts = read(spec["inputs"][0])["rows"]
        source = read(spec["inputs"][2])
        matched_profiles = read(spec["inputs"][3])["rows"]
        source11 = read(spec["inputs"][4])
        frozen = read(spec["inputs"][5])["base_protocol"]
        families = read(spec["inputs"][6])["public_families"]
        expected_folds = {(s, g, h) for s in seeds for g in groups for h in range(8)}
        expected_profiles = {(*f, a) for f in expected_folds for a in cohort["prior_arms"]}
        ctx = {(c["seed"], c["archive_group"], c["held_task"]): c for c in contexts}
        matched = {(r["seed"], r["archive_group"], r["held_task"], r["arm"]): r for r in matched_profiles}
        source_forecasts = {(r["seed"], r["archive_group"], r["held_task"], r["arm"]): r for r in source11["rows"]}
        if not source["complete"] or not source11["complete"] or len(ctx) != 256 or len(contexts) != 256 or set(ctx) != expected_folds or any(
                len(d) != 512 or set(d) != expected_profiles for d in (matched, source_forecasts)):
            raise ValueError("A23 source cohort or recovered contexts changed")
        if sorted(s for pair in pairs for s in pair) != seeds or len(pairs) != 8 or any(len(p) != 2 or p[0] == p[1] for p in pairs):
            raise ValueError("A23 seed pairs no longer partition the fixed source cohort")
        fits = {(f["arm"], f["held_seed"]): f for f in source["calibrations"]}
        if set(fits) != {(a, s) for a in cohort["prior_arms"] for s in seeds} or any(
                f["training_seeds"] != [s for s in seeds if s != f["held_seed"]] for f in fits.values()):
            raise ValueError("A23 scored source seed entered macro calibration")
        for pair_id, pair in enumerate(pairs):
            for group in groups:
                generation_start = time.process_time()
                tasks = {s: prepare_sources(s, "shared_structure" if group == "shared" else "separate_head_structure", frozen)[0] for s in pair}
                references = {s: np.random.default_rng(51000 + s).uniform(-2., 2., (frozen["reference_bank_size"], frozen["psi_dim"])) for s in pair}
                for source_seed, scored_seed in (tuple(pair), tuple(reversed(pair))):
                    basis = FixedFunctionBasis(references[source_seed])
                    for held in range(8):
                        fold_start = generation_start if not result["folds"] else time.process_time()
                        saved = ctx[source_seed, group, held]
                        training_ids = [i for i in range(8) if i != held]
                        if saved["training_tasks"] != training_ids or saved["source_label_count"] != 448:
                            raise ValueError("A23 seven-task source conditioning changed")
                        training = [tasks[source_seed][i] for i in training_ids]
                        full = SourceFunctionPrior.fit(head_batches(training, 2), [t["noise"][:, 2] for t in training], references[source_seed], "fixed_source")
                        structural = SourceFunctionPrior(basis, np.r_[saved["eta_intercept_mean"], np.zeros(basis.feature_dim)],
                            np.diag([.25] + saved["eta_prior_function_variance"]), saved["eta_discrepancy_variance"], {})
                        selected = saved["fg_selected_indices"]
                        for name, value in covariance_controls([structural, full], selected).items():
                            result["controls"][name] = max(result["controls"].get(name, 0.), value)
                        full_contexts.append({"seed": source_seed, "archive_group": group, "held_task": held,
                            "intercept_mean": float(full.mean[0]), "coefficient_variance": np.diag(full.covariance).tolist(),
                            "discrepancy_variance": full.discrepancy_variance})
                        if not result["folds"]:
                            reproduced = fold_forecasts([structural, full], tasks[source_seed][held], selected, families, (32,))
                            for arm in cohort["prior_arms"]:
                                old_key = source_seed, group, held, arm
                                old_e, old_p = profile_arrays(matched[old_key], 32)
                                new = reproduced[arm]["32"]
                                error = max(float(np.max(np.abs(np.asarray(new["relative_log_evidence"]) - old_e))),
                                            float(np.max(np.abs(np.asarray(new["component_log_density"]) - old_p))))
                                result["controls"]["max_matched_reproduction_error"] = max(result["controls"]["max_matched_reproduction_error"], error)
                                old11 = source_forecasts[old_key]["forecasts"]
                                ratio = np.array([np.log1p(-f["null_posterior"]) - np.log(f["null_posterior"]) for f in old11])
                                result["controls"]["max_matched_a9_evidence_error"] = max(result["controls"]["max_matched_a9_evidence_error"], float(np.max(np.abs(np.array(new["relative_log_evidence"])[:, 3] - ratio))))
                            if max(result["controls"].values()) > 1e-8:
                                raise ValueError("A23 matched-fold reproduction failed before transfer")
                        transfer = fold_forecasts([structural, full], tasks[scored_seed][held], selected, families)
                        for arm in cohort["prior_arms"]:
                            old_key = scored_seed, group, held, arm
                            old_forecasts = source_forecasts[old_key]["forecasts"]
                            if [(f["step"], f["candidate"]) for f in old_forecasts] != [(t + 1, t) for t in range(8, 24)]:
                                raise ValueError("A23 scored source function chronology changed")
                            profile = {"applied_source_seed": source_seed, "seed": scored_seed, "pair_id": pair_id,
                                "archive_group": group, "held_task": held, "arm": arm, "scores_by_rho": transfer[arm]}
                            profiles.append(profile)
                            e, p = profile_arrays(profile, 32)
                            old_e, old_p = profile_arrays(matched[old_key], 32)
                            prior = fits[arm, scored_seed]["macro_prior_mass"]
                            score = predictive_scores(e, p, prior)
                            old_score = predictive_scores(old_e, old_p, prior)
                            posterior = posterior_mass(e, prior)
                            for branch in range(4):
                                err = float(np.max(np.abs(predictive_scores(e, p, np.eye(4)[branch]) - p[:, branch])))
                                result["controls"]["max_pure_branch_error"] = max(result["controls"]["max_pure_branch_error"], err)
                            if arm == "fixed_source":
                                q = prior[3]
                                err = float(np.max(np.abs(score - predictive_scores(e, p, [1. - q, 0., 0., q]))))
                                result["controls"]["max_generic_ir_error"] = max(result["controls"]["max_generic_ir_error"], err)
                            for nodes in (16, 48):
                                ee, pp = profile_arrays(profile, nodes)
                                change = numerical[str(nodes)]
                                change["max_pure_branch_score_change"] = max(change["max_pure_branch_score_change"], float(np.max(np.abs(pp - p))))
                                change["max_mixture_score_change"] = max(change["max_mixture_score_change"], float(np.max(np.abs(predictive_scores(ee, pp, prior) - score))))
                            result["rows"].append({"source_seed": source_seed, "scored_seed": scored_seed, "pair_id": pair_id,
                                "archive_group": group, "held_task": held, "arm": arm, "q": prior[3],
                                "log_scores": score.tolist(), "matched_log_scores": old_score.tolist(),
                                "pure_r_gain": float(np.mean(p[:, 3] - p[:, 0])), "matched_pure_r_gain": float(np.mean(old_p[:, 3] - old_p[:, 0])),
                                "pure_r_gain_change": float(np.mean((p[:, 3] - p[:, 0]) - (old_p[:, 3] - old_p[:, 0]))),
                                "mixture_gain": float(np.mean(score - p[:, 0])), "matched_mixture_gain": float(np.mean(old_score - old_p[:, 0])),
                                "mixture_gain_change": float(np.mean((score - p[:, 0]) - (old_score - old_p[:, 0]))),
                                "i_transfer_loss": float(np.mean(p[:, 0] - old_p[:, 0])), "mixture_transfer_loss": float(np.mean(score - old_score)),
                                "pure_f_gain": float(np.mean(p[:, 1] - p[:, 0])), "pure_g_gain": float(np.mean(p[:, 2] - p[:, 0])),
                                "step9_posterior_mass": posterior[0].tolist(), "step24_posterior_mass": posterior[-1].tolist()})
                        result["folds"].append({"source_seed": source_seed, "scored_seed": scored_seed, "archive_group": group,
                            "held_task": held, "training_tasks": training_ids, "source_labels": 448,
                            "covariance_context_reused": True, "fold_cpu_seconds": time.process_time() - fold_start})
                        if len(result["folds"]) == 1:
                            first = time.process_time() - fold_start
                            projection = time.process_time() - start + first * 255 * 1.25 + 20.
                            result["admission"] = {"retained_first_complete_fold_cpu_seconds": first,
                                "projected_stage_cpu_seconds": projection, "summary_cpu_reserve": 20.,
                                "safety_multiplier": 1.25, "cap_cpu_seconds": cap, "admitted": projection < cap}
                            print(json.dumps({"admission": result["admission"]}), flush=True)
                            if projection >= cap:
                                raise TimeoutError("A23 complete-cohort projection exceeds CPU cap")
            print(json.dumps({"transfer_folds_completed": len(result["folds"])}), flush=True)
        result["numerical_check"] = {"primary_order": 32, "checks": numerical, "tolerance": 1e-4,
            "passed": max(v for c in numerical.values() for v in c.values()) <= 1e-4, "primary_parameters_changed": False}
        if max(result["controls"].values()) > 1e-8 or not result["numerical_check"]["passed"]:
            raise ValueError("A23 transfer control or quadrature check failed")
        metrics = ("pure_r_gain", "matched_pure_r_gain", "pure_r_gain_change", "mixture_gain", "matched_mixture_gain", "mixture_gain_change",
                   "i_transfer_loss", "mixture_transfer_loss", "pure_f_gain", "pure_g_gain")
        indices = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(8, size=(20000, 8))
        summaries = []
        for group in groups:
            for arm in cohort["prior_arms"]:
                directions = [{"scored_seed": s, **{m: float(np.mean([r[m] for r in result["rows"] if (r["scored_seed"], r["archive_group"], r["arm"]) == (s, group, arm)])) for m in metrics}} for s in seeds]
                blocks = {m: [float(np.mean([d[m] for d in directions if d["scored_seed"] in pair])) for pair in pairs] for m in metrics}
                rr = [r for r in result["rows"] if (r["archive_group"], r["arm"]) == (group, arm)]
                summaries.append({"archive_group": group, "arm": arm, "analysis_units": "eight disjoint seed-pair means", "pair_blocks": 8,
                    **{m: block_contrast(v, indices) for m, v in blocks.items()},
                    "direction_values": directions, "mean_q": float(np.mean([r["q"] for r in rr])),
                    "mean_step9_macro_mass": np.mean([r["step9_posterior_mass"] for r in rr], axis=0).tolist(),
                    "mean_step24_macro_mass": np.mean([r["step24_posterior_mass"] for r in rr], axis=0).tolist()})
        result.update(complete=True, groups=summaries, transfer_whole_function_folds=len(result["folds"]),
            transfer_prior_profiles=len(profiles), transfer_forecasts=8192, full_eta_prior_refits=len(full_contexts),
            historical_source_information=source["historical_source_information"],
            interpretation="Source-only fixed-pair transfer assay; no target outcomes or new calibration fits.")
    except Exception as exc:
        result["failure"] = str(exc)
        raise
    finally:
        result.update(cpu_seconds=time.process_time() - start, wall_seconds=time.monotonic() - wall,
            current_process_vm_hwm_mb=float(next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("VmHWM:"))) / 1024)
        (ARTIFACTS / "transfer_forecasts.json.gz").write_bytes(gzip.compress(json.dumps({"rows": profiles}, allow_nan=False).encode()))
        (ARTIFACTS / "full_source_contexts.json.gz").write_bytes(gzip.compress(json.dumps({"rows": full_contexts}, allow_nan=False).encode()))
        (ARTIFACTS / "diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: result[k] for k in ("complete", "cpu_seconds", "current_process_vm_hwm_mb")}), flush=True)


if __name__ == "__main__":
    main()

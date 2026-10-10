"""A25: held-pair empirical covariance meta-prior, with cached paid labels."""
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
from performance.diagnose_meta_prior_source_cross_archive import block_contrast
from performance.run_meta_prior_mechanism_v2 import prepare_sources
from performance.validate_meta_prior_function_covariance_source import profile_arrays
from performance.validate_meta_prior_scale_preserving_source_covariance import read
from representation.archive_covariance_prior import ArchiveCovariancePrior, mixture_forecasts, prequential_components
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_archive_covariance_a25_20261003"


def local_priors(context, full, basis):
    return [SourceFunctionPrior(basis, np.r_[context["eta_intercept_mean"], np.zeros(basis.feature_dim)],
                np.diag([.25] + context["eta_prior_function_variance"]), context["eta_discrepancy_variance"], {}),
            SourceFunctionPrior(basis, np.r_[full["intercept_mean"], np.zeros(basis.feature_dim)],
                np.diag(full["coefficient_variance"]), full["discrepancy_variance"], {})]


def main():
    start, wall = time.process_time(), time.monotonic()
    spec = read("performance/meta_prior_archive_covariance_plan.json")
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def stop(signum, frame):
        raise TimeoutError("A25 local CPU cap reached")
    signal.signal(signal.SIGXCPU, stop)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    result = {"stage": spec["stage"], "complete": False, "rows": [], "population_folds": [],
        "new_eta_ard_refits": 0, "new_source_calls": 0, "new_target_calls": 0,
        "server_jobs": 0, "server_pulls": 0, "target_paths_scored": 0,
        "kg_calls": 0, "candidate_recommendations": 0}
    profiles = []
    try:
        contexts = read(spec["inputs"][0])["rows"]
        full = read(spec["inputs"][1])["rows"]
        ctx = {(c["seed"], c["archive_group"], c["held_task"]): c for c in contexts}
        ff = {(c["seed"], c["archive_group"], c["held_task"]): c for c in full}
        saved = {condition: {(r["seed"], r["archive_group"], r["held_task"], r["arm"]): r
                            for r in read(spec["inputs"][i])["rows"]}
                 for i, condition in ((2, "matched"), (3, "transfer"))}
        frozen = read(read(spec["inputs"][4])["inputs"][5])["base_protocol"]
        if not read(spec["inputs"][5])["proceed_to_covariance_meta_prior_source_validation"]:
            raise ValueError("A25 requires the A24 covariance-study decision")
        pairs, groups = spec["cohort"]["seed_pairs"], spec["cohort"]["archive_groups"]
        seeds = [s for pair in pairs for s in pair]
        expected = {(s, g, h) for s in seeds for g in groups for h in range(8)}
        expected_profiles = {(*key, a) for key in expected for a in ("fixed_source", "risk_ard_mixture")}
        if len(contexts) != 256 or len(full) != 256 or set(ctx) != expected or set(ff) != expected or any(
                set(d) != expected_profiles for d in saved.values()):
            raise ValueError("A25 source cohort changed")
        bases = {s: FixedFunctionBasis(np.random.default_rng(51000+s).uniform(-2, 2,
                    (frozen["reference_bank_size"], frozen["psi_dim"]))) for s in seeds}
        controls = {"max_original_i_error": 0., "max_original_full_error": 0.}
        for pair_id, pair in enumerate(pairs):
            populations = {h: ArchiveCovariancePrior.from_contexts(contexts, bases, pair, h) for h in range(8)}
            for h, population in populations.items():
                keys = {(s, g) for s in seeds if s not in pair for g in groups}
                if len(population.components) != 28 or set(population.archive_keys) != keys or any(
                        c["training_tasks"] != [i for i in range(8) if i != h]
                        for c in contexts if c["held_task"] == h):
                    raise ValueError("A25 population included a held archive or changed source task conditioning")
                result["population_folds"].append({"pair_id": pair_id, "held_task": h,
                    "excluded_seeds": pair, "population_archives": population.archive_keys,
                    "source_eta_labels": 28 * 448})
            for scored_seed in pair:
                partner = next(s for s in pair if s != scored_seed)
                for group in groups:
                    block_start = time.process_time()
                    tasks = prepare_sources(scored_seed, "shared_structure" if group == "shared" else "separate_head_structure", frozen)[0]
                    for held, task in enumerate(tasks):
                        path_start = block_start if not profiles else time.process_time()
                        population = populations[held]
                        priors = population.components + local_priors(ctx[scored_seed, group, held], ff[scored_seed, group, held], bases[scored_seed]) + local_priors(ctx[partner, group, held], ff[partner, group, held], bases[partner])
                        output = prequential_components(priors, task["psi"], np.arange(24),
                            task["response"][:24, 2], task["noise"][:24, 2])
                        e, p = output["log_evidence"][8:24], output["log_density"][8:24]
                        matched_i, _ = mixture_forecasts(e[:, 28:30], p[:, 28:30], [.5, .5])
                        for condition, local in (("matched", [28, 29]), ("transfer", [30, 31])):
                            columns = list(range(28)) + local
                            ce, cp = e[:, columns], p[:, columns]
                            weights = {"I": np.r_[np.zeros(28), .5, .5],
                                       "H": np.r_[population.weights * .25, .25, .5],
                                       "M": np.r_[population.weights * .5, 0., .5],
                                       "P": np.r_[population.weights, 0., 0.],
                                       "full": np.r_[np.zeros(29), 1.]}
                            scores, posteriors = {}, {}
                            for name, w in weights.items():
                                scores[name], posteriors[name] = mixture_forecasts(ce, cp, w)
                            _, old_p = profile_arrays(saved[condition][scored_seed, group, held, "risk_ard_mixture"], 32)
                            _, old_full = profile_arrays(saved[condition][scored_seed, group, held, "fixed_source"], 32)
                            controls["max_original_i_error"] = max(controls["max_original_i_error"], float(np.max(np.abs(scores["I"] - old_p[:, 0]))))
                            controls["max_original_full_error"] = max(controls["max_original_full_error"], float(np.max(np.abs(scores["full"] - old_full[:, 0]))))
                            if max(controls.values()) > 1e-8:
                                raise ValueError("A25 original I or full-dictionary forecast changed")
                            metrics = {"h_minus_i": float(np.mean(scores["H"] - scores["I"])),
                                "h_minus_matched_i": float(np.mean(scores["H"] - matched_i)),
                                "m_minus_i": float(np.mean(scores["M"] - scores["I"])),
                                "p_minus_i": float(np.mean(scores["P"] - scores["I"])),
                                "h_minus_full": float(np.mean(scores["H"] - scores["full"])),
                                "i_minus_matched_i": float(np.mean(scores["I"] - matched_i))}
                            result["rows"].append({"seed": scored_seed, "pair_id": pair_id, "archive_group": group,
                                "held_task": held, "condition": condition, **metrics,
                                "step9_h_mass": [float(posteriors["H"][0, :28].sum()), *posteriors["H"][0, 28:].tolist()],
                                "step24_h_mass": [float(posteriors["H"][-1, :28].sum()), *posteriors["H"][-1, 28:].tolist()]})
                            profiles.append({"seed": scored_seed, "pair_id": pair_id, "archive_group": group,
                                "held_task": held, "condition": condition,
                                "log_evidence": ce.tolist(), "log_density": cp.tolist(),
                                "scores": {k: v.tolist() for k, v in scores.items()}})
                        if len(profiles) == 2:
                            first = time.process_time() - path_start
                            projection = time.process_time() - start + first * 255 * 1.25 + 15
                            result["admission"] = {"first_path_cpu_seconds": first,
                                "projected_stage_cpu_seconds": projection, "cap_cpu_seconds": cap,
                                "admitted": projection < cap}
                            print(json.dumps({"admission": result["admission"]}), flush=True)
                            if projection >= cap:
                                raise TimeoutError("A25 projection exceeds frozen CPU cap")
            print(json.dumps({"source_paths_completed": len(profiles)//2}), flush=True)
        indices = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(8, size=(20000, 8))
        summaries = []
        for condition in spec["cohort"]["conditions"]:
            for group in groups:
                selected = [r for r in result["rows"] if (r["condition"], r["archive_group"]) == (condition, group)]
                summaries.append({"condition": condition, "archive_group": group,
                    **{k: block_contrast([np.mean([r[k] for r in selected if r["pair_id"] == i]) for i in range(8)], indices) for k in metrics},
                    "step9_h_mass": np.mean([r["step9_h_mass"] for r in selected], axis=0).tolist(),
                    "step24_h_mass": np.mean([r["step24_h_mass"] for r in selected], axis=0).tolist()})
        result.update(complete=True, groups=summaries, controls=controls,
            source_paths=256, scored_profiles=512, scored_forecasts=8192,
            posterior_mass_order=["population_structure", "local_structure", "full_dictionary"])
    except Exception as exc:
        result["failure"] = str(exc)
        raise
    finally:
        result.update(cpu_seconds=time.process_time()-start, wall_seconds=time.monotonic()-wall)
        (ARTIFACTS / "source_forecasts.json.gz").write_bytes(gzip.compress(json.dumps({"rows": profiles}, allow_nan=False).encode()))
        (ARTIFACTS / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        print(json.dumps({k: result[k] for k in ("complete", "cpu_seconds")}), flush=True)


if __name__ == "__main__":
    main()

"""A30: uniform sharing reference, source-only held-pair validation."""
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
from performance.validate_meta_prior_archive_covariance import local_priors
from performance.validate_meta_prior_scale_preserving_source_covariance import read
from representation.archive_covariance_prior import (
    mixture_forecasts, prequential_components, sharing_branches, sharing_forecasts,
)
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior, head_batches

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_conditional_sharing_a30_20261003"


def main():
    start, wall = time.process_time(), time.monotonic()
    spec = read("performance/meta_prior_conditional_sharing_plan.json")
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap+2))
    def stop(signum, frame):
        raise TimeoutError("A30 local CPU cap reached")
    signal.signal(signal.SIGXCPU, stop)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2)+"\n")
    result = {"stage": spec["stage"], "complete": False, "rows": [],
        "new_eta_ard_refits": 0, "new_fg_selection_fits": 0, "cheap_fg_moment_fits": 0,
        "new_source_calls": 0, "new_target_calls": 0, "server_jobs": 0, "server_pulls": 0,
        "target_paths_scored": 0, "kg_calls": 0, "recommendations": 0}
    raw, profiles, head_contexts = [], [], []
    try:
        executed = read(spec["inputs"][0])
        if not read(spec["inputs"][3])["complete"]:
            raise ValueError("A30 requires completed A29 evidence")
        contexts, full = [read(p)["rows"] for p in spec["inputs"][1:3]]
        ctx = {(c["seed"], c["archive_group"], c["held_task"]): c for c in contexts}
        ff = {(c["seed"], c["archive_group"], c["held_task"]): c for c in full}
        old_source = read("paper_artifacts/meta_prior_joint_archive_a27_20261003/validation.json")
        old = {(r["seed"], r["archive_group"], r["held_task"], r["condition"]): r
               for r in read("paper_artifacts/meta_prior_joint_archive_a27_20261003/source_forecasts.json.gz")["rows"]}
        frozen = read(read(executed["inputs"][4])["inputs"][5])["base_protocol"]
        pairs, groups = executed["cohort"]["seed_pairs"], executed["cohort"]["archive_groups"]
        seeds = [s for pair in pairs for s in pair]
        expected = {(s, g, h) for s in seeds for g in groups for h in range(8)}
        if not old_source["complete"] or len(contexts) != 256 or len(full) != 256 or set(ctx) != expected or set(ff) != expected or set(old) != {(*k,c) for k in expected for c in ("matched", "transfer")}:
            raise ValueError("A30 source cohort changed")
        tasks = {(s,g): prepare_sources(s, "shared_structure" if g == "shared" else "separate_head_structure", frozen)[0]
                 for s in seeds for g in groups}
        references = {s: np.random.default_rng(51000+s).uniform(-2,2,
            (frozen["reference_bank_size"], frozen["psi_dim"])) for s in seeds}
        bases = {s: FixedFunctionBasis(references[s]) for s in seeds}
        archive_keys = [(s,g) for s in seeds for g in groups]
        archive_index = {k: i for i,k in enumerate(archive_keys)}
        heads, risk_priors = {}, {}
        for key,c in ctx.items():
            s,g,h = key
            if c["training_tasks"] != [i for i in range(8) if i != h]:
                raise ValueError("A30 held task entered source learning")
            training = [tasks[s,g][i] for i in c["training_tasks"]]
            heads[key] = []
            for index,name in enumerate(("f","g")):
                basis = FixedFunctionBasis(references[s], c["fg_selected_indices"][name])
                prior = SourceFunctionPrior.from_basis(head_batches(training,index),
                    [t["noise"][:,index] for t in training], basis, "learned")
                heads[key].append(prior)
                result["cheap_fg_moment_fits"] += 1
            risk_priors[key] = local_priors(c,ff[key],bases[s])
            head_contexts.append({"seed":s,"archive_group":g,"held_task":h,
                "heads":[{"selected_indices":c["fg_selected_indices"][name],"mean":p.mean.tolist(),
                    "covariance":p.covariance.tolist(),"discrepancy_variance":p.discrepancy_variance}
                    for name,p in zip(("f","g"), heads[key])]})
        controls = {"max_a27_reproduction_error": 0., "max_forecast_mixture_error": 0.}
        for pair_id,pair in enumerate(pairs):
            population = [archive_index[k] for k in archive_keys if k[0] not in pair]
            if len(population) != 28:
                raise ValueError("A30 held pair entered population")
            for s in pair:
                partner = next(a for a in pair if a != s)
                for group in groups:
                    for held,task in enumerate(tasks[s,group]):
                        path_start = time.process_time()
                        eta = prequential_components([p for a,g in archive_keys for p in risk_priors[a,g,held]],
                            task["psi"], np.arange(24), task["response"][:24,2], task["noise"][:24,2])
                        fg_e = sum(prequential_components([heads[a,g,held][index] for a,g in archive_keys],
                            task["psi"], np.arange(24), task["response"][:24,index], task["noise"][:24,index])["log_evidence"]
                            for index in (0,1))
                        raw.append({"seed":s,"pair_id":pair_id,"archive_group":group,"held_task":held,
                            "risk_log_evidence":eta["log_evidence"][8:24].tolist(),
                            "risk_log_density":eta["log_density"][8:24].tolist(),
                            "auxiliary_log_evidence":fg_e[8:24].tolist()})
                        for condition,local_seed in (("matched",s),("transfer",partner)):
                            indices = population + [archive_index[local_seed,group]]
                            columns = [j for i in indices for j in (2*i, 2*i+1)]
                            risk = {k:v[8:24,columns] for k,v in eta.items()}
                            branches = sharing_branches(risk, fg_e[8:24,indices], np.r_[np.full(28,.5/28), .5])
                            u, sharing, posterior = sharing_forecasts(branches)
                            scores = {**branches["scores"], "U":u}
                            scores["I"],_ = mixture_forecasts(risk["log_evidence"][:,-2:], risk["log_density"][:,-2:], [.5,.5])
                            previous = old[s,group,held,condition]["scores"]
                            controls["max_a27_reproduction_error"] = max(controls["max_a27_reproduction_error"],
                                *(float(np.max(np.abs(scores[name] - previous[name]))) for name in ("J","A","G","I")))
                            # One Gaussian-mixture score computed directly from risk weights.
                            with np.errstate(divide="ignore"):
                                check = np.logaddexp.reduce(np.log(posterior)+risk["log_density"],axis=1)
                            controls["max_forecast_mixture_error"] = max(controls["max_forecast_mixture_error"],float(np.max(np.abs(u-check))))
                            if max(controls.values()) > 1e-8:
                                raise ValueError("A30 old predictions or risk-mixture identity changed")
                            delta = u-scores["I"]
                            metrics = {"u_minus_i":float(delta.mean()),"u_minus_j":float(np.mean(u-scores["J"])),
                                "u_minus_a":float(np.mean(u-scores["A"])),"u_minus_g":float(np.mean(u-scores["G"])),
                                "early_u_minus_i":float(delta[:4].mean()),"late_u_minus_i":float(delta[-4:].mean())}
                            row = {"seed":s,"pair_id":pair_id,"archive_group":group,"held_task":held,"condition":condition,**metrics}
                            for name,t in (("step9",0),("step24",-1)):
                                shared = float(sharing[t,0])
                                q_mean = (1+shared)/3
                                q_sd = np.sqrt((1+2*shared)/6-q_mean**2)
                                row[name] = {"shared_state_probability":shared,"sharing_probability_mean":q_mean,
                                    "sharing_probability_sd":float(q_sd),
                                    "log_shared_to_separate_bayes_factor":float(branches["log_evidence"][t,0]-branches["log_evidence"][t,1]),
                                    "population_risk_mass":float(posterior[t,:56].sum()),
                                    "local_risk_mass":float(posterior[t,-2:].sum()),
                                    "learned_risk_mass":float(posterior[t,::2].sum())}
                            result["rows"].append(row)
                            profiles.append({"seed":s,"pair_id":pair_id,"archive_group":group,"held_task":held,"condition":condition,
                                "branch_log_evidence":branches["log_evidence"].tolist(),
                                "branch_log_density":branches["log_density"].tolist(),
                                "scores":{k:v.tolist() for k,v in scores.items()},
                                "shared_state_probability":sharing[:,0].tolist()})
                        if len(raw) == 1:
                            first = time.process_time()-path_start
                            projection = time.process_time()-start+first*255*1.25+20
                            result["admission"] = {"first_path_cpu_seconds":first,"projected_stage_cpu_seconds":projection,
                                "cap_cpu_seconds":cap,"admitted":projection<cap}
                            print(json.dumps({"admission":result["admission"]}),flush=True)
                            if projection >= cap:
                                raise TimeoutError("A30 projection exceeds frozen CPU cap")
            print(json.dumps({"source_paths_completed":len(raw)}),flush=True)
        indices = np.random.default_rng(20261108).integers(8,size=(20000,8))
        summaries = []
        for condition in ("matched","transfer"):
            for group in groups:
                selected = [r for r in result["rows"] if (r["condition"],r["archive_group"]) == (condition,group)]
                summaries.append({"condition":condition,"archive_group":group,
                    **{k:block_contrast([np.mean([r[k] for r in selected if r["pair_id"] == i]) for i in range(8)],indices) for k in metrics},
                    **{name:{k:float(np.mean([r[name][k] for r in selected])) for k in selected[0][name]}
                       for name in ("step9","step24")}})
        result.update(complete=True, groups=summaries, controls=controls, source_paths=256,
            source_profiles=512, source_forecasts=8192, raw_archive_components=32,
            raw_evidence_cache_preserves_outer_exclusion=True, sharing_reference="Beta(1,1)",
            new_calibration_fits=0, record_recombination_stress_performed=False)
    except Exception as exc:
        result["failure"] = str(exc)
        raise
    finally:
        result.update(cpu_seconds=time.process_time()-start,wall_seconds=time.monotonic()-wall)
        for name,data in (("source_forecasts",{"rows":profiles}),
                          ("raw_source_component_evidence",{"archive_keys":archive_keys if raw else [],"rows":raw}),
                          ("source_head_contexts",{"rows":head_contexts})):
            (ARTIFACTS / (name+".json.gz")).write_bytes(gzip.compress(json.dumps(data,allow_nan=False).encode()))
        (ARTIFACTS / "validation.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(json.dumps({k:result[k] for k in ("complete","cpu_seconds")}),flush=True)


if __name__ == "__main__":
    main()

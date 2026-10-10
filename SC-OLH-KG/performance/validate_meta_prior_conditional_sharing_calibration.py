"""A31: source empirical sharing prior with nested archive exclusion."""
import gzip
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from performance.conditional_sharing_calibration import (
    cached_context, fit_sharing_prior, posterior_q_moments, training_profiles,
)
from performance.diagnose_meta_prior_source_cross_archive import block_contrast
from performance.validate_meta_prior_scale_preserving_source_covariance import read
from representation.archive_covariance_prior import sharing_branches, sharing_forecasts

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_conditional_sharing_calibration_a31_20261003"


def main():
    start,wall = time.process_time(),time.monotonic()
    spec = read("performance/meta_prior_conditional_sharing_calibration_plan.json")
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU,(cap,cap+2))
    def stop(signum,frame):
        raise TimeoutError("A31 local CPU cap reached")
    signal.signal(signal.SIGXCPU,stop)
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec,indent=2)+"\n")
    result = {"stage":spec["stage"],"complete":False,"rows":[],"calibrations":[],
        "new_gaussian_fits":0,"new_source_calls":0,"new_target_calls":0,"server_jobs":0,
        "server_pulls":0,"target_paths_scored":0,"kg_calls":0}
    profiles = []
    try:
        cache = read(spec["inputs"][0])
        old = read(spec["inputs"][1])["rows"]
        previous = {(r["seed"],r["archive_group"],r["held_task"],r["condition"]):r for r in old}
        source = read(spec["inputs"][2])
        pairs = spec["cohort"]["seed_pairs"]
        groups = spec["cohort"]["groups"]
        expected = {(s,g,h) for pair in pairs for s in pair for g in groups for h in range(8)}
        if not source["complete"] or not read(spec["inputs"][4])["proceed_to_nested_source_calibration"] or len(cache["rows"]) != 256 or {(r["seed"],r["archive_group"],r["held_task"]) for r in cache["rows"]} != expected or len(previous) != 512:
            raise ValueError("A31 source evidence or cohort changed")
        controls = {"max_outer_exclusion_error":0.,"max_a30_reproduction_error":0.}
        checks = {str(n):0. for n in (128,256)}
        for held,pair in enumerate(pairs):
            fold_start = time.process_time()
            e,p = training_profiles(cache,pairs,held)
            if e.shape != (7168,2) or p.shape != (7168,2):
                raise ValueError("A31 training cohort changed")
            fits = {n:fit_sharing_prior(e,p,n) for n in (64,128,256)}
            fit = fits[64]
            result["calibrations"].append({"held_pair_id":held,"held_seeds":pair,
                "training_pair_ids":[i for i in range(8) if i != held],
                "training_population_archive_count":24,"held_population_archive_count":28,**fit})
            if held == 0:
                # Remove both held teacher records and their prior columns before fitting.
                excluded_indices = [i for i,(s,g) in enumerate(cache["archive_keys"]) if s in pair]
                changed_rows = []
                for row in cache["rows"]:
                    changed = dict(row)
                    for name in ("risk_log_evidence","risk_log_density","auxiliary_log_evidence"):
                        array = np.array(row[name])
                        cols = excluded_indices if name.startswith("auxiliary") else [j for i in excluded_indices for j in (2*i,2*i+1)]
                        array[:,cols] += 100.
                        if row["pair_id"] == held:
                            array += 300.
                        changed[name] = array.tolist()
                    changed_rows.append(changed)
                alternate_cache = {"archive_keys":cache["archive_keys"],"rows":changed_rows}
                changed_e,changed_p = training_profiles(alternate_cache,pairs,held)
                controls["max_outer_exclusion_error"] = max(float(np.max(np.abs(e-changed_e))),float(np.max(np.abs(p-changed_p))))
            for row in cache["rows"]:
                if row["pair_id"] != held:
                    continue
                for condition,local in (("matched",row["seed"]),("transfer",next(s for s in pair if s != row["seed"]))):
                    risk,auxiliary,weights = cached_context(row,cache["archive_keys"],pair,local)
                    branches = sharing_branches(risk,auxiliary,weights)
                    c,sharing,posterior = sharing_forecasts(branches,fit["q_mean"])
                    q_mean,q_sd = posterior_q_moments(branches,fit)
                    u,_,_ = sharing_forecasts(branches)
                    key = row["seed"],row["archive_group"],row["held_task"],condition
                    saved = previous[key]["scores"]
                    controls["max_a30_reproduction_error"] = max(controls["max_a30_reproduction_error"],
                        float(np.max(np.abs(u-saved["U"]))),
                        *(float(np.max(np.abs(branches["scores"][name]-saved[name]))) for name in ("J","A","G")))
                    for n in (128,256):
                        alternative,_,_ = sharing_forecasts(branches,fits[n]["q_mean"])
                        checks[str(n)] = max(checks[str(n)],float(np.max(np.abs(c-alternative))))
                    full = {name:np.repeat(values[:,1::2],2,axis=1) for name,values in risk.items()}
                    generic = sharing_branches(full,auxiliary,weights)
                    cg,_,_ = sharing_forecasts(generic,fit["q_mean"])
                    delta = c-np.array(saved["I"])
                    metrics = {"c_minus_i":float(delta.mean()),"c_minus_j":float(np.mean(c-saved["J"])),
                        "c_minus_u":float(np.mean(c-saved["U"])),"c_minus_a":float(np.mean(c-saved["A"])),
                        "c_minus_cg":float(np.mean(c-cg)),"early_c_minus_i":float(delta[:4].mean()),
                        "late_c_minus_i":float(delta[-4:].mean())}
                    summary = {"seed":row["seed"],"pair_id":held,"archive_group":row["archive_group"],"held_task":row["held_task"],"condition":condition,**metrics}
                    for name,t in (("step9",0),("step24",-1)):
                        summary[name] = {"shared_state_probability":float(sharing[t,0]),
                            "sharing_probability_mean":float(q_mean[t]),"sharing_probability_sd":float(q_sd[t]),
                            "population_risk_mass":float(posterior[t,:56].sum()),"learned_risk_mass":float(posterior[t,::2].sum())}
                    result["rows"].append(summary)
                    profiles.append({**{k:summary[k] for k in ("seed","pair_id","archive_group","held_task","condition")},
                        "scores":{"C":c.tolist(),"CG":cg.tolist()},"source_q_mean":fit["q_mean"],
                        "shared_state_probability":sharing[:,0].tolist()})
            if held == 0:
                first = time.process_time()-fold_start
                projection = time.process_time()-start+first*7*1.25+5
                result["admission"] = {"first_outer_pair_cpu_seconds":first,"projected_stage_cpu_seconds":projection,"cap_cpu_seconds":cap,"admitted":projection<cap}
                print(json.dumps({"admission":result["admission"]}),flush=True)
                if projection >= cap:
                    raise TimeoutError("A31 projection exceeds frozen CPU cap")
            print(json.dumps({"outer_pairs_completed":held+1}),flush=True)
        if max(controls.values()) > 1e-8 or max(checks.values()) > 1e-4:
            raise ValueError("A31 exclusion, reproduction or quadrature check failed")
        indices = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(8,size=(20000,8))
        summaries = []
        for condition in ("matched","transfer"):
            for group in groups:
                selected = [r for r in result["rows"] if (r["condition"],r["archive_group"]) == (condition,group)]
                summaries.append({"condition":condition,"archive_group":group,
                    **{k:block_contrast([np.mean([r[k] for r in selected if r["pair_id"] == i]) for i in range(8)],indices) for k in metrics},
                    **{name:{k:float(np.mean([r[name][k] for r in selected])) for k in selected[0][name]} for name in ("step9","step24")}})
        result.update(complete=True,groups=summaries,controls=controls,
            numerical_check={"primary_order":64,"confirm_orders":[128,256],"max_forecast_changes":checks,"passed":True,"primary_parameters_changed":False},
            source_profiles=512,source_forecasts=8192,source_q_mean=float(np.mean([f["q_mean"] for f in result["calibrations"]])))
    except Exception as exc:
        result["failure"] = str(exc)
        raise
    finally:
        result.update(cpu_seconds=time.process_time()-start,wall_seconds=time.monotonic()-wall)
        (ARTIFACTS / "source_forecasts.json.gz").write_bytes(gzip.compress(json.dumps({"rows":profiles},allow_nan=False).encode()))
        (ARTIFACTS / "validation.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(json.dumps({k:result[k] for k in ("complete","cpu_seconds")}),flush=True)


if __name__ == "__main__":
    main()

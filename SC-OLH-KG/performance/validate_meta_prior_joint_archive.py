"""A27: jointly infer archive covariance identity from past f/g/eta."""
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
from performance.validate_meta_prior_function_covariance_source import profile_arrays
from performance.validate_meta_prior_scale_preserving_source_covariance import read
from representation.archive_covariance_prior import joint_archive_forecasts, mixture_forecasts, prequential_components
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior, head_batches

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_joint_archive_a27_20261003"


def main():
    start, wall = time.process_time(), time.monotonic()
    spec = read("performance/meta_prior_joint_archive_plan.json")
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap+2))
    def stop(signum, frame):
        raise TimeoutError("A27 local CPU cap reached")
    signal.signal(signal.SIGXCPU, stop)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS/"execution_plan.json").write_text(json.dumps(spec,indent=2)+"\n")
    result = {"stage":spec["stage"],"complete":False,"rows":[],"new_eta_ard_refits":0,
        "new_fg_selection_fits":0,"cheap_fg_moment_fits":0,"new_source_calls":0,"new_target_calls":0,
        "server_jobs":0,"server_pulls":0,"target_paths_scored":0,"kg_calls":0}
    profiles=[]
    try:
        contexts=read(spec["inputs"][0])["rows"]
        full=read(spec["inputs"][1])["rows"]
        ctx={(c["seed"],c["archive_group"],c["held_task"]):c for c in contexts}
        ff={(c["seed"],c["archive_group"],c["held_task"]):c for c in full}
        saved={condition:{(r["seed"],r["archive_group"],r["held_task"],r["arm"]):r for r in read(spec["inputs"][i])["rows"]}
            for i,condition in ((2,"matched"),(3,"transfer"))}
        frozen=read(read(spec["inputs"][4])["inputs"][5])["base_protocol"]
        if not read(spec["inputs"][5])["weight_only_modification_closed"]:
            raise ValueError("A27 requires the completed A26 diagnosis")
        pairs,groups=spec["cohort"]["seed_pairs"],spec["cohort"]["archive_groups"]
        seeds=[s for pair in pairs for s in pair]
        expected={(s,g,h) for s in seeds for g in groups for h in range(8)}
        if len(contexts)!=256 or len(full)!=256 or set(ctx)!=expected or set(ff)!=expected:
            raise ValueError("A27 source contexts changed")
        tasks={(s,g):prepare_sources(s,"shared_structure" if g=="shared" else "separate_head_structure",frozen)[0]
               for s in seeds for g in groups}
        references={s:np.random.default_rng(51000+s).uniform(-2,2,(frozen["reference_bank_size"],frozen["psi_dim"])) for s in seeds}
        bases={s:FixedFunctionBasis(references[s]) for s in seeds}
        heads,risk_priors={},{}
        for key,c in ctx.items():
            s,g,h=key
            train_ids=[i for i in range(8) if i!=h]
            if c["training_tasks"]!=train_ids:
                raise ValueError("A27 held task entered source learning")
            training=[tasks[s,g][i] for i in train_ids]
            heads[key]=[]
            for index,name in enumerate(("f","g")):
                basis=FixedFunctionBasis(references[s],c["fg_selected_indices"][name])
                heads[key].append(SourceFunctionPrior.from_basis(head_batches(training,index),
                    [t["noise"][:,index] for t in training],basis,"learned"))
                result["cheap_fg_moment_fits"]+=1
            risk_priors[key]=local_priors(c,ff[key],bases[s])
        controls={"max_original_i_error":0.,"max_original_full_error":0.,"max_future_auxiliary_error":0.}
        for pair_id,pair in enumerate(pairs):
            population=[(s,g) for s in seeds if s not in pair for g in groups]
            if len(population)!=28:
                raise ValueError("A27 held pair entered population")
            for s in pair:
                partner=next(a for a in pair if a!=s)
                for group in groups:
                    for held,task in enumerate(tasks[s,group]):
                        path_start=time.process_time()
                        archive_keys=population+[(s,group),(partner,group)]
                        eta_priors=[p for a,g in archive_keys for p in risk_priors[a,g,held]]
                        eta=prequential_components(eta_priors,task["psi"],np.arange(24),task["response"][:24,2],task["noise"][:24,2])
                        fg_e=np.zeros((24,len(archive_keys)))
                        for index in (0,1):
                            priors=[heads[a,g,held][index] for a,g in archive_keys]
                            aux=prequential_components(priors,task["psi"],np.arange(24),task["response"][:24,index],task["noise"][:24,index])
                            fg_e+=aux["log_evidence"]
                            if not profiles:
                                altered=task["response"][:24,index].copy();altered[8:]+=100.
                                changed=prequential_components(priors,task["psi"],np.arange(24),altered,task["noise"][:24,index])
                                controls["max_future_auxiliary_error"]=max(controls["max_future_auxiliary_error"],float(np.max(np.abs(aux["log_evidence"][:9]-changed["log_evidence"][:9]))))
                        for condition,local in (("matched",28),("transfer",29)):
                            columns=list(range(56))+[2*local,2*local+1]
                            risk={k:v[8:24,columns] for k,v in eta.items()}
                            aux_e=fg_e[8:24,list(range(28))+[local]]
                            scores,post=joint_archive_forecasts(risk,aux_e,np.r_[np.full(28,.5/28),.5])
                            scores["I"],_=mixture_forecasts(risk["log_evidence"][:,-2:],risk["log_density"][:,-2:],[.5,.5])
                            _,old=profile_arrays(saved[condition][s,group,held,"risk_ard_mixture"],32)
                            _,generic=profile_arrays(saved[condition][s,group,held,"fixed_source"],32)
                            controls["max_original_i_error"]=max(controls["max_original_i_error"],float(np.max(np.abs(scores["I"]-old[:,0]))))
                            controls["max_original_full_error"]=max(controls["max_original_full_error"],float(np.max(np.abs(risk["log_density"][:,-1]-generic[:,0]))))
                            if max(controls.values())>1e-8:
                                raise ValueError("A27 reproduction or future-information check failed")
                            delta=scores["J"]-scores["I"]
                            metrics={"j_minus_i":float(delta.mean()),"j_minus_a":float(np.mean(scores["J"]-scores["A"])),
                                "j_minus_g":float(np.mean(scores["J"]-scores["G"])),
                                "early_j_minus_i":float(delta[:4].mean()),"late_j_minus_i":float(delta[-4:].mean())}
                            result["rows"].append({"seed":s,"pair_id":pair_id,"archive_group":group,"held_task":held,"condition":condition,**metrics,
                                "step9_mass":[float(post["J"][0,:56].sum()),float(post["J"][0,-2:].sum()),float(post["J"][0,::2].sum())],
                                "step24_mass":[float(post["J"][-1,:56].sum()),float(post["J"][-1,-2:].sum()),float(post["J"][-1,::2].sum())]})
                            profiles.append({"seed":s,"pair_id":pair_id,"archive_group":group,"held_task":held,"condition":condition,
                                "scores":{k:v.tolist() for k,v in scores.items()}})
                        if len(profiles)==2:
                            first=time.process_time()-path_start
                            projection=time.process_time()-start+first*255*1.25+15
                            result["admission"]={"first_path_cpu_seconds":first,"projected_stage_cpu_seconds":projection,"cap_cpu_seconds":cap,"admitted":projection<cap}
                            print(json.dumps({"admission":result["admission"]}),flush=True)
                            if projection>=cap:
                                raise TimeoutError("A27 projection exceeds frozen CPU cap")
            print(json.dumps({"source_paths_completed":len(profiles)//2}),flush=True)
        indices=np.random.default_rng(spec["paired_bootstrap_seed"]).integers(8,size=(20000,8))
        summaries=[]
        for condition in spec["cohort"]["conditions"]:
            for group in groups:
                selected=[r for r in result["rows"] if (r["condition"],r["archive_group"])==(condition,group)]
                summaries.append({"condition":condition,"archive_group":group,
                    **{k:block_contrast([np.mean([r[k] for r in selected if r["pair_id"]==i]) for i in range(8)],indices) for k in metrics},
                    "step9_mass":np.mean([r["step9_mass"] for r in selected],axis=0).tolist(),
                    "step24_mass":np.mean([r["step24_mass"] for r in selected],axis=0).tolist()})
        result.update(complete=True,groups=summaries,controls=controls,source_paths=256,source_forecasts=8192,
            mass_order=["population_archive","local_archive","learned_risk_component"])
    except Exception as exc:
        result["failure"]=str(exc)
        raise
    finally:
        result.update(cpu_seconds=time.process_time()-start,wall_seconds=time.monotonic()-wall)
        (ARTIFACTS/"source_forecasts.json.gz").write_bytes(gzip.compress(json.dumps({"rows":profiles},allow_nan=False).encode()))
        (ARTIFACTS/"validation.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(json.dumps({k:result[k] for k in ("complete","cpu_seconds")}),flush=True)


if __name__=="__main__":
    main()

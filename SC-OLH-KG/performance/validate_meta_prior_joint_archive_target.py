"""A28: unchanged joint archive prior on all historical target query paths."""
import gzip
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from performance.diagnose_meta_prior_source_cross_archive import block_contrast
from performance.run_meta_prior_mechanism import observe, task_truth
from performance.run_meta_prior_mechanism_v2 import prepare_sources
from performance.validate_meta_prior_function_covariance_source import profile_arrays
from performance.validate_meta_prior_scale_preserving_source_covariance import read
from representation.archive_covariance_prior import joint_archive_forecasts, mixture_forecasts, prequential_components
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior, head_batches

ARTIFACTS=ROOT/"paper_artifacts/meta_prior_joint_archive_target_a28_20261003"


def main():
    start,wall=time.process_time(),time.monotonic()
    spec=read("performance/meta_prior_joint_archive_target_plan.json")
    cap=spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU,(cap,cap+2))
    def stop(signum,frame):
        raise TimeoutError("A28 local CPU cap reached")
    signal.signal(signal.SIGXCPU,stop)
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    (ARTIFACTS/"execution_plan.json").write_text(json.dumps(spec,indent=2)+"\n")
    result={"stage":spec["stage"],"complete":False,"rows":[],"new_eta_ard_refits":0,
        "new_fg_selection_fits":0,"cheap_fg_moment_fits":0,"new_source_calls":0,"new_target_calls":0,
        "server_jobs":0,"server_pulls":0,"kg_calls":0,"candidate_recommendations":0,"new_verifier_calls":0}
    profiles=[]
    try:
        contexts=read(spec["inputs"][0])["rows"]
        context={(c["seed"],c["regime"]):c for c in contexts}
        previous={(r["seed"],r["regime"],r["arm"]):r for r in read(spec["inputs"][1])["rows"]}
        records=read(spec["inputs"][2])["records"]
        base=read(spec["inputs"][3])["base_protocol"]
        protocol={**base,**base["allocations"][spec["cohort"]["allocation"]]}
        if not read(spec["inputs"][4])["source_gate_passed_for_cached_target_forecasts"]:
            raise ValueError("A28 requires the complete A27 source gate")
        cohort=spec["cohort"]
        seeds,regimes,policies=cohort["seeds"],cohort["regimes"],cohort["path_policies"]
        pairs=cohort["seed_pairs"]
        pair_of={s:i for i,pair in enumerate(pairs) for s in pair}
        indexed={(r["row"]["seed"],r["row"]["regime"],r["row"]["arm"]):r["row"] for r in records}
        expected={(s,r,a) for s in seeds for r in regimes for a in policies}
        if len(records)!=160 or set(indexed)!=expected or set(previous)!=expected or len(contexts)!=80 or set(context)!={(s,r) for s in seeds for r in regimes}:
            raise ValueError("A28 target cohort changed")
        controls={"max_saved_i_error":0.,"max_saved_f_error":0.,"max_shared_source_context_error":0.}
        for s in seeds:
            shared=context[s,"shared_structure"]
            for regime in regimes:
                c=context[s,regime]
                if regime!="separate_head_structure":
                    if c["fg_selected_indices"]!=shared["fg_selected_indices"]:
                        raise ValueError("A28 shared source selections changed between regimes")
                    for one,two in zip(c["eta_prior_components"],shared["eta_prior_components"]):
                        for name in ("coefficient_variance","intercept_mean","discrepancy_variance"):
                            controls["max_shared_source_context_error"]=max(controls["max_shared_source_context_error"],float(np.max(np.abs(np.array(one[name])-two[name]))))
        if controls["max_shared_source_context_error"]>1e-8:
            raise ValueError("A28 shared source priors differ across target regimes")
        psi={s:np.random.default_rng(51000+s).uniform(-2,2,(protocol["candidates"],protocol["psi_dim"])) for s in seeds}
        heads,risk_priors={},{}
        for s in seeds:
            reference=psi[s][:protocol["reference_bank_size"]]
            basis=FixedFunctionBasis(reference)
            for group,regime in (("shared","shared_structure"),("separate_heads","separate_head_structure")):
                tasks=prepare_sources(s,regime,protocol)[0]
                c=context[s,regime]
                risk_priors[s,group]=[SourceFunctionPrior(basis,np.r_[p["intercept_mean"],np.zeros(basis.feature_dim)],
                    np.diag(p["coefficient_variance"]),p["discrepancy_variance"],{}) for p in c["eta_prior_components"]]
                heads[s,group]=[]
                for index,name in enumerate(("f","g")):
                    selected=FixedFunctionBasis(reference,c["fg_selected_indices"][name])
                    heads[s,group].append(SourceFunctionPrior.from_basis(head_batches(tasks,index),[t["noise"][:,index] for t in tasks],selected,"learned"))
                    result["cheap_fg_moment_fits"]+=1
        for s in seeds:
            population=[(a,g) for a in seeds if a not in pairs[pair_of[s]] for g in ("shared","separate_heads")]
            for regime in regimes:
                group="separate_heads" if regime=="separate_head_structure" else "shared"
                archive_keys=population+[(s,group)]
                _,supports=prepare_sources(s,regime,protocol)
                target_seed=61000+100*s+protocol["regimes"].index(regime)
                truth=task_truth(psi[s],supports,target_seed)
                truth[:,1]+=protocol["difficulty_offsets"][cohort["difficulty"]]
                for policy in policies:
                    path_start=time.process_time()
                    row=indexed[s,regime,policy]
                    ids=np.array(row["queries"])
                    if row["difficulty"]!=cohort["difficulty"] or row["allocation"]!=cohort["allocation"] or len(ids)!=24 or row["queries"][:8]!=row["initial_design"]:
                        raise ValueError("A28 path budgets or initial candidates changed")
                    visits=np.zeros(len(psi[s]),dtype=int)
                    labels,noises,prior_visits=[],[],[]
                    for candidate in ids:
                        v=int(visits[candidate]);prior_visits.append(v)
                        y,n=observe(truth[candidate],target_seed,int(candidate),v,protocol["replicates"])
                        labels.append(y);noises.append(n);visits[candidate]+=1
                    labels,noises=np.array(labels),np.array(noises)
                    eta=prequential_components([p for a,g in archive_keys for p in risk_priors[a,g]],psi[s],ids,labels[:,2],noises[:,2])
                    risk={k:v[8:24] for k,v in eta.items()}
                    fg_e=np.zeros((24,29))
                    for index in (0,1):
                        aux=prequential_components([heads[a,g][index] for a,g in archive_keys],psi[s],ids,labels[:,index],noises[:,index])
                        fg_e+=aux["log_evidence"]
                    scores,post=joint_archive_forecasts(risk,fg_e[8:24],np.r_[np.full(28,.5/28),.5])
                    scores["I"],_=mixture_forecasts(risk["log_evidence"][:,-2:],risk["log_density"][:,-2:],[.5,.5])
                    scores["F"]=risk["log_density"][:,-1]
                    _,old=profile_arrays(previous[s,regime,policy],32)
                    baseline=scores["I"] if policy=="risk_ard_mixture" else scores["F"]
                    name="max_saved_i_error" if policy=="risk_ard_mixture" else "max_saved_f_error"
                    controls[name]=max(controls[name],float(np.max(np.abs(baseline-old[:,0]))))
                    if max(controls.values())>1e-8:
                        raise ValueError("A28 old observation chronology or original eta prediction changed")
                    delta=scores["J"]-scores["I"]
                    metrics={"j_minus_i":float(delta.mean()),"j_minus_a":float(np.mean(scores["J"]-scores["A"])),
                        "j_minus_g":float(np.mean(scores["J"]-scores["G"])),"j_minus_f":float(np.mean(scores["J"]-scores["F"])),
                        "early_j_minus_i":float(delta[:4].mean()),"late_j_minus_i":float(delta[-4:].mean())}
                    result["rows"].append({"seed":s,"pair_id":pair_of[s],"regime":regime,"path_policy":policy,**metrics,
                        "step9_mass":[float(post["J"][0,:56].sum()),float(post["J"][0,-2:].sum()),float(post["J"][0,::2].sum())],
                        "step24_mass":[float(post["J"][-1,:56].sum()),float(post["J"][-1,-2:].sum()),float(post["J"][-1,::2].sum())]})
                    profiles.append({"seed":s,"pair_id":pair_of[s],"regime":regime,"path_policy":policy,
                        "candidates":ids.tolist(),"prior_visits":prior_visits,
                        "scores":{k:v.tolist() for k,v in scores.items()}})
                    if len(profiles)==1:
                        first=time.process_time()-path_start
                        projection=time.process_time()-start+first*159*1.25+15
                        result["admission"]={"first_path_cpu_seconds":first,"projected_stage_cpu_seconds":projection,"cap_cpu_seconds":cap,"admitted":projection<cap}
                        print(json.dumps({"admission":result["admission"]}),flush=True)
                        if projection>=cap:
                            raise TimeoutError("A28 projection exceeds frozen CPU cap")
            print(json.dumps({"target_paths_completed":len(profiles)}),flush=True)
        indices=np.random.default_rng(spec["paired_bootstrap_seed"]).integers(8,size=(20000,8))
        groups=[]
        for regime in regimes:
            for policy in policies:
                selected=[r for r in result["rows"] if (r["regime"],r["path_policy"])==(regime,policy)]
                groups.append({"regime":regime,"path_policy":policy,
                    **{k:block_contrast([np.mean([r[k] for r in selected if r["pair_id"]==i]) for i in range(8)],indices) for k in metrics},
                    "step9_mass":np.mean([r["step9_mass"] for r in selected],axis=0).tolist(),
                    "step24_mass":np.mean([r["step24_mass"] for r in selected],axis=0).tolist()})
        result.update(complete=True,groups=groups,controls=controls,target_paths_scored=160,target_forecasts=2560,
            repeated_candidate_paths=sum(len(set(p["candidates"]))<24 for p in profiles),
            mass_order=["population_archive","local_archive","learned_risk_component"])
    except Exception as exc:
        result["failure"]=str(exc)
        raise
    finally:
        result.update(cpu_seconds=time.process_time()-start,wall_seconds=time.monotonic()-wall)
        (ARTIFACTS/"target_forecasts.json.gz").write_bytes(gzip.compress(json.dumps({"rows":profiles},allow_nan=False).encode()))
        (ARTIFACTS/"validation.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(json.dumps({k:result[k] for k in ("complete","cpu_seconds")}),flush=True)


if __name__=="__main__":
    main()

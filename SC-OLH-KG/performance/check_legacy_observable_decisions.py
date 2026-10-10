"""Numerical followup on saved positive and negative observable decisions."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from baselines.transfer_archive import FrozenTransferArchive,resample_normalized_profiles
from core.budgeted_guarded_kg import BudgetedTerminalActions,budgeted_marginal_lookahead
from core.designs import load_frozen_source_informed_design
from core.guarded_deployment_kg import posterior_draws
from performance.benchmark_frozen_proposal_only import _select_shortlist
from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from performance.replay_guarded_deployment import shortlist_indices
from performance.replay_legacy_observable_budget import fit_arms
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.legacy_function_prior import legacy_source_labels
from representation.legacy_observable_prior import legacy_exposure_descriptors


def check():
    path=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/"performance/manifests/legacy_observable_precision_v1.json"
    spec=json.loads(path.read_text())
    replay=json.loads((ROOT/spec["replay"]).read_text()); protocol=replay["protocol"]
    old=json.loads((ROOT/protocol["paid_paths"]).read_text()); base=old["protocol"]
    started=time.process_time(); rows=[]; truth_jobs=[]
    for domain in dict.fromkeys(c["domain"] for c in spec["cases"]):
        inputs=ROOT/"paper_artifacts/legacy_joint_bridge_v1/inputs"/domain
        archive=FrozenTransferArchive.load(inputs/"archive.json")
        initial,_=load_frozen_source_informed_design(inputs/"design.json",heldout=domain,
            seed=base["target_seed"],n0=10,dimension=base["target_dimension"])
        problem=build_scalarized_problem(domain,base["target_dimension"],base["L"],base["sigma"],base["alpha"],base["weights"])
        lifted=resample_normalized_profiles(np.vstack([t.X for t in archive.tasks]),problem.d)
        candidates=list(dict.fromkeys(list(initial)+[tuple(map(int,problem.continuous_to_int(x))) for x in lifted]))
        desc=[]
        for task in archive.tasks:
            provider=build_scalarized_problem(task.name,task.X.shape[1],base["L"],base["sigma"],base["alpha"],base["weights"])
            desc.extend(legacy_exposure_descriptors(provider,task.X_integer,protocol["dynamics_grid_dimension"]))
        coordinate,arms=fit_arms(legacy_source_labels(archive),np.asarray(desc),protocol)
        psi=coordinate.transform(legacy_exposure_descriptors(problem,candidates,protocol["dynamics_grid_dimension"]))
        paid=next(r for r in old["rows"] if r["domain"]==domain and r["arm"]=="source_orthogonal" and r["search_calls"]==13)
        for arm in dict.fromkeys(c["arm"] for c in spec["cases"] if c["domain"]==domain):
            p=arms[arm]; f=JointFunctionRiskPosterior([p[0]],[p[3]],psi); g=JointFunctionRiskPosterior([p[1]],[p[2]],psi)
            for prefix,(index,value) in enumerate(zip(paid["paid_candidate_indices"],paid["paid_observations"]),1):
                f.update_batch(index,value[0],None,1); g.update_batch(index,value[1]-problem.tau,None,1)
                if not any(c=={"domain":domain,"prefix":prefix,"arm":arm} for c in spec["cases"]): continue
                n0=min(prefix,10); obs=paid["paid_observations"][:n0]
                orig,_=_select_shortlist(problem,initial[:n0],obs)
                pos=[initial[:n0].index(tuple(r["point"])) for r in orig]
                table,inc=shortlist_indices(pos,list(range(n0)),obs,len(candidates),range(10))
                terminal=BudgetedTerminalActions(table,inc,{"objective_reference":float(np.dot(problem.weights,problem.ref_point)),
                    "chance_alpha":problem.alpha,"familywise_delta":.05,"comparison_budget":8,"comparison_delta":.05})
                remaining=protocol["total_target_call_cap"]-prefix
                samples=[]; values=[]
                for seed in spec["seeds"]:
                    if spec["integration"]=="terminal_marginal":
                        r,c,a=terminal.evaluate_posterior(f,g,remaining,spec["draw_power"],seed)
                    else:
                        draws=posterior_draws(f,g,spec["draw_power"],seed)
                        r,c,a=terminal.evaluate(draws,remaining)
                    best=int(np.lexsort((c,-r))[0])
                    result={"seed":seed,"action":a[best],"predicted_reward":float(r[best]),"predicted_calls":float(c[best])}
                    samples.append(result); values.append(r)
                    truth_jobs.append((result,terminal,remaining,best,problem,candidates))
                original=next(r for r in replay["rows"] if r["domain"]==domain and r["arm"]==arm and r["prefix"]==prefix)
                rows.append({"domain":domain,"arm":arm,"prefix":prefix,"integrations":samples,
                    "original_4096_action":original["terminal_action"],
                    "max_reward_difference":float(np.abs(values[0]-values[-1]).max()) if len(values)>1 else None})
                probe=spec.get("lookahead_probe")
                if probe and probe["domain"]==domain and probe["arm"]==arm and probe["prefix"]==prefix:
                    points=list(dict.fromkeys([pos[1],pos[0],samples[0]["action"]["primary"]]))
                    rows[-1]["lookahead"]=budgeted_marginal_lookahead(f,g,terminal,remaining,points,
                        draw_power=probe["draw_power"],fantasy_power=probe["fantasy_power"],seed=protocol["integration_seed"])
                print(domain,arm,prefix,"precision frozen",flush=True)
    for row,terminal,remaining,best,problem,candidates in truth_jobs:
        truth=np.array([[problem.true_objective(p),np.log(problem.true_sigma(p)[0]**2),
            problem.true_constraint_mean(p)-problem.tau,np.log(problem.true_sigma(p)[1]**2)] for p in candidates])[None]
        r,c,_=terminal.evaluate(truth,remaining)
        row["truth"]={"reward":float(r[best]),"calls":float(c[best]),
            "primary_feasible":bool(problem.is_truly_feasible(candidates[row["action"]["primary"]]))}
    output={"protocol":spec,"rows":rows,"new_calls":0,"cpu_seconds":time.process_time()-started}
    (ROOT/"paper_artifacts/legacy_observable_budget_v1"/spec["output"]).write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({"rows":len(rows),"new_calls":0,"cpu_seconds":output["cpu_seconds"]}),flush=True)


if __name__=="__main__":
    check()

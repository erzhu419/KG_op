"""Compare fresh repeated batches at the retained policy using saved data."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from core.budgeted_guarded_kg import BudgetedTerminalActions
from core.raw_likelihood_lookahead import likelihood_query_lookahead
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.replay_legacy_mixture_budget import build_models,terminal_at_prefix


def run(protocol_path='performance/manifests/legacy_repeat_batch_v1.json'):
    started=time.process_time()
    spec=json.loads((ROOT/protocol_path).read_text())
    diagnosis=json.loads((ROOT/spec['diagnosis']).read_text()); pilot=json.loads((ROOT/spec['pilot']).read_text())
    p=pilot['protocol']; replay=json.loads((ROOT/p['replay']).read_text())
    old=json.loads((ROOT/replay['protocol']['paid_paths']).read_text()); source_spec=replay['source_fit_contract']
    problem,initial,candidates,model,sources=build_models(p['domain'],source_spec,old['protocol'],replay['protocol']['hypotheses'])
    no_outcomes(problem)
    paid=next(r for r in old['rows'] if r['domain']==p['domain'] and r['arm']=='source_orthogonal' and r['search_calls']==13)
    for index,value in zip(paid['paid_candidate_indices'][:4],paid['paid_observations'][:4]):
        model.update_raw(index,[value[0],value[1]-problem.tau])
    row=next(r for r in pilot['rows'] if r['group']==2 and r['arm']=='observe_primary')
    model.update_raw(row['query'],[row['raw_observation'][0],row['raw_observation'][1]-problem.tau])
    np.testing.assert_allclose(model.weights,diagnosis['target_model_weights'],rtol=1e-12,atol=1e-12)
    original,_=terminal_at_prefix(problem,initial,paid,4,source_spec,len(candidates))
    mask=np.isin(original.shortlists[:,0],diagnosis['target_primary_roster'])
    terminal=BudgetedTerminalActions(original.shortlists[mask],original.incumbents[mask],original.protocol)
    terminal.retain_recommendation(spec['retained_policy'])
    probes=[]
    for seed in spec['seeds']:
        for count in spec['fresh_replicates']:
            print('fresh batch probe starts',seed,count,flush=True)
            probe=likelihood_query_lookahead(model,terminal,p['total_target_call_cap']-row['search_calls'],spec['query'],
                draw_power=spec['draw_power'],fantasy_power=spec['fantasy_power'],seed=seed,replicates=count)
            probe['seed']=seed; probes.append(probe)
            print('fresh batch probe complete',seed,count,probe['net_reward_of_batch'],probe['projected_net_reward_of_batch'],
                probe['exact_moment_variance_reduction_by_candidate'][str(spec['query'])][3],flush=True)
    output={'protocol':spec,'source_fit':sources,'model_weights':model.weights.tolist(),'probes':probes,
        'new_simulator_calls':0,'target_scoring_truth_queries':0,'cpu_seconds':time.process_time()-started}
    out=ROOT/spec['output']; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'probes':len(probes),'cpu_seconds':output['cpu_seconds'],'new_simulator_calls':0}),flush=True)


if __name__=='__main__':
    run()

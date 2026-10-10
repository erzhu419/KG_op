"""Refine the saved-state repeat/new probe without repeating source folds."""

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


def run():
    started=time.process_time()
    spec=json.loads((ROOT/'performance/manifests/legacy_risk_probe_precision_v1.json').read_text())
    diagnosis=json.loads((ROOT/spec['diagnosis']).read_text()); pilot=json.loads((ROOT/spec['pilot']).read_text())
    p=pilot['protocol']; replay=json.loads((ROOT/p['replay']).read_text())
    old=json.loads((ROOT/replay['protocol']['paid_paths']).read_text()); source_spec=replay['source_fit_contract']
    problem,initial,candidates,model,_=build_models(p['domain'],source_spec,old['protocol'],replay['protocol']['hypotheses'])
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
        for query in spec['queries']:
            print('precision probe starts',seed,query,flush=True)
            probe=likelihood_query_lookahead(model,terminal,p['total_target_call_cap']-row['search_calls'],query,
                draw_power=spec['draw_power'],fantasy_power=spec['fantasy_power'],seed=seed)
            probe['seed']=seed; probes.append(probe)
            print('precision probe complete',seed,query,probe['net_reward_of_paid_episode'],
                probe['likelihood_tower_drift'],probe['effective_sample_size_min'],flush=True)
    output={'protocol':spec,'target_probes':probes,'new_simulator_calls':0,
        'target_scoring_truth_queries':0,'cpu_seconds':time.process_time()-started}
    (ROOT/spec['output']).write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'probes':len(probes),'cpu_seconds':output['cpu_seconds'],'new_simulator_calls':0}),flush=True)


if __name__=='__main__':
    run()

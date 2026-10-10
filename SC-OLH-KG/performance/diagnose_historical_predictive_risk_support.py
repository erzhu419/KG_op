"""Locate single-call predictive integration error using existing head caches."""

import argparse
import json
import time

import numpy as np
from scipy.stats import qmc

from performance.complete_historical_source_heads import ROOT, read, save
from performance.refine_historical_achievable_precision import assemble_case, compose_records


def diagnose(plan):
    started = time.process_time(); spec = read(plan); parent = read(spec['parent_plan'])
    registry = read(spec['output']+'/registered_jobs.json')['jobs']; cases = []
    reference = read(spec['output']+'/results_at16.json')
    for case in spec['cases']:
        assert case['n'] == 1
        heads, bank, _ = assemble_case(spec, case, registry)
        source = parent['accepted_baseline_inputs'][case['state']]
        baseline = read(source+'/'+case['state']+'/n1/baseline/terminal_rules.json')['rows'][-1]
        stop = read(parent['stop_inputs'][case['state']])['rows'][-1]['best_reward']
        fixed = baseline['best_action_index']; debited = baseline['best_reward']
        probabilities = qmc.Sobol(2, scramble=True, seed=parent['predictive_seed']).random_base2(
            case['predictive_power'])[:, 1]
        adaptation = 0.; tower = 0.; origins = []
        for origin in bank['origins']:
            f = heads[0, origin][:max(case['objective_prefixes'])]
            g = heads[1, origin][:max(parent['predictive_prefixes'])]
            raw, _, _, _ = compose_records(f, g, bank['prior_weights'], 'raw', 1)
            native, costs, _, _ = compose_records(f, g, bank['prior_weights'], 'projected', 1)
            choices = np.lexsort((costs, -native), axis=2)[:, :, 0]
            achieved = np.take_along_axis(raw, choices[:, :, None], axis=2)[:, :, 0]
            gain = (achieved-raw[:, :, fixed]).mean(axis=0)
            fixed_values = raw[:, :, fixed].mean(axis=0); mass = bank['prior_weights'][origin]
            adaptation += mass*float(gain.mean()); tower += mass*float(fixed_values.mean()-debited)
            rows = [dict(position=p, quantile=float(probabilities[p]), label=r['label'],
                achieved_minus_fixed=float(gain[p]), fixed_reward=float(fixed_values[p]),
                recommendation_switch_fraction=float(np.mean(choices[:, p] != fixed))) for p, r in enumerate(g)]
            ordered = sorted(rows, key=lambda r:r['quantile'])
            jumps = [dict(left=a['quantile'], right=b['quantile'],
                gain_change=b['achieved_minus_fixed']-a['achieved_minus_fixed']) for a,b in zip(ordered,ordered[1:])]
            origins.append(dict(origin=origin, prior_mass=mass, rows=rows,
                largest_adjacent_changes=sorted(jumps,key=lambda r:abs(r['gain_change']),reverse=True)[:8]))
        net = adaptation+debited-stop
        fine = next(c for c in reference['cases'] if c['case']==case['name'])['rows'][-1]
        assert abs(net-fine['raw_net_reward_of_projected_choice_control_variate']) <= 1e-12
        assert abs(tower-fine['raw_fixed_action_tower_residual']) <= 1e-12
        cases.append(dict(case=case['name'],net=net,fixed_action_tower_residual=tower,origins=origins))
    result = dict(cases=cases, scope='Current cached16 development diagnostic; not numerical admission.',
        cpu_seconds=time.process_time()-started,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0)
    save(ROOT/spec['output']/'risk_support_preview.json',result)
    print('SCHEDULEURM_DONE '+json.dumps(dict(complete=True,cpu_seconds=result['cpu_seconds'],
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0)),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True)
    diagnose(parser.parse_args().plan)

"""Extend only the sensitive one-call source objective predictive axis."""

import argparse
import copy
import json
import signal
import time

import numpy as np
from scipy.stats import qmc

from core.positive_mixture_lookahead import RawPredictiveCDF
from core.raw_terminal_head_grid import evaluate_head_records
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.historical_full_bank_batch_kg import restore
from performance.historical_achievable_batch_kg import grid

PLAN = 'performance/manifests/historical_achievable_objective_v3.json'


def run(origin, plan=PLAN):
    spec = read(plan); parent = read(spec['parent_plan']); name = spec['state']
    out = ROOT/spec['output']/str(origin); out.mkdir(parents=True, exist_ok=True)
    if (out/'results.json').exists() and read(str(out/'results.json')).get('complete'):
        return read(str(out/'results.json'))
    started = time.process_time(); handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_case_cpu_seconds'])
    result = dict(origin=origin, complete=False, new_physical_calls=0, new_source_fits=0, new_verifier_calls=0)
    try:
        model, terminal, _, _ = restore(parent, name)
        old = read(parent['output']+'/'+name+'/n1/head/0_'+str(origin)+'/head_records.json')
        cached = copy.deepcopy(old['records'])
        cached.sort(key=lambda r:r['position'])
        for position, row in enumerate(cached): row['position'] = position
        nodes = qmc.Sobol(2, scramble=True, seed=parent['predictive_seed']).random_base2(spec['predictive_power'])
        cdf = RawPredictiveCDF(model.components[origin][0], parent['query'])
        labels = cdf.inverse(nodes[:,0]); old_error = float(np.max(abs(labels[:len(cached)]-[r['label'] for r in cached])))
        assert old_error <= 1e-8
        labels[:len(cached)] = [r['label'] for r in cached]
        cdf_values, bound = cdf.adaptive(labels)
        error = float(np.max(abs(cdf_values-nodes[:,0])))+bound
        if error > 1e-8: raise ValueError('Raw predictive objective quantiles remain inaccurate')
        partial = out/'head_records.json'
        if partial.exists(): cached = read(str(partial))['records']
        def checkpoint(rows, actions):
            assert actions == old['actions']
            save(partial, dict(records=rows, actions=actions, prior_weights=model.weights.tolist()))
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(step=len(rows), total=len(labels), unit='objective_states',
                metrics=dict(origin=origin, cpu_seconds=time.process_time()-started))), flush=True)
        rows = evaluate_head_records(model, terminal, 360, parent['query'], 0, labels, range(len(labels)),
            powers=parent['powers'], projected_powers=parent['powers'], seed=parent['seed'], completed=cached,
            checkpoint=checkpoint, utility_identities=cached[0]['utility_identities'], guarded_mean_position=0)
        result.update(complete=True, objective_states=len(rows), new_objective_states=len(rows)-8,
            old_label_roundoff=old_error, inverse_cdf_error=error)
    except TimeoutError:
        result['cpu_budget_stopped'] = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
        result['cpu_seconds'] = time.process_time()-started; save(out/'results.json', result)
    print('SCHEDULEURM_DONE '+json.dumps(result), flush=True)
    return result


def collect(plan=PLAN):
    spec = read(plan); parent = read(spec['parent_plan']); name = spec['state']
    directory = parent['output']+'/'+name; bank = read(directory+'/bank.json')
    heads = {}; jobs = []
    for origin in bank['origins']:
        path = spec['output']+'/'+str(origin)
        job = read(path+'/results.json'); jobs.append(job)
        if not job['complete']: raise ValueError('Objective extension incomplete')
        heads[0,origin] = sorted(read(path+'/head_records.json')['records'],key=lambda r:r['position'])
        heads[1,origin] = sorted(read(directory+'/n1/head/1_'+str(origin)+'/head_records.json')['records'],key=lambda r:r['position'])
    stop = read(parent['stop_inputs'][name])['rows'][-1]
    debited = read(directory+'/n1/baseline/terminal_rules.json')['rows'][-1]
    baseline = stop['best_reward'], debited['best_reward'], debited['best_action_index']
    rows = [grid(heads, np.asarray(bank['prior_weights']), bank['origins'], [prefix,8],
        parent['powers'][-1],parent['powers'][-1],baseline) for prefix in spec['prefixes']]
    changes = [abs(rows[i]['raw_net_reward_of_projected_choice_control_variate']-
                   rows[i-1]['raw_net_reward_of_projected_choice_control_variate']) for i in range(1,len(rows))]
    result = dict(rows=rows, executions=jobs, objective_prefix_changes=changes,
        objective_axis_passed=bool(changes[-1] <= parent['maximum_numerical_adaptation_difference']),
        cpu_seconds=sum(r['cpu_seconds'] for r in jobs), new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        numerical_diagnostics_passed=False, recommendation_frozen=False, independent_confirmation=False,
        scope='Only source one-call objective axis8/16/32; reuse eight risk states per origin and all terminal caches. '
              'The original raw/native action and evidence HOLD remain open.')
    save(ROOT/spec['output']/'outcome_summary.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--origin', type=int)
    parser.add_argument('--collect', action='store_true'); parser.add_argument('--plan', default=PLAN)
    args = parser.parse_args()
    if args.collect: collect(args.plan)
    else: run(args.origin,args.plan)

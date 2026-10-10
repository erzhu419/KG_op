"""Measure shared allocations on an already opened joint-risk record."""

import argparse
import copy
import json
import resource
import time

import numpy as np

from core.budgeted_guarded_kg import BudgetedTerminalActions
from core.conditional_raw_reference import component_risk_allocations, component_risk_statistics
from performance.complete_historical_source_heads import ROOT, read, save
from performance.diagnose_historical_component_guarded import compressed_read, mixed_values
from performance.diagnose_historical_shared_risk import BASE, OUT
from performance.refine_historical_observed_prediction import field_from_snapshot
from representation.vectorized_joint_function_risk import update_raw_observation

PLAN = 'performance/manifests/historical_risk_allocations_v1.json'


def run(identity, action_index, kind, plan=PLAN):
    spec = read(plan)
    output = ROOT / spec['output']
    path = output / 'parts' / f'{identity}_{action_index}_{kind}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.process_time()
    resource.setrlimit(resource.RLIMIT_CPU, (spec['maximum_task_cpu_seconds'],
        spec['maximum_task_cpu_seconds'] + 5))
    bundle = compressed_read(BASE / 'marginals' / (spec['record'] + '.json.gz'))
    assert identity in bundle['utility_identities'] and bundle['replicates'] == 1
    block = bundle['blocks'][action_index]
    action = block['action']
    assert action['kind'] == 'guarded'
    prior = field_from_snapshot(block['fields'][identity])
    future = copy.deepcopy(prior)
    score = update_raw_observation(future, 0, bundle['label'], None, 1)
    assert abs(score - bundle['head_log_scores'][identity]) <= 1e-10
    builder = BudgetedTerminalActions([action['shortlist']], [action['incumbent_position']], bundle['protocol'])
    actions = [dict(action, budgets=budgets) for k, _, _, budgets in builder.allocations(spec['remaining_calls'])
        if k == 'guarded']
    original = next(i for i, a in enumerate(actions) if a['budgets'] == action['budgets'])
    _, rank = np.unique(prior.means, axis=0, return_inverse=True)
    anchor = read(str(OUT / 'coordinate_parts' / f'{identity}_{action_index}_{kind}.json'))['rows']['E7']
    rows = {}
    for rule in spec['rules']:
        kwargs = dict(query=0, raw=kind == 'raw', order=rule['order'], power=rule['power'],
            seed=bundle['seed'] + rank * 104729, integrated_position=action['incumbent_position'])
        tick = time.process_time()
        separate = [component_risk_statistics(prior, future, a, bundle['label'], score,
            bundle['protocol'], **kwargs) for a in actions]
        separate_cpu = time.process_time() - tick
        tick = time.process_time()
        statistics, costs, diagnostics = component_risk_allocations(prior, future, actions,
            bundle['label'], score, bundle['protocol'], **kwargs)
        shared_cpu = time.process_time() - tick
        value_difference = float(np.max(abs(statistics - np.asarray([r[0] for r in separate]))))
        cost_difference = float(np.max(abs(costs - np.asarray([r[1] for r in separate]))))
        assert value_difference <= 1e-12 and cost_difference <= 1e-10
        assert diagnostics['all_positive_components_retained'] and diagnostics['certificate_power_pairs'] == 6
        row = dict(statistics=statistics.tolist(), costs=costs.tolist(), diagnostics=diagnostics,
            separate_cpu_seconds=separate_cpu, shared_cpu_seconds=shared_cpu,
            cpu_speedup=separate_cpu/shared_cpu, maximum_value_difference=value_difference,
            maximum_cost_difference=cost_difference)
        if rule['name'] == 'coarse':
            row['maximum_frozen_anchor_value_difference'] = float(np.max(abs(statistics[original] - anchor['statistics'])))
            row['frozen_anchor_cost_difference'] = float(abs(costs[original] - anchor['cost']))
            assert row['maximum_frozen_anchor_value_difference'] <= 1e-12
            assert row['frozen_anchor_cost_difference'] <= 1e-10
        rows[rule['name']] = row
        save(path, dict(complete=False, identity=identity, action=action_index, kind=kind, rows=rows))
        print('SCHEDULEURM_PROGRESS ' + json.dumps(dict(current=len(rows), total=len(spec['rules']),
            unit='matched_allocation_rules', identity=identity, action=action_index, kind=kind,
            cpu_speedup=row['cpu_speedup'])), flush=True)
    result = dict(complete=True, record=spec['record'], identity=identity, action=action_index, kind=kind,
        original_allocation_position=original, actions=actions, rows=rows,
        cpu_seconds=time.process_time()-started, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        new_physical_calls=0, new_source_fits=0, new_verifier_calls=0)
    save(path, result)
    print('SCHEDULEURM_DONE ' + json.dumps({k:v for k,v in result.items() if k not in ('rows','actions')}), flush=True)
    return result


def collect(plan=PLAN):
    spec = read(plan)
    output = ROOT / spec['output']
    parts = [read(str(output / 'parts' / f'{h}_{a}_{k}.json')) for h in spec['identities']
        for a in spec['guarded_actions'] for k in ('raw','projected')]
    assert all(p['complete'] for p in parts)
    bundle = compressed_read(BASE / 'marginals' / (spec['record'] + '.json.gz'))
    historical = read(str(OUT / 'coordinate_summary.json'))
    tables = {k:np.asarray(historical['statistics']['E7'][k]).copy() for k in ('raw','projected')}
    costs = {k:np.asarray(historical['costs']['E7'][k]).copy() for k in ('raw','projected')}
    for p in parts:
        h,a,k = p['identity'],p['action'],p['kind']
        i = p['original_allocation_position']
        tables[k][h,a] = p['rows']['coarse']['statistics'][i]
        costs[k][h,a] = p['rows']['coarse']['costs'][i]
    mixed_differences = {k:float(np.max(abs(mixed_values(bundle,tables[k],costs[k],k) -
        mixed_values(bundle,historical['statistics']['E7'][k],historical['costs']['E7'][k],k))))
        for k in ('raw','projected')}
    assert max(mixed_differences.values()) <= 1e-12
    rules = {}
    for rule in spec['rules']:
        rows = [p['rows'][rule['name']] for p in parts]
        separate = sum(r['separate_cpu_seconds'] for r in rows)
        shared = sum(r['shared_cpu_seconds'] for r in rows)
        group_cpu = shared/len(spec['guarded_actions'])
        rules[rule['name']] = dict(order=rule['order'], power=rule['power'], separate_cpu_seconds=separate,
            shared_cpu_seconds=shared, cpu_speedup=separate/shared,
            minimum_part_speedup=min(r['cpu_speedup'] for r in rows), maximum_part_speedup=max(r['cpu_speedup'] for r in rows),
            maximum_value_difference=max(r['maximum_value_difference'] for r in rows),
            maximum_cost_difference=max(r['maximum_cost_difference'] for r in rows),
            mean_cpu_seconds_per_three_allocation_group=group_cpu,
            estimated_73_groups_96_records_guarded_core_hours=group_cpu*73*96/3600)
    result = dict(complete=True, record=spec['record'], parts=len(parts), rules=rules,
        maximum_frozen_anchor_value_difference=max(p['rows']['coarse']['maximum_frozen_anchor_value_difference'] for p in parts),
        maximum_frozen_anchor_cost_difference=max(p['rows']['coarse']['frozen_anchor_cost_difference'] for p in parts),
        all_identity_mixed_action_differences=mixed_differences,
        cpu_seconds=sum(p['cpu_seconds'] for p in parts), maximum_task_peak_rss_mb=max(p['peak_rss_mb'] for p in parts),
        full_predictive_integration_admitted=False, independent_confirmation=False,
        new_physical_calls=0, new_source_fits=0, new_verifier_calls=0,
        scope=spec['scope'], cost_estimate_scope='Measured two opened shortlists; extrapolation excludes direct actions, objective integrals and controller overhead.')
    save(output / 'summary.json', result)
    print('SCHEDULEURM_DONE ' + json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run','collect'])
    parser.add_argument('--identity', type=int)
    parser.add_argument('--action', type=int)
    parser.add_argument('--kind', choices=['raw','projected'])
    args = parser.parse_args()
    if args.mode == 'run':
        run(args.identity,args.action,args.kind)
    else:
        collect()

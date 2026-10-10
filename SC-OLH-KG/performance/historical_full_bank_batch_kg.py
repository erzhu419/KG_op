"""Numerical admission for full-bank replicated positive-mixture KG."""

import argparse
import copy
import json
import signal
import time

import numpy as np
from scipy.special import digamma

from core.positive_raw_lookahead import evaluate_raw_futures
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.historical_full_bank_development import context, restore_full_prior, restricted_snapshot, terminal_rule
from performance.prepare_historical_independent_kg import check_replayed_snapshot, terminal_from_initialization
from performance.run_historical_independent_kg_pilot import utility_identities
from performance.run_legacy_related_inventory_source import problem_for
from performance.diagnose_legacy_risk_transfer import no_outcomes

PLAN = 'performance/manifests/historical_full_bank_batch_kg_v1.json'
STAGES = [PLAN]+['performance/manifests/historical_full_bank_batch_kg_'+suffix+'.json'
    for suffix in ('precision_v2','bounded_utility_v2','primary_mean_v3','component_factor_v4','local_blocks_v5',
                   'raw_precision_v6')]


def restore(spec, name):
    if 'posterior_inputs' in spec:
        import pickle
        with (ROOT/spec['posterior_inputs'][name]).open('rb') as f:
            return pickle.load(f)
    base = read(spec['full_bank_plan'])
    pilot, frozen, states, public, connection, old, bank = context(base)
    state = next(s for s in states if s['name'] == name)
    paid = next(r for r in read(base['initializations'])['rows'] if r['task_id'] == state['task_id'])
    task = next(t for t in frozen['tasks'] if t['task_id'] == state['task_id'])
    model, restoration = restore_full_prior(base, pilot, state, connection, old, len(bank))
    for candidate, value in zip(paid['candidate_bank_indices'], paid['paired_observations']):
        model.update_raw(candidate, [value[0], value[1]-paid['tau']])
    check_replayed_snapshot(restricted_snapshot(model, state['candidate_bank_indices']),
        state['posterior_moment_diagnostic'], roundoff_units=max(64, 2*len(bank)))
    terminal = terminal_from_initialization(no_outcomes(problem_for(public, task)), [tuple(p) for p in bank],
        paid['paired_observations'], state['terminal_protocol'], paid_indices=paid['candidate_bank_indices'])
    return model, terminal, state, restoration


def run(name, mode, plan=PLAN):
    spec = read(plan); out = ROOT/spec['output']/mode/name; out.mkdir(parents=True, exist_ok=True)
    path = out/'results.json'
    if path.exists() and read(str(path)).get('complete'): return read(str(path))
    started = time.process_time(); handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_case_cpu_seconds'])
    result = {'name': name, 'mode': mode, 'complete': False, 'new_physical_calls': 0,
              'new_source_fits': 0, 'independent_confirmation': False, 'truth_scoring_used': False}
    try:
        model, terminal, state, restoration = restore(spec, name)
        result.update(native_subset_replay_passed=True, restoration=restoration)
        bound = spec['maximum_identity_integration_error_bound']
        if mode == 'precision':
            identities = utility_identities(model, state['arm'], bound)
            input_directory = spec.get('precision_rule_inputs', read(spec['full_bank_plan'])['output']+'/shards')
            previous = read(input_directory+'/'+name+'/terminal_rules.json')['rows']
            rows = [next(r for r in previous if r['power'] == spec['terminal_powers'][0])]
            caches = {}
            for power in spec['terminal_powers'][1:]:
                rows.append(terminal_rule(model, terminal, 360, identities, power, spec['seed'], caches))
                save(out/'terminal_rules.json', {'rows': rows})
                print('SCHEDULEURM_PROGRESS '+json.dumps({'step': len(rows)-1, 'total': len(spec['terminal_powers'])-1,
                    'unit': 'refined_rules', 'metrics': {'name': name, 'power': power,
                    'cpu_seconds': time.process_time()-started}}), flush=True)
            delta = abs(np.asarray(rows[-1]['values'])-rows[-2]['values'])
            worst = int(delta.argmax()); omitted = float(sum(w for i, w in enumerate(model.weights) if i not in identities))
            result.update(action_count=len(rows[-1]['actions']),
                maximum_action_resolution_difference=float(delta.max()), worst_action=rows[-1]['actions'][worst],
                worst_action_gap=rows[-1]['best_reward']-rows[-1]['values'][worst],
                selected_action=rows[-1]['best_action'], selected_reward=rows[-1]['best_reward'],
                choice_stable=rows[-1]['best_action_index'] == rows[-2]['best_action_index'],
                omitted_utility_mass=omitted,
                numerical_diagnostics_passed=bool(delta.max() <= spec['maximum_action_resolution_difference']
                    and omitted <= bound and rows[-1]['best_action_index'] == rows[-2]['best_action_index']),
                old_8_10_HOLD_preserved=True)
        else:
            query = spec['anchor_query']; n = spec['anchor_replicates']
            batch = next(r for r in read(spec['observations'])['rows']
                         if (r['task_id'], r['candidate']) == (state['task_id'], query))
            observations = np.asarray(batch['paired_observations'][:n]); observations[:, 1] -= batch['tau']
            labels = observations.mean(axis=0); k = (n-1)/2
            eta = np.log(observations.var(axis=0, ddof=1))-digamma(k)+np.log(k)
            future = copy.deepcopy(model); future.update_replicates(query, observations)
            ids = utility_identities(future, state['arm'], bound)
            actual, actual_costs, actual_actions = future.terminal_values(terminal, 360-n,
                spec['projected_power'], spec['seed'],utility_identities=ids)
            identity_weights = future.weights.copy(); del future
            records, actions = evaluate_raw_futures(model, terminal, 360, query, [labels],
                raw_powers=spec['raw_powers'], projected_power=spec['projected_power'], seed=spec['seed'],
                utility_identities=ids, projected_utility_identities=ids, fantasy_coordinates=[(0, 0)],
                guarded_mean_position=spec.get('guarded_mean_position'),
                replicates=n, eta_labels=[eta],
                checkpoint=lambda r, a: save(out/'future_records.json',
                    {'records': r, 'actions': a, 'cached_observation_anchor': True}))
            record = records[0]
            error = float(np.max(abs(actual-record['projected_values'])))
            weight_error = float(np.max(abs(identity_weights-record['shared_identity_weights'])))
            assert actions == actual_actions and error <= 1e-10 and weight_error <= 1e-10
            assert model.paid_episodes == 2 and len(model.history) == 2
            first, last = [record['raw'][str(p)] for p in (spec['raw_powers'][0], spec['raw_powers'][-1])]
            raw_difference = float(np.max(abs(np.asarray(first['values'])-last['values'])))
            result.update(query=query, replicates=n, future_paid_episodes=2+n, future_remaining_calls=360-n,
                guarded_mean_position=spec.get('guarded_mean_position'),
                action_count=len(actions), maximum_native_projected_action_difference=error,
                maximum_native_identity_weight_difference=weight_error,
                positive_components_per_head=restoration['positive_components_per_head'],
                maximum_raw_action_resolution_difference=raw_difference,
                predictive_log_evidence_max_error=last['predictive_log_evidence_max_error'],
                omitted_utility_mass=last['omitted_posterior_utility_mass'],
                omitted_projected_utility_mass=record['omitted_projected_utility_mass'],
                native_batch_admission_passed=True,
                raw_terminal_precision_passed=bool(raw_difference <= spec['maximum_action_resolution_difference']
                    and last['predictive_log_evidence_max_error'] <= 1e-4 and last['omitted_posterior_utility_mass'] <= bound),
                algorithmic_query_selection_complete=False,
                scope='One cached paid-batch anchor, not a predictive KG expectation or a deployed KG policy.')
        result['complete'] = True
    except TimeoutError:
        result['cpu_budget_stopped'] = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    result['cpu_seconds'] = time.process_time()-started
    save(path, result)
    print('SCHEDULEURM_DONE '+json.dumps(result), flush=True)
    return result


def collect():
    rows = []
    for plan in STAGES:
        spec = read(plan)
        for mode in ('precision', 'batch'):
            for name in spec[mode+'_states']:
                path = spec['output']+'/'+mode+'/'+name+'/results.json'
                if not (ROOT/path).exists(): raise ValueError('Numerical admission is still running: '+path)
                rows.append(dict(read(path), execution_plan=plan))
    precision = {}; batch = {}
    for row in rows:
        (precision if row['mode'] == 'precision' else batch)[row['name']] = row
    result = {'rows': rows, 'latest_precision': list(precision.values()), 'latest_batch': list(batch.values()),
        'precision_passed_count': sum(r.get('numerical_diagnostics_passed', False) for r in precision.values()),
        'native_batch_passed_count': sum(r.get('native_batch_admission_passed', False) for r in batch.values()),
        'raw_batch_precision_passed_count': sum(r.get('raw_terminal_precision_passed', False) for r in batch.values()),
        'cpu_seconds_all_admissions_including_timeouts': sum(r['cpu_seconds'] for r in rows),
        'new_physical_calls': 0, 'new_source_fits': 0, 'new_verifier_calls': 0,
        'independent_confirmation': False, 'algorithmic_query_selection_complete': False,
        'limitations': 'Three selected stopping-rule precision cases and two cached four-call batch anchors. '
            'No predictive KG expectation, complete24-case precision audit or independent deployment comparison.'}
    save(ROOT/read(PLAN)['output']/'outcome_summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['precision', 'batch', 'collect'])
    parser.add_argument('--name'); parser.add_argument('--plan', default=PLAN)
    args = parser.parse_args()
    if args.mode == 'collect':
        result = collect(); print(json.dumps({k: v for k, v in result.items() if k not in ('rows','latest_precision','latest_batch')}))
    else:
        if not args.name: parser.error('--name is required for a numerical case')
        run(args.name, args.mode, args.plan)

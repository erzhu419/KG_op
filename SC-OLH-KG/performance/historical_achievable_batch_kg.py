"""Bounded full-bank predictive KG, priced at the native recommendation."""

import argparse
import json
import signal
import time

import numpy as np

from core.positive_mixture_lookahead import predictive_positive_batches
from core.raw_terminal_head_grid import evaluate_head_records, compose_head_grid, summarize_achievable_grid
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.historical_full_bank_batch_kg import restore
from performance.historical_full_bank_development import terminal_rule
from performance.run_historical_independent_kg_pilot import utility_identities

PLAN = 'performance/manifests/historical_achievable_batch_kg_v1.json'


def run(mode, name, replicates=None, head=None, origin=None, plan=PLAN):
    spec = read(plan); out = ROOT/spec['output']/name
    if mode != 'bank': out = out/('n'+str(replicates))/mode
    if mode == 'head': out = out/(str(head)+'_'+str(origin))
    out.mkdir(parents=True, exist_ok=True); path = out/'results.json'
    if path.exists() and read(str(path)).get('complete'): return read(str(path))
    started = time.process_time(); handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_case_cpu_seconds'])
    result = dict(mode=mode, name=name, complete=False, new_physical_calls=0,
        new_source_fits=0, new_verifier_calls=0, independent_confirmation=False)
    try:
        model, terminal, state, restoration = restore(spec, name)
        ids = utility_identities(model, state['arm'], spec['maximum_identity_integration_error_bound'])
        origins = spec['origins'][state['arm']]
        result.update(native_subset_replay_passed=True, restoration=restoration)
        if mode == 'bank':
            batches = {}
            for n in spec['replicates']:
                labels, eta, weights = predictive_positive_batches(model, spec['query'], n,
                    spec['predictive_power'], spec['predictive_seed'])
                batches[str(n)] = dict(labels=labels.tolist(), eta=None if eta is None else eta.tolist())
                assert abs(weights.sum()-1) <= 1e-12
            save(out/'bank.json', dict(batches=batches, prior_weights=model.weights.tolist(),
                utility_identities=ids, origins=origins, query=spec['query'], truth_used=False))
            result.update(predictive_origins=origins, count=2**spec['predictive_power'])
        elif mode == 'baseline':
            before = spec.get('baseline_inputs')
            cached = [] if before is None else read(before+'/'+name+'/n'+str(replicates)+'/baseline/terminal_rules.json')['rows']
            rules = [r for r in cached if r['power'] in spec['baseline_powers']]; cache = {}
            for power in spec['baseline_powers']:
                if not any(r['power'] == power for r in rules):
                    rules.append(terminal_rule(model, terminal, 360-replicates, ids, power, spec['seed'], cache))
                rules.sort(key=lambda r:r['power'])
                save(out/'terminal_rules.json', dict(rows=rules))
            delta = np.max(abs(np.asarray(rules[-1]['values'])-rules[0]['values']))
            omitted = float(sum(w for i,w in enumerate(model.weights) if i not in ids))
            result.update(maximum_action_resolution_difference=float(delta), omitted_utility_mass=omitted,
                numerical_diagnostics_passed=bool(delta <= spec['maximum_action_resolution_difference']
                    and omitted <= spec['maximum_identity_integration_error_bound']
                    and rules[0]['best_action_index'] == rules[-1]['best_action_index']),
                selected_action=rules[-1]['best_action'], selected_reward=rules[-1]['best_reward'])
        else:
            bank = read(spec['output']+'/'+name+'/bank.json'); count = 2**spec['predictive_power']
            batch = bank['batches'][str(replicates)]; labels = np.asarray(batch['labels'])[:, head]
            eta = None if batch['eta'] is None else np.asarray(batch['eta'])[:, head]
            positions = list(range(origin*count, (origin+1)*count))
            old = read(str(out/'head_records.json'))['records'] if (out/'head_records.json').exists() else None
            def checkpoint(rows, actions):
                save(out/'head_records.json', dict(records=rows, actions=actions, prior_weights=model.weights.tolist()))
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(step=len(rows), total=count, unit='head_states',
                    metrics=dict(head=head, origin=origin, replicates=replicates,
                                 cpu_seconds=time.process_time()-started))), flush=True)
            rows = evaluate_head_records(model, terminal, 360, spec['query'], head, labels, positions,
                powers=spec['powers'], projected_powers=spec['powers'], seed=spec['seed'],
                completed=old, checkpoint=checkpoint, utility_identities=bank['utility_identities'],
                guarded_mean_position=0, replicates=replicates, eta_labels=eta)
            result.update(head=head, origin=origin, replicates=replicates, head_states=len(rows))
        result['complete'] = True
    except TimeoutError:
        result['cpu_budget_stopped'] = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
        result['cpu_seconds'] = time.process_time()-started; save(path, result)
    print('SCHEDULEURM_DONE '+json.dumps(result), flush=True)
    return result


def grid(heads, prior, origins, prefixes, raw_power, projected_power, baseline):
    totals = []; raw_delta = 0.; projected_delta = 0.; omitted_max = 0.
    for origin in origins:
        f, g = [heads[h, origin][:prefixes[h]] for h in (0, 1)]
        def compose(kind, power):
            return compose_head_grid(*[np.asarray([r[kind][str(power)]['statistics'] for r in rows]) for rows in (f,g)],
                np.asarray([r[kind][str(power)]['costs'] for r in g]),
                *[[r['head_log_scores'] for r in rows] for rows in (f,g)], prior)
        values, costs, posterior, _ = compose('raw', raw_power)
        projected, pc, native_posterior, _ = compose('projected', projected_power)
        np.testing.assert_allclose(posterior, native_posterior, atol=1e-14, rtol=0.)
        mass = float(prior[origin]); weights = np.full(prefixes, mass/np.prod(prefixes))
        stop, debited, fixed = baseline
        row = summarize_achievable_grid(values, costs, projected, pc, posterior, weights,
            (mass*stop, mass*debited, fixed), raw_power, sum(prefixes))
        utilities = f[0]['utility_identities']
        assert all(r['utility_identities'] == utilities for r in f+g)
        omitted = posterior[:, :, sorted(set(range(len(prior)))-set(utilities))].sum(axis=2)
        row['weighted_omitted_mass'] = float(np.sum(weights*omitted))
        omitted_max = max(omitted_max, float(omitted.max()))
        totals.append(row)
        low = min(int(p) for p in f[0]['raw'])
        raw_delta = max(raw_delta, float(np.max(abs(values-compose('raw', low)[0]))))
        projected_delta = max(projected_delta, float(np.max(abs(projected-compose('projected', low)[0]))))
    keys = ['raw_adaptation_value', 'raw_net_reward_control_variate', 'raw_adaptation_of_projected_choice',
        'raw_net_reward_of_projected_choice_control_variate', 'action_selection_loss',
        'projected_value_bias_at_selected_action', 'raw_fixed_action_tower_residual',
        'weighted_omitted_mass', 'quadrature_joint_probability_mass']
    result = {k: sum(r[k] for r in totals) for k in keys}
    expected = np.sum([r['expected_identity_weights'] for r in totals], axis=0)
    rare = float(sum(w for i,w in enumerate(prior) if i not in origins))
    # Sampling consumes the same verification budget in every predictive origin,
    # including origins left uncomputed under their original-mass bound.
    for key in ('raw_net_reward_control_variate', 'raw_net_reward_of_projected_choice_control_variate'):
        result[key] += rare*(baseline[1]-baseline[0])
    bound = result['weighted_omitted_mass']+rare
    net = result['raw_net_reward_of_projected_choice_control_variate']
    result.update(prefixes=prefixes, raw_power=raw_power, projected_power=projected_power,
        identity_error_bound=bound, omitted_predictive_origin_mass=rare,
        achieved_net_identity_interval=[net-bound, net+bound],
        maximum_future_omitted_utility_mass=omitted_max,
        maximum_raw_action_resolution_difference=raw_delta,
        maximum_native_action_resolution_difference=projected_delta,
        expected_identity_weights=expected.tolist(),
        covered_model_weight_martingale_residual=(expected-prior).tolist(),
        origin_recommendation_probabilities=[r['projected_selected_action_probabilities'] for r in totals])
    return result


def collect(plan=PLAN):
    spec = read(plan); out = ROOT/spec['output']; cases = []; executions = []
    for name in spec['states']:
        directory = spec['output']+'/'+name; bank = read(directory+'/bank.json')
        prior = np.asarray(bank['prior_weights']); origins = bank['origins']
        executions.append(read(directory+'/results.json'))
        stops = read(spec['stop_inputs'][name])['rows']; stop = stops[-1]
        stop_delta = float(np.max(abs(np.asarray(stops[-1]['values'])-stops[-2]['values'])))
        for n in spec['replicates']:
            base = directory+'/n'+str(n)
            accepted = spec.get('accepted_baseline_inputs', {}).get(name, spec['output'])+'/'+name+'/n'+str(n)
            if accepted != base: executions.append(read(base+'/baseline/results.json'))
            admission = read(accepted+'/baseline/results.json'); executions.append(admission)
            rules = read(accepted+'/baseline/terminal_rules.json')['rows']; debited = rules[-1]
            baseline = stop['best_reward'], debited['best_reward'], debited['best_action_index']
            heads = {}; actions = None
            for origin in origins:
                for h in (0, 1):
                    path = base+'/head/'+str(h)+'_'+str(origin)
                    job = read(path+'/results.json'); executions.append(job)
                    if not job['complete']: raise ValueError('Predictive head cache is incomplete: '+path)
                    data = read(path+'/head_records.json')
                    assert data['actions'] == debited['actions']
                    actions = data['actions']; heads[h,origin] = sorted(data['records'], key=lambda r:r['position'])
            small, large = spec['predictive_prefixes']; low, high = spec['powers']
            rows = [grid(heads, prior, origins, prefix, raw, native, baseline)
                for prefix,raw,native in [([small,small],high,high), ([small,large],high,high),
                    ([large,small],high,high), ([large,large],low,high),
                    ([large,large],high,low), ([large,large],high,high)]]
            fine = rows[-1]; net = fine['raw_net_reward_of_projected_choice_control_variate']
            changes = [abs(net-r['raw_net_reward_of_projected_choice_control_variate']) for r in rows[:-1]]
            norm = max(r['raw'][str(high)]['predictive_log_evidence_max_error']
                for head in heads.values() for r in head)
            gates = dict(baseline=admission['numerical_diagnostics_passed']
                and stop_delta <= spec['maximum_action_resolution_difference']
                and stops[-2]['best_action_index'] == stop['best_action_index'],
                predictive=max(changes) <= spec['maximum_numerical_adaptation_difference'],
                raw_action=fine['maximum_raw_action_resolution_difference'] <= spec['maximum_action_resolution_difference'],
                native_action=fine['maximum_native_action_resolution_difference'] <= spec['maximum_action_resolution_difference'],
                evidence=norm <= 1e-4,
                identity=fine['identity_error_bound'] <= spec['maximum_identity_integration_error_bound']
                    and fine['maximum_future_omitted_utility_mass'] <= spec['maximum_identity_integration_error_bound'])
            cases.append(dict(name=name, query=spec['query'], replicates=n, action_count=len(actions),
                future_paid_episodes=2+n, future_remaining_calls=360-n,
                stop_reward=baseline[0], debited_reward=baseline[1],
                verification_budget_opportunity_cost=baseline[0]-baseline[1], rows=rows,
                predictive_and_terminal_changes=changes, predictive_log_evidence_max_error=norm,
                gates=gates, numerical_diagnostics_passed=all(gates.values()),
                recommendation_frozen=False, independent_confirmation=False))
    result = dict(cases=cases, executions=executions, all_executions_complete=all(r['complete'] for r in executions),
        cpu_seconds=sum(r['cpu_seconds'] for r in executions), new_physical_calls=0,
        new_source_fits=0, new_verifier_calls=0, independent_confirmation=False,
        algorithmic_query_selection_complete=False, query_truth_used=False,
        scope='One frozen query on an opened development task, complete365-action class. '
              'Predictive pilot; no independent deployed advantage or complete query search.')
    save(out/'outcome_summary.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','executions')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['bank', 'baseline', 'head', 'collect'])
    parser.add_argument('--name'); parser.add_argument('--replicates', type=int)
    parser.add_argument('--head', type=int); parser.add_argument('--origin', type=int)
    parser.add_argument('--plan', default=PLAN); args = parser.parse_args()
    if args.mode == 'collect': collect(args.plan)
    else: run(args.mode, args.name, args.replicates, args.head, args.origin, args.plan)

"""Native recommendation precision and raw-law loss at cached batch anchors."""

import argparse
import json
import signal
import time

import numpy as np

from core.raw_terminal_head_grid import compose_head_grid
from core.conditional_terminal_kg import ConditionalTerminalActions
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.historical_full_bank_batch_kg import restore
from performance.historical_full_bank_development import terminal_rule
from performance.run_historical_independent_kg_pilot import utility_identities
from performance.historical_batch_head_precision import action_key

PLAN = 'performance/manifests/historical_batch_native_precision_v6.json'


def raw_choice_diagnostic(setup, fine, weights):
    source = read(setup['raw_plan']); name = setup['state']
    f = read(source['head_inputs']['0']+'/'+name+'/0/head_records.json')
    g = read(source['output']+'/'+name+'/1/head_records.json')
    fr, gr = f['records'][0], g['records'][0]
    fp, gp = [source['head_powers'][str(h)][-1] for h in (0, 1)]
    values, _, posterior, _ = compose_head_grid(
        np.asarray(fr['raw'][str(fp)]['statistics'])[None],
        np.asarray(gr['raw'][str(gp)]['statistics'])[None],
        np.asarray(gr['raw'][str(gp)]['costs'])[None],
        [fr['head_log_scores']], [gr['head_log_scores']], f['prior_weights'])
    np.testing.assert_allclose(posterior[0, 0], weights, atol=1e-12, rtol=0.)
    assert fine['actions'] == g['actions']
    values = values[0, 0]; best = int(values.argmax()); selected = fine['best_action_index']
    return dict(raw_optimal_action=fine['actions'][best], raw_optimal_value=float(values[best]),
        raw_value_of_native_choice=float(values[selected]),
        raw_reward_loss_of_native_choice=float(values[best]-values[selected]),
        maximum_raw_projected_action_difference=float(np.max(abs(values-fine['values']))),
        scope='Fixed observed batch; projection-choice diagnostic, not predictive KG or deployment benefit.')


def run(case, plan=PLAN, shard=None):
    spec = read(plan); setup = next(c for c in spec['cases'] if c['name'] == case)
    out = ROOT/spec['output']/case
    if shard is not None: out = out/('shard'+str(shard))
    out.mkdir(parents=True, exist_ok=True)
    if (out/'results.json').exists() and read(str(out/'results.json')).get('complete'):
        return read(str(out/'results.json'))
    start = time.process_time(); handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_case_cpu_seconds'])
    result = dict(name=case, state=setup['state'], mode=setup['mode'], complete=False,
        new_physical_calls=0, new_source_fits=0, independent_confirmation=False)
    try:
        model, terminal, state, restoration = restore(spec, setup['state'])
        if shard is not None:
            selected = np.arange(shard, len(terminal.shortlists), spec['shards'])
            terminal = ConditionalTerminalActions(terminal.shortlists[selected], terminal.incumbents[selected], terminal.protocol)
        remaining = 360
        if setup['mode'] == 'batch':
            batch = next(r for r in read(spec['observations'])['rows'] if
                (r['task_id'], r['candidate']) == (state['task_id'], spec['query']))
            observations = np.asarray(batch['paired_observations'][:spec['replicates']])
            observations[:, 1] -= batch['tau']
            model.update_replicates(spec['query'], observations); remaining -= spec['replicates']
        ids = utility_identities(model, state['arm'], spec['maximum_identity_integration_error_bound'])
        cache = {}; rules = []
        for power in spec['powers']:
            rules.append(terminal_rule(model, terminal, remaining, ids, power, spec['seed'], cache))
            save(out/'terminal_rules.json', {'rows': rules})
        fine = rules[-1]; delta = abs(np.asarray(fine['values'])-rules[0]['values'])
        omitted = float(sum(w for i,w in enumerate(model.weights) if i not in ids))
        stable = fine['best_action_index'] == rules[0]['best_action_index']
        result.update(complete=True, native_subset_replay_passed=True, restoration=restoration,
            powers=spec['powers'], paid_episodes=model.paid_episodes, remaining_calls=remaining,
            action_count=len(fine['actions']), maximum_action_resolution_difference=float(delta.max()),
            omitted_utility_mass=omitted, selected_action=fine['best_action'],
            selected_value=fine['best_reward'], choice_stable=stable,
            native_precision_passed=bool(delta.max() <= spec['maximum_action_resolution_difference']
                and omitted <= spec['maximum_identity_integration_error_bound'] and stable))
        result['future_weights'] = model.weights.tolist()
        if setup['mode'] == 'batch' and shard is None:
            result.update(raw_choice_diagnostic(setup, fine, model.weights))
    except TimeoutError:
        result['cpu_budget_stopped'] = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
        result['cpu_seconds'] = time.process_time()-start; save(out/'results.json', result)
    print('SCHEDULEURM_DONE '+json.dumps(result), flush=True)
    return result


def collect(case, plan):
    spec = read(plan); setup = next(c for c in spec['cases'] if c['name'] == case)
    out = ROOT/spec['output']/case
    jobs = [read(str((out/('shard'+str(i))/'results.json').relative_to(ROOT))) for i in range(spec['shards'])]
    assert all(r['complete'] for r in jobs)
    inputs = [read(str((out/('shard'+str(i))/'terminal_rules.json').relative_to(ROOT)))['rows'] for i in range(spec['shards'])]
    raw = read(setup['raw_plan'])
    actions = read(raw['output']+'/'+setup['state']+'/1/head_records.json')['actions']
    positions = {action_key(a): i for i,a in enumerate(actions)}; rules = []
    for p, power in enumerate(spec['powers']):
        values = np.empty(len(actions)); costs = np.empty(len(actions)); covered = []
        for bank in inputs:
            row = bank[p]; assert row['power'] == power
            indices = [positions[action_key(a)] for a in row['actions']]
            values[indices] = row['values']; costs[indices] = row['costs']; covered.extend(indices)
        assert sorted(covered) == list(range(len(actions)))
        best = int(np.lexsort((costs, -values))[0])
        rules.append(dict(power=power, actions=actions, values=values.tolist(), costs=costs.tolist(),
            best_action_index=best, best_action=actions[best], best_reward=float(values[best])))
    save(out/'terminal_rules.json', {'rows': rules}); fine = rules[-1]
    delta = float(np.max(abs(np.asarray(fine['values'])-rules[0]['values'])))
    stable = fine['best_action_index'] == rules[0]['best_action_index']
    assert all(r['future_weights'] == jobs[0]['future_weights'] for r in jobs)
    result = dict(jobs[0], shards=jobs, maximum_action_resolution_difference=delta,
        action_count=len(actions), choice_stable=stable, selected_action=fine['best_action'],
        selected_value=fine['best_reward'], cpu_seconds=sum(r['cpu_seconds'] for r in jobs),
        native_precision_passed=bool(delta <= spec['maximum_action_resolution_difference'] and stable
            and jobs[0]['omitted_utility_mass'] <= spec['maximum_identity_integration_error_bound']),
        **raw_choice_diagnostic(setup, fine, jobs[0]['future_weights']))
    save(out/'results.json', result); print(json.dumps(result), flush=True); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--case', required=True)
    parser.add_argument('--plan', default=PLAN); parser.add_argument('--shard', type=int)
    parser.add_argument('--collect', action='store_true'); args = parser.parse_args()
    if args.collect: collect(args.case, args.plan)
    else: run(args.case, args.plan, args.shard)

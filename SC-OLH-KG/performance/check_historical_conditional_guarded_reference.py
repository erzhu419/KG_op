"""Check the unchanged actual guarded incumbent with an independent reference."""

import signal
import time

from core.conditional_fixed_reference import fixed_guarded_projected_tower
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import build_connection

PLAN = 'performance/manifests/historical_conditional_guarded_reference_v1.json'


def run(plan=PLAN):
    spec = read(plan); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out/'results.json'; checkpoint_path = out/'heads.json'
    previous = read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all_registered_rules_complete'):
        return previous
    heads = read(str(checkpoint_path.relative_to(ROOT)))['heads'] if checkpoint_path.exists() else []
    if not heads and 'completed_reference' in spec:
        completed = read(spec['completed_reference']); assert completed['all_registered_rules_complete']
        parent_spec = read(completed['execution_plan'])
        for key in ('connection_plan', 'terminal_results', 'layout', 'candidate', 'terminal_power', 'seed'):
            assert spec[key] == parent_spec[key]
        heads = read(spec['completed_heads'])['heads']
    started = time.process_time(); rows = []; stopped = False
    connection, _, models = build_connection(spec['connection_plan'])
    assert connection['all_registered_cases_complete']
    state, model, terminal, remaining = next(r for r in models if r[0]['layout'] == spec['layout'])
    source = read(spec['terminal_results'])['rows']
    baseline = next(r for r in source if (r['layout'], r['remaining_calls'], r['power']) ==
        (spec['layout'], remaining-1, spec['terminal_power']))
    action = baseline['best_action']; assert action['kind'] == 'guarded' and len(model.components) == 8
    identity_powers = None; retained_mass = 0.
    if 'retained_identity_indices' in spec:
        retained = spec['retained_identity_indices']
        retained_mass = float(sum(model.weights[i] for i in retained))
        assert retained_mass <= spec['maximum_retained_identity_expectation_error']
        identity_powers = {i: spec['retained_fantasy_power'] if i in retained else max(spec['fantasy_powers'])
            for i in range(len(model.components))}
        parent_heads = read(spec['completed_heads'])
        assert (parent_heads['layout'], parent_heads['action'], parent_heads['terminal_power']) == (
            spec['layout'], action, spec['terminal_power'])
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds']-(time.process_time()-started)))
    def checkpoint(current):
        nonlocal heads
        heads = current
        save(checkpoint_path, {'execution_plan': plan, 'layout': spec['layout'], 'action': action,
            'terminal_power': spec['terminal_power'], 'identity_weights': model.weights.tolist(), 'heads': heads})
        print('guarded head', current[-1]['identity_name'], current[-1]['head'],
            'fantasies', len(current[-1]['future_statistics']), 'complete', len(heads),
            'CPU', round(time.process_time()-started, 2), flush=True)
    try:
        rows, heads = fixed_guarded_projected_tower(model, action, spec['candidate'],
            terminal_power=spec['terminal_power'], fantasy_powers=spec['fantasy_powers'],
            seed=spec['seed'], protocol=terminal.protocol, completed_heads=heads, checkpoint=checkpoint,
            identity_fantasy_powers=identity_powers)
        for row in rows:
            assert abs(row['prior_fixed_reward']-baseline['best_reward']) < 1e-11
            row.update(layout=spec['layout'], same_rule_prior_reference_error=abs(row['prior_fixed_reward']-baseline['best_reward']))
        if 'completed_reference' in spec:
            old_rows = read(spec['completed_reference'])['rows']
            for row in rows:
                old = next((r for r in old_rows if r['fantasy_power'] == row['fantasy_power']), None)
                if old is not None:
                    row['completed_prefix_reward_difference'] = abs(
                        row['expected_future_fixed_reward']-old['expected_future_fixed_reward'])
                    assert row['completed_prefix_reward_difference'] < 1e-12
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    result = {'execution_plan': plan, 'rows': rows, 'completed_heads': len(heads),
        'all_registered_rules_complete': len(rows) == len(spec['fantasy_powers']),
        'cpu_seconds': previous.get('cpu_seconds', 0.)+time.process_time()-started,
        'cpu_seconds_this_execution': time.process_time()-started, 'cpu_budget_stopped': stopped,
        'previous_cpu_budget_stop': previous.get('cpu_budget_stopped', False),
        'identity_weights': model.weights.tolist(), 'identity_names': model.names,
        'retained_identity_expectation_error_bound': retained_mass,
        'cost': spec['cost'], 'scope': spec['scope'], 'independent_confirmation': False,
        'exact_likelihood_drift_isolated': False}
    save(path, result); return result


if __name__ == '__main__':
    result = run()
    print({key: result[key] for key in ('all_registered_rules_complete', 'completed_heads', 'cpu_seconds', 'cpu_budget_stopped')})

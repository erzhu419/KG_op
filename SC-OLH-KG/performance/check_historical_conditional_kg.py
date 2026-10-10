"""Run a bounded actual adaptive KG using the corrected deployment integrator."""

import signal
import time

import numpy as np

from core.conditional_terminal_kg import ConditionalTerminalActions
from core.mixture_guarded_kg import mixture_raw_lookahead
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import build_connection

PLAN = 'performance/manifests/historical_conditional_kg_pilot_v1.json'


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists() and read(str(path.relative_to(ROOT))).get('all_registered_cases_complete'):
        return read(str(path.relative_to(ROOT)))
    started = time.process_time(); rows = []; stopped = False
    reference_path = spec.get('fixed_action_reference', spec.get('fixed_direct_tower'))
    reference_power = spec.get('fixed_action_fantasy_power', spec.get('fixed_direct_fantasy_power'))
    prefix = None
    if 'completed_prefix' in spec:
        assert read(spec['efficiency_results'])['all_cases_equivalent']
        assert read(spec['paid_anchor_results'])['all_anchors_equivalent']
        saved = read(spec['completed_prefix']); assert saved['all_registered_cases_complete']
        previous_plan = read(saved['execution_plan'])
        for key in ('connection_plan', 'terminal_results', 'layout', 'candidate', 'terminal_power', 'seed'):
            assert spec[key] == previous_plan[key]
        assert reference_path == previous_plan.get('fixed_action_reference', previous_plan.get('fixed_direct_tower'))
        assert reference_power == previous_plan.get('fixed_action_fantasy_power', previous_plan.get('fixed_direct_fantasy_power'))
        prefix = saved['rows'][0].copy()
        prefix['fantasies_per_component'] = 2**prefix['joint_fantasy_power']
        assert prefix['fantasies_per_component'] <= 2**(spec['joint_fantasy_power']-1)
    connection, _, models = build_connection(spec['connection_plan'])
    assert connection['all_registered_cases_complete']
    state, model, parent, remaining = next(r for r in models if r[0]['layout'] == spec['layout'])
    terminal = ConditionalTerminalActions(parent.shortlists, parent.incumbents, parent.protocol)
    registered = read(spec['terminal_results'])['rows']
    before, after = [next(r for r in registered if (r['layout'], r['remaining_calls'], r['power']) ==
        (spec['layout'], budget, spec['terminal_power'])) for budget in (remaining, remaining - 1)]
    fixed = next(r for r in read(reference_path)['rows'] if r['fantasy_power'] == reference_power)
    assert (fixed['candidate'], fixed['action']) == (spec['candidate'], after['best_action'])
    assert abs(fixed['prior_fixed_reward'] - after['best_reward']) < 1e-11
    baseline = (np.array(before['values']), np.array(before['costs']), before['actions'],
        np.array(after['values']), after['actions'])
    old = next(r for r in read(spec['old_kg_results'])['rows']
        if (r['layout'], r['candidate']) == (spec['layout'], spec['candidate']))
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds'] - (time.process_time() - started)))
    try:
        result = mixture_raw_lookahead(model, terminal, remaining, [spec['candidate']],
            draw_power=spec['terminal_power'], fantasy_power=spec['joint_fantasy_power'],
            seed=spec['seed'], baseline=baseline,
            fixed_action_expectations={spec['candidate']: fixed['expected_future_fixed_reward']}, completed_prefix=prefix)
        rows.append({'layout': spec['layout'], 'joint_fantasy_power': spec['joint_fantasy_power'],
            'terminal_power': spec['terminal_power'], 'fixed_reference_fantasy_power': reference_power,
            'stop_expected_reward': result['stop_expected_reward'], 'stop_action': result['stop_action'],
            'provisional_choice': result['choice'], **result['lookahead'][0],
            'old_coarse_net_reward': old['net_reward_of_paid_episode'],
            'old_coarse_adaptation_value': old['adaptation_value'],
            'completed_prefix_artifact': spec.get('completed_prefix'),
            'adaptive_maximum_evaluated': True, 'terminal_action_count': len(before['actions']),
            'all_eight_complete_identities_retained': len(model.components) == 8})
        assert model.paid_episodes == len(state['paid_observations'])
        print('conditional KG pilot', spec['layout'], spec['candidate'], rows[0], flush=True)
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    result = {'execution_plan': plan, 'rows': rows, 'all_registered_cases_complete': len(rows) == 1,
        'cpu_seconds': time.process_time() - started, 'cpu_budget_stopped': stopped,
        'cost': spec['cost'], 'scope': spec['scope'], 'decision_ranking_certified': False,
        'independent_confirmation': False}
    save(path, result); return result


if __name__ == '__main__':
    result = run()
    print({k: result[k] for k in ('all_registered_cases_complete', 'cpu_seconds', 'cpu_budget_stopped')})

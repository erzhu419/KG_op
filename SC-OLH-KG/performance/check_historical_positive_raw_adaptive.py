"""Frozen positive-mixture raw adaptive pilot with resumable action values."""

import signal
import time

import numpy as np

from core.conditional_terminal_kg import ConditionalTerminalActions
from core.positive_raw_lookahead import evaluate_raw_futures, summarize_raw_futures
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import build_connection

PLAN = 'performance/manifests/historical_positive_raw_adaptive_v1.json'


def run(plan=PLAN):
    spec = read(plan); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out/'results.json'; checkpoint_path = out/'futures.json'
    previous = read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all_registered_cases_complete'):
        return previous
    records = read(str(checkpoint_path.relative_to(ROOT)))['records'] if checkpoint_path.exists() else []
    started = time.process_time(); stopped = False; rows = []
    phase = read(spec['raw_reference_phase'])
    assert phase['next_episode_projection_difference_resolved'] and phase['actual_field_anchors_equivalent']
    _, _, models = build_connection(spec['connection_plan'])
    state, model, parent, remaining = next(r for r in models if r[0]['layout'] == spec['layout'])
    terminal = ConditionalTerminalActions(parent.shortlists, parent.incumbents, parent.protocol)
    bank = read(spec['projected_heads']); np.testing.assert_array_equal(model.weights, bank['identity_weights'])
    count = 2**spec['fantasy_power']
    fantasies = []
    for identity in range(len(model.components)):
        pair = [next(r for r in bank['heads'] if (r['identity'], r['head']) == (identity, head)) for head in (0, 1)]
        fantasies.extend(zip(pair[0]['predictive_labels'][:count], pair[1]['predictive_labels'][:count]))
    source = read(spec['terminal_results'])['rows']
    before, after = [next(r for r in source if (r['layout'], r['remaining_calls'], r['power']) ==
        (spec['layout'], budget, spec['projected_draw_power'])) for budget in (remaining, remaining-1)]
    baseline = (before['best_reward'], after['best_reward'], after['best_action_index'])
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds']-(time.process_time()-started)))
    def checkpoint(current, actions):
        nonlocal records
        records = current
        save(checkpoint_path, {'execution_plan': plan, 'source_weights': model.weights.tolist(),
            'actions': actions, 'records': records})
        print('raw adaptive future', len(records), '/', len(fantasies), 'CPU', round(time.process_time()-started, 2), flush=True)
    try:
        records, actions = evaluate_raw_futures(model, terminal, remaining, spec['candidate'], fantasies,
            raw_powers=spec['raw_draw_powers'], projected_power=spec['projected_draw_power'],
            seed=spec['seed'], completed_records=records, checkpoint=checkpoint)
        assert actions == after['actions']
        rows = summarize_raw_futures(records, model.weights, count, actions, baseline, spec['raw_draw_powers'])
        old = read(spec['old_projected_pilot'])['rows'][0]
        for row in rows:
            error = max(abs(row['projected_adaptation_value']-old['adaptation_value']),
                abs(row['projected_raw_joint_expected_future_reward']-old['raw_joint_expected_future_reward']))
            assert error <= spec['maximum_projected_pilot_reproduction_error']
            row['old_projected_pilot_reproduction_error'] = error
        assert model.paid_episodes == len(state['paid_observations'])
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    result = {'execution_plan': plan, 'rows': rows, 'completed_futures': len(records),
        'all_registered_cases_complete': len(rows) == len(spec['raw_draw_powers']),
        'cpu_seconds': previous.get('cpu_seconds', 0.)+time.process_time()-started,
        'cpu_budget_stopped': stopped, 'shared_identity_and_all_source_components_retained': True,
        'all40_actions_evaluated': bool(len(rows) and rows[-1]['terminal_action_count'] == 40),
        'previous_paid_projection_retained': True, 'independent_confirmation': False,
        'decision_ranking_certified': False, 'cost': spec['cost'], 'scope': spec['scope']}
    save(path, result); return result


if __name__ == '__main__':
    r = run()
    print({k: r[k] for k in ('all_registered_cases_complete', 'completed_futures', 'cpu_seconds', 'cpu_budget_stopped')})

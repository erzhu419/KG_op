"""Extend only the cached GP query whose ordinary predictive stratum is open."""

import signal
import time

import numpy as np
from scipy.stats import qmc

from core.conditional_terminal_kg import ConditionalTerminalActions
from core.positive_mixture_lookahead import predictive_values
from core.positive_raw_lookahead import evaluate_raw_futures
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_matched_gp import refine_primary_records, summarize_query
from performance.connect_historical_prior_posterior import snapshot
from representation.joint_response_mixture import JointResponseMixture
from representation.joint_function_risk import JointFunctionRiskPosterior

PLAN = 'performance/manifests/historical_matched_gp_extension_v2.json'


def restore_saved_model(row):
    """This declared control has one Gaussian per response; restore it exactly."""
    state = row['posterior']; assert len(state['components']) == 1
    component = state['components'][0]
    pair = tuple(JointFunctionRiskPosterior.from_moments(component[f'{h}_mean'],
        component[f'{h}_covariance']) for h in ('f', 'g'))
    for field in pair:
        field.paid_batches = field.paid_episodes = state['paid_episodes']
    model = JointResponseMixture(state['names'], [pair], prior_weights=state['weights'])
    model.paid_batches = model.paid_episodes = state['paid_episodes']; model.history = state['history']
    restored = snapshot(model)
    assert restored == state
    return model


def run(plan=PLAN):
    spec = read(plan); parent = read(spec['parent_plan']); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out/'results.json'; previous = read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all_registered_cases_complete'):
        return previous
    old_result = read(spec['parent_results']); assert old_result['all_unit_tests_successful']
    started = time.process_time(); stopped = False; summary = {}
    original = read(spec['completed_futures']); checkpoint_path = out/'futures.json'
    records = read(str(checkpoint_path.relative_to(ROOT)))['records'] if checkpoint_path.exists() else original['records']
    layout = spec['layout']; row = next(r for r in read(spec['connection_states'])['rows'] if
        (r['layout'], r['arm']) == (layout, parent['lookahead_arm']))
    model = restore_saved_model(row); query = parent['queries'][layout]
    state = next(r for r in read(read(parent['connection_plan'])['development_states'])['rows'] if r['layout'] == layout)
    guarded = state['terminal_actions'][:8]
    terminal = ConditionalTerminalActions([a['shortlist'] for a in guarded],
        [a['incumbent_position'] for a in guarded], state['terminal_protocol'])
    rule = dict(parent, **{k: spec[k] for k in ('predictive_power', 'tail_counts', 'complement_prefixes')})
    uniforms = qmc.Sobol(2, scramble=True, seed=rule['seed']+1).random_base2(rule['predictive_power'])
    positions = np.union1d(np.flatnonzero(uniforms[:, 1] < rule['low_constraint_probability'])[:max(rule['tail_counts'])],
        np.flatnonzero(uniforms[:max(rule['complement_prefixes']), 1] >= rule['low_constraint_probability']))
    def checkpoint(records, actions):
        save(checkpoint_path, dict(original, records=records, actions=actions, execution_plan=plan))
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_single_thread_cpu_seconds'])
    try:
        fantasies, _ = predictive_values(model, query, rule['predictive_power'], rule['seed']+1)
        for record in original['records']:
            np.testing.assert_allclose(record['labels'], fantasies[record['identity_fantasy_position']], atol=1e-14, rtol=1e-12)
        records, actions = evaluate_raw_futures(model, terminal, 360, query, fantasies[positions],
            raw_powers=rule['raw_powers'], projected_power=rule['raw_powers'][-1], seed=rule['seed'],
            completed_records=records, checkpoint=checkpoint, fantasy_coordinates=[(0, int(p)) for p in positions])
        refine_primary_records(model, terminal, 360, query, records, actions, rule['raw_powers'], rule['seed'], checkpoint)
        old = {r['identity_fantasy_position']: r for r in original['records']}
        assert all(r == old[r['identity_fantasy_position']] for r in records if r['identity_fantasy_position'] in old)
        summary = summarize_query(records, model, row['terminal'], actions, uniforms, rule)
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    result = {'execution_plan': plan, 'layout': layout, 'candidate': query, **summary,
        'all_registered_cases_complete': bool(summary), 'original22_futures_preserved': bool(summary),
        'original_source_fits_and_paid_state_preserved': True,
        'new_joint_futures': len(records)-len(original['records']), 'reused_joint_futures': len(original['records']),
        'total_joint_futures': len(records), 'cpu_seconds': previous.get('cpu_seconds', 0.)+time.process_time()-started,
        'cpu_budget_stopped': stopped, 'cost': dict(parent['cost'], new_source_fits=0),
        'independent_confirmation': False, 'decision_ranking_certified': False,
        'limitations': ['Adjacent fantasy/terminal resolutions are numerical diagnostics, not certified error bounds.',
            'GP and meta-prior KG values are expectations under different models, not a deployment-benefit estimate.',
            'Only the two opened queries have controls; complete query policies and independent decisions remain open.']}
    save(path, result)
    print({k: result[k] for k in ('all_registered_cases_complete', 'new_joint_futures', 'cpu_seconds', 'cpu_budget_stopped')})
    if summary:
        print(summary['rows'][-1]['raw_net_reward_control_variate'], summary['numerical_changes'], summary['numerical_gates_passed'])
    return result


if __name__ == '__main__':
    run()

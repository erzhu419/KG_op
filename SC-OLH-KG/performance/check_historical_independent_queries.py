"""Reuse native head grids for all frozen independent model/query states."""

import signal
import time

from performance.check_historical_remaining_queries import evaluate_case, validate_anchors
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.prepare_historical_independent_kg import restore_target_models

PLAN = 'performance/manifests/historical_independent_queries_v1.json'


def registered_queries(states):
    return [dict(name=f"{s['name']}_query{q}", state_name=s['name'], task_id=s['task_id'],
        layout=s['layout'], arm=s['arm'], query=q) for s in states for q in s['queries']]


def cohort_status(states, cases):
    """Missing or HOLD queries cannot produce a ready model policy."""
    expected = registered_queries(states); names = {r['name'] for r in expected}
    assert len(names) == len(expected) and len({c['name'] for c in cases}) == len(cases)
    assert all(c['name'] in names for c in cases)
    complete = len(cases) == len(expected)
    ready = []
    for state in states:
        rows = [c for c in cases if c['state_name'] == state['name']]
        if len(rows) == len(state['queries']) and all(c.get('adaptation_diagnostics_passed', False) for c in rows):
            ready.append(state['name'])
    return {'registered_query_cases': len(expected), 'initial_query_cases_complete': len(cases),
        'all_initial_cases_complete': complete,
        'all_registered_cases_complete': complete and all(c.get('processing_complete', False) for c in cases),
        'passed_query_cases': sum(c.get('adaptation_diagnostics_passed', False) for c in cases),
        'ready_model_states': ready, 'all_model_states_ready': len(ready) == len(states)}


def run(plan=PLAN):
    started = time.process_time(); spec = read(plan); pilot = read(spec['pilot_plan'])
    numerical = read(pilot['numerical_plan']); preflight = read(spec['preflight'])
    baselines = read(spec['baselines']); states = preflight['states']
    assert preflight['all_density_and_rare_origin_prechecks_passed']
    assert baselines['all_baseline_diagnostics_passed'] and baselines['passed_model_states'] == len(states)
    assert not preflight['independent_outcomes_opened'] and not baselines['independent_outcomes_opened']
    assert read(spec['unit_tests'])['successful']
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); path = out/'results.json'
    previous = read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all_registered_cases_complete'):
        return previous
    cases = list(previous.get('cases', [])); stopped = False
    descriptors = registered_queries(states)
    assert len(descriptors) == pilot['new_task_count']*len(pilot['models'])*2*6 == 144
    frozen = {'pilot_plan': spec['pilot_plan'], 'queries': descriptors,
        'source_preflight': spec['preflight'], 'source_baselines': spec['baselines']}
    registration = out/'registered_queries.json'
    if registration.exists():
        assert read(str(registration.relative_to(ROOT))) == frozen
    else:
        save(registration, frozen)
    rule = dict(numerical, **spec['numerical_overrides'])
    parent = dict(pilot, maximum_anchor_branch_error=pilot['maximum_anchor_branch_error'])
    state_models = {}; baseline_by_name = {c['name']: c for c in baselines['cases']}

    def publish():
        result = dict(cohort_status(states, cases), execution_plan=plan, cases=cases,
            cpu_seconds=previous.get('cpu_seconds', 0.)+time.process_time()-started,
            cpu_budget_stopped=stopped,
            cost={'new_source_fits': 0, 'new_physical_calls': 0, 'new_search_calls': 0,
                'new_verifier_calls': 0, 'server_jobs': 0},
            independent_outcomes_opened=False, independent_confirmation=False,
            decision_ranking_certified=False,
            limitations=['Adjacent-rule and native-anchor checks do not certify strict query-ranking error bounds.',
                'Preserve raw fixed-action tower HOLD diagnostics and earlier paid moment projections.'],
            next='Complete all144 diagnostics before query decisions;freeze all48 final recommendations before independent verification.')
        save(path, result); return result

    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds']-
        previous.get('cpu_seconds', 0.)-(time.process_time()-started)))
    try:
        pending_states = [s for s in states if not all(any(c['name'] == f"{s['name']}_query{q}" and
            c.get('processing_complete', False) for c in cases) for q in s['queries'])]
        for state, model, terminal in restore_target_models(pilot, pending_states):
            state_models[state['name']] = state, model, terminal
        for descriptor in descriptors:
            if any(c['name'] == descriptor['name'] for c in cases):
                continue
            state, model, _ = state_models[descriptor['state_name']]
            baseline = baseline_by_name[state['name']]['baseline']
            cached = ROOT/spec['preflight_output']/descriptor['name']
            bank = read(str((cached/'label_banks.json').relative_to(ROOT)))
            check = next(c for c in preflight['cases'] if c['name'] == descriptor['name'])
            assert check['density_precheck_passed'] and check['rare_origin_precheck_passed']
            assert bank['source_weights'] == model.weights.tolist() == state['source_weights']
            identities, proposal = ([4, 5], 4) if state['arm'] == 'source_structural_mixture' else ([0], 0)
            directory = out/descriptor['name']; directory.mkdir(exist_ok=True)
            summary = evaluate_case(model, state, 360, baseline, bank, directory, identities, proposal,
                rule['initial_raw_powers'], dict(rule, current_query=descriptor['query']), parent)
            save(directory/'initial_results.json', summary)
            current = dict(descriptor, baseline=baseline, source_weights=model.weights.tolist(),
                bank=str((cached/'label_banks.json').relative_to(ROOT)), stages=[summary])
            cases.append(current); publish()
            print('independent initial', descriptor['name'], 'grid', summary['grid_diagnostics_passed'],
                'changes', summary['numerical_changes'], 'CPU', round(time.process_time()-started, 2), flush=True)
        # Keep every initial case visible before refining or accepting any policy.
        for current in cases:
            if current.get('processing_complete', False):
                continue
            state, model, _ = state_models[current['state_name']]
            baseline = baseline_by_name[state['name']]['baseline']
            directory = out/current['name']; identities, proposal = ([4, 5], 4) if state['arm'] == 'source_structural_mixture' else ([0], 0)
            summary = current['stages'][-1]
            if not summary['grid_diagnostics_passed'] and summary['raw_powers'] != rule['final_raw_powers']:
                summary = evaluate_case(model, state, 360, baseline, read(current['bank']), directory, identities,
                    proposal, rule['final_raw_powers'], dict(rule, current_query=current['query']), parent)
                current['stages'].append(summary); save(directory/'refined_results.json', summary); publish()
            anchors = validate_anchors(model, state, 360, baseline, directory, identities, current['query'],
                summary['raw_powers'], rule)
            current.update(processing_complete=True, anchors=anchors,
                adaptation_diagnostics_passed=summary['grid_diagnostics_passed'] and anchors['all_passed'])
            save(directory/'results.json', current); publish()
            print('independent final', current['name'], 'net', summary['rows'][-1]['raw_net_reward_control_variate'],
                'pass', current['adaptation_diagnostics_passed'], 'tower', summary['fixed_action_tower_gate_passed'],
                'CPU', round(time.process_time()-started, 2), flush=True)
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    return publish()


if __name__ == '__main__':
    result = run(); print('independent cohort', result['initial_query_cases_complete'],
        result['passed_query_cases'], 'CPU', round(result['cpu_seconds'], 2), flush=True)

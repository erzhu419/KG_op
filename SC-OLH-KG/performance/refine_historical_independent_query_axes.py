"""Apply historical risk-only and positive-tail refinements to new targets."""

import argparse
import json
import signal
import time
import numpy as np

from core.raw_terminal_head_grid import evaluate_head_records
from performance.check_historical_gauss_risk_tail import summarize
from performance.check_historical_independent_queries import cohort_status, registered_queries
from performance.check_historical_raw_head_grid import terminal_for
from performance.check_historical_remaining_queries import validate_anchors
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.diagnose_historical_independent_query_axes import terminal_axes, refinement_axes
from performance.prepare_historical_independent_kg import restore_target_models
from performance.refine_historical_remaining_query_failures import positive_risk_tail

PLAN = 'performance/manifests/historical_independent_query_completion_v2.json'


def grid_gates(summary, pilot):
    rows = summary['rows']; lo, hi = pilot['predictive_density_ratio_limits']
    return {'terminal_and_predictive_resolution': summary['numerical_gates_passed'],
        'identity_remainder': max(r['identity_and_rare_origin_error_bound'] for r in rows) <= pilot['maximum_identity_integration_error_bound'],
        'predictive_mass': max(abs(r['predictive_mass_residual']) for r in rows) <= pilot['maximum_predictive_mass_residual'],
        'density_ratio': all(lo <= r['predictive_density_ratio_range'][0] and r['predictive_density_ratio_range'][1] <= hi for r in rows)}


def merge_cases(states, original, refinements):
    """Replacing a numerical result retains the complete original registration."""
    expected = registered_queries(states); by_name = {c['name']: c for c in original}
    if len(by_name) != len(original) or set(by_name) != {d['name'] for d in expected}:
        raise ValueError('All original registered query cases must be available before refinement')
    replacement = {c['name']: c for c in refinements}
    if len(replacement) != len(refinements) or not set(replacement) <= set(by_name):
        raise ValueError('Duplicate or unregistered independent query refinement')
    result = []
    for descriptor in expected:
        old = by_name[descriptor['name']]; current = replacement.get(old['name'], old)
        if any(current[k] != v for k, v in descriptor.items()):
            raise ValueError('Query refinement changed the target/model/query identity')
        if old.get('adaptation_diagnostics_passed', False) and current is not old:
            raise ValueError('An accepted original query does not need replacement')
        result.append(current)
    return result


def compact(case):
    fine = case['stages'][-1]['rows'][-1]; stage = case['stages'][-1]
    return {k: case[k] for k in ('name', 'state_name', 'task_id', 'layout', 'arm', 'query')} | {
        'processing_complete': case.get('processing_complete', False),
        'adaptation_diagnostics_passed': case.get('adaptation_diagnostics_passed', False),
        'grid_diagnostics_passed': stage['grid_diagnostics_passed'],
        'numerical_changes': stage['numerical_changes'], 'estimated_net_kg': fine['raw_net_reward_control_variate'],
        'raw_fixed_action_tower_residual': fine['raw_fixed_action_tower_residual'],
        'fixed_action_tower_gate_passed': stage['fixed_action_tower_gate_passed'],
        'raw_net_identity_interval': fine['raw_net_control_variate_identity_interval']}


def run(plan=PLAN):
    started = time.process_time(); spec = read(plan); parent = read(spec['parent_plan'])
    pilot = read(parent['pilot_plan']); rule = dict(read(pilot['numerical_plan']), **parent['numerical_overrides'])
    original = read(parent['output']+'/results.json'); preflight = read(parent['preflight']); states = preflight['states']
    if not original['all_initial_cases_complete']:
        raise ValueError('Finish all original144 initial cases before selective axis refinement')
    merge_cases(states, original['cases'], [])
    assert preflight['all_density_and_rare_origin_prechecks_passed'] and not original['independent_outcomes_opened']
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); path = out/'results.json'
    previous = read(str(path)) if path.exists() else {}; cases = list(previous.get('cases', [])); stopped = False
    def publish():
        merged = merge_cases(states, original['cases'], cases)
        result = dict(cohort_status(states, merged), execution_plan=plan, cases=cases,
            cpu_seconds=previous.get('cpu_seconds', 0.)+time.process_time()-started, cpu_budget_stopped=stopped,
            cost={'new_source_fits': 0, 'new_physical_calls': 0, 'new_verifier_calls': 0},
            independent_outcomes_opened=False, independent_confirmation=False)
        save(path, result)
        save(out/'phase_results.json', dict(result, cases=[compact(c) for c in merged],
            raw_fixed_action_tower_hold_cases=[c['name'] for c in merged if not c['stages'][-1]['fixed_action_tower_gate_passed']],
            limitations=['Paired resolution and native anchors remain diagnostics, not certified ranking bounds.',
                        'Earlier paid projections and independent deployment advantage remain open.']))
        return result
    pending = [c for c in original['cases'] if not c.get('adaptation_diagnostics_passed', False)
        and (not spec.get('case_names') or c['name'] in spec['case_names'])
        and not any(r['name'] == c['name'] and r.get('processing_complete', False) for r in cases)]
    workers = spec.get('workers', 1)
    state_names = {c['state_name'] for c in pending}
    ordered_states = sorted([s for s in states if s['name'] in state_names],
        key=lambda s: not all(c['stages'][-1]['grid_diagnostics_passed']
            for c in original['cases'] if c['state_name'] == s['name']))
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds']-
        previous.get('cpu_seconds', 0.)-(time.process_time()-started)))
    try:
        for state, model, _ in restore_target_models(pilot, ordered_states):
            for old in sorted([c for c in pending if c['state_name'] == state['name']],
                    key=lambda c: not c['stages'][-1]['grid_diagnostics_passed']):
                before = ROOT/parent['output']/old['name']; directory = out/old['name']; directory.mkdir(exist_ok=True)
                f = read(str(before/'head0.json'))['records']; old_g = read(str(before/'head1.json'))
                hp = directory/'head1.json'; records = read(str(hp))['records'] if hp.exists() else old_g['records']
                bank = read(old['bank']); stage = old['stages'][-1]
                if stage['grid_diagnostics_passed']:
                    anchors = validate_anchors(model, state, 360, old['baseline'], before,
                        [4, 5] if old['arm'] == 'source_structural_mixture' else [0],
                        old['query'], stage['raw_powers'], rule, workers=workers)
                    current = dict(old, processing_complete=True, anchors=anchors,
                        adaptation_diagnostics_passed=anchors['all_passed'])
                    save(directory/'results.json', current); cases.append(current); publish()
                    print('independent cached-grid anchors', old['name'], 'pass', current['adaptation_diagnostics_passed'], flush=True)
                    continue
                split = terminal_axes(old, bank, [f, old_g['records']], rule)
                axes = refinement_axes(stage['numerical_changes'], split, pilot['maximum_numerical_adaptation_difference'])
                if not axes or set(axes)-{'terminal_risk', 'risk_tail_order'}:
                    print('independent HOLD requires another axis or anchor', old['name'], axes, flush=True)
                    continue
                identities, proposal = ([4, 5], 4) if old['arm'] == 'source_structural_mixture' else ([0], 0)
                assert old['source_weights'] == bank['source_weights']
                np.testing.assert_allclose(model.weights, old['source_weights'],
                    rtol=64*np.finfo(float).eps, atol=0.)
                risk_rules = dict(bank['risk_rules']); labels = dict(zip(bank['heads'][1]['positions'], bank['heads'][1]['labels']))
                row_path = directory/'results.json'; current = read(str(row_path)) if row_path.exists() else dict(
                    {k: old[k] for k in ('name', 'state_name', 'task_id', 'layout', 'arm', 'query', 'baseline', 'source_weights')},
                    original_case=str((before/'results.json').relative_to(ROOT)), axis_diagnosis=split, stages=[])
                def checkpoint(records, actions):
                    assert actions == state['future_terminal_actions']
                    hp.write_text(json.dumps({'records': records, 'actions': actions}, separators=(',', ':'))+'\n')
                def evaluate(powers, tail_orders):
                    processing_rule = dict(rule, raw_powers=powers, objective_raw_powers=stage['raw_powers'],
                        risk_tail_orders=tail_orders, proposal_origin=proposal, predictive_origins=identities)
                    summary = summarize([f, records], risk_rules, model.weights, processing_rule,
                        objective_rules=bank['objective_rules'], baseline=old['baseline'])
                    gates = grid_gates(summary, pilot)
                    summary.update(gates=gates, grid_diagnostics_passed=all(gates.values()),
                        raw_powers=powers, objective_raw_powers=stage['raw_powers'], risk_tail_orders=tail_orders)
                    current['stages'].append(summary); save(row_path, current)
                    return summary
                powers = stage['raw_powers']; summary = stage
                if 'terminal_risk' in axes:
                    higher_powers = [p for p in spec['risk_terminal_powers'] if p > powers[-1]]
                    for lower, higher in zip([powers[-1]]+higher_powers, higher_powers):
                        powers = [lower, higher]
                        records = evaluate_head_records(model, terminal_for(state), 360, old['query'], 1,
                            labels, bank['heads'][1]['positions'], powers=[higher], seed=rule['seed'], completed=records,
                            checkpoint=checkpoint, utility_identities=identities, guarded_mean_position=0, workers=workers)
                        summary = next((s for s in current['stages'] if s['raw_powers'] == powers
                            and s['risk_tail_orders'] == rule['risk_tail_orders']), None)
                        if summary is None:
                            summary = evaluate(powers, rule['risk_tail_orders'])
                        if summary['numerical_changes']['terminal_absolute_contrast'] <= pilot['maximum_numerical_adaptation_difference']:
                            break
                if summary['numerical_changes']['risk_tail_order'] > pilot['maximum_numerical_adaptation_difference']:
                    lp = directory/'risk_tail64.json'
                    added = read(str(lp)) if lp.exists() else positive_risk_tail(model, old['query'], proposal, rule,
                        order=spec['risk_tail_orders'][-1], position=max(labels)+1)
                    save(lp, added); labels.update(zip(added['positions'], added['labels'])); risk_rules['64'] = added
                    records = evaluate_head_records(model, terminal_for(state), 360, old['query'], 1,
                        labels, added['positions'], powers=powers, seed=rule['seed'], completed=records,
                        checkpoint=checkpoint, utility_identities=identities, guarded_mean_position=0, workers=workers)
                    summary = next((s for s in current['stages'] if s['raw_powers'] == powers
                        and s['risk_tail_orders'] == spec['risk_tail_orders']), None)
                    if summary is None:
                        summary = evaluate(powers, spec['risk_tail_orders'])
                anchors = validate_anchors(model, state, 360, old['baseline'], directory, identities, old['query'], powers, rule,
                    workers=workers)
                current.update(processing_complete=True, anchors=anchors,
                    adaptation_diagnostics_passed=summary['grid_diagnostics_passed'] and anchors['all_passed'])
                save(row_path, current); cases.append(current); publish()
                print('independent axis refinement', old['name'], 'pass', current['adaptation_diagnostics_passed'], flush=True)
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    return publish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default=PLAN)
    args = parser.parse_args()
    started = time.monotonic()
    print('SCHEDULEURM_PROGRESS ' + json.dumps({'current': 0, 'total': 1, 'unit': 'query',
        'elapsed_seconds': 0}), flush=True)
    result = run(args.plan)
    if not all(c.get('processing_complete', False) for c in result['cases']) or not result['cases']:
        raise SystemExit('Query processing incomplete; saved numerical work remains resumable')
    print('SCHEDULEURM_PROGRESS ' + json.dumps({'current': 1, 'total': 1, 'unit': 'query',
        'elapsed_seconds': time.monotonic()-started}), flush=True)
    print('DONE', flush=True)

"""Locate numerical HOLD axes by recomposing saved native head statistics."""

import time

import numpy as np
from scipy.stats import qmc

from core.raw_terminal_head_grid import summarize_proposal_head_grid
from performance.complete_historical_source_heads import ROOT, read, save

PLAN = 'performance/manifests/historical_independent_queries_completion_v2.json'


def terminal_axes(case, bank, heads, rule):
    stage = case['stages'][-1]; low, high = stage['raw_powers']
    index = [{r['position']: r for r in records} for records in heads]
    f_rule = bank['objective_rules'][str(rule['objective_prefixes'][-1])]
    g_rule = bank['risk_rules'][str(rule['risk_tail_orders'][-1])]
    u = qmc.Sobol(2, scramble=True, seed=rule['seed']+1).random_base2(8)
    body = np.flatnonzero(u[:rule['risk_complement_prefixes'][-1], 1] >= rule['low_constraint_probability'])
    f = [index[0][p] for p in f_rule['positions']]
    g = [index[1][p] for p in g_rule['positions']+body.tolist()]
    g_weights = np.r_[g_rule['weights'], np.full(len(body), (1-rule['low_constraint_probability'])/len(body))]
    source = case['arm'] == 'source_structural_mixture'
    def compose(fp, gp):
        return summarize_proposal_head_grid(f, g, f_rule['weights'], g_weights, case['source_weights'],
            4 if source else 0, [4, 5] if source else [0], power=gp,
            objective_power=fp, baseline=case['baseline'])
    fine = compose(high, high)
    weights = np.asarray(fine['joint_quadrature_weights'])
    contrast = np.asarray(fine['weighted_adaptive_contrasts'])
    def difference(other):
        return float(np.sum(weights*abs(contrast-np.asarray(other['weighted_adaptive_contrasts']))))
    together = difference(compose(low, low))
    if abs(together-stage['numerical_changes']['terminal_absolute_contrast']) > 1e-14:
        raise ValueError('Saved terminal resolution does not reproduce from its native head cache')
    return {'objective_only_difference': difference(compose(low, high)),
        'risk_only_difference': difference(compose(high, low)), 'joint_difference': together,
        'cached_head_powers': [low, high], 'new_native_integrations': 0}


def refinement_axes(changes, split, tolerance):
    axes = []
    if changes['terminal_absolute_contrast'] > tolerance:
        # Reserve a tenth of the unchanged gate for the objective head that stays fixed.
        if split['objective_only_difference'] <= tolerance/10:
            axes.append('terminal_risk')
        elif split['risk_only_difference'] <= tolerance/10:
            axes.append('terminal_objective')
        else:
            axes.append('terminal_both_heads')
    for key in ('objective_prefix', 'risk_complement_prefix', 'risk_tail_order'):
        if changes[key] > tolerance:
            axes.append(key)
    return axes


def run(plan=PLAN):
    started = time.process_time(); spec = read(plan); pilot = read(spec['pilot_plan'])
    rule = dict(read(pilot['numerical_plan']), **spec['numerical_overrides'])
    cohort = read(spec['output']+'/results.json'); cases = []
    for case in cohort['cases']:
        stage = case['stages'][-1]
        if stage['grid_diagnostics_passed']:
            continue
        directory = ROOT/spec['output']/case['name']
        heads = [read(str(directory/f'head{h}.json'))['records'] for h in (0, 1)]
        split = terminal_axes(case, read(case['bank']), heads, rule)
        axes = refinement_axes(stage['numerical_changes'], split, pilot['maximum_numerical_adaptation_difference'])
        cases.append({k: case[k] for k in ('name', 'state_name', 'task_id', 'layout', 'arm', 'query')} | {
            'numerical_changes': stage['numerical_changes'], 'terminal_axis_resolution': split,
            'refinement_axes': axes, 'grid_gates': stage['gates'],
            'historical_risk_only_or_tail_method_applicable': bool(axes) and set(axes) <= {'terminal_risk', 'risk_tail_order'}})
    result = {'execution_plan': plan, 'initial_cases_observed': len(cohort['cases']),
        'all_initial_cases_complete': cohort['all_initial_cases_complete'], 'cases': cases,
        'grid_hold_count': len(cases), 'historical_method_applicable_count':
            sum(c['historical_risk_only_or_tail_method_applicable'] for c in cases),
        'cached_diagnosis_cpu_seconds': time.process_time()-started,
        'cost': {'new_native_integrations': 0, 'new_physical_calls': 0, 'new_verifier_calls': 0, 'server_jobs': 0},
        'independent_outcomes_opened': False,
        'next': 'Complete the original registered diagnostics before selective new-task risk/tail refinement.'}
    save(ROOT/spec['output']/'axis_diagnosis.json', result)
    print('cached axes', len(cases), 'HOLD;', result['historical_method_applicable_count'],
        'use historical risk/tail method;CPU', round(result['cached_diagnosis_cpu_seconds'], 3), flush=True)
    return result


if __name__ == '__main__':
    run()

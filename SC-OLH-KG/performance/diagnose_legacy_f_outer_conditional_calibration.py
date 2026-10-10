"""Compare already-scored outer and inner calibration without new forecasts."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_inner_conditional_calibration import local_calibration
from performance.diagnose_legacy_f_level_shape_mean_error import decompose
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save

EXECUTION = 'performance/manifests/legacy_joint_population_f_outer_conditional_calibration_execution_v1.json'


def summarize(sites):
    return {key: float(np.mean([site[key] for site in sites])) for key in sites[0]}


def calibration(filename, labels):
    sites = []
    with np.load(ROOT / filename) as state:
        assert state['component_means'].shape[1] == len(labels) == 12
        for j, label in enumerate(labels):
            record, _ = local_calibration(state['component_means'][:, j],
                state['component_local_covariances'][:, j], state['log_weights'], label[0], label[3])
            sites.append(record)
    return sites


def run():
    started = time.process_time(); spec = read(EXECUTION)
    coverage, _ = source_paths(read(spec['source_context_execution']))
    inner = read(spec['inner_calibration_results']); assert inner['diagnostic_interpreted']
    cases = []
    for key in ('control_pilot_results', 'q1_pilot_results', 'learned_pilot_results'):
        source = read(spec[key]); assert source['scientific_gate_interpreted']
        folder = Path(source['execution']['output'])
        for row in source['rows']:
            if row['design'] not in spec['designs']: continue
            if row['arm'] not in spec['structural_arms'] and not (
                row['arm'] == 'source_gp' and row['design'] == 'intercept_mixture_greedy'): continue
            if row['arm'] == 'source_variance_mixture' and row['design'] == 'fixed_public_prefix':
                selected = Path(spec['retained_fixed_component_directory']) / row['component_marginals_file']
                refined = selected.with_name(selected.stem + '_refined.npz')
            else:
                selected = folder / row['component_marginals_file']
                refined = folder / row['refined_component_marginals_file'] if row['arm'] != 'source_gp' else None
            cases.append((row, selected, refined))
    assert len(cases) == 14
    rows, checks = [], []
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    base = {'execution': spec, 'cost': spec['cost'], 'no_model_adopted': True,
        'no_new_unscored_outcomes_read': True, 'source6_7_evaluated': False,
        'target_promoted': False, 'independent_confirmation': False, 'training_diagnostic_only': True}
    for row, selected, refined in cases:
        if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']:
            save(out / 'results.json', {**base, 'rows': rows, 'diagnostic_interpreted': False,
                'stopped_at_cpu_budget': True, 'cpu_seconds': time.process_time() - started}); return
        layout, design, arm = (row[key] for key in ('layout', 'design', 'arm'))
        labels = np.asarray(read(Path(coverage['output']) / f'layout{layout}' / 'inputs.json')['source_tasks'][1]['response'])
        assert row['scoring_sites'] == coverage['layouts'][layout][1]
        sites = calibration(selected, labels); average = summarize(sites)
        density_error = abs(average['joint_batch_density'] - row['scores']['f_raw_batch_density'])
        assert density_error <= spec['cached_density_tolerance']
        breakdown = decompose(row['prediction']['mean'], labels[:, 0])
        assert abs(breakdown['mse'] - row['scores']['f_mse']) <= spec['cached_mse_tolerance']
        if refined is not None:
            reference = calibration(refined, labels)
            error = {
                'conditional_mean_max_absolute': max(abs(a['conditional_mean'] - b['conditional_mean']) for a, b in zip(sites, reference)),
                'variance_relative': max(abs(a['conditional_batch_mean_variance'] / b['conditional_batch_mean_variance'] - 1) for a, b in zip(sites, reference)),
                'cdf_max_absolute': max(abs(a['conditional_cdf'] - b['conditional_cdf']) for a, b in zip(sites, reference)),
                'gradient_max_absolute': max(abs(a['log_conditional_f_variance_score_gradient'] - b['log_conditional_f_variance_score_gradient']) for a, b in zip(sites, reference))}
            checks.append({'layout': layout, 'design': design, 'arm': arm, 'errors': error,
                'passed': all(error[k] <= v for k, v in spec['refinement_tolerances'].items())})
        rows.append({'layout': layout, 'design': design, 'arm': arm, 'selected_component_file': str(selected),
            'sites': sites, 'summary': average, 'unconditional_mean_error': breakdown,
            'cached_joint_score_error': density_error})
        save(out / 'partial_results.json', {**base, 'rows': rows, 'refinement_checks': checks})
        print('outer calibration', layout, design, arm, {k: average[k] for k in (
            'squared_standardized_error', 'covered_90', 'log_conditional_f_variance_score_gradient')}, flush=True)
    assert len(checks) == 12
    comparisons, density_contrasts = [], []
    for row in rows:
        layout, arm = row['layout'], row['arm']
        baseline_arm = {'source_predictive_shape_mixture': 'shape_mixture', 'source_gp': 'source_gp'}.get(arm)
        if baseline_arm is not None:
            baseline = inner['summary'][layout][baseline_arm]
            before, after = baseline['log_conditional_f_variance_score_gradient'], row['summary']['log_conditional_f_variance_score_gradient']
            comparisons.append({'layout': layout, 'design': row['design'], 'arm': arm,
                'inner_fixed_prefix_summary': baseline, 'outer_summary': row['summary'],
                'local_variance_score_direction_reversed': bool(before * after < 0)})
        if arm == 'source_predictive_shape_scale_mixture':
            for control in ('source_variance_mixture', 'source_predictive_shape_mixture'):
                matched = next(r for r in rows if r['layout'] == layout and r['design'] == row['design'] and r['arm'] == control)
                differences = {k: row['summary'][k] - matched['summary'][k] for k in (
                    'joint_batch_density', 'variance_label_density', 'conditional_mean_density')}
                density_contrasts.append({'layout': layout, 'design': row['design'], 'reference_arm': control,
                    'density_differences': differences, 'mse_differences': {k: row['unconditional_mean_error'][k] - matched['unconditional_mean_error'][k]
                        for k in ('mse', 'level_mse', 'centered_shape_mse')}})
    verified = all(check['passed'] for check in checks)
    save(out / 'results.json', {**base, 'rows': rows, 'refinement_checks': checks,
        'inner_outer_comparisons': comparisons, 'learned_scale_contrasts': density_contrasts,
        'diagnostic_interpreted': verified, 'cpu_seconds': time.process_time() - started,
        'interpretation': 'Future-variance-conditioned diagnostics at already-scored sites;local within-component f|eta variance tangent fixes means,eta loading and weights. Inner/outer comparison differs in tasks,source size and sites;not a causal task-shift test or a new prior.'})
    print('outer calibration complete', verified, 'CPU', time.process_time() - started, flush=True)


if __name__ == '__main__':
    run()

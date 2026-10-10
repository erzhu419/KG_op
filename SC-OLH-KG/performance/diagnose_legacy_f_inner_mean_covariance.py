"""Matched source4 mean/covariance swaps at the frozen four-query prefix."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_reference_training_budget import compact, prediction, score
from performance.check_legacy_f_source_information_design import gaussian_field
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_level_shape_mean_error import decompose
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates, source_prior_moments
from representation.source_kernel_population import predict_unequal_site_kernel_population
from representation.vectorized_joint_function_risk import update_raw_observation

EXECUTION = 'performance/manifests/legacy_joint_population_f_inner_mean_covariance_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION)
    assert read(spec['exact_mixture_results'])['diagnostic_interpreted']
    controls = read(spec['control_results']); assert controls['diagnostic_interpreted']
    fitting = read(spec['source_fit_results']); assert fitting['all_inner_fits_converged']
    coverage, _ = source_paths(read(spec['source_context_execution']))
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    branches = []
    for folder in controls['execution']['source_packages']:
        branches += read(Path(folder) / 'source_branches.json')['rows']
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    fields, frozen, prior_checks = [], [], []
    for branch in branches:
        layout, query, sites = branch['layout'], branch['query_task'], branch['scoring_sites']
        inputs = read(Path(coverage['output']) / f'layout{layout}' / 'task1' / 'inputs.json')
        psi = (public_control_mean_coordinates(no_outcomes(problem_for(public['plan'], public['tasks'][query])), bank)
            - inputs['mean_coordinate_center']) / inputs['mean_coordinate_scale']
        np.testing.assert_allclose(psi[coverage['layouts'][layout][query]], branch['query_mean_psi'], atol=1e-12, rtol=1e-12)
        record = next(r for r in fitting['rows'] if r['layout'] == layout and r['inner_query_task'] == f'related_inventory_{query:02d}')
        fitted = read(record['source_fit']); assert fitted['inner_training_task_names'] == branch['source_task_names']
        assert 'related_inventory_01' not in branch['source_task_names'] and f'related_inventory_{query:02d}' not in branch['source_task_names']
        selection = spec['paid_prefix'] + sites
        basis = OrthogonalLowOrderBasis(branch['public_mean_reference'])
        native = source_prior_moments(restore_joint(basis, fitted['fits']['structural']), psi[selection])
        gp = predict_unequal_site_kernel_population(psi[selection], fitted['fits']['gp'])
        for name, values in [('structural', native), ('source_gp', gp)]:
            old = record['prediction'][name]
            errors = {'mean_max_absolute': float(np.max(np.abs(values[0][4:] - np.asarray(old['mean'])[4:]))),
                'covariance_relative_frobenius': float(np.linalg.norm(values[1][4:, 4:] - np.asarray(old['covariance'])[4:, 4:]) /
                    np.linalg.norm(np.asarray(old['covariance'])[4:, 4:]))}
            assert all(errors[k] <= v for k, v in spec['prior_tolerances'].items())
            prior_checks.append({'layout': layout, 'query_task': query, 'arm': name, 'errors': errors})
        eta = np.asarray(branch['eta_mean'])[selection], np.asarray(branch['eta_covariance'])[np.ix_(selection, selection)]
        for arm, f in [('structural_mean_gp_covariance', (native[0], gp[1])),
                       ('gp_mean_structural_covariance', (gp[0], native[1]))]:
            row = {'layout': layout, 'query_task': query, 'design': 'fixed_public_prefix', 'arm': arm,
                'episode_count': 4, 'paid_prefix': spec['paid_prefix'], 'scoring_sites': sites}
            fields.append({'row': row, 'field': gaussian_field(f, eta)})
            frozen.append({**row, 'source_task_names': branch['source_task_names'], 'same_source4_eta': True})
    assert len(fields) == 20 and len(prior_checks) == 20
    save(out / 'frozen_inputs.json', {'execution': spec, 'rows': frozen, 'prior_checks': prior_checks,
        'all_twenty_swapped_priors_frozen_before_paid_values': True})
    raw = read(coverage['source_data'])
    paid = {b['query_task']: [raw['tasks'][b['query_task']]['Y_replicates'][j][0][0] for j in spec['paid_prefix']] for b in branches}; del raw
    rows = []
    for work in fields:
        row, field = work['row'], work.pop('field')
        evidence = sum(update_raw_observation(field, j, float(y)) for j, y in enumerate(paid[row['query_task']]))
        state = compact(field, list(range(4, 12)), evidence)
        filename = f"layout{row['layout']}_task{row['query_task']}_{row['arm']}.npz"
        np.savez_compressed(out / filename, **state)
        row.update(prediction=prediction(state), component_file=filename); rows.append(row)
    save(out / 'predictions.json', {'execution': spec, 'rows': rows, 'all_twenty_forecasts_frozen_before_score_labels': True})
    base = {'execution': spec, 'cost': spec['cost'], 'no_model_adopted': True, 'source6_7_evaluated': False,
        'target_promoted': False, 'independent_confirmation': False, 'training_diagnostic_only': True,
        'no_new_unscored_outcomes_read': True, 'prior_checks': prior_checks}
    for row in rows:
        if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']:
            save(out / 'results.json', {**base, 'rows': rows, 'diagnostic_interpreted': False,
                'stopped_at_cpu_budget': True, 'cpu_seconds': time.process_time() - started}); return
        inputs = read(Path(coverage['output']) / f"layout{row['layout']}" / 'task1' / 'inputs.json')
        task = next(t for t in inputs['source_tasks'] if t['name'] == f"related_inventory_{row['query_task']:02d}")
        labels = {k: np.asarray(task[k])[4:] for k in ('response', 'noise')}
        with np.load(out / row['component_file']) as state: row['scores'], row['density_integration_error_bound'] = score(row, state, labels)
        row['mean_error'] = decompose(row['prediction']['mean'], labels['response'][:, 0])
        comparisons = {}
        for control in ('native_point', 'source_gp'):
            reference = next(r for r in controls['rows'] if r['layout'] == row['layout'] and r['query_task'] == row['query_task']
                and r['arm'] == control and r['design'] == row['design'])
            assert reference['paid_prefix'] == row['paid_prefix'] and reference['scoring_sites'] == row['scoring_sites']
            differences = {k: row['scores'][k] - reference['scores'][k] for k in spec['metrics']}
            comparisons[control] = {'differences': differences, 'mse_relative_change': row['scores']['f_mse'] / reference['scores']['f_mse'] - 1}
        row['comparisons'] = comparisons
    summary = {layout: {arm: {control: {
        'mean_differences': {k: float(np.mean([r['comparisons'][control]['differences'][k] for r in rows if r['layout'] == layout and r['arm'] == arm])) for k in spec['metrics']},
        'mse_improved_tasks': sum(r['comparisons'][control]['differences']['f_mse'] < 0 for r in rows if r['layout'] == layout and r['arm'] == arm)}
        for control in ('native_point', 'source_gp')} for arm in spec['arms']} for layout in spec['layouts']}
    save(out / 'results.json', {**base, 'rows': rows, 'summary': summary, 'diagnostic_interpreted': True,
        'cpu_seconds': time.process_time() - started, 'component_artifact_bytes': sum(p.stat().st_size for p in out.glob('*.npz')),
        'interpretation': 'Controlled swaps between two source4 Gaussian priors at identical queries and eta law. Each swap is a valid Gaussian prior,but it is not a coherent fitted structural population law or a new adopted method. The inner tasks participated in earlier development;not independent confirmation.'})
    print('inner mean/covariance swaps', summary, 'CPU', time.process_time() - started, flush=True)


if __name__ == '__main__':
    run()

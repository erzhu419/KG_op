"""Compress both source-population kernels before source conditioning."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_reference_training_budget import compact, prediction, score
from performance.check_legacy_f_source_information_design import gaussian_field
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_coherent_common_compression import condition
from performance.diagnose_legacy_f_level_shape_mean_error import decompose
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates
from representation.source_kernel_population import rbf_cross, shared_kernel
from representation.vectorized_joint_function_risk import update_raw_observation

EXECUTION = 'performance/manifests/legacy_joint_population_f_inner_joint_compression_execution_v1.json'


def source_laws(tasks, reference, query, fitted, tail_multiplier):
    gp, native = fitted['fits']['gp'], fitted['fits']['structural']
    basis = OrthogonalLowOrderBasis(reference)
    design = lambda x: np.column_stack([np.ones(len(x)), basis.transform(x)])
    d, ds, dq = design(reference), np.vstack([design(t['psi']) for t in tasks]), design(query)
    xs = np.vstack([t['features'] for t in tasks]); np.testing.assert_array_equal(xs, gp['source_features'])
    anchor = gp['anchor']; scale = anchor['normalization']['scale']; scale2 = scale**2
    center = anchor['normalization']['mean'] + scale * anchor['constant']
    amplitude, intercept, discrepancy = gp['residual_parameters']; white = scale2 * discrepancy
    noise = [np.asarray(t['noise'])[:, 0] for t in tasks]
    error = np.concatenate([np.asarray(t['response'])[:, 0] for t in tasks]) - center
    residual = block_diag(*[scale2 * (amplitude * rbf_cross(np.asarray(t['features']), np.asarray(t['features']), anchor['lengthscale']) + intercept)
        + np.diag(v + white) for t, v in zip(tasks, noise)])
    nll, alpha, factor = condition(scale2 * shared_kernel(xs, xs, anchor), residual, error)
    np.testing.assert_allclose(nll / len(error), gp['source_negative_log_likelihood_per_label'] + np.log(scale), atol=1e-8, rtol=1e-8)
    cross = scale2 * shared_kernel(query, xs, anchor)
    mean = center + cross @ alpha
    covariance = (scale2 * (shared_kernel(query, query, anchor) + amplitude * rbf_cross(query, query, anchor['lengthscale']) + intercept)
        + white * np.eye(len(query)) - cross @ cho_solve(factor, cross.T))
    common = scale2 * shared_kernel(reference, reference, anchor)
    task = scale2 * (amplitude * rbf_cross(reference, reference, anchor['lengthscale']) + intercept)
    active = (np.asarray(native['common_group_variances']) > 0) | (np.asarray(native['source_group_variances']) > 0)
    scopes = {'joint_full_dictionary': list(range(d.shape[1])),
        'joint_native_support': [0] + [j + 1 for j, group in enumerate(basis.group_index) if active[group]]}
    tail = np.diag(np.r_[0., np.full(basis.feature_dim, tail_multiplier * native['tail_variance_budget'] / basis.feature_dim)])
    beta0 = np.r_[center, np.zeros(basis.feature_dim)]; laws = []
    for arm, columns in scopes.items():
        z = d[:, columns]; operator, _, rank, _ = np.linalg.lstsq(z, np.eye(len(z)), rcond=None)
        assert rank == len(columns)
        a, r = tail.copy(), tail.copy()
        a[np.ix_(columns, columns)] += operator @ common @ operator.T
        r[np.ix_(columns, columns)] += operator @ task @ operator.T
        a, r = .5 * (a + a.T), .5 * (r + r.T)
        assert min(np.linalg.eigvalsh(a).min(), np.linalg.eigvalsh(r).min()) >= -1e-10
        v = block_diag(*[design(t['psi']) @ r @ design(t['psi']).T + np.diag(n + white) for t, n in zip(tasks, noise)])
        value, solved, cf = condition(ds @ a @ ds.T, v, error)
        beta = beta0 + a @ ds.T @ solved
        h = a - a @ ds.T @ cho_solve(cf, ds @ a); h = .5 * (h + h.T)
        assert np.linalg.eigvalsh(h + r).min() >= -1e-10
        laws.append({'arm': arm, 'columns': columns, 'common_prior_coefficient_covariance': a.tolist(),
            'task_coefficient_covariance': r.tolist(), 'source_posterior_coefficient_mean': beta.tolist(),
            'source_posterior_common_covariance': h.tolist(), 'weak_tail_per_coefficient': float(tail[1, 1]),
            'white_discrepancy': float(white), 'source_nll_per_label': value / len(error),
            'source_total_nll_change_vs_gp': value - nll,
            'f': (dq @ beta, dq @ (h + r) @ dq.T + white * np.eye(len(query)))})
    return laws, (mean, covariance)


def run(execution=EXECUTION):
    started = time.process_time(); spec = read(execution)
    assert read(spec['swap_results'])['diagnostic_interpreted']
    controls, fitting = read(spec['control_results']), read(spec['source_fit_results'])
    assert controls['diagnostic_interpreted'] and fitting['all_inner_fits_converged']
    coverage, _ = source_paths(read(spec['source_context_execution']))
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    branches = sum([read(Path(p) / 'source_branches.json')['rows'] for p in controls['execution']['source_packages']], [])
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True); fields, frozen = [], []
    for branch in branches:
        layout, query, sites = branch['layout'], branch['query_task'], branch['scoring_sites']
        inputs = read(Path(coverage['output']) / f'layout{layout}' / 'task1' / 'inputs.json')
        tasks = [t for t in inputs['source_tasks'] if t['name'] in branch['source_task_names']]
        assert len(tasks) == 4 and [t['name'] for t in tasks] == branch['source_task_names']
        assert f'related_inventory_{query:02d}' not in branch['source_task_names'] and 'related_inventory_01' not in branch['source_task_names']
        record = next(r for r in fitting['rows'] if r['layout'] == layout and r['inner_query_task'] == f'related_inventory_{query:02d}')
        fitted = read(record['source_fit']); assert fitted['inner_training_task_names'] == branch['source_task_names']
        psi = (public_control_mean_coordinates(no_outcomes(problem_for(public['plan'], public['tasks'][query])), bank)
            - inputs['mean_coordinate_center']) / inputs['mean_coordinate_scale']
        selection = spec['paid_prefix'] + sites
        np.testing.assert_allclose(psi[coverage['layouts'][layout][query]], branch['query_mean_psi'], atol=1e-12, rtol=1e-12)
        laws, gp = source_laws(tasks, np.asarray(branch['public_mean_reference']), psi[selection], fitted, spec['weak_tail_multiplier'])
        for key, actual in zip(('mean', 'covariance'), gp):
            np.testing.assert_allclose(actual[4:] if key == 'mean' else actual[4:, 4:], np.asarray(record['prediction']['source_gp'][key])[4:] if key == 'mean'
                else np.asarray(record['prediction']['source_gp'][key])[4:, 4:], atol=1e-8, rtol=1e-6)
        eta = np.asarray(branch['eta_mean'])[selection], np.asarray(branch['eta_covariance'])[np.ix_(selection, selection)]
        for law in laws:
            f = law.pop('f'); row = {'layout': layout, 'query_task': query, 'arm': law['arm'], 'design': 'fixed_public_prefix',
                'episode_count': 4, 'paid_prefix': spec['paid_prefix'], 'scoring_sites': sites}
            fields.append({'row': row, 'field': gaussian_field(f, eta)})
            frozen.append({**row, **law, 'source_task_names': branch['source_task_names']})
    assert len(fields) == 20
    save(out / 'source_laws.json', {'execution': spec, 'rows': frozen, 'all_twenty_source_laws_frozen_before_paid_values': True,
        'all_ten_full_gp_source_likelihood_and_forecast_reconstructions_passed': True})
    raw = read(coverage['source_data'])
    paid = {b['query_task']: [raw['tasks'][b['query_task']]['Y_replicates'][j][0][0] for j in spec['paid_prefix']] for b in branches}; del raw
    rows = []
    for work in fields:
        row, field = work['row'], work.pop('field')
        evidence = sum(update_raw_observation(field, j, float(y)) for j, y in enumerate(paid[row['query_task']]))
        state = compact(field, list(range(4, 12)), evidence); filename = f"layout{row['layout']}_task{row['query_task']}_{row['arm']}.npz"
        np.savez_compressed(out / filename, **state); row.update(prediction=prediction(state), component_file=filename); rows.append(row)
    save(out / 'predictions.json', {'execution': spec, 'rows': rows, 'all_twenty_forecasts_frozen_before_score_labels': True})
    base = {'execution': spec, 'cost': spec['cost'], 'no_model_adopted': True, 'training_diagnostic_only': True,
        'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    for row in rows:
        if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']:
            save(out / 'results.json', {**base, 'rows': rows, 'diagnostic_interpreted': False, 'stopped_at_cpu_budget': True,
                'cpu_seconds': time.process_time() - started}); return
        inputs = read(Path(coverage['output']) / f"layout{row['layout']}" / 'task1' / 'inputs.json')
        task = next(t for t in inputs['source_tasks'] if t['name'] == f"related_inventory_{row['query_task']:02d}")
        labels = {k: np.asarray(task[k])[4:] for k in ('response', 'noise')}
        with np.load(out / row['component_file']) as state: row['scores'], row['density_integration_error_bound'] = score(row, state, labels)
        row['mean_error'] = decompose(row['prediction']['mean'], labels['response'][:, 0]); row['comparisons'] = {}
        for control in ('native_point', 'source_gp'):
            reference = next(r for r in controls['rows'] if r['layout'] == row['layout'] and r['query_task'] == row['query_task']
                and r['arm'] == control and r['design'] == row['design'])
            assert reference['paid_prefix'] == row['paid_prefix'] and reference['scoring_sites'] == row['scoring_sites']
            row['comparisons'][control] = {k: row['scores'][k] - reference['scores'][k] for k in spec['metrics']}
    summary = {layout: {arm: {control: {k: float(np.mean([r['comparisons'][control][k] for r in rows if r['layout'] == layout and r['arm'] == arm]))
        for k in spec['metrics']} for control in ('native_point', 'source_gp')} for arm in spec['arms']} for layout in spec['layouts']}
    save(out / 'results.json', {**base, 'rows': rows, 'summary': summary, 'diagnostic_interpreted': True,
        'cpu_seconds': time.process_time() - started, 'component_artifact_bytes': sum(p.stat().st_size for p in out.glob('*.npz'))})
    print('joint prior compression', summary, 'CPU', time.process_time() - started, flush=True)


if __name__ == '__main__':
    run()

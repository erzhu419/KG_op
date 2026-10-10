"""Source-only spectra and weak-tail budgets of the joint finite laws."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_information_ceiling import variance_parts
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, spectral_weights

EXECUTION = 'performance/manifests/legacy_joint_population_f_inner_joint_spectral_compatibility_execution_v1.json'


def fit_spectrum(basis, covariance, columns, native_exponent, design):
    diagonal = np.diag(covariance); shape = diagonal[1:]
    selected = np.asarray(columns[1:]) - 1
    assert (shape[selected] > 0).all()
    groups = np.unique(basis.group_index[selected])
    budgets = np.bincount(basis.group_index, weights=shape, minlength=len(basis.group_names))
    group_weights = budgets / budgets.sum()
    fractions = np.zeros_like(shape)
    fractions[selected] = shape[selected] / budgets[basis.group_index[selected]]

    def loss(exponent):
        weights = spectral_weights(basis, exponent)[0]
        return float(sum(group_weights[g] * np.mean((np.log(weights[basis.group_index == g])
            - np.log(fractions[basis.group_index == g])) ** 2) for g in groups))

    fit = minimize_scalar(loss, bounds=(0., 4.), method='bounded', options={'xatol': 1e-9, 'maxiter': 500})
    assert fit.success
    exponent = min((0., float(fit.x), 4.), key=loss)
    weights = spectral_weights(basis, exponent)[0]
    target = np.r_[diagonal[0], budgets[basis.group_index] * weights]
    ratios = np.ones_like(diagonal); enabled = np.r_[0, np.asarray(columns[1:])]
    assert (diagonal[enabled] > 0).all()
    ratios[enabled] = np.sqrt(target[enabled] / diagonal[enabled])
    constrained = ratios[:, None] * covariance * ratios[None, :]
    assert np.linalg.eigvalsh(constrained).min() >= -1e-10
    np.testing.assert_allclose(np.diag(constrained), target, atol=1e-14, rtol=1e-10)
    np.testing.assert_allclose(np.bincount(basis.group_index, weights=np.diag(constrained)[1:], minlength=len(budgets)), budgets, atol=1e-14, rtol=1e-10)
    before_correlation = covariance[np.ix_(enabled, enabled)] / np.sqrt(np.outer(diagonal[enabled], diagonal[enabled]))
    after_correlation = constrained[np.ix_(enabled, enabled)] / np.sqrt(np.outer(target[enabled], target[enabled]))
    np.testing.assert_allclose(after_correlation, before_correlation, atol=1e-12, rtol=1e-10)
    group_rows = []
    for g in groups:
        indices = np.flatnonzero(basis.group_index == g)
        group_rows.append({'group': basis.group_names[g], 'coefficient_columns': (indices + 1).tolist(),
            'roughness': basis.roughness[indices].tolist(), 'total_variance': float(budgets[g]),
            'source_variance_fractions': fractions[indices].tolist(), 'constrained_variance_fractions': weights[indices].tolist(),
            'total_variation': float(.5 * np.sum(np.abs(fractions[indices] - weights[indices])))})
    bank, changed = design @ covariance @ design.T, design @ constrained @ design.T
    before, after = variance_parts(bank), variance_parts(changed)
    centered = design - design.mean(axis=0)
    centered_before, centered_after = centered @ covariance @ centered.T, centered @ constrained @ centered.T
    return constrained, {'fitted_exponent': exponent, 'native_exponent': native_exponent,
        'weighted_log_fraction_rmse': float(np.sqrt(loss(exponent))),
        'native_weighted_log_fraction_rmse': float(np.sqrt(loss(native_exponent))),
        'flat_weighted_log_fraction_rmse': float(np.sqrt(loss(0.))),
        'weighted_group_total_variation': float(sum(group_weights[g] * r['total_variation'] for g, r in zip(groups, group_rows))),
        'coefficient_covariance_relative_change': float(np.linalg.norm(constrained - covariance) / np.linalg.norm(covariance)),
        'bank_covariance_relative_change': float(np.linalg.norm(changed - bank) / np.linalg.norm(bank)),
        'centered_bank_covariance_relative_change': float(np.linalg.norm(centered_after - centered_before) / np.linalg.norm(centered_before)),
        'bank_variance_before': before, 'bank_variance_after': after,
        'centered_shape_variance_ratio': after['centered_shape_variance'] / before['centered_shape_variance'],
        'groups': group_rows, 'psd_group_budgets_and_correlations_preserved': True}


def run():
    started = time.process_time(); spec = read(EXECUTION); plan = read(spec['plan'])
    with_tail, without_tail = read(plan['with_tail_laws']), read(plan['without_tail_laws'])
    assert with_tail['all_twenty_source_laws_frozen_before_paid_values'] and without_tail['all_twenty_source_laws_frozen_before_paid_values']
    fitting = read(plan['source_fit_results']); coverage, _ = source_paths(read(plan['source_context_execution']))
    out = ROOT / plan['output']; out.mkdir(parents=True, exist_ok=True); rows, laws = [], []
    for old in without_tail['rows']:
        if time.process_time() - started >= plan['maximum_single_thread_cpu_seconds']:
            save(out / 'results.json', {'execution': spec, 'rows': rows, 'diagnostic_interpreted': False,
                'stopped_at_cpu_budget': True, 'cpu_seconds': time.process_time() - started}); return
        layout, query, arm = old['layout'], old['query_task'], old['arm']
        added = next(r for r in with_tail['rows'] if (r['layout'], r['query_task'], r['arm']) == (layout, query, arm))
        assert old['source_task_names'] == added['source_task_names'] and old['columns'] == added['columns']
        assert 'related_inventory_01' not in old['source_task_names'] and f'related_inventory_{query:02d}' not in old['source_task_names']
        record = next(r for r in fitting['rows'] if r['layout'] == layout and r['inner_query_task'] == f'related_inventory_{query:02d}')
        fitted = read(record['source_fit']); assert fitted['inner_training_task_names'] == old['source_task_names']
        native = fitted['fits']['structural']
        reference = np.asarray(read(Path(coverage['output']) / f'layout{layout}' / 'task1' / 'inputs.json')['public_mean_reference'])
        basis = OrthogonalLowOrderBasis(reference); design = np.column_stack([np.ones(len(reference)), basis.transform(reference)])
        np.testing.assert_allclose(basis.roughness, native['roughness_eigenvalues'], atol=1e-12, rtol=1e-12)
        assert basis.names == native['selected_names']
        tau = native['tail_variance_budget'] / basis.feature_dim
        tail = np.diag(np.r_[0., np.full(basis.feature_dim, tau)])
        tail_parts = variance_parts(design @ tail @ design.T)
        for name, key, exponent_key in [('common', 'common_prior_coefficient_covariance', 'common_spectral_exponent'),
                                      ('task', 'task_coefficient_covariance', 'spectral_exponent')]:
            covariance = np.asarray(old[key]); actual_tail = np.asarray(added[key]) - covariance
            np.testing.assert_allclose(actual_tail, tail, atol=1e-14, rtol=1e-12)
            assert np.linalg.eigvalsh(covariance).min() >= -1e-10
            before = variance_parts(design @ covariance @ design.T)
            combined = variance_parts(design @ np.asarray(added[key]) @ design.T)
            for metric in before:
                np.testing.assert_allclose(before[metric] + tail_parts[metric], combined[metric], atol=1e-12, rtol=1e-10)
            constrained, spectrum = fit_spectrum(basis, covariance, old['columns'], native[exponent_key], design)
            row = {'layout': layout, 'query_task': query, 'arm': arm, 'component': name,
                'original_tail_budget': native['tail_variance_budget'], 'smooth_bank_variance': before, 'tail_bank_variance': tail_parts,
                'tail_to_smooth_ratio': {k: tail_parts[k] / before[k] for k in before}, 'spectrum': spectrum}
            rows.append(row)
            laws.append({'layout': layout, 'query_task': query, 'arm': arm, 'component': name, 'source_task_names': old['source_task_names'],
                'columns': old['columns'], 'spectral_exponent': spectrum['fitted_exponent'], 'coefficient_covariance': constrained.tolist()})
    assert len(rows) == len(laws) == 40
    summary = {layout: {arm: {component: {
        'tail_to_smooth_centered_shape_mean': float(np.mean([r['tail_to_smooth_ratio']['centered_shape_variance'] for r in rows if (r['layout'], r['arm'], r['component']) == (layout, arm, component)])),
        **{metric: float(np.mean([r['spectrum'][metric] for r in rows if (r['layout'], r['arm'], r['component']) == (layout, arm, component)]))
            for metric in ('fitted_exponent', 'native_exponent', 'weighted_log_fraction_rmse', 'weighted_group_total_variation',
                          'centered_bank_covariance_relative_change', 'centered_shape_variance_ratio')}}
        for component in ('common', 'task')} for arm in spec['arms']} for layout in spec['layouts']}
    save(out / 'spectral_laws.json', {'execution': spec, 'rows': laws, 'source_only_covariance_diagnostics': True,
        'source_conditioned_or_query_scored': False})
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'summary': summary, 'diagnostic_interpreted': True,
        'cpu_seconds': time.process_time() - started, 'cost': plan['cost'], 'all_forty_covariance_checks_passed': True,
        'no_model_adopted': True, 'query_outcomes_indexed': False, 'source_response_arrays_indexed': False,
        'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False})
    print('source spectral compatibility', summary, 'CPU', time.process_time() - started, flush=True)


if __name__ == '__main__':
    run()

"""Source-only coefficient diagnostic for the frozen risk-mean basis.

The original six-column selector remains frozen. These are small weighted
linear diagnostic fits, not replacement source laws or target optimizers.
"""

import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.adapt_historical_source_structure import restore_selector
from representation.exchangeable_mean import ExchangeableBoundaryMeanCoordinate

PLAN = 'performance/manifests/historical_source_structure_adaptation_v1.json'
MAPS = 'paper_artifacts/historical_source_structure_adaptation_v1/maps.json'
OUTPUT = 'paper_artifacts/mainline_d1000_risk_basis_diagnostic_v1/results.json'
RIDGE = .001


def read(path):
    return json.loads((ROOT / path).read_text())


def _pooled_fit(features, y, noise, domains):
    """Common slopes and zero-sum task intercepts, unchanged across bases."""
    names = sorted(set(domains))
    contrasts = np.asarray([[float(t == n) - float(t == names[-1])
                             for n in names[:-1]] for t in domains])
    coefficients, _ = ExchangeableBoundaryMeanCoordinate._solve(
        np.column_stack([features, contrasts]), y, 1 / np.maximum(noise, 1e-8), RIDGE)
    return coefficients[:1 + features.shape[1]]


def _weighted_projection(design, values, noise):
    weight = 1 / np.maximum(noise, 1e-8)
    return np.linalg.pinv(design.T @ (weight[:, None] * design)) @ (design.T @ (weight * values))


def residual_model(design, dispersion, y, noise, domains):
    """One shared public residual coordinate, with taskwise FWL slopes."""
    projection = _weighted_projection(design, dispersion, noise)
    residual = dispersion - design @ projection
    tasks = []
    for name in sorted(set(domains)):
        mask = domains == name
        D, r, v = design[mask], residual[mask], noise[mask]
        orthogonal = r - D @ _weighted_projection(D, r, v)
        weight = 1 / np.maximum(v, 1e-8)
        information = float(orthogonal @ (weight * orthogonal))
        influence = weight * orthogonal / information
        beta = float(influence @ y[mask])
        variance = float((influence ** 2) @ v)
        tasks.append({'task': name, 'beta': beta, 'sampling_standard_error': np.sqrt(variance),
                      'sampling_variance': variance, 'residual_information': information})
    betas = np.asarray([t['beta'] for t in tasks])
    variances = np.asarray([t['sampling_variance'] for t in tasks])
    between = float(np.var(betas, ddof=1))
    deconvolved = max(0., between - float(variances.mean()))
    return {'dispersion_projection_on_intercept_and_g6': projection.tolist(),
            'tasks': tasks, 'beta_equal_task_mean': float(betas.mean()),
            'observed_between_task_variance': between,
            'mean_task_sampling_variance': float(variances.mean()),
            'deconvolved_between_task_variance': deconvolved,
            'common_mean_sampling_variance': float(variances.sum() / len(tasks) ** 2),
            'common_mean_total_variance_deconvolved': float(deconvolved / len(tasks) + variances.sum() / len(tasks) ** 2),
            'new_task_beta_variance_deconvolved': float(deconvolved * (1 + 1 / len(tasks)) + variances.sum() / len(tasks) ** 2),
            'weighted_residual_orthogonality_error': float(np.max(abs(
                design.T @ (residual / np.maximum(noise, 1e-8)))))}, residual


def _metrics(prediction, observed):
    error = prediction - observed
    return {'mse': float(np.mean(error ** 2)), 'bias_prediction_minus_observed': float(error.mean()),
            'predictions': prediction.tolist(), 'observed_batch_means': observed.tolist()}


def full_mean_residual_model(design, coordinates, y, noise, domains):
    """Ridge Schur/FWL decomposition with the unchanged task intercepts."""
    names = sorted(set(domains))
    contrasts = np.asarray([[float(t == n) - float(t == names[-1])
                             for n in names[:-1]] for t in domains])
    Z = np.column_stack([design, contrasts])
    weight = 1 / np.maximum(noise, 1e-8)
    penalty = RIDGE * np.eye(Z.shape[1])
    penalty[0, 0] = 0.
    inverse = np.linalg.pinv(Z.T @ (weight[:, None] * Z) + penalty)
    base = inverse @ (Z.T @ (weight * y))
    projection = inverse @ (Z.T @ (weight[:, None] * coordinates))
    cross = coordinates.T @ (weight[:, None] * Z)
    schur = coordinates.T @ (weight[:, None] * coordinates) + RIDGE * np.eye(4) - cross @ projection
    beta_inverse = np.linalg.pinv(schur)
    influence = beta_inverse @ (coordinates.T * weight - cross @ inverse @ (Z.T * weight))
    beta = influence @ y
    fitted = Z @ (base - projection @ beta) + coordinates @ beta
    error = y - fitted
    cluster = sum(np.outer(influence[:, domains == name] @ error[domains == name],
                           influence[:, domains == name] @ error[domains == name]) for name in names)
    correction = len(names) / (len(names) - 1) * (len(y) - 1) / (len(y) - Z.shape[1] - 4)
    cluster *= correction
    known = (influence * noise) @ influence.T
    return {'projection_coefficients': projection[:design.shape[1]].tolist(),
            'training_task_contrast_projection': projection[design.shape[1]:].tolist(),
            'beta_mean': beta.tolist(), 'beta_covariance': cluster.tolist(),
            'known_noise_sampling_covariance': known.tolist(),
            'covariance_method': 'Source-task clustered sandwich CR1; no addition of overlapping known-noise covariance',
            'cluster_count': len(names), 'CR1_multiplier': correction,
            'residual_coordinate_rank': int(np.linalg.matrix_rank(coordinates - Z @ projection)),
            'baseline_common_coefficients': base[:design.shape[1]].tolist()}


def run():
    started = time.process_time()
    spec = read(PLAN)
    inputs = read(spec['source_inputs_by_layout']['5'])
    tasks = [t for t in inputs['source_tasks'] if t['name'] in spec['source_tasks']]
    record = next(r for r in read(MAPS)['rows'] if
                  (r['layout'], r['head'], r['selector']) == ('5', 'g', 'spectral'))
    coordinates = np.vstack([t['psi'] for t in tasks])
    features = restore_selector(record, spec).transform(coordinates)
    # A row-order mismatch would invalidate both the source labels and FWL map.
    np.testing.assert_allclose(features, record['source_features'], atol=1e-12, rtol=1e-12)
    y = np.concatenate([np.asarray(t['response'])[:, 1] for t in tasks])
    noise = np.concatenate([np.asarray(t['noise'])[:, 1] for t in tasks])
    domains = np.concatenate([np.repeat(t['name'], len(t['psi'])) for t in tasks])
    designs = {'current_g6': features,
               'g6_plus_dispersion': np.column_stack([features, coordinates[:, 3]]),
               'g6_plus_mean4': np.column_stack([features, coordinates])}
    D = np.column_stack([np.ones(len(features)), features])
    full, residual = residual_model(D, coordinates[:, 3], y, noise, domains)
    pooled_residual = full_mean_residual_model(D, coordinates, y, noise, domains)
    full_direct = _pooled_fit(designs['g6_plus_mean4'], y, noise, domains)
    full_residual_prediction = (D @ pooled_residual['baseline_common_coefficients']
        + (coordinates - D @ pooled_residual['projection_coefficients']) @ pooled_residual['beta_mean'])
    full_direct_prediction = np.column_stack([D, coordinates]) @ full_direct
    full_fwl_error = float(np.max(abs(full_residual_prediction - full_direct_prediction)))
    np.testing.assert_allclose(full_residual_prediction, full_direct_prediction, atol=1e-8, rtol=1e-8)
    folds, oof = [], {name: np.zeros(len(y)) for name in
                      (*designs, 'g6_plus_task_mean_residual', 'g6_plus_pooled_mean4_residual')}
    for heldout in spec['source_tasks']:
        test = domains == heldout
        train = ~test
        rows = {}
        for name, design in designs.items():
            coefficient = _pooled_fit(design[train], y[train], noise[train], domains[train])
            prediction = np.column_stack([np.ones(test.sum()), design[test]]) @ coefficient
            oof[name][test] = prediction
            rows[name] = _metrics(prediction, y[test])
        fitted, _ = residual_model(D[train], coordinates[train, 3], y[train], noise[train], domains[train])
        heldout_residual = coordinates[test, 3] - D[test] @ fitted['dispersion_projection_on_intercept_and_g6']
        prediction = oof['current_g6'][test] + fitted['beta_equal_task_mean'] * heldout_residual
        oof['g6_plus_task_mean_residual'][test] = prediction
        rows['g6_plus_task_mean_residual'] = _metrics(prediction, y[test])
        multivariate = full_mean_residual_model(D[train], coordinates[train], y[train], noise[train], domains[train])
        multi_prediction = (D[test] @ multivariate['baseline_common_coefficients']
            + (coordinates[test] - D[test] @ multivariate['projection_coefficients']) @ multivariate['beta_mean'])
        fwl_error = float(np.max(abs(multi_prediction - oof['g6_plus_mean4'][test])))
        np.testing.assert_allclose(multi_prediction, oof['g6_plus_mean4'][test], atol=1e-8, rtol=1e-8)
        oof['g6_plus_pooled_mean4_residual'][test] = multi_prediction
        rows['g6_plus_pooled_mean4_residual'] = _metrics(multi_prediction, y[test])
        folds.append({'heldout_task': heldout, 'training_tasks': sorted(set(domains[train])),
                      'training_rows': int(train.sum()), 'heldout_rows': int(test.sum()),
                      'models': rows, 'residual_model': fitted,
                      'pooled_mean4_residual_model': multivariate, 'pooled_mean4_FWL_max_error': fwl_error})
    aggregate = {name: _metrics(prediction, y) for name, prediction in oof.items()}
    baseline_error = y - oof['current_g6']
    centered_error, centered_residual = baseline_error.copy(), residual.copy()
    for name in spec['source_tasks']:
        mask = domains == name
        centered_error[mask] -= centered_error[mask].mean()
        centered_residual[mask] -= centered_residual[mask].mean()
    result = {'source_inputs': spec['source_inputs_by_layout']['5'], 'source_maps': MAPS,
        'source_tasks': spec['source_tasks'], 'source_rows': len(y),
        'selected_g_functions': record['diagnostics']['selected_names'],
        'method': {'ridge': RIDGE, 'weights': 'inverse observed three-replicate mean variance, floor1e-8',
                   'task_intercepts': 'zero-sum training-task effects; unseen task effect zero in every model',
                   'residual_coordinate': 'dispersion minus fixed weighted projection on intercept/current g6',
                   'residual_beta': 'taskwise weighted FWL, equal task mean; no strength tuning'},
        'folds': folds, 'aggregate': aggregate, 'full_source_residual_model': full,
        'full_source_pooled_mean4_residual_model': pooled_residual,
        'full_source_pooled_mean4_FWL_max_error': full_fwl_error,
        'omitted_dimension_relation': {'oof_observed_minus_prediction_vs_dispersion_residual_correlation':
            float(np.corrcoef(baseline_error, residual)[0, 1]),
            'within_task_centered_correlation': float(np.corrcoef(centered_error, centered_residual)[0, 1])},
        'limitations': ['Coefficient-only leave-one-task-out: the frozen selector and its whitening already used all48 source labels.',
            'This compares weighted linear diagnostic fits, not the native raw-source mixture or its cross-policy covariance.',
            'No tasks00/01/06/07 or target observations/outcomes choose parameters; batch means remain noisy source labels.'],
        'cost': {'diagnostic_pooled_coefficient_fits': 13, 'taskwise_FWL_slopes': 16,
                 'pooled_mean4_FWL_decompositions': 5, 'native_source_law_fits': 0,
                 'new_simulator_calls': 0, 'new_verifier_calls': 0, 'target_truth_calls': 0,
                 'cpu_seconds': time.process_time() - started}}
    path = ROOT / OUTPUT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    common = {'layout': '5', 'source_task_ids': spec['source_tasks'], 'source_rows': len(y),
              'maps': MAPS, 'selection_plan': PLAN, 'source_inputs': spec['source_inputs_by_layout']['5'],
              'affected_identity_prefix': 'spectral/head_specific/', 'diagnostic_results': OUTPUT,
              'target_observations_used': False, 'new_source_labels': 0}
    rejected = dict(common, status='diagnostic_rejected', adopted=False,
        projection_coefficients=full['dispersion_projection_on_intercept_and_g6'],
        beta_mean=full['beta_equal_task_mean'],
        beta_between_task_variance=full['deconvolved_between_task_variance'],
        beta_mean_variance=full['common_mean_total_variance_deconvolved'],
        beta_predictive_variance=full['new_task_beta_variance_deconvolved'],
        rejection='Source coefficient-LOTO MSE worsens for both single-dispersion variants', task_beta=full['tasks'])
    calibration = dict(common, **pooled_residual,
        status='source_coefficient_diagnostic_supported_pending_native_application', adopted=False,
        coordinate='Frozen standardized public mean4 residualized against current g6/intercept and training-task contrasts',
        application='For unseen tasks R=mean4-D@projection_coefficients; add R@beta_mean and R@beta_covariance@R.T to native g only',
        baseline_source_LOTO_mse=aggregate['current_g6']['mse'],
        corrected_source_LOTO_mse=aggregate['g6_plus_pooled_mean4_residual']['mse'],
        improved_folds=sum(f['models']['g6_plus_mean4']['mse'] < f['models']['current_g6']['mse'] for f in folds),
        limitations=result['limitations'] + ['Only four source clusters determine coefficient covariance.',
            'FWL equality concerns the diagnostic pooled baseline; adding the residual to the different native mixture requires its own replay assessment.'])
    for name, value in [('rejected_dispersion_calibration.json', rejected), ('calibration.json', calibration)]:
        path.with_name(name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': OUTPUT, 'aggregate': {k: {n: r[n] for n in
        ('mse', 'bias_prediction_minus_observed')} for k, r in aggregate.items()},
        'task_slopes': full['tasks'], 'residual_relation': result['omitted_dimension_relation'],
        'cost': result['cost']}, indent=2))
    return result


if __name__ == '__main__':
    run()

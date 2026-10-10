"""Cached deletion-mixture failures and source-only paired-function moments."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_counterbalanced_mse_uncertainty import paired_interval
from performance.diagnose_legacy_task_covariance_direction import dense_nll
from performance.run_legacy_related_inventory_source import save, training_view
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.source_function_prior import head_batches
from representation.source_joint_population import JointCovariancePopulationLikelihood

EXECUTION = 'performance/manifests/legacy_joint_population_counterbalanced_source_dependence_v1.json'


def mixture_changes(spec, coverage):
    data = read(coverage['source_data'])
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json'))
    reference = read(spec['reference_profiles'])['rows']
    labels = {i: training_view(data, [i], public['plan']['task_distribution']['L'])[0] for i in spec['heldout_tasks']}
    rows = []
    for row in read(spec['mixture_profiles'])['rows']:
        old = next(r for r in reference if all(r[k] == row[k] for k in ('layout', 'task_index', 'mean_arm')))
        label = labels[row['task_index']]
        for h, head in enumerate(('f', 'g')):
            interval, _ = paired_interval(np.asarray(row['mean'])[2:, h], np.asarray(old['mean'])[2:, h],
                label['response'][2:, h], label['noise'][2:, h], 3)
            np.testing.assert_allclose(interval['paired_mse_difference'],
                row['scores']['observed_head_mse'][h]-old['scores']['observed_head_mse'][h], atol=1e-12)
            rows.append({'layout': row['layout'], 'task_id': row['task_id'], 'mean_arm': row['mean_arm'], 'head': head,
                **interval, 'median_mean_variance_ratio': float(np.median(np.asarray(row['variance'])[2:, h]/np.asarray(old['variance'])[2:, h])),
                'mean_prediction_shift_rms': float(np.sqrt(np.mean((np.asarray(row['mean'])[2:, h]-np.asarray(old['mean'])[2:, h])**2))),
                'full_identity_posterior_weight': row['identity_weights'][0],
                'previous_full_identity_posterior_weight': old['identity_weights'][0]})
    return rows


def source_head_state(inputs, native, h):
    tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
    basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
    prior = restore_regularizer(basis, native['regularizer'], tasks, h)
    model = JointCovariancePopulationLikelihood(prior, head_batches(tasks, h), [t['noise'][:, h] for t in tasks])
    value, _, state = model.evaluate(np.asarray(native['joint_structural']['optimizer_parameters']))
    r = np.diag(state['residual_diagonal']); means, covariances = [], []
    for x, error, noise in model.blocks:
        variance = x@r@x.T+np.diag(noise+state['discrepancy'])
        gain = r@cho_solve(cho_factor(variance, lower=True), x).T
        means.append(gain@(error-x@(state['mean']-prior.mean)))
        covariances.append(r-gain@x@r+(gain@x)@state['mean_covariance']@(gain@x).T)
    return value, state, model.blocks, means, covariances


def paired_direction(inputs, native, layout):
    heads = [source_head_state(inputs, native['heads'][h], h) for h in (0, 1)]
    count = len(heads[0][2]); width = len(heads[0][1]['residual_diagonal'])
    rdiag = np.concatenate([h[1]['residual_diagonal'] for h in heads])
    cdiag = np.concatenate([h[1]['common_diagonal'] for h in heads])
    moments, blocks = [], []
    for i in range(count):
        means = np.concatenate([h[3][i] for h in heads])
        moments.append(np.outer(means, means)+block_diag(*[h[4][i] for h in heads]))
        blocks.append((block_diag(*[h[2][i][0] for h in heads]),
            np.concatenate([h[2][i][1] for h in heads]),
            np.concatenate([h[2][i][2]+h[1]['discrepancy'] for h in heads])))
    moment = np.mean(moments, axis=0); support = rdiag > 0; scales = np.sqrt(rdiag[support])
    whitened = moment[np.ix_(support, support)]/np.outer(scales, scales)
    values, vectors = np.linalg.eigh((whitened+whitened.T)/2)
    direction = np.zeros(len(rdiag)); direction[support] = scales*vectors[:, -1]
    if direction[np.argmax(np.abs(direction))] < 0:
        direction *= -1
    fmask = np.arange(len(rdiag))[support] < width
    diagonal_max = max(np.linalg.eigvalsh(whitened[np.ix_(mask, mask)])[-1] for mask in (fmask, ~fmask))
    base = dense_nll(blocks, cdiag, np.diag(rdiag), 0.)
    np.testing.assert_allclose(base, np.mean([h[0] for h in heads]), atol=1e-8, rtol=1e-8)
    n = sum(len(b[1]) for b in blocks); derivative = .5*count/n*(1-values[-1]); step = 1e-5
    actual = (dense_nll(blocks, cdiag, np.diag(rdiag)+step*np.outer(direction, direction), 0.)-base)/step
    np.testing.assert_allclose(actual, derivative, atol=2e-5, rtol=2e-4)
    basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
    return {'layout': layout, 'source_tasks': count, 'source_paired_records': n//2,
        'independent_head_leading_eigenvalue': float(diagonal_max), 'paired_leading_eigenvalue': float(values[-1]),
        'whitened_cross_head_moment_frobenius_norm': float(np.linalg.norm(whitened[np.ix_(fmask, ~fmask)])),
        'direction_white_squared_norm_by_head': {'f': float(np.sum(vectors[fmask, -1]**2)), 'g': float(np.sum(vectors[~fmask, -1]**2))},
        'rank_one_variance_source_nll_derivative': float(derivative), 'finite_difference_derivative': float(actual),
        'independent_head_average_source_nll': float(np.mean([h[0] for h in heads])), 'dense_joint_source_nll': base,
        'fixed_unit_scale_source_nll': dense_nll(blocks, cdiag, np.diag(rdiag)+np.outer(direction, direction), 0.),
        'coefficient_names': [f'{head}:{name}' for head in ('f', 'g') for name in ['intercept']+basis.names],
        'direction': direction.tolist()}


def run():
    started = time.process_time(); spec = read(EXECUTION); coverage = read(spec['coverage_execution'])
    mixture = mixture_changes(spec, coverage); paired = []
    for layout in spec['layouts']:
        folder = Path(coverage['output'])/f'layout{layout}'
        paired.append(paired_direction(read(folder/'inputs.json'), read(folder/'source_fit.json'), layout))
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    result = {'execution': spec, 'mixture_diagnostics': mixture, 'paired_source_directions': paired,
        'cost': {'new_source_fits': 0, 'new_simulator_calls': 0, 'new_verification_calls': 0, 'new_terminal_integrations': 0},
        'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result)
    print({'structural_g_mixture_changes': [r for r in mixture if r['mean_arm'] == 'joint_structural' and r['head'] == 'g'],
        'paired_source_directions': [{k: v for k, v in r.items() if k not in ('direction', 'coefficient_names')} for r in paired],
        'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()

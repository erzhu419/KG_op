"""Reconstruct correlated common function/risk heads from source eta components."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_diverse_source_design import encode_diverse_source
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.legacy_observable_prior import legacy_exposure_descriptors
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis

PLAN = 'performance/manifests/legacy_joint_population_f_raw_source_shared_mixture_plan_v1.json'


def restore_risk(spec, field):
    contract = read(spec['source_contract_plan']); known = read(contract['known_source_plan'])
    coverage, _ = source_paths(read(known['source_context_execution']))
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json'))
    raw = read(coverage['source_data']); ids = contract['source_tasks']
    data = {'bank': raw['bank'], 'tasks': {i: raw['tasks'][i] for i in ids}}; del raw
    bank = [tuple(p) for p in public['bank']]
    descriptors = {i: legacy_exposure_descriptors(no_outcomes(problem_for(public['plan'], public['tasks'][i])), bank, 50) for i in ids}
    execution = read(known['risk_source_execution'])
    case = next(c for c in execution['cases'] if (c['layout'], c['query_task']) == (field['layout'], 0))
    folder = Path(execution['output']) / case['folder']
    inputs = read(folder / 'risk_inputs.json'); fits = read(folder / 'risk_source_fit.json')['fits']
    assert inputs['training_task_ids'] == ids
    coordinate, tasks, _, _ = encode_diverse_source(data, ids, inputs['source_site_indices_by_task'], descriptors,
        public['plan']['task_distribution']['L'])
    reference = coordinate.transform(descriptors[ids[0]])
    np.testing.assert_allclose(reference, inputs['public_reference_psi'], atol=1e-8, rtol=1e-8)
    for task, saved in zip(tasks, inputs['source_tasks']):
        np.testing.assert_allclose(task['psi'], saved['psi'], atol=1e-8, rtol=1e-8)
    basis = OrthogonalLowOrderBasis(reference); ds = []
    for source in ids:
        sites = [s['policy'] for s in field['sites'] if s['source_task'] == source]
        ds.append(np.column_stack([np.ones(12), basis.transform(coordinate.transform(descriptors[source][sites]))]))
    d = np.vstack(ds); law = fits['joint_structural']; mean = np.asarray(fits['regularizer']['coefficient_mean'])
    a = np.diag(law['common_prior_coefficient_diagonal']); r = np.diag(law['residual_coefficient_diagonal'])
    k = d @ a @ d.T + block_diag(*[dt @ r @ dt.T + law['discrepancy_variance'] * np.eye(12) for dt in ds])
    np.testing.assert_allclose(d @ mean, field['eta_mean'], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(k, field['eta_covariance'], atol=1e-8, rtol=1e-8)
    cross = a @ d.T; gain = cho_solve(cho_factor(k, lower=True), cross.T).T
    covariance = a - gain @ cross.T; covariance = .5 * (covariance + covariance.T)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-10
    return {'prior_mean': mean, 'conditional_covariance': covariance, 'source_eta_gain': gain,
        'common_prior_covariance': a, 'task_coefficient_covariance': r, 'white_discrepancy': law['discrepancy_variance'],
        'source_design': d, 'public_reference_psi': reference}


def moment_errors(mean, covariance, reference_mean, reference_covariance):
    diagonal = np.diag(reference_covariance); active = diagonal > 0
    if np.any(~active):
        np.testing.assert_allclose(mean[~active], reference_mean[~active], atol=1e-10, rtol=0)
        np.testing.assert_allclose(np.diag(covariance)[~active], 0, atol=1e-10, rtol=0)
    return {'mean_rms_difference_in_full_posterior_sd': float(np.sqrt(np.mean((mean[active] - reference_mean[active]) ** 2 / diagonal[active]))),
        'maximum_diagonal_variance_relative_difference': float(np.max(np.abs(np.diag(covariance)[active] / diagonal[active] - 1)))}


def run():
    started = time.process_time(); spec = read(PLAN); posterior = read(spec['source_results'])
    assert posterior['all_monte_carlo_gates_passed'] and posterior['all_mixture_retention_gates_passed']
    fields = read(spec['source_inputs'])['rows']; mixtures = read(spec['eta_components'])['rows']
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True); rows, reconstructions = [], []
    for field in fields:
        risk = restore_risk(spec, field); gm = risk['source_eta_gain']; rc = risk['conditional_covariance']
        reconstructions.append({'layout': field['layout'], **{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in risk.items()},
            'source_eta_prior_mean': field['eta_mean'], 'source_eta_prior_covariance': field['eta_covariance'],
            'components': spec['eta_components'], 'function_law_and_raw_labels': spec['source_inputs'],
            'within_component_f_eta_cross_covariance': 0., 'task_residuals_added_only_for_future_task': True})
        for index, arm in enumerate(field['arms']):
            component = next(r for r in mixtures if (r['layout'], r['arm']) == (field['layout'], arm))
            assert component['mixture_gate_passed']; states = np.asarray(component['eta_states_by_chain']).reshape(512, 48)
            old = next(r for r in posterior['rows'] if (r['layout'], r['arm']) == (field['layout'], arm))
            model = VarianceReferencePosterior(field['f_means'][index], field['f_covariances'][index], field['batch_means'], field['sample_variances'])
            cross = np.asarray(field['common_latent_to_source_f_covariances'][index]); a = np.asarray(field['common_latent_prior_covariances'][index])
            fm = np.asarray(field['common_latent_prior_means'][index]); nf = len(fm); means, covariances = [], []
            for eta in states:
                if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']: raise TimeoutError('registered CPU budget reached')
                _, factor, alpha = model.evaluate(eta)
                cm = fm + cross @ alpha; cc = a - cross @ cho_solve(factor, cross.T)
                means.append(np.r_[cm, risk['prior_mean'] + gm @ (eta - field['eta_mean'])])
                covariances.append(block_diag(cc, rc))
            means, covariances = np.asarray(means), np.asarray(covariances)
            mean = means.mean(axis=0); centered = means - mean
            within = covariances.mean(axis=0); between = centered.T @ centered / len(states); covariance = within + between
            second = (covariances + means[:, :, None] * means[:, None, :]).mean(axis=0) - np.outer(mean, mean)
            np.testing.assert_allclose(covariance, second, atol=1e-8, rtol=1e-8)
            minimum = float(np.linalg.eigvalsh(covariance).min()); assert minimum >= -1e-9
            rm = risk['prior_mean'] + gm @ (np.asarray(old['eta_posterior_mean']) - field['eta_mean'])
            rh = rc + gm @ np.asarray(old['eta_posterior_covariance']) @ gm.T
            errors = {'f': moment_errors(mean[:nf], covariance[:nf, :nf], np.asarray(old['common_posterior_mean']), np.asarray(old['common_posterior_covariance'])),
                'eta': moment_errors(mean[nf:], covariance[nf:, nf:], rm, rh)}
            passed = all(e['mean_rms_difference_in_full_posterior_sd'] <= .1 and e['maximum_diagonal_variance_relative_difference'] <= .15 for e in errors.values())
            scale = np.sqrt(np.diag(covariance)[:nf, None] * np.diag(covariance)[None, nf:]); active = scale > 0
            association = np.zeros_like(scale); association[active] = covariance[:nf, nf:][active] / scale[active]
            rows.append({'layout': field['layout'], 'arm': arm, 'joint_mixture_gate_passed': passed,
                'common_latent_order': ['f_common_coordinates', 'eta_common_orthogonal_coefficients'], 'f_common_dimension': nf,
                'common_joint_mean': mean.tolist(), 'within_component_common_covariance': within.tolist(),
                'between_component_common_covariance': between.tolist(), 'common_joint_covariance': covariance.tolist(),
                'maximum_absolute_cross_head_correlation': float(np.max(np.abs(association))),
                'minimum_joint_covariance_eigenvalue': minimum, 'full_posterior_projection_errors': errors,
                'positive_mixture_components': len(states), 'equal_weight': component['equal_positive_component_weight'],
                'joint_Gaussian_not_substituted': True})
            print('shared raw source mixture', field['layout'], arm, 'gate', passed, 'projection errors', errors,
                'max cross-head correlation', rows[-1]['maximum_absolute_cross_head_correlation'], flush=True)
    passed = all(r['joint_mixture_gate_passed'] for r in rows)
    save(out / 'reconstruction.json', {'execution': spec, 'rows': reconstructions, 'all_joint_mixture_gates_passed': passed})
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'all_four_joint_mixtures_complete': True,
        'all_joint_mixture_gates_passed': passed, 'diagnostic_interpreted': passed,
        'cpu_seconds': time.process_time() - started, 'cost': {**spec['cost'], 'actual_coordinate_restorations': 2},
        'validation_task0_or_task1_outcomes_indexed': False, 'new_raw_paid_forecasts': 0,
        'no_model_adopted': True, 'independent_confirmation': False,
        'interpretation': 'Positive source posterior mixture with common function/risk conditional heads and between-component cross dependence. Compact law reconstructs components;aggregate moments are diagnostics only. Task residuals remain separate. No held-task score,complete EM fit,risk-factor recovery or decision advantage.'})


if __name__ == '__main__':
    run()

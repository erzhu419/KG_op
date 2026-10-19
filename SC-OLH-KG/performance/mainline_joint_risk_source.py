"""Joint raw-source conditioning of the old g basis and public mean4 directions.

Only two nonnegative covariance amplitudes are learned per affected identity.
The original eta states are importance proposals, not unchanged posteriors.
Both their masses and the outer source-identity masses use the new likelihood.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import digamma, logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.connect_complete_historical_reuse import design_for
from performance.connect_historical_learned_prior_kg import PLAN, learned_law
from performance.continue_historical_raw_source_learning import load_mixture
from performance.check_legacy_f_raw_source_predictive_mixture import project_source_components
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior
from representation.joint_function_risk import JointFunctionRiskPosterior

OUTPUT = 'paper_artifacts/mainline_joint_risk_source_v1'
IDENTITY = ('layout', 'selector', 'support', 'treatment')
MIN_ESS = 100.
MAX_SPLIT_LOGZ_GAP = .1


def read(path):
    return json.loads((ROOT / path).read_text())


def save(path, value):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_calibration(path):
    return read(path)


def _source_context(plan=PLAN, layout='5'):
    connection = read(plan)
    parent = read(connection['source_plan'])
    selection = read(parent['selection_plan'])
    inputs = read(selection['source_inputs_by_layout'][layout])
    tasks = [t for t in inputs['source_tasks'] if t['name'] in selection['source_tasks']]
    assert [t['name'] for t in tasks] == selection['source_tasks']
    response = np.vstack([t['response'] for t in tasks])
    noise = np.vstack([t['noise'] for t in tasks])
    sample_variance = np.exp(response[:, 2] + digamma(1.))
    np.testing.assert_allclose(sample_variance / 3, noise[:, 1], atol=1e-14, rtol=1e-10)
    laws = {tuple(r[k] for k in (*IDENTITY, 'head')): r for r in read(parent['source_laws'])['rows']}
    cases = {tuple(r[k] for k in (*IDENTITY, 'head')): r for r in read(connection['source_learning'])['evidence_rows']}
    records = {tuple(r[k] for k in (*IDENTITY, 'head')): r for r in read(connection['source_precision'])['rows']}
    weights = read(connection['source_structure_weights'])['by_layout'][layout]
    pairs = []
    for identity in weights['identities']:
        if identity[1:3] != ['spectral', 'head_specific']:
            continue
        key = (*identity, 'g')
        case, record = cases[key], records[key]
        law = learned_law(laws[key], case['f_common_task_scales'])
        risk = learned_law(laws[(*identity, 'eta_g')], case['eta_common_task_scales'])
        eta, masses = load_mixture(record, case)
        pairs.append({'identity': identity, 'name': '/'.join(identity[1:]),
                      'law': law, 'risk': risk, 'eta': eta, 'masses': masses,
                      'component_file': record['positive_mixture_component_file']})
    assert len(pairs) == 2 and all(p['eta'].shape == (4, 256, 48) for p in pairs)
    return {'connection': connection, 'selection': selection,
            'maps': read(connection['maps'])['rows'], 'source_tasks': selection['source_tasks'],
            'source_inputs': selection['source_inputs_by_layout'][layout],
            'selection_plan': parent['selection_plan'], 'mean_coordinates': np.vstack([t['psi'] for t in tasks]),
            'labels': response[:, 1], 'sample_variances': sample_variance,
            'noise': noise[:, 1], 'pairs': pairs,
            'identities': weights['identities'], 'outer_weights': weights['equal_prior_source_posterior_weights']}


def joint_prior_moments(law, source_residual, query_design, query_residual, variances):
    """One augmented common/task Gaussian prior, before any source labels."""
    common, task = np.asarray(law['unconditioned_common_covariance']), np.asarray(law['task_covariance'])
    source_design = np.asarray(law['source_design'])
    shared_variance, task_variance = np.asarray(variances)
    source_kernel = np.asarray(law['source_covariance']) + shared_variance * source_residual @ source_residual.T
    source_kernel += task_variance * block_diag(*[h @ h.T for h in np.split(source_residual, 4)])
    query_kernel = query_design @ (common + task) @ query_design.T
    query_kernel += (shared_variance + task_variance) * query_residual @ query_residual.T
    query_kernel += law['point_discrepancy_variance'] * np.eye(len(query_design))
    cross = query_design @ common @ source_design.T + shared_variance * query_residual @ source_residual.T
    return {'source_mean': np.asarray(law['source_mean']), 'source_covariance': source_kernel,
            'query_mean': query_design @ law['unconditioned_common_mean'],
            'query_covariance': query_kernel, 'cross_covariance': cross}


def law_with_original_task_scale(law, multiplier):
    """Scale the original g task prior before joint source conditioning."""
    if multiplier == 1.:
        return law
    design = np.asarray(law['source_design'])
    task = np.asarray(law['task_covariance'])
    blocks = block_diag(*[d @ task @ d.T for d in np.split(design, 4)])
    return dict(law, task_covariance=multiplier*task,
        source_covariance=np.asarray(law['source_covariance'])+(multiplier-1)*blocks)


class SourceEvidence:
    """Finite positive-mixture estimate of Z(new)/Z(old) and its gradient."""

    def __init__(self, law, residual, labels, eta, masses):
        self.base = np.asarray(law['source_covariance'])
        self.error = labels - law['source_mean']
        self.noise = np.exp(eta.reshape(-1, len(labels))) / 3
        self.shape = masses.shape
        self.log_masses = np.log(masses.reshape(-1))
        self.common = residual
        self.task = block_diag(*np.split(residual, 4))
        self.loadings = np.column_stack([self.common, self.task])
        self.directions = [self.common @ self.common.T, self.task @ self.task.T]
        self.scale = float(np.var(labels))
        self.old_logs = self._evaluate(np.zeros(2), gradient=False)[0]

    def _evaluate(self, scaled_variances, gradient=True):
        covariance = self.base + self.scale * sum(t * k for t, k in zip(scaled_variances, self.directions))
        logs, gradients = [], []
        for noise in self.noise:
            factor = cho_factor(covariance + np.diag(noise), lower=True)
            alpha = cho_solve(factor, self.error)
            logs.append(-.5 * (len(noise) * np.log(2*np.pi)
                + 2*np.log(np.diag(factor[0])).sum() + self.error @ alpha))
            if gradient:
                inverse_loadings = cho_solve(factor, self.loadings)
                influence = self.loadings.T @ alpha
                terms = influence**2 - np.sum(self.loadings * inverse_loadings, axis=0)
                gradients.append(.5*self.scale*np.array([terms[:4].sum(), terms[4:].sum()]))
        return np.asarray(logs), np.asarray(gradients)

    def objective(self, scaled_variances):
        logs, gradient = self._evaluate(scaled_variances)
        log_weight = self.log_masses + logs - self.old_logs
        logz = logsumexp(log_weight)
        weights = np.exp(log_weight-logz)
        return -float(logz)/len(self.error), -(weights @ gradient)/len(self.error)

    def summarize(self, scaled_variances):
        logs, _ = self._evaluate(scaled_variances, gradient=False)
        ratios = logs-self.old_logs
        logz = float(logsumexp(self.log_masses+ratios))
        weights = np.exp(self.log_masses+ratios-logz)
        split = []
        for part in (slice(0, self.shape[1]//2), slice(self.shape[1]//2, None)):
            mass = self.log_masses.reshape(self.shape)[:, part]
            split.append(float(logsumexp(mass+ratios.reshape(self.shape)[:, part])-logsumexp(mass)))
        ess = float(1/(weights @ weights))
        gap = abs(split[0]-split[1])
        return {'new_component_weights': weights.tolist(), 'log_likelihood_ratios': ratios.tolist(),
                'log_source_evidence_ratio': logz, 'importance_diagnostics': {
                    'importance_ESS': ess, 'maximum_component_mass': float(weights.max()),
                    'source_chain_masses': weights.reshape(self.shape).sum(axis=1).tolist(),
                    'split_log_evidence_ratios': split, 'split_log_evidence_gap': gap,
                    'usable': bool(ess >= MIN_ESS and gap <= MAX_SPLIT_LOGZ_GAP and np.all(weights > 0))}}


def _fit_source(output=OUTPUT):
    started = time.process_time()
    context = _source_context()
    design = np.asarray(context['pairs'][0]['law']['source_design'])
    # This geometric projection uses the frozen source coordinates, no target
    # labels, task outcomes, prior residual coefficients, or new map selection.
    projection = np.linalg.lstsq(design, context['mean_coordinates'], rcond=None)[0]
    residual = context['mean_coordinates'] - design @ projection
    rows = []
    for pair in context['pairs']:
        phase = time.process_time()
        np.testing.assert_allclose(pair['law']['source_design'], design, atol=0., rtol=0.)
        evidence = SourceEvidence(pair['law'], residual, context['labels'], pair['eta'], pair['masses'])
        zero = evidence.summarize(np.zeros(2))
        np.testing.assert_allclose(zero['log_source_evidence_ratio'], 0., atol=1e-12, rtol=0.)
        np.testing.assert_allclose(zero['new_component_weights'], pair['masses'].reshape(-1), atol=1e-14, rtol=1e-12)
        fit = minimize(evidence.objective, np.zeros(2), method='L-BFGS-B', jac=True,
            bounds=[(0., None)]*2, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-7})
        summary = evidence.summarize(fit.x)
        row = {'identity': pair['identity'], 'name': pair['name'],
            'source_component_file': pair['component_file'], 'new_variances': (fit.x*evidence.scale).tolist(),
            'variance_parameter_scale': evidence.scale, 'optimizer_converged': bool(fit.success),
            'optimizer_iterations': int(fit.nit), 'optimizer_message': str(fit.message),
            'optimizer_scaled_gradient': fit.jac.tolist(), 'source_component_shape': list(pair['eta'].shape),
            'variance_at_zero_boundary': (fit.x == 0).tolist(),
            'zero_increment_log_evidence_ratio': zero['log_source_evidence_ratio'],
            **summary, 'cpu_seconds': time.process_time()-phase}
        rows.append(row)
        save(f'{output}/{pair["name"].replace("/", "_")}.json', row)
        print(pair['name'], 'variance', row['new_variances'], 'ESS', summary['importance_diagnostics']['importance_ESS'],
              'split', summary['importance_diagnostics']['split_log_evidence_gap'], flush=True)
    ratios = {r['name']: r['log_source_evidence_ratio'] for r in rows}
    outer_logs = np.log(context['outer_weights']) + [ratios.get('/'.join(i[1:]), 0.) for i in context['identities']]
    new_outer = np.exp(outer_logs-logsumexp(outer_logs))
    usable = all(r['optimizer_converged'] and r['importance_diagnostics']['usable'] for r in rows)
    calibration = {'layout': '5', 'source_connection_plan': PLAN,
        'source_inputs': context['source_inputs'], 'source_task_ids': context['source_tasks'], 'source_rows': 48,
        'source_replicates_per_row': 3, 'maps': context['connection']['maps'],
        'selection_plan': context['selection_plan'], 'projection_coefficients': projection.tolist(),
        'residual_coordinate_rank': int(np.linalg.matrix_rank(residual)), 'rows': rows,
        'identities': context['identities'], 'old_outer_source_weights': context['outer_weights'],
        'new_outer_source_weights': new_outer.tolist(), 'source_importance_usable': usable,
        'importance_limits': {'minimum_ESS': MIN_ESS, 'maximum_split_log_evidence_gap': MAX_SPLIT_LOGZ_GAP},
        'construction': 'Old g basis and four extra mean directions jointly conditioned through one raw-source likelihood; g/eta component masses and complete identity masses updated by the same evidence ratio.',
        'source_fit_objective': 'Maximize logsumexp(log old eta mass + Gaussian batch-mean loglik(new)-loglik(old)); old eta prior and three-replicate sample-variance likelihood cancel in the ratio. Divide by48 for optimizer scale.',
        'optimizer': {'method': 'L-BFGS-B', 'start': [0., 0.], 'bounds': 'two nonnegative variances',
                      'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-7},
        'f_and_eta_prior_parameters_changed': False, 'target_observations_used': False,
        'posterior_precision_certified': False, 'source_fits': 2, 'new_source_labels': 0,
        'new_target_calls': 0, 'source_sampling_sweeps': 0, 'cpu_seconds': time.process_time()-started}
    save(f'{output}/calibration.json', calibration)
    return calibration


def fit_source(output=OUTPUT):
    started = time.process_time()
    attempt = {'status': 'failed', 'new_target_calls': 0}
    try:
        result = _fit_source(output)
        attempt['status'] = 'completed_usable' if result['source_importance_usable'] else 'importance_refresh_required'
        attempt['source_fits'] = result['source_fits']
        return result
    except Exception as error:
        attempt['exception'] = repr(error)
        raise
    finally:
        attempt['cpu_seconds'] = time.process_time()-started
        path = ROOT/output/'attempts.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as stream:
            stream.write(json.dumps(attempt)+'\n')


def _restore_joint_g(pair, context, coordinates, projection, row):
    law = law_with_original_task_scale(pair['law'], row.get('original_task_covariance_multiplier', 1.))
    risk = pair['risk']
    mean, eta_coordinates = (np.asarray(coordinates[k]) for k in ('mean', 'risk'))
    D = design_for(law, (mean, eta_coordinates), context['maps'], context['selection'])
    E = design_for(risk, (mean, eta_coordinates), context['maps'], context['selection'])
    Hs = context['mean_coordinates'] - np.asarray(law['source_design']) @ projection
    Hq = mean-D @ projection
    prior = joint_prior_moments(law, Hs, D, Hq, row['new_variances'])
    eta = pair['eta'].reshape(-1, 48)
    B, Es = np.asarray(risk['unconditioned_common_covariance']), np.asarray(risk['source_design'])
    risk_cross = E @ B @ Es.T
    gain = cho_solve(cho_factor(np.asarray(risk['source_covariance']), lower=True), risk_cross.T).T
    risk_mean = (E @ risk['unconditioned_common_mean'])[None] + (eta-risk['source_mean']) @ gain.T
    risk_covariance = E @ (B+risk['task_covariance']) @ E.T
    risk_covariance += risk['point_discrepancy_variance']*np.eye(len(E))-gain @ risk_cross.T
    risk_covariance = .5*(risk_covariance+risk_covariance.T)
    model = VarianceReferencePosterior(prior['source_mean'], prior['source_covariance'],
                                      context['labels'], context['sample_variances'])
    completed = project_source_components(model, eta, prior['query_mean'], prior['query_covariance'],
        prior['cross_covariance'], risk_mean, risk_covariance)
    count, n = len(eta), len(D)
    covariance = np.zeros((count, 2*n, 2*n))
    covariance[:, ::2, ::2] = completed['f_component_covariances']
    covariance[:, 1::2, 1::2] = completed['risk_component_covariances']
    posterior = JointFunctionRiskPosterior.from_moments(completed['component_means'][0],
                                                      covariance[0].reshape(n, 2, n, 2))
    posterior.means = completed['component_means'].reshape(count, 2*n).copy()
    posterior.covariances = covariance
    posterior.log_weights = np.log(row['new_component_weights'])
    posterior.component_pairs = [(0, 0)]*count
    posterior.g_structural = posterior.eta_structural = [True]
    return posterior


def apply_joint_risk_source(model, coordinates, calibration):
    """Replace two jointly conditioned g fields before any target update."""
    assert model.paid_episodes == 0 and not model.history
    if not calibration['source_importance_usable']:
        raise ValueError('The changed source posterior requires a source-only importance refresh')
    context = _source_context(calibration['source_connection_plan'], calibration['layout'])
    names = ['/'.join(i[1:]) for i in calibration['identities']]
    assert model.names == names
    np.testing.assert_allclose(model.weights, calibration['old_outer_source_weights'], atol=1e-14, rtol=1e-12)
    projection = np.asarray(calibration['projection_coefficients'])
    affected = []
    for pair, row in zip(context['pairs'], calibration['rows']):
        assert pair['name'] == row['name']
        index = model.names.index(row['name'])
        f, old_g = model.components[index]
        new_g = _restore_joint_g(pair, context, coordinates, projection, row)
        model.components[index] = (f, new_g)
        affected.append({'identity': row['name'], 'component_count': len(new_g.means),
                         'original_task_covariance_multiplier': row.get('original_task_covariance_multiplier', 1.),
                         'importance_ESS': row['importance_diagnostics']['importance_ESS'],
                         'log_source_evidence_ratio': row['log_source_evidence_ratio']})
        del old_g
    model.log_weights = np.log(calibration['new_outer_source_weights'])
    return {'affected_identities': affected, 'outer_source_weights_before': calibration['old_outer_source_weights'],
        'outer_source_weights_after': model.weights.tolist(), 'source_component_masses_reweighted': True,
        'original_and_extra_g_directions_jointly_conditioned': True,
        'paid_episodes': 0, 'source_fits': 0, 'new_target_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=OUTPUT)
    args = parser.parse_args()
    result = fit_source(args.output)
    print(json.dumps({'source_importance_usable': result['source_importance_usable'],
                      'cpu_seconds': result['cpu_seconds']}))

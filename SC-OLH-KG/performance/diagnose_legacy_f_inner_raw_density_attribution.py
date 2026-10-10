"""Split cached joint scores into log-variance and conditional-mean densities."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import minimize_scalar
from scipy.special import digamma, logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save
from performance.vectorized_joint_batch_density import component_batch_log_evidence
from representation.physical_risk_prior import log_sample_variance_density

EXECUTION = 'performance/manifests/legacy_joint_population_f_inner_raw_density_attribution_execution_v1.json'


def log_integral(at):
    count = len(at(0.))
    modes = np.asarray([minimize_scalar(lambda t: -at(t, i), bounds=(-12., 12.), method='bounded').x for i in range(count)])
    shifts = at(modes)
    integral, error, info = quad_vec(lambda t: np.exp(at(t)-shifts), -12., 12.,
        points=np.unique([modes.min(), np.median(modes), modes.max()]), norm='max', full_output=True,
        epsabs=1e-10, epsrel=1e-10, workers=1)
    if not info.success: raise RuntimeError(info.message)
    assert np.all(integral > 0)
    relative = float(error/integral.min()); assert relative <= 1e-7
    return shifts+np.log(integral), relative


def component_variance_density(eta_mean, eta_variance, eta_label, replicates=3):
    sd = np.sqrt(eta_variance); k = (replicates-1)/2
    s2 = float(np.exp(eta_label+digamma(k)-np.log(k)))
    def at(t, index=slice(None)):
        return -.5*(np.log(2*np.pi)+t*t)+log_sample_variance_density(eta_mean[index]+sd[index]*t, s2, replicates)
    return log_integral(at)


def factorized_conditional_mean(means, covariance, label, eta_label, log_variance_evidence, replicates=3):
    """Direct expectation under eta|z;keep conditional f/eta loading and weights."""
    gm, em = means.T; gv, ev = covariance[:, 0, 0], covariance[:, 1, 1]
    sd = np.sqrt(ev); cross = covariance[:, 0, 1]
    conditional = np.maximum(gv-cross**2/ev, 0.); loading = cross/sd
    k = (replicates-1)/2; s2 = float(np.exp(eta_label+digamma(k)-np.log(k)))
    def at(t, index=slice(None)):
        eta = em[index]+sd[index]*t; variance = conditional[index]+np.exp(eta)/replicates
        risk_posterior = (-.5*(np.log(2*np.pi)+t*t)+log_sample_variance_density(eta, s2, replicates)-log_variance_evidence[index])
        return risk_posterior-.5*(np.log(2*np.pi*variance)+(label-gm[index]-loading[index]*t)**2/variance)
    return log_integral(at)


def decomposition(state, labels, verify_first_site=False):
    sites, checks, max_error = [], [], 0.
    for j, label in enumerate(labels):
        means, covariance = state['component_means'][:, j], state['component_local_covariances'][:, j]
        log_variance, error = component_variance_density(means[:, 1], covariance[:, 1, 1], label[3])
        joint, bound = component_batch_log_evidence(means, covariance, label[0], label[3], 3)
        variance_density = float(logsumexp(state['log_weights']+log_variance))
        joint_density = float(logsumexp(state['log_weights']+joint))
        sites.append({'bias_corrected_log_sample_variance_density': variance_density,
            'batch_mean_given_log_sample_variance_density': joint_density-variance_density,
            'joint_batch_density': joint_density})
        max_error = max(max_error, error, bound)
        if verify_first_site and j == 0:
            conditional, conditional_error = factorized_conditional_mean(means, covariance, label[0], label[3], log_variance)
            posterior_log_weights = state['log_weights']+log_variance-variance_density
            combined = variance_density+float(logsumexp(posterior_log_weights+conditional))
            difference = abs(combined-joint_density)
            np.testing.assert_allclose(combined, joint_density, atol=1e-8, rtol=1e-8)
            checks.append({'independent_factorized_joint_density_error': difference,
                'conditional_integration_relative_error_bound': conditional_error, 'f_eta_correlation_retained': True})
    means = {key: float(np.mean([s[key] for s in sites])) for key in sites[0]}
    return {'means': means, 'sites': sites, 'factorization_checks': checks, 'maximum_integration_relative_error_bound': max_error}


def run():
    started = time.process_time(); spec = read(EXECUTION); plan = read(spec['plan']); source = read(plan['source_results'])
    assert source['diagnostic_interpreted'] and source['numerical_density_verified']
    coverage, _ = source_paths(read(plan['source_context_execution'])); folder = ROOT/source['execution']['output']
    rows, refinements = [], []
    for row in source['rows']:
        if row['design'] != 'fixed_public_prefix' or row['arm'] not in spec['arms']: continue
        inputs = read(Path(coverage['output'])/f"layout{row['layout']}"/'task1'/'inputs.json')
        task = next(t for t in inputs['source_tasks'] if t['name'] == f"related_inventory_{row['query_task']:02d}")
        labels = np.asarray(task['response'])[4:]
        assert row['scoring_sites'] == coverage['layouts'][row['layout']][row['query_task']][4:]
        with np.load(folder/row['selected_component_file']) as state:
            values = decomposition(state, labels, row['query_task'] == 0)
        replay_error = abs(values['means']['joint_batch_density']-row['scores']['f_raw_batch_density'])
        assert replay_error <= spec['cached_score_tolerance']
        if 'refined_component_file' in row:
            with np.load(folder/row['refined_component_file']) as state: refined = decomposition(state, labels)
            differences = {k: abs(values['means'][k]-refined['means'][k]) for k in values['means']}
            refinements.append({'layout': row['layout'], 'query_task': row['query_task'], 'errors': differences,
                'passed': max(differences.values()) <= spec['refinement_tolerance']})
        rows.append({'layout': row['layout'], 'query_task': row['query_task'], 'arm': row['arm'],
            **values, 'cached_joint_score_replay_error': replay_error})
    assert len(rows) == 30 and len(refinements) == 10
    summary = {}; contrasts = []
    for layout in ('5', '6'):
        summary[layout] = {}
        for arm in ('shape_point', 'shape_mixture'):
            members = []
            for query in (0, 2, 3, 4, 5):
                gp = next(r for r in rows if r['layout'] == layout and r['query_task'] == query and r['arm'] == 'source_gp')
                candidate = next(r for r in rows if r['layout'] == layout and r['query_task'] == query and r['arm'] == arm)
                delta = {k: gp['means'][k]-candidate['means'][k] for k in gp['means']}
                np.testing.assert_allclose(delta['joint_batch_density'], delta['bias_corrected_log_sample_variance_density']+delta['batch_mean_given_log_sample_variance_density'], atol=1e-10, rtol=1e-10)
                members.append(delta); contrasts.append({'layout': layout, 'query_task': query, 'gp_minus_arm': arm, 'differences': delta})
            mean = {k: float(np.mean([d[k] for d in members])) for k in members[0]}
            mean['variance_density_fraction_of_gp_joint_gain'] = mean['bias_corrected_log_sample_variance_density']/mean['joint_batch_density']
            summary[layout][arm] = mean
    verified = all(r['passed'] for r in refinements)
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    result = {'execution': spec, 'rows': rows, 'contrasts': contrasts, 'summary': summary, 'refinement_checks': refinements,
        'decomposition_numerically_verified': verified, 'diagnostic_interpreted': verified,
        'observation_measure': 'batch mean and bias-corrected log sample variance;log-variance transformation Jacobian included',
        'cpu_seconds': time.process_time()-started, 'cost': plan['cost'], 'no_model_adopted': True,
        'no_new_unscored_outcomes_read': True, 'source6_7_evaluated': False, 'target_promoted': False,
        'independent_confirmation': False, 'training_diagnostic_only': True}
    save(out/'results.json', result)
    print('raw density decomposition', summary, 'verified', verified, 'CPU', result['cpu_seconds'], flush=True)


if __name__ == '__main__':
    run()

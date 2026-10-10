"""Cover variance tails with the old source kernel and undo the sampling tilt."""

import signal
import time

import numpy as np
from scipy.linalg import cho_solve
from scipy.special import digamma, polygamma

from performance.complete_historical_source_heads import (
    ROOT, IDENTITY, common_moments, cpu_timeout, log_density, read, sample,
    save, source_case, variance_reference, VarianceReferencePosterior, verify_learned,
)
from performance.check_legacy_f_correlated_source_raw_update import diagnostics
from performance.historical_source_block_transition import source_block_transition

PLAN = 'performance/manifests/historical_source_variance_resolution_v1.json'


class VarianceTiltedPosterior(VarianceReferencePosterior):
    """Sample p(eta)*h(eta); inverse-h weights recover the frozen posterior p.

    h is positive and fixed from the earlier source-only precision pilot.
    Gaussian population parameters and the raw likelihood are unchanged.
    """

    def __init__(self, mean, covariance, labels, sample_variances, sites, center, scale):
        super().__init__(mean, covariance, labels, sample_variances)
        self.sites = np.asarray(sites)
        self.center, self.scale = np.asarray(center), np.asarray(scale)
        self.rows = self.covariance[self.sites]

    def log_tilt(self, evaluation):
        _, factor, alpha = evaluation
        mean = self.mean[self.sites] + self.rows @ alpha
        variance = np.diag(self.covariance)[self.sites] - np.sum(
            self.rows * cho_solve(factor, self.rows.T).T, axis=1)
        functional = (variance + (mean - self.center[self.sites]) ** 2) / self.scale[self.sites]
        return float(np.log1p(functional.mean()))

    def evaluate(self, eta):
        evaluation = super().evaluate(eta)
        return evaluation[0] + self.log_tilt(evaluation), evaluation[1], evaluation[2]

    def inverse_weights(self, displacements, covariances):
        means = self.mean[self.sites] + displacements[..., self.sites]
        diagonal = np.diagonal(covariances, axis1=-2, axis2=-1)[..., self.sites]
        functional = (diagonal + (means - self.center[self.sites]) ** 2) / self.scale[self.sites]
        return 1. / (1. + functional.mean(axis=-1))


def weighted_moments(draws, weights):
    flat = np.asarray(draws).reshape(-1, draws.shape[-1])
    w = np.asarray(weights).reshape(-1); w = w / w.sum()
    mean = w @ flat; centered = flat - mean
    return mean, (centered.T * w) @ centered


def weighted_function_moments(displacements, covariances, weights):
    mean, between = weighted_moments(displacements, weights)
    w = weights / weights.sum()
    within = np.einsum('ct,ctij->ij', w, covariances)
    return mean, within + between


def original_posterior_summary(model, eta, ds, cs, inverse, spec):
    """Ratio-estimator influence chains diagnose the original p expectations."""
    count = eta.shape[1]; delta, covariance = weighted_function_moments(ds, cs, inverse)
    em, ec = weighted_moments(eta, inverse)
    half = [weighted_function_moments(ds[:, part], cs[:, part], inverse[:, part])
            for part in (slice(0, count // 2), slice(count // 2, None))]
    half_variance = [np.diag(item[1]) for item in half]
    difference = np.abs(half_variance[0] - half_variance[1]) / np.mean(half_variance, axis=0)
    moments = np.concatenate([eta, ds, np.diagonal(cs, axis1=-2, axis2=-1) + ds ** 2], axis=-1)
    expectation = np.einsum('ct,ctj->j', inverse / inverse.sum(), moments)
    influence = inverse[..., None] * (moments - expectation) / inverse.mean()
    # Variance uses a central second moment. Include its influence and the
    # normalizer as well as the existing eta/mean/second-moment coordinates.
    variance_function = np.diagonal(cs, axis1=-2, axis2=-1) + (ds - delta) ** 2
    variance_influence = inverse[..., None] * (variance_function - np.diag(covariance)) / inverse.mean()
    coordinates = [influence, variance_influence]
    normalizer = inverse / inverse.mean() - 1.
    # Equal retained weights have an exact normalizer,not an MCMC coordinate.
    if np.any(normalizer != 0.):
        coordinates.append(normalizer[..., None])
    diagnostic = diagnostics(np.concatenate(coordinates, axis=-1))
    max_rhat = max(diagnostic['split_rhat']); min_ess = min(diagnostic['autocorrelation_ess'])
    gate = spec['monte_carlo_gate']
    passed = (max_rhat <= gate['maximum_split_rhat_eta_and_rao_blackwell_f_moments']
        and min_ess >= gate['minimum_autocorrelation_ess_eta_and_rao_blackwell_f_moments']
        and difference.max() <= gate['maximum_first_second_half_f_variance_relative_difference'])
    return {'posterior_mean': (model.mean + delta).tolist(), 'posterior_covariance': covariance.tolist(),
        'eta_posterior_mean': em.tolist(), 'eta_posterior_covariance': ec.tolist(),
        'first_half_posterior_mean': (model.mean + half[0][0]).tolist(),
        'second_half_posterior_mean': (model.mean + half[1][0]).tolist(),
        'first_half_posterior_variance': half_variance[0].tolist(),
        'second_half_posterior_variance': half_variance[1].tolist(),
        'maximum_first_second_half_f_variance_relative_difference': float(difference.max()),
        'half_variance_relative_differences': difference.tolist(),
        'maximum_split_rhat': float(max_rhat), 'minimum_autocorrelation_ess': float(min_ess),
        'monte_carlo_gate_passed': bool(passed), 'diagnostics': diagnostic,
        'diagnostic_coordinate_order': 'ratio-influence:eta48,conditional_f_mean48,second_about_prior48,central_variance48'
            + (',normalizer1' if np.any(normalizer != 0.) else ';constant normalizer omitted'),
        'normalized_inverse_weight_ESS': float(inverse.sum() ** 2 / np.sum(inverse ** 2))}


def retained_weighted_mixture(model, eta, ds, cs, inverse, posterior, spec):
    stride = spec['retained_mixture']['stride_per_chain']
    kept = eta[:, ::stride]; weights = inverse[:, ::stride]
    weights = weights / weights.sum()
    delta, covariance = weighted_function_moments(ds[:, ::stride], cs[:, ::stride], weights)
    full_mean = np.asarray(posterior['posterior_mean']); full_var = np.diag(posterior['posterior_covariance'])
    mean_error = np.sqrt(np.mean((model.mean + delta - full_mean) ** 2 / full_var))
    variance_error = np.max(np.abs(np.diag(covariance) - full_var) / full_var)
    gate = spec['retained_mixture']
    return {'eta_states_by_chain': kept.tolist(), 'positive_component_weights': weights.reshape(-1).tolist(),
        'component_count': len(weights.reshape(-1)), 'stride_per_chain': stride,
        'source_f_mean_rms_difference_in_full_posterior_sd': float(mean_error),
        'maximum_source_f_diagonal_variance_relative_difference_vs_all_draws': float(variance_error),
        'mixture_gate_passed': bool(mean_error <= gate['maximum_source_f_mean_rms_difference_in_full_posterior_sd']
            and variance_error <= gate['maximum_source_f_diagonal_variance_relative_difference_vs_all_draws']),
        'weights_correct_to_original_frozen_posterior': True}


def run(plan=PLAN):
    spec = read(plan)
    result_path = ROOT / spec['output'] / 'source_results.json'
    if result_path.exists(): return read(str(result_path.relative_to(ROOT)))
    previous = read(spec['previous_source_registry'])
    rows = [dict(r) for r in previous['rows']]
    parent = read(spec['covariance_plan']); selection = read(spec['selection_plan'])
    laws = read(spec['source_laws'])['rows']; fields = read(parent['source_fields'])['rows']
    out = ROOT / spec['output']; (out / 'source_cases').mkdir(parents=True, exist_ok=True)
    started = time.process_time(); checks = []; costs = []; stopped = False
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_single_thread_cpu_seconds'])
    try:
        for i, old in enumerate(previous['rows']):
            if old['source_posterior_usable']: continue
            one = time.process_time(); identity = {k: old[k] for k in IDENTITY}; head = old['head']
            law = next(r for r in laws if r['head'] == head and all(r[k] == old[k] for k in IDENTITY))
            risk = next(r for r in laws if r['head'] == ('eta_f' if head == 'f' else 'eta_g')
                        and all(r[k] == old[k] for k in IDENTITY))
            source = dict(next(r for r in fields if (r['layout'], r['active_arm']) ==
                          (old['layout'], parent['proposal_arm'])))
            if head == 'g':
                tasks = read(selection['source_inputs_by_layout'][old['layout']])['source_tasks']
                response = np.vstack([t['response'] for t in tasks if t['name'] in selection['source_tasks']])
                source['batch_means'] = response[:, 1].tolist()
                source['sample_variances'] = np.exp(response[:, 2] + digamma(1.)).tolist()
            field = source_case(law, risk, source); pilot = read(old['component_file'])
            center = np.asarray(pilot['source_f_posterior_mean']); scale = np.asarray(pilot['source_f_posterior_variance'])
            hv = [np.asarray(pilot[k + '_half_posterior_variance']) for k in ('first', 'second')]
            sites = np.argsort(np.abs(hv[0] - hv[1]) / np.mean(hv, axis=0))[-spec['tilt_sites_per_case']:]
            if old.get('variance_tilt_corrected', False):
                center = np.asarray(pilot['tilt']['center']); scale = np.asarray(pilot['tilt']['scale'])
                sites = np.asarray(pilot['tilt']['sites'])
            model = VarianceTiltedPosterior(field['f_means'][0], field['f_covariances'][0],
                field['batch_means'], field['sample_variances'], sites, center, scale)
            base = VarianceReferencePosterior(model.mean, model.covariance, model.labels, model.s2)
            mean, covariance = variance_reference(field['eta_mean'], field['eta_covariance'], model.s2)
            reference = {'reference_mean': mean, 'reference_covariance': covariance,
                'approximate_Gaussian_variance_evidence': log_density(np.log(model.s2) - digamma(1.) - field['eta_mean'],
                    np.asarray(field['eta_covariance']) + float(polygamma(1, 1.)) * np.eye(48))}
            transition, block_check = source_block_transition(mean, covariance,
                spec['sampler'].get('joint_direction_sites', ()))
            errors = [abs(model.evaluate(p)[0] - model.log_tilt(model.evaluate(p)) - base.evaluate(p)[0])
                      for p in (mean, mean + .1 * np.sqrt(np.diag(covariance)))]
            assert max(errors) < 1e-10
            checks.append({**identity, 'head': head, **verify_learned(field, 0, reference), **block_check,
                'inverse_tilt_density_identity_max_error': max(errors), 'frozen_tilt_source_sites': sites.tolist()})
            execution = {**spec, 'sampler': {**spec['sampler'], 'seed': spec['sampler']['seed']
                + 1000 * (old['selector'] == 'additive') + 10 * (old['support'] == 'joint')
                + (10000 if head == 'g' else 0)}}
            captured = []; moments = []
            proposal = sample(field, 0, execution, started, model, mean, covariance, captured,
                transition=transition, retained_moments=moments)
            eta = captured[0]; ds, cs = moments[0]; inverse = model.inverse_weights(ds, cs)
            # Check the weights obtained from retained moments against the
            # exact likelihood evaluation actually used by the transition.
            weight_errors = [abs(np.log(inverse[c, t]) + model.log_tilt(model.evaluate(eta[c, t])))
                             for c, t in ((0, 0), (1, 500), (3, -1))]
            assert max(weight_errors) < 1e-9
            checks[-1]['inverse_weights_from_moments_max_log_error'] = max(weight_errors)
            posterior = original_posterior_summary(model, eta, ds, cs, inverse, execution)
            mixture = retained_weighted_mixture(model, eta, ds, cs, inverse, posterior, execution)
            cm, cc = common_moments(field, 0, np.asarray(posterior['posterior_mean']),
                                   np.asarray(posterior['posterior_covariance']))
            path = out / 'source_cases' / (head + '_layout' + '_'.join(identity.values()) + '.json')
            row = {**identity, 'head': head, 'reused_completed_case': False, 'variance_tilt_corrected': True,
                **{k: posterior[k] for k in ('monte_carlo_gate_passed', 'maximum_split_rhat', 'minimum_autocorrelation_ess',
                    'maximum_first_second_half_f_variance_relative_difference')},
                'elliptical_likelihood_evaluations_by_chain': proposal['elliptical_likelihood_evaluations_by_chain'],
                'mixture_gate_passed': mixture['mixture_gate_passed'],
                'source_posterior_usable': bool(posterior['monte_carlo_gate_passed'] and mixture['mixture_gate_passed']),
                'retained_sweeps_per_chain': execution['sampler']['retained_sweeps_per_chain'],
                'component_file': str(path.relative_to(ROOT)), 'previous_component_file': old['component_file']}
            save(path, {'execution_plan': plan, **identity, 'head': head, 'gates': row, 'source_laws': spec['source_laws'],
                'positive_eta_mixture': mixture, 'common_coefficient_posterior_mean': cm.tolist(),
                'common_coefficient_posterior_covariance': cc.tolist(),
                'source_f_posterior_mean': posterior['posterior_mean'],
                'source_f_posterior_variance': np.diag(posterior['posterior_covariance']).tolist(),
                'full_source_eta_posterior_mean': posterior['eta_posterior_mean'],
                'full_source_eta_posterior_covariance': posterior['eta_posterior_covariance'],
                **{k: posterior[k] for k in ('diagnostics', 'diagnostic_coordinate_order', 'first_half_posterior_mean',
                    'second_half_posterior_mean', 'first_half_posterior_variance', 'second_half_posterior_variance',
                    'normalized_inverse_weight_ESS')},
                'tilt': {'sites': sites.tolist(), 'center': center.tolist(), 'scale': scale.tolist(),
                    'rule': 'h=1+mean((conditional_f_variance+(conditional_f_mean-center)^2)/scale);weights=1/h'},
                'proposal_diagnostics': {k: proposal[k] for k in ('maximum_split_rhat', 'minimum_autocorrelation_ess')}})
            rows[i] = row; costs.append({**identity, 'head': head, 'cpu_seconds': time.process_time() - one})
            print('variance resolved', head, *identity.values(), 'usable', row['source_posterior_usable'],
                'ESS', round(row['minimum_autocorrelation_ess'], 1),
                'halfvar', round(row['maximum_first_second_half_f_variance_relative_difference'], 3),
                'retention', round(mixture['maximum_source_f_diagonal_variance_relative_difference_vs_all_draws'], 3), flush=True)
            del captured, moments, eta, ds, cs
    except TimeoutError: stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    save(out / 'source_verification.json', {'checks': checks, 'execution_plan': plan})
    result = {'rows': rows, 'execution_plan': plan, 'usable_cases': sum(r['source_posterior_usable'] for r in rows),
        'reused_usable_cases': sum(r['source_posterior_usable'] for r in previous['rows']), 'resolved_cases_completed': len(costs),
        'all_source_posteriors_usable': all(r['source_posterior_usable'] for r in rows),
        'case_costs': costs, 'cpu_seconds': time.process_time() - started, 'stopped_at_cpu_budget': stopped,
        'cost': spec['cost'], 'original_population_prior_and_raw_likelihood_unchanged': True,
        'no_model_adopted': True, 'independent_confirmation': False}
    save(out / 'source_results.json', result)
    return result


if __name__ == '__main__': run()

"""Proper full-support uncertainty in the source task root amplitude."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_source_spectral_uncertainty import source_model
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_information_ceiling import variance_parts
from performance.run_legacy_related_inventory_source import save

PLAN = 'performance/manifests/legacy_joint_population_f_source_task_amplitude_uncertainty_plan_v1.json'


def finite_moment_verification(model):
    shape = model.columns[1:]
    assert np.linalg.eigvalsh(model.templates[1][np.ix_(shape, shape)]).min() > 0
    rank = sum(int(np.linalg.matrix_rank(d[:, shape])) for d in model.designs)
    assert rank > 1
    return {'source_large_amplitude_shape_rank': rank, 'posterior_amplitude_second_moment_finite': True,
        'reason': 'Source likelihood is O(a^-rank);proper half-Cauchy density is O(a^-2). Thus the second-moment tail is O(a^-rank),integrable for rank>1. Positive known noise and fixed positive tails keep the finite-a source covariance nonsingular.'}


def amplitude_rule(model, parameters, order):
    z, w = np.polynomial.legendre.leggauss(order); u, mass = [], []
    for lower, upper in ((0., .25), (.25, .5), (.5, .75), (.75, 1.)):
        u.extend(.5 * (upper - lower) * z + .5 * (upper + lower)); mass.extend(.5 * (upper - lower) * w)
    nodes = np.tan(.5 * np.pi * np.asarray(u)); states, values = [], []
    for amplitude in nodes:
        theta = np.asarray(parameters).copy(); theta[2] = amplitude
        value, _, state = model.evaluate(theta); values.append(value); states.append(state)
    log_mass = np.log(mass) - model.label_count * np.asarray(values)
    log_evidence = float(logsumexp(log_mass)); weights = np.exp(log_mass - log_evidence)
    np.testing.assert_allclose(weights.sum(), 1., atol=1e-12, rtol=0.)
    return {'order_per_interval': order, 'amplitude_nodes': nodes, 'cdf_nodes': np.asarray(u),
        'quadrature_prior_mass': np.asarray(mass), 'source_nll_per_label': np.asarray(values),
        'log_evidence': log_evidence, 'posterior_weights': weights, 'states': states}


def moments(rule, model, public):
    weights, states = rule['posterior_weights'], rule['states']
    beta = np.asarray([s['source_posterior_coefficient_mean'] for s in states]); mean = weights @ beta
    covariances = np.asarray([s['source_posterior_common_covariance'] + s['task_coefficient_covariance'] for s in states])
    within = np.einsum('k,kij->ij', weights, covariances); centered = beta - mean
    between = (centered.T * weights) @ centered; covariance = within + between
    second = np.einsum('k,kij->ij', weights, covariances + beta[:, :, None] * beta[:, None, :]) - np.outer(mean, mean)
    np.testing.assert_allclose(covariance, second, atol=1e-11, rtol=1e-9)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-10
    design = np.column_stack([np.ones(len(public)), model.basis.transform(public)])
    f_mean = design @ mean; f_covariance = design @ covariance @ design.T + model.white * np.eye(len(public))
    a_mean = float(weights @ rule['amplitude_nodes']); a_second = float(weights @ rule['amplitude_nodes'] ** 2)
    assert np.isfinite(a_second) and np.all(np.isfinite(f_covariance))
    record = {k: rule[k].tolist() for k in ('amplitude_nodes', 'cdf_nodes', 'quadrature_prior_mass', 'source_nll_per_label', 'posterior_weights')}
    record.update(order_per_interval=rule['order_per_interval'], log_evidence=rule['log_evidence'],
        amplitude_mean=a_mean, amplitude_second_moment=a_second, amplitude_variance=a_second - a_mean ** 2,
        probability_amplitude_ge_1=float(weights[rule['cdf_nodes'] >= .5].sum()),
        coefficient_mean=mean.tolist(), within_coefficient_covariance=within.tolist(), between_coefficient_covariance=between.tolist(),
        coefficient_covariance=covariance.tolist(), public_variance_parts=variance_parts(f_covariance),
        between_component_public_variance_parts=variance_parts(design @ between @ design.T))
    return record, f_mean, f_covariance


def run():
    started = time.process_time(); spec = read(PLAN); information = read(spec['information_results']); assert information['diagnostic_interpreted']
    source_spec = read(spec['source_probe_plan']); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True); rows = []
    base = {'execution': spec, 'query_outcomes_indexed': False, 'no_model_adopted': True, 'point_candidate_remains_stopped': True,
        'source6_7_evaluated': False, 'independent_confirmation': False}
    try:
        for layout in spec['layouts']:
            model, law, public = source_model(source_spec, layout, spec['query_task']); theta = np.asarray(law['parameters'])
            nll, _, state = model.evaluate(theta); np.testing.assert_allclose(nll, law['source_nll_per_label'], atol=1e-10, rtol=1e-10)
            for k in state: np.testing.assert_allclose(state[k], law[k], atol=1e-10, rtol=1e-10)
            row = {'layout': layout, 'query_task': spec['query_task'], 'source_task_names': law['source_task_names'],
                'point_parameters': law['parameters'], 'source_point_law_reproduced': True,
                'finite_moment_verification': finite_moment_verification(model), 'rules': []}; rows.append(row); public_moments = []
            for order in (16, 32):
                if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']: raise TimeoutError('amplitude integration CPU cap')
                record, mean, covariance = moments(amplitude_rule(model, theta, order), model, public)
                row['rules'].append(record); public_moments.append((mean, covariance))
            a, r = row['rules']; tolerances = spec['tolerances']
            errors = {'log_evidence_absolute': abs(a['log_evidence'] - r['log_evidence']),
                'amplitude_mean_absolute': abs(a['amplitude_mean'] - r['amplitude_mean']),
                'amplitude_second_moment_absolute': abs(a['amplitude_second_moment'] - r['amplitude_second_moment']),
                'f_mean_max_absolute': float(np.max(np.abs(public_moments[0][0] - public_moments[1][0]))),
                'f_covariance_relative_frobenius': float(np.linalg.norm(public_moments[0][1] - public_moments[1][1]) / np.linalg.norm(public_moments[1][1]))}
            row['numerical_errors'] = errors; row['numerical_passed'] = all(errors[k] <= tolerances[k] for k in tolerances)
            d = np.column_stack([np.ones(len(public)), model.basis.transform(public)]); point_mean = d @ state['source_posterior_coefficient_mean']
            point_covariance = d @ (state['source_posterior_common_covariance'] + state['task_coefficient_covariance']) @ d.T + model.white * np.eye(len(public))
            row['changes_vs_point_law'] = {'public_mean_max_absolute': float(np.max(np.abs(public_moments[0][0] - point_mean))),
                'public_mean_rms': float(np.sqrt(np.mean((public_moments[0][0] - point_mean) ** 2))),
                'public_covariance_relative_frobenius': float(np.linalg.norm(public_moments[0][1] - point_covariance) / np.linalg.norm(point_covariance)),
                'point_public_variance_parts': variance_parts(point_covariance), 'mixture_public_variance_parts': a['public_variance_parts']}
            print('source amplitude uncertainty', layout, 'E[a],E[a^2]', a['amplitude_mean'], a['amplitude_second_moment'], 'numerical', row['numerical_passed'], flush=True)
    except (AssertionError, TimeoutError) as error:
        save(out / 'results.json', {**base, 'rows': rows, 'diagnostic_interpreted': False, 'stopped_reason': str(error), 'cpu_seconds': time.process_time() - started}); raise
    save(out / 'results.json', {**base, 'rows': rows, 'diagnostic_interpreted': all(r['numerical_passed'] for r in rows),
        'cost': {**spec['cost'], 'actual_source_conditionings': 386}, 'cpu_seconds': time.process_time() - started,
        'interpretation': 'Proper conditional empirical-Bayes amplitude posterior on full unbounded support;all other source parameters/templates fixed. No raw forecasts.'})


if __name__ == '__main__':
    run()

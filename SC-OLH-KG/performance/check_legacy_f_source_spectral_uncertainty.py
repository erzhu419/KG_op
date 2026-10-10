"""Source-only bounded task-spectrum uncertainty, conditional on frozen fits."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_information_ceiling import variance_parts
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.source_joint_spectral_geometry import JointSpectralGeometryLikelihood

PLAN = 'performance/manifests/legacy_joint_population_f_source_spectral_uncertainty_plan_v1.json'


def source_model(spec, layout, query):
    law = next(r for r in read(spec['source_laws'])['rows'] if (r['layout'], r['query_task']) == (layout, query))
    coverage, _ = source_paths(read(spec['source_context_execution']))
    inputs = read(Path(coverage['output']) / f'layout{layout}' / 'task1' / 'inputs.json')
    tasks = [t for t in inputs['source_tasks'] if t['name'] in law['source_task_names']]
    assert len(tasks) == 4 and [t['name'] for t in tasks] == law['source_task_names']
    assert f'related_inventory_{query:02d}' not in law['source_task_names'] and 'related_inventory_01' not in law['source_task_names']
    template = next(r for r in read(spec['source_templates'])['rows']
        if (r['layout'], r['query_task'], r['arm']) == (layout, query, 'joint_native_support'))
    assert template['columns'] == law['columns']
    record = next(r for r in read(spec['source_registry'])['rows']
        if r['layout'] == layout and r['inner_query_task'] == f'related_inventory_{query:02d}')
    anchor = read(record['source_fit'])['fits']['gp']['anchor']
    center = anchor['normalization']['mean'] + anchor['normalization']['scale'] * anchor['constant']
    public = np.asarray(inputs['public_mean_reference']); basis = OrthogonalLowOrderBasis(public)
    model = JointSpectralGeometryLikelihood(basis, tasks, template['common_prior_coefficient_covariance'],
        template['task_coefficient_covariance'], law['columns'], center, law['white_discrepancy'], .01)
    assert model.label_count == 48
    np.testing.assert_allclose(model.tails, [law['common_tail_per_coefficient'], law['task_tail_per_coefficient']], atol=1e-14, rtol=1e-10)
    return model, law, public


def spectral_rule(model, parameters, order):
    nodes, measures, intervals = [], [], []
    z, w = np.polynomial.legendre.leggauss(order)
    for index, (lower, upper) in enumerate(((0., 2.), (2., 3.), (3., 4.))):
        nodes.extend(.5 * (upper - lower) * z + .5 * (upper + lower))
        measures.extend(w * (upper - lower) / 8.)  # dp/4: normalized uniform prior.
        intervals.extend([index] * order)
    nodes, measures = np.asarray(nodes), np.asarray(measures)
    states, values = [], []
    for p in nodes:
        theta = np.asarray(parameters).copy(); theta[3] = p
        value, _, state = model.evaluate(theta); values.append(value); states.append(state)
    log_mass = np.log(measures) - model.label_count * np.asarray(values)
    log_evidence = float(logsumexp(log_mass)); weights = np.exp(log_mass - log_evidence)
    np.testing.assert_allclose(weights.sum(), 1., atol=1e-12, rtol=0.)
    return {'order_per_interval': order, 'nodes': nodes, 'quadrature_prior_mass': measures,
        'interval_index': np.asarray(intervals), 'source_nll_per_label': np.asarray(values),
        'log_evidence': log_evidence, 'posterior_weights': weights, 'states': states}


def prior_summary(rule, model, public):
    weights, states, nodes = rule['posterior_weights'], rule['states'], rule['nodes']
    beta = np.asarray([s['source_posterior_coefficient_mean'] for s in states]); mean = weights @ beta
    covariances = np.asarray([s['source_posterior_common_covariance'] + s['task_coefficient_covariance'] for s in states])
    within = np.einsum('k,kij->ij', weights, covariances)
    centered = beta - mean; between = (centered.T * weights) @ centered; covariance = within + between
    # A second total-variance expression detects omitted between-component uncertainty.
    second = np.einsum('k,kij->ij', weights, covariances + beta[:, :, None] * beta[:, None, :]) - np.outer(mean, mean)
    np.testing.assert_allclose(covariance, second, atol=1e-11, rtol=1e-9)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-10
    design = np.column_stack([np.ones(len(public)), model.basis.transform(public)])
    f_mean = design @ mean; f_covariance = design @ covariance @ design.T + model.white * np.eye(len(public))
    p_mean = float(weights @ nodes); p_variance = float(weights @ (nodes - p_mean) ** 2)
    record = {k: rule[k].tolist() for k in ('nodes', 'quadrature_prior_mass', 'interval_index', 'source_nll_per_label', 'posterior_weights')}
    record.update(order_per_interval=rule['order_per_interval'], log_evidence=rule['log_evidence'],
        p_mean=p_mean, p_variance=p_variance, p_sd=float(np.sqrt(p_variance)),
        probability_p_le_2=float(weights[rule['interval_index'] == 0].sum()),
        probability_p_ge_3=float(weights[rule['interval_index'] == 2].sum()),
        coefficient_mean=mean.tolist(), within_coefficient_covariance=within.tolist(),
        between_coefficient_covariance=between.tolist(), coefficient_covariance=covariance.tolist(),
        public_variance_parts=variance_parts(f_covariance),
        between_component_public_variance_parts=variance_parts(design @ between @ design.T),
        coefficient_covariance_min_eigenvalue=float(np.linalg.eigvalsh(covariance).min()))
    return record, f_mean, f_covariance


def run():
    started = time.process_time(); spec = read(PLAN); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    rows = []; base = {'execution': spec, 'query_outcomes_indexed': False, 'no_model_adopted': True,
        'point_candidate_remains_stopped': True, 'source6_7_evaluated': False, 'independent_confirmation': False}
    try:
        for layout in spec['layouts']:
            model, law, public = source_model(spec, layout, spec['query_task'])
            nll, _, state = model.evaluate(np.asarray(law['parameters'])); assert law['parameters'][3] == 4.
            np.testing.assert_allclose(nll, law['source_nll_per_label'], atol=1e-10, rtol=1e-10)
            for k, value in state.items(): np.testing.assert_allclose(value, law[k], atol=1e-10, rtol=1e-10)
            row = {'layout': layout, 'query_task': spec['query_task'], 'source_task_names': law['source_task_names'],
                'point_parameters': law['parameters'], 'source_point_law_reproduced': True, 'rules': []}; rows.append(row)
            moments = []
            for order in (16, 32):
                if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']: raise TimeoutError('spectral uncertainty CPU cap')
                record, mean, covariance = prior_summary(spectral_rule(model, law['parameters'], order), model, public)
                row['rules'].append(record); moments.append((mean, covariance))
            a, r = row['rules']; tolerances = spec['tolerances']
            errors = {'log_evidence_absolute': abs(a['log_evidence'] - r['log_evidence']),
                'p_mean_absolute': abs(a['p_mean'] - r['p_mean']), 'p_variance_absolute': abs(a['p_variance'] - r['p_variance']),
                'f_mean_max_absolute': float(np.max(np.abs(moments[0][0] - moments[1][0]))),
                'f_covariance_relative_frobenius': float(np.linalg.norm(moments[0][1] - moments[1][1]) / np.linalg.norm(moments[1][1]))}
            row['numerical_errors'] = errors; row['numerical_passed'] = all(errors[k] <= tolerances[k] for k in tolerances)
            d = np.column_stack([np.ones(len(public)), model.basis.transform(public)])
            point_mean = d @ state['source_posterior_coefficient_mean']
            point_covariance = d @ (state['source_posterior_common_covariance'] + state['task_coefficient_covariance']) @ d.T + model.white * np.eye(len(public))
            row['changes_vs_point_law'] = {'public_mean_max_absolute': float(np.max(np.abs(moments[0][0] - point_mean))),
                'public_mean_rms': float(np.sqrt(np.mean((moments[0][0] - point_mean) ** 2))),
                'public_covariance_relative_frobenius': float(np.linalg.norm(moments[0][1] - point_covariance) / np.linalg.norm(point_covariance)),
                'point_public_variance_parts': variance_parts(point_covariance),
                'mixture_public_variance_parts': a['public_variance_parts']}
            print('source spectral uncertainty', layout, 'p mean/sd', a['p_mean'], a['p_sd'], 'numerical', row['numerical_passed'], flush=True)
    except (AssertionError, TimeoutError) as error:
        save(out / 'results.json', {**base, 'rows': rows, 'diagnostic_interpreted': False, 'stopped_reason': str(error),
            'cpu_seconds': time.process_time() - started}); raise
    save(out / 'results.json', {**base, 'rows': rows, 'diagnostic_interpreted': all(r['numerical_passed'] for r in rows),
        'cost': {**spec['cost'], 'actual_source_conditionings': 290}, 'cpu_seconds': time.process_time() - started,
        'interpretation': 'Conditional empirical-Bayes spectrum posterior;all other source parameters/templates/support fixed. No paid updates or forecasts.'})


if __name__ == '__main__':
    run()

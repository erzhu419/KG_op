"""One cached full-tail variance-mixture check without sequential projection."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_exact_raw_filter import prepare_cases
from performance.check_legacy_f_reference_training_budget import compact, prediction, errors
from performance.check_legacy_f_task_level_reference_development import mixture_field
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_level_shape_mean_error import decompose
from performance.exact_gaussian_raw_posterior import quadrature, moments
from performance.prepare_legacy_f_task_level_reference_quadrature import rule_on_intervals
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis

EXECUTION = 'performance/manifests/legacy_joint_population_f_exact_mixture_moments_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION)
    gaussian = read(spec['gaussian_exact_results']); assert gaussian['diagnostic_interpreted']
    original = read(gaussian['execution']['original_results'])['execution']
    cases, _, coverage = prepare_cases(original)
    case = next(c for c in cases if c['identity']['layout'] == spec['layout'] and
        c['identity']['design'] == spec['design'] and c['identity']['arm'] == 'native_f')
    assert case['identity']['paid_prefix'] == spec['paid_prefix']
    sites = case['identity']['scoring_sites']; selection = spec['paid_prefix'] + sites
    inputs = read(Path(coverage['output']) / f"layout{spec['layout']}" / 'task1' / 'inputs.json')
    package = read(spec['source_rules']); assert package['all14_rules_frozen_without_query_data']
    rule = next(r for r in package['rows'] if r['layout'] == spec['layout'] and r['deleted_task'] == spec['deleted_task'])
    source = rule['source_reference_law']
    assert source['source_task_names'] == [t['name'] for t in inputs['source_tasks']]
    assert 'related_inventory_01' not in source['source_task_names']
    control = read(spec['control_results'])
    projected = next(r for r in control['rows'] if r['layout'] == spec['layout'] and
        r['design'] == spec['design'] and r['arm'] == 'source_variance_mixture')
    assert projected['paid_prefix'] == spec['paid_prefix'] and projected['scoring_sites'] == sites
    eta = case['prior']['eta_mean'], case['prior']['eta_covariance']
    public = np.asarray(inputs['public_mean_reference']); basis = OrthogonalLowOrderBasis(public)
    design = np.column_stack([np.ones(len(selection)), basis.transform(public[selection])])
    diagonal = np.asarray(source['native_law']['residual_coefficient_diagonal']).copy()
    diagonal[0] = source['integrated_task_intercept_variance']
    reference = {'mean': (design @ np.asarray(source['integrated_common_coefficient_mean'])).tolist(),
        'covariance': (design @ np.asarray(source['integrated_common_coefficient_covariance']) @ design.T
            + (design * diagonal) @ design.T + source['native_law']['discrepancy_variance'] * np.eye(16)).tolist(),
        'paid_predictive_log_score': 0.}
    fields, prior_checks = {}, []
    for size in spec['source_nodes_per_interval']:
        nodes, weights = rule_on_intervals(source, rule['adaptive_u_intervals'], size)
        field = mixture_field(basis, public[selection], source, nodes, weights, eta)
        actual = prediction(compact(field, list(range(16)), 0.)); difference = errors(actual, reference)
        assert all(difference[k] <= v for k, v in spec['prediction_tolerances'].items())
        fields[size] = field
        prior_checks.append({'source_nodes_per_interval': size, 'component_count': len(nodes),
            'source_variance_min': float(nodes.min()), 'source_variance_max': float(nodes.max()), 'errors': difference})
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    base = {'execution': spec, 'cost': spec['cost'], 'no_model_adopted': True,
        'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False,
        'no_new_unscored_outcomes_read': True, 'training_diagnostic_only': True}
    save(out / 'frozen_inputs.json', {'execution': spec, 'source_task_names': source['source_task_names'],
        'paid_prefix': spec['paid_prefix'], 'scoring_sites': sites, 'prior_checks': prior_checks,
        'both_source_rules_frozen_before_paid_values': True})
    raw = read(coverage['source_data']); paid = [raw['tasks'][1]['Y_replicates'][j][0][0] for j in spec['paid_prefix']]; del raw
    rows = []
    for size, field in fields.items():
        for order in spec['paid_eta_orders']:
            component_predictions, log_evidences = [], []
            for j in range(len(field.log_weights)):
                if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']:
                    save(out / 'results.json', {**base, 'rows': rows, 'prior_checks': prior_checks,
                        'partial_cell': {'source_nodes_per_interval': size, 'paid_eta_order': order, 'completed_components': j},
                        'stopped_at_cpu_budget': True, 'diagnostic_interpreted': False, 'cpu_seconds': time.process_time() - started}); return
                indices = np.arange(0, 32, 2)
                prior = {'f_mean': field.means[j, indices], 'f_covariance': field.covariances[j][np.ix_(indices, indices)],
                    'eta_mean': eta[0], 'eta_covariance': eta[1]}
                grid = quadrature(prior, paid, order); m = moments(prior, grid)
                component_predictions.append((m['f_mean'][4:], m['f_covariance'][4:, 4:])); log_evidences.append(grid['evidence'])
            evidence = float(logsumexp(field.log_weights + log_evidences))
            weights = np.exp(field.log_weights + log_evidences - evidence)
            means = np.asarray([p[0] for p in component_predictions]); covariance = np.asarray([p[1] for p in component_predictions])
            mean = weights @ means; displacement = means - mean
            covariance = np.einsum('k,kij->ij', weights, covariance) + (displacement.T * weights) @ displacement
            rows.append({'source_nodes_per_interval': size, 'paid_eta_order': order,
                'component_count': len(weights), 'component_paid_log_evidences': log_evidences,
                'posterior_source_weights': weights.tolist(), 'prediction': {'mean': mean.tolist(),
                    'covariance': covariance.tolist(), 'paid_predictive_log_score': evidence}})
            save(out / 'partial_results.json', {**base, 'rows': rows, 'prior_checks': prior_checks})
            print('exact mixture cell', size, order, len(weights), 'CPU', time.process_time() - started, flush=True)
    save(out / 'predictions.json', {**base, 'rows': rows, 'all_four_moments_frozen_before_score_labels': True})
    def cell(size, order): return next(r['prediction'] for r in rows if r['source_nodes_per_interval'] == size and r['paid_eta_order'] == order)
    a, b = spec['source_nodes_per_interval']; c, d = spec['paid_eta_orders']; checks = []
    for dimension, pairs in [('source', [((a, c), (b, c)), ((a, d), (b, d))]),
                             ('paid_eta', [((a, c), (a, d)), ((b, c), (b, d))])]:
        for left, right in pairs:
            difference = errors(cell(*left), cell(*right))
            checks.append({'dimension': dimension, 'left': left, 'right': right, 'errors': difference,
                'passed': all(difference[k] <= v for k, v in spec['prediction_tolerances'].items())})
    verified = all(r['passed'] for r in checks)
    result = {**base, 'rows': rows, 'prior_checks': prior_checks, 'quadrature_checks': checks,
        'diagnostic_interpreted': verified, 'cpu_seconds': time.process_time() - started,
        'future_raw_density_evaluated': False, 'single_gaussian_checks_reused': True}
    if verified:
        labels = np.asarray(read(Path(coverage['output']) / f"layout{spec['layout']}" / 'inputs.json')['source_tasks'][1]['response'])[:, 0]
        actual = cell(b, d); before = decompose(projected['prediction']['mean'], labels); after = decompose(actual['mean'], labels)
        np.testing.assert_allclose(before['mse'], projected['scores']['f_mse'], atol=1e-12, rtol=1e-10)
        result.update(exact_mean_error=after, projected_mean_error=before,
            exact_minus_projection={k: after[k] - before[k] for k in ('mse', 'level_mse', 'centered_shape_mse')},
            mse_relative_change=after['mse'] / before['mse'] - 1,
            prediction_difference_from_projection=errors(actual, projected['prediction']))
    save(out / 'results.json', result)
    print('exact mixture complete', verified, 'CPU', result['cpu_seconds'], flush=True)


if __name__ == '__main__':
    run()

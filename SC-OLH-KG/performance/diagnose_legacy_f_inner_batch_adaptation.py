"""Separate prior prediction from few-batch adaptation on the same inner tasks."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_inner_shape_allocation import shape_system
from performance.diagnose_legacy_f_inner_shape_geometry import condition
from performance.diagnose_legacy_f_inner_task_level_shape import prepare
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save
from performance.validate_legacy_paired_taskout import gaussian_score

EXECUTION = 'performance/manifests/legacy_joint_population_f_inner_batch_adaptation_execution_v1.json'


def score(mean, covariance, observed):
    error = mean-observed; level = float(error.mean())
    return {'mse': float(np.mean(error**2)), 'level_mse': level**2,
        'centered_shape_mse': float(np.mean((error-level)**2)),
        'batch_log_density_per_label': gaussian_score(mean, covariance, observed)/len(observed)}


def run():
    started = time.process_time(); spec = read(EXECUTION); fitted = read(spec['fitted_results']); pilot = read(spec['failed_pilot_results'])
    assert fitted['all_fits_complete'] and pilot['scientific_gate_interpreted'] and not pilot['retained_mixture_gate_both_designs']
    probe = read(fitted['execution']['probe_results']); joint = read(probe['execution']['joint_results'])
    inner = read(joint['execution']['inner_results']); coverage, _ = source_paths(read(inner['execution']['source_context_execution']))
    observed_sites = np.arange(4); future_sites = np.arange(4, 12); rows = []
    for record in inner['rows']:
        layout, query = record['layout'], record['inner_query_task']; task_index = int(query.rsplit('_', 1)[1])
        indices = coverage['layouts'][layout][task_index]
        assert indices[:4] == spec['source_anchor_policy_indices'] and len(indices) == 12
        inputs = read(Path(coverage['output'])/f'layout{layout}'/f"task{spec['outer_deleted_task']}"/'inputs.json')
        fits = read(record['source_fit']); assert query not in fits['inner_training_task_names']
        system = shape_system(prepare(inputs, fits['fits'], query))
        q = next(r['q'] for r in fitted['rows'] if r['layout'] == layout); forecasts = {}
        for arm, coefficient in [('native', 0.), ('source_predictive_shape', q)]:
            result = condition(system, coefficient); mean = np.asarray(result['predictive_mean'])
            covariance = np.asarray(result['predictive_covariance_including_known_label_noise'])
            factor = cho_factor(covariance[np.ix_(observed_sites, observed_sites)], lower=True)
            cross = covariance[np.ix_(future_sites, observed_sites)]
            posterior_mean = mean[future_sites]+cross @ cho_solve(factor, system['query_labels'][observed_sites]-mean[observed_sites])
            posterior_covariance = covariance[np.ix_(future_sites, future_sites)]-cross @ cho_solve(factor, cross.T)
            forecasts[arm] = {'prior': score(mean[future_sites], covariance[np.ix_(future_sites, future_sites)], system['query_labels'][future_sites]),
                'after_four_source_batches': score(posterior_mean, posterior_covariance, system['query_labels'][future_sites])}
            # The same conditional density also follows the joint-minus-paid identity.
            conditional = (gaussian_score(mean, covariance, system['query_labels'])
                -gaussian_score(mean[observed_sites], covariance[np.ix_(observed_sites, observed_sites)], system['query_labels'][observed_sites]))/len(future_sites)
            np.testing.assert_allclose(conditional, forecasts[arm]['after_four_source_batches']['batch_log_density_per_label'], atol=1e-8, rtol=1e-8)
        contrast = {stage: {key: forecasts['source_predictive_shape'][stage][key]-forecasts['native'][stage][key]
                            for key in forecasts['native'][stage]} for stage in forecasts['native']}
        rows.append({'layout': layout, 'inner_query_task': query, 'source_q': q, 'scores': forecasts,
                     'candidate_minus_native': contrast, 'conditional_density_identity_verified': True})
    summary = {layout: {stage: {key: float(np.mean([r['candidate_minus_native'][stage][key] for r in rows if r['layout'] == layout]))
                              for key in rows[0]['candidate_minus_native'][stage]}
                        for stage in rows[0]['candidate_minus_native']} for layout in fitted['execution']['layouts']}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    result = {'execution': spec, 'rows': rows, 'summary': summary, 'cpu_seconds': time.process_time()-started, 'cost': spec['cost'],
        'no_model_adopted': True, 'outer_query_paid_or_score_outcomes_read': False, 'source6_7_evaluated': False,
        'target_promoted': False, 'independent_confirmation': False,
        'diagnostic_contract': 'Point source4 priors,known Gaussian source-label errors,four three-replicate batch observations (12 physical calls),eight unobserved source batches. Isolates Gaussian adaptation on the same tasks;not the four-single-episode unknown-eta raw mixture protocol.'}
    save(out/'results.json', result); print({'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()

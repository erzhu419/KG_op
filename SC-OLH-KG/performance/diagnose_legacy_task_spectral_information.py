"""Continuous conditional source evidence for the existing task exponent."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad_vec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.source_function_prior import head_batches
from representation.source_joint_population import JointCovariancePopulationLikelihood

EXECUTION = 'performance/manifests/legacy_joint_population_task_spectral_information_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION); native = read(spec['native_execution'])
    profile = read(spec['profile_execution']); coverage = read(spec['coverage_execution']); rows = []
    for layout in spec['layouts']:
        folders = [(None, Path(coverage['output'])/f'layout{layout}', Path(native['output'])/f'layout{layout}')]
        folders += [(i, Path(coverage['output'])/f'layout{layout}'/f'task{i}', Path(profile['output'])/f'layout{layout}'/f'task{i}')
            for i in spec['training_tasks']]
        for deleted, inputs_folder, fit_folder in folders:
            inputs = read(inputs_folder/'inputs.json'); means = read(fit_folder/'selected_source_fit.json')
            law = means['heads'][1]['joint_structural']; basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
            tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
            prior = restore_regularizer(basis, means['heads'][1]['regularizer'], tasks, 1)
            likelihood = JointCovariancePopulationLikelihood(prior, head_batches(tasks, 1), [t['noise'][:, 1] for t in tasks])
            parameters = np.asarray(law['optimizer_parameters']).copy(); native_value = likelihood.evaluate(parameters)[0]
            np.testing.assert_allclose(native_value, law['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
            gamma4 = parameters.copy(); gamma4[-2] = 4.
            reference = likelihood.evaluate(gamma4)[0]
            evaluations = []
            def moments(gamma):
                point = parameters.copy(); point[-2] = gamma
                value = likelihood.evaluate(point)[0]; log_weight = -likelihood.label_count*(value-reference)
                evaluations.append({'gamma': float(gamma), 'conditional_log_likelihood_minus_gamma4': float(log_weight)})
                return np.exp(log_weight)*np.array([1., gamma, gamma**2])
            lo, low_error = quad_vec(moments, 0., 2., epsabs=1e-8, epsrel=1e-8)
            middle, mid_error = quad_vec(moments, 2., 3., epsabs=1e-8, epsrel=1e-8)
            hi, high_error = quad_vec(moments, 3., 4., epsabs=1e-8, epsrel=1e-8)
            total = lo+middle+hi; mean = float(total[1]/total[0]); sd = float(np.sqrt(total[2]/total[0]-mean**2))
            moments(0.)
            rows.append({'layout': layout, 'deleted_task': deleted, 'source_label_count': likelihood.label_count,
                'native_fitted_gamma': float(parameters[-2]),
                'conditional_likelihood_mean_gamma': mean, 'conditional_likelihood_sd_gamma': sd,
                'mass_gamma_le2': float(lo[0]/total[0]), 'mass_gamma_ge3': float(hi[0]/total[0]),
                'log_likelihood_gamma0_minus_gamma4': evaluations[-1]['conditional_log_likelihood_minus_gamma4'],
                'normalizer_against_uniform_reference': float(total[0]/4),
                'quadrature_relative_error_bound': float((low_error+mid_error+high_error)/total[0]),
                'evaluations': evaluations})
        print('continuous source task-spectrum evidence complete', layout, flush=True)
    result = {'execution': spec, 'rows': rows, 'native_source_objectives_reproduced': True,
        'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'target_promoted': False, 'independent_confirmation': False}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); save(out/'results.json', result)
    print({'full_source': [{k: v for k, v in r.items() if k != 'evaluations'} for r in rows if r['deleted_task'] is None],
        'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()

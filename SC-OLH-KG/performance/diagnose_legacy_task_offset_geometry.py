"""Source intercept boundary support and unchanged public-bank task geometry."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates
from representation.source_function_prior import head_batches
from representation.source_joint_population import JointCovariancePopulationLikelihood

EXECUTION = 'performance/manifests/legacy_joint_population_task_offset_geometry_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION)
    profile = read(spec['profile_execution']); coverage = read(spec['coverage_execution'])
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json'))
    bank = [tuple(p) for p in public['bank']]; rows = []
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            inputs = read(Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'/'inputs.json')
            means = read(Path(profile['output'])/f'layout{layout}'/f'task{deleted}'/'selected_source_fit.json')
            native = means['heads'][1]['joint_structural']; basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
            tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
            prior = restore_regularizer(basis, means['heads'][1]['regularizer'], tasks, 1)
            likelihood = JointCovariancePopulationLikelihood(prior, head_batches(tasks, 1), [t['noise'][:, 1] for t in tasks])
            parameters = np.asarray(native['optimizer_parameters']).copy(); index = len(basis.group_names)+1
            reference, gradient, state = likelihood.evaluate(parameters)
            np.testing.assert_allclose(reference, native['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
            fitted = parameters[index]; calls = 0
            def loss(amplitude):
                nonlocal calls
                point = parameters.copy(); point[index] = amplitude; calls += 1
                return likelihood.label_count*(likelihood.evaluate(point)[0]-reference)
            at_zero = float(loss(0.)); drop = spec['likelihood_drop']
            tolerance = spec['root_absolute_tolerance_on_signal_scaled_amplitude']
            lower = 0. if at_zero <= drop else brentq(lambda p: loss(p)-drop, 0., fitted, xtol=tolerance)
            upper_bracket = fitted+1.
            while loss(upper_bracket) < drop:
                upper_bracket = 2*upper_bracket+1.
            upper = brentq(lambda p: loss(p)-drop, fitted, upper_bracket, xtol=tolerance)
            provider = no_outcomes(problem_for(public['plan'], public['tasks'][deleted]))
            psi = (public_control_mean_coordinates(provider, bank)-inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
            phi = basis.transform(psi); design = np.column_stack([np.ones(len(phi)), phi])
            average = phi.mean(axis=0); centered = phi-average
            residual = np.asarray(native['residual_coefficient_diagonal'])
            function_offset = float((average**2)@residual[1:])
            intercept = float(residual[0]); mean_marginal = float(intercept+np.mean((phi**2)@residual[1:]))
            offset_shape = centered@(residual[1:]*average)
            common_offset = float(design.mean(axis=0)@np.asarray(native['population_mean_covariance'])@design.mean(axis=0))
            rows.append({'layout': layout, 'deleted_task': deleted,
                'native_task_intercept_variance': intercept,
                'source_task_intercept_likelihood_support_variances': [float(lower*likelihood.signal), float(upper*likelihood.signal)],
                'source_task_intercept_likelihood_support_standard_deviations': [float(np.sqrt(lower*likelihood.signal)), float(np.sqrt(upper*likelihood.signal))],
                'conditional_total_nll_increase_at_zero': at_zero,
                'conditional_source_derivative_at_native_amplitude': float(likelihood.label_count*gradient[index]),
                'task_bank_average_variance_from_intercept': intercept,
                'task_bank_average_variance_from_function': function_offset,
                'task_mean_marginal_variance': mean_marginal,
                'task_centered_shape_variance_mean': float(np.mean((centered**2)@residual[1:])),
                'task_bank_average_shape_covariance_rms_over_mean_marginal': float(np.sqrt(np.mean(offset_shape**2))/mean_marginal),
                'common_bank_average_variance': common_offset,
                'white_bank_average_variance': native['discrepancy_variance']/len(bank),
                'source_likelihood_evaluations': calls})
        print('native source objectives, intercept support and public task geometry complete', layout, flush=True)
    result = {'execution': spec, 'rows': rows, 'native_source_objectives_reproduced': True,
        'zero_in_conditional_source_support_count': sum(r['source_task_intercept_likelihood_support_variances'][0] == 0 for r in rows),
        'cost': {**spec['cost'], 'conditional_source_likelihood_evaluations': sum(r['source_likelihood_evaluations'] for r in rows)},
        'cpu_seconds': time.process_time()-started, 'source_holdouts_evaluated': False,
        'target_promoted': False, 'independent_confirmation': False}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); save(out/'results.json', result)
    print({'task1_rows': [r for r in rows if r['deleted_task'] == 1],
        'zero_in_conditional_source_support_count': result['zero_in_conditional_source_support_count'],
        'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()

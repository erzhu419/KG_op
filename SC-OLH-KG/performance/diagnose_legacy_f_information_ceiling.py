"""Conditional covariance bounds without responses, fits or query data."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis

EXECUTION = 'performance/manifests/legacy_joint_population_f_information_ceiling_execution_v1.json'


def inverse_positive(value):
    return cho_solve(cho_factor(value, lower=True), np.eye(len(value)))


def variance_parts(value):
    total = float(np.trace(value)/len(value)); level = float(np.mean(value))
    return {'bank_average_variance': total, 'bank_average_level_variance': level,
        'centered_shape_variance': total-level}


def run():
    started = time.process_time(); spec = read(EXECUTION)
    conditioning = read(spec['conditioning_execution']); coverage = read(conditioning['coverage_execution'])
    parent = read(coverage['source_execution']); native = read(conditioning['native_execution']); rows = []
    for layout in spec['layouts']:
        sources = [('old144', parent, coverage['parent_layouts'][layout], 'source_fit.json'),
            ('native216', native, layout, 'selected_source_fit.json')]
        for name, fit_spec, directory, fit_filename in sources:
            inputs_spec = parent if name == 'old144' else coverage
            inputs = read(Path(inputs_spec['output'])/f'layout{directory}'/'inputs.json')
            law = read(Path(fit_spec['output'])/f'layout{directory}'/fit_filename)['heads'][0]['joint_structural']
            public = np.asarray(inputs['public_mean_reference']); basis = OrthogonalLowOrderBasis(public)
            design = np.column_stack([np.ones(len(public)), basis.transform(public)])
            a, r = (np.asarray(law[key]) for key in ('common_prior_coefficient_diagonal', 'residual_coefficient_diagonal'))
            delta = law['discrepancy_variance']; task_count = len(spec['training_tasks'])
            assert len(inputs['source_tasks']) == task_count == law['source_task_count'] == 6
            assert (a > 0).all() and (r > 0).all()
            precision = np.diag(1/a)
            for task, sites in zip(inputs['source_tasks'], inputs['execution']['source_site_indices_by_task']):
                local = design[sites]
                v = (local*r)@local.T+np.diag(np.asarray(task['noise'])[:, 0]+delta)
                precision += local.T@cho_solve(cho_factor(v, lower=True), local)
            current = inverse_positive(precision)
            np.testing.assert_allclose(current, law['population_mean_covariance'], atol=1e-8, rtol=1e-8)
            exact6 = np.diag(1/(1/a+task_count/r)); exact12 = np.diag(1/(1/a+2*task_count/r))
            # All four reached f laws have delta=0. Full-rank noiseless bank
            # observations therefore reveal the same coefficient vectors.
            assert delta == 0 and np.linalg.matrix_rank(design) == len(a)
            all73_zero_noise = exact6.copy()
            assert np.linalg.eigvalsh(current-all73_zero_noise).min() >= -1e-10
            assert np.linalg.eigvalsh(exact6-exact12).min() >= -1e-10
            common = {key: design@h@design.T for key, h in
                (('current', current), ('all73_zero_noise', all73_zero_noise), ('exact6', exact6), ('exact12', exact12))}
            task_covariance = (design*r)@design.T
            new_task = {key: h+task_covariance+delta*np.eye(len(public)) for key, h in common.items()}
            paid = np.asarray(spec['paid_candidates']); after_paid = {}
            for key, k in new_task.items():
                cross = k[:, paid]; conditioned = k-cross@np.linalg.solve(k[np.ix_(paid, paid)], cross.T)
                conditioned = .5*(conditioned+conditioned.T)
                assert np.linalg.eigvalsh(k-conditioned).min() >= -1e-10
                assert np.linalg.eigvalsh(conditioned).min() >= -1e-10
                np.testing.assert_allclose(conditioned[paid], 0, atol=1e-10, rtol=0)
                after_paid[key] = variance_parts(conditioned)
            common_parts = {key: variance_parts(k) for key, k in common.items()}
            total_parts = {key: variance_parts(k) for key, k in new_task.items()}
            fractions = {key: {'current_common_fraction': common_parts['current'][key]/value,
                'max_more_same_six_source_reduction': (value-total_parts['exact6'][key])/value,
                'additional_six_perfect_source_task_reduction': (total_parts['exact6'][key]-total_parts['exact12'][key])/value,
                'max_fixed_two_exact_paid_reduction': (value-after_paid['current'][key])/value}
                for key, value in total_parts['current'].items()}
            rows.append({'layout': layout, 'law': name, 'source_task_count': task_count,
                'public_bank_size': len(public), 'dictionary_dimension': len(a), 'discrepancy_variance': delta,
                'common_uncertainty': common_parts, 'task_function_variance': variance_parts(task_covariance),
                'total_new_task_uncertainty': total_parts, 'after_exact_paid0_1': after_paid,
                'fractions_of_current_total_uncertainty': fractions,
                'current_H_reproduced': True, 'information_ordering_verified': True,
                'all73_noiseless_equals_exact_coefficients_because_delta_zero_and_full_rank': True})
            print('information ceiling', layout, name, fractions, flush=True)
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    result = {'execution': spec, 'rows': rows, 'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'response_outcomes_indexed': False, 'query_paid_outcomes_read': False, 'source6_7_evaluated': False,
        'no_model_adopted': True, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'results.json', result); print('cpu_seconds', result['cpu_seconds'], flush=True)


if __name__ == '__main__':
    run()

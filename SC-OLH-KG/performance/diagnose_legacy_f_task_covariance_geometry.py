"""Source-only task/common covariance geometry in the original basis."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_information_ceiling import variance_parts
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates
from representation.source_kernel_population import rbf_cross

EXECUTION = 'performance/manifests/legacy_joint_population_f_task_covariance_geometry_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION); transplant = read(spec['transplant_execution'])
    mixture = read(transplant['mixture_execution']); reference = read(mixture['reference_execution'])
    centering = read(read(reference['source_execution'])['centering_execution'])
    coverage = read(centering['coverage_execution']); profile = read(centering['profile_execution'])
    compressed = read(spec['common_compression_laws'])['rows']; priors = read(spec['frozen_gp_priors'])['rows']
    rules = read(mixture['source_rules'])['rows']
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(x) for x in public['bank']]
    rows = []
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            suffix = Path(f'layout{layout}')/f'task{deleted}'
            inputs = read(Path(coverage['output'])/suffix/'inputs.json')
            source = next(r['source_reference_law'] for r in rules if r['layout'] == layout and r['deleted_task'] == deleted)
            gp = read(Path(profile['output'])/suffix/'selected_source_fit.json')['heads'][0]['joint_gp']
            anchor = gp['anchor']; scale2 = anchor['normalization']['scale']**2
            features = np.asarray(inputs['public_mean_reference']); basis = OrthogonalLowOrderBasis(features)
            query = (public_control_mean_coordinates(no_outcomes(problem_for(public['plan'], public['tasks'][deleted])), bank)
                -inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
            np.testing.assert_allclose(query, features, atol=1e-12, rtol=1e-12)
            d = np.column_stack([np.ones(len(features)), basis.transform(features)]); n = len(d)
            operator = np.linalg.lstsq(d, np.eye(n), rcond=None)[0]
            saved = next(r for r in compressed if r['layout'] == layout and r['deleted_task'] == deleted and r['scope'] == 'full_dictionary')
            amplitude, intercept, discrepancy = gp['residual_parameters']; assert anchor['signal_variance'] > 0
            a = np.asarray(saved['prior_coefficient_covariance']).copy(); a[0, 0] -= scale2*anchor['constant_variance']
            task_coefficient = a*(amplitude/anchor['signal_variance']); task_coefficient[0, 0] += scale2*intercept
            task_coefficient = .5*(task_coefficient+task_coefficient.T)
            task_bank = scale2*(amplitude*rbf_cross(features, features, anchor['lengthscale'])+intercept)
            np.testing.assert_allclose(task_coefficient, operator@task_bank@operator.T, atol=1e-10, rtol=1e-7)
            total = np.asarray(next(r['gp_prior_covariance'] for r in priors if r['layout'] == layout and r['deleted_task'] == deleted))
            white = scale2*discrepancy*np.eye(n); common = total-task_bank-white
            for value in (task_bank, common, white): assert np.linalg.eigvalsh(.5*(value+value.T)).min() >= -1e-10
            np.testing.assert_allclose(operator@common@operator.T, saved['posthoc_projected_coefficient_covariance'], atol=1e-8, rtol=1e-6)
            native = source['native_law']; r = np.asarray(native['residual_coefficient_diagonal']).copy()
            r[0] = source['integrated_task_intercept_variance']
            native_common = d@np.asarray(source['integrated_common_coefficient_covariance'])@d.T
            native_task = (d*r)@d.T
            components = {name: variance_parts(value) for name, value in
                (('gp_total', total), ('gp_common', common), ('gp_task', task_bank), ('gp_white', white),
                 ('native_common', native_common), ('native_task', native_task))}
            for metric in components['gp_total']:
                np.testing.assert_allclose(sum(components[key][metric] for key in ('gp_common', 'gp_task', 'gp_white')),
                    components['gp_total'][metric], atol=1e-12, rtol=1e-8)
            active = np.asarray(native['source_group_variances']) > 0
            union = active | (np.asarray(native['common_group_variances']) > 0)
            scopes = {'full_dictionary': list(range(len(r))),
                'task_active_groups': [0]+[j+1 for j, g in enumerate(basis.group_index) if active[g]],
                'common_task_union': [0]+[j+1 for j, g in enumerate(basis.group_index) if union[g]]}
            outside = {}
            for name, columns in scopes.items():
                local = d[:, columns]; projector = local@np.linalg.lstsq(local, np.eye(n), rcond=None)[0]
                residual = np.eye(n)-projector
                outside[name] = {'column_count': len(columns), 'fraction_of_gp_task_centered_shape_variance':
                    float(np.trace(residual@task_bank@residual.T)/n/components['gp_task']['centered_shape_variance'])}
            shape = task_coefficient[1:, 1:]; cross = task_coefficient[0, 1:]
            association = float(cross@np.linalg.pinv(shape, rcond=1e-10, hermitian=True)@cross/task_coefficient[0, 0])
            assert -1e-8 <= association <= 1+1e-8
            rows.append({'layout': layout, 'deleted_task': deleted, 'components': components,
                'gp_shape_variance_fractions': {key: components[key]['centered_shape_variance']/components['gp_total']['centered_shape_variance']
                    for key in ('gp_common', 'gp_task', 'gp_white')}, 'gp_task_outside_native_scope': outside,
                'gp_extra_task_intercept_parameter': float(intercept), 'gp_task_bank_level_variance': float(task_coefficient[0, 0]),
                'gp_task_level_linear_association_with_shape_fraction': association,
                'gp_task_shape_offdiagonal_relative_frobenius': float(np.linalg.norm(shape-np.diag(np.diag(shape)))/np.linalg.norm(shape)),
                'coordinate_projection_and_covariance_decomposition_verified': True})
    summary = {layout: {'gp_shape_variance_fractions': {key: float(np.mean([r['gp_shape_variance_fractions'][key] for r in rows if r['layout'] == layout]))
        for key in rows[0]['gp_shape_variance_fractions']},
        'gp_task_outside_native_scope': {name: float(np.mean([r['gp_task_outside_native_scope'][name]['fraction_of_gp_task_centered_shape_variance']
            for r in rows if r['layout'] == layout])) for name in rows[0]['gp_task_outside_native_scope']},
        'gp_task_level_linear_association_with_shape_fraction': float(np.mean([r['gp_task_level_linear_association_with_shape_fraction'] for r in rows if r['layout'] == layout])),
        'gp_task_shape_offdiagonal_relative_frobenius': float(np.mean([r['gp_task_shape_offdiagonal_relative_frobenius'] for r in rows if r['layout'] == layout]))}
        for layout in spec['layouts']}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    result = {'execution': spec, 'rows': rows, 'summary': summary, 'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'no_model_adopted': True, 'response_outcomes_indexed': False, 'query_paid_outcomes_read': False,
        'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'results.json', result); print({'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()

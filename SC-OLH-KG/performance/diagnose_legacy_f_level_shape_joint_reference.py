"""Source-only geometry diagnostic for the joint physical variance measure."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.diagnose_legacy_f_task_shape_uncertainty import source_system
from performance.run_legacy_related_inventory_source import save
from representation.task_level_shape_reference import TaskLevelShapeReference

EXECUTION = 'performance/manifests/legacy_joint_population_f_level_shape_joint_reference_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION); shape_spec = read(spec['shape_execution'])
    coverage, profile = source_paths(read(shape_spec['source_context_execution']))
    shapes = read(spec['shape_results'])['rows']; levels = read(spec['intercept_results'])['rows']; rows, laws = [], []
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            suffix = Path(f'layout{layout}')/f'task{deleted}'
            inputs = read(Path(coverage['output'])/suffix/'inputs.json')
            head = read(Path(profile['output'])/suffix/'selected_source_fit.json')['heads'][0]
            system = source_system(inputs, head, deleted); joint = TaskLevelShapeReference(system)
            shape = next(r for r in shapes if r['layout'] == layout and r['deleted_task'] == deleted)
            level = next(r for r in levels if r['layout'] == layout and r['deleted_task'] == deleted)
            native = joint.evaluate(joint.native_level, joint.native_shape)
            np.testing.assert_allclose(native['total_source_nll']/len(system['error']),
                system['native']['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
            np.testing.assert_allclose(native['beta'], system['native']['coefficient_mean'], atol=1e-8, rtol=1e-8)
            np.testing.assert_allclose(native['h'], system['native']['population_mean_covariance'], atol=1e-8, rtol=1e-8)
            points = []
            for pv in spec['quantile_probabilities']:
                v = level['reference_posterior']['quantiles'][str(pv)]
                for pa in spec['quantile_probabilities']:
                    a = shape['reference_posterior']['quantiles'][str(pa)]
                    evaluated = joint.evaluate(v, a, moments=False)
                    cross = evaluated['total_source_nll']-joint.evaluate(v, joint.native_shape, moments=False)['total_source_nll']
                    cross -= joint.evaluate(joint.native_level, a, moments=False)['total_source_nll']-native['total_source_nll']
                    points.append({'conditional_level_quantile': pv, 'conditional_shape_quantile': pa,
                        'level_variance': v, 'shape_variance': a, 'total_source_nll': evaluated['total_source_nll'],
                        'fisher_correlation': evaluated['fisher_correlation'],
                        'joint_to_product_conditional_jeffreys_factor': evaluated['joint_to_product_conditional_jeffreys_factor'],
                        'likelihood_nonseparability_nll_contrast': float(cross)})
            verification = {'native_nll_beta_h_reproduced': True, 'joint_design_rank': joint.rank,
                'level_direction_rank': joint.level_count, 'shape_direction_rank': system['rank'],
                'full_tail_second_moment_rank_condition': True}
            if deleted == spec['dense_verification_deleted_task']:
                interior = spec['dense_verification_variance_over_signal']*joint.signal
                verification.update(joint.dense_verification(interior, interior))
            rows.append({'layout': layout, 'deleted_task': deleted, 'selected_group': shape['selected_group'],
                'native_fisher': native['fisher'].tolist(), 'native_fisher_correlation': native['fisher_correlation'],
                'native_joint_to_product_conditional_jeffreys_factor': native['joint_to_product_conditional_jeffreys_factor'],
                'maximum_absolute_likelihood_nonseparability_contrast': max(abs(p['likelihood_nonseparability_nll_contrast']) for p in points),
                'minimum_joint_to_product_conditional_jeffreys_factor': min(p['joint_to_product_conditional_jeffreys_factor'] for p in points),
                'source_diagnostic_points': points, 'verification': verification})
            laws.append({'layout': layout, 'deleted_task': deleted, 'source_task_names': [t['name'] for t in system['tasks']],
                'native_law': system['native'], 'selected_group_index': system['group'], 'source_signal': joint.signal,
                'native_level_variance': joint.native_level, 'native_shape_variance': joint.native_shape,
                'joint_variance_source_gram': joint.gram.tolist(), 'joint_variance_source_rhs': joint.rhs.tolist(),
                'level_column_count': joint.level_count, 'common_coefficient_update_columns': joint.update.tolist(),
                'coefficient_mean_at_zero_level_and_selected_shape_variances': joint.beta00.tolist(),
                'common_covariance_at_zero_level_and_selected_shape_variances': joint.h00.tolist(),
                'total_source_nll_at_zero_level_and_selected_shape_variances': joint.nll00,
                'residual_diagonal_without_selected_shape_excess_at_native_level': system['r0'].tolist(),
                'task_group_unit_coefficient_diagonal': system['unit'].tolist(),
                'reference_measure': spec['reference_measure'], 'not_a_fitted_or_integrated_replacement_prior': True})
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'source_joint_laws.json', {'execution': spec, 'rows': laws})
    result = {'execution': spec, 'rows': rows, 'all12_joint_source_laws_verified': len(rows) == 12,
        'cpu_seconds': time.process_time()-started, 'cost': spec['cost'], 'no_prior_or_model_adopted': True,
        'query_paid_outcomes_read': False, 'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'results.json', result)
    print({'task1': [{k:r[k] for k in ('layout', 'native_fisher_correlation',
        'maximum_absolute_likelihood_nonseparability_contrast', 'minimum_joint_to_product_conditional_jeffreys_factor', 'verification')}
        for r in rows if r['deleted_task'] == 1], 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()

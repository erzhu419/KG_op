"""Source coverage and ideal information bounds for the current correlated law."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_information_ceiling import inverse_positive, variance_parts
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis

PLAN = 'performance/manifests/legacy_joint_population_f_joint_source_information_ceiling_plan_v1.json'


def run():
    started = time.process_time(); spec = read(PLAN); assert read(spec['latent_noise_results'])['diagnostic_interpreted']
    inner, outer = read(spec['inner_source_laws']), read(spec['outer_source_laws'])
    assert inner['all_ten_source_laws_frozen_without_query_values'] and outer['both_source5_laws_frozen_before_query_values']
    coverage, _ = source_paths(read(spec['source_context_execution'])); rows = []
    for law in [r for r in inner['rows'] if r['query_task'] == spec['inner_query_task']] + outer['rows']:
        layout, names = law['layout'], law['source_task_names']; m = len(names); assert m in (4, 5)
        assert 'related_inventory_01' not in names
        inputs = read(Path(coverage['output']) / f'layout{layout}' / 'task1' / 'inputs.json')
        public = np.asarray(inputs['public_mean_reference']); basis = OrthogonalLowOrderBasis(public)
        design = np.column_stack([np.ones(len(public)), basis.transform(public)])
        assert np.linalg.matrix_rank(design) == design.shape[1]
        tasks = [t for t in inputs['source_tasks'] if t['name'] in names]; assert [t['name'] for t in tasks] == names
        a, r, h = (np.asarray(law[k]) for k in ('common_prior_coefficient_covariance', 'task_coefficient_covariance', 'source_posterior_common_covariance'))
        ai, ri = inverse_positive(a), inverse_positive(r)
        floor = inverse_positive(ai + m * ri); doubled = inverse_positive(ai + 2 * m * ri)
        # Independent Gaussian covariance conditional on exact source task vectors.
        checked = a - a @ np.linalg.solve(a + r / m, a); checked = .5 * (checked + checked.T)
        np.testing.assert_allclose(floor, checked, atol=1e-10, rtol=1e-8)
        eigen_current = float(np.linalg.eigvalsh(h - floor).min()); eigen_more = float(np.linalg.eigvalsh(floor - doubled).min())
        assert eigen_current >= -1e-10 and eigen_more >= -1e-10
        source_ids = [int(n.rsplit('_', 1)[1]) for n in names]
        union = sorted(set(j for q in source_ids for j in coverage['layouts'][layout][q])); geometry = []
        for task, q in zip(tasks, source_ids):
            sites = coverage['layouts'][layout][q]
            np.testing.assert_allclose(public[sites], task['psi'], atol=1e-12, rtol=1e-12)
            geometry.append({'source_task': q, 'current_policy_count': len(sites), 'union_policy_count': len(union),
                'current_full_design_rank': int(np.linalg.matrix_rank(design[sites])),
                'current_selected_design_rank': int(np.linalg.matrix_rank(design[np.ix_(sites, law['columns'])])),
                'union_full_design_rank': int(np.linalg.matrix_rank(design[union])),
                'union_selected_design_rank': int(np.linalg.matrix_rank(design[np.ix_(union, law['columns'])]))})
        common = {k: variance_parts(design @ v @ design.T) for k, v in (('current', h), ('perfect_same_sources', floor), ('perfect_double_sources', doubled))}
        task = design @ r @ design.T + law['white_discrepancy'] * np.eye(len(public))
        total = {k: variance_parts(design @ v @ design.T + task) for k, v in (('current', h), ('perfect_same_sources', floor), ('perfect_double_sources', doubled))}
        reductions = {key: {'maximum_same_source_uncertainty_reduction_fraction': (value - total['perfect_same_sources'][key]) / value,
            'additional_perfect_double_sources_reduction_fraction': (total['perfect_same_sources'][key] - total['perfect_double_sources'][key]) / value,
            'current_common_fraction': common['current'][key] / value} for key, value in total['current'].items()}
        row = {'layout': layout, 'source_task_count': m, 'law_query_task': law['query_task'], 'source_task_names': names,
            'strong_column_count': len(law['columns']), 'dictionary_column_count': design.shape[1], 'union_policies': union,
            'source_policy_geometry': geometry, 'current_common_variance_parts': common['current'], 'common_information_bounds': common,
            'full_predictive_variance_bounds': total, 'uncertainty_reduction_bounds': reductions,
            'information_order_min_eigenvalues': [eigen_current, eigen_more], 'independent_floor_covariance_error': float(np.max(np.abs(checked - floor))),
            'geometry_and_information_order_verified': True}; rows.append(row)
        print('joint source information ceiling', m, layout, reductions, 'union policies', len(union), flush=True)
        assert time.process_time() - started < spec['maximum_single_thread_cpu_seconds']
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'diagnostic_interpreted': True, 'cpu_seconds': time.process_time() - started,
        'cost': spec['cost'], 'source_or_query_responses_indexed': False, 'no_model_adopted': True, 'independent_confirmation': False,
        'interpretation': 'Ideal exact coefficient observations,strictly more informative than finite noisy source-policy samples. Conditional on fixed A/R,white,templates,support and center. No promised MSE gain,no task distribution fit,no observations from hypothetical doubled sources.'})


if __name__ == '__main__':
    run()

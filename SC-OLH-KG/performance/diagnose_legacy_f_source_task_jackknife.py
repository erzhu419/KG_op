"""Delete-one source-task stability of the complete cached spectral learner."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_information_ceiling import variance_parts
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis

PLAN = 'performance/manifests/legacy_joint_population_f_source_task_jackknife_plan_v1.json'


def run():
    started = time.process_time(); spec = read(PLAN); inner, outer = read(spec['inner_source_laws']), read(spec['outer_source_laws'])
    assert inner['all_ten_source_laws_frozen_without_query_values'] and outer['both_source5_laws_frozen_before_query_values']
    coverage, _ = source_paths(read(spec['source_context_execution'])); rows = []
    for layout in spec['layouts']:
        full = next(r for r in outer['rows'] if r['layout'] == layout); assert full['query_task'] == 1
        names = set(full['source_task_names']); assert len(names) == 5 and 'related_inventory_01' not in names
        folds = [r for r in inner['rows'] if r['layout'] == layout]; assert len(folds) == 5
        for fold in folds:
            assert set(fold['source_task_names']) == names - {f"related_inventory_{fold['query_task']:02d}"}
        inputs = read(Path(coverage['output']) / f'layout{layout}' / 'task1' / 'inputs.json')
        public = np.asarray(inputs['public_mean_reference']); basis = OrthogonalLowOrderBasis(public)
        design = np.column_stack([np.ones(len(public)), basis.transform(public)])
        beta = np.asarray([r['source_posterior_coefficient_mean'] for r in folds]); full_beta = np.asarray(full['source_posterior_coefficient_mean'])
        assert beta.shape == (5, design.shape[1]) and full_beta.shape == (design.shape[1],)
        mean = beta.mean(axis=0); centered = beta - mean; scatter = centered.T @ centered
        jackknife = 4 / 5 * scatter; ensemble = scatter / 5
        assert np.linalg.eigvalsh(jackknife).min() >= -1e-10
        public_centered = (design @ beta.T).T - design @ mean
        np.testing.assert_allclose(design @ jackknife @ design.T, 4 / 5 * public_centered.T @ public_centered, atol=1e-12, rtol=1e-10)
        common = design @ np.asarray(full['source_posterior_common_covariance']) @ design.T
        task = design @ np.asarray(full['task_coefficient_covariance']) @ design.T
        total = common + task + full['white_discrepancy'] * np.eye(len(public)); deletions = []
        for fold, coefficient in zip(folds, beta):
            delta = design @ (coefficient - full_beta); level = float(delta.mean())
            covariance = design @ (np.asarray(fold['source_posterior_common_covariance']) + np.asarray(fold['task_coefficient_covariance'])) @ design.T + fold['white_discrepancy'] * np.eye(len(public))
            deletions.append({'omitted_source_task': fold['query_task'], 'source_task_names': fold['source_task_names'], 'parameters': fold['parameters'],
                'strong_column_count': len(fold['columns']), 'strong_columns_added_vs_full': sorted(set(fold['columns']) - set(full['columns'])),
                'strong_columns_removed_vs_full': sorted(set(full['columns']) - set(fold['columns'])),
                'public_mean_level_change': level, 'public_mean_centered_shape_rms_change': float(np.sqrt(np.mean((delta - level) ** 2))),
                'public_mean_max_absolute_change': float(np.max(np.abs(delta))),
                'full_predictive_covariance_relative_frobenius_change': float(np.linalg.norm(covariance - total) / np.linalg.norm(total)),
                'public_variance_parts': variance_parts(covariance)})
        jk_parts = variance_parts(design @ jackknife @ design.T); common_parts = variance_parts(common); total_parts = variance_parts(total)
        row = {'layout': layout, 'source_task_names': full['source_task_names'], 'full_parameters': full['parameters'], 'deletions': deletions,
            'delete_one_coefficient_mean': mean.tolist(), 'task_jackknife_coefficient_covariance': jackknife.tolist(),
            'equal_ensemble_coefficient_covariance': ensemble.tolist(), 'task_jackknife_public_variance_parts': jk_parts,
            'equal_ensemble_public_variance_parts': variance_parts(design @ ensemble @ design.T),
            'full_common_posterior_variance_parts': common_parts, 'full_predictive_variance_parts': total_parts,
            'jackknife_to_common_posterior_shape_ratio': jk_parts['centered_shape_variance'] / common_parts['centered_shape_variance'],
            'jackknife_to_full_predictive_shape_ratio': jk_parts['centered_shape_variance'] / total_parts['centered_shape_variance'],
            'jackknife_to_full_predictive_level_ratio': jk_parts['bank_average_level_variance'] / total_parts['bank_average_level_variance'],
            'rosters_coefficient_coordinates_and_covariance_identity_verified': True}; rows.append(row)
        print('source task jackknife', layout, 'ratios shape/common,shape/total,level/total', row['jackknife_to_common_posterior_shape_ratio'],
            row['jackknife_to_full_predictive_shape_ratio'], row['jackknife_to_full_predictive_level_ratio'], flush=True)
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'diagnostic_interpreted': True, 'cpu_seconds': time.process_time() - started,
        'cost': spec['cost'], 'query_outcomes_indexed': False, 'no_model_adopted': True, 'independent_confirmation': False,
        'interpretation': 'Delete-one task jackknife of complete empirical source learning,not a Bayesian posterior or deployed covariance correction. Templates,centers,support and parameter changes are combined;their separate effects are not identified.'})


if __name__ == '__main__':
    run()

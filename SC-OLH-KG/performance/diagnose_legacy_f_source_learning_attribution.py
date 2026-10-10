"""Separate missing source labels from changes in the empirical source learner."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_source_spectral_uncertainty import source_model
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_information_ceiling import variance_parts
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save
from representation.source_joint_spectral_geometry import JointSpectralGeometryLikelihood

PLAN = 'performance/manifests/legacy_joint_population_f_source_learning_attribution_plan_v1.json'


def change_parts(delta):
    level = float(np.mean(delta)); shape = delta - level
    return {'level_change': level, 'centered_shape_rms_change': float(np.sqrt(np.mean(shape ** 2))),
        'max_absolute_change': float(np.max(np.abs(delta)))}


def coefficient_check(model, state):
    a, r = state['common_prior_coefficient_covariance'], state['task_coefficient_covariance']
    precision = cho_solve(cho_factor(a, lower=True), np.eye(len(a))); rhs = np.zeros(len(a)); offset = 0
    for d in model.designs:
        n = len(d); factor = cho_factor(d @ r @ d.T + np.diag(model.noise[offset:offset + n]), lower=True)
        precision += d.T @ cho_solve(factor, d); rhs += d.T @ cho_solve(factor, model.error[offset:offset + n]); offset += n
    h = cho_solve(cho_factor(precision, lower=True), np.eye(len(a)))
    beta = np.r_[model.center, np.zeros(model.basis.feature_dim)] + h @ rhs
    np.testing.assert_allclose(h, state['source_posterior_common_covariance'], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(beta, state['source_posterior_coefficient_mean'], atol=1e-8, rtol=1e-8)
    return {'independent_coefficient_conditioning_verified': True, 'beta_max_absolute_error': float(np.max(np.abs(beta - state['source_posterior_coefficient_mean']))),
        'H_max_absolute_error': float(np.max(np.abs(h - state['source_posterior_common_covariance'])))}


def run():
    started = time.process_time(); spec = read(PLAN); before = read(spec['jackknife_results']); assert before['diagnostic_interpreted']
    source_spec = read(spec['source_probe_plan']); full_laws = read(spec['outer_source_laws'])['rows']
    coverage, _ = source_paths(read(spec['source_context_execution'])); rows = []; verification = None
    for layout in spec['layouts']:
        full = next(r for r in full_laws if r['layout'] == layout); inputs = read(Path(coverage['output']) / f'layout{layout}' / 'task1' / 'inputs.json')
        tasks = inputs['source_tasks']; assert len(tasks) == 5 and [t['name'] for t in tasks] == full['source_task_names']
        assert 'related_inventory_01' not in full['source_task_names']
        full_beta = np.asarray(full['source_posterior_coefficient_mean']); betas, deletions = [], []
        for query in spec['query_tasks']:
            old, law, public = source_model(source_spec, layout, query)
            model = JointSpectralGeometryLikelihood(old.basis, tasks, *old.templates, old.columns, old.center, old.white, .01)
            _, _, state = model.evaluate(np.asarray(law['parameters'])); assert model.label_count == 60
            for k in ('common_prior_coefficient_covariance', 'task_coefficient_covariance'):
                np.testing.assert_allclose(state[k], law[k], atol=1e-10, rtol=1e-10)
            if verification is None: verification = coefficient_check(model, state)
            assert np.linalg.eigvalsh(state['source_posterior_common_covariance'] + state['task_coefficient_covariance']).min() >= -1e-10
            design = np.column_stack([np.ones(len(public)), old.basis.transform(public)])
            beta = state['source_posterior_coefficient_mean']; betas.append(beta)
            added = design @ (beta - law['source_posterior_coefficient_mean']); learned = design @ (full_beta - beta)
            total = design @ (full_beta - law['source_posterior_coefficient_mean'])
            np.testing.assert_allclose(added + learned, total, atol=1e-12, rtol=1e-10)
            cross = float(2 * np.mean((added - added.mean()) * (learned - learned.mean())))
            parts = [change_parts(z) for z in (added, learned, total)]
            np.testing.assert_allclose(parts[0]['centered_shape_rms_change'] ** 2 + parts[1]['centered_shape_rms_change'] ** 2 + cross,
                parts[2]['centered_shape_rms_change'] ** 2, atol=1e-12, rtol=1e-10)
            deletions.append({'omitted_from_law_training_task': query, 'law_training_source_names': law['source_task_names'],
                'conditioning_source_names': full['source_task_names'], 'frozen_law_source5_beta': beta.tolist(),
                'extra_source_label_conditioning_change': parts[0], 'empirical_learning_change': parts[1], 'total_source4_to_source5_change': parts[2],
                'centered_shape_cross_term': cross, 'source4_A_R_retained': True})
        beta = np.asarray(betas); centered = beta - beta.mean(axis=0); covariance = centered.T @ centered / 5
        scatter = variance_parts(design @ covariance @ design.T)
        old = next(r for r in before['rows'] if r['layout'] == layout); old_scatter = old['equal_ensemble_public_variance_parts']
        row = {'layout': layout, 'deletions': deletions, 'equal_observation_coefficient_scatter': covariance.tolist(),
            'equal_observation_public_scatter': scatter, 'original_delete_one_public_scatter': old_scatter,
            'remaining_shape_scatter_ratio': scatter['centered_shape_variance'] / old_scatter['centered_shape_variance'],
            'remaining_level_scatter_ratio': scatter['bank_average_level_variance'] / old_scatter['bank_average_level_variance'],
            'same_observation_law_scatter_to_common_shape_ratio': scatter['centered_shape_variance'] / old['full_common_posterior_variance_parts']['centered_shape_variance'],
            'same_observation_law_scatter_to_full_predictive_shape_ratio': scatter['centered_shape_variance'] / old['full_predictive_variance_parts']['centered_shape_variance'],
            'decomposition_verified': True}; rows.append(row)
        print('equal-observation source law variation', layout, 'remaining shape/level scatter', row['remaining_shape_scatter_ratio'], row['remaining_level_scatter_ratio'], flush=True)
        assert time.process_time() - started < spec['maximum_single_thread_cpu_seconds']
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'verification': verification, 'diagnostic_interpreted': True,
        'cost': spec['cost'], 'cpu_seconds': time.process_time() - started, 'query_outcomes_indexed': False, 'no_model_adopted': True,
        'independent_confirmation': False, 'interpretation': 'Same five conditioning tasks isolate combined empirical center/template/support/parameter variation from missing labels. Scatter is a sensitivity diagnostic,not posterior uncertainty or a deployed ensemble.'})


if __name__ == '__main__':
    run()

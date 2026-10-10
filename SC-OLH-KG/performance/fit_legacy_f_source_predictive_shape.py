"""Learn one shape allocation parameter from honest inner source predictions."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_inner_shape_allocation import shape_system
from performance.diagnose_legacy_f_inner_shape_geometry import condition
from performance.diagnose_legacy_f_inner_task_level_shape import joint_template, prepare
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis

EXECUTION = 'performance/manifests/legacy_joint_population_f_source_predictive_shape_execution_v1.json'


def outer_law(inputs, head, q):
    native = head['joint_structural']; basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
    public = np.asarray(inputs['public_mean_reference'])
    def design(psi):
        return np.column_stack([np.ones(len(psi)), basis.transform(np.asarray(psi))])
    direction, geometry = joint_template(basis, design(public), public, {'structural': native, 'gp': head['joint_gp']})
    direction = np.diag(np.diag(direction)); direction[0, 0] = 0.
    residual = np.asarray(native['residual_coefficient_diagonal'])+q*np.diag(direction)
    a = np.diag(native['common_prior_coefficient_diagonal']); m0 = np.asarray(head['regularizer']['coefficient_mean'])
    tasks = inputs['source_tasks']; local = [design(t['psi']) for t in tasks]; d = np.vstack(local)
    error = np.concatenate([np.asarray(t['response'])[:, 0] for t in tasks])-d @ m0
    delta = native['discrepancy_variance']
    def posterior(task_diagonal):
        c = d @ a @ d.T+block_diag(*[(z*task_diagonal) @ z.T+np.diag(np.asarray(t['noise'])[:, 0]+delta)
                                    for z, t in zip(local, tasks)])
        factor = cho_factor(c, lower=True); alpha = cho_solve(factor, error)
        beta = m0+a @ d.T @ alpha; h = a-a @ d.T @ cho_solve(factor, d @ a); h = .5*(h+h.T)
        nll = .5*(len(error)*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+error @ alpha)/len(error)
        return beta, h, float(nll)
    baseline = posterior(np.asarray(native['residual_coefficient_diagonal']))
    for actual, key in zip(baseline, ('coefficient_mean', 'population_mean_covariance', 'source_negative_log_likelihood_per_label')):
        np.testing.assert_allclose(actual, native[key], atol=1e-8, rtol=1e-8)
    beta, h, nll = posterior(residual)
    assert np.all(residual >= 0) and residual[0] == native['residual_coefficient_diagonal'][0]
    return {'layout': inputs['execution']['output'].split('layout')[-1].split('/')[0],
        'source_task_names': [t['name'] for t in tasks], 'source_task_count': len(tasks), 'source_label_count': len(error),
        'fitted_shape_allocation_q': q, 'native_law': native, 'coefficient_mean': beta.tolist(),
        'population_mean_covariance': h.tolist(), 'residual_coefficient_diagonal': residual.tolist(),
        'common_prior_coefficient_diagonal': native['common_prior_coefficient_diagonal'],
        'empirical_center_coefficient_mean': m0.tolist(), 'discrepancy_variance': delta,
        'source_signal_scale': native['source_signal_scale'], 'source_negative_log_likelihood_per_label': nll,
        'native_source_negative_log_likelihood_per_label': baseline[2], 'geometry': geometry,
        'native_full_source_law_reproduced': True,
        'law_contract': 'New shape group allocation at fixed native spectrum/support union;independent native task intercept and tail. Common posterior coherently reconditioned;not an additive posterior covariance correction.'}


def run():
    started = time.process_time(); spec = read(EXECUTION); probe = read(spec['probe_results'])
    assert probe['source_predictive_shape_allocation_supported']
    joint = read(probe['execution']['joint_results']); inner = read(joint['execution']['inner_results'])
    coverage, profile = source_paths(read(inner['execution']['source_context_execution']))
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); records, laws = [], []
    for layout in spec['layouts']:
        inputs = read(Path(coverage['output'])/f'layout{layout}'/f"task{spec['outer_deleted_task']}"/'inputs.json')
        assert {t['name'] for t in inputs['source_tasks']} == {f'related_inventory_{i:02d}' for i in range(6) if i != spec['outer_deleted_task']}
        systems = []; members = [r for r in inner['rows'] if r['layout'] == layout]
        for member in members:
            fits = read(member['source_fit']); query = member['inner_query_task']
            assert query not in fits['inner_training_task_names']
            systems.append(shape_system(prepare(inputs, fits['fits'], query)))
        def objective(parameters):
            values = [condition(system, float(parameters[0])) for system in systems]
            return float(np.mean([v['predictive_nll_per_label'] for v in values])), np.asarray([
                np.mean([v['predictive_q_gradient_per_label'] for v in values])])
        baseline, baseline_gradient = objective([0.]); optimizer = spec['optimizer']
        result = minimize(objective, [0.], jac=True, bounds=[(0., 1.)], method=optimizer['method'],
            options={'maxiter': optimizer['max_iterations'], 'ftol': optimizer['ftol'], 'gtol': optimizer['gtol']})
        q = float(result.x[0]); loss, gradient = objective([q]); folds = []
        for member, system in zip(members, systems):
            old, new = condition(system, 0.), condition(system, q)
            folds.append({'inner_query_task': member['inner_query_task'],
                'predictive_log_density_gain_per_label': old['predictive_nll_per_label']-new['predictive_nll_per_label']})
        record = {'layout': layout, 'q': q, 'source_predictive_nll_per_label': loss,
            'native_source_predictive_nll_per_label': baseline, 'density_gain_per_label': baseline-loss,
            'initial_predictive_q_gradient': float(baseline_gradient[0]), 'final_predictive_q_gradient': float(gradient[0]),
            'optimizer_converged': bool(result.success), 'optimizer_iterations': int(result.nit),
            'optimizer_evaluations': int(result.nfev), 'optimizer_message': str(result.message), 'folds': folds}
        records.append(record)
        if not result.success:
            save(out/'results.json', {'execution': spec, 'rows': records, 'all_fits_complete': False,
                'laws_frozen_before_query_outcomes': False, 'cpu_seconds': time.process_time()-started}); return
        assert loss <= baseline+1e-10
        head = read(Path(profile['output'])/f'layout{layout}'/f"task{spec['outer_deleted_task']}"/'selected_source_fit.json')['heads'][0]
        law = outer_law(inputs, head, q); law.update(layout=layout, deleted_task=spec['outer_deleted_task'])
        laws.append(law); print('source-predictive shape fit', record, flush=True)
    save(out/'source_laws.json', {'execution': spec, 'both_laws_frozen_without_outer_query_outcomes': True, 'rows': laws})
    result = {'execution': spec, 'rows': records, 'all_fits_complete': True, 'laws_frozen_before_query_outcomes': True,
        'cost': {**spec['cost'], 'actual_source_predictive_q_fits': len(records)}, 'cpu_seconds': time.process_time()-started,
        'no_model_adopted': True, 'outer_query_paid_or_score_outcomes_read': False, 'source6_7_evaluated': False,
        'target_promoted': False, 'independent_confirmation': False,
        'source_predictive_selection_is_training_evidence': True}
    save(out/'results.json', result)


if __name__ == '__main__':
    run()

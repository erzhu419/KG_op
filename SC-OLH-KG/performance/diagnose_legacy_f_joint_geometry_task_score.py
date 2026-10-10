"""Compare source joint and conditional-block objectives at frozen parameters."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis

EXECUTION = 'performance/manifests/legacy_joint_population_f_joint_geometry_task_score_execution_v1.json'


def gaussian_loss_and_derivative(c,error,direction):
    factor = cho_factor(c,lower=True); alpha = cho_solve(factor,error)
    inverse = cho_solve(factor,np.eye(len(error)))
    loss = .5*(len(error)*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+error@alpha)
    derivative = .5*(np.sum(inverse*direction)-alpha@direction@alpha)
    return float(loss),float(derivative)


def run():
    started = time.process_time(); spec = read(EXECUTION); parent = read(spec['source_execution'])
    coverage,profile = source_paths(read(parent['template_execution'])); laws = read(spec['source_laws'])['rows']; rows = []
    for law in laws:
        layout,deleted = law['layout'],law['deleted_task']; suffix = Path(f'layout{layout}')/f'task{deleted}'
        inputs = read(Path(coverage['output'])/suffix/'inputs.json'); tasks = inputs['source_tasks']
        assert {t['name'] for t in tasks}=={f'related_inventory_{i:02d}' for i in range(6) if i!=deleted}
        basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
        designs = [np.column_stack([np.ones(len(t['psi'])),basis.transform(np.asarray(t['psi']))]) for t in tasks]
        d = np.vstack(designs); error = np.concatenate([np.asarray(t['response'])[:,0] for t in tasks])-d@np.asarray(law['empirical_center_coefficient_mean'])
        noise = block_diag(*[np.diag(np.asarray(t['noise'])[:,0]+law['discrepancy_variance']) for t in tasks])
        def covariance(a,r): return d@a@d.T+block_diag(*[local@r@local.T for local in designs])+noise
        a,r = np.asarray(law['common_prior_coefficient_covariance']),np.asarray(law['residual_coefficient_covariance'])
        assert law['task_template_strength']==0.
        base = np.asarray(law['task_covariance_at_zero_independent_intercept'])
        weak = np.diag(np.r_[0.,np.full(basis.feature_dim,law['independent_tail_per_coefficient'])])
        direction = block_diag(*[local@(np.asarray(law['task_template_at_fitted_spectrum'])-(base-weak))@local.T for local in designs])
        c = covariance(a,r); masks = []; offset = 0
        for local in designs:
            masks.append(np.r_[np.arange(offset),np.arange(offset+len(local),len(error))]); offset += len(local)
        def objectives(matrix):
            total,gradient = gaussian_loss_and_derivative(matrix,error,direction)
            losses,derivatives = zip(*[gaussian_loss_and_derivative(matrix[np.ix_(m,m)],error[m],direction[np.ix_(m,m)]) for m in masks])
            return {'joint_nll_per_label':total/len(error),'sum_conditional_block_nll_per_label':(len(tasks)*total-sum(losses))/len(error),
                'joint_q_gradient_per_label':gradient/len(error),'conditional_block_q_gradient_per_label':(len(tasks)*gradient-sum(derivatives))/len(error)}
        current = objectives(c)
        np.testing.assert_allclose(current['joint_nll_per_label'],law['source_negative_log_likelihood_per_label'],atol=1e-8,rtol=1e-8)
        native = read(Path(profile['output'])/suffix/'selected_source_fit.json')['heads'][0]['joint_structural']
        original = objectives(covariance(np.diag(native['common_prior_coefficient_diagonal']),np.diag(native['residual_coefficient_diagonal'])))
        probe = objectives(c+spec['fixed_probe_q']*direction)
        verification = {'fitted_joint_source_nll_reproduced':True}
        if deleted==spec['gradient_verification_deleted_task']:
            step=spec['gradient_step']; one,two=objectives(c+step*direction),objectives(c+2*step*direction)
            finite=(4*one['sum_conditional_block_nll_per_label']-two['sum_conditional_block_nll_per_label']-3*current['sum_conditional_block_nll_per_label'])/(2*step)
            np.testing.assert_allclose(finite,current['conditional_block_q_gradient_per_label'],atol=1e-5,rtol=1e-4)
            verification['conditional_q_gradient_error']=float(abs(finite-current['conditional_block_q_gradient_per_label']))
        rows.append({'layout':layout,'deleted_task':deleted,'fitted':current,
            'joint_vs_native_conditional_log_density_gain_per_label':original['sum_conditional_block_nll_per_label']-current['sum_conditional_block_nll_per_label'],
            'q_probe_conditional_log_density_gain_per_label':current['sum_conditional_block_nll_per_label']-probe['sum_conditional_block_nll_per_label'],
            'verification':verification})
    summary={layout:{'mean_q_joint_gradient':float(np.mean([r['fitted']['joint_q_gradient_per_label'] for r in rows if r['layout']==layout])),
        'mean_q_conditional_block_gradient':float(np.mean([r['fitted']['conditional_block_q_gradient_per_label'] for r in rows if r['layout']==layout])),
        'mean_q_probe_conditional_density_gain':float(np.mean([r['q_probe_conditional_log_density_gain_per_label'] for r in rows if r['layout']==layout])),
        'mean_joint_vs_native_conditional_density_gain':float(np.mean([r['joint_vs_native_conditional_log_density_gain_per_label'] for r in rows if r['layout']==layout]))} for layout in ('5','6')}
    out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    result={'execution':spec,'rows':rows,'summary':summary,'cpu_seconds':time.process_time()-started,'cost':spec['cost'],
        'hyperparameters_and_empirical_center_use_all_five_source_tasks':True,'honest_inner_task_validation':False,
        'query_paid_outcomes_read':False,'source6_7_evaluated':False,'target_promoted':False,'independent_confirmation':False,'no_model_adopted':True}
    save(out/'results.json',result);print({'summary':summary,'cpu_seconds':result['cpu_seconds']},flush=True)


if __name__=='__main__':
    run()

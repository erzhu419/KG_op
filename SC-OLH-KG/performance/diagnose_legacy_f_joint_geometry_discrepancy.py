"""Source-only score for the original f white discrepancy parameter."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis

EXECUTION = 'performance/manifests/legacy_joint_population_f_joint_geometry_discrepancy_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION); fitted = read(spec['fitted_source_laws'])['rows']
    parent = read(spec['source_execution']); coverage, _ = source_paths(read(parent['template_execution'])); rows = []
    for law in fitted:
        layout, deleted = law['layout'], law['deleted_task']
        inputs = read(Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'/'inputs.json')
        basis = OrthogonalLowOrderBasis(inputs['public_mean_reference']); tasks = inputs['source_tasks']
        assert {t['name'] for t in tasks} == {f'related_inventory_{i:02d}' for i in range(6) if i != deleted}
        designs = [np.column_stack([np.ones(len(t['psi'])), basis.transform(np.asarray(t['psi']))]) for t in tasks]
        d = np.vstack(designs); error = np.concatenate([np.asarray(t['response'])[:, 0] for t in tasks])-d@np.asarray(law['empirical_center_coefficient_mean'])
        a = np.asarray(law['common_prior_coefficient_covariance']); r = np.asarray(law['residual_coefficient_covariance'])
        covariance = d@a@d.T+block_diag(*[local@r@local.T+np.diag(np.asarray(t['noise'])[:, 0]+law['discrepancy_variance'])
            for local, t in zip(designs, tasks)])
        signal = law['source_signal_scale']
        def likelihood(amount):
            factor = cho_factor(covariance+signal*amount*np.eye(len(error)), lower=True)
            alpha = cho_solve(factor, error)
            return .5*(len(error)*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+error@alpha)/len(error)
        baseline = likelihood(0.)
        np.testing.assert_allclose(baseline, law['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
        factor = cho_factor(covariance, lower=True); inverse = cho_solve(factor, np.eye(len(error))); alpha = cho_solve(factor, error)
        gradient = .5*signal*(np.trace(inverse)-alpha@alpha)/len(error)
        information = .5*signal**2*np.sum(inverse**2)
        verification = {'cached_fitted_source_likelihood_reproduced': True}
        if deleted == spec['gradient_verification_deleted_task']:
            step = spec['gradient_step']; finite = (4*likelihood(step)-likelihood(2*step)-3*baseline)/(2*step)
            np.testing.assert_allclose(finite, gradient, atol=1e-6, rtol=1e-4)
            verification.update(one_sided_gradient_error=float(abs(finite-gradient)))
        rows.append({'layout':layout, 'deleted_task':deleted, 'original_discrepancy_variance':law['discrepancy_variance'],
            'scaled_discrepancy_gradient_per_label':float(gradient), 'total_fisher_information':float(information),
            'fixed_probe_source_log_likelihood_gain_per_label':float(baseline-likelihood(spec['fixed_probe_over_signal'])),
            'verification':verification})
    summary = {layout:{'mean_gradient_per_label':float(np.mean([r['scaled_discrepancy_gradient_per_label'] for r in rows if r['layout']==layout])),
        'mean_fixed_probe_log_likelihood_gain_per_label':float(np.mean([r['fixed_probe_source_log_likelihood_gain_per_label'] for r in rows if r['layout']==layout])),
        'improving_folds':sum(r['fixed_probe_source_log_likelihood_gain_per_label'] > 0 for r in rows if r['layout']==layout)} for layout in ('5','6')}
    passed = all(r['mean_fixed_probe_log_likelihood_gain_per_label'] > 0 for r in summary.values())
    out = ROOT/spec['output']; out.mkdir(parents=True,exist_ok=True)
    result = {'execution':spec, 'rows':rows, 'summary':summary, 'source_discrepancy_direction_passed':passed,
        'cpu_seconds':time.process_time()-started, 'cost':spec['cost'], 'query_paid_outcomes_read':False,
        'source6_7_evaluated':False, 'target_promoted':False, 'independent_confirmation':False, 'no_model_adopted':True}
    save(out/'results.json',result); print({'summary':summary,'gate':passed,'cpu_seconds':result['cpu_seconds']},flush=True)


if __name__ == '__main__':
    run()

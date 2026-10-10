"""Positive source-v rules for the frozen full common/task covariance laws."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import roots_legendre

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.prepare_legacy_f_task_level_reference_quadrature import approximation_errors, passes
from performance.run_legacy_related_inventory_source import save

EXECUTION = 'performance/manifests/legacy_joint_population_f_joint_geometry_quadrature_execution_v1.json'


def posterior_density_u(u, law):
    signal = law['source_signal']; u = np.asarray(u); v = signal*(1-u**2)/u**2
    eigenvalues, rhs = np.asarray(law['task_variance_eigenvalues']), np.asarray(law['task_variance_rhs'])
    def difference(variance):
        x = np.asarray(variance)[..., None]*eigenvalues
        return .5*np.sum(np.log1p(x)-np.asarray(variance)[..., None]/(1+x)*rhs**2, axis=-1)
    information = .5*np.sum((eigenvalues/(1+v[..., None]*eigenvalues))**2, axis=-1)
    return np.exp(-difference(v)+difference(law['fitted_independent_task_intercept_variance'])+.5*np.log(information)
        +np.log(2*signal)-3*np.log(u)-law['reference_log_normalizer_relative_to_fitted_nll'])


def rule_on_intervals(law, intervals, size):
    x, w = roots_legendre(size); nodes, weights = [], []
    for lo, hi in intervals:
        u = .5*(lo+hi)+.5*(hi-lo)*x
        nodes.extend((law['source_signal']*(1-u**2)/u**2).tolist())
        weights.extend((.5*(hi-lo)*w*posterior_density_u(u, law)).tolist())
    return np.asarray(nodes), np.asarray(weights)


def run():
    started = time.process_time(); spec = read(EXECUTION); reference = read(spec['reference_results'])
    assert reference['all12_source_fits_and_v_integrations_verified']
    laws = read(spec['reference_laws'])['rows']; rules, rows = [], []
    for law in laws:
        layout, deleted = law['layout'], law['deleted_task']; signal = law['source_signal']
        moments = next(r['integration']['reference_posterior'] for r in reference['rows'] if r['layout']==layout and r['deleted_task']==deleted)
        record = {'reference_posterior':moments}; mean = moments['mean']; second = mean**2+moments['standard_deviation']**2
        eigenvalues, rhs = np.asarray(law['task_variance_eigenvalues']), np.asarray(law['task_variance_rhs'])
        def integrand(u):
            if u == 0.:
                v = law['fitted_independent_task_intercept_variance']; x = v*eigenvalues
                difference = .5*np.sum(np.log1p(x)-v/(1+x)*rhs**2)
                log_tail = difference+.5*np.sum(rhs**2/eigenvalues-np.log(eigenvalues))+.5*np.log(len(eigenvalues)/2)-law['reference_log_normalizer_relative_to_fitted_nll']
                endpoint = np.exp(np.log(2)+(2-len(eigenvalues)/2)*np.log(signal)+log_tail)/second
                return np.r_[0.,0.,endpoint,np.zeros(len(eigenvalues)+len(eigenvalues)**2)]
            v = signal*(1-u**2)/u**2; x = v*eigenvalues; shrinkage = x/(1+x)
            return posterior_density_u(u,law)*np.r_[1.,v/mean,v**2/second,shrinkage,np.outer(shrinkage,shrinkage).ravel()]
        points = [np.sqrt(signal/(signal+v)) for v in (law['fitted_independent_task_intercept_variance'],.1*signal) if v>0]
        t = spec['quadrature']; integral,error,info = quad_vec(integrand,0.,1.,points=points,norm='max',full_output=True,workers=1,
            epsabs=t['absolute_tolerance_for_scaled_integrals'],epsrel=t['relative_tolerance'],limit=t['maximum_subintervals'])
        assert info.success and error <= max(t['absolute_tolerance_for_scaled_integrals'],t['relative_tolerance']*np.max(np.abs(integral)))
        intervals = sorted(info.intervals.tolist()); trials = []; selected = None
        for size in spec['nodes_per_subinterval_resolutions']:
            nodes,weights = rule_on_intervals(law,intervals,size); errors = approximation_errors(nodes,weights,law,record)
            passed = passes(errors,law,record,spec); trials.append({'nodes_per_interval':size,'node_count':len(nodes),'errors':errors,'passed':passed})
            if passed:
                selected = {'layout':layout,'deleted_task':deleted,'source_reference_law':law,'variance_nodes':nodes.tolist(),
                    'posterior_weights':(weights/weights.sum()).tolist(),'adaptive_u_intervals':intervals,'nodes_per_interval':size}
                rules.append(selected); break
        if selected is None: raise RuntimeError(f'Source quadrature inadequate for {layout}/{deleted};no paid forecast allowed')
        rows.append({'layout':layout,'deleted_task':deleted,'adaptive_vector_error':float(error),'trials':trials,'selected_node_count':len(selected['variance_nodes'])})
    out = ROOT/spec['output']; out.mkdir(parents=True,exist_ok=True)
    save(out/'source_quadrature.json',{'execution':spec,'all12_source_rules_frozen_without_query_data':True,'rows':rules})
    result = {'execution':spec,'rows':rows,'all12_source_rules_verified':True,'cpu_seconds':time.process_time()-started,
        'cost':spec['cost'],'query_paid_outcomes_read':False,'source6_7_evaluated':False,'target_promoted':False,'independent_confirmation':False}
    save(out/'results.json',result); print({'node_counts':[r['selected_node_count'] for r in rows],'cpu_seconds':result['cpu_seconds']},flush=True)


if __name__ == '__main__':
    run()

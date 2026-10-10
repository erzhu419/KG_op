"""Exact density ratio for an affine pushforward of the cached raw posterior."""

import signal
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.adapt_historical_source_covariance import (
    CollapsedRawSourcePosterior, gaussian_rows_log_density, importance_summary, read, save)
from performance.prepare_legacy_f_variance_reference import variance_reference
from performance.refine_historical_raw_lookahead import cpu_timeout

PLAN = 'performance/manifests/historical_source_sample_transport_v1.json'


def reference_transport(draws, old_reference, new_reference):
    """Push forward the actual samples; compensate the full Jacobian.

    References are deterministic functions of the fixed source law and paid
    labels. The source samples themselves do not fit the transformation.
    The resulting proposal is a transformed non-Gaussian old posterior.
    """
    old_mean, old_covariance = old_reference; new_mean, new_covariance = new_reference
    old_root, new_root = map(np.linalg.cholesky, (old_covariance, new_covariance))
    whitened = solve_triangular(old_root, (draws-old_mean).T, lower=True).T
    transformed = new_mean + whitened @ new_root.T
    log_jacobian = np.log(np.diag(new_root)).sum()-np.log(np.diag(old_root)).sum()
    return transformed, float(log_jacobian)


def run():
    started = time.process_time(); spec = read(PLAN); parent = read(spec['covariance_plan'])
    covariance = read(spec['covariance_results']); assert covariance['all_registered_cases_complete']
    assert covariance['eligible_raw_importance_cases'] == 0
    laws = read(spec['source_laws'])['rows']; fields = read(parent['source_fields'])['rows']
    mixtures = read(parent['source_components'])['rows']; rows, checks = [], []
    output = ROOT/spec['output']
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_single_thread_cpu_seconds'])
    try:
        for layout in ('5','6'):
            source = next(r for r in fields if (r['layout'],r['active_arm']) == (layout,parent['proposal_arm']))
            mixture = next(r for r in mixtures if (r['layout'],r['arm']) == (layout,parent['proposal_arm']))
            draws = np.asarray(mixture['eta_states_by_chain']).reshape(512,48)
            old_model = CollapsedRawSourcePosterior(source['f_means'][source['active_arm_index']],
                source['f_covariances'][source['active_arm_index']],source['batch_means'],source['sample_variances'])
            old_logs = (gaussian_rows_log_density(draws,np.asarray(source['eta_mean']),np.asarray(source['eta_covariance']))
                + np.array([old_model.evaluate(e)[0] for e in draws]))
            old_reference = variance_reference(source['eta_mean'],source['eta_covariance'],source['sample_variances'])
            unchanged, jacobian = reference_transport(draws,old_reference,old_reference)
            np.testing.assert_allclose(unchanged,draws,atol=1e-12,rtol=0.); assert abs(jacobian)<1e-12
            checks.append({'layout':layout,'unchanged_transport_max_error':float(abs(unchanged-draws).max()),
                'unchanged_log_Jacobian':jacobian,'reference_is_not_replacing_the_scientific_prior':True,'passed':True})
            for kind in ('spectral','additive'):
                for support in ('head_specific','joint'):
                    for treatment in ('observed_task','deconvolved_task'):
                        heads = [next(r for r in laws if (r['layout'],r['selector'],r['support'],r['treatment'],r['head']) ==
                            (layout,kind,support,treatment,h)) for h in ('f','eta_f')]
                        f,eta = heads
                        reference = variance_reference(eta['source_mean'],eta['source_covariance'],source['sample_variances'])
                        transformed,jacobian = reference_transport(draws,old_reference,reference)
                        model = CollapsedRawSourcePosterior(f['source_mean'],f['source_covariance'],
                            source['batch_means'],source['sample_variances'])
                        logs = (gaussian_rows_log_density(transformed,np.asarray(eta['source_mean']),np.asarray(eta['source_covariance']))
                            + np.array([model.evaluate(e)[0] for e in transformed]) + jacobian-old_logs)
                        row = {'layout':layout,'selector':kind,'support':support,'treatment':treatment,
                            'affine_log_Jacobian':jacobian,**importance_summary(logs,parent['precision_gates'])}
                        rows.append(row)
                        save(output,{'execution':spec,'rows':rows,'checks':checks,'all_registered_cases_complete':len(rows)==16,
                            'eligible_raw_importance_cases':sum(r['precision_gate_passed'] for r in rows),
                            'cpu_seconds':time.process_time()-started,'no_model_adopted':True,'independent_confirmation':False,
                            'cost':{'new_source_chains':0,'new_source_data':0,'new_simulator_calls':0,'new_verifier_calls':0},
                            'limitations':'Exact change-of-variables ratio for a transformed cached source posterior. Reference maps use fixed labels and unconditioned source laws,not the cached draws or target labels. A Gaussian reference does not add a Gaussian variance-label likelihood. Failed overlap or chain gates cannot rank covariance treatments or attach a source posterior.'})
                        print('transport',layout,kind,support,treatment,'ESS',row['weight_ESS'],'eligible',row['precision_gate_passed'],flush=True)
    except TimeoutError:
        signal.setitimer(signal.ITIMER_PROF,0.)
        saved=read(spec['output']) if rows else {'execution':spec,'rows':[],'checks':checks}
        saved.update(stopped_at_cpu_budget=True,all_registered_cases_complete=False,cpu_seconds=time.process_time()-started)
        save(output,saved)
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.); signal.signal(signal.SIGPROF,handler)


if __name__=='__main__':
    run()

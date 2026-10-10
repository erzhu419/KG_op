"""Locate variance instability using saved source mixtures, without new draws."""

import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_collapsed_source_raw_update import CollapsedRawSourcePosterior, aggregate
from performance.check_legacy_f_correlated_source_raw_update import diagnostics
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.refine_historical_raw_lookahead import cpu_timeout
from performance.run_legacy_related_inventory_source import save

PLAN = 'performance/manifests/historical_source_reconditioning_v1.json'


def run():
    started = time.process_time(); spec = read(PLAN); parent = read(spec['covariance_plan'])
    results = read(Path(spec['output'])/'results.json')
    laws = read(spec['source_laws'])['rows']; sources = read(parent['source_fields'])['rows']; rows = []
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['saved_failure_diagnostic']['maximum_single_thread_cpu_seconds'])
    stopped = False
    try:
        for row in (r for r in results['rows'] if not r['source_posterior_usable']):
            saved = read(row['component_file']); states = np.asarray(saved['positive_eta_mixture']['eta_states_by_chain'])
            law = next(r for r in laws if r['head']=='f' and all(r[k]==row[k]
                for k in ('layout','selector','support','treatment')))
            source = next(r for r in sources if (r['layout'],r['active_arm'])==(row['layout'],parent['proposal_arm']))
            model = CollapsedRawSourcePosterior(law['source_mean'],law['source_covariance'],source['batch_means'],source['sample_variances'])
            moments = [[model.moments(model.evaluate(eta)) for eta in chain] for chain in states]
            ds = np.asarray([[m[0] for m in chain] for chain in moments])
            cs = np.asarray([[m[1] for m in chain] for chain in moments])
            _, first, _, _ = aggregate(ds[:,:64],cs[:,:64]); _, second, _, _ = aggregate(ds[:,64:],cs[:,64:])
            half_error = abs(np.diag(first)-np.diag(second))/((np.diag(first)+np.diag(second))/2)
            site = int(np.argmax(half_error))
            delta = np.asarray(saved['source_f_posterior_mean'])-model.mean
            variance_functional = np.diagonal(cs,axis1=2,axis2=3)+(ds-delta)**2
            diagnostic = diagnostics(variance_functional)
            full = np.asarray(saved['source_f_posterior_variance'])
            conditional_variance = np.diagonal(cs,axis1=2,axis2=3).mean(axis=(0,1))
            rows.append({**row,'retained_worst_variance_site':source['sites'][site],
                'retained_worst_variance_site_index':site,'retained_half_variance_relative_difference':float(half_error[site]),
                'retained_variance_functional_ESS_at_site':diagnostic['autocorrelation_ess'][site],
                'retained_variance_functional_Rhat_at_site':diagnostic['split_rhat'][site],
                'retained_within_eta_conditional_variance_fraction_at_site':float(
                    conditional_variance[site]/variance_functional[:,:,site].mean()),
                'retained_variance_functional_mean_to_full_variance_ratio_at_site':float(variance_functional[:,:,site].mean()/full[site]),
                'failed_full_chain_gate_unchanged':True})
    except TimeoutError:
        signal.setitimer(signal.ITIMER_PROF,0.); stopped=True
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.); signal.signal(signal.SIGPROF,handler)
    save(ROOT/spec['output']/'failed_precision_diagnostic.json',{'execution':spec['saved_failure_diagnostic'],
        'rows':rows,'all_saved_failures_processed':len(rows)==sum(not r['source_posterior_usable'] for r in results['rows']),
        'stopped_at_cpu_budget':stopped,'cpu_seconds':time.process_time()-started,'new_source_draws':0,
        'interpretation':'Diagnostics of the saved fixed-stride128 states per chain only. Variance functionals use the saved full-run mean. Locations and ESS are coarse retained-state diagnostics;they do not recover the unsaved full-chain site ranking or reverse its failed gate.'})


if __name__ == '__main__':
    run()

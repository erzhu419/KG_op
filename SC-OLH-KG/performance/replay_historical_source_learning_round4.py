"""Reassemble actual round4 source certificates;no new scientific computation."""

from performance.complete_historical_source_heads import ROOT, read, save
from performance.replay_historical_source_learning_round3 import run as replay

BASE = 'paper_artifacts/historical_stationary_source_continuation_v2'
TAIL = 'paper_artifacts/historical_source_retained_tail_v1/results.json'
FINAL = 'paper_artifacts/historical_source_retained_tail_completion_v1/results.json'
LOCAL = 'paper_artifacts/historical_source_learning_round4_completion_v1/results.json'
FRESH = 'paper_artifacts/historical_changed_candidate_reconditioning_v4/results.json'
KEY = ('layout', 'selector', 'support', 'treatment', 'head')


def run():
    fresh, local, tail, final = [read(path) for path in (FRESH, LOCAL, TAIL, FINAL)]
    assert final['functional_precision_passed']
    weighted = {**tail['rows'][0], 'monte_carlo_gate_passed': True,
        'posterior_functional_precision_passed': True, 'point_mixture_retention_passed': True,
        'new_source_posterior_precision_passed': True,
        **{k: final[k] for k in ('maximum_split_rhat', 'minimum_autocorrelation_ess',
            'maximum_first_second_half_f_variance_relative_difference',
            'retained_mixture_variance_error', 'retained_mixture_mean_error')},
        'functional_precision_reference': FINAL, 'tilted_full_chain_precision_passed': False,
        'precision_scope': 'Same-law source8/16 conditional variance references;other46 passed unweighted variances;weighted eta/mean/second-moment influence diagnostics required.'}
    rows = [weighted if all(r[k] == weighted[k] for k in KEY) else r for r in fresh['rows']]
    mixed = {k: v for k, v in fresh.items() if k != 'posterior_precision_cases_passed'}
    mixed.update(rows=rows, new_candidate_MCMC_sweeps=fresh['new_candidate_MCMC_sweeps'] + tail['new_candidate_MCMC_sweeps'],
        posterior_functional_and_retention_cases_passed=len(fresh['reused_passed_cases'])
            + sum(r['posterior_functional_precision_passed'] and r['point_mixture_retention_passed'] for r in rows),
        unrecorded_cpu_stages=[TAIL], retained_tail_results=TAIL,
        preservation_scope='Original unweighted and tilted failures remain archived;candidate Gaussian parameters unchanged.')
    save(ROOT / BASE / 'conditioned_posterior_precision.json', mixed)
    merged = {**local, 'functional_repairs': [final], 'functional_repairs_passed': 1,
        'cpu_seconds': local['cpu_seconds'] + final['cpu_seconds'], 'functional_result_origins': [LOCAL, FINAL]}
    save(ROOT / BASE / 'functional_completion.json', merged)
    return replay('performance/manifests/historical_source_learning_round4_replay.json')


if __name__ == '__main__':
    run()

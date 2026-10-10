"""Cached Louis diagonal-curvature diagnostic;no parameter proposal or sampling."""

import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from performance.complete_historical_source_heads import ROOT, IDENTITY, read, save
from performance.learn_historical_raw_source_amplitudes import covariance_builder


def expected_complete_curvature(covariance, directions, second):
    inverse = cho_solve(cho_factor(covariance, lower=True), np.eye(len(covariance)))
    return np.asarray([[(-.5 * np.trace(inverse @ a @ inverse @ b)
        + np.trace(inverse @ a @ inverse @ b @ inverse @ second)) / len(covariance)
        for b in directions] for a in directions])


def run():
    started = time.process_time()
    registry = read('paper_artifacts/historical_source_risk_score_banks_v1/results.json')
    assert registry['all_registered_cases_complete'] and registry['risk_score_precision_cases_passed'] == 32
    source = read('performance/manifests/historical_source_variance_resolution_v4.json')
    laws = read(source['source_laws'])['rows']; rows = []
    for record in registry['rows']:
        bank = read(record['results']); case = bank['current_candidate_parameters']
        law = next(r for r in laws if r['head'] == ('eta_g' if case['head'] == 'g' else 'eta_f')
            and all(r[k] == case[k] for k in IDENTITY))
        theta = np.asarray(case['eta_common_task_scales']); covariance, directions = covariance_builder(law)(theta)[:2]
        summaries = {}
        for name in ('full', 'first_half', 'second_half'):
            moment = bank['risk_sufficient_moments'][name]; delta = np.asarray(moment['eta_posterior_mean']) - law['source_mean']
            second = np.asarray(moment['eta_posterior_covariance']) + np.outer(delta, delta)
            complete = np.diag(expected_complete_curvature(covariance, directions, second))
            score = bank['score_summaries'][name]
            # Equal-weight chains:MCSE^2*ESS recovers the sampled normalized score variance.
            score_variance = (np.asarray(score['scaled_MCSE']) / theta) ** 2 * score['diagnostics']['autocorrelation_ess']
            missing = len(covariance) * score_variance
            summaries[name] = {'scaled_complete_curvature': (complete * theta ** 2).tolist(),
                'scaled_observed_diagonal_curvature': ((complete - missing) * theta ** 2).tolist(),
                'local_missing_information_fraction': (missing / complete).tolist()}
        for coordinate in range(2):
            rows.append({**{k: case[k] for k in (*IDENTITY, 'head')}, 'coordinate': coordinate,
                'bank_results': record['results'], 'scaled_source_score': bank['score_summaries']['full']['scaled_scores'][coordinate],
                **{name: {k: v[coordinate] for k, v in value.items()} for name, value in summaries.items()}})
    fractions = [r['full']['local_missing_information_fraction'] for r in rows]
    result = {'rows': rows, 'source_cases': 32, 'risk_coordinates': len(rows),
        'median_local_missing_information_fraction': float(np.median(fractions)),
        'coordinates_observed_diagonal_below_quarter_complete': sum(v > .75 for v in fractions),
        'negative_observed_diagonal_coordinates': sum(r['full']['scaled_observed_diagonal_curvature'] <= 0 for r in rows),
        'positive_NLL_score_coordinates': sum(r['scaled_source_score'] > 0 for r in rows),
        'cpu_seconds': time.process_time() - started, 'new_candidate_MCMC_sweeps': 0,
        'new_source_labels': 0, 'new_target_simulator_calls': 0, 'new_verifier_calls': 0, 'server_jobs': 0,
        'no_model_adopted': True, 'independent_confirmation': False,
        'scope': 'Local sixth-law diagonal Louis curvature from current posterior moments and equal-weight score variance. Curvature MC uncertainty and off-diagonal score covariance are not estimated;no Newton step,MLE boundary or identifiability conclusion follows.'}
    save(ROOT / 'paper_artifacts/historical_source_risk_score_banks_v1/em_rate.json', result)
    print('local missing-information fraction median', result['median_local_missing_information_fraction'], flush=True)
    return result


if __name__ == '__main__':
    run()

"""Saved-stream prefixes and diagnostic power; never generate missing raw."""
from collections import Counter

import numpy as np
from scipy.stats import nct, norm

from core.certification import gaussian_quantile_tolerance_certificate
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save


MANIFEST = ROOT / 'performance/manifests/mainline_role_terminal_v1.json'
BUDGETS = (80, 112, 128)


def certificate_power(mean, sigma, n, alpha, delta):
    """P(bar Y + k S <= 0) for centered iid Gaussian constraint raw."""
    critical = nct.ppf(1. - delta, n - 1, norm.ppf(1. - alpha) * np.sqrt(n))
    return float(nct.sf(critical, n - 1, -mean / sigma * np.sqrt(n)))


def analyze():
    plan = read(MANIFEST)
    source, output = ROOT / plan['input_dir'], ROOT / plan['output_dir']
    with attempt(output, 'saved-prefix-and-power') as cost:
        cost.update(new_source_fits=0, new_truth_function_calls=0, download_bytes=0)
        parent = read(ROOT / plan['parent_manifest'])
        alpha = parent['task']['alpha']
        delta = parent['decision_protocol']['familywise_delta'] / 3
        # A sign/tail error would invalidate every power interpretation.
        errors = [abs(certificate_power(-norm.ppf(1-alpha), 1., n, alpha, delta)
                      - delta) for n in BUDGETS]
        assert max(errors) < 1e-12, errors
        prefixes, powers, missing_truth = [], [], []
        for seed in parent['target_seeds']:
            base = source / f'seed_{seed}'
            cache = {tuple(r['point']): r['truth'] for r in read(base/'truth.json')}
            initial = tuple(read(base/'initialization.json')['history'][0]['point'])
            neighbors = {tuple(map(int, p)): i for i, p in enumerate(
                np.load(base/'neighbors.npz')['points'])}
            seen = set()
            for arm in parent['arms']:
                directory = base / arm['name']
                result = read(directory/'verification.json')['verification']
                raw = read(directory/'paid_verification_raw.json')
                cursor = 0
                for item in result['attempts']:
                    index = item['candidate_index']
                    record = result['frozen_shortlist'][index]
                    point = tuple(record['point'])
                    count = item['sample_count']
                    segment = raw[cursor:cursor+count]
                    cursor += count
                    assert len(segment) == count
                    assert all(tuple(r['point']) == point for r in segment)
                    values = np.asarray([r['raw'][1] for r in segment])
                    by_budget = {}
                    for n in BUDGETS:
                        if n > count:
                            by_budget[str(n)] = {'status': 'missing_raw',
                                                 'missing_count': n-count}
                        else:
                            cert = gaussian_quantile_tolerance_certificate(
                                values[:n], tau=item['tau'], alpha=item['alpha'],
                                delta=item['delta'])
                            by_budget[str(n)] = dict(status='available_prefix',
                                certified=cert.certified, upper_margin=cert.upper_margin)
                    prefixes.append(dict(seed=seed, arm=arm['name'],
                        shortlist_position=index+1, initial0=point == initial,
                        neighbor_index=neighbors.get(point),
                        stream_tag=item['verification_stream_tag'],
                        stored_sample_count=count, prefixes=by_budget))
                assert cursor + result['objective_comparison_budget'] == len(raw)
                # Include every frozen candidate with existing truth, not just winners.
                for record in result['frozen_shortlist']:
                    point = tuple(record['point'])
                    if point in seen:
                        continue
                    seen.add(point)
                    if point not in cache:
                        missing_truth.append(dict(seed=seed, arm=arm['name'],
                            shortlist_position=record['shortlist_position']))
                        continue
                    truth = cache[point]
                    powers.append(dict(seed=seed, initial0=point == initial,
                        neighbor_index=neighbors.get(point), truth=truth,
                        power_by_budget={str(n): certificate_power(
                            truth['constraint_mean_centered'], truth['constraint_sigma'],
                            n, alpha, delta) for n in BUDGETS}))
        available = [row for row in prefixes
                     if row['prefixes']['112']['status'] == 'available_prefix']
        result = dict(status='complete', budgets=BUDGETS, candidate_delta=delta,
            boundary_power_max_error=max(errors), saved_stream_prefixes=prefixes,
            unique_task_policy_powers=powers, missing_cached_truth=missing_truth,
            prefix_summary=dict(attempts=len(prefixes), available112=len(available),
                missing112=len(prefixes)-len(available),
                available112_vs128=Counter(
                    f"{r['prefixes']['112']['certified']}->{r['prefixes']['128']['certified']}"
                    for r in available)),
            scope='Prefix outcomes are conditional on old attempted streams. '
                  'All18 primary streams lack32 raw for112. No complete alternative '
                  'deployment can be inferred. Truth-based power is descriptive only.')
    result['cost'] = cost
    save(output/'power_and_prefixes.json', result)
    print(result['prefix_summary'], flush=True)


if __name__ == '__main__':
    analyze()

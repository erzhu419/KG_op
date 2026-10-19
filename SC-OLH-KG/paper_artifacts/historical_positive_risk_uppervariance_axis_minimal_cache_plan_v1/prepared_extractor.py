"""Summarize existing origin4 caches; no numerical model evaluation."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

PLAN = 'performance/manifests/historical_positive_risk_uppervariance_axis_minimal_cache_plan_v1.json'


def read(path):
    return json.loads(Path(path).read_text())


def new_group():
    return dict(nodes=0, weight_sum=0.0, risk_weighted_sum=0.0,
                order_counts={}, order_weight_sums={})


def add(group, weight, value, order):
    group['nodes'] += 1
    group['weight_sum'] += weight
    group['risk_weighted_sum'] += weight * value
    key = str(order)
    group['order_counts'][key] = group['order_counts'].get(key, 0) + 1
    group['order_weight_sums'][key] = group['order_weight_sums'].get(key, 0.0) + weight


def summarize(spec):
    intervals = spec['intervals']

    def bucket(mean_cdf):
        return next(i for i, (low, high) in enumerate(intervals) if low <= mean_cdf < high)

    current = [new_group() for _ in intervals]
    gauss = [new_group() for _ in intervals]
    task = read(spec['missing_sources']['current_fixed'])
    assert task['complete']
    fixed = {r['record']: r['fixed'] for r in task['rows'] if r['policy'] == 63}
    for row in spec['current_metadata']:
        saved = fixed[row['record']]
        group = current[bucket(row['cdf'][0])]
        add(group, row['outer_weight'], saved['raw'], 128)
        group['weighted64_128_perturbation'] = group.get('weighted64_128_perturbation', 0.0) + row['outer_weight'] * saved['raw64_128_difference']
        group['maximum_normalization_error'] = max(group.get('maximum_normalization_error', 0.0), saved['raw_evidence_error'])

    for path in spec['missing_sources']['gauss_shards']:
        bank = read(path)
        columns = [bank[key] for key in ('probabilities', 'weights', 'values', 'inner_orders')]
        assert len({len(column) for column in columns}) == 1
        for point, weight, value, order in zip(*columns):
            add(gauss[bucket(point[0])], weight, value, order)

    pi_f = spec['prior_mass'] * spec['current_f16_mean']
    rows = []
    for i, interval in enumerate(intervals):
        for group in (current[i], gauss[i]):
            group['fixed_reward_contribution'] = pi_f * group['risk_weighted_sum']
        rows.append(dict(mean_cdf_interval=interval, target=i in spec['target_interval_indices'],
                         current=current[i], old_gauss_node_mask=gauss[i]))

    current_total = sum(g['risk_weighted_sum'] for g in current)
    gauss_total = sum(g['risk_weighted_sum'] for g in gauss)
    # Detect omitted/duplicated records and accidental renormalization of shards.
    assert sum(g['nodes'] for g in current) == 112
    assert sum(g['nodes'] for i, g in enumerate(current) if i in spec['target_interval_indices']) == 80
    assert sum(g['nodes'] for g in gauss) == 2048
    assert abs(sum(g['weight_sum'] for g in current) - 1) < 1e-12
    assert abs(sum(g['weight_sum'] for g in gauss) - 1) < 1e-12
    assert abs(current_total - spec['expected_current_risk_sum']) < 1e-12
    assert abs(gauss_total - spec['expected_gauss_risk_sum']) < 1e-12

    r2048 = read(spec['missing_sources']['r2048_risk_orders'])
    assert r2048['complete'] and len(r2048['orders']) == 2048
    counts = Counter(r2048['orders'])
    return dict(complete=True, origin=4, fixed_action=355, new_integrations=0,
                intervals=rows, current_risk_sum=current_total, gauss_risk_sum=gauss_total,
                r2048_eta_order_counts=dict(counts),
                r2048_eta_order_weight_sums={str(k): n / 2048 for k, n in counts.items()},
                r2048_all_inner_passed=r2048['all_inner_rules_passed'],
                matched_interval_integrals_available=False,
                same_node_adaptive_vs128_differences_available=False,
                weights_renormalized=False,
                limitations='Gauss interval masks partition the old discrete rule;their mass need not equal interval length. They are not matched interval integrals or interval error estimates. Order histograms do not recover missing128-stage values.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=PLAN)
    args = parser.parse_args()
    started = time.process_time()
    result = dict(complete=False, new_integrations=0)
    try:
        result = summarize(read(args.plan))
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
    finally:
        result['reader_process_cpu_seconds'] = time.process_time() - started
        print(json.dumps(result))


if __name__ == '__main__':
    main()

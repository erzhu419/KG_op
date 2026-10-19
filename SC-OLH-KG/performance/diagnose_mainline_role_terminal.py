"""Trace the frozen stage29 terminal decisions using saved JSON and raw only.

No engines, source models, simulator, or truth functions are restored. Policy
identity is exact physical-vector equality; output labels point into the saved
initialization, neighbor bank, or query history rather than creating hashes.
"""
import argparse
from collections import Counter
from dataclasses import asdict

import numpy as np
from scipy.stats import t as student_t

from core.certification import gaussian_quantile_tolerance_certificate
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save

SOURCE = ROOT / 'paper_artifacts/mainline_role_fresh_v1'
OUTPUT = ROOT / 'paper_artifacts/mainline_role_terminal_v1'


def policy_key(point):
    return tuple(int(value) for value in point)


def replay_certificates(verified, raw):
    """Check each original certificate and both original paired guards."""
    terminal = verified['verification']
    shortlist = terminal['frozen_shortlist']
    cursor, records = 0, []
    for saved in terminal['attempts']:
        index, count = saved['candidate_index'], saved['sample_count']
        segment = raw[cursor:cursor + count]
        point = policy_key(shortlist[index]['point'])
        if len(segment) != count or any(policy_key(x['point']) != point for x in segment):
            raise ValueError('Saved certificate raw does not match its frozen policy and count')
        replayed = asdict(gaussian_quantile_tolerance_certificate(
            [row['raw'][1] for row in segment], tau=saved['tau'],
            alpha=saved['alpha'], delta=saved['delta']))
        fields = ('upper_margin', 'sample_mean', 'sample_std', 'quantile_upper',
                  'tolerance_factor', 'noncentral_t_quantile', 'noncentrality')
        error = max(abs(replayed[name] - saved[name]) for name in fields)
        passed = replayed['certified'] == saved['certified']
        if error > 1e-12 or not passed:
            raise ValueError('Saved certificate cannot be reproduced from its paid raw')
        records.append(dict(candidate_index=index, shortlist_position=index + 1,
            raw_start=cursor, raw_stop=cursor + count, sample_count=count,
            stream_tag=saved['verification_stream_tag'], delta=saved['delta'],
            alpha=saved['alpha'], tau=saved['tau'], upper_margin=saved['upper_margin'],
            certified=saved['certified'], replay_max_absolute_error=error,
            replay_pass_matches=passed))
        cursor += count

    guard = terminal['objective_comparison']
    comparison = dict(status=guard['status'], simulation_calls=guard['simulation_calls'],
                      delta=guard['delta'])
    if guard['simulation_calls']:
        count = guard['comparison_budget_per_policy']
        segment = raw[cursor:cursor + 2 * count]
        challenger = policy_key(shortlist[0]['point'])
        incumbent = policy_key(shortlist[terminal['objective_incumbent_position'] - 1]['point'])
        if len(segment) != 2 * count or any(
                policy_key(row['point']) != (challenger if i % 2 == 0 else incumbent)
                for i, row in enumerate(segment)):
            raise ValueError('Saved paired guard raw order is not challenger/incumbent')
        differences = np.asarray([segment[2 * i]['raw'][0] - segment[2 * i + 1]['raw'][0]
                                  for i in range(count)])
        mean, std = float(np.mean(differences)), float(np.std(differences, ddof=1))
        se = float(std / np.sqrt(count))
        critical = float(student_t.ppf(1. - guard['delta'], count - 1))
        upper = mean + critical * se
        replayed = dict(sample_mean_difference=mean, sample_std_difference=std,
                        standard_error=se, critical_value=critical,
                        one_sided_upper_confidence_bound=upper)
        error = max(abs(value - guard[name]) for name, value in replayed.items())
        passed = bool(upper < 0.) == guard['challenger_dominates']
        if error > 1e-12 or not passed:
            raise ValueError('Saved paired objective guard cannot be reproduced')
        comparison.update(replayed, sample_count_per_policy=count,
            stream_tag=guard['comparison_stream_tag'], raw_start=cursor,
            raw_stop=cursor + 2 * count, challenger_dominates=guard['challenger_dominates'],
            replay_max_absolute_error=error, replay_pass_matches=passed)
        cursor += 2 * count
    if cursor != len(raw) or cursor != terminal['verification_budget']:
        raise ValueError('Paid verification raw is not fully accounted for')
    return records, comparison


def summarize(source=SOURCE):
    results = read(source / 'results.json')
    plan = results['protocol']
    rows, seed_catalogs = [], []
    for seed in plan['target_seeds']:
        task = source / f'seed_{seed}'
        initial = read(task / 'initialization.json')['history']
        neighbors = [policy_key(x) for x in np.load(task / 'neighbors.npz')['points']]
        truth = {policy_key(row['point']): row['truth'] for row in read(task / 'truth.json')}
        initial_keys = [policy_key(row['point']) for row in initial]
        initial_truth = truth[initial_keys[0]]
        searches = {arm['name']: read(task / arm['name'] / 'search.json') for arm in plan['arms']}
        labels, catalog = {}, []
        # Use the existing neighbor index whenever available, including initial0.
        def label(point, location):
            point = policy_key(point)
            if point not in labels:
                ni = neighbors.index(point) if point in neighbors else None
                ii = [i for i, candidate in enumerate(initial_keys) if point == candidate]
                name = f'neighbor:{ni}' if ni is not None else (f'initial:{ii[0]}' if ii else location)
                labels[point] = name
                catalog.append(dict(policy=name, neighbor_index=ni,
                    initial_indices=ii, source_location=location,
                    truth=truth.get(point), truth_available=point in truth))
            return labels[point]

        for i, point in enumerate(initial_keys):
            label(point, f'initialization.json#/history/{i}/point')
        for arm, search in searches.items():
            for i, history in enumerate(search['history']):
                label(history['point'], f'{arm}/search.json#/history/{i}/point')
            for i, entry in enumerate(search['terminal']['frozen_shortlist']):
                label(entry['point'], f'{arm}/search.json#/terminal/frozen_shortlist/{i}/point')

        for arm, search in searches.items():
            verified = read(task / arm / 'verification.json')
            terminal = verified['verification']
            raw = read(task / arm / 'paid_verification_raw.json')
            certificates, comparison = replay_certificates(verified, raw)
            attempts_by_position = {x['shortlist_position']: x for x in certificates}
            deployed = label(verified['deployed'], f'{arm}/verification.json#/deployed')
            primary = label(verified['primary'], f'{arm}/verification.json#/primary')
            initial_label = labels[initial_keys[0]]
            shortlist = []
            for entry in terminal['frozen_shortlist']:
                position = entry['shortlist_position']
                policy = label(entry['point'], f'{arm}/verification.json#/verification/frozen_shortlist/{position - 1}/point')
                cert = attempts_by_position.get(position)
                if cert is None:
                    disposition = 'not_verified_after_earlier_terminal_decision'
                elif not cert['certified']:
                    disposition = 'safety_certificate_failed'
                elif policy == deployed:
                    disposition = 'deployed'
                elif position == 1 and comparison['status'] == 'incumbent_retained':
                    disposition = 'objective_guard_failed'
                else:
                    disposition = 'certified_not_deployed'
                shortlist.append(dict(policy=policy, shortlist_position=position,
                    shortlist_role=entry['shortlist_role'], posterior_rank=entry.get('posterior_rank'),
                    objective_incumbent=entry.get('objective_incumbent', False),
                    planned_safety_budget=terminal['candidate_verification_budgets'][position - 1],
                    certificate=cert, disposition=disposition,
                    truth=truth.get(policy_key(entry['point']))))
            shortlist_by_policy = {x['policy']: x for x in shortlist}
            query_events, distinct = [], {}
            for i, history in enumerate(search['history'][10:]):
                key = policy_key(history['point'])
                policy = labels[key]
                actual = truth[key]
                improving = actual['true_feasible'] and actual['objective'] < initial_truth['objective']
                entry = shortlist_by_policy.get(policy)
                disposition = entry['disposition'] if entry else 'not_on_frozen_shortlist'
                event = dict(query_index=i, history_index=10 + i, policy=policy,
                    truth=actual, feasible_improving=improving,
                    objective_gain=initial_truth['objective'] - actual['objective'],
                    quality_gain=actual['quality'] - initial_truth['quality'],
                    shortlist_position=entry['shortlist_position'] if entry else None,
                    posterior_rank=entry['posterior_rank'] if entry else None,
                    disposition=disposition)
                query_events.append(event)
                if policy not in distinct:
                    distinct[policy] = dict(event, query_indices=[])
                distinct[policy]['query_indices'].append(i)
            improving_events = [event for event in query_events if event['feasible_improving']]
            improving_policies = [event for event in distinct.values() if event['feasible_improving']]
            logical_cap = 362 if arm == 'initialization_only' else 365
            rows.append(dict(seed=seed, arm=arm, initial=initial_label, primary=primary,
                deployed=deployed, deployment_changed_from_initial=deployed != initial_label,
                certified=terminal['certified'], shortlist=shortlist,
                primary_selection=terminal['objective_challenger'],
                query_events=query_events, distinct_query_policies=list(distinct.values()),
                feasible_improving_events=len(improving_events),
                distinct_feasible_improving_policies=len(improving_policies),
                improving_query_dispositions=dict(Counter(x['disposition'] for x in improving_events)),
                improving_policy_dispositions=dict(Counter(x['disposition'] for x in improving_policies)),
                certificates=certificates, objective_comparison=comparison,
                familywise_delta=terminal['familywise_delta'],
                per_candidate_delta=terminal['per_candidate_delta'],
                budgets=dict(logical_search_calls=search['logical_search_calls'],
                    safety_calls=terminal['safety_verification_budget'],
                    objective_guard_calls=terminal['objective_comparison_budget'],
                    actual_verification_calls=terminal['verification_budget'],
                    maximum_verification_calls=terminal['max_verification_budget'],
                    unused_verification_cap=terminal['max_verification_budget'] - terminal['verification_budget'],
                    actual_logical_calls=search['logical_search_calls'] + terminal['verification_budget'],
                    maximum_logical_calls=logical_cap)))
        seed_catalogs.append(dict(seed=seed, initial=labels[initial_keys[0]], policies=catalog))

    arm_summary = []
    for arm in [entry['name'] for entry in plan['arms']]:
        group = [row for row in rows if row['arm'] == arm]
        event_dispositions, policy_dispositions = Counter(), Counter()
        for row in group:
            event_dispositions.update(row['improving_query_dispositions'])
            policy_dispositions.update(row['improving_policy_dispositions'])
        arm_summary.append(dict(arm=arm, cases=len(group),
            query_events=sum(len(row['query_events']) for row in group),
            feasible_improving_events=sum(row['feasible_improving_events'] for row in group),
            distinct_feasible_improving_policies=sum(row['distinct_feasible_improving_policies'] for row in group),
            improving_event_dispositions=dict(event_dispositions),
            improving_policy_dispositions=dict(policy_dispositions),
            actual_verification_calls=sum(row['budgets']['actual_verification_calls'] for row in group),
            unused_verification_cap=sum(row['budgets']['unused_verification_cap'] for row in group)))
    all_certificates = [certificate for row in rows for certificate in row['certificates']]
    guards = [row['objective_comparison'] for row in rows if row['objective_comparison']['simulation_calls']]
    same_initial = []
    for row in rows:
        if row['seed'] != 2026101652:
            continue
        entry = next(x for x in row['shortlist'] if x['policy'] == row['initial'])
        same_initial.append(dict(arm=row['arm'], policy=row['initial'],
            shortlist_position=entry['shortlist_position'], certificate=entry['certificate'],
            deployed=row['deployed'], certified=row['certified']))
    return dict(status='complete', source_directory=str(source.relative_to(ROOT)),
        scope='Saved trajectories only; no new posterior, target observation, truth evaluation or certificate claim.',
        posterior_rank_scope='Only ranks explicitly stored on the frozen shortlist; null means not recorded.',
        policy_identity='Exact physical vector equality. Neighbor and initial indices are zero-based; shortlist positions are one-based.',
        cases=rows, arms=arm_summary, policy_catalog_by_seed=seed_catalogs,
        same_initial_certificate_by_position_seed_2026101652=same_initial,
        certificate_replay=dict(count=len(all_certificates),
            max_absolute_error=max(x['replay_max_absolute_error'] for x in all_certificates),
            all_pass_decisions_match=all(x['replay_pass_matches'] for x in all_certificates),
            objective_guard_count=len(guards),
            objective_guard_max_absolute_error=max(x['replay_max_absolute_error'] for x in guards),
            all_objective_guard_decisions_match=all(x['replay_pass_matches'] for x in guards),
            accounted_paid_raw=sum(row['budgets']['actual_verification_calls'] for row in rows)),
        source_stage_completion=read(source / 'completion.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=str, default=str(OUTPUT))
    args = parser.parse_args()
    from pathlib import Path
    output = Path(args.output)
    with attempt(output, 'saved-terminal-trace-diagnosis', 'FactorShockStatePolicyRZDT1') as cost:
        cost.update(new_target_initial_calls=0, new_source_fits=0,
                    new_truth_function_calls=0, download_bytes=0,
                    replayed_certificate_count=0, replayed_objective_guard_count=0)
        result = summarize()
        cost['replayed_certificate_count'] = result['certificate_replay']['count']
        cost['replayed_objective_guard_count'] = result['certificate_replay']['objective_guard_count']
        save(output / 'diagnosis.json', result)
    result['diagnosis_stage_cost'] = dict(cost)
    save(output / 'diagnosis.json', result)
    print({'cases': len(result['cases']), 'replay': result['certificate_replay'],
           'cpu_seconds': cost['cpu_seconds'], 'new_simulation_calls': 0})


if __name__ == '__main__':
    main()

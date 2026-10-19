"""Audit every observed-policy objective pair using saved posteriors and own raw data."""
import argparse
from collections import Counter
from itertools import combinations

import numpy as np

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save

MANIFEST = ROOT/'performance/manifests/mainline_objective_contrast_exposure_v1.json'
STRATA = ('all', 'initial_initial', 'initial_new', 'new_new',
          'no_repeated_policy', 'one_repeated_policy', 'both_repeated_policies',
          'both_true_feasible', 'both_predicted_eligible')
METRICS = ('posterior_contrast_mse', 'raw_contrast_mse', 'posterior_contrast_mae',
           'raw_contrast_mae', 'posterior_order_accuracy', 'raw_order_accuracy',
           'posterior_repaired_fraction', 'posterior_introduced_fraction')


def key(row):
    return row['replicate'], row['seed'], row['arm']


def sign(value):
    return None if value is None else int(value > 1e-12)-int(value < -1e-12)


def policies(case, outcome):
    result = []
    for row in case['observed']:
        index = row['bank_index']
        assert outcome['moments']['mean'][index][0] == row['objective']
        assert outcome['moments']['variance'][index][0] == row['objective_variance']
        assert outcome['moments']['noise'][index][0] == row['episode_noise'][0]
        raw = [float(y[0]) for i, y in zip(outcome['history_indices'], outcome['history_raw']) if i == index]
        assert len(raw) == len(row['history_indices']) > 0
        variance = float(np.var(raw, ddof=1)) if len(raw) > 1 else None
        result.append(dict(policy=row['policy'], origin='initial' if min(row['history_indices']) < 10 else 'new',
            history_indices=row['history_indices'], raw_count=len(raw), raw_objectives=raw,
            raw_mean=float(np.mean(raw)), empirical_raw_variance=variance,
            empirical_raw_mean_variance=None if variance is None else variance/len(raw),
            predicted_objective=row['objective'], predicted_epistemic_variance=row['objective_variance'],
            predicted_log_noise_mean=outcome['moments']['mean'][index][1],
            predicted_log_noise_variance=outcome['moments']['variance'][index][1],
            predicted_episode_noise_variance=row['episode_noise'][0],
            predicted_raw_mean_variance=row['episode_noise'][0]/len(raw),
            true_objective=None if row['truth'] is None else row['truth']['objective'],
            true_feasible=None if row['truth'] is None else row['truth']['true_feasible'],
            eligible=row['eligible'], probability_violation=row['probability_violation'],
            original_shortlist_position=row['original_shortlist_position'],
            transformed_shortlist_position=row['transformed_shortlist_position'],
            safety_attempted=row['safety_attempted'], safety_passed=row['safety_passed'],
            actual_certified_deployment=row['actual_certified_deployment']))
    assert sum(row['raw_count'] for row in result) == 13
    return result


def pair_record(left, right, native):
    forecast = left['predicted_objective']-right['predicted_objective']
    raw = left['raw_mean']-right['raw_mean']
    truth = (None if left['true_objective'] is None or right['true_objective'] is None
             else left['true_objective']-right['true_objective'])
    origins = sorted((left['origin'], right['origin']))
    repeated = int(left['raw_count'] > 1)+int(right['raw_count'] > 1)
    # Covariance is absent: retain marginal scales, never add them as a difference variance.
    native_difference = (None if native is None else
        native[left['policy']]['objective']-native[right['policy']]['objective'])
    posterior_correct = None if sign(truth) in (None, 0) else sign(forecast) == sign(truth)
    raw_correct = None if sign(truth) in (None, 0) else sign(raw) == sign(truth)
    attribution = ('unknown_or_true_tie' if posterior_correct is None else
        'both_correct' if posterior_correct and raw_correct else
        'posterior_repaired_raw_misorder' if posterior_correct else
        'posterior_introduced_misorder' if raw_correct else 'both_misordered')
    return dict(left=left['policy'], right=right['policy'], direction='left_minus_right',
        origin_pair='_'.join(origins), repeated_group=STRATA[4+repeated],
        predicted_difference=forecast, observed_raw_mean_difference=raw, cached_true_difference=truth,
        posterior_error=None if truth is None else forecast-truth,
        raw_error=None if truth is None else raw-truth,
        posterior_minus_raw_difference=forecast-raw, posterior_order_correct=posterior_correct,
        raw_order_correct=raw_correct, ordering_origin=attribution,
        posterior_difference_variance=None, raw_difference_variance=None,
        native_reference_difference=native_difference,
        native_reference_order_correct=None if sign(truth) in (None, 0) or native_difference is None
            else sign(native_difference) == sign(truth),
        both_true_feasible=(None if left['true_feasible'] is None or right['true_feasible'] is None
                            else left['true_feasible'] and right['true_feasible']),
        both_predicted_eligible=left['eligible'] and right['eligible'])


def pair_metrics(rows):
    known = [row for row in rows if row['cached_true_difference'] is not None]
    non_ties = [row for row in known if sign(row['cached_true_difference']) != 0]
    result = dict(pair_count=len(rows), known_truth_pairs=len(known), non_tie_pairs=len(non_ties),
                  ordering_counts=dict(Counter(row['ordering_origin'] for row in rows)))
    for name, values in (
        ('posterior_contrast_mse', [row['posterior_error']**2 for row in known]),
        ('raw_contrast_mse', [row['raw_error']**2 for row in known]),
        ('posterior_contrast_mae', [abs(row['posterior_error']) for row in known]),
        ('raw_contrast_mae', [abs(row['raw_error']) for row in known]),
        ('posterior_order_accuracy', [row['posterior_order_correct'] for row in non_ties]),
        ('raw_order_accuracy', [row['raw_order_correct'] for row in non_ties]),
        ('posterior_repaired_fraction', [row['ordering_origin']=='posterior_repaired_raw_misorder' for row in non_ties]),
        ('posterior_introduced_fraction', [row['ordering_origin']=='posterior_introduced_misorder' for row in non_ties])):
        result[name] = float(np.mean(values)) if values else None
    return result


def analyze_case(case, outcome, native):
    rows = policies(case, outcome)
    pairs = [pair_record(left, right, native) for left, right in combinations(rows, 2)]
    summaries = {group: pair_metrics([row for row in pairs if group == 'all' or
        row['origin_pair'] == group or row['repeated_group'] == group or row.get(group) is True]) for group in STRATA}
    return dict(**{field: case[field] for field in ('replicate', 'seed', 'domain', 'arm', 'status')},
        paid_observations=13, observed_policy_count=len(rows), policies=rows, pairs=pairs,
        task_metrics=summaries, original_primary=case['original_primary'],
        transformed_primary=case['transformed_primary'], original_shortlist=case['original_shortlist'],
        transformed_shortlist=case['transformed_shortlist'],
        reported_deployed=case['reference']['deployed'], certified=case['reference']['certified'],
        actual_safe_reward=case['reference']['actual_safe_reward'],
        best_initial_quality_increment=case['reference']['best_initial_quality_increment'],
        quality_baseline_defined=case['reference']['baseline_defined'],
        objective_comparison=case['reference']['objective_comparison'],
        native_view_available=native is not None)


def aggregate(cases, arm, registered, domain=None):
    group = [case for case in cases if case['arm']==arm and (domain is None or case['domain']==domain)]
    complete = [case for case in group if case['status']=='complete']
    strata = {}
    for stratum in STRATA:
        available = [case['task_metrics'][stratum] for case in complete if case['task_metrics'][stratum]['pair_count']]
        strata[stratum] = dict(task_count=len(available), pair_count=sum(row['pair_count'] for row in available),
            **{metric: float(np.mean([row[metric] for row in available if row[metric] is not None]))
               if any(row[metric] is not None for row in available) else None for metric in METRICS})
    all_metrics = [case['task_metrics']['all'] for case in complete]
    return dict(arm=arm, domain=domain, registered_tasks=registered, completed_tasks=len(complete),
        missing_tasks=registered-len(complete), strata=strata,
        posterior_minus_raw_order_accuracy_task_signs=dict(
            better=sum(row['posterior_order_accuracy']>row['raw_order_accuracy']+1e-12 for row in all_metrics),
            tied=sum(abs(row['posterior_order_accuracy']-row['raw_order_accuracy'])<=1e-12 for row in all_metrics),
            worse=sum(row['posterior_order_accuracy']<row['raw_order_accuracy']-1e-12 for row in all_metrics)))


def run(plan):
    output = ROOT/plan['output_dir']
    with attempt(output/'target_contrasts', 'saved-target-objective-contrasts') as cost:
        cost.update(new_target_initial_calls=0, new_source_fits=0, new_truth_function_calls=0,
            engine_restorations=0, posterior_restorations=0, raw_conditioning=0, raw_replays=0,
            new_rng_draws=0, download_bytes=0)
        parent = read(ROOT/plan['parent_manifest'])
        saved = read(ROOT/plan['inputs']['target_summary'])
        assert saved['status']=='complete' and saved['registered_units']==parent['registered_units']==72
        lookup = {key(case): case for case in saved['cases']}
        native_views = {(row['replicate'],row['seed'],row['history_arm']):
                        {item['policy']:item for item in row['observed']} for row in saved['saved_native_observed']}
        cases, column_examples = [], []
        for unit in parent['units']:
            case = lookup[key(unit)]
            if case['status'] != 'complete':
                cases.append(dict(**{k:case[k] for k in ('replicate','seed','domain','arm','status')},reason=case.get('reason')))
                continue
            path = ROOT/plan['inputs']['target_outcomes']/unit['domain']/f"seed_{unit['seed']}"/unit['arm']/'outcome.json'
            outcome = read(path)
            assert outcome['status']=='complete'
            analyzed = analyze_case(case,outcome,native_views.get(key(unit)))
            cases.append(analyzed)
            audit_policy = {2026103017:'initial:2', 2026103023:'initial:8', 2026103024:'initial:2'}.get(unit['seed'])
            if audit_policy is not None:
                policy = next(row for row in analyzed['policies'] if row['policy']==audit_policy)
                search = read(ROOT/unit['search_file'])
                raw = [search['history'][i]['raw'] for i in policy['history_indices']]
                assert raw == [outcome['history_raw'][i] for i in policy['history_indices']]
                assert [value[0] for value in raw] == policy['raw_objectives']
                column_examples.append(dict(seed=unit['seed'],arm=unit['arm'],**policy,
                    original_search_raw=raw,source_search_file=unit['search_file'],
                    search_raw_matches_restored_history=True,objective_column=0,eta_f_column=1,
                    source_saved_mean_matches_stage60_objective=True))
        domains = list(dict.fromkeys(case['domain'] for case in cases))
        aggregates = [aggregate(cases,arm,parent['registered_geometry_count']) for arm in parent['arms']]
        families = [aggregate(cases,arm,len({case['seed'] for case in cases if case['domain']==domain}),domain)
                    for arm in parent['arms'] for domain in domains]
        focal = []
        for case in cases:
            if case['seed']==2026103016 and case['status']=='complete':
                selected = {'neighbor:18','neighbor:53','initial:6'}
                focal.append(dict(arm=case['arm'], policies=[p for p in case['policies'] if p['policy'] in selected],
                    pairs=[p for p in case['pairs'] if {p['left'],p['right']} <= selected],
                    original_shortlist=case['original_shortlist'], transformed_shortlist=case['transformed_shortlist'],
                    certified=case['certified'], reported_deployed=case['reported_deployed'],
                    objective_comparison=case['objective_comparison']))
        result = dict(status='complete', registered_geometries=24, registered_units=72,
            outcome_counts=dict(Counter(case['status'] for case in cases)), cases=cases, arms=aggregates,
            by_family=families, focal_3016=focal,
            objective_column_audit=dict(checked_policies=sum(len(c.get('policies',[])) for c in cases),
                all_objective_means_variances_and_noise_match_original_saved_head=True,
                examples=column_examples,
                interpretation='Original search raw first column equals restored first column; saved posterior head 0 equals the actual terminal objective. Cached truth is original objective, not normalized quality. No column rescaling or replacement was applied.'),
            same_history_scope=[{k:v for k,v in row.items() if k!='risk_channel_substitutions'} for row in saved['pairs']],
            references=dict(source_file=plan['inputs']['target_summary'],
                registered_terminal_cells=saved['references']['registered_cells'],
                completed_terminal_cells=saved['references']['completed_cells'],
                arms=saved['references']['arms'], missing_cells=saved['references']['missing_cells'],
                undefined_baseline_cells=saved['references']['undefined_baseline_cells'],
                saved_native_observed_views=len(native_views)),
            definitions=dict(pair_direction='left_minus_right, objective minimized', true_tie_tolerance=1e-12,
                policy_origin='Initial if present among first ten observations, new otherwise; repeated is raw_count>1.',
                aggregation='Each pair metric averages within one task/arm/stratum first; tasks receive equal weight within each arm and family.',
                raw_mean='Only that arm own saved search raw observations; no terminal verification observations enter the means.',
                uncertainty='Per-policy marginal epistemic, predicted episode noise and repeat empirical variance are retained. Cross-policy covariance and difference variance are unknown.'),
            limitations=['Retrospective diagnosis; cached truth never selects or refits a model.',
                'Pair rows and overlapping own-history arms are not independent task replicates.',
                'Most policies have one raw observation; absent empirical variance is null, not zero.',
                'Marginal noise/count is a per-policy model scale, not a calibrated confidence interval for a difference.',
                'Recorded unverified candidates are distinct from certified deployment; altered exposure has no new certificate.'])
        save(output/'target_contrasts.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default=str(MANIFEST))
    run(read(parser.parse_args().manifest))

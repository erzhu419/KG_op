"""Describe actual saved safety/comparison exposure; never replay a verifier."""
import argparse
from collections import Counter

import numpy as np

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save

MANIFEST = ROOT / 'performance/manifests/mainline_objective_contrast_exposure_v1.json'
PROTOCOL = 'same_policy_distinct_challenger'


def stop_reason(verification):
    status = verification['objective_comparison']['status']
    if status in ('challenger_dominates', 'incumbent_retained'):
        return 'paired_objective_decision_after_two_safety_passes'
    if status == 'challenger_not_safety_certified':
        return 'incumbent_safe_primary_failed'
    if status == 'incumbent_not_safety_certified':
        return 'primary_safe_incumbent_failed'
    if status == 'neither_primary_nor_incumbent_safety_certified':
        return 'fallback_safety_passed' if verification['certified'] else 'all_shortlisted_safety_failed'
    return status


def raw_comparison(directory, item, verification, labels):
    """Read only actual paid blocks, preserving their original pair/stream."""
    raw = read(directory / 'paid_verification_raw.json')
    cursor = 0
    for row in verification['attempts']:
        count = row['verification_budget']
        point = item['frozen_shortlist'][row['candidate_index']]['point']
        block = raw[cursor:cursor+count]
        assert len(block) == count and all(r['point'] == point for r in block)
        cursor += count
    count = verification['objective_comparison_budget']
    assert cursor + count == len(raw) == verification['verification_budget']
    if not count:
        return dict(status='no_paid_objective_comparison', raw_file=str(directory.relative_to(ROOT) / 'paid_verification_raw.json'))
    points = [item['frozen_shortlist'][i]['point'] for i in (0, item['objective_incumbent_position']-1)]
    tail = raw[cursor:]
    assert count == 2*item['objective_comparison_budget']
    assert all(r['point'] == points[i % 2] for i, r in enumerate(tail))
    values = np.array([r['raw'][0] for r in tail]).reshape(-1, 2)
    delta = values[:, 0] - values[:, 1]
    comparison = verification['objective_comparison']
    # A wrong block boundary or pair must stop this descriptive extraction.
    np.testing.assert_allclose([delta.mean(), delta.std(ddof=1)],
        [comparison['sample_mean_difference'], comparison['sample_std_difference']], rtol=1e-10, atol=1e-12)
    return dict(status='existing_pair_confirmed', pair=[labels[tuple(p)] for p in points],
        verification_seed=item['verification_seed'], stream_tag=comparison['comparison_stream_tag'],
        raw_start=cursor, raw_count=count, paired_objective_values=values.tolist(),
        paired_differences=delta.tolist(), saved_upper_bound=comparison['one_sided_upper_confidence_bound'],
        saved_status=comparison['status'], raw_file=str(directory.relative_to(ROOT) / 'paid_verification_raw.json'))


def analyze_case(reference, directory, truth):
    item, saved = read(directory/'input.json'), read(directory/'verification.json')
    search = read(ROOT/item['source_case']/'search.json')
    verification = saved['verification']
    labels = {tuple(r['point']): f'initial:{i}' for i, r in reversed(list(enumerate(search['history'][:10])))}
    for row, label in zip(search['history'][10:], reference['queries']):
        labels[tuple(row['point'])] = label
    for row, ref in zip(item['frozen_shortlist'], reference['shortlist']):
        labels[tuple(row['point'])] = ref['policy']
    points = list(dict.fromkeys(tuple(r['point']) for r in search['history']))
    shortlist = [tuple(r['point']) for r in item['frozen_shortlist']]
    attempts = {r['candidate_index']: r for r in verification['attempts']}
    comparison = verification['objective_comparison']
    scheduled = [shortlist[0], shortlist[item['objective_incumbent_position']-1]]
    compared = scheduled if comparison.get('simulation_calls', 0) else []
    deployed = tuple(saved['deployed']) if verification['certified'] else None
    assert verification['certified'] == reference['certified']
    assert labels[tuple(saved['deployed'])] == reference['deployed']
    best = reference['best_feasible_initial_truth']
    observed = []
    for point in points:
        actual = truth[point]
        position = shortlist.index(point) if point in shortlist else None
        safety = attempts.get(position)
        is_deployed = point == deployed
        disposition = ('certified_deployment' if is_deployed else
            'not_shortlisted' if position is None else
            'unexposed_after_terminal_decision' if safety is None else
            'safety_failed' if not safety['certified'] else
            'paired_objective_not_selected' if point in compared else 'safety_passed_without_comparison')
        diff = actual['quality']-truth[deployed]['quality'] if deployed is not None else None
        observed.append(dict(policy=labels[point], true_feasible=actual['true_feasible'],
            good_new=None if best is None else point not in {tuple(r['point']) for r in search['history'][:10]}
                and actual['true_feasible'] and actual['objective'] < best['objective']-1e-12,
            shortlisted_position=None if position is None else position+1,
            safety_attempted=safety is not None, safety_passed=None if safety is None else safety['certified'],
            objective_comparison_participant=point in compared, disposition=disposition,
            quality_difference_to_certified_deployment=diff,
            better_than_certified_deployment=None if diff is None else actual['true_feasible'] and diff > 1e-12))
    row = dict(**{k: reference[k] for k in ('replicate','domain','seed','arm','history_arm','terminal_backend')},
        status='complete', baseline_defined=reference['baseline_defined'],
        G=reference['best_initial_quality_increment'], certified=verification['certified'],
        reported_deployed=reference['deployed'], certified_deployment=None if deployed is None else labels[deployed],
        shortlist=[labels[p] for p in shortlist],
        safety_attempts=[dict(policy=labels[shortlist[r['candidate_index']]], index=r['candidate_index'],
            certified=r['certified'], upper_margin=r['upper_margin'], budget=r['verification_budget'])
            for r in verification['attempts']],
        scheduled_objective_pair=[labels[p] for p in scheduled],
        actual_objective_pair=[labels[p] for p in compared] or None,
        objective_comparison={k:v for k,v in comparison.items() if 'fingerprint' not in k},
        stop_reason=stop_reason(verification), observed=observed,
        physical_verification_calls=saved['new_verification_calls'],
        logical_verification_calls=verification['verification_budget'],
        maximum_terminal_calls=verification['max_verification_budget'],
        unused_terminal_allowance=verification['max_verification_budget']-verification['verification_budget'],
        safety_calls=verification['safety_verification_budget'],
        objective_comparison_calls=verification['objective_comparison_budget'],
        logical_search_calls=item['search_evaluation_count'],
        files=dict(input=str((directory/'input.json').relative_to(ROOT)),
                   verification=str((directory/'verification.json').relative_to(ROOT))))
    if reference['seed'] == 2026103016:
        row['paid_raw_objective_audit'] = raw_comparison(directory, item, verification, labels)
    return row


def run(plan):
    output = ROOT/plan['output_dir']
    with attempt(output/'comparison_exposure', 'saved-terminal-objective-comparison-exposure') as cost:
        cost.update(new_target_initial_calls=0, new_source_fits=0, new_truth_function_calls=0,
            engine_restorations=0, posterior_restorations=0, raw_conditioning=0, raw_replays=0,
            new_rng_draws=0, download_bytes=0)
        summary = read(ROOT/plan['inputs']['target_summary'])
        parent = read(ROOT/plan['parent_manifest'])
        source = read(ROOT/parent['source_manifest'])
        prepared = read(ROOT/source['output_dir']/'prepared.json')
        paths = {(r['replicate'], f['domain']): ROOT/read(ROOT/f['terminal_manifest'])['output_dir']
                 for r in prepared['replicates'] for f in r['families']}
        truth_paths = {(u['replicate'], u['seed']): u['cached_truth_file'] for u in parent['units']}
        truth = {}
        cases = []
        for reference in summary['references']['cases']:
            key = reference['replicate'], reference['seed']
            if key not in truth:
                truth[key] = {tuple(r['point']):r['truth'] for r in read(ROOT/truth_paths[key])}
            directory = paths[reference['replicate'], reference['domain']]/f"seed_{reference['seed']}"/reference['arm']/PROTOCOL
            cases.append(analyze_case(reference, directory, truth[key]))
        assert len(cases)+len(summary['references']['missing_cells']) == summary['references']['registered_cells'] == 216
        arms = []
        for arm in dict.fromkeys(r['arm'] for r in cases):
            rows = [r for r in cases if r['arm']==arm]
            policies = [p for r in rows for p in r['observed']]
            better = [p for p in policies if p['better_than_certified_deployment'] is True]
            arms.append(dict(arm=arm, registered=24, completed=len(rows), missing=24-len(rows),
                baseline_undefined=sum(not r['baseline_defined'] for r in rows),
                actual_objective_comparisons=sum(r['actual_objective_pair'] is not None for r in rows),
                logical_terminal_calls=sum(r['logical_verification_calls'] for r in rows),
                physical_terminal_calls=sum(r['physical_verification_calls'] for r in rows),
                unused_terminal_allowance=sum(r['unused_terminal_allowance'] for r in rows),
                stop_reasons=dict(Counter(r['stop_reason'] for r in rows)),
                better_policy_cells=len(better), better_dispositions=dict(Counter(p['disposition'] for p in better)),
                better_without_objective_comparison=sum(not p['objective_comparison_participant'] for p in better),
                policy_comparison_unknown_without_certified_deployment=sum(p['better_than_certified_deployment'] is None for p in policies)))
        focal = [r for r in cases if r['seed']==2026103016]
        mechanism = read(ROOT/parent['output_dir']/'terminal_mechanism_analysis.json')
        result = dict(status='complete', registered_cells=216, completed_cells=len(cases),
            cases=cases, missing_cells=summary['references']['missing_cells'], arms=arms,
            queue3016_direct_18_vs_53=[r['arm'] for r in focal
                if set(r['actual_objective_pair'] or [])=={'neighbor:18','neighbor:53'}],
            role_native_crosses=mechanism['role_native_same_history'],
            prior_failures=summary['references']['execution_failures'],
            limits=['Observed-policy cells and terminal crosses are not independent tasks.',
                'An unexposed policy has no counterfactual safety/comparison certificate.',
                'Raw extraction only confirms each already paid pair; no blocks are combined into another certificate.',
                'Quality advantage is undefined without an actual certified deployment; best-initial labels remain unknown for Queue3009.',
                'The all-policy comparison uses cached target truth retrospectively, not for selecting a new terminal rule.'])
        save(output/'comparison_exposure.json', result)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(MANIFEST))
    run(read(parser.parse_args().manifest))

"""Trace good new policies through saved original or modified terminal decisions."""
import argparse
from collections import Counter

import numpy as np

from performance.diagnose_mainline_terminal_omission import (
    raw_summary, role_eligibility, snapshot_evidence,
)
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.run_mainline_terminal_equal import cases as terminal_cases

PARENT = ROOT / 'performance/manifests/mainline_three_family_v1.json'
OUTPUT = ROOT / 'paper_artifacts/mainline_three_family_funnel_v1'
FIELDS = ('queried', 'shortlisted', 'safety_attempted', 'safety_passed', 'deployed')


def funnel_summary(group, registered):
    known = [row for row in group if row['baseline_defined']]
    observed = {key: sum(row['funnel'][key] for row in known) for key in FIELDS}
    return dict(tasks=len(group), registered_tasks=registered,
        missing_terminal_tasks=registered-len(group), known_baseline_tasks=len(known),
        baseline_undefined_tasks=len(group)-len(known), known_baseline_funnel=observed,
        funnel=observed if len(known) == registered else dict.fromkeys(FIELDS))


def analyze(parent_path=PARENT, variant='original', terminal_protocol='original', result_rows=None):
    cases, missing_scores = [], []
    registered_by_family = {}
    parent = parent_path if isinstance(parent_path, dict) else read(parent_path)
    baseline_file = ROOT / parent['output_dir'] / 'initialization_baseline.json'
    baseline = ({(r['domain'], r['seed'], r['arm']): r for r in
        read(baseline_file)['rows']} if baseline_file.is_file() else {})
    modified = (read(ROOT / parent['modified_terminal_manifest'])
                if 'modified_terminal_manifest' in parent else None)
    if variant == 'modified' and modified is None:
        raise ValueError('Modified terminal requires its registered manifest')
    result_root = ROOT / (modified['output_dir'] if variant == 'modified' else parent['output_dir'])
    if result_rows is None:
        result_rows = {(r['domain'], r['seed'], r['arm']): r for r in
                       read(result_root / 'results.json')['rows']}
    for family in parent['families']:
        plan = read(ROOT / family['terminal_manifest'])
        registered_by_family[family['domain']] = len(plan['target_seeds'])
        source, terminal_root = ROOT / plan['source_output_dir'], ROOT / plan['output_dir']
        if variant == 'modified':
            terminal_root = ROOT / modified['output_dir'] / family['domain']
        completed = {(seed, arm) for seed, arm, _ in terminal_cases(plan)}
        for seed in plan['target_seeds']:
            task = source / f'seed_{seed}'
            truth = {tuple(r['point']): r['truth'] for r in read(task / 'truth.json')}
            neighbors = {tuple(p): i for i, p in enumerate(np.load(task / 'neighbors.npz')['points'])}
            searches = {arm['history_arm']: read(task / arm['history_arm'] / 'search.json')
                        for arm in plan['arms'] if (seed, arm['name']) in completed}
            initial = [tuple(r['point']) for r in next(iter(searches.values()))['history'][:10]]
            best = min((p for p in initial if truth[p]['true_feasible']),
                       key=lambda p: truth[p]['objective'], default=None)

            def label(point):
                if point in initial:
                    return 'initial:' + str(initial.index(point))
                if point in neighbors:
                    return 'neighbor:' + str(neighbors[point])
                return next(arm + ':history:' + str(i) for arm, search in searches.items()
                            for i, row in enumerate(search['history']) if tuple(row['point']) == point)

            def good(point):
                return None if best is None else (point not in initial and truth[point]['true_feasible'] and
                        truth[point]['objective'] < truth[best]['objective'] - 1e-12)

            for cell in plan['arms']:
                arm, history_arm = cell['name'], cell['history_arm']
                if (seed, arm) not in completed:
                    continue
                directory, search = task / history_arm, searches[history_arm]
                native = (read(directory / 'native_terminal.json')
                          if cell['terminal_backend'] == 'native' else None)
                frozen = (native or search)['terminal']
                terminal_directory = terminal_root / f'seed_{seed}' / arm / terminal_protocol
                item = read(terminal_directory / 'input.json')
                saved = read(terminal_directory / 'verification.json')
                verification = saved['verification']
                attempts = {r['candidate_index'] + 1: r for r in verification['attempts']}
                deployed = tuple(saved['deployed'])
                certified_feasible = verification['certified'] and truth[deployed]['true_feasible']
                shortlist, by_key = [], {}
                for entry in item['frozen_shortlist']:
                    point, position = tuple(entry['point']), entry['shortlist_position']
                    certificate = attempts.get(position)
                    row = dict(policy=label(point), position=position,
                        posterior_rank=entry.get('posterior_rank'),
                        role=entry['shortlist_role'], candidate_universe=entry.get('candidate_universe'),
                        posterior_objective=entry.get('selected_posterior_objective'),
                        violation_probability=entry.get('selected_posterior_violation_probability',
                            entry.get('selected_probability_violation')),
                        good_new=good(point), true_objective=truth[point]['objective'],
                        true_feasible=truth[point]['true_feasible'],
                        safety_attempted=certificate is not None,
                        safety_passed=certificate['certified'] if certificate else None,
                        upper_margin=certificate['upper_margin'] if certificate else None,
                        safety_budget=certificate['verification_budget'] if certificate else None,
                        deployed=point == deployed and verification['certified'])
                    shortlist.append(row); by_key[point] = row
                queried = list(dict.fromkeys(tuple(r['point']) for r in search['history'][10:]))
                policies = []
                native_records = ({tuple(r['point']): r for r in native['observed']} if native else {})
                for point in filter(good, queried):
                    entry = by_key.get(point)
                    if entry is None:
                        disposition = 'not_on_frozen_shortlist'
                    elif not entry['safety_attempted']:
                        disposition = 'not_verified_after_earlier_terminal_decision'
                    elif not entry['safety_passed']:
                        disposition = 'safety_certificate_failed'
                    elif entry['deployed']:
                        disposition = 'deployed'
                    else:
                        disposition = 'objective_guard_retained_incumbent'
                    final = (dict(backend='native', **{k: v for k, v in native_records[point].items()
                                                     if k != 'point'}) if native else
                             dict(backend=cell['terminal_backend'], **role_eligibility(point, frozen)))
                    if not native and entry:
                        final.update(probability_violation=entry['violation_probability'],
                                     final_objective_mean=entry['posterior_objective'])
                    if not native:
                        missing_scores.append(dict(domain=family['domain'], seed=seed, arm=arm,
                            policy=label(point), disposition=disposition,
                            final_rank=final['posterior_rank'], eligibility=final['eligible']))
                    policies.append(dict(policy=label(point),
                        candidate_origin='shared_neighbor' if point in neighbors else 'dynamic',
                        query_history_indices=[i for i, r in enumerate(search['history'])
                                               if tuple(r['point']) == point],
                        disposition=disposition, shortlist_position=entry['position'] if entry else None,
                        truth=truth[point], quality_gain_over_best_initial=truth[point]['quality']-truth[best]['quality'],
                        quality_difference_to_certified_deployment=(truth[point]['quality']-truth[deployed]['quality']
                                                                   if certified_feasible else None),
                        better_than_certified_deployment=(truth[point]['quality'] > truth[deployed]['quality']+1e-12
                                                          if certified_feasible else None),
                        raw=raw_summary(search['history'], point, truth[point]), final=final,
                        saved_prequery=snapshot_evidence(directory, point, best, truth[point])))
                funnel = dict.fromkeys(FIELDS) if best is None else dict(queried=len(policies),
                    shortlisted=sum(r['shortlist_position'] is not None for r in policies),
                    safety_attempted=sum(r['good_new'] and r['safety_attempted'] for r in shortlist),
                    safety_passed=sum(r['good_new'] and r['safety_passed'] is True for r in shortlist),
                    deployed=sum(r['disposition'] == 'deployed' for r in policies))
                old = baseline.get((family['domain'], seed, arm))
                # These comparisons detect a mismatched physical policy or best-initial definition.
                if old is not None and best is not None:
                    assert funnel['queried'] == old['good_new_queries']
                    assert funnel['shortlisted'] == old['good_new_queries_shortlisted']
                scored = result_rows[family['domain'], seed, arm]
                if 'certified_new_policy_beats_best_initial' in scored:
                    expected = scored['certified_new_policy_beats_best_initial']
                else:
                    expected = (certified_feasible and deployed not in initial and good(deployed))
                assert funnel['deployed'] == (None if best is None else int(expected))
                assert list(deployed) == scored['deployed'] and verification['certified'] == scored['certified']
                cases.append(dict(domain=family['domain'], seed=seed, arm=arm,
                    terminal_variant=variant, input_changed=item.get('input_changed', False),
                    history_arm=history_arm, terminal_backend=cell['terminal_backend'],
                    baseline_defined=best is not None,
                    baseline_unavailable_reason=None if best is not None else 'no_true_feasible_initial_policy',
                    best_feasible_initial=None if best is None else label(best),
                    best_feasible_initial_truth=None if best is None else truth[best],
                    queries=[label(tuple(r['point'])) for r in search['history'][10:]],
                    primary=label(tuple(item['primary'])), deployed=label(deployed),
                    deployed_truth=truth[deployed], certified=verification['certified'],
                    objective_challenger=frozen['objective_challenger'],
                    objective_comparison=verification['objective_comparison'],
                    verification_status=verification['status'],
                    safety_attempts=[{key: row[key] for key in (
                        'candidate_index', 'certified', 'upper_margin', 'verification_budget',
                        'sample_mean', 'sample_std')} for row in verification['attempts']],
                    verification_calls=saved['new_verification_calls'],
                    logical_verification_calls=verification['verification_budget'],
                    actual_safe_reward=scored['actual_safe_reward'],
                    certified_feasible_quality_gain=scored['certified_feasible_quality_gain'],
                    certified_initial_quality_component=scored['certified_initial_quality_component'],
                    shortlist=shortlist, good_new_policies=None if best is None else policies, funnel=funnel))
    arms = []
    for arm in dict.fromkeys(r['arm'] for r in cases):
        group = [r for r in cases if r['arm'] == arm]
        rows = [p for r in group for p in (r['good_new_policies'] or [])]
        families = []
        for family in parent['families']:
            domain = family['domain']
            summary = funnel_summary([row for row in group if row['domain'] == domain],
                                     registered_by_family[domain])
            families.append(dict(domain=domain, **summary, **summary['funnel']))
        arms.append(dict(arm=arm, **funnel_summary(group, len(parent['target_seeds'])),
            origins=dict(Counter(r['candidate_origin'] for r in rows)),
            dispositions=dict(Counter(r['disposition'] for r in rows)),
            better_than_actual_deployment_by_disposition=dict(Counter(r['disposition'] for r in rows
                if r['better_than_certified_deployment'] is True)),
            unselected_not_better_than_actual_deployment=sum(r['disposition'] != 'deployed' and
                r['better_than_certified_deployment'] is False for r in rows),
            no_certified_deployment=sum(r['better_than_certified_deployment'] is None for r in rows),
            by_family=families))
    if modified is not None:
        scored = read(ROOT / modified['output_dir'] / 'results.json')['best_initial_deployments'][
            'baseline' if variant == 'original' else 'modified']
        expected = {r['arm']: r for r in scored}
        for row in arms:
            assert row['tasks'] == expected[row['arm']]['tasks']
            assert row['known_baseline_funnel']['deployed'] == expected[row['arm']]['certified_new_policy_beats_best_initial']
    return dict(status='complete', task_count=len(parent['target_seeds']), terminal_cells=len(cases),
        terminal_variant=variant, execution_failures=parent.get('execution_failures', []),
        registered_terminal_cells=len(parent['target_seeds'])*len(plan['arms']),
        failed_terminal_cells=len(parent['target_seeds'])*len(plan['arms'])-len(cases),
        cases=cases, arms=arms,
        scope='Saved JSON/NPZ and cached truth; no posterior restoration or new outcomes.',
        definition='Unique task-policy outside shared initial10; true feasible and objective below best true-feasible initial by more than 1e-12.',
        missing_final_scores=missing_scores,
        limitations=['Only crossed native terminals saved complete final observed-policy scores. Original terminals retain selected scores/ranks and eligible counts, not complete paid13 scores.',
            'Without a true-feasible initial policy, best-initial good-policy labels and quality funnels are undefined. Known-baseline counts do not replace the registered-cohort denominator.',
            'Saved prefix10/11/12 predictions precede the last observation; they cannot identify missing paid13 risk/noise/uncertainty contributions.',
            'A shortlisted but unexposed good policy has no counterfactual safety or objective certificate.',
            'Terminal crosses reuse the same query history and are paired terminal interventions, not independent discoveries.'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest')
    args = parser.parse_args()
    plan = read(ROOT / args.manifest) if args.manifest else None
    output = ROOT / plan['output_dir'] if plan else OUTPUT
    parent = ROOT / plan['source_manifest'] if plan else PARENT
    for variant in ('original', 'modified') if plan else ('original',):
        with attempt(output, 'saved-' + variant + '-terminal-funnel-audit') as cost:
            cost.update(new_source_fits=0, new_target_initial_calls=0,
                        new_truth_function_calls=0, posterior_restorations=0, download_bytes=0)
            result = analyze(parent, variant)
            save(output / (variant + '_terminal.json' if plan else 'terminal.json'), result)
        print(variant, {r['arm']: r['funnel'] for r in result['arms']}, flush=True)


if __name__ == '__main__':
    main()

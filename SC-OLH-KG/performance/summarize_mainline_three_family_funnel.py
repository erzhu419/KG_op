"""Join saved query/terminal audits without restoring models or evaluating truth."""
from datetime import datetime, timezone
import argparse
import json

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save

MANIFEST = ROOT / 'performance/manifests/mainline_three_family_funnel_v1.json'
FULL = 'legacy_role_full'
AFTER = 'legacy_role_after_init_point_only'
POINT = 'legacy_role_point_only'


def decision_chains(plan, query, terminals):
    """Link saved selections to actual certificates under both frozen terminals."""
    source = read(ROOT / plan['source_manifest'])
    scored = {
        'original': read(ROOT / source['output_dir'] / 'results.json'),
        'modified': read(ROOT / read(ROOT / source['modified_terminal_manifest'])['output_dir'] / 'results.json'),
    }
    scores = {variant: {(r['seed'], r['arm']): r for r in data['rows']}
              for variant, data in scored.items()}
    cells = {variant: {(r['seed'], r['arm']): r for r in data['cases']}
             for variant, data in terminals.items()}
    rows = []
    for task in query['tasks']:
        seed = task['seed']; best = task['best_initial_truth']['quality']
        arms = []
        for arm in (FULL, AFTER, POINT):
            case = cells['modified'][seed, arm]
            states = [r for r in query['states'] if r['seed'] == seed and r['arm'] == arm]
            selections = [dict(paid_prefix=state['paid_prefix'], policy=policy,
                selection_mode=state['choices']['actual']['mode'],
                **state['choices']['actual']['selected'])
                for state, policy in zip(states, case['queries'], strict=True)]
            outcomes = {}
            for variant in terminals:
                cell, score = cells[variant][seed, arm], scores[variant][seed, arm]
                feasible = score['certified_true_feasible'] is True
                reference = best if feasible else 0.
                improvement = score['deployed_truth']['quality'] - best if feasible else 0.
                assert abs(score['actual_safe_reward'] - reference - improvement) <= 1e-12
                outcomes[variant] = dict(deployed=cell['deployed'], certified=cell['certified'],
                    funnel=cell['funnel'], branch=cell['objective_comparison']['status'],
                    actual_safe_reward=score['actual_safe_reward'],
                    first_initial_reference=score['certified_initial_quality_component'],
                    first_initial_quality_component=score['certified_feasible_quality_gain'],
                    best_initial_reference_component=reference,
                    improvement_over_best_initial=improvement,
                    good_new_policies=cell['good_new_policies'])
            arms.append(dict(arm=arm, queries=selections, outcomes=outcomes))
        by_arm = {r['arm']: r for r in arms}
        comparisons = []
        for left, right in ((AFTER, POINT), (FULL, AFTER)):
            first = next((r for r in query['paired_states'] if r['seed'] == seed
                          and r['left'] == left and r['right'] == right and r['first_divergence']), None)
            differences = {}
            for variant in terminals:
                a, b = (by_arm[arm]['outcomes'][variant] for arm in (left, right))
                differences[variant] = {key: a[key] - b[key] for key in (
                    'actual_safe_reward', 'first_initial_reference', 'first_initial_quality_component',
                    'best_initial_reference_component', 'improvement_over_best_initial')}
                differences[variant]['jointly_certified'] = a['certified'] and b['certified']
            comparisons.append(dict(left=left, right=right, first_divergence=first,
                                    outcome_differences=differences))
        rows.append(dict(domain=task['domain'], seed=seed, arms=arms, comparisons=comparisons,
                         best_initial_truth=task['best_initial_truth']))
    cohorts = []
    for left, right in ((AFTER, POINT), (FULL, AFTER)):
        selected = [next(c for c in r['comparisons'] if c['left'] == left and c['right'] == right)
                    for r in rows]
        cohorts.append(dict(left=left, right=right, task_count=len(rows),
            variants={variant: {key: sum(c['outcome_differences'][variant][key] for c in selected)/len(rows)
                for key in ('actual_safe_reward', 'first_initial_reference', 'first_initial_quality_component',
                            'best_initial_reference_component', 'improvement_over_best_initial')}
                for variant in terminals}))
    return dict(tasks=rows, comparisons=cohorts,
        focus_seeds=plan['focus_seeds'],
        definition=('Reward is exactly decomposed into certification-gated best-feasible-initial '
            'quality and improvement over it. This is a descriptive reference, not a counterfactual '
            'certificate for that initial policy. Query swaps hold saved draws and pools fixed.'))


def main(manifest=MANIFEST):
    plan = read(manifest)
    output = ROOT / plan['output_dir']
    with attempt(output, 'aggregate-saved-three-family-funnel') as cost:
        cost.update(new_target_initial_calls=0, new_source_fits=0,
                    new_truth_function_calls=0, posterior_restorations=0,
                    download_bytes=0)
        query = read(output / 'query_diagnosis.json')
        terminal = read(output / plan.get('terminal_file', 'terminal.json'))
        assert query['status'] == terminal['status'] == 'complete'
        paired = []
        for left, right in ((FULL, AFTER), (AFTER, POINT)):
            cases = [r for r in query['paired_states'] if r['left'] == left
                     and r['right'] == right and r['first_divergence']]
            names = list(dict.fromkeys(s['substitution'] for r in cases
                                       for s in r['substitutions']))
            paired.append(dict(left=left, right=right, first_divergence_count=len(cases),
                seeds=[r['seed'] for r in cases],
                all_same_history=all(r['same_paid_history'] for r in cases),
                all_same_pool=all(r['same_physical_pool'] for r in cases),
                swaps=[dict(receiver=receiver, substitution=name,
                    donor_choice_recovered=sum(s['donor_choice_recovered']
                        for r in cases for s in r['substitutions']
                        if s['receiver'] == receiver and s['substitution'] == name))
                    for receiver in (left, right) for name in names]))

        source = read(ROOT / plan['source_manifest'])
        history_comparisons = []
        for family in source['families']:
            search = read(ROOT / family['search_manifest'])
            for seed in search['target_seeds']:
                directory = ROOT / search['output_dir'] / f'seed_{seed}'
                histories = {arm: read(directory / arm / 'search.json')['history']
                             for arm in (FULL, AFTER)}
                cells = {arm: next(c for c in terminal['cases']
                         if c['seed'] == seed and c['arm'] == arm) for arm in (FULL, AFTER)}
                history_comparisons.append(dict(domain=family['domain'], seed=seed,
                    identical_full_paid_history=histories[FULL] == histories[AFTER],
                    same_deployed_policy=cells[FULL]['deployed'] == cells[AFTER]['deployed'],
                    full_deployed=cells[FULL]['deployed'], after_deployed=cells[AFTER]['deployed']))

        opportunity = [dict(domain=t['domain'], seed=t['seed'],
            shared_new_good=len(t['shared_new_good'])) for t in query['tasks']]
        lost = []
        for cell in terminal['cases']:
            for policy in cell['good_new_policies']:
                if policy['disposition'] == 'deployed':
                    continue
                actual = cell['deployed_truth']
                difference = (policy['truth']['quality'] - actual['quality']
                    if cell['certified'] and actual['true_feasible'] else None)
                lost.append(dict(domain=cell['domain'], seed=cell['seed'], arm=cell['arm'],
                    policy=policy['policy'], disposition=policy['disposition'],
                    shortlist_position=policy['shortlist_position'],
                    eligible=policy['final'].get('eligible'),
                    eligible_count=policy['final'].get('eligible_count'),
                    posterior_rank=policy['final'].get('posterior_rank'),
                    terminal_status=cell['objective_comparison']['status'],
                    quality_difference_vs_certified_deployment=difference,
                    improves_certified_deployment=None if difference is None else difference > 1e-12))
        result = dict(status='complete', task_count=len(query['tasks']),
            query_states=len(query['states']), terminal_cells=terminal['terminal_cells'],
            reconstruction=query['reconstruction'], first_divergences=paired,
            shared_opportunities=opportunity,
            shared_opportunities_by_family={domain: sum(r['shared_new_good']
                for r in opportunity if r['domain'] == domain)
                for domain in dict.fromkeys(r['domain'] for r in opportunity)},
            missing_shared_good_exposures=sum(len(s['shared_good_absent']) for s in query['states']),
            unknown_dynamic_truth_state_exposures=sum(s['unknown_truth_action_count']
                                                       for s in query['states']),
            terminal_funnels=terminal['arms'], unselected_good_policies=lost,
            full_after_history_comparisons=history_comparisons,
            scope=plan['scope'], limits=plan['limits'])
        if 'terminal_variants' in plan:
            terminals = {variant: read(output / f'{variant}_terminal.json')
                         for variant in plan['terminal_variants']}
            assert sum(t['terminal_cells'] for t in terminals.values()) == plan['available_terminal_cells']
            result.update(terminal_cells_by_variant={key: value['terminal_cells'] for key, value in terminals.items()},
                terminal_funnels_by_variant={key: value['arms'] for key, value in terminals.items()},
                registered_terminal_cells=plan['registered_terminal_cells'],
                execution_failures=plan['execution_failures'],
                propagation_to_deployment=decision_chains(plan, query, terminals))
        save(output / 'summary.json', result)
    attempts = [json.loads(line) for line in (output / 'attempts.jsonl').read_text().splitlines()]
    fields = ('cpu_seconds', 'wall_seconds', 'new_source_calls', 'new_target_initial_calls',
              'new_target_search_calls', 'new_verification_calls', 'new_source_fits',
              'new_truth_function_calls', 'posterior_restorations', 'source_model_restorations',
              'raw_replays', 'download_bytes')
    save(output / 'completion.json', dict(status='complete',
        cost={key: sum(row.get(key, 0) for row in attempts) for key in fields},
        failed_attempts=[row for row in attempts if row['status'] != 'completed'],
        attempts=len(attempts), completed_utc=datetime.now(timezone.utc).isoformat()))
    plan['status'] = 'complete'
    save(manifest, plan)
    print(dict(status='complete', queries=len(query['states']), terminals=terminal['terminal_cells']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(MANIFEST))
    main(parser.parse_args().manifest)

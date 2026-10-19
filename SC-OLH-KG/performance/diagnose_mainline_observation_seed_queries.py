"""Attribute registered frozen task/observation streams using saved arrays only."""
import argparse
from collections import Counter
import json

from performance.diagnose_mainline_three_family_queries import (
    AFTER, ARMS, FULL, POINT, summarize as summarize_replica)
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save

MANIFEST = ROOT/'performance/manifests/mainline_observation_seed_attribution_v1.json'
STAGE = 'saved-observation-query-attribution'
RISK_SWAP = 'risk_risk_mean+noise_margin+risk_draw'


def counts(states, pairs):
    registered = {(r['replicate'], r['domain'], r['seed']) for r in states}
    comparisons = []
    for left, right in ((AFTER, POINT), (FULL, AFTER)):
        selected = [p for p in pairs if (p['left'], p['right']) == (left, right)]
        first = [p for p in selected if p['first_divergence']]
        causal = [p for p in first if p['same_paid_history'] and p['same_physical_pool']]
        swaps = []
        for receiver, donor in ((left, right), (right, left)):
            matching = [{s['substitution']: s for s in p['substitutions']
                         if s['receiver'] == receiver and s['donor'] == donor}
                        for p in causal]
            risk = [s[RISK_SWAP]['donor_choice_recovered'] for s in matching]
            objective = [s['all_objective']['donor_choice_recovered'] for s in matching]
            swaps.append(dict(receiver=receiver, donor=donor, first_divergences=len(causal),
                all_risk_recovers_donor=sum(risk), all_objective_recovers_donor=sum(objective),
                component_recovers_donor={mode:sum(s[mode]['donor_choice_recovered'] for s in matching)
                    for mode in ('risk_risk_mean', 'risk_noise_margin', 'risk_risk_draw',
                                 'objective_mean', 'objective_draw')},
                risk_only=sum(r and not o for r, o in zip(risk, objective)),
                objective_only=sum(o and not r for r, o in zip(risk, objective)),
                both_alone_recovers=sum(r and o for r, o in zip(risk, objective)),
                neither_alone_recovers=sum(not r and not o for r, o in zip(risk, objective))))
        comparisons.append(dict(left=left, right=right,
            registered_task_noise_units=len(registered),
            unavailable_task_noise_units=[dict(replicate=r, domain=d, seed=s) for r,d,s in sorted(
                registered-{(p['replicate'], p['domain'], p['seed']) for p in selected})],
            task_noise_units=len({(p['replicate'], p['domain'], p['seed']) for p in selected}),
            paired_states=len(selected), changed_queries=sum(p['query_changed'] for p in selected),
            first_divergences=len(first), first_divergence_by_prefix=dict(Counter(
                p['paid_prefix'] for p in first)),
            same_history_and_pool_first_divergences=len(causal),
            same_history_states=sum(p['same_paid_history'] for p in selected),
            same_pool_states=sum(p['same_physical_pool'] for p in selected), swaps=swaps))
    choices = []
    for arm in ARMS:
        rows = [r for r in states if r['arm'] == arm]
        actual = [r['choices']['actual']['selected'] for r in rows]
        good = [p for r in rows for p in (r['good_candidates'] or [])]
        choices.append(dict(arm=arm, recorded_choices=len(rows),
            registered_task_noise_units=len(registered),
            available_task_noise_units=len({(r['replicate'], r['domain'], r['seed']) for r in rows}),
            actual_selected_good=sum(p['new_good'] is True for p in actual),
            actual_selected_unknown_good=sum(p['new_good'] is None for p in actual),
            quality_baseline_undefined_states=sum(not r['quality_baseline_defined'] for r in rows),
            actual_selected_unknown_truth=sum(p['truth'] is None for p in actual),
            good_candidate_state_exposures=len(good),
            reasons=dict(Counter(p['actual_selection_reason'] for p in good))))
    return dict(comparisons=comparisons, choices=choices)


def summarize(plan):
    prepared = read(ROOT/plan['prepared_file'])
    tasks, states, pairs, summaries, errors = [], [], [], [], []
    per_replica = []
    for replica in prepared['replicates']:
        result = summarize_replica(replica)
        for field in ('tasks', 'states', 'paired_states', 'summary'):
            for row in result[field]:
                row['replicate'] = replica['replicate']
        # The reused helper's membership test is binary; uncached truth remains unknown.
        for row in result['states']:
            for choice in row['choices'].values():
                if choice['selected']['truth'] is None:
                    choice['selected']['new_good'] = None
        for row in result['paired_states']:
            choices = [*row['selected'].values(),
                       *(s['selected'] for s in row['substitutions']),
                       *(s['policy'] for s in row.get('selected_pair_components', []))]
            for choice in choices:
                if choice['truth'] is None:
                    choice['new_good'] = None
        tasks.extend(result['tasks']); states.extend(result['states'])
        pairs.extend(result['paired_states']); summaries.extend(result['summary'])
        errors.append(result['reconstruction']['max_score_error'])
        per_replica.append(dict(replicate=replica['replicate'],
            **counts(result['states'], result['paired_states'])))
    focus_neighbors = plan.get('focus_neighbors', plan.get('focal_queries', {}))
    focus = set(plan.get('focus_seeds', map(int, focus_neighbors)))
    focal = [dict(replicate=r['replicate'], domain=r['domain'], seed=r['seed'],
        arm=r['arm'], paid_prefix=r['paid_prefix'], actual=r['choices']['actual'],
        tracked_good_candidates=None if r['good_candidates'] is None else
            [p for p in r['good_candidates'] if str(r['seed']) not in focus_neighbors or
             p['neighbor_index'] in focus_neighbors[str(r['seed'])]])
        for r in states if r['seed'] in focus]
    unavailable = [dict(replicate=t['replicate'], domain=t['domain'], seed=t['seed'], arm=arm,
        reason='Registered search execution failure; saved complete query trajectory unavailable')
        for t in tasks for arm in t['unavailable_arms']]
    assert len(states) == plan.get('expected_recorded_choices', len(states))
    return dict(status='complete', tasks=tasks, states=states, paired_states=pairs,
        summary=summaries, by_replicate=per_replica, combined=counts(states, pairs),
        focal_queries=focal, reconstruction=dict(recorded_choices=len(states),
            max_score_error=max(errors)),
        unavailable_query_histories=unavailable,
        unavailable_final_score_states=[dict(replicate=t['replicate'], domain=t['domain'],
            seed=t['seed'], arm=arm, paid_prefix=13,
            reason='No saved query scores after the final paid observation; no posterior restoration')
            for t in tasks for arm in ARMS if arm not in t['unavailable_arms']],
        limits=(f'{len(tasks)} registered task-observation units; geometry and observation identities stay separate. '
            f'All {len(states)} saved choices are reconstructed unchanged. Registered failed histories remain missing. '
            'Absent feasible-initial baselines leave good-policy labels undefined. Good means outside initial10 and '
            'better than its best true-feasible member. Unknown dynamic-policy truth remains null; '
            'candidate-state exposures repeat policies. Bidirectional component swaps are fixed-state '
            'diagnoses: only same-history, same-physical-pool first divergences isolate the propagation '
            'contrast. Later differing histories are descriptive, and saved risk draw is the realized '
            'TS innovation. No new trajectory, certification or truth is inferred. Final paid13 '
            'query scores were not saved and remain unavailable.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(MANIFEST))
    plan = read(parser.parse_args().manifest)
    output = ROOT/plan['output_dir']
    with attempt(output, STAGE) as cost:
        cost.update(new_target_initial_calls=0, new_source_fits=0,
            new_truth_function_calls=0, posterior_restorations=0, download_bytes=0,
            new_rng_draws=0, raw_replays=0)
        result = summarize(plan)
        save(output/'query_diagnosis.json', result)
    attempts = [r for line in (output/'attempts.jsonl').read_text().splitlines()
                if (r := json.loads(line))['stage'] == STAGE]
    fields = ('cpu_seconds', 'wall_seconds', 'new_source_calls', 'new_target_initial_calls',
        'new_target_search_calls', 'new_verification_calls', 'new_source_fits',
        'new_truth_function_calls', 'source_model_restorations', 'posterior_restorations',
        'download_bytes', 'new_rng_draws', 'raw_replays')
    save(output/'query_completion.json', dict(status='complete',
        reconstruction=result['reconstruction'], attempts=len(attempts),
        cost={k:sum(r.get(k, 0) for r in attempts) for k in fields},
        failed_attempts=[r for r in attempts if r['status'] != 'completed']))
    print(json.dumps(dict(reconstruction=result['reconstruction'],
        by_replicate=result['by_replicate'], combined=result['combined'])))


if __name__ == '__main__':
    main()

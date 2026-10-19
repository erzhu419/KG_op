"""Summarize recovered candidate pools without model or simulator evaluation."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json

import numpy as np

from performance.run_mainline_candidate_pool_audit import MANIFEST, case_dir, cases
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save


def summarize(plan):
    output = ROOT / plan['output_dir']
    with attempt(output, 'summarize', plan['domain']) as cost:
        cost.update(new_truth_function_calls=0, new_source_fits=0, download_bytes=0)
        truth = [row['truth'] for row in read(output / 'truth.json')]
        points = np.load(output / 'truth_policies.npz')['points']
        assert len(points) == len(truth)
        global_ids = {tuple(map(int, point)): i for i, point in enumerate(points)}
        state = np.column_stack((points[:, 0] / 100.,
                                 np.mean(points[:, 1:] / 100., axis=1),
                                 np.std(points[:, 1:] / 100., axis=1)))
        rows = []
        initial_id = None
        unique_pools = {name: set() for name in ('generated', 'action', 'terminal_union')}

        def coverage(ids, observed, initial):
            ids = list(dict.fromkeys(ids))
            feasible = [i for i in ids if truth[i]['true_feasible']]
            improving = [i for i in feasible if truth[i]['objective'] < initial - 1e-12]
            unobserved = [i for i in ids if i not in observed]
            best = min(feasible, key=lambda i: truth[i]['objective']) if feasible else None
            nearest = min(unobserved, key=lambda i: np.linalg.norm(state[i] - state[initial_id])) if unobserved else None
            return {
                'count': len(ids), 'unobserved_count': len(unobserved),
                'feasible_count': len(feasible), 'feasible_improving_count': len(improving),
                'unobserved_feasible_count': sum(i not in observed for i in feasible),
                'unobserved_feasible_improving_count': sum(i not in observed for i in improving),
                'best_objective': min((truth[i]['objective'] for i in ids), default=None),
                'best_feasible_bank_id': best,
                'best_feasible_objective': truth[best]['objective'] if best is not None else None,
                'objective_headroom_over_initial': max(0., initial - truth[best]['objective']) if best is not None else 0.,
                'quality_headroom_over_initial': max(0., (initial - truth[best]['objective']) / 1.5) if best is not None else 0.,
                'feasible_improving_bank_ids': improving,
                'nearest_unobserved_to_initial': None if nearest is None else {
                    'bank_id': nearest, 'policy_state': state[nearest].tolist(),
                    'state_distance': float(np.linalg.norm(state[nearest] - state[initial_id]))},
            }

        for seed, arm in cases(plan):
            directory = case_dir(output, seed, arm)
            recovery = read(directory / 'recovery.json')
            assert recovery['selected_policy_score_route_parity']
            local = np.load(directory / 'policies.npz')['points']
            mapping = [global_ids[tuple(map(int, point))] for point in local]
            first = mapping[recovery['history_ids'][0]]
            if initial_id is None:
                initial_id = first
            assert first == initial_id
            initial = truth[initial_id]['objective']

            for stage in recovery['stages']:
                assert stage['parity']
                convert = lambda name: [mapping[i] for i in stage[name]]
                generated, action = convert('generated_ids'), convert('action_ids')
                terminal = convert('terminal_union_ids')
                for name, pool in (('generated', generated), ('action', action), ('terminal_union', terminal)):
                    unique_pools[name].update(pool)
                observed = set(convert('observed_ids'))
                scores = stage['score']
                count = len(action)
                assert len(set(action)) == count
                total = np.asarray(scores['total'], dtype=float)
                ranks = np.empty(count, dtype=int)
                ranks[np.argsort(-total, kind='stable')] = np.arange(1, count + 1)
                action_positions = {bank_id: position for position, bank_id in enumerate(action)}

                def detail(bank_id):
                    position = action_positions.get(bank_id)
                    item = {'bank_id': bank_id, 'observed': bank_id in observed,
                            'policy_state': state[bank_id].tolist(), 'truth': truth[bank_id],
                            'in_action_pool': position is not None}
                    if position is not None:
                        posterior = {key: value[position] for key, value in scores.items()
                                     if isinstance(value, list) and len(value) == count
                                     and all(isinstance(v, (float, int)) for v in value)}
                        item.update(score_rank=int(ranks[position]),
                                    strictly_better_score_count=int(np.sum(total > total[position])),
                                    tied_score_count=int(np.sum(total == total[position])),
                                    route=stage['action_sources'][position], prediction=posterior)
                    return item

                coverage_by_pool = {
                    'generated': coverage(generated, observed, initial),
                    'action': coverage(action, observed, initial),
                    'terminal_union': coverage(terminal, observed, initial),
                    'terminal_only': coverage([i for i in terminal if i not in action_positions], observed, initial),
                    'generated_removed_before_scoring': coverage([i for i in generated if i not in action_positions], observed, initial),
                    'added_after_generation': coverage([i for i in action if i not in set(generated)], observed, initial),
                }
                by_route = {}
                for pool, ids, routes in [('generated', generated, stage['generated_sources']),
                                          ('action', action, stage['action_sources'])]:
                    by_route[pool] = {route: coverage([i for i, r in zip(ids, routes) if r == route], observed, initial)
                                      for route in sorted(set(routes))}

                unobserved_positions = [i for i, bank_id in enumerate(action) if bank_id not in observed]
                prediction = {'count': len(unobserved_positions)}
                if unobserved_positions:
                    positions = np.asarray(unobserved_positions, dtype=int)
                    values = [truth[action[i]] for i in positions]
                    for name, predicted, actual in [
                        ('objective', np.asarray(scores['objective_mean'])[positions], [v['objective'] for v in values]),
                        ('risk_mean', np.asarray(scores['constraint_mean'])[positions], [v['constraint_mean_centered'] for v in values]),
                        ('chance_margin', np.asarray(scores['stochastic_margin_mean'])[positions], [v['chance_margin'] for v in values]),
                        ('constraint_sigma', np.sqrt(np.maximum(np.asarray(scores['constraint_aleatoric'])[positions], 0)), [v['constraint_sigma'] for v in values]),
                        ('violation_probability', np.asarray(scores['probability_violation'])[positions], [float(not v['true_feasible']) for v in values]),
                    ]:
                        error = predicted - np.asarray(actual)
                        prediction[name + '_mse'] = float(np.mean(error ** 2))
                        prediction[name + '_bias'] = float(np.mean(error))

                selected = detail(mapping[stage['selected_id']])
                assert selected['score_rank'] == 1
                best = coverage_by_pool['action']['best_feasible_objective']
                selected.update(feasible_improving=bool(selected['truth']['true_feasible']
                    and selected['truth']['objective'] < initial - 1e-12),
                    feasible_objective_regret=None if best is None or not selected['truth']['true_feasible']
                    else max(0., selected['truth']['objective'] - best))
                improvements = list(dict.fromkeys(i for pool in coverage_by_pool.values()
                                                   for i in pool['feasible_improving_bank_ids']))
                rows.append({'seed': seed, 'arm': arm, 'iteration': stage['iteration'],
                    'paid_prefix': stage['paid_prefix'], 'coverage': coverage_by_pool,
                    'by_route': by_route, 'selected': selected, 'initial_policy0': detail(initial_id),
                    'unobserved_action_prediction_errors': prediction,
                    'feasible_improvements': [detail(i) for i in improvements],
                    'sampled_feasible_count': scores['sampled_feasible_count'],
                    'sampled_feasibility_mode': scores['sampled_feasibility_mode'],
                    'sampled_expert': scores.get('sampled_expert'),
                    'posterior_source': scores['posterior_source'],
                    'decision_aleatoric_mode': scores['decision_aleatoric_mode'],
                    'decision_ambiguity_mode': scores['decision_ambiguity_mode']})

        aggregates = []
        for arm in plan['arms']:
            group = [row for row in rows if row['arm'] == arm]
            agg = {'arm': arm, 'query_stages': len(group),
                   'selected_feasible': sum(row['selected']['truth']['true_feasible'] for row in group),
                   'selected_feasible_improving': sum(row['selected']['feasible_improving'] for row in group),
                   'sampled_feasibility_modes': dict(Counter(row['sampled_feasibility_mode'] for row in group)),
                   'initial0_in_action': sum(row['initial_policy0']['in_action_pool'] for row in group)}
            for pool in ('generated', 'action', 'terminal_only'):
                agg[pool] = {
                    'policy_occurrences': sum(row['coverage'][pool]['count'] for row in group),
                    'unobserved_occurrences': sum(row['coverage'][pool]['unobserved_count'] for row in group),
                    'feasible_occurrences': sum(row['coverage'][pool]['feasible_count'] for row in group),
                    'feasible_improving_occurrences': sum(row['coverage'][pool]['feasible_improving_count'] for row in group),
                    'stages_with_feasible_improvement': sum(row['coverage'][pool]['feasible_improving_count'] > 0 for row in group),
                    'maximum_quality_headroom': max(row['coverage'][pool]['quality_headroom_over_initial'] for row in group),
                }
            agg['by_route'] = {}
            for pool in ('generated', 'action'):
                agg['by_route'][pool] = {}
                for route in sorted(set().union(*(row['by_route'][pool] for row in group))):
                    covered = [row['by_route'][pool][route] for row in group if route in row['by_route'][pool]]
                    agg['by_route'][pool][route] = {
                        'stage_count': len(covered),
                        **{field: sum(item[field] for item in covered) for field in (
                            'count', 'unobserved_count', 'feasible_count', 'feasible_improving_count')},
                    }
            n = sum(row['unobserved_action_prediction_errors']['count'] for row in group)
            agg['unobserved_prediction_instances'] = n
            agg['unobserved_prediction_errors'] = {
                key: sum(row['unobserved_action_prediction_errors'].get(key, 0.)
                         * row['unobserved_action_prediction_errors']['count'] for row in group) / n
                for key in sorted(set().union(*(row['unobserved_action_prediction_errors'] for row in group)))
                if key != 'count' and n}
            aggregates.append(agg)
        assert len(rows) == 27
        save(output / 'summary.json', {
            'role': 'Post hoc audit of exactly recovered query pools; no new optimization, posterior intervention or verification.',
            'bank_index_space': 'Global truth_policies.npz order; truth.json has the same order.',
            'score_rank_rule': 'Descending original acquisition total, stable original candidate order; ties also reported.',
            'prediction_error_scope': 'Unobserved at each scoring checkpoint, including repeated predictions across stages; no matched-pool treatment effect.',
            'initial_policy0': {'bank_id': initial_id, 'policy_state': state[initial_id].tolist(), 'truth': truth[initial_id]},
            'case_count': len(cases(plan)), 'query_stages': len(rows), 'arms': aggregates, 'stages': rows,
            'unique_pool_coverage': {name: {
                'count': len(pool),
                'feasible_bank_ids': sorted(i for i in pool if truth[i]['true_feasible']),
                'minimum_noninitial_chance_margin': min(
                    truth[i]['chance_margin'] for i in pool if i != initial_id),
            } for name, pool in unique_pools.items()},
            'any_action_feasible_improvement': any(row['coverage']['action']['feasible_improving_count'] for row in rows),
            'any_terminal_only_feasible_improvement': any(row['coverage']['terminal_only']['feasible_improving_count'] for row in rows),
        })
    # The summarize attempt is now closed, so its actual cost is included once.
    attempts = [json.loads(line) for path in sorted(output.rglob('attempts.jsonl'))
                for line in path.read_text().splitlines() if line.strip()]
    totals = {key: sum(row.get(key, 0) for row in attempts) for key in (
        'cpu_seconds', 'wall_seconds', 'new_truth_function_calls', 'new_source_calls',
        'new_target_search_calls', 'new_verification_calls', 'new_source_fits', 'download_bytes')}
    save(output / 'completion.json', {
        'status': 'complete', 'case_count': len(cases(plan)), 'parity_query_stages': len(rows),
        'summary': str((output / 'summary.json').relative_to(ROOT)),
        'recorded_attempt_count': len(attempts),
        'failed_attempt_count': sum(row['status'] == 'failed' for row in attempts),
        'failed_attempts': [{key: row.get(key) for key in ('stage', 'family', 'arm', 'seed', 'error')}
                            for row in attempts if row['status'] == 'failed'],
        'cost': totals, 'wall_time_definition': 'Sum of attempt wall seconds, not elapsed wall time; cases may run concurrently.',
        'new_simulator_calls': totals['new_source_calls'] + totals['new_target_search_calls'] + totals['new_verification_calls'],
        'completed_utc': datetime.now(timezone.utc).isoformat(),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(MANIFEST))
    args = parser.parse_args()
    summarize(read(args.manifest))


if __name__ == '__main__':
    main()

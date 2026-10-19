"""Partition saved native TS score changes without replay or model updates.

The two partitions are policies sampled since initial10 and all other policies.
This decomposes recorded scoring changes; it is not a point-only posterior.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from performance.diagnose_mainline_shared_neighbors import load_truth_cache
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.run_mainline_query_channels import constrained_ts


MANIFEST = ROOT / 'performance/manifests/mainline_initialization_audit_v1.json'
ARM = 'legacy_native_neighbors'
FIELDS = ('sampled_objective', 'sampled_constraint', 'constraint_aleatoric')


def assert_parity(actual, expected, selected_id):
    # Detect an incorrect tau, action mapping or TS branch before attribution.
    for field in ('total', 'sampled_margin'):
        np.testing.assert_allclose(actual[field], expected[field],
                                   rtol=1e-12, atol=1e-12, err_msg=field)
    assert actual['selected_id'] == selected_id
    for field in ('sampled_feasible_count', 'sampled_feasibility_mode'):
        assert actual[field] == expected[field]


def run(plan):
    output = ROOT / plan['output_dir']
    parent = ROOT / plan['parent_output']
    channels = ROOT / plan['query_channels_output']
    problem = SimpleNamespace(alpha=plan['task']['alpha'], tau=plan['task']['tau'])
    with attempt(output, 'native-score-partitions', plan['domain'], ARM) as cost:
        cost.update(new_truth_function_calls=0, new_source_fits=0,
                    download_bytes=0, model_replays=0, score_stage_count=0)
        destination = output / 'native_self_cross.json'
        if destination.exists():
            cost['reused'] = True
            return read(destination)
        rows, case_points, case_neighbors, histories = [], {}, {}, {}
        for seed in plan['target_seeds']:
            source = parent / f'seed_{seed}' / ARM
            audit = read(source / 'query_audit.json')
            search = read(source / 'search.json')
            frozen = read(channels / f'seed_{seed}' / ARM / 'native_frozen10.json')
            assert frozen['status'] == 'complete' and frozen['first_query_full_parity']
            points = [tuple(map(int, x)) for x in np.load(source / 'query_policies.npz')['points']]
            neighbors = [tuple(map(int, x)) for x in
                         np.load(parent / f'seed_{seed}' / 'neighbors.npz')['points']]
            case_points[seed], case_neighbors[seed] = points, neighbors
            histories[seed] = search['history']
            stages = []
            for actual, fixed in zip(audit['stages'], frozen['stages'], strict=True):
                count, ids = actual['paid_prefix'], actual['scored_ids']
                assert count == fixed['paid_prefix']
                sampled = {tuple(row['point']) for row in search['history'][10:count]}
                mask = np.asarray([points[i] in sampled for i in ids])
                full_score, fixed_score = actual['score'], fixed['score']
                controls = {}
                for name, use_full in (
                    ('frozen10', np.zeros(len(ids), dtype=bool)),
                    ('own_partition', mask), ('cross_partition', ~mask),
                    ('full', np.ones(len(ids), dtype=bool)),
                ):
                    f, g, noise = [np.where(use_full, full_score[field], fixed_score[field])
                                   for field in FIELDS]
                    controls[name] = constrained_ts(f, g, noise, problem, ids)
                    controls[name]['constraint_aleatoric'] = noise
                assert_parity(controls['full'], full_score, actual['selected_id'])
                assert_parity(controls['frozen10'], fixed_score, fixed['selected_id'])
                if count == 10:
                    for control in controls.values():
                        assert_parity(control, full_score, actual['selected_id'])
                stages.append({'query_index': actual['iteration'] + 1,
                    'paid_prefix': count, 'action_ids': ids,
                    'own_partition_ids': [i for i, own in zip(ids, mask) if own],
                    'own_partition_neighbor_indices': [neighbors.index(point)
                        if point in neighbors else None for point in sampled],
                    'full_expert': full_score['sampled_expert'],
                    'frozen_expert': fixed_score['sampled_expert'],
                    'experts_match': full_score['sampled_expert'] == fixed_score['sampled_expert'],
                    'full_and_frozen_score_parity': True, 'controls': controls})
                cost['score_stage_count'] += 1
            rows.append({'seed': seed, 'stages': stages})

        # All nine score interventions are fixed before consulting cached truth.
        cache = load_truth_cache(plan, parent)
        for row in read(channels / 'truth_new.json'):
            cache.setdefault(tuple(row['point']), {}).update(row['truth'])
        for row in rows:
            seed = row['seed']
            points, neighbors = case_points[seed], case_neighbors[seed]
            initial = cache[tuple(histories[seed][0]['point'])]
            improving = [i for i, point in enumerate(neighbors)
                if cache.get(point, {}).get('true_feasible', False)
                and cache[point]['objective'] < initial['objective'] - 1e-12]
            row['improving_neighbor_indices'] = improving
            for stage in row['stages']:
                ids = stage['action_ids']
                focus = set(improving)
                for control in stage['controls'].values():
                    point = points[control['selected_id']]
                    if point in neighbors:
                        focus.add(neighbors.index(point))
                for control in stage['controls'].values():
                    total = np.asarray(control['total'])
                    ranks = np.empty(len(total), dtype=int)
                    ranks[np.argsort(-total, kind='stable')] = np.arange(1, len(total) + 1)

                    def detail(index):
                        point = points[ids[index]]
                        truth = cache.get(point)
                        return {'policy_id': ids[index],
                            'neighbor_index': neighbors.index(point) if point in neighbors else None,
                            'own_partition': ids[index] in stage['own_partition_ids'],
                            'rank': int(ranks[index]),
                            'strictly_better_count': int(np.sum(total > total[index])),
                            'sampled_objective': float(control['sampled_objective'][index]),
                            'sampled_constraint': float(control['sampled_constraint'][index]),
                            'noise_sigma': float(np.sqrt(control['constraint_aleatoric'][index])),
                            'sampled_margin': float(control['sampled_margin'][index]),
                            'total': float(total[index]), 'truth': truth,
                            'cached_truth_missing': truth is None,
                            'feasible_improving': None if truth is None else bool(
                                truth['true_feasible'] and truth['objective'] < initial['objective'] - 1e-12)}

                    control['selected'] = detail(control['selected_index'])
                    positions = {points[pid]: i for i, pid in enumerate(ids)}
                    control['focus_neighbors'] = [detail(positions[neighbors[i]]) for i in sorted(focus)
                                                   if neighbors[i] in positions]
        summary = []
        for name in plan['native_partition']['controls']:
            selections = [stage['controls'][name]['selected'] for row in rows
                          for stage in row['stages'] if stage['paid_prefix'] > 10]
            summary.append({'control': name, 'queries': len(selections),
                'cached_truth_missing_count': sum(x['cached_truth_missing'] for x in selections),
                'safe_count': sum(x['truth']['true_feasible'] for x in selections if x['truth']),
                'feasible_improving_count': sum(x['feasible_improving'] is True for x in selections),
                'selected_neighbor_indices': [x['neighbor_index'] for x in selections]})
        result = {'status': 'complete', 'rows': rows, 'paid11_12_summary': summary,
            'score_stage_count': cost['score_stage_count'], 'all_full_and_frozen_scores_match': True,
            'all_full_and_frozen_experts_match': all(s['experts_match'] for r in rows for s in r['stages']),
            'scores_frozen_before_cached_truth': True,
            'definition': plan['native_partition'],
            'limitations': 'Partitions combine saved draws and noise; they are not native conditional posteriors. Actual histories and pools remain fixed, and no alternate rollout or deployment was executed.',
            'cost': dict(cost, status='completed')}
        save(destination, result)
        print({'stages': result['score_stage_count'], 'summary': summary}, flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    args = parser.parse_args()
    run(read(args.manifest))


if __name__ == '__main__':
    main()

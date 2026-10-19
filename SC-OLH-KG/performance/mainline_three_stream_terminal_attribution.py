"""Join three sealed streams by physical policy, reusing the original audit."""
import copy
from pathlib import Path

from performance import mainline_confirmation_terminal_attribution as terminal


AFTER = 'legacy_role_after_init_point_only'


def register(registry, point):
    from performance.mainline_three_stream_query_attribution import register_policy
    return register_policy(registry, point)


def load_terminal(root, stream, seed):
    """Reuse Stage66's original terminal attribution without evaluating it again."""
    if stream.get('reuse_attribution_dir'):
        path = root / stream['reuse_attribution_dir'] / 'cases' / f'seed_{seed}' / 'analysis.json'
        return terminal.read(path)['terminal'], True, str(path.relative_to(root))
    return terminal.analyze_task(root, stream, seed), False, None


def attach_policy_ids(result, histories, shortlists, registry):
    """Labels remain readable aliases; only exact physical vectors define identity."""
    result = copy.deepcopy(result)
    for arm in result['arms']:
        name = arm['arm']
        history, points = histories.get(name), shortlists.get(name)
        arm['history_available'] = history is not None
        arm['history_policy_ids'] = ([] if history is None else
            [register(registry, row['point']) for row in history])
        arm['observed_policy_ids'] = list(dict.fromkeys(arm['history_policy_ids']))
        labels = {} if points is None else {tuple(point): candidate['policy']
            for point, candidate in zip(points, arm['candidates'])}
        arm['adaptive_queries'] = []
        for index, row in enumerate(history or []):
            if index < 10:
                continue
            point = tuple(row['point'])
            arm['adaptive_queries'].append(dict(history_index=index,
                policy_id=arm['history_policy_ids'][index],
                label=labels.get(point, f'{arm["history_arm"]}:history:{index}'),
                label_is_history_anchor=point not in labels))
        label_ids = {}
        for index, candidate in enumerate(arm['candidates']):
            candidate['policy_id'] = None if points is None else register(registry, points[index])
            label_ids[candidate['policy']] = candidate['policy_id']
        for protocol in arm['protocols'].values():
            protocol['reported_policy_id'] = label_ids.get(protocol['reported_deployed'])
            protocol['deployed_policy_id'] = label_ids.get(protocol['deployed'])
            for comparison in protocol['comparisons']:
                for field in ('challenger', 'incumbent', 'winner'):
                    comparison[field + '_policy_id'] = label_ids[comparison[field]]
        third = arm.get('third_comparison', {}).get('actual')
        if third is not None:
            for field in ('challenger', 'incumbent', 'winner'):
                third[field + '_policy_id'] = label_ids[third[field]]
    return result


def exposure_paths(streams):
    """Track AFTER's actual queried/shortlisted policies through every retained arm."""
    tracked = []
    for stream in streams:
        after = next(row for row in stream['terminal']['arms'] if row['arm'] == AFTER)
        tracked.extend(row['policy_id'] for row in after['candidates'])
        tracked.extend(row['policy_id'] for row in after['adaptive_queries'])
    result = []
    for policy_id in dict.fromkeys(pid for pid in tracked if pid is not None):
        paths, aliases, truth = [], [], None
        for stream in streams:
            for arm in stream['terminal']['arms']:
                candidate = next((row for row in arm['candidates'] if row['policy_id'] == policy_id), None)
                if candidate is not None:
                    aliases.append(dict(stream=stream['name'], arm=arm['arm'], label=candidate['policy']))
                    if truth is None:
                        truth = candidate['cached_truth']
                observed_at = ([i for i, pid in enumerate(arm['history_policy_ids']) if pid == policy_id]
                               if arm['history_available'] else None)
                path = dict(stream=stream['name'], arm=arm['arm'], history_arm=arm['history_arm'],
                    observed_at=observed_at,
                    shortlisted_slot=None if candidate is None else candidate['index'] + 1,
                    shortlisted_role=None if candidate is None else candidate['role'])
                for name in ('old', 'new'):
                    protocol = arm['protocols'].get(name)
                    if protocol is None:
                        path[name] = dict(status='unavailable', safety=None, comparisons=[], deployed=None)
                        continue
                    comparisons = [dict(block=row['block_index'],
                        challenger_policy_id=row['challenger_policy_id'],
                        incumbent_policy_id=row['incumbent_policy_id'], winner_policy_id=row['winner_policy_id'],
                        pair=[row['challenger'], row['incumbent']], winner=row['winner'],
                        mean_difference=row['sample_mean_difference'],
                        true_difference=row['true_objective_difference'],
                        upper_bound=row['one_sided_upper_confidence_bound'])
                        for row in protocol['comparisons']
                        if policy_id in (row['challenger_policy_id'], row['incumbent_policy_id'])]
                    path[name] = dict(status=protocol['status'],
                        safety=(dict(status='not_shortlisted', passed=None) if candidate is None else candidate[name]['safety']),
                        comparisons=comparisons,
                        deployed=None if protocol['status'] != 'complete' else protocol['deployed_policy_id'] == policy_id,
                        reported=protocol['reported_policy_id'] == policy_id)
                paths.append(path)
        result.append(dict(policy_id=policy_id, aliases=aliases, cached_truth=truth, paths=paths))
    return dict(tracked_arm=AFTER, policies=result,
        identity_basis='Exact integer policy vectors within geometry; history labels are not identities.',
        scope='Actual observed and terminal exposure only; unperformed tests remain unknown.')


def registered_negatives(root, plan, seed):
    if seed not in (2026104013, 2026104005):
        return []
    file = plan['registered_negative_records']
    saved = terminal.read(root / file)
    if seed == 2026104013:
        rows = [dict(recorded_stream=row['stream'], record=row) for row in saved['focal_rows']
                if row['seed'] == seed]
    else:
        rows = [dict(recorded_stream=1, record=row) for row in saved['stream1_struct_negative_rows']
                if row['seed'] == seed]
    return [dict(source_file=file, handling='Copied saved finding; no certificate or comparison recomputation.',
                 **row) for row in rows]


def analyze_task(root, plan, seed, policy_registry=None):
    root = Path(root)
    registry = {} if policy_registry is None else policy_registry
    streams = []
    for descriptor in plan['streams']:
        result, reused, path = load_terminal(root, descriptor, seed)
        histories, shortlists, history_cache = {}, {}, {}
        for arm in result['arms']:
            source_case = arm.get('source_case')
            if source_case:
                if source_case not in history_cache:
                    history_cache[source_case] = terminal.read(root / source_case / 'search.json')['history']
                histories[arm['arm']] = history_cache[source_case]
            protocols = arm['protocols']
            input_file = (protocols.get('new', {}).get('input_file') or
                          protocols.get('old', {}).get('input_file'))
            if input_file:
                shortlists[arm['arm']] = [row['point'] for row in terminal.read(root / input_file)['frozen_shortlist']]
        identified = attach_policy_ids(result, histories, shortlists, registry)
        streams.append(dict(name=descriptor['name'], reused=reused, reused_file=path,
                            source_output_dir=descriptor['source_output_dir'], terminal=identified))
    exposure = (exposure_paths(streams) if seed in plan['focus_terminal_seeds'] else
                dict(status='not_a_registered_terminal_focus', policies=[]))
    return dict(seed=seed, domain=streams[0]['terminal']['domain'], status='complete', streams=streams,
        exposure=exposure, registered_negatives=registered_negatives(root, plan, seed),
        new_work=dict(terminal.ZERO))


def compact_task(result):
    streams = []
    for stream in result['streams']:
        compact = terminal.compact_task(stream['terminal'])
        indexed = {row['arm']: row for row in stream['terminal']['arms']}
        for arm in compact['arms']:
            source = indexed[arm['arm']]
            arm.update({key: source[key] for key in ('history_available', 'history_policy_ids',
                                                    'observed_policy_ids', 'adaptive_queries')})
            for candidate, original in zip(arm['candidates'], source['candidates']):
                candidate['policy_id'] = original['policy_id']
            for name, protocol in arm['protocols'].items():
                original = source['protocols'][name]
                protocol.update({key: original[key] for key in ('reported_policy_id', 'deployed_policy_id')})
                for comparison, actual in zip(protocol['comparisons'], original['comparisons']):
                    comparison.update({key: actual[key] for key in ('challenger_policy_id',
                        'incumbent_policy_id', 'winner_policy_id')})
        streams.append(dict(name=stream['name'], reused=stream['reused'], reused_file=stream['reused_file'],
                            source_output_dir=stream['source_output_dir'], terminal=compact))
    return {**{key: result[key] for key in ('seed', 'domain', 'status', 'exposure',
             'registered_negatives', 'new_work')}, 'streams': streams}


def aggregate_terminals(compact_tasks):
    names = list(dict.fromkeys(stream['name'] for task in compact_tasks for stream in task['streams']))
    streams = []
    for name in names:
        selected = [stream for task in compact_tasks for stream in task['streams'] if stream['name'] == name]
        streams.append(dict(name=name, reused_tasks=sum(stream['reused'] for stream in selected),
            summary=terminal.aggregate_terminals([stream['terminal'] for stream in selected])))
    return dict(task_count=len(compact_tasks), streams=streams,
        registered_pairs=sum(len(stream['terminal']['arms']) for task in compact_tasks for stream in task['streams']),
        focal_exposure=[dict(seed=task['seed'], domain=task['domain'], **task['exposure'])
                        for task in compact_tasks if task['exposure']['policies']],
        registered_negatives=[row for task in compact_tasks for row in task['registered_negatives']],
        new_work=dict(terminal.ZERO),
        interpretation='Streams stay separate; paired terminal effects and their historical fees are reused, not new confirmation samples.')

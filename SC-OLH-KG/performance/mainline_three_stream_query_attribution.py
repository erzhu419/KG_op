"""Saved three-stream query attribution with exact physical-policy alignment."""
from pathlib import Path

import numpy as np

from performance import mainline_confirmation_query_attribution as single
from performance.run_mainline_legacy_backend_bridge import read

ARMS = single.ARMS
ARM_LABEL = single.ARM_LABEL


def register_policy(registry, point):
    key = tuple(int(value) for value in point)
    if key not in registry:
        registry[key] = f'P{len(registry):03d}'
    return registry[key]


def register_alias(registry, catalog, point, origin, alias):
    identifier = register_policy(registry, point)
    row = catalog.setdefault(identifier, dict(policy_id=identifier, origin=origin, aliases=[]))
    if alias not in row['aliases']:
        row['aliases'].append(alias)
    return identifier


def load_query(root, stream, seed):
    """The opened original result is read verbatim, never reconstructed again."""
    if stream.get('reuse_attribution_dir'):
        path = Path(root) / stream['reuse_attribution_dir'] / 'cases' / f'seed_{seed}' / 'analysis.json'
        return read(path)['query'], True
    return single.analyze_task(root, dict(source_manifest=stream['source_manifest'],
        source_output_dir=stream['source_output_dir']), seed), False


def stream_files(root, stream, seed):
    cases = {r['arm']:r for r in read(root / stream['source_output_dir'] /
        'execution_status.json')['cases'] if r['seed']==seed and r['arm'] in ARMS}
    directory = (root / cases[single.AFTER]['source_case']).parent
    initial = [tuple(r['point']) for r in read(directory/'initialization.json')['history']]
    with np.load(directory/'neighbors.npz') as saved:
        shared = [tuple(map(int,p)) for p in saved['points']]
    bundles, deployments = {}, []
    for arm in ARMS:
        case = cases[arm]
        path = root / case['source_case']
        with np.load(path/'query_policies.npz') as saved:
            bank = [tuple(map(int,p)) for p in saved['points']]
        bundles[arm] = dict(bank=bank,audit=read(path/'query_audit.json'),
            history=read(path/'search.json')['history'])
        for protocol in ('old','new'):
            descriptor = case[protocol]
            if descriptor['status']=='complete':
                record = read(root/descriptor['verification_file'])
                deployments.append(dict(arm=arm,protocol=protocol,
                    point=tuple(record['deployed']),certified=record['verification']['certified']))
    return initial, shared, bundles, deployments


def label(point, initial, shared, bundles):
    if point in initial:
        return 'initial', f'initial:{initial.index(point)}'
    if point in shared:
        return 'shared_neighbor', f'neighbor:{shared.index(point)}'
    for arm,bundle in bundles.items():
        for i,row in enumerate(bundle['history']):
            if tuple(row['point'])==point:
                return 'dynamic',f'{ARM_LABEL[arm]}:history:{i}'
    return 'dynamic',None


def analyze_task(root, plan, seed, policy_registry=None):
    """Reconstruct only the two new streams and align all three saved physical paths."""
    root = Path(root)
    registry = policy_registry if policy_registry is not None else {}
    streams, catalog, contexts, targets = [], {}, {}, {}
    for descriptor in plan['streams']:
        name = descriptor['name']
        query,reused = load_query(root,descriptor,seed)
        initial,shared,bundles,deployments = stream_files(root,descriptor,seed)
        contexts[name] = dict(initial=initial,shared=shared,bundles=bundles)

        def identify(point, arm, kind, **location):
            origin,display = label(point,initial,shared,bundles)
            return register_alias(registry,catalog,point,origin,
                dict(stream=name,arm=ARM_LABEL[arm],kind=kind,label=display,**location))

        for state in query['states']:
            arm = state['arm']
            selected = state['choices']['actual']['selected']
            point = bundles[arm]['bank'][selected['policy_id']]
            selected['physical_policy_id'] = identify(point,arm,'query',prefix=state['paid_prefix'])
        for pair in query['paired_states']:
            for arm in (pair['left'],pair['right']):
                bundle = bundles[arm]
                stage = next(s for s in bundle['audit']['stages'] if s['paid_prefix']==pair['paid_prefix'])
                pair['selected'][arm]['physical_policy_id'] = identify(
                    bundle['bank'][stage['selected_id']],arm,'query',prefix=pair['paid_prefix'])
            for swap in pair['substitutions']:
                arm = pair['left']
                point = bundles[arm]['bank'][swap['selected']['left_bank_id']]
                swap['selected']['physical_policy_id'] = identify(point,arm,
                    'fixed_state_substitution',prefix=pair['paid_prefix'])
        physical_deployments = []
        for deployed in deployments:
            arm,point = deployed['arm'],deployed['point']
            identifier = identify(point,arm,'reported_deployment',protocol=deployed['protocol'])
            physical_deployments.append(dict(arm=ARM_LABEL[arm],protocol=deployed['protocol'],
                policy_id=identifier,certified=deployed['certified']))
            targets.setdefault(identifier,dict(point=point,reported_deployments=[]))[
                'reported_deployments'].append(dict(stream=name,arm=ARM_LABEL[arm],
                    protocol=deployed['protocol'],certified=deployed['certified']))
            if arm==single.AFTER:
                path = next(r for r in query['deployment_paths'] if r['protocol']==deployed['protocol'])
                path['physical_policy_id'] = identifier
        streams.append(dict(name=name,reused=reused,query=query,
            physical_deployments=physical_deployments))
    focal = []
    if seed in plan.get('focus_query_seeds',plan.get('focus_seeds',[])):
        for identifier,target in targets.items():
            paths = []
            for name,context in contexts.items():
                for arm,bundle in context['bundles'].items():
                    paths.append(dict(stream=name,arm=ARM_LABEL[arm],
                        **single.candidate_path(target['point'],bundle)))
            focal.append(dict(policy_id=identifier,origin=catalog[identifier]['origin'],
                reported_deployments=target['reported_deployments'],paths=paths))
    return dict(status='complete',seed=seed,domain=streams[0]['query']['domain'],streams=streams,
        physical_policies=list(catalog.values()),focal_chains=focal,
        reconstruction=dict(reused_choices=sum(s['query']['reconstruction']['recorded_choices']
            for s in streams if s['reused']),newly_reconstructed_choices=sum(
            s['query']['reconstruction']['recorded_choices'] for s in streams if not s['reused'])),
        limits=['Policy IDs compare exact physical integer vectors within this geometry, not history labels.',
            'Original query-score attribution is reused; only two new streams are reconstructed.',
            'Only same-stream, same-history, same-pool first-divergence substitutions are reported.',
            'Cross-stream paths have different observations and are descriptive, not counterfactual trajectories.',
            'Missing cached truth and unexecuted certificates remain unknown; no new truth is evaluated.'])


def compact_query(query):
    result = single.compact_task(query)
    for compact,raw in zip(result['actual_queries'],query['states']):
        compact['physical_policy_id'] = raw['choices']['actual']['selected']['physical_policy_id']
    for comparison in result['comparisons']:
        first = comparison['first_divergence']
        if first is None:
            continue
        pair = next(r for r in query['paired_states'] if r['first_divergence'] and
            ARM_LABEL[r['left']]==comparison['left'] and ARM_LABEL[r['right']]==comparison['right'])
        first['selected_policy_ids'] = {ARM_LABEL[a]:p['physical_policy_id']
            for a,p in pair['selected'].items()}
        for compact,raw in zip(first['swaps'],pair['substitutions']):
            compact['selected_policy_id'] = raw['selected']['physical_policy_id']
    if result['after_new'] is not None:
        path = next(r for r in query['deployment_paths'] if r['protocol']=='new')
        result['after_new']['physical_policy_id'] = path['physical_policy_id']
    return result


def compact_path(path):
    states = []
    for state in path['states']:
        row = dict(prefix=state['paid_prefix'],present=state['present'])
        if state['present']:
            row.update(observed=state['previously_observed'],selected=state['selected'],
                reason=state['reason'],margin=state['sampled_margin'],
                risk_mean=state['constraint_mean'],
                noise_margin=state['stochastic_margin_mean']-state['constraint_mean'],
                risk_draw=state['sampled_constraint']-state['constraint_mean'],
                objective_mean=state['objective_mean'],
                objective_draw=state['sampled_objective']-state['objective_mean'],
                ts_rank=state['sampled_objective_rank_all'],
                feasible_ts_rank=state['sampled_objective_rank_within_feasible'])
        states.append(row)
    return dict(stream=path['stream'],arm=path['arm'],in_bank=path['present_in_saved_bank'],
        first_candidate_prefix=path['first_candidate_paid_prefix'],
        observed_indices=path['observation_history_indices'],states=states)


def compact_task(result):
    return dict(status=result['status'],seed=result['seed'],domain=result['domain'],
        streams=[dict(name=s['name'],reused=s['reused'],query=compact_query(s['query']),
            physical_deployments=s['physical_deployments']) for s in result['streams']],
        physical_policies=result['physical_policies'],
        focal_chains=[dict(policy_id=r['policy_id'],origin=r['origin'],
            reported_deployments=r['reported_deployments'],paths=[compact_path(p) for p in r['paths']])
            for r in result['focal_chains']],reconstruction=result['reconstruction'],limits=result['limits'])


def aggregate_queries(compact_tasks):
    names = list(dict.fromkeys(s['name'] for t in compact_tasks for s in t['streams']))
    streams = []
    for name in names:
        rows = [s for task in compact_tasks for s in task['streams'] if s['name']==name]
        streams.append(dict(name=name,reused=all(s['reused'] for s in rows),
            **single.aggregate_queries([s['query'] for s in rows])))
    return dict(task_count=len(compact_tasks),streams=streams,
        reused_choices=sum(t['reconstruction']['reused_choices'] for t in compact_tasks),
        newly_reconstructed_choices=sum(t['reconstruction']['newly_reconstructed_choices']
            for t in compact_tasks),focal_seeds=[t['seed'] for t in compact_tasks if t['focal_chains']],
        limits='Stream summaries reuse the same24 geometries; they are not72 independent tasks or new confirmation estimates.')

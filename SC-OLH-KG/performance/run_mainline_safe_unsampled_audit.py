"""Post-terminal full-pool truth inventory, per-task scoring, and AP-old summary."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import socket
from unittest.mock import patch

import numpy as np

from performance import summarize_mainline_two_comparison_confirmation as terminal
from performance import run_mainline_two_comparison_confirmation as runner
from performance.diagnose_mainline_shared_neighbors import truth_value
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.run_mainline_two_comparison_development import benchmark, build_problem

MANIFEST = ROOT/'performance/manifests/mainline_safe_unsampled_validation_v1.json'
ROLES = ('legacy_role_after_init_point_only','legacy_role_point_only','legacy_role_full')
ZERO = dict(new_target_initial_calls=0,new_target_search_calls=0,new_verification_calls=0,
    new_source_calls=0,new_source_fits=0,new_truth_function_calls=0,source_model_restorations=0,
    engine_restorations=0,posterior_restorations=0,download_bytes=0,new_rng_draws=0,
    truth_problem_constructions=0)


def directory(plan):
    return ROOT/plan['output_dir']/'mechanism'


def physical(points):
    return [tuple(map(int,point)) for point in points]


def load_bank(path):
    with np.load(path) as saved:
        return physical(saved['points'])


def history_bundle(path, initial):
    """An unsuccessful captured choice is not a returned observation."""
    completed = path/'search.json'
    if completed.exists():
        result = read(completed)
        if not (path/'query_audit.json').exists():
            return dict(status='missing',points=[],audit={'stages':[]},history=result['history'],
                reason='completed_history_missing_query_audit')
        return dict(status='complete',points=load_bank(path/'query_policies.npz'),
            audit=read(path/'query_audit.json'),history=result['history'])
    audits = sorted(path.glob('failed_query_audit_*.json'))
    if audits:
        audit = read(audits[-1])
        raw = read(path/audit['paid_raw_file'])
        assert len(raw)==audit['returned_raw_count']
        return dict(status='partial',points=load_bank(path/audit['query_policies_file']),
            audit=audit,history=[*initial,*raw],failure_audit=str(audits[-1].relative_to(ROOT)),
            history_semantics='Returned observations; partial final response may not be conditioned by backend.')
    return dict(status='missing',points=[],audit={'stages':[]},history=list(initial),
        reason='search_failed_before_query_audit_or_dependency_failed')


def task_material(plan, task, cases):
    seed = task['geometry_task_seed']
    prepared = read(ROOT/plan['output_dir']/'prepared.json')
    family = next(f for f in prepared['replicates'][0]['families'] if f['domain']==task['domain'])
    source = read(ROOT/family['search_manifest'])
    root = ROOT/source['output_dir']/f'seed_{seed}'
    initial_path = root/'initialization.json'
    if initial_path.exists():
        initial = read(initial_path)['history']
    else:
        failed = sorted((root/'initialization_only').glob('failed_search_raw_*.json'))
        initial = read(failed[-1]) if failed else []
    shared = load_bank(root/'neighbors.npz') if (root/'neighbors.npz').exists() else []
    bundles = {arm:history_bundle(root/arm,initial) for arm in ROLES}
    selected = [case for case in cases if case['seed']==seed]
    return root,source,initial,shared,bundles,selected


def inventory_points(initial, bundles, cases):
    """Whole actual action union, not generated-only or chosen-only candidates."""
    points = dict.fromkeys(tuple(row['point']) for row in initial)
    for bundle in bundles.values():
        for stage in bundle['audit']['stages']:
            points.update(dict.fromkeys(bundle['points'][i] for i in stage['action_ids']))
    for case in cases:
        for protocol in ('old','new'):
            descriptor = case[protocol]
            if descriptor['item'] is not None:
                points.update(dict.fromkeys(tuple(row['point']) for row in descriptor['item']['frozen_shortlist']))
            if descriptor['status']=='complete':
                points[tuple(descriptor['record']['deployed'])] = None
    return list(points)


def freeze(plan):
    # Validate the entire terminal barrier before reading any task's candidates.
    seal,cases = terminal.load_sealed(plan)
    output = directory(plan)
    if (output/'inventory.json').exists():
        return read(output/'inventory.json')
    with attempt(output/'inventory','freeze-full-action-pool-inventory') as cost:
        cost.update(ZERO)
        rows = []
        for task in plan['tasks']:
            root,source,initial,shared,bundles,selected = task_material(plan,task,cases)
            points = inventory_points(initial,bundles,selected)
            path = output/f"seed_{task['geometry_task_seed']}"/'policies.npz'
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,points=np.asarray(points,dtype=np.int16).reshape((-1,plan['frozen']['d'])))
            rows.append(dict(seed=task['geometry_task_seed'],domain=task['domain'],policies=str(path.relative_to(ROOT)),
                distinct_policies=len(points), maximum_analytic_function_calls=3*len(points),
                source_directory=str(root.relative_to(ROOT)),initial_rows=len(initial),
                role_status={arm:bundle['status'] for arm,bundle in bundles.items()},
                captured_stages={arm:len(bundle['audit']['stages']) for arm,bundle in bundles.items()},
                frozen_utc=datetime.now(timezone.utc).isoformat()))
        result = dict(status='sealed',tasks=rows,registered_tasks=len(plan['tasks']),
            terminal_pairs=seal['registered_pairs'],truth_used=False,new_truth_function_calls=0,
            sealed_utc=datetime.now(timezone.utc).isoformat(),
            distinct_policy_task_units=sum(r['distinct_policies'] for r in rows),
            maximum_analytic_function_calls=sum(r['maximum_analytic_function_calls'] for r in rows))
        save(output/'inventory.json',result)
    return result


def cache_read(path):
    return {tuple(row['point']):row['truth'] for row in read(path)} if path.exists() else {}


def cache_save(path,cache):
    save(path,[dict(point=list(point),truth=value) for point,value in cache.items()])


def truth_problem(task,source,cases):
    anchor = next((case[p]['item'] for case in cases for p in ('old','new') if case[p]['item']),None)
    if anchor is not None:
        return build_problem(anchor)
    config,kwargs = source['task'],dict(task['geometry']['problem_kwargs'])
    if task['domain']=='FactorShockStatePolicyRZDT1':
        kwargs['shared_shock_scale']=config['shared_shock_scale']
    return benchmark.build_scalarized_problem(task['domain'],config['d'],config['L'],config['sigma'],
        config['alpha'],tuple(config['weights']),problem_kwargs=kwargs)


def score_points(points,cache,problem_factory,cost):
    problem,extra = None,{}
    for point in points:
        if point in cache and all(k in cache[point] for k in
                ('objective','constraint_mean_centered','constraint_sigma','quality','true_feasible')):
            continue
        if problem is None:
            problem = problem_factory()
            cost['truth_problem_constructions']+=1
            cost['new_rng_draws']=None  # Factory internals are not instrumented.
        with patch.object(problem,'simulate',side_effect=RuntimeError('Post-seal audit cannot simulate')):
            truth_value(problem,point,cache,extra,cost)


def context(initial,shared,bundles,cases,cache):
    best,status = terminal.initial_reference(initial,cache)
    histories = {}
    for case in cases:
        path = case['source_case']
        if path not in histories and any(case[p]['item'] for p in ('old','new')):
            saved = read(ROOT/path/'search.json')
            histories[path] = terminal.policy_labels(initial,shared,saved['history'],case['history_arm'])
    return dict(cache=cache,initial=initial,neighbors=shared,best=best,reference_status=status,histories=histories)


def task_score(plan,seed):
    from performance.mainline_safe_unsampled_competition import analyze_task,compact_task
    _,cases = terminal.load_sealed(plan)
    output = directory(plan)/f'seed_{seed}'
    if (output/'result.json').exists():
        return read(output/'result.json')
    if (output/'failure.json').exists():
        raise RuntimeError('Retain the failed audit and its truth cache; no automatic retry')
    inventory = read(directory(plan)/'inventory.json')
    assert inventory['status']=='sealed' and inventory['registered_tasks']==len(plan['tasks'])
    row = next(r for r in inventory['tasks'] if r['seed']==seed)
    task = next(t for t in plan['tasks'] if t['geometry_task_seed']==seed)
    root,source,initial,shared,bundles,selected = task_material(plan,task,cases)
    points = load_bank(ROOT/row['policies'])
    cache = cache_read(root/'truth.json')
    try:
        with attempt(output,'score-full-action-pool',task['domain']) as cost:
            cost.update(ZERO,seed=seed)
            try:
                score_points(points,cache,lambda:truth_problem(task,source,selected),cost)
            finally:
                cache_save(root/'truth.json',cache)
            ctx = context(initial,shared,bundles,selected,cache)
            outcomes = [terminal.score_pair(case,ctx) for case in selected]
            result = analyze_task(seed,task['domain'],bundles,cache,initial,shared,selected,outcomes)
            save(output/'analysis.json',result)
            assert result['coverage']['invalid_replays']==0, 'Saved scores do not reproduce captured queries'
            result = dict(status='complete',seed=seed,domain=task['domain'],
                distinct_inventory_policies=len(points),cached_inventory_policies=sum(p in cache for p in points),
                new_truth_function_calls=cost['new_truth_function_calls'],
                truth_problem_constructions=cost['truth_problem_constructions'],mechanism=compact_task(result))
            assert cost['new_truth_function_calls'] <= row['maximum_analytic_function_calls']
            save(output/'result.json',result)
    except Exception as error:
        save(output/'failure.json',dict(status='failed',seed=seed,domain=task['domain'],reason=repr(error),
            truth_cache=str((root/'truth.json').relative_to(ROOT)),distinct_inventory_policies=len(points)))
        raise
    return result


def collect(plan,manifest):
    from performance.mainline_safe_unsampled_competition import analyze_task,compact_task,aggregate_tasks
    from performance.summarize_mainline_safe_unsampled_validation import summarize,compact_summary,_costs
    output = ROOT/plan['output_dir']
    if (output/'compact_summary.json').exists():
        return read(output/'compact_summary.json')
    seal,cases = terminal.load_sealed(plan)
    inventory = read(directory(plan)/'inventory.json')
    assert inventory['status']=='sealed'
    task_results,contexts,truth_counts,compacts = [],{},{},[]
    # Check every task result/failure before beginning any combined summary.
    for task in plan['tasks']:
        folder = directory(plan)/f"seed_{task['geometry_task_seed']}"
        assert (folder/'result.json').exists() or (folder/'failure.json').exists(), 'Unfinished full-pool audit'
    with attempt(directory(plan)/'aggregate','aggregate-saved-full-pool') as cost:
        cost.update(ZERO)
        for task in plan['tasks']:
            seed = task['geometry_task_seed']
            folder = directory(plan)/f'seed_{seed}'
            root,source,initial,shared,bundles,selected = task_material(plan,task,cases)
            cache = cache_read(root/'truth.json')
            contexts[task['domain'],seed] = context(initial,shared,bundles,selected,cache)
            result = read(folder/('result.json' if (folder/'result.json').exists() else 'failure.json'))
            task_results.append(result)
            if result['status']=='complete':
                compacts.append(result['mechanism'])
            else:
                empty = {arm:dict(status='missing',points=[],audit={'stages':[]},history=initial) for arm in ROLES}
                compacts.append(compact_task(analyze_task(seed,task['domain'],empty,cache,initial,shared,selected,
                    [terminal.score_pair(case,contexts[task['domain'],seed]) for case in selected])))
            truth_counts[seed]=dict(seed=seed,domain=task['domain'],status=result['status'],
                cached_policies=len(cache),new_truth_function_calls=0,scope='Read-only cache; original calls in mechanism attempts')
        mechanism = aggregate_tasks(compacts)
        save(directory(plan)/'summary.json',dict(status='complete',tasks=compacts,aggregate=mechanism,
            audit_failures=[r for r in task_results if r['status']!='complete']))
    # Reuse the legacy terminal accounting, but with already saved truth only.
    base = terminal.score(plan,saved_contexts=contexts,saved_truth_counts=list(truth_counts.values()))
    if any(r['status']!='complete' for r in task_results):
        base['status']='complete_with_audit_failures'
        save(output/'summary.json',base)
    with attempt(directory(plan)/'primary_summary','summarize-registered-ap-old') as cost:
        cost.update(ZERO)
        mainline = summarize(plan,base)
        mainline['status'] = base['status']
        mainline['mechanism'] = dict(aggregate=mechanism,tasks=compacts)
        compact = compact_summary(mainline)
        compact['mechanism'] = dict(aggregate=mechanism,tasks=compacts)
    completion = runner.finalize_accounting(plan,manifest)
    audit_costs = dict(inventory_policy_task_units=inventory['distinct_policy_task_units'],
        registered_function_call_cap=inventory['maximum_analytic_function_calls'],
        new_truth_function_calls=sum(r.get('new_truth_function_calls',0) for r in completion['attempts']
            if r['stage']=='score-full-action-pool'),
        failures=[r for r in task_results if r['status']!='complete'],
        accounting='Included once in completion.new_work; do not add again.',
        registration_account=plan.get('registration_account',
            'paper_artifacts/mainline_safe_unsampled_validation_registration_v1/scheduler_execution.json'))
    mainline['costs'] = _costs(plan,base,completion,audit_costs)
    compact['costs'] = dict(mainline['costs'])
    compact['costs']['failed_attempt_count'] = len(compact['costs'].pop('failed_attempts'))
    save(output/'mainline_summary.json',mainline)
    save(output/'compact_summary.json',compact)
    return compact


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','task','collect'))
    parser.add_argument('--manifest',default=str(MANIFEST))
    parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    node=socket.gethostname().split('.')[0]
    assert node in {f'node{i:03d}' for i in range(1,7)}, node
    plan=read(args.manifest)
    if args.stage=='freeze':
        result=freeze(plan)
    elif args.stage=='task':
        result=task_score(plan,args.seed)
    else:
        result=collect(plan,args.manifest)
    print(result.get('status','complete'),flush=True)


if __name__=='__main__':
    main()

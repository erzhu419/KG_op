"""Reuse historical policy partitions for unresolved rules of either head."""

import argparse
import copy
import json
import signal
import time

import numpy as np

from core.conditional_terminal_kg import ConditionalTerminalActions
from core.raw_likelihood_lookahead import _actions
from core.raw_terminal_head_grid import evaluate_head_records
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.historical_batch_head_precision import action_key
from performance.historical_full_bank_batch_kg import restore
from performance.refine_historical_achievable_precision import PLAN, original_heads, head_checks, compose_records, all_head_checks

SHARDS = 6


def current_head(spec, job):
    case = next(c for c in spec['cases'] if c['name'] == job['case'])
    data = original_heads(spec, case, job['origin'])
    for h in (0, 1):
        for ordinal in range(len(data[h]['records'])):
            name = case['name']+'_o'+str(job['origin'])+'_h'+str(h)+'_p'+str(ordinal)
            path = ROOT/spec['output']/'shards'/name/'head_records.json'
            if path.exists():
                cached = read(str(path))
                assert cached['actions'] == data[h]['actions']
                data[h]['records'][ordinal] = cached['records'][0]
    return case, data


def partition_record(record, positions):
    row = copy.deepcopy(record)
    for kind in ('raw', 'projected'):
        for rule in row[kind].values():
            rule['statistics'] = np.asarray(rule['statistics'])[:, positions].tolist()
            rule['costs'] = np.asarray(rule['costs'])[:, positions].tolist()
    return row


def merge_rule(cached_actions, pieces, kind, power):
    positions = {action_key(a): i for i, a in enumerate(cached_actions)}
    assert len(positions) == len(cached_actions)
    coverage = []
    first = pieces[0]['records'][0][kind][str(power)]
    result = copy.deepcopy(first)
    n = len(first['statistics'])
    statistics = np.empty((n, len(positions), 4)); costs = np.empty((n, len(positions)))
    diagnostics = []
    for piece in pieces:
        ids = [positions[action_key(a)] for a in piece['actions']]
        coverage.extend(ids)
        rule = piece['records'][0][kind][str(power)]
        statistics[:, ids] = rule['statistics']; costs[:, ids] = rule['costs']
        if kind == 'raw': diagnostics.append(rule['diagnostics_by_identity'])
    assert sorted(coverage) == list(range(len(positions)))
    result.update(statistics=statistics.tolist(), costs=costs.tolist())
    if kind == 'raw':
        result['diagnostics_by_identity'] = [None if diagnostics[0][i] is None else dict(
            max((d[i] for d in diagnostics), key=lambda d:d['predictive_log_evidence_max_error']),
            effective_sample_size_min=min(d[i]['effective_sample_size_min'] for d in diagnostics))
            for i in range(n)]
        result['predictive_log_evidence_max_error'] = max(
            p['records'][0]['raw'][str(power)]['predictive_log_evidence_max_error'] for p in pieces)
        result['effective_sample_size_min'] = min(
            p['records'][0]['raw'][str(power)]['effective_sample_size_min'] for p in pieces)
    return result


def prepare(power, plan=PLAN):
    spec = read(plan); parent = read(spec['parent_plan']); jobs = []; contexts = {}; joint_failures = {}; checked={}
    for job in read(spec['output']+'/registered_jobs.json')['jobs']:
        path = spec['output']+'/shards/'+job['name']+'/results.json'
        execution = read(path)
        assert execution['complete'] or execution.get('cpu_budget_stopped')
        key = job['case'], job['origin']
        if key not in contexts:
            contexts[key] = current_head(spec, job)
            case, data = contexts[key]
            prior = read(parent['output']+'/'+case['state']+'/bank.json')['prior_weights']
            checked[key]=all_head_checks(data,prior,parent['maximum_action_resolution_difference'])
            f = data[0]['records'][:max(case['objective_prefixes'])]
            g = data[1]['records'][:max(parent['predictive_prefixes'])]
            joint_failures[key] = {}
            for kind in ('raw', 'projected'):
                high = compose_records(f,g,prior,kind,1)[0]
                delta = abs(high-compose_records(f,g,prior,kind,0)[0])
                head_delta = [abs(high-compose_records(f,g,prior,kind,levels)[0])
                    for levels in ((0,1),(1,0))]
                # A failure caused by f must not force another g refinement.
                # If the two small errors add up, refine the larger contributor.
                joint_failures[key][kind] = {h:set(np.flatnonzero(np.any(
                    (delta > parent['maximum_action_resolution_difference'])
                    & (head_delta[h] >= head_delta[1-h]), axis=(1-h,2))).tolist())
                    for h in (0, 1)}
        case, data = contexts[key]
        head = job['head']; row = data[head]['records'][job['ordinal']]
        prior = read(parent['output']+'/'+case['state']+'/bank.json')['prior_weights']
        checks = checked[key][head][job['ordinal']]
        raw = not (checks['raw_passed'] and checks['evidence_passed']) or job['ordinal'] in joint_failures[key]['raw'][head]
        native = not checks['projected_passed'] or job['ordinal'] in joint_failures[key]['projected'][head]
        if not (raw or native): continue
        assert not raw or power > max(map(int, row['raw']))
        assert not native or power > max(map(int, row['projected']))
        next_power = max(max(map(int,row[k])) for k,needed in [('raw',raw),('projected',native)] if needed)+2
        jobs.append(dict(job, raw=raw, native=native, initial_checks=checks, target_power=min(power,next_power)))
    directory = ROOT/spec['output']/('policy'+str(power)); directory.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for job in jobs:
        key = tuple(job[k] for k in ('case','origin','head','raw','native','target_power'))
        grouped.setdefault(key, []).append(job)
    groups = [dict(name='group'+str(i),jobs=rows) for i,rows in enumerate(grouped.values())]
    shards=spec.get('policy_shards',SHARDS)
    save(directory/'registered_jobs.json', dict(power=power, jobs=jobs, shards=shards, groups=groups))
    print(json.dumps(dict(power=power, records=len(jobs), tasks=shards*len(jobs),
        batched_tasks=shards*len(groups))), flush=True)
    return jobs


def run_group(name, power, shard, workers, plan=PLAN):
    """Share one posterior and Sobol grid across a group of head observations."""
    spec = read(plan); parent = read(spec['parent_plan'])
    registry=read(spec['output']+'/policy'+str(power)+'/registered_jobs.json')
    group = next(g for g in registry['groups'] if g['name']==name)
    root = ROOT/spec['output']/('policy'+str(power)); out = root/'groups'/name/('shard'+str(shard))
    out.mkdir(parents=True, exist_ok=True)
    pending = [j for j in group['jobs'] if not (root/j['name']/('shard'+str(shard))/'results.json').exists()
        or not read(str(root/j['name']/('shard'+str(shard))/'results.json'))['complete']]
    if not pending: return
    started = time.process_time(); handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_case_cpu_seconds']*len(pending))
    result = dict(name=name, shard=shard, complete=False, workers=min(workers,len(pending)),
        new_physical_calls=0, new_source_fits=0, new_verifier_calls=0)
    completed_cpu = 0.
    try:
        first = pending[0]; case = next(c for c in spec['cases'] if c['name']==first['case'])
        model, terminal, _, _ = restore(parent, case['state'])
        selected = np.arange(shard, len(terminal.shortlists), registry['shards'])
        terminal = ConditionalTerminalActions(terminal.shortlists[selected], terminal.incumbents[selected], terminal.protocol)
        actions = _actions(terminal, 360-case['n']); rows = []; by_position = {}
        for job in pending:
            cached = read(spec['output']+'/shards/'+job['name']+'/head_records.json'); row = cached['records'][0]
            full = {action_key(a):i for i,a in enumerate(cached['actions'])}
            rows.append(partition_record(row,[full[action_key(a)] for a in actions]))
            by_position[row['position']] = job, row['cpu_seconds']
        assert len(by_position)==len(rows)
        assert all(r['utility_identities']==rows[0]['utility_identities'] for r in rows)
        labels = {r['position']:r['label'] for r in rows}
        eta = None if case['n']==1 else {r['position']:r['eta_label'] for r in rows}
        target = first['target_power']; finished = set()
        def checkpoint(records, current_actions):
            nonlocal completed_cpu
            assert current_actions==actions
            for row in records:
                p = row['position']
                if p in finished: continue
                if any(str(target) not in row[k] for k,needed in [('raw',first['raw']),('projected',first['native'])] if needed): continue
                job, before_cpu = by_position[p]; directory = root/job['name']/('shard'+str(shard))
                directory.mkdir(parents=True,exist_ok=True)
                save(directory/'head_records.json',dict(records=[row],actions=actions,prior_weights=model.weights.tolist()))
                cpu = row['cpu_seconds']-before_cpu; completed_cpu += cpu
                save(directory/'results.json',dict(name=job['name'],power=target,phase_power=power,shard=shard,
                    complete=True,action_count=len(actions),raw=first['raw'],native=first['native'],
                    cpu_seconds=cpu,new_physical_calls=0,group=name))
                finished.add(p)
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=len(finished),total=len(rows),
                    unit='policy_head_states',metrics=dict(group=name,shard=shard))),flush=True)
        evaluate_head_records(model,terminal,360,parent['query'],first['head'],labels,list(labels),
            powers=[target] if first['raw'] else [],projected_powers=[target] if first['native'] else [],
            seed=parent['seed'],completed=rows,utility_identities=rows[0]['utility_identities'],
            guarded_mean_position=0,replicates=case['n'],eta_labels=eta,workers=workers,checkpoint=checkpoint)
        result.update(complete=len(finished)==len(rows),completed_records=len(finished))
    except TimeoutError: result['cpu_budget_stopped']=True
    finally:
        signal.setitimer(signal.ITIMER_PROF,0.); signal.signal(signal.SIGPROF,handler)
        result['cpu_seconds']=time.process_time()-started
        result['completed_record_cpu_seconds']=completed_cpu
        result['shared_cpu_seconds']=result['cpu_seconds']-completed_cpu
        save(out/'results.json',result)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)


def run(name, power, shard, plan=PLAN):
    started = time.process_time()
    spec = read(plan); parent = read(spec['parent_plan'])
    registry=read(spec['output']+'/policy'+str(power)+'/registered_jobs.json')
    job = next(j for j in registry['jobs'] if j['name']==name)
    target_power = job.get('target_power', power)
    out = ROOT/spec['output']/('policy'+str(power))/name/('shard'+str(shard)); out.mkdir(parents=True, exist_ok=True)
    if (out/'results.json').exists() and read(str(out/'results.json')).get('complete'): return
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_case_cpu_seconds'])
    result = dict(name=name, power=target_power, phase_power=power, shard=shard, complete=False, new_physical_calls=0)
    try:
        case = next(c for c in spec['cases'] if c['name']==job['case'])
        cached = read(spec['output']+'/shards/'+name+'/head_records.json')
        row = cached['records'][0]
        model, terminal, _, _ = restore(parent, case['state'])
        selected = np.arange(shard, len(terminal.shortlists), registry['shards'])
        terminal = ConditionalTerminalActions(terminal.shortlists[selected], terminal.incumbents[selected], terminal.protocol)
        full = {action_key(a):i for i,a in enumerate(cached['actions'])}
        actions = _actions(terminal, 360-case['n'])
        positions = [full[action_key(a)] for a in actions]
        partial = partition_record(row, positions)
        records = evaluate_head_records(model, terminal, 360, parent['query'], job['head'],
            {row['position']:row['label']}, [row['position']],
            powers=[target_power] if job['raw'] else [], projected_powers=[target_power] if job['native'] else [],
            seed=parent['seed'], completed=[partial], utility_identities=row['utility_identities'],
            guarded_mean_position=0, replicates=case['n'],
            eta_labels=None if row['eta_label'] is None else {row['position']:row['eta_label']},
            checkpoint=lambda r,a:save(out/'head_records.json',dict(records=r,actions=a,prior_weights=model.weights.tolist())))
        assert records[0]['head_log_scores'] == row['head_log_scores']
        result.update(complete=True, action_count=len(actions), raw=job['raw'], native=job['native'])
    except TimeoutError:
        result['cpu_budget_stopped'] = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
        result['cpu_seconds'] = time.process_time()-started; save(out/'results.json', result)
    print('SCHEDULEURM_DONE '+json.dumps(result), flush=True)


def merge(power, plan=PLAN):
    spec = read(plan); parent = read(spec['parent_plan']); executions = []; contexts = {}; updates=[]
    registry = read(spec['output']+'/policy'+str(power)+'/registered_jobs.json')
    for job in registry['jobs']:
        directory = ROOT/spec['output']/('policy'+str(power))/job['name']
        marker = directory/'merged.json'
        if marker.exists(): executions.extend(read(str(marker))['executions']); continue
        pieces = [read(str(directory/('shard'+str(i))/'head_records.json')) for i in range(registry['shards'])]
        costs = [read(str(directory/('shard'+str(i))/'results.json')) for i in range(registry['shards'])]
        assert all(r['complete'] for r in costs)
        key = job['case'], job['origin']
        if key not in contexts: contexts[key] = current_head(spec, job)
        case, data = contexts[key]; head = job['head']; row = data[head]['records'][job['ordinal']]
        for piece in pieces:
            r = piece['records'][0]
            assert all(r[k] == row[k] for k in ('position','head_log_scores','label','eta_label','utility_identities'))
            assert piece['prior_weights'] == pieces[0]['prior_weights']
        for kind, needed in [('raw', job['raw']), ('projected', job['native'])]:
            if needed:
                target_power = job.get('target_power',power)
                row[kind][str(target_power)] = merge_rule(data[head]['actions'], pieces, kind, target_power)
        point = ROOT/spec['output']/'shards'/job['name']
        old = read(str(point/'results.json'))
        save(point/('results_before_policy'+str(power)+'.json'), old)
        save(point/'head_records.json',dict(records=[row],actions=data[head]['actions'],prior_weights=pieces[0]['prior_weights']))
        updates.append((key,job,point,marker,old,costs,pieces[0]['prior_weights']))
    checked={}
    for key,job,point,marker,old,costs,prior in updates:
        if key not in checked:
            checked[key]=all_head_checks(contexts[key][1],prior,parent['maximum_action_resolution_difference'])
        checks=checked[key][job['head']][job['ordinal']]
        passed = checks['raw_passed'] and checks['projected_passed'] and checks['evidence_passed']
        result = dict(old,complete=True,checks=checks,numerical_diagnostics_passed=passed,
            old_caches_preserved=True,cpu_seconds=old['cpu_seconds']+sum(r['cpu_seconds'] for r in costs),
            policy_refinement_powers=old.get('policy_refinement_powers',[])+[job.get('target_power',power)])
        result.pop('cpu_budget_stopped',None)
        save(point/'results.json',result)
        save(marker,dict(checks=checks,executions=costs)); executions.extend(costs)
    result = dict(power=power, records=len(registry['jobs']), executions=executions,
        cpu_seconds=sum(r['cpu_seconds'] for r in executions), new_physical_calls=0)
    save(ROOT/spec['output']/('policy'+str(power))/'execution_summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='executions'}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','case','group','merge'])
    parser.add_argument('--name');parser.add_argument('--power',type=int,required=True)
    parser.add_argument('--shard',type=int);parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--plan',default=PLAN);args=parser.parse_args()
    if args.mode=='prepare':prepare(args.power,args.plan)
    elif args.mode=='merge':merge(args.power,args.plan)
    elif args.mode=='group':run_group(args.name,args.power,args.shard,args.workers,args.plan)
    else:run(args.name,args.power,args.shard,args.plan)

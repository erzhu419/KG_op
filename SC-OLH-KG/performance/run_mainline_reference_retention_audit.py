"""Freeze references from saved inputs, then audit completed terminal crosses."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import socket
import time
import traceback

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.run_mainline_safe_unsampled_audit import ZERO


MANIFEST = 'performance/manifests/mainline_reference_retention_v1.json'


def register(plan, manifest):
    output = ROOT/plan['output_dir']
    if (output/'registration/result.json').exists():
        return read(output/'registration/result.json')
    with attempt(output/'registration', 'freeze-initial-references-without-truth') as cost:
        cost.update(ZERO)
        source = read(ROOT/plan['reference_manifest'])
        completed = read(ROOT/plan['reference_output_dir']/'completion.json')
        assert source['status'] == completed['status'] == 'complete'
        assert completed['complete_task_streams'] == 48
        assert completed['registered_terminal_outcomes'] == 864
        assert source['tasks'] == plan['tasks'] and source['terminal_arms'] == plan['terminal_arms']
        from performance.mainline_reference_retention import freeze_references
        references = []
        for stream in plan['streams']:
            child = read(ROOT/stream['manifest'])
            for task in plan['tasks']:
                path = output/'registration'/f"seed_{task['seed']}"/(stream['name']+'.json')
                if not path.exists():
                    frozen = freeze_references(ROOT, child, task['seed'], stream=stream['name'])
                    save(path, frozen)
                references.append(dict(seed=task['seed'], stream=stream['name'],
                    file=str(path.relative_to(ROOT))))
        result = dict(status='registered', references=references,
            registered_task_streams=48, registered_terminal_outcomes=864,
            reference_identity_uses_truth=False, reference_identity_uses_verification=False, **ZERO)
        save(output/'registration/result.json', result)
        plan.update(status='registered', registered_utc=datetime.now(timezone.utc).isoformat())
        save(ROOT/manifest, plan)
    return result


def geometry_worker(args):
    manifest, seed = args
    plan = read(ROOT/manifest)
    task = next(t for t in plan['tasks'] if t['seed'] == seed)
    output = ROOT/plan['output_dir']
    outcomes = []
    for stream in plan['streams']:
        directory = output/'cases'/f'seed_{seed}'/stream['name']
        path = directory/'result.json'
        if path.exists():
            result = read(path)
        else:
            frozen = {}
            try:
                with attempt(directory, 'attribute-initial-reference-retention', task['domain']) as cost:
                    cost.update(ZERO, seed=seed, stream=stream['name'])
                    from performance.mainline_reference_retention import analyze_task
                    frozen = read(output/'registration'/f'seed_{seed}'/(stream['name']+'.json'))
                    previous = read(ROOT/plan['reference_output_dir']/'cases'/f'seed_{seed}'/stream['name']/'result.json')
                    result = analyze_task(ROOT, previous, frozen)
                    save(path, result)
            except Exception:
                result = dict(status='failed', seed=seed, domain=task['domain'], stream=stream['name'],
                    arms=[], references=[{k:v for k,v in r.items() if k != 'point'}
                        for r in frozen.get('references', [])], error=traceback.format_exc())
                save(path, result)
        outcomes.append(dict(seed=seed, stream=stream['name'], status=result['status']))
    return outcomes


def batch(plan, manifest, start, end, workers):
    output = ROOT/plan['output_dir']
    assert (output/'registration/result.json').exists()
    assert 0 <= start < end <= len(plan['tasks']) == 24 and workers > 0
    shard = output/'dispatch'/f'shard_{start}_{end}'
    assert not (shard/'result.json').exists(), 'Preserve the completed analysis shard'
    cpu, wall = time.process_time(), time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=min(workers, end-start)) as executor:
        pending = {executor.submit(geometry_worker, (manifest, t['seed'])):t for t in plan['tasks'][start:end]}
        for future in as_completed(pending):
            task = pending[future]
            try:
                rows.extend(future.result())
            except Exception:
                error = traceback.format_exc()
                for stream in plan['streams']:
                    path = output/'cases'/f"seed_{task['seed']}"/stream['name']/'result.json'
                    if not path.exists():
                        frozen = read(output/'registration'/f"seed_{task['seed']}"/(stream['name']+'.json'))
                        save(path, dict(status='failed', seed=task['seed'], domain=task['domain'],
                            stream=stream['name'], arms=[], error=error, cpu_seconds=None,
                            references=[{k:v for k,v in r.items() if k != 'point'} for r in frozen['references']]))
                    rows.append(dict(seed=task['seed'], stream=stream['name'], status=read(path)['status']))
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=len(rows)//2, total=end-start,
                unit='geometry', elapsed_seconds=time.monotonic()-wall)), flush=True)
    result = dict(status='finished', outcomes=rows, node=socket.gethostname(),
        coordinator_cpu_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall,
        failures=[r for r in rows if r['status'] != 'complete'], **ZERO)
    save(shard/'result.json', result)
    return result


def collect(plan, manifest):
    output = ROOT/plan['output_dir']
    if (output/'completion.json').exists():
        return read(output/'completion.json')
    tasks = []
    for task in plan['tasks']:
        for stream in plan['streams']:
            path = output/'cases'/f"seed_{task['seed']}"/stream['name']/'result.json'
            assert path.exists(), 'Every registered stream needs a result or a retained failure'
            tasks.append(read(path))
    with attempt(output/'aggregation', 'aggregate-same-history-terminal-reference-effects') as cost:
        cost.update(ZERO)
        from performance.summarize_mainline_reference_retention import aggregate, compact_summary
        scientific = read(ROOT/plan['scientific_output_dir']/'summary.json')
        result = aggregate(tasks, plan, scientific_summary=scientific)
        save(output/'summary.json', result)
        save(output/'compact_summary.json', compact_summary(result))
    attempts = [dict(json.loads(line), ledger=str(path.relative_to(ROOT)))
        for scope in ('registration','cases','aggregation')
        for path in (output/scope).rglob('attempts.jsonl')
        for line in path.read_text().splitlines() if line.strip()]
    shards = [read(path) for path in (output/'dispatch').glob('shard_*/*result*.json')]
    checks = [read(path) for path in sorted((output/'cluster_control/validation').glob('attempt*/checks.json'))]
    result = dict(status=result['status'], completed_utc=datetime.now(timezone.utc).isoformat(),
        registered_task_streams=48, complete_task_streams=sum(t['status']=='complete' for t in tasks),
        registered_terminal_outcomes=864,
        failures=[{k:t.get(k) for k in ('seed','stream','error')} for t in tasks if t['status'] != 'complete'],
        attempts=attempts, failed_attempts=[a for a in attempts if a['status']=='failed'],
        recorded_analysis_cpu_seconds=sum(a['cpu_seconds'] for a in attempts)
            +sum(s['coordinator_cpu_seconds'] for s in shards),
        validations=checks, new_work=dict(ZERO), historical_accounts=plan['historical_accounts'],
        accounting='Attempt and coordinator CPU counted once; validation separate; wrapper overlaps. Historical science and failed analyses stay in referenced ledgers.')
    save(output/'completion.json', result)
    save(output/'account_summary.json', {k:v for k,v in result.items() if k not in ('attempts','failed_attempts')})
    plan.update(status=result['status'], completed_utc=result['completed_utc'],
        summary_file=plan['output_dir']+'/summary.json', completion_file=plan['output_dir']+'/completion.json')
    save(ROOT/manifest, plan)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('register','batch','collect'))
    parser.add_argument('--manifest', default=MANIFEST)
    for field in ('start','end','workers'):
        parser.add_argument('--'+field, type=int)
    args = parser.parse_args()
    assert socket.gethostname().split('.')[0] in {f'node{i:03d}' for i in range(1,7)}
    plan = read(ROOT/args.manifest)
    result = register(plan,args.manifest) if args.stage == 'register' else (
        batch(plan,args.manifest,args.start,args.end,args.workers) if args.stage == 'batch' else collect(plan,args.manifest))
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()

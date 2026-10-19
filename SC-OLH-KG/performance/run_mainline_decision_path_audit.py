"""Analyze the two completed streams from saved files only, on scheduler nodes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import time
import traceback

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.run_mainline_safe_unsampled_audit import ZERO


MANIFEST = 'performance/manifests/mainline_decision_path_attribution_v1.json'


def register(plan, manifest):
    output = ROOT/plan['output_dir']
    if (output/'registration/result.json').exists():
        return read(output/'registration/result.json')
    with attempt(output/'registration', 'register-frozen-decision-path-analysis') as cost:
        cost.update(ZERO)
        parent = read(ROOT/plan['reference_manifest'])
        source = ROOT/plan['reference_output_dir']
        completion = read(source/'completion.json')
        assert parent['status'] == 'complete' and completion['completed_outcome_cells'] == 864
        corrected = read(source/'terminal_recording/revision_1.json')
        assert corrected['scientific_results_unchanged']
        expected = [(t['seed'], t['domain']) for t in plan['tasks']]
        assert len(expected) == 24 and len(set(expected)) == 24
        assert len(plan['streams']) == 2 and len(plan['terminal_arms']) == 9
        streams = []
        for descriptor in plan['streams']:
            child = read(ROOT/descriptor['manifest'])
            assert [(t['geometry_task_seed'], t['domain']) for t in child['tasks']] == expected
            assert child['terminal_arms'] == plan['terminal_arms']
            seal = read(ROOT/child['output_dir']/'execution_status.json')
            assert seal['status'] == 'sealed' and seal['registered_terminal_cells'] == 432
            record = ROOT/descriptor['terminal_recording_file']
            coverage = read(record.with_name('summary.json'))['coverage']
            assert coverage['registered_pairs'] == coverage['valid_records'] == 216
            assert coverage['issue_pairs'] == 0
            streams.append(dict(name=descriptor['name'], coverage=coverage,
                terminal_recording_file=descriptor['terminal_recording_file']))
        result = dict(status='registered', registered_task_streams=48,
            registered_terminal_outcomes=864, streams=streams, **ZERO)
        save(output/'registration/result.json', result)
        plan.update(status='registered', registered_utc=datetime.now(timezone.utc).isoformat())
        save(ROOT/manifest, plan)
    return result


def geometry_worker(args):
    manifest, seed = args
    plan = read(ROOT/manifest)
    task = next(t for t in plan['tasks'] if t['seed'] == seed)
    output = ROOT/plan['output_dir']/'cases'/f'seed_{seed}'
    outcomes = []
    for descriptor in plan['streams']:
        directory = output/descriptor['name']
        result_file = directory/'result.json'
        if result_file.exists():
            result = read(result_file)
        else:
            try:
                with attempt(directory, 'attribute-saved-decision-paths', task['domain']) as cost:
                    cost.update(ZERO, seed=seed, stream=descriptor['name'])
                    from performance.mainline_decision_path_attribution import analyze_task
                    child = read(ROOT/descriptor['manifest'])
                    result = analyze_task(ROOT, child, seed, stream=descriptor['name'],
                        terminal_recording_file=ROOT/descriptor['terminal_recording_file'])
                    save(result_file, result)
            except Exception:
                result = dict(status='failed', seed=seed, domain=task['domain'],
                    stream=descriptor['name'], cells=[], error=traceback.format_exc())
                save(result_file, result)
        outcomes.append(dict(seed=seed, stream=descriptor['name'], status=result['status']))
    return outcomes


def batch(plan, manifest, start, end, workers):
    assert (ROOT/plan['output_dir']/'registration/result.json').exists()
    assert 0 <= start < end <= len(plan['tasks']) == 24 and workers > 0
    output = ROOT/plan['output_dir']/'dispatch'/f'shard_{start}_{end}'
    assert not (output/'result.json').exists(), 'Retain the completed analysis shard'
    cpu, wall = time.process_time(), time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=min(workers, end-start)) as executor:
        pending = {executor.submit(geometry_worker, (manifest, t['seed'])):t
                   for t in plan['tasks'][start:end]}
        for future in as_completed(pending):
            task = pending[future]
            try:
                rows.extend(future.result())
            except Exception:
                error = traceback.format_exc()
                for stream in plan['streams']:
                    path = ROOT/plan['output_dir']/'cases'/f"seed_{task['seed']}"/stream['name']/'result.json'
                    if not path.exists():
                        save(path, dict(status='failed', seed=task['seed'], domain=task['domain'],
                            stream=stream['name'], cells=[], error=error,
                            cpu_seconds=None, accounting='Worker process failure; wrapper retains process CPU.'))
                    saved = read(path)
                    rows.append(dict(seed=task['seed'], stream=stream['name'], status=saved['status']))
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=len(rows)//2, total=end-start,
                unit='geometry', elapsed_seconds=time.monotonic()-wall)), flush=True)
    result = dict(status='finished', outcomes=rows, node=socket.gethostname(),
        coordinator_cpu_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall,
        failures=[r for r in rows if r['status'] != 'complete'], **ZERO)
    save(output/'result.json', result)
    return result


def collect(plan, manifest):
    output = ROOT/plan['output_dir']
    if (output/'completion.json').exists():
        return read(output/'completion.json')
    tasks = []
    for task in plan['tasks']:
        for stream in plan['streams']:
            path = output/'cases'/f"seed_{task['seed']}"/stream['name']/'result.json'
            assert path.exists(), 'Every registered task-stream requires a saved result or failure'
            tasks.append(read(path))
    with attempt(output/'aggregation', 'aggregate-saved-decision-paths') as cost:
        cost.update(ZERO)
        from performance.summarize_mainline_decision_path_attribution import aggregate, compact_summary
        scientific = read(ROOT/plan['reference_output_dir']/'summary.json')
        result = aggregate(tasks, plan, scientific_summary=scientific)
        save(output/'summary.json', result)
        save(output/'compact_summary.json', compact_summary(result))
    attempts = [dict(json.loads(line), ledger=str(path.relative_to(ROOT)))
        for scope in ('registration', 'cases', 'aggregation')
        for path in (output/scope).rglob('attempts.jsonl')
        for line in path.read_text().splitlines() if line.strip()]
    shards = [read(path) for path in (output/'dispatch').glob('shard_*/*result*.json')]
    checks = read(output/'cluster_control/validation/attempt1/checks.json')
    extra_checks = [read(path) for path in sorted((output/'cluster_control/validation').glob('attempt*/checks.json'))
        if path.parent.name != 'attempt1']
    completion = dict(status=result['status'], completed_utc=datetime.now(timezone.utc).isoformat(),
        registered_task_streams=48, complete_task_streams=sum(t['status']=='complete' for t in tasks),
        registered_terminal_outcomes=864, failures=[{k:t.get(k) for k in ('seed','stream','error')}
            for t in tasks if t['status'] != 'complete'],
        attempts=attempts, failed_attempts=[a for a in attempts if a['status']=='failed'],
        recorded_analysis_cpu_seconds=sum(a['cpu_seconds'] for a in attempts)
            +sum(s['coordinator_cpu_seconds'] for s in shards),
        validation=checks, additional_validation=extra_checks,
        new_work=dict(ZERO), historical_accounts=plan['historical_accounts'],
        accounting='Analysis attempt and coordinator CPU are added once; validation is separate. Scheduler wrapper CPU overlaps subprocess CPU. Historical scientific spending is referenced, not recharged.')
    save(output/'completion.json', completion)
    plan.update(status=result['status'], completed_utc=completion['completed_utc'],
        summary_file=plan['output_dir']+'/summary.json', completion_file=plan['output_dir']+'/completion.json')
    save(ROOT/manifest, plan)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('register','batch','collect'))
    parser.add_argument('--manifest', default=MANIFEST)
    for field in ('start','end','workers'):
        parser.add_argument('--'+field, type=int)
    args = parser.parse_args()
    assert socket.gethostname().split('.')[0] in {f'node{i:03d}' for i in range(1,7)}
    plan = read(ROOT/args.manifest)
    if args.stage == 'register':
        result = register(plan, args.manifest)
    elif args.stage == 'batch':
        result = batch(plan, args.manifest, args.start, args.end, args.workers)
    else:
        result = collect(plan, args.manifest)
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()

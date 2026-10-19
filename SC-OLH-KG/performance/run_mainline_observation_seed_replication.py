"""Run registered task/observation streams with global execution barriers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from performance import run_mainline_terminal_equal as verifier
from performance.mainline_modified_terminal import prepare as prepare_terminal, PROTOCOL
from performance.run_mainline_legacy_backend_bridge import ROOT, read, save
from performance.run_mainline_terminal_confirmation import native_terminal, costs, ROLES

MANIFEST = ROOT / 'performance/manifests/mainline_observation_seed_replication_v1.json'
STOP = 'initialization_only'
STRUCTURAL = 'current_structural_mixture'


def materialize(plan):
    """Write only child manifests; no models, observations or target truth are read."""
    output = ROOT / plan['output_dir']
    if (output / 'prepared.json').exists():
        return read(output / 'prepared.json')
    parent = read(ROOT / plan['source_manifest'])
    replicas = []
    for replica in plan['replicates']:
        families = []
        for family in parent['families']:
            domain = family['domain']
            tasks = [t for t in plan['tasks'] if t['domain'] == domain]
            seeds = [t['geometry_task_seed'] for t in tasks]
            source = deepcopy(read(ROOT / family['search_manifest']))
            terminal = deepcopy(read(ROOT / family['terminal_manifest']))
            directory = Path(replica['output_dir']) / domain
            source_path, terminal_path = directory / 'search_plan.json', directory / 'terminal_plan.json'
            for child in (source, terminal):
                for key in ('execution_failures', 'failure_resolution', 'started_utc', 'completed_utc'):
                    child.pop(key, None)
                child.update(status='registered', target_seeds=seeds,
                             registered_utc=plan['registered_utc'])
            source.update(name=f"observation_replica_{replica['replicate']}_{domain}_search",
                output_dir=str(directory / 'search'),
                geometry_by_seed={str(t['geometry_task_seed']): t['geometry'] for t in tasks},
                observation_seed_by_task={str(s): replica['seed_by_task'][str(s)]['observation_seed'] for s in seeds},
                arms=[dict(name=a, online=a != STOP) for a in plan['search_arms']],
                initialization=plan['frozen']['initial_policies'],
                streams=plan['randomness'], barriers=plan['barriers'],
                geometry=plan.get('geometry_description',
                    'All registered Stage50 geometries reused; no new geometry draw.'),
                continuation=plan['objective'])
            if 'geometry_seed' in plan:
                source['geometry_seed'] = plan['geometry_seed']
            for mapping, field in (('fixed_initialization_file_by_task', 'initial_policy_file'),
                                   ('fixed_neighbors_file_by_task', 'shared_neighbors_file')):
                source.pop(mapping, None)
                frozen_files = {str(t['geometry_task_seed']): t[field] for t in tasks if field in t}
                if frozen_files:
                    source[mapping] = frozen_files
            terminal.update(name=f"observation_replica_{replica['replicate']}_{domain}_terminal",
                output_dir=str(directory / 'terminal'), source_manifest=str(source_path),
                source_output_dir=source['output_dir'],
                verification_seed_by_task={str(s): replica['seed_by_task'][str(s)]['verification_seed'] for s in seeds},
                arms=plan['terminal_arms'], protocols={PROTOCOL: [80, 128, 128]},
                inherited_replay_check_file=str(Path(read(ROOT / family['terminal_manifest'])['output_dir']) / 'replay_check.json'),
                statistical_unit=plan.get('statistical_unit',
                    'Fixed geometry with two new observation/verification streams; paired cells are not independent.'),
                scope=plan['objective'], truth_barrier=plan['barriers']['before_scoring'])
            save(ROOT / source_path, source)
            save(ROOT / terminal_path, terminal)
            families.append(dict(domain=domain, search_manifest=str(source_path), terminal_manifest=str(terminal_path)))
        replicas.append(dict(replicate=replica['replicate'], output_dir=replica['output_dir'], families=families))
    prepared = dict(status='prepared', replicates=replicas, new_target_calls=0,
                    models_loaded=0, truth_read=False)
    save(output / 'prepared.json', prepared)
    return prepared


def run_stage(output, name, jobs, workers):
    """Run one global barrier stage; callers decide how recorded failures proceed."""
    logs = output / 'execution_logs'; logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=str(ROOT))

    def invoke(job):
        with (logs / f"{job['label']}.log").open('a') as stream:
            process = subprocess.run(job['command'], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        return job, process.returncode

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, future in enumerate(as_completed([pool.submit(invoke, job) for job in jobs]), 1):
            job, code = future.result()
            if code:
                failed.append(dict(job, returncode=code))
            save(output / 'progress.json', dict(stage=name, completed=count, registered_jobs=len(jobs)))
    print(name, 'completed', len(jobs), 'failed', len(failed), flush=True)
    return failed


def search_jobs(prepared, initial):
    jobs = []
    for replica in prepared['replicates']:
        for family in replica['families']:
            source = read(ROOT / family['search_manifest'])
            failed = {(r['seed'], r['arm']) for r in source.get('execution_failures', [])}
            for seed in source['target_seeds']:
                for arm in source['arms']:
                    name = arm['name']
                    if (name == STOP) != initial or (seed, name) in failed:
                        continue
                    directory = ROOT / source['output_dir'] / f'seed_{seed}' / name
                    if (directory / 'search.json').exists():
                        continue
                    jobs.append(dict(label=f"r{replica['replicate']}_search_{seed}_{name}", seed=seed, arm=name,
                        search_manifest=family['search_manifest'], terminal_manifest=family['terminal_manifest'],
                        replicate=replica['replicate'], command=[sys.executable, '-u', '-m',
                        'performance.run_mainline_role_fresh', 'search', '--manifest', family['search_manifest'],
                        '--seed', str(seed), '--arm', name]))
    return jobs


def retain_structural_failure(job):
    """Keep the registered failing reference and its paid prefix, without retrying."""
    if job['arm'] != STRUCTURAL:
        raise RuntimeError(f"Search failed; preserve paid state before continuing: {job['label']}")
    source = read(ROOT / job['search_manifest'])
    directory = ROOT / source['output_dir'] / f"seed_{job['seed']}" / job['arm']
    raw_file = sorted(directory.glob('failed_search_raw_*.json'))[-1]
    observed = read(raw_file)
    ledger = [json.loads(line) for line in (directory / 'attempts.jsonl').read_text().splitlines()]
    record = dict(domain=source['domain'], seed=job['seed'], arm=job['arm'], status='search_failed',
        logical_search_calls=10+len(observed), new_search_calls=len(observed),
        reason=ledger[-1]['error'], failed_raw_file=str(raw_file.relative_to(ROOT)),
        failed_state_file=str((directory / 'failed_search_engine.pkl').relative_to(ROOT)))
    assert (directory / 'failed_search_engine.pkl').exists()
    for path in (job['search_manifest'], job['terminal_manifest']):
        child = read(ROOT / path)
        child.setdefault('execution_failures', []).append(record)
        save(ROOT / path, child)
    return dict(replicate=job['replicate'], **record)


def run(plan, workers, manifest_path=MANIFEST):
    assert plan['execution_preparation']['status'] == 'validated'
    prepared = materialize(plan)
    output = ROOT / plan['output_dir']
    plan.update(status='running', execution_started=True)
    save(manifest_path, plan)
    failures = run_stage(output, 'initialization', search_jobs(prepared, True), workers)
    if failures:
        raise RuntimeError('Initialization failed; preserve paid raw and state before continuing')
    children = [(replica, family) for replica in prepared['replicates'] for family in replica['families']]
    jobs = [dict(label=f"r{replica['replicate']}_banks_{family['domain']}", command=[sys.executable, '-u', '-m',
        'performance.run_mainline_role_fresh', 'prepare', '--manifest', family['search_manifest']])
        for replica, family in children]
    if run_stage(output, 'fixed-neighbors', jobs, workers):
        raise RuntimeError('Fixed-neighbor preparation failed')
    failures = run_stage(output, 'online-search', search_jobs(prepared, False), workers)
    for job in failures:
        record = retain_structural_failure(job)
        plan.setdefault('execution_failures', []).append(record)
        save(manifest_path, plan)
    jobs = []
    for replica, family in children:
        source = read(ROOT / family['search_manifest'])
        for seed in source['target_seeds']:
            for arm in ROLES:
                if (ROOT / source['output_dir'] / f'seed_{seed}' / arm / 'native_terminal.json').exists():
                    continue
                jobs.append(dict(label=f"r{replica['replicate']}_native_{seed}_{arm}", command=[sys.executable, '-u', '-m',
                    'performance.run_mainline_observation_seed_replication', 'native',
                    '--child-manifest', family['search_manifest'], '--seed', str(seed), '--arm', arm]))
    if run_stage(output, 'native-terminals', jobs, workers):
        raise RuntimeError('Native terminal freezing failed')
    for _, family in children:
        source = read(ROOT / family['search_manifest'])
        save(ROOT / source['output_dir'] / 'completion.json', dict(status='search_complete', **costs(ROOT / source['output_dir'])))
        prepare_terminal(source, read(ROOT / family['terminal_manifest']))
    jobs = []
    for replica, family in children:
        terminal = read(ROOT / family['terminal_manifest'])
        for seed, arm, protocol in verifier.cases(terminal):
            if (verifier.case_dir(ROOT / terminal['output_dir'], seed, arm, protocol) / 'verification.json').exists():
                continue
            jobs.append(dict(label=f"r{replica['replicate']}_verify_{seed}_{arm}", command=[sys.executable, '-u', '-m',
                'performance.run_mainline_observation_seed_replication', 'verify',
                '--child-manifest', family['terminal_manifest'], '--seed', str(seed), '--arm', arm]))
    if run_stage(output, 'verification', jobs, workers):
        raise RuntimeError('Verification failed; recover saved paid raw before continuing')
    from performance.summarize_mainline_observation_seed_replication import score
    result = score(plan, prepared)
    total = costs(output)
    failed_logical = sum(r['logical_search_calls'] for r in plan.get('execution_failures', []))
    total.update(status='complete_with_search_failure' if failed_logical else 'complete',
        new_physical_target_calls=sum(total['cost'][k] for k in
            ('new_target_initial_calls', 'new_target_search_calls', 'new_verification_calls')),
        logical_target_calls=sum(r['logical_total_calls'] for r in result['rows']
                                 if r['logical_total_calls'] is not None) + failed_logical,
        execution_failures=plan.get('execution_failures', []),
        completed_utc=datetime.now(timezone.utc).isoformat())
    assert total['new_physical_target_calls'] <= plan['budget']['maximum_target_physical_calls']
    assert total['logical_target_calls'] <= plan['budget']['maximum_logical_target_calls']
    save(output / 'completion.json', total)
    plan.update(status=total['status'], completed_utc=total['completed_utc'])
    save(manifest_path, plan)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'run', 'native', 'verify'))
    parser.add_argument('--manifest', default=str(MANIFEST))
    parser.add_argument('--child-manifest'); parser.add_argument('--seed', type=int); parser.add_argument('--arm')
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    if args.stage in ('prepare', 'run'):
        plan = read(args.manifest)
        if args.stage == 'prepare':
            materialize(plan)
        else:
            run(plan, args.workers, Path(args.manifest))
    else:
        child = read(ROOT / args.child_manifest)
        if args.stage == 'native':
            assert args.seed in child['target_seeds'] and args.arm in ROLES
            native_terminal(child, args.seed, args.arm)
        else:
            assert (args.seed, args.arm, PROTOCOL) in verifier.cases(child)
            verifier.verify(ROOT / child['output_dir'], child, args.seed, args.arm, PROTOCOL)


if __name__ == '__main__':
    main()

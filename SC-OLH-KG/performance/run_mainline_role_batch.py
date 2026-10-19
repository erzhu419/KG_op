"""Run a registered role-prior batch through the existing experiment barriers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import os
import subprocess
import sys

from performance.run_mainline_legacy_backend_bridge import ROOT, read, save


def run(manifest, workers):
    plan = read(manifest)
    output = ROOT/plan['output_dir']
    logs = output/'execution_logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONPATH=str(ROOT))
    runner = [sys.executable, '-u', 'performance/run_mainline_role_fresh.py']
    manifest_args = ['--manifest', str(manifest)]

    def invoke(label, command):
        with (logs/f'{label}.log').open('a') as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        return label, result.returncode

    def stage(name, jobs):
        failures = []
        print(name, 'started', len(jobs), flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(invoke, label, command) for label,command in jobs]
            for count,future in enumerate(as_completed(futures), 1):
                label, code = future.result()
                print(name, count, '/', len(jobs), label, 'exit', code, flush=True)
                if code:
                    failures.append(label)
        if failures:
            raise RuntimeError(f'{name} failed; paid stages are not retried: {failures}')

    names = [a['name'] for a in plan['arms']]
    seeds = plan['target_seeds']
    def jobs(operation, arms):
        return [(f'{operation}_{seed}_{arm}', runner+[operation, '--seed', str(seed),
                    '--arm', arm]+manifest_args) for seed in seeds for arm in arms]

    stage('initialization', jobs('search', ['initialization_only']))
    stage('neighbors', [('prepare', runner+['prepare']+manifest_args)])
    stage('online-search', jobs('search', [a for a in names if a!='initialization_only']))
    stage('verification', jobs('verify', names))
    stage('truth-scoring', [('summarize', [sys.executable, '-u',
        'performance/summarize_mainline_role_fresh.py']+manifest_args)])
    stage('completion', [('complete', runner+['complete']+manifest_args)])
    stage('decision-summary', [('decisions', [sys.executable, '-u',
        'performance/summarize_mainline_role_decisions.py']+manifest_args)])
    stage('final-ledger', [('complete-final', runner+['complete']+manifest_args)])
    completion = read(output/'completion.json')
    cost = completion['cost']
    plan.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
        completion_file=str((output/'completion.json').relative_to(ROOT)),
        execution=dict(completed_initializations=len(seeds), completed_searches=len(seeds)*len(names),
            completed_verifications=len(seeds)*len(names), new_target_observations_started=True,
            new_initial_calls=cost['new_target_initial_calls'], new_search_calls=cost['new_target_search_calls'],
            new_verification_calls=cost['new_verification_calls'],
            new_physical_target_calls=completion['new_physical_target_calls'],
            logical_target_calls=completion['logical_target_calls'],
            new_truth_function_calls=cost['new_truth_function_calls'],
            source_fits=cost['new_source_fits'], downloads=cost['download_bytes'],
            failed_attempts=len(completion['failed_attempts'])))
    save(manifest, plan)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=lambda value: ROOT/value, required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    run(args.manifest, args.workers)

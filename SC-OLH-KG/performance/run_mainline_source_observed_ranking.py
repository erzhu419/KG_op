"""Frozen source-only initial10+3 histories, with a global forecast/score barrier."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from baselines.transfer_archive import FrozenTransferArchive
from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.mainline_legacy_source_models import fit_source_models, SourceFitError
from performance.mainline_role_source_holdout import fit_native
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.run_mainline_source_policy_holdout import training_archive

MANIFEST = ROOT/'performance/manifests/mainline_source_observed_ranking_v1.json'


def frozen_split(plan, archive, fold):
    prior = read(ROOT/plan['partition_manifest'])
    split = read(ROOT/prior['output_dir']/f'fold_{fold}'/'split.json')
    np.testing.assert_array_equal(archive.tasks[0].X_integer, archive.tasks[1].X_integer)
    assert len(set(map(tuple, archive.tasks[0].X_integer))) == 64
    assert len(split['training']) == 48 and len(split['held']) == 16
    assert set(split['training']).isdisjoint(split['held'])
    assert sorted(split['training']+split['held']) == list(range(64))
    return split


def run_fold(plan, heldout, fold):
    from performance.mainline_source_observed_ranking import forecast
    source = next(row for row in plan['source_archives'] if row['heldout_target_family'] == heldout)
    output = ROOT/plan['output_dir']/heldout/f'fold_{fold}'
    if (output/'forecasts.json').exists():
        print('reuse frozen fold', heldout, fold, flush=True)
        return
    cases = []
    with attempt(output, 'source-observed-fit-and-forecast', heldout) as cost:
        cost.update(fold=fold, new_source_fits=0, current_bundle_fits=0, native_prior_fits=0,
            new_target_initial_calls=0, new_truth_function_calls=0, download_bytes=0,
            paired_raw_adaptation_exposures=0)
        try:
            archive = FrozenTransferArchive.load(ROOT/source['path'])
            assert [task.name for task in archive.tasks] == source['source_families']
            split = frozen_split(plan, archive, fold)
            save(output/'split.json', split)
            training = training_archive(archive, split['training'])
            assert training.simulator_calls == 288
            providers = {task.name: no_outcomes(build_scalarized_problem(
                task.name, task.X.shape[1], 100, .04, .05, [.5, .5])) for task in training.tasks}
            cost['new_source_fits'] += 1
            cost['current_bundle_fits'] += 1
            try:
                bundle = fit_source_models(training, providers)
            except SourceFitError as error:
                save(output/'failed_source_fit.json', error.diagnostics)
                raise
            assert bundle['encoder'].source_count == bundle['diagnostics']['source_sites'] == 96
            save(output/'fit_diagnostics.json', bundle['diagnostics'])
            cost['new_source_fits'] += 1
            cost['native_prior_fits'] += 1
            prior = fit_native(training, heldout_family=heldout)
            save(output/'native_fit_diagnostics.json', prior.diagnostics())
            for task in archive.tasks:
                case = forecast(bundle, prior, task, split, cost, heldout)
                cases.append(case)
            save(output/'forecasts.json', dict(status='complete', archive_heldout_family=heldout,
                fold=fold, cases=cases, frozen_utc=datetime.now(timezone.utc).isoformat()))
        except Exception as error:
            save(output/'forecasts.json', dict(status='fold_failed', archive_heldout_family=heldout,
                fold=fold, cases=cases, failure=repr(error),
                frozen_utc=datetime.now(timezone.utc).isoformat()))
            raise
    print('fold frozen', heldout, fold, flush=True)


def finish(plan, manifest_path=MANIFEST):
    from performance.summarize_mainline_source_observed_ranking import score
    result = score(plan)
    output = ROOT/plan['output_dir']
    attempts = [json.loads(line) for path in output.rglob('attempts.jsonl')
                for line in path.read_text().splitlines()]
    keys = ('cpu_seconds', 'wall_seconds', 'new_source_calls', 'new_target_initial_calls',
        'new_target_search_calls', 'new_verification_calls', 'new_source_fits',
        'current_bundle_fits', 'native_prior_fits', 'new_truth_function_calls',
        'download_bytes', 'paired_raw_adaptation_exposures', 'native_replayed_raw_rows',
        'role_posterior_evaluations')
    flags, fold_failures = [], []
    for source in plan['source_archives']:
        for fold in range(plan['folds']):
            directory = output/source['heldout_target_family']/f'fold_{fold}'
            frozen = read(directory/'forecasts.json')
            if frozen['status'] == 'fold_failed':
                fold_failures.append(dict(archive=source['heldout_target_family'], fold=fold,
                                          reason=frozen['failure']))
            if (directory/'fit_diagnostics.json').exists():
                for record in read(directory/'fit_diagnostics.json')['source_identity_heads']:
                    if record['low_importance_ess']:
                        flags.append(dict(archive=source['heldout_target_family'], fold=fold,
                            identity=record['identity'], head=record['head'], weight_ess=record['weight_ess']))
    completion = dict(status=result['status'], completed_utc=datetime.now(timezone.utc).isoformat(),
        attempts=len(attempts), cost={key:sum(row.get(key, 0) for row in attempts) for key in keys},
        failed_attempts=[row for row in attempts if row['status'] != 'completed'],
        fold_failures=fold_failures, source_low_ess_flags=flags, deployed_model_changed=False)
    save(output/'completion.json', completion)
    plan.update(status=completion['status'], completed_utc=completion['completed_utc'],
                results_file=str((output/'summary.json').relative_to(ROOT)),
                completion_file=str((output/'completion.json').relative_to(ROOT)))
    save(manifest_path, plan)
    print(json.dumps(completion), flush=True)
    return completion


def run(plan, workers, manifest_path=MANIFEST):
    output = ROOT/plan['output_dir']
    logs = output/'execution_logs'
    logs.mkdir(parents=True, exist_ok=True)
    plan.update(status='running', started=True)
    plan.setdefault('started_utc', datetime.now(timezone.utc).isoformat())
    save(manifest_path, plan)
    jobs = [(source['heldout_target_family'], fold) for source in plan['source_archives']
            for fold in range(plan['folds'])
            if not (output/source['heldout_target_family']/f'fold_{fold}'/'forecasts.json').exists()]
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def invoke(job):
        heldout, fold = job
        with (logs/f'{heldout}_{fold}.log').open('a') as stream:
            result = subprocess.run([sys.executable, '-u', '-m',
                'performance.run_mainline_source_observed_ranking', 'fold',
                '--manifest', str(manifest_path), '--heldout', heldout, '--fold', str(fold)],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        return dict(heldout=heldout, fold=fold, returncode=result.returncode)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, future in enumerate(as_completed([pool.submit(invoke, job) for job in jobs]), 1):
            record = future.result()
            save(output/'progress.json', dict(stage='freeze-source-forecasts',
                completed_this_run=count, jobs_this_run=len(jobs), last_job=record))
            print('finished', record, flush=True)
    return finish(plan, manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('fold', 'run', 'score'))
    parser.add_argument('--manifest', default=str(MANIFEST))
    parser.add_argument('--heldout')
    parser.add_argument('--fold', type=int, choices=range(4))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    plan = read(args.manifest)
    if args.stage == 'fold':
        run_fold(plan, args.heldout, args.fold)
    elif args.stage == 'score':
        finish(plan, Path(args.manifest))
    else:
        run(plan, args.workers, Path(args.manifest))


if __name__ == '__main__':
    main()

"""Whole-policy source holdout of the unchanged deployed role-prior operator."""
import argparse
from datetime import datetime, timezone
import json

import numpy as np

from baselines.transfer_archive import FrozenTransferArchive
from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.mainline_legacy_source_models import fit_source_models, SourceFitError
from performance.mainline_role_prior import role_trajectory
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.run_mainline_source_policy_holdout import partition, training_archive

MANIFEST = ROOT/'performance/manifests/mainline_role_source_holdout_v1.json'
FIELDS = ('mean', 'variance', 'noise')


def forecast(bundle, prior, task, split, cost):
    from performance.mainline_role_source_holdout import build_source_engine
    engine = build_source_engine(prior, task.name, d=task.X.shape[1])
    assert engine.problem.tau == task.tau and engine.problem.alpha == task.alpha
    held = split['held']
    bank = [tuple(map(int, point)) for point in task.X_integer[held]]
    rows = [dict(point=list(bank[i]), raw=task.Y_replicates[held[i]][0].tolist()) for i in range(2)]
    snapshots, scores, events = role_trajectory(bundle, engine, bank, rows)
    # The source0 point control needs the second anchor's own posterior. It does
    # not observe the first anchor. This extra replay consumes no new raw.
    own, _, own_events = role_trajectory(bundle, engine, bank, rows[1:])
    cost['unique_reused_adaptation_raw_rows'] += 2
    cost['replayed_raw_rows'] += 3
    predictions = {arm: {} for arm in ('full', 'point_from_source0', 'point_after_first')}
    for key in FIELDS:
        full = np.asarray([s[key] for s in snapshots])
        local = np.repeat(full[:1], 3, axis=0)
        local[1:, 0] = full[1, 0]
        local[2, 1] = own[1][key][1]
        after = full.copy()
        after[2] = full[1]
        after[2, 1] = full[2, 1]
        predictions['full'][key] = full
        predictions['point_from_source0'][key] = local
        predictions['point_after_first'][key] = after
    return dict(family=task.name, d=task.X.shape[1], tau=task.tau,
        held_policy_ids=held, adaptation_raw=rows,
        predictions=predictions, head_order=['f','eta_f','g_centered','eta_g'],
        noise_head_order=['f','g'], identity_weights=[s['identity_weights'] for s in snapshots],
        adaptation_log_scores=scores, roundoff_events=events+own_events,
        operator='Unchanged role_trajectory source0 projection; no training raw replay.',
        controls='Point controls update all four own-policy heads and its identity weights; never-observed policies retain the declared source0 or first-anchor state.')


def run_fold(plan, fold):
    from performance.mainline_role_source_holdout import fit_native
    output = ROOT/plan['output_dir']/f'fold_{fold}'
    if (output/'forecasts.json').exists():
        print('reuse completed fold', fold, flush=True)
        return
    with attempt(output, 'fit-source-fold-and-freeze-role-forecasts') as cost:
        cost.update(new_source_fits=0, current_bundle_fits=0, native_prior_fits=0,
            new_target_initial_calls=0, new_truth_function_calls=0, download_bytes=0,
            unique_reused_adaptation_raw_rows=0, replayed_raw_rows=0)
        archive = FrozenTransferArchive.load(ROOT/plan['archive'])
        split = partition(plan, archive)[fold]
        old = read(ROOT/read(ROOT/plan['partition_manifest'])['output_dir']/f'fold_{fold}'/'split.json')
        for key in ('training','held','anchors','scoring'):
            np.testing.assert_array_equal(split[key], old[key])
        save(output/'split.json', split)
        train = training_archive(archive, split['training'])
        assert train.simulator_calls == 288
        providers = {task.name: no_outcomes(build_scalarized_problem(
            task.name, task.X.shape[1], 100, .04, .05, [.5,.5])) for task in train.tasks}
        cost['new_source_fits'] += 1; cost['current_bundle_fits'] += 1
        try:
            bundle = fit_source_models(train, providers)
        except SourceFitError as error:
            save(output/'failed_source_fit.json', error.diagnostics)
            raise
        assert bundle['encoder'].source_count == bundle['diagnostics']['source_sites'] == 96
        save(output/'fit_diagnostics.json', bundle['diagnostics'])
        cost['new_source_fits'] += 1; cost['native_prior_fits'] += 1
        prior = fit_native(train)
        save(output/'native_fit_diagnostics.json', prior.diagnostics())
        cases = [forecast(bundle, prior, task, split, cost) for task in archive.tasks]
        save(output/'forecasts.json', dict(fold=fold, cases=cases,
            frozen_utc=datetime.now(timezone.utc).isoformat()))
    print('fold complete', fold, cost['cpu_seconds'], flush=True)


def summarize(plan):
    from performance.summarize_mainline_role_source_holdout import summarize as score
    output = ROOT/plan['output_dir']
    score(plan)
    attempts = [json.loads(line) for path in output.rglob('attempts.jsonl')
                for line in path.read_text().splitlines()]
    fields = ('cpu_seconds','wall_seconds','new_source_calls','new_target_initial_calls',
              'new_target_search_calls','new_verification_calls','new_source_fits',
              'current_bundle_fits','native_prior_fits','new_truth_function_calls',
              'download_bytes','unique_reused_adaptation_raw_rows','replayed_raw_rows')
    low_ess = []
    for fold in range(plan['folds']):
        diagnostics = read(output/f'fold_{fold}'/'fit_diagnostics.json')
        for record in diagnostics['source_identity_heads']:
            if record['low_importance_ess']:
                low_ess.append(dict(fold=fold, identity=record['identity'],
                    head=record['head'], weight_ess=record['weight_ess']))
    result = dict(status='complete', cost={k:sum(r.get(k,0) for r in attempts) for k in fields},
        attempts=len(attempts), failed_attempts=[r for r in attempts if r['status']!='completed'],
        source_low_ess_flags=low_ess, completed_utc=datetime.now(timezone.utc).isoformat(),
        deployed_model_changed=False)
    save(output/'completion.json', result)
    plan.update(status='complete', completed_utc=result['completed_utc'])
    save(MANIFEST, plan)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', type=int, choices=range(4))
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args(); plan = read(MANIFEST)
    if args.summarize:
        summarize(plan)
    elif args.fold is not None:
        run_fold(plan, args.fold)
    else:
        parser.error('Choose --fold or --summarize')

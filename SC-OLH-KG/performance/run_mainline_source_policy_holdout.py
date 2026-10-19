"""Grouped source-policy holdout with two raw objective updates and no new calls."""
import argparse
import json
from dataclasses import replace

import numpy as np

from baselines.transfer_archive import FrozenTransferArchive
from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.diagnose_mainline_objective_attribution import collect_nodes, attribute
from performance.mainline_legacy_source_models import fit_source_models, _query_design, SourceFitError
from performance.mainline_source_task_conditioning import condition_existing_source_task
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from representation.legacy_function_prior import legacy_public_coordinates, legacy_source_labels
from representation.legacy_observable_prior import legacy_exposure_descriptors

MANIFEST = ROOT/'performance/manifests/mainline_source_policy_holdout_v1.json'


def partition(plan, archive):
    np.testing.assert_array_equal(archive.tasks[0].X_integer, archive.tasks[1].X_integer)
    assert len(set(map(tuple, archive.tasks[0].X_integer))) == 64
    held = np.random.default_rng(plan['fold_seed']).permutation(64).reshape(4, 16)
    rows = [dict(fold=i, training=np.setdiff1d(np.arange(64), group),
                 held=group, anchors=group[:2], scoring=group[2:]) for i,group in enumerate(held)]
    for row in rows:
        assert not set(row['training']) & set(row['held'])
        assert len(row['training']) == 48 and len(row['scoring']) == 14
    return rows


def training_archive(archive, indices):
    tasks = []
    for task in archive.tasks:
        arrays = {key:getattr(task, key)[indices].copy() for key in (
            'X', 'X_integer', 'Y_mean', 'replicate_variance', 'mean_observation_variance', 'constraint_sigma')}
        tasks.append(replace(task, **arrays,
            Y_replicates=tuple(task.Y_replicates[i].copy() for i in indices),
            origins=tuple(task.origins[i] for i in indices)))
    # An in-memory subset only; no new archive identifier or serialization.
    return replace(archive, tasks=tuple(tasks))


def forecast(bundle, training, original_task, provider, held, family_index):
    tasks = legacy_source_labels(training)
    labels = np.concatenate([t['response'][:, 0] for t in tasks])
    variances = np.concatenate([t.replicate_variance[:, 0] for t in training.tasks])
    bank = original_task.X_integer[held]
    coordinates = (
        (legacy_public_coordinates(original_task.X[held])-bundle['mean_center'])/bundle['mean_scale'],
        (bundle['encoder'].transform(legacy_exposure_descriptors(provider, bank, 50))
         -bundle['risk_center'])/bundle['risk_scale'])
    history = [(i, [float(original_task.Y_replicates[held[i]][0, 0])]) for i in range(2)]
    nodes, masses, validations, events = [], [], [], []
    family_slice = slice(48*family_index, 48*(family_index+1))
    for identity, pair in zip(bundle['identities'], bundle['components']):
        record, validation = condition_existing_source_task(
            pair[0], labels, variances, bundle['settings'], family_slice)
        D = _query_design(bundle, identity, 0, coordinates, record['law'])
        E = _query_design(bundle, identity, 3, coordinates, record['risk'])
        projected, trajectory, roundoff = collect_nodes(record, D, E, history, list(range(16)))
        nodes.append(projected)
        masses.append([[step[0], 0.] for step in trajectory])
        validations.append(dict(identity=identity, **validation))
        events.extend(dict(event, identity=identity) for event in roundoff)
    stacked = {key:np.stack([n[key] for n in nodes]) for key in nodes[0]}
    result = attribute(stacked, np.asarray(masses), bundle['log_weights'])
    full = result['mean']
    point_only = np.repeat(full[:1], 3, axis=0)
    point_only[1:, 0] = full[1, 0]
    # The second anchor is not a scoring policy; its own posterior is unnecessary.
    return dict(family=original_task.name, held_policy_ids=held, history=history,
        full_mean=full, point_only_scoring_and_reference_mean=point_only,
        full_difference_variance=result['difference_variance'],
        full_probability_improving=result['probability_improving'],
        source_identity_weights=np.exp(bundle['log_weights']),
        identity_weights=result['identity_weights'], first_update=result['steps'][0],
        validation=validations, roundoff_events=events)


def run_fold(plan, fold):
    output = ROOT/plan['output_dir']/f'fold_{fold}'
    with attempt(output, 'source-policy-fold') as cost:
        cost.update(new_source_fits=0, new_target_initial_calls=0, new_truth_function_calls=0,
                    download_bytes=0, reused_adaptation_raws=0)
        archive = FrozenTransferArchive.load(ROOT/plan['archive'])
        split = partition(plan, archive)[fold]
        save(output/'split.json', split)
        train = training_archive(archive, split['training'])
        assert train.simulator_calls == 288
        providers = {task.name:no_outcomes(build_scalarized_problem(
            task.name, task.X.shape[1], 100, .04, .05, [.5, .5])) for task in train.tasks}
        cost['new_source_fits'] = 1
        try:
            bundle = fit_source_models(train, providers)
        except SourceFitError as error:
            save(output/'failed_source_fit.json', error.diagnostics)
            raise
        assert bundle['encoder'].source_count == 96
        assert bundle['diagnostics']['source_sites'] == 96
        save(output/'fit_diagnostics.json', bundle['diagnostics'])
        rows = []
        for i,task in enumerate(archive.tasks):
            rows.append(forecast(bundle, train, task, providers[task.name], split['held'], i))
            cost['reused_adaptation_raws'] += 2
        save(output/'forecasts.json', dict(fold=fold, cases=rows))
    print('fold complete', fold, cost['cpu_seconds'], flush=True)


def metrics(mean, observed, observation_variance):
    error = np.asarray(mean)-observed
    return dict(mse=float(np.mean(error**2)), bias=float(np.mean(error)),
        noise_subtracted_mse=float(np.mean(error**2-observation_variance)))


def summarize(plan):
    output = ROOT/plan['output_dir']
    # Read every frozen forecast before opening the score labels.
    folds = [read(output/f'fold_{i}'/'forecasts.json') for i in range(plan['folds'])]
    with attempt(output, 'score-frozen-source-forecasts') as cost:
        cost.update(new_source_fits=0, new_target_initial_calls=0, new_truth_function_calls=0,
                    download_bytes=0)
        archive = FrozenTransferArchive.load(ROOT/plan['archive'])
        by_name = {t.name:t for t in archive.tasks}
        rows = []
        for fold in folds:
            for case in fold['cases']:
                task = by_name[case['family']]
                held = case['held_policy_ids']
                reference = task.Y_replicates[held[0]][1:, 0]
                anchor = float(reference.mean())
                anchor_noise = float(reference.var(ddof=1)/len(reference))
                observed = np.array([task.Y_replicates[i][:, 0].mean() for i in held[2:]])
                noise = np.array([task.Y_replicates[i][:, 0].var(ddof=1)/3 for i in held[2:]])
                delta, delta_noise = observed-anchor, noise+anchor_noise
                informative = np.abs(delta) >= 2*np.sqrt(delta_noise)
                scores = []
                for arm,key in [('full', 'full_mean'), ('point_only', 'point_only_scoring_and_reference_mean')]:
                    for t,mean in enumerate(np.asarray(case[key])):
                        predicted_delta = mean[2:]-mean[0]
                        scores.append(dict(arm=arm, round=t,
                            absolute=metrics(mean[2:], observed, noise),
                            difference=metrics(predicted_delta, delta, delta_noise),
                            direction_agreement=float(np.mean(predicted_delta*delta > 0)),
                            informative_count=int(informative.sum()),
                            informative_direction_correct=int(np.sum((predicted_delta*delta > 0)&informative))))
                full = np.asarray(case['full_mean'])[-1]
                point = np.asarray(case['point_only_scoring_and_reference_mean'])[-1]
                rows.append(dict(fold=fold['fold'], family=task.name, scored_policy_ids=held[2:],
                    anchor_policy_id=held[0], independent_anchor_mean=anchor,
                    independent_anchor_mean_variance=anchor_noise, observed_mean=observed,
                    observed_mean_variance=noise, scores=scores,
                    final_direction_changes=int(np.sum(np.sign(full[2:]-full[0]) != np.sign(point[2:]-point[0]))),
                    final_scoring_argmin=dict(full=held[2+int(np.argmin(full[2:]))],
                                             point_only=held[2+int(np.argmin(point[2:]))])))
        aggregate = []
        for family in [*by_name, 'both_families_equal']:
            selected = [r for r in rows if family == 'both_families_equal' or r['family'] == family]
            for arm in ('full', 'point_only'):
                for t in range(3):
                    values = [next(s for s in r['scores'] if s['arm']==arm and s['round']==t) for r in selected]
                    aggregate.append(dict(family=family, arm=arm, round=t,
                        **{metric:{key:float(np.mean([v[metric][key] for v in values]))
                                   for key in ('mse', 'bias', 'noise_subtracted_mse')}
                           for metric in ('absolute', 'difference')},
                        direction_agreement=float(np.mean([v['direction_agreement'] for v in values])),
                        informative_count=sum(v['informative_count'] for v in values),
                        informative_direction_correct=sum(v['informative_direction_correct'] for v in values)))
        save(output/'summary.json', dict(cases=rows, aggregate=aggregate,
            source_family_count=2, folds=4, scored_policy_family_pairs=112,
            final_direction_changes=sum(r['final_direction_changes'] for r in rows),
            final_scoring_argmin_changes=sum(r['final_scoring_argmin']['full'] !=
                                            r['final_scoring_argmin']['point_only'] for r in rows),
            interpretation='Same-source-task missing-policy objective validation; dependent folds, '
                'no target geometry, feasibility, optimizer or verified decision result.'))
    attempts = [json.loads(line) for directory in [output]+[output/f'fold_{i}' for i in range(4)]
                for line in (directory/'attempts.jsonl').read_text().splitlines()]
    cost_keys = ('cpu_seconds', 'wall_seconds', 'new_source_calls', 'new_target_search_calls',
                'new_verification_calls', 'new_source_fits', 'new_truth_function_calls',
                'download_bytes', 'reused_adaptation_raws')
    completion = dict(status='complete', attempts=len(attempts),
        failed_attempts=[r for r in attempts if r['status']!='completed'],
        cost={key:sum(r.get(key,0) for r in attempts) for key in cost_keys},
        validation=[dict(fold=f['fold'], family=c['family'], identities=c['validation'])
                    for f in folds for c in f['cases']],
        maximum_source_reconstruction_error=max(v['maximum_reconstruction_error']
            for f in folds for c in f['cases'] for v in c['validation']),
        roundoff_events=sum(len(c['roundoff_events']) for f in folds for c in f['cases']))
    low_ess = []
    for fold in range(4):
        fit = read(output/f'fold_{fold}'/'fit_diagnostics.json')
        weights = dict(zip(fit['source_identities'], fit['source_identity_weights']))
        low_ess.extend(dict(fold=fold, identity='/'.join(r['identity']), head=r['head'],
            weight_ess=r['weight_ess'], source_identity_weight=weights['/'.join(r['identity'])])
            for r in fit['source_identity_heads'] if r['low_importance_ess'])
    completion['source_low_ess_flags'] = low_ess
    save(output/'completion.json', completion)
    plan.update(status='complete', entrypoint='performance/run_mainline_source_policy_holdout.py')
    save(MANIFEST, plan)
    print(json.dumps(dict(cost=completion['cost'], aggregate=aggregate), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', type=int, choices=range(4))
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.summarize:
        summarize(read(MANIFEST))
    elif args.fold is not None:
        run_fold(read(MANIFEST), args.fold)
    else:
        parser.error('Choose --fold or --summarize')

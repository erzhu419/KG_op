"""Frozen source objective contrasts and raw-order errors; no posterior calls."""
from itertools import combinations

import numpy as np
from scipy.stats import norm

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.summarize_mainline_source_observed_ranking import load_frozen_cases, scoring_labels

MANIFEST = ROOT/'performance/manifests/mainline_objective_contrast_exposure_v1.json'
BASE = ('archive_heldout_family','fold','family','arm','prefix')
GROUP = ('partition','pair_group','subset')
FIELDS = ('label_absolute_gap_mean','prediction_difference_mse','prediction_difference_mae',
          'prediction_order_error','gap_weighted_prediction_order_error',
          'raw_difference_mse','raw_order_error','prediction_order_error_on_raw_pairs',
          'prediction_minus_raw_difference_mse','raw_wrong_repaired_fraction',
          'raw_correct_spoiled_fraction','endpoint_epistemic_mean','endpoint_noise_mean')


def metrics(pairs, labels, first_raw, prefix, snapshot, held):
    empty = dict.fromkeys(FIELDS)
    if not pairs:
        return dict(empty,pairs=0,raw_pairs=0,untied_label_pairs=0,
                    boundary_policy0_error_sum=None,prediction_squared_error_sum=None)
    i,j = np.asarray(pairs).T
    gap = labels['mean'][i,0]-labels['mean'][j,0]
    untied = gap != 0
    available_raw = (i < prefix) & (j < prefix)
    raw_gap = first_raw[i[available_raw]]-first_raw[j[available_raw]]
    raw_error = raw_gap-gap[available_raw]
    raw_right = np.sign(raw_gap)==np.sign(gap[available_raw])
    result = dict(empty,pairs=len(pairs),raw_pairs=int(available_raw.sum()),
        untied_label_pairs=int(untied.sum()),label_absolute_gap_mean=float(np.mean(np.abs(gap))),
        raw_difference_mse=float(np.mean(raw_error**2)) if len(raw_error) else None,
        raw_order_error=float(np.mean(~raw_right)) if len(raw_error) else None,
        boundary_policy0_error_sum=None,prediction_squared_error_sum=None)
    if snapshot.get('status') != 'complete':
        return result
    f = np.asarray(snapshot['mean'],dtype=float)[:,0]
    delta = f[i]-f[j]
    error = delta-gap
    right = np.sign(delta)==np.sign(gap)
    endpoint = np.r_[i,j]
    result.update(prediction_difference_mse=float(np.mean(error**2)),
        prediction_difference_mae=float(np.mean(np.abs(error))),
        prediction_order_error=float(np.mean(~right[untied])) if untied.any() else None,
        gap_weighted_prediction_order_error=float(np.sum(np.abs(gap)*(~right))/np.sum(np.abs(gap)))
            if np.sum(np.abs(gap)) else None,
        endpoint_epistemic_mean=float(np.mean(np.asarray(snapshot['variance'],dtype=float)[endpoint,0])),
        endpoint_noise_mean=float(np.mean(np.asarray(snapshot['noise'],dtype=float)[endpoint,0])),
        boundary_policy0_error_sum=float(np.sum(error[(np.asarray(held)[i]==0)|(np.asarray(held)[j]==0)]**2)),
        prediction_squared_error_sum=float(np.sum(error**2)))
    if len(raw_error):
        corrected,spoiled = (~raw_right)&right[available_raw],raw_right&(~right[available_raw])
        result.update(prediction_order_error_on_raw_pairs=float(np.mean(~right[available_raw])),
            prediction_minus_raw_difference_mse=float(np.mean(error[available_raw]**2)-np.mean(raw_error**2)),
            raw_wrong_repaired_fraction=float(np.mean(corrected)),
            raw_correct_spoiled_fraction=float(np.mean(spoiled)))
    return result


def aggregate(rows, **identity):
    result = {}
    for name in FIELDS:
        values = [r['metrics'][name] for r in rows if r['metrics'][name] is not None]
        result[name] = dict(registered_cases=len(rows),available_cases=len(values),
            mean=float(np.mean(values)) if len(values)==len(rows) and values else None,
            conditional_mean=float(np.mean(values)) if values else None,
            conditional_median=float(np.median(values)) if values else None)
    known = [r for r in rows if r['metrics']['prediction_difference_mse'] is not None]
    total = sum(r['metrics']['prediction_difference_mse'] for r in known)
    squares = sum(r['metrics']['prediction_squared_error_sum'] for r in known)
    return dict(identity,metrics=result,
        registered_pairs=sum(r['metrics']['pairs'] for r in rows),
        raw_comparison_pairs=sum(r['metrics']['raw_pairs'] for r in rows),
        boundary_policy0_fraction_of_pair_squared_error=(sum(r['metrics']['boundary_policy0_error_sum']
            for r in known)/squares if squares else None),
        largest_case_mse=[dict(**{k:r[k] for k in BASE},mse=r['metrics']['prediction_difference_mse'],
            fraction_of_available_case_mse=r['metrics']['prediction_difference_mse']/total if total else None)
            for r in sorted(known,key=lambda r:r['metrics']['prediction_difference_mse'],reverse=True)[:3]])


def run(plan):
    parent = read(ROOT/plan['inputs']['source_manifest'])
    records = load_frozen_cases(parent)
    source = ROOT/parent['output_dir']
    rows,arm_cases,snapshots,flags = [],[],[],[]
    for archive in parent['source_archives']:
        heldout = archive['heldout_target_family']
        tasks = {t['name']:t for t in read(ROOT/archive['path'])['tasks']}
        for fold in range(parent['folds']):
            frozen = records[heldout,fold]
            directory = source/heldout/f'fold_{fold}'
            split = read(directory/'split.json')
            fit = read(directory/'fit_diagnostics.json')
            flags.append(dict(archive_heldout_family=heldout,fold=fold,
                precision_certified=fit['precision_certified'],
                source_identity_heads=fit['source_identity_heads']))
            by_family = {c['family']:c for c in frozen['cases']}
            for family in archive['source_families']:
                case = by_family.get(family,{})
                held = case.get('held_policy_ids',split['held'])
                task = tasks[family]
                labels = scoring_labels(task,held)
                first_raw = np.full(16,np.nan)
                first_raw[:13] = np.asarray(task['Y_replicates'])[held[:13],0,0]
                if case:
                    np.testing.assert_array_equal(first_raw[:13],[r['raw'][0] for r in case['adaptation_raw']])
                x = np.asarray(task['X_integer'])[held]/100.
                distance = np.mean((x[:,None,:]-x[None,:,:])**2,axis=2)
                np.fill_diagonal(distance,np.inf)
                nearest = {tuple(sorted((i,int(np.argmin(distance[i]))))) for i in range(16)}
                empirical = labels['mean'][:,1]+norm.ppf(1-task['alpha'])*np.sqrt(labels['variance'][:,1]) <= 0
                for arm in parent['arms']:
                    prediction = case.get('predictions',{}).get(arm,{})
                    identity = dict(archive_heldout_family=heldout,fold=fold,family=family,arm=arm)
                    arm_cases.append(dict(identity,status=prediction.get('status','failed'),
                        failures=prediction.get('failures',frozen.get('failure'))))
                    for prefix in (10,13):
                        snap = prediction.get('snapshots',{}).get(str(prefix),{})
                        snapshots.append(dict(identity,prefix=prefix,status=snap.get('status','missing'),
                            roundoff_events=snap.get('roundoff_events'),held_policy_ids=held,
                            objective_mean=[v[0] for v in snap['mean']] if snap.get('status')=='complete' else None,
                            objective_epistemic=[v[0] for v in snap['variance']] if snap.get('status')=='complete' else None,
                            objective_noise=[v[0] for v in snap['noise']] if snap.get('status')=='complete' else None))
                        for partition,n_observed in (('current_prefix',prefix),('fixed_final13',13)):
                            for group in ('observed_observed','observed_unobserved','unobserved_unobserved'):
                                pairs = [(i,j) for i,j in combinations(range(16),2)
                                    if ('observed_observed' if j<n_observed else 'observed_unobserved'
                                        if i<n_observed else 'unobserved_unobserved') == group]
                                # Fixed descriptive strata, not alternative decision rules.
                                small = set(sorted(pairs,key=lambda ij:abs(labels['mean'][ij[0],0]-labels['mean'][ij[1],0]))
                                            [:int(np.ceil(len(pairs)/4))])
                                subsets = dict(all=pairs,small_label_gap_quarter=[p for p in pairs if p in small],
                                    nearest_physical_pairs=[p for p in pairs if p in nearest],
                                    both_empirically_feasible=[p for p in pairs if empirical[p[0]] and empirical[p[1]]])
                                for subset,selected in subsets.items():
                                    rows.append(dict(identity,prefix=prefix,partition=partition,pair_group=group,
                                        subset=subset,metrics=metrics(selected,labels,first_raw,prefix,snap,held)))
    aggregates = []
    keys = tuple(dict.fromkeys(tuple(r[k] for k in ('arm','prefix')+GROUP) for r in rows))
    for key in keys:
        group = [r for r in rows if tuple(r[k] for k in ('arm','prefix')+GROUP)==key]
        for family in [None]+sorted({r['family'] for r in group}):
            selected = group if family is None else [r for r in group if r['family']==family]
            aggregates.append(aggregate(selected,**dict(zip(('arm','prefix')+GROUP,key)),family=family))
    indexed = {tuple(r[k] for k in BASE+GROUP):r for r in rows}
    contrasts = []
    comparisons = [('legacy_role_after_init_point_only','legacy_role_point_only'),
        ('legacy_role_full','legacy_role_after_init_point_only'),('legacy_role_after_init_point_only','legacy_native')]
    for left,right in comparisons:
        for prefix in (10,13):
            for group in ('observed_observed','observed_unobserved','unobserved_unobserved'):
                for subset in ('all','small_label_gap_quarter','nearest_physical_pairs','both_empirically_feasible'):
                    matches = [r for r in rows if (r['arm'],r['prefix'],r['partition'],r['pair_group'],r['subset'])
                               ==(left,prefix,'fixed_final13',group,subset)]
                    differences = []
                    for r in matches:
                        key = tuple(right if k=='arm' else r[k] for k in BASE+GROUP)
                        other = indexed[key]
                        differences.append(dict(**{k:r[k] for k in BASE[:3]},metrics={m:
                            None if r['metrics'][m] is None or other['metrics'][m] is None else r['metrics'][m]-other['metrics'][m]
                            for m in FIELDS}))
                    summary = {}
                    for m in FIELDS:
                        values = [d['metrics'][m] for d in differences if d['metrics'][m] is not None]
                        summary[m] = dict(registered=24,available=len(values),
                            mean=float(np.mean(values)) if len(values)==24 else None,
                            conditional_mean=float(np.mean(values)) if values else None,
                            conditional_median=float(np.median(values)) if values else None,
                            lower=sum(v < -1e-12 for v in values),tied=sum(abs(v)<=1e-12 for v in values),
                            higher=sum(v > 1e-12 for v in values))
                    contrasts.append(dict(left=left,right=right,prefix=prefix,partition='fixed_final13',
                        pair_group=group,subset=subset,metrics=summary,cases=differences))
    completion = read(source/'completion.json')
    return dict(status='complete',registered_source_cases=24,registered_arm_cases=96,
        completed_arm_cases=sum(c['status']=='complete' for c in arm_cases),registered_snapshots=192,
        completed_snapshots=sum(s['status']=='complete' for s in snapshots),
        arm_cases=arm_cases,snapshots=snapshots,case_group_metrics=rows,aggregates=aggregates,contrasts=contrasts,
        source_fit_flags=flags,prior_paid_accounting=dict(file=str((source/'completion.json').relative_to(ROOT)),
            cost=completion['cost'],failed_attempts=completion['failed_attempts']),
        definitions=dict(current_prefix='Observed10/unobserved6 at prefix10; observed13/unobserved3 at prefix13.',
            fixed_final13='The same final13/final3 partition at both prefixes; future three adaptation raws are not available as prefix10 raw baselines.',
            raw='First adaptation raw only for pairs with both endpoint observations already available at this prefix.',
            scoring='Original unused two repeats at final13 policies, all three at never-observed3; these are noisy labels, not truth.',
            nearest='Union of each held policy nearest neighbor under mean squared normalized physical-coordinate distance; ties use held order.',
            small_gap='Lowest ceil(pair_count/4) absolute unused-label objective gaps within each case/partition/group; descriptive strata only.',
            feasible='Both endpoints have the original empirical g mean + z*sample standard deviation <= 0; not a safety certificate.',
            comparison_ties='Absolute paired metric changes <=1e-12 counted as ties; no model or decision tolerance changes.'),
        limits=['Aggregate within each case before family/overall means. Overlapping archives, folds and policy pairs are not independent task replications.',
            'Only saved marginal epistemic variances and expected episode noises are exposed. Cross-policy covariance and posterior difference variance remain unavailable.',
            'No source-model choice, tuning, new observation, fit, posterior restoration, raw update or target-data join is performed. '
            'Previously rejected head-specific representations are not reopened.'])


def main():
    plan = read(MANIFEST)
    output = ROOT/plan['output_dir']
    with attempt(output/'source_contrasts','saved-source-objective-contrasts') as cost:
        cost.update(new_target_initial_calls=0,new_source_fits=0,new_truth_function_calls=0,
            engine_restorations=0,posterior_restorations=0,raw_conditioning=0,raw_replays=0,new_rng_draws=0,download_bytes=0)
        result = run(plan)
        save(output/'source_contrasts.json',result)
    print({'status':result['status'],'completed_arm_cases':result['completed_arm_cases'],
           'cpu_seconds':cost['cpu_seconds']})


if __name__ == '__main__':
    main()

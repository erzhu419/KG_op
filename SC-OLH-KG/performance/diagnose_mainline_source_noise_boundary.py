"""Describe frozen, never-observed source noise forecasts; no posterior replay."""
import math
from statistics import mean, median

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.summarize_mainline_source_observed_ranking import load_frozen_cases, scoring_labels

MANIFEST = ROOT/'performance/manifests/mainline_source_risk_channel_attribution_v1.json'
FIELDS = ('saved_noise', 'empirical_noise', 'noise_squared_error', 'mean_eta', 'variance_eta',
          'exp_mean_eta', 'matched_gaussian_noise', 'log_saved_noise', 'log_gaussian_proxy',
          'log_mixture_over_gaussian_proxy')
KEYS = ('archive_heldout_family', 'fold', 'family', 'arm', 'held_position', 'policy_id', 'head')


def summarize(rows, **identity):
    metrics = {}
    for field in FIELDS:
        values = [r[field] for r in rows if r[field] is not None]
        metrics[field] = dict(available=len(values), registered=len(rows),
            mean=mean(values) if len(values) == len(rows) and values else None,
            conditional_mean=mean(values) if values else None,
            conditional_median=median(values) if values else None)
    known = [r for r in rows if r['saved_noise'] is not None]
    total = sum(r['saved_noise'] for r in known)
    largest = sorted(known, key=lambda r:r['saved_noise'], reverse=True)[:3]
    return dict(identity, metrics=metrics, top_saved_noise=[dict(
        **{k:r[k] for k in KEYS}, saved_noise=r['saved_noise'],
        fraction_of_available_noise=r['saved_noise']/total if total else None) for r in largest])


def diagnose(plan):
    parent = read(ROOT/plan['parent_manifest'])
    frozen = load_frozen_cases(parent)  # All forecast/failure files precede label reads.
    source_root = ROOT/parent['output_dir']
    rows, snapshots, source_flags, support, failures = [], [], [], [], []
    for archive in parent['source_archives']:
        heldout = archive['heldout_target_family']
        tasks = {t['name']:t for t in read(ROOT/archive['path'])['tasks']}
        deployed = ROOT/'paper_artifacts/mainline_legacy_backend_bridge_v1/source_models'/heldout
        deployed_diagnostics = read(deployed/'source_diagnostics.json')
        deployed_attempts = read(deployed/'source_fit_attempts.json')
        # Existing full-source fit provenance, not restored deployment predictions.
        assert any(r['status'] == 'complete' and r['archive'] == archive['path']
                   for r in deployed_attempts)
        assert deployed_diagnostics['source_sites'] == sum(len(t['X']) for t in tasks.values()) == 128
        support.append(dict(archive_heldout_family=heldout, archive=archive['path'],
            deployment_diagnostics=str((deployed/'source_diagnostics.json').relative_to(ROOT)),
            deployment_fit_attempts=str((deployed/'source_fit_attempts.json').relative_to(ROOT)),
            deployment_source_sites=128,
            family_policy_counts={name:len(t['X']) for name,t in tasks.items()},
            policy0_in_full_source=True, policy0_origin={name:t['origins'][0] for name,t in tasks.items()}))
        for fold in range(parent['folds']):
            record = frozen[heldout, fold]
            directory = source_root/heldout/f'fold_{fold}'
            split = read(directory/'split.json')
            diagnostics = read(directory/'fit_diagnostics.json')
            source_flags.append(dict(archive_heldout_family=heldout, fold=fold,
                source_sites=diagnostics['source_sites'],
                precision_certified=diagnostics['precision_certified'],
                identities=diagnostics['source_identities'],
                source_identity_weights=diagnostics['source_identity_weights'],
                heads=[{k:h.get(k) for k in ('identity','head','precision_certified',
                    'low_importance_ess','weight_ess','split_log_normalizer_gap')}
                    for h in diagnostics['source_identity_heads']]))
            cases = {c['family']:c for c in record['cases']}
            for family in archive['source_families']:
                case = cases.get(family, {})
                held = case.get('held_policy_ids', split['held'])
                assert held == split['held'] and set(held).isdisjoint(split['training'])
                labels = scoring_labels(tasks[family], held)
                for arm in parent['arms']:
                    prediction = case.get('predictions', {}).get(arm, {})
                    if prediction.get('status') != 'complete':
                        failures.append(dict(archive_heldout_family=heldout, fold=fold,
                            family=family, arm=arm, status=prediction.get('status', 'missing'),
                            failures=prediction.get('failures', record.get('failure'))))
                    for prefix in (10, 13):
                        snapshot = prediction.get('snapshots', {}).get(str(prefix), {})
                        complete = snapshot.get('status') == 'complete'
                        identity = dict(archive_heldout_family=heldout, fold=fold,
                                        family=family, arm=arm, prefix=prefix)
                        weights = snapshot.get('identity_weights', snapshot.get('task_weights'))
                        snapshots.append(dict(identity, status=snapshot.get('status', 'missing'),
                            arm_status=prediction.get('status', 'missing'),
                            roundoff_events=snapshot.get('roundoff_events'),
                            weight_kind='native_decision_weights' if arm == 'legacy_native' else 'role_identity_weights',
                            identity_names=None if arm == 'legacy_native' else diagnostics['source_identities'],
                            unobserved_weights={str(held[j]): (weights[j] if weights and isinstance(weights[0], list)
                                else weights) for j in range(13,16)}))
                        for j in range(13,16):
                            for head,h,eta in (('f',0,1), ('g',1,3)):
                                noise = snapshot['noise'][j][h] if complete else None
                                m = snapshot['mean'][j][eta] if complete else None
                                v = snapshot['variance'][j][eta] if complete else None
                                empirical = float(labels['variance'][j,h])
                                logproxy = m+v/2 if m is not None and v is not None else None
                                lognoise = math.log(noise) if noise is not None and noise > 0 else None
                                rows.append(dict(identity, held_position=j, policy_id=held[j], head=head,
                                    status=snapshot.get('status', 'missing'), never_observed=True,
                                    excluded_from_whole_policy_training=True, scoring_repeats=3,
                                    empirical_noise=empirical, saved_noise=noise,
                                    noise_squared_error=(noise-empirical)**2 if noise is not None else None,
                                    mean_eta=m, variance_eta=v,
                                    exp_mean_eta=math.exp(m) if m is not None else None,
                                    matched_gaussian_noise=math.exp(logproxy) if logproxy is not None else None,
                                    log_saved_noise=lognoise, log_gaussian_proxy=logproxy,
                                    log_mixture_over_gaussian_proxy=(lognoise-logproxy
                                        if lognoise is not None and logproxy is not None else None)))
    indexed = {(tuple(r[k] for k in KEYS), r['prefix']):r for r in rows}
    changes = []
    for r in rows:
        if r['prefix'] != 13:
            continue
        old = indexed[tuple(r[k] for k in KEYS), 10]
        fields = ('mean_eta','variance_eta','log_gaussian_proxy','log_saved_noise',
                  'log_mixture_over_gaussian_proxy','saved_noise')
        delta = {k: r[k]-old[k] if r[k] is not None and old[k] is not None else None for k in fields}
        variance_delta = delta.pop('variance_eta')
        delta['half_variance_eta'] = variance_delta/2 if variance_delta is not None else None
        if delta['mean_eta'] is not None:
            # Detect accidental assignment of mixture Eexpeta to a matched-Gaussian proxy.
            assert math.isclose(delta['log_gaussian_proxy'], delta['mean_eta']+delta['half_variance_eta'], abs_tol=1e-10)
            assert math.isclose(delta['log_saved_noise'], delta['log_gaussian_proxy']+
                                delta['log_mixture_over_gaussian_proxy'], abs_tol=1e-10)
        changes.append(dict(**{k:r[k] for k in KEYS}, delta_13_minus_10=delta))
    groups = []
    for arm in parent['arms']:
        for prefix in (10,13):
            for head in ('f','g'):
                group = [r for r in rows if (r['arm'],r['prefix'],r['head']) == (arm,prefix,head)]
                for family in [None]+sorted({r['family'] for r in group}):
                    selected = group if family is None else [r for r in group if r['family'] == family]
                    groups.append(summarize(selected, arm=arm,prefix=prefix,head=head,family=family,
                        subset='all_unobserved3'))
                groups.append(summarize([r for r in group if r['policy_id'] == 0],
                    arm=arm,prefix=prefix,head=head,family=None,subset='policy0'))
    return dict(status='complete' if not failures else 'complete_with_forecast_failures',
        registered_cases=24, registered_arm_cases=96, registered_snapshots=192,
        registered_policy_head_rows=1152, available_policy_head_rows=sum(r['saved_noise'] is not None for r in rows),
        failures=failures, rows=rows, summaries=groups, prefix_changes=changes,
        frozen_snapshot_metadata=snapshots, source_fit_flags=source_flags, full_source_support=support,
        policy0_rows=[r for r in rows if r['policy_id'] == 0],
        policy0_changes=[r for r in changes if r['policy_id'] == 0],
        definitions=dict(empirical_noise='Unbiased sample variance of all three unused raw repeats for never-observed3.',
            saved_noise='Role: saved mixture E[exp(eta)]. Native: saved expert-weighted aleatoric variance; eta unavailable.',
            log_proxy='mean_eta + variance_eta/2; changes split additively into delta mean_eta and delta variance_eta/2.',
            mixture_gap='log(saved noise) minus the matched-Gaussian log proxy; its change completes an accounting identity, not component-level causality.'),
        limits=['The actual mixture Eexpeta is not exp(aggregate mean_eta + aggregate variance_eta/2). '
                'Saved identity weights do not recover unsaved within-identity/node moments or their individual noise contributions.',
                'Source precision/ESS and roundoff flags are retained. No more precise quadrature or posterior reconstruction is performed.',
                'The 24 source archive/fold/family cases overlap; three-repeat empirical variances are noisy development labels.',
                'Policy0 is excluded from both source families throughout fold3 and never adapted; deployment full-source fits include it. '
                'This holdout boundary failure alone does not establish a deployed d1000 failure.',
                'No target labels, new observations, model fitting, restoration, raw replay, or RNG enter this analysis.'])


def main():
    plan = read(MANIFEST)
    output = ROOT/plan['output_dir']
    with attempt(output/'noise_boundary', 'saved-source-noise-boundary') as cost:
        cost.update(new_target_initial_calls=0,new_truth_function_calls=0,new_source_fits=0,
                    posterior_restorations=0,raw_replays=0,new_rng_draws=0,download_bytes=0)
        result = diagnose(plan)
        result['preparation_errors'] = [
            dict(stage='interpreter_startup', interpreter='scheduleurm-or-exact/bin/python',
                 error='ModuleNotFoundError: numpy', analysis_started=False, model_calls=0),
            dict(stage='interpreter_startup', interpreter='scheduleurm-or-exact-site/bin/python',
                 error='No such file or directory', analysis_started=False, model_calls=0)]
        save(output/'noise_boundary.json', result)
    print({'status':result['status'], 'rows':len(result['rows']), 'cpu_seconds':cost['cpu_seconds']})


if __name__ == '__main__':
    main()

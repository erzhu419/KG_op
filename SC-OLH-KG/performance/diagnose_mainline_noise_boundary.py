"""Source0 eta variance attribution at a real held-out boundary policy."""
from datetime import datetime, timezone
import json

import numpy as np
from scipy.special import logsumexp

from baselines.transfer_archive import FrozenTransferArchive
from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.mainline_legacy_source_models import fit_source_models, _query_design, SourceFitError
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.run_mainline_source_policy_holdout import training_archive
from representation.legacy_function_prior import legacy_public_coordinates
from representation.legacy_observable_prior import legacy_exposure_descriptors

MANIFEST = ROOT/'performance/manifests/mainline_noise_boundary_v1.json'
PARTS = ('source_common', 'new_task', 'point_discrepancy', 'source_nodes', 'identities')


def parity(actual, expected, errors, name):
    # Any mismatch invalidates the attribution of the previously saved model.
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10, err_msg=name)
    errors[name] = float(np.max(np.abs(np.asarray(actual)-expected)))


def decompose(bundle, coordinates, head, focus):
    identities = []
    a = np.exp(bundle['log_weights'])
    for identity, pair in zip(bundle['identities'], bundle['components']):
        record = pair[head]
        E = _query_design(bundle, identity, 3 if head == 0 else 2, coordinates, record['risk'])
        node_mean = record['eta_mean']@E.T
        q = np.exp(record['log_weights'])
        mean = q@node_mean
        common = np.einsum('np,pq,nq->n', E, record['eta_covariance'], E)
        task = np.einsum('np,pq,nq->n', E, record['risk']['task'], E)
        white = np.full(len(E), record['risk']['white'])
        nodes = q@(node_mean-mean)**2
        node_log_moment = logsumexp(record['log_weights'][:, None]+node_mean, axis=0)
        log_noise = (common+task+white)/2+node_log_moment
        tails = []
        for policy in focus:
            rho = np.exp(record['log_weights']+node_mean[:, policy]-node_log_moment[policy])
            top = np.argsort(rho)[-3:][::-1]
            tails.append(dict(policy_id=policy, source_node_ess=float(1/(q@q)),
                noise_contribution_ess=float(1/(rho@rho)),
                top_nodes=[dict(node=int(k), posterior_mass=q[k],
                    noise_fraction=rho[k], mean_eta=node_mean[k, policy]) for k in top]))
        identities.append(dict(identity=identity, mean=mean,
            variance_parts=dict(zip(PARTS[:4], (common, task, white, nodes))),
            log_noise=log_noise, node_jensen_gap=node_log_moment-mean, tails=tails))
    means = np.asarray([r['mean'] for r in identities])
    mean = a@means
    parts = {key:sum(w*r['variance_parts'][key] for w,r in zip(a,identities)) for key in PARTS[:4]}
    parts['identities'] = a@(means-mean)**2
    logs = np.asarray([r['log_noise'] for r in identities])
    log_noise = logsumexp(bundle['log_weights'][:, None]+logs, axis=0)
    log_parts = {key:parts[key]/2 for key in PARTS[:3]}
    log_parts['source_nodes_jensen'] = sum(w*r['node_jensen_gap'] for w,r in zip(a, identities))
    log_parts['identity_jensen'] = log_noise-a@logs
    np.testing.assert_allclose(mean+sum(log_parts.values()), log_noise, rtol=1e-10, atol=1e-10)
    for i,r in enumerate(identities):
        r['source_identity_mass'] = a[i]
        r['noise_fraction'] = np.exp(bundle['log_weights'][i]+r['log_noise']-log_noise)
    return dict(head=('f','g')[head], mean_eta=mean, variance_eta=sum(parts.values()),
        noise=np.exp(log_noise), variance_parts=parts, log_noise_parts=log_parts,
        identities=identities,
        interpretation='Source0 total variance; log-noise identity Jensen is not eta identity variance. '
            'Node noise concentration describes the fixed rule, not an integration certificate.')


def main():
    plan = read(MANIFEST)
    output = ROOT/plan['output_dir']
    if (output/'completion.json').exists():
        print('reuse completed attribution', flush=True)
        return
    parent = read(ROOT/plan['parent_manifest'])
    old_dir = ROOT/parent['output_dir']/f"fold_{plan['fold']}"
    errors = {}
    with attempt(output, 'refit-one-source-fold-and-attribute-source0-eta') as cost:
        cost.update(new_source_fits=0, native_prior_fits=0, new_target_initial_calls=0,
                    new_truth_function_calls=0, download_bytes=0, saved_raw_rows_replayed=0)
        archive = FrozenTransferArchive.load(ROOT/parent['archive'])
        split = read(old_dir/'split.json')
        training = np.asarray(split['training'])
        task0 = archive.tasks[0]
        focus = [plan['boundary_policy_id']]
        focus.append(int(training[np.argmin(np.linalg.norm(task0.X[training]-task0.X[focus[0]], axis=1))]))
        focus.append(int(training[np.argmin(np.abs(task0.X[training].mean(axis=1)-.5))]))
        assert focus == plan['focus_policy_ids'], focus
        train = training_archive(archive, training)
        providers = {t.name:no_outcomes(build_scalarized_problem(t.name, t.X.shape[1], 100, .04, .05, [.5,.5]))
                     for t in train.tasks}
        cost['new_source_fits'] += 1
        try:
            bundle = fit_source_models(train, providers)
        except SourceFitError as error:
            save(output/'failed_source_fit.json', error.diagnostics)
            raise
        save(output/'fit_diagnostics.json', bundle['diagnostics'])
        old = read(old_dir/'forecasts.json')
        cases = []
        from performance.mainline_noise_boundary_geometry import geometry
        for task in archive.tasks:
            provider = providers[task.name]
            descriptor = legacy_exposure_descriptors(provider, task.X_integer, 50)
            coordinates = ((legacy_public_coordinates(task.X)-bundle['mean_center'])/bundle['mean_scale'],
                (bundle['encoder'].transform(descriptor)-bundle['risk_center'])/bundle['risk_scale'])
            heads = [decompose(bundle, coordinates, h, focus) for h in range(2)]
            saved = next(c for c in old['cases'] if c['family']==task.name)
            reference = saved['predictions']['full']
            held = saved['held_policy_ids']
            parity(np.exp(bundle['log_weights']), saved['identity_weights'][0], errors, task.name+'/identity')
            for h,row in enumerate(heads):
                for actual,key,col in [('mean_eta','mean',2*h+1),('variance_eta','variance',2*h+1),('noise','noise',h)]:
                    parity(row[actual][held], np.asarray(reference[key])[0,:,col], errors, task.name+'/'+row['head']+'/'+actual)
            cases.append(dict(family=task.name, heads=heads,
                geometry=geometry(bundle, task, provider, training, focus)))
        save(output/'attribution.json', dict(fold=plan['fold'], prefix=0, focus_policy_ids=focus,
            cases=cases, parity=errors, statistical_scope='Opened source fold diagnostic; no held response fitting or target outcome use.'))
    records = [json.loads(line) for line in (output/'attempts.jsonl').read_text().splitlines()]
    keys = ('cpu_seconds','wall_seconds','new_source_fits','native_prior_fits','new_source_calls',
            'new_target_initial_calls','new_target_search_calls','new_verification_calls',
            'new_truth_function_calls','source_model_restorations','saved_raw_rows_replayed','download_bytes')
    result = dict(status='complete', attempts=len(records),
        failed_attempts=[r for r in records if r['status']!='completed'],
        cost={k:sum(r.get(k,0) for r in records) for k in keys},
        reconstruction_max_absolute_error=max(errors.values()), deployed_model_changed=False,
        completed_utc=datetime.now(timezone.utc).isoformat())
    save(output/'completion.json', result)
    plan.update(status='complete', completed_utc=result['completed_utc'])
    save(MANIFEST, plan)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

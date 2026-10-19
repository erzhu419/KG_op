"""One bounded mean-encoder candidate on the existing four source folds."""
import argparse
from datetime import datetime, timezone
import json

import numpy as np

from baselines.transfer_archive import FrozenTransferArchive
from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from performance.benchmark_quality import json_safe
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.mainline_bounded_mean_encoder import fit_candidate_source_models
from performance.mainline_legacy_source_models import SourceFitError, _query_design
from performance.mainline_role_source_holdout import fit_native
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.run_mainline_role_source_holdout import forecast
from performance.run_mainline_source_policy_holdout import training_archive
from representation.legacy_function_prior import legacy_public_coordinates
from representation.legacy_observable_prior import legacy_exposure_descriptors

MANIFEST = ROOT/'performance/manifests/mainline_bounded_mean_source_v1.json'


def compare_diagnostics(actual, expected, errors, path='native'):
    """A different native fit would confound the representation comparison."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            compare_diagnostics(actual[key], expected[key], errors, path+'/'+key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i,(a,b) in enumerate(zip(actual,expected)):
            compare_diagnostics(a,b,errors,path+f'/{i}')
    elif isinstance(expected, (float,int)) and not isinstance(expected,bool):
        np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-10,err_msg=path)
        errors.append(abs(actual-expected))
    else:
        assert actual == expected, path


def encoder_audit(bundle, task, provider, split):
    encoder = bundle['encoder']
    descriptor = legacy_exposure_descriptors(provider,task.X_integer,50)
    values = encoder.transform(descriptor)
    assert all(getattr(encoder,name).latent_transform=='source_tanh' for name in ('f_mean','g_mean'))
    assert all(getattr(encoder,name).coordinate.latent_transform=='identity' for name in ('f_variance','g_variance'))
    assert np.max(np.abs(values[:,:4])) <= 1+1e-12
    coordinates = ((legacy_public_coordinates(task.X)-bundle['mean_center'])/bundle['mean_scale'],
                   (values-bundle['risk_center'])/bundle['risk_scale'])
    maps = []
    for identity,pair,mass in zip(bundle['identities'],bundle['components'],np.exp(bundle['log_weights'])):
        for h,record in enumerate(pair):
            E = _query_design(bundle,identity,3 if h==0 else 2,coordinates,record['risk'])
            norm = np.linalg.norm(E,axis=1)
            maps.append(dict(identity=identity,head=('eta_f','eta_g')[h],source_mass=mass,
                policy0_design_norm=norm[0],training_design_max=norm[split['training']].max()))
    result = dict(family=task.name, encoder_values=values, eta_design_norms=maps)
    if split['fold']==3:
        old = read(ROOT/'paper_artifacts/mainline_noise_boundary_v1/attribution.json')
        g = next(c['geometry'] for c in old['cases'] if c['family']==task.name)
        expected = np.asarray(g['encoder']['focus'])[:,4:]
        actual = values[g['focus_policy_ids'],4:]
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
        result['unchanged_eta_encoder_max_error'] = float(np.max(np.abs(actual-expected)))
    return result


def run_fold(plan,fold):
    output = ROOT/plan['output_dir']/f'fold_{fold}'
    if (output/'forecasts.json').exists():
        print('reuse completed fold',fold,flush=True)
        return
    parent = read(ROOT/plan['parent_manifest'])
    previous = ROOT/parent['output_dir']/f'fold_{fold}'
    with attempt(output,'fit-bounded-mean-source-and-freeze-forecasts') as cost:
        cost.update(new_source_fits=0,current_bundle_fits=0,native_prior_fits=0,
            new_target_initial_calls=0,new_truth_function_calls=0,download_bytes=0,
            unique_reused_adaptation_raw_rows=0,replayed_raw_rows=0)
        archive = FrozenTransferArchive.load(ROOT/plan['archive'])
        split = read(previous/'split.json')
        save(output/'split.json',split)
        train = training_archive(archive,np.asarray(split['training']))
        assert train.simulator_calls==288 and not set(split['training']) & set(split['held'])
        providers = {t.name:no_outcomes(build_scalarized_problem(t.name,t.X.shape[1],100,.04,.05,[.5,.5]))
                     for t in train.tasks}
        cost['new_source_fits']+=1; cost['current_bundle_fits']+=1
        try:
            bundle = fit_candidate_source_models(train,providers)
        except SourceFitError as error:
            save(output/'failed_source_fit.json',error.diagnostics)
            raise
        assert bundle['encoder'].source_count==bundle['diagnostics']['source_sites']==96
        save(output/'fit_diagnostics.json',bundle['diagnostics'])
        save(output/'encoder_audit.json',[encoder_audit(bundle,t,providers[t.name],split) for t in archive.tasks])
        # The candidate constructor replacement has ended before native fitting.
        cost['new_source_fits']+=1; cost['native_prior_fits']+=1
        prior = fit_native(train)
        diagnostics = json_safe(prior.diagnostics())
        save(output/'native_fit_diagnostics.json',diagnostics)
        errors = []
        compare_diagnostics(diagnostics,read(previous/'native_fit_diagnostics.json'),errors)
        cases = [forecast(bundle,prior,t,split,cost) for t in archive.tasks]
        old_cases = read(previous/'forecasts.json')['cases']
        parity = dict(native_diagnostics_max_error=max(errors,default=0.))
        for case in cases:
            old = next(c for c in old_cases if c['family']==case['family'])
            assert case['held_policy_ids']==old['held_policy_ids'] and case['adaptation_raw']==old['adaptation_raw']
            for key in ('mean','variance'):
                actual = np.asarray(case['predictions']['full'][key])[0,:,2]
                expected = np.asarray(old['predictions']['full'][key])[0,:,2]
                np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-10)
                parity[case['family']+'/source0_g_'+key] = float(np.max(np.abs(actual-expected)))
        save(output/'forecasts.json',dict(fold=fold,cases=cases,parity=parity,
            frozen_utc=datetime.now(timezone.utc).isoformat()))
    print('fold complete',fold,cost['cpu_seconds'],flush=True)


def summarize(plan):
    from performance.summarize_mainline_bounded_source import summarize as score
    output = ROOT/plan['output_dir']
    # Completed scores survive a later accounting-only interruption.
    if not (output/'comparison.json').exists():
        score(plan)
    records = [json.loads(line) for p in output.rglob('attempts.jsonl') for line in p.read_text().splitlines()]
    fields = ('cpu_seconds','wall_seconds','new_source_calls','new_target_initial_calls',
        'new_target_search_calls','new_verification_calls','new_truth_function_calls','new_source_fits',
        'current_bundle_fits','native_prior_fits','download_bytes','unique_reused_adaptation_raw_rows','replayed_raw_rows')
    low_ess,parities,roundoff = [],[],0
    for fold in range(4):
        sub = output/f'fold_{fold}'
        d = read(sub/'fit_diagnostics.json')
        low_ess.extend(dict(fold=fold,**r) for r in d['source_identity_heads'] if r['low_importance_ess'])
        f = read(sub/'forecasts.json'); parities.extend(f['parity'].values())
        roundoff += sum(len(c['roundoff_events']) for c in f['cases'])
    checks = {name:read(output/(name+'.json')) for name in ('encoder_check','comparison_check')}
    result = dict(status='complete',cost={k:sum(r.get(k,0) for r in records) for k in fields},
        preparation_checks=checks,attempts=len(records),
        failed_attempts=[r for r in records if r['status']!='completed'],source_low_ess_flags=low_ess,
        native_parity_max_absolute_error=max(parities),roundoff_events=roundoff,
        deployed_model_changed=False,completed_utc=datetime.now(timezone.utc).isoformat())
    result['total_recorded_cpu_seconds'] = (result['cost']['cpu_seconds']
        +checks['encoder_check']['cpu_seconds']+checks['comparison_check']['actual_cpu_seconds'])
    result['unknown_cpu_attempts'] = sum(bool(r.get('cpu_seconds_unknown')) for r in records)
    save(output/'completion.json',result)
    plan.update(status='complete',completed_utc=result['completed_utc'])
    save(MANIFEST,plan)
    print(json.dumps({k:v for k,v in result.items() if k not in ('preparation_checks','source_low_ess_flags')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold',type=int,choices=range(4)); parser.add_argument('--summarize',action='store_true')
    args=parser.parse_args(); plan=read(MANIFEST)
    if args.summarize: summarize(plan)
    elif args.fold is not None: run_fold(plan,args.fold)
    else: parser.error('Choose --fold or --summarize')

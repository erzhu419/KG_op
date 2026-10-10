"""Validate cached controls against finer direct references, keeping legacy failures."""
import json
import resource
import signal
import time

from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.diagnose_historical_component_guarded import compressed_read, mixed_values
from performance.validate_historical_component_bank import group_checks
from performance.validate_historical_component_transfer import part_path, record_bundle
from performance.validate_historical_label_bank import valid_tables
from performance.validate_historical_label_control import controlled_row, tables

PLAN='performance/manifests/historical_reference_upgrade_v1.json'
REFERENCES=('E8','F8','H8','E9')
CANDIDATES={'control32':'E5','control128':'E7'}


def upgrade(bundle,anchor,target,kind,tolerance,legacy,anchor_rule):
    direct={n:tables(bundle,{h:target[h][n] for h in bundle['utility_identities']}) for n in REFERENCES}
    reference=group_checks(bundle,[dict(identity=h,rows=target[h]) for h in bundle['utility_identities']],kind,
        [('E8','F8'),('E8','H8'),('E8','E9')],tolerance)
    reference['valid_unclipped_ranges']=all(valid_tables(bundle,v) for v in direct.values())
    reference['passed']=bool(reference['passed'] and reference['valid_unclipped_ranges'])
    values={n:mixed_values(bundle,*data,kind) for n,data in direct.items()}
    checks={};candidates={}
    for method,low in CANDIDATES.items():
        data=tables(bundle,{h:controlled_row(anchor[h][anchor_rule],target[h][low],anchor[h][low])
            for h in bundle['utility_identities']})
        candidate=mixed_values(bundle,*data,kind)
        differences={n:float(abs(candidate-value).max()) for n,value in values.items()}
        normalization=max(reference['maximum_log_evidence_error'],max(
            abs(rows[h][n]['diagnostics']['predictive_log_evidence_error'])
            for rows,n in ((anchor,anchor_rule),(anchor,low),(target,low)) for h in bundle['utility_identities']))
        valid=valid_tables(bundle,data)
        checks[method]=dict(direct_differences=differences,valid_unclipped_ranges=valid,
            maximum_log_evidence_error=normalization,
            passed=bool(reference['passed'] and valid and normalization<=tolerance and max(differences.values())<=tolerance))
        candidates[method]=data
    method=next((m for m in CANDIDATES if checks[m]['passed']),
        'direct256' if reference['passed'] else 'HOLD')
    selected=candidates[method] if method in candidates else direct['E8'] if method=='direct256' else None
    route=('legacy_and_finer_direct' if legacy['selected_method']==method else 'finer_direct_upgrade') if method in candidates else 'direct_reference' if method=='direct256' else 'HOLD'
    return dict(selected_method=method,validation_route=route,candidate_checks=checks,
        direct_reference_checks=reference,hybrid_direct_admitted=method!='HOLD',
        selected_statistics=selected[0].tolist() if selected is not None else None,
        selected_costs=selected[1].tolist() if selected is not None else None,
        legacy_check={k:v for k,v in legacy.items() if k not in ('selected_statistics','selected_costs')},
        selected_target_rule_cpu_seconds=sum(target[h][CANDIDATES.get(method,'E8')]['cpu_seconds']
            for h in bundle['utility_identities']) if method!='HOLD' else None)


def replay():
    tick=time.process_time();spec=read(PLAN);out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    assert spec['direct_reference_rules']==list(REFERENCES) and spec['candidate_rules']==list(CANDIDATES.values())
    budget=sum(spec['cpu_reserves'].values());assert budget<=spec['maximum_estimated_cpu_seconds']
    base=ROOT/read(spec['parent_plan'])['output'];original=read(str(base/'context.json'))
    near=ROOT/spec['nearest_anchor_output'];near_context=read(str(near/'contexts.json'))
    layout=read('paper_artifacts/historical_cdf_bank_v1/bank_layout.json')
    assert original['actions']==layout['actions'] and original['groups']==layout['groups']
    contexts={};results=[];counterfactual=[];target_cpu=0.
    for cohort in spec['cohorts']:
        directory=ROOT/'paper_artifacts'/cohort['directory']
        summary=read(str(directory/'summary.json'));assert summary['complete']
        current=read(str(directory/'contexts.json'))
        work={(r['record'],r['group'],r['kind']):r for r in read(str(directory/'pilot_plan.json'))['selected']} if cohort['format']=='cdf' else {}
        for row in summary['groups']:
            context=current[row['record']];gid=row['group'];kind=row['kind'];identities=context['utility_identities']
            assert len(context['head_log_scores'])==8 and context['prior_weights']==original['prior_weights']
            if row['record'] in contexts:
                saved=contexts[row['record']]
                for key in ('label','head','head_log_scores','prior_weights','replicates','eta_label','utility_identities'):
                    assert saved[key]==context[key]
                for group,local in context['groups'].items():
                    if group in saved['groups']:assert saved['groups'][group]==local
                    saved['groups'][group]=local
            else:contexts[row['record']]=context
            shared=compressed_read(base/'marginals'/f'{gid:03d}.json.gz')
            bundle=record_bundle(shared,context)
            assert {h:len(shared['fields'][h]['means']) for h in identities}=={4:512,5:1024}
            target={h:read(str(part_path(directory,row,h)))['rows'] for h in identities}
            source={h:read(str(base/'parts'/f'{gid:03d}_{h}_{kind}.json'))['rows'] for h in identities}
            for h in identities:
                part=read(str(part_path(directory,row,h)))
                assert all(part[k]==v for k,v in dict(record=row['record'],group=gid,kind=kind,identity=h).items())
            anchor=source;anchor_name=original['name'];anchor_rule='E8'
            if cohort['anchor']!='original':
                anchor_name=cohort['anchor'];anchor_context=near_context[anchor_name]
                assert anchor_context['prior_weights']==context['prior_weights']
                assert anchor_context['utility_identities']==identities
                assert anchor_context['groups'][str(gid)]['action_indices']==context['groups'][str(gid)]['action_indices']
                anchor={h:read(str(part_path(near,dict(row,record=anchor_name),h)))['rows'] for h in identities}
            if cohort['format']=='cdf':anchor_rule=work[row['record'],gid,kind]['anchor_rule']
            result=upgrade(bundle,anchor,target,kind,spec['maximum_action_difference'],row,anchor_rule)
            old_rule=CANDIDATES.get(row['selected_method'],'E8')
            old_cpu=sum(target[h][old_rule]['cpu_seconds'] for h in identities)
            results.append(dict(result,record=row['record'],group=gid,kind=kind,anchor=anchor_name,anchor_rule=anchor_rule,
                action_indices=context['groups'][str(gid)]['action_indices'],source_directory=cohort['directory'],
                legacy_target_rule_cpu_seconds=old_cpu))
            target_cpu+=sum(target[h][n]['cpu_seconds'] for h in identities for n in target[h])
            if cohort['directory']=='historical_nearby_anchor_v1':
                legacy=row['anchor_comparisons']['original']
                checked=upgrade(bundle,source,target,kind,spec['maximum_action_difference'],legacy,'E8')
                counterfactual.append(dict(checked,record=row['record'],group=gid,kind=kind,anchor=original['name'],
                    action_indices=context['groups'][str(gid)]['action_indices']))
    assert len(results)==33 and len(counterfactual)==5
    save(out/'contexts.json',contexts)
    actions=lambda rows,method:sum(len(r['action_indices']) for r in rows if r['selected_method']==method)
    result=dict(complete=True,groups=results,original_anchor_p197=counterfactual,
        all_target_references_admitted=all(r['hybrid_direct_admitted'] for r in results),
        unique_target_labels=len(contexts),group_kind_record_checks=len(results),
        selected_actions={m:actions(results,m) for m in ('control32','control128','direct256','HOLD')},
        legacy_selected_actions={m:sum(len(r['action_indices']) for r in results if r['legacy_check']['selected_method']==m)
            for m in ('control32','control128','direct256','HOLD')},
        changed_cases=[{k:r[k] for k in ('record','group','kind','selected_method')}|dict(
            legacy_method=r['legacy_check']['selected_method']) for r in results if r['selected_method']!=r['legacy_check']['selected_method']],
        legacy_target_rule_cpu_seconds=sum(r['legacy_target_rule_cpu_seconds'] for r in results),
        selected_target_rule_cpu_seconds=sum(r['selected_target_rule_cpu_seconds'] for r in results)
            if all(r['selected_target_rule_cpu_seconds'] is not None for r in results) else None,
        cached_target_integral_cpu_seconds=target_cpu,new_integrations=0,new_integral_cpu_seconds=0.,
        cumulative_estimated_cpu_seconds=budget,maximum_estimated_cpu_seconds=spec['maximum_estimated_cpu_seconds'],
        cpu_seconds=time.process_time()-tick,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False,
        full_predictive_integration_admitted=False,
        limitations='Opened development caches; p197 motivated this numerical protocol. Same fixed32 objective heads and all identity weights/components. '
        'Finite comparisons are not rigorous error bounds or independent decision evidence. Selected-rule timers exclude anchor/reference acquisition and validation; '
        'the upgrade requires available direct target references and does not close missing CDF coverage.')
    save(out/'summary.json',result)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k not in ('groups','original_anchor_p197')}),flush=True)
    return result


if __name__=='__main__':
    signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,240.)
    try:replay()
    finally:print('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=time.process_time(),
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)

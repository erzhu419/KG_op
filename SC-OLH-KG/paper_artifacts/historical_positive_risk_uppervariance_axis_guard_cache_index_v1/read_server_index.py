import gzip, json, signal, time
from pathlib import Path
started=time.process_time()
root=Path('.')
result={'scope':'Only current deploy historical metadata and exact frozen observation coordinates; no integrations or reference admission.'}
def read(path):
    return json.loads((root/path).read_text())
def stop(*args):
    raise TimeoutError('Two CPU-second cache-index budget exhausted')
signal.signal(signal.SIGPROF,stop)
signal.setitimer(signal.ITIMER_PROF,2.)
try:
    plan='performance/manifests/historical_positive_risk_uppervariance_axis_cached_pair_references_screen_v2.json'
    screen=read(plan)
    attached_outputs={read(p)['output'] for p in screen['joint_reference_plans']}
    live={(r['record'],r['kind']):r for r in read(screen['output']+'/summary.json')['records'] if r['candidate_actions']}
    bank=read(screen['bank_plan'])
    with gzip.open(root/bank['output']/'context.json.gz','rt') as f: context=json.load(f)
    coords={(int(o),r['label'],r['eta_label']):r['record'] for o,v in context['origins'].items() for r in v['risk']}
    matches=[];missing=[];skipped=[];directories=0;groups=0
    for folder in sorted((root/'paper_artifacts').iterdir()):
        if not folder.is_dir() or not folder.name.startswith(('historical_positive_risk','historical_conditional','historical_component_guarded')): continue
        output=str(folder)
        if output in attached_outputs: continue
        prep=folder/'preparation.json';compact=folder/'compact_summary.json'
        if not prep.exists() or not compact.exists(): continue
        if max(prep.stat().st_size,compact.stat().st_size)>262144:
            skipped.append(output);continue
        directories+=1
        selected={r['record']:r for r in read(str(prep)).get('selected',[])}
        for check in read(str(compact)).get('groups',[]):
            groups+=1
            name=check.get('record');kind=check.get('kind');row=selected.get(name,{})
            target=coords.get((row.get('origin'),row.get('label'),row.get('eta_label')))
            if (name,kind) not in live and (target,kind) not in live: continue
            matches.append(dict(output=output,record=name,kind=kind,coordinate_target=target,
                group=check.get('group'),action_indices=row.get('action_indices'),
                candidate_passed=check.get('candidate_passed'),selected_candidate=check.get('selected_candidate'),
                selected_rules_by_identity=check.get('selected_rules_by_identity')))
    result.update(complete=True,source_screen_plan=plan,attached_reference_plans=len(attached_outputs),
        unattached_metadata_outputs_scanned=directories,unattached_groups_scanned=groups,
        maximum_metadata_bytes=262144,skipped_large_metadata_outputs=skipped,matches=matches,
        live_record_kinds=len(live),exact_projected_p17_matches=[m for m in matches if m['kind']=='projected' and (m['record']=='positive_midmean_v1_n4_m4_localv_o4_h1_p17' or m['coordinate_target']=='positive_midmean_v1_n4_m4_localv_o4_h1_p17')])
    gauss='performance/manifests/historical_positive_risk_uppervariance_axis_gauss_pilot_v1.json'
    spec=read(gauss)
    name='positive_body_v1_n4_m4_v4_o4_h1_p0';kind='raw'
    case=next(r for r in spec['registered_cases'] if (r['record'],r['kind'])==(name,kind))
    source=read(case['source_plan']); comparisons=[]
    for h in (4,5):
        suffix=f'/parts/{name}/{h}_{kind}.json'
        before=read(source['output']+suffix)['rows'][case['candidate']]
        after=read(spec['output']+suffix)['rows'][case['candidate']]
        comparisons.append(dict(identity=h,statistics_equal=before['statistics']==after['statistics'],
            costs_equal=before['costs']==after['costs']))
    result['gauss_existing_selected_controls']=dict(record=name,kind=kind,source_plan=case['source_plan'],
        already_attached=case['source_plan'] in screen['joint_reference_plans'],
        selected_candidate=case['candidate'],comparisons=comparisons)
except Exception as exc:
    result.update(complete=False,error_type=type(exc).__name__,error=str(exc))
finally:
    signal.setitimer(signal.ITIMER_PROF,0)
    result['reader_process_cpu_seconds']=time.process_time()-started
    print(json.dumps(result))

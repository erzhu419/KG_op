"""Compare existing label anchors at one opened target, using shared integrals."""
import argparse
import json
import signal
import time

from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.diagnose_historical_component_guarded import compressed_read, mixed_values
from performance.validate_historical_component_transfer import prepare_contexts, part_path, record_bundle
from performance.validate_historical_label_groups import check, VARIANTS
from performance.validate_historical_label_control import controlled_row, tables

PLAN = 'performance/manifests/historical_nearby_anchor_v1.json'
RULES = ['E5', 'E7', 'F7', 'G7', 'E8', 'F8', 'H8', 'E9']


def nearest_record(target, candidates):
    return min(candidates, key=lambda n: (abs(candidates[n]['label']-target['label']), n))


def priced_row(row, identities, cached, references):
    missing = 0.; reused = 0.
    for h in identities:
        for rule in row['rules']:
            if rule in cached[h]:
                reused += cached[h][rule]['cpu_seconds']
            else:
                missing += max(r[h][rule]['cpu_seconds'] for r in references if rule in r[h])
        missing += 1.  # Existing worker setup allowance, before 20% headroom.
    return dict(row, cached_cpu_seconds=reused, estimated_new_cpu_seconds=1.2*missing,
        estimated_cpu_seconds=1.2*missing)


def anchor_rows(base, nearby, row, identities):
    original = {h:read(str(base/'parts'/f'{row["group"]:03d}_{h}_{row["kind"]}.json'))['rows'] for h in identities}
    adjacent = {h:read(str(part_path(nearby, row, h)))['rows'] for h in identities}
    return original, adjacent


def finer_reference_differences(bundle, anchor, target, kind):
    candidate=mixed_values(bundle,*tables(bundle,{h:controlled_row(anchor[h]['E8'],
        target[h]['E7'],anchor[h]['E7']) for h in bundle['utility_identities']}),kind)
    return {n:float(abs(candidate-mixed_values(bundle,*tables(bundle,
        {h:target[h][n] for h in bundle['utility_identities']}),kind)).max())
        for n in ('E8','F8','H8','E9')}


def prepare(plan=PLAN):
    tick = time.process_time(); spec = read(plan); out = ROOT/spec['output']
    out.mkdir(parents=True, exist_ok=True)
    base = ROOT/read(spec['parent_plan'])['output']; nearby = ROOT/spec['anchor_output']
    contexts, shared, original = prepare_contexts(dict(spec, records=spec['records']+[spec['anchor_record']]))
    anchor = contexts.pop(spec['anchor_record'])
    saved_anchor = read(str(nearby/'contexts.json'))[anchor['name']]
    assert saved_anchor == anchor
    candidates = dict(read('paper_artifacts/historical_label_groups_v1/contexts.json'))
    candidates[original['name']] = original; candidates[anchor['name']] = anchor
    target = contexts[spec['records'][0]]
    assert nearest_record(target, candidates) == spec['anchor_record']
    selection = read(spec['selection'])['selected']
    assert sorted({r['group'] for r in selection}) == spec['groups']
    admitted = read(str(nearby/'summary.json')); assert admitted['complete']
    approved = {(r['group'],r['kind']):r for r in admitted['groups'] if r['record']==anchor['name']}
    source = read(str(base/'bank_tail_resolution_summary.json'))['groups']
    selected = []
    for old in selection:
        gid, kind = old['group'], old['kind']
        assert approved[gid,kind]['direct_reference_checks']['passed']
        assert source[gid]['checks'][kind]['passed'] and source[gid]['selected_rules'][kind]=='E8'
        row = dict(record=target['name'], group=gid, kind=kind, rules=RULES, direct_reference=True)
        reference = anchor_rows(base, nearby, dict(row, record=anchor['name']), target['utility_identities'])
        for bank in reference:
            for h in target['utility_identities']:
                assert all(n in bank[h] for pair in VARIANTS.values() for n in pair)
                assert max(abs(bank[h][n]['diagnostics']['predictive_log_evidence_error'])
                    for pair in VARIANTS.values() for n in pair) <= spec['maximum_action_difference']
        cached = {h:read(str(part_path(out,row,h)))['rows'] if part_path(out,row,h).exists() else {}
            for h in target['utility_identities']}
        selected.append(priced_row(row, target['utility_identities'], cached, reference))
    estimate = sum(r['estimated_new_cpu_seconds'] for r in selected)
    cumulative = estimate+sum(spec['cpu_reserves'].values())
    work = dict(selected=selected, estimated_cpu_seconds=estimate,
        cumulative_estimated_cpu_seconds=cumulative, maximum_estimated_cpu_seconds=spec['maximum_estimated_cpu_seconds'],
        estimated_cost_admitted=cumulative<=spec['maximum_estimated_cpu_seconds'])
    save(out/'contexts.json', contexts); save(out/'resolution_plan.json', work)
    result = dict(complete=True, source_components={str(h):len(shared['fields'][h]['means']) for h in target['utility_identities']},
        target=target['name'], target_label=target['label'], nearest_anchor=anchor['name'], anchor_label=anchor['label'],
        original_anchor=original['name'], original_anchor_label=original['label'],
        anchor_label_distances={n:abs(c['label']-target['label']) for n,c in candidates.items()},
        reference_metadata_replay_passed=True, formal_actions=len(original['actions']),
        reused_integral_cpu_seconds=sum(r['cached_cpu_seconds'] for r in selected),
        estimated_new_integral_cpu_seconds=estimate, cumulative_estimated_cpu_seconds=cumulative,
        estimated_cost_admitted=work['estimated_cost_admitted'], cpu_seconds=time.process_time()-tick,
        full_posterior_restorations=0, copied_joint_marginal_bytes=0,
        new_physical_calls=0, new_source_fits=0, new_verifier_calls=0, independent_confirmation=False)
    save(out/'preparation.json',result)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)
    return result


def collect(plan=PLAN):
    tick = time.process_time(); spec = read(plan); out = ROOT/spec['output']
    base = ROOT/read(spec['parent_plan'])['output']; nearby = ROOT/spec['anchor_output']
    contexts = read(str(out/'contexts.json')); work = read(str(out/'resolution_plan.json'))
    rows = []; tasks = []; integral_cpu = 0.
    for kind in ('raw','projected'):
        for shard,row in enumerate(r for r in work['selected'] if r['kind']==kind):
            context = contexts[row['record']]; identities = context['utility_identities']
            bundle = record_bundle(compressed_read(base/'marginals'/f'{row["group"]:03d}.json.gz'),context)
            old, adjacent = anchor_rows(base,nearby,dict(row,record=spec['anchor_record']),identities)
            target = {h:read(str(part_path(out,row,h)))['rows'] for h in identities}
            integral_cpu += sum(target[h][n]['cpu_seconds'] for h in identities for n in RULES)
            tasks.extend(read(str(out/'tasks'/'resolution'/f'{h}_{kind}_{shard}.json')) for h in identities)
            comparisons = {n:check(bundle,a,target,kind,spec['maximum_action_difference'],True)
                for n,a in (('original',old),('nearest',adjacent))}
            chosen = comparisons['nearest']
            compact = {n:{k:v for k,v in r.items() if k not in ('selected_statistics','selected_costs')}
                for n,r in comparisons.items()}
            errors = {n:dict(control128=max(r['direct_differences'].values()),
                control32=max(r['coarse_direct_differences'].values())) for n,r in comparisons.items()}
            selected_rule_cpu = {n:sum(target[h][{'control32':'E5','control128':'E7',
                'direct256':'E8'}[r['selected_method']]]['cpu_seconds'] for h in identities)
                if r['selected_method']!='HOLD' else None for n,r in comparisons.items()}
            rows.append(dict(chosen, record=row['record'],group=row['group'],kind=kind,
                action_indices=context['groups'][str(row['group'])]['action_indices'],
                anchor_comparisons=compact, direct_action_errors=errors,
                selected_target_rule_cpu_seconds=selected_rule_cpu))
    complete = all(t['complete'] for t in tasks)
    prep = read(str(out/'preparation.json'))
    result = dict(complete=complete, groups=rows,
        all_target_references_admitted=complete and all(r['hybrid_direct_admitted'] for r in rows),
        selected_actions_by_anchor={a:{method:sum(len(r['action_indices']) for r in rows
            if r['anchor_comparisons'][a]['selected_method']==method)
            for method in ('control32','control128','direct256','HOLD')} for a in ('original','nearest')},
        maximum_direct_action_errors={a:{method:max(r['direct_action_errors'][a][method] for r in rows)
            for method in ('control32','control128')} for a in ('original','nearest')},
        selected_target_rule_cpu_seconds={a:sum(r['selected_target_rule_cpu_seconds'][a] for r in rows)
            if all(r['selected_target_rule_cpu_seconds'][a] is not None for r in rows) else None
            for a in ('original','nearest')},
        selected_rule_cpu_scope='Matched target-rule timers only. Excludes cached anchor acquisition, '
        'independent/GH/refinement validation and composition; total new phase work is reported separately.',
        cpu_seconds=integral_cpu, new_integral_cpu_seconds=integral_cpu-prep['reused_integral_cpu_seconds'],
        successful_task_cpu_seconds=sum(t['cpu_seconds'] for t in tasks),
        maximum_task_peak_rss_mb=max(t['peak_rss_mb'] for t in tasks),
        collection_cpu_seconds=time.process_time()-tick,
        target=prep['target'], nearest_anchor=prep['nearest_anchor'],original_anchor=prep['original_anchor'],
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False,
        full_predictive_integration_admitted=False,
        limitations='One opened target, five frozen worst group-kind pairs, same32 objective heads. '
        'Both anchors reuse the same target integrals and direct references; these are finite numerical checks, '
        'not full outer KG, a rigorous error bound, measured end-to-end speedup or independent deployment evidence.')
    save(out/'summary.json',result)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k!='groups'}),flush=True)
    return result


def compare_library(plan=PLAN):
    """Explore other already validated anchors after the primary comparison."""
    tick=time.process_time();spec=read(plan);out=ROOT/spec['output']
    primary=read(str(out/'summary.json'));assert primary['complete']
    context=read(str(out/'contexts.json'))[primary['target']]
    historical=ROOT/'paper_artifacts/historical_label_groups_v1'
    anchors=read(str(historical/'contexts.json'))
    old_results=read(str(historical/'summary.json'));assert old_results['complete']
    passed={(r['record'],r['group'],r['kind']) for r in old_results['groups']
        if r['direct_reference_checks'] is not None and r['direct_reference_checks']['passed']}
    base=ROOT/read(spec['parent_plan'])['output'];rows=[];finer=[]
    for row in primary['groups']:
        bundle=record_bundle(compressed_read(base/'marginals'/f'{row["group"]:03d}.json.gz'),context)
        target={h:read(str(part_path(out,row,h)))['rows'] for h in context['utility_identities']}
        old,near=anchor_rows(base,ROOT/spec['anchor_output'],dict(row,record=spec['anchor_record']),context['utility_identities'])
        finer.append(dict(group=row['group'],kind=row['kind'],differences={n:
            finer_reference_differences(bundle,a,target,row['kind']) for n,a in (('original',old),('nearest',near))}))
        for name,anchor in sorted(anchors.items(),key=lambda kv:abs(kv[1]['label']-context['label'])):
            assert (name,row['group'],row['kind']) in passed
            for key in ('prior_weights','utility_identities','replicates','eta_label','head'):
                assert context[key]==anchor[key]
            assert context['groups'][str(row['group'])]['action_indices']==anchor['groups'][str(row['group'])]['action_indices']
            cached={h:read(str(part_path(historical,dict(row,record=name),h)))['rows']
                for h in context['utility_identities']}
            assert max(abs(cached[h][n]['diagnostics']['predictive_log_evidence_error'])
                for h in cached for pair in VARIANTS.values() for n in pair)<=spec['maximum_action_difference']
            result=check(bundle,cached,target,row['kind'],spec['maximum_action_difference'],True)
            rows.append(dict(record=row['record'],group=row['group'],kind=row['kind'],anchor=name,
                label_distance=abs(anchor['label']-context['label']),
                **{k:v for k,v in result.items() if k not in ('selected_statistics','selected_costs')}))
    result=dict(complete=True,groups=rows,primary_candidate_finer_reference_differences=finer,
        cpu_seconds=time.process_time()-tick,new_integrations=0,
        independent_confirmation=False,full_predictive_integration_admitted=False,
        scope='Exploratory cache-only comparison after the preselected original/nearest comparison. '
        'Same target and direct references. No replacement of the primary nearest-anchor results or main bank.')
    save(out/'library_comparison.json',result)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k!='groups'}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','collect','compare-library'])
    parser.add_argument('--plan',default=PLAN);args=parser.parse_args()
    signal.signal(signal.SIGPROF,cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF,120.)
    {'prepare':prepare,'collect':collect,'compare-library':compare_library}[args.mode](args.plan)

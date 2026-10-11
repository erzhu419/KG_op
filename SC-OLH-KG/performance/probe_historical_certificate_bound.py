"""Price and test a marginal bound before more joint terminal integration."""
import argparse
import copy
import json
import resource
import signal
import time

import numpy as np
from scipy.special import logsumexp

from core.marginal_certificate_bound import raw_feasibility_upper, projected_feasibility_upper, certificate_upper, branch_quality_upper
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.diagnose_historical_component_guarded import compressed_read
from performance.refine_historical_observed_prediction import field_from_snapshot
from performance.refine_historical_achievable_precision import rule_pair
from representation.vectorized_joint_function_risk import raw_fantasy_moments, apply_raw_moments

PLAN = 'performance/manifests/historical_certificate_bound_v1.json'
TAGS = [(81,3),(130,3),(132,3),(83,3),(128,1),(359,1)]


def run(identity, shard=0, plan=PLAN):
    start = time.process_time(); spec = read(plan); out = ROOT/spec['output']
    (out/'tasks').mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGPROF,cpu_timeout); signal.setitimer(signal.ITIMER_PROF,spec['maximum_worker_cpu_seconds'])
    contexts = read('paper_artifacts/historical_decision_regions_v1/direct_contexts.json')
    names = [f'source_n1_o{o}_h1_p{p}' for o in spec['origins'] for p in spec['positions']]
    base = ROOT/'paper_artifacts/historical_component_bank_v1/marginals'
    policies = spec['policy_shards'][shard] if 'policy_shards' in spec else spec['policies']
    shared = {p:compressed_read(base/f'{73+p:03d}.json.gz') for p in dict.fromkeys([63]+policies)}
    fields = {p:field_from_snapshot(s['fields'][identity]) for p,s in shared.items()}
    query = fields[63]
    moments,_ = raw_fantasy_moments(query,0,np.asarray([contexts[n]['label'] for n in names]))
    cached = {}
    if 'reuse_plan' in spec:
        previous = read(spec['reuse_plan'])
        assert previous['orders'] == spec['orders'] and previous['identities'] == spec['identities']
        old = read(previous['output']+'/tasks/'+str(identity)+'.json');assert old['complete']
        cached = {(r['record'],r['policy']):r for r in old['rows']}
    rows = []; numerical_cpu = 0.; reused = 0
    for policy in policies:
        field = fields[policy]
        np.testing.assert_array_equal(field.means[:,:2],query.means)
        np.testing.assert_array_equal(field.covariances[:,:2,:2],query.covariances)
        np.testing.assert_array_equal(field.log_weights,query.log_weights)
        position = shared[policy]['points'].index(policy)
        assert shared[policy]['points'][0] == 63
        protocol = shared[policy]['protocol']
        for name,moment in zip(names,moments):
            if (name,policy) in cached:
                rows.append(cached[name,policy]);reused += 1;continue
            context = contexts[name]
            assert shared[policy]['prior_weights'] == context['prior_weights']
            future = copy.deepcopy(field); score = apply_raw_moments(future,0,moment)
            assert abs(score-context['head_log_scores'][identity]) <= 1e-10
            tick = time.process_time(); raw = {}
            for order in spec['orders']:
                value,diagnostics = raw_feasibility_upper(field,future,0,position,context['label'],score,protocol['chance_alpha'],order)
                raw[str(order)] = dict(feasible_upper=value,diagnostics=diagnostics,
                    certificates={f'{b}_{s}':certificate_upper(value,b,s,protocol) for b,s in TAGS})
            native = projected_feasibility_upper(future,position,protocol['chance_alpha'])
            numerical_cpu += time.process_time()-tick
            rows.append(dict(record=name,policy=policy,raw=raw,projected_feasible_upper=native,
                projected_certificates={f'{b}_{s}':certificate_upper(native,b,s,protocol) for b,s in TAGS}))
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(identity=identity,policy=policy,records=len(rows),cpu_seconds=time.process_time()-start)),flush=True)
    result = dict(complete=True,identity=identity,shard=shard,policies=policies,rows=rows,components=len(query.means),
        reused_pilot_rows=reused,marginal_rule_evaluations=(len(rows)-reused)*len(spec['orders']),marginal_numerical_cpu_seconds=numerical_cpu,
        cpu_seconds=time.process_time()-start,new_joint_terminal_integrations=0,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False)
    name = f'{identity}_{shard}' if 'policy_shards' in spec else str(identity)
    save(out/'tasks'/f'{name}.json',result);signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


def collect(plan=PLAN):
    start = time.process_time(); spec = read(plan); out = ROOT/spec['output']
    signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,240.)
    names = [f'{h}_{s}' for h in spec['identities'] for s in range(len(spec['policy_shards']))] if 'policy_shards' in spec else list(map(str,spec['identities']))
    tasks = [read(str(out/'tasks'/f'{name}.json')) for name in names]
    assert all(t['complete'] for t in tasks)
    parts = {(t['identity'],r['record'],r['policy']):r for t in tasks for r in t['rows']}
    maximum_change = max(abs(r['raw']['128']['feasible_upper']-r['raw']['64']['feasible_upper']) for r in parts.values())
    maximum_evidence = max(abs(r['raw'][order]['diagnostics']['log_evidence_error']) for r in parts.values() for order in ('64','128'))
    admitted = max(maximum_change,maximum_evidence) <= spec['maximum_numerical_difference']
    contexts = read('paper_artifacts/historical_decision_regions_v1/direct_contexts.json')
    layout = read('paper_artifacts/historical_cdf_bank_v1/bank_layout.json')
    covered = {}
    for entry in read('paper_artifacts/historical_body_reference_assembly_v1/reference_sources.json'):
        covered.setdefault((entry['record'],entry['kind']),set()).update(entry['action_indices'])
    native_names = {r['record'] for r in read('paper_artifacts/historical_decision_regions_v1/native_reference_summary.json')['records']}
    parent = read('performance/manifests/historical_predictive_positive_body_precision_v7.json')
    anchor = read('paper_artifacts/historical_component_bank_v1/context.json')
    anchor_bank = read('paper_artifacts/historical_component_bank_v1/bank-tail-resolution_risk_bank.json')
    records = []; maximum_reference_excess = 0.; maximum_anchor_excess = 0.;potential = {}
    for origin in spec['origins']:
        f = read(parent['head_inputs']['source_n1'][0].format(origin=origin)+'/head_records.json')
        assert f['actions'] == layout['actions']
        objective = {kind:np.asarray([r[kind][str(rule_pair(r,kind)[1])]['statistics'] for r in f['records']])
            for kind in ('raw','projected')}
        for p in spec['positions']:
            name = f'source_n1_o{origin}_h1_p{p}';context = contexts[name];prior = np.asarray(context['prior_weights'])
            assert f['prior_weights'] == prior.tolist()
            risk = read(parent['output']+'/shards/'+name+'/head_records.json')
            assert risk['actions'] == layout['actions'] and risk['prior_weights'] == prior.tolist()
            row = risk['records'][0]
            assert all(row[k] == context[k] for k in ('label','head_log_scores'))
            native = row['projected'][str(rule_pair(row,'projected')[1])]['statistics']
            scores = np.log(prior)+np.asarray([r['head_log_scores'] for r in f['records']])+context['head_log_scores']
            posterior = np.exp(scores-logsumexp(scores,axis=-1)[:,None])
            missing = posterior[:,[h for h in range(len(prior)) if h not in spec['identities']]].sum(axis=-1)
            reference = {h:read(f'paper_artifacts/historical_decision_regions_v1/direct_parts/{name}/{h}.json')['rows']['G7']['statistics'][1][0]
                for h in spec['identities']}
            for kind in ('raw','projected'):
                coverage = set(covered.get((name,kind),set()))
                if kind == 'raw':coverage.update(layout['groups'][136]['action_indices'])
                else:
                    coverage.update(i for i,a in enumerate(layout['actions']) if a['kind']=='direct')
                    if name in native_names:coverage.update(layout['groups'][63]['action_indices'])
                potential[name,kind] = np.zeros(len(f['records']))
                for h in spec['identities']:
                    part = parts[h,name,63]
                    certificates = part['raw']['128']['certificates'] if kind=='raw' else part['projected_certificates']
                    actual = reference[h] if kind=='raw' else native[h][355][0]
                    maximum_reference_excess = max(maximum_reference_excess,actual-certificates['359_1'])
                fixed = sum(posterior[:,h]*objective[kind][:,h,355,0]*(reference[h] if kind=='raw' else native[h][355][0]) for h in spec['identities'])
                for gid in spec['groups']:
                    group = layout['groups'][gid];ix = group['action_indices']
                    upper = np.zeros((len(f['records']),len(ix)));old_upper = np.zeros_like(upper)
                    for h in spec['identities']:
                        for j,a in enumerate(ix):
                            action = layout['actions'][a];size = len(action['shortlist'])
                            caps = []
                            for policy,budget in zip(action['shortlist'],action['budgets']):
                                part = parts[h,name,policy]
                                cert = part['raw']['128']['certificates'] if kind=='raw' else part['projected_certificates']
                                caps.append(cert[f'{budget}_{size}'])
                            quality = objective[kind][:,h,a]
                            old_upper[:,j] += posterior[:,h]*quality.max(axis=-1)
                            bounded = quality[:,0]*caps[0] if size==1 else branch_quality_upper(quality,np.asarray(caps),action['incumbent_position'])
                            upper[:,j] += posterior[:,h]*bounded
                    upper += missing[:,None];old_upper += missing[:,None]
                    if name == anchor['name']:
                        assert anchor['prior_weights'] == prior.tolist() and anchor['head_log_scores'] == context['head_log_scores']
                        actual = sum(posterior[:,h,None]*np.sum(objective[kind][:,h][:,ix]*np.asarray(anchor_bank['statistics'][kind])[h,ix],axis=-1)
                            for h in spec['identities'])
                        maximum_anchor_excess = max(maximum_anchor_excess,float((actual-upper).max()))
                    overlap = upper+2e-4 >= fixed[:,None]
                    old_overlap = old_upper+2e-4 >= fixed[:,None]
                    assert np.all(upper <= old_upper+1e-12)
                    unreferenced = np.asarray([a not in coverage for a in ix])
                    unresolved = overlap & unreferenced[None,:]
                    uplift = np.maximum(np.max(np.where(unreferenced[None,:],upper-fixed[:,None],-np.inf),axis=-1),0.)
                    potential[name,kind] = np.maximum(potential[name,kind],uplift)
                    records.append(dict(record=name,kind=kind,group=gid,action_head_pairs=int(overlap.size),
                        old_overlapping_action_head_pairs=int(old_overlap.sum()),new_overlapping_action_head_pairs=int(overlap.sum()),
                        unreferenced_overlapping_action_head_pairs=int(unresolved.sum()),
                        maximum_upper_minus_fixed=float((upper-fixed[:,None]).max()),
                        maximum_upper_reduction=float((old_upper-upper).max()),
                        all_candidates_excluded=bool(not overlap.any())))
    from scipy.special import roots_legendre
    weights = dict(zip(range(192,224),.9*roots_legendre(32)[1]/2))
    envelope = {kind:sum(contexts[name]['prior_weights'][int(name.split('_o')[1].split('_')[0])]*weights[int(name.rsplit('_p',1)[1])]*float(value.mean())
        for (name,k),value in potential.items() if k==kind) for kind in ('raw','projected')}
    result = dict(complete=True,raw_marginal_numerical_gate_passed=admitted,
        cached_reference_containment_passed=max(maximum_reference_excess,maximum_anchor_excess)<=spec['maximum_numerical_difference'],
        maximum_validated_query_certificate_above_bound=float(maximum_reference_excess),
        maximum_validated_anchor_action_above_bound=float(maximum_anchor_excess),
        maximum_raw64_128_feasibility_bound_difference=maximum_change,maximum_raw_log_evidence_error=maximum_evidence,
        records=records,totals={kind:{key:sum(r[key] for r in records if r['kind']==kind)
            for key in ('action_head_pairs','old_overlapping_action_head_pairs','new_overlapping_action_head_pairs','unreferenced_overlapping_action_head_pairs')} for kind in ('raw','projected')},
        conditional_unreferenced_body_uplift_envelope=envelope,
        components={str(t['identity']):t['components'] for t in tasks},
        successful_worker_cpu_seconds=sum(t['cpu_seconds'] for t in tasks),
        marginal_numerical_cpu_seconds=sum(t['marginal_numerical_cpu_seconds'] for t in tasks),
        marginal_rule_evaluations=sum(t['marginal_rule_evaluations'] for t in tasks),
        reused_pilot_rows=sum(t.get('reused_pilot_rows',0) for t in tasks),collection_cpu_seconds=time.process_time()-start,
        new_joint_terminal_integrations=0,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        independent_confirmation=False,full_predictive_integration_admitted=False,
        scope='Opened body32 labels and the manifest action groups. Gaussian tangent and verifier branch relaxations are analytic upper bounds; '
            'raw query-eta expectation uses all positive components and64/128 numerical checks. Projection uses the original paid moments. '
            'Original prior,all8 identity likelihoods,365-action contract and359 budget remain unchanged. '
            'Exclusion is conditional on matched cached objective tables and numerical fixed-action references,with the historical2e-4 margin. '
            'No algorithm action is removed and no continuous quadrature error or fullKG admission is claimed.')
    save(out/'summary.json',result);signal.setitimer(signal.ITIMER_PROF,0.)
    compact = {k:v for k,v in result.items() if k!='records'}
    compact['group_record_checks'] = len(records)
    compact['unreferenced_group_record_checks_remaining'] = sum(r['unreferenced_overlapping_action_head_pairs']>0 for r in records)
    compact['closest_remaining'] = sorted([r for r in records if r['unreferenced_overlapping_action_head_pairs']],
        key=lambda r:-r['maximum_upper_minus_fixed'])[:32]
    save(out/'compact_summary.json',compact)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('mode',choices=['run','collect']);p.add_argument('--identity',type=int,choices=[4,5]);p.add_argument('--shard',type=int,default=0);p.add_argument('--plan',default=PLAN);args=p.parse_args()
    try:
        run(args.identity,args.shard,args.plan) if args.mode=='run' else collect(args.plan)
    finally:
        print('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=time.process_time(),
            peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)

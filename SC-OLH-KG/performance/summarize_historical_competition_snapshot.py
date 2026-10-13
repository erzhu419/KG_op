"""Export small diagnostics from an already closed competition phase."""
import argparse
import collections
import json
from pathlib import Path

from performance.complete_historical_source_heads import ROOT, read, save
from performance.diagnose_historical_component_guarded import compressed_read
from performance.validate_historical_component_bank import action_groups


def summarize(plan, output):
    spec=read(plan);completion=read(spec['phase_completion'])
    assert completion['state']=='phase_complete' and completion['all_reference_precision_passed']
    screen=read(spec['output']+'/summary.json')
    context=compressed_read(ROOT/read(spec['bank_plan'])['output']/'context.json.gz')
    coordinates={r['record']:(int(o),r['cdf']) for o,data in context['origins'].items() for r in data['risk']}
    groups=action_groups(context['actions']);diagnostics={}
    mean_edges=[0.,.8,.875,.9125,.95,.99,1.]
    variance_edges=[0.,.2,.8,1.]
    def interval(value,edges):
        i=next(i for i in range(len(edges)-1) if value<edges[i+1])
        return f'[{edges[i]},{edges[i+1]}]'
    def total(rows):
        return dict(records=len(rows),candidate_groups=sum(len(r['candidate_groups']) for r in rows),
            weighted_upper_gap=sum(r['weighted_unresolved_upper_gap'] for r in rows))
    for kind in ('raw','projected'):
        rows=[r for r in screen['records'] if r['kind']==kind]
        by_mean=collections.defaultdict(list);by_variance=collections.defaultdict(list);by_origin=collections.defaultdict(list)
        priority=collections.defaultdict(lambda:dict(record_count=0,weighted_group_priority=0.))
        for r in rows:
            origin,cdf=coordinates[r['record']]
            by_mean[interval(cdf[0],mean_edges)].append(r)
            by_variance[interval(cdf[1],variance_edges)].append(r)
            by_origin[str(origin)].append(r)
            for gid in r['candidate_groups']:
                priority[gid]['record_count']+=1
                priority[gid]['weighted_group_priority']+=r['outer_grid_weight']*r['candidate_group_expected_positive_gap'][str(gid)]
        envelope=screen['weighted_bank_envelope']['kinds'][kind]
        assert abs(total(rows)['weighted_upper_gap']-envelope['unresolved_upper_gap'])<1e-14
        zero=[r for r in rows if r['candidate_groups'] and r['mean_unresolved_upper_gap']==0.]
        diagnostics[kind]=dict(by_mean_cdf={k:total(v) for k,v in by_mean.items()},
            by_variance_cdf={k:total(v) for k,v in by_variance.items()},by_origin={k:total(v) for k,v in by_origin.items()},
            zero_positive_gap_but_still_overlapping=total(zero),
            remaining_groups=[dict(group=g,kind=groups[g]['kind'],action_indices=groups[g]['action_indices'],
                shortlist=context['actions'][groups[g]['action_indices'][0]]['shortlist'],**p)
                for g,p in sorted(priority.items(),key=lambda x:-x[1]['weighted_group_priority'])],
            group_priority_scope='Group priorities overlap and must not be summed as disjoint contributions to the total gap.',
            largest_records=[dict(record=r['record'],origin=coordinates[r['record']][0],cdf=coordinates[r['record']][1],
                weight=r['outer_grid_weight'],weighted_upper_gap=r['weighted_unresolved_upper_gap'],
                mean_upper_gap=r['mean_unresolved_upper_gap'],candidate_groups=r['candidate_groups'])
                for r in sorted(rows,key=lambda r:-r['weighted_unresolved_upper_gap'])[:12]],
            bank_reference_contrast=envelope['best_reference_contrast'],
            nominal_bank_upper_contrast=envelope['best_reference_contrast']+envelope['unresolved_upper_gap'])
    number=int(Path(spec['phase_completion']).parent.name.rsplit('_v',1)[1]);history=[]
    prefix=spec['competition_prefix']
    for n in range(1,number+1):
        folder=ROOT/'paper_artifacts'/f'{prefix}_competition_v{n}'
        c=read(str(folder/'completion.json'))
        if 'weighted_bank_envelope' not in c:
            old_plan=read(f'performance/manifests/{prefix}_competition_v{n}.json')
            old_bound=read(old_plan['followup_bound_plan'])
            c['weighted_bank_envelope']=read(old_bound['output']+'/compact_summary.json')['weighted_bank_envelope']
        checks=read(str(folder/'compact_summary.json'))
        history.append(dict(phase=n,workers=c.get('reference_workers',len(read(str(folder/'preparation.json'))['jobs'])),groups=len(checks['groups']),
            recorded_cpu_seconds=c.get('total_recorded_process_cpu_seconds',c.get('recorded_process_cpu_seconds')),
            estimated_cpu_seconds=c['estimated_phase_cpu_seconds'],
            maximum_reference_difference=max(v['maximum_difference'] for g in checks['groups'] for v in g['checks']['comparisons']),
            raw_gap=c['weighted_bank_envelope']['kinds']['raw']['unresolved_upper_gap'],
            projected_gap=c['weighted_bank_envelope']['kinds']['projected']['unresolved_upper_gap'],
            raw_best_reference_contrast=c['weighted_bank_envelope']['kinds']['raw']['best_reference_contrast']))
    phase_folder=ROOT/Path(spec['phase_completion']).parent
    replay=read(str(phase_folder/'incumbent_replay_final.json'))
    result=dict(source_bound_plan=plan,closed_phase=number,
        protocol={k:spec[k] for k in ('risk_observations','objective_heads_per_origin','all_identity_likelihoods','source_components','action_count','remaining_calls')},
        prior_weights=context['prior_weights'],control_variate=spec['control_variate'],
        envelope=screen['weighted_bank_envelope'],totals=screen['totals'],diagnostics=diagnostics,
        recommendation=replay['raw_recommendation']['policy_totals'],
        maximum_omitted_utility_identity_mass=replay['maximum_omitted_utility_identity_mass'],
        all_conditional_actions_closed=replay['all_conditional_actions_closed'],
        history=history,history_recorded_cpu_seconds=sum(r['recorded_cpu_seconds'] for r in history),
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,independent_confirmation=False,
        scope='Read-only export of the existing finite224/F16 bank; no new integrations, likelihood changes or confirmation reads.')
    save(ROOT/output,result);print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();summarize(args.plan,args.output)

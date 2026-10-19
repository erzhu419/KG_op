"""Diagnose cap/branch slack on recently verified cached guarded actions."""
import json
import resource
import signal
import time

import numpy as np
from scipy.special import logsumexp

from core.branch_coupled_certificate_bound import coupled_branch_quality_upper
from core.marginal_certificate_bound import branch_quality_upper, certificate_upper
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.diagnose_historical_component_guarded import compressed_read
from performance.validate_historical_component_bank import selected_reference_rule


def run(plan,output):
    started=time.process_time();signal.signal(signal.SIGPROF,cpu_timeout);signal.setitimer(signal.ITIMER_PROF,30.)
    spec=read(plan);base=ROOT/read(spec['bank_plan'])['output'];context=compressed_read(base/'context.json.gz')
    parts={(t['identity'],r['record'],r['policy']):r
        for h in (4,5) for shard in range(len(spec['policy_shards']))
        for t in [read(str(base/'tasks'/f'{h}_{shard}.json'))] for r in t['rows']}
    risk={r['record']:(int(o),r) for o,data in context['origins'].items() for r in data['risk']}
    candidates=[]
    for reference in spec['joint_reference_plans'][-4:]:
        ref=read(reference);prep=read(ref['output']+'/preparation.json')
        selected={r['record']:r for r in prep['selected']}
        for check in read(ref['output']+'/compact_summary.json')['groups']:
            if check['kind']!='raw' or check['group'] not in (34,63,71):continue
            row=selected[check['record']];a=context['actions'][row['action_indices'][0]]
            if a['incumbent_position']==0:continue
            candidates.append((ref,check,row))
    rows=[];prior=np.asarray(context['prior_weights'])
    for ref,check,row in candidates[:8]:
        name=row['record'];origin,r=risk[name];data=context['origins'][str(origin)]
        scores=np.log(prior)+np.asarray(data['objective_scores'])+r['head_log_scores']
        posterior=np.exp(scores-logsumexp(scores,axis=-1)[:,None])
        ix=row['action_indices'];values={k:np.zeros((16,len(ix))) for k in ('old','coupled','cap_proxy','reference')}
        margins=[]
        for h in (4,5):
            part=read(ref['output']+'/parts/'+name+f'/{h}_raw.json')
            statistics=np.asarray(part['rows'][selected_reference_rule(check,h)]['statistics'])
            for j,aidx in enumerate(ix):
                a=context['actions'][aidx];inc=a['incumbent_position'];assert inc in (1,2)
                q=np.asarray(data['objective']['raw'])[:,h,aidx]
                caps=np.asarray([certificate_upper(parts[h,name,p]['raw']['128']['feasible_upper'],budget,3,context['protocol'])
                    for p,budget in zip(a['shortlist'],a['budgets'])])
                s=statistics[j];proxy=caps.copy();proxy[0]=s[0]+s[3];proxy[inc]=s[inc]+s[3]
                weight=posterior[:,h]
                values['old'][:,j]+=weight*branch_quality_upper(q,caps,inc)
                values['coupled'][:,j]+=weight*coupled_branch_quality_upper(q,caps,inc)
                values['cap_proxy'][:,j]+=weight*coupled_branch_quality_upper(q,proxy,inc)
                values['reference'][:,j]+=weight*(q@s)
                margins.append(dict(identity=h,action=aidx,incumbent_position=inc,
                    primary_cap=float(caps[0]),primary_point_estimate=float(proxy[0]),
                    incumbent_cap=float(caps[inc]),incumbent_point_estimate=float(proxy[inc])))
        for j,aidx in enumerate(ix):
            rows.append(dict(record=name,group=check['group'],action=aidx,cdf=r['cdf'],
                old_slack=float(np.mean(values['old'][:,j]-values['reference'][:,j])),
                coupled_slack=float(np.mean(values['coupled'][:,j]-values['reference'][:,j])),
                point_cap_proxy_slack=float(np.mean(values['cap_proxy'][:,j]-values['reference'][:,j]))))
        rows[-1]['marginal_examples']=margins
    assert rows
    totals={k:float(np.mean([r[k] for r in rows])) for k in ('old_slack','coupled_slack','point_cap_proxy_slack')}
    result=dict(source_bound_plan=plan,records=len(candidates[:8]),action_comparisons=len(rows),means=totals,
        rows=rows,analysis_cpu_seconds=time.process_time()-started,new_joint_reference_integrations=0,
        scope='Selected recently verified raw guarded cases. P/I certificate marginals are point estimates reconstructed from the same cached branch statistics; proxy values are diagnostic, not new valid screening caps or global G contributions.')
    save(ROOT/output,result);signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)
    return result

"""Compare cached action rankings at matched objective records; no integration."""
import json
import numpy as np

from performance.complete_historical_source_heads import ROOT, read, save
from performance.diagnose_historical_component_guarded import mixed_values

out=ROOT/'paper_artifacts/historical_cdf_bank_v1'
near=ROOT/'paper_artifacts/historical_label_bank_v1'
contexts=read(str(out/'contexts.json'));layout=read(str(out/'bank_layout.json'))
near_name='source_n1_o4_h1_p207';near_context=read(str(near/'contexts.json'))[near_name]
assert layout==read(str(near/'bank_layout.json'))
banks={name:read(str(out/(name+'_risk_bank.json'))) for name in ('coarse','selected')}
near_bank=read(str(near/'independent_risk_bank.json'))[near_name];results={}
precision=read(str(out/'summary.json'))['groups']
for name,context in contexts.items():
    assert context['prior_weights']==near_context['prior_weights']
    first=next(iter(context['groups'].values()))['opposite']
    opposite={k:np.zeros((len(first),len(context['prior_weights']),len(layout['actions']),4)) for k in ('raw','projected')}
    for group in layout['groups']:
        local=context['groups'][str(group['group'])];ix=group['action_indices']
        assert local['opposite']==near_context['groups'][str(group['group'])]['opposite']
        for kind in opposite:opposite[kind][:,:,ix]=np.asarray([r[kind] for r in local['opposite']])
    full=dict(context,actions=layout['actions'],opposite=[dict(head_log_scores=r['head_log_scores'],
        **{k:data[i].tolist() for k,data in opposite.items()}) for i,r in enumerate(first)])
    near_full=dict(full,head_log_scores=near_context['head_log_scores']);results[name]={}
    for kind in opposite:
        values={n:mixed_values(full,b[name]['statistics'][kind],b[name]['costs'][kind],kind) for n,b in banks.items()}
        prior=mixed_values(near_full,near_bank['statistics'][kind],near_bank['costs'][kind],kind)
        winner=values['selected'].argmax(1);old=prior.argmax(1)
        gain=values['selected'].max(1)-values['selected'][np.arange(len(old)),old]
        order=np.argsort(values['selected'],axis=1);challenger=order[:,-2]
        ordered=np.sort(values['selected'],axis=1);gap=ordered[:,-1]-ordered[:,-2]
        admitted={i for group in precision if group['record']==name and group['kind']==kind and
            group['selected_method']!='HOLD' for i in group['action_indices']}
        if kind=='projected':admitted.update(i for i,a in enumerate(layout['actions']) if a['kind']=='direct')
        results[name][kind]=dict(changed_near_recommendations=int(np.count_nonzero(winner!=old)),
            changed_by_pilot_updates=int(np.count_nonzero(winner!=values['coarse'].argmax(1))),
            maximum_switch_advantage=float(gain.max()),heads_with_switch_advantage_above_1e_minus4=int(np.count_nonzero(gain>1e-4)),
            minimum_runner_up_gap=float(gap.min()),maximum_runner_up_gap=float(gap.max()),
            winner_actions={str(i):dict(action=layout['actions'][i],target_precision_admitted=int(i) in admitted)
                for i in np.unique(winner)},
            runner_up_actions={str(i):layout['actions'][i] for i in np.unique(challenger)},
            selected_recommendations=winner.tolist(),near_recommendations=old.tolist())
result=dict(records=results,new_integrations=0,new_physical_calls=0,independent_confirmation=False,
    scope='Same32 frozen objective records, all365 actions and all identity scores/weights. Rankings use partial coarse banks with explicit pilot updates; remaining actions are unvalidated. Switch advantages are conditional model values, not a full KG integral or independent deployment evidence.')
save(out/'decision_relevance.json',result);print(json.dumps(result))

"""Compare cached GH32/GH64 rules only where the original screen passed."""
from collections import Counter
import numpy as np
from performance.validate_historical_component_bank import PLAN, ROOT, read, save, compose_head_grid

spec=read(PLAN);out=ROOT/spec['output'];context=read(str(out/'context.json'))
summary=read(str(out/'adaptive_summary.json'));result={}
for kind in ('raw','projected'):
    original=np.asarray(context['baseline'][kind]['statistics']).copy();cheaper=original.copy()
    costs=np.asarray(context['baseline'][kind]['costs']).copy();cheap_costs=costs.copy()
    current_cpu=0.;cheap_cpu=0.;matched=Counter()
    for group in summary['groups']:
        name=group['selected_rules'][kind]
        if name=='native_reference':continue
        candidate='E7' if name=='G7' else name
        if candidate=='E7':
            checks=group['checks'][kind]
            assert checks['passed']
            assert {('E5','E7'),('E7','F7'),('E7','G7')}<=set((c['low'],c['high']) for c in checks['comparisons'])
        for h in context['utility_identities']:
            rows=read(str(out/'parts'/f'{group["group"]:03d}_{h}_{kind}.json'))['rows']
            for table,price,rule in [(original,costs,name),(cheaper,cheap_costs,candidate)]:
                table[h,group['action_indices']]=rows[rule]['statistics']
                price[h,group['action_indices']]=rows[rule]['costs']
            if candidate=='E7' and group['kind']=='guarded' and rows['E7']['cpu_seconds']>0.:
                current_cpu+=rows[name]['cpu_seconds'];cheap_cpu+=rows[candidate]['cpu_seconds'];matched['identity_group_pairs']+=1
    f=np.asarray([r[kind] for r in context['opposite']]);fs=[r['head_log_scores'] for r in context['opposite']]
    old=compose_head_grid(f,original[None],costs[None],fs,[context['head_log_scores']],context['prior_weights'])[0][:,0]
    new=compose_head_grid(f,cheaper[None],cheap_costs[None],fs,[context['head_log_scores']],context['prior_weights'])[0][:,0]
    result[kind]=dict(maximum_action_difference=float(abs(new-old).max()),
        changed_recommendation_count=int(np.count_nonzero(new.argmax(axis=1)!=old.argmax(axis=1))),
        matched_guarded_pairs=matched['identity_group_pairs'],gh64_cpu_seconds=current_cpu,
        gh32_cpu_seconds=cheap_cpu,matched_speedup=current_cpu/cheap_cpu)
result.update(new_integral_evaluations=0,full_predictive_integration_admitted=False,independent_confirmation=False,
    scope='Reuse existing E7/F7/G7 checks on already passed groups; no change to stored admission summaries. Timers compare matched nonzero cached guarded evaluations only, not total KG runtime.')
save(out/'cheaper_rule_diagnostic.json',result);print(result)

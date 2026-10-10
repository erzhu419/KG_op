"""Price missing target checks from saved matching anchor timers; no integration."""
import json

from performance.complete_historical_source_heads import ROOT, read, save
from performance.validate_historical_component_transfer import part_path

out=ROOT/'paper_artifacts/historical_label_bank_v1'
base=ROOT/'paper_artifacts/historical_component_bank_v1'
groups=read(str(out/'fine_groups.json'));context=read(str(out/'contexts.json'))
selected={(r['record'],r['group'],r['kind']):r for r in read(str(out/'fine_plan.json'))['selected']}
results={}
for name,rule in [('independent128','F7'),('gh64_128','G7')]:
    costs={k:0. for k in ('raw','projected')};counts={k:0 for k in costs}
    for group in groups:
        if group['target_precision_admitted']:continue
        row=selected[group['record'],group['group'],group['kind']];counts[row['kind']]+=1
        for h in context[row['record']]['utility_identities']:
            if rule in read(str(part_path(out,row,h)))['rows']:continue
            source=read(str(base/'parts'/f'{row["group"]:03d}_{h}_{row["kind"]}.json'))['rows']
            costs[row['kind']]+=source[rule]['cpu_seconds']
    estimate=1.2*(sum(costs.values())+32)
    results[name]=dict(missing_group_counts=counts,estimated_missing_integral_cpu_seconds=costs,
        estimated_cpu_seconds=estimate,maximum_estimated_cpu_seconds=3600,
        estimated_cost_admitted=estimate<=3600,dispatched=False)
result=dict(plans=results,scope='Remaining unadmitted target groups only. Reuse historical matching rule timers, plus32 worker CPU seconds and20 percent headroom. Full GH check exceeding the phase cap must be redesigned before dispatch; no precision admission or new integration.')
save(out/'next_precision_cost.json',result);print(json.dumps(result))

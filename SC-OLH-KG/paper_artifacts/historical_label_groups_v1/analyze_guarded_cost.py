"""Project guarded-bank candidate cost from existing saved rule timers."""
import json
from pathlib import Path

out=Path(__file__).resolve().parent
base=out.parent/'historical_component_bank_v1'
summary=json.loads((base/'bank_tail_resolution_summary.json').read_text())
kinds={}
for kind in ('raw','projected'):
    rows=[]
    for group in summary['groups']:
        if group['kind']!='guarded':continue
        for h in (4,5):
            part=json.loads((base/'parts'/f'{group["group"]:03d}_{h}_{kind}.json').read_text())['rows']
            selected=group['selected_rules'][kind]
            rows.append(dict(group=group['group'],identity=h,selected_rule=selected,
                low_cpu=part['E7']['cpu_seconds'],selected_cpu=part[selected]['cpu_seconds']))
    valid=[r for r in rows if r['low_cpu'] and r['selected_cpu']]
    kinds[kind]=dict(groups=len(rows)//2,matched_group_heads=len(valid),
        missing_group_heads=[r for r in rows if not r['low_cpu'] or not r['selected_cpu']],
        low_cpu=sum(r['low_cpu'] for r in valid),selected_cpu=sum(r['selected_cpu'] for r in valid))
result=dict(kinds=kinds,projected_guarded_cost_ratio=sum(r['selected_cpu'] for r in kinds.values())/
    sum(r['low_cpu'] for r in kinds.values()),
    scope='Historic timers from separate runs, not a matched full-bank benchmark. Exclude reused zero-timer pairs, direct actions, initialization and precision checks. E7 control has not been validated for the full guarded bank.')
(out/'guarded_cost_probe.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))

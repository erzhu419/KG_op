"""Read saved same-worker timers; do not perform any new integrals."""
import json
from pathlib import Path

from performance.validate_historical_component_transfer import part_path

out=Path(__file__).resolve().parent
summary=json.loads((out/'summary.json').read_text())
contexts=json.loads((out/'contexts.json').read_text())
pairs=[];selected_errors=[]
for row in summary['groups']:
    method=row['selected_method']
    if method=='control32':
        selected_errors.extend(row['coarse_direct_differences'].values())
        for h in contexts[row['record']]['utility_identities']:
            part=json.loads(part_path(out,row,h).read_text())['rows']
            pairs.append(dict(record=row['record'],group=row['group'],kind=row['kind'],identity=h,
                low_cpu_seconds=part['E5']['cpu_seconds'],old_cpu_seconds=part['E7']['cpu_seconds']))
    elif method=='control128':selected_errors.extend(row['direct_differences'].values())
    elif method=='direct256':
        selected_errors.extend(c['maximum_difference'] for c in row['direct_reference_checks']['comparisons']
            if c['low']=='E8' and c['high']=='E9')
low=sum(r['low_cpu_seconds'] for r in pairs);old=sum(r['old_cpu_seconds'] for r in pairs)
result=dict(same_worker_pairs=pairs,low_cpu_seconds=low,old_cpu_seconds=old,
    candidate_integral_speedup=old/low,maximum_selected_difference_to_direct_reference=max(selected_errors),
    new_integrals=0,scope='Saved E5/E7 integral timers from the same original worker for the six combinations admitted to control32. Includes all retained components; excludes anchor setup, mixing, validation and other actions. Not a full-bank or full-KG benchmark.')
(out/'direct_cost_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='same_worker_pairs'}))

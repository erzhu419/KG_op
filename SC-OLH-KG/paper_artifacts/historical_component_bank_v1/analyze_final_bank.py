"""Summarize cached final-bank timers on the server; evaluate no new integrals."""
import json
from collections import Counter
from pathlib import Path

out = Path(__file__).resolve().parent
summary = json.loads((out / 'bank_tail_resolution_summary.json').read_text())
result = {}
for kind in ('raw', 'projected'):
    counts = Counter()
    times = {h: 0. for h in (4, 5)}
    cheaper = 0.
    for group in summary['groups']:
        rule = group['selected_rules'][kind]
        counts[rule] += len(group['action_indices'])
        if rule == 'native_reference':
            continue
        for h in times:
            rows = json.loads((out / 'parts' / f'{group["group"]:03d}_{h}_{kind}.json').read_text())['rows']
            times[h] += rows[rule]['cpu_seconds']
            cheaper += rows['E7' if rule == 'G7' else rule]['cpu_seconds']
    result[kind] = dict(actions_by_rule=dict(counts), selected_rule_cpu_seconds=sum(times.values()),
        by_identity=times, cached_gh32_substitution_cpu_seconds=cheaper)
result.update(summary='bank_tail_resolution_summary.json',
    scope='One cached bank only. Selected-rule timers exclude precision checks, preparation and native direct reference. GH32 substitution costs are diagnostic and do not change selected rules. No runtime guarantee for other observations.')
(out / 'final_bank_cost.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))

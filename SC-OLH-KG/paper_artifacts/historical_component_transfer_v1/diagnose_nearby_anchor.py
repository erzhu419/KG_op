"""Cache-only feasibility check for an observation-level control variate."""
import json
from pathlib import Path

import numpy as np

from performance.diagnose_historical_component_guarded import compressed_read, mixed_values
from performance.validate_historical_component_transfer import record_bundle

out = Path(__file__).resolve().parent
base = out.parent / 'historical_component_bank_v1'
kind, gid = 'raw', 52
contexts = json.loads((out / 'contexts.json').read_text())
shared = compressed_read(base / 'marginals' / f'{gid:03d}.json.gz')


def diagnose(target):
    context = contexts[target]
    bundle = record_bundle(shared, context)
    values, ranges = {}, {}
    replay_error = 0.
    for name, high, low in [('candidate', 'E9', 'E7'), ('refined', 'E10', 'E8'),
            ('independent', 'F9', 'F7'), ('gh', 'L9', 'G7')]:
        stats, costs = np.zeros((8, 3, 4)), np.zeros((8, 3))
        for h in bundle['utility_identities']:
            anchor = json.loads((base / 'parts' / f'{gid:03d}_{h}_{kind}.json').read_text())['rows']
            current = json.loads((out / 'parts' / target / f'{gid:03d}_{h}_{kind}.json').read_text())['rows']
            a, b, c = anchor[high], current[low], anchor[low]
            ma, mb, mc = [np.exp(r['diagnostics']['predictive_log_evidence_error']) for r in (a, b, c)]
            for key, table in [('statistics', stats), ('costs', costs)]:
                av, bv, cv = [np.asarray(r[key]) for r in (a, b, c)]
                table[h] = (ma * av + mb * bv - mc * cv) / mb
                replay_error = max(replay_error, float(abs((ma * av + mc * cv - mc * cv) / ma - av).max()))
        values[name] = mixed_values(bundle, stats, costs, kind)
        active = stats[bundle['utility_identities']]
        ranges[name] = dict(minimum_statistic=float(active.min()), maximum_statistic=float(active.max()),
            minimum_cost=float(costs[bundle['utility_identities']].min()), maximum_cost=float(costs.max()))
    comparisons = {name: float(abs(v - values['candidate']).max()) for name, v in values.items() if name != 'candidate'}
    result = dict(target=target, anchor=shared['name'], group=gid, kind=kind,
        target_label=context['label'], anchor_label=shared['label'], label_difference=context['label']-shared['label'],
        same_label_replay_error=replay_error, comparisons=comparisons, ranges=ranges,
        all_control_diagnostics_passed=replay_error<=1e-12 and max(comparisons.values())<=1e-4 and
            all(-1e-12<=r['minimum_statistic'] and r['maximum_statistic']<=1+1e-12 and
                r['minimum_cost']>=0 and r['maximum_cost']<=359 for r in ranges.values()),
        direct_high_target_check_available=False, new_integral_evaluations=0,
        new_physical_calls=0, new_source_fits=0, new_verifier_calls=0, independent_confirmation=False,
        full_predictive_integration_admitted=False,
        scope='One label pair on the identical frozen joint prior, same component rules and all three budgets. High anchor plus paired unnormalized numerator differences, divided by target mass; all identity scores/weights remain target-specific. Feasibility diagnostic only; no change to admission summaries. Direct high-rule target comparison is still required.')
    assert replay_error <= 1e-12
    return result


results = [diagnose(target) for target in contexts]
nearby = next(r for r in results if r['target'] == 'source_n1_o4_h1_p207')
assert abs(nearby['label_difference']) < 1e-4
(out / 'nearby_anchor_diagnostic.json').write_text(json.dumps(nearby, indent=2) + '\n')
result = dict(records=results, all_records_control_diagnostics_passed=all(r['all_control_diagnostics_passed'] for r in results),
    new_integral_evaluations=0, independent_confirmation=False, full_predictive_integration_admitted=False)
(out / 'cross_label_anchor_diagnostic.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))

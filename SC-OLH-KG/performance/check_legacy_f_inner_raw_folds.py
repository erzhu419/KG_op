"""Complete the registered training-protocol diagnosis,reusing canonical folds."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.prepare_legacy_f_inner_raw_contract import run as prepare
from performance.check_legacy_f_inner_raw_contract import run as forecast
from performance.run_legacy_related_inventory_source import save

EXECUTION = 'performance/manifests/legacy_joint_population_f_inner_raw_folds_execution_v1.json'


def run():
    spec = read(EXECUTION); old = read(spec['reused_canonical_raw_results'])
    assert old['diagnostic_interpreted'] and old['numerical_density_verified'] and len(old['rows']) == 8
    prepare(spec['remaining_source_execution']); source_spec = read(spec['remaining_source_execution'])
    source = read(Path(source_spec['output'])/'results.json'); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    if not source['all_source_laws_complete']:
        save(out/'results.json', {'execution': spec, 'all_ten_inner_folds_complete': False, 'remaining_source_fit_or_rule_failed': True}); return
    forecast(spec['remaining_raw_execution']); raw_spec = read(spec['remaining_raw_execution'])
    new = read(Path(raw_spec['output'])/'results.json')
    if not new['diagnostic_interpreted']:
        save(out/'results.json', {'execution': spec, 'all_ten_inner_folds_complete': False, 'remaining_raw_numerical_failure': True}); return
    rows = old['rows']+new['rows']; keys = [(r['layout'], r['query_task'], r['arm']) for r in rows]
    assert len(keys) == len(set(keys)) == 40
    assert {(r['layout'], r['query_task']) for r in rows} == {(l, i) for l in ('5', '6') for i in (0, 2, 3, 4, 5)}
    summary = {}
    for layout in ('5', '6'):
        summary[layout] = {}
        for contract in ('point', 'mixture'):
            native = [r for r in rows if r['layout'] == layout and r['arm'] == f'native_{contract}']
            shape = [r for r in rows if r['layout'] == layout and r['arm'] == f'shape_{contract}']
            baseline = {k: float(np.mean([r['scores'][k] for r in native])) for k in native[0]['scores']}
            selected = {k: float(np.mean([r['scores'][k] for r in shape])) for k in shape[0]['scores']}
            comparisons = [(a['scores'], next(b['scores'] for b in native if b['query_task'] == a['query_task'])) for a in shape]
            summary[layout][contract] = {'native': baseline, 'shape': selected,
                'shape_minus_native': {k: selected[k]-baseline[k] for k in baseline},
                'mse_relative_change': selected['f_mse']/baseline['f_mse']-1,
                'mse_improved_folds': sum(a['f_mse'] < b['f_mse'] for a, b in comparisons),
                'raw_density_improved_folds': sum(a['f_raw_batch_density'] > b['f_raw_batch_density'] for a, b in comparisons)}
    original_source = read(spec['reused_canonical_source_results'])
    result = {'execution': spec, 'rows': rows, 'summary': summary, 'all_ten_inner_folds_complete': True,
        'all_twenty_mixture_numerical_references_verified': True,
        'cost': {'new_risk_coordinate_refits_this_expansion': source['cost']['actual_risk_coordinate_refits'],
            'new_risk_objective_fits_this_expansion': source['cost']['actual_risk_objective_fits'],
            'new_variance_integrations_this_expansion': source['cost']['actual_source_variance_integrations'],
            'new_raw_forecasts_this_expansion': len(new['rows']), 'new_numerical_reference_forecasts_this_expansion': len(new['numerical_checks']),
            'reused_raw_forecasts': len(old['rows']), 'reused_numerical_reference_forecasts': len(old['numerical_checks']),
            'new_simulator_calls': 0, 'new_verification_calls': 0, 'new_terminal_integrations': 0},
        'cpu_seconds_for_remaining_folds': source['cpu_seconds']+new['cpu_seconds'],
        'cpu_seconds_for_stages160_163': original_source['cpu_seconds']+old['cpu_seconds']+source['cpu_seconds']+new['cpu_seconds'],
        'component_artifact_bytes_including_reused': old['component_artifact_bytes']+new['component_artifact_bytes'],
        'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False,
        'training_diagnostic_only': True, 'stopped_shape_candidate_not_promoted': True,
        'interpretation': 'Overlapping source4 folds participated in q selection. Paired training-protocol diagnosis,not independent validation,physical-factor recovery or decision superiority.'}
    save(out/'results.json', result)
    print('all-ten source4 raw diagnosis', {l: {c: {k: v[k] for k in ('mse_relative_change', 'mse_improved_folds', 'raw_density_improved_folds')}
        for c, v in group.items()} for l, group in summary.items()}, flush=True)


if __name__ == '__main__':
    run()

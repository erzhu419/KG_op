"""Validate one-source predictive integration against saved two-origin grids."""

from performance.check_historical_query_grids import grid_rules, PLAN
from performance.check_historical_source_head_grid import baseline_for
from performance.complete_historical_source_heads import ROOT, read, save


def run(plan=PLAN):
    registration = read(plan); source_plan = 'performance/manifests/historical_source_head_grid_v2.json'
    spec = read(source_plan); source_out = ROOT/spec['output']; result = read(str((source_out/'results.json').relative_to(ROOT)))
    heads = [read(str((source_out/f'origin4_head{h}.json').relative_to(ROOT)))['records'] for h in (0, 1)]
    for key in ('maximum_predictive_mass_residual', 'predictive_density_ratio_limits'):
        spec[key] = registration[key]
    spec['maximum_proposal_reference_difference'] = registration['maximum_proposal_reference_difference']
    replay = grid_rules(heads, result['source_weights'], baseline_for(spec), 4, [4, 5], spec)
    differences = [abs(a['raw_net_reward_control_variate']-b['raw_net_reward_control_variate'])
        for a, b in zip(replay['rows'], result['rows'])]
    assert len(replay['rows']) == len(result['rows'])
    passed = max(differences) <= spec['maximum_proposal_reference_difference'] and all(replay[k] for k in
        ('numerical_gates_passed', 'identity_remainder_gate_passed', 'proposal_mass_gate_passed', 'proposal_ratio_gate_passed'))
    result = {'execution_plan': plan, 'reference_plan': source_plan, **replay,
        'maximum_proposal_reference_difference': max(differences), 'all_registered_proposal_gates_passed': passed,
        'source_models_and_shared_posterior_weights_changed': False, 'head_state_requirement_reduction': [240, 120],
        'independent_confirmation': False}
    save(ROOT/registration['proposal_calibration'], result)
    print('proposal calibration', passed, 'maximum difference', max(differences), 'fine net', replay['rows'][-1]['raw_net_reward_control_variate'])
    return result


if __name__ == '__main__':
    run()

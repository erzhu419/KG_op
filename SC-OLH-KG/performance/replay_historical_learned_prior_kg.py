"""Replay the completed learned-prior KG artifacts without repeating calculations."""

import time

from performance.complete_historical_source_heads import ROOT, read, save
from performance.connect_historical_learned_prior_kg import run as connection
from performance.check_historical_learned_prior_kg_precision import run as precision
from performance.check_historical_learned_terminal_resolution import run as resolution
from performance.check_historical_learned_direct_terminal import run as direct
from performance.check_historical_learned_direct_tower import run as tower


def run():
    started = time.process_time(); connected = connection()
    kg = precision('performance/manifests/historical_learned_prior_kg_completion_v1.json')
    terminal = resolution(); analytic = direct(); fixed = tower()
    tests = [read(f'paper_artifacts/{name}/unit_tests.json') for name in (
        'historical_learned_prior_kg_precision_v1', 'historical_learned_direct_terminal_v1')]
    diagnostics = read('paper_artifacts/historical_learned_prior_kg_completion_v1/decision_diagnostics.json')
    result = {'connected_heads': connected['connected_heads'],
        'connected_complete_identities': connected['connected_complete_identities'],
        'all12_actions_complete': kg['all12_registered_actions_complete'],
        'historical_absolute_precision_gate_actions_passed': kg['actions_precision_passed'],
        'terminal_resolution_complete': terminal['all_registered_cases_complete'],
        'analytic_direct_actions_complete': analytic['all64_direct_actions_complete'],
        'fixed_direct_tower_rules_complete': fixed['all_registered_rules_complete'],
        'unit_tests_passed': sum(r['tests_run'] for r in tests),
        'all_unit_tests_successful': all(r['successful'] for r in tests),
        'positive_nets_without_positive_adaptation_minus_budget_cost': diagnostics[
            'positive_nominal_nets_with_nonpositive_adaptation_minus_budget_cost'],
        'cpu_seconds_recorded_this_stage': sum(r['cpu_seconds'] for r in (kg, terminal, analytic, fixed)),
        'cached_replay_cpu_seconds': time.process_time() - started,
        'server_jobs': 0, 'new_target_simulator_calls': 0, 'new_verifier_calls': 0,
        'decision_ranking_certified': False, 'independent_confirmation': False,
        'next': 'Connect analytic direct utility to the acquisition and resolve guarded-terminal precision. '
            'Separate remaining paid-filter projection drift,then compare matched GP controls and independent decisions.'}
    save(ROOT/'paper_artifacts/historical_learned_prior_kg_completion_v1/phase_results.json', result)
    print(result)
    return result


if __name__ == '__main__':
    run()

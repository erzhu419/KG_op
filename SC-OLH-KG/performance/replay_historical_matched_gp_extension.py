"""Cached-only GP extension replay and joint development-stage summary."""

import time

import numpy as np
from scipy.stats import qmc

from performance.complete_historical_source_heads import ROOT, read, save
from performance.connect_historical_matched_gp import summarize_query
from performance.extend_historical_matched_gp import PLAN, restore_saved_model


def run():
    started = time.process_time(); spec = read(PLAN); parent = read(spec['parent_plan'])
    out = ROOT/spec['output']; path = out/'results.json'
    saved = read(str(path.relative_to(ROOT))); assert saved['all_registered_cases_complete']
    tests = [read(str((out/n).relative_to(ROOT))) for n in ('unit_tests.json', 'adjacent_prefix_test.json')]
    assert all(t['successful'] for t in tests)
    bank = read(str((out/'futures.json').relative_to(ROOT))); old = read(spec['completed_futures'])
    before = {r['identity_fantasy_position']: r for r in old['records']}
    assert len(bank['records']) == saved['total_joint_futures'] == 72
    for record in bank['records']:
        assert record['primary_mean_refined']
        if record['identity_fantasy_position'] in before:
            assert record == before[record['identity_fantasy_position']]
    assert len(before) == 22 and bank['actions'] == old['actions']
    row = next(r for r in read(spec['connection_states'])['rows'] if
        (r['layout'], r['arm']) == (spec['layout'], parent['lookahead_arm']))
    model = restore_saved_model(row)
    rule = dict(parent, **{k: spec[k] for k in ('predictive_power', 'tail_counts', 'complement_prefixes')})
    uniforms = qmc.Sobol(2, scramble=True, seed=rule['seed']+1).random_base2(rule['predictive_power'])
    replay = summarize_query(bank['records'], model, row['terminal'], bank['actions'], uniforms, rule)
    # Earlier summarizer incorrectly used first-to-last rather than adjacent prefixes.
    # Preserve that diagnostic, then repair summaries without new model evaluations.
    for new, previous in zip(replay['rows'], saved['rows']):
        for key in ('raw_adaptation_value', 'raw_net_reward_control_variate', 'action_selection_loss'):
            assert abs(new[key]-previous[key]) < 1e-14
    if saved['numerical_changes'] != replay['numerical_changes']:
        old_path = out/'summary_before_adjacent_prefix_fix.json'
        if not old_path.exists():
            save(old_path, saved)
    corrected = dict(saved, **replay)
    save(path, corrected)
    previous = read(spec['parent_results'])
    layout6 = next(r for r in previous['lookahead'] if r['layout'] == '6')
    phase = dict(corrected, both_opened_gp_query_numerical_gates_passed=
        corrected['numerical_gates_passed'] and layout6['numerical_gates_passed'],
        layout6_existing_result_preserved=layout6,
        both_common_task_gp_stop_choices_match_learned_prior=previous['both_common_task_gp_stop_choices_match_learned_prior'],
        terminal_summary=previous['terminal_summary'],
        layout5_learned_raw_kg_reference=previous['layout5_learned_raw_kg_reference'],
        successful_tests=previous['successful_tests']+sum(t['tests_run'] for t in tests),
        cpu_seconds_including_parent_and_tests=previous['cpu_seconds_including_tests']+
            saved['cpu_seconds']+sum(t['cpu_seconds'] for t in tests),
        cached_replay_cpu_seconds=time.process_time()-started,
        next='Reuse historical per-head raw terminal statistics on a Cartesian predictive grid, '
            'reconstruct each complete action value and keep the maximum inside each joint fantasy. '
            'Use that cheaper calculation to freeze complete query policies before opening a '
            'matched independent evaluation block. The development controls do not establish prior superiority.')
    save(out/'phase_results.json', phase)
    print({k: phase[k] for k in ('both_opened_gp_query_numerical_gates_passed',
        'both_common_task_gp_stop_choices_match_learned_prior', 'successful_tests',
        'cpu_seconds_including_parent_and_tests')})
    print(phase['rows'][-1]['raw_net_reward_control_variate'], phase['numerical_changes'])
    return phase


if __name__ == '__main__':
    run()

"""Reproduce matched GP diagnostics from saved action arrays; no new fits."""

import time

import numpy as np
from scipy.stats import qmc

from performance.complete_historical_source_heads import ROOT, read, save
from performance.connect_historical_matched_gp import PLAN, summarize_query
from representation.joint_response_mixture import JointResponseMixture


def run():
    started = time.process_time(); spec = read(PLAN); out = ROOT/spec['output']
    result = read(str((out/'results.json').relative_to(ROOT)))
    tests = read(str((out/'unit_tests.json').relative_to(ROOT)))
    assert result['all_registered_cases_complete'] and tests['successful']
    fits = read(str((out/'source_fit.json').relative_to(ROOT)))['by_layout']
    connected = read(str((out/'connected_states.json').relative_to(ROOT)))
    states = read(read(spec['connection_plan'])['development_states'])['rows']
    source = read(spec['reference_terminal'])['rows']
    learned_states = read('paper_artifacts/historical_learned_prior_kg_connection_v1/connected_states.json')['rows']
    selection = read(spec['selection_plan']); terminal_summary = []
    for row in connected['rows']:
        state = next(s for s in states if s['layout'] == row['layout'])
        fit = fits[row['layout']]
        assert fit['source_task_ids'] == selection['source_tasks'] == row['source_task_ids']
        assert fit['source_label_count_per_head'] == 48
        assert row['candidate_bank_indices'] == state['candidate_bank_indices']
        assert row['paid_observations'] == state['paid_observations'] and row['paid_variance_labels'] == 0
        assert row['posterior']['paid_episodes'] == 2
        assert all(h['replicates'] == 1 for h in row['posterior']['history'])
        by_budget = {}
        for budget in (360, 359):
            pair = [next(r for r in row['terminal'] if (r['remaining_calls'], r['power']) ==
                (budget, p)) for p in spec['terminal_powers']]
            low, high = pair
            reference = next(r for r in source if (r['layout'], r['remaining_calls'], r['power']) ==
                (row['layout'], budget, high['power']))
            assert low['actions'] == high['actions'] == reference['actions']
            reserve = 2*state['terminal_protocol']['comparison_budget']
            assert max(sum(a['budgets'])+(reserve if a['kind'] == 'guarded' else 0) for a in high['actions']) == budget
            for key, calculated in (
                ('own_model_gap_for_learned_choice', high['best_reward']-high['values'][reference['best_action_index']]),
                ('learned_model_gap_for_this_choice', reference['best_reward']-reference['values'][high['best_action_index']])):
                assert abs(high[key]-calculated) < 1e-14
            by_budget[str(budget)] = {'action': high['best_action'],
                'same_action_as_learned_prior': high['best_action'] == reference['best_action'],
                'same_choice_at_both_powers': low['best_action_index'] == high['best_action_index'],
                'maximum_absolute_terminal_change': float(np.max(abs(np.asarray(high['values'])-low['values']))),
                'own_model_gap_for_learned_choice': high['own_model_gap_for_learned_choice'],
                'learned_model_gap_for_this_choice': high['learned_model_gap_for_this_choice']}
        learned = next(s for s in learned_states if s['layout'] == row['layout'])
        terminal_summary.append({'layout': row['layout'], 'arm': row['arm'], 'by_budget': by_budget,
            'paid_joint_log_score': sum(h['mixture_log_score'] for h in row['posterior']['history']),
            'learned_prior_paid_joint_log_score': sum(h['mixture_log_score'] for h in learned['posterior']['history'])})
    uniforms = qmc.Sobol(2, scramble=True, seed=spec['seed']+1).random_base2(spec['predictive_power'])
    for layout in spec['queries']:
        bank = read(str((out/f'layout{layout}_futures.json').relative_to(ROOT)))
        row = next(r for r in connected['rows'] if (r['layout'], r['arm']) == (layout, spec['lookahead_arm']))
        assert len(bank['records']) == 22 and all(r['primary_mean_refined'] for r in bank['records'])
        assert all(a['budgets'] for a in bank['actions']) and len(bank['actions']) == 40
        model = JointResponseMixture(['cached GP'], [(None, None)])
        replay = summarize_query(bank['records'], model, row['terminal'], bank['actions'], uniforms, spec)
        saved = next(r for r in result['lookahead'] if r['layout'] == layout)
        assert replay['numerical_gates_passed'] == saved['numerical_gates_passed']
        for current, previous in zip(replay['rows'], saved['rows']):
            for key in ('raw_adaptation_value', 'raw_net_reward_control_variate', 'action_selection_loss',
                    'weighted_absolute_terminal_contrast_difference', 'raw_fixed_action_tower_residual'):
                assert abs(current[key]-previous[key]) < 1e-14
        assert replay['numerical_changes'] == saved['numerical_changes']
    source_kg = read(spec['reference_raw_kg'])['rows'][-1]
    result.update(terminal_summary=terminal_summary,
        all_same_source_roster_paid_data_actions_budgets_preserved=True,
        both_common_task_gp_stop_choices_match_learned_prior=all(r['by_budget']['360']['same_action_as_learned_prior']
            for r in terminal_summary if r['arm'] == spec['lookahead_arm']),
        raw_kg_cached_reproduction_max_tolerance=1e-14,
        layout5_learned_raw_kg_reference=source_kg['raw_net_reward_control_variate'],
        successful_tests=tests['tests_run'], all_unit_tests_successful=tests['successful'],
        cpu_seconds_including_tests=result['cpu_seconds']+tests['cpu_seconds'],
        cached_replay_cpu_seconds=time.process_time()-started,
        next='Reuse the44 saved GP futures to diagnose its complementary predictive region and extend the '
            'fixed numerical rule without refitting. Then freeze complete query policies and compare them '
            'on a matched independent block. Neither own-model KG differences nor two development '
            'predictive log scores establish deployment superiority.')
    save(out/'phase_results.json', result)
    print({k: result[k] for k in ('all_registered_cases_complete', 'source_fits_converged',
        'both_common_task_gp_stop_choices_match_learned_prior', 'lookahead_numerical_gates_passed',
        'successful_tests', 'cpu_seconds_including_tests')})
    for row in result['lookahead']:
        print(row['layout'], row['rows'][-1]['raw_net_reward_control_variate'], row['numerical_changes'])
    return result


if __name__ == '__main__':
    run()

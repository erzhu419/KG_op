"""Compare adaptive and frozen queries using fully verified saved results only."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
DUAL = 'source_structural_dual_channel'
FROZEN = 'source_structural_frozen_query'
DEFAULT = 'paper_artifacts/mainline_d1000_frozen_query_v1'


def read(path):
    return json.loads(path.read_text())


def outcome(row):
    return {'primary': row['action']['primary'], 'certified': row['verification']['certified'],
            **{key: row['scores'][key] for key in ('actual_safe_reward', 'conditional_safe_reward',
                'true_feasible', 'false_certificate', 'quality')}}


def analyze(output=DEFAULT):
    folder = ROOT / output
    started, wall = time.process_time(), time.monotonic()
    attempt = {'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'failed',
               'new_simulator_calls': 0, 'new_truth_calls': 0, 'new_integrations': 0,
               'new_source_fits': 0, 'new_bootstrap_draws': 0}
    try:
        result = read(folder / 'results.json')
        assert result['all_recommendations_frozen_before_verification']
        assert result['all_verification_finished_before_truth_scoring']
        summary = read(folder / 'summary.json')
        spec = read(folder / 'frozen_inputs.json')['plan']
        online = spec['policies'][1]
        expected = spec['task_count'] * sum(len(spec['policies_by_model'][m]) for m in spec['models'])
        assert len(result['rows']) == expected
        paired = next(r for r in summary['paired_reward_differences']
                      if r['lhs'] == [DUAL, online] and r['rhs'] == [FROZEN, online])
        rows = {(r['task_index'], r['arm'], r['policy']): r for r in result['rows']}
        tasks, frozen_changes, terminal_changes, paid_counts = [], [], [], []
        for task in range(spec['task_count']):
            cases = {arm: read(folder / f'task{task:02d}' / arm / 'case.json') for arm in (DUAL, FROZEN)}
            dual, frozen = (cases[arm]['rounds'] for arm in (DUAL, FROZEN))
            records = {arm: rows[task, arm, online] for arm in (DUAL, FROZEN)}
            tasks.append({'task_index': task, 'task_id': records[DUAL]['task_id'],
                'dual_queries': [r['decision']['query'] for r in dual],
                'frozen_queries': [r['decision']['query'] for r in frozen],
                'first_query_same': dual[0]['decision']['query'] == frozen[0]['decision']['query'],
                'first_raw_same': dual[0]['raw_observation'] == frozen[0]['raw_observation'],
                'second_query_changed': dual[1]['decision']['query'] != frozen[1]['decision']['query'],
                'second_raw_same_when_query_same': (dual[1]['raw_observation'] == frozen[1]['raw_observation']
                    if dual[1]['decision']['query'] == frozen[1]['decision']['query'] else None),
                'dual': outcome(records[DUAL]), 'frozen': outcome(records[FROZEN])})
            frozen_changes.extend(max(r['frozen_query_max_changes'].values()) for r in frozen)
            terminal_changes.extend(max(r['no_cross_nonqueried_max_changes'].values())
                                    for case in cases.values() for r in case['rounds'])
            paid_counts.extend({'task_index': task, 'arm': arm, 'search_calls': records[arm]['search_calls'],
                'round_paid': [r['paid_episodes'] for r in cases[arm]['rounds']],
                'verification_budget': records[arm]['action']['budgets'][0]} for arm in cases)
        classifications = []
        for arm in spec['models']:
            for policy in spec['policies_by_model'][arm]:
                counts = Counter()
                for row in result['rows']:
                    if (row['arm'], row['policy']) == (arm, policy):
                        certified, feasible = row['verification']['certified'], row['scores']['true_feasible']
                        category = ('certified_true_feasible' if feasible else 'false_certificate') if certified else (
                            'uncertified_true_feasible' if feasible else 'uncertified_true_infeasible')
                        counts[category] += 1
                classifications.append({'arm': arm, 'policy': policy, 'counts': dict(counts)})
        wlt = {}
        for metric in ('actual_safe_reward', 'conditional_safe_reward'):
            delta = paired[metric]['per_unit_difference']
            wins, losses = sum(x > 1e-12 for x in delta), sum(x < -1e-12 for x in delta)
            wlt[metric] = {'wins': wins, 'losses': losses, 'ties': len(delta) - wins - losses}
        analysis = {'source_results': str(folder / 'results.json'), 'primary_dual_minus_frozen': paired,
            'win_loss_tie': wlt, 'tasks': tasks,
            'certificate_transitions_frozen_to_dual': dict(Counter(
                f'{int(t["frozen"]["certified"])}->{int(t["dual"]["certified"])}' for t in tasks)),
            'verification_outcome_categories': classifications,
            'controls': {'first_query_and_raw_same_all_tasks': all(t['first_query_same'] and t['first_raw_same'] for t in tasks),
                'second_query_changed_tasks': sum(t['second_query_changed'] for t in tasks),
                'frozen_rounds': len(frozen_changes), 'expected_frozen_rounds': spec['task_count'] * spec['rounds'],
                'frozen_query_max_change': max(frozen_changes), 'all_frozen_query_changes_zero': all(x == 0. for x in frozen_changes),
                'terminal_nonqueried_max_change': max(terminal_changes),
                'all_terminal_nonqueried_changes_zero': all(x == 0. for x in terminal_changes),
                'paid_counts': paid_counts, 'all_real_paid_four_and_verifier_358': all(
                    r['search_calls'] == 4 and r['round_paid'] == [3, 4] and r['verification_budget'] == 358 for r in paid_counts)},
            'full_dual_agreement': summary['dual_channel']['full_query_channel_agreement'],
            'scope': 'Paired total decision benefit of updating the query posterior after initialization; '
                'not a pure cross-policy propagation effect. Both terminal channels use their own real observations. '
                'Bootstrap intervals are copied from the completed summary, without new resampling.'}
        (folder / 'query_adaptation_analysis.json').write_text(json.dumps(analysis, indent=2, allow_nan=False) + '\n')
        attempt['status'] = 'completed'
        return analysis
    except Exception:
        attempt['traceback'] = traceback.format_exc()
        raise
    finally:
        attempt.update(cpu_seconds=time.process_time() - started, wall_seconds=time.monotonic() - wall,
                       maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        with (folder / 'query_adaptation_analysis_attempts.jsonl').open('a') as handle:
            handle.write(json.dumps(attempt, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=DEFAULT)
    print(json.dumps(analyze(parser.parse_args().output)['win_loss_tie'], allow_nan=False))

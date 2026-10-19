"""Five-role nonempty saved pools, seal barriers and accounting substitutes."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from performance import summarize_mainline_source_alignment as runner
from performance import run_mainline_two_comparison_development as development
from performance import mainline_pathway_replication as pathway


INITIAL = [dict(point=[index], raw=[5., -1.]) for index in range(10)]


def bundle(points):
    bank = [(index,) for index in range(10)] + points
    actions = [(0,), *points]
    history, stages = deepcopy(INITIAL), []
    for step, choice in enumerate((1, 2, 1)):
        objective = [9., 3., 3., 3.]
        objective[choice] = 1.
        margin = [1., -1., -1., -1.]
        chosen, total, mode, count = runner.previous.competition.score_choice(objective, margin)
        stages.append(dict(paid_prefix=10 + step, action_ids=[bank.index(point) for point in actions],
            observed_ids=[bank.index(tuple(row['point'])) for row in history],
            selected_index=chosen, selected_id=bank.index(actions[chosen]), action_sources=['substitute'] * 4,
            score=dict(objective_mean=objective, sampled_objective=objective, sampled_margin=margin,
                total=total.tolist(), sampled_feasibility_mode=mode, sampled_feasible_count=count,
                objective_epistemic=[.1] * 4, constraint_mean=[-1.] * 4, constraint_epistemic=[.1] * 4,
                constraint_aleatoric=[.1] * 4, sampled_constraint=margin, probability_violation=[.01] * 4)))
        history.append(dict(point=list(actions[chosen]), raw=[1., -1.]))
    return dict(points=bank, history=history, audit=dict(stages=stages))


def source_fixture(root):
    plan = dict(output_dir='alignment', frozen=dict(d=1),
        tasks=[dict(geometry_task_seed=1, domain='family', geometry=dict(problem_kwargs={}))],
        streams=[dict(name=f'stream{index}', manifest=f'child_{index}.json') for index in (1, 2)],
        terminal_arms=[dict(name=arm, history_arm=arm.removesuffix('_native_terminal')) for arm in runner.factorial.ARMS],
        budget=dict(terminal_cells=22, verification_cap_per_cell=352),
        historical_accounts=['old_batch/completion.json', 'alignment/cluster_control/stage91_accounting.json'],
        source_asset_seal_file='alignment/source_preparation/source_asset_seal.json',
        source_warning_document='docs/source_fit_limits.md', reference_manifest='original_protocol.json')
    runner.save(root / plan['source_asset_seal_file'], dict(status='sealed_sources_only', source_models_ready=True,
        numerical_limits='Queue dominant objective heads have low source ESS; precision_certified=false.',
        source_fit_cpu_seconds=58., target_execution_allowed=False, failures=[]))
    cases = []
    for number, stream in enumerate(plan['streams']):
        child = dict(output_dir=f'alignment/stream_{number + 1}', frozen=dict(d=1))
        source_path = f'source_{number}.json'
        source = dict(output_dir=child['output_dir'] + '/search', execution_failures=[],
            task=dict(d=1, L=100, sigma=.04, alpha=.05, weights=[1.]))
        runner.save(root / stream['manifest'], child)
        runner.save(root / source_path, source)
        runner.save(root / child['output_dir'] / 'prepared.json',
            dict(replicates=[dict(families=[dict(domain='family', search_manifest=source_path)])]))
        directory = root / source['output_dir'] / 'seed_1'
        runner.save(directory / 'initialization.json', dict(history=deepcopy(INITIAL)))
        np.savez_compressed(directory / 'neighbors.npz', points=np.asarray([[10 + number * 20]], dtype=np.int16))
        role_points = {}
        for index, arm in enumerate(runner.ROLES):
            points = [(10 + number * 20 + index * 3 + offset,) for offset in range(3)]
            role_points[arm] = points
            saved = bundle(points)
            runner.save(directory / arm / 'search.json', dict(history=saved['history']))
            runner.save(directory / arm / 'query_audit.json', saved['audit'])
            np.savez_compressed(directory / arm / 'query_policies.npz', points=np.asarray(saved['points'], dtype=np.int16))
        for index, cell in enumerate(plan['terminal_arms']):
            arm, history_arm = cell['name'], cell['history_arm']
            if history_arm in role_points:
                points = role_points[history_arm]
            elif history_arm == 'initialization_only':
                points = [(0,), (1,), (2,)]
                runner.save(directory / history_arm / 'search.json', dict(history=deepcopy(INITIAL)))
            else:
                points = [(100 + number * 20 + index,), (0,), (1,)]
                runner.save(directory / history_arm / 'search.json', dict(history=deepcopy(INITIAL) +
                    [dict(point=list(points[0]), raw=[1., -1.])] * 3))
            item = dict(frozen_shortlist=[dict(point=list(point)) for point in points])
            terminal = Path('alignment') / stream['name'] / 'terminals' / arm
            runner.save(root / terminal / 'input.json', item)
            runner.save(root / terminal / 'verification.json', dict(deployed=list(points[0]), verification=dict(
                certified=True, verification_budget=80, max_verification_budget=352,
                attempts=[dict(candidate_index=0, certified=True, upper_margin=-1.)], objective_comparisons=[])))
            cases.append(dict(stream=stream['name'], seed=1, domain='family', arm=arm, history_arm=history_arm,
                source_case=str((directory / history_arm).relative_to(root)), search_manifest=source_path,
                old=dict(status='complete', input_file=str(terminal / 'input.json'),
                         verification_file=str(terminal / 'verification.json'))))
    runner.save(root / 'plan.json', plan)
    runner.save(root / plan['output_dir'] / 'execution_status.json',
        dict(status='sealed', truth_access_allowed=True, cases=cases))
    return plan


class TruthProblem:
    tau, alpha, weights, ref_point = 0., .05, np.array([1.]), np.array([10.])

    def simulate(self, *args, **kwargs):
        raise AssertionError('A postseal analytic substitute cannot simulate')

    def true_objective(self, point):
        return 5. if point[0] < 10 else 1. + point[0] / 100

    def true_constraint_mean(self, point):
        return -1.

    def true_sigma(self, point):
        return [0., 0.]


def ledger(root, path, rows):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def test_all_five_nonempty_pools_and_all_eleven_terminals_survive_real_file_adapter(tmp_path, monkeypatch):
    plan = source_fixture(tmp_path)
    monkeypatch.setattr(development, 'build_problem', lambda item: TruthProblem())
    monkeypatch.setattr(runner.previous.competition, '_swaps', lambda *args: pytest.fail('No new score substitutions'))
    original_roles = runner.previous.audit.ROLES, runner.previous.competition.ARMS, pathway.ROLES
    inventory = runner.freeze(plan, root=tmp_path)
    assert inventory['registered_terminal_cells'] == 22
    assert inventory['query_arms'] == list(runner.ROLES) and inventory['registered_query_states'] == 30
    assert inventory['inventory_complete'] and inventory['known_distinct_policy_task_units'] == 46
    assert not (tmp_path / 'alignment/cases').exists()
    index = runner.read(tmp_path / inventory['tasks'][0]['index_file'])
    assert all(len(row['cases']) == 11 and len(row['captured_stages']) == 5 for row in index['streams'])
    assert all(all(count == 3 for count in row['captured_stages'].values()) for row in index['streams'])
    result = runner.score_task(plan, 1, root=tmp_path)
    assert result['status'] == 'complete', result
    assert result['cached_policies'] == 46
    assert sum(row['new_truth_function_calls'] for row in result['streams']) == 138
    for stream in result['streams']:
        assert len(stream['outcomes']) == 11 and len(stream['queries']) == 15
        assert stream['coverage']['registered_states'] == 15 and stream['coverage']['unavailable_states'] == 0
        assert stream['coverage']['registered_query_arms'] == 5
        for arm in (runner.factorial.PERMUTED_AFTER, runner.factorial.PERMUTED_POINT):
            outcome = next(row for row in stream['outcomes'] if row['arm'] == arm)
            assert outcome['metrics']['C'] == 1 and outcome['metrics']['G'] > 0
            assert outcome['discovery_category'] == 'multiple_safe_unobserved_competition'
            assert outcome['returned_repeat_count'] == 1
            repeats = [row for row in stream['queries'] if row['arm'] == arm and row['paid_prefix'] == 12]
            assert repeats[0]['query_action_kind'] == 'repeat'
        assert all(not row.get('substitutions') for row in stream['mechanism']['comparisons'])
    assert (runner.previous.audit.ROLES, runner.previous.competition.ARMS, pathway.ROLES) == original_roles
    monkeypatch.setattr(runner.previous.audit, 'score_points', lambda *args: pytest.fail('Completed scoring repeated'))
    assert runner.score_task(plan, 1, root=tmp_path) == result


@pytest.mark.parametrize('entry', ['freeze', 'score_task', 'score_batch', 'collect'])
def test_last_terminal_cell_blocks_every_entry_even_with_a_saved_inventory(tmp_path, monkeypatch, entry):
    plan = source_fixture(tmp_path)
    runner.freeze(plan, root=tmp_path)
    path = tmp_path / 'alignment/execution_status.json'
    seal = runner.read(path)
    seal['cases'].pop()
    runner.save(path, seal)
    monkeypatch.setattr(runner.previous.audit, 'task_material', lambda *args: pytest.fail('Pools read before seal'))
    monkeypatch.setattr(runner.previous.audit, 'score_points', lambda *args: pytest.fail('Truth read before seal'))
    with pytest.raises(ValueError, match='globally sealed'):
        if entry == 'freeze':
            runner.freeze(plan, root=tmp_path)
        elif entry == 'score_task':
            runner.score_task(plan, 1, root=tmp_path)
        elif entry == 'score_batch':
            runner.score_batch(plan, 'plan.json', 0, 1, 1, root=tmp_path)
        else:
            runner.collect(plan, 'plan.json', root=tmp_path)
    assert not (tmp_path / 'alignment/cases').exists()


def test_old_three_role_inventory_is_not_reused_as_full_five_role_coverage(tmp_path):
    plan = source_fixture(tmp_path)
    inventory = runner.freeze(plan, root=tmp_path)
    inventory['query_arms'] = inventory['query_arms'][:3]
    inventory['registered_query_states'] = 18
    runner.save(tmp_path / 'alignment/mechanism/inventory.json', inventory)
    with pytest.raises(ValueError, match='all five saved role'):
        runner.score_task(plan, 1, root=tmp_path)


def test_missing_saved_role_states_remain_partial_without_hiding_its_terminal_result(tmp_path, monkeypatch):
    plan = source_fixture(tmp_path)
    arm = runner.factorial.PERMUTED_POINT
    (tmp_path / 'alignment/stream_2/search/seed_1' / arm / 'query_audit.json').unlink()
    monkeypatch.setattr(development, 'build_problem', lambda item: TruthProblem())
    inventory = runner.freeze(plan, root=tmp_path)
    assert not inventory['inventory_complete']
    assert inventory['maximum_analytic_function_calls'] is None
    result = runner.score_task(plan, 1, root=tmp_path)
    assert result['status'] == 'complete'
    second = result['streams'][1]
    assert second['search_coverage_status'] == 'partial'
    assert second['coverage']['captured_states'] == 12 and second['coverage']['unavailable_states'] == 3
    assert len(second['outcomes']) == 11
    assert next(row for row in second['outcomes'] if row['arm'] == arm)['metrics']['C'] == 1


def test_failed_terminal_scoring_failure_and_fees_preserve_all_controls_and_source_limits(tmp_path, monkeypatch):
    plan = source_fixture(tmp_path)
    seal_path = tmp_path / 'alignment/execution_status.json'
    seal = runner.read(seal_path)
    failed = next(row for row in seal['cases'] if row['stream'] == 'stream1' and row['arm'] == runner.factorial.PERMUTED_AFTER)
    failed['old'] = dict(status='failed', input_file=failed['old']['input_file'],
        failure=dict(error='terminal failed after paid prefix', new_verification_calls=None, call_accounting_complete=False))
    runner.save(seal_path, seal)
    ledger(tmp_path, 'alignment/source_models/family/fit_native/attempts.jsonl',
        [dict(stage='fit-source-alignment-native', status='completed', new_source_fits=1, cpu_seconds=58.)])
    ledger(tmp_path, 'alignment/source_preparation/attempts.jsonl',
        [dict(stage='describe-source-alignment-fits', status='completed', cpu_seconds=.5)])
    ledger(tmp_path, 'alignment/cluster_control/validation/attempts.jsonl',
        [dict(stage='substitute-check', status='completed', cpu_seconds=2.)])
    ledger(tmp_path, 'alignment/stream_1/execution/attempts.jsonl',
        [dict(stage='search', status='completed', new_target_initial_calls=20, new_target_search_calls=48, cpu_seconds=3.),
         dict(stage='verify', status='failed', new_verification_calls=31, cpu_seconds=4.)])
    monkeypatch.setattr(development, 'build_problem', lambda item: TruthProblem())
    runner.freeze(plan, root=tmp_path)
    result = runner.score_task(plan, 1, root=tmp_path)
    # A subsequent saved scoring failure may coexist with the earlier terminal failure.
    result['status'] = 'scoring_failed'
    result['streams'][0] = runner.previous.failed_stream(plan, plan['tasks'][0], plan['streams'][0], 'saved scoring failure')
    runner.save(tmp_path / 'alignment/cases/seed_1/result.json', result)
    completed = runner.collect(plan, 'plan.json', root=tmp_path)
    assert completed['status'] == 'complete_with_retained_failures'
    assert len(completed['terminal_failures']) == len(completed['scoring_failures']) == 1
    accounts = completed['costs']
    assert accounts['known_recorded']['new_source_fits'] == 0
    assert accounts['retained_source_known_recorded']['new_source_fits'] == 1
    assert accounts['retained_source_known_recorded']['cpu_seconds'] == 58.5
    assert accounts['retained_check_known_recorded']['cpu_seconds'] == 2.
    assert accounts['known_recorded_physical_target_calls'] == 99
    assert accounts['physical_target_calls'] is None
    assert len(accounts['failed_attempts']) == 1
    assert accounts['historical_accounts'] == plan['historical_accounts']
    headline = runner.read(tmp_path / completed['headline_file'])
    assert len(headline['independent']['controls']) == 11
    assert len(headline['independent']['comparisons']) == 6
    assert all(row['delta']['C']['all_registered_mean'] is None for row in headline['independent']['comparisons'])
    assert 'Queue' in headline['source_numerical_limits'] and 'low source ESS' in headline['source_numerical_limits']
    assert runner.read(tmp_path / 'child_1.json')['status'] == 'complete_with_retained_failures'
    assert runner.read(tmp_path / 'child_2.json')['status'] == 'complete'
    assert runner.collect(plan, 'plan.json', root=tmp_path) == completed


def test_five_role_context_restores_old_defaults_after_a_scoring_exception(tmp_path, monkeypatch):
    plan = source_fixture(tmp_path)
    runner.freeze(plan, root=tmp_path)
    before = runner.previous.audit.ROLES, runner.previous.competition.ARMS, pathway.ROLES
    def failure(*args, **kwargs):
        assert runner.previous.audit.ROLES == runner.ROLES
        raise RuntimeError('synthetic outer failure')
    monkeypatch.setattr(runner.previous, 'score_task', failure)
    with pytest.raises(RuntimeError, match='synthetic outer failure'):
        runner.score_task(plan, 1, root=tmp_path)
    assert (runner.previous.audit.ROLES, runner.previous.competition.ARMS, pathway.ROLES) == before

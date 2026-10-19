"""Four-cell raw reuse, independent blocks, paid failure and explicit resume."""
from copy import deepcopy
from itertools import product
import json
from types import SimpleNamespace

import numpy as np
import pytest

from core import terminal_two_comparison as tournament
from core import terminal_verification as original
from performance import mainline_reference_coverage as wiring
from performance import mainline_reference_coverage_execution as execution


def inputs(*, same_third=False):
    item = dict(seed=8001, arm='legacy_role_after_init_point_only', domain='fixture',
        task=dict(d=1000, alpha=.05), geometry=dict(task_geometry_shift=[.1, -.2]),
        protocol=wiring.OLD_PROTOCOL, search_evaluation_count=13,
        candidate_budgets=[80, 128, 128], objective_incumbent_position=2,
        objective_comparison_budget=8, objective_comparison_delta=.05 / 3,
        familywise_delta=.05, verification_seed=9001, tau=0.,
        method='normal_quantile_tolerance', mean_delta_fraction=.5,
        shortlist_mode='posterior_objective_challenger_then_safe',
        frozen_shortlist=[dict(point=[point], shortlist_position=i + 1)
                          for i, point in enumerate((11, 1, 2))])
    order = dict(status='frozen', seed=8001, domain='fixture', stream='observation_1',
        paid_prefix=10, input_file='stop/input.json', order=[dict(point=[2 if same_third else 0],
            source_initial_rank=1, source_pool_index=0, first_initial_index=0,
            observation_indices=[0], probability_violation=.01, objective=.2)])
    return wiring.make_variants(item, order)


def substitute_verifiers(monkeypatch, safe, *, flip_128=False):
    """Keep real terminal branching; substitute only repeated scalar evaluations."""
    calls = []

    def safety(problem, point, **kw):
        index, count = kw['candidate_index'], kw['verification_budget']
        calls.append(('safety', index, count))
        for replication in range(count):
            problem.simulate(point, ('safety', kw['seed'], index, replication, 0))
        certified = safe[index]
        if flip_128 and index == 1:
            certified = count == 128
        return dict(candidate_index=index, certified=certified, verification_budget=count)

    def compare(problem, challenger, incumbent, **kw):
        block = kw.get('comparison_block_index', 0)
        calls.append(('objective', block, kw['delta']))
        for replication in range(kw['comparison_budget']):
            for side, point in enumerate((challenger, incumbent)):
                problem.simulate(point, ('objective', kw['seed'], block, replication, side))
        return dict(challenger_dominates=True, simulation_calls=2 * kw['comparison_budget'])

    for module in (original, tournament):
        monkeypatch.setattr(module, 'verify_frozen_policy', safety)
        monkeypatch.setattr(module, 'verify_paired_objective_dominance', compare)
    monkeypatch.setattr(np.random, 'default_rng', lambda *a, **k: pytest.fail('Real RNG was constructed'))
    return calls


class Substitute:
    def __init__(self, fail=None):
        self.calls, self.fail = [], fail
        self.failed = False

    def simulate(self, point, rng=None):
        self.calls.append((list(point), rng))
        if self.fail is not None and self.fail(point, rng) and not self.failed:
            self.failed = True
            raise RuntimeError('substitute half-pair failure')
        return [float(point[0]), -1.]


@pytest.mark.parametrize('safe', list(product((False, True), repeat=3)))
def test_all_four_cells_keep_real_branches_caps_and_recompute_certificates(tmp_path, monkeypatch, safe):
    events = substitute_verifiers(monkeypatch, safe)
    problem = Substitute()
    variants = inputs()
    result = execution.execute_four_variants(tmp_path, variants, stream='observation_1',
                                            problem_factory=lambda item: problem)
    assert result['status'] == 'complete'
    assert list(result['variants']) == list(wiring.VARIANTS)
    assert result['physical_verification_calls'] == len(problem.calls)
    assert result['failed_simulation_calls'] == 0
    for name, row in result['variants'].items():
        assert row['logical_verification_calls'] <= 352
        assert row['logical_verification_calls'] == row['new_verification_calls'] + row['reused_raw_calls']
        assert row['verification']['certified'] == any(safe)
        assert row['verification']['total_evaluation_count'] <= 365
        assert row['truth_used'] is False
        if name.endswith('two_comparison'):
            assert [a['candidate_index'] for a in row['verification']['attempts']] == [0, 1, 2]
            objective = row['verification']['objective_comparisons']
            assert any(a['block_index'] == 1 for a in objective) == (safe[2] and any(safe[:2]))
    assert sum(event[0] == 'safety' for event in events) == (10 if any(safe[:2]) else 12)
    # First two exact old safety blocks are reused but re-certified in the other old cell.
    ref_old = result['variants']['reference_old']
    assert ref_old['reused_previous_variant_raw_calls'] >= 208
    before = len(problem.calls)
    assert execution.execute_four_variants(tmp_path, variants, stream='observation_1') == result
    assert len(problem.calls) == before


def test_prefix120_consumes_no_old_tail_and_changes_certificate_without_inheriting_it(tmp_path, monkeypatch):
    substitute_verifiers(monkeypatch, (False, True, True), flip_128=True)
    problem = Substitute()
    result = execution.execute_four_variants(tmp_path, inputs(), stream='observation_1',
                                            problem_factory=lambda item: problem)
    old = result['variants']['original_old']
    new = result['variants']['original_two_comparison']
    assert old['verification']['attempts'][1]['certified']
    assert not new['verification']['attempts'][1]['certified']
    incumbent = next(block for block in new['blocks_used'] if block['kind'] == 'safety' and block['index'] == 1)
    assert incumbent['stored_rows_before'] == 128
    assert incumbent['count'] == incumbent['reused_raw_calls'] == 120
    assert incumbent['new_verification_calls'] == 0
    assert old['verification']['selected_shortlist_rank'] == 2
    assert new['verification']['selected_shortlist_rank'] == 3


def test_same_list_is_raw_identity_but_four_distinct_inferences(tmp_path, monkeypatch):
    events = substitute_verifiers(monkeypatch, (True, True, True))
    problem = Substitute()
    result = execution.execute_four_variants(tmp_path, inputs(same_third=True), stream='observation_1',
                                            problem_factory=lambda item: problem)
    assert result['variants']['reference_old']['new_verification_calls'] == 0
    assert result['variants']['reference_two_comparison']['new_verification_calls'] == 0
    assert len(problem.calls) == 360  # Old224 plus third120 and independent comparison1 sixteen.
    assert len(events) == 16
    objectives = [rng for point, rng in problem.calls if rng[0] == 'objective']
    assert {rng[2] for rng in objectives} == {0, 1}
    assert len(objectives) == 32
    old = result['variants']['original_old']['verification']
    new = result['variants']['original_two_comparison']['verification']
    assert old['selected_shortlist_rank'] == 1 and new['selected_shortlist_rank'] == 3


def test_half_pair_failure_is_retained_and_resume_buys_only_missing_side(tmp_path, monkeypatch):
    substitute_verifiers(monkeypatch, (True, True, True))
    # Block1 of the original third is unique to its cell; another cell cannot fill it.
    failed = {'armed': False}

    def fail(point, rng):
        if rng[:3] == ('objective', 9001, 1) and rng[3:] == (0, 0) and point[0] == 2:
            failed['armed'] = True
            return False
        return failed['armed'] and rng[:3] == ('objective', 9001, 1) and rng[3:] == (0, 1)

    problem = Substitute(fail)
    variants = inputs()
    first = execution.execute_four_variants(tmp_path, variants, stream='observation_1',
                                            problem_factory=lambda item: problem)
    assert first['status'] == 'complete_with_failures'
    broken = first['variants']['original_two_comparison']
    assert broken['status'] == 'failed' and broken['failed_simulation_calls'] == 1
    assert broken['blocks_used'][-1]['count'] == 1
    before = len(problem.calls)
    assert execution.execute_four_variants(tmp_path, variants, stream='observation_1')['status'] == 'complete_with_failures'
    assert len(problem.calls) == before
    resumed = execution.execute_four_variants(tmp_path, variants, stream='observation_1',
        problem_factory=lambda item: problem, resume=True)
    repaired = resumed['variants']['original_two_comparison']
    assert resumed['status'] == 'complete'
    assert len(problem.calls) - before == 15
    assert repaired['new_verification_calls'] == 15
    assert repaired['reused_same_variant_raw_calls'] == 121  # Third120 plus saved half-pair.
    assert repaired['cumulative_failed_simulation_calls'] == 1
    assert resumed['physical_simulation_invocations'] == resumed['physical_verification_calls'] + 1
    assert (tmp_path / 'original_two_comparison/failure.json').exists()
    attempts = [json.loads(line) for line in (tmp_path / 'original_two_comparison/attempts.jsonl').read_text().splitlines()]
    assert [row['status'] for row in attempts] == ['failed', 'completed']
    assert sum(row['failed_simulation_calls'] for row in attempts) == 1
    assert sum(row['new_verification_calls'] for row in attempts) == repaired['cumulative_new_verification_calls']


@pytest.mark.parametrize('change', ['arm', 'stream', 'verification_seed', 'geometry', 'slot'])
def test_saved_raw_cannot_cross_arm_stream_seed_geometry_or_slot(tmp_path, monkeypatch, change):
    substitute_verifiers(monkeypatch, (True, True, True))
    variants = inputs()
    execution.execute_four_variants(tmp_path, variants, stream='observation_1',
                                    problem_factory=lambda item: Substitute())
    changed, stream = deepcopy(variants), 'observation_1'
    if change == 'stream':
        stream = 'observation_2'
    else:
        for row in changed.values():
            if change == 'arm':
                row['arm'] = 'gp'
            elif change == 'verification_seed':
                row['verification_seed'] += 1
            elif change == 'geometry':
                row['geometry']['task_geometry_shift'][0] += .01
            else:
                row['frozen_shortlist'][0], row['frozen_shortlist'][1] = row['frozen_shortlist'][1], row['frozen_shortlist'][0]
    with pytest.raises(ValueError, match='changed'):
        execution.execute_four_variants(tmp_path, changed, stream=stream)


def test_no_factory_cannot_start_simulation_and_keeps_all_failures(tmp_path, monkeypatch):
    substitute_verifiers(monkeypatch, (True, True, True))
    result = execution.execute_four_variants(tmp_path, inputs(), stream='observation_1')
    assert result['status'] == 'complete_with_failures'
    assert all('explicitly admitted' in row['error'] for row in result['variants'].values())
    assert result['physical_verification_calls'] == result['failed_simulation_calls'] == 0


def test_paid_prefix_without_completion_requires_explicit_resume(tmp_path, monkeypatch):
    substitute_verifiers(monkeypatch, (True, True, True))
    variants, problem = inputs(), Substitute()
    first = execution.execute_four_variants(tmp_path, variants, stream='observation_1',
                                            problem_factory=lambda item: problem)
    # A worker interruption can leave all paid raws without its final result file.
    (tmp_path / 'original_old/verification.json').unlink()
    with pytest.raises(RuntimeError, match='explicit resume'):
        execution.execute_four_variants(tmp_path, variants, stream='observation_1')
    recovered = execution.execute_four_variants(tmp_path, variants, stream='observation_1', resume=True)
    assert recovered['status'] == 'complete'
    assert recovered['physical_verification_calls'] == first['physical_verification_calls']
    assert recovered['variants']['original_old']['new_verification_calls'] == 0
    assert recovered['variants']['original_old']['reused_same_variant_raw_calls'] == 224

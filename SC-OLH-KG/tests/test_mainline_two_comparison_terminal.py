"""Registered tournament branches and stream domains, using deterministic substitutes."""
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from core import terminal_two_comparison as tournament
from core import terminal_verification as old


POINTS = ((0,), (1,), (2,))
SHORTLIST = [dict(point=list(point), shortlist_position=i+1, posterior_rank=i+1,
                  shortlist_role=role) for i, (point, role) in enumerate(zip(
                      POINTS, ('primary', 'initial_incumbent', 'third')))]


@pytest.fixture(autouse=True)
def forbid_real_rng(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('This preparation test cannot construct a real random stream')
    monkeypatch.setattr(np.random, 'default_rng', forbidden)
    monkeypatch.setattr(np.random, 'SeedSequence', forbidden)


class DeterministicProblem:
    alpha, tau, simulation_noise_model = .05, 0., 'iid_gaussian'

    def __init__(self):
        self.calls = []

    def simulate(self, point, rng=None):
        self.calls.append((tuple(point), rng))
        return np.array([float(point[0]), -10.])


# Each row declares an externally expected winner and actual comparisons.
# Every safety mask appears; both comparisons' success/failure combinations appear.
BRANCHES = [
    ((0, 0, 0), (0, 0), None, []),
    ((0, 0, 1), (0, 0), 2, []),
    ((0, 1, 0), (0, 0), 1, []),
    ((0, 1, 1), (0, 0), 1, [(1, 2, 1, 1)]),
    ((0, 1, 1), (0, 1), 2, [(1, 2, 1, 2)]),
    ((1, 0, 0), (0, 0), 0, []),
    ((1, 0, 1), (0, 0), 0, [(1, 2, 0, 0)]),
    ((1, 0, 1), (0, 1), 2, [(1, 2, 0, 2)]),
    ((1, 1, 0), (0, 0), 1, [(0, 0, 1, 1)]),
    ((1, 1, 0), (1, 0), 0, [(0, 0, 1, 0)]),
    ((1, 1, 1), (0, 0), 1, [(0, 0, 1, 1), (1, 2, 1, 1)]),
    ((1, 1, 1), (0, 1), 2, [(0, 0, 1, 1), (1, 2, 1, 2)]),
    ((1, 1, 1), (1, 0), 0, [(0, 0, 1, 0), (1, 2, 0, 0)]),
    ((1, 1, 1), (1, 1), 2, [(0, 0, 1, 0), (1, 2, 0, 2)]),
]


@pytest.mark.parametrize('search_count,total_cap', [(13, 365), (10, 362)])
@pytest.mark.parametrize('safe,wins,winner,comparisons', BRANCHES)
def test_all_safety_masks_keep_only_certified_winners_and_pay_exact_calls(
        monkeypatch, safe, wins, winner, comparisons, search_count, total_cap):
    problem, events = DeterministicProblem(), []
    original = deepcopy(SHORTLIST)

    def safety(task, point, **kwargs):
        index = POINTS.index(tuple(point))
        assert kwargs['candidate_index'] == index
        assert kwargs['verification_budget'] == (80, 120, 120)[index]
        assert kwargs['delta'] == pytest.approx(.05/3)
        assert kwargs['seed'] == 91 and kwargs['search_evaluation_count'] == search_count
        events.append(('safety', index))
        for _ in range(kwargs['verification_budget']):
            task.simulate(point)
        return dict(candidate_index=index, certified=bool(safe[index]),
                    verification_budget=kwargs['verification_budget'], delta=kwargs['delta'])

    def paired(task, challenger, incumbent, **kwargs):
        block = kwargs['comparison_block_index']
        indices = POINTS.index(tuple(challenger)), POINTS.index(tuple(incumbent))
        expected = next(row for row in comparisons if row[0] == block)
        assert indices == expected[1:3]
        assert kwargs['comparison_budget'] == 8 and kwargs['delta'] == pytest.approx(.05/6)
        assert kwargs['seed'] == 91
        events.append(('comparison', block))
        for _ in range(8):
            task.simulate(challenger)
            task.simulate(incumbent)
        return dict(challenger_dominates=bool(wins[block]), simulation_calls=16,
                    comparison_budget_per_policy=8, delta=kwargs['delta'])

    monkeypatch.setattr(tournament, 'verify_frozen_policy', safety)
    monkeypatch.setattr(tournament, 'verify_paired_objective_dominance', paired)
    deployed, result = tournament.verify_frozen_three_candidate_tournament(
        problem, SHORTLIST, seed=91, search_evaluation_count=search_count)

    assert SHORTLIST == original
    assert deployed == POINTS[0 if winner is None else winner]
    assert result['certified'] is (winner is not None)
    assert result['selected_shortlist_rank'] == (None if winner is None else winner+1)
    assert [row['candidate_index'] for row in result['attempts']] == [0, 1, 2]
    assert [(r['block_index'], r['challenger_index'], r['incumbent_index'], r['winner_index'])
            for r in result['objective_comparisons']] == comparisons
    expected_events = [('safety', 0), ('safety', 1)]
    if any(row[0] == 0 for row in comparisons):
        expected_events.append(('comparison', 0))
    expected_events.append(('safety', 2))
    if any(row[0] == 1 for row in comparisons):
        expected_events.append(('comparison', 1))
    assert events == expected_events
    assert result['safety_verification_budget'] == 320
    assert result['objective_comparison_budget'] == 16*len(comparisons)
    assert result['verification_budget'] == len(problem.calls) == 320+16*len(comparisons)
    assert result['max_verification_budget'] == 352
    assert result['total_evaluation_count'] == search_count+len(problem.calls) <= total_cap
    assert result['candidate_verification_budgets'] == [80, 120, 120]
    assert result['objective_familywise_delta'] == pytest.approx(.05/3)
    assert result['objective_delta_per_comparison'] == pytest.approx(.05/6)
    assert result['search_samples_reused'] is False
    assert result['posterior_updated_from_verification'] is False


@dataclass(frozen=True)
class StreamToken:
    components: tuple


@pytest.fixture
def token_streams(monkeypatch):
    seeds = []

    def sequence(components):
        value = tuple(int(component) for component in components)
        seeds.append(value)
        return value

    monkeypatch.setattr(np.random, 'SeedSequence', sequence)
    monkeypatch.setattr(np.random, 'default_rng', lambda sequence: StreamToken(sequence))
    return seeds


class TokenProblem(DeterministicProblem):
    def simulate(self, point, rng=None):
        assert isinstance(rng, StreamToken)
        self.calls.append((tuple(point), rng))
        # Same-token offsets cancel within a pair; no RNG method exists or is used.
        replication = rng.components[1]
        return np.array([(2., 5., 1.)[point[0]]+replication, -10.])


def test_old_paired_default_fields_and_crn_loop_are_preserved(token_streams):
    problem = TokenProblem()
    default = old.verify_paired_objective_dominance(problem, POINTS[0], POINTS[1],
        seed=123, comparison_budget=8, delta=.05/6)
    default_calls = list(problem.calls)
    problem.calls.clear()
    explicit = old.verify_paired_objective_dominance(problem, POINTS[0], POINTS[1],
        seed=123, comparison_budget=8, delta=.05/6, comparison_block_index=0)
    assert default == explicit
    assert default['sample_mean_difference'] == -3.
    assert default['sample_std_difference'] == default['standard_error'] == 0.
    assert default['one_sided_upper_confidence_bound'] == -3.
    assert default['degrees_of_freedom'] == 7 and default['simulation_calls'] == 16
    assert default['challenger_dominates'] is True
    assert default['paired_common_random_numbers'] is True
    assert default['comparison_stream_independent_from_search_and_safety'] is True
    assert default['target_oracle_used'] is False and default['comparison_samples_logged'] is False
    assert 'comparison_block_index' not in default  # Default helper schema remains unchanged.
    for rep in range(8):
        challenger, incumbent = default_calls[2*rep:2*rep+2]
        expected = (123, rep, old.TERMINAL_OBJECTIVE_COMPARISON_STREAM_TAG)
        assert challenger[0] == POINTS[0] and incumbent[0] == POINTS[1]
        assert challenger[1].components == incumbent[1].components == expected
        assert problem.calls[2*rep][1].components == expected


def test_actual_safety_and_both_paired_loops_use_disjoint_token_domains(token_streams):
    from algorithms.single_olhkg import SingleOLHKGAlgorithm, SIMULATION_STREAM_TAG

    problem = TokenProblem()
    surrogate = SimpleNamespace(config=SimpleNamespace(seed=123, observation_seed=None),
                                history=[], observations={}, problem=problem)
    for _ in range(13):
        SingleOLHKGAlgorithm._simulate_and_store(surrogate, POINTS[0])
    search_tokens = {rng.components for _, rng in problem.calls}
    assert search_tokens == {(123, i, SIMULATION_STREAM_TAG) for i in range(13)}
    problem.calls.clear()
    deployed, result = tournament.verify_frozen_three_candidate_tournament(
        problem, SHORTLIST, seed=123, search_evaluation_count=13)

    assert deployed == POINTS[2] and result['certified'] is True
    assert len(problem.calls) == result['verification_budget'] == 352
    assert result['total_evaluation_count'] == 365
    assert [row['sample_count'] for row in result['attempts']] == [80, 120, 120]
    assert all(row['certified'] and row['upper_margin'] < 0 for row in result['attempts'])
    first = {(123, rep, old.TERMINAL_OBJECTIVE_COMPARISON_STREAM_TAG) for rep in range(8)}
    second = {token+(1,) for token in first}
    tokens = [rng.components for _, rng in problem.calls]
    safety = set(tokens)-first-second
    assert len(safety) == 320 and len(first) == len(second) == 8
    assert not (first & second or first & safety or second & safety)
    assert not ((first | second | safety) & search_tokens)
    for domain in (first, second):
        for token in domain:
            assert tokens.count(token) == 2
            points = [point for point, rng in problem.calls if rng.components == token]
            assert points == ([POINTS[0], POINTS[1]] if domain is first else [POINTS[2], POINTS[0]])
    assert {(row['block_index'], row['incumbent_index']) for row in result['objective_comparisons']} == {(0, 1), (1, 0)}

"""Exact paid-block reuse and continuation with deterministic synthetic observations."""
from dataclasses import dataclass
import json

import numpy as np
import pytest

from core import terminal_verification as old
from performance import prepare_mainline_two_comparison_terminal as candidate
from performance import register_mainline_two_comparison_development as registration
from performance import run_mainline_two_comparison_development as runner
from performance.run_mainline_legacy_backend_bridge import read, save


@dataclass(frozen=True)
class StreamToken:
    components: tuple


@pytest.fixture(autouse=True)
def token_streams(monkeypatch):
    """No generator exists: helpers receive only immutable seed-component tokens."""
    monkeypatch.setattr(np.random, 'SeedSequence', lambda values: tuple(map(int, values)))
    monkeypatch.setattr(np.random, 'default_rng', lambda values: StreamToken(values))


class SyntheticProblem:
    d, alpha, tau, simulation_noise_model = 1000, .05, 0., 'iid_gaussian'

    def __init__(self):
        self.rows, self.tokens = [], []
        self.fail_before = None

    def simulate(self, point, rng=None):
        assert isinstance(rng, StreamToken)
        if self.fail_before is not None and len(self.rows) == self.fail_before:
            raise RuntimeError('synthetic observation unavailable')
        components = rng.components
        index = int(point[0])
        objective, risk = (2., 5., 1.)[index], -10.
        if components[2] == old.TERMINAL_OBJECTIVE_COMPARISON_STREAM_TAG:
            if len(components) == 3 and index == 0:
                # Old delta passes; the tighter frozen delta retains incumbent1.
                objective = 5. - 1.1 + (1. if components[1] % 2 == 0 else -1.)
        else:
            replication = components[1] if len(components) == 3 else components[2]
            risk += .001 * replication
            if replication >= 120:
                risk += 1.  # Makes an accidental 128-row statistic detectable.
        value = [objective, risk]
        self.rows.append(dict(point=list(point), raw=value))
        self.tokens.append(components)
        return np.asarray(value)


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(registration, 'ROOT', tmp_path)
    points = [[i] + [0] * 999 for i in range(3)]
    item = dict(domain='InventorySupplyChain', seed=3016, arm='legacy_role_full',
        verification_seed=3216, task=dict(d=1000, L=3, sigma=.04, alpha=.05, weights=[1.]),
        geometry={}, tau=0., method='normal_quantile_tolerance', mean_delta_fraction=.5,
        shortlist_mode='posterior_objective_challenger_then_safe', search_evaluation_count=13,
        protocol='same_policy_distinct_challenger', candidate_budgets=[80,128,128],
        objective_incumbent_position=2, objective_comparison_budget=8,
        objective_comparison_delta=.05/3, familywise_delta=.05,
        primary=points[0], frozen_shortlist=[dict(point=point, posterior_rank=i+1,
            shortlist_position=i+1, shortlist_role=('primary','initial_incumbent','third')[i])
            for i, point in enumerate(points)])
    source = SyntheticProblem()
    deployed, result = old.verify_frozen_shortlist(source, item['frozen_shortlist'], seed=3216,
        **{key:item[key] for key in ('search_evaluation_count','candidate_budgets',
            'familywise_delta','method','mean_delta_fraction','shortlist_mode',
            'objective_incumbent_position','objective_comparison_budget','objective_comparison_delta')})
    assert len(source.rows) == result['verification_budget'] == 224
    assert result['objective_comparison']['challenger_dominates']
    save(tmp_path/'old/input.json', item)
    save(tmp_path/'old/verification.json', dict(verification_seed=3216, verification=result,
        deployed=list(deployed), new_verification_calls=224))
    save(tmp_path/'old/paid_verification_raw.json', source.rows)
    save(tmp_path/'candidate/input.json', candidate.candidate_input(item))
    row = registration.block_inventory(dict(seed=item['seed'], arm=item['arm'], domain=item['domain'],
        input_file='candidate/input.json', original_input_file='old/input.json',
        original_verification_file='old/verification.json', quality_baseline_defined=True))
    plan = dict(output_dir='new', candidate_protocol=candidate.PROTOCOL,
        old_protocol='same_policy_distinct_challenger')
    actual = SyntheticProblem()
    monkeypatch.setattr(runner, 'build_problem', lambda given: actual)
    directory = tmp_path/'new/cases/seed_3016/legacy_role_full'
    return plan, row, directory, actual, source, tmp_path


def attempts(directory):
    return [json.loads(line) for line in (directory/'attempts.jsonl').read_text().splitlines()]


def test_exact_prefix_recomputes_stricter_delta_and_separates_costs(case):
    plan, row, directory, actual, source, root = case
    original_bytes = (root/'old/paid_verification_raw.json').read_bytes()
    result = runner.verify_case(plan, row)
    assert result == read(directory/'verification.json')
    verified = result['verification']
    assert verified['attempts'][1]['sample_count'] == 120
    assert verified['attempts'][1]['sample_mean'] == pytest.approx(
        np.mean([r['raw'][1] for r in source.rows[80:200]]))
    first, second = verified['objective_comparisons']
    assert first['delta'] == .05/6 and not first['challenger_dominates']
    assert (first['block_index'], first['incumbent_index'], first['winner_index']) == (0,1,1)
    assert (second['block_index'], second['incumbent_index'], second['winner_index']) == (1,1,2)
    assert result['deployed'][0] == 2
    assert result['logical_verification_calls'] == 352
    assert result['reused_old_raw_calls'] == 216
    assert result['reused_new_raw_calls'] == 0
    assert result['cumulative_new_verification_calls'] == len(actual.rows) == 136
    assert result['old_verification_calls'] == 224
    assert result['historical_verification_calls'] == 360
    assert verified['total_evaluation_count'] == 365
    assert (root/'old/paid_verification_raw.json').read_bytes() == original_bytes
    assert len(read(root/'old/paid_verification_raw.json')) == 224
    assert not (set(actual.tokens) & set(source.tokens))
    assert actual.tokens[:120] == [(3216,2,r,old.TERMINAL_SHORTLIST_VERIFICATION_STREAM_TAG) for r in range(120)]
    for rep in range(8):
        assert actual.tokens[120+2*rep:122+2*rep] == [
            (3216,rep,old.TERMINAL_OBJECTIVE_COMPARISON_STREAM_TAG,1)] * 2
    assert sum(r['new_verification_calls'] for r in attempts(directory)) == 136
    # A completed read is idempotent, including the historical cost account.
    assert runner.verify_case(plan, row) == result
    assert len(actual.rows) == 136
    assert sum(r['new_verification_calls'] for r in attempts(directory)) == 136


def test_half_pair_failure_requires_explicit_resume_and_buys_only_suffix(case):
    plan, row, directory, actual, source, root = case
    actual.fail_before = 121  # safety2 complete; only block1 challenger0 returned.
    with pytest.raises(RuntimeError, match='synthetic observation unavailable'):
        runner.verify_case(plan, row)
    assert len(actual.rows) == 121
    assert not (directory/'verification.json').exists()
    failure = read(directory/'failure.json')
    assert failure['failed_simulation_calls'] == 1
    assert failure['cumulative_new_simulation_invocations'] == 122
    block_path = directory/'new_blocks/objective_1.json'
    partial = read(block_path)
    assert partial['status'] == 'partial' and len(partial['raw']) == 1
    assert partial['raw'][0] == actual.rows[-1]['raw']
    assert partial['ordered_points'][0][0] == 2 and partial['ordered_points'][1][0] == 1
    assert partial['expected_count'] == 16
    with pytest.raises(RuntimeError):
        runner.verify_case(plan, row)
    assert read(directory/'failure.json') == failure and read(block_path) == partial
    assert len(actual.rows) == 121
    assert sum(r['new_verification_calls'] for r in attempts(directory)) == 121
    actual.fail_before = None
    result = runner.verify_case(plan, row, resume=True)
    assert len(actual.rows) == result['cumulative_new_verification_calls'] == 136
    assert actual.rows[121]['point'][0] == 1
    assert actual.tokens[120] == actual.tokens[121] == (
        3216,0,old.TERMINAL_OBJECTIVE_COMPARISON_STREAM_TAG,1)
    assert [r['new_verification_calls'] for r in attempts(directory) if r['new_verification_calls']] == [121,15]
    assert result['reused_old_raw_calls'] == 216 and result['reused_new_raw_calls'] == 121
    assert result['logical_verification_calls'] == 352 and result['historical_verification_calls'] == 360
    assert result['failed_simulation_calls'] == 1
    assert result['cumulative_new_simulation_invocations'] == 137
    assert result['historical_simulation_invocations'] == 361
    assert sum(r['new_simulation_invocations'] for r in attempts(directory)) == 137
    assert read(block_path)['status'] == 'complete' and len(read(block_path)['raw']) == 16
    assert runner.verify_case(plan, row) == result and len(actual.rows) == 136


@pytest.mark.parametrize('kind,index,paid', [('safety',2,120),('objective',1,136)])
def test_statistic_failure_preserves_completed_raw_block_metadata(case, monkeypatch, kind, index, paid):
    plan, row, directory, actual, source, root = case
    name = 'verify_frozen_policy' if kind == 'safety' else 'verify_paired_objective_dominance'
    original = getattr(runner.tournament, name)

    def fail_after_raw(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs['candidate_index' if kind == 'safety' else 'comparison_block_index'] == index:
            raise RuntimeError('synthetic statistic failure after returned observations')
        return result

    monkeypatch.setattr(runner.tournament, name, fail_after_raw)
    with pytest.raises(RuntimeError, match='synthetic statistic failure'):
        runner.verify_case(plan, row)
    assert len(actual.rows) == paid and not (directory/'verification.json').exists()
    block = read(directory/f'new_blocks/{kind}_{index}.json')
    assert len(block['raw']) == block['expected_count'] == (120 if kind == 'safety' else 16)
    assert block['status'] == 'complete'  # Raw completion does not certify failed statistics.
    failure = read(directory/'failure.json')
    assert 'synthetic statistic failure' in str(failure)
    affected = failure['blocks_used'][-1]
    assert (affected['kind'], affected['index'], affected['status']) == (kind,index,'partial')
    assert affected['count'] == block['expected_count']
    assert sum(r['new_verification_calls'] for r in attempts(directory)) == paid
    with pytest.raises(RuntimeError):
        runner.verify_case(plan, row)
    assert len(actual.rows) == paid
    monkeypatch.setattr(runner.tournament, name, original)
    result = runner.verify_case(plan, row, resume=True)
    assert result['cumulative_new_verification_calls'] == len(actual.rows) == 136
    assert attempts(directory)[-1]['new_verification_calls'] == 136-paid

"""Deterministic substitutes exercise the real terminal recording/failure path."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core import terminal_two_comparison as tournament
from performance import prepare_mainline_two_comparison_terminal as candidate
from performance import run_mainline_terminal_equal as recorder
from performance.run_mainline_legacy_backend_bridge import ROOT, read, save


def wired_case(tmp_path, monkeypatch, arm='legacy_role_full', fail_after=None):
    original = read(ROOT / ('paper_artifacts/mainline_unseen_geometry_confirmation_v1/'
        f'replicate_1/InventorySupplyChain/terminal/seed_2026103001/{arm}/'
        'same_policy_distinct_challenger/input.json'))
    item = candidate.candidate_input(original)
    plan = dict(target_seeds=[item['seed']], arms=[arm], protocols={candidate.PROTOCOL:[80,120,120]})
    directory = recorder.case_dir(tmp_path, item['seed'], arm, candidate.PROTOCOL)
    save(directory / 'input.json', item)
    save(tmp_path / 'replay_check.json', dict(status='passed', evidence='deterministic substitutes only'))
    returned = []

    def simulate(point, rng=None):
        returned.append(tuple(point))
        return np.asarray([float(len(returned)), -.5])

    problem = SimpleNamespace(d=1000, tau=item['tau'], alpha=item['task']['alpha'],
        base=SimpleNamespace(**item['geometry']), simulate=simulate)
    monkeypatch.setattr(recorder.benchmark, 'build_scalarized_problem', lambda *a, **kw: problem)

    def no_rng(*args, **kwargs):
        raise AssertionError('Recorded-flow substitute tests must not draw random numbers')
    monkeypatch.setattr(np.random, 'default_rng', no_rng)
    monkeypatch.setattr(np.random, 'SeedSequence', no_rng)

    def observe(point):
        problem.simulate(point)
        if fail_after is not None and len(returned) == fail_after:
            raise RuntimeError('injected failure after completed synthetic observation')

    def safety(task, point, **kw):
        assert task is problem and kw['seed'] == item['verification_seed']
        assert kw['delta'] == .05/3
        for _ in range(kw['verification_budget']):
            observe(point)
        return dict(certified=True, candidate_index=kw['candidate_index'],
                    verification_budget=kw['verification_budget'])

    def paired(task, challenger, incumbent, **kw):
        assert task is problem and kw['seed'] == item['verification_seed']
        assert kw['delta'] == .05/6 and kw['comparison_budget'] == 8
        for _ in range(8):
            observe(challenger)
            observe(incumbent)
        return dict(challenger_dominates=True, simulation_calls=16)

    monkeypatch.setattr(tournament, 'verify_frozen_policy', safety)
    monkeypatch.setattr(tournament, 'verify_paired_objective_dominance', paired)
    return item, plan, directory, returned


@pytest.mark.parametrize('arm,total', [('legacy_role_full',365), ('initialization_only',362)])
def test_recorded_candidate_cap_and_completed_reuse(tmp_path, monkeypatch, arm, total):
    item, plan, directory, returned = wired_case(tmp_path, monkeypatch, arm)
    recorder.verify(tmp_path, plan, item['seed'], arm, candidate.PROTOCOL, runner=candidate.run_verifier)
    result = read(directory / 'verification.json')
    assert result['deployed'] == item['frozen_shortlist'][2]['point']
    assert result['verification']['total_evaluation_count'] == total
    assert result['new_verification_calls'] == len(returned) == 352
    assert len(read(directory / 'paid_verification_raw.json')) == 352
    assert [r['block_index'] for r in result['verification']['objective_comparisons']] == [0,1]
    recorder.verify(tmp_path, plan, item['seed'], arm, candidate.PROTOCOL, runner=candidate.run_verifier)
    attempts = [json.loads(line) for line in (directory/'attempts.jsonl').read_text().splitlines()]
    assert attempts[-1]['reused'] and attempts[-1]['new_verification_calls'] == 0
    assert sum(r['new_verification_calls'] for r in attempts) == 352 and len(returned) == 352
    print(json.dumps(dict(synthetic=True, arm=arm, completed_observations=352,
                         logical_total=total, completed_reuse_no_extra_calls=True)))


@pytest.mark.parametrize('fail_after', [81,337])
def test_partial_safety_and_half_pair_failure_are_not_repeated(tmp_path, monkeypatch, fail_after):
    item, plan, directory, returned = wired_case(tmp_path, monkeypatch, fail_after=fail_after)
    with pytest.raises(RuntimeError, match='injected failure'):
        recorder.verify(tmp_path, plan, item['seed'], item['arm'], candidate.PROTOCOL,
                        runner=candidate.run_verifier)
    assert not (directory/'verification.json').exists()
    failure_files = list(directory.glob('failed_verification_raw_*.json'))
    assert len(failure_files) == 1
    rows = read(failure_files[0])
    assert len(rows) == len(returned) == fail_after
    if fail_after == 337:
        assert rows[-1]['point'] == item['frozen_shortlist'][2]['point']
    with pytest.raises(RuntimeError, match='Recover saved paid verification'):
        recorder.verify(tmp_path, plan, item['seed'], item['arm'], candidate.PROTOCOL,
                        runner=candidate.run_verifier)
    attempts = [json.loads(line) for line in (directory/'attempts.jsonl').read_text().splitlines()]
    assert [r['status'] for r in attempts] == ['failed','failed']
    assert [r['new_verification_calls'] for r in attempts] == [fail_after,0]
    assert len(returned) == fail_after and read(failure_files[0]) == rows
    print(json.dumps(dict(synthetic=True, completed_observations=fail_after,
                         raw_retained=True, retry_blocked=True, retry_extra_calls=0)))


def test_default_recorder_still_routes_to_original_terminal(tmp_path, monkeypatch):
    item, plan, directory, returned = wired_case(tmp_path, monkeypatch)
    called = []

    def old_runner(problem, given):
        called.append(copy.deepcopy(given))
        return given['frozen_shortlist'][0]['point'], dict(verification_budget=0,max_verification_budget=352)

    monkeypatch.setattr(recorder, 'run_verifier', old_runner)
    recorder.verify(tmp_path, plan, item['seed'], item['arm'], candidate.PROTOCOL)
    assert called == [item] and not returned

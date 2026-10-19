"""Exercise modified-only preparation and seed routing without real task calls."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance import mainline_modified_terminal as modified
from performance import run_mainline_terminal_equal as verifier
from performance import run_mainline_terminal_confirmation as confirmation
from performance.run_mainline_legacy_backend_bridge import read, save


@pytest.fixture
def saved_inputs(tmp_path, monkeypatch):
    source = read(ROOT / 'performance/manifests/mainline_same_policy_confirmation_inventory_search_v1.json')
    terminal = read(ROOT / 'performance/manifests/mainline_same_policy_confirmation_inventory_cells_v1.json')
    original_source = ROOT / source['output_dir']
    source.update(output_dir='search', target_seeds=[2026102111, 2026102112])
    terminal.update(output_dir='terminal', target_seeds=source['target_seeds'],
        protocols={modified.PROTOCOL: [80, 128, 128]},
        inherited_replay_check_file='inherited_verifier.json',
        verification_seed_by_task={'2026102111': 2026102241, '2026102112': 2026102242})
    for seed in source['target_seeds']:
        for cell in terminal['arms']:
            relative = Path(f'seed_{seed}') / cell['history_arm']
            save(tmp_path / 'search' / relative / 'search.json', read(original_source / relative / 'search.json'))
            if cell['terminal_backend'] == 'native':
                save(tmp_path / 'search' / relative / 'native_terminal.json',
                     read(original_source / relative / 'native_terminal.json'))
    save(tmp_path / 'inherited_verifier.json', dict(status='passed', new_raw_replay=False))
    monkeypatch.setattr(modified, 'ROOT', tmp_path)

    def no_real_call(*args, **kwargs):
        raise AssertionError('Modified terminal preparation cannot restore a model or build a task')

    monkeypatch.setattr(confirmation, 'load_pickle', no_real_call)
    monkeypatch.setattr(verifier.benchmark, 'build_scalarized_problem', no_real_call)
    return source, terminal


def test_prepare_reorders_saved_same_policy_case_without_original_verification(saved_inputs, tmp_path):
    source, terminal = saved_inputs
    modified.prepare(source, terminal)
    output = tmp_path / terminal['output_dir']
    assert read(output / 'prepared.json')['case_count'] == 18
    assert not list(tmp_path.rglob('verification.json'))
    assert not list(tmp_path.rglob('*raw*.json'))
    inherited = read(output / 'replay_check.json')
    assert inherited['inherited'] == 'inherited_verifier.json'
    assert inherited['new_raw_replay'] is False
    for seed, arm, protocol in verifier.cases(terminal):
        item = read(verifier.case_dir(output, seed, arm, protocol) / 'input.json')
        assert item['protocol'] == modified.PROTOCOL
        assert item['verification_seed'] == terminal['verification_seed_by_task'][str(seed)]
        assert sum(item['candidate_budgets']) + 2 * item['objective_comparison_budget'] == 352
        assert item['tau'] == 0.
        assert item['search_evaluation_count'] == (10 if arm == 'initialization_only' else 13)
    old = read(tmp_path / 'search/seed_2026102112/legacy_role_full/search.json')['terminal']
    item = read(verifier.case_dir(output, 2026102112, 'legacy_role_full', modified.PROTOCOL) / 'input.json')
    assert item['input_changed'] is True
    assert [row['point'] for row in item['frozen_shortlist']] == [
        old['frozen_shortlist'][i]['point'] for i in (2, 0, 1)]
    assert item['objective_incumbent_position'] == 2
    assert item['primary'] == old['frozen_shortlist'][2]['point']
    unchanged = read(verifier.case_dir(output, 2026102111, 'legacy_role_full', modified.PROTOCOL) / 'input.json')
    old_unchanged = read(tmp_path / 'search/seed_2026102111/legacy_role_full/search.json')['terminal']
    assert unchanged['input_changed'] is False
    assert unchanged['frozen_shortlist'] == old_unchanged['frozen_shortlist']
    native = read(verifier.case_dir(output, 2026102112, 'legacy_role_full_native_terminal', modified.PROTOCOL) / 'input.json')
    native_source = read(tmp_path / 'search/seed_2026102112/legacy_role_full/native_terminal.json')['terminal']
    assert {tuple(row['point']) for row in native['frozen_shortlist']} == {
        tuple(row['point']) for row in native_source['frozen_shortlist']}


@pytest.mark.parametrize('arm,logical_cap', [('legacy_role_full', 365), ('initialization_only', 362)])
def test_direct_verifier_receives_new_seed_and_keeps_budget(saved_inputs, tmp_path, monkeypatch, arm, logical_cap):
    source, terminal = saved_inputs
    modified.prepare(source, terminal)
    output = tmp_path / terminal['output_dir']
    seed = 2026102112
    item = read(verifier.case_dir(output, seed, arm, modified.PROTOCOL) / 'input.json')
    captured = {}
    problem = SimpleNamespace(d=1000, tau=0., alpha=source['task']['alpha'],
        base=SimpleNamespace(**item['geometry']), simulate=lambda point, rng=None: np.zeros(2))

    def deterministic_verifier(task, shortlist, **kwargs):
        captured.update(kwargs)
        assert task is problem
        assert shortlist == item['frozen_shortlist']
        for _ in range(352):
            task.simulate(shortlist[0]['point'])
        return shortlist[0]['point'], dict(verification_budget=352, max_verification_budget=352)

    monkeypatch.setattr(verifier.benchmark, 'build_scalarized_problem', lambda *args, **kwargs: problem)
    monkeypatch.setattr(verifier, 'verify_frozen_shortlist', deterministic_verifier)
    verifier.verify(output, terminal, seed, arm, modified.PROTOCOL)
    assert captured['seed'] == 2026102242
    assert captured['seed'] != seed
    assert captured['candidate_budgets'] == [80, 128, 128]
    assert captured['familywise_delta'] == .05
    assert captured['objective_comparison_delta'] == .05 / 3
    assert captured['objective_incumbent_position'] == item['objective_incumbent_position']
    saved = read(verifier.case_dir(output, seed, arm, modified.PROTOCOL) / 'verification.json')
    assert saved['verification_seed'] == 2026102242
    assert saved['new_verification_calls'] + item['search_evaluation_count'] == logical_cap
    assert not (tmp_path / 'search' / f'seed_{seed}' / arm / 'verification.json').exists()


def test_prepare_refuses_missing_frozen_native_terminal(saved_inputs, tmp_path):
    source, terminal = saved_inputs
    (tmp_path / 'search/seed_2026102111/legacy_role_full/native_terminal.json').unlink()
    with pytest.raises(FileNotFoundError):
        modified.prepare(source, terminal)
    assert not (tmp_path / 'terminal/prepared.json').exists()
    assert not list((tmp_path / 'terminal').rglob('input.json'))

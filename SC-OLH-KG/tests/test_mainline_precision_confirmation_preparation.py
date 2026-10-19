import json
from pathlib import Path

import pytest

from performance import run_mainline_precision_confirmation as prep


def store(root, path, value):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value))


def paired(name='observation_1'):
    rows = [dict(stage=stage, cpu_seconds=cpu, status=status)
        for stage, cpu, status in [('search', 8., 'completed'), ('prepare-neighbors', 1., 'completed'),
            ('freeze-modified-only-terminal-inputs', 3., 'completed'),
            ('verify', 2., 'completed'), ('verify-two-comparison-development', 100., 'failed')]]
    return dict(status='complete', registered_geometries=1, new_work=dict(
        new_target_initial_calls=10, new_target_search_calls=18, new_truth_function_calls=1),
        terminal_costs=dict(old_physical_calls=200, additional_new_protocol_physical_calls=50),
        new_physical_target_calls=278, recorded_cpu_seconds=114., attempts=rows,
        failed_attempts=[rows[-1]], execution_failures=[dict(protocol='new', status='failed')])


def test_old_only_cost_keeps_failure_without_charging_other_verifier_cpu():
    result = prep.paired_source_cost(paired(), 'observation_1')
    assert result['old_only_physical_calls'] == 228
    assert result['old_only_search_terminal_cpu_seconds'] == 14
    assert result['source_all_protocols_target_calls'] == 278
    assert result['source_failed_attempts'][0]['cpu_seconds'] == 100
    missing = paired()
    missing['attempts'] = missing['attempts'][:2]
    with pytest.raises(ValueError, match='two-terminal total'):
        prep.paired_source_cost(missing, 'observation_1')


def new_geometry_source(root):
    cells = []
    for arm in range(9):
        file = f'terminal/{arm}/original_old/result.json'
        store(root, str(Path(file).parent / 'attempts.jsonl'), dict(
            stage='verify-reference-coverage', cpu_seconds=1., status='completed'))
        cells.append(dict(arm=str(arm), variant='original_old', result_file=file, status='complete'))
    cells.append(dict(variant='reference_two_comparison', result_file='forbidden/result.json', status='failed'))
    completion = dict(status='complete', registered_geometries=1, costs=dict(
        shared_initial_physical_calls=10, shared_online_physical_calls=18,
        terminal_logical_and_raw_ownership_by_variant=dict(original_old=dict(cumulative_new_verification_calls=200)),
        by_stage={'search':dict(cpu_seconds=8.), 'verify-reference-coverage':dict(cpu_seconds=1000.)},
        successful_target_physical_calls=999, attempt_cpu_seconds=1008., analytic_truth_field_calls=30,
        failed_attempts=[dict(stage='other_terminal', status='failed')],
        call_accounting_complete=True, search_failed_invocation_counter=None))
    return completion, dict(status='sealed', cells=cells)


def test_four_cell_old_cpu_reads_only_actual_old_ledgers(tmp_path):
    completion, seal = new_geometry_source(tmp_path)
    result = prep.four_cell_source_cost(completion, seal, tmp_path)
    assert result['old_terminal_cpu_seconds'] == 9
    assert result['old_only_search_terminal_cpu_seconds'] == 17
    assert result['old_only_physical_calls'] == 228
    assert len(result['source_execution_failures']) == 1
    assert result['source_search_failed_invocation_counter'] is None


def test_cold_truth_projection_uses_full_inventory_not_paid_cache_misses():
    sources = [prep.paired_source_cost(paired(), name) for name in ('reference', 'observation_1', 'observation_2')]
    sources.append(dict(name='new_geometries_one_stream', task_streams=1,
        old_only_physical_calls=300, old_only_search_terminal_cpu_seconds=20.))
    inventories = [dict(status='sealed', tasks=[dict(seed=7, distinct_policies=n)]) for n in (7, 8)]
    result = prep.cost_projection(sources, inventories, dict(candidate_geometries=[24, 48]), 48)
    selected = result['selected']
    assert selected['reachable_successful_physical_call_cap'] == 6104*48
    assert selected['expanded_logical_call_cap'] == 6564*48
    assert selected['historical_no_cross_stream_dedup_cold_truth_fields'] == 45*48
    assert selected['historical_cost_scenarios'][0]['projected_old_only_physical_calls'] == 228*96
    assert result['cold_truth']['policy_arrays_read'] is False


def preparation_fixture(root):
    plan = dict(name='mainline_precision_confirmation_preparation_v1', output_dir='output',
        source83_readout='old.json', source82_readout='new.json', historical_accounts=['paid-before.json'])
    store(root, 'checks.json', dict(status='passed', cpu_seconds=.1))
    store(root, 'old.json', dict(streams=[dict(controls=['nine-old-reference'])],
        independent=dict(controls=['nine-old-independent']), analysis_failures=[]))
    store(root, 'new.json', dict(controls=['nine-new-geometries'], analysis_failures=['retained-source-failure']))
    return plan


def test_prepare_wires_analysis_then_registration_without_science(tmp_path, monkeypatch):
    from performance import prepare_mainline_precision_confirmation as registration
    plan = preparation_fixture(tmp_path)
    events = []
    def precision(*args):
        events.append('precision')
        return dict(status='selected', selected_N=24, selection_pass=True)
    def fees(*args):
        events.append('costs')
        return dict(selected=dict(geometries=24))
    def register(plan, result, checks, *, root):
        events.append('registration')
        assert (root / 'output/planning.json').is_file()
        assert result['selected_N'] == 24 and checks == 'checks.json'
        return dict(status='registered_not_started', execution_started=False)
    monkeypatch.setattr(prep, 'analyze_precision', precision)
    monkeypatch.setattr(prep, 'analyze_costs', fees)
    monkeypatch.setattr(registration, 'register', register)
    result = prep.prepare(plan, 'manifest.json', 'checks.json', root=tmp_path)
    assert events == ['precision', 'costs', 'registration']
    assert result['scientific_execution_started'] is False
    assert not any(result['new_work'].values())
    saved = json.loads((tmp_path / 'output/readout.json').read_text())
    assert saved['all_nine_controls']['new_geometry'] == ['nine-new-geometries']
    assert saved['original_analysis_failures']['newer'] == ['retained-source-failure']
    prep.prepare(plan, 'manifest.json', 'checks.json', root=tmp_path)
    assert events == ['precision', 'costs', 'registration']


def test_failed_checks_stop_before_reading_or_planning(tmp_path, monkeypatch):
    store(tmp_path, 'checks.json', dict(status='failed'))
    def forbidden(*args):
        raise AssertionError('No planning or registration before admission')
    monkeypatch.setattr(prep, 'analyze_precision', forbidden)
    with pytest.raises(ValueError, match='checks must pass'):
        prep.prepare(dict(output_dir='output'), 'manifest.json', 'checks.json', root=tmp_path)

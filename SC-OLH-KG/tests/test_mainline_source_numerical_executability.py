"""Deterministic parser substitutes; no real scientific artifacts are opened."""
import ast
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

from performance import audit_mainline_source_numerical_executability as audit


ARMS = ['legacy_role_full', 'legacy_role_full_native_terminal',
        'legacy_role_after_init_point_only', 'legacy_role_after_init_point_only_native_terminal',
        'legacy_native_neighbors', 'same_source_gp', 'legacy_role_point_only',
        'current_structural_mixture', 'initialization_only',
        'source_permuted_role_after_init_point_only', 'source_permuted_role_point_only']


def put(root, path, value):
    audit.save(root / path, value)


def fixture(tmp_path):
    plan = dict(output_dir='audit', source_output_dir='science', saved_record_audit_authorized=True,
        source_asset_seal_file='sources/seal.json', failure_index_file='saved/failures.json',
        failure_execution_index_file='saved/online_failure_log_index.json',
        source_diagnostics=[], retained_terminal_arms=ARMS,
        historical_accounts=['saved/accounting.json'], source_warning='Queue warning retained',
        coverage=dict(registered_terminal_cells=44, retained_failures=4,
                      complete_source_C_geometries=1, complete_source_G_geometries=1,
                      full144_source_C=None, full144_source_G=None))
    families = []
    for family in ('Inventory', 'Queue', 'FactorShock'):
        entry = dict(family=family)
        models = []
        for association in ('real', 'permuted'):
            entry[association] = {}
            for kind in ('structural', 'native'):
                name = f'sources/{family}/{association}/{kind}.json'
                entry[association][kind] = name
                if kind == 'structural':
                    diagnostic = dict(source_sites=128, source_quadrature_nodes=1024,
                        source_identities=['spectral/head_specific/observed_task'],
                        source_identity_weights=[.9], precision_certified=False, cpu_seconds=2.,
                        source_identity_heads=[dict(identity=['spectral', 'head_specific', 'observed_task'],
                            head='f', weight_ess=2.6, maximum_component_mass=.4,
                            split_log_normalizer_gap=.25, precision_certified=False,
                            low_importance_ess=True, joint_risk_fit=None)])
                else:
                    diagnostic = dict(status='fit', component_stage='spectral_hvd', n_records=128,
                                      training=dict(source_simulator_calls=384, source_observation_replicates=3),
                                      disabled_proposal=dict(status='not_materialized'))
                put(tmp_path, name, diagnostic)
                if association == 'permuted':
                    receipt = f'sources/{family}/{association}/{kind}_receipt.json'
                    put(tmp_path, receipt, dict(status='complete', cost=dict(cpu_seconds=3., source_archive_calls_reused=384)))
                    models.append(dict(kind=kind, receipt_file=receipt))
        plan['source_diagnostics'].append(entry)
        families.append(dict(family=family, models=models))
    put(tmp_path, 'sources/seal.json', dict(status='sealed_sources_only', families=families))
    cases, index, failed_jobs = [], [], []
    traces = [
        '  File "mainline_role_after_init.py", line 40, in _initial_conditioning\n',
        '  File "mainline_role_after_init.py", line 189, in after_init_point_only_moment_state\n    model.update_raw(0, raw)\n',
        '  File "mainline_legacy_backend.py", line 158, in _decision_backend_terminal_recommendation\n',
        '  File "unrelated.py", line 8, in opaque_saved_operation\n',
    ]
    for seed in range(1, 5):
        for arm in ARMS:
            folder = f'science/stream_1/replicate_1/FactorShock/search/seed_{seed}/{arm}'
            failed = arm == ARMS[-2]
            cases.append(dict(stream='stream1', seed=seed, arm=arm, domain='FactorShock',
                              source_case=folder, old=dict(status='missing' if failed else 'complete'),
                              forbidden_scientific_score='must not appear in audit outputs'))
            if not failed:
                continue
            label = f'stream1_r1_search_{seed}_{arm}'
            item = dict(label=label, arm=arm, attempt_count=1, statuses=['failed'],
                        search_calls=[seed-1], cpu_seconds=[10. + seed], raw_counts=[seed-1], raw_files=1)
            index.append(item)
            failed_jobs.append(dict(label=label, seed=seed, arm=arm, returncode=1))
            ledger = tmp_path / folder / 'attempts.jsonl'
            ledger.parent.mkdir(parents=True)
            record = dict(status='failed', stage='search', new_target_search_calls=seed-1,
                          cpu_seconds=10. + seed, posterior_restorations=None,
                          error=f"RuntimeError('{audit.GUARD}: 2e-8')")
            ledger.write_text(json.dumps(record) + '\n')
            log = tmp_path / f'science/cluster/online/shard_0_4/execution_logs/{label}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            message = f'RuntimeError: {audit.GUARD}: 2e-8\n'
            log.write_text('Traceback (most recent call last):\n' + traces[seed-1] + message + message)
    put(tmp_path, 'science/execution_status.json', dict(status='sealed', cases=cases))
    put(tmp_path, 'saved/online_failure_log_index.json',
        dict(provenance=dict(source_state_file='local_saved_source_alignment_science_state.json'),
             online_shards=[dict(start=0, end=4)],
             registered_failure_labels=[row['label'] for row in failed_jobs]))
    put(tmp_path, 'saved/failures.json', dict(returncode=0, result=dict(failed_count=4, rows=index)))
    account = dict(target_ledger_cpu_seconds=50., failed_prefix_cpu_seconds=50.,
                   wrapper_cpu_seconds=60., source_model_restorations=None,
                   note='failure and ledger are included in wrapper; not additive')
    put(tmp_path, 'saved/accounting.json', account)
    put(tmp_path, 'checks.json', dict(status='passed'))
    return plan


@pytest.mark.parametrize(('trace', 'phase'), [
    ('in _initial_conditioning', 'initial_conditioning'),
    ('in after_init_point_only_moment_state\n    model.update_raw(0, raw)', 'post_initial_own_update'),
    ('in _decision_backend_terminal_recommendation\nin _initial_conditioning', 'terminal_recommendation'),
    ('returned_query_count=2\nin update_batch', 'unknown'),
])
def test_phase_uses_stack_not_returned_query_count(trace, phase):
    row = audit.parse_log(trace + f'\nRuntimeError: {audit.GUARD}: 8e-8')
    assert row['traceback_phase'] == phase
    assert row['recorded']['reported_error_ratio'] == 8e-8
    assert row['recorded']['head'] is None
    assert row['recorded']['integration_status'] is None
    assert row['recorded']['minimum_normalizer'] is None
    assert row['code_inference']['integrator_success'] is True


def test_integration_failure_does_not_get_success_inference():
    row = audit.parse_log('RuntimeError: Batched raw moment integration failed: roundoff; '
                          'status=2, error=1e-9, relative_error=2e-8')
    assert row['code_inference'] is None
    assert row['recorded']['integration_status'] == 2
    assert row['recorded']['integration_absolute_error'] == 1e-9
    assert row['recorded']['reported_error_ratio'] is None


def test_complete_saved_audit_preserves_science_status_and_nested_fees(tmp_path):
    plan = fixture(tmp_path)
    before = copy.deepcopy(plan)
    result = audit.audit(plan, root=tmp_path, checks_file='checks.json')
    assert plan == before
    assert result['status'] == 'complete'
    assert result['diagnosis'] == 'cause_unidentified'
    assert result['source_diagnostic_files'] == dict(registered=12, read=12)
    assert result['failure_logs'] == dict(registered=4, read=4)
    assert result['retained_terminal_status']['status_counts'] == dict(complete=40, missing=4)
    assert len(result['retained_terminal_status']['arms']) == 11
    assert set(result['phase_counts']) == {'initial_conditioning', 'post_initial_own_update', 'terminal_recommendation', 'unknown'}
    assert all(value == 0 for value in result['new_work'].values())
    compact = json.loads((tmp_path / 'audit/conclusion.json').read_text())
    assert len(compact['source_diagnostic_summary']) == 12
    assert compact['failure_summary']['recorded_attempt_count'] == 4
    assert compact['failure_summary']['recorded_attempt_cpu_seconds'] == 50.
    assert compact['failure_summary']['returned_query_count'] == 6
    assert compact['failure_summary']['minimum_recorded_error_ratio'] == 2e-8
    assert compact['failure_summary']['maximum_recorded_error_ratio'] == 2e-8
    assert len(compact['failure_summary']['arm_phase_counts']) == 4
    failures = json.loads((tmp_path / 'audit/failure_locations_and_costs.json').read_text())
    for row in failures['rows']:
        assert row['attempt_count'] == 1
        assert row['guard_message_occurrences'] == 2
        assert row['retained_scientific_status'] == 'missing'
        assert row['recorded_attempt_cpu_seconds'] == 10. + row['seed']
        assert row['saved_attempts'][0]['posterior_restorations'] is None
    accounting = json.loads((tmp_path / 'audit/accounting.json').read_text())
    retained = accounting['retained_historical_accounts'][0]['saved_account']
    assert retained['wrapper_cpu_seconds'] == 60.
    assert retained['target_ledger_cpu_seconds'] == retained['failed_prefix_cpu_seconds'] == 50.
    assert retained['source_model_restorations'] is None
    assert 'total_cpu_seconds' not in accounting  # No sum of nested historical fees.
    for path in (tmp_path / 'audit').glob('*.json'):
        assert 'forbidden_scientific_score' not in path.read_text()
    assert audit.audit(plan, root=tmp_path) == result  # Completed work is reused.
    assert len((tmp_path / 'audit/attempts.jsonl').read_text().splitlines()) == 1


def test_source_heads_preserved_native_fields_not_fabricated(tmp_path):
    plan = fixture(tmp_path)
    audit.audit(plan, root=tmp_path, checks_file='checks.json')
    rows = json.loads((tmp_path / 'audit/source_diagnostics.json').read_text())['rows']
    structural = next(row for row in rows if row['kind'] == 'structural')
    head = structural['saved_diagnostics']['source_identity_heads'][0]
    assert head['split_log_normalizer_gap'] == .25
    assert head['precision_certified'] is False
    assert head['joint_risk_fit'] is None
    native = next(row for row in rows if row['kind'] == 'native')
    assert native['importance_sampling_scope'] == 'not_applicable_to_native_schema'
    assert 'source_identity_heads' not in native['saved_diagnostics']
    assert native['saved_diagnostics']['precision_certified'] is None
    assert native['saved_diagnostics']['status'] == 'fit'
    assert native['saved_diagnostics']['training']['source_simulator_calls'] == 384
    compact = json.loads((tmp_path / 'audit/conclusion.json').read_text())
    structural = next(row for row in compact['source_diagnostic_summary'] if row['kind'] == 'structural')
    native = next(row for row in compact['source_diagnostic_summary'] if row['kind'] == 'native')
    assert structural['heads'] == 1 and structural['low_ess_heads'] == 1
    assert structural['minimum_ess'] == 2.6
    assert native['saved_optimizer_diagnostics'] is None
    assert native['precision_certified'] is None


def test_missing_log_does_not_drop_original_failure_or_ledger(tmp_path):
    plan = fixture(tmp_path)
    log = next((tmp_path / 'science/cluster/online/shard_0_4/execution_logs').glob('*_1_*.log'))
    log.unlink()
    result = audit.audit(plan, root=tmp_path, checks_file='checks.json')
    assert result['status'] == 'complete_with_missing_records'
    assert result['failure_logs'] == dict(registered=4, read=3)
    failures = json.loads((tmp_path / 'audit/failure_locations_and_costs.json').read_text())
    assert failures['retained_failure_rows'] == 4
    row = next(row for row in failures['rows'] if row['seed'] == 1)
    assert row['retained_scientific_status'] == 'missing'
    assert row['recorded_attempt_cpu_seconds'] == 11.
    assert row['traceback_phase'] == 'unknown'
    assert row['code_inference'] is None
    assert row['recorded']['reported_error_ratio'] is None


def test_missing_diagnostic_and_unknown_cpu_remain_unknown(tmp_path):
    plan = fixture(tmp_path)
    (tmp_path / plan['source_diagnostics'][0]['real']['structural']).unlink()
    ledger = next((tmp_path / 'science/stream_1/replicate_1/FactorShock/search/seed_1').glob('*/attempts.jsonl'))
    value = json.loads(ledger.read_text())
    del value['cpu_seconds']
    ledger.write_text(json.dumps(value) + '\n')
    result = audit.audit(plan, root=tmp_path, checks_file='checks.json')
    assert result['source_diagnostic_files']['read'] == 11
    assert result['status'] == 'complete_with_missing_records'
    failures = json.loads((tmp_path / 'audit/failure_locations_and_costs.json').read_text())
    row = next(row for row in failures['rows'] if row['seed'] == 1)
    assert row['recorded_attempt_cpu_seconds'] is None
    assert row['original_failure_index']['cpu_seconds'] == [11.]


def test_actual_ledger_attempt_count_is_not_log_signature_count(tmp_path):
    plan = fixture(tmp_path)
    ledger = next((tmp_path / 'science/stream_1/replicate_1/FactorShock/search/seed_1').glob('*/attempts.jsonl'))
    ledger.write_text(ledger.read_text() * 3)
    result = audit.audit(plan, root=tmp_path, checks_file='checks.json')
    failures = json.loads((tmp_path / 'audit/failure_locations_and_costs.json').read_text())
    row = next(row for row in failures['rows'] if row['seed'] == 1)
    assert row['attempt_count'] == 3
    assert row['guard_message_occurrences'] == 2
    assert row['original_failure_index']['attempt_count'] == 1
    assert any('Attempt count differs' in issue['error'] for issue in result['input_issues'])


def test_reader_rejects_models_and_module_has_no_scientific_imports(tmp_path):
    reader = audit.SavedRecords(tmp_path)
    with pytest.raises(ValueError, match='Expected saved'):
        reader.read('failed_search_engine.pkl')
    with pytest.raises(ValueError, match='Expected saved'):
        reader.read('source_models.npz')
    tree = ast.parse(Path(audit.__file__).read_text())
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module.split('.')[0])
    assert set(modules) <= sys.stdlib_module_names


def test_failed_checks_and_unfinished_attempt_stop_before_reading(tmp_path):
    plan = fixture(tmp_path)
    put(tmp_path, 'checks.json', dict(status='failed'))
    with patch.object(audit.SavedRecords, 'read', side_effect=AssertionError('must not read')):
        with pytest.raises(ValueError, match='substitutes must pass'):
            audit.audit(plan, root=tmp_path, checks_file='checks.json')
    (tmp_path / 'audit').mkdir()
    (tmp_path / 'audit/attempts.jsonl').write_text('{"status":"failed"}\n')
    with pytest.raises(RuntimeError, match='unfinished audit attempt retained'):
        audit.audit(plan, root=tmp_path, checks_file='checks.json')


def test_registration_without_execution_authorization_does_not_read(tmp_path):
    plan = fixture(tmp_path)
    plan['saved_record_audit_authorized'] = False
    with patch.object(audit.SavedRecords, 'read', side_effect=AssertionError('must not read')):
        with pytest.raises(ValueError, match='authorized separately'):
            audit.audit(plan, root=tmp_path, checks_file='checks.json')

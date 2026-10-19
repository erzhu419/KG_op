"""Saved-record audit fixtures; no scientific cohort, simulation or truth calls."""
from copy import deepcopy

from performance.mainline_terminal_recording_audit import audit_record, audit_stream, compact_audit
from performance.run_mainline_legacy_backend_bridge import save
from performance.run_mainline_same_policy import transform
from test_mainline_terminal_recording import construct, engine_fixture


AFTER = 'legacy_role_after_init_point_only'
ARMS = ['legacy_role_full', 'legacy_role_full_native_terminal', AFTER, AFTER + '_native_terminal',
    'legacy_native_neighbors', 'same_source_gp', 'legacy_role_point_only',
    'current_structural_mixture', 'initialization_only']


def saved_fixture():
    engine = engine_fixture()
    frozen = construct(engine)
    item = transform(dict(**deepcopy(frozen['terminal']), primary=frozen['primary']))
    history = [dict(point=list(x), raw=y.tolist()) for x, y in engine.history]
    return engine, item, history


def test_actual_record_connects_every_observed_policy_and_incumbent_displacement():
    """Catch silently incomplete pool/slot records even when the chosen point survives."""
    engine, item, history = saved_fixture()
    calls = deepcopy(engine.calls)
    result = audit_record(item['terminal_recording'], history, item['frozen_shortlist'], 1, AFTER)
    assert result['status'] == 'complete' and result['issues'] == []
    assert len(result['observed']) == 12 and engine.calls == calls
    assert result['observed'][11]['observation_indices'] == [11, 12]
    assert result['observed'][11]['disposition'] == 'incumbent_insertion_displaced'
    assert result['observed'][9]['disposition'] == 'omitted_and_objective_challenger_ineligible'
    assert result['observed'][8]['objective_eligible'] is True
    assert result['observed'][0]['pre_incumbent_slot'] is None
    assert result['observed'][0]['post_incumbent_slot'] == result['observed'][0]['final_slot'] == 2
    assert result['shortlist_observed_indices'] == [10, 0, 3]


def test_missing_fields_wrong_physical_pool_and_native_context_are_explicit():
    """Catch saved-score truncation, same-index/different-policy joins and native contamination."""
    _, item, history = saved_fixture()
    record = item['terminal_recording']
    broken = deepcopy(record)
    next(s for s in broken['stages'] if s['phase'] == 'objective_gate')['components'].pop('objective')
    assert 'objective_gate:missing_or_unaligned_objective' in audit_record(
        broken, history, item['frozen_shortlist'], 1, AFTER)['issues']
    broken = deepcopy(record)
    next(s for s in broken['stages'] if s['phase'] == 'primary_pool')['pool'][0][0] = 99
    result = audit_record(broken, history, item['frozen_shortlist'], 1, AFTER)
    assert 'primary_pool:incomplete_observed_physical_pool' in result['issues']
    assert result['status'] == 'invalid_record'
    assert 'posterior_context_mismatch' in audit_record(record, history, item['frozen_shortlist'], 0, AFTER)['issues']
    broken = deepcopy(record)
    next(s for s in broken['stages'] if s['phase'] == 'primary_order')['selected'] = history[9]['point']
    assert 'selected_not_first_saved_rank' in audit_record(broken, history, item['frozen_shortlist'], 1, AFTER)['issues']
    assert 'transformed_shortlist_not_actual_input' in audit_record(record, history,
        list(reversed(item['frozen_shortlist'])), 1, AFTER)['issues']
    assert audit_record(None, history, item['frozen_shortlist'], 1)['status'] == 'missing_record'


def test_initial_safe_pool_uses_registered_atlas_without_relaxing_observed_pools():
    """Accept the real initial-only fallback; reject an aligned but wrong subset."""
    engine = engine_fixture()
    engine.config.terminal_safe_interior_candidate_scope = 'initial'
    frozen = construct(engine)
    item = transform(dict(**deepcopy(frozen['terminal']), primary=frozen['primary']))
    history = [dict(point=list(x), raw=y.tolist()) for x, y in engine.history]
    record = item['terminal_recording']
    calls = deepcopy(engine.calls)
    stages = {stage['phase']: stage for stage in record['stages']}
    assert len(stages['safe_pool']['pool']) == 10
    assert len(stages['primary_pool']['pool']) == len(stages['objective_gate']['pool']) == 12
    assert audit_record(record, history, item['frozen_shortlist'], 1, AFTER)['issues'] == []
    broken = deepcopy(record)
    safe = next(s for s in broken['stages'] if s['phase'] == 'safe_pool')
    safe['pool'][-1] = history[10]['point']
    result = audit_record(broken, history, item['frozen_shortlist'], 1, AFTER)
    assert result['status'] == 'invalid_record'
    assert result['issues'] == ['safe_pool:incomplete_initial_physical_pool']
    broken = deepcopy(record)
    next(s for s in broken['stages'] if s['phase'] == 'primary_pool')['pool'].pop()
    assert 'primary_pool:incomplete_observed_physical_pool' in audit_record(
        broken, history, item['frozen_shortlist'], 1, AFTER)['issues']
    assert engine.calls == calls


def test_stream_retains_all_nine_controls_failed_outcomes_and_missing_records(tmp_path):
    """Catch denominator shrinkage and treating unexecuted verification as a lost comparison."""
    _, item, history = saved_fixture()
    cells = [dict(name=name,
        history_arm=name.removesuffix('_native_terminal'),
        terminal_backend='native' if name.endswith('_native_terminal') else 'original') for name in ARMS]
    plan = dict(output_dir='stream', tasks=[dict(geometry_task_seed=1, domain='fixture')], terminal_arms=cells)
    cases, summary_rows = [], []
    for cell in cells:
        name = cell['name']
        source = 'search/' + cell['history_arm']
        descriptor = dict(status='missing', input_file=None, failure=dict(status='search_failed'))
        case = dict(seed=1, arm=name, domain='fixture', history_arm=cell['history_arm'], source_case=source,
            old=deepcopy(descriptor), new=deepcopy(descriptor))
        if name == 'same_source_gp':
            # Existing failed/missing descriptors may name input files that were never written.
            case['old'] = dict(status='failed', input_file='absent/gp_old.json')
            case['new'] = dict(status='missing', input_file='absent/gp_new.json', reason='old_verification_failed')
        if name in (AFTER, 'legacy_role_point_only'):
            current = deepcopy(item)
            current.update(source_case=source, verification_seed=91)
            current['terminal_recording']['backend_arm'] = cell['history_arm']
            if name == 'legacy_role_point_only':
                current.pop('terminal_recording')
            save(tmp_path/source/'search.json', dict(history=history))
            for protocol in ('old', 'new'):
                path = f'inputs/{name}/{protocol}.json'
                if not (name == AFTER and protocol == 'new'):
                    save(tmp_path/path, current)
                case[protocol] = dict(status='failed', input_file=path, failure_file=f'{name}/failure.json')
            if name == AFTER:
                # No comparison was reached: incumbent is the only safety-certified point.
                result = dict(deployed=current['frozen_shortlist'][1]['point'], new_verification_calls=208,
                    verification=dict(certified=True, attempts=[dict(candidate_index=0, certified=False,
                        upper_margin=.2, verification_budget=80), dict(candidate_index=1, certified=True,
                        upper_margin=-.1, verification_budget=128)],
                        objective_comparison=dict(status='challenger_not_safety_certified', simulation_calls=0)))
                save(tmp_path/'verification.json', result)
                case['old'] = dict(status='complete', input_file=case['old']['input_file'], verification_file='verification.json')
        cases.append(case)
        summary_rows.append(dict(seed=1, arm=name, old=dict(C=None, G=None), new=dict(C=None, G=None),
            quality_reference_status='no_true_feasible_initial', quality_baseline_defined=False))
    save(tmp_path/'stream'/'execution_status.json', dict(status='sealed', cases=cases))
    audit = audit_stream(tmp_path, plan, terminal_summary=dict(rows=summary_rows))
    assert audit['coverage'] == dict(registered_pairs=9, sealed_pairs=9, valid_records=1, issue_pairs=8)
    assert len(audit['cases']) == len(audit['by_arm']) == 9
    after = next(row for row in audit['cases'] if row['arm'] == AFTER)
    assert after['protocols']['old']['comparisons'][0]['executed'] is False
    assert after['protocols']['old']['comparisons'][0]['status'] == 'challenger_not_safety_certified'
    assert after['protocols']['new']['status'] == 'failed'
    assert after['protocols']['new']['input_available'] is False
    assert after['saved_results']['old']['G'] is None
    gp = next(row for row in audit['cases'] if row['arm'] == 'same_source_gp')
    assert gp['status'] == 'no_frozen_input'
    assert gp['protocols']['old']['status'] == 'failed' and not gp['protocols']['old']['input_available']
    assert gp['protocols']['new']['status'] == 'missing' and not gp['protocols']['new']['input_available']
    assert compact_audit(audit)['coverage']['issue_pairs'] == 8
    assert audit['truth_read'] is False and audit['new_scientific_calls'] == 0

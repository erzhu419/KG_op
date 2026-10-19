"""Catch real schema, contrast-direction and fee mismatches across four-cell stages."""
import json

import numpy as np
import pytest

from performance import mainline_reference_coverage_cohort as cohort
from performance import mainline_reference_coverage_execution as execution
from performance import summarize_mainline_reference_coverage_confirmation as summary
from test_mainline_reference_coverage_cohort import setup_cohort, write_online
from test_mainline_reference_coverage_execution import Substitute, substitute_verifiers


def test_saved_search_to_four_terminals_and_sealed_summary_keeps_all_nine_controls(tmp_path, monkeypatch):
    plan, source = setup_cohort(tmp_path, seeds=[1])
    source['decision_protocol']['verification_method'] = 'normal_quantile_tolerance'
    cohort.save(tmp_path / 'source.json', source)
    cohort.freeze_initial_references(plan, root=tmp_path)
    write_online(tmp_path, source)
    names = [summary.FULL, summary.AFTER, summary.POINT, 'same_source_gp',
             'legacy_native_neighbors', 'current_structural_mixture']
    renamed = {f'online_{i}': name for i, name in enumerate(names)}
    for old, new in renamed.items():
        (tmp_path / 'search/seed_1' / old).rename(tmp_path / 'search/seed_1' / new)
    source['arms'] = [dict(name=renamed.get(row['name'], row['name'])) for row in source['arms']]
    terminal = cohort.read(tmp_path / 'terminal.json')
    for cell in terminal['arms']:
        cell['history_arm'] = renamed.get(cell['history_arm'], cell['history_arm'])
        cell['name'] = (cell['history_arm'] + '_native_terminal'
                        if cell['terminal_backend'] == 'native' else cell['history_arm'])
    plan.update(search_arms=[row['name'] for row in source['arms']], terminal_arms=terminal['arms'],
        tasks=[dict(geometry_task_seed=1, domain='fixture')], statistical_unit='one synthetic geometry',
        analysis=dict(primary='AFTER joint-minus-original'),
        budget=dict(maximum_target_physical_calls=10 + 6 * 3 + 9 * 512))
    cohort.save(tmp_path / 'source.json', source)
    cohort.save(tmp_path / 'terminal.json', terminal)
    initial = cohort.read(tmp_path / 'search/seed_1/initialization_only/search.json')['history']
    cohort.save(tmp_path / 'search/seed_1/initialization.json', dict(history=initial))
    # Existing search ledgers charge shared initial10 once and six online3 histories once.
    for arm in plan['search_arms']:
        ledger = tmp_path / 'cohort/search_costs' / arm / 'attempts.jsonl'
        ledger.parent.mkdir(parents=True, exist_ok=True)
        row = dict(stage='search', status='completed', cpu_seconds=0., wall_seconds=0.,
            new_target_initial_calls=10 if arm == cohort.STOP else 0,
            new_target_search_calls=0 if arm == cohort.STOP else 3)
        ledger.write_text(json.dumps(row) + '\n')
    frozen = cohort.freeze_cohort_inputs(plan, root=tmp_path)
    assert frozen['registered_terminal_cells'] == 36
    with pytest.raises(ValueError, match='barrier is absent'):
        summary.freeze_truth_inventory(plan, root=tmp_path)

    substitute_verifiers(monkeypatch, (True, True, True))
    problem = Substitute()
    for row in frozen['rows']:
        variants = {name: cohort.read(tmp_path / descriptor['input_file'])
                    for name, descriptor in row['variants'].items()}
        result = execution.execute_four_variants(tmp_path / row['execution_dir'], variants,
            stream=row['stream'], problem_factory=lambda item: problem)
        assert result['status'] == 'complete'
        assert all(cell['logical_verification_calls'] <= 352 for cell in result['variants'].values())
    seal = cohort.seal_cohort_results(plan, root=tmp_path)
    assert seal['completed_cells'] == 36
    assert seal['physical_verification_calls'] == len(problem.calls)

    class AnalyticSubstitute:
        alpha, tau = .05, 0.
        weights, ref_point = np.array([1.]), np.array([1.])

        def simulate(self, *args, **kwargs):
            pytest.fail('Post-seal scoring must never simulate')

        def true_objective(self, point):
            return .9 - .05 * point[0]

        def true_constraint_mean(self, point):
            return -1.

        def true_sigma(self, point):
            return [0., .01]

    monkeypatch.setattr(summary, 'build_problem', lambda item: AnalyticSubstitute())
    inventory = summary.freeze_truth_inventory(plan, root=tmp_path)
    assert inventory['required_policies'] == 10
    scored = summary.score_task(plan, 1, root=tmp_path)
    assert scored['new_truth_function_calls'] == 30
    assert len(scored['outcomes']) == 36
    collected = summary.collect(plan, root=tmp_path)
    assert collected['registered_terminal_cells'] == collected['completed_terminal_cells'] == 36
    assert len(collected['arms']) == 9
    primary = collected['primary']
    assert primary['joint']['C']['all_registered_mean'] == 0.
    assert primary['joint']['G']['all_registered_mean'] == pytest.approx(.45)
    assert primary['coverage_only']['G']['all_registered_mean'] == pytest.approx(.10)
    assert primary['reference_only']['G']['all_registered_mean'] == 0.
    assert primary['interaction']['G']['all_registered_mean'] == pytest.approx(.35)
    ap = next(row for row in collected['secondary'] if (row['left'], row['right']) ==
              (summary.AFTER, summary.POINT))
    assert all(ap[name]['G']['all_registered_mean'] == 0. for name in cohort.VARIANTS)
    costs = collected['costs']
    assert costs['shared_initial_physical_calls'] == 10
    assert costs['shared_online_physical_calls'] == 18
    assert costs['unique_terminal_physical_calls'] == len(problem.calls)
    assert costs['successful_target_physical_calls'] == 28 + len(problem.calls)
    assert costs['analytic_truth_field_calls'] == 30
    assert costs['historical_accounts'] == plan['historical_accounts']
    # Completion reuse must not construct another simulator or add paid raw rows.
    assert summary.score_task(plan, 1, root=tmp_path) == scored

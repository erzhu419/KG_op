"""Admission preserves the full factorial grid without opening outcomes."""
from copy import deepcopy

import pytest

from performance import register_mainline_reference_coverage as reg
from performance import mainline_reference_coverage as wiring


def reference():
    return dict(tasks=[dict(seed=i, domain='fixture') for i in range(24)],
        streams=[dict(name=f'flow{i}', manifest=f'flow{i}.json') for i in range(2)],
        terminal_arms=[dict(name=name) for name in ['initialization_only']+[f'arm{i}' for i in range(8)]],
        historical_accounts=['old_failure_account.json'], output_dir='old_stage78')


def test_registration_keeps_controls_budget_tradeoffs_and_science_closed():
    source = reference()
    original = deepcopy(source)
    plan = reg.build_plan(source)
    assert source == original
    assert plan['terminal_arms'] == source['terminal_arms']
    assert set(plan['variants']) == set(reg.VARIANTS)
    assert plan['budgets']['new_safety_calls']+max(plan['budgets']['new_comparison_calls']) == 352
    assert plan['budgets']['old_actual_path_maximum'] == 336
    assert plan['budgets']['four_terminal_logical_cap_per_arm'] == 1408
    assert plan['scientific_validation']['execution_authorized'] is False
    assert plan['scientific_validation']['new_geometry_seeds'] == []
    assert plan['historical_accounts'][0] == 'old_failure_account.json'
    assert not any(plan['new_work'].values())


def test_all_initial_orders_precede_final_wiring_and_missing_arm_keeps_four_cells(monkeypatch,tmp_path):
    plan = reg.build_plan(reference())
    documents, events = {}, []
    for stream in plan['streams']:
        directory = stream['name']
        documents[stream['manifest']] = dict(output_dir=directory)
        cells = []
        for task in plan['tasks']:
            for arm in plan['terminal_arms']:
                path = f"{directory}/{task['seed']}/{arm['name']}/input.json"
                item = dict(protocol='old', seed=task['seed'], arm=arm['name'],
                    frozen_shortlist=[dict(point=[p]) for p in (0,1,2)],
                    candidate_budgets=[80,128,128], objective_comparison_delta=.05/3,
                    objective_incumbent_position=2, search_evaluation_count=10 if arm['name']=='initialization_only' else 13,
                    verification_seed=100+task['seed'])
                documents[path] = item
                cells.append(dict(seed=task['seed'], arm=arm['name'],old=dict(input_file=path)))
        documents[f'{directory}/execution_status.json'] = dict(status='sealed',cases=cells)
    del documents['flow1/23/arm7/input.json']

    def safe_read(path):
        relative = str(path.relative_to(tmp_path))
        assert relative.endswith(('input.json','execution_status.json','flow0.json','flow1.json'))
        if relative not in documents:
            raise FileNotFoundError(relative)
        return deepcopy(documents[relative])

    def freeze(item, **kwargs):
        assert not any(event=='variants' for event in events)
        events.append('initial')
        return dict(status='frozen', seed=item['seed'])

    def variants(item, order):
        assert events.count('initial') == 48
        events.append('variants')
        return {name:deepcopy(item) for name in reg.VARIANTS}

    monkeypatch.setattr(reg,'read',safe_read)
    monkeypatch.setattr(wiring,'freeze_initial_order',freeze)
    monkeypatch.setattr(wiring,'make_variants',variants)
    result = reg.prepare_inputs(tmp_path,plan)
    assert result['prepared_reference_orders'] == 48
    assert result['retained_arm_inputs'] == 432 and result['wired_arm_inputs'] == 431
    assert result['retained_variant_cells'] == 1728
    assert len(result['failures']) == 1
    failure = result['failures'][0]
    assert failure['arm']=='arm7' and failure['seed']==23 and failure['stream']=='flow1'
    assert set(failure['variants']) == set(reg.VARIANTS)
    assert all(v['executed'] is False for v in failure['variants'].values())


def test_failed_checks_block_admission_and_preserve_attempt_cost(monkeypatch,tmp_path):
    plan = reg.build_plan(reference())
    reg.save(tmp_path/'plan.json',plan)
    reg.save(tmp_path/'checks.json',dict(status='failed'))

    def forbidden(*args,**kwargs):
        raise AssertionError('Admission must not run after failed substitutes')

    monkeypatch.setattr(reg,'prepare_inputs',forbidden)
    with pytest.raises(ValueError,match='must pass'):
        reg.register('plan.json','checks.json',root=tmp_path)
    record=reg.read(tmp_path/plan['output_dir']/'registration_attempts/attempt_1.json')
    assert record['status']=='failed' and record['cpu_seconds']>=0
    assert 'old_failure_account.json' in record['historical_accounts']
    assert not any(record['new_work'].values())
    assert reg.read(tmp_path/'plan.json')['status']=='planned'


def test_resume_reuses_completed_orders_and_wiring_without_reading_them_again(monkeypatch,tmp_path):
    plan = reg.build_plan(reference())
    plan['tasks'] = plan['tasks'][:2]
    plan['streams'] = plan['streams'][:1]
    plan['terminal_arms'] = plan['terminal_arms'][:2]
    output = tmp_path/plan['output_dir']
    prior_order = dict(status='frozen', seed=0, exact_saved_value=1.)
    order_path = output/'reference_orders/flow0/seed_0.json'
    reg.save(order_path,prior_order)
    before = order_path.read_bytes()
    completed = [dict(stream='flow0',seed=0,domain='fixture',arm=arm['name'],
        status='wired_not_executed',preserved_metadata='original_complete') for arm in plan['terminal_arms']]
    reg.save(output/'wiring.json',dict(cases=completed+[
        dict(stream='flow0',seed=1,arm=arm['name'],status='admission_failed')
        for arm in plan['terminal_arms']]))
    original_read = reg.read
    calls = []
    cells = [dict(seed=seed,arm=arm['name'],old=dict(input_file=f'input/{seed}/{arm["name"]}.json'))
             for seed in range(2) for arm in plan['terminal_arms']]

    def read(path):
        if path.exists():
            return original_read(path)
        name = str(path.relative_to(tmp_path))
        if name=='flow0.json':
            return dict(output_dir='flow0')
        if name=='flow0/execution_status.json':
            return dict(status='sealed',cases=cells)
        assert name.startswith('input/1/'), 'Completed input must never be reopened'
        calls.append(name)
        return dict(seed=1, protocol='old',frozen_shortlist=[dict(point=[p]) for p in range(3)],
            candidate_budgets=[80,128,128],objective_comparison_delta=.05/3,
            objective_incumbent_position=2,search_evaluation_count=13,verification_seed=11)

    monkeypatch.setattr(reg,'read',read)
    monkeypatch.setattr(wiring,'freeze_initial_order',lambda item,**kw:dict(status='frozen',seed=item['seed']))
    monkeypatch.setattr(wiring,'make_variants',lambda item,order:{name:deepcopy(item) for name in reg.VARIANTS})
    result = reg.prepare_inputs(tmp_path,plan)
    assert result['reused_reference_orders']==result['newly_prepared_reference_orders']==1
    assert result['reused_wired_arm_inputs']==result['newly_wired_arm_inputs']==2
    assert result['failures']==[] and len(calls)==3
    assert order_path.read_bytes()==before
    saved = original_read(output/'wiring.json')['cases']
    assert saved[:2]==completed

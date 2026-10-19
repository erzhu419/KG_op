"""Full-population inventory, terminal barrier and analytic-only fee semantics."""
from types import SimpleNamespace

import numpy as np
import pytest

from performance import run_mainline_safe_unsampled_audit as audit


def test_full_action_union_keeps_unchosen_and_failed_history_exposure():
    initial=[dict(point=[0,0],raw=[0.,-1.])]
    bundle=dict(status='partial',points=[(0,0),(1,0),(2,0),(3,0)],
        audit={'stages':[dict(action_ids=[0,1,2],generated_ids=[0,1,2,3],selected_id=1)]},history=initial)
    cases=[dict(old=dict(status='complete',item={'frozen_shortlist':[{'point':[4,0]}]},
        record={'deployed':[5,0]}),new=dict(status='missing',item=None,record=None))]
    points=audit.inventory_points(initial,{'role':bundle},cases)
    assert set(points)=={(0,0),(1,0),(2,0),(4,0),(5,0)}
    assert (2,0) in points and (3,0) not in points
    assert len(points)==5


def test_inventory_cannot_open_candidates_before_complete_terminal_seal(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    monkeypatch.setattr(audit.terminal,'ROOT',tmp_path)
    plan=dict(output_dir='cohort',tasks=[{'geometry_task_seed':17}],terminal_arms=['a'])
    def forbidden(*args,**kwargs):
        raise AssertionError('Candidate inventory opened before terminal seal')
    monkeypatch.setattr(audit,'task_material',forbidden)
    with pytest.raises(RuntimeError,match='globally sealed'):
        audit.freeze(plan)
    audit.save(tmp_path/'cohort/execution_status.json',dict(status='sealed',cases=[dict(seed=17,arm='a',
        old=dict(status='failed'),new=dict(status='complete',input_file='absent.json',verification_file='absent2.json'))]))
    with pytest.raises(RuntimeError,match='absent'):
        audit.freeze(plan)
    assert not (tmp_path/'cohort/mechanism/inventory.json').exists()


def test_analytic_calls_charge_missing_fields_and_do_not_sample_or_restore():
    calls=[]
    def record(name,value):
        calls.append(name)
        return value
    problem=SimpleNamespace(alpha=.05,tau=0.,weights=np.array([1.]),ref_point=np.array([1.]),
        true_objective=lambda p:record('f',p[0]/10),
        true_constraint_mean=lambda p:record('g',-1.),true_sigma=lambda p:record('s',np.array([.2,.1])),
        simulate=lambda *args:pytest.fail('Analytic scoring must not simulate'))
    cache={(1,0):dict(objective=.1,constraint_mean_centered=-1.,constraint_sigma=.1,quality=.9,true_feasible=True),
           (2,0):dict(objective=.2)}
    cost=dict(audit.ZERO)
    audit.score_points([(1,0),(2,0),(3,0)],cache,lambda:problem,cost)
    assert calls==['g','s','f','g','s'] and cost['new_truth_function_calls']==5
    assert cost['truth_problem_constructions']==1
    assert all(cache[p]['true_feasible'] for p in cache)
    before=cost.copy()
    audit.score_points([(1,0),(2,0),(3,0)],cache,lambda:pytest.fail('Complete cache cannot build problem'),cost)
    assert cost==before and cost['new_target_search_calls']==cost['new_verification_calls']==0


def test_partial_audit_does_not_invent_returned_observations(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    initial=[dict(point=[i,0],raw=[i,-1.]) for i in range(10)]
    folder=tmp_path/'role'
    audit.save(folder/'failed_search_raw_1.json',[dict(point=[10,0],raw=[1.,-1.])])
    np.savez_compressed(folder/'failed_query_policies_1.npz',points=np.array([[10,0],[11,0]]))
    audit.save(folder/'failed_query_audit_1.json',dict(status='partial',returned_raw_count=1,
        paid_raw_file='failed_search_raw_1.json',query_policies_file='failed_query_policies_1.npz',
        stages=[dict(paid_prefix=10,action_ids=[0,1],selected_id=0),dict(paid_prefix=11,action_ids=[0,1],selected_id=1)]))
    bundle=audit.history_bundle(folder,initial)
    assert bundle['status']=='partial' and len(bundle['history'])==11
    assert len(bundle['audit']['stages'])==2 and tuple(bundle['history'][-1]['point'])==(10,0)
    assert (11,0) in audit.inventory_points(initial,{'role':bundle},[])


def test_terminal_summary_with_saved_contexts_never_fills_missing_truth(tmp_path,monkeypatch):
    base=audit.terminal
    monkeypatch.setattr(base,'ROOT',tmp_path)
    case=dict(domain='fixture',seed=17,arm=base.AFTER)
    monkeypatch.setattr(base,'load_sealed',lambda plan:({'costs':{}},[case]))
    monkeypatch.setattr(base,'collect_task_truth',lambda *args:pytest.fail('Read-only summary retried oracle'))
    monkeypatch.setattr(base,'score_pair',lambda *args:dict(status='complete',old={},new={}))
    monkeypatch.setattr(base,'outcome_record',lambda c,p,v:dict(status='complete',protocol=p))
    monkeypatch.setattr(base,'summarize_rows',lambda *args:([dict(arm=base.AFTER)],[]))
    monkeypatch.setattr(base,'add_sensitivity',lambda *args:None)
    monkeypatch.setattr(base,'recorded_attempt_costs',lambda *args:[])
    result=base.score(dict(output_dir='cohort',scope='fixture',tasks=[{}],terminal_arms=[base.AFTER]),
        saved_contexts={('fixture',17):{}},saved_truth_counts=[dict(status='missing_truth_retained')])
    assert result['costs']['scoring_new_truth_function_calls']==0
    assert result['truth_by_task']==[dict(status='missing_truth_retained')]

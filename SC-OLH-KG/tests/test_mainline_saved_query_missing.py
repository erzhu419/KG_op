"""Missing quality baselines and registered histories must not erase saved choices."""
import numpy as np
import pytest

from performance import diagnose_mainline_three_family_queries as query
from performance.diagnose_mainline_observation_seed_queries import counts


def saved_task(directory, *, feasible_initial):
    points = [(0,), (1,), (2,)]
    truth = [dict(point=[0], truth=dict(objective=.5, true_feasible=feasible_initial,
        constraint_mean_centered=-2., constraint_sigma=1., chance_margin=-1.)),
        dict(point=[1], truth=dict(objective=.4, true_feasible=True,
        constraint_mean_centered=-2., constraint_sigma=1., chance_margin=-1.))]
    query.save(directory/'truth.json', truth)
    history = [dict(point=[0], raw=[.5, -2.]) for _ in range(10)]
    query.save(directory/'initialization.json', dict(history=history))
    np.savez(directory/'neighbors.npz', points=np.asarray(points[:2]))
    for arm in (query.FULL, query.AFTER):
        case = directory/arm; case.mkdir()
        np.savez(case/'query_policies.npz', points=np.asarray(points))
        stages = []
        for prefix in (10, 11, 12):
            score = dict(objective_mean=[.5, .4, .3], objective_epistemic=[1.]*3,
                sampled_objective=[.5, .4, .3], constraint_mean=[-2.]*3,
                constraint_epistemic=[1.]*3, constraint_aleatoric=[1.]*3,
                sampled_constraint=[-2.]*3, sampled_margin=[-1.]*3,
                stochastic_margin_mean=[-1.]*3, total=[-.5, -.4, -.3],
                sampled_feasibility_mode='feasible_objective', sampled_feasible_count=3)
            stages.append(dict(paid_prefix=prefix, action_ids=[0, 1, 2],
                observed_ids=[0] if prefix==10 else [0, 2], selected_index=2, selected_id=2,
                score=score))
        query.save(case/'query_audit.json', dict(stages=stages))
        query.save(case/'search.json', dict(history=history+[dict(point=[2],raw=[.3,-2.])]*3))


@pytest.mark.parametrize('feasible_initial', [False, True])
def test_missing_point_history_and_baseline_keep_scores_and_unknowns(tmp_path, feasible_initial):
    saved_task(tmp_path, feasible_initial=feasible_initial)
    task, states, pairs, errors = query.audit_task('synthetic', 1, tmp_path, 1., [query.POINT])
    assert task['unavailable_arms'] == [query.POINT]
    assert task['quality_baseline_defined'] is feasible_initial
    assert len(states)==6 and len(pairs)==3 and errors==[0.]*6
    assert all((p['left'],p['right'])==(query.FULL,query.AFTER) for p in pairs)
    assert all(s['choices']['actual']['selected']['new_good'] is None for s in states)
    assert all(s['choices']['actual']['selected']['truth'] is None for s in states)
    if not feasible_initial:
        assert task['best_initial_truth'] is None and task['shared_new_good'] is None
        assert all(s['good_candidates'] is None and s['shared_good_present'] is None for s in states)
        assert all(s['choices']['actual']['selected']['feasible_improving'] is None for s in states)
        stage=query.read(tmp_path/query.FULL/'query_audit.json')['stages'][0]
        known=query.point_record(1,stage,[(0,),(1,),(2,)],{(0,):0,(1,):1},
            {tuple(r['point']):r['truth'] for r in query.read(tmp_path/'truth.json')},None,1.)
        assert known['feasible_improving'] is None and known['true_objective_difference_to_initial'] is None
        assert known['objective_mean_difference_to_initial'] is None
        assert known['risk_mean_bias']==0. and known['objective_mean_bias']==0.
    else:
        assert all(len(s['good_candidates'])==1 for s in states)
    for row in states+pairs:
        row['replicate']=1
    aggregate = counts(states,pairs)
    first = aggregate['comparisons'][0]
    assert first['registered_task_noise_units']==1 and first['task_noise_units']==0
    assert first['unavailable_task_noise_units']==[dict(replicate=1,domain='synthetic',seed=1)]
    assert aggregate['choices'][0]['actual_selected_unknown_truth']==3
    # An unregistered missing arm remains a real error, never silently excluded.
    with pytest.raises(FileNotFoundError):
        query.audit_task('synthetic',1,tmp_path,1.)

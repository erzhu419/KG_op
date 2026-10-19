"""Catch inherited completed cohorts, stream mistakes and budget drift using metadata."""
from copy import deepcopy

import pytest

from performance import register_mainline_safe_unsampled_validation as reg
from performance import run_mainline_observation_seed_replication as replication


def plan_fixture():
    reference = reg.read(reg.ROOT/reg.REFERENCE)
    vectors = [dict(task_geometry_shift=[i/10000,0.,0.],task_geometry_radius_scale=1.) for i in range(24)]
    return reference, reg.make_plan(reference,vectors,'substitute-time')


def test_new_primary_preserves_frozen_controls_without_completed_or_old_policy_state(tmp_path, monkeypatch):
    reference, plan = plan_fixture()
    before = deepcopy(reference)
    for key in ('frozen','search_arms','terminal_arms','old_protocol','candidate_protocol'):
        assert plan[key] == reference[key]
    assert plan['status']=='registered_not_started' and not plan['execution_started']
    assert not {'completed_utc','summary_file','execution_started_utc','execution_failures'} & plan.keys()
    assert plan['primary_terminal']=='old' and (plan['primary_candidate'],plan['primary_reference'])==(reg.AFTER,reg.POINT)
    assert len(plan['tasks'])==24 and all('initial_policy_file' not in t for t in plan['tasks'])
    assert plan['budget']['maximum_target_physical_calls']==24*(10+6*3+9*368)
    assert plan['budget']['maximum_logical_target_calls_both_protocols']==2*24*(8*365+362)
    parent = reg.read(reg.ROOT/plan['source_manifest'])
    for family in parent['families']:
        for key in ('search_manifest','terminal_manifest'):
            family[key]=str(reg.ROOT/family[key])
    reg.save(tmp_path/'parent.json',parent)
    plan['source_manifest']=str(tmp_path/'parent.json')
    monkeypatch.setattr(replication,'ROOT',tmp_path)
    def forbidden(*args,**kwargs):
        raise AssertionError('Registration metadata cannot sample or run workers')
    monkeypatch.setattr(replication.subprocess,'run',forbidden)
    monkeypatch.setattr(reg.np.random,'default_rng',forbidden)
    prepared=replication.materialize(plan)
    for family in prepared['replicates'][0]['families']:
        source=reg.read(tmp_path/family['search_manifest'])
        terminal=reg.read(tmp_path/family['terminal_manifest'])
        assert len(source['target_seeds'])==8
        assert 'fixed_initialization_file_by_task' not in source and 'fixed_neighbors_file_by_task' not in source
        for seed in source['target_seeds']:
            assert source['observation_seed_by_task'][str(seed)]==seed+100
            assert terminal['verification_seed_by_task'][str(seed)]==seed+200
    assert reference==before
    assert not list(tmp_path.rglob('search.json')) and not list(tmp_path.rglob('truth.json'))


def test_reject_seed_or_geometry_reuse_before_observation():
    reference, plan=plan_fixture()
    reg.check_freshness(plan,[reference])
    with pytest.raises(AssertionError,match='seed collision'):
        reg.check_freshness(plan,[dict(observation_seed=2026107101)])
    with pytest.raises(AssertionError):
        reg.check_freshness(plan,[plan['tasks'][0]['geometry']])

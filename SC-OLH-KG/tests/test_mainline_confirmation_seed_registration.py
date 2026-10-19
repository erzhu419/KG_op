"""Metadata routing checks; no task construction, observations or RNG."""
from copy import deepcopy

from performance.register_mainline_confirmation_seed_replication import plans


def reference():
    return dict(name='reference',status='complete',output_dir='saved_stage65',
        source_manifest='frozen_parent.json',parent_manifest='development.json',
        geometry_seed=2026104000,completed_utc='old',summary_file='old/summary.json',
        tasks=[dict(domain=('Inventory','Queue','Factor')[i//8],
            geometry_task_seed=2026104001+i,optimizer_seed=2026104001+i,
            geometry=dict(task_id=str(i),problem_kwargs=dict(shift=[.01,.02,.03],scale=1.)))
            for i in range(24)],
        replicates=[dict(replicate=1,output_dir='saved_stage65/replicate_1')],
        frozen=dict(d=1000),randomness={},barriers={},analysis={},
        execution_preparation=dict(inherited_preflight='oldpreflight',
            inherited_seed_routing='oldrouting',inherited_reuse_checks={}),
        search_arms=['FULL','AFTER','POINT','OLD','GP','STRUCT','STOP'],
        terminal_arms=[dict(name=str(i)) for i in range(9)],
        historical_accounts=['old/completion.json'],budget=dict(geometry_count=24,
            maximum_verification_calls_per_cell=352,maximum_target_physical_calls=80160))


def test_two_streams_preserve_geometry_optimizer_policy_and_budget_identity():
    original = reference(); before = deepcopy(original)
    outer,children = plans(original,'registered-time')
    assert original==before
    assert len(outer['streams'])==len(children)==2
    assert outer['budget']['maximum_target_physical_calls']==160320
    for (_,child),stream in zip(children,outer['streams']):
        assert child['source_manifest']==original['source_manifest']
        assert child['budget']==original['budget']
        assert len(child['replicates'])==1 and len(child['tasks'])==24
        assert 'completed_utc' not in child and 'summary_file' not in child
        assert child['replicates'][0]['replicate']==stream['replicate']
        for task,old in zip(child['tasks'],original['tasks']):
            assert {k:task[k] for k in old}==old
            assert task['initial_policy_file'].startswith('saved_stage65/replicate_1/')
            assert task['shared_neighbors_file'].endswith('/neighbors.npz')


def test_new_observation_ids_never_replace_original_target_or_optimizer_seeds():
    _,children = plans(reference(),'registered-time')
    all_stream_seeds = []
    for replicate,(_,child) in enumerate(children,1):
        mapping = child['replicates'][0]['seed_by_task']
        assert list(mapping)==[str(2026104001+i) for i in range(24)]
        for index,row in enumerate(mapping.values(),1):
            base = 2026105000 if replicate==1 else 2026106000
            assert row==dict(observation_seed=base+100+index,verification_seed=base+200+index)
            all_stream_seeds.extend(row.values())
    assert len(set(all_stream_seeds))==96

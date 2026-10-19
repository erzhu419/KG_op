"""Two reachable risks: tie identity and accidental response-dependent pairing."""
from pathlib import Path

from performance import mainline_source_neighbor_geometry as geometry


def test_nearest_excludes_self_breaks_ties_by_held_order_and_deduplicates():
    # Two distinct policies coincide in this public representation; keep the edge.
    coordinates = [[0.] * 10, [0.] * 10, [1.] * 10]
    result = geometry.nearest_pairs(coordinates, [90, 20, 10])
    assert [(p['i'], p['j']) for p in result['pairs']] == [(0, 1), (0, 2)]
    assert result['pairs'][0]['selected_from'] == [0, 1]
    assert result['pairs'][0]['policy_ids'] == [90, 20]
    assert result['pairs'][0]['distance_squared'] == 0.
    assert result['nodes'][2]['nearest_tied_positions'] == [0, 1]
    assert result['nodes'][2]['nearest_j'] == 0
    assert all(node['i'] != node['nearest_j'] for node in result['nodes'])


def test_response_changes_cannot_change_pairs_or_registered_prefix_membership(monkeypatch):
    task = dict(name='family', X_integer=[[i] * 50 for i in range(16)],
        Y_replicates=[[[1., 2.]]] * 16, constraint_sigma=[3.] * 16)
    archive = dict(tasks=[task])
    source = dict(source_archives=[dict(heldout_target_family='heldout', path='archive.json',
        source_families=['family'])], folds=1, output_dir='source')
    reads = []
    def read(path):
        name = str(Path(path).relative_to('/repo'))
        reads.append(name)
        return {'manifest.json': source, 'archive.json': archive,
            'source/heldout/fold_0/split.json': dict(held=list(range(16)))}[name]
    monkeypatch.setattr(geometry, 'read', read)
    plan = dict(inputs=dict(source_manifest='manifest.json'))
    before = geometry.build_fixed_pairs('/repo', plan)
    task['Y_replicates'] = [[[9999., -9999.]]] * 16
    task['constraint_sigma'] = [-9999.] * 16
    after = geometry.build_fixed_pairs('/repo', plan)
    assert before == after
    group = before['groups'][0]
    assert group['nodes'][10]['observed'] == {'10': False, '13': True}
    assert group['nodes'][13]['observed'] == {'10': False, '13': False}
    assert all('forecast' not in path for path in reads)
    assert before['response_fields_used'] is False

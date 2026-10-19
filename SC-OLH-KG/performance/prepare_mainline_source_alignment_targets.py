"""Future one-time target metadata materialization from the sealed source cohort.

Importing this module or validating its substitutes does not draw the registered
geometries. The explicit register action prepares metadata, not target execution.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import socket

import numpy as np

from performance import run_mainline_observation_seed_replication as replication
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save

CONTROL = 'performance/manifests/mainline_source_alignment_control_v1.json'
MANIFEST = 'performance/manifests/mainline_source_alignment_targets_v1.json'
OUTPUT = 'paper_artifacts/mainline_source_alignment_control_v1'
SEAL = f'{OUTPUT}/source_preparation/source_asset_seal.json'
NAME = 'mainline_source_alignment_targets_v1'


def budget_for(n=144, streams=2):
    units = n * streams
    return dict(geometry_count=n, observation_replicates=streams, task_stream_units=units,
        unique_search_histories=9 * units, online_search_histories=8 * units,
        registered_terminal_cells=11 * units, terminal_cells=11 * units,
        terminal_cells_per_protocol=11 * units,
        maximum_verification_calls_per_cell=352, verification_cap_per_cell=352,
        reachable_successful_verification_calls_per_cell=336,
        online_logical_cap=365, initial_only_logical_cap=362,
        initial_physical_calls=10 * units, online_search_physical_calls=24 * units,
        expanded_logical_search_calls=140 * units,
        maximum_verification_physical_calls=11 * 352 * units,
        maximum_target_physical_calls=(34 + 11 * 352) * units,
        reachable_successful_target_physical_calls=(34 + 11 * 336) * units,
        maximum_expanded_logical_target_calls=(10 * 365 + 362) * units,
        new_source_calls=0, new_source_fits=0)


def require_sources(control, seal, root=ROOT):
    """Check the six sealed paths and metadata without deserializing models."""
    assert seal['status'] == 'sealed_sources_only' and seal['source_models_ready']
    assert seal['completed_fit_pipelines'] == 6 and seal['source_permutations_generated'] == 3
    assert not seal['failures'] and not seal['unknown_attempt_costs']
    assert seal['source_fitting_recipes_unchanged']
    families = {row['family']: row for row in seal['families']}
    assert set(families) == {row['domain'] for row in control['families']}
    for family in control['families']:
        record = families[family['domain']]
        assert record['permutation_seed'] == family['source_permutation_seed']
        models = {row['kind']: row for row in record['models']}
        assert set(models) == {'structural', 'native'}
        for kind, key in (('structural', 'source_models_file'), ('native', 'native_source_prior_file')):
            model = models[kind]
            assert model['receipt_status'] == 'complete'
            assert model['model_file'] == family['source_permuted_assets'][key]
            stat = (Path(root) / model['model_file']).stat()
            assert stat.st_size == model['size_bytes'] and stat.st_mtime_ns == model['mtime_ns'], model['model_file']
    return deepcopy(seal)


def make_plans(control, reference, geometries, registered_utc):
    """Pure construction: substitute geometries can exercise all 3168 cells."""
    assert len(geometries) == control['cohort']['geometry_count'] == 144
    assert reference['terminal_arms'] == control['preserved_terminal_arms']
    assert control['search_arms'][:7] == reference['search_arms']
    assert len(control['search_arms']) == 9 and len(control['terminal_arms']) == 11
    tasks = []
    base = control['cohort']['geometry_optimizer_seed_base']
    for index, geometry in enumerate(geometries, 1):
        family = next(f for f in control['families']
            if f['global_geometry_index_first'] <= index <= f['global_geometry_index_last'])
        assert len(geometry['task_geometry_shift']) == 3
        assert all(-.035 <= x <= .035 for x in geometry['task_geometry_shift'])
        assert .95 <= geometry['task_geometry_radius_scale'] <= 1.05
        tasks.append(dict(domain=family['domain'], geometry_task_seed=base + index,
            optimizer_seed=base + index, source_search_manifest=family['source_search_manifest'],
            geometry=dict(task_id=f"source_alignment_{family['domain']}_{index:04d}",
                problem_kwargs=deepcopy(geometry))))
    frozen = deepcopy(reference['frozen'])
    frozen['initial_policies'] = 'Unchanged source-only design80; same physical initial10 across streams, with independent stream observations shared across its nine histories.'
    frozen['source_models'] = 'Original family assets for frontend and nine controls; six sealed permuted assets only for the two added role backends.'
    barriers = dict(before_sampling='Explicit target admission after substitute checks and one-time metadata registration.',
        before_online='All 288 stream-specific initializations and banks complete or retained failures.',
        before_verification='All 2592 histories and all 3168 terminal inputs frozen or explicit missing/failed.',
        before_scoring='All 3168 old-terminal cells complete or failed/missing and globally sealed before complete-pool truth access.',
        failure='Preserve paid prefixes and fees. No automatic retry, replacement geometry, refit or redraw.')
    common = dict(source_manifest=reference['source_manifest'], source_control_manifest=CONTROL,
        source_asset_seal_file=SEAL, source_warning_document='docs/mainline_source_alignment_fit_results_20261018.md',
        reference_manifest=control['reference_manifest'], protocol_document=control['protocol_document'],
        status='registered_not_started', registered_utc=registered_utc, execution_started=False,
        target_execution_started=False, target_execution_authorized=False,
        target_execution_preparation=dict(status='pending', independent_scientific_execution=False),
        tasks=tasks, selected_N=144, search_arms=deepcopy(control['search_arms']),
        terminal_arms=deepcopy(control['terminal_arms']), old_protocol=control['old_protocol'],
        terminal_protocols=['old'], frozen=frozen, randomness=deepcopy(reference['randomness']),
        geometry_seed=control['cohort']['geometry_seed'], geometry_description=control['cohort']['geometry_generator'],
        objective=control['hypothesis'], statistical_unit='144 geometries, two paired observation streams within each geometry.',
        barriers=barriers, analysis=deepcopy(control['analysis']), saved_score_contract=deepcopy(control['saved_score_contract']),
        source_permuted_assets_by_family={f['domain']: deepcopy(f['source_permuted_assets']) for f in control['families']},
        historical_accounts=list(dict.fromkeys([*reference['historical_accounts'],
            f"{reference['output_dir']}/completion.json", *control['historical_accounts'],
            *[f'{OUTPUT}/cluster_control/stage{s}_accounting.json' for s in (90, 91, 92)]])))
    children, streams = [], []
    for number, stream in enumerate(control['cohort']['streams'], 1):
        name = f'mainline_source_alignment_stream_{number}_v1'
        path, output = f'performance/manifests/{name}.json', f'{OUTPUT}/stream_{number}'
        child = deepcopy(common)
        child.update(name=name, output_dir=output, parent_manifest=MANIFEST,
            observation_replicate=number, budget=budget_for(streams=1),
            replicates=[dict(replicate=number, output_dir=f'{output}/replicate_{number}',
                seed_by_task={str(t['geometry_task_seed']): dict(
                    observation_seed=stream['observation_seed_base'] + i,
                    verification_seed=stream['verification_seed_base'] + i)
                    for i, t in enumerate(tasks, 1)})])
        children.append((path, child))
        streams.append(dict(name=stream['name'], replicate=number, manifest=path))
    plan = deepcopy(common)
    plan.update(name=NAME, output_dir=OUTPUT, streams=streams, budget=budget_for())
    return plan, children


def materialize(child, root=ROOT):
    """Reuse family recipes; add isolated source paths to the new family plans."""
    assert Path(root) == Path(replication.ROOT)
    prepared = replication.materialize(child)
    for replica in prepared['replicates']:
        for family in replica['families']:
            path = Path(root) / family['search_manifest']
            source = read(path)
            source['source_permuted_assets'] = deepcopy(child['source_permuted_assets_by_family'][family['domain']])
            save(path, source)
    return prepared


def check_freshness(plan, children, documents):
    proposed = {t['geometry_task_seed'] for t in plan['tasks']}
    streams = [v for _, child in children
        for row in child['replicates'][0]['seed_by_task'].values() for v in row.values()]
    assert len(streams) == len(set(streams)) == 576 and not proposed.intersection(streams)
    proposed.update(streams)
    geometries = [t['geometry']['problem_kwargs'] for t in plan['tasks']]
    assert all(g not in geometries[:i] for i, g in enumerate(geometries)), 'Duplicate geometry'

    def visit(value, seed_field=False):
        if isinstance(value, dict):
            if 'task_geometry_shift' in value and 'task_geometry_radius_scale' in value:
                geometry = {k: value[k] for k in ('task_geometry_shift', 'task_geometry_radius_scale')}
                assert geometry not in geometries, 'Registered geometry collision'
            for key, item in value.items():
                if seed_field and str(key).isdigit():
                    assert int(key) not in proposed, 'Registered seed collision'
                visit(item, seed_field or 'seed' in str(key).lower())
        elif isinstance(value, list):
            for item in value:
                visit(item, seed_field)
        elif seed_field and type(value) is int:
            assert value not in proposed, 'Registered seed collision'
    for document in documents:
        visit(document)


def register(checks, root=ROOT):
    """Explicit future node action; never called by the Stage92 check runner."""
    root = Path(root)
    assert socket.gethostname().split('.')[0] in {f'node{i:03d}' for i in range(1, 7)}
    assert read(root / checks)['status'] == 'passed', 'Target substitute checks must pass'
    receipt = root / OUTPUT / 'target_registration.json'
    if receipt.exists():
        return read(receipt)
    assert not (root / MANIFEST).exists(), 'Preserve partial registration; do not redraw'
    registration_dir = root / OUTPUT / 'target_registration'
    assert not (registration_dir / 'started.json').exists(), 'Preserve interrupted registration; do not redraw'
    control = read(root / CONTROL)
    require_sources(control, read(root / SEAL), root)
    with attempt(registration_dir, 'register-source-alignment-targets') as cost:
        save(registration_dir / 'started.json', dict(status='started', new_target_calls=0))
        cost.update(new_target_initial_calls=0, new_target_search_calls=0, new_verification_calls=0,
            new_source_fits=0, new_source_calls=0, new_truth_function_calls=0,
            source_model_restorations=0, posterior_restorations=0, new_candidate_scores=0)
        rng = np.random.Generator(np.random.PCG64(control['cohort']['geometry_seed']))
        geometries = [dict(task_geometry_shift=rng.uniform(-.035, .035, 3).tolist(),
            task_geometry_radius_scale=float(rng.uniform(.95, 1.05))) for _ in range(144)]
        cost['geometry_uniform_draws'] = 576
        plan, children = make_plans(control, read(root / control['reference_manifest']), geometries,
            datetime.now(timezone.utc).isoformat())
        check_freshness(plan, children, [read(p) for p in (root / 'performance/manifests').glob('*.json')
            if p != root / CONTROL])
        for path, child in children:
            save(root / path, child)
            materialize(child, root)
        save(root / MANIFEST, plan)
        result = dict(status='registered_not_started', manifest=MANIFEST,
            registered_geometries=144, streams=2, terminal_cells=3168,
            geometry_uniform_draws=576, target_execution_started=False, target_execution_authorized=False,
            source_asset_seal_file=SEAL, new_target_calls=0)
        save(receipt, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register'])
    parser.add_argument('--checks', required=True)
    args = parser.parse_args()
    print(register(args.checks), flush=True)

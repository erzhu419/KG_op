"""Register two new streams on the fixed Stage73 geometries; metadata only."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from performance import run_mainline_observation_seed_replication as replication
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save


NAME = 'mainline_terminal_recording_seed_replication_v1'
MANIFEST = ROOT / f'performance/manifests/{NAME}.json'
REFERENCE = 'performance/manifests/mainline_safe_unsampled_validation_v1.json'
PROTOCOL = 'docs/mainline_terminal_recording_seed_replication_protocol_20261017.md'
AFTER = 'legacy_role_after_init_point_only'
POINT = 'legacy_role_point_only'
ZERO = dict(new_target_initial_calls=0, new_target_search_calls=0, new_verification_calls=0,
    new_source_calls=0, new_source_fits=0, new_truth_function_calls=0,
    source_model_restorations=0, engine_restorations=0, posterior_restorations=0,
    new_rng_draws=0, new_raw_replays=0, geometry_uniform_draws=0, download_bytes=0)


def plans(reference, registered_utc):
    """Copy physical identities and frozen rules, replacing only registered streams."""
    assert len(reference['tasks']) == 24
    assert {t['geometry_task_seed'] for t in reference['tasks']} == set(range(2026107001, 2026107025))
    assert all(t['optimizer_seed'] == t['geometry_task_seed'] for t in reference['tasks'])
    assert len(reference['terminal_arms']) == 9 and len(reference['search_arms']) == 7
    directory = f'paper_artifacts/{NAME}'
    initial_source = Path(reference['replicates'][0]['output_dir'])
    tasks = deepcopy(reference['tasks'])
    for task in tasks:
        source = initial_source / task['domain'] / 'search' / f"seed_{task['geometry_task_seed']}"
        task.update(initial_policy_file=str(source/'initialization.json'),
            shared_neighbors_file=str(source/'neighbors.npz'))
    frozen = deepcopy(reference['frozen'])
    frozen.update(initial_design_seed=80,
        initial_policies='Reuse exact Stage73 initial10 physical policies; discard old raw values and generate10 fresh observations shared by the7 histories.',
        shared_neighbors='Reuse the exact Stage73 shared64 physical bank and metadata, without drawing a new bank.')
    frozen['new_terminal'].update(
        independence='Use the frozen comparison mapping: block0 SeedSequence([verification_seed, replication, TERMINAL_OBJECTIVE_COMPARISON_STREAM_TAG]); block1 appends1. Search and safety use their distinct namespaces; common random numbers are shared only within each compared pair.',
        reuse='Use the existing core.terminal_two_comparison verifier and saved-block reuse implementation, including128-to120 safety prefixes and independent objective blocks0/1; retain the frozen old verifier as the default reference.')
    randomness = deepcopy(reference['randomness'])
    randomness['optimizer_seed'] = 'Keep geometry_task_seed and optimizer_seed2026107001..2026107024 as config.seed in both streams. Observation seeds do not replace optimizer seeds.'
    barriers = dict(
        before_sampling='Freeze both stream maps, source references, physical initial10/shared64 files and passive terminal recording before any new observation.',
        before_online='Both streams must finish all48 shared initial histories and freeze all48 reused neighbor banks before any adaptive search.',
        before_verification='Both streams must seal all336 histories or retained failures and both protocols of all432 paired terminal inputs before terminal simulation.',
        before_scoring='Seal all864 old/new terminal outcomes or retained failures across both streams, then seal the complete saved action-pool physical inventory before any target-truth access or scoring.',
        failure='Retain every failed arm, paid prefix, partial audit and CPU; no automatic resampling, seed replacement, task dropping or numerical model patch.')
    analysis = dict(primary='Old/default terminal AFTER-minus-POINT C and signedG: form each stream effect, then average the two new-stream effects within each of24 fixed geometries before between-geometry summaries.',
        metrics=deepcopy(reference['analysis']['metrics']),
        secondary='New-terminal AFTER-minus-POINT; FULL-minus-AFTER; all9 old/new controls, GP, OLD, STRUCT, STOP and native crosses. Paired new-minus-old terminal effects remain secondary.',
        fixed_size='Exactly24 frozen geometries and2 new observation/verification streams; no interim scoring, optional expansion, filtering or model/threshold/terminal changes.',
        concentration=reference['analysis']['concentration'],
        uncertainty='24 geometry units, not48 independent samples. Average the two streams within geometry before descriptive paired95%t intervals; also retain each stream separately.',
        missing='An undefined baseline or failed effect remains null. If either stream effect is null, that geometry average is null; report full24 and explicitly defined-subset denominators separately.',
        truth='Only after the864-outcome global seal, score every frozen action_ids candidate from all3 prefixes of FULL/AFTER/POINT, initial policies and all9-arm old/new terminal/deployment union. Reuse same-geometry exact-physical-policy deterministic truth; only missing fields require new analytic calls, accounted separately from simulation observations.',
        reference='Opened Stage73 outcomes are historical reference only and never enter the new primary mean or stream-average denominator.',
        interpretation='Discovery, shortlist inclusion, certification and comparison are distinct. A good omitted point is a loss candidate only relative to the actual deployed policy; true quality does not certify an unexecuted alternative.')
    audit = deepcopy(reference['mechanism_audit'])
    audit['execution_gap'] = 'A future execution adapter must enforce the two-stream864-outcome seal before all-pool scoring. Registration and family materialization do not run experiments.'
    audit['terminal_recording'] = 'Save already computed final observed-pool components and ranking, objective gate, safe fallback, pre/post incumbent and same-policy transformation, preserving physical points and repeated observation indices. No extra prediction, restoration, ranking or random draw.'
    statistical_unit = '24 fixed Stage73 geometries. The two new-stream paired effects are averaged within each geometry; policies, prefixes, streams and terminal cells are not independent geometry samples.'
    historical = [*reference['historical_accounts'], f"{reference['output_dir']}/completion.json",
        'paper_artifacts/mainline_observed_terminal_attribution_v1/completion.json']
    children, streams = [], []
    for replicate, obs_base, verify_base in ((1,2026108100,2026108200), (2,2026109100,2026109200)):
        name = f'mainline_terminal_recording_seed_replication_stream_{replicate}_v1'
        path = f'performance/manifests/{name}.json'
        output = f'{directory}/stream_{replicate}'
        child = {key:deepcopy(reference[key]) for key in (
            'source_manifest', 'search_arms', 'terminal_arms', 'old_protocol', 'candidate_protocol', 'budget')}
        child.update(name=name, status='registered_not_started', execution_started=False,
            registered_utc=registered_utc, parent_manifest=REFERENCE, reference_manifest=REFERENCE,
            output_dir=output, protocol_document=PROTOCOL, tasks=deepcopy(tasks),
            geometry_seed=reference['geometry_seed'], observation_replicate=replicate,
            replicates=[dict(replicate=replicate, output_dir=f'{output}/replicate_{replicate}',
                seed_by_task={str(t['geometry_task_seed']):dict(
                    observation_seed=obs_base+t['geometry_task_seed']-2026107000,
                    verification_seed=verify_base+t['geometry_task_seed']-2026107000) for t in tasks})],
            objective='Replicate frozen AFTER cross-policy propagation versus POINT under a fresh observation/verification stream, while passively saving the complete existing terminal selection process.',
            scope='One new stream on the same24 Stage73 geometries; it is one half of the registered two-stream replication, not24 new geometries.',
            geometry_description='Exact Stage73 geometry vectors and optimizer seeds; exact initial10 and shared64 physical files. No geometry, initialization-policy or neighbor-bank draw.',
            primary_candidate=AFTER, primary_reference=POINT, primary_terminal='old',
            frozen=deepcopy(frozen), randomness=deepcopy(randomness), barriers=deepcopy(barriers),
            analysis=deepcopy(analysis), mechanism_audit=deepcopy(audit),
            statistical_unit=statistical_unit, historical_accounts=deepcopy(historical),
            new_target_calls_at_registration=0, truth_used_at_registration=False,
            execution_preparation=dict(status='pending_global_two_stream_adapter', workers=None,
                entrypoint=None, metadata_materializer='performance.run_mainline_observation_seed_replication.materialize',
                execution_policy='scheduleurm submit-cpu-batch node001-node006; CPU-only vram=0; BLAS threads1.',
                inherited_preflight=deepcopy(reference['execution_preparation']['inherited_preflight']),
                inherited_seed_routing=deepcopy(reference['execution_preparation']['inherited_seed_routing']),
                inherited_reuse_checks=deepcopy(reference['execution_preparation']['inherited_reuse_checks']),
                remaining='Connect and validate a global two-stream execution/scoring adapter before any sampling; child-only run/finalize is not an admitted execution path.'))
        children.append((path,child))
        streams.append(dict(name=f'observation_{replicate}', replicate=replicate, manifest=path))
    budget = deepcopy(reference['budget'])
    for key in ('unique_search_histories', 'terminal_cells_per_protocol', 'registered_terminal_cells',
                'initial_physical_calls', 'online_search_physical_calls',
                'maximum_verification_logical_calls_per_protocol', 'expanded_logical_search_calls_per_protocol',
                'maximum_logical_target_calls_both_protocols', 'maximum_verification_physical_calls',
                'maximum_target_physical_calls'):
        budget[key] *= 2
    budget.update(observation_replicates=2, task_stream_units=48)
    assert budget['maximum_target_physical_calls'] == 160320
    assert budget['maximum_logical_target_calls_both_protocols'] == 315072
    outer = dict(name=NAME, status='registered_not_started', execution_started=False,
        registered_utc=registered_utc, reference_manifest=REFERENCE, output_dir=directory,
        protocol_document=PROTOCOL, tasks=tasks, streams=streams,
        search_arms=deepcopy(reference['search_arms']), terminal_arms=deepcopy(reference['terminal_arms']),
        objective='Two-stream confirmation of frozen old/default AFTER-minus-POINT C and signedG with passive full final-pool terminal evidence.',
        scope='Two fresh observation/verification streams on24 fixed Stage73 geometries,8per family; no new geometry or model selection.',
        primary_candidate=AFTER, primary_reference=POINT, primary_terminal='old',
        statistical_unit=statistical_unit, frozen=frozen, randomness=randomness, budget=budget,
        barriers=barriers, analysis=analysis, mechanism_audit=audit, historical_accounts=historical,
        execution_preparation=dict(status='pending_global_two_stream_adapter', execution_started=False,
            remaining='Only metadata is registered/materialized. Global execution/scoring adapter admission remains pending.'),
        new_target_calls_at_registration=0, truth_used_at_registration=False)
    return outer, children


def check_seed_collisions(children, existing_documents):
    proposed = [value for _,child in children for row in child['replicates'][0]['seed_by_task'].values()
                for value in row.values()]
    assert len(proposed) == len(set(proposed)) == 96
    proposed = set(proposed)

    def visit(value, seed_field=False):
        if isinstance(value, dict):
            for key,item in value.items():
                visit(item, seed_field or 'seed' in key.lower())
        elif isinstance(value, list):
            for item in value:
                visit(item, seed_field)
        elif seed_field and type(value) is int:
            assert value not in proposed, f'Observation/verification seed collision: {value}'
    for document in existing_documents:
        visit(document)


def register():
    output = ROOT / f'paper_artifacts/{NAME}'
    with attempt(output/'registration', 'register-terminal-recording-seed-replication') as cost:
        cost.update(ZERO)
        assert not MANIFEST.exists(), 'Preserve the existing registration'
        reference = read(ROOT/REFERENCE)
        assert reference['status'] == 'complete'
        outer, children = plans(reference, datetime.now(timezone.utc).isoformat())
        own = {MANIFEST, *(ROOT/path for path,_ in children)}
        assert all(not path.exists() for path in own), 'Preserve existing child registrations'
        check_seed_collisions(children, (read(path) for path in
            (ROOT/'performance/manifests').glob('mainline*.json') if path not in own))
        assets = {t[field] for t in outer['tasks'] for field in ('initial_policy_file','shared_neighbors_file')}
        assets.update(str(Path(t['shared_neighbors_file']).with_suffix('.json')) for t in outer['tasks'])
        parent = read(ROOT/reference['source_manifest'])
        for family in parent['families']:
            source = read(ROOT/family['search_manifest'])
            read(ROOT/family['terminal_manifest'])
            assets.update(value for key,value in source['frozen_assets'].items() if key.endswith('_file'))
        for path in assets:
            assert (ROOT/path).is_file(), f'Required frozen asset absent: {path}'
        for path,child in children:
            save(ROOT/path,child)
        save(MANIFEST,outer)
        prepared = [replication.materialize(child) for _,child in children]
        family_plans = [family for group in prepared for replica in group['replicates'] for family in replica['families']]
        assert len(family_plans) == 6
        save(output/'registration'/'result.json', dict(status='registered_not_started',
            manifest=str(MANIFEST.relative_to(ROOT)), children=[path for path,_ in children],
            family_plans=family_plans, checked_asset_files=len(assets), new_stream_seeds=96,
            execution_started=False, models_loaded=0, truth_read=False, **ZERO))
    return outer


if __name__ == '__main__':
    print(register()['status'])

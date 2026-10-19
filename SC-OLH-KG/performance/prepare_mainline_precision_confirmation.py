"""Register a precision-selected fresh cohort using metadata only; never run it."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import socket

import numpy as np

from performance import run_mainline_observation_seed_replication as replication
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save


NAME = 'mainline_precision_confirmation_v1'
MANIFEST = f'performance/manifests/{NAME}.json'
OUTPUT = f'paper_artifacts/{NAME}'
REFERENCE = 'performance/manifests/mainline_safe_unsampled_validation_v1.json'
PROTOCOL = 'docs/mainline_precision_confirmation_protocol_20261018.md'
CANDIDATE_N = (24, 48, 72, 96, 120, 144, 192, 288, 384)
GEOMETRY_SEED = 2026120000
STREAM_BASES = ((2026130000, 2026140000), (2026150000, 2026160000))
AFTER, POINT = 'legacy_role_after_init_point_only', 'legacy_role_point_only'
ZERO = dict(new_target_initial_calls=0, new_target_search_calls=0,
    new_verification_calls=0, new_source_calls=0, new_source_fits=0,
    new_truth_function_calls=0, source_model_restorations=0, engine_restorations=0,
    posterior_restorations=0, new_candidate_scores=0, new_raw_replays=0,
    simulation_rng_draws=0, download_bytes=0)


def budget_for(n, streams=2):
    """Separate the retained 352 cap from the old verifier's reachable 336 path."""
    units = n * streams
    return dict(geometry_count=n, observation_replicates=streams, task_stream_units=units,
        unique_search_histories=7 * units, registered_terminal_cells=9 * units,
        terminal_cells_per_protocol=9 * units, initial_physical_calls=10 * units,
        online_search_physical_calls=6 * 3 * units,
        maximum_verification_calls_per_cell=352, reachable_successful_verification_calls_per_cell=336,
        online_logical_cap=365, initial_only_logical_cap=362,
        expanded_logical_search_calls=units * (8 * 13 + 10),
        maximum_verification_physical_calls=9 * 352 * units,
        maximum_target_physical_calls=(10 + 6 * 3 + 9 * 352) * units,
        reachable_successful_verification_physical_calls=9 * 336 * units,
        reachable_successful_target_physical_calls=(10 + 6 * 3 + 9 * 336) * units,
        maximum_expanded_logical_target_calls=(8 * 365 + 362) * units,
        bound_basis='One shared initial10 per stream/geometry; six histories pay three adaptive calls; nine separate old terminals. In the unchanged old verifier, the two-certified comparison path costs224 and the third-candidate fallback path costs336; those paths are mutually exclusive. The protocol ceiling remains352. Failed paid attempts and explicitly admitted recovery are additional retained costs, not hidden inside a successful-run bound.',
        analytic_truth_function_call_bound='Three times the sealed distinct complete audit-policy inventory; no outcome-independent numeric inventory size is assumed. Charge actual missing analytic fields separately from simulation.',
        new_source_calls=0, new_source_fits=0, registration_target_calls=0)


def make_plans(reference, geometries, preparation_plan, precision_result, registered_utc):
    """Pure metadata construction with caller-supplied geometries for small tests."""
    n = precision_result['selected_N']
    assert type(n) is int and n in CANDIDATE_N and len(geometries) == n
    assert reference['status'] == 'complete'
    assert len(reference['search_arms']) == 7 and len(reference['terminal_arms']) == 9
    assert reference['frozen']['d'] == 1000 and reference['frozen']['search_calls'] == 13
    assert reference['frozen']['candidate_budgets'] == [80, 128, 128]
    domains = tuple(dict.fromkeys(t['domain'] for t in reference['tasks']))
    assert len(domains) == 3 and n % 3 == 0
    tasks, per_family = [], n // 3
    for index, geometry in enumerate(geometries, 1):
        shift = geometry['task_geometry_shift']
        assert len(shift) == 3 and all(-.035 <= x <= .035 for x in shift)
        assert .95 <= geometry['task_geometry_radius_scale'] <= 1.05
        domain, seed = domains[(index - 1) // per_family], GEOMETRY_SEED + index
        source = next(t['source_search_manifest'] for t in reference['tasks'] if t['domain'] == domain)
        tasks.append(dict(domain=domain, geometry_task_seed=seed, optimizer_seed=seed,
            source_search_manifest=source, geometry=dict(
                task_id=f'precision_confirmation_{domain}_{(index - 1) % per_family}',
                problem_kwargs=deepcopy(geometry))))
    frozen = deepcopy(reference['frozen'])
    frozen.pop('new_terminal')
    frozen.update(terminal='Unchanged old/default same_policy_distinct_challenger only; original shortlist and objective guard, safety80/128/128, comparison8 per policy, total cap352.',
        initial_policies='Use unchanged source-only design80 to construct the same physicalinitial10 in both streams of each geometry; each stream draws its own10 observations shared across its seven histories.',
        shared_neighbors='Each stream independently applies the unchanged shared64 generator to its own STOP engine; physically shared across arms within that stream. No bank or raw observation is copied from the other stream.')
    randomness = deepcopy(reference['randomness'])
    randomness.update(optimizer_seed='Same geometry_task_seed as config.seed in both streams; observation and verification seeds do not replace the optimizer seed.',
        cross_stream='Independent registered observation and verification namespaces. Same source-only initial physical design and geometry; each stream generates its own shared64 from its own STOP state, so realized banks and adaptive pools may differ.',
        planning_scope='Historical Stage76 streams reused a previously frozen shared64 bank. Their two-stream variance is a planning scenario, not a guarantee for fresh geometries whose banks follow their own STOP observations; the separate new-geometry single-stream planning batch is not divided by two.')
    barriers = dict(
        before_sampling='Register all geometry metadata and both seed maps, frozen sources and rules; admit a dedicated old-only two-stream scheduler adapter before any science.',
        before_online=f'All{2*n} stream-specific initial histories and shared64 banks are sealed or explicitly failed before any adaptive search. Initial physical policies must match across streams; raw observations and derived banks are not copied.',
        before_verification=f'Seal all{14*n} histories or retained failures, then all{18*n} old-terminal inputs or explicit missing records before verification.',
        before_scoring=f'Seal all{18*n} old-terminal outcomes or retained failures across both streams, then freeze the complete saved action-pool and terminal/deployment policy inventory before target truth access or scoring.',
        failure='Keep every failure, paid prefix and CPU. No automatic retry, geometry replacement, task dropping, interim scoring or parameter changes.')
    analysis = dict(primary='Old/default AFTER-minus-POINT C and signedG; average the two observation-stream paired effects within each geometry before between-geometry summaries.',
        metrics=deepcopy(reference['analysis']['metrics']),
        secondary='FULL-minus-AFTER; all nine controls including same-source GP, old backend, structural, STOP and both same-search native crosses under the same old terminal.',
        fixed_size=f'Exactly{n} new geometries, {per_family} per known family, two observation/verification streams; no optional expansion or outcome-based screening.',
        uncertainty='Geometry is the independent unit. Report two streams separately and within-geometry averages; no division of historical single-stream variance by two. Planning half-widths are not guaranteed realized precision or power.',
        missing='Retain all registered denominators. Missing baseline or failed effects remain null; report defined-subset statistics separately and never replace missing values by zero.',
        concentration=reference['analysis']['concentration'],
        decomposition='Report deltaG=deltaP-deltaL, with positive_gain_difference and avoided_POINT_loss signed separately; positive AFTER quality alone is not an AFTER-minus-POINT gain.',
        interpretation='Keep historical batches separate from this cohort. Same winner, same discovery branch and repeat sampling are distinct; discovery mechanisms require saved actual-returned observations, and repeat counts do not establish causal necessity.')
    audit = deepcopy(reference['mechanism_audit'])
    audit.update(population='Every saved action_ids candidate in all three prefixes of AFTER, POINT and FULL, initial10 and all nine old-terminal shortlists/deployments; exact physical deduplication within geometry. All other controls retain their complete outcomes even where saved mechanism evidence is unavailable.',
        ranking='Use existing complete-pool saved mean and sampled scores after the global seal; no new posterior restoration, prediction or counterfactual score substitutions.',
        classification='Reuse the frozen Stage83 first-actual-returned-query classes: multiple safe unobserved competition, single safe unobserved competition, minimum violation, initial, other, unknown and unavailable saved path; repeat sampling is an orthogonal tag.',
        attribution_limit='Only the first divergence with identical saved history and pool supports same-state propagation comparison. Later differences are trajectory associations; no causal mediation or unseen alternative-history reward.',
        terminal_recording='Passively retain already computed final observed-pool components, ranking, shortlist construction and observation indices; no extra candidate scores or RNG draws.',
        execution_gap='Dedicated old-only two-stream execution and complete-pool scoring adapter remains unimplemented and unadmitted. Historical old/new runners are not execution entrypoints for this registration.')
    execution = dict(status='pending_old_only_two_stream_adapter', execution_started=False,
        independent_scientific_execution=False, entrypoint=None, workers=None,
        metadata_materializer='performance.run_mainline_observation_seed_replication.materialize',
        execution_policy='scheduleurm submit-cpu-batch explicit node001-node006, vram0, BLAS1; size workers from live free physical cores.',
        remaining='Registration only. Connect and check the dedicated old-only two-stream global barriers and post-seal full-pool audit before any scientific execution.')
    preparation_manifest = f"performance/manifests/{preparation_plan['name']}.json"
    common = dict(source_manifest=reference['source_manifest'], reference_manifest=REFERENCE,
        preparation_manifest=preparation_manifest,
        protocol_document=preparation_plan.get('protocol_document', PROTOCOL),
        status='registered_not_started', execution_started=False, registered_utc=registered_utc,
        search_arms=deepcopy(reference['search_arms']), terminal_arms=deepcopy(reference['terminal_arms']),
        old_protocol=reference['old_protocol'], terminal_protocols=['old'],
        geometry_seed=GEOMETRY_SEED, tasks=tasks, frozen=frozen, randomness=randomness,
        primary_candidate=AFTER, primary_reference=POINT, primary_terminal='old',
        objective='Independently confirm whether frozen prior cross-policy propagation improves actual deployment under the unchanged old terminal.',
        scope='New known-family geometries with two independent observation/verification streams. The original shared-neighbor generation rule is frozen; new realized banks depend on each stream own STOP state.',
        geometry_description=f'RNG{GEOMETRY_SEED}; original family order; {per_family} successive shift U(-.035,.035)^3 and radius U(.95,1.05) draws per family. No geometry screening, replacement or truth access.',
        statistical_unit=f'{n} geometry units; average paired stream effects within each geometry. Streams, histories, policies and terminal cells are not independent geometry units.',
        barriers=barriers, analysis=analysis, mechanism_audit=audit,
        execution_preparation=execution, new_target_calls_at_registration=0,
        truth_used_at_registration=False,
        historical_accounts=[*reference['historical_accounts'], f"{reference['output_dir']}/completion.json",
            'paper_artifacts/mainline_terminal_recording_seed_replication_v1/completion.json',
            'paper_artifacts/mainline_reference_coverage_confirmation_v1/completion.json',
            'paper_artifacts/mainline_pathway_replication_v1/completion.json',
            f"{preparation_plan['output_dir']}/completion.json"])
    children, streams = [], []
    for number, (observation_base, verification_base) in enumerate(STREAM_BASES, 1):
        name = f'mainline_precision_confirmation_stream_{number}_v1'
        path, output = f'performance/manifests/{name}.json', f'{OUTPUT}/stream_{number}'
        child = deepcopy(common)
        child.update(name=name, output_dir=output, parent_manifest=MANIFEST,
            observation_replicate=number, budget=budget_for(n, streams=1),
            replicates=[dict(replicate=number, output_dir=f'{output}/replicate_{number}',
                seed_by_task={str(t['geometry_task_seed']): dict(
                    observation_seed=observation_base + i,
                    verification_seed=verification_base + i) for i, t in enumerate(tasks, 1)})])
        children.append((path, child))
        streams.append(dict(name=f'observation_{number}', replicate=number, manifest=path))
    outer = deepcopy(common)
    outer.update(name=NAME, output_dir=OUTPUT, streams=streams, budget=budget_for(n),
        precision_selection=deepcopy(precision_result), selected_N=n)
    return outer, children


def check_freshness(plan, children, existing_documents):
    proposed = {GEOMETRY_SEED, *(t['geometry_task_seed'] for t in plan['tasks'])}
    stream_seeds = [v for _, child in children
        for row in child['replicates'][0]['seed_by_task'].values() for v in row.values()]
    assert len(stream_seeds) == len(set(stream_seeds)) == 4 * len(plan['tasks'])
    assert not proposed.intersection(stream_seeds)
    proposed.update(stream_seeds)
    geometries = [t['geometry']['problem_kwargs'] for t in plan['tasks']]
    assert all(g not in geometries[:i] for i, g in enumerate(geometries)), 'Duplicate proposed geometry'

    def visit(value, seed_field=False):
        if isinstance(value, dict):
            if 'task_geometry_shift' in value and 'task_geometry_radius_scale' in value:
                geometry = {key: value[key] for key in ('task_geometry_shift', 'task_geometry_radius_scale')}
                assert geometry not in geometries, 'Registered geometry collision'
            for key, item in value.items():
                if seed_field and str(key).isdigit():
                    assert int(key) not in proposed, f'Registered seed collision: {key}'
                visit(item, seed_field or 'seed' in str(key).lower())
        elif isinstance(value, list):
            for item in value:
                visit(item, seed_field)
        elif seed_field and type(value) is int:
            assert value not in proposed, f'Registered seed collision: {value}'
    for document in existing_documents:
        visit(document)


def validate_metadata(plan, children, prepared, root=ROOT):
    """Check real source keys and the old-only family manifests, without models."""
    root = Path(root)
    n, assets = len(plan['tasks']), set()
    assert len(prepared) == 2 and plan['budget'] == budget_for(n)
    assert plan['terminal_protocols'] == ['old'] and not plan['execution_started']
    for (_, child), group in zip(children, prepared, strict=True):
        assert len(group['replicates']) == 1
        assert len(group['replicates'][0]['families']) == 3
        mapping = child['replicates'][0]['seed_by_task']
        for family in group['replicates'][0]['families']:
            source = read(root / family['search_manifest'])
            terminal = read(root / family['terminal_manifest'])
            original = read(root / next(t['source_search_manifest'] for t in plan['tasks'] if t['domain'] == family['domain']))
            for key in ('task', 'anchor', 'frozen_assets', 'neighbor_count', 'neighbor_radius',
                        'neighbor_safe_margin_scale', 'frontends', 'role_prior', 'point_only', 'after_init_point_only'):
                assert source[key] == original[key], key
            assert len(source['target_seeds']) == n // 3
            assert source['arms'] == [dict(name=a, online=a != 'initialization_only') for a in plan['search_arms']]
            assert 'fixed_initialization_file_by_task' not in source and 'fixed_neighbors_file_by_task' not in source
            assert terminal['arms'] == plan['terminal_arms']
            assert terminal['protocols'] == {plan['old_protocol']: [80, 128, 128]}
            for seed in source['target_seeds']:
                assert source['observation_seed_by_task'][str(seed)] == mapping[str(seed)]['observation_seed']
                assert terminal['verification_seed_by_task'][str(seed)] == mapping[str(seed)]['verification_seed']
            assets.update(value for key, value in source['frozen_assets'].items() if key.endswith('_file'))
            assets.add(terminal['inherited_replay_check_file'])
    for path in assets:
        assert (root / path).is_file(), f'Required frozen asset absent: {path}'
    return dict(checked_asset_files=len(assets), family_plans=6, registered_geometries=n,
        search_histories=14 * n, terminal_cells=18 * n, independent_scientific_execution=False)


def register(preparation_plan, precision_result, checks, root=ROOT):
    """Node-only registration after the frozen precision rule and substitute checks."""
    root = Path(root)
    node = socket.gethostname().split('.')[0]
    assert node in {f'node{i:03d}' for i in range(1, 7)}, node
    assert read(root / checks)['status'] == 'passed', 'Substitute checks must pass before registration'
    n = precision_result['selected_N']
    if n is None:
        return dict(status='not_registered_precision_target_unmet', selected_N=None,
            execution_started=False, geometry_uniform_draws=0, **ZERO)
    assert type(n) is int and n in CANDIDATE_N
    assert precision_result['status'] == 'selected' and precision_result['selection_pass'] is True
    output = root / OUTPUT
    if (output / 'registration.json').exists():
        result = read(output / 'registration.json')
        assert result['selected_N'] == n, 'Preserve the existing selected cohort'
        return result
    with attempt(root / preparation_plan['output_dir'] / 'registration', 'register-precision-confirmation') as cost:
        cost.update(ZERO, geometry_uniform_draws=0)
        child_paths = [f'performance/manifests/mainline_precision_confirmation_stream_{i}_v1.json' for i in (1, 2)]
        own = {root / MANIFEST, *(root / p for p in child_paths)}
        assert all(not path.exists() for path in own), 'Preserve existing registrations; partial preparation requires explicit inspection'
        reference = read(root / REFERENCE)
        rng = np.random.default_rng(GEOMETRY_SEED)
        geometries = [dict(task_geometry_shift=rng.uniform(-.035, .035, 3).tolist(),
            task_geometry_radius_scale=float(rng.uniform(.95, 1.05))) for _ in range(n)]
        cost['geometry_uniform_draws'] = 4 * n
        outer, children = make_plans(reference, geometries, preparation_plan, precision_result,
            datetime.now(timezone.utc).isoformat())
        excluded = own | {root / outer['preparation_manifest']}
        paths = [p for p in sorted((root / 'performance/manifests').rglob('*.json')) if p not in excluded]
        check_freshness(outer, children, (read(path) for path in paths))
        outer['execution_preparation']['validation_file'] = str(checks)
        assert Path(replication.ROOT) == root, 'Metadata materializer must use the same checkout'
        for path, child in children:
            child['execution_preparation']['validation_file'] = str(checks)
            save(root / path, child)
        save(root / MANIFEST, outer)
        prepared = [replication.materialize(child) for _, child in children]
        validation = validate_metadata(outer, children, prepared, root)
        result = dict(status='registered_preparation_complete', manifest=MANIFEST,
            selected_N=n, children=[p for p, _ in children], node=node, checks_file=str(checks),
            collision_checked_manifests=len(paths), geometry_uniform_draws=4 * n,
            execution_started=False, models_loaded=0, truth_read=False,
            budget=outer['budget'], **validation, **ZERO)
        save(output / 'registration.json', result)
    return result

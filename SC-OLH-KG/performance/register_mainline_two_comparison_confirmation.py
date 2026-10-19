"""Register one fresh geometry cohort before any target observations."""
from copy import deepcopy
from datetime import datetime, timezone

import numpy as np

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save

NAME = 'mainline_terminal_two_comparison_confirmation_v1'
MANIFEST = ROOT / f'performance/manifests/{NAME}.json'


def register():
    assert not MANIFEST.exists(), 'Keep the registered cohort unchanged'
    previous = read(ROOT / 'performance/manifests/mainline_unseen_geometry_confirmation_v1.json')
    development = read(ROOT / 'performance/manifests/mainline_terminal_two_comparison_development_v1.json')
    output = ROOT / 'paper_artifacts' / NAME
    with attempt(output / 'registration', 'register-fresh-two-comparison-confirmation') as cost:
        cost.update(new_source_fits=0, new_target_initial_calls=0,
            new_truth_function_calls=0, download_bytes=0, geometry_uniform_draws=96,
            source_model_restorations=0, engine_restorations=0, posterior_restorations=0)
        rng = np.random.default_rng(2026104000)
        families = list(dict.fromkeys(t['domain'] for t in previous['tasks']))
        tasks, streams = [], {}
        for domain in families:
            source = next(t['source_search_manifest'] for t in previous['tasks'] if t['domain'] == domain)
            for i in range(8):
                number = len(tasks) + 1
                seed = 2026104000 + number
                geometry = dict(task_geometry_shift=rng.uniform(-.035, .035, 3).tolist(),
                    task_geometry_radius_scale=float(rng.uniform(.95, 1.05)))
                tasks.append(dict(domain=domain, geometry_task_seed=seed, optimizer_seed=seed,
                    source_search_manifest=source,
                    geometry=dict(task_id=f'two_comparison_confirmation_{domain}_{i}',
                                  problem_kwargs=geometry)))
                streams[str(seed)] = dict(observation_seed=2026104100 + number,
                                          verification_seed=2026104200 + number)
        # A registration collision would invalidate freshness before observing outcomes.
        proposed = {t['geometry_task_seed'] for t in tasks} | {
            v for row in streams.values() for v in row.values()} | {2026104000}
        old_geometries = []
        def walk(value):
            if isinstance(value, dict):
                if 'task_geometry_shift' in value and 'task_geometry_radius_scale' in value:
                    old_geometries.append(value)
                for key, item in value.items():
                    if 'seed' in key and isinstance(item, int):
                        assert item not in proposed, f'Registered seed collision: {item}'
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
        for path in (ROOT / 'performance/manifests').glob('mainline*.json'):
            walk(read(path))
        assert all(t['geometry']['problem_kwargs'] not in old_geometries for t in tasks)
        frozen = deepcopy(previous['frozen'])
        frozen['terminal'] = 'Paired frozen old same_policy and new three-candidate/two-comparison rules; identical shortlist per history. No model, threshold or allocation adjustment.'
        frozen['new_terminal'] = deepcopy(development['frozen_candidate'])
        plan = dict(name=NAME, status='registered_not_started',
            registered_utc=datetime.now(timezone.utc).isoformat(), execution_started=False,
            source_manifest=previous['source_manifest'],
            parent_manifest='performance/manifests/mainline_terminal_two_comparison_development_v1.json',
            output_dir=f'paper_artifacts/{NAME}', geometry_seed=2026104000,
            objective='Independently confirm AFTER old-to-new terminal certification and signed quality gains on24 fresh geometries, and quantify contribution concentration without model or threshold selection.',
            scope='24 fresh geometries,8per known family; one independent observation and verification stream each. Previous opened cohorts excluded from estimates.',
            geometry_description='Single RNG2026104000; Inventory,Queue,FactorShock order;8 successive shift U(-.035,.035)^3 and radius U(.95,1.05) draws per family, without filtering.',
            tasks=tasks, replicates=[dict(replicate=1, output_dir=f'paper_artifacts/{NAME}/replicate_1', seed_by_task=streams)],
            search_arms=deepcopy(previous['search_arms']), terminal_arms=deepcopy(previous['terminal_arms']),
            old_protocol=development['old_protocol'], candidate_protocol=development['candidate_protocol'],
            primary_candidate='legacy_role_after_init_point_only', primary_reference='legacy_role_point_only',
            frozen=frozen, randomness=deepcopy(previous['randomness']),
            statistical_unit='24 independent registered geometries; paired protocols and nine arms are not independent samples.',
            budget=dict(geometry_count=24, observation_replicates=1, unique_search_histories=168,
                terminal_cells_per_protocol=216, registered_terminal_cells=432,
                initial_physical_calls=240, online_search_physical_calls=432,
                maximum_verification_calls_per_cell=352, online_logical_cap=365, initial_only_logical_cap=362,
                maximum_verification_logical_calls_per_protocol=76032,
                expanded_logical_search_calls_per_protocol=2736,
                maximum_logical_target_calls_both_protocols=157536,
                maximum_unique_terminal_calls_per_paired_cell=368,
                maximum_verification_physical_calls=79488, maximum_target_physical_calls=80160,
                bound_basis='Same-cell union: safety80+128+128 and objective block0+block1 of16 each. Per-protocol352 cap remains; historical spending and unused old safety tails stay charged.',
                new_source_calls=0, new_source_fits=0, download_bytes=0),
            barriers=dict(before_sampling='Freeze all geometry vectors, source/model references and streams; validate only changed orchestration with deterministic substitutes.',
                before_online='All24 initialization histories and64-neighbor banks frozen before any adaptive search.',
                before_verification='All168 registered histories or retained failures and both sets of216 terminal inputs/missing records frozen before any terminal simulation.',
                before_scoring='Seal every old/new terminal outcome or failure before opening/evaluating any target truth. Terminal results never update searches.',
                failure='Preserve any arm failure, paid prefixes and CPU; no automatic resampling, precision patch, seed replacement or dropping tasks.'),
            execution_preparation=dict(status='pending', workers=12,
                inherited_preflight=previous['execution_preparation']['inherited_preflight'],
                inherited_seed_routing=previous['execution_preparation']['inherited_seed_routing'],
                inherited_reuse_checks=development['execution_checks']),
            analysis=dict(primary='AFTER paired new-minus-old C and signedG; report both without substitution.',
                metrics=deepcopy(development['primary_analysis']['metrics']),
                secondary='AFTER-minus-POINT and FULL-minus-AFTER under each terminal and change in those contrasts; all native/GP/structural/STOP controls.',
                decomposition=deepcopy(development['mechanism']),
                concentration='Report all per-geometry effects, family directions, wins/ties/losses, positive and negative sums, largest positive contribution share. Also remove the single largest positive contribution for a labeled descriptive sensitivity only; never replace the full registered primary estimate.',
                uncertainty='Descriptive paired95%t intervals over geometry-level effects. If any baseline or execution missing, full24 mean/interval remain null; completed defined subset explicitly labeled. Binary certification support bounds are not confidence intervals.',
                failures='Retain new failures as new failures. Historical POINT3018 does not imply that fresh POINT tasks fail; attempt all9 registered arms.',
                fixed_size='Finish all24 tasks without interim scoring, optional expansion, threshold/model changes or adopting another arm from results.',
                truth='Only after both-terminal seal, cache analytic truth for sharedinitial10 and frozen terminal policy union; count every new truth evaluation separately.',
                interpretation='Known-family/new-geometry confirmation, not unseen-family or within-geometry noise replication. Equal352 ceilings do not imply equal actual usage.'),
            historical_accounts=['paper_artifacts/mainline_unseen_geometry_confirmation_v1/completion.json',
                'paper_artifacts/mainline_terminal_two_comparison_development_v1/completion.json'],
            protocol_document='docs/mainline_terminal_two_comparison_confirmation_protocol_20261017.md',
            new_target_calls_at_registration=0, truth_used_at_registration=False)
        plan['analysis']['decomposition'].pop('focal_cases', None)
        save(MANIFEST, plan)
    return plan


if __name__ == '__main__':
    print(register()['status'])

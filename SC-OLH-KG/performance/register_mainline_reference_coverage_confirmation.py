"""Register one independent reference/coverage cohort; never execute science."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import socket

import numpy as np

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.run_mainline_observation_seed_replication import materialize
from performance.register_mainline_safe_unsampled_validation import check_freshness
from performance.mainline_reference_coverage import VARIANTS


NAME = 'mainline_reference_coverage_confirmation_v1'
MANIFEST = f'performance/manifests/{NAME}.json'
REFERENCE = 'performance/manifests/mainline_terminal_two_comparison_confirmation_v1.json'
METHOD = 'performance/manifests/mainline_reference_coverage_preparation_v1.json'
OUTPUT = f'paper_artifacts/{NAME}'
PROTOCOL = 'docs/mainline_reference_coverage_confirmation_protocol_20261017.md'
GEOMETRY_SEED = 2026110000
AFTER, POINT = 'legacy_role_after_init_point_only', 'legacy_role_point_only'


def make_plan(reference, geometries, registered_utc):
    """Reuse source/search recipes, replacing only prospective metadata/terminal design."""
    keys = ('source_manifest', 'search_arms', 'terminal_arms', 'old_protocol',
            'candidate_protocol', 'frozen', 'randomness')
    plan = {key: deepcopy(reference[key]) for key in keys}
    domains = tuple(dict.fromkeys(t['domain'] for t in reference['tasks']))
    assert len(domains) == 3 and len(geometries) == 24
    tasks, streams = [], {}
    for number, geometry in enumerate(geometries, 1):
        domain, seed = domains[(number - 1) // 8], GEOMETRY_SEED + number
        source = next(t['source_search_manifest'] for t in reference['tasks'] if t['domain'] == domain)
        tasks.append(dict(domain=domain, geometry_task_seed=seed, optimizer_seed=seed,
            source_search_manifest=source, geometry=dict(
                task_id=f'reference_coverage_confirmation_{domain}_{(number - 1) % 8}',
                problem_kwargs=geometry)))
        streams[str(seed)] = dict(observation_seed=seed + 100, verification_seed=seed + 200)
    plan.update(name=NAME, status='registered_not_started', execution_started=False,
        registered_utc=registered_utc, parent_manifest=REFERENCE, method_manifest=METHOD,
        output_dir=OUTPUT, protocol_document=PROTOCOL, geometry_seed=GEOMETRY_SEED,
        tasks=tasks, replicates=[dict(replicate=1, output_dir=f'{OUTPUT}/replicate_1', seed_by_task=streams)],
        objective='Independently compare the frozen common-initial-reference and actual-comparison-coverage joint terminal on the unchanged AFTER search, retaining all nine controls and all four factorial cells.',
        scope='24 new known-family geometries, eight per family, one fresh observation/verification stream; previously opened48 flows are historical only.',
        geometry_description='RNG2026110000; original family order; eight successive shift U(-.035,.035)^3 and radius U(.95,1.05) draws per family; no screening, replacement or outcome-dependent expansion.',
        terminal_variants=list(VARIANTS), primary_arm=AFTER,
        primary_contrast=['reference_two_comparison', 'original_old'],
        statistical_unit='24 paired geometry-level effects; arms, four terminal cells, policies and comparisons are not independent samples.',
        new_target_calls_at_registration=0, truth_used_at_registration=False)
    plan['frozen']['terminal'] = 'Stage79 registered four-cell reference/coverage rule; unchanged verifiers and thresholds. Original_old remains the default baseline.'
    plan['reference_selection'] = dict(method_manifest=METHOD,
        input='STOP prefix-ten saved complete primary_pool and observation indices',
        order=['probability_violation', 'objective', 'first_initial_index'],
        freeze='All24 complete orders sealed before online search; finite saved scores kept without clipping.',
        shortlist='Retain original primary and incumbent slots1/2; replace slot3 by first distinct physical initial policy in shared order.',
        truth_used=False, final_search_scores_used=False, new_scores=0,
        failure='Missing initial recording or no distinct reference stays an admission failure; no fallback or posterior restoration.',
        augmented_controls='All nine reference_* arms receive the same STOP reference order; reference GP is augmented, original_old GP remains intact.')
    plan['budget'] = dict(geometry_count=24, observation_replicates=1, unique_search_histories=168,
        terminal_cells_per_variant=216, registered_terminal_cells=864,
        initial_physical_calls=240, online_search_physical_calls=432,
        maximum_verification_calls_per_cell=352, maximum_four_cell_logical_terminal_calls=1408,
        online_logical_cap=365, initial_only_logical_cap=362,
        maximum_verification_logical_calls_per_variant=76032,
        expanded_logical_search_calls_per_variant=2736,
        maximum_logical_target_calls_four_variants=315072,
        maximum_unique_terminal_calls_per_four_cell=512,
        maximum_unique_terminal_calls_when_third_identical=368,
        maximum_verification_physical_calls=110592, maximum_target_physical_calls=111264,
        bound_basis='Conservative same-arm raw union: safety80+128+128+128=464, block0=16, two distinct third-policy block1 pairs=32; no cross-arm deduplication. Per-cell352 ceilings remain. Failed physical calls and any explicit recovery are charged separately.',
        analytic_truth_function_call_bound='At most3N fields for sealed initial10 plus terminal/deployed union, N<=24*28; at most2016 analytic calls, separate from simulation.',
        new_source_calls=0, new_source_fits=0, registration_target_calls=0, download_bytes=0)
    plan['barriers'] = dict(
        before_sampling='Register all24 geometry vectors, source recipes and streams; pass deterministic changed-wiring checks. This preparation does not launch science.',
        before_online='All24 STOP initial histories and shared neighbor banks frozen; seal24 common initial reference orders or explicit initialization failures before adaptive search.',
        before_verification='All168 histories or retained failures sealed, then all864 four-cell terminal inputs or explicit missing records globally sealed.',
        before_scoring='Seal all864 terminal outcomes or retained failures, then freeze initial10 plus four-cell terminal/deployed policy union before any target truth access.',
        failure='Preserve all failures, paid raw prefixes, CPU and historical costs; no automatic retry, seed replacement, precision patch or completed-cell rerun.')
    plan['execution_preparation'] = dict(status='registration_and_substitute_wiring_only',
        independent_scientific_execution=False, workers=None,
        execution_policy='scheduleurm explicit node001-node006, vram0, per-worker BLAS1; independent items via submit-cpu-batch using live free physical cores.',
        remaining='Connect admitted cohort helpers to scheduled scientific stages and post-seal summary before explicit launch; no science descriptors submitted during preparation.')
    plan['analysis'] = dict(primary='Within AFTER: reference_two_comparison minus original_old paired C and signed G, both retained.',
        metrics=deepcopy(reference['analysis']['metrics']),
        secondary='Within each of four terminals AFTER-minus-POINT and FULL-minus-AFTER; nine controls, same-search native crosses, reference-only effect, coverage-only effect and interaction.',
        factorial='For each metric retain (reference_old-original_old), (original_two_comparison-original_old), and joint minus those two effects; no causal mediation percentage.',
        concentration=reference['analysis']['concentration'], uncertainty=reference['analysis']['uncertainty'],
        missing='All24 denominators fixed. Any undefined baseline or failed cell keeps full-cohort G mean/interval null; defined subset explicitly separate. Retain false certificates, ties and losses.',
        fixed_size='Complete all24 registered geometries, no interim scoring, new stream, tuning, screening or outcome-dependent expansion.',
        truth='After global terminal seal only: exact physical-policy union of initial10, all four frozen shortlists and deployments. No complete unsampled-pool audit or posterior inference.',
        interpretation='Terminal gains shared by controls do not establish propagation gains. Report within-cell AFTER-minus-POINT and changes in that contrast. Known-family new-geometry validation only.')
    plan['historical_accounts'] = list(dict.fromkeys([
        *read(ROOT / METHOD)['historical_accounts'],
        'paper_artifacts/mainline_reference_coverage_preparation_v1/scheduler_execution.json']))
    return plan


def validate_metadata(plan, prepared):
    assert plan['status'] == 'registered_not_started' and not plan['execution_started']
    assert len(plan['tasks']) == 24 and len(plan['search_arms']) == 7 and len(plan['terminal_arms']) == 9
    assert plan['frozen']['d'] == 1000 and plan['frozen']['search_calls'] == 13
    assert plan['terminal_variants'] == list(VARIANTS)
    assert plan['budget']['maximum_target_physical_calls'] == 24 * (10 + 6 * 3 + 9 * 512)
    assert plan['budget']['maximum_logical_target_calls_four_variants'] == 4 * 24 * (8 * 365 + 362)
    for family in prepared['replicates'][0]['families']:
        source, terminal = read(ROOT / family['search_manifest']), read(ROOT / family['terminal_manifest'])
        original = read(ROOT / next(t['source_search_manifest'] for t in plan['tasks'] if t['domain'] == family['domain']))
        for key in ('task', 'anchor', 'frozen_assets', 'neighbor_count', 'neighbor_radius',
                    'neighbor_safe_margin_scale', 'frontends', 'role_prior', 'point_only', 'after_init_point_only'):
            assert source[key] == original[key], key
        assert len(source['target_seeds']) == 8 and terminal['arms'] == plan['terminal_arms']
        assert 'fixed_initialization_file_by_task' not in source and 'fixed_neighbors_file_by_task' not in source
        for seed in source['target_seeds']:
            assert source['observation_seed_by_task'][str(seed)] == seed + 100
            assert terminal['verification_seed_by_task'][str(seed)] == seed + 200
        for value in source['frozen_assets'].values():
            if isinstance(value, str) and value.endswith(('.pkl', '.json', '.npz')):
                assert (ROOT / value).is_file(), value
    return dict(status='registration_validated', geometry_count=24, search_histories=168,
        terminal_cells=864, independent_scientific_execution=False,
        new_target_calls=0, new_truth_function_calls=0, new_fits=0,
        models_loaded=0, posteriors_restored=0, new_candidate_scores=0,
        maximum_target_physical_calls=111264, maximum_expanded_logical_calls=315072)


def register(manifest=MANIFEST, checks=None):
    node = socket.gethostname().split('.')[0]
    assert node in {f'node{i:03d}' for i in range(1, 7)}, node
    assert checks and read(ROOT / checks)['status'] == 'passed', 'Substitute checks must pass before registration'
    output, path = ROOT / OUTPUT, ROOT / manifest
    if (output / 'registration.json').exists():
        return read(output / 'registration.json')
    with attempt(output / 'registration', 'register-reference-coverage-confirmation') as cost:
        cost.update(new_target_initial_calls=0, new_truth_function_calls=0, new_source_fits=0,
            engine_restorations=0, posterior_restorations=0, geometry_uniform_draws=0, download_bytes=0)
        current = read(path) if path.exists() else None
        if current and current['status'] == 'registered_not_started':
            plan = current
        else:
            assert not current or current['status'] == 'planned_preparation_only'
            rng = np.random.default_rng(GEOMETRY_SEED)
            vectors = [dict(task_geometry_shift=rng.uniform(-.035, .035, 3).tolist(),
                task_geometry_radius_scale=float(rng.uniform(.95, 1.05))) for _ in range(24)]
            cost['geometry_uniform_draws'] = 96
            plan = make_plan(read(ROOT / REFERENCE), vectors, datetime.now(timezone.utc).isoformat())
            check_freshness(plan, [read(p) for p in (ROOT / 'performance/manifests').glob('mainline*.json') if p != path])
            plan['execution_preparation']['validation_file'] = str(checks)
            save(path, plan)
        prepared = materialize(plan)
        summary = validate_metadata(plan, prepared)
        summary.update(manifest=str(path.relative_to(ROOT)), node=node, geometry_uniform_draws=96,
            checks_file=str(checks), status='registered_preparation_complete')
        save(output / 'registration.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=MANIFEST)
    parser.add_argument('--checks', required=True)
    args = parser.parse_args()
    print(register(args.manifest, args.checks))

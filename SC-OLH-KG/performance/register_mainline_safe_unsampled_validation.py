"""Register fresh geometry metadata; never start searches or open target truth."""
from copy import deepcopy
from datetime import datetime, timezone
import socket

import numpy as np

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.run_mainline_observation_seed_replication import materialize

NAME = 'mainline_safe_unsampled_validation_v1'
MANIFEST = ROOT / f'performance/manifests/{NAME}.json'
REFERENCE = 'performance/manifests/mainline_terminal_two_comparison_confirmation_v1.json'
OUTPUT = ROOT / 'paper_artifacts/mainline_safe_unsampled_validation_registration_v1'
PROTOCOL = 'docs/mainline_safe_unsampled_validation_protocol_20261017.md'
GEOMETRY_SEED = 2026107000
AFTER, POINT, FULL = ('legacy_role_after_init_point_only', 'legacy_role_point_only', 'legacy_role_full')


def make_plan(reference, geometries, registered_utc):
    """Use explicit new vectors and frozen recipes, without old completion state."""
    keys = ('source_manifest', 'search_arms', 'terminal_arms', 'old_protocol',
            'candidate_protocol', 'frozen', 'randomness', 'budget', 'barriers')
    plan = {key: deepcopy(reference[key]) for key in keys}
    tasks, streams = [], {}
    # Preserve the repository's family spelling and order.
    families = tuple(dict.fromkeys(t['domain'] for t in reference['tasks']))
    assert len(families) == 3 and len(geometries) == 24
    for number, geometry in enumerate(geometries, 1):
        domain = families[(number-1)//8]
        source = next(t['source_search_manifest'] for t in reference['tasks'] if t['domain'] == domain)
        seed = GEOMETRY_SEED + number
        tasks.append(dict(domain=domain, geometry_task_seed=seed, optimizer_seed=seed,
            source_search_manifest=source, geometry=dict(
                task_id=f'safe_unsampled_validation_{domain}_{(number-1)%8}', problem_kwargs=geometry)))
        streams[str(seed)] = dict(observation_seed=2026107100+number, verification_seed=2026107200+number)
    output = f'paper_artifacts/{NAME}'
    plan.update(name=NAME, status='registered_not_started', execution_started=False,
        registered_utc=registered_utc, parent_manifest=REFERENCE, output_dir=output,
        protocol_document=PROTOCOL, geometry_seed=GEOMETRY_SEED, tasks=tasks,
        replicates=[dict(replicate=1, output_dir=f'{output}/replicate_1', seed_by_task=streams)],
        objective='Independent frozen AFTER-minus-POINT deployment comparison, with complete post-seal audit of safe unobserved candidate competition.',
        scope='24 new geometries,8per known family,one new observation/verification stream; opened cohorts are development context only.',
        geometry_description='RNG2026107000; same family order/distribution as Stage65:8 successive shift U(-.035,.035)^3 and radius U(.95,1.05) per family; no screening or replacement.',
        primary_candidate=AFTER, primary_reference=POINT, primary_terminal='old',
        statistical_unit='24 geometry-level paired effects; policies, pairs, prefixes, arms and terminal protocols are not independent samples.',
        new_target_calls_at_registration=0, truth_used_at_registration=False)
    plan['execution_preparation'] = dict(status='audit_wiring_pending', workers=None,
        entrypoint='performance/mainline_cluster_dispatch.py',
        execution_policy='scheduleurm node001-node006 explicit pool; vram0; BLAS1; stage barriers',
        **{key: deepcopy(reference['execution_preparation'][key]) for key in
           ('inherited_preflight', 'inherited_seed_routing', 'inherited_reuse_checks')},
        remaining='Implement the registered full-action-pool truth audit and AP-old primary summary after global terminal seal; check population, selection replay, failure retention and cost separation with substitutes before starting the cohort.')
    plan['barriers']['before_scoring'] = (
        'Seal all432 old/new terminal outcomes or retained failures, then seal the complete audit policy inventory from saved action_ids, before opening any target truth. No scoring or truth-driven stopping during execution.')
    plan['analysis'] = dict(primary='AFTER-minus-POINT under old/default terminal: paired C and signed G; retain A,R,P,L and false certificates.',
        metrics=deepcopy(reference['analysis']['metrics']),
        secondary='Same contrast under new terminal; FULL-minus-AFTER; old frontend/backend, same-source GP, structural, STOP and both native terminal crosses. New terminal remains secondary.',
        concentration=reference['analysis']['concentration'], uncertainty=reference['analysis']['uncertainty'],
        fixed_size='All24 registered tasks, no interim scoring, extra stream, optional expansion, outcome filtering, seed replacement, model or terminal changes.',
        missing='Missing baseline or failed execution stays unknown, not zero; all24 denominators and conditional defined-subset denominators are shown.',
        interpretation='A positive mean alone is insufficient for a stable-benefit claim. Interpret paired uncertainty, certification, false certificates, family effects, largest-contributor sensitivity and actual fees together; never promote a secondary contrast to replace a failed primary.')
    plan['mechanism_audit'] = dict(
        arms=[AFTER, POINT, FULL], paid_prefixes=[10,11,12],
        population='Union of every saved action_ids point at all3 prefixes in all3 role histories, plus initial10 and all9-arm old/new terminal/deployed policy union; exact physical-policy dedup within geometry only. Freeze before truth; no shortlist/selected-only restriction.',
        observations='Unobserved means absent from that arm history strictly before the current query; also track the same initially-unobserved set through all3 prefixes without replacing it by final survivors.',
        safety='True feasibility is chance_margin<=0 from the analytic target functions after sealing; sampled search eligibility is saved sampled_margin<=0. Report all4 combinations. Source-terminal probability<=0.5 is a different quantity.',
        coverage='For every arm/prefix/task record unobserved count, truly feasible unobserved count, sampled-eligible feasible unobserved count, safe improving count, and whether at least2 safe unobserved policies compete. Zero coverage and unavailable rows stay in the24-task report.',
        ranking='Over all truly feasible currently-unobserved action candidates, retain all unordered objective-gap pairs, predicted objective-mean and sampled-objective misorder/tie counts and best-eligible choice regret. Aggregate within geometry before between-geometry summaries; shared-neighbor versus dynamic origin remains separate.',
        pairing='At prefix10 compare common physical candidates and initial history, checking pool/order equality before fixed-state attribution. In the common pool, retain both-unobserved true-feasible pairs and distinguish both sampled-eligible, one eligible, neither eligible and zero/one-policy sets. Later own-history comparisons are trajectory associations; intersections never replace full-pool coverage.',
        selection='Replay actual score argmax and fallback from saved arrays without model restoration or new draws; separate risk exclusion, mean ranking, sample reversal, repeated query and candidate absence. Keep every gain, tie and loss.',
        deployment='Follow every queried initially-unobserved policy into observed pool, frozen shortlist, safety, objective comparison and final deployment; join C,R,G,P,L. Merely containing a good candidate is not deployed value.',
        attribution_limit='Saved risk/objective substitutions describe fixed-state selection only; no causal mediation percentage, unseen alternative-history rewards, independent policy-pair sample count or restored posterior.',
        truth_account='Reuse same-geometry exact-policy truth only. At most3 new analytic function calls per distinct uncached policy (objective, constraint mean, constraint sigma); inventory N is fixed before truth, planned bound3N, actual missing-field calls charged separately. No simulation in audit.',
        execution_gap='Successful query_audit.json and query_policies.npz already save the needed arrays. Retain partial failure audit with same-attempt raw linkage; a captured choice need not have a returned observation. Add post-seal inventory/scoring/aggregation and explicit AP-old primary reporting; existing terminal summary still labels new-minus-old primary. No search-rule change or posterior inference.')
    plan['budget'].update(analytic_truth_function_call_bound='3 times sealed distinct audit policy inventory N; account separately from target simulation and reused fields.',
        registration_target_calls=0)
    plan['historical_accounts'] = list(dict.fromkeys([
        *reference['historical_accounts'], f"{reference['output_dir']}/completion.json",
        'paper_artifacts/mainline_terminal_two_comparison_seed_replication_v1/completion.json',
        'paper_artifacts/mainline_three_stream_attribution_v1/completion.json',
        'paper_artifacts/mainline_source_neighbor_coverage_v1/scheduler_execution.json',
        'paper_artifacts/mainline_source_neighbor_propagation_v1/scheduler_execution.json']))
    return plan


def check_freshness(plan, registered):
    proposed = {plan['geometry_seed']} | {t['geometry_task_seed'] for t in plan['tasks']}
    proposed.update(v for row in plan['replicates'][0]['seed_by_task'].values() for v in row.values())
    assert len(proposed) == 73
    old_geometry = []
    def walk(value):
        if isinstance(value, dict):
            if 'task_geometry_shift' in value and 'task_geometry_radius_scale' in value:
                old_geometry.append(value)
            for key, item in value.items():
                if 'seed' in key and isinstance(item, int):
                    assert item not in proposed, f'Registered seed collision: {item}'
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    for item in registered:
        walk(item)
    assert all(t['geometry']['problem_kwargs'] not in old_geometry for t in plan['tasks'])


def validate_metadata(plan, prepared):
    assert plan['status'] == 'registered_not_started' and not plan['execution_started']
    assert len(plan['tasks']) == 24 and len(plan['search_arms']) == 7 and len(plan['terminal_arms']) == 9
    assert plan['frozen']['d'] == 1000 and plan['frozen']['search_calls'] == 13
    budget = plan['budget']
    assert budget['maximum_target_physical_calls'] == 24*10 + 24*6*3 + 24*9*368 == 80160
    assert budget['maximum_logical_target_calls_both_protocols'] == 2*24*(8*365+362) == 157536
    for family in prepared['replicates'][0]['families']:
        source, terminal = read(ROOT/family['search_manifest']), read(ROOT/family['terminal_manifest'])
        original = read(ROOT/next(t['source_search_manifest'] for t in plan['tasks'] if t['domain']==family['domain']))
        for key in ('task','anchor','frozen_assets','neighbor_count','neighbor_radius','neighbor_safe_margin_scale','frontends','role_prior','point_only','after_init_point_only'):
            assert source[key] == original[key], key
        assert len(source['target_seeds']) == 8 and terminal['arms'] == plan['terminal_arms']
        for seed in source['target_seeds']:
            assert source['observation_seed_by_task'][str(seed)] == seed+100
            assert terminal['verification_seed_by_task'][str(seed)] == seed+200
        assert 'fixed_initialization_file_by_task' not in source and 'fixed_neighbors_file_by_task' not in source
        for value in source['frozen_assets'].values():
            if isinstance(value, str) and value.endswith(('.pkl','.json','.npz')):
                assert (ROOT/value).is_file(), value
    return dict(status='registration_validated', geometry_count=24, search_histories=168,
        terminal_cells=432, primary='AFTER-minus-POINT / old terminal',
        target_cohort_started=False, full_pool_audit_wiring='pending', new_target_calls=0,
        new_truth_function_calls=0, models_loaded=0, posteriors_restored=0,
        maximum_target_physical_calls=80160, maximum_expanded_logical_calls=157536)


def register():
    node = socket.gethostname().split('.')[0]
    assert node in {f'node{i:03d}' for i in range(1,7)}, node
    assert not MANIFEST.exists(), 'Preserve the registered independent cohort'
    with attempt(OUTPUT, 'register-safe-unsampled-validation') as cost:
        cost.update(new_target_initial_calls=0,new_truth_function_calls=0,new_source_fits=0,
            engine_restorations=0,posterior_restorations=0,geometry_uniform_draws=96,download_bytes=0)
        rng = np.random.default_rng(GEOMETRY_SEED)
        vectors = [dict(task_geometry_shift=rng.uniform(-.035,.035,3).tolist(),
            task_geometry_radius_scale=float(rng.uniform(.95,1.05))) for _ in range(24)]
        plan = make_plan(read(ROOT/REFERENCE), vectors, datetime.now(timezone.utc).isoformat())
        check_freshness(plan, [read(p) for p in (ROOT/'performance/manifests').glob('mainline*.json')])
        save(MANIFEST, plan)
        prepared = materialize(plan)
        summary = validate_metadata(plan, prepared)
        summary.update(manifest=str(MANIFEST.relative_to(ROOT)),node=node,geometry_uniform_draws=96)
        save(OUTPUT/'summary.json',summary)
    return summary


if __name__ == '__main__':
    print(register())

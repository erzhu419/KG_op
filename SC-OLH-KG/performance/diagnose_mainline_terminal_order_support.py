"""Saved terminal stopping evidence and source support, without model execution."""
from collections import Counter
import numpy as np
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save

MANIFEST = ROOT/'performance/manifests/mainline_unseen_geometry_attribution_v1.json'
FULL, AFTER = 'legacy_role_full', 'legacy_role_after_init_point_only'


def stopping(group):
    third = [(case, row) for case in group for row in case['shortlist'] if row['position'] == 3]
    unexposed = [(case, row) for case, row in third if not row['safety_attempted']]
    good = [(case, row) for case, row in third if row['good_new'] is True]
    omitted = [dict(seed=case['seed'], arm=case['arm'], policy=policy['policy'], policy_id=policy['policy_id'],
        position=policy['shortlist_position'], disposition=policy['disposition'],
        final=policy['final'], stop_status=case['objective_comparison']['status'],
        better_than_actual_deployment=policy['better_than_certified_deployment'],
        quality_difference_to_deployment=policy['quality_difference_to_certified_deployment'])
        for case in group for policy in case['good_new_policies'] or []
        if policy['disposition'] != 'deployed']
    third_omitted = [row for row in omitted if row['position']==3]
    unique = {row['policy_id'] for row in third_omitted}
    return dict(completed_cells=len(group), known_baseline_cells=sum(c['baseline_defined'] for c in group),
        third_slot_count=len(third), third_slot_unexposed=len(unexposed),
        unexposed_stop_status=dict(Counter(c['objective_comparison']['status'] for c, _ in unexposed)),
        good_third_slot_count=len(good), good_third_slot_unexposed=sum(not r['safety_attempted'] for _, r in good),
        good_third_slot_better_than_deployment=sum(r['position']==3 and
            r['better_than_actual_deployment'] is True for r in omitted),
        good_third_unique_task_policies=len(unique),
        good_third_unique_geometries=len({row['seed'] for row in third_omitted}),
        good_third_unique_better_in_any_cell=len({row['policy_id'] for row in third_omitted
            if row['better_than_actual_deployment'] is True}),
        good_third_unknown_deployment_comparisons=sum(row['better_than_actual_deployment'] is None
            for row in third_omitted),
        omitted_good=omitted)


def crossing(rows):
    known = [row for row in rows if row['role_minus_native_quality_increment'] is not None]
    values = [row['role_minus_native_quality_increment'] for row in known]
    return dict(registered_pairs=len(rows), known_quality_pairs=len(known),
        role_wins=sum(v > 1e-12 for v in values), ties=sum(abs(v) <= 1e-12 for v in values),
        native_wins=sum(v < -1e-12 for v in values),
        quality_sum_known=float(sum(values)), quality_mean_known=float(np.mean(values)),
        reward_mean=float(np.mean([r['role_minus_native_reward'] for r in rows])),
        role_certificate_advantage=sum(int(r['role_certified'])-int(r['native_certified']) for r in rows),
        nonzero_cases=[r for r in rows if abs(r['role_minus_native_reward']) > 1e-12])


def source_support():
    manifest = read(ROOT/'performance/manifests/mainline_role_source_holdout_v1.json')
    source = ROOT/manifest['output_dir']
    saved = read(source/'summary.json')
    archive = read(ROOT/manifest['archive'])
    source_archives = []
    for path in sorted((ROOT/'paper_artifacts/legacy_joint_bridge_v1/inputs').glob('*/archive.json')):
        rows = read(path)['tasks']
        source_archives.append(dict(heldout_target_family=path.parent.name,
            path=str(path.relative_to(ROOT)), source_families=[task['name'] for task in rows],
            source_policy_counts=[len(task['X_integer']) for task in rows]))
    schemas = []
    for fold in range(manifest['folds']):
        for case in read(source/f'fold_{fold}'/'forecasts.json')['cases']:
            schemas.append(dict(fold=fold, family=case['family'], d=case['d'],
                held_policies=len(case['held_policy_ids']), adaptation_rows=len(case['adaptation_raw']),
                forecast_arms=list(case['predictions']),
                fields=list(case['predictions']['full']),
                mean_array_shape=np.shape(case['predictions']['full']['mean'])))
    return dict(manifest='performance/manifests/mainline_role_source_holdout_v1.json',
        saved_summary=str((source/'summary.json').relative_to(ROOT)), forecast_schemas=schemas,
        frozen_source_archives=source_archives,
        archive=[dict(family=t['name'], d=len(t['X_integer'][0]), policies=len(t['X_integer']),
            repeats_per_policy=sorted({len(r) for r in t['Y_replicates']}),
            stored_constraint_sigma=True, stored_true_chance_labels=False) for t in archive['tasks']],
        source_final_contrasts=saved['final_contrasts'],
        previous_candidate=dict(manifest='performance/manifests/mainline_bounded_mean_source_v1.json',
            report='docs/mainline_bounded_mean_source_results_20261016.md', adopted=False),
        available='Role moments on held d50 policies, source raws and native fit diagnostics.',
        missing=['Matched native forecast/ranking on the same held policies and observed adaptation histories.',
            'Held-source observed13 terminal ranking and stopping comparison with independent outcomes.',
            'Independent80/128 safety blocks and paired objective-guard outcomes. Three source repeats do not supply them.'],
        conclusion='Existing heldout evidence does not select one uniform role/native ranking or third-slot promotion correction.',
        next_source_check='Predeclare one source-only matched observed-policy ranking comparison, retaining allfolds/families and noisy held labels. No target-specific gate, weight, family selection or assumed certificate recovery.')


def main():
    plan = read(MANIFEST); output = ROOT/plan['output_dir']
    with attempt(output, 'saved-terminal-order-source-support') as cost:
        cost.update(new_target_initial_calls=0, new_source_fits=0, new_truth_function_calls=0,
                    posterior_restorations=0, raw_replays=0, new_rng_draws=0, download_bytes=0)
        terminal = read(ROOT/plan['terminal_diagnosis_file'])
        cases = terminal['cases']; crosses = terminal['native_crosses']
        result = dict(status='complete', source_file=plan['terminal_diagnosis_file'],
            registered_cells=terminal['registered_terminal_cells'], completed_cells=len(cases),
            missing_cells=terminal['missing_cells'], undefined_baseline_cells=terminal['undefined_baseline_cells'],
            all_completed=stopping(cases), by_arm=[dict(arm=arm, **stopping([c for c in cases if c['arm']==arm]))
                for arm in dict.fromkeys(c['arm'] for c in cases)],
            crossings=[dict(history_arm=arm, **crossing([r for r in crosses if r['history_arm']==arm]),
                by_family=[dict(domain=domain, **crossing([r for r in crosses if r['history_arm']==arm and r['domain']==domain]))
                           for domain in dict.fromkeys(c['domain'] for c in cases)]) for arm in (FULL, AFTER)],
            focal=[{key: c[key] for key in ('seed','arm','queries','primary','deployed','certified',
                'best_initial_quality_increment','objective_challenger','objective_comparison','shortlist','good_new_policies')}
                for c in cases if c['seed'] in (2026103021,2026103007) and c['history_arm']==AFTER],
            selector=dict(order='Same feasible-first ordering: p_violation<=.5 first, objective mean ascending; no feasible points -> violation probability then objective.',
                posterior_difference='Role decision context supplies current role means/covariance/episode variance. Native bypasses that context and uses native models already updated by the identical paid history.',
                shortlist='Observed policies only: objective challenger, frozen initial empirical-objective incumbent, remaining posterior-ranked fallback; existing same-policy transform chooses a distinct challenger when necessary.',
                stop='Verify challenger and incumbent. If either is safe, deploy challenger/guardedincumbent and do not expose third slot. Third slot is attempted only when both fail.',
                code=['acquisition/decision_backends.py:55','performance/mainline_legacy_backend.py:140',
                    'performance/diagnose_mainline_terminal_backend.py:34','core/terminal_verification.py:829',
                    'performance/run_mainline_same_policy.py:28']), source_support=source_support(),
            limits=['Crossed terminals are paired interventions sharing histories, not independent searches.',
                'Changing slots may jointly change ordering, exposure, budget and verifier stream.',
                'Good-policy quality is undefined without a feasible initial baseline; failures are not zero-gain rows.',
                'Omitted true-good policies have no unobserved counterfactual certificate; no correction is selected from target outcomes.'])
        save(output/'terminal_order_support.json', result)
    print(dict(completed_cells=len(cases), crosses=len(crosses), cost=cost), flush=True)


if __name__ == '__main__':
    main()

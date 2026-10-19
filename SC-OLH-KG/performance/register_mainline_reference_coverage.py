"""Register a frozen reference-selection/coverage design; never run a verifier."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
NAME = 'mainline_reference_coverage_preparation_v1'
MANIFEST = f'performance/manifests/{NAME}.json'
REFERENCE = 'performance/manifests/mainline_reference_retention_v1.json'
VARIANTS = ('original_old', 'original_two_comparison', 'reference_old',
            'reference_two_comparison')
ZERO = dict.fromkeys(('new_target_initial_calls', 'new_target_search_calls',
    'new_verification_calls', 'new_source_calls', 'new_source_fits',
    'new_truth_function_calls', 'new_candidate_scores', 'source_model_restorations',
    'engine_restorations', 'posterior_restorations', 'new_raw_replays',
    'new_rng_draws'), 0)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def build_plan(reference):
    """Only the method and saved-input admission grid are registered here."""
    return dict(name=NAME, status='planned', output_dir=f'paper_artifacts/{NAME}',
        protocol_document='docs/mainline_reference_coverage_protocol_20261017.md',
        reference_manifest=REFERENCE, tasks=deepcopy(reference['tasks']),
        streams=deepcopy(reference['streams']),
        terminal_arms=deepcopy(reference['terminal_arms']),
        frozen=dict(AFTER=True, source_model=True, representation=True, d=1000,
            search=True, initialization=True, observation_streams=True,
            default_terminal=True, numerical_precision=True, all9_controls=True),
        selection=dict(
            source='Complete saved STOP primary_pool at prefix10; common to all9 controls.',
            key=['probability_violation', 'objective_mean', 'first_initial_position'],
            ordering='Ascending lexicographic order on every distinct initial physical policy; no probability cutoff.',
            timing='Freeze the full initial order before reading any final arm shortlist. Select R after excluding the frozen first two physical policies.',
            preserve='Existing distinct-challenger transform and first two final slots, with empirical incumbent in slot2.',
            replace='Third slot only; keep identity cases, displaced original thirds and all admission failures.',
            forbidden=['target truth', 'verification winners', 'final search scores',
                'per-family rules', 'selection of the best Stage78 role label'],
            missing='Record admission failure; no model restore, extra scoring, final-score fallback or sample exclusion.'),
        variants={name:dict(shortlist='reference' if name.startswith('reference') else 'original',
            coverage='two_comparison' if name.endswith('two_comparison') else 'old',
            candidate_budgets=[80,120,120] if name.endswith('two_comparison') else [80,128,128],
            comparison_budget_per_policy=8,
            objective_delta=.05/6 if name.endswith('two_comparison') else .05/3,
            maximum_objective_comparisons=2 if name.endswith('two_comparison') else 1,
            terminal_cap=352) for name in VARIANTS},
        budgets=dict(safety_familywise_delta=.05, safety_delta_per_candidate=.05/3,
            objective_familywise_delta=.05/3, independent_comparison_blocks=[0,1],
            old_actual_path_maximum=336, new_safety_calls=320,
            new_comparison_calls=[0,16,32], new_terminal_maximum=352,
            search_calls_online=13, search_calls_STOP=10,
            online_total_cap_per_terminal=365, STOP_total_cap_per_terminal=362,
            four_terminal_logical_cap_per_arm=1408,
            interpretation='352 is per terminal; a four-cell comparison is not one352-call experiment. Future physical reuse requires identical point, slot, stream and replicate, and preserves all historical fees.'),
        analysis=dict(primary='Prospective AFTER reference_two_comparison minus original_old: C and signed G jointly, with failures, false certificates and actual fees.',
            factorial='Coverage on original lists: original_two_comparison-original_old; reference selection under full coverage: reference_two_comparison-original_two_comparison; also reference_old-original_old and the interaction.',
            propagation='Report AFTER-POINT within each fixed terminal and the change of that difference; a common terminal gain is not prior propagation gain.',
            controls='All9 controls in all4 cells. The common STOP reference adds the same source-prior information to each augmented cell; keep original baselines separate from these controlled terminal comparisons.',
            statistics='Geometry is the unit; average repeated streams within geometry, retain missing baselines and failed cells, positive/negative/tied outcomes, family summaries and largest-gain sensitivity.',
            reporting='Track actual safety, actual objective coverage, reference/displaced-third deployment, certification loss, quality gain and loss, logical calls, reused calls, new physical calls and failure spending separately.'),
        preparation=dict(saved_task_streams=48, saved_arm_inputs=432,
            configured_variant_cells=1728, reference_order_before_final_inputs=True,
            read_target_truth=False, read_verification_records=False,
            read_raw_observations=False, run_verifiers=False,
            scope='Saved-input wiring and synthetic reachability only; no re-certification or retrospective scientific score.'),
        scientific_validation=dict(status='not_registered_not_started',
            execution_authorized=False, new_geometry_seeds=[],
            requirement='A later independent cohort/execution registration must freeze geometry and observation seeds, sampling plan, physical reuse and global pre-truth seals before any science. Opened48 streams cannot become that independent estimate.'),
        historical_accounts=list(dict.fromkeys([*reference['historical_accounts'],
            f"{reference['output_dir']}/completion.json",
            f"{reference['output_dir']}/scheduler_execution.json"])),
        execution=dict(node_pool=[f'node{i:03}' for i in range(1,7)], vram=0,
            threads_per_worker=1, scheduler='submit-cpu-batch',
            stages=['deterministic_substitute_checks', 'register_saved_input_wiring']),
        new_work=deepcopy(ZERO))


def prepare_inputs(root, plan):
    from performance.mainline_reference_coverage import freeze_initial_order, make_variants

    root = Path(root)
    output = root / plan['output_dir']
    orders, cases, failures, seals = {}, [], [], {}
    prior_path = output / 'wiring.json'
    prior = read(prior_path) if prior_path.exists() else dict(cases=[])
    completed = {(r['stream'], r['seed'], r['arm']): r for r in prior['cases']
                 if r['status'] == 'wired_not_executed'}
    reused_orders = reused_inputs = 0
    # Only the sealed path index is read here; verification files and truth stay unopened.
    for stream in plan['streams']:
        child = read(root / stream['manifest'])
        seal = read(root / child['output_dir'] / 'execution_status.json')
        if seal['status'] != 'sealed':
            raise ValueError('Original saved inputs require a sealed execution index')
        seals[stream['name']] = {(r['seed'], r['arm']): r for r in seal['cases']}
    # Complete the entire query-before order stage before inspecting final arm inputs.
    for stream in plan['streams']:
        for task in plan['tasks']:
            key = (stream['name'], task['seed'])
            try:
                path = output / 'reference_orders' / key[0] / f'seed_{key[1]}.json'
                if path.exists():
                    order = read(path)
                    reused_orders += 1
                else:
                    cell = seals[key[0]][(key[1], 'initialization_only')]
                    source = cell['old']['input_file']
                    order = freeze_initial_order(read(root / source), stream=key[0], input_file=source)
                    save(path, order)
                orders[key] = (order, str(path.relative_to(root)))
            except Exception as error:
                failures.append(dict(stage='initial_order', stream=key[0], seed=key[1],
                    error=f'{type(error).__name__}: {error}'))
    for stream in plan['streams']:
        for task in plan['tasks']:
            key = (stream['name'], task['seed'])
            for arm in plan['terminal_arms']:
                row = dict(stream=key[0], seed=key[1], domain=task['domain'], arm=arm['name'])
                old_row = completed.get((*key, arm['name']))
                if old_row is not None:
                    cases.append(deepcopy(old_row))
                    reused_inputs += 1
                    continue
                try:
                    order, order_path = orders[key]
                    source = seals[key[0]][(key[1], arm['name'])]['old']['input_file']
                    item = read(root / source)
                    variants = make_variants(item, order)
                    if set(variants) != set(VARIANTS):
                        raise ValueError('All four declared cells must be retained')
                    old_third = item['frozen_shortlist'][2]['point']
                    new_third = variants['reference_old']['frozen_shortlist'][2]['point']
                    row.update(status='wired_not_executed', original_input_file=source,
                        reference_order_file=order_path, reference_point=new_third,
                        displaced_third=old_third, shortlist_changed=new_third != old_third,
                        original_protocol=item['protocol'],
                        search_evaluation_count=item['search_evaluation_count'],
                        verification_seed=item['verification_seed'],
                        variants={name:dict(protocol=v['protocol'],
                            candidate_budgets=v['candidate_budgets'],
                            objective_comparison_delta=v['objective_comparison_delta'],
                            objective_incumbent_position=v['objective_incumbent_position'],
                            executed=False) for name,v in variants.items()})
                except Exception as error:
                    row.update(status='admission_failed', error=f'{type(error).__name__}: {error}',
                        variants={name:dict(status='admission_failed', executed=False) for name in VARIANTS})
                    failures.append(dict(stage='arm_input', **row))
                cases.append(row)
    save(output / 'wiring.json', dict(cases=cases, failures=failures,
        scientific_execution_started=False, **ZERO))
    return dict(registered_reference_orders=48, prepared_reference_orders=len(orders),
        reused_reference_orders=reused_orders, newly_prepared_reference_orders=len(orders)-reused_orders,
        registered_arm_inputs=432, retained_arm_inputs=len(cases),
        wired_arm_inputs=sum(row['status']=='wired_not_executed' for row in cases),
        reused_wired_arm_inputs=reused_inputs,
        newly_wired_arm_inputs=sum(row['status']=='wired_not_executed' for row in cases)-reused_inputs,
        registered_variant_cells=1728, retained_variant_cells=4*len(cases),
        failures=failures, wiring_file=str((output/'wiring.json').relative_to(root)))


def register(manifest, checks, *, root=ROOT):
    root = Path(root)
    manifest = root / manifest
    plan = read(manifest)
    if plan['status'] == 'registered_preparation_only':
        return read(root / plan['output_dir'] / 'registration.json')
    output = root / plan['output_dir']
    attempt_dir = output / 'registration_attempts'
    number = len(list(attempt_dir.glob('attempt_*.json'))) + 1
    started, wall = time.process_time(), time.monotonic()
    record = dict(attempt=number, started_utc=datetime.now(timezone.utc).isoformat(),
        new_work=deepcopy(ZERO), checks_file=str(checks), historical_accounts=plan['historical_accounts'])
    try:
        validation = read(root / checks)
        if validation['status'] != 'passed':
            raise ValueError('Targeted substitute checks must pass before registration')
        reference = read(root / plan['reference_manifest'])
        if reference['status'] != 'complete':
            raise ValueError('Keep the completed Stage78 reference; do not rerun it')
        if len(plan['tasks']) != 24 or len(plan['streams']) != 2 or len(plan['terminal_arms']) != 9:
            raise ValueError('Preserve the full saved admission grid')
        prior_wiring = output / 'wiring.json'
        if prior_wiring.exists():
            attempt_dir.mkdir(parents=True, exist_ok=True)
            archive = attempt_dir / f'prior_wiring_before_attempt_{number}.json'
            shutil.copyfile(prior_wiring, archive)
            record['preserved_prior_wiring_file'] = str(archive.relative_to(root))
        details = prepare_inputs(root, plan)
        record.update(details)
        record['status'] = 'registered_preparation_only' if not details['failures'] else 'preparation_incomplete'
        record['independent_scientific_validation_started'] = False
        record['default_terminal_changed'] = False
        plan.update(status=record['status'], registered_utc=record['started_utc'],
            registration_file=f"{plan['output_dir']}/registration.json")
        save(manifest, plan)
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        record.update(cpu_seconds=time.process_time()-started, wall_seconds=time.monotonic()-wall,
            completed_utc=datetime.now(timezone.utc).isoformat())
        save(attempt_dir / f'attempt_{number}.json', record)
        save(output / 'registration.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=MANIFEST)
    parser.add_argument('--checks', required=True)
    args = parser.parse_args()
    result = register(args.manifest, args.checks)
    print(json.dumps({key:result.get(key) for key in ('status','prepared_reference_orders',
        'wired_arm_inputs','retained_variant_cells','cpu_seconds')}))
    raise SystemExit(0 if result['status']=='registered_preparation_only' else 1)

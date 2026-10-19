"""Register paired terminal development from metadata; never read or score raw rows."""
from datetime import datetime, timezone
from pathlib import Path

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.prepare_mainline_two_comparison_terminal import ZERO

NAME = 'mainline_terminal_two_comparison_development_v1'
OUTPUT = ROOT / 'paper_artifacts' / NAME
MANIFEST = ROOT / 'performance/manifests' / (NAME + '.json')
PREPARATION = ROOT / 'paper_artifacts/mainline_terminal_two_comparison_preparation_v1'


def block_inventory(registration):
    old_input = read(ROOT / registration['original_input_file'])
    old = read(ROOT / registration['original_verification_file'])
    result = old['verification']
    assert old['verification_seed'] == old_input['verification_seed']
    raw_path = Path(registration['original_verification_file']).with_name('paid_verification_raw.json')
    assert (ROOT / raw_path).is_file()
    assert old_input['objective_incumbent_position'] == 2
    cursor, blocks, safety = 0, [], {}
    for row in result['attempts']:
        index, count = row['candidate_index'], row['verification_budget']
        assert count == (80 if index == 0 else 128)
        safety[index] = row
        blocks.append(dict(kind='safety', candidate_index=index, raw_start=cursor,
            stored_count=count, reusable_prefix_count=80 if index == 0 else 120,
            old_certified=bool(row['certified'])))
        cursor += count
    assert 0 in safety and 1 in safety
    comparison_count = result['objective_comparison_budget']
    assert comparison_count in (0, 16)
    if comparison_count:
        blocks.append(dict(kind='objective', comparison_block_index=0,
            challenger_index=0, incumbent_index=1, raw_start=cursor,
            stored_count=16, reusable_prefix_count=16))
    cursor += comparison_count
    assert cursor == result['verification_budget'] == old['new_verification_calls']
    primary_safe = bool(safety[0]['certified'])
    missing_third = 2 not in safety
    missing_first_possible = primary_safe and not comparison_count
    reusable = sum(b['reusable_prefix_count'] for b in blocks)
    new_bound = 120*missing_third + 16*missing_first_possible + 16
    logical_bound = 320 + 16*primary_safe + 16
    assert reusable + new_bound == logical_bound
    return dict(seed=registration['seed'], domain=registration['domain'], arm=registration['arm'],
        status='registered_not_executed', candidate_input=registration['input_file'],
        original_input=registration['original_input_file'],
        old_verification=registration['original_verification_file'], old_raw_file=str(raw_path),
        verification_seed=old_input['verification_seed'], old_raw_rows=cursor,
        search_evaluation_count=old_input['search_evaluation_count'],
        quality_baseline_defined=registration['quality_baseline_defined'],
        available_blocks=blocks, old_primary80_certified=primary_safe,
        missing_third_safety=missing_third, existing_comparison0=bool(comparison_count),
        possible_new_comparison0=missing_first_possible,
        comparison1=dict(status='new_if_required', challenger_index=2,
            incumbent_index='actual first-stage certified winner', maximum_calls=16),
        conditionally_reusable_rows=reusable, new_physical_calls_upper_bound=new_bound,
        candidate_logical_terminal_cap=352, tightened_logical_terminal_upper_bound=logical_bound,
        old_plus_new_physical_upper_bound=cursor+new_bound)


def register():
    with attempt(OUTPUT, 'register-paired-development-and-block-coverage') as cost:
        cost.update(ZERO, raw_rows_read=0, certificate_rescorings=0)
        prepared = read(PREPARATION / 'prepared.json')
        checks = read(PREPARATION / 'summary.json')
        assert checks['checks_passed'] == 35 and checks['checks_failed'] == 0
        available = [r for r in prepared['registrations'] if r['status'] == 'prepared_only']
        rows = [block_inventory(r) for r in available]
        missing = [r for r in prepared['registrations'] if r['status'] != 'prepared_only']
        assert len(rows) == 215 and len(missing) == 1
        fields = ('old_raw_rows', 'conditionally_reusable_rows', 'new_physical_calls_upper_bound',
                  'candidate_logical_terminal_cap', 'tightened_logical_terminal_upper_bound',
                  'old_plus_new_physical_upper_bound')
        totals = {k:sum(r[k] for r in rows) for k in fields}
        assert totals['new_physical_calls_upper_bound'] == 24248
        arms = []
        for arm in prepared['arm_counts']:
            group = [r for r in rows if r['arm'] == arm]
            arms.append(dict(arm=arm, registered=24, available=len(group),
                existing_third_safety=sum(not r['missing_third_safety'] for r in group),
                missing_third_safety=sum(r['missing_third_safety'] for r in group),
                existing_comparison0=sum(r['existing_comparison0'] for r in group),
                possible_new_comparison0=sum(r['possible_new_comparison0'] for r in group),
                **{k:sum(r[k] for r in group) for k in fields}))
        inventory = dict(status='metadata_registration_complete', registered_cells=216,
            available_cells=215, arms=arms, rows=rows, missing_cells=missing, totals=totals,
            raw_rows_read=0, certificate_rescorings=0,
            scope='Same old cell only. Available prefixes are conditional reuse capacity, not necessarily consumed. Raw contents must be checked against their exact policy/block at execution.')
        save(OUTPUT / 'block_inventory.json', inventory)
        plan = dict(name=NAME, status='registered_not_started', registered_utc=datetime.now(timezone.utc).isoformat(),
            execution_started=False, parent_manifest='performance/manifests/mainline_terminal_two_comparison_preparation_v1.json',
            output_dir=str(OUTPUT.relative_to(ROOT)), block_inventory=str((OUTPUT/'block_inventory.json').relative_to(ROOT)),
            preparation_file=str((PREPARATION/'prepared.json').relative_to(ROOT)),
            protocol_document='docs/mainline_terminal_two_comparison_development_protocol_20261017.md',
            scope='Paired development on opened24 geometries; old terminal results retained, one frozen new terminal. No source/model/search/shortlist/risk/precision change; not independent confirmation.',
            geometry_count=24, registered_cells_per_protocol=216, available_candidate_cells=215,
            terminal_arms=list(prepared['arm_counts']), old_protocol='same_policy_distinct_challenger',
            candidate_protocol='frozen_three_candidate_two_objective_comparisons',
            frozen_candidate=read(ROOT/'performance/manifests/mainline_terminal_two_comparison_preparation_v1.json')['single_candidate'],
            reuse=dict(unit='same original geometry/seed/arm cell only; no cross-arm pooling',
                safety='Exact same slot, point and seed, first80/120/120 rows. Recompute candidate certificates; do not inherit128-sample status.',
                comparison0='Reuse the original ordered primary/incumbent8+8 block only when required; recompute with delta .05/6.',
                comparison1='New independent block1 for third versus actual first-stage winner, only when both are safety-certified.',
                identity='Raw identity is geometry/task/verification_seed/stream/candidate_index or comparison_block_index/ordered points/replication/side. Budget and delta describe inference, not raw identity.',
                failure='Persist exact partial blocks and spending. Ordinary repeat is blocked; an explicit resume reuses every recorded prefix, including a half CRN pair, and buys only missing samples. No resampling or precision change.'),
            budget=dict(new_source_calls=0,new_initial_calls=0,new_search_calls=0,new_source_fits=0,
                engine_restorations=0,posterior_restorations=0,new_truth_function_calls=0,download_bytes=0,
                maximum_candidate_terminal_logical_calls=75680,
                tightened_candidate_terminal_logical_bound=73584,
                maximum_new_physical_verification_calls=24248,
                conditional_existing_raw_capacity=49336,per_cell_terminal_cap=352,
                online_total_cap=365,stop_total_cap=362,
                cumulative_note='Old physical spending remains charged, including unused last8 safety rows. Historical old plus new spending is not bounded by one candidate352 cap.'),
            primary_analysis=dict(candidate_arm='legacy_role_after_init_point_only',
                comparisons='Report new-minus-old C and signed G for each of all9 arms; AFTER is predesignated, not selected from results.',
                metrics=dict(A='actual certificate',C='I(certified and true feasible)',R='C*Qdeployed',
                    G='C*(Qdeployed-Qbest_true_feasible_initial10), signed',P='max(G,0)',L='max(-G,0)'),
                secondary='Report change in AFTER-minus-POINT and FULL-minus-AFTER under old versus new terminal, plus native/GP/structural/STOP references; fixed histories mean this is terminal conversion, not new search learning.',
                aggregation='Statistical unit geometry, not arm or pair. Report per-geometry differences, family directions, wins/ties/losses, contribution concentration and descriptive paired t intervals on explicitly stated available subsets.',
                missing='POINT3018 stays missing. Queue3009 G/P/L and all baseline-dependent metrics stay null even when C=0. All24 G means/intervals remain null; conditional n=23, POINT n=22 if no further failures. POINT all24 C mean/interval stays null; known paired sum S permits support bounds[(S-1)/24,(S+1)/24], not confidence intervals.'),
            mechanism=dict(intermediate_B='Use new first-stage certified winner when present; otherwise use new third-only fallback final outcome.',
                decomposition='For defined metric M: Mnew-Mold=(Mnew-MB)+(MB-Mold). First term is third challenging an existing winner; second is joint safety120 and stricter-delta change, retaining interactions.',
                matched_diagnostics='Report128-to120 safety pass/fail flips and old/new block0 decisions only for actual matched candidates/streams; do not invent additive causal components or new variants.',
                focal_cases={'3016':'Record actual block1 opponent; stricter block0 delta may retain initial6, so53 need not directly face18.',
                    '3021':'Record49 safety120, actual opponent, objective comparison and deployed quality.',
                    '3007':'Keep both role arms already deploying61 and native crossings omitting61; report losses as well as recoveries.'}),
            barriers=['Complete exact-prefix/block reuse adapter and deterministic budget/resume/metric checks before real execution. The35 previous checks do not validate cache reuse.',
                'Use all215 frozen candidate inputs and preserve the216th search failure; no intermediate outcome-based truncation, new seeds, variants or precision patches.',
                'Seal all available candidate results and failures before scoring cached truth; no true-function calls. Missing cached labels remain unknown.',
                'Keep old results immutable. Report actual logical, reused, newly paid, failed and historical costs separately. Existing searches are charged once in the inherited global ledger.'],
            decision='Assess certification, signed quality and actual spending jointly. Do not adopt automatically from this opened cohort, fill missing values, or attribute generic terminal gains to prior propagation. No further patch screening in this batch.',
            inherited_failures=prepared['inherited_failures'], inherited_accounts=prepared['inherited_accounts'],
            new_work_at_registration=ZERO)
        if MANIFEST.exists():
            raise RuntimeError('Registration already exists; preserve the frozen protocol')
        save(MANIFEST, plan)
    return inventory


if __name__ == '__main__':
    result = register()
    print(result['status'], result['totals'])

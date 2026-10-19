"""Wire saved three-policy lists to the registered candidate; CLI prepares only."""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone

from core.terminal_two_comparison import verify_frozen_three_candidate_tournament
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.run_mainline_terminal_equal import case_dir

MANIFEST = ROOT / 'performance/manifests/mainline_terminal_two_comparison_preparation_v1.json'
PROTOCOL = 'frozen_three_candidate_two_objective_comparisons'
ZERO = dict.fromkeys(('new_source_calls', 'new_target_initial_calls', 'new_target_search_calls',
    'new_verification_calls', 'new_truth_function_calls', 'new_source_fits',
    'source_model_restorations', 'engine_restorations', 'posterior_restorations',
    'raw_conditioning', 'raw_replays', 'new_rng_draws', 'download_bytes'), 0)


def candidate_input(original):
    """Change only the registered terminal allocation; preserve every policy record."""
    assert original['task']['d'] == 1000
    assert original['candidate_budgets'] == [80, 128, 128]
    assert original['objective_incumbent_position'] == 2
    assert original['objective_comparison_budget'] == 8
    assert original['objective_comparison_delta'] == .05 / 3
    assert original['familywise_delta'] == .05
    assert len(original['frozen_shortlist']) == 3
    assert len({tuple(r['point']) for r in original['frozen_shortlist']}) == 3
    assert original['search_evaluation_count'] == (10 if original['arm'] == 'initialization_only' else 13)
    item = copy.deepcopy(original)
    item.update(protocol=PROTOCOL, candidate_budgets=[80, 120, 120],
                objective_comparison_delta=.05/6, maximum_objective_comparisons=2)
    return item


def run_verifier(problem, item):
    """Adapter for the existing recorded verifier; preparation never calls it."""
    assert item['protocol'] == PROTOCOL and item['candidate_budgets'] == [80, 120, 120]
    assert item['objective_incumbent_position'] == 2
    assert item['objective_comparison_budget'] == 8 and item['maximum_objective_comparisons'] == 2
    assert item['objective_comparison_delta'] == .05/6 and item['familywise_delta'] == .05
    return verify_frozen_three_candidate_tournament(problem, item['frozen_shortlist'],
        seed=item['verification_seed'], **{key:item[key] for key in
            ('search_evaluation_count', 'method', 'mean_delta_fraction', 'shortlist_mode')})


def prepare(plan):
    output = ROOT / plan['output_dir']
    with attempt(output, 'prepare-frozen-two-comparison-inputs') as cost:
        cost.update(ZERO)
        reference = read(ROOT / plan['evidence']['terminal'])
        assert reference['registered_cells'] == 216 and reference['completed_cells'] == 215
        rows = reference['cases']
        # A missing available input must block preparation, not silently remove a control.
        inputs = [(r, candidate_input(read(ROOT / r['files']['input']))) for r in rows]
        seeds = sorted({r['seed'] for r in rows})
        arms = [r['arm'] for r in reference['arms']]
        missing = [dict(seed=r['seed'], arm=r['arm'], domain=r['domain'], status=r['status'])
                   for r in reference['missing_cells']]
        assert len(seeds) == 24 and len(arms) == 9
        assert {(r['seed'], r['arm']) for r in rows} | {(r['seed'], r['arm']) for r in missing} == {
            (s, a) for s in seeds for a in arms}
        assert len(inputs) == 215 and len(missing) == 1
        registrations = []
        for row, item in inputs:
            path = case_dir(output, item['seed'], item['arm'], PROTOCOL) / 'input.json'
            if path.exists():
                assert read(path) == item, 'Existing prepared candidate input changed'
            else:
                save(path, item)
            registrations.append(dict(seed=item['seed'], arm=item['arm'], domain=item['domain'],
                status='prepared_only', input_file=str(path.relative_to(ROOT)),
                original_input_file=row['files']['input'],
                original_verification_file=row['files']['verification'],
                search_evaluation_count=item['search_evaluation_count'],
                quality_baseline_defined=row['baseline_defined']))
        terminal_plan = dict(name=PROTOCOL, status='prepared_not_executed',
            target_seeds=seeds, arms=arms, protocols={PROTOCOL:[80, 120, 120]},
            execution_failures=reference['prior_failures'], output_dir=plan['output_dir'],
            real_execution_authorized=False)
        save(output / 'terminal_plan.json', terminal_plan)
        result = dict(status='prepared_not_executed', registered_cells=216, prepared_inputs=215,
            registered_geometries=24, arm_counts=dict(Counter(r['arm'] for r in registrations)),
            observed_prefix_counts=dict(Counter(r['search_evaluation_count'] for r in registrations)),
            registrations=registrations + missing, inherited_failures=reference['prior_failures'],
            seed_mapping=dict(block0='[verification_seed, replication, 0x4F424A44]',
                block1='[verification_seed, replication, 0x4F424A44, 1]',
                within_pair='Two independent RNG objects initialized with identical block/replication components.',
                first_comparison_skipped='Third versus winner still uses block1.',
                search='[observation_seed or config.seed, history_index, 0x53434F4C]',
                primary_safety='[verification_seed, replication, 0x56455249]',
                other_safety='[verification_seed, candidate_index, replication, 0x56534C54]'),
            inherited_accounts=['paper_artifacts/mainline_unseen_geometry_confirmation_v1/completion.json',
                'paper_artifacts/mainline_objective_contrast_exposure_v1/completion.json'],
            new_work=ZERO, prepared_utc=datetime.now(timezone.utc).isoformat())
        save(output / 'prepared.json', result)
        cost['prepared_inputs'] = len(inputs)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(MANIFEST))
    args = parser.parse_args()
    result = prepare(read(args.manifest))
    print(result['status'], result['prepared_inputs'], '/', result['registered_cells'])

"""Read saved d1000 context for the source risk-channel diagnosis; no replay."""
import argparse
from collections import Counter
import math

import numpy as np
from scipy.special import ndtr

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save

MANIFEST = ROOT/'performance/manifests/mainline_source_risk_channel_attribution_v1.json'
FULL, AFTER = 'legacy_role_full', 'legacy_role_after_init_point_only'


def spread(values):
    values = [float(x) for x in values if x is not None]
    return dict(count=len(values), minimum=min(values) if values else None,
                median=float(np.median(values)) if values else None,
                maximum=max(values) if values else None)


def summarize(plan):
    prior = read(ROOT/plan['inputs']['target_trace_context'])
    query = read(ROOT/prior['artifacts']['queries'])
    terminal = read(ROOT/prior['terminal_diagnosis_file'])
    focal = read(ROOT/prior['artifacts']['focal_chains'])
    support = read(ROOT/prior['artifacts']['terminal_source_support'])
    assets = []
    for family in read(ROOT/prior['prepared_file'])['replicates'][0]['families']:
        search = read(ROOT/family['search_manifest'])
        assets.append(dict(domain=family['domain'], search_directory=search['output_dir'],
            frozen_assets=search['frozen_assets'], available_by_arm={arm: {
                name: sum((ROOT/search['output_dir']/f'seed_{seed}'/arm/name).exists()
                          for seed in search['target_seeds'])
                for name in ('search.json','search_engine.pkl','native_terminal.json')}
                for arm in prior['query_arms']}))
    selected = []
    for row in query['states']:
        point = row['choices']['actual']['selected']
        z = point['stochastic_margin_mean']/math.sqrt(max(point['constraint_epistemic'], 1e-12))
        selected.append(dict(seed=row['seed'], domain=row['domain'], arm=row['arm'],
            paid_prefix=row['paid_prefix'], policy_id=point['policy_id'],
            quality_baseline_defined=row['quality_baseline_defined'],
            cached_truth_available=point['truth'] is not None,
            sampled_feasible_count=row['choices']['actual']['feasible_count'],
            derived_query_probability=float(ndtr(z)), standardized_mean_margin=z,
            sigma_ratio=point.get('sigma_ratio'), risk_mean_bias=point.get('risk_mean_bias'),
            risk_noise_margin_bias=point.get('risk_noise_margin_bias')))
    query_arms = []
    for arm in prior['query_arms']:
        rows = [r for r in selected if r['arm'] == arm]
        query_arms.append(dict(arm=arm, recorded_selected_queries=len(rows),
            sampled_empty_feasible_states=sum(r['sampled_feasible_count'] == 0 for r in rows),
            selected_probability_exact_zero=sum(r['derived_query_probability'] == 0 for r in rows),
            selected_probability_exact_one=sum(r['derived_query_probability'] == 1 for r in rows),
            selected_sigma_ratio=spread(r['sigma_ratio'] for r in rows),
            selected_standardized_margin=spread(r['standardized_mean_margin'] for r in rows)))
    cells = []
    for row in terminal['cases']:
        challenger = row.get('objective_challenger') or {}
        cells.append(dict(seed=row['seed'], domain=row['domain'], arm=row['arm'],
            baseline_defined=row['baseline_defined'], primary=row['primary'], deployed=row['deployed'],
            certified=row['certified'], best_initial_quality_increment=row['best_initial_quality_increment'],
            challenger=challenger, saved_shortlist_probabilities=[p['violation_probability']
                for p in row['shortlist'] if p.get('violation_probability') is not None],
            saved_endpoint_rows=[{k:p[k] for k in ('policy','position','role','violation_probability',
                'true_feasible','safety_attempted','deployed')} for p in row['shortlist']
                if p.get('violation_probability') in (0., 1.)]))
    terminal_arms = []
    for arm in prior['terminal_arms']:
        rows = [r for r in cells if r['arm'] == arm['name']]
        empty = [r for r in rows if r['challenger'].get('eligible_count') == 0]
        probabilities = [p for r in rows for p in r['saved_shortlist_probabilities']]
        terminal_arms.append(dict(arm=arm['name'], registered=24, completed=len(rows),
            challenger_empty=len(empty), challenger_empty_seeds=[r['seed'] for r in empty],
            challenger_empty_certified=sum(r['certified'] for r in empty),
            shortlist_probabilities_saved=len(probabilities),
            saved_probability_exact_zero=sum(p == 0 for p in probabilities),
            saved_probability_exact_one=sum(p == 1 for p in probabilities),
            final_full_probability_vector=None, whole_observed_pool_empty=None))
    contrasts, noise_swaps = [], []
    for pair in query['paired_states']:
        if pair['left'] != FULL or not (pair['first_divergence'] and
                pair['same_paid_history'] and pair['same_physical_pool']):
            continue
        for point in pair['selected_pair_components']:
            full, after = point['arms'][FULL], point['arms'][AFTER]
            truth = point['policy']['truth']
            true_noise = None if truth is None else truth['chance_margin']-truth['constraint_mean_centered']
            contrasts.append(dict(seed=pair['seed'], domain=pair['domain'], paid_prefix=pair['paid_prefix'],
                policy=point['policy'], full_noise_margin=full['noise_margin'],
                after_noise_margin=after['noise_margin'],
                noise_margin_change=full['noise_margin']-after['noise_margin'],
                full_to_after_noise_margin_ratio=full['noise_margin']/after['noise_margin'],
                cached_true_noise_margin=true_noise,
                absolute_noise_bias_change=None if true_noise is None else
                    abs(full['noise_margin']-true_noise)-abs(after['noise_margin']-true_noise)))
        swaps = [s for s in pair['substitutions'] if s['substitution'] == 'risk_noise_margin']
        if any(s['donor_choice_recovered'] for s in swaps):
            noise_swaps.append(dict(seed=pair['seed'], paid_prefix=pair['paid_prefix'],
                                    selected=pair['selected'], swaps=swaps))
    return dict(status='complete', registered_geometry_count=len(query['tasks']),
        registered_terminal_cells=216, completed_terminal_cells=len(cells),
        task_roster=[{k:t[k] for k in ('replicate','domain','seed','quality_baseline_defined','unavailable_arms')}
                     for t in query['tasks']], missing_query_histories=query['unavailable_query_histories'],
        missing_terminal_cells=terminal['missing_cells'], undefined_quality_reference_tasks=prior['undefined_quality_reference_tasks'],
        query_prefix_evidence=dict(arms=query_arms, selected=selected, first_divergences=query['combined']['comparisons']),
        final_terminal_evidence=dict(arms=terminal_arms, cells=cells, selector=support['selector'],
            unavailable_final_role_score_states=query['unavailable_final_score_states']),
        continued_noise_propagation=dict(same_history_pool_first_divergences=13,
            selected_policy_state_comparisons=contrasts, noise_only_choice_recovery_cases=noise_swaps,
            strict_float_change_sign_counts=dict(Counter('increased' if r['noise_margin_change'] > 0 else
                'decreased' if r['noise_margin_change'] < 0 else 'unchanged' for r in contrasts)),
            strict_float_absolute_noise_bias_change=dict(Counter('worse' if r['absolute_noise_bias_change'] > 0
                else 'better' if r['absolute_noise_bias_change'] < 0 else 'unchanged'
                for r in contrasts if r['absolute_noise_bias_change'] is not None)),
            full_to_after_noise_margin_ratio=spread(r['full_to_after_noise_margin_ratio'] for r in contrasts)),
        focal_chains=[{k:c[k] for k in ('seed','domain','full_after_complete_paid_history_equal',
            'interpretation','final_paid13_candidate_risk_scores','final_score_limitation')} for c in focal['cases']],
        replay_feasibility=dict(status='read_only_feasibility_only_no_restore_or_replay_performed', assets=assets,
            scope='71 complete role histories: FULL24, AFTER24, POINT23 across24 geometries; failed POINT3018 remains missing.48 native terminal observed-pool records already exist.',
            entrypoints=['performance/diagnose_mainline_terminal_backend.py:freeze/signature/challenger_error',
                'performance/mainline_role_prior.py:role_moment_state',
                'performance/mainline_role_after_init.py:after_init_point_only_moment_state',
                'performance/mainline_role_point_only.py:point_only_moment_state',
                'performance/run_mainline_same_policy.py:transform'],
            missing='Final role whole-pool M/N/V and identity weights are absent from JSON, reconstructible from unchanged saved engines, source bundles and13-row histories.',
            labels='Existing search/seed_*/truth.json plus terminal input/verification JSON and Stage57 terminal.json; join only after all reconstructed forecasts freeze.',
            cost='No new fit/simulation/truth/verification/download needed; model restorations and raw conditioning must be registered and charged.Original precision and failures retained.'),
        judgment='Saved d1000 evidence supports risk-sensitive query and terminal changes, but no uniform risk/noise or CDF-fallback correction. Keep the model frozen.',
        next_step='Separately register one all24-geometry final-terminal replay from saved71 complete role histories, retain original failures and freeze reconstructed moments before joining existing true/verified labels. Do not alter the model or numerical precision.',
        limits=[
            'Challenger-subset eligibility can exclude the primary; its empty count is not whole-observed-pool emptiness.',
            'Query probabilities above are deterministic diagnostics from saved selected-policy prefix10/11/12 Gaussian moments, not terminal13 probabilities or the TS selection rule.',
            'Only selected queries and logged shortlist probabilities are covered; unsaved candidate probabilities, ties and final role moments remain unknown.',
            'Native expert-mixture probabilities cannot be reconstructed with one Gaussian standardized margin.',
            'Noise contrasts are 26 selected-policy/state exposures across13 first divergences, not26 independent tasks; tiny floating differences are retained.',
            'Source boundary policy0 and target neighbor0 are different local identifiers; this does not establish the same physical boundary mechanism.',
            'Uncached dynamic truth and Queue3009 undefined quality reference remain missing; POINT3018 stays failed. All9 references remain registered.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(MANIFEST))
    plan = read(parser.parse_args().manifest)
    output = ROOT/plan['output_dir']
    with attempt(output/'target_context', 'saved-target-risk-context') as cost:
        cost.update(new_target_initial_calls=0, new_source_fits=0, new_truth_function_calls=0,
                    posterior_restorations=0, raw_replays=0, new_rng_draws=0, download_bytes=0)
        result = summarize(plan)
        save(output/'target_context.json', result)
    print('target context complete', result['completed_terminal_cells'], cost['cpu_seconds'])


if __name__ == '__main__':
    main()

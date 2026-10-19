"""Join saved query and terminal evidence; no models, simulators or truth evaluation."""
from collections import Counter
from datetime import datetime, timezone
import json

import numpy as np
from scipy.stats import t

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save

MANIFEST = ROOT / 'performance/manifests/mainline_observation_seed_attribution_v1.json'
METRICS = ('certification', 'reward', 'certificate_reference', 'signed_gain',
           'new_policy_gain', 'regression', 'initial_regression', 'new_policy_regression',
           'observed_opportunity', 'unrealized_opportunity')
AFTER = 'legacy_role_after_init_point_only'
POINT = 'legacy_role_point_only'
FULL = 'legacy_role_full'


def key(row):
    return row['replicate'], row['seed'], row['arm']


def means(rows):
    return {metric: float(np.mean([r[metric] for r in rows])) for metric in METRICS}


def decompose(case, scored):
    certificate = int(case['certified'] and case['deployed_truth']['true_feasible'])
    best = case['best_feasible_initial_truth']['quality']
    quality = case['deployed_truth']['quality']
    gain = certificate * max(quality - best, 0.)
    regression = certificate * max(best - quality, 0.)
    opportunity = max([0., *[p['quality_gain_over_best_initial']
                             for p in case['good_new_policies']]])
    initial = case['deployed'].startswith('initial:')
    # Detect physical-policy or baseline mismatches before interpreting the decomposition.
    assert initial or case['deployed'] in case['queries']
    assert not gain or not initial
    assert certificate == scored['certification']
    assert abs(best - scored['best_feasible_initial_truth']['quality']) < 1e-12
    assert abs(gain - regression - scored['quality_increment']) < 1e-12
    assert abs(certificate * best + gain - regression - scored['reward']) < 1e-12
    assert opportunity >= gain - 1e-12
    best_observed = [dict(policy=p['policy'], origin=p['candidate_origin'],
        gain=p['quality_gain_over_best_initial'], disposition=p['disposition'],
        better_than_certified_deployment=p['better_than_certified_deployment'])
        for p in case['good_new_policies']
        if abs(p['quality_gain_over_best_initial'] - opportunity) < 1e-12]
    return dict(replicate=case['replicate'], seed=case['seed'], domain=case['domain'],
        arm=case['arm'], queries=case['queries'], deployed=case['deployed'],
        deployed_quality=quality if certificate else None, best_initial_quality=best,
        certification=certificate, reward=scored['reward'], certificate_reference=certificate*best,
        signed_gain=scored['quality_increment'], new_policy_gain=gain, regression=regression,
        initial_regression=regression if initial else 0.,
        new_policy_regression=0. if initial else regression,
        observed_opportunity=opportunity, unrealized_opportunity=opportunity-gain,
        best_observed_good=best_observed, funnel=case['funnel'],
        deployed_new_origins=[p['candidate_origin'] for p in case['good_new_policies']
                             if p['disposition'] == 'deployed'])


def contrast(rows, source, seeds):
    indexed = {key(r): r for r in rows}
    left, right = source['left'], source['right']
    by_replicate = []
    for replicate in (1, 2):
        paired = []
        for seed in seeds:
            a, b = (indexed[replicate, seed, arm] for arm in (left, right))
            paired.append(dict(replicate=replicate, seed=seed, domain=a['domain'],
                **{metric: a[metric]-b[metric] for metric in METRICS}))
        by_replicate.append(dict(replicate=replicate, rows=paired, means=means(paired)))
    geometry = [dict(seed=seed, domain=by_replicate[0]['rows'][i]['domain'],
        **means([rep['rows'][i] for rep in by_replicate])) for i, seed in enumerate(seeds)]
    average = means(geometry)
    assert abs(average['signed_gain'] - source['task_equal_means']['quality_increment']) < 1e-12
    assert abs(average['reward'] - source['task_equal_means']['reward']) < 1e-12
    for row in [average, *geometry, *[r for rep in by_replicate for r in rep['rows']]]:
        assert abs(row['signed_gain'] - row['new_policy_gain'] + row['regression']) < 1e-12
        assert abs(row['signed_gain'] - row['observed_opportunity'] +
                   row['unrealized_opportunity'] + row['regression']) < 1e-12
    pure = np.asarray([r['new_policy_gain'] for r in geometry])
    half = float(t.ppf(.975, len(seeds)-1)*np.std(pure, ddof=1)/np.sqrt(len(seeds)))
    return dict(left=left, right=right, by_replicate=by_replicate,
        geometry_mean_effects=geometry, geometry_equal_means=average,
        by_family=[dict(domain=domain, means=means([r for r in geometry if r['domain']==domain]))
                   for domain in dict.fromkeys(r['domain'] for r in geometry)],
        pure_gain_descriptive_95pct_t_interval=[float(np.mean(pure))-half, float(np.mean(pure))+half],
        pure_gain_geometry_wins_ties_losses=[int(np.sum(pure>1e-12)),
            int(np.sum(np.abs(pure)<=1e-12)), int(np.sum(pure<-1e-12))])


def summarize(plan, query, terminal, source):
    indexed = {key(r): r for r in source['rows']}
    cases = {key(r): r for r in terminal['cases']}
    expected = {(rep, seed, arm['name']) for rep in plan['replicates']
                for seed in plan['target_seeds'] for arm in plan['terminal_arms']}
    assert set(indexed) == set(cases) == expected
    assert len(terminal['cases']) == len(source['rows']) == plan['expected_terminal_cells']
    assert query['reconstruction']['recorded_choices'] == plan['expected_recorded_choices']
    rows = [decompose(case, indexed[key(case)]) for case in terminal['cases']]
    contrasts = [contrast(rows, c, plan['target_seeds']) for c in source['comparisons']]
    decomposed = {key(r): r for r in rows}
    chains = []
    for comparison in contrasts[:2]:
        left, right = comparison['left'], comparison['right']
        for replica in comparison['by_replicate']:
            for effect in replica['rows']:
                rep, seed = effect['replicate'], effect['seed']
                first = next((p for p in query['paired_states'] if p['replicate']==rep and
                    p['seed']==seed and p['left']==left and p['right']==right and
                    p['first_divergence']), None)
                evidence = None if first is None else {k: first[k] for k in (
                    'paid_prefix', 'same_paid_history', 'same_physical_pool', 'selected')}
                if evidence is not None:
                    evidence['swaps'] = [s for s in first['substitutions'] if s['substitution'] in
                        ('risk_risk_mean+noise_margin+risk_draw', 'all_objective')]
                chains.append(dict(replicate=rep, seed=seed, domain=effect['domain'],
                    left=left, right=right, first_divergence=evidence, decomposition=effect,
                    outcomes={arm: decomposed[rep, seed, arm] for arm in (left, right)}))
    arms = []
    for arm in plan['terminal_arms']:
        selected = [r for r in rows if r['arm']==arm['name']]
        arms.append(dict(arm=arm['name'], task_noise_units=len(selected), means=means(selected),
            by_deployed_origin=[dict(origin=origin,
                deployment_cells=sum(origin in r['deployed_new_origins'] for r in selected),
                geometry_count=len({r['seed'] for r in selected if origin in r['deployed_new_origins']}),
                gain_mean_over_all_units=sum(r['new_policy_gain'] for r in selected
                    if origin in r['deployed_new_origins'])/len(selected))
                for origin in ('shared_neighbor', 'dynamic')],
            by_replicate=[dict(replicate=rep, means=means([r for r in selected if r['replicate']==rep]),
                deployed_new_origins=dict(Counter(origin for r in selected if r['replicate']==rep
                    for origin in r['deployed_new_origins']))) for rep in plan['replicates']]))
    primary = contrasts[0]['geometry_equal_means']
    return dict(status='complete', task_noise_units=24, geometry_count=12,
        query_reconstruction=query['reconstruction'], terminal_cells=len(rows), rows=rows,
        arms=arms, primary=contrasts[0], comparisons=contrasts, decision_chains=chains,
        primary_gain_fractions=dict(new_policy_gain=primary['new_policy_gain']/primary['signed_gain'],
            avoided_regression=-primary['regression']/primary['signed_gain']),
        query_attribution=query['combined'], terminal_funnels=terminal['arms'],
        terminal_native_crosses=terminal['native_crosses'],
        source_execution_failures=source.get('execution_failures', []),
        historical_failure=plan['historical_failure'],
        definitions=dict(signed_gain=plan['decomposition'],
            observed_opportunity=plan['observed_opportunity_decomposition']),
        scope=plan['scope'], limits=plan['limits'])


def main():
    plan = read(MANIFEST)
    output = ROOT / plan['output_dir']
    with attempt(output, 'join-saved-observation-attribution') as cost:
        cost.update(new_target_initial_calls=0, new_source_fits=0, new_truth_function_calls=0,
                    posterior_restorations=0, raw_replays=0, new_rng_draws=0, download_bytes=0)
        result = summarize(plan, read(output/'query_diagnosis.json'), read(output/'terminal.json'),
                           read(ROOT/plan['source_output_dir']/'results.json'))
        save(output/'summary.json', result)
    attempts = [json.loads(line) for line in (output/'attempts.jsonl').read_text().splitlines()]
    fields = ('cpu_seconds', 'wall_seconds', 'new_source_calls', 'new_target_initial_calls',
        'new_target_search_calls', 'new_verification_calls', 'new_source_fits',
        'new_truth_function_calls', 'source_model_restorations', 'posterior_restorations',
        'raw_replays', 'new_rng_draws', 'download_bytes')
    completion = dict(status='complete', cost={k:sum(r.get(k,0) for r in attempts) for k in fields},
        attempts=len(attempts), failed_attempts=[r for r in attempts if r['status']!='completed'],
        completed_utc=datetime.now(timezone.utc).isoformat())
    save(output/'completion.json', completion)
    plan.update(status='complete', completed_utc=completion['completed_utc'])
    save(MANIFEST, plan)
    print(json.dumps(dict(status='complete', cells=len(result['rows']),
                          primary=result['primary']['geometry_equal_means'], cost=completion['cost'])))


if __name__ == '__main__':
    main()

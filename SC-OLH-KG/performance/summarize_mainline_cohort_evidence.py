"""Compare the two saved geometry cohorts without pooling their evidence roles."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'performance/manifests/mainline_cohort_evidence_v1.json'
FULL = 'legacy_role_full'
AFTER = 'legacy_role_after_init_point_only'
POINT = 'legacy_role_point_only'
METRICS = ('certification', 'reward', 'best_initial_reference', 'quality_increment')


def read(path):
    return json.loads((ROOT / path).read_text())


def save(path, value):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def aggregate(rows, expected):
    means = {key: sum(r[key] for r in rows) / len(rows) for key in METRICS}
    return dict(completed=len(rows), expected=expected, missing=expected-len(rows),
        means=means if len(rows) == expected else {key: None for key in METRICS},
        completed_case_means=means)


def comparison(rows, left, right, seeds):
    index = {(r['seed'], r['arm']): r for r in rows}
    paired = []
    for seed in seeds:
        if (seed, left) not in index or (seed, right) not in index:
            continue
        a, b = index[seed, left], index[seed, right]
        paired.append(dict(seed=seed, domain=a['domain'],
            **{key: a[key]-b[key] for key in METRICS}))
    return dict(left=left, right=right, **aggregate(paired, len(seeds)), rows=paired,
        win_tie_loss={key: dict(wins=sum(r[key] > 1e-12 for r in paired),
            ties=sum(abs(r[key]) <= 1e-12 for r in paired),
            losses=sum(r[key] < -1e-12 for r in paired), missing=len(seeds)-len(paired))
            for key in ('reward', 'quality_increment')},
        by_family=[dict(domain=domain, **aggregate(
            [r for r in paired if r['domain'] == domain], len(domain_seeds)))
            for domain, domain_seeds in seeds_by_family(rows, seeds).items()])


def seeds_by_family(rows, seeds):
    return {domain: sorted({r['seed'] for r in rows if r['domain'] == domain and r['seed'] in seeds})
            for domain in dict.fromkeys(r['domain'] for r in rows)}


def cohort(spec):
    source = read(spec['source_manifest'])
    modified = read(spec['modified_results'])
    seeds = source['target_seeds']
    best = {r['seed']: r['best_feasible_initial_truth'] for r in modified['rows']}
    initial = {}
    for family in source['families']:
        search = read(family['search_manifest'])
        for seed in search['target_seeds']:
            saved = read(Path(search['output_dir']) / f'seed_{seed}/initialization.json')
            initial[seed] = {tuple(r['point']) for r in saved['history']}
    assert set(best) == set(initial) == set(seeds)
    variants = {}
    for variant, data in (('original', read(spec['original_results'])), ('modified', modified)):
        rows = []
        for r in data['rows']:
            assert r['certified_true_feasible'] in (True, False)
            certified = int(r['certified_true_feasible'])
            truth = r['deployed_truth']
            increment = truth['quality'] - best[r['seed']]['quality'] if certified else 0.
            reference = certified * best[r['seed']]['quality']
            # A bad task join or use of the first-row baseline invalidates this table.
            assert abs(r['actual_safe_reward'] - reference - increment) < 1e-12
            good = bool(certified and tuple(r['deployed']) not in initial[r['seed']]
                        and truth['objective'] < best[r['seed']]['objective'] - 1e-12)
            if variant == 'modified':
                assert abs(increment - r['certified_quality_gain_vs_best_feasible_initial']) < 1e-12
                assert good == r['certified_new_policy_beats_best_initial']
            rows.append(dict(seed=r['seed'], domain=r['domain'], arm=r['arm'],
                certification=certified, reward=r['actual_safe_reward'],
                best_initial_reference=reference, quality_increment=increment,
                good_new_deployment=good, false_certificate=r['false_certificate'],
                logical_verification_calls=r['logical_total_calls']-r['logical_search_calls']))
        arms = list(dict.fromkeys(r['arm'] for r in rows))
        failures = source.get('execution_failures', [])
        assert len(rows) + len(failures) == len(seeds) * len(arms)
        variants[variant] = dict(terminal_cells=len(rows), rows=rows,
            arms=[dict(arm=arm, **aggregate([r for r in rows if r['arm'] == arm], len(seeds)),
                certified=sum(r['certification'] for r in rows if r['arm'] == arm),
                good_new_deployments=sum(r['good_new_deployment'] for r in rows if r['arm'] == arm),
                false_certificates=sum(r['false_certificate'] for r in rows if r['arm'] == arm),
                logical_verification_calls=sum(r['logical_verification_calls'] for r in rows if r['arm'] == arm))
                for arm in arms],
            primary=comparison(rows, AFTER, POINT, seeds),
            full_minus_after=comparison(rows, FULL, AFTER, seeds),
            references=[comparison(rows, AFTER, arm, seeds) for arm in arms if arm not in (FULL, AFTER, POINT)])
    primary = variants['modified']['primary']
    totals = {key: sum(r[key] for r in primary['rows']) for key in ('reward', 'quality_increment')}
    contributions = [dict(seed=r['seed'], domain=r['domain'],
        mean_contribution={key: r[key]/len(seeds) for key in totals},
        share_of_net_difference={key: r[key]/total if total != 0 else None for key, total in totals.items()})
        for r in primary['rows']]
    return dict(name=spec['name'], evidence_role=spec['evidence_role'],
        task_count=len(seeds), execution_failures=source.get('execution_failures', []),
        variants=variants, modified_primary_contributions=contributions)


def main():
    plan = read(MANIFEST)
    started = time.perf_counter(); cpu = time.process_time()
    results = [cohort(spec) for spec in plan['cohorts']]
    output = Path(plan['output_dir'])
    save(output / 'summary.json', dict(status='complete', cohorts=results,
        pooling='None: opened development and frozen confirmation remain separate.',
        metric_definitions=plan['metric_definitions']))
    cost = dict(cpu_seconds=time.process_time()-cpu, wall_seconds=time.perf_counter()-started,
        new_source_calls=0, new_target_initial_calls=0, new_target_search_calls=0,
        new_verification_calls=0, new_truth_function_calls=0, new_source_fits=0,
        posterior_restorations=0, source_model_restorations=0, raw_replays=0, download_bytes=0)
    completed = datetime.now(timezone.utc).isoformat()
    save(output / 'completion.json', dict(status='complete', cost=cost, completed_utc=completed))
    plan.update(status='complete', completed_utc=completed)
    save(MANIFEST, plan)
    for result in results:
        for variant, data in result['variants'].items():
            arms = {r['arm']: r for r in data['arms']}
            print(result['name'], variant, 'certified', arms[AFTER]['certified'], arms[POINT]['certified'],
                  'good_new', arms[AFTER]['good_new_deployments'], arms[POINT]['good_new_deployments'],
                  'differences', data['primary']['means'])


if __name__ == '__main__':
    main()

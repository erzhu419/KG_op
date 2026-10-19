"""Summarize two fully scored streams; this module never scores target truth."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import socket

from performance import summarize_mainline_safe_unsampled_validation as fresh
from performance.summarize_mainline_confirmation_seed_replication import (
    ORIGINS, geometry_average, origin_rows, sign,
)
from performance.summarize_mainline_two_comparison_confirmation import add_sensitivity, arm_names
from performance.summarize_mainline_two_comparison_development import (
    AFTER, FULL, POINT, FIELDS, METRICS, difference, summarize_rows,
)
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save


MANIFEST = ROOT/'performance/manifests/mainline_terminal_recording_seed_replication_v1.json'
ZERO = dict(new_target_initial_calls=0, new_target_search_calls=0, new_verification_calls=0,
    new_source_calls=0, new_source_fits=0, new_truth_function_calls=0,
    source_model_restorations=0, engine_restorations=0, posterior_restorations=0,
    new_raw_replays=0, new_rng_draws=0, download_bytes=0)


def summarize_unit(plan, rows):
    """The primary is the old-terminal AP contrast, never AFTER's new-minus-old."""
    arms = arm_names(plan)
    groups, secondary = summarize_rows(rows, arms)
    add_sensitivity(groups, secondary, rows)
    ap, fa = fresh.contrast(rows, AFTER, POINT), fresh.contrast(rows, FULL, AFTER)
    controls = [fresh.contrast(rows, AFTER, arm) for arm in arms if arm not in (AFTER, POINT, FULL)]
    indexed = {(row['seed'], row['arm']):row for row in rows}
    primary_rows = []
    for pair in ap['rows']:
        a, p = indexed[pair['seed'], AFTER], indexed[pair['seed'], POINT]
        primary_rows.append(dict(seed=pair['seed'], domain=pair['domain'], metrics=pair['old_contrast'],
            candidate=a['old'], reference=p['old'],
            candidate_status=a['old_status'], reference_status=p['old_status'],
            quality_reference_status=a['quality_reference_status'], best_initial_quality=a['best_initial_quality'],
            candidate_deployed=a.get('old_deployed'), reference_deployed=p.get('old_deployed'),
            origin_contributions={origin:difference(a['origin_contributions'][origin]['old'],
                p['origin_contributions'][origin]['old']) for origin in ORIGINS}))
    origins = []
    for origin in ORIGINS:
        selected = [dict(seed=row['seed'], domain=row['domain'], arm=row['arm'],
            status=row['status'], **row['origin_contributions'][origin]) for row in rows]
        origins.append(dict(origin=origin, AFTER_minus_POINT=fresh.contrast(selected, AFTER, POINT),
            FULL_minus_AFTER=fresh.contrast(selected, FULL, AFTER)))
    return dict(registered_geometries=len(plan['tasks']), rows=rows, arms=groups,
        primary=dict(left=AFTER, right=POINT, protocol='old', co_reported_endpoints=['C','G'],
            metrics=ap['old_contrast'], rows=primary_rows,
            by_family=[dict(domain=f['domain'], metrics=f['old_contrast']) for f in ap['by_family']]),
        secondary=dict(AFTER_minus_POINT=ap, FULL_minus_AFTER=fa, AFTER_minus_controls=controls),
        origin_contributions=origins)


def directions(stream_rows):
    indexed = [{(r['seed'],r['arm']):r for r in rows} for rows in stream_rows]
    results = []
    for label,left,right,field in (
        ('AFTER_minus_POINT_old',AFTER,POINT,'old'), ('AFTER_minus_POINT_new',AFTER,POINT,'new'),
        ('FULL_minus_AFTER_old',FULL,AFTER,'old'), ('FULL_minus_AFTER_new',FULL,AFTER,'new'),
        ('AFTER_new_minus_old',AFTER,None,'new_minus_old')):
        rows = []
        for seed in dict.fromkeys(r['seed'] for r in stream_rows[0]):
            effects = [index[seed,left][field] if right is None else
                difference(index[seed,left][field],index[seed,right][field]) for index in indexed]
            rows.append(dict(seed=seed, domain=indexed[0][seed,left]['domain'], stream_effects=effects,
                directions={m:[sign(effect[m]) for effect in effects] for m in ('C','R','G')}))
        results.append(dict(contrast=label, rows=rows, direction_counts={m:dict(Counter(
            '/'.join(row['directions'][m]) for row in rows)) for m in ('C','R','G')}))
    return results


def summarize_loaded(plan, streams, reference):
    """Pure saved-metric interface; both complete grids include explicit failures."""
    if len(streams) != 2:
        raise ValueError('Both registered new streams are required')
    arms = arm_names(plan)
    expected = {(t['geometry_task_seed'],a) for t in plan['tasks'] for a in arms}
    expected_outcomes = {(s,a,p) for s,a in expected for p in ('old','new')}
    units, rows_by_stream, outcomes, accounts, mechanisms = [], [], [], [], []
    for stream in streams:
        summary = stream['summary']
        rows = summary['rows']
        if len(rows) != len(expected) or {(r['seed'],r['arm']) for r in rows} != expected:
            raise ValueError('Every stream must retain the registered geometry-arm grid')
        saved = summary['outcomes']
        if len(saved) != len(expected_outcomes) or {(r['seed'],r['arm'],r['protocol']) for r in saved} != expected_outcomes:
            raise ValueError('Both terminal outcomes, including failures, are required for every cell')
        rows = origin_rows(rows)
        rows_by_stream.append(rows)
        units.append(dict(name=stream['name'], replicate=stream['replicate'], status=summary['status'],
            **summarize_unit(plan, rows)))
        outcomes.extend(dict(stream=stream['name'], **deepcopy(row)) for row in saved)
        accounts.append(dict(name=stream['name'], replicate=stream['replicate'],
            output_dir=stream.get('output_dir'), completion=deepcopy(stream['completion']),
            scored_costs=deepcopy(summary.get('costs',{}))))
        mechanisms.append(dict(name=stream['name'], replicate=stream['replicate'],
            aggregate=deepcopy(summary.get('mechanism',{}).get('aggregate')),
            audit_failures=deepcopy(summary.get('costs',{}).get('analytic_truth',{}).get('mechanism_stage',{}).get('failures',[])),
            scope='Saved per-stream competition diagnostics; repeated prefixes/policies are not independent samples.'))
    averaged = geometry_average(rows_by_stream)
    indexed = [{(r['seed'],r['arm']):r for r in rows} for rows in rows_by_stream]
    for row in averaged:
        sources = [index[row['seed'],row['arm']] for index in indexed]
        for protocol in ('old','new'):
            row[protocol+'_status'] = ('complete' if all(s[protocol+'_status']=='complete' for s in sources)
                                      else 'incomplete_streams')
    failures = [row for row in outcomes if row['status'] != 'complete']
    false = [dict(stream=stream['name'], seed=row['seed'], domain=row['domain'], arm=row['arm'],
        protocol=protocol, reported_deployed=row.get(protocol+'_reported_deployed'), metrics=row[protocol])
        for stream in streams for row in stream['summary']['rows'] for protocol in ('old','new')
        if row[protocol]['false_certificate'] is not None and row[protocol]['false_certificate'] > 0]
    status = ('complete_with_missing_cells' if failures else 'complete_with_audit_failures'
              if any(stream['summary']['status'] != 'complete' for stream in streams) else 'complete')
    return dict(status=status,
        statistical_unit=plan['statistical_unit'], registered_geometry_count=len(plan['tasks']),
        registered_streams=2, registered_outcome_cells=2*len(expected_outcomes),
        completed_outcome_cells=sum(row['status']=='complete' for row in outcomes),
        primary_definition=plan['analysis']['primary'],
        geometry_averaged=summarize_unit(plan, averaged), streams=units,
        replication_directions=directions(rows_by_stream), outcomes=outcomes,
        failures=failures, false_certificates=false, mechanism=mechanisms,
        reference=dict(role='Opened Stage73 stream, descriptive context only; excluded from every new-stream estimate.',
            manifest=plan['reference_manifest'], primary=deepcopy(reference['primary']),
            secondary=deepcopy(reference['secondary']), arms=deepcopy(reference['arms'])),
        costs=dict(streams=accounts, historical_accounts=deepcopy(plan.get('historical_accounts',[]))),
        limits=['Two new streams test observation and verification noise on 24 fixed geometries, not new geometry generalization.',
            'Each required stream is averaged within geometry before uncertainty summaries. Missing either stream metric makes that geometry metric null.',
            'Undefined initial baselines remain null; complete-cohort and available-subset estimates keep separate denominators.',
            'Every origin contribution is signed deployment accounting, not a causal intervention.',
            'All 9 controls and both terminals, failed outcomes, false certificates and actual costs remain in the report.',
            'The old/default AFTER-minus-POINT C and signed G comparison remains primary; new terminal results are secondary.'])


def compact_metrics(values):
    return {metric:fresh.compact_estimate(values[metric]) for metric in METRICS}


def compact_contrast(group):
    return dict(left=group['left'], right=group['right'],
        **{field:compact_metrics(group[field]) for field in ('old_contrast','new_contrast','change')},
        by_family=[dict(domain=row['domain'], **{field:compact_metrics(row[field])
            for field in ('old_contrast','new_contrast','change')}) for row in group['by_family']])


def compact_unit(unit):
    primary = unit['primary']
    tasks = []
    for row in primary['rows']:
        tasks.append(dict(row, arms=[dict(arm=r['arm'], status=r['status'],
            old_status=r['old_status'], new_status=r['new_status'], old=r['old'], new=r['new'],
            new_minus_old=r['new_minus_old'], old_deployed=r.get('old_deployed'), new_deployed=r.get('new_deployed'))
            for r in unit['rows'] if r['seed']==row['seed']]))
    return dict(registered_geometries=unit['registered_geometries'],
        primary=dict(left=AFTER, right=POINT, protocol='old', co_reported_endpoints=['C','G'],
            metrics=compact_metrics(primary['metrics']), by_family=[dict(domain=r['domain'],
                metrics=compact_metrics(r['metrics'])) for r in primary['by_family']]),
        arms=[dict(arm=g['arm'], registered=g['registered'], complete=g['complete'],
            **{f:compact_metrics(g[f]) for f in FIELDS}) for g in unit['arms']],
        secondary=dict(AFTER_minus_POINT=compact_contrast(unit['secondary']['AFTER_minus_POINT']),
            FULL_minus_AFTER=compact_contrast(unit['secondary']['FULL_minus_AFTER']),
            AFTER_minus_controls=[compact_contrast(g) for g in unit['secondary']['AFTER_minus_controls']]),
        origin_contributions=[dict(origin=g['origin'], AFTER_minus_POINT=compact_contrast(g['AFTER_minus_POINT']),
            FULL_minus_AFTER=compact_contrast(g['FULL_minus_AFTER'])) for g in unit['origin_contributions']], tasks=tasks)


def compact_summary(result):
    costs = dict(historical_accounts=result['costs']['historical_accounts'], streams=[])
    for source in result['costs']['streams']:
        account = source['completion']
        costs['streams'].append(dict(name=source['name'], replicate=source['replicate'],
            completion_file=source['output_dir']+'/completion.json',
            **{key:account.get(key) for key in ('new_work','new_physical_target_calls',
                'recorded_cpu_seconds','recorded_attempt_wall_seconds','terminal_costs','execution_failures')},
            failed_attempt_count=len(account.get('failed_attempts',[])),
            failed_attempts=account.get('failed_attempts',[]),
            logical=source['scored_costs'].get('logical'),
            analytic_truth=source['scored_costs'].get('analytic_truth')))
    if 'parent_account' in result['costs']:
        costs['parent_account'] = {key:value for key,value in result['costs']['parent_account'].items()
                                  if key != 'parent_attempts'}
    if 'validation_account' in result['costs']:
        costs['validation_account'] = result['costs']['validation_account']
    return dict(status=result['status'], statistical_unit=result['statistical_unit'],
        registered_geometry_count=result['registered_geometry_count'], registered_streams=2,
        registered_outcome_cells=result['registered_outcome_cells'], completed_outcome_cells=result['completed_outcome_cells'],
        primary_definition=result['primary_definition'], geometry_averaged=compact_unit(result['geometry_averaged']),
        streams=[dict(name=s['name'], replicate=s['replicate'], status=s['status'], **compact_unit(s)) for s in result['streams']],
        replication_directions=deepcopy(result['replication_directions']),
        cell_accounts=[{key:row.get(key) for key in ('stream','seed','domain','arm','protocol','status','costs','files')}
                       for row in result['outcomes']],
        failures=deepcopy(result['failures']), false_certificates=deepcopy(result['false_certificates']),
        mechanism=deepcopy(result['mechanism']), terminal_recording=deepcopy(result.get('terminal_recording')),
        reference=dict(role=result['reference']['role'],
            manifest=result['reference']['manifest'], primary=dict(left=AFTER,right=POINT,protocol='old',
                metrics=compact_metrics(result['reference']['primary']['metrics']))),
        costs=costs, limits=result['limits'], detailed_summary_file='summary.json', completion_file='completion.json')


def completion(plan, result, parent_attempts):
    accounts = result['costs']['streams']
    child = [row['completion'] for row in accounts]
    keys = set().union(*(a['new_work'] for a in child))
    work = {}
    for key in keys:
        values = [a['new_work'].get(key) for a in child]
        if key == 'randomized_new_observations':
            work[key] = any(values)
        else:
            work[key] = None if any(v is None for v in values) else sum(values)
    parent_cpu = sum(r.get('cpu_seconds',0.) for r in parent_attempts)
    child_cpu = sum(a['recorded_cpu_seconds'] for a in child)
    physical = sum(a['new_physical_target_calls'] for a in child)
    return dict(status=result['status'], completed_utc=datetime.now(timezone.utc).isoformat(),
        registered_geometries=len(plan['tasks']), registered_new_streams=2,
        registered_outcome_cells=result['registered_outcome_cells'], completed_outcome_cells=result['completed_outcome_cells'],
        new_work=work, new_physical_target_calls=physical,
        recorded_cpu_seconds=parent_cpu+child_cpu, recorded_parent_cpu_seconds=parent_cpu,
        recorded_child_cpu_seconds=child_cpu, parent_attempts=parent_attempts,
        failed_parent_attempts=[r for r in parent_attempts if r['status']=='failed'],
        streams=accounts, execution_failure_cells=result['failures'],
        historical_accounts=plan.get('historical_accounts',[]),
        accounting_note='New physical and analytic calls are the sum of two finalized child accounts. Recorded CPU adds only child accounts and current parent-only seal, terminal-audit and aggregation attempts. Historical registration accounts are references; their CPU/calls are not added. Validation and controller/wrapper CPU are separately reported because wrapper CPU overlaps its child process.')


def summarize(plan):
    """Read child scoring outputs only; caller has already enforced both global seals."""
    output = ROOT/plan['output_dir']
    streams = []
    for descriptor in plan['streams']:
        child = read(ROOT/descriptor['manifest'])
        directory = ROOT/child['output_dir']
        streams.append(dict(name=descriptor['name'], replicate=descriptor['replicate'],
            output_dir=child['output_dir'], summary=read(directory/'mainline_summary.json'),
            completion=read(directory/'completion.json')))
    reference_plan = read(ROOT/plan['reference_manifest'])
    reference = read(ROOT/reference_plan['output_dir']/'mainline_summary.json')
    with attempt(output/'aggregation', 'aggregate-recorded-terminal-replication') as cost:
        cost.update(ZERO)
        result = summarize_loaded(plan, streams, reference)
    children = [ROOT/stream['output_dir'] for stream in streams]
    parent_attempts = []
    # Registration belongs to the previous preparation stage. Current outer
    # work has separate, nonnested metered scopes; wrapper CPU is not added.
    paths = [path for scope in ('mechanism','terminal_recording','aggregation')
             for path in (output/scope).rglob('attempts.jsonl')]
    for path in sorted(paths):
        if any(path.is_relative_to(child) for child in children):
            continue
        parent_attempts.extend(dict(json.loads(line), ledger=str(path.relative_to(ROOT)))
                               for line in path.read_text().splitlines() if line.strip())
    account = completion(plan,result,parent_attempts)
    checks = output/'cluster_control/validation/attempt1/checks.json'
    if checks.exists():
        validation = dict(file=str(checks.relative_to(ROOT)), account=read(checks),
            included_in_recorded_cpu_seconds=False,
            wrapper_account='cluster_control/jobs/validation/attempt1_0_1.json',
            accounting='Validation subprocess CPU is reported separately; its job wrapper overlaps it.')
        account['validation_account'] = validation
        result['costs']['validation_account'] = validation
    result['terminal_recording'] = read(output/'terminal_recording/summary.json')
    result['costs']['parent_account'] = {key:value for key,value in account.items() if key not in ('streams','execution_failure_cells')}
    save(output/'summary.json',result)
    save(output/'completion.json',account)
    (output/'compact_summary.json').write_text(json.dumps(compact_summary(result),separators=(',',':'))+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default=str(MANIFEST))
    args = parser.parse_args()
    if socket.gethostname().split('.')[0] not in {f'node{i:03d}' for i in range(1,7)}:
        raise RuntimeError('Run replication aggregation through scheduler on node001-node006')
    print(summarize(read(args.manifest))['status'],flush=True)


if __name__ == '__main__':
    main()

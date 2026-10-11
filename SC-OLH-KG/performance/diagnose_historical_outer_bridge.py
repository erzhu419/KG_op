"""Reuse corrected risk references in historical outer rules and action bounds."""
import argparse
import json
import resource
import signal
import time
from collections import defaultdict

import numpy as np

from core.raw_terminal_head_grid import summarize_achievable_grid
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.diagnose_historical_decision_regions import PLAN, OUTPUT as REGIONS, objective_upper, overlay_direct
from performance.extend_historical_upper_risk import head_rules
from performance.refine_historical_achievable_precision import assemble_case, compose_records, rule_pair
from performance.assemble_historical_risk_references import (
    OUTPUT as ASSEMBLY, load_entries, apply_entry, reference_masks, entry_summary)

OUTPUT = 'paper_artifacts/historical_outer_bridge_v1'
CONFIGURATIONS = [(16, 32, 32), (32, 16, 32), (32, 32, 16), (32, 32, 32)]


def outer_selections(spec, rules, origins):
    # The historical API takes objective, retained tail, then body order.
    return {(objective,body,tail):head_rules(spec, rules, origins, objective, tail, body)
        for objective,body,tail in CONFIGURATIONS}


def outer_diagnostics(rows, tolerance):
    fine = rows[-1]; field = 'raw_net_reward_of_projected_choice_control_variate'
    changes = {axis:float(abs(fine[field]-row[field]))
        for axis,row in zip(('objective16_32','body16_32','tail16_32'),rows[:-1])}
    passed = bool(max(changes.values()) <= tolerance and abs(fine['raw_fixed_action_tower_residual']) <= tolerance)
    return dict(outer_changes=changes,finite_outer_diagnostics_passed=passed)


def competition_envelope(values, upper, covered, weights, groups, margin=2e-4):
    """Conditional envelope with referenced values fixed, not an error certificate."""
    values, upper, covered, weights = map(np.asarray, (values, upper, covered, weights))
    assert covered.any()
    known = np.max(np.where(covered[None, :], values, -np.inf), axis=-1)
    unseen = np.where(~covered[None, :], upper, -np.inf)
    excess = np.maximum(np.max(unseen, axis=-1)-known, 0.)
    overlap = (~covered[None, :]) & (upper+margin >= known[:, None])
    priority = []
    for group in groups:
        ix = group['action_indices']
        if not overlap[:, ix].any():
            continue
        potential = np.maximum(np.max(unseen[:, ix], axis=-1)-known, 0.)
        priority.append(dict(group=group['group'],
            action_head_pairs=int(overlap[:, ix].sum()),
            weighted_potential_uplift=float(weights @ potential)))
    return dict(known_best=known, potential_uplift=float(weights @ excess),
        overlap_probability_mass=float(weights[overlap.any(axis=-1)].sum()),
        overlapping_action_head_pairs=int(overlap.sum()), groups=priority)


def run(plan='performance/manifests/historical_outer_bridge_v1.json'):
    started = time.process_time()
    signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, 240.)
    out = ROOT/read(plan)['output']
    out.mkdir(parents=True, exist_ok=True)
    spec = read(PLAN); precision = read(spec['precision_plan']); case = precision['cases'][0]
    heads, bank, _ = assemble_case(precision, case, read(precision['output']+'/registered_jobs.json')['jobs'])
    prior = np.asarray(bank['prior_weights'])
    layout = read('paper_artifacts/historical_cdf_bank_v1/bank_layout.json')
    base = spec['accepted_baseline_inputs'][case['state']]+'/'+case['state']+'/n1/baseline'
    debited = read(base+'/terminal_rules.json')['rows'][-1]
    stop = read(spec['stop_inputs'][case['state']])['rows'][-1]
    assert layout['actions'] == debited['actions'] and len(layout['actions']) == 365
    fixed = debited['best_action_index']
    direct = read(REGIONS+'/direct_summary.json'); native = read(REGIONS+'/native_reference_summary.json')
    assert direct['complete'] and direct['all_references_admitted']
    assert native['complete'] and native['all_references_admitted']
    direct_names = {r['record'] for r in direct['records']}
    native_names = {r['record'] for r in native['records']}
    direct_contexts = read(REGIONS+'/direct_contexts.json'); native_contexts = read(REGIONS+'/contexts.json')
    rules = read(spec['output']+'/risk_rules.json')
    entries = load_entries(layout); by_record = defaultdict(list)
    for (name, kind, _), entry in entries.items():
        by_record[name, kind].append(entry)
    selections = outer_selections(spec, rules, bank['origins'])
    totals = defaultdict(list); coverage = []; priorities = defaultdict(float)
    expected_decisions = {}; maximum_upper_violation = 0.
    for origin in bank['origins']:
        f = heads[0, origin]
        assert [r['position'] for r in f] == list(range(32))
        positions = sorted({p for selection in selections.values() for p in selection[1, origin]['positions']})
        risk_index = {r['position']:r for r in heads[1, origin]}
        g = [risk_index[p] for p in positions]; masks = {}
        for row in g:
            name = f'source_n1_o{origin}_h1_p{row["position"]}'
            if name in direct_names:
                context = direct_contexts[name]
                assert all(row[k] == context[k] for k in ('label', 'head_log_scores'))
                assert context['prior_weights'] == list(prior)
                parts = {h:read(f'{REGIONS}/direct_parts/{name}/{h}.json') for h in bank['utility_identities']}
                overlay_direct(row, parts, layout['groups'][136]['action_indices'])
            if name in native_names:
                context = native_contexts[name]
                assert all(row[k] == context[k] for k in ('label', 'head_log_scores'))
                assert context['prior_weights'] == list(prior)
                parts = {h:read(f'{REGIONS}/parts/{name}/063_{h}_projected.json') for h in bank['utility_identities']}
                overlay_direct(row, parts, layout['groups'][63]['action_indices'], kind='projected', pair=('F8','E8'))
            for kind in ('raw', 'projected'):
                applied = by_record[name, kind]
                for entry in applied:
                    apply_entry(row, f, prior, entry, origin)
                mask = np.logical_or.reduce(list(reference_masks(layout, applied).values()))
                if kind == 'raw' and name in direct_names:
                    mask[layout['groups'][136]['action_indices']] = True
                if kind == 'projected':
                    ix = [i for i,a in enumerate(layout['actions']) if a['kind'] == 'direct']
                    levels = [row[kind][str(p)] for p in rule_pair(row, kind)]
                    assert all(np.max(abs(np.asarray(levels[0][k])[:,ix]-np.asarray(levels[1][k])[:,ix])) <= 1e-12
                        for k in ('statistics','costs'))
                    mask[ix] = True
                    if name in native_names:
                        mask[layout['groups'][63]['action_indices']] = True
                masks[row['position'],kind] = mask
        raw, costs, posterior, _ = compose_records(f, g, prior, 'raw', 1)
        projected, pcosts, pposterior, _ = compose_records(f, g, prior, 'projected', 1)
        np.testing.assert_array_equal(posterior, pposterior)
        gi = {p:i for i,p in enumerate(positions)}
        for cfg, selection in selections.items():
            fx = np.asarray(selection[0,origin]['positions']); gx = np.asarray([gi[p] for p in selection[1,origin]['positions']])
            ix = np.ix_(fx,gx)
            weights = prior[origin]*np.outer(selection[0,origin]['weights'],selection[1,origin]['weights'])
            totals[cfg].append(summarize_achievable_grid(raw[ix],costs[ix],projected[ix],pcosts[ix],posterior[ix],weights,
                (prior[origin]*stop['best_reward'],prior[origin]*debited['best_reward'],fixed),'corrected_cached',len(fx)+len(gx)))
        fine = selections[32,32,32]
        for j,p in enumerate(fine[1,origin]['positions']):
            k = gi[p]; name = f'source_n1_o{origin}_h1_p{p}'
            expected_decisions[name] = dict(raw_winners=sorted(map(int,np.unique(np.lexsort((costs[:,k],-raw[:,k]),axis=-1)[:,0]))),
                native_winners=sorted(map(int,np.unique(np.lexsort((pcosts[:,k],-projected[:,k]),axis=-1)[:,0]))))
        for kind, values in (('raw',raw),('projected',projected)):
            upper = objective_upper([r[kind][str(rule_pair(r,kind)[1])]['statistics'] for r in f],posterior,bank['utility_identities'])
            maximum_upper_violation = max(maximum_upper_violation,float(np.max(values-upper)))
            for order in (16,32):
                selection = selections[32,order,32]
                for j,p in enumerate(selection[1,origin]['positions'][:order]):
                    k = gi[p]; mask = masks[p,kind]
                    weights = prior[origin]*selection[1,origin]['weights'][j]*np.asarray(selection[0,origin]['weights'])
                    envelope = competition_envelope(values[:,k],upper[:,k],mask,weights,layout['groups'])
                    row = dict(origin=origin,position=p,kind=kind,body_order=order,reference_actions=int(mask.sum()),
                        **{key:envelope[key] for key in ('potential_uplift','overlap_probability_mass','overlapping_action_head_pairs')})
                    coverage.append(row)
                    if order == 32:
                        for priority in envelope['groups']:
                            priorities[kind,priority['group']] += priority['weighted_potential_uplift']
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(origin=origin,cpu_seconds=time.process_time()-started)),flush=True)
    keys = ['raw_adaptation_value','raw_net_reward_control_variate','raw_adaptation_of_projected_choice',
        'raw_net_reward_of_projected_choice_control_variate','action_selection_loss',
        'raw_fixed_action_tower_residual','quadrature_joint_probability_mass']
    rows = [dict(objective_prefix=f,body_order=b,tail_order=t,**{key:sum(r[key] for r in totals[f,b,t]) for key in keys})
        for f,b,t in CONFIGURATIONS]
    fine = rows[-1]; field = 'raw_net_reward_of_projected_choice_control_variate'
    assembled = read(ASSEMBLY+'/summary.json')
    assert assembled['prior_weights'] == prior.tolist() and assembled['reference_assembly']['groups'] == len(entries)
    difference = abs(fine[field]-assembled['raw_model_net_preview'])
    assert difference <= 1e-12
    assert all(expected_decisions[r['record']] == {k:r[k] for k in ('raw_winners','native_winners')} for r in assembled['records'])
    envelopes = {f'{kind}_body{order}':{key:sum(r[key] for r in coverage if (r['kind'],r['body_order']) == (kind,order))
        for key in ('potential_uplift','overlap_probability_mass','overlapping_action_head_pairs')}
        for kind in ('raw','projected') for order in (16,32)}
    result = dict(complete=True,rows=rows,**outer_diagnostics(rows,spec['maximum_numerical_adaptation_difference']),
        original_assembly_net_difference=difference,original_128_recommendations_unchanged=True,
        reference_assembly=entry_summary(entries),conditional_missing_action_envelopes=envelopes,
        priority_groups=[dict(kind=k,group=g,weighted_potential_uplift=v) for (k,g),v in sorted(priorities.items(),key=lambda item:-item[1])[:12]],
        coverage=coverage,maximum_observed_objective_upper_violation=maximum_upper_violation,
        action_count=365,query=63,remaining_budget=359,prior_weights=prior.tolist(),
        omitted_predictive_origin_mass=float(sum(w for k,w in enumerate(prior) if k not in bank['origins'])),
        cpu_seconds=time.process_time()-started,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        new_integrations=0,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,
        full_predictive_integration_admitted=False,independent_confirmation=False,
        scope='Corrected cached16/32 outer comparisons, with original positive weights and all identity scores. '
            'An outer-rule comparison still uses unreferenced historical competitors. Conditional missing-action envelopes '
            'hold matched cached objective tables and selected numerical references fixed; they are neither quadrature '
            'error certificates nor confidence intervals. The2e-4 overlap margin is the historical screening margin. '
            'Body16 and body32 coverage are separate. No continuous outerKG or independent decision claim.')
    save(out/'summary.json',result)
    signal.setitimer(signal.ITIMER_PROF,0.)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k not in ('coverage','prior_weights')}),flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--plan',default='performance/manifests/historical_outer_bridge_v1.json')
    args = parser.parse_args()
    try:
        run(args.plan)
    finally:
        print('SCHEDULEURM_RESOURCE '+json.dumps(dict(process_cpu_seconds=time.process_time(),
            peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)

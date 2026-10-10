"""Attach existing numerical references to the opened risk grid, without integration."""
from collections import Counter

import numpy as np

from performance.complete_historical_source_heads import ROOT, read
from performance.refine_historical_achievable_precision import rule_pair
from performance.validate_historical_component_transfer import part_path
from performance.validate_historical_label_control import tables

OUTPUT = 'paper_artifacts/historical_body_reference_assembly_v1'
SCOPES = ('direct_target', 'paired_residual', 'native_analytic')


def bank_group_tables(context, group, bank, kind, actions):
    ix = group['action_indices']
    s, c = np.asarray(bank['statistics'][kind]), np.asarray(bank['costs'][kind])
    return tables(dict(context, actions=[actions[i] for i in ix]),
        {h:dict(statistics=s[h, ix], costs=c[h, ix]) for h in context['utility_identities']})


def add_entry(entries, context, group, kind, statistics, costs, scope, source, method, opposite=None):
    key = (context['name'], kind, group['group'])
    entry = dict(context=context, group=group, kind=kind, statistics=np.asarray(statistics),
        costs=np.asarray(costs), scope=scope, sources=[source], method=method, opposite=opposite)
    if key in entries:
        old = entries[key]
        sources = list(dict.fromkeys(old['sources'] + [source]))
        # A direct target reference has wider numerical evidence than a paired
        # residual comparison. Neither becomes a new independent experiment.
        if old['scope'] != 'paired_residual' and scope == 'paired_residual':
            old['sources'] = sources
            return
        entry['sources'] = sources
    entries[key] = entry


def load_entries(layout):
    entries = {}; groups = layout['groups']
    base = ROOT/'paper_artifacts/historical_component_bank_v1'
    context = read(str(base/'context.json'))
    assert context['actions'] == layout['actions'] and context['groups'] == groups
    result = read(str(base/'bank_tail_resolution_summary.json'))
    assert result['complete'] and result['single_record_risk_bank_precision_passed']
    bank = read(str(base/'bank-tail-resolution_risk_bank.json'))
    for g in result['groups']:
        for kind in ('raw', 'projected'):
            assert g['checks'][kind]['passed']
            ix = g['action_indices']
            scope = 'native_analytic' if g['selected_rules'][kind] == 'native_reference' else 'direct_target'
            add_entry(entries, context, groups[g['group']], kind,
                *bank_group_tables(context, g, bank, kind, layout['actions']),
                scope, str(base/'bank_tail_resolution_summary.json'), g['selected_rules'][kind],
                [dict(r, **{k:np.asarray(r[k])[:, ix] for k in ('raw', 'projected')}) for r in context['opposite']])

    base = ROOT/'paper_artifacts/historical_label_bank_v1'
    assert read(str(base/'bank_layout.json')) == layout
    result = read(str(base/'gh_full_summary.json'))
    assert result['complete'] and result['all_label_comparisons_passed']
    checked = {(r['record'], r['group'], r['kind']) for r in read(str(base/'gh_full_groups.json')) if r['checks']['passed']}
    contexts = read(str(base/'contexts.json')); bank = read(str(base/'independent_risk_bank.json'))
    for row in read(str(base/'independent_groups.json')):
        assert row['valid_unclipped_ranges'] and row['coarse_fine_comparison_passed'] and row['independent_comparison_passed']
        if row['selected_method'] == 'native_reference': scope = 'native_analytic'
        elif row['target_precision_admitted']: scope = 'direct_target'
        else:
            assert (row['record'], row['group'], row['kind']) in checked
            scope = 'paired_residual'
        kind = row['kind']; ix = row['action_indices']; actual = bank[row['record']]
        add_entry(entries, contexts[row['record']], groups[row['group']], kind,
            *bank_group_tables(contexts[row['record']], groups[row['group']], actual, kind, layout['actions']),
            scope, str(base/'gh_full_summary.json'), row['selected_method'])

    for directory in ('historical_label_groups_v1', 'historical_body_reference_pilot_v1',
            'historical_nearby_anchor_v1'):
        base = ROOT/'paper_artifacts'/directory
        result = read(str(base/'summary.json')); assert result['complete']
        contexts = read(str(base/'contexts.json'))
        for row in result['groups']:
            if not row['hybrid_direct_admitted']: continue
            assert row['direct_reference_checks']['passed']
            add_entry(entries, contexts[row['record']], groups[row['group']], row['kind'],
                row['selected_statistics'], row['selected_costs'], 'direct_target',
                str(base/'summary.json'), row['selected_method'])

    base = ROOT/'paper_artifacts/historical_label_control_v1'
    result = read(str(base/'summary.json')); assert result['complete']
    contexts = read(str(base/'contexts.json'))
    for row in result['groups']:
        assert row['checks']['passed']
        context = contexts[row['record']]; group = groups[row['group']]
        cached = {}
        for h in context['utility_identities']:
            cached[h] = {}
            for directory in ('historical_component_transfer_v1', 'historical_label_control_v1'):
                path = part_path(ROOT/'paper_artifacts'/directory, row, h)
                if path.exists(): cached[h].update(read(str(path))['rows'])
        data = tables(dict(context, actions=[layout['actions'][i] for i in group['action_indices']]),
            {h:cached[h][row['selected_rule']] for h in context['utility_identities']})
        add_entry(entries, context, group, row['kind'], *data, 'direct_target',
            str(base/'summary.json'), 'direct_'+row['selected_rule'])

    base = ROOT/'paper_artifacts/historical_cdf_bank_v1'
    assert read(str(base/'bank_layout.json')) == layout
    result = read(str(base/'summary.json')); assert result['complete'] and result['all_pilot_references_admitted']
    contexts = read(str(base/'contexts.json')); bank = read(str(base/'selected_risk_bank.json'))
    # The remaining coarse-bank groups are deliberately absent: a complete
    # screen is not a complete numerical reference.
    selected = [(r, 'direct_target') for r in result['groups']]
    selected += [(r, 'native_analytic') for r in read(str(base/'coarse_groups.json')) if r['selected_method'] == 'native_reference']
    for row, scope in selected:
        if scope == 'direct_target': assert row['direct_reference_checks']['passed'] and row['selected_method'] != 'HOLD'
        kind = row['kind']; ix = row['action_indices']; actual = bank[row['record']]
        add_entry(entries, contexts[row['record']], groups[row['group']], kind,
            *bank_group_tables(contexts[row['record']], groups[row['group']], actual, kind, layout['actions']),
            scope, str(base/'summary.json'), row['selected_method'])

    base = ROOT/'paper_artifacts/historical_upper_decision_refs_v1'
    result = read(str(base/'full_summary.json')); assert result['complete'] and result['all_references_admitted']
    contexts = read(str(base/'contexts.json'))
    for row in result['checks']:
        assert row['checks']['passed']
        context = contexts[row['record']]; group = groups[row['group']]
        data = tables(dict(context, actions=[layout['actions'][i] for i in group['action_indices']]),
            {h:read(str(base/'parts'/row['record']/f'{row["group"]}_{h}.json'))['rows']['E7'] for h in context['utility_identities']})
        add_entry(entries, context, group, 'raw', *data, 'direct_target', str(base/'full_summary.json'), 'E7')
    return entries


def apply_entry(row, objectives, prior, entry, origin):
    context = entry['context']; kind = entry['kind']; ix = entry['group']['action_indices']
    assert context['name'] == f'source_n1_o{origin}_h1_p{row["position"]}'
    for key in ('label', 'head_log_scores', 'utility_identities', 'replicates'):
        assert row[key] == context[key], key
    assert row['head'] == 1 and row['eta_label'] is None and context['prior_weights'] == list(prior)
    local = context.get('groups', {})
    local = local[str(entry['group']['group'])] if isinstance(local, dict) else None
    if local is not None: assert local['action_indices'] == ix
    opposite = entry['opposite'] if entry['opposite'] is not None else local['opposite']
    objective_index = {r['position']:r for r in objectives}
    assert set(objective_index) == {r['position'] for r in opposite}
    for reference in opposite:
        actual = objective_index[reference['position']]
        assert actual['head_log_scores'] == reference['head_log_scores']
        np.testing.assert_array_equal(np.asarray(actual[kind][str(rule_pair(actual, kind)[1])]['statistics'])[:, ix], reference[kind])
    value = row[kind][str(rule_pair(row, kind)[1])]
    s, c = entry['statistics'], entry['costs']; active = row['utility_identities']
    assert s.shape == (len(prior), len(ix), 4) and c.shape == (len(prior), len(ix))
    assert np.isfinite(s).all() and np.isfinite(c).all() and s[active].min() >= -1e-12 and s[active].max() <= 1+1e-12
    assert c[active].min() >= 0 and c[active].max() <= 359
    for h in range(len(prior)):
        for j, a in enumerate(ix):
            value['statistics'][h][a] = s[h, j].tolist(); value['costs'][h][a] = float(c[h, j])


def reference_masks(layout, applied):
    masks = {scope:np.zeros(len(layout['actions']), dtype=bool) for scope in SCOPES}
    for entry in applied: masks[entry['scope']][entry['group']['action_indices']] = True
    for scope in ('direct_target', 'native_analytic'): masks['paired_residual'][masks[scope]] = False
    return masks


def remaining_risk_checks(difference, dominated, masks):
    covered = np.logical_or.reduce(list(masks.values()))
    missing = (difference > 1e-4) & ~covered[None, :]
    relevant = missing & ~dominated
    return dict(reference_actions={k:int(v.sum()) for k,v in masks.items()},
        unreferenced_actions=int((~covered).sum()), unreferenced_imprecise_action_head_pairs=int(missing.sum()),
        unreferenced_relevant_action_head_pairs=int(relevant.sum()),
        unreferenced_imprecise_action_indices=np.flatnonzero(missing.any(axis=0)).tolist(),
        unreferenced_relevant_action_indices=np.flatnonzero(relevant.any(axis=0)).tolist())


def entry_summary(entries):
    return dict(groups=len(entries), group_scopes=dict(Counter(r['scope'] for r in entries.values())),
        groups_by_source=dict(Counter(source for r in entries.values() for source in r['sources'])))

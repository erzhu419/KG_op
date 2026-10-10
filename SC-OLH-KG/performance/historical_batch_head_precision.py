"""Attribute cached full-bank batch errors using the existing head-grid law."""

import argparse
import json
import signal
import time

import numpy as np

from core.raw_terminal_head_grid import evaluate_head_records, compose_head_grid
from core.conditional_terminal_kg import ConditionalTerminalActions
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.historical_full_bank_batch_kg import restore

PLAN = 'performance/manifests/historical_batch_head_precision_v1.json'


def action_key(action):
    return action['kind'], tuple(action['shortlist']), action['incumbent_position'], tuple(action['budgets'])


def run(name, head, plan=PLAN, shard=None):
    spec = read(plan); out = ROOT/spec['output']/name/str(head)
    if shard is not None:
        out = out/('shard'+str(shard))
    out.mkdir(parents=True, exist_ok=True)
    if (out/'results.json').exists() and read(str(out/'results.json')).get('complete'):
        return read(str(out/'results.json'))
    started = time.process_time(); handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_case_cpu_seconds'])
    result = {'name': name, 'head': head, 'complete': False, 'new_physical_calls': 0}
    try:
        anchor = read(spec['anchors']+'/'+name+'/future_records.json')['records'][0]
        weights = np.asarray(anchor['shared_identity_weights'])
        identities = list(range(len(weights)))
        for identity in np.argsort(weights):
            trial = [i for i in identities if i != identity]
            if 1-weights[trial].sum() <= spec['maximum_identity_integration_error_bound']:
                identities = trial
        assert anchor['replicates'] == spec['replicates']
        if spec.get('require_archived_parity', True):
            assert anchor['guarded_mean_position'] == spec['guarded_mean_position']
        model, terminal, state, restoration = restore(spec, name)
        if shard is not None:
            selected = np.arange(shard, len(terminal.shortlists), spec['shards'])
            terminal = ConditionalTerminalActions(terminal.shortlists[selected], terminal.incumbents[selected], terminal.protocol)
        completed = None
        if str(head) in spec.get('completed_inputs', {}):
            cached = read(spec['completed_inputs'][str(head)]+'/'+name+'/'+str(head)+'/head_records.json')
            completed = cached['records']
            if shard is not None:
                from core.raw_likelihood_lookahead import _actions
                positions = {action_key(a): i for i,a in enumerate(cached['actions'])}
                selected = [positions[action_key(a)] for a in _actions(terminal, 360-spec['replicates'])]
                for r in completed:
                    for raw in r['raw'].values():
                        raw['statistics'] = np.asarray(raw['statistics'])[:, selected].tolist()
                        raw['costs'] = np.asarray(raw['costs'])[:, selected].tolist()
        records = evaluate_head_records(model, terminal, 360, spec['query'], head,
            [anchor['labels'][head]], [0], powers=spec['powers'], seed=spec['seed'], completed=completed,
            utility_identities=identities, guarded_mean_position=spec['guarded_mean_position'],
            replicates=spec['replicates'], eta_labels=[anchor['eta_labels'][head]],
            checkpoint=lambda r, a: save(out/'head_records.json', {'records': r, 'actions': a,
                'prior_weights': model.weights.tolist(), 'cached_observation_anchor': True}))
        result.update(complete=True, native_subset_replay_passed=True, restoration=restoration,
            shard=shard,
            utility_identities=identities, diagnostics={p: r['diagnostics_by_identity']
                for p, r in records[0]['raw'].items()})
    except TimeoutError:
        result['cpu_budget_stopped'] = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
        result['cpu_seconds'] = time.process_time()-started
        save(out/'results.json', result)
    print('SCHEDULEURM_DONE '+json.dumps(result), flush=True)
    return result


def merge_shards(spec, name, head):
    directory = ROOT/spec['output']/name/str(head)
    shards = [read(str((directory/('shard'+str(i))/'head_records.json').relative_to(ROOT)))
        for i in range(spec['shards'])]
    executions = [read(str((directory/('shard'+str(i))/'results.json').relative_to(ROOT)))
        for i in range(spec['shards'])]
    assert all(r['complete'] for r in executions)
    cached = read(spec['completed_inputs'][str(head)]+'/'+name+'/'+str(head)+'/head_records.json')
    positions = {action_key(a): i for i,a in enumerate(cached['actions'])}
    first = shards[0]['records'][0]; record = dict(first, raw={})
    coverage = []
    for s in shards:
        row = s['records'][0]
        assert row['head_log_scores'] == first['head_log_scores']
        assert row['label'] == first['label'] and row['eta_label'] == first['eta_label']
        assert row['utility_identities'] == first['utility_identities']
        assert s['prior_weights'] == cached['prior_weights']
        coverage.extend(positions[action_key(a)] for a in s['actions'])
    assert sorted(coverage) == list(range(len(positions)))
    for power in spec['powers']:
        old = first['raw'][str(power)]; n = len(old['statistics'])
        statistics = np.empty((n, len(positions), 4)); costs = np.empty((n, len(positions)))
        for s in shards:
            ids = [positions[action_key(a)] for a in s['actions']]; raw = s['records'][0]['raw'][str(power)]
            statistics[:, ids] = raw['statistics']; costs[:, ids] = raw['costs']
            assert raw['diagnostics_by_identity'] == old['diagnostics_by_identity']
        record['raw'][str(power)] = dict(old, statistics=statistics.tolist(), costs=costs.tolist())
    save(directory/'head_records.json', dict(cached, records=[record]))
    save(directory/'results.json', dict(name=name, head=head, complete=True,
        shards=executions, cpu_seconds=sum(r['cpu_seconds'] for r in executions), new_physical_calls=0))


def collect(plan=PLAN):
    spec = read(plan); rows = []; jobs = []
    for name in spec['states']:
        for head in spec.get('sharded_heads', []):
            merge_shards(spec, name, head)
        directories = [spec.get('head_inputs', {}).get(str(h), spec['output'])+'/'+name+'/'+str(h)
            for h in (0, 1)]
        heads = [read(d+'/head_records.json') for d in directories]
        jobs.extend(read(d+'/results.json') for d in directories)
        assert all(r['complete'] for r in jobs)
        assert heads[0]['actions'] == heads[1]['actions']
        np.testing.assert_array_equal(heads[0]['prior_weights'], heads[1]['prior_weights'])
        fr, gr = [h['records'][0] for h in heads]
        assert fr['utility_identities'] == gr['utility_identities']
        anchor_file = read(spec['anchors']+'/'+name+'/future_records.json')
        anchor = anchor_file['records'][0]; actions = anchor_file['actions']
        assert actions == heads[0]['actions']
        refined_primaries = []
        if 'failed_guarded_input' in spec:
            old = read(spec['failed_guarded_input']+'/'+name+'/future_records.json')['records'][0]
            difference = abs(np.asarray(old['raw']['16']['values'])-old['raw']['14']['values'])
            refined_primaries = sorted({a['primary'] for a,d in zip(actions,difference)
                if a['kind']=='guarded' and d > spec['maximum_action_resolution_difference']})
            original = read(spec['original_risk_input']+'/'+name+'/1/head_records.json')['records'][0]
            retained = [i for i,a in enumerate(actions) if a['kind']=='direct' or a['primary'] not in refined_primaries]
            for power, raw in gr['raw'].items():
                for key in ('statistics','costs'):
                    values = np.asarray(raw[key]); values[:, retained] = np.asarray(original['raw'][power][key])[:, retained]
                    raw[key] = values.tolist()
        def compose(fp, gp):
            f = np.asarray(fr['raw'][str(fp)]['statistics'])[None]
            g = np.asarray(gr['raw'][str(gp)]['statistics'])[None]
            c = np.asarray(gr['raw'][str(gp)]['costs'])[None]
            v, c, w, _ = compose_head_grid(f, g, c, [fr['head_log_scores']],
                [gr['head_log_scores']], heads[0]['prior_weights'])
            return v[0, 0], c[0, 0], w[0, 0]
        fp, gp = [spec.get('head_powers', {}).get(str(h), spec['powers']) for h in (0, 1)]
        low, _, _ = compose(fp[0], gp[0]); fine, costs, weights = compose(fp[-1], gp[-1])
        f_low, _, _ = compose(fp[0], gp[-1]); g_low, _, _ = compose(fp[-1], gp[0])
        archived = np.asarray(anchor['raw'][str(spec.get('archived_power', spec['powers'][-1]))]['values'])
        parity = float(np.max(abs(fine-archived)))
        # Later precision stages keep the archived old law for a comparison,
        # while attribution of the unchanged law must reproduce it exactly.
        if spec.get('require_archived_parity', True):
            assert parity <= 1e-12
        np.testing.assert_allclose(weights, anchor['shared_identity_weights'], atol=1e-12, rtol=0.)
        d = abs(fine-low); df = abs(fine-f_low); dg = abs(fine-g_low)
        ids = fr['utility_identities']; omitted = float(sum(w for i, w in enumerate(weights) if i not in ids))
        diagnostics = {str(h): r['raw'][str(p[-1])]['diagnostics_by_identity']
            for h, r, p in zip((0, 1), (fr, gr), (fp, gp))}
        norm = max(abs(d['predictive_log_evidence_error']) for h in diagnostics.values() for d in h if d)
        selected = int(np.lexsort((costs, -fine))[0]); worst = int(d.argmax())
        rows.append({'name': name, 'head_powers': [fp, gp], 'action_count': len(actions),
            'incumbent_mean_refined_primaries': refined_primaries,
            'maximum_archived_action_difference': parity,
            'maximum_action_resolution_difference': float(d.max()),
            'maximum_objective_contribution_difference': float(df.max()),
            'maximum_risk_contribution_difference': float(dg.max()),
            'direct_resolution_difference': max(float(v) for v,a in zip(d,actions) if a['kind']=='direct'),
            'worst_action': actions[worst], 'worst_objective_difference': float(df[worst]),
            'worst_risk_difference': float(dg[worst]), 'selected_action': actions[selected],
            'selected_value': float(fine[selected]), 'selected_resolution_difference': float(d[selected]),
            'predictive_log_evidence_max_error': norm, 'diagnostics_by_head_identity': diagnostics,
            'omitted_utility_mass': omitted,
            'precision_passed': bool(d.max() <= spec['maximum_action_resolution_difference']
                and norm <= 1e-4 and omitted <= spec['maximum_identity_integration_error_bound']),
            'top_errors': [dict(action=actions[i], difference=float(d[i]),
                objective_difference=float(df[i]), risk_difference=float(dg[i]), gap=float(fine[selected]-fine[i]))
                for i in np.argsort(-d)[:5]]})
    result = {'rows': rows, 'jobs': jobs, 'cpu_seconds': sum(r['cpu_seconds'] for r in jobs),
        'new_physical_calls': 0, 'new_source_fits': 0, 'new_verifier_calls': 0,
        'independent_confirmation': False, 'algorithmic_query_selection_complete': False}
    save(ROOT/spec['output']/'outcome_summary.json', result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['head', 'collect'])
    parser.add_argument('--name'); parser.add_argument('--head', type=int, choices=[0, 1])
    parser.add_argument('--shard', type=int)
    parser.add_argument('--plan', default=PLAN); args = parser.parse_args()
    if args.mode == 'collect': collect(args.plan)
    else:
        if args.name is None or args.head is None: parser.error('--name and --head are required')
        run(args.name, args.head, args.plan, args.shard)

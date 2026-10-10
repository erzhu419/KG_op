"""Reuse admitted observed-coordinate records and extend the full bank."""

import argparse
import copy
import json
import pickle
import signal
import time

import numpy as np

from core.observed_batch_predictive import predictive_observed_batches
from core.raw_likelihood_lookahead import _actions
from core.raw_terminal_head_grid import evaluate_head_records
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.historical_full_bank_batch_kg import restore
from performance.refine_historical_achievable_precision import assemble_case, original_heads

PLAN = 'performance/manifests/historical_predictive_extension_v1.json'


def head_labels(bank, n, origin, head):
    batch = bank['batches'][str(n)]; count = batch['count']
    part = slice(origin*count, (origin+1)*count)
    labels = np.asarray(batch['labels'])[part, head]
    eta = None if batch['eta'] is None else np.asarray(batch['eta'])[part, head]
    assert len(labels) == count and (eta is None or len(eta) == count)
    return labels, eta


def merge_records(old, pieces, actions, count):
    rows = {r['position']: r for r in old}
    for piece in pieces:
        assert piece['actions'] == actions
        for row in piece['records']:
            assert row['position'] not in rows
            rows[row['position']] = row
    assert sorted(rows) == list(range(count))
    return [rows[p] for p in range(count)]


def prepare(name, plan=PLAN):
    spec = read(plan); old_spec = read(spec['reuse_precision_plan'])
    parent = read(old_spec['parent_plan']); out = ROOT/spec['output']/name
    out.mkdir(parents=True, exist_ok=True)
    if (out/'preparation.json').exists(): return
    started = time.process_time(); snapshot = ROOT/spec['posterior_inputs'][name]
    reused_snapshot = snapshot.exists()
    restored = restore(spec if reused_snapshot else parent, name); model, terminal, _, _ = restored
    cases = [c for c in spec['cases'] if c['state'] == name]
    # The expensive restoration and complete covariance stay on the server.
    if not reused_snapshot:
        with snapshot.open('wb') as f: pickle.dump(restored, f)
    batches = {}; jobs = []; identities = None; origins = None
    old_registry = read(old_spec['output']+'/registered_jobs.json')['jobs']
    for case in cases:
        previous_case = next(c for c in old_spec['cases'] if c['name'] == case['name'])
        original_bank = read(parent['output']+'/'+name+'/bank.json')
        origins = original_bank['origins']; identities = original_bank['utility_identities']
        count = 2**case['predictive_power']
        labels, eta, weights = predictive_observed_batches(model, parent['query'], case['n'],
            case['predictive_power'], parent['predictive_seed'])
        assert abs(weights.sum()-1.) <= 1e-12
        batches[str(case['n'])] = dict(count=count, labels=labels.tolist(), eta=None if eta is None else eta.tolist())
        current = dict(batches=batches)
        admitted = assemble_case(old_spec, previous_case, old_registry)[0]
        for origin in origins:
            actions = _actions(terminal, 360-case['n'])
            historical_actions = original_heads(old_spec, previous_case, origin)[0]['actions']
            assert actions == historical_actions
            for head in (0, 1):
                old = copy.deepcopy(admitted.get((head, origin), []))
                y, t = head_labels(current, case['n'], origin, head)
                if old:
                    matched = np.max(abs(y[:len(old)]-[r['label'] for r in old])) <= 1e-8
                    if t is not None:
                        matched = matched and all(r['eta_label'] is not None for r in old) and \
                            np.max(abs(t[:len(old)]-[r['eta_label'] for r in old])) <= 1e-8
                    if case['n'] == 1: assert matched
                    # Historical batch proxies require new observed-coordinate records.
                    if not matched: old = []
                if old:
                    # Retain historical labels and every already admitted rule.
                    for p, row in enumerate(old):
                        row['position'] = p
                        batches[str(case['n'])]['labels'][origin*count+p][head] = row['label']
                        if t is not None:
                            batches[str(case['n'])]['eta'][origin*count+p][head] = row['eta_label']
                directory = out/('n'+str(case['n']))/'head'/(str(head)+'_'+str(origin))
                directory.mkdir(parents=True, exist_ok=True)
                save(directory/'head_records.json', dict(records=old, actions=actions, prior_weights=model.weights.tolist()))
                for p in range(len(old), case['head_counts'][head]):
                    jobs.append(dict(name=case['name']+'_o'+str(origin)+'_h'+str(head)+'_p'+str(p),
                        case=case['name'], state=name, n=case['n'], origin=origin, head=head, position=p))
    save(out/'bank.json', dict(batches=batches, prior_weights=model.weights.tolist(),
        utility_identities=identities, origins=origins, query=parent['query'], truth_used=False))
    result = dict(complete=True, state=name, jobs=jobs, cpu_seconds=time.process_time()-started,
        source_state_restored_once=True, posterior_snapshot_reused=reused_snapshot,
        new_physical_calls=0, new_source_fits=0, new_verifier_calls=0)
    save(out/'preparation.json', result)
    print('SCHEDULEURM_DONE '+json.dumps({k:v for k,v in result.items() if k!='jobs'}), flush=True)


def run(name, plan=PLAN):
    spec = read(plan); out = ROOT/spec['output']
    case = next(c for c in spec['cases'] if name.startswith(c['name']+'_o'))
    job = next(j for j in read(spec['output']+'/'+case['state']+'/preparation.json')['jobs'] if j['name'] == name)
    directory = out/'new_heads'/name; directory.mkdir(parents=True, exist_ok=True)
    if (directory/'results.json').exists() and read(str(directory/'results.json')).get('complete'): return
    started = time.process_time(); handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_case_cpu_seconds'])
    result = dict(name=name, complete=False, new_physical_calls=0, new_source_fits=0, new_verifier_calls=0)
    try:
        model, terminal, _, _ = restore(spec, job['state'])
        bank = read(spec['output']+'/'+job['state']+'/bank.json')
        y, eta = head_labels(bank, job['n'], job['origin'], job['head']); p = job['position']
        path = directory/'head_records.json'; cached = read(str(path))['records'] if path.exists() else []
        expected = read(spec['output']+'/'+job['state']+'/n'+str(job['n'])+'/head/'+str(job['head'])+'_'+str(job['origin'])+'/head_records.json')['actions']
        def checkpoint(records, actions):
            assert actions == expected
            save(path, dict(records=records, actions=actions, prior_weights=model.weights.tolist()))
        evaluate_head_records(model, terminal, 360, spec['query'], job['head'], {p:y[p]}, [p],
            powers=spec['powers'], projected_powers=spec['powers'], seed=spec['seed'], completed=cached,
            utility_identities=bank['utility_identities'], guarded_mean_position=0, replicates=job['n'],
            eta_labels=None if eta is None else {p:eta[p]},
            checkpoint=checkpoint)
        result.update(complete=True, action_count=len(expected), posterior_restored_from_server_snapshot=True)
    except TimeoutError: result['cpu_budget_stopped'] = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
        result['cpu_seconds'] = time.process_time()-started; save(directory/'results.json', result)
    print('SCHEDULEURM_DONE '+json.dumps(result), flush=True)


def assemble(plan=PLAN):
    spec = read(plan); costs = []
    for case in spec['cases']:
        state = case['state']; prep = read(spec['output']+'/'+state+'/preparation.json')
        for origin in read(spec['output']+'/'+state+'/bank.json')['origins']:
            for head in (0, 1):
                path = ROOT/spec['output']/state/('n'+str(case['n']))/'head'/(str(head)+'_'+str(origin))/'head_records.json'
                before = read(str(path)); pieces = []
                for job in prep['jobs']:
                    if (job['case'], job['origin'], job['head']) != (case['name'], origin, head): continue
                    directory = spec['output']+'/new_heads/'+job['name']; result = read(directory+'/results.json')
                    assert result['complete']; costs.append(result)
                    piece = read(directory+'/head_records.json')
                    if piece['records'][0]['position'] not in {r['position'] for r in before['records']}: pieces.append(piece)
                rows = merge_records(before['records'], pieces, before['actions'], case['head_counts'][head])
                save(path, dict(before, records=rows))
    preparations = [read(spec['output']+'/'+s+'/preparation.json') for s in {c['state'] for c in spec['cases']}]
    result = dict(complete=True, executions=costs, new_head_states=len(costs),
        cpu_seconds=sum(r['cpu_seconds'] for r in costs+preparations), new_physical_calls=0,
        new_source_fits=0, new_verifier_calls=0, independent_confirmation=False)
    save(ROOT/spec['output']/'execution_summary.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='executions'}), flush=True)


def compact_cases(result):
    cases=[]
    for case in result['cases']:
        fine=case['rows'][-1]
        cases.append(dict(case=case['case'],replicates=case['replicates'],action_count=case['action_count'],
            prefixes=fine['prefixes'],net=fine['raw_net_reward_of_projected_choice_control_variate'],
            raw_optimal_net=fine['raw_net_reward_control_variate'],
            native_choice_loss=fine['action_selection_loss'],
            projected_bias_at_selected_action=fine['projected_value_bias_at_selected_action'],
            fixed_action_tower_residual=fine['raw_fixed_action_tower_residual'],
            numerical_changes=case['numerical_changes'],gates=case['gates'],
            terminal_diagnostics_passed=case['terminal_diagnostics_passed'],
            fixed_action_predictive_tower_diagnostic_passed=case['fixed_action_predictive_tower_diagnostic_passed'],
            full_predictive_integration_admitted=case['full_predictive_integration_admitted'],
            maximum_raw_action_difference=fine['maximum_raw_action_resolution_difference'],
            maximum_native_action_difference=fine['maximum_native_action_resolution_difference'],
            predictive_log_evidence_error=case['predictive_log_evidence_error']))
    return cases


def collect_initial(plan=PLAN):
    from performance.refine_historical_achievable_precision import collect
    started=time.process_time();spec=read(plan);precision=read(spec['precision_plan'])
    result=collect(spec['precision_plan'])
    save(ROOT/precision['output']/'results_at16.json',result)
    summary=dict(cases=compact_cases(result),scope='Development preview with unresolved terminal precision; not admitted.',
        collection_cpu_seconds=time.process_time()-started,new_physical_calls=0,new_source_fits=0,
        new_verifier_calls=0,independent_confirmation=False)
    save(ROOT/precision['output']/'initial_precision_summary.json',summary)
    print('SCHEDULEURM_DONE '+json.dumps(dict(complete=True,cpu_seconds=summary['collection_cpu_seconds'],
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0)),flush=True)


def finish(power,plan=PLAN):
    from performance.refine_historical_achievable_policies import merge
    from performance.refine_historical_achievable_precision import collect
    started=time.process_time();spec=read(plan);precision=read(spec['precision_plan'])
    merge(power,spec['precision_plan']);result=collect(spec['precision_plan'])
    save(ROOT/precision['output']/('results_at'+str(power)+'.json'),result)
    summary=dict(cases=compact_cases(result),scope='Frozen-query development, same365 actions and original gates.',
        collection_cpu_seconds=time.process_time()-started,new_physical_calls=0,new_source_fits=0,
        new_verifier_calls=0,independent_confirmation=False)
    save(ROOT/precision['output']/('precision_summary_at'+str(power)+'.json'),summary)
    print('SCHEDULEURM_DONE '+json.dumps(dict(complete=True,cpu_seconds=summary['collection_cpu_seconds'],
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0)),flush=True)


def extend(plan=PLAN):
    """Register unused, already validated quantiles without restoring a model."""
    started=time.process_time();spec=read(plan);added=[]
    for case in spec['cases']:
        directory=ROOT/spec['output']/case['state'];prep=read(str(directory/'preparation.json'))
        bank=read(str(directory/'bank.json'));known={j['name'] for j in prep['jobs']}
        for origin in bank['origins']:
            for head in (0,1):
                before=read(str(directory/('n'+str(case['n']))/'head'/(str(head)+'_'+str(origin))/'head_records.json'))
                assert case['head_counts'][head]<=bank['batches'][str(case['n'])]['count']
                for p in range(len(before['records']),case['head_counts'][head]):
                    name=case['name']+'_o'+str(origin)+'_h'+str(head)+'_p'+str(p)
                    if name in known:continue
                    job=dict(name=name,case=case['name'],state=case['state'],n=case['n'],origin=origin,head=head,position=p)
                    prep['jobs'].append(job);added.append(job);known.add(name)
        save(directory/'preparation.json',prep)
    result=dict(complete=True,jobs=added,cpu_seconds=time.process_time()-started,
        new_physical_calls=0,new_source_fits=0,new_verifier_calls=0,posterior_restorations=0)
    save(ROOT/spec['output']/'risk32_registered_jobs.json',result)
    print('SCHEDULEURM_DONE '+json.dumps(dict({k:v for k,v in result.items() if k!='jobs'},new_head_states=len(added))),flush=True)


def stage_risk(plan):
    """Keep the closed inputs intact while reusing their admitted rules."""
    from performance.refine_historical_achievable_precision import prepare as prepare_precision
    started=time.process_time();spec=read(plan);old=read(spec['reuse_precision_plan'])
    precision=read(spec['precision_plan']);registry=read(old['output']+'/registered_jobs.json')['jobs']
    previous=read(old['output']+'/outcome_summary.json')
    for case in spec['cases']:
        assert next(c for c in previous['cases'] if c['case']==case['name'])['terminal_diagnostics_passed']
        old_case=next(c for c in old['cases'] if c['name']==case['name'])
        heads,bank,_=assemble_case(old,old_case,registry)
        prep=read(spec['output']+'/'+case['state']+'/preparation.json')
        for origin in bank['origins']:
            data=original_heads(old,old_case,origin)
            for head in (0,1):
                admitted=heads[head,origin];pieces=[]
                for job in prep['jobs']:
                    if (job['case'],job['origin'],job['head'])!=(case['name'],origin,head):continue
                    if job['position']<len(admitted):continue
                    directory=spec['output']+'/new_heads/'+job['name']
                    assert read(directory+'/results.json')['complete']
                    pieces.append(read(directory+'/head_records.json'))
                rows=merge_records(admitted,pieces,data[head]['actions'],case['head_counts'][head])
                path=ROOT/(precision['head_inputs'][case['name']][head].format(origin=origin))
                if 'head_inputs' in old:
                    assert path!=ROOT/(old['head_inputs'][case['name']][head].format(origin=origin))
                path.mkdir(parents=True,exist_ok=True)
                save(path/'head_records.json',dict(data[head],records=rows))
    jobs=prepare_precision(spec['precision_plan'])
    result=dict(complete=True,precision_jobs=len(jobs),old_inputs_preserved=True,
        cpu_seconds=time.process_time()-started,new_physical_calls=0,new_source_fits=0,new_verifier_calls=0)
    save(ROOT/precision['output']/'staging_summary.json',result)
    print('SCHEDULEURM_DONE '+json.dumps(result),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['prepare', 'case', 'assemble', 'collect', 'extend', 'finish', 'stage-risk'])
    parser.add_argument('--name'); parser.add_argument('--plan', default=PLAN)
    parser.add_argument('--power',type=int,default=18);args = parser.parse_args()
    if args.mode == 'prepare': prepare(args.name, args.plan)
    elif args.mode == 'case': run(args.name, args.plan)
    elif args.mode=='assemble': assemble(args.plan)
    elif args.mode=='collect': collect_initial(args.plan)
    elif args.mode=='extend': extend(args.plan)
    elif args.mode=='stage-risk':stage_risk(args.plan)
    else: finish(args.power,args.plan)

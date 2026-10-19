"""Registered new-geometry role-prior closed loops with the V69 verifier."""
import argparse
from datetime import datetime, timezone
import json
from shutil import copyfile
import signal
import time

import numpy as np

from performance.mainline_role_backend import build_fresh_backend, reload_backend_bundle, STOP
from performance.mainline_terminal_recording import snapshot
from performance.mainline_shared_neighbors import make_shared_neighbors, shared_neighbor_candidates
from performance.run_mainline_legacy_backend_bridge import (
    ROOT, attempt, count_simulations, load_pickle, read, save, save_engine,
)
from performance.run_mainline_local_noise_control import case_dir, cases, require_all


MANIFEST = ROOT/'performance/manifests/mainline_role_fresh_v1.json'


def search(output, plan, seed, arm):
    directory = case_dir(output, seed, arm)
    observations = []
    with attempt(directory, 'search', plan['domain'], arm) as cost:
        cost.update(seed=seed,new_target_initial_calls=0,new_source_fits=0,download_bytes=0)
        if (directory/'search.json').exists():
            cost['reused']=True
            return
        if list(directory.glob('failed_search_raw_*.json')):
            raise RuntimeError('Recover the saved paid state before repeating a failed search')
        initial=[]
        if arm!=STOP:
            for s in plan['target_seeds']:
                read(output/f'seed_{s}'/'neighbors.json')
            initial=read(case_dir(output,seed,STOP)/'search.json')['history']
        engine=build_fresh_backend(plan,seed,arm,initial)
        fixed_initialization=plan.get('fixed_initialization_file_by_task',{}).get(str(seed))
        if fixed_initialization:
            # Reuse physical policies only. STOP draws this replicate's raws;
            # all other histories replay that newly generated STOP history.
            frozen_rows=read(ROOT/fixed_initialization)['history']
            assert len(frozen_rows)==10
            fixed_points=tuple(tuple(map(int,row['point'])) for row in frozen_rows)
            engine.config.initial_design_points=fixed_points
        geometry=plan['geometry_by_seed'][str(seed)]['problem_kwargs']
        target=engine.problem.base.base
        np.testing.assert_array_equal(target.task_geometry_shift,geometry['task_geometry_shift'])
        assert target.task_geometry_radius_scale==geometry['task_geometry_radius_scale']
        assert engine.problem.d==1000 and not engine.config.truth_pool_diagnostics
        points=[] if arm==STOP else np.load(output/f'seed_{seed}'/'neighbors.npz')['points']
        field='new_target_initial_calls' if arm==STOP else 'new_target_search_calls'
        audit=None
        try:
            with shared_neighbor_candidates(engine,points) as audit:
                with count_simulations(engine,cost,field,observations):
                    engine.run_search()
        except BaseException:
            failure_id=time.time_ns()
            raw_path=directory/f'failed_search_raw_{failure_id}.json'
            save(raw_path,observations)
            recording = snapshot(engine)
            if recording is not None:
                save(directory/f'failed_terminal_recording_{failure_id}.json',
                     dict(terminal_recording=recording, paid_raw_file=raw_path.name,
                          returned_raw_count=len(observations), truth_used=False))
            if audit is not None:
                bank_path=directory/f'failed_query_policies_{failure_id}.npz'
                np.savez_compressed(bank_path,points=np.asarray(audit['points'],dtype=np.int16))
                save(directory/f'failed_query_audit_{failure_id}.json',{
                    **{key:value for key,value in audit.items() if key!='points'},
                    'status':'partial','paid_raw_file':raw_path.name,
                    'query_policies_file':bank_path.name,'paths_relative_to':'audit_directory',
                    'returned_raw_count':len(observations),'truth_used':False,
                    'stage_semantics':'Pre-observation decision snapshots; a captured stage need not have a returned observation.',
                })
            save_engine(directory/'failed_search_engine.pkl',engine)
            raise
        save(directory/'paid_search_raw.json',observations)
        assert cost[field]==(10 if arm==STOP else 3)
        history=[{'point':list(p),'raw':np.asarray(y).tolist()} for p,y in engine.history]
        assert arm==STOP or history[:10]==initial
        if fixed_initialization:
            assert tuple(tuple(row['point']) for row in history[:10])==fixed_points
        assert [s['paid_prefix'] for s in audit['stages']]==([] if arm==STOP else [10,11,12])
        if arm!=STOP:
            for stage,row in zip(audit['stages'],engine.iteration_log,strict=True):
                assert list(audit['points'][stage['selected_id']])==row['x_selected']==history[stage['paid_prefix']]['point']
                assert set(audit['shared_ids']).issubset(stage['action_ids'])
            np.savez_compressed(directory/'query_policies.npz',points=np.asarray(audit.pop('points'),dtype=np.int16))
            save(directory/'query_audit.json',audit)
        result={'seed':seed,'arm':arm,'family':plan['domain'],**engine.frozen_terminal,
            'geometry':geometry,'history':history,'iterations':engine.iteration_log,
            'backend_trace':getattr(engine,'backend_trace',[]),
            'role_restoration_trace':getattr(engine,'role_restoration_trace',[]),
            'logical_search_calls':len(history),'new_initial_calls':cost['new_target_initial_calls'],
            'new_search_calls':cost['new_target_search_calls'],'initial_design_seed':80,
            'truth_used':False,'frozen_utc':datetime.now(timezone.utc).isoformat()}
        result['observation_seed']=getattr(engine.config,'observation_seed',None)
        if result['observation_seed'] is None:
            result['observation_seed']=seed
        if fixed_initialization:
            result['fixed_initialization_file']=fixed_initialization
        save_engine(directory/'search_engine.pkl',engine)
        save(directory/'search.json',result)
        if arm==STOP:
            save(output/f'seed_{seed}'/'initialization.json',{'history':history,'geometry':geometry,
                'observation_seed':result['observation_seed'],
                'fixed_initialization_file':fixed_initialization})


def prepare(output,plan):
    for seed in plan['target_seeds']:
        directory=output/f'seed_{seed}'
        with attempt(directory,'prepare-neighbors',plan['domain']) as cost:
            cost.update(seed=seed,new_source_fits=0,download_bytes=0)
            if (directory/'neighbors.json').exists():
                cost['reused']=True
                continue
            fixed_neighbors=plan.get('fixed_neighbors_file_by_task',{}).get(str(seed))
            if fixed_neighbors:
                source=ROOT/fixed_neighbors
                # Preserve the registered physical bank and its original
                # metadata, independent of the new initialization responses.
                copyfile(source,directory/'neighbors.npz')
                copyfile(source.with_suffix('.json'),directory/'neighbors.json')
                cost['fixed_neighbors_file']=fixed_neighbors
                continue
            engine=load_pickle(case_dir(output,seed,STOP)/'search_engine.pkl')
            neighbors=make_shared_neighbors(engine,seed,plan['neighbor_count'])
            assert neighbors['metadata']['radius']==plan['neighbor_radius']
            assert neighbors['metadata']['safe_margin_scale']==plan['neighbor_safe_margin_scale']
            np.savez_compressed(directory/'neighbors.npz',points=np.asarray(neighbors['points'],dtype=np.int16))
            save(directory/'neighbors.json',{**neighbors['metadata'],
                'geometry':plan['geometry_by_seed'][str(seed)],'frozen_utc':datetime.now(timezone.utc).isoformat()})


def verify(output,plan,seed,arm):
    directory=case_dir(output,seed,arm); observations=[]
    with attempt(directory,'verify',plan['domain'],arm) as cost:
        cost.update(seed=seed,new_source_fits=0,download_bytes=0)
        if (directory/'verification.json').exists():
            cost['reused']=True
            return
        require_all(output,plan,'search.json')
        if list(directory.glob('failed_verification_raw_*.json')):
            raise RuntimeError('Recover the saved verification before repeating paid observations')
        result=read(directory/'search.json')
        engine=load_pickle(directory/'search_engine.pkl')
        previous_restorations=len(getattr(engine,'role_restoration_trace',[]))
        reload_backend_bundle(engine, plan, arm)
        engine.defer_terminal=False
        try:
            with count_simulations(engine,cost,'new_verification_calls',observations):
                deployed,verified=engine._terminal_verification_protocol(tuple(result['primary']))
        except BaseException:
            save(directory/f'failed_verification_raw_{time.time_ns()}.json',observations)
            save_engine(directory/'failed_verification_engine.pkl',engine)
            raise
        save(directory/'paid_verification_raw.json',observations)
        assert [r['point'] for r in verified['frozen_shortlist']]==[r['point'] for r in result['terminal']['frozen_shortlist']]
        assert cost['new_verification_calls']==verified['verification_budget']
        assert len(engine.history)+cost['new_verification_calls']<=(362 if arm==STOP else 365)
        save(directory/'verification.json',{'seed':seed,'arm':arm,'primary':result['primary'],
            'deployed':list(map(int,deployed)),'verification':verified,
            'new_verification_calls':cost['new_verification_calls'],'truth_used':False,
            'role_restoration_trace':getattr(engine,'role_restoration_trace',[])[previous_restorations:],
            'completed_utc':datetime.now(timezone.utc).isoformat()})


def complete(output,plan):
    require_all(output,plan,'verification.json')
    results,diagnostic=read(output/'results.json'),read(output/'diagnosis.json')
    rows=[json.loads(line) for p in output.rglob('attempts.jsonl') for line in p.read_text().splitlines()]
    fields=('cpu_seconds','wall_seconds','new_source_calls','new_target_initial_calls',
            'new_target_search_calls','new_verification_calls','new_source_fits','download_bytes','new_truth_function_calls')
    cost={k:sum(r.get(k,0) for r in rows) for k in fields}
    integration_events=[event for seed,arm in cases(plan)
        for name in ('search.json','verification.json')
        for trace in read(case_dir(output,seed,arm)/name).get('role_restoration_trace',[])
        for event in trace['roundoff_events']]
    save(output/'completion.json',{'status':'complete','recommendations':len(results['rows']),
        'attempt_count':len(rows),'failed_attempts':[r for r in rows if r['status']=='failed'],
        'cost':cost,'role_filter_roundoff_events':sum(e['integration_status']==2 for e in integration_events),
        'role_filter_interval_limit_events':sum(e['integration_status']==1 for e in integration_events),
        'new_physical_target_calls':sum(cost[k] for k in
            ('new_target_initial_calls','new_target_search_calls','new_verification_calls')),
        'logical_target_calls':sum(r['logical_total_calls'] for r in results['rows']),
        'completed_utc':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__':
    # Let the existing paid-stage exception handlers save raw/state on SIGTERM.
    # A default SIGTERM exits before those handlers and attempt ledgers run.
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'External termination signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['search','prepare','verify','complete'])
    p.add_argument('--seed',type=int);p.add_argument('--arm')
    p.add_argument('--manifest',default=str(MANIFEST))
    args=p.parse_args();plan=read(args.manifest);output=ROOT/plan['output_dir']
    if args.stage in ('search','verify'):
        assert (args.seed,args.arm) in cases(plan)
        {'search':search,'verify':verify}[args.stage](output,plan,args.seed,args.arm)
    else:
        {'prepare':prepare,'complete':complete}[args.stage](output,plan)
    print(args.stage,args.seed,args.arm,'complete',flush=True)

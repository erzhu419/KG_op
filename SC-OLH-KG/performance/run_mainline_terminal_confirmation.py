"""New-geometry search once per history, followed by crossed frozen terminals."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

from performance import run_mainline_terminal_equal as verifier
from performance.mainline_terminal_recording import snapshot
from performance.diagnose_mainline_terminal_backend import freeze
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, load_pickle, read, save

MANIFEST = ROOT/'performance/manifests/mainline_terminal_confirmation_v1.json'
ROLES = ('legacy_role_full', 'legacy_role_after_init_point_only')


def input_record(search, frozen, parent, cell, seed, verification_seed, source_case, tau):
    shortlist = frozen['frozen_shortlist']
    assert len(shortlist)==3 and frozen['objective_incumbent_guard_enabled']
    assert frozen['candidate_verification_budgets']==[80,128,128]
    assert frozen['max_verification_budget']==352
    assert frozen['objective_comparison_budget_per_policy']==8
    assert frozen['familywise_delta']==.05
    assert frozen['objective_comparison']['delta']==.05/3
    assert search['geometry']==parent['geometry_by_seed'][str(seed)]['problem_kwargs']
    assert search['logical_search_calls']==(10 if cell['history_arm']=='initialization_only' else 13)
    observed={tuple(r['point']) for r in search['history']}
    assert all(tuple(r['point']) in observed for r in shortlist)
    result = dict(seed=seed,arm=cell['name'],history_arm=cell['history_arm'],
        terminal_backend=cell['terminal_backend'],protocol='original',
        domain=parent['domain'],task=parent['task'],geometry=search['geometry'],
        verification_seed=verification_seed,primary=shortlist[0]['point'],
        initial=search['history'][0]['point'],frozen_shortlist=shortlist,
        candidate_budgets=[80,128,128],search_evaluation_count=search['logical_search_calls'],
        familywise_delta=frozen['familywise_delta'],
        method=parent['decision_protocol']['verification_method'],
        mean_delta_fraction=frozen['mean_delta_fraction'],shortlist_mode=frozen['shortlist_mode'],
        objective_incumbent_position=frozen['objective_incumbent_position'],
        objective_comparison_budget=8,objective_comparison_delta=.05/3,
        tau=tau,source_case=source_case,truth_used=False)
    if frozen.get('terminal_recording') is not None:
        result['terminal_recording'] = deepcopy(frozen['terminal_recording'])
    return result


def preflight(root, terminal_plan):
    """Check the new input adapter against three existing real frozen cases."""
    with attempt(root,'input-adapter-preflight') as cost:
        cost.update(new_source_fits=0,new_truth_function_calls=0,download_bytes=0)
        old=read(ROOT/'performance/manifests/mainline_terminal_cross_v1.json')
        parent=read(ROOT/old['source_manifest']); seed=2026101736
        checked=[]
        for name in ('legacy_role_full','legacy_role_full_native_terminal','initialization_only'):
            expected=read(verifier.case_dir(ROOT/old['output_dir'],seed,name,'original')/'input.json')
            cell=next(a for a in old['arms'] if a['name']==name)
            search=read(ROOT/expected['source_case']/'search.json')
            frozen=search['terminal'] if cell['terminal_backend']!='native' else read(
                ROOT/old['decision_output_dir']/f'seed_{seed}'/cell['history_arm']/'decisions.json')['native']['terminal']
            actual=input_record(search,frozen,parent,cell,seed,expected['verification_seed'],
                                expected['source_case'],expected['tau'])
            assert all(actual[k]==expected[k] for k in actual), 'New input adapter changed verifier contract'
            checked.append(name)
        evidence=ROOT/old['output_dir']/'replay_check.json'
        assert read(evidence)['status']=='passed'
        save(ROOT/terminal_plan['output_dir']/'replay_check.json',dict(status='passed',
            inherited_adapter_replay=str(evidence.relative_to(ROOT)),historical_replayed_cases=84,
            new_histories_replayed=0,new_input_cases_checked=checked,new_simulation_calls=0))


def native_terminal(source_plan, seed, arm):
    directory=ROOT/source_plan['output_dir']/f'seed_{seed}'/arm
    if (directory/'native_terminal.json').exists():
        return
    with attempt(directory,'freeze-native-terminal',source_plan['domain'],arm) as cost:
        cost.update(seed=seed,new_source_fits=0,new_truth_function_calls=0,download_bytes=0)
        saved=read(directory/'search.json'); engine=load_pickle(directory/'search_engine.pkl')
        history=[dict(point=list(p),raw=np.asarray(y).tolist()) for p,y in engine.history]
        assert history==saved['history']
        try:
            with ExitStack() as stack:
                for obj,name in ((engine.problem,'simulate'),(engine,'_true_chance_margin'),
                                 (engine,'_evaluate_recommendation')):
                    stack.enter_context(patch.object(obj,name,side_effect=AssertionError('Freeze cannot simulate or score truth')))
                frozen=freeze(engine,native=True)
        except BaseException:
            recording = snapshot(engine)
            if recording is not None:
                save(directory/f'failed_native_terminal_recording_{time.time_ns()}.json',
                     dict(terminal_recording=recording, truth_used=False))
            raise
        assert history==[dict(point=list(p),raw=np.asarray(y).tolist()) for p,y in engine.history]
        save(directory/'native_terminal.json',dict(**frozen,tau=float(engine.problem.tau),
            truth_used=False,verification_performed=False,frozen_utc=datetime.now(timezone.utc).isoformat()))


def prepare(source_plan, terminal_plan):
    source=ROOT/source_plan['output_dir']; output=ROOT/terminal_plan['output_dir']
    active=set(verifier.cases(terminal_plan))
    # All available histories and alternative terminals precede every verifier.
    for seed in source_plan['target_seeds']:
        for cell in terminal_plan['arms']:
            if (seed,cell['name'],'original') not in active:
                continue
            path=source/f'seed_{seed}'/cell['history_arm']
            read(path/('native_terminal.json' if cell['terminal_backend']=='native' else 'search.json'))
    with attempt(output,'freeze-terminal-inputs',source_plan['domain']) as cost:
        cost.update(new_source_fits=0,new_truth_function_calls=0,download_bytes=0)
        for seed in source_plan['target_seeds']:
            tau=float(load_pickle(source/f'seed_{seed}'/'initialization_only'/'search_engine.pkl').problem.tau)
            assert tau==0.
            for cell in terminal_plan['arms']:
                if (seed,cell['name'],'original') not in active:
                    continue
                path=source/f'seed_{seed}'/cell['history_arm']; search=read(path/'search.json')
                frozen=read(path/'native_terminal.json')['terminal'] if cell['terminal_backend']=='native' else search['terminal']
                item=input_record(search,frozen,source_plan,cell,seed,
                    terminal_plan['verification_seed_by_task'][str(seed)],str(path.relative_to(ROOT)),tau)
                target=verifier.case_dir(output,seed,cell['name'],'original')/'input.json'
                if target.exists():
                    assert read(target)==item
                else:
                    save(target,item)
        save(output/'prepared.json',dict(case_count=len(verifier.cases(terminal_plan)),
            truth_used=False,frozen_utc=datetime.now(timezone.utc).isoformat()))


def costs(directory):
    rows=[json.loads(line) for path in directory.rglob('attempts.jsonl') for line in path.read_text().splitlines()]
    fields=('cpu_seconds','wall_seconds','new_source_calls','new_target_initial_calls',
            'new_target_search_calls','new_verification_calls','new_source_fits',
            'new_truth_function_calls','download_bytes')
    return dict(cost={k:sum(r.get(k,0) for r in rows) for k in fields},
                attempt_count=len(rows),failed_attempts=[r for r in rows if r['status']=='failed'])


def run(plan, workers):
    root=ROOT/plan['output_dir']; source=read(ROOT/plan['search_manifest'])
    terminal=read(ROOT/plan['terminal_manifest']); output=ROOT/terminal['output_dir']
    logs=root/'execution_logs'; logs.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(ROOT))
    def invoke(job):
        label,command=job
        with (logs/f'{label}.log').open('a') as stream:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        return label,result.returncode
    def stage(name,jobs):
        print(name,'started',len(jobs),flush=True); failed=[]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for count,future in enumerate(as_completed([pool.submit(invoke,j) for j in jobs]),1):
                label,code=future.result()
                print(name,count,'/',len(jobs),label,'exit',code,flush=True)
                if code: failed.append(label)
        if failed: raise RuntimeError(f'{name} failed; preserve paid state, do not resample: {failed}')
    base=[sys.executable,'-u','-m','performance.run_mainline_role_fresh']
    seeds=source['target_seeds']; names=[a['name'] for a in source['arms']]
    def searches(arms):
        return [(f'search_{seed}_{arm}',base+['search','--manifest',str(ROOT/plan['search_manifest']),
                 '--seed',str(seed),'--arm',arm]) for seed in seeds for arm in arms
                if not (ROOT/source['output_dir']/f'seed_{seed}'/arm/'search.json').exists()]
    if not (output/'replay_check.json').exists(): preflight(root,terminal)
    stage('initialization',searches(['initialization_only']))
    if not all((ROOT/source['output_dir']/f'seed_{s}'/'neighbors.json').exists() for s in seeds):
        stage('neighbors',[('prepare-neighbors',base+['prepare','--manifest',str(ROOT/plan['search_manifest'])])])
    stage('online-search',searches([a for a in names if a!='initialization_only']))
    own=[sys.executable,'-u','-m','performance.run_mainline_terminal_confirmation']
    stage('native-terminals',[(f'native_{s}_{a}',own+['native','--seed',str(s),'--arm',a])
        for s in seeds for a in ROLES if not (ROOT/source['output_dir']/f'seed_{s}'/a/'native_terminal.json').exists()])
    save(ROOT/source['output_dir']/'completion.json',dict(status='search_complete',**costs(ROOT/source['output_dir'])))
    if not (output/'prepared.json').exists(): prepare(source,terminal)
    stage('verification',[(f'verify_{s}_{a}',own+['verify','--seed',str(s),'--arm',a])
        for s,a,p in verifier.cases(terminal) if not (verifier.case_dir(output,s,a,p)/'verification.json').exists()])
    from performance.summarize_mainline_terminal_confirmation import summarize
    summarize(output,terminal)
    completion=read(output/'completion.json')
    total=costs(root)
    total.update(status='complete',recommendations=108,
        new_physical_target_calls=sum(total['cost'][k] for k in
            ('new_target_initial_calls','new_target_search_calls','new_verification_calls')),
        logical_target_calls=completion['logical_target_calls'],
        completed_utc=datetime.now(timezone.utc).isoformat())
    save(root/'completion.json',total)
    plan.update(status='complete',completion_file=str((root/'completion.json').relative_to(ROOT)),
                completed_utc=total['completed_utc'])
    save(MANIFEST,plan)
    for path in (plan['search_manifest'],plan['terminal_manifest']):
        child=read(ROOT/path); child.update(status='complete',completed_utc=total['completed_utc'])
        save(ROOT/path,child)
    print('complete',json.dumps(total['cost']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['run','native','verify'])
    parser.add_argument('--seed',type=int); parser.add_argument('--arm')
    parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args(); plan=read(MANIFEST)
    if args.stage=='run':
        run(plan,args.workers)
    elif args.stage=='native':
        assert args.arm in ROLES
        native_terminal(read(ROOT/plan['search_manifest']),args.seed,args.arm)
    else:
        terminal=read(ROOT/plan['terminal_manifest'])
        assert (args.seed,args.arm,'original') in verifier.cases(terminal)
        verifier.verify(ROOT/terminal['output_dir'],terminal,args.seed,args.arm,'original')

"""Run the fixed three-family cohort with global search/verification barriers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
import subprocess
import sys

from performance import run_mainline_terminal_equal as verifier
from performance.run_mainline_legacy_backend_bridge import ROOT, read, save
from performance.run_mainline_terminal_confirmation import native_terminal, prepare, costs, ROLES

MANIFEST=ROOT/'performance/manifests/mainline_three_family_v1.json'


def run(plan, workers, manifest_path=MANIFEST):
    root=ROOT/plan['output_dir']; logs=root/'execution_logs'; logs.mkdir(parents=True,exist_ok=True)
    assert read(ROOT/plan['preflight_file'])['status']=='passed'
    assert plan['status'] in ('registered','running')
    plan.update(status='running',started_utc=plan.get('started_utc',datetime.now(timezone.utc).isoformat()))
    save(manifest_path,plan)
    families=[(entry,read(ROOT/entry['search_manifest']),read(ROOT/entry['terminal_manifest']))
              for entry in plan['families']]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(ROOT))
    def invoke(job):
        label,command=job
        with (logs/f'{label}.log').open('a') as stream:
            code=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
        return label,code
    def stage(name,jobs):
        save(root/'progress.json',dict(stage=name,job_count=len(jobs),completed=0))
        print(name,'started',len(jobs),flush=True); failed=[]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for count,future in enumerate(as_completed([pool.submit(invoke,j) for j in jobs]),1):
                label,code=future.result()
                print(name,count,'/',len(jobs),label,'exit',code,flush=True)
                if code: failed.append(label)
                save(root/'progress.json',dict(stage=name,job_count=len(jobs),completed=count,failed=failed))
        if failed: raise RuntimeError(f'Preserve paid raw/state before recovery: {failed}')
    base=[sys.executable,'-u','-m','performance.run_mainline_role_fresh']
    own=[sys.executable,'-u','-m','performance.run_mainline_three_family','--manifest',str(manifest_path)]
    def searches(initial):
        jobs=[]
        for entry,source,_ in families:
            failed_cases={(row['seed'],row['arm']) for row in source.get('execution_failures',[])}
            for seed in source['target_seeds']:
                for arm in source['arms']:
                    name=arm['name']
                    if (seed,name) in failed_cases: continue
                    if (name=='initialization_only')!=initial: continue
                    path=ROOT/source['output_dir']/f'seed_{seed}'/name/'search.json'
                    if not path.exists():
                        jobs.append((f'search_{seed}_{name}',base+['search','--manifest',str(ROOT/entry['search_manifest']),
                                     '--seed',str(seed),'--arm',name]))
        return jobs
    stage('initialization',searches(True))
    stage('neighbors',[(f'neighbors_{entry["domain"]}',base+['prepare','--manifest',str(ROOT/entry['search_manifest'])])
        for entry,source,_ in families if not all((ROOT/source['output_dir']/f'seed_{s}'/'neighbors.json').exists()
                                                for s in source['target_seeds'])])
    stage('online-search',searches(False))
    stage('native-terminals',[(f'native_{seed}_{arm}',own+['native','--family',entry['domain'],
        '--seed',str(seed),'--arm',arm]) for entry,source,_ in families for seed in source['target_seeds']
        for arm in ROLES if not (ROOT/source['output_dir']/f'seed_{seed}'/arm/'native_terminal.json').exists()])
    inherited=ROOT/'paper_artifacts/mainline_terminal_confirmation_v1/terminal/replay_check.json'
    assert read(inherited)['status']=='passed'
    for entry,source,terminal in families:
        save(ROOT/source['output_dir']/'completion.json',dict(status='search_complete',**costs(ROOT/source['output_dir'])))
        output=ROOT/terminal['output_dir']
        if not (output/'prepared.json').exists(): prepare(source,terminal)
        save(output/'replay_check.json',dict(status='passed',inherited=str(inherited.relative_to(ROOT)),
            family_dispatch_preflight=plan['preflight_file'],new_raw_replay=False))
    modified = None
    if 'modified_terminal_manifest' in plan:
        from performance import run_mainline_same_policy as challenge
        modified = read(ROOT/plan['modified_terminal_manifest'])
        challenge.prepare(modified)
        assert read(ROOT/modified['output_dir']/'prepared.json')['case_count'] == sum(
            len(verifier.cases(terminal)) for _,_,terminal in families)
    # Both terminal variants for every family freeze before any verifier runs.
    stage('verification',[(f'verify_{seed}_{arm}',own+['verify','--family',entry['domain'],
        '--seed',str(seed),'--arm',arm]) for entry,_,terminal in families for seed,arm,protocol in verifier.cases(terminal)
        if not (verifier.case_dir(ROOT/terminal['output_dir'],seed,arm,protocol)/'verification.json').exists()])
    for _,_,terminal in families:
        verifier.require_all(ROOT/terminal['output_dir'],terminal,'verification.json')
    if modified is not None:
        pool = challenge.load_blocks(modified)
        jobs = list(challenge.cases(modified))
        for count,(baseline,directory) in enumerate(jobs,1):
            challenge.verify_case(modified,directory,pool,baseline)
            save(root/'progress.json',dict(stage='modified-verification',completed=count,job_count=len(jobs)))
            if count % 12 == 0:
                print('modified-verification',count,'/',len(jobs),flush=True)
        assert all((directory/'verification.json').exists() for _,directory in jobs)
        save(ROOT/modified['output_dir']/'verification_complete.json',dict(status='complete',cases=len(jobs),
            completed_utc=datetime.now(timezone.utc).isoformat()))
    # Every registered verification must exist before any family opens target truth.
    save(root/'progress.json',dict(stage='scoring',completed=0))
    from performance.summarize_mainline_terminal_confirmation import summarize
    for _,_,terminal in families:
        summarize(ROOT/terminal['output_dir'],terminal)
    from performance.summarize_mainline_three_family import summarize as aggregate
    result=aggregate(root,plan)
    rows = list(result['rows'])
    if modified is not None:
        from performance.summarize_mainline_same_policy import summarize as summarize_modified, completion
        modified_result = summarize_modified(ROOT/modified['output_dir'],modified)
        completion(ROOT/modified['output_dir'],modified_result)
        rows.extend(modified_result['rows'])
    total=costs(root); completed=datetime.now(timezone.utc).isoformat()
    failures=plan.get('execution_failures',[])
    failed_logical_calls=(2 if modified is not None else 1)*sum(r['logical_search_calls'] for r in failures)
    status='complete_with_search_failure' if failures else 'complete'
    total.update(status=status,recommendations=len(rows),registered_terminal_cells=plan['budget'].get('terminal_cells',len(rows)),
        execution_failures=failures,logical_failed_cell_calls=failed_logical_calls,
        new_physical_target_calls=sum(total['cost'][k] for k in
            ('new_target_initial_calls','new_target_search_calls','new_verification_calls')),
        logical_target_calls=sum(r['logical_total_calls'] for r in rows)+failed_logical_calls,completed_utc=completed)
    assert total['new_physical_target_calls']<=plan['budget']['maximum_target_physical_calls']
    assert total['logical_target_calls']<=plan['budget']['maximum_logical_target_calls']
    save(root/'completion.json',total)
    for entry,source,terminal in families:
        for key,child in (('search_manifest',source),('terminal_manifest',terminal)):
            child.update(status='complete_with_search_failure' if child.get('execution_failures') else 'complete',completed_utc=completed);save(ROOT/entry[key],child)
    if modified is not None:
        modified.update(status=status,completed_utc=completed)
        save(ROOT/plan['modified_terminal_manifest'],modified)
    plan.update(status=status,completed_utc=completed);save(manifest_path,plan)
    save(root/'progress.json',dict(stage=status,completed=len(rows),failed_terminal_cells=plan['budget'].get('terminal_cells',len(rows))-len(rows)))
    print('complete',json.dumps(total['cost']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['run','native','verify'])
    parser.add_argument('--family');parser.add_argument('--seed',type=int);parser.add_argument('--arm')
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--manifest',type=str,default=str(MANIFEST))
    args=parser.parse_args();manifest_path=ROOT/args.manifest;plan=read(manifest_path)
    if args.stage=='run':
        run(plan,args.workers,manifest_path)
    else:
        entry=next(row for row in plan['families'] if row['domain']==args.family)
        if args.stage=='native':
            assert args.arm in ROLES
            native_terminal(read(ROOT/entry['search_manifest']),args.seed,args.arm)
        else:
            terminal=read(ROOT/entry['terminal_manifest'])
            assert (args.seed,args.arm,'original') in verifier.cases(terminal)
            verifier.verify(ROOT/terminal['output_dir'],terminal,args.seed,args.arm,'original')

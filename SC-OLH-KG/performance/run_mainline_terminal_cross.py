"""Fresh verification of frozen search-history x terminal-backend decisions."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import os
import subprocess
import sys

from performance import run_mainline_terminal_equal as terminal
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save

MANIFEST = ROOT / 'performance/manifests/mainline_terminal_cross_v1.json'


def prepare(output, plan):
    parent = read(ROOT/plan['source_manifest'])
    source, decisions = ROOT/plan['source_output_dir'], ROOT/plan['decision_output_dir']
    with attempt(output, 'freeze-terminal-inputs', parent['domain']) as cost:
        cost.update(new_source_fits=0, new_truth_function_calls=0, download_bytes=0)
        for seed in plan['target_seeds']:
            for cell in plan['arms']:
                arm, history_arm = cell['name'], cell['history_arm']
                old = source/f'seed_{seed}'/history_arm
                search, verified = read(old/'search.json'), read(old/'verification.json')
                frozen = search['terminal']
                decision_file = old/'search.json'
                if cell['terminal_backend'] == 'native':
                    decision_file = decisions/f'seed_{seed}'/history_arm/'decisions.json'
                    saved = read(decision_file)
                    assert saved['saved_decision_parity'] and not saved['truth_used']
                    assert saved['original']['primary'] == search['primary']
                    assert [r['point'] for r in saved['original']['terminal']['frozen_shortlist']] == [r['point'] for r in frozen['frozen_shortlist']]
                    frozen = saved['native']['terminal']
                shortlist = frozen['frozen_shortlist']
                assert len(shortlist) == 3 and frozen['objective_incumbent_guard_enabled']
                assert frozen['candidate_verification_budgets'] == [80,128,128]
                assert frozen['max_verification_budget'] == 352
                assert frozen['objective_comparison_budget_per_policy'] == 8
                assert frozen['familywise_delta'] == .05
                assert frozen['objective_comparison']['delta'] == .05/3
                assert search['geometry'] == parent['geometry_by_seed'][str(seed)]['problem_kwargs']
                assert search['logical_search_calls'] == (10 if history_arm=='initialization_only' else 13)
                observed = {tuple(row['point']) for row in search['history']}
                assert all(tuple(r['point']) in observed for r in shortlist)
                # The core verifier computes outcomes from fresh raw, not these
                # frozen construction-time enabled/certified/status fields.
                first = verified['verification']['attempts'][0]
                item = dict(seed=seed, arm=arm, history_arm=history_arm,
                    terminal_backend=cell['terminal_backend'], protocol='original',
                    domain=parent['domain'], task=parent['task'], geometry=search['geometry'],
                    verification_seed=plan['verification_seed_by_task'][str(seed)],
                    primary=shortlist[0]['point'], initial=search['history'][0]['point'],
                    frozen_shortlist=shortlist, candidate_budgets=[80,128,128],
                    search_evaluation_count=search['logical_search_calls'],
                    familywise_delta=frozen['familywise_delta'], method=first['method_mode'],
                    mean_delta_fraction=frozen['mean_delta_fraction'],
                    shortlist_mode=frozen['shortlist_mode'],
                    objective_incumbent_position=frozen['objective_incumbent_position'],
                    objective_comparison_budget=frozen['objective_comparison_budget_per_policy'],
                    objective_comparison_delta=frozen['objective_comparison']['delta'],
                    tau=first['tau'], source_case=str(old.relative_to(ROOT)),
                    frozen_decision_file=str(decision_file.relative_to(ROOT)), truth_used=False)
                assert item['verification_seed'] != seed
                path = terminal.case_dir(output,seed,arm,'original')/'input.json'
                if path.exists():
                    assert read(path) == item, 'Frozen terminal input changed'
                else:
                    save(path,item)
        save(output/'prepared.json',dict(case_count=len(terminal.cases(plan)),
            truth_used=False, new_simulation_calls=0,
            frozen_utc=datetime.now(timezone.utc).isoformat()))


def replay(output, plan):
    # This tests the shared verifier adapter and branches on all 84 originals;
    # new native shortlists cannot inherit their old paid verification outcomes.
    originals = dict(plan, arms=[a for a in plan['arms'] if a['name']==a['history_arm']])
    terminal.replay(output, originals)


def run(output, plan, workers):
    if not (output/'prepared.json').exists():
        prepare(output,plan)
    if not (output/'replay_check.json').exists():
        replay(output,plan)
    logs=output/'execution_logs'
    logs.mkdir(exist_ok=True)
    env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
             MKL_NUM_THREADS='1', PYTHONPATH=str(ROOT))
    def invoke(case):
        seed, arm, _ = case
        with (logs/f'{seed}_{arm}.log').open('a') as stream:
            result=subprocess.run([sys.executable,'-u','-m',
                'performance.run_mainline_terminal_cross','verify','--seed',str(seed),
                '--arm',arm], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        return seed,arm,result.returncode
    pending=[case for case in terminal.cases(plan)
             if not (terminal.case_dir(output,*case)/'verification.json').exists()]
    failures=[]
    print('verification started',len(pending),flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(invoke,case) for case in pending]
        for count,future in enumerate(as_completed(futures),1):
            seed,arm,code=future.result()
            print(count,'/',len(pending),seed,arm,'exit',code,flush=True)
            if code:
                failures.append([seed,arm])
    if failures:
        raise RuntimeError(f'Verification failed; saved paid stages are not retried: {failures}')
    from performance.summarize_mainline_terminal_cross import summarize
    summarize(output,plan)
    completion=read(output/'completion.json')
    plan.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
        completion_file=str((output/'completion.json').relative_to(ROOT)),
        execution=completion)
    save(MANIFEST,plan)
    print('complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','replay','verify','summarize','run'])
    parser.add_argument('--seed',type=int)
    parser.add_argument('--arm')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    plan=read(MANIFEST); output=ROOT/plan['output_dir']
    if args.stage=='verify':
        assert (args.seed,args.arm,'original') in terminal.cases(plan)
        terminal.verify(output,plan,args.seed,args.arm,'original')
    elif args.stage=='run':
        run(output,plan,args.workers)
    elif args.stage=='summarize':
        from performance.summarize_mainline_terminal_cross import summarize
        summarize(output,plan)
    else:
        {'prepare':prepare,'replay':replay}[args.stage](output,plan)

"""Frozen post-review nine-point controls on the existing 240 tasks."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json, os
from pathlib import Path
import sys
from unittest.mock import patch
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import norm
from core.profile_atlas import AtlasSelection, ProfileAtlasConfig, SourceScoredProfileAtlas, percentile_ranks
from problems.randomized_profiles import RandomizedOrderedProfileProblem, generate_structural_profile_library, source_profile_records
from performance import benchmark_profile_stress_suite as benchmark
from performance.submission_revision_attribution import FixedSelection, selections, paired
from performance.submission_revision_controls import compact_row, summarize
PROTOCOL=ROOT/'performance/manifests/source_first_remainders_v1_20261002.json'
OUTPUT=ROOT/'paper_artifacts/source_first_remainders_v1_20261002'


def read(path):return json.loads(Path(path).read_text())
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')
def rowkey(row):return tuple(row[k] for k in ('family_seed','regime','target_seed','design_seed'))


def random_remainder(ids,first,family,design_seed,tag=53107):
    remaining=[pid for pid in ids if pid!=first]
    rng=np.random.default_rng(np.random.SeedSequence([family,design_seed,tag]))
    return [first]+[remaining[int(i)] for i in rng.choice(len(remaining),9,replace=False)]


def fixed_first_portfolio(scores,first,count=10):
    scores=np.asarray(scores);overall=scores.mean(axis=0);chosen=[first];incumbent=scores[:,first].copy()
    for _ in range(count-1):
        candidates=[j for j in range(scores.shape[1]) if j not in chosen]
        choice=min(candidates,key=lambda j:(float(np.minimum(incumbent,scores[:,j]).mean()),float(overall[j]),j))
        chosen.append(choice);incumbent=np.minimum(incumbent,scores[:,choice])
    return chosen


def portfolio_scores(records,ids):
    tasks=sorted({r.task_id for r in records});pairs={(r.task_id,r.profile_id):r for r in records};scores=[]
    for task in tasks:
        rows=[pairs[(task,pid)] for pid in ids]
        margins=[float(np.mean(r.constraint_samples))+norm.ppf(1-r.alpha)*np.sqrt(max(float(np.var(r.constraint_samples,ddof=1)),1e-6))-r.tau for r in rows]
        objective=[float(np.mean(r.objective_samples)) for r in rows]
        scores.append(.5*percentile_ranks(margins)+.5*percentile_ranks(objective))
    return np.array(scores)


def prepare(protocol):
    if (OUTPUT/'plan.json').exists():raise ValueError('comparison plan already frozen')
    saved=[r for r in read(ROOT/protocol['reference_rows'])['rows'] if r['study']=='confirmation']
    refs={(rowkey(r),r['arm']):r for r in saved};tasks=[];reused=[];archives=[]
    for family,regime in sorted({(r['family_seed'],r['regime']) for r in saved}):
        library=generate_structural_profile_library(64,dimension=128,seed=family+991,maximum_frequency=40)
        sources=[RandomizedOrderedProfileProblem(regime=regime,role='source',task_seed=10000+i,family_seed=family+benchmark._stable_seed(regime),d=256) for i in range(2)]
        records=source_profile_records(sources,library,replications=3,seed=family+1237)
        designs=selections(records);members=sorted(SourceScoredProfileAtlas(ProfileAtlasConfig(n0=64)).fit(records).selected().members,key=lambda m:m.profile_id)
        ids=[m.profile_id for m in members];first=designs['source_atlas'].members[0].profile_id
        fixed=[ids[i] for i in fixed_first_portfolio(portfolio_scores(records,ids),ids.index(first))]
        original=[m.profile_id for m in designs['source_greedy_portfolio'].members]
        archives.append(dict(family_seed=family,regime=regime,first_profile_id=first,fixed_first_portfolio=fixed,original_portfolio=original,source_replayed_calls=384))
        group=sorted([r for r in saved if r['family_seed']==family and r['regime']==regime and r['arm']=='source_atlas'],key=rowkey)
        for base in group:
            for arm in ('source_atlas','source_best_z','source_greedy_portfolio'):
                assert [m.profile_id for m in designs[arm].members]==refs[(rowkey(base),arm)]['selected_profile_ids'],'frozen source design differs'
            meta={k:base[k] for k in ('family_seed','regime','target_seed','design_seed')}
            random_ids=random_remainder(ids,first,family,base['design_seed'],protocol['random_stream_tag'])
            tasks.append(dict(**meta,arm='source_first_random9',selected_profile_ids=random_ids))
            if fixed==original:
                reused.append(dict(refs[(rowkey(base),'source_greedy_portfolio')],arm='source_first_fixed_portfolio',evaluation_reused_from='source_greedy_portfolio',selected_profile_ids=fixed))
            else:tasks.append(dict(**meta,arm='source_first_fixed_portfolio',selected_profile_ids=fixed))
    assert len(tasks)+len(reused)==480 and len(archives)==24
    write(OUTPUT/'plan.json',dict(protocol_id=protocol['protocol_id'],fixed_at_utc=datetime.now(timezone.utc).isoformat(),tasks=tasks,reused_rows=reused,archives=archives,original_source_samples_replayed=9216,new_independent_source_samples=0))
    print(json.dumps(dict(new_arm_task_runs=len(tasks),reused_arm_task_runs=len(reused),source_replay_calls=9216,new_independent_source_samples=0)))


def stage_metrics(target,points,shortlist):
    probabilities=np.array([target.true_feasibility_probability(p) for p in points]);safe=np.array([target.is_truly_feasible(p) for p in points]);margins=np.array([target.true_chance_margin(p) for p in points])
    index={tuple(p):i for i,p in enumerate(points)};chosen=[index[tuple(r['point'])] for r in shortlist]
    survival=1.;expected_true=0.;expected_false=0.
    for i in chosen:
        q=float(probabilities[i]**80)
        if safe[i]:expected_true+=survival*q
        else:expected_false+=survival*q
        survival*=1-q
    return dict(design_coverage=bool(safe.any()),shortlist_coverage=bool(safe[chosen].any()),design_has_p99=bool(np.any(probabilities>=.99)),shortlist_has_p99=bool(np.any(probabilities[chosen]>=.99)),expected_true_certificate=expected_true,expected_false_certificate=expected_false,best_true_margin=float(margins.min()),first_safe=bool(safe[0]),first_in_shortlist=bool(0 in chosen))


def run(protocol,index):
    task=read(OUTPUT/'plan.json')['tasks'][index];path=OUTPUT/'tasks'/f'{index:03d}'/'result.json'
    if path.exists():return
    family=task['family_seed'];library=generate_structural_profile_library(64,dimension=128,seed=family+991,maximum_frequency=40);byid={p.profile_id:p for p in library}
    # Frozen choices already contain the source information. Dummy member ranks are never consumed by this evaluator.
    from core.profile_atlas import AtlasMember
    members=tuple(AtlasMember(profile_id=pid,profile=byid[pid].values,nodes=byid[pid].nodes,safety_rank=0.,objective_rank=0.,robust_source_feasible=False,selected_order=i+1) for i,pid in enumerate(task['selected_profile_ids']))
    selection=AtlasSelection(members,dict(first_center=task['selected_profile_ids'][0],selection_frozen_before_new_target_outcomes=True,source_outcomes_used=True,target_outcomes_used=False,target_oracle_used=False))
    kwargs=dict(regime=task['regime'],target_seed=task['target_seed'],design_seed=task['design_seed'],dimension=1000,family_seed=family,library_size=64,N=10)
    with patch.object(benchmark,'source_profile_records',return_value=()),patch.object(benchmark,'SourceScoredProfileAtlas',lambda config:FixedSelection(selection)):
        result=benchmark.run_task(arm='source_atlas',**kwargs)
    row=compact_row(result,arm=task['arm'],provenance='new_postdecision_comparator_under_frozen_20261002_rules')
    row.update(family_seed=family,library_size=64,study='remainder_diagnostic')
    assert row['selected_profile_ids']==task['selected_profile_ids']
    target=RandomizedOrderedProfileProblem(regime=task['regime'],role='target',task_seed=task['target_seed'],family_seed=family+benchmark._stable_seed(task['regime']),d=1000)
    points=[target.point_from_structural_profile(byid[pid]) for pid in task['selected_profile_ids']]
    metrics=stage_metrics(target,points,result['shortlist']);assert metrics['design_coverage']==row['initial_design_contains_true_feasible']
    row['stage_metrics']=metrics;write(path,row);print(json.dumps(dict(index=index,arm=row['arm'],certified=row['certified_true_feasible'])),flush=True)


def collect(protocol,new_rows_path=None):
    plan=read(OUTPUT/'plan.json');new=read(new_rows_path) if new_rows_path is not None else [read(OUTPUT/'tasks'/f'{i:03d}'/'result.json') for i in range(len(plan['tasks']))]
    assert [(rowkey(r),r['arm']) for r in new]==[(rowkey(r),r['arm']) for r in plan['tasks']]
    saved=[r for r in read(ROOT/protocol['reference_rows'])['rows'] if r['study']=='confirmation' and r['arm'] in ('source_atlas','source_best_z','source_greedy_portfolio','standardized_generic_dct_maximin')]
    stagekey=lambda r:tuple(r[k] for k in ('family_seed','regime','target_seed'))
    stages={(stagekey(r),r['arm']):r for r in read(ROOT/protocol['stage_rows'])}
    for row in saved:row['stage_metrics']=stages[(stagekey(row),row['arm'])]
    reused=plan['reused_rows']
    for row in reused:row['stage_metrics']=stages[(stagekey(row),'source_greedy_portfolio')]
    rows=saved+reused+new;arms=sorted({r['arm'] for r in rows});groups=[]
    for arm in arms:
        chosen=[r for r in rows if r['arm']==arm];assert len(chosen)==240 and len({rowkey(r) for r in chosen})==240
        stats=summarize(chosen);stats['arm']=arm
        stats['stages']={k:sum(r['stage_metrics'][k] for r in chosen) for k in ('design_coverage','shortlist_coverage','design_has_p99','shortlist_has_p99','expected_true_certificate','expected_false_certificate')}
        stats['median_best_true_margin']=float(np.median([r['stage_metrics']['best_true_margin'] for r in chosen])) if all('best_true_margin' in r['stage_metrics'] for r in chosen) else None
        paid=0 if arm=='standardized_generic_dct_maximin' else 9216
        stats['source_calls_paid_once']=paid;stats['cumulative_all_in_calls']=paid+sum(r['target_search_calls']+r['verification_calls'] for r in chosen)
        stats['calls_per_true_certificate']=stats['cumulative_all_in_calls']/stats['certified_true_feasible_deployment_count'] if stats['certified_true_feasible_deployment_count'] else None;groups.append(stats)
    contrasts=[]
    for reference in ('source_atlas','source_best_z'):
        subset=[]
        for r in rows:
            if r['arm']==reference:subset.append(dict(r,arm='source_atlas'))
            elif r['arm'] in ('source_first_random9','source_first_fixed_portfolio'):subset.append(r)
        for control in ('source_first_random9','source_first_fixed_portfolio'):
            contrasts.append(dict(reference=reference,**paired(subset,control)))
    by_family=[dict(family_seed=family,summaries=[summarize([r for r in rows if r['family_seed']==family and r['arm']==arm]) for arm in arms]) for family in sorted({r['family_seed'] for r in rows})]
    by_regime=[dict(regime=regime,summaries=[summarize([r for r in rows if r['regime']==regime and r['arm']==arm]) for arm in arms]) for regime in sorted({r['regime'] for r in rows})]
    summary=dict(by_family=by_family,by_regime=by_regime,protocol_id=protocol['protocol_id'],status='complete',role=protocol['role'],new_arm_task_runs=len(new),reused_arm_task_runs=len(reused),groups=groups,paired=contrasts,new_independent_source_samples=0,source_archive_calls_paid_once_per_source_arm=9216,actual_new_target_search_calls=sum(r['target_search_calls'] for r in new),actual_new_verification_calls=sum(r['verification_calls'] for r in new),confirmation_year_access=False,limitations=protocol['limitations'])
    write(OUTPUT/'rows.json',rows);write(OUTPUT/'new_rows.json',new);write(OUTPUT/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('paired','by_family','by_regime')}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','run','collect'));parser.add_argument('--index',type=int);parser.add_argument('--new-rows',type=Path,help='Analyze compact recorded outcomes instead of individual task files.');args=parser.parse_args();protocol=read(PROTOCOL)
    if args.mode=='prepare':prepare(protocol)
    elif args.mode=='run':
        run(protocol,args.index)
        print('DONE',flush=True)
    else:collect(protocol,args.new_rows)

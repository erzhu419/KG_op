"""Postdecision finite-library truth diagnostic; never a selection rule."""
from pathlib import Path
import json, sys, os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from core.profile_atlas import SourceScoredProfileAtlas, ProfileAtlasConfig, percentile_ranks, profile_cosine_coordinate
from problems.randomized_profiles import RandomizedOrderedProfileProblem, generate_structural_profile_library, source_profile_records
from performance.benchmark_profile_stress_suite import _stable_seed


def margin_bound(z, margins, estimated, population):
    epsilon=float(np.max(np.abs(estimated-population)))
    members=np.flatnonzero(estimated <= estimated.min()+2*epsilon)
    distances=np.linalg.norm(z[:,None]-z[None,:],axis=2)
    differences=np.abs(margins[:,None]-margins[None,:])
    zero=distances==0
    collision=bool(np.any(differences[zero]>1e-12))
    lipschitz=None if collision else float(np.max(np.divide(differences,distances,out=np.zeros_like(distances),where=~zero)))
    diameter=float(distances[np.ix_(members,members)].max())
    # Stable population tie break: score, safety, objective, id is supplied upstream.
    j=int(np.argmin(population))
    gamma=float(-margins[j])
    upper=None if collision else float(-gamma+lipschitz*diameter)
    return dict(oracle_score_error=epsilon,compatible_set_size=len(members),compatible_diameter=diameter,finite_library_lipschitz=lipschitz,coordinate_collision=collision,population_first_margin=float(margins[j]),population_first_safe=bool(gamma>=0),margin_upper_bound=upper,sufficient_condition_holds=bool(gamma>=0 and upper is not None and upper<=0))


def main():
    saved=json.loads((ROOT/'paper_artifacts/submission_revision_20260930/profile_rows.json').read_text())['rows']
    rows=[r for r in saved if r['study']=='confirmation' and r['arm']=='source_atlas']
    output=[]; replayed=0
    for family,regime in sorted({(r['family_seed'],r['regime']) for r in rows}):
        library=generate_structural_profile_library(64,dimension=128,seed=family+991,maximum_frequency=40)
        sources=[RandomizedOrderedProfileProblem(regime=regime,role='source',task_seed=10000+i,family_seed=family+_stable_seed(regime),d=256) for i in range(2)]
        records=source_profile_records(sources,library,replications=3,seed=family+1237)
        replayed+=384
        fit=SourceScoredProfileAtlas(ProfileAtlasConfig(n0=64)).fit(records).selected()
        members=sorted(fit.members,key=lambda m:m.profile_id)
        ids=[m.profile_id for m in members]; byid={p.profile_id:p for p in library}
        coordinates=np.vstack([profile_cosine_coordinate(m.profile,nodes=m.nodes,max_frequency=8,frequency_penalty=.25,include_diagonal_quadratic=True) for m in members])
        scale=coordinates.std(axis=0); z=(coordinates-coordinates.mean(axis=0))/np.where(scale>1e-10,scale,1)
        estimated=np.array([.5*(m.safety_rank+m.objective_rank) for m in members])
        g=[]; f=[]
        for source in sources:
            points=[source.point_from_structural_profile(byid[pid]) for pid in ids]
            g.append(percentile_ranks([source.true_chance_margin(x) for x in points]))
            f.append(percentile_ranks([source.true_objective(x) for x in points]))
        g=np.mean(g,axis=0); f=np.mean(f,axis=0); population=.5*(g+f)
        order=sorted(range(64),key=lambda i:(population[i],g[i],f[i],ids[i]))
        # Order all arrays by the declared population tie break for argmin.
        for row in [r for r in rows if r['family_seed']==family and r['regime']==regime]:
            selected=min(range(64),key=lambda i:(estimated[i],members[i].safety_rank,members[i].objective_rank,ids[i]))
            assert ids[selected]==row['selected_profile_ids'][0], 'source replay differs from frozen decision'
            target=RandomizedOrderedProfileProblem(regime=regime,role='target',task_seed=row['target_seed'],family_seed=family+_stable_seed(regime),d=1000)
            margins=np.array([target.true_chance_margin(target.point_from_structural_profile(byid[pid])) for pid in ids])
            result=margin_bound(z[order],margins[order],estimated[order],population[order])
            result.update(family_seed=family,regime=regime,target_seed=row['target_seed'],estimated_first_margin=float(margins[selected]),estimated_first_safe=bool(margins[selected]<=0))
            if result['sufficient_condition_holds']: assert result['estimated_first_safe'], 'bound implication violated'
            output.append(result)
    summary=dict(task_count=len(output),population_first_safe=sum(r['population_first_safe'] for r in output),estimated_first_safe=sum(r['estimated_first_safe'] for r in output),sufficient_condition_holds=sum(r['sufficient_condition_holds'] for r in output),coordinate_collisions=sum(r['coordinate_collision'] for r in output),compatible_set_size_range=[min(r['compatible_set_size'] for r in output),max(r['compatible_set_size'] for r in output)],source_samples_replayed=replayed,new_independent_source_samples=0,target_truth_evaluations=len(output)*64,new_target_simulations=0,role='post hoc oracle diagnostic; exact observed source error is not a confidence radius; Lipschitz constant applies only to fixed mapped library; z_star chosen as implemented z so coordinate error is zero')
    summary['by_regime']=[dict(regime=regime,tasks=sum(r['regime']==regime for r in output),population_first_safe=sum(r['population_first_safe'] for r in output if r['regime']==regime),estimated_first_safe=sum(r['estimated_first_safe'] for r in output if r['regime']==regime),sufficient_condition_holds=sum(r['sufficient_condition_holds'] for r in output if r['regime']==regime)) for regime in sorted({r['regime'] for r in output})]
    out=ROOT/'paper_artifacts/first_center_margin_diagnostic_20261002';out.mkdir(exist_ok=True)
    (out/'rows.json').write_text(json.dumps(output,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))

if __name__=='__main__': main()

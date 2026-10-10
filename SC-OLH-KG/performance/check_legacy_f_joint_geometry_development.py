"""Matched cached eight-episode test of joint source geometry and v mixture."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_reference_training_budget import compact, prediction, errors, score, raw_density
from performance.check_legacy_f_training_information_budget import group_summary
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.prepare_legacy_f_joint_geometry_quadrature import rule_on_intervals
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates, source_prior_moments

EXECUTION = 'performance/manifests/legacy_joint_population_f_joint_geometry_development_execution_v1.json'
ARM = 'joint_common_task_geometry_variance_mixture'


def mixture_field(basis, psi, source, variances, weights, eta):
    design = np.column_stack([np.ones(len(psi)), basis.transform(psi)])
    law = source['fitted_source_law']; task_base = np.asarray(source['task_covariance_at_zero_independent_intercept'])
    beta0 = np.asarray(source['coefficient_mean_at_zero_task_intercept_variance'])
    h0 = np.asarray(source['common_covariance_at_zero_task_intercept_variance'])
    update = design@np.asarray(source['common_coefficient_update_columns'])
    eigenvalues, rhs = np.asarray(source['task_variance_eigenvalues']), np.asarray(source['task_variance_rhs'])
    conditional = variances[:, None]/(1+variances[:, None]*eigenvalues)
    base = design@(h0+task_base)@design.T+law['discrepancy_variance']*np.eye(len(psi))
    fmean = design@beta0-(conditional*rhs)@update.T
    n, k = len(psi), len(variances); means = np.empty((k,n,2)); means[:,:,0] = fmean; means[:,:,1] = eta[0]
    covariance = np.zeros((k,n,2,n,2)); covariance[:,:,1,:,1] = eta[1]
    covariance[:,:,0,:,0] = base+np.einsum('ki,ai,bi->kab',conditional,update,update)+variances[:,None,None]*np.outer(design[:,0],design[:,0])
    field = JointFunctionRiskPosterior.from_moments(means[0],covariance[0])
    field.means,field.covariances = means.reshape(k,2*n),covariance.reshape(k,2*n,2*n)
    field.log_weights = np.log(weights/weights.sum()); field.component_pairs = [(0,0)]*k
    return field


def run():
    started = time.process_time(); spec = read(EXECUTION); source_spec = read(spec['source_execution'])
    coverage,_ = source_paths(read(source_spec['template_execution'])); parent = read(coverage['source_execution'])
    old_spec = read(spec['mixture_execution']); adaptation = read(read(old_spec['reference_execution'])['adaptation_execution'])
    assert read(spec['source_rule_results'])['all12_source_rules_verified']
    rules = read(spec['source_rules']); assert rules['all12_source_rules_frozen_without_query_data']
    prefixes = read(old_spec['public_prefixes'])['rows']
    controls = read(old_spec['controls_profiles'])['rows']; mixtures = read(spec['mixture_profiles'])['rows']
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(x) for x in public['bank']]
    branches = []; frozen = []; count = spec['raw_episode_budgets'][0]
    for rule in rules['rows']:
        layout,deleted = rule['layout'],rule['deleted_task']; suffix = Path(f'layout{layout}')/f'task{deleted}'
        inputs = read(Path(coverage['output'])/suffix/'inputs.json'); source = rule['source_reference_law']; law = source['fitted_source_law']
        names = law['source_task_names']
        assert names == [t['name'] for t in inputs['source_tasks']] == [t['name'] for t in inputs['risk_source_tasks']]
        assert set(names) == {f'related_inventory_{i:02d}' for i in range(6) if i != deleted}
        assert law['source_task_count'] == 5 and law['source_label_count'] == 60
        prefix = next(r['public_prefix'] for r in prefixes if r['layout']==layout and r['deleted_task']==deleted)[:count]
        sites = coverage['layouts'][layout][deleted]; assert not set(prefix).intersection(sites)
        basis,risk_basis = (OrthogonalLowOrderBasis(inputs[key]) for key in ('public_mean_reference','public_risk_reference'))
        psi = (public_control_mean_coordinates(no_outcomes(problem_for(public['plan'],public['tasks'][deleted])),bank)
            -inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
        np.testing.assert_allclose(psi,inputs['public_mean_reference'],atol=1e-12,rtol=1e-12)
        risk = read(Path(parent['output'])/f"layout{adaptation['parent_layouts'][layout]}"/f'task{deleted}'/'risk_source_fit.json')
        eta_law = next(h['joint_structural'] for h in risk['heads'] if h['head']==3)
        eta = source_prior_moments(restore_joint(risk_basis,eta_law),np.asarray(inputs['public_risk_reference']))
        # One fitted-v component must equal the coherently conditioned source law.
        point = mixture_field(basis,psi,source,np.asarray([source['fitted_independent_task_intercept_variance']]),np.ones(1),eta)
        design = np.column_stack([np.ones(len(psi)),basis.transform(psi)])
        expected_mean = design@np.asarray(law['coefficient_mean'])
        expected_covariance = design@(np.asarray(law['population_mean_covariance'])+np.asarray(law['residual_coefficient_covariance']))@design.T+law['discrepancy_variance']*np.eye(len(psi))
        np.testing.assert_allclose(point.means[0,::2],expected_mean,atol=1e-8,rtol=1e-8)
        np.testing.assert_allclose(point.covariances[0,::2,::2],expected_covariance,atol=1e-8,rtol=1e-8)
        np.testing.assert_allclose(point.means[0,1::2],eta[0],atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(point.covariances[0,1::2,1::2],eta[1],atol=1e-12,rtol=1e-12)
        branches.append((layout,deleted,rule,basis,psi,eta,prefix,sites))
        frozen.append({'layout':layout,'deleted_task':deleted,'source_task_names':names,'paid_prefix':prefix,'scoring_sites':sites,
            'source_node_count':len(rule['variance_nodes']),'common_p':law['common_correlation_strength'],'task_q':law['task_template_strength'],
            'fitted_v_component_and_unchanged_eta_verified':True})
    out = ROOT/spec['output']; out.mkdir(parents=True,exist_ok=True)
    save(out/'frozen_inputs.json',{'execution':spec,'all12_source_rules_and_prefixes_frozen_before_paid_values':True,'rows':frozen})
    raw = read(coverage['source_data'])
    paid = {(layout,deleted):[raw['tasks'][deleted]['Y_replicates'][j][0][0] for j in prefix]
        for layout,deleted,_,_,_,_,prefix,_ in branches}; del raw
    rows,numerical = [],[]
    for layout,deleted,rule,basis,psi,eta,prefix,sites in branches:
        source = rule['source_reference_law']; nodes,weights = rule_on_intervals(source,rule['adaptive_u_intervals'],32)
        selected = mixture_field(basis,psi,source,np.asarray(rule['variance_nodes']),np.asarray(rule['posterior_weights']),eta)
        refined = mixture_field(basis,psi,source,nodes,weights,eta)
        selected_evidence = refined_evidence = 0.
        for j,y in zip(prefix,paid[(layout,deleted)]):
            selected_evidence += selected.update_batch(j,float(y),None,1)
            refined_evidence += refined.update_batch(j,float(y),None,1)
        state,refined_state = compact(selected,sites,selected_evidence),compact(refined,sites,refined_evidence)
        p,rp = prediction(state),prediction(refined_state); e = errors(p,rp); t = spec['predictive_tolerances']
        passed = e['mean_max_absolute']<=t['mean_max_absolute'] and e['covariance_relative_frobenius']<=t['covariance_relative_frobenius'] and e['paid_log_evidence_absolute']<=t['paid_log_evidence_absolute']
        numerical.append({'layout':layout,'deleted_task':deleted,'errors':e,'passed':passed})
        filename = f'layout{layout}_task{deleted}_episodes{count}.npz'; np.savez_compressed(out/filename,**state)
        refined_filename = None
        if deleted == 1:
            refined_filename = f'layout{layout}_task{deleted}_episodes{count}_refined.npz'; np.savez_compressed(out/refined_filename,**refined_state)
        rows.append({'layout':layout,'deleted_task':deleted,'episode_count':count,'arm':ARM,'scoring_sites':sites,'paid_prefix':prefix,
            'prediction':p,'refined_prediction':rp,'component_marginals_file':filename,'refined_component_marginals_file':refined_filename,
            'component_count':len(rule['variance_nodes']),'refined_component_count':len(nodes)})
        print('joint geometry forecast frozen',layout,deleted,passed,flush=True)
        del selected,refined,state,refined_state
    save(out/'predictions.json',{'execution':spec,'all12_forecasts_frozen_before_score_labels':True,'numerical_checks':numerical,'rows':rows})
    base = {'execution':spec,'numerical_checks':numerical,'no_model_adopted':True,'cost':spec['cost'],
        'source6_7_evaluated':False,'target_promoted':False,'independent_confirmation':False,'original_complete_gate_unchanged':True}
    if not all(r['passed'] for r in numerical):
        save(out/'results.json',{**base,'numerical_forecasts_verified':False,'forecasts_scored':False,'scientific_gate_interpreted':False,'cpu_seconds':time.process_time()-started})
        return
    score_frozen(rows,numerical,spec,out,coverage,controls,mixtures,count,base,started)


def score_frozen(rows,numerical,spec,out,coverage,controls,mixtures,count,base,started):
    labels = {layout:read(Path(coverage['output'])/f'layout{layout}'/'inputs.json')['source_tasks'] for layout in spec['layouts']}
    density_checks = []
    for row in rows:
        label = labels[row['layout']][row['deleted_task']]
        with np.load(out/row['component_marginals_file']) as state:
            row['scores'],row['density_integration_error_bound'] = score(row,state,label)
        if row['deleted_task']==1:
            with np.load(out/row['refined_component_marginals_file']) as state: density,_ = raw_density(state,label)
            difference = abs(density-row['scores']['f_raw_batch_density'])
            density_checks.append({'layout':row['layout'],'raw_density_refinement_error':difference,'passed':difference<=spec['density_refinement_tolerance']})
    groups = group_summary(rows,spec); summary = {layout:groups[layout]['all6'][str(count)] for layout in spec['layouts']}
    gates,contrasts = {},[]
    for layout in spec['layouts']:
        gates[layout] = {}
        for arm in ('native_f','flat','source_gp','source_variance_mixture'):
            members = [r for r in (mixtures if arm=='source_variance_mixture' else controls) if r['layout']==layout and r['episode_count']==count and r['arm']==arm]
            assert len(members)==6
            reference = {key:float(np.mean([r['scores'][key] for r in members])) for key in ('f_mse','f_raw_batch_density','sequential_transcript_density')}
            actual = summary[layout]
            gates[layout][arm] = {'mse_nonincrease':actual['f_mse']<=reference['f_mse'],
                'raw_density_strict_improvement':actual['f_raw_batch_density']>reference['f_raw_batch_density'],
                'transcript_density_nondecrease':actual['sequential_transcript_density']>=reference['sequential_transcript_density']}
            for row in rows:
                if row['layout']!=layout: continue
                control = next(r for r in members if r['deleted_task']==row['deleted_task'])
                contrasts.append({'layout':layout,'deleted_task':row['deleted_task'],'reference':arm,
                    'candidate_minus_control':{key:row['scores'][key]-control['scores'][key] for key in ('f_mse','f_raw_batch_density','sequential_transcript_density')},
                    'actual_mixture_transcript_minus_control':row['scores']['actual_mixture_transcript_density']-control['scores']['sequential_transcript_density']})
    verified = all(r['passed'] for r in density_checks)
    result = {**base,'summary':summary,'descriptive_groups':groups,'gates':gates,'matched_contrasts':contrasts,
        'advantage_against_all_four_controls':all(all(g.values()) for layout in gates.values() for g in layout.values()) if verified else None,
        'numerical_forecasts_verified':True,'numerical_density_verified':verified,'density_checks':density_checks,
        'forecasts_scored':True,'scientific_gate_interpreted':verified,'cpu_seconds':time.process_time()-started,
        'compact_component_artifact_bytes':sum(p.stat().st_size for p in out.glob('*.npz'))}
    save(out/'calibration_profiles.json',{'execution':spec,'rows':rows}); save(out/'results.json',result)
    print({'summary':summary,'gates':gates,'advantage':result['advantage_against_all_four_controls'],'cpu_seconds':result['cpu_seconds']},flush=True)


if __name__ == '__main__':
    run()

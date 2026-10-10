"""Source-prior initial-design information at the unchanged four-call budget."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_level_information import coefficient_field, gp_field, level_information
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import source_prior_moments

EXECUTION = 'performance/manifests/legacy_joint_population_f_source_information_design_execution_v1.json'


def posterior_covariance(covariance, selected, noise):
    factor = cho_factor(covariance[np.ix_(selected, selected)]+np.diag(noise[selected]), lower=True)
    return covariance-covariance[:, selected] @ cho_solve(factor, covariance[selected])


def greedy_design(covariance, noise, pool, count):
    updated = covariance.copy(); selected = []; steps = []
    for _ in range(count):
        available = [j for j in pool if j not in selected]
        reductions = (updated[:, available]**2).sum(axis=0)/(np.diag(updated)[available]+noise[available])
        position = int(np.argmax(reductions)); index = available[position]
        steps.append({'site': index, 'bank_average_variance_reduction': float(reductions[position]/len(updated))})
        column = updated[:, index].copy()
        updated -= np.outer(column, column)/(updated[index, index]+noise[index])
        selected.append(index)
    exact = posterior_covariance(covariance, selected, noise)
    np.testing.assert_allclose(updated, exact, atol=1e-12, rtol=1e-10)
    return selected, steps


def run():
    started = time.process_time(); spec = read(EXECUTION); context = read(spec['source_context_execution'])
    coverage, profile = source_paths(context); old = read(spec['retained_source_execution'])
    adaptation = read(read(old['reference_execution'])['adaptation_execution']); parent = read(coverage['source_execution'])
    laws = read(spec['intercept_source_laws'])['rows']; public_prefixes = read(old['public_prefixes'])['rows']; rows = []
    for layout in spec['layouts']:
        suffix = Path(f'layout{layout}')/f"task{spec['deleted_task']}"
        inputs = read(Path(coverage['output'])/suffix/'inputs.json')
        head = read(Path(profile['output'])/suffix/'selected_source_fit.json')['heads'][0]; native = head['joint_structural']
        names = [t['name'] for t in inputs['source_tasks']]
        assert set(names) == {f'related_inventory_{i:02d}' for i in range(6) if i != spec['deleted_task']}
        public = np.asarray(inputs['public_mean_reference']); basis = OrthogonalLowOrderBasis(public)
        design = np.column_stack([np.ones(len(public)), basis.transform(public)])
        law = next(r for r in laws if r['layout'] == layout and r['deleted_task'] == spec['deleted_task'])
        assert law['source_task_names'] == names
        residual = np.asarray(native['residual_coefficient_diagonal']).copy(); residual[0] = law['integrated_task_intercept_variance']
        fields = {'intercept_mixture': coefficient_field(design, np.asarray(law['integrated_common_coefficient_mean']),
            np.asarray(law['integrated_common_coefficient_covariance']), residual, native['discrepancy_variance']),
            'source_gp': gp_field(public, head['joint_gp'])}
        risk = read(Path(parent['output'])/f"layout{adaptation['parent_layouts'][layout]}"/f"task{spec['deleted_task']}"/'risk_source_fit.json')
        eta_law = next(h['joint_structural'] for h in risk['heads'] if h['head'] == 3)
        risk_public = np.asarray(inputs['public_risk_reference']); risk_basis = OrthogonalLowOrderBasis(risk_public)
        eta_mean, eta_covariance = source_prior_moments(restore_joint(risk_basis, eta_law), risk_public)
        noise = np.exp(eta_mean+.5*np.diag(eta_covariance))
        fixed = next(r['public_prefix'] for r in public_prefixes if r['layout'] == layout and r['deleted_task'] == spec['deleted_task'])[:4]
        sites = coverage['layouts'][layout][spec['deleted_task']]; pool = [j for j in range(len(public)) if j not in sites]
        assert all(j in pool for j in fixed)
        selections = {'fixed_public_prefix': fixed}; steps = {}
        for arm, field in fields.items():
            selections[arm+'_greedy'], steps[arm] = greedy_design(field[1], noise, pool, spec['prefix_length'])
        score_weights = np.zeros(len(public)); score_weights[sites] = 1/len(sites)
        weights = {'public_bank': np.ones(len(public))/len(public), 'fixed_score_roster': score_weights}
        comparisons = {}
        for arm, field in fields.items():
            comparisons[arm] = {}
            for name, selected in selections.items():
                covariance = posterior_covariance(field[1], selected, noise)
                comparisons[arm][name] = {'bank_average_conditional_f_variance': float(np.trace(covariance)/len(public)),
                    'original_bank_average_f_variance': float(np.trace(field[1])/len(public)),
                    'level_information': {roster: level_information(field, w, selected, noise[selected]) for roster, w in weights.items()}}
        rows.append({'layout': layout, 'deleted_task': spec['deleted_task'], 'source_task_names': names,
            'available_policy_indices': pool, 'scoring_site_indices': sites, 'frozen_prefixes': selections,
            'greedy_steps': steps, 'comparisons': comparisons, 'sequential_and_joint_covariance_updates_match': True})
        print({'layout': layout, 'prefixes': selections, 'variance': {arm: {name: value['bank_average_conditional_f_variance']
            for name, value in values.items()} for arm, values in comparisons.items()}}, flush=True)
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    result = {'execution': spec, 'rows': rows, 'cpu_seconds': time.process_time()-started, 'cost': spec['cost'],
        'frozen_before_query_outcomes': True, 'query_paid_or_score_outcomes_read': False,
        'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False,
        'information_contract': 'Source-derived linear expected posterior integrated f variance on the full public bank;approximate initial-design diagnostic for the raw mixture filter. No decision,KG or empirical prediction gain is inferred.'}
    save(out/'results.json', result)


if __name__ == '__main__':
    run()

"""Conditional innovations of the24 appended f source records, not a new fit."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_nested_source_conditioning import curve_change, source_system
from performance.run_legacy_related_inventory_source import save

EXECUTION = 'performance/manifests/legacy_joint_population_f_extra_source_innovations_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION); previous = read(spec['conditioning_results'])
    assert previous['all_source_conditionings_verified']
    conditioning = read(spec['conditioning_execution']); coverage = read(conditioning['coverage_execution'])
    parent = read(coverage['source_execution']); rows = []
    for layout in spec['layouts']:
        old = read(Path(parent['output'])/f"layout{conditioning['parent_layouts'][layout]}"/'inputs.json')
        new = read(Path(coverage['output'])/f'layout{layout}'/'inputs.json')
        head = read(Path(parent['output'])/f"layout{conditioning['parent_layouts'][layout]}"/'source_fit.json')['heads'][0]
        old_sites, new_sites = (x['execution']['source_site_indices_by_task'] for x in (old, new))
        public, design, y, m0, a, c = source_system(head, old['public_mean_reference'], new['source_tasks'], new_sites)
        keep, extra, task_ids, policy_ids, offset = [], [], [], [], 0
        for task_id, before, after in zip(spec['training_tasks'], old_sites, new_sites):
            keep.extend(offset+after.index(site) for site in before)
            for position, site in enumerate(after):
                if site not in before:
                    extra.append(offset+position); task_ids.append(task_id); policy_ids.append(site)
            offset += len(after)
        keep, extra = np.asarray(keep), np.asarray(extra); assert len(keep) == 48 and len(extra) == 24
        factor = cho_factor(c[np.ix_(keep, keep)], lower=True)
        cross = c[np.ix_(keep, extra)]; inverse_cross = cho_solve(factor, cross)
        old_error = y[keep]-design[keep]@m0
        prediction = design[extra]@m0+cross.T@cho_solve(factor, old_error)
        innovation = y[extra]-prediction
        s = c[np.ix_(extra, extra)]-cross.T@inverse_cross; s = .5*(s+s.T)
        conditional_factor = cho_factor(s, lower=True); alpha = cho_solve(conditional_factor, innovation)
        energy = float(innovation@alpha)
        extra_nll = .5*(len(extra)*np.log(2*np.pi)+2*np.log(np.diag(conditional_factor[0])).sum()+energy)
        saved = next(r['cases'] for r in previous['rows'] if r['layout'] == layout)
        expected_nll = 72*saved['old144_all72']['source_negative_log_likelihood_per_label']-48*saved['old144_old48']['source_negative_log_likelihood_per_label']
        np.testing.assert_allclose(extra_nll, expected_nll, atol=1e-8, rtol=1e-8)
        coefficient_cross = a@design[extra].T-a@design[keep].T@inverse_cross
        gain = public@cho_solve(conditional_factor, coefficient_cross.T).T
        delta = gain@innovation
        expected_delta = np.asarray(saved['old144_all72']['common_mean_bank'])-saved['old144_old48']['common_mean_bank']
        np.testing.assert_allclose(delta, expected_delta, atol=1e-10, rtol=1e-8)
        h_before = np.asarray(saved['old144_old48']['population_mean_covariance'])
        h_after = h_before-coefficient_cross@cho_solve(conditional_factor, coefficient_cross.T)
        np.testing.assert_allclose(h_after, saved['old144_all72']['population_mean_covariance'], atol=1e-10, rtol=1e-8)
        effects = []; contribution_sum = np.zeros(len(public)); task_ids = np.asarray(task_ids)
        for task_id in spec['training_tasks']:
            selected = task_ids == task_id; contribution = gain[:, selected]@innovation[selected]; contribution_sum += contribution
            effects.append({'source_task': task_id, 'extra_policy_indices': np.asarray(policy_ids)[selected].tolist(),
                'signed_common_curve_update': curve_change(contribution)})
        np.testing.assert_allclose(contribution_sum, delta, atol=1e-12)
        noise = np.concatenate([np.asarray(t['noise'])[:, 0] for t in new['source_tasks']])[extra]
        standardized = innovation/np.sqrt(np.diag(s))
        records = [{'source_task': int(task_id), 'policy': policy, 'conditional_innovation': float(value),
            'marginal_standardized_innovation': float(z), 'conditional_variance': float(variance),
            'observed_mean_noise_fraction': float(n/variance)}
            for task_id, policy, value, z, variance, n in zip(task_ids, policy_ids, innovation, standardized, np.diag(s), noise)]
        rows.append({'layout': layout, 'extra_record_count': len(extra), 'conditional_mahalanobis_energy': energy,
            'energy_per_extra_dimension': energy/len(extra), 'conditional_extra_nll': float(extra_nll),
            'conditional_nll_reproduction_error': float(abs(extra_nll-expected_nll)), 'common_curve_update': curve_change(delta),
            'maximum_absolute_marginal_standardized_innovation': float(np.max(np.abs(standardized))),
            'median_observed_mean_noise_fraction': float(np.median(noise/np.diag(s))),
            'source_task_effects': effects, 'records': records,
            'conditional_mean_covariance_and_log_density_reproduce_stage116': True})
        print('extra source innovations', layout, energy, energy/len(extra), flush=True)
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'results.json', {'execution': spec, 'rows': rows, 'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'no_model_adopted': True, 'source6_7_evaluated': False, 'query_paid_outcomes_read': False,
        'target_promoted': False, 'independent_confirmation': False})


if __name__ == '__main__':
    run()

"""Separate extra source conditioning from changed f population parameters."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis

EXECUTION = 'performance/manifests/legacy_joint_population_f_nested_source_conditioning_execution_v1.json'


def source_system(head, reference, tasks, site_lists):
    basis = OrthogonalLowOrderBasis(reference); public = np.asarray(reference)
    design = np.column_stack([np.ones(len(public)), basis.transform(public)])
    designs = [design[sites] for sites in site_lists]
    law = head['joint_structural']; m0 = np.asarray(head['regularizer']['coefficient_mean'])
    a, r = np.diag(law['common_prior_coefficient_diagonal']), np.asarray(law['residual_coefficient_diagonal'])
    d = np.vstack(designs); y = np.concatenate([np.asarray(t['response'])[:, 0] for t in tasks])
    error = y-d@m0
    c = d@a@d.T+block_diag(*[(local*r)@local.T+np.diag(np.asarray(task['noise'])[:, 0]+law['discrepancy_variance'])
        for local, task in zip(designs, tasks)])
    return design, d, y, m0, a, c


def condition_fixed_law(head, reference, tasks, site_lists):
    design, d, y, m0, a, c = source_system(head, reference, tasks, site_lists)
    error = y-d@m0
    factor = cho_factor(c, lower=True); alpha = cho_solve(factor, error)
    beta = m0+a@d.T@alpha
    h = a-a@d.T@cho_solve(factor, d)@a; h = .5*(h+h.T)
    nll = .5*(len(y)*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+error@alpha)/len(y)
    bank_covariance = design@h@design.T
    return {'source_label_count': len(y), 'source_negative_log_likelihood_per_label': float(nll),
        'coefficient_mean': beta.tolist(), 'population_mean_covariance': h.tolist(),
        'common_mean_bank': (design@beta).tolist(), 'common_covariance_bank': bank_covariance.tolist(),
        'common_bank_average_variance': float(np.trace(bank_covariance)/len(design)),
        'common_public_average_level_variance': float(design.mean(axis=0)@h@design.mean(axis=0))}


def curve_change(value):
    value = np.asarray(value)
    return {'bank_rms': float(np.sqrt(np.mean(value**2))), 'bank_average_level_change': float(value.mean()),
        'centered_shape_rms': float(np.std(value))}


def run():
    started = time.process_time(); spec = read(EXECUTION); coverage = read(spec['coverage_execution'])
    parent, native = read(coverage['source_execution']), read(spec['native_execution'])
    rows = []
    for layout in spec['layouts']:
        old_layout = spec['parent_layouts'][layout]
        old = read(Path(parent['output'])/f'layout{old_layout}'/'inputs.json')
        new = read(Path(coverage['output'])/f'layout{layout}'/'inputs.json')
        old_head = read(Path(parent['output'])/f'layout{old_layout}'/'source_fit.json')['heads'][0]
        new_head = read(Path(native['output'])/f'layout{layout}'/'selected_source_fit.json')['heads'][0]
        old_sites, new_sites = (x['execution']['source_site_indices_by_task'] for x in (old, new))
        assert [t['name'] for t in old['source_tasks']] == [t['name'] for t in new['source_tasks']]
        assert len(old['source_tasks']) == len(spec['training_tasks']) == 6
        for old_task, new_task, before, after in zip(old['source_tasks'], new['source_tasks'], old_sites, new_sites):
            assert set(before) <= set(after)
            positions = [after.index(site) for site in before]
            for key in ('response', 'noise'):
                np.testing.assert_array_equal(old_task[key], np.asarray(new_task[key])[positions])
        for inputs, sites in ((old, old_sites), (new, new_sites)):
            public = np.asarray(inputs['public_mean_reference'])
            for task, indices in zip(inputs['source_tasks'], sites):
                np.testing.assert_allclose(task['psi'], public[indices], atol=1e-12, rtol=1e-12)
        cases = {}; native_checks = []
        for law_name, head, reference in (('old144', old_head, old['public_mean_reference']), ('native216', new_head, new['public_mean_reference'])):
            for data_name, inputs, sites in (('old48', old, old_sites), ('all72', new, new_sites)):
                value = condition_fixed_law(head, reference, inputs['source_tasks'], sites); cases[f'{law_name}_{data_name}'] = value
                if (law_name, data_name) in (('old144', 'old48'), ('native216', 'all72')):
                    law = head['joint_structural']; t = spec['native_reproduction_tolerance']
                    for key in ('coefficient_mean', 'population_mean_covariance', 'source_negative_log_likelihood_per_label'):
                        np.testing.assert_allclose(value[key], law[key], atol=t['absolute'], rtol=t['relative'])
                    native_checks.append({'law': law_name, 'source_records': data_name, 'native_moments_and_nll_reproduced': True})
            before, after = (np.asarray(cases[f'{law_name}_{name}']['population_mean_covariance']) for name in ('old48', 'all72'))
            assert np.linalg.eigvalsh(after-before).max() <= spec['fixed_law_common_covariance_contraction_tolerance']
        oo, oe, no, nn = (np.asarray(cases[name]['common_mean_bank']) for name in
            ('old144_old48', 'old144_all72', 'native216_old48', 'native216_all72'))
        changes = {'total_native_change': nn-oo, 'extra_data_under_old_law': oe-oo, 'law_change_after_all72': nn-oe,
            'law_change_before_extra_data': no-oo, 'extra_data_under_new_law': nn-no, 'data_law_interaction': nn-no-oe+oo}
        np.testing.assert_allclose(changes['total_native_change'], changes['extra_data_under_old_law']+changes['law_change_after_all72'], atol=1e-14)
        np.testing.assert_allclose(changes['total_native_change'], changes['law_change_before_extra_data']+changes['extra_data_under_new_law'], atol=1e-14)
        rows.append({'layout': layout, 'parent_layout': old_layout, 'cases': cases, 'curve_changes': {k: curve_change(v) for k, v in changes.items()},
            'verification': {'old_records_nested_and_observations_matched': True, 'native': native_checks,
                'source_sites_match_each_public_basis': True, 'fixed_law_common_covariance_contracts_with_extra_records': True}})
        print('nested source conditioning', layout, rows[-1]['curve_changes'], flush=True)
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    result = {'execution': spec, 'rows': rows, 'all_source_conditionings_verified': True, 'no_model_adopted': True,
        'cost': spec['cost'], 'cpu_seconds': time.process_time()-started, 'source6_7_evaluated': False,
        'query_paid_outcomes_read': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'results.json', result); print({'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()

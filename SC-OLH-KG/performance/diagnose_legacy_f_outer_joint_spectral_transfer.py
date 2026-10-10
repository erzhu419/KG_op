"""Attribute the stopped outer law using frozen states and cached controls."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_information_ceiling import variance_parts
from performance.diagnose_legacy_f_inner_conditional_calibration import local_calibration
from performance.diagnose_legacy_f_level_shape_mean_error import decompose
from performance.diagnose_legacy_f_outer_conditional_calibration import summarize
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save

EXECUTION = 'performance/manifests/legacy_joint_population_f_outer_joint_spectral_transfer_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION); outer = read(spec['outer_results'])
    assert outer['diagnostic_interpreted'] and outer['candidate_stopped']
    calibration = read(spec['cached_calibration_results']); assert calibration['diagnostic_interpreted']
    plan = read(outer['execution']['plan']); controls = read(plan['control_results'])
    coverage, _ = source_paths(read(plan['source_context_execution']))
    folder, out = ROOT / outer['execution']['output'], ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    base = {'execution': spec, 'cost': spec['cost'], 'no_model_adopted': True, 'candidate_remains_stopped': True,
        'no_new_forecasts_or_unscored_outcomes': True, 'source6_7_evaluated': False, 'target_promoted': False,
        'independent_confirmation': False, 'training_diagnostic_only': True}
    rows = []
    for forecast in outer['rows']:
        if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']:
            save(out / 'results.json', {**base, 'rows': rows, 'diagnostic_interpreted': False,
                'stopped_at_cpu_budget': True, 'cpu_seconds': time.process_time() - started}); return
        layout = forecast['layout']; inputs = read(Path(coverage['output']) / f'layout{layout}' / 'inputs.json')
        task = next(t for t in inputs['source_tasks'] if t['name'] == 'related_inventory_01')
        labels = np.asarray(task['response']); observed = labels[:, 0]
        reference = next(r for r in calibration['rows'] if r['layout'] == layout and r['design'] == 'fixed_public_prefix' and r['arm'] == 'source_variance_mixture')
        mixture = next(r for r in controls['rows'] if r['layout'] == layout and r['design'] == forecast['design'] and r['arm'] == 'source_variance_mixture')
        gp = next(r for r in controls['rows'] if r['layout'] == layout and r['design'] == forecast['design'] and r['arm'] == 'source_gp')
        assert Path(reference['selected_component_file']).name == mixture['component_marginals_file']
        assert mixture['scoring_sites'] == forecast['scoring_sites'] == coverage['layouts'][layout][1]
        assert mixture['paid_prefix'] == gp['paid_prefix'] == forecast['paid_prefix'] == plan['paid_prefix']
        sites = []
        with np.load(folder / forecast['component_file']) as state:
            assert state['component_means'].shape == (1, 12, 2)
            for j, label in enumerate(labels):
                record, _ = local_calibration(state['component_means'][:, j], state['component_local_covariances'][:, j], state['log_weights'], label[0], label[3])
                sites.append(record)
        average = summarize(sites)
        np.testing.assert_allclose(average['joint_batch_density'], forecast['scores']['f_raw_batch_density'], atol=1e-8, rtol=0.)
        np.testing.assert_allclose(reference['summary']['joint_batch_density'], mixture['scores']['f_raw_batch_density'], atol=1e-8, rtol=0.)
        density_delta = {k: average[k] - reference['summary'][k] for k in ('joint_batch_density', 'variance_label_density', 'conditional_mean_density')}
        np.testing.assert_allclose(density_delta['joint_batch_density'], density_delta['variance_label_density'] + density_delta['conditional_mean_density'], atol=1e-12, rtol=1e-10)
        mse_geometry = {name: decompose(row['prediction']['mean'], observed) for name, row in
            [('candidate', forecast), ('retained_mixture', mixture), ('source_gp', gp)]}
        for name, row in [('candidate', forecast), ('retained_mixture', mixture), ('source_gp', gp)]:
            np.testing.assert_allclose(mse_geometry[name]['mse'], row['scores']['f_mse'], atol=1e-12, rtol=1e-10)
        per_site = []
        for j, (candidate_site, retained_site) in enumerate(zip(sites, reference['sites'])):
            density = {k: candidate_site[k] - retained_site[k] for k in density_delta}
            values = {name: (row['prediction']['mean'][j] - observed[j]) ** 2 for name, row in
                [('candidate', forecast), ('retained_mixture', mixture), ('source_gp', gp)]}
            per_site.append({'scoring_policy': forecast['scoring_sites'][j], 'candidate_calibration': candidate_site,
                'density_difference_from_retained': density, 'squared_mean_errors': values})
        row = {'layout': layout, 'query_task': 1, 'paid_prefix': plan['paid_prefix'], 'scoring_sites': forecast['scoring_sites'],
            'candidate_summary': average, 'retained_summary': reference['summary'], 'density_differences_from_retained': density_delta,
            'mse_geometry': mse_geometry, 'latent_f_bank_variance': {name: variance_parts(np.asarray(r['prediction']['covariance']))
                for name, r in [('candidate', forecast), ('retained_mixture', mixture), ('source_gp', gp)]}, 'sites': per_site,
            'negative_density_sites': sum(s['density_difference_from_retained']['joint_batch_density'] < 0 for s in per_site),
            'mse_improved_sites_vs_gp': int(sum(s['squared_mean_errors']['candidate'] < s['squared_mean_errors']['source_gp'] for s in per_site))}
        rows.append(row)
        print('stopped outer attribution', layout, density_delta, 'calibration', {k: average[k] for k in
            ('covered_90', 'squared_standardized_error', 'log_conditional_f_variance_score_gradient')}, flush=True)
    save(out / 'results.json', {**base, 'rows': rows, 'diagnostic_interpreted': True, 'cpu_seconds': time.process_time() - started,
        'interpretation': 'Future-variance-conditioned diagnostic only. Local variance tangents do not specify an admissible learned repair. Retained mixture calibration and all GP forecasts are reused without replay;no candidate fit or query selection.'})


if __name__ == '__main__':
    run()

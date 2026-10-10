"""Cached-truth attribution to frozen source-selected groups and spectral modes."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_covered_mean_attribution import design
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, spectral_weights

EXECUTION = 'performance/manifests/legacy_joint_population_group_spectral_attribution_execution_v1.json'


def projection(x, truth, columns, candidates):
    matrix = x[:, columns]
    coefficient, _, rank, singular = np.linalg.lstsq(matrix, truth, rcond=None)
    residual = matrix@coefficient-truth
    return {'columns': [int(c) for c in columns], 'rank': int(rank), 'column_count': len(columns),
        'condition_number': float(singular[0]/singular[-1]),
        'coefficient_solution_unique': int(rank) == len(columns),
        'full_catalog_mse': float(np.mean(residual**2)), 'unpaid_mse': float(np.mean(residual[2:]**2)),
        'critical_residuals': residual[candidates].tolist()}, coefficient, residual


def run():
    started = time.process_time(); spec = read(EXECUTION); source = read(spec['source_execution'])
    snapshots = read(spec['target_snapshot'])['snapshots']; folds = read(spec['target_scoring'])['folds']
    profiles = read(spec['target_profiles'])['rows']; rows, geometries = [], []
    for layout in spec['layouts']:
        folder = Path(source['output'])/f'layout{layout}'
        inputs, fits = read(folder/'inputs.json'), read(folder/'source_fit.json')
        basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
        group_columns = {'intercept': [0], **{g: (np.flatnonzero(basis.group_index == j)+1).tolist()
            for j, g in enumerate(basis.group_names)}}
        low_columns = {k: [0]+[j+1 for j, name in enumerate(basis.names)
            if name.startswith('pair:') or int(name.rsplit(':', 1)[1]) < k] for k in (1, 2)}
        geometries.append({'layout': layout, **basis.diagnostics()})
        for frozen in (s for s in snapshots if s['layout'] == layout):
            task = frozen['query']['task_id']; x = design(basis, frozen['mean_queries'][0])
            truth = np.asarray(next(f for f in folds if f['task_id'] == task)['scoring_truth'])[:, [0, 2]]
            for head, name in enumerate(('f', 'g')):
                full, oracle, residual = projection(x, truth[:, head], list(range(x.shape[1])), spec['diagnostic_candidates'])
                low = {str(k): projection(x, truth[:, head], c, spec['diagnostic_candidates'])[0]
                    for k, c in low_columns.items()}
                for arm in spec['mean_arms']:
                    fit = fits['heads'][head][arm]; beta = np.asarray(fit['coefficient_mean'])
                    row = next(r for r in profiles if r['layout'] == layout and r['task_id'] == task and r['mean_arm'] == arm)
                    prior = np.asarray(row['prior']['mean'])[:, head]
                    # A wrong head/basis would attribute an error that the frozen model never made.
                    np.testing.assert_allclose(x@beta, prior, rtol=1e-8, atol=1e-8)
                    active = (np.asarray(fit['common_group_variances']) > 0) if arm == 'joint_structural' else np.ones(len(basis.group_names), dtype=bool)
                    columns = [0]+[j+1 for j, g in enumerate(basis.group_index) if active[g]]
                    selected = projection(x, truth[:, head], columns, spec['diagnostic_candidates'])[0]
                    mode_error = x*(beta-oracle)[None, :]
                    group_error = {g: mode_error[:, c].sum(axis=1) for g, c in group_columns.items()}
                    # The decomposition must close; otherwise a column or projection residual was lost.
                    np.testing.assert_allclose(mode_error.sum(axis=1)+residual, prior-truth[:, head], atol=1e-10)
                    weights = spectral_weights(basis, fit['common_spectral_exponent'])[0] if arm == 'joint_structural' else np.array([
                        1/np.sum(basis.group_index == g) for g in basis.group_index])
                    diagonal = np.asarray(fit['common_prior_coefficient_diagonal'])
                    rows.append({'layout': layout, 'task_id': task, 'head': name, 'mean_arm': arm,
                        'full_projection': full, 'low_mode_projections_with_all_pairs': low,
                        'source_selected_excess_support_projection': selected,
                        'active_common_groups': [g for g, a in zip(basis.group_names, active) if a],
                        'common_spectral_exponent': fit.get('common_spectral_exponent', 0),
                        'prior_unpaid_mse': float(np.mean((prior[2:]-truth[2:, head])**2)),
                        'group_error_catalog_rms': {g: float(np.sqrt(np.mean(v**2))) for g, v in group_error.items()},
                        'coefficients': [{'name': n, 'oracle': float(o), 'source_mean': float(b),
                            'common_prior_variance': float(v), 'group_spectral_weight': float(w),
                            'roughness': float(r)} for n, o, b, v, w, r in zip(
                                ['intercept']+basis.names, oracle, beta, diagonal, np.r_[1., weights], np.r_[0., basis.roughness])],
                        'points': [{'candidate': j, 'prior_error': float(prior[j]-truth[j, head]),
                            'projection_residual': float(residual[j]),
                            'intercept_contribution': float(mode_error[j, 0]),
                            'group_contributions': {g: float(v[j]) for g, v in group_error.items() if g != 'intercept'},
                            'mode_contributions': {n: float(v) for n, v in zip(basis.names, mode_error[j, 1:])},
                            'sum_absolute_mode_contributions': float(np.abs(mode_error[j]).sum()),
                            'sum_absolute_oracle_function_contributions': float(np.abs(x[j]*oracle).sum()),
                            'oracle_projected_value': float(x[j]@oracle)} for j in spec['diagnostic_candidates']]})
    summary = {layout: {arm: {head: {
        'full_projection_mse': float(np.mean([r['full_projection']['full_catalog_mse'] for r in rows if r['layout'] == layout and r['mean_arm'] == arm and r['head'] == head])),
        'selected_support_projection_mse': float(np.mean([r['source_selected_excess_support_projection']['full_catalog_mse'] for r in rows if r['layout'] == layout and r['mean_arm'] == arm and r['head'] == head])),
        'low_mode_projection_mse': {str(k): float(np.mean([r['low_mode_projections_with_all_pairs'][str(k)]['full_catalog_mse']
            for r in rows if r['layout'] == layout and r['mean_arm'] == arm and r['head'] == head])) for k in (1, 2)},
        'g63_or_f63': {'mean_prior_error': float(np.mean([r['points'][1]['prior_error'] for r in rows if r['layout'] == layout and r['mean_arm'] == arm and r['head'] == head])),
            'mean_group_contributions': {g: float(np.mean([r['points'][1]['group_contributions'][g] for r in rows
                if r['layout'] == layout and r['mean_arm'] == arm and r['head'] == head])) for g in geometries[0]['function_groups']}}}
        for head in ('f', 'g')} for arm in spec['mean_arms']} for layout in spec['layouts']}
    result = {'execution': spec, 'geometries': geometries, 'rows': rows, 'summary': summary,
        'interpretation': 'Projection uses already-opened full-catalog truth only for diagnosis. Zero common excess variance still retains the native weak tail. Product-marginal orthogonality does not imply orthogonality on this correlated catalog; contributions can cancel and coefficients do not identify physical factors. No diagnostic coefficients or supports enter a predictor.',
        'cost': {'new_source_fits': 0, 'new_simulator_calls': 0, 'new_verification_calls': 0, 'new_terminal_integrations': 0},
        'cpu_seconds': time.process_time()-started}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); save(out/'results.json', result)
    print({'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()

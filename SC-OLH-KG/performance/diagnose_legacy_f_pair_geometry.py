"""Noiseless two-site information geometry under frozen f covariances."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_information_ceiling import variance_parts
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis

EXECUTION = 'performance/manifests/legacy_joint_population_f_pair_geometry_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION); information = read(spec['information_execution'])
    conditioning = read(information['conditioning_execution']); coverage = read(conditioning['coverage_execution'])
    parent, native = read(coverage['source_execution']), read(conditioning['native_execution'])
    previous = read(spec['information_results'])['rows']; rows = []
    for layout in spec['layouts']:
        for name, fit_spec, directory, filename in (
            ('old144', parent, coverage['parent_layouts'][layout], 'source_fit.json'),
            ('native216', native, layout, 'selected_source_fit.json')):
            input_spec = parent if name == 'old144' else coverage
            reference = read(Path(input_spec['output'])/f'layout{directory}'/'inputs.json')['public_mean_reference']
            law = read(Path(fit_spec['output'])/f'layout{directory}'/filename)['heads'][0]['joint_structural']
            basis = OrthogonalLowOrderBasis(reference)
            design = np.column_stack([np.ones(len(reference)), basis.transform(reference)])
            h = np.asarray(law['population_mean_covariance']); r = np.asarray(law['residual_coefficient_diagonal'])
            k = design@(h+np.diag(r))@design.T+law['discrepancy_variance']*np.eye(len(reference))
            n = len(k); parts = variance_parts(k); pairs = np.asarray([(i, j) for i in range(n) for j in range(i+1, n)])
            cross = np.transpose(k[:, pairs], (1, 0, 2)); local = k[pairs[:, :, None], pairs[:, None, :]]
            inverse = np.linalg.pinv(local, rcond=1e-12, hermitian=True)
            total_gain = np.einsum('bni,bij,bnj->b', cross, inverse, cross)/n
            average = cross.mean(axis=1); level_gain = np.einsum('bi,bij,bj->b', average, inverse, average)
            shape_gain = total_gain-level_gain
            chosen = int(np.argmax(shape_gain)); original = int(np.flatnonzero((pairs == spec['paid_candidates']).all(axis=1))[0])
            former = next(v for v in previous if v['layout'] == layout and v['law'] == name)
            gains = np.column_stack([total_gain, level_gain, shape_gain]); keys = list(parts)
            np.testing.assert_allclose(gains[original], [parts[key]-former['after_exact_paid0_1']['current'][key]
                for key in keys], atol=1e-10, rtol=1e-8)
            x = cross[chosen]; conditioned = k-x@inverse[chosen]@x.T; conditioned = .5*(conditioned+conditioned.T)
            selected_parts = variance_parts(conditioned)
            np.testing.assert_allclose(gains[chosen], [parts[key]-selected_parts[key] for key in keys], atol=1e-10, rtol=1e-8)
            assert np.linalg.eigvalsh(conditioned).min() >= -1e-10
            assert np.linalg.eigvalsh(k-conditioned).min() >= -1e-10
            eigenvalues = np.linalg.eigvalsh(local); ranks = (eigenvalues > 1e-12*eigenvalues[:, -1:]).sum(axis=1)
            row = {'layout': layout, 'law': name, 'pair_count': len(pairs),
                'numerical_pair_rank_counts': {str(rank): int((ranks == rank).sum()) for rank in np.unique(ranks)},
                'fixed0_1_reduction_fraction': {key: float(gains[original, i]/parts[key]) for i, key in enumerate(keys)},
                'shape_optimal_pair': pairs[chosen].tolist(),
                'shape_optimal_pair_reduction_fraction': {key: float(gains[chosen, i]/parts[key]) for i, key in enumerate(keys)},
                'scalar_and_batched_conditioning_agree': True, 'previous_fixed_pair_bound_reproduced': True}
            rows.append(row); print('pair geometry', row, flush=True)
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'results.json', {'execution': spec, 'rows': rows, 'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'response_outcomes_indexed': False, 'query_paid_outcomes_read': False, 'source6_7_evaluated': False,
        'no_model_adopted': True, 'target_promoted': False, 'independent_confirmation': False})


if __name__ == '__main__':
    run()

"""Restore frozen source laws on actual cross-dimension public policy inputs.

The old risk encoder saved its transformed source rows but not its numerical
map. Rebuild that encoder from the same 32 source rows and fixed algorithm,
then verify the entire original public reference before using new policies.
Source laws, spectral/additive maps, mixture masses and GP fits stay frozen.
"""

import json
from pathlib import Path

import numpy as np

from performance.connect_historical_learned_prior_kg import (
    PLAN, read, restore_learned_source_prior,
)
from performance.connect_historical_matched_gp import controls_from_fit
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.historical_common_bank import _require_zero_paid
from performance.run_legacy_related_inventory_source import problem_for
from representation.legacy_observable_prior import (
    LegacyObservableCoordinate, legacy_exposure_descriptors,
)
from representation.public_control_mean import public_control_mean_coordinates


def query_coordinates(layout, problem, bank, plan=PLAN, *, output_path=None):
    """Compute and optionally freeze mean4/risk10 from real public controls.

The caller supplies the actual target bank and public-schema provider. Neither
target outcomes nor task geometry parameters enter the feature calculation.
This function performs one frozen encoder reconstruction, not a source-law or
GP optimization. The returned record is directly JSON serializable.
"""
    connection = read(plan)
    selection = read(read(connection['source_plan'])['selection_plan'])
    inputs_path = selection['source_inputs_by_layout'][layout]
    inputs = read(inputs_path)
    old = read(connection['legacy_connection_plan'])
    risk_path = old['query_risk_inputs_by_layout'][layout]
    risk = read(risk_path)
    public = read(old['public_inputs'])
    source_bank = [tuple(p) for p in public['bank']]
    ids, sites = risk['training_task_ids'], risk['source_site_indices_by_task']
    providers = {i: no_outcomes(problem_for(public['plan'], public['tasks'][i])) for i in ids}
    descriptors = {i: legacy_exposure_descriptors(providers[i], source_bank, 50) for i in ids}
    tasks = [dict(t, response=np.asarray(t['response'])) for t in risk['source_tasks']]
    assert [t['name'] for t in tasks] == [public['tasks'][i]['task_id'] for i in ids]
    training = np.vstack([descriptors[i][indices] for i, indices in zip(ids, sites)])
    assert len(training) == sum(len(t['response']) for t in tasks) == 32
    coordinate = LegacyObservableCoordinate().fit(training, tasks)
    replayed_risk = coordinate.transform(descriptors[ids[0]])
    frozen_risk = np.asarray(risk['public_reference_psi'])
    np.testing.assert_allclose(replayed_risk, frozen_risk, atol=1e-8, rtol=1e-8)
    center, scale = np.asarray(inputs['mean_coordinate_center']), np.asarray(inputs['mean_coordinate_scale'])
    replayed_mean = (public_control_mean_coordinates(providers[ids[0]], source_bank)-center)/scale
    frozen_mean = np.asarray(inputs['public_mean_reference'])
    np.testing.assert_allclose(replayed_mean, frozen_mean, atol=1e-12, rtol=1e-12)
    points = np.asarray(bank)
    if points.ndim != 2 or points.shape[1] != problem.d:
        raise ValueError('Actual policy bank dimension must match the public-schema provider')
    mean = (public_control_mean_coordinates(problem, bank)-center)/scale
    risk_coordinates = coordinate.transform(legacy_exposure_descriptors(problem, bank, 50))
    if not (np.isfinite(mean).all() and np.isfinite(risk_coordinates).all()):
        raise ValueError('Actual policy coordinates must be finite')
    result = {'mean': mean.tolist(), 'risk': risk_coordinates.tolist(), 'diagnostics': {
        'layout': layout, 'target_dimension': int(problem.d), 'candidate_count': len(bank),
        'mean_dimension': 4, 'risk_dimension': 10, 'source_mean_inputs': inputs_path,
        'source_risk_inputs': risk_path, 'source_task_names': [t['name'] for t in tasks],
        'frozen_encoder_reconstructions': 1, 'frozen_encoder_training_rows': len(training),
        'source_law_optimizations': 0, 'gp_optimizations': 0, 'new_simulator_calls': 0,
        'old_mean_reference_max_absolute_error': float(np.max(abs(replayed_mean-frozen_mean))),
        'old_risk_reference_max_absolute_error': float(np.max(abs(replayed_risk-frozen_risk))),
        'new_bank_representation_recomputed': True,
    }}
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def restore_cross_dimension_prior(reuse, pilot, layout, arm, coordinates):
    """Restore one source-only arm on the caller's frozen actual-bank features."""
    mean, risk = (np.asarray(coordinates[key], dtype=float) for key in ('mean', 'risk'))
    count = len(mean)
    if mean.shape != (count, 4) or risk.shape != (count, 10):
        raise ValueError('Actual bank coordinates must have mean[N,4] and risk[N,10]')
    if arm == 'source_structural_mixture':
        model, identities, checks = restore_learned_source_prior(
            layout, list(range(count)), pilot['source_connection_plan'], coordinates=(mean, risk))
        diagnostics = {'complete_identities': len(identities),
            'positive_components_per_head': [len(f.means) for pair in model.components for f in pair],
            'maximum_source_projection_error': max(c['common_projection_max_error'] for c in checks)}
    elif arm == 'source_common_task_rbf':
        fit = read(reuse['gp_fit'])['by_layout'][layout]
        assert fit['source_task_ids'] == [f'related_inventory_{i:02d}' for i in pilot['training_tasks']]
        model = controls_from_fit(fit, mean, risk, .04)[arm]
        diagnostics = {'frozen_gp_fit': reuse['gp_fit'], 'positive_components_per_head': [1, 1]}
    else:
        raise ValueError('Cross-dimension pilot supports the structural and same-source GP arms')
    _require_zero_paid(model, count)
    return model, {**diagnostics, 'candidate_count': count, 'paid_episodes': 0,
        'actual_bank_coordinates_used': True, 'source_law_optimizations': 0, 'gp_optimizations': 0,
        'frozen_encoder_reconstructions': 0}

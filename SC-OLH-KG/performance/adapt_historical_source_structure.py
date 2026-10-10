"""Fit existing historical source selectors to the current cached source rows."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import digamma

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save
from representation.additive_groups import TransferableAdditiveGroupBank
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.physical_risk_prior import log_sample_variance_density
from representation.task_posterior import ReplicationVarianceTaskPosterior
from representation.transferable_spectral import SourceDomainBatch, TransferableSpectralBasis

PLAN = 'performance/manifests/historical_source_structure_adaptation_v1.json'


def selector_state(model):
    """Save the existing learner's numerical map,without copying its algorithm."""
    names = ('psi_mean_', 'psi_scale_', 'library_mean_', 'library_scale_',
        'library_names_', 'selected_idx_', 'whitening_', 'group_names_',
        'function_names_', 'source_domains_', 'source_weights_', 'clip_low_', 'clip_high_')
    return {name: value.tolist() if isinstance(value, np.ndarray) else value
            for name in names if (value := getattr(model, name, None)) is not None}


def restore_selector(record, spec):
    classes = {'spectral': TransferableSpectralBasis, 'additive': TransferableAdditiveGroupBank}
    model = classes[record['selector']](**spec[record['selector'] + '_config'])
    string_lists = {'library_names_', 'group_names_', 'function_names_', 'source_domains_'}
    for name, value in record['map_state'].items():
        setattr(model, name, value if name in string_lists else np.asarray(value))
    model.feature_dim = model.whitening_.shape[1]
    model.diagnostics_ = dict(record['diagnostics'])
    return model


def run():
    started = time.process_time(); spec = read(PLAN)
    fields = read(spec['raw_source_fields'])['rows']; rows = []; checks = []; maps = []
    risk_laws = read(spec['risk_laws'])['rows']
    for layout, path in spec['source_inputs_by_layout'].items():
        inputs = read(path)
        field = next(r for r in fields if r['layout'] == layout)
        tasks = [t for t in inputs['source_tasks'] if t['name'] in spec['source_tasks']]
        assert [t['name'] for t in tasks] == spec['source_tasks']
        response = np.vstack([t['response'] for t in tasks])
        risk = next(r for r in risk_laws if (r['layout'], r['arm']) == (layout, field['active_arm']))
        reference = np.asarray(risk['public_reference_psi'])
        risk_psi = reference[[site['policy'] for site in field['sites']]]
        design = np.column_stack([np.ones(48), OrthogonalLowOrderBasis(reference).transform(risk_psi)])
        np.testing.assert_allclose(design, risk['source_design'], atol=1e-12, rtol=0.)
        np.testing.assert_allclose(response[:, 0], field['batch_means'], atol=1e-14, rtol=0.)
        sample_variance = np.exp(response[:, 3] + digamma(1.))
        np.testing.assert_allclose(sample_variance, field['sample_variances'], atol=1e-14, rtol=0.)
        # The historical discrete variance posterior and the current exact
        # log-label density must assign the same mass on these real batches.
        variance_experts = np.array([.5, 1., 2.]) * np.mean(sample_variance)
        posterior = ReplicationVarianceTaskPosterior(['half', 'pooled', 'double'],
            [1/3, 1/3, 1/3], minimum_weight=0.)
        for i, variance in enumerate(sample_variance):
            posterior.update_from_replication(i, variance, variance_experts, 3)
        logs = log_sample_variance_density(np.log(variance_experts), sample_variance[:, None], 3).sum(axis=0)
        expected = np.exp(logs - np.max(logs)); expected /= expected.sum()
        np.testing.assert_allclose(posterior.posterior_weights(), expected, atol=1e-12, rtol=1e-10)
        checks.append({'layout': layout, 'raw_mean_and_sample_variance_rows_matched': True,
            'historical_and_current_variance_posterior_max_error': float(np.max(np.abs(expected - posterior.posterior_weights()))),
            'replicated_policies': posterior.evidence_count, 'effective_variance_dof': posterior.effective_dof})
        mean_coordinates = [np.asarray(t['psi']) for t in tasks]
        risk_coordinates = np.split(risk_psi, 4)
        for head, name in enumerate(('f', 'g', 'eta_g', 'eta_f', 'joint_f_g_eta_g_eta_f')):
            # Old risk caches held eight rows per source. The frozen public
            # risk map supplies the actual twelve raw-batch sites per source.
            # Current mean and risk maps are distinct fitted coordinate blocks;
            # the joint selector retains both,without equating their axes.
            coordinates = ([np.column_stack([m, v]) for m, v in zip(mean_coordinates, risk_coordinates)]
                if head == 4 else mean_coordinates if head < 2 else risk_coordinates)
            batches = [SourceDomainBatch(t['name'], psi,
                np.asarray(t['response']) if head == 4 else np.asarray(t['response'])[:, head, None])
                for t, psi in zip(tasks, coordinates)]
            pooled = np.vstack([b.psi for b in batches])
            for kind, model in (
                ('spectral', TransferableSpectralBasis(**spec['spectral_config'])),
                ('additive', TransferableAdditiveGroupBank(**spec['additive_config'])),
            ):
                if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']:
                    raise TimeoutError('historical selector adaptation CPU budget reached')
                model.fit(batches); values = model.transform(pooled); gram = values.T @ values / len(values)
                diagnostic = {k: v for k, v in model.diagnostics().items() if k != 'fingerprint'}
                orthogonal = bool(np.max(np.abs(gram - np.eye(model.feature_dim))) <= spec['source_Gram_tolerance'])
                assert len(values) == 48 and np.all(np.isfinite(values)) and model.feature_dim > 0
                row = {'layout': layout, 'head': name, 'selector': kind,
                    'input_dimension': pooled.shape[1], 'source_record_count': len(values),
                    'operational_adaptation_passed': True, 'source_orthogonality_passed': orthogonal,
                    'source_Gram_max_error': float(np.max(np.abs(gram - np.eye(model.feature_dim)))),
                    'diagnostics': diagnostic}
                record = {'layout': layout, 'head': name, 'selector': kind,
                    'map_state': selector_state(model), 'source_features': values.tolist(), 'diagnostics': diagnostic}
                restored = restore_selector(record, spec).transform(pooled)
                np.testing.assert_allclose(restored, values, atol=1e-12, rtol=0.)
                row['cached_map_reconstruction_passed'] = True
                rows.append(row); maps.append(record)
                print('historical selector', layout, name, kind, 'dimensions', model.feature_dim,
                      'Gram', row['source_Gram_max_error'], flush=True)
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out / 'maps.json', {'execution': spec, 'rows': maps})
    save(out / 'results.json', {'execution': spec, 'checks': checks, 'rows': rows,
        'all_operational_adaptations_passed': all(r['operational_adaptation_passed'] for r in rows),
        'all_source_orthogonality_passed': all(r['source_orthogonality_passed'] for r in rows),
        'cpu_seconds': time.process_time() - started, 'cost': {'existing_selector_fits': len(rows),
            'new_simulator_calls': 0, 'new_source_chains': 0, 'new_target_predictions': 0},
        'no_model_adopted': True, 'independent_confirmation': False,
        'interpretation': 'Historical source spectral/additive selectors now consume the same current48 source batches,with their existing algorithms. This establishes interface reuse and pooled-source orthogonality. Cached upstream coordinates,empirical source measure and auxiliary source selection scores remain unchanged;this is not independent validation,raw joint-likelihood hyperparameter convergence or decision evidence.'})


if __name__ == '__main__':
    run()

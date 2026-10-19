"""Supply new posterior moments to the complete original V69 engine.

The original proposal posterior, candidate routes, raw feedback and independent
terminal verifier remain native. Only constrained-TS scoring and the final
observed-policy ordering receive the new posterior. TS uses a coherent Gaussian
moment projection; source identities and raw updates remain in the raw mixture.
"""
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np

from acquisition.decision_backends import score_decision_backend
from performance.mainline_legacy_native import (
    LegacyNativeAlgorithm, build_native, cached_initial_rows,
)


class FieldView:
    def __init__(self, bank, mean, covariance):
        self.index = {tuple(point): i for i, point in enumerate(bank)}
        self.a = np.asarray(mean)
        self.C = np.asarray(covariance)

    def _indices(self, points):
        return np.asarray([self.index[tuple(point)] for point in points], dtype=int)

    def posterior_mean_many(self, points):
        return self.a[self._indices(points)]

    def posterior_var_many(self, points):
        return np.maximum(np.diag(self.C)[self._indices(points)], 0.)

    def augmented_feature_matrix(self, points):
        return np.eye(len(self.a))[self._indices(points)]


class VarianceView:
    def __init__(self, bank, noise):
        self.index = {tuple(point): i for i, point in enumerate(bank)}
        self.noise = np.asarray(noise)

    def predict_variance_many(self, output, points, problem):
        return self.noise[[self.index[tuple(point)] for point in points], output]

    # This view is only used inside posterior decision calculations. Actual
    # certification uses the unchanged independent raw-observation verifier.
    predict_certification_variance_many = predict_variance_many


def posterior_views(bank, state, tau):
    return (
        [FieldView(bank, state['mean'][:, 0], state['cov'][:, 0, :, 0]),
         FieldView(bank, state['mean'][:, 2] + tau, state['cov'][:, 2, :, 2])],
        VarianceView(bank, state['noise']),
    )


def restore_point_only_noise(bundle, problem, bank, history):
    """Source-zero marginals, each conditioned on only its own paired raws.

    Projecting one policy at a time is the point-only product used in the
    frozen-history diagnosis, without retaining the full candidate covariance.
    Identity weights and E[exp(eta)] are updated by the native raw mixture.
    """
    from performance.mainline_legacy_source_models import restore_model
    from performance.mainline_moment_policy import moment_state
    noise = []
    for point in bank:
        own = [row for row in history if tuple(row['point']) == tuple(point)]
        model = restore_model(bundle, 'source_structural_mixture', problem, [point], own)
        noise.append(moment_state(model)['noise'][0])
    return np.asarray(noise)


class LegacyPosteriorAlgorithm(LegacyNativeAlgorithm):
    def __init__(self, *args, **kwargs):
        self.backend_arm = None
        self.backend_bundle = None
        self.backend_trace = []
        self._decision_context = 0
        self._moment_cache = {}
        self._local_noise_cache = {}
        super().__init__(*args, **kwargs)

    @contextmanager
    def _posterior_decision(self):
        self._decision_context += 1
        try:
            yield
        finally:
            self._decision_context -= 1

    def _posterior_views(self, pool):
        from performance.mainline_legacy_source_models import restore_moment_state
        history = [{'point': list(point), 'raw': list(raw)} for point, raw in self.history]
        bank = list(dict.fromkeys([tuple(row['point']) for row in history]
                                  + [tuple(point) for point in pool]))
        key = tuple(bank)
        paid = len(history)
        if getattr(self, '_cache_paid', None) != paid:
            self._moment_cache.clear()
            self._local_noise_cache = {}
            self._cache_paid = paid
        if key not in self._moment_cache:
            model_arm = {'same_source_gp': 'source_common_task_rbf',
                         'current_structural_mixture': 'source_structural_mixture',
                         'current_structural_local_noise': 'source_structural_mixture'}[self.backend_arm]
            state = restore_moment_state(self.backend_bundle, model_arm, self.problem, bank, history)
            if not all(np.isfinite(state[name]).all() for name in ('mean', 'cov', 'noise')):
                raise ValueError('Nonfinite posterior moments')
            self._moment_cache[key] = posterior_views(bank, state, self.problem.tau)
        return self._moment_cache[key]

    def _terminal_noise_view(self, pool, full_noise):
        history = [{'point': list(point), 'raw': list(raw)} for point, raw in self.history]
        missing = list(dict.fromkeys(tuple(point) for point in pool
                                     if tuple(point) not in self._local_noise_cache))
        if missing:
            local = restore_point_only_noise(self.backend_bundle, self.problem, missing, history)
            self._local_noise_cache.update(zip(missing, local[:, 1]))
        noise = np.column_stack([
            full_noise.predict_variance_many(0, pool, self.problem),
            [self._local_noise_cache[tuple(point)] for point in pool],
        ])
        return VarianceView(pool, noise)

    def _score_backend(self, backend, candidates, obj, con, variance, problem, **kwargs):
        views, noise = self._posterior_views(candidates)
        kwargs['task_ensemble'] = None
        result = score_decision_backend(backend, candidates, *views, noise, problem, **kwargs)
        self.backend_trace.append({
            'paid_episodes': len(self.history), 'candidate_count': len(candidates),
            'unobserved_candidate_count': sum(tuple(x) not in self.observations for x in candidates),
            'ranked_first': list(candidates[int(np.argmax(result['total']))]),
            'objective_mean_range': [float(np.min(views[0].posterior_mean_many(candidates))),
                                     float(np.max(views[0].posterior_mean_many(candidates)))],
            'constraint_mean_range': [float(np.min(views[1].posterior_mean_many(candidates))),
                                      float(np.max(views[1].posterior_mean_many(candidates)))],
            'moment_projection': 'joint Gaussian per response head; exact mixture episode variance',
        })
        return result

    def _terminal_bayes_risk_components(self, gpr_models, variance_model, pool,
                                        task_ensemble=None, **kwargs):
        if self._decision_context:
            gpr_models, variance_model = self._posterior_views(pool)
            if getattr(self, 'backend_arm', None) == 'current_structural_local_noise':
                variance_model = self._terminal_noise_view(pool, variance_model)
            task_ensemble = None
            kwargs.pop('objective_expert_moments', None)
            kwargs.pop('constraint_expert_moments', None)
        return super()._terminal_bayes_risk_components(
            gpr_models, variance_model, pool, task_ensemble=task_ensemble, **kwargs)

    def _decision_backend_terminal_recommendation(self, pool):
        with self._posterior_decision():
            return super()._decision_backend_terminal_recommendation(pool)

    def _terminal_verification_protocol(self, primary):
        with self._posterior_decision():
            return super()._terminal_verification_protocol(primary)

    def run_search(self):
        with patch('algorithms.single_olhkg.score_decision_backend', self._score_backend):
            return super().run_search()


def build_backend(domain, arm, bundle=None, source_prior=None, N=13, seed=80, *, initial_rows=None):
    new = arm in {'same_source_gp', 'current_structural_mixture', 'current_structural_local_noise'}
    if arm not in {'legacy_native_v69', 'initialization_only',
                   'same_source_gp', 'current_structural_mixture', 'current_structural_local_noise'}:
        raise ValueError(arm)
    engine = build_native(
        domain, seed=seed, N=10 if arm == 'initialization_only' else N,
        source_prior=source_prior, initial_rows=(cached_initial_rows(domain, seed)
                                               if initial_rows is None else initial_rows),
        algorithm_class=LegacyPosteriorAlgorithm if new else LegacyNativeAlgorithm,
    )
    engine.backend_arm = arm
    if new:
        if bundle is None:
            raise ValueError('A fitted source bundle is required')
        engine.backend_bundle = bundle
    return engine

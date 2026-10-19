"""Frozen role prior and point-local control in the original V69 decisions."""
import numpy as np

from performance.mainline_legacy_backend import LegacyPosteriorAlgorithm, posterior_views
from performance.mainline_legacy_native import LegacyNativeAlgorithm, build_native
from performance.mainline_role_prior import role_moment_state
from performance.mainline_role_point_only import point_only_moment_state
from performance.mainline_role_after_init import after_init_point_only_moment_state
from performance.mainline_source_alignment_backend import (
    SOURCE_PERMUTED_ROLE_ARMS, SOURCE_PERMUTED_ROLE_OPERATORS,
    build_role_projection_context,
)
from performance.run_mainline_legacy_backend_bridge import ROOT, load_pickle


ROLE_ARMS = {'legacy_role_full', 'legacy_role_point_only', 'legacy_role_after_init_point_only'}
# These two historical sets deliberately identify original-source assets.
# New controls must not be restored with frozen_assets/source_models_file.
MODEL_ARMS = ROLE_ARMS | {'same_source_gp', 'current_structural_mixture'}
ALL_ROLE_ARMS = ROLE_ARMS | SOURCE_PERMUTED_ROLE_ARMS
ROLE_OPERATORS = {
    'legacy_role_full': 'full',
    'legacy_role_after_init_point_only': 'after_init_point_only',
    'legacy_role_point_only': 'point_only',
    **SOURCE_PERMUTED_ROLE_OPERATORS,
}
STOP = 'initialization_only'


class RolePosteriorAlgorithm(LegacyPosteriorAlgorithm):
    def _posterior_views(self, pool):
        history = [{'point': list(p), 'raw': list(y)} for p,y in self.history]
        bank = list(dict.fromkeys([tuple(r['point']) for r in history]+[tuple(p) for p in pool]))
        key = tuple(bank)
        if getattr(self, '_cache_paid', None) != len(history):
            self._moment_cache.clear()
            self._cache_paid = len(history)
        if key not in self._moment_cache:
            context = getattr(self, 'role_projection_context', self)
            operator = ROLE_OPERATORS[self.backend_arm]
            if operator == 'full':
                state = role_moment_state(self.backend_bundle, context, bank, history)
            else:
                if not hasattr(self, '_point_only_cache'):
                    self._point_only_cache = {}
                moment_function = (after_init_point_only_moment_state
                    if operator == 'after_init_point_only'
                    else point_only_moment_state)
                state = moment_function(self.backend_bundle, context, bank, history,
                                        cache=self._point_only_cache)
            if not all(np.isfinite(state[k]).all() for k in ('mean','cov','noise')):
                raise ValueError('Nonfinite role posterior')
            if not hasattr(self, 'role_restoration_trace'):
                self.role_restoration_trace = []
            self.role_restoration_trace.append({'paid_prefix':len(history), 'bank_count':len(bank),
                'roundoff_events':state.get('roundoff_events',[]), 'cost':state.get('cost',{})})
            self._moment_cache[key] = posterior_views(bank,state,self.problem.tau)
        return self._moment_cache[key]


def reload_backend_bundle(engine, plan, arm):
    """Restore the bundle stripped by save_engine, respecting source identity.

    The separate native risk context is retained inside the search checkpoint;
    it is neither reconstructed nor replaced during paid-prefix recovery.
    """
    if arm in SOURCE_PERMUTED_ROLE_ARMS:
        assets = plan['source_permuted_assets']
    elif arm in MODEL_ARMS:
        assets = plan['frozen_assets']
    else:
        return
    engine.backend_bundle = load_pickle(ROOT/assets['source_models_file'])


def build_fresh_backend(plan, seed, arm, initial_rows):
    prior = load_pickle(ROOT/plan['frozen_assets']['native_source_prior_file'])
    geometry = plan['geometry_by_seed'][str(seed)]
    cls = (RolePosteriorAlgorithm if arm in ALL_ROLE_ARMS else
           LegacyPosteriorAlgorithm if arm in MODEL_ARMS else LegacyNativeAlgorithm)
    engine = build_native(plan['domain'],seed=seed,N=10 if arm==STOP else 13,
        source_prior=prior,initial_rows=initial_rows,algorithm_class=cls,
        problem_kwargs=geometry['problem_kwargs'])
    if 'observation_seed_by_task' in plan:
        engine.config.observation_seed = int(plan['observation_seed_by_task'][str(seed)])
    engine.backend_arm = 'legacy_native_v69' if arm=='legacy_native_neighbors' else arm
    if arm in SOURCE_PERMUTED_ROLE_ARMS:
        permuted_prior = load_pickle(
            ROOT/plan['source_permuted_assets']['native_source_prior_file'])
        engine.role_projection_context = build_role_projection_context(engine, permuted_prior)
    reload_backend_bundle(engine, plan, arm)
    return engine

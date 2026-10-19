"""Restore frozen paid13 terminal moments; never run search or verification."""
import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np

from performance.diagnose_mainline_terminal_backend import freeze, signature, challenger_error
from performance.mainline_legacy_backend import posterior_views
from performance.mainline_role_prior import role_moment_state
from performance.mainline_role_after_init import after_init_point_only_moment_state
from performance.mainline_role_point_only import point_only_moment_state
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, load_pickle, read, save
from performance.run_mainline_same_policy import transform
from performance.run_mainline_terminal_confirmation import input_record
from representation.joint_function_risk import JointFunctionRiskPosterior

MANIFEST = ROOT/'performance/manifests/mainline_target_terminal_risk_reconstruction_v1.json'
FUNCTIONS = {'legacy_role_full': role_moment_state,
             'legacy_role_after_init_point_only': after_init_point_only_moment_state,
             'legacy_role_point_only': point_only_moment_state}
ZERO = dict(new_source_calls=0, new_target_initial_calls=0, new_target_search_calls=0,
            new_verification_calls=0, new_truth_function_calls=0, new_source_fits=0,
            new_rng_draws=0, download_bytes=0)


def directory(plan, unit):
    return ROOT/plan['output_dir']/unit['domain']/f"seed_{unit['seed']}"/unit['arm']


@contextmanager
def measure_conditioning(cost):
    """Count the existing primitive work, including the prefix of a failed call."""
    construct = JointFunctionRiskPosterior.from_moments
    update = JointFunctionRiskPosterior.update_batch
    cost.update(posterior_field_restorations=0, head_batch_updates_attempted=0,
                head_batch_updates_completed=0, component_node_raw_updates_attempted=0)

    def counted_construct(*args, **kwargs):
        cost['posterior_field_restorations'] += 1
        return construct(*args, **kwargs)

    def counted_update(self, candidate, g_label, eta_label, replicates):
        cost['head_batch_updates_attempted'] += 1
        cost['component_node_raw_updates_attempted'] += len(self.means)*replicates
        value = update(self, candidate, g_label, eta_label, replicates)
        cost['head_batch_updates_completed'] += 1
        return value

    with patch.object(JointFunctionRiskPosterior, 'from_moments', counted_construct), \
         patch.object(JointFunctionRiskPosterior, 'update_batch', counted_update):
        yield


def reconstruct(plan, unit, bundle, parent):
    output = directory(plan, unit)
    if (output/'outcome.json').exists():
        return
    result = {k: unit[k] for k in ('domain', 'seed', 'arm')}
    result.update(source_paths=unit, truth_used=False, verification_performed=False)
    if unit['status'] != 'registered_available_not_started':
        failure = next(r for r in plan['prior_failures']
                       if r['seed'] == unit['seed'] and r['arm'] == unit['arm'])
        save(output/'outcome.json', dict(result, status='registered_prior_search_failure',
             prior_failure=failure, sealed_utc=datetime.now(timezone.utc).isoformat()))
        return
    try:
        with attempt(output, 'restore-frozen-terminal', unit['domain'], unit['arm']) as cost:
            cost.update(ZERO, seed=unit['seed'], engine_restorations=0,
                        full_history_conditionings=0, full_history_raw_rows=0,
                        moment_states_attempted=0, moment_states_completed=0)
            saved = read(ROOT/unit['search_file'])
            expected = read(ROOT/unit['transformed_terminal_input'])
            engine = load_pickle(ROOT/unit['engine_file'])
            cost['engine_restorations'] = 1
            engine.backend_bundle = bundle
            history = [dict(point=list(p), raw=np.asarray(y).tolist()) for p, y in engine.history]
            assert history == saved['history'] and len(history) == 13
            assert engine.backend_arm == unit['arm'] and saved['seed'] == unit['seed']
            assert engine.problem.d == 1000 and engine.config.n0 == 10
            assert saved['geometry'] == expected['geometry']
            for key, value in saved['geometry'].items():
                np.testing.assert_array_equal(getattr(engine.problem.base.base, key), value)
            assert engine.problem.alpha == expected['task']['alpha']
            assert engine.problem.tau == expected['tau']
            bank = list(dict.fromkeys(tuple(row['point']) for row in history))
            index = {p: i for i, p in enumerate(bank)}
            result.update(bank=bank, history_indices=[index[tuple(r['point'])] for r in history],
                history_raw=[r['raw'] for r in history],
                task=dict(d=1000, L=engine.problem.L, alpha=engine.problem.alpha,
                          tau=engine.problem.tau, geometry=saved['geometry'],
                          guard=max(float(engine.problem.pilot_constraint_guard()), 0.)
                          if hasattr(engine.problem, 'pilot_constraint_guard') else 0.))
            rng_before = {name: deepcopy(getattr(engine, name).bit_generator.state)
                          for name in ('rng', 'rec_rng')}
            with ExitStack() as stack:
                for obj, name in ((engine.problem, 'simulate'), (engine, '_true_chance_margin'),
                                  (engine, '_evaluate_recommendation')):
                    stack.enter_context(patch.object(obj, name,
                        side_effect=AssertionError('Registered reconstruction cannot simulate or score truth')))
                stack.enter_context(measure_conditioning(cost))
                function = FUNCTIONS[unit['arm']]
                cost['moment_states_attempted'] += 1
                if unit['arm'] == 'legacy_role_full':
                    cost['full_history_conditionings'] += 1
                    cost['full_history_raw_rows'] += len(history)
                    state = function(bundle, engine, bank, history)
                else:
                    state = function(bundle, engine, bank, history,
                                     cache=engine._point_only_cache)
                cost['moment_states_completed'] += 1
                cost.update(state.get('cost', {}))
                assert all(np.isfinite(state[k]).all() for k in ('mean', 'cov', 'noise'))
                n = len(bank)
                result['moments'] = dict(mean=state['mean'],
                    variance=np.diag(state['cov'].reshape(4*n, 4*n)).reshape(n, 4),
                    noise=state['noise'], weights=state['weights'],
                    policy_weights=state.get('policy_weights'), identity_names=bundle['identities'],
                    covariance_semantics=state.get('covariance_semantics', 'full raw mixture moments'),
                    roundoff_events=state.get('roundoff_events', []), cost=state.get('cost', {}),
                    head_order=['f', 'eta_f', 'g_minus_tau', 'eta_g'])
                # Reuse this exact state in the original decision path, not a second replay.
                engine._cache_paid = len(history)
                engine._moment_cache = {tuple(bank): posterior_views(bank, state, engine.problem.tau)}
                original = freeze(engine, native=False)
                result['original'] = original
                assert signature(original) == signature(saved), 'Original terminal ordering mismatch'
                error = challenger_error(original, saved)
                cell = next(c for c in plan['reference_terminal_arms'] if c['name'] == unit['arm'])
                item = input_record(saved, original['terminal'], parent, cell, unit['seed'],
                    expected['verification_seed'], expected['source_case'], engine.problem.tau)
                changed = transform(item)
                assert changed == expected, 'Transformed terminal input mismatch'
                result['transformed'] = {k: changed[k] for k in ('primary', 'frozen_shortlist',
                    'objective_incumbent_position', 'candidate_budgets', 'protocol', 'input_changed')}
                assert set(tuple(r['point']) for r in original['observed']) == set(bank)
            assert history == [dict(point=list(p), raw=np.asarray(y).tolist()) for p, y in engine.history]
            assert rng_before == {name: getattr(engine, name).bit_generator.state for name in rng_before}
            result.update(status='complete', parity=dict(history=True, original=True,
                transformed=True, challenger_max_error=error), new_rng_draws=0)
    except Exception as error:
        result.update(status='reconstruction_failed', reason=repr(error))
    result['sealed_utc'] = datetime.now(timezone.utc).isoformat()
    save(output/'outcome.json', result)
    print(unit['domain'], unit['seed'], unit['arm'], result['status'],
          result.get('reason', ''), flush=True)


def run_family(plan, domain, first_only=False):
    output = ROOT/plan['output_dir']/domain
    units = [u for u in plan['units'] if u['domain'] == domain]
    if first_only:
        units = units[:1]
    pending = [u for u in units if not (directory(plan, u)/'outcome.json').exists()]
    if not pending:
        return
    source = next(a for a in plan['frozen_assets'] if a['domain'] == domain)
    parent_plan = read(ROOT/plan['source_manifest'])
    prepared = read(ROOT/parent_plan['output_dir']/'prepared.json')
    family = next(f for r in prepared['replicates'] for f in r['families'] if f['domain'] == domain)
    parent = read(ROOT/family['search_manifest'])
    with attempt(output, 'load-frozen-source', domain) as cost:
        cost.update(ZERO)
        bundle = load_pickle(ROOT/source['frozen_assets']['source_models_file'])
        cost['source_model_restorations'] = 1
    for unit in pending:
        reconstruct(plan, unit, bundle, parent)


def seal(plan):
    rows = []
    for unit in plan['units']:
        path = directory(plan, unit)/'outcome.json'
        record = read(path)
        assert record['status'] in ('complete', 'reconstruction_failed', 'registered_prior_search_failure')
        rows.append(dict(domain=unit['domain'], seed=unit['seed'], arm=unit['arm'],
                         status=record['status'], file=str(path.relative_to(ROOT))))
    save(ROOT/plan['output_dir']/'posterior_sealed.json', dict(status='sealed',
         registered_units=72, records=rows, truth_used=False,
         counts={s:sum(r['status'] == s for r in rows) for s in set(r['status'] for r in rows)},
         sealed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(MANIFEST))
    parser.add_argument('--family')
    parser.add_argument('--first-only', action='store_true')
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    plan = read(args.manifest)
    if args.seal:
        seal(plan)
    else:
        run_family(plan, args.family, args.first_only)

"""Frozen native/role forecasts on identical archived initial10+3 histories.

No search or terminal verification is run. The native posterior uses the real
initial fit, pilot and HVD feedback; role arms retain their deployed operators.
Scoring repeats are never read here.
"""
from unittest.mock import patch

import numpy as np

from acquisition.decision_backends import feasible_first_terminal_order
from algorithms.single_olhkg import SingleOLHKGAlgorithm
from performance import mainline_legacy_native as native
from performance import mainline_role_after_init as after_init
from performance import mainline_role_point_only as point_only
from performance import mainline_role_prior as role
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.mainline_legacy_backend import posterior_views
from performance.mainline_role_source_holdout import build_source_engine

ARMS = ('legacy_native', 'legacy_role_full',
        'legacy_role_after_init_point_only', 'legacy_role_point_only')


def _json(value):
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def initialize_native(prior, task, rows, heldout_family, cost):
    """Use archived d50 policies with the complete frozen native initializer."""
    materialize = native.materialize_v69_task
    build_problem = native.benchmark.build_scalarized_problem

    def source_task(domain, seed=80, N=13, **kwargs):
        result = materialize(heldout_family, seed=seed, N=N, **kwargs)
        result['heldout'] = domain
        args = result['args']
        # This registered history is not the old d1000 source-informed design.
        # _fit_initial_belief receives the fixed held rows directly; no automatic
        # design generation is invoked.
        for key in list(args):
            if key.startswith('initial_design'):
                del args[key]
        args.update(d=int(task.X.shape[1]), initial_design='auto')
        return result

    def provider(*args, **kwargs):
        return no_outcomes(build_problem(*args, **kwargs))

    with patch.object(native, 'materialize_v69_task', source_task), patch.object(
            native.benchmark, 'build_scalarized_problem', provider):
        engine = native.build_native(task.name, initial_rows=rows, source_prior=prior)
    no_outcomes(engine.problem)
    assert engine.problem.d == task.X.shape[1]
    assert engine.problem.tau == task.tau and engine.problem.alpha == task.alpha
    try:
        engine._fit_initial_belief([tuple(row['point']) for row in rows[:10]])
    finally:
        cost['native_replayed_raw_rows'] = cost.get('native_replayed_raw_rows', 0) + engine.replayed_initial_calls
    return engine


def update_native(engine, row):
    """The deployed sequential feedback block, without selecting a new query."""
    engine._refresh_sequential_basis()
    point = tuple(map(int, row['point']))
    x = np.asarray(point, dtype=float)
    mean = [model.posterior_mean(x) for model in engine.gpr]
    epistemic = [model.posterior_var(x) for model in engine.gpr]
    noise = [engine.variance_model.predict_variance(i, x, engine.problem)
             for i in range(2)]
    raw = engine._simulate_and_store(point)
    for i, model in enumerate(engine.gpr):
        model.update(x, raw[i], noise[i])
    if str(engine.config.source_constraint_mean_adaptation_mode).strip().lower() in {
            'sequential_evidence_mixture', 'sequential_mixture'}:
        engine._configure_hvd_source_task_posterior(engine.variance_model, engine.gpr)
    stored = engine.observations[point]
    for i, model in enumerate(engine.gpr):
        values = [float(value[i]) for value in stored]
        engine.variance_model.update(i, x, raw[i], mean[i], model, engine.problem,
            epistemic_var=epistemic[i], replicate_count=len(values),
            replicate_variance=float(np.var(values, ddof=1)) if len(values) > 1 else None)
    if engine.task_ensemble is not None:
        engine.task_ensemble.update(x, raw, existing_observations=stored[:-1], tau=engine.problem.tau)
    engine._update_posterior_dominance_incumbent(
        stage=len(engine.history), reason='budgeted_target_observation')


def terminal_rank(snapshot, held_policy_ids):
    """Apply the actual frozen feasible-first rule to the observed13 only."""
    objective = np.asarray(snapshot['mean'], dtype=float)[:13, 0]
    probability = np.asarray(snapshot['probability_violation'], dtype=float)[:13]
    order, mode = feasible_first_terminal_order(
        objective, probability, maximum_violation_probability=.5)
    eligible = np.flatnonzero(probability <= .5).tolist()
    return dict(selected_index=int(order[0]), eligible_indices=eligible,
        ordered_indices=order.tolist(), selected_policy_id=int(held_policy_ids[int(order[0])]),
        no_eligible=not eligible, mode=mode, maximum_violation_probability=.5)


def native_snapshot(engine, bank):
    """Expose the moments and probabilities used by the frozen native decision."""
    ensemble = engine.task_ensemble
    assert engine._certification_head_authority() == 'task_joint'
    assert engine.config.decision_aleatoric_mode == 'posterior_central'
    assert engine.config.decision_ambiguity_mode == 'posterior_nominal'
    components = engine._terminal_bayes_risk_components(engine.gpr, engine.variance_model,
        bank, task_ensemble=ensemble, risk_penalty=engine.config.decision_risk_penalty)
    # These are the same expert moments/weights used inside terminal components;
    # the probability is the expert mixture, not a Gaussian of averaged moments.
    _, _, obj_noise = ensemble.expert_moments_many(0, bank, certification=False)
    con_mu, con_epi, con_noise = ensemble.expert_moments_many(1, bank, certification=False)
    weights = ensemble.posterior.decision_weights()
    objective_weights = (ensemble.posterior.posterior_weights()
                         if ensemble.posterior.safe_generalized else weights)
    constraint = weights @ con_mu
    constraint_variance = weights @ np.maximum(con_epi, 0.) + weights @ (con_mu-constraint[None, :])**2
    for values in (components['objective'], components['objective_variance'],
                   components['probability_violation'], constraint,
                   constraint_variance, obj_noise, con_noise):
        if not np.isfinite(values).all():
            raise RuntimeError('Nonfinite native posterior moments')
    missing = np.full(len(bank), np.nan)
    return dict(status='complete',
        mean=np.column_stack([components['objective'], missing, constraint-engine.problem.tau, missing]),
        variance=np.column_stack([components['objective_variance'], missing, constraint_variance, missing]),
        noise=np.column_stack([objective_weights @ np.maximum(obj_noise, 0.), weights @ np.maximum(con_noise, 0.)]),
        probability_violation=components['probability_violation'],
        roundoff_events=[], task_weights=weights,
        moment_semantics='Native terminal expert-mixture response moments, including any frozen expert constraint guard; eta moments unavailable.',
        probability_semantics='Actual native terminal mixture of expert violation probabilities.')


def role_snapshot(engine, bank, state):
    if not all(np.isfinite(state[key]).all() for key in ('mean', 'cov', 'noise')):
        raise RuntimeError('Nonfinite posterior moments')
    models, variance = posterior_views(bank, state, engine.problem.tau)
    components = SingleOLHKGAlgorithm._terminal_bayes_risk_components(
        engine, models, variance, bank, task_ensemble=None,
        risk_penalty=engine.config.decision_risk_penalty)
    return dict(status='complete', mean=state['mean'],
        variance=np.diag(state['cov'].reshape(4*len(bank), 4*len(bank))).reshape(len(bank), 4),
        noise=state['noise'], probability_violation=components['probability_violation'],
        roundoff_events=state.get('roundoff_events', []),
        identity_weights=state.get('policy_weights', state.get('weights')),
        operator_cost=state.get('cost', {}),
        probability_semantics='Deployed role Gaussian moment terminal probability.')


def _failed(record, prefix, error):
    failure = dict(prefix=prefix, error=repr(error))
    record['snapshots'][str(prefix)] = dict(status='failed', **failure)
    record['failures'].append(failure)
    print('forecast prefix failed', prefix, repr(error), flush=True)


def forecast(bundle, prior, task, split, cost, heldout_family):
    held = list(map(int, split['held']))
    bank = [tuple(map(int, point)) for point in task.X_integer[held]]
    rows = [dict(point=list(bank[i]), raw=task.Y_replicates[held[i]][0].tolist())
            for i in range(13)]
    cost['paired_raw_adaptation_exposures'] = cost.get('paired_raw_adaptation_exposures', 0) + 13
    predictions = {arm: dict(status='complete', snapshots={}, terminal13=None, failures=[]) for arm in ARMS}
    record = predictions['legacy_native']
    engine = None
    print('forecast starts', task.name, 'legacy_native', 10, flush=True)
    try:
        engine = initialize_native(prior, task, rows, heldout_family, cost)
        record['snapshots']['10'] = native_snapshot(engine, bank)
    except Exception as error:
        _failed(record, 10, error)
    if record['snapshots']['10']['status'] == 'complete':
        before = engine.replayed_initial_calls
        print('forecast starts', task.name, 'legacy_native', 13, flush=True)
        try:
            for row in rows[10:]:
                update_native(engine, row)
            record['snapshots']['13'] = native_snapshot(engine, bank)
            record['terminal13'] = terminal_rank(record['snapshots']['13'], held)
        except Exception as error:
            _failed(record, 13, error)
        finally:
            cost['native_replayed_raw_rows'] = cost.get('native_replayed_raw_rows', 0) + engine.replayed_initial_calls-before
    else:
        _failed(record, 13, RuntimeError('Native prefix10 failed; no continuation or retry.'))

    try:
        projection = build_source_engine(prior, task.name, d=task.X.shape[1], heldout_family=heldout_family)
        assert projection.problem.tau == task.tau and projection.problem.alpha == task.alpha
    except Exception as error:
        for arm in ARMS[1:]:
            for prefix in (10, 13):
                _failed(predictions[arm], prefix, error)
    else:
        operators = ((ARMS[1], role.role_moment_state),
                     (ARMS[2], after_init.after_init_point_only_moment_state),
                     (ARMS[3], point_only.point_only_moment_state))
        for arm, operator in operators:
            record, cache = predictions[arm], {}
            for prefix in (10, 13):
                if prefix == 13 and record['snapshots']['10']['status'] != 'complete':
                    _failed(record, prefix, RuntimeError('Prefix10 failed; no continuation or retry.'))
                    continue
                cost['role_posterior_evaluations'] = cost.get('role_posterior_evaluations', 0) + 1
                print('forecast starts', task.name, arm, prefix, flush=True)
                try:
                    kwargs = {} if arm == ARMS[1] else dict(cache=cache)
                    state = operator(bundle, projection, bank, rows[:prefix], **kwargs)
                    record['snapshots'][str(prefix)] = role_snapshot(projection, bank, state)
                    if prefix == 13:
                        record['terminal13'] = terminal_rank(record['snapshots']['13'], held)
                except Exception as error:
                    _failed(record, prefix, error)
    for record in predictions.values():
        if record['failures']:
            record['status'] = ('partial_failure' if any(
                item['status'] == 'complete' for item in record['snapshots'].values()) else 'failed')
    print('forecast case frozen', task.name,
          {arm: record['status'] for arm, record in predictions.items()}, flush=True)
    return _json(dict(family=task.name, archive_heldout_family=heldout_family,
        fold=split.get('fold'), d=int(task.X.shape[1]), tau=float(task.tau), alpha=float(task.alpha),
        held_policy_ids=held, adaptation_raw=rows, predictions=predictions,
        head_order=['f', 'eta_f', 'g_centered', 'eta_g'], noise_head_order=['f', 'g'],
        native_initialization='Actual initial10 GPR fit, HVD initialization and task pilot; fixed3 native updates.',
        role_projection='Unchanged source-only role law and deployed FULL/AFTER/POINT operators.'))

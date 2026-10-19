"""Own-policy raw updates from the fixed role prior, starting at source zero.

Every policy has a separate copy of source identity and quadrature-node weights.
Only that policy's paid observations update its four latent marginals and noise.
For TS, C = D C_source D retains source correlations while matching these local
posterior variances. This is a Gaussian moment control, not a joint Bayesian
posterior: even within-policy cross-head correlations are held at source zero.
The native one-policy filters used for subsequent raw updates retain their own
complete covariance. A cache belongs to one engine and its frozen source bundle.
"""

import numpy as np

from performance.mainline_moment_policy import moment_state
from performance.mainline_role_prior import role_inputs, role_moment_state
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture


def _point_model(bundle, engine, point):
    """Project directly to two-dimensional fields instead of coefficient slots."""
    source, designs = role_inputs(bundle, engine, [point])
    components = []
    for pair, design_pair in zip(source['components'], designs):
        fields = []
        for record, (D, E) in zip(pair, design_pair):
            d, e = D[0], E[0]
            means = np.column_stack([record['mean'] @ d, record['eta_mean'] @ e])
            covariance = np.zeros((len(means), 2, 2))
            covariance[:, 0, 0] = (
                np.einsum('p,kpq,q->k', d, record['covariance'], d)
                + d @ record['law']['task'] @ d + record['law']['white'])
            covariance[:, 1, 1] = (e @ (record['eta_covariance']
                + record['risk']['task']) @ e + record['risk']['white'])
            field = JointFunctionRiskPosterior.from_moments(
                means[:1], covariance[0].reshape(1, 2, 1, 2))
            field.means = means.copy()
            field.covariances = covariance
            field.log_weights = record['log_weights'].copy()
            field.component_pairs = [(0, 0)] * len(means)
            fields.append(field)
        components.append(tuple(fields))
    return JointResponseMixture(source['identities'], components,
                                np.exp(source['log_weights']))


def point_only_moment_state(bundle, engine, bank, rows=(), *,
                            component_batch_size=64, cache=None):
    """Return source-correlated TS moments with strictly local raw learning.

The source bank is projected once per changed bank. Already observed policies
keep small native filters and consume only newly appended own raw rows. Returned
roundoff events and cost counters cover work performed by this call only.
``weights`` denotes unchanged source weights; ``policy_weights`` contains the
different local identity posteriors, including source weights at unsampled points.
"""
    if cache is None:
        cache = {}
    points = tuple(tuple(point) for point in bank)
    cost = dict(source0_projections=0, own_model_restorations=0, own_raw_updates=0)
    events = []
    if cache.get('source_bank') != points:
        cache['source_state'] = role_moment_state(bundle, engine, bank,
            component_batch_size=component_batch_size)
        cache['source_bank'] = points
        cost['source0_projections'] += 1
        events.extend(cache['source_state']['roundoff_events'])
    source = cache['source_state']
    mean, noise = source['mean'].copy(), source['noise'].copy()
    source_covariance = source['cov'].reshape(4 * len(bank), 4 * len(bank))
    source_variance = np.diag(source_covariance)
    variance = source_variance.copy().reshape(len(bank), 4)
    policy_weights = np.broadcast_to(source['weights'],
                                    (len(bank), len(source['weights']))).copy()
    own_rows = {}
    for row in rows:
        own_rows.setdefault(tuple(row['point']), []).append(tuple(row['raw']))
    local_cache = cache.setdefault('own', {})
    for index, point in enumerate(points):
        raws = tuple(own_rows.get(point, ()))
        if not raws:
            continue
        entry = local_cache.get(point)
        if entry is None or raws[:len(entry['raws'])] != entry['raws']:
            entry = dict(raws=(), model=_point_model(bundle, engine, point))
            local_cache[point] = entry
            cost['own_model_restorations'] += 1
        model = entry['model']
        before = [[len(getattr(field, 'roundoff_events', [])) for field in pair]
                  for pair in model.components]
        new_raws = raws[len(entry['raws']):]
        for raw in new_raws:
            model.update_raw(0, [raw[0], raw[1] - engine.problem.tau])
        cost['own_raw_updates'] += len(new_raws)
        if new_raws:
            entry['raws'], entry['state'] = raws, moment_state(model)
        state = entry['state']
        mean[index], noise[index] = state['mean'][0], state['noise'][0]
        variance[index] = np.diag(state['cov'].reshape(4, 4))
        policy_weights[index] = model.weights
        for identity, pair, starts in zip(model.names, model.components, before):
            for head, (field, start) in enumerate(zip(pair, starts)):
                events.extend(dict(event, policy=list(point), identity=identity,
                                   head=('f', 'g')[head])
                              for event in getattr(field, 'roundoff_events', [])[start:])
    scale = np.sqrt(variance.reshape(-1) / source_variance)
    covariance = source_covariance * scale[:, None] * scale[None, :]
    return dict(mean=mean, cov=covariance.reshape(len(bank), 4, len(bank), 4),
                noise=noise, weights=source['weights'].copy(),
                policy_weights=policy_weights, paid_episodes=len(rows),
                roundoff_events=events, cost=cost,
                covariance_semantics='fixed source0 correlations; own-only marginal variances')

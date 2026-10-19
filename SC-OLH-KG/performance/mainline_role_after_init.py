"""Normal initial10 learning, followed by strictly own-policy raw updates.

The frozen initial10 posterior retains every conditional quadrature node and
identity, including within-node response/log-variance covariance. New policies
are projected from that same posterior. Only subsequent own-policy observations
change each policy's node and identity weights and four latent marginals.

Decision covariance retains initial10 correlations, scaled to current marginal
standard deviations. This is a Gaussian moment control, not a joint posterior
after the switch; a covariance block involving an updated point may rescale.
"""

import numpy as np
from scipy.special import logsumexp

from performance.diagnose_mainline_initialization import stream_head
from performance.mainline_legacy_source_models import _merge_moments
from performance.mainline_moment_policy import moment_state
from performance.mainline_role_prior import _history, role_inputs, role_moment_state
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture


def _initial_conditioning(bundle, engine, rows, batch_size):
    points = list(dict.fromkeys(tuple(row['point']) for row in rows))
    source, designs = role_inputs(bundle, engine, points)
    history = _history(engine, points, rows)
    pairs, masses, events = [], [], []
    for identity, pair, prior_mass, design_pair in zip(
            source['identities'], source['components'], source['log_weights'], designs):
        fields, head_masses = [], []
        for head, (record, (D, E)) in enumerate(zip(pair, design_pair)):
            chunks = []

            def retain(prefix, start, mass, field, *_):
                if prefix == len(history):
                    chunks.append((field.means.copy(), field.covariances.copy(),
                                   mass + field.log_weights))

            _, roundoff = stream_head(record, D, E, history, head, batch_size,
                                     observer=retain)
            means = np.concatenate([chunk[0] for chunk in chunks])
            covariances = np.concatenate([chunk[1] for chunk in chunks])
            logs = np.concatenate([chunk[2] for chunk in chunks])
            mass = float(logsumexp(logs))
            count = means.shape[1] // 2
            field = JointFunctionRiskPosterior.from_moments(
                means[0].reshape(count, 2), covariances[0].reshape(count, 2, count, 2))
            field.means, field.covariances = means, covariances
            field.log_weights = logs - mass
            field.component_pairs = [(0, 0)] * len(means)
            fields.append(field)
            head_masses.append(mass)
            events.extend(dict(event, identity=identity) for event in roundoff)
        pairs.append(tuple(fields))
        masses.append(prior_mass + sum(head_masses))
    return dict(points=points, components=pairs,
                log_weights=np.asarray(masses) - logsumexp(masses)), events


def _projection(initial, points, D, E):
    """Keep initialized policies' white terms; new policies get fresh latents."""
    no, p = len(initial['points']), max(D.shape[1], E.shape[1])
    LF, LE = np.zeros((len(points), no+p)), np.zeros((len(points), no+p))
    LF[:, no:no+D.shape[1]], LE[:, no:no+E.shape[1]] = D, E
    original = {point: index for index, point in enumerate(initial['points'])}
    unobserved = np.ones(len(points), dtype=bool)
    for index, point in enumerate(points):
        if point in original:
            LF[index] = 0.; LE[index] = 0.
            LF[index, original[point]] = LE[index, original[point]] = 1.
            unobserved[index] = False
    return LF, LE, unobserved


def _initial_state(source, designs, initial, points):
    n, total = len(points), None
    for pair, records, design_pair, mass in zip(initial['components'],
            source['components'], designs, initial['log_weights']):
        means, covariances, noises = [], [], []
        for field, record, (D, E) in zip(pair, records, design_pair):
            LF, LE, unobserved = _projection(initial, points, D, E)
            mean, covariance = field.predict(full_cov=True)
            transform = np.zeros((2*n, field.means.shape[1]))
            transform[0::2, 0::2], transform[1::2, 1::2] = LF, LE
            projected_covariance = transform @ covariance.reshape(
                field.means.shape[1], field.means.shape[1]) @ transform.T
            projected_covariance += np.diag(np.column_stack([
                record['law']['white'] * unobserved,
                record['risk']['white'] * unobserved]).reshape(-1))
            means.append((transform @ mean.reshape(-1)).reshape(n, 2))
            covariances.append(projected_covariance.reshape(n, 2, n, 2))
            eta_mean = field.means[:, 1::2] @ LE.T
            eta_variance = np.einsum('ni,kij,nj->kn', LE,
                field.covariances[:, 1::2, 1::2], LE)
            eta_variance += record['risk']['white'] * unobserved
            noises.append(np.exp(field.log_weights) @ np.exp(eta_mean + eta_variance/2))
        covariance = np.zeros((n, 4, n, 4))
        covariance[:, :2, :, :2], covariance[:, 2:, :, 2:] = covariances
        total = _merge_moments(total, mass, np.column_stack(means).reshape(-1),
                              covariance.reshape(4*n, 4*n), np.column_stack(noises))
    _, mean, covariance, noise = total
    return dict(mean=mean.reshape(n, 4), cov=covariance.reshape(n, 4, n, 4),
                noise=noise, weights=np.exp(initial['log_weights']))


def _conditional_point_model(source, designs, initial, point, index):
    components = []
    for pair, records, design_pair in zip(initial['components'],
                                        source['components'], designs):
        fields = []
        for field, record, (D, E) in zip(pair, records, design_pair):
            LF, LE, unobserved = _projection(initial, [point], D[index:index+1], E[index:index+1])
            d, e = LF[0], LE[0]
            means = np.column_stack([field.means[:, 0::2] @ d, field.means[:, 1::2] @ e])
            covariance = np.zeros((len(means), 2, 2))
            covariance[:, 0, 0] = np.einsum('p,kpq,q->k', d,
                field.covariances[:, 0::2, 0::2], d) + record['law']['white'] * unobserved[0]
            covariance[:, 1, 1] = np.einsum('p,kpq,q->k', e,
                field.covariances[:, 1::2, 1::2], e) + record['risk']['white'] * unobserved[0]
            covariance[:, 0, 1] = covariance[:, 1, 0] = np.einsum('p,kpq,q->k', d,
                field.covariances[:, 0::2, 1::2], e)
            local = JointFunctionRiskPosterior.from_moments(
                means[:1], covariance[0].reshape(1, 2, 1, 2))
            local.means, local.covariances = means, covariance
            local.log_weights = field.log_weights.copy()
            local.component_pairs = [(0, 0)] * len(means)
            fields.append(local)
        components.append(tuple(fields))
    return JointResponseMixture(source['identities'], components, np.exp(initial['log_weights']))


def after_init_point_only_moment_state(bundle, engine, bank, rows=(), *,
                                       component_batch_size=64, cache=None):
    """Full initial10, then own raw learning with fixed initial10 correlations."""
    if len(rows) < 10:
        return role_moment_state(bundle, engine, bank, rows,
                                 component_batch_size=component_batch_size)
    if cache is None:
        cache = {}
    initial_key = tuple((tuple(row['point']), tuple(row['raw'])) for row in rows[:10])
    cost = dict(initial10_conditionings=0, initial10_projections=0,
                own_model_restorations=0, own_raw_updates=0)
    events = []
    if cache.get('initial_key') != initial_key:
        cache.clear()
        cache['initial'], events = _initial_conditioning(
            bundle, engine, rows[:10], component_batch_size)
        cache['initial_key'] = initial_key
        cost['initial10_conditionings'] += 1
    initial, points = cache['initial'], tuple(tuple(point) for point in bank)
    if cache.get('bank') != points:
        source, designs = role_inputs(bundle, engine, bank)
        # Search checkpoints retain conditioned nodes, not another source archive.
        cache['source'] = dict(identities=source['identities'], components=[tuple(
            dict(law=dict(white=record['law']['white']),
                 risk=dict(white=record['risk']['white'])) for record in pair)
            for pair in source['components']])
        cache['designs'] = designs
        cache['base'] = _initial_state(cache['source'], designs, initial, points)
        cache['bank'] = points
        cost['initial10_projections'] += 1
    base = cache['base']
    mean, noise = base['mean'].copy(), base['noise'].copy()
    base_covariance = base['cov'].reshape(4*len(bank), 4*len(bank))
    base_variance = np.diag(base_covariance)
    variance = base_variance.copy().reshape(len(bank), 4)
    policy_weights = np.broadcast_to(base['weights'],
                                    (len(bank), len(base['weights']))).copy()
    own_rows = {}
    for row in rows[10:]:
        own_rows.setdefault(tuple(row['point']), []).append(tuple(row['raw']))
    own_cache = cache.setdefault('own', {})
    for index, point in enumerate(points):
        raws = tuple(own_rows.get(point, ()))
        if not raws:
            continue
        entry = own_cache.get(point)
        if entry is None or raws[:len(entry['raws'])] != entry['raws']:
            entry = dict(raws=(), model=_conditional_point_model(
                cache['source'], cache['designs'], initial, point, index))
            own_cache[point] = entry
            cost['own_model_restorations'] += 1
        model = entry['model']
        before = [[len(getattr(field, 'roundoff_events', [])) for field in pair]
                  for pair in model.components]
        new_raws = raws[len(entry['raws']):]
        for raw in new_raws:
            model.update_raw(0, [raw[0], raw[1]-engine.problem.tau])
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
    scale = np.sqrt(variance.reshape(-1) / base_variance)
    covariance = base_covariance * scale[:, None] * scale[None, :]
    return dict(mean=mean, cov=covariance.reshape(len(bank), 4, len(bank), 4),
                noise=noise, weights=base['weights'].copy(), policy_weights=policy_weights,
                paid_episodes=len(rows), roundoff_events=events, cost=cost,
                covariance_semantics='fixed initial10 correlations; post-initial own-only marginal variances')

"""Frozen legacy constraint-mean marginal in the current raw target filter.

The legacy source-only Gaussian coefficient law replaces g at every current
source node. The current eta marginal, source-node and identity weights, and
objective pair are retained. This is a marginal-prior intervention: source
observations are not conditioned on again and source g/eta dependence is cut.
"""

import numpy as np
from scipy.special import logsumexp

from performance.diagnose_mainline_initialization import merge, stream_head
from performance.mainline_legacy_source_models import (
    _merge_moments, _query_design, _stream_head,
)
from representation.legacy_function_prior import legacy_public_coordinates
from representation.legacy_observable_prior import legacy_exposure_descriptors


def role_prior(engine):
    """Read the source coefficient law, including the frozen latent split.

    The returned mean uses native raw-g units. No fitted target coefficients,
    observations, or target likelihood scores enter these two native methods.
    """
    model = engine.gpr[1]
    prior = engine._source_constraint_coefficient_prior(model, 1)
    return engine._calibrate_source_constraint_deviation(model, prior)


def role_inputs(bundle, engine, bank):
    """Return a shallow source bundle and [(D_f,E_f),(D_g,E_g)] per identity."""
    problem = engine.problem
    prior = role_prior(engine)
    coefficient_mean = np.asarray(prior['mean']).copy()
    coefficient_mean[0] -= float(problem.tau)
    coefficient_covariance = np.asarray(prior['covariance'])
    role_design = engine.gpr[1].basis_matrix(bank)
    public = (legacy_public_coordinates(np.asarray(bank)/problem.L)
              - bundle['mean_center'])/bundle['mean_scale']
    risk = (bundle['encoder'].transform(legacy_exposure_descriptors(problem, bank, 50))
            - bundle['risk_center'])/bundle['risk_scale']
    coordinates = (public, risk)
    replacement = dict(bundle)
    replacement['components'] = []
    designs = []
    for identity, pair in zip(bundle['identities'], bundle['components']):
        objective, constraint = pair
        count = len(constraint['mean'])
        law = dict(mean=coefficient_mean, common=coefficient_covariance,
                   task=np.zeros_like(coefficient_covariance),
                   white=float(prior['deviation_variance']))
        constraint_role = dict(constraint, law=law,
            mean=np.broadcast_to(coefficient_mean, (count, len(coefficient_mean))),
            covariance=np.broadcast_to(coefficient_covariance,
                                       (count, *coefficient_covariance.shape)))
        replacement['components'].append((objective, constraint_role))
        designs.append((
            (_query_design(bundle, identity, 0, coordinates, objective['law']),
             _query_design(bundle, identity, 3, coordinates, objective['risk'])),
            (role_design,
             _query_design(bundle, identity, 2, coordinates, constraint['risk'])),
        ))
    return replacement, designs


def _history(engine, bank, rows):
    indices = {tuple(point): index for index, point in enumerate(bank)}
    return [(indices[tuple(row['point'])],
             [row['raw'][0], row['raw'][1]-engine.problem.tau]) for row in rows]


def role_trajectory(bundle, engine, bank, rows, batch_size=64):
    """Replay saved raw rows once and return the existing prefix snapshot API."""
    replacement, designs = role_inputs(bundle, engine, bank)
    history = _history(engine, bank, rows)
    totals = [None]*(len(rows)+1)
    identity_masses, head_masses, events = [], [], []
    for identity, pair, prior_mass, design_pair in zip(
            replacement['identities'], replacement['components'],
            replacement['log_weights'], designs):
        heads = []
        for head, (record, (D, E)) in enumerate(zip(pair, design_pair)):
            trajectory, roundoff = stream_head(record, D, E, history, head, batch_size)
            heads.append(trajectory)
            events.extend([{**event, 'identity': identity} for event in roundoff])
        masses = []
        for prefix, (f, g) in enumerate(zip(*heads)):
            mass = float(prior_mass+f[0]+g[0])
            masses.append(mass)
            totals[prefix] = merge(totals[prefix], mass,
                np.column_stack([f[1], g[1]]), np.column_stack([f[2], g[2]]),
                np.column_stack([f[3], g[3]]))
        identity_masses.append(masses)
        head_masses.append([[f[0], g[0]] for f, g in zip(*heads)])
    identity_masses, head_masses = np.asarray(identity_masses), np.asarray(head_masses)
    snapshots = [dict(paid_prefix=prefix, mean=mean, variance=variance, noise=noise,
                      identity_weights=np.exp(identity_masses[:, prefix]-mass))
                 for prefix, (mass, mean, variance, noise) in enumerate(totals)]
    scores = []
    for prefix in range(len(rows)):
        before = identity_masses[:, prefix]-logsumexp(identity_masses[:, prefix])
        increments = head_masses[:, prefix+1]-head_masses[:, prefix]
        scores.append(dict(heads=logsumexp(before[:, None]+increments, axis=0),
                           joint=float(logsumexp(before+increments.sum(axis=1)))))
    return snapshots, scores, events


def role_moment_state(bundle, engine, bank, rows=(), *, component_batch_size=64):
    """Project full bank moments using the existing streamed raw posterior."""
    replacement, designs = role_inputs(bundle, engine, bank)
    history = _history(engine, bank, rows)
    total, identity_masses, events = None, [], []
    n = len(bank)
    for identity, pair, prior_mass, design_pair in zip(
            replacement['identities'], replacement['components'],
            replacement['log_weights'], designs):
        heads = [_stream_head(record, D, E, history, head, component_batch_size)
                 for head, (record, (D, E)) in enumerate(zip(pair, design_pair))]
        mass = float(prior_mass+sum(result[0] for result in heads))
        identity_masses.append(mass)
        events.extend([{**event, 'identity': identity}
                       for result in heads for event in result[4]])
        mean = np.column_stack([result[1] for result in heads]).reshape(-1)
        covariance = np.zeros((n, 4, n, 4))
        covariance[:, :2, :, :2], covariance[:, 2:, :, 2:] = heads[0][2], heads[1][2]
        noise = np.column_stack([result[3] for result in heads])
        total = _merge_moments(total, mass, mean,
                               covariance.reshape(4*n, 4*n), noise)
    mass, mean, covariance, noise = total
    return dict(mean=mean.reshape(n, 4), cov=covariance.reshape(n, 4, n, 4),
                noise=noise, weights=np.exp(np.asarray(identity_masses)-mass),
                paid_episodes=len(rows), roundoff_events=events)

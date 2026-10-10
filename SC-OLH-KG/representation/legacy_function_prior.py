"""Adapt the historical scalarized source observations, without oracle fields."""

import numpy as np
from scipy.special import digamma, polygamma

from baselines.transfer_archive import dimension_equivariant_profile_features


def legacy_public_coordinates(profiles):
    """Ten public ordered coordinates: mean, SD and eight weighted cosines.

    These are the linear part of the existing normalized 65-column map. The
    function dictionary supplies interactions; no target labels fit this map.
    """
    return dimension_equivariant_profile_features(profiles)[:, :10]


def replicated_source_labels(name, profiles, replicates, tau):
    """Observed-only four-head labels for one independently identified task."""
    response, noise = [], []
    for values in replicates:
        values = np.asarray(values)
        if values.shape[1] != 2 or len(values) < 2:
            raise ValueError("legacy source fitting requires replicated (f,g) observations")
        n = len(values)
        variance = np.var(values, axis=0, ddof=1)
        k = (n - 1) / 2
        eta = np.log(variance[1]) - digamma(k) + np.log(k)
        eta_f = np.log(variance[0]) - digamma(k) + np.log(k)
        response.append([values[:, 0].mean(), values[:, 1].mean() - tau, eta, eta_f])
        noise.append([variance[0] / n, variance[1] / n, float(polygamma(1, k)), float(polygamma(1, k))])
    return {"name": name, "psi": legacy_public_coordinates(profiles),
            "response": np.asarray(response), "noise": np.asarray(noise)}


def legacy_source_labels(archive):
    """Return f, centered g, log constraint variance and log objective variance.

    Archived repeats already contain (scalarized objective, constraint), not
    the base simulator's (f1, f2, g). Only their observed values are consumed;
    constraint_sigma and chance_margin are deliberately not input labels.
    """
    return [replicated_source_labels(task.name, task.X, task.Y_replicates, task.tau)
            for task in archive.tasks]

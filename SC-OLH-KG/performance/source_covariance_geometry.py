"""Function-covariance movement caused by equal allocation on a selected set."""
import numpy as np


def direction_geometry(variances, selected):
    p = np.asarray(variances) / np.sum(variances)
    selected = sorted(set(selected))
    flat = np.zeros(len(p))
    flat[selected] = 1. / len(selected)
    covered = float(p[selected].sum())
    outside = 1. - covered
    distance = float(.5 * np.abs(p - flat).sum())
    extra = float(np.maximum(p[selected] - 1. / len(selected), 0.).sum())
    return {"selected_covariance_mass": covered, "outside_covariance_mass": outside,
            "flat_reallocation_total_variation": distance,
            "extra_within_selection_redistribution": extra,
            "decomposition_error": abs(distance - outside - extra)}

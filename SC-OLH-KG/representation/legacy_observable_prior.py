"""Source-only observable coordinates for the legacy function/risk bridge."""

from dataclasses import replace

import numpy as np

from baselines.transfer_archive import resample_normalized_profiles
from representation.boundary_coordinate import SourceAlignedBoundaryCoordinate
from representation.observable_exposure import (
    canonical_observable_state_descriptor, grouped_policy_state_exposure)
from representation.variance_coordinate import SourceAlignedVarianceRiskCoordinate


def legacy_exposure_descriptors(problem, points, grid_dimension=50):
    """Keep actual control groups, express curve dynamics on one public grid.

    The existing curve DCT scales with sqrt(d). Resampling just its dynamics
    prevents that units change in the supported d=50 to d=1000 transfer.
    Channel means/scales and occupancy retain the actual public action schema.
    No simulator outcomes or cumulative-risk provider are queried.
    """
    profiles = np.vstack([problem.normalize(p) for p in points])
    curves = resample_normalized_profiles(profiles, grid_dimension)
    descriptors = []
    for point, curve in zip(points, curves):
        exposure = problem.observable_state_exposure(point)
        dynamics = grouped_policy_state_exposure(curve, (curve,)).dynamics
        descriptors.append(canonical_observable_state_descriptor(
            replace(exposure, dynamics=dynamics)))
    return np.asarray(descriptors)


class LegacyObservableCoordinate:
    """Two mean axes per response and three variance axes per response.

    All four maps consume the same observable descriptor and archived source
    labels. The orthogonal function prior, rather than these maps' auxiliary
    ridge coefficient laws, defines target functions and uncertainty.
    """

    def fit(self, descriptors, tasks):
        labels = np.vstack([t["response"] for t in tasks])
        domains = np.concatenate([np.repeat(t["name"], len(t["response"])) for t in tasks])
        centered_f = np.concatenate([t["response"][:,0] - t["response"][:,0].mean() for t in tasks])
        self.f_mean = SourceAlignedBoundaryCoordinate(latent_dim=2, input_mode="observable_state_exposure")
        self.g_mean = SourceAlignedBoundaryCoordinate(latent_dim=2, input_mode="observable_state_exposure")
        self.f_mean.fit(descriptors, centered_f, domains, coefficient_targets=labels[:,0])
        self.g_mean.fit(descriptors, labels[:,1], domains, coefficient_targets=labels[:,1])
        self.g_variance = SourceAlignedVarianceRiskCoordinate(local_dim=3, shared_dim=3)
        self.f_variance = SourceAlignedVarianceRiskCoordinate(local_dim=3, shared_dim=3)
        self.g_variance.fit(descriptors, np.exp(labels[:,2]), domains)
        self.f_variance.fit(descriptors, np.exp(labels[:,3]), domains)
        self.source_count = len(labels)
        return self

    def transform(self, descriptors):
        return np.vstack([np.r_[self.f_mean.features_profile(d), self.g_mean.features_profile(d),
            self.g_variance.latent_from_descriptor(d), self.f_variance.latent_from_descriptor(d)]
            for d in descriptors])

    def diagnostics(self):
        return {"coordinate_dimension":10, "source_record_count":self.source_count,
            "axes":["f_mean_0","f_mean_1","g_mean_0","g_mean_1",
                    "eta_g_0","eta_g_1","eta_g_2","eta_f_0","eta_f_1","eta_f_2"],
            "input":"declared observable control groups, occupancy and common-grid curve dynamics",
            "mean_maps":[self.f_mean.diagnostics(), self.g_mean.diagnostics()],
            "variance_maps":[self.g_variance.diagnostics(), self.f_variance.diagnostics()],
            "target_labels_used":False, "target_risk_parameters_used":False}

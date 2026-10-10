"""The public billing functions embedded in the orthogonal low-order class."""

import numpy as np

from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.source_function_prior import SourceFunctionPrior


class OrthogonalInventoryBasis(OrthogonalLowOrderBasis):
    billing_axes = {3: "order_quantity", 5: "holding_quantity",
                    4: "lost_sales_quantity", 6: "order_frequency"}

    def _library(self, psi):
        psi = np.atleast_2d(np.asarray(psi, dtype=float))
        values, names = super()._library(psi)
        return np.column_stack([values, psi[:, list(self.billing_axes)]]), names + list(self.billing_axes.values())

    def _axis_spec(self, reference, j, lookup):
        indices, derivative, order = super()._axis_spec(reference, j, lookup)
        z = np.tanh(.5 * reference[:, j])
        # Derivative energy in standardized physical coordinates. In particular
        # a raw linear billing quantity has unit standardized roughness.
        derivative = derivative / z.std() * (.5 * (1 - z ** 2) * reference[:, j].std())[:, None] if z.std() else derivative
        if j in self.billing_axes:
            indices.append(lookup[self.billing_axes[j]])
            derivative = np.column_stack([derivative, np.full(len(z), reference[:, j].std())])
            order = [5] + order  # retain exact charges before numerical rank reduction
        return indices, derivative, order

    def _pair_roughness(self, reference):
        z = np.tanh(.5 * reference)
        gradients = [float(np.mean((.5 * (1 - z[:, j] ** 2) * reference[:, j].std() / self.axes[j][3]) ** 2))
                     if self.axes[j][3] else 0. for j in range(self.dim)]
        return [gradients[i] + gradients[j] for i, j in self.pairs]

    def diagnostics(self):
        return {**super().diagnostics(), "derivative_coordinate": "standardized_public_physical_coordinate",
                "public_billing_functions": list(self.billing_axes.values())}


def embed_inventory_prior(prior, basis, reference):
    """Change function coordinates without changing the source Gaussian law.

    Unary functions are projected within their own public marginal. Products
    expand into constant, unary and centered pair terms, so the transport never
    relies on correlations in the observed trajectory geometry.
    """
    raw, names = basis._library(reference)
    offsets = np.cumsum([0] + [axis[2].shape[1] for axis in basis.axes])
    unary, constants = {}, {}
    for j, (indices, center, transform, scale) in enumerate(basis.axes):
        features = (raw[:, indices] - center) @ transform
        for k, index in enumerate(indices):
            coefficient = np.zeros(basis.feature_dim)
            coefficient[offsets[j]:offsets[j + 1]] = features.T @ (raw[:, index] - center[k]) / len(raw)
            unary[names[index]], constants[names[index]] = coefficient, center[k]
    conversion = np.zeros((basis.feature_dim + 1, len(prior.mean)))
    conversion[0, 0] = 1.
    for column, name in enumerate(prior.basis.names, 1):
        if name.startswith("psi") and "*" in name:
            left, right = name.split("*")
            i, j = int(left[3:]), int(right[3:])
            coefficient = constants[right] * unary[left] + constants[left] * unary[right]
            coefficient[offsets[-1] + basis.pairs.index((i, j))] += basis.axes[i][3] * basis.axes[j][3]
            constant = constants[left] * constants[right]
        else:
            coefficient, constant = unary[name], constants[name]
        conversion[0, column] = (constant - prior.basis.mean[column - 1]) / prior.basis.scale[column - 1]
        conversion[1:, column] = coefficient / prior.basis.scale[column - 1]
    info = {**prior.diagnostics, **basis.diagnostics(), "mode": "source_orthogonal_inventory_support",
            "selected_function_names": prior.basis.names,
            "coefficient_transport": "product_marginal_projection; full source second moments retained",
            "selected_support_dim": prior.basis.feature_dim}
    return SourceFunctionPrior(basis, conversion @ prior.mean,
        conversion @ prior.covariance @ conversion.T, prior.discrepancy_variance, info)

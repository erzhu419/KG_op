"""Public Inventory function class, including the four billing quantities."""

import numpy as np
from dataclasses import replace
from scipy.stats import norm

from representation.conditional_function_prior import conditional_selection
from representation.source_function_prior import PublicCoordinateSpectralBasis, SourceFunctionPrior


class InventoryFunctionBasis:
    billing_indices = [68, 69, 70, 71]

    def __init__(self, reference_psi, indices=None):
        self.dictionary = PublicCoordinateSpectralBasis(max_library_size=64)
        raw, names = self._library(reference_psi)
        self.indices = np.arange(raw.shape[1]) if indices is None else np.asarray(indices, dtype=int)
        raw = raw[:, self.indices]
        self.mean = raw.mean(axis=0)
        self.scale = np.maximum(raw.std(axis=0), 1e-8)
        self.feature_dim = len(self.indices)
        self.names = [names[i] for i in self.indices]

    def _library(self, psi):
        psi = np.atleast_2d(np.asarray(psi, dtype=float))
        spectral, names = self.dictionary._library(psi)
        # These quantities are calculated from the public trajectory. Prices,
        # outcome means and active response functions are not supplied here.
        quantities = psi[:, [3, 5, 4, 6]]
        return np.column_stack([spectral, quantities]), names + [
            "order_quantity", "holding_quantity", "lost_sales_quantity", "order_frequency"]

    def transform(self, psi):
        raw, _ = self._library(psi)
        return (raw[:, self.indices] - self.mean) / self.scale

    def diagnostics(self):
        return {"selected_names": self.names, "active_dim": self.feature_dim,
                "orthogonalization": "none", "input_coordinates": "public_inventory_trajectory"}


def fit_inventory_group_prior(batches, noises, reference, adaptation_positions, test_positions):
    """Source-only joint billing-block selection and residual calibration."""
    dictionary = InventoryFunctionBasis(reference)
    greedy, history = conditional_selection(batches, noises, dictionary, 4)

    def support_losses(support):
        losses = []
        for task, (batch, noise) in enumerate(zip(batches, noises)):
            Z = np.column_stack([np.ones(len(batch.psi)), dictionary.transform(batch.psi)[:, support]])
            y, noise = np.asarray(batch.signals).reshape(-1), np.asarray(noise)
            halves = np.array_split(np.random.default_rng(92000 + task).permutation(len(y)), 2)
            scale = max(float(np.var(y) - np.mean(noise)), float(np.mean(noise)), 1e-8)
            folded = []
            for train, test in (halves, halves[::-1]):
                weights = 1. / np.maximum(noise[train], 1e-8)
                penalty = np.diag([1e-8] + [1e-3] * len(support))
                beta = np.linalg.solve(Z[train].T @ (weights[:, None] * Z[train]) + penalty,
                                       Z[train].T @ (weights * y[train]))
                folded.append(float(np.mean((y[test] - Z[test] @ beta) ** 2) / scale))
            losses.append(np.mean(folded))
        return np.asarray(losses)

    gain = support_losses(greedy) - support_losses(dictionary.billing_indices)
    se = float(np.std(gain, ddof=1) / np.sqrt(len(gain)))
    use_block = bool(gain.mean() > se)
    support = dictionary.billing_indices if use_block else greedy
    basis = InventoryFunctionBasis(reference, support)
    # Residual calibration is nested inside the outer held-source evaluation.
    # Only this training archive supplies support, moments and both scores.
    scores = {0.: [], .02: []}
    for left_out, held in enumerate(batches):
        keep = [i for i in range(len(batches)) if i != left_out]
        prior = SourceFunctionPrior.from_basis([batches[i] for i in keep], [noises[i] for i in keep], basis, "group_source")
        for discrepancy in scores:
            model = replace(prior, discrepancy_variance=discrepancy).target_model(held.psi)
            for index in adaptation_positions:
                model.update([index], held.signals[index], noises[left_out][index])
            candidates = np.asarray(test_positions)[:, None]
            mean = model.posterior_mean_many(candidates)
            variance = model.posterior_var_many(candidates) + noises[left_out][test_positions]
            scores[discrepancy].append(float(np.mean(norm.logpdf(held.signals[test_positions], mean, np.sqrt(variance)))))
    chosen = max(scores, key=lambda value: float(np.mean(scores[value])))
    prior = SourceFunctionPrior.from_basis(batches, noises, basis, "group_source")
    info = {**prior.diagnostics, "selected_indices": support, "greedy_indices": greedy,
            "greedy_cv": history, "billing_block_selected": use_block,
            "billing_block_cv_gain": float(gain.mean()), "billing_block_task_se": se,
            "discrepancy_source_scores": {str(k): v for k, v in scores.items()},
            "chosen_discrepancy_variance": chosen}
    return replace(prior, discrepancy_variance=chosen, diagnostics=info)


def fit_inventory_posterior_group_prior(batches, noises, reference, adaptation_positions, test_positions):
    """Select support and residual using the posterior that will be deployed.

    Compare a jointly declared billing block with greedy support. Every score
    conditions a new task on its paid adaptation labels, using coefficient
    moments learned on the remaining task laws. No target task is supplied.
    """
    dictionary = InventoryFunctionBasis(reference)
    greedy, history = conditional_selection(batches, noises, dictionary, 4)
    supports = {"greedy": greedy, "billing_block": dictionary.billing_indices}
    scores = {(name, residual): [] for name in supports for residual in (0., .02)}
    for name, support in supports.items():
        basis = InventoryFunctionBasis(reference, support)
        for left_out, held in enumerate(batches):
            keep = [i for i in range(len(batches)) if i != left_out]
            prior = SourceFunctionPrior.from_basis([batches[i] for i in keep], [noises[i] for i in keep], basis, "posterior_group_source")
            for residual in (0., .02):
                model = replace(prior, discrepancy_variance=residual).target_model(held.psi)
                for index in adaptation_positions:
                    model.update([index], held.signals[index], noises[left_out][index])
                candidates = np.asarray(test_positions)[:, None]
                mean = model.posterior_mean_many(candidates)
                variance = model.posterior_var_many(candidates) + noises[left_out][test_positions]
                scores[name, residual].append(float(np.mean(norm.logpdf(held.signals[test_positions], mean, np.sqrt(variance)))))
    chosen, residual = max(scores, key=lambda key: float(np.mean(scores[key])))
    support = supports[chosen]
    prior = SourceFunctionPrior.from_basis(batches, noises, InventoryFunctionBasis(reference, support), "posterior_group_source")
    info = {**prior.diagnostics, "selected_indices": support, "greedy_indices": greedy, "greedy_cv": history,
            "billing_block_selected": chosen == "billing_block", "chosen_discrepancy_variance": residual,
            "source_posterior_scores": {name + ":" + str(r): v for (name, r), v in scores.items()}}
    return replace(prior, discrepancy_variance=residual, diagnostics=info)

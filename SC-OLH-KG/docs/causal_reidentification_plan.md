# Causal Re-identification Plan

## Why this reset is necessary

The completed matrix shows that the frozen source-informed `n0` proposal is
the dominant observed effect. Exact KG and risk-aware Thompson sampling do not
beat Sobol continuation, cumulative HVD does not beat pooled variance, and the
theory certificate is empty in every completed structural run.

The old structural-prior table is not causal. It reused one full-prior
proposal and changed only spectral posterior flags; the ordered implementation
still retained low-frequency weighting, orthogonal coordinates, sparse model
selection, group ridge, and diagonal-additive blocks.

## Corrected estimands

All structural experiments use Sobol continuation, pooled variance, adaptive
source-discrepancy updating, identical source-call budgets, and one seed per
scheduler task.

1. `proposal_only`: profile-specific source proposal plus a no-four-prior
   target posterior.
2. `posterior_only`: common Sobol `n0` plus a profile-specific posterior.
3. `joint`: profile-specific source proposal and the matching posterior.

Every profile is retrained from the same immutable source archive and emits a
new proposal fingerprint. Both spectral and ordered realizations of each
assumption are switched together. If fingerprints remain identical, the
proposal gain belongs to source-consensus ranking rather than that prior.

## Dimension transfer

The `risk_coordinate_atlas` proposal represents source policies with
dimensionless ordered moments and cosine coordinates, selects source-ranked
maximin coverage, and synthesizes policies directly at the target dimension.
It is compared with legacy rank-spanning interpolation.

Domain and dimension are held out simultaneously: for example, source domains
are observed only at `d_source=50`, while a held-out target domain is optimized
at `d_target=200` or `1000`. Source and target dimensions, archive
fingerprints, and proposal fingerprints are recorded separately. No `d=10000`
experiment is admissible until the `d=1000,N=20` gate succeeds.

## HVD identification

HVD is tested outside objective search so candidate quality cannot hide its
statistical value. The FactorShock benchmark sweeps:

- shared-shock scale: `0, 0.25, 1, 4`;
- replications per policy: `2, 4, 8, 16`;
- pooled, class, orthogonal-pointwise, factor-pointwise, and cumulative-factor
  variance models.

Fit data are ordinary replicated simulator observations. Oracle variance is
used only after fitting to report log-variance RMSE, risk ordering,
upper-variance coverage, false feasibility, missed feasibility, and whether
the certificate is non-vacuous.

The first 400-cell gate identified two implementation defects rather than a
failed risk coordinate: sample-variance replication degrees of freedom were
discarded by the tail guard, and a one-shot PSD/nonnegative projection inflated
the unconstrained ridge scale. The promoted estimator uses replication-aware
projected IRLS. In the corrected five-seed gate, cumulative factor-HVD ranks
variance best in all 16 shock/replication cells and has the lowest log-variance
RMSE in 14, with zero false-certified fraction. Certification remains
deliberately conservative at the highest shock levels, so recall and precision
are reported separately rather than collapsed into population error rates.

## Gates

1. Unit tests and local tiny smoke.
2. Corrected causal matrix at `d=50,N=20,n0=10`, 5 seeds.
3. HVD identification, 5 seeds per cell.
4. Dimension holdout at `d_target=200,N=20`, source `d=50`.
5. Only promoted proposal variants proceed to `d_target=1000,N=20`.
6. Increase seeds only after a positive paired effect; do not use `d=10000`
   to obscure a failed `d=1000` gate.

Scheduler entrypoints:

- `scripts/submit_scolhkg_causal_prior_matrix_scheduler.py`
- `scripts/submit_scolhkg_hvd_identifiability_scheduler.py`

Runtime checkpoints stay in remote checkpoint directories. Result collection
accepts only `result.json`; pickle files, model weights, NumPy arrays, and
checkpoint trees are never synchronized for aggregation.

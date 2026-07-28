# V24 Bounded Nullspace Semiparametric Expert

## Failure Being Repaired

V23 enforces `Phi^T K_perp = 0` only on a frozen unlabeled policy pool by
subtracting an ordered polynomial from each RBF feature.  Two FactorShock
recommendations leave that support, reach leverage above `2e6`, and one is
falsely certified.  Inventory keeps every local residual PIP at its lower
bound, so it does not benefit from the extension.

## Single Structural Change

Let `K` be a bounded RBF dictionary on the frozen pool and let `Phi` contain an
intercept and the ordered cumulative features.  V24 computes an orthonormal
basis `P` for the coefficient nullspace of `Phi^T K` and uses

`k_perp(x) = k(x)^T P`.

Then `Phi^T K P = 0` on the frozen pool.  More importantly, no polynomial is
subtracted at a new point.  Since every RBF coordinate lies in `[0,1]` and `P`
has orthonormal columns, the local residual norm is globally bounded by the
parent dictionary norm for every candidate.

## Controls Held Fixed

- six finite task experts;
- one combined ordered semiparametric expert;
- six optional local residual directions;
- total posterior effective dimension at most `0.35 N`;
- source PIP `0.5` for each residual direction;
- factor HVD, theory certification, IID exact-MC2 KG, candidate generation,
  initial design, `d=50`, `N=20`, and seven paired seeds.

## Gate And Interpretation

- FactorShock must be 7/7 true feasible with no false certificate;
- Inventory must be at least 4/7 true feasible with at most one false
  feasible recommendation;
- every projection error must remain below `1e-10` and every effective
  dimension must remain below its cap;
- Queue remains unopened until both conditions pass.

If FactorShock is restored but Inventory remains below 4/7, the local residual
construction is accepted as a safety repair but V24 is not promoted.  The next
independent intervention is rotation/permutation-robust transfer of ordered
coefficient groups or subspaces.  If FactorShock still fails, the local
residual feature model is rejected rather than protected by another guard or
threshold.

## Result

The gate failed: FactorShock reached 4/7 and Inventory 2/7, with no false
certificate.  The bounded projection repaired V23's extrapolation defect but
did not restore local-model utility.  V24 is rejected.  The next challenger
uses a latent mutually exclusive structure model and group/subspace-level
ordered transfer; the direct-sum residual is not retained.

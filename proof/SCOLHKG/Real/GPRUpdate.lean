import Mathlib

namespace SCOLHKG.Real

/-!
Code-level bridge for `SC-OLH-KG/core/gpr.py`.

`ParametricGPR.update` is a rank-one Kalman update.  `core/kg.py` uses the same
update algebra to compute the response slope

```text
sigma_tilde(u; x) = cov(u,x) / sqrt(sigma2(x) + cov(x,x)).
```

The theorem below proves that this is exactly the coefficient of a standard
normal predictive shock in the updated posterior mean.
-/

universe u

structure RankOneGPRState (Design : Type u) where
  mean : Design → ℝ
  covariance : Design → Design → ℝ
  observationVariance : Design → ℝ

def predictiveVariance
    {Design : Type u}
    (s : RankOneGPRState Design)
    (x : Design) : ℝ :=
  s.observationVariance x + s.covariance x x

noncomputable def kalmanGain
    {Design : Type u}
    (s : RankOneGPRState Design)
    (sample target : Design) : ℝ :=
  s.covariance target sample / predictiveVariance s sample

noncomputable def updatedPosteriorMean
    {Design : Type u}
    (s : RankOneGPRState Design)
    (sample : Design)
    (y : ℝ)
    (target : Design) : ℝ :=
  s.mean target + kalmanGain s sample target * (y - s.mean sample)

noncomputable def kgUpdateWeight
    {Design : Type u}
    (s : RankOneGPRState Design)
    (sample target : Design)
    (denom : ℝ) : ℝ :=
  s.covariance target sample / denom

theorem rank_one_update_standard_shock_slope
    {Design : Type u}
    (s : RankOneGPRState Design)
    (sample target : Design)
    {denom z y : ℝ}
    (hdenom : denom ≠ 0)
    (hdenomSq : predictiveVariance s sample = denom ^ 2)
    (hobs : y = s.mean sample + denom * z) :
    updatedPosteriorMean s sample y target =
      s.mean target + kgUpdateWeight s sample target denom * z := by
  unfold updatedPosteriorMean kalmanGain kgUpdateWeight
  rw [hobs, hdenomSq]
  have hsq : denom ^ 2 ≠ 0 := pow_ne_zero 2 hdenom
  field_simp [hdenom, hsq]
  ring

def covarianceResponse
    {Design : Type u}
    (s : RankOneGPRState Design)
    (sample target : Design) : ℝ :=
  s.covariance target sample

theorem kg_sigma_tilde_matches_code_formula
    {Design : Type u}
    (s : RankOneGPRState Design)
    (sample target : Design)
    {denom : ℝ}
    (_hdenom : denom ≠ 0) :
    kgUpdateWeight s sample target denom =
      covarianceResponse s sample target / denom := by
  rfl

/-!
Repeated evaluation is an admissible exact-KG action. At an already observed
design with epistemic variance `q` and observation variance `r`, the scalar
rank-one update removes `q^2 / (q + r)` from the epistemic variance. This is
the quantity used to rank the optional replication candidates in
`algorithms/single_olhkg.py`; it only proposes candidates, while exact KG still
decides whether evaluating one is preferable to a new design.
-/

noncomputable def replicationVarianceReduction (q r : ℝ) : ℝ :=
  q ^ 2 / (q + r)

theorem replication_variance_update_identity
    {q r : ℝ}
    (hq : 0 ≤ q)
    (hr : 0 < r) :
    q - replicationVarianceReduction q r = q * r / (q + r) := by
  have hsum : q + r ≠ 0 := ne_of_gt (add_pos_of_nonneg_of_pos hq hr)
  unfold replicationVarianceReduction
  field_simp [hsum]
  ring

theorem replication_variance_reduction_nonnegative
    {q r : ℝ}
    (hq : 0 ≤ q)
    (hr : 0 < r) :
    0 ≤ replicationVarianceReduction q r := by
  unfold replicationVarianceReduction
  exact div_nonneg (sq_nonneg q) (le_of_lt (add_pos_of_nonneg_of_pos hq hr))

theorem replication_variance_reduction_le_epistemic
    {q r : ℝ}
    (hq : 0 ≤ q)
    (hr : 0 < r) :
    replicationVarianceReduction q r ≤ q := by
  have hsum : 0 < q + r := add_pos_of_nonneg_of_pos hq hr
  unfold replicationVarianceReduction
  rw [div_le_iff₀ hsum]
  nlinarith [mul_nonneg hq (le_of_lt hr)]

theorem replication_updated_variance_nonnegative
    {q r : ℝ}
    (hq : 0 ≤ q)
    (hr : 0 < r) :
    0 ≤ q - replicationVarianceReduction q r := by
  exact sub_nonneg.mpr (replication_variance_reduction_le_epistemic hq hr)

end SCOLHKG.Real

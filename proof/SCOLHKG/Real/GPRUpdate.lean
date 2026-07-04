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

end SCOLHKG.Real

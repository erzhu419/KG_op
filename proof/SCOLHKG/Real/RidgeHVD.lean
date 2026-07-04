import Mathlib

namespace SCOLHKG.Real

/-!
Concrete ridge-HVD residual-square oracle inequality.

The HVD implementation fits a log/residual-square variance model with a ridge
penalty.  This file proves the deterministic oracle step used by the statistical
argument: a ridge empirical minimizer plus a uniform residual-square
concentration event gives an oracle inequality for the true residual-square
risk.
-/

universe u

def RidgeObjective
    {Param : Type u}
    (empiricalRisk penalty : Param → ℝ)
    (lambda : ℝ)
    (theta : Param) : ℝ :=
  empiricalRisk theta + lambda * penalty theta

def RidgeMinimizer
    {Param : Type u}
    (empiricalRisk penalty : Param → ℝ)
    (lambda : ℝ)
    (thetaHat : Param) : Prop :=
  ∀ theta,
    RidgeObjective empiricalRisk penalty lambda thetaHat
      ≤ RidgeObjective empiricalRisk penalty lambda theta

def UniformResidualSquareConcentration
    {Param : Type u}
    (trueRisk empiricalRisk : Param → ℝ)
    (radius : ℝ) : Prop :=
  ∀ theta, |trueRisk theta - empiricalRisk theta| ≤ radius

theorem ridge_basic_inequality
    {Param : Type u}
    {empiricalRisk penalty : Param → ℝ}
    {lambda : ℝ}
    {thetaHat thetaStar : Param}
    (hMin : RidgeMinimizer empiricalRisk penalty lambda thetaHat)
    (hlambda : 0 ≤ lambda)
    (hPenaltyHat : 0 ≤ penalty thetaHat) :
    empiricalRisk thetaHat
      ≤ empiricalRisk thetaStar + lambda * penalty thetaStar := by
  unfold RidgeMinimizer RidgeObjective at hMin
  have hObj := hMin thetaStar
  have hNonneg : 0 ≤ lambda * penalty thetaHat := by
    exact mul_nonneg hlambda hPenaltyHat
  linarith

theorem ridge_hvd_residual_square_oracle
    {Param : Type u}
    {trueRisk empiricalRisk penalty : Param → ℝ}
    {lambda radius : ℝ}
    {thetaHat thetaStar : Param}
    (hMin : RidgeMinimizer empiricalRisk penalty lambda thetaHat)
    (hConc : UniformResidualSquareConcentration trueRisk empiricalRisk radius)
    (hlambda : 0 ≤ lambda)
    (hPenaltyHat : 0 ≤ penalty thetaHat) :
    trueRisk thetaHat
      ≤ trueRisk thetaStar + 2 * radius + lambda * penalty thetaStar := by
  have hHatAbs := abs_le.mp (hConc thetaHat)
  have hStarAbs := abs_le.mp (hConc thetaStar)
  have hHat :
      trueRisk thetaHat ≤ empiricalRisk thetaHat + radius := by
    linarith [hHatAbs.2]
  have hStar :
      empiricalRisk thetaStar ≤ trueRisk thetaStar + radius := by
    linarith [hStarAbs.1]
  have hEmp :
      empiricalRisk thetaHat
        ≤ empiricalRisk thetaStar + lambda * penalty thetaStar := by
    exact ridge_basic_inequality
      (empiricalRisk := empiricalRisk)
      (penalty := penalty)
      (lambda := lambda)
      (thetaHat := thetaHat)
      (thetaStar := thetaStar)
      hMin hlambda hPenaltyHat
  linarith

structure RidgeHVDOracleTerms where
  fittedTrueRisk : ℝ
  oracleTrueRisk : ℝ
  residualConcentrationRadius : ℝ
  ridgePenaltyOracle : ℝ
  approximationError : ℝ
  slack : ℝ

def RidgeHVDOracleInequality (t : RidgeHVDOracleTerms) : Prop :=
  t.fittedTrueRisk ≤
    t.oracleTrueRisk
      + 2 * t.residualConcentrationRadius
      + t.ridgePenaltyOracle
      + t.approximationError
      + t.slack

theorem ridge_hvd_oracle_terms_from_bound
    (t : RidgeHVDOracleTerms)
    (hBound :
      t.fittedTrueRisk ≤
        t.oracleTrueRisk
          + 2 * t.residualConcentrationRadius
          + t.ridgePenaltyOracle)
    (hApprox : 0 ≤ t.approximationError)
    (hSlack : 0 ≤ t.slack) :
    RidgeHVDOracleInequality t := by
  unfold RidgeHVDOracleInequality
  linarith

end SCOLHKG.Real

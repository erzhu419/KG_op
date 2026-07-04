import Mathlib

namespace SCOLHKG.Real

/-!
Real-valued HVD oracle inequality skeleton.

The concentration proof for residual-square regression will eventually
instantiate `estimationError ≤ complexity + concentrationRadius`.  This file
proves the deterministic oracle inequality once that event is available.
-/

structure HVDOracleTerms where
  fittedRisk : ℝ
  oracleRisk : ℝ
  approximationError : ℝ
  estimationError : ℝ
  slack : ℝ

def HVDOracleInequality (t : HVDOracleTerms) : Prop :=
  t.fittedRisk ≤
    t.oracleRisk + t.approximationError + t.estimationError + t.slack

theorem hvd_oracle_from_basic_inequality
    (t : HVDOracleTerms)
    (hBasic :
      t.fittedRisk ≤
        t.oracleRisk + t.approximationError + t.estimationError)
    (hSlack : 0 ≤ t.slack) :
    HVDOracleInequality t := by
  unfold HVDOracleInequality
  linarith

structure ResidualSquareConcentration where
  estimationError : ℝ
  complexity : ℝ
  concentrationRadius : ℝ

def ResidualSquareConcentration.Valid (c : ResidualSquareConcentration) : Prop :=
  c.estimationError ≤ c.complexity + c.concentrationRadius

theorem hvd_oracle_from_residual_concentration
    (t : HVDOracleTerms)
    (c : ResidualSquareConcentration)
    (hSame : t.estimationError = c.estimationError)
    (hConc : c.Valid)
    (hBasic :
      t.fittedRisk ≤
        t.oracleRisk + t.approximationError + t.estimationError)
    (hSlack : 0 ≤ t.slack) :
    t.fittedRisk ≤
      t.oracleRisk + t.approximationError
        + (c.complexity + c.concentrationRadius) + t.slack := by
  unfold ResidualSquareConcentration.Valid at hConc
  rw [hSame] at hBasic
  linarith

structure ConservativeVariance where
  trueVariance : ℝ
  predictedVariance : ℝ
  modelUncertainty : ℝ

def ConservativeVariance.Valid (c : ConservativeVariance) : Prop :=
  c.trueVariance ≤ c.predictedVariance + c.modelUncertainty

def certificationVariance (c : ConservativeVariance) : ℝ :=
  c.predictedVariance + c.modelUncertainty

theorem conservative_variance_upper
    (c : ConservativeVariance)
    (h : c.Valid) :
    c.trueVariance ≤ certificationVariance c := by
  exact h

end SCOLHKG.Real


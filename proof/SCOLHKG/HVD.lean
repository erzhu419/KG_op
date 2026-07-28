namespace SCOLHKG

/-!
Formal skeleton for HVD estimation guarantees.

This file does not yet prove sub-exponential concentration for squared
residuals.  Instead, it formalizes the deterministic implication used after
that concentration step: a basic inequality plus approximation and estimation
terms yields the oracle-inequality shape used in the manuscript.
-/

structure HVDOracleTerms where
  fittedRisk : Nat
  oracleRisk : Nat
  approximationError : Nat
  estimationError : Nat
  slack : Nat
deriving Repr, BEq

def HVDOracleInequality (t : HVDOracleTerms) : Prop :=
  t.fittedRisk ≤
    t.oracleRisk + t.approximationError + t.estimationError + t.slack

theorem hvdOracleInequality_from_basic
    (t : HVDOracleTerms)
    (hBasic :
      t.fittedRisk ≤
        t.oracleRisk + t.approximationError + t.estimationError) :
    HVDOracleInequality t := by
  unfold HVDOracleInequality
  exact Nat.le_trans hBasic (Nat.le_add_right _ _)

theorem hvdOracleInequality_from_concentration
    (t : HVDOracleTerms)
    (complexity concentration : Nat)
    (hEstimation : t.estimationError ≤ complexity + concentration)
    (hBasic :
      t.fittedRisk ≤
        t.oracleRisk + t.approximationError + t.estimationError) :
    t.fittedRisk ≤
      t.oracleRisk + t.approximationError + (complexity + concentration) + t.slack := by
  have h1 :
      t.fittedRisk ≤
        t.oracleRisk + t.approximationError + (complexity + concentration) := by
    exact Nat.le_trans hBasic (Nat.add_le_add_left hEstimation _)
  exact Nat.le_trans h1 (Nat.le_add_right _ _)

structure HVDCertification where
  trueVariance : Nat
  predictedVariance : Nat
  modelUncertainty : Nat
deriving Repr, BEq

def ConservativeVariance (c : HVDCertification) : Prop :=
  c.trueVariance ≤ c.predictedVariance + c.modelUncertainty

def certificationVariance (c : HVDCertification) : Nat :=
  c.predictedVariance + c.modelUncertainty

theorem conservativeVariance_upper
    (c : HVDCertification)
    (h : ConservativeVariance c) :
    c.trueVariance ≤ certificationVariance c := by
  exact h

theorem conservativeVariance_monotone_uncertainty
    (c : HVDCertification)
    {extra : Nat}
    (h : ConservativeVariance c) :
    c.trueVariance ≤ c.predictedVariance + c.modelUncertainty + extra := by
  exact Nat.le_trans h (Nat.le_add_right _ _)

end SCOLHKG


import Mathlib
import SCOLHKG.Real.SafeRegret

namespace SCOLHKG.Real

/-!
Information-gain regret accounting.

This is the deterministic endpoint of a GP-style regret proof.  The kernel-
specific work is to upper-bound the posterior error by an information-gain
radius.  Once that event is available, the finite-budget safe-regret conclusion
is arithmetic.
-/

noncomputable def informationGainRadius (beta gammaT : ℝ) : ℝ :=
  Real.sqrt (beta * gammaT)

structure InformationGainRegretTerms where
  actualRegret : ℝ
  beta : ℝ
  gammaT : ℝ
  candidateSetError : ℝ
  certificationError : ℝ
  kgApproximationError : ℝ

def InformationGainRegretBound (t : InformationGainRegretTerms) : Prop :=
  t.actualRegret ≤
    informationGainRadius t.beta t.gammaT
      + t.candidateSetError
      + t.certificationError
      + t.kgApproximationError

theorem information_gain_regret_le_budget
    (t : InformationGainRegretTerms)
    {eps : ℝ}
    (hRegret : InformationGainRegretBound t)
    (hBudget :
      informationGainRadius t.beta t.gammaT
        + t.candidateSetError
        + t.certificationError
        + t.kgApproximationError ≤ eps) :
    t.actualRegret ≤ eps := by
  exact hRegret.trans hBudget

theorem safe_simple_regret_from_information_gain
    {Design : Type*}
    (p : ChanceOptimization Design)
    (xRec xStar : Design)
    (t : InformationGainRegretTerms)
    {eps : ℝ}
    (hObj :
      p.objective xRec - p.objective xStar = t.actualRegret)
    (hRegret : InformationGainRegretBound t)
    (hBudget :
      informationGainRadius t.beta t.gammaT
        + t.candidateSetError
        + t.certificationError
        + t.kgApproximationError ≤ eps) :
    SafeSimpleRegretBound p xRec xStar eps := by
  unfold SafeSimpleRegretBound
  rw [hObj]
  exact information_gain_regret_le_budget t hRegret hBudget

end SCOLHKG.Real

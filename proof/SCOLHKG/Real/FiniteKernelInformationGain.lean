import Mathlib
import SCOLHKG.Real.InformationGainRegret

namespace SCOLHKG.Real

open scoped BigOperators

/-!
Finite-kernel information-gain accounting.

This is the code-facing version of the information-gain term for finite
candidate/posterior pools.  Each sampled design contributes a scalar

```text
0.5 log(1 + posterior_variance / observation_noise).
```

If every scalar contribution is bounded by the same cap, the total information
gain is bounded by `T * cap`.  Kernel-specific determinant arguments can later
replace the per-step cap without changing downstream regret accounting.
-/

noncomputable def scalarInformationGain (variance noise : ℝ) : ℝ :=
  (1 / 2 : ℝ) * Real.log (1 + variance / noise)

noncomputable def finiteInformationGain
    {Time : Type*}
    (steps : Finset Time)
    (variance noise : Time → ℝ) : ℝ :=
  ∑ t ∈ steps, scalarInformationGain (variance t) (noise t)

theorem scalarInformationGain_le_of_ratio
    {variance noise varianceCap noiseFloor : ℝ}
    (hposLeft : 0 < 1 + variance / noise)
    (hratio : variance / noise ≤ varianceCap / noiseFloor) :
    scalarInformationGain variance noise
      ≤ scalarInformationGain varianceCap noiseFloor := by
  unfold scalarInformationGain
  have harg : 1 + variance / noise ≤ 1 + varianceCap / noiseFloor := by
    linarith
  exact mul_le_mul_of_nonneg_left
    (Real.log_le_log hposLeft harg)
    (by norm_num)

theorem finiteInformationGain_le_uniform_cap
    {Time : Type*}
    (steps : Finset Time)
    (variance noise : Time → ℝ)
    {cap : ℝ}
    (hEach :
      ∀ t ∈ steps,
        scalarInformationGain (variance t) (noise t) ≤ cap) :
    finiteInformationGain steps variance noise ≤ steps.card * cap := by
  unfold finiteInformationGain
  calc
    ∑ t ∈ steps, scalarInformationGain (variance t) (noise t)
      ≤ ∑ _t ∈ steps, cap := by
        exact Finset.sum_le_sum (by
          intro t ht
          exact hEach t ht)
    _ = steps.card * cap := by
        simp

theorem finiteInformationGain_regret_budget
    {Time : Type*}
    (steps : Finset Time)
    (variance noise : Time → ℝ)
    (t : InformationGainRegretTerms)
    {eps cap : ℝ}
    (hGamma :
      finiteInformationGain steps variance noise ≤ cap)
    (hRadius :
      finiteInformationGain steps variance noise ≤ cap →
        informationGainRadius t.beta t.gammaT
        ≤ informationGainRadius t.beta cap)
    (hRegret : InformationGainRegretBound t)
    (hBudget :
      informationGainRadius t.beta cap
        + t.candidateSetError
        + t.certificationError
        + t.kgApproximationError ≤ eps) :
    t.actualRegret ≤ eps := by
  unfold InformationGainRegretBound at hRegret
  have hRadius' := hRadius hGamma
  linarith

end SCOLHKG.Real

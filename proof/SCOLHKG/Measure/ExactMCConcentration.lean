import SCOLHKG.Measure.SubGaussianConfidence
import SCOLHKG.Real.ExactKGImplementation

namespace SCOLHKG.Measure

open MeasureTheory ProbabilityTheory
open scoped MeasureTheory ProbabilityTheory BigOperators

/-!
Concentration bridge for the exact-MC posterior-update KG estimator.

The implementation evaluates a finite candidate pool.  If the Monte-Carlo
estimator error for each candidate is centered sub-Gaussian, then a finite
union bound gives a uniform-error event.  On that event, the deterministic
`ExactMCEstimator` bridge in `Real/ExactKGImplementation.lean` yields the
same `2 eta` maximizer gap.
-/

variable {Ω Design : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω}

def ExactMCUniformErrorEvent
    (candidates : Finset Design)
    (exact : Design → ℝ)
    (estimate : Ω → Design → ℝ)
    (eta : ℝ) : Set Ω :=
  {ω | ∀ x ∈ candidates, |estimate ω x - exact x| ≤ eta}

def ExactMCBadEvent
    (candidates : Finset Design)
    (exact : Design → ℝ)
    (estimate : Ω → Design → ℝ)
    (eta : ℝ) : Set Ω :=
  ⋃ x ∈ candidates,
    CenteredSubGaussianBadEvent
      (fun ω ↦ estimate ω x - exact x)
      eta

theorem exactMC_uniform_error_of_not_bad
    (candidates : Finset Design)
    (exact : Design → ℝ)
    (estimate : Ω → Design → ℝ)
    {eta : ℝ}
    {ω : Ω}
    (hnot : ω ∉ ExactMCBadEvent candidates exact estimate eta) :
    ω ∈ ExactMCUniformErrorEvent candidates exact estimate eta := by
  intro x hx
  by_contra hgt
  have heta : eta ≤ |estimate ω x - exact x| := le_of_lt (not_le.mp hgt)
  have hmem :
      ω ∈ CenteredSubGaussianBadEvent
        (fun ω ↦ estimate ω x - exact x) eta := by
    exact heta
  exact hnot (by
    unfold ExactMCBadEvent
    exact Set.mem_iUnion₂.mpr ⟨x, hx, hmem⟩)

theorem exactMC_uniform_error_implies_estimator
    (candidates : Finset Design)
    (exact : Design → ℝ)
    (estimate : Ω → Design → ℝ)
    {eta : ℝ}
    {ω : Ω}
    (hevent : ω ∈ ExactMCUniformErrorEvent candidates exact estimate eta)
    (heta : 0 ≤ eta)
    (houtside : ∀ x, x ∉ candidates → estimate ω x = exact x) :
    SCOLHKG.Real.ExactMCEstimator exact (estimate ω) eta := by
  intro x
  by_cases hx : x ∈ candidates
  · simpa [abs_sub_comm] using hevent x hx
  · simp [houtside x hx]
    exact heta

theorem exactMC_bad_event_le_sum
    (candidates : Finset Design)
    (exact : Design → ℝ)
    (estimate : Ω → Design → ℝ)
    (c : Design → NNReal)
    (eta delta : Design → ℝ)
    [IsFiniteMeasure μ]
    (hsub :
      ∀ x ∈ candidates,
        HasSubgaussianMGF
          (fun ω ↦ estimate ω x - exact x)
          (c x)
          μ)
    (heta : ∀ x ∈ candidates, 0 ≤ eta x)
    (htail :
      ∀ x ∈ candidates,
        2 * Real.exp (-(eta x) ^ 2 / (2 * (c x : ℝ))) ≤ delta x) :
    μ.real
        (⋃ x ∈ candidates,
          CenteredSubGaussianBadEvent
            (fun ω ↦ estimate ω x - exact x)
            (eta x))
      ≤ ∑ x ∈ candidates, delta x := by
  exact centeredSubGaussian_finite_candidate_bad_event_le_sum
    (μ := μ)
    (s := candidates)
    (X := fun x ω ↦ estimate ω x - exact x)
    (c := c)
    (radius := eta)
    (delta := delta)
    hsub
    heta
    htail

theorem exactMC_constant_radius_bad_event_le_sum
    (candidates : Finset Design)
    (exact : Design → ℝ)
    (estimate : Ω → Design → ℝ)
    (c : Design → NNReal)
    {eta : ℝ}
    (delta : Design → ℝ)
    [IsFiniteMeasure μ]
    (hsub :
      ∀ x ∈ candidates,
        HasSubgaussianMGF
          (fun ω ↦ estimate ω x - exact x)
          (c x)
          μ)
    (heta : 0 ≤ eta)
    (htail :
      ∀ x ∈ candidates,
        2 * Real.exp (-eta ^ 2 / (2 * (c x : ℝ))) ≤ delta x) :
    μ.real (ExactMCBadEvent candidates exact estimate eta)
      ≤ ∑ x ∈ candidates, delta x := by
  unfold ExactMCBadEvent
  exact exactMC_bad_event_le_sum
    (μ := μ)
    candidates
    exact
    estimate
    c
    (fun _ ↦ eta)
    delta
    hsub
    (fun x hx ↦ heta)
    (fun x hx ↦ by simpa using htail x hx)

structure ExactMCSchedule where
  mcSamples : ℕ
  perDrawVarianceProxy : ℝ
  poolDelta : ℝ
  radius : ℝ

def ExactMCSchedule.Valid
    (s : ExactMCSchedule)
    (poolSize : ℕ) : Prop :=
  0 < s.mcSamples ∧
  0 < s.perDrawVarianceProxy ∧
  0 ≤ s.radius ∧
  0 ≤ s.poolDelta ∧
  2 * Real.exp
      (-(s.radius) ^ 2
        / (2 * (s.perDrawVarianceProxy / s.mcSamples)))
    ≤ s.poolDelta / poolSize

theorem exactMC_schedule_bad_event_le_pool_delta
    (candidates : Finset Design)
    (exact : Design → ℝ)
    (estimate : Ω → Design → ℝ)
    (schedule : ExactMCSchedule)
    [IsFiniteMeasure μ]
    (hvalid : schedule.Valid candidates.card)
    (hsub :
      ∀ x ∈ candidates,
        HasSubgaussianMGF
          (fun ω ↦ estimate ω x - exact x)
          ⟨schedule.perDrawVarianceProxy / schedule.mcSamples,
            by
              have hpos :
                  0 < schedule.perDrawVarianceProxy / schedule.mcSamples := by
                exact div_pos hvalid.2.1 (Nat.cast_pos.mpr hvalid.1)
              exact hpos.le⟩
          μ) :
    μ.real (ExactMCBadEvent candidates exact estimate schedule.radius)
      ≤ schedule.poolDelta := by
  let mcProxy : NNReal :=
    ⟨schedule.perDrawVarianceProxy / schedule.mcSamples,
      by
        have hpos :
            0 < schedule.perDrawVarianceProxy / schedule.mcSamples := by
          exact div_pos hvalid.2.1 (Nat.cast_pos.mpr hvalid.1)
        exact hpos.le⟩
  have hmcProxy_coe :
      (mcProxy : ℝ)
        = schedule.perDrawVarianceProxy / schedule.mcSamples := rfl
  calc
    μ.real (ExactMCBadEvent candidates exact estimate schedule.radius)
      ≤ ∑ x ∈ candidates, schedule.poolDelta / candidates.card := by
        exact exactMC_constant_radius_bad_event_le_sum
          (μ := μ)
          candidates
          exact
          estimate
          (fun _ ↦ mcProxy)
          (fun _ ↦ schedule.poolDelta / candidates.card)
          (by simpa [mcProxy] using hsub)
          hvalid.2.2.1
          (fun x hx ↦ by
            simpa [hmcProxy_coe] using hvalid.2.2.2.2)
    _ = candidates.card * (schedule.poolDelta / candidates.card) := by
        simp
    _ ≤ schedule.poolDelta := by
        by_cases hzero : candidates.card = 0
        · simp [hzero, hvalid.2.2.2.1]
        · have hpos : (0 : ℝ) < candidates.card := by
            exact_mod_cast Nat.pos_of_ne_zero hzero
          have hne : (candidates.card : ℝ) ≠ 0 := ne_of_gt hpos
          calc
            (candidates.card : ℝ)
                * (schedule.poolDelta / candidates.card)
              = schedule.poolDelta := by field_simp [hne]
            _ ≤ schedule.poolDelta := le_rfl

end SCOLHKG.Measure

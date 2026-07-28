import Mathlib

namespace SCOLHKG.Real

/-!
Finite-family implementation contract for TCB-V3.

Source data determine a frozen finite library.  Budgeted target pilots only
change a posterior over that library.  Certification uses an envelope over a
posterior credible family set, so averaging cannot hide an unsafe family that
is still credible.
-/

structure FiniteFamilyPosterior (Family : Type*) [Fintype Family] where
  weight : Family → ℝ
  weight_nonnegative : ∀ family, 0 ≤ weight family
  weight_sum_one : ∑ family, weight family = 1

def credibleFamilyMass
    {Family : Type*} [Fintype Family]
    (posterior : FiniteFamilyPosterior Family)
    (credible : Finset Family) : ℝ :=
  ∑ family ∈ credible, posterior.weight family

structure CredibleFamilySet
    (Family : Type*) [Fintype Family]
    (posterior : FiniteFamilyPosterior Family) where
  members : Finset Family
  delta : ℝ
  delta_nonnegative : 0 ≤ delta
  mass_requirement :
    1 - delta ≤ credibleFamilyMass posterior members

theorem credibleFamilyMass_reaches_declared_level
    {Family : Type*} [Fintype Family]
    {posterior : FiniteFamilyPosterior Family}
    (credible : CredibleFamilySet Family posterior) :
    1 - credible.delta ≤
      credibleFamilyMass posterior credible.members := by
  exact credible.mass_requirement

structure FamilyEnvelopeCertificate
    (Family Design : Type*) where
  credible : Finset Family
  familyUpper : Family → Design → ℝ
  envelope : Design → ℝ
  dominates :
    ∀ family ∈ credible, ∀ design,
      familyUpper family design ≤ envelope design

theorem credible_family_envelope_covers_true_margin
    {Family Design : Type*}
    (certificate : FamilyEnvelopeCertificate Family Design)
    (trueFamily : Family)
    (trueMargin : Design → ℝ)
    (hFamilyCredible : trueFamily ∈ certificate.credible)
    (hFamilyCoverage :
      ∀ design,
        trueMargin design ≤ certificate.familyUpper trueFamily design)
    (design : Design) :
    trueMargin design ≤ certificate.envelope design := by
  exact (hFamilyCoverage design).trans
    (certificate.dominates trueFamily hFamilyCredible design)

def guardedFamilyEnvelope
    {Family Design : Type*}
    (certificate : FamilyEnvelopeCertificate Family Design)
    (guard : Design → ℝ) : Design → ℝ :=
  fun design => certificate.envelope design + guard design

theorem nonnegative_family_guard_cannot_relax_upper
    {Family Design : Type*}
    (certificate : FamilyEnvelopeCertificate Family Design)
    (guard : Design → ℝ)
    (hGuard : ∀ design, 0 ≤ guard design)
    (design : Design) :
    certificate.envelope design ≤
      guardedFamilyEnvelope certificate guard design := by
  unfold guardedFamilyEnvelope
  linarith [hGuard design]

theorem guarded_credible_family_envelope_covers_true_margin
    {Family Design : Type*}
    (certificate : FamilyEnvelopeCertificate Family Design)
    (guard : Design → ℝ)
    (hGuard : ∀ design, 0 ≤ guard design)
    (trueFamily : Family)
    (trueMargin : Design → ℝ)
    (hFamilyCredible : trueFamily ∈ certificate.credible)
    (hFamilyCoverage :
      ∀ design,
        trueMargin design ≤ certificate.familyUpper trueFamily design)
    (design : Design) :
    trueMargin design ≤ guardedFamilyEnvelope certificate guard design := by
  exact (credible_family_envelope_covers_true_margin
    certificate trueFamily trueMargin hFamilyCredible hFamilyCoverage design).trans
    (nonnegative_family_guard_cannot_relax_upper
      certificate guard hGuard design)

theorem guarded_credible_recommendation_is_safe
    {Family Design : Type*}
    (certificate : FamilyEnvelopeCertificate Family Design)
    (guard : Design → ℝ)
    (hGuard : ∀ design, 0 ≤ guard design)
    (trueFamily : Family)
    (trueMargin : Design → ℝ)
    (hFamilyCredible : trueFamily ∈ certificate.credible)
    (hFamilyCoverage :
      ∀ design,
        trueMargin design ≤ certificate.familyUpper trueFamily design)
    {design : Design}
    (hCertified : guardedFamilyEnvelope certificate guard design ≤ 0) :
    trueMargin design ≤ 0 := by
  exact (guarded_credible_family_envelope_covers_true_margin
    certificate guard hGuard trueFamily trueMargin
    hFamilyCredible hFamilyCoverage design).trans hCertified

def generalizedBayesLogWeight
    {Family Pilot : Type*}
    (sourceLogPrior : Family → ℝ)
    (pilotLogEvidence : Pilot → Family → ℝ)
    (temperature : ℝ)
    (pilots : Finset Pilot)
    (family : Family) : ℝ :=
  sourceLogPrior family
    + temperature * ∑ pilot ∈ pilots, pilotLogEvidence pilot family

theorem generalizedBayesLogWeight_target_name_independent
    {Family Pilot TargetName : Type*}
    (sourceLogPrior : Family → ℝ)
    (pilotLogEvidence : Pilot → Family → ℝ)
    (temperature : ℝ)
    (pilots : Finset Pilot)
    (family : Family)
    (firstTargetName secondTargetName : TargetName) :
    (fun _ : TargetName =>
      generalizedBayesLogWeight
        sourceLogPrior pilotLogEvidence temperature pilots family)
        firstTargetName =
      (fun _ : TargetName =>
        generalizedBayesLogWeight
          sourceLogPrior pilotLogEvidence temperature pilots family)
        secondTargetName := by
  rfl

theorem credible_family_certificate_failure_le
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : MeasureTheory.Measure Ω)
    (familyContained familyUpperCovers safeFailure : Set Ω)
    (delta alpha : ENNReal)
    (hFamily : μ familyContainedᶜ ≤ delta)
    (hUpper : μ familyUpperCoversᶜ ≤ alpha)
    (hFailure :
      safeFailure ⊆ familyContainedᶜ ∪ familyUpperCoversᶜ) :
    μ safeFailure ≤ delta + alpha := by
  calc
    μ safeFailure ≤ μ (familyContainedᶜ ∪ familyUpperCoversᶜ) :=
      MeasureTheory.measure_mono hFailure
    _ ≤ μ familyContainedᶜ + μ familyUpperCoversᶜ :=
      MeasureTheory.measure_union_le _ _
    _ ≤ delta + alpha := add_le_add hFamily hUpper

end SCOLHKG.Real

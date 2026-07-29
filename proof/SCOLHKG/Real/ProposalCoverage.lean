import Mathlib
import SCOLHKG.Real.TransferGeneralization

namespace SCOLHKG.Real

open scoped BigOperators

/-!
Source-to-target feasible-basin coverage for the frozen proposal.

The source archive controls a posterior-weighted target miss probability by
PAC-Bayes transfer.  The effective structural dimension enters through an
upper bound on the finite-expert KL term.  Independent frozen proposal draws
then convert a one-draw feasible-mass lower bound into an explicit
`1 - (1 - pLower)^n0` probability of at least one feasible initial policy.
-/

noncomputable section

def effectiveDimensionTransferRadius
    (effectiveDim logLibrary invConfidence sourceSamples : ℝ) : ℝ :=
  (effectiveDim * logLibrary + invConfidence) / sourceSamples

def targetFeasibleMass (targetMiss : ℝ) : ℝ :=
  1 - targetMiss

def proposalFeasibleMassLower
    (sourceMiss domainShift complexityRadius : ℝ) : ℝ :=
  max 0 (1 - (sourceMiss + domainShift + complexityRadius))

def iidProposalHitLowerBound (feasibleMass : ℝ) (n0 : ℕ) : ℝ :=
  1 - (1 - feasibleMass) ^ n0

def targetMissIndicator
    {Expert : Type*} (feasible : Expert → Prop) [DecidablePred feasible]
    (i : Expert) : ℝ :=
  if feasible i then 0 else 1

def finiteAtlasFeasibleMass
    {Expert : Type*} [Fintype Expert]
    (posterior : Expert → ℝ)
    (feasible : Expert → Prop) [DecidablePred feasible] : ℝ :=
  ∑ i, posterior i * if feasible i then 1 else 0

theorem effective_dimension_transfer_radius_eq_pac_radius
    (effectiveDim logLibrary delta sourceSamples : ℝ) :
    effectiveDimensionTransferRadius effectiveDim logLibrary
      (Real.log (1 / delta)) sourceSamples =
      transferGeneralizationRadius
        (effectiveDim * logLibrary) delta sourceSamples := by
  rfl

theorem proposal_feasible_mass_lower_valid
    {sourceMiss targetMiss domainShift complexityRadius : ℝ}
    (hTransfer :
      targetMiss ≤ sourceMiss + domainShift + complexityRadius)
    (hTargetMissLeOne : targetMiss ≤ 1) :
    proposalFeasibleMassLower sourceMiss domainShift complexityRadius ≤
      targetFeasibleMass targetMiss := by
  unfold proposalFeasibleMassLower targetFeasibleMass
  apply max_le
  · linarith
  · linarith

theorem iid_proposal_hit_lower_bound_mono
    {pLower pTarget : ℝ}
    (n0 : ℕ)
    (hLower : pLower ≤ pTarget)
    (hTargetLeOne : pTarget ≤ 1) :
    iidProposalHitLowerBound pLower n0 ≤
      iidProposalHitLowerBound pTarget n0 := by
  have hTargetMissNonnegative : 0 ≤ 1 - pTarget := by
    linarith
  have hMissOrder : 1 - pTarget ≤ 1 - pLower := by
    linarith
  have hPow :
      (1 - pTarget) ^ n0 ≤ (1 - pLower) ^ n0 :=
    pow_le_pow_left₀ hTargetMissNonnegative hMissOrder n0
  unfold iidProposalHitLowerBound
  linarith

theorem finite_source_to_target_proposal_coverage
    {Expert : Type*} [Fintype Expert] [Nonempty Expert]
    {posterior prior : Expert → ℝ}
    {sourceMiss targetMiss domainShift : Expert → ℝ}
    {effectiveDim logLibrary delta sourceSamples : ℝ}
    {n0 : ℕ}
    {hitProbability : ℝ}
    (hPosterior : ∀ i, 0 < posterior i)
    (hPrior : ∀ i, 0 < prior i)
    (hPosteriorNorm : ∑ i, posterior i = 1)
    (hPriorNorm : ∑ i, prior i = 1)
    (hKL :
      finiteTaskKL posterior prior ≤ effectiveDim * logLibrary)
    (hDelta : 0 < delta)
    (hSamples : 0 < sourceSamples)
    (hMoment :
      (∑ i, prior i * Real.exp
        (sourceSamples * sourceTargetGap
          sourceMiss targetMiss domainShift i)) ≤ 1 / delta)
    (hTargetMissNonnegative :
      0 ≤ ∑ i, posterior i * targetMiss i)
    (hTargetMissLeOne :
      (∑ i, posterior i * targetMiss i) ≤ 1)
    (hIID :
      hitProbability =
        iidProposalHitLowerBound
          (targetFeasibleMass (∑ i, posterior i * targetMiss i)) n0) :
    let sourcePosteriorMiss := ∑ i, posterior i * sourceMiss i
    let posteriorShift := ∑ i, posterior i * domainShift i
    let radius := effectiveDimensionTransferRadius effectiveDim logLibrary
      (Real.log (1 / delta)) sourceSamples
    let pLower :=
      proposalFeasibleMassLower sourcePosteriorMiss posteriorShift radius
    pLower ≤ targetFeasibleMass (∑ i, posterior i * targetMiss i) ∧
      iidProposalHitLowerBound pLower n0 ≤ hitProbability := by
  dsimp
  have hTransfer := finite_source_to_target_pac_bayes
    hPosterior hPrior hPosteriorNorm hPriorNorm hKL hDelta hSamples hMoment
  rw [← effective_dimension_transfer_radius_eq_pac_radius] at hTransfer
  have hMass := proposal_feasible_mass_lower_valid
    hTransfer hTargetMissLeOne
  refine ⟨hMass, ?_⟩
  rw [hIID]
  apply iid_proposal_hit_lower_bound_mono n0
  · exact hMass
  · unfold targetFeasibleMass
    linarith

theorem proposal_hit_probability_at_least_one_minus_miss_power
    {pLower hitProbability : ℝ}
    {n0 : ℕ}
    (hCoverage :
      iidProposalHitLowerBound pLower n0 ≤ hitProbability) :
    1 - (1 - pLower) ^ n0 ≤ hitProbability := by
  exact hCoverage

theorem target_feasible_mass_indicator_eq_finite_atlas_mass
    {Expert : Type*} [Fintype Expert]
    {posterior : Expert → ℝ}
    {feasible : Expert → Prop} [DecidablePred feasible]
    (hPosteriorNorm : ∑ i, posterior i = 1) :
    targetFeasibleMass
      (∑ i, posterior i * targetMissIndicator feasible i) =
        finiteAtlasFeasibleMass posterior feasible := by
  unfold targetFeasibleMass finiteAtlasFeasibleMass
  calc
    1 - (∑ i, posterior i * targetMissIndicator feasible i) =
        (∑ i, posterior i) -
          (∑ i, posterior i * targetMissIndicator feasible i) := by
            rw [hPosteriorNorm]
    _ = ∑ i, (
        posterior i - posterior i * targetMissIndicator feasible i) := by
          rw [Finset.sum_sub_distrib]
    _ = ∑ i, posterior i * if feasible i then 1 else 0 := by
          apply Finset.sum_congr rfl
          intro i _
          unfold targetMissIndicator
          by_cases hFeasible : feasible i <;> simp [hFeasible]

theorem positive_finite_atlas_mass_has_feasible_support
    {Expert : Type*} [Fintype Expert]
    {posterior : Expert → ℝ}
    {feasible : Expert → Prop} [DecidablePred feasible]
    (hPositive : 0 < finiteAtlasFeasibleMass posterior feasible) :
    ∃ i, feasible i := by
  by_contra hExists
  have hNone : ∀ i, ¬ feasible i := by
    intro i hFeasible
    exact hExists ⟨i, hFeasible⟩
  simp [finiteAtlasFeasibleMass, hNone] at hPositive

/-!
The deployed paper proposal is a deterministic finite atlas, not `n0`
independent draws.  This theorem is its implementation-matched coverage
statement.  PAC-Bayes transfer lower-bounds posterior feasible mass on the
frozen atlas; if that lower bound is strictly positive, at least one policy in
the atlas support is feasible.  The IID theorem above remains available only
for a genuinely randomized proposal backend.
-/

theorem finite_source_to_target_atlas_coverage
    {Expert : Type*} [Fintype Expert] [Nonempty Expert]
    {posterior prior : Expert → ℝ}
    {sourceMiss domainShift : Expert → ℝ}
    {feasible : Expert → Prop} [DecidablePred feasible]
    {effectiveDim logLibrary delta sourceSamples : ℝ}
    (hPosterior : ∀ i, 0 < posterior i)
    (hPrior : ∀ i, 0 < prior i)
    (hPosteriorNorm : ∑ i, posterior i = 1)
    (hPriorNorm : ∑ i, prior i = 1)
    (hKL :
      finiteTaskKL posterior prior ≤ effectiveDim * logLibrary)
    (hDelta : 0 < delta)
    (hSamples : 0 < sourceSamples)
    (hMoment :
      (∑ i, prior i * Real.exp
        (sourceSamples * sourceTargetGap sourceMiss
          (targetMissIndicator feasible) domainShift i)) ≤ 1 / delta)
    (hTargetMissLeOne :
      (∑ i, posterior i * targetMissIndicator feasible i) ≤ 1)
    (hLowerPositive :
      0 < proposalFeasibleMassLower
        (∑ i, posterior i * sourceMiss i)
        (∑ i, posterior i * domainShift i)
        (effectiveDimensionTransferRadius effectiveDim logLibrary
          (Real.log (1 / delta)) sourceSamples)) :
    let sourcePosteriorMiss := ∑ i, posterior i * sourceMiss i
    let posteriorShift := ∑ i, posterior i * domainShift i
    let radius := effectiveDimensionTransferRadius effectiveDim logLibrary
      (Real.log (1 / delta)) sourceSamples
    let pLower :=
      proposalFeasibleMassLower sourcePosteriorMiss posteriorShift radius
    pLower ≤ finiteAtlasFeasibleMass posterior feasible ∧
      ∃ i, feasible i := by
  dsimp
  have hTransfer := finite_source_to_target_pac_bayes
    hPosterior hPrior hPosteriorNorm hPriorNorm hKL hDelta hSamples hMoment
  rw [← effective_dimension_transfer_radius_eq_pac_radius] at hTransfer
  have hMassTarget := proposal_feasible_mass_lower_valid
    hTransfer hTargetMissLeOne
  rw [target_feasible_mass_indicator_eq_finite_atlas_mass
    hPosteriorNorm] at hMassTarget
  refine ⟨hMassTarget, ?_⟩
  apply positive_finite_atlas_mass_has_feasible_support
  exact lt_of_lt_of_le hLowerPositive hMassTarget

end

end SCOLHKG.Real

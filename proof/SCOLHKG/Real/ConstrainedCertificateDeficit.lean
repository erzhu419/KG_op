import Mathlib

namespace SCOLHKG.Real

/-!
V53/V54 constrained certificate-deficit policy improvement.

The implementation evaluates every action with two higher-is-better posterior
scores on the same pre-update action universe and nested common random numbers:

* Bayes-risk reduction relative to the promoted V51 terminal functional;
* reduction of the certificate deficit
  `max (min_x M_D(x)) 0`.

A supplemental action may replace the best V51 action only when both estimated
score improvements exceed twice their separately calibrated uniform Monte
Carlo errors. The results below show that, on those two numerical-fidelity
events, a switch strictly improves both exact posterior scores. Retaining the
fallback is definitionally noninferior.

V54 replaces the global worst-action radius by a separately audited
common-random-number pair-difference radius for each challenger relative to
the literal V51 fallback. The nested-prefix discrepancy is an observable
radius construction, while its domination of the exact integration error
remains an explicit numerical-fidelity assumption rather than a theorem.

The theorem is conditional on posterior-model and numerical-fidelity events. It does
not assert simulator calibration, candidate-pool coverage, or certificate
nonvacuity without their separately stated assumptions.
-/

def certificateDeficit (minimumMargin : ℝ) : ℝ :=
  max minimumMargin 0

theorem certificateDeficit_nonnegative (minimumMargin : ℝ) :
    0 ≤ certificateDeficit minimumMargin := by
  exact le_max_right _ _

theorem certificateDeficit_eq_zero_iff (minimumMargin : ℝ) :
    certificateDeficit minimumMargin = 0 ↔ minimumMargin ≤ 0 := by
  simp [certificateDeficit]

def terminalReduction (current expectedAfter : ℝ) : ℝ :=
  current - expectedAfter

/--
V53-v3's robust numerical score. The scale is frozen by the current posterior
before any action fantasy is evaluated. Clipping is applied to every fantasy
improvement before integration, rather than to the final Monte Carlo mean.
-/
noncomputable def boundedCurrentGain (scale current afterFantasy : ℝ) : ℝ :=
  max (-1) (min 1 ((current - afterFantasy) / scale))

theorem boundedCurrentGain_mem_Icc
    (scale current afterFantasy : ℝ) :
    boundedCurrentGain scale current afterFantasy ∈ Set.Icc (-1) 1 := by
  constructor
  · exact le_max_left _ _
  · exact max_le (by norm_num) (min_le_left _ _)

theorem boundedCurrentGain_abs_le_one
    (scale current afterFantasy : ℝ) :
    |boundedCurrentGain scale current afterFantasy| ≤ 1 := by
  exact abs_le.mpr (boundedCurrentGain_mem_Icc scale current afterFantasy)

theorem boundedCurrentGain_pair_difference_le_two
    (scale current firstAfter secondAfter : ℝ) :
    |boundedCurrentGain scale current firstAfter -
        boundedCurrentGain scale current secondAfter| ≤ 2 := by
  have hFirst := boundedCurrentGain_mem_Icc scale current firstAfter
  have hSecond := boundedCurrentGain_mem_Icc scale current secondAfter
  rw [abs_le]
  constructor <;> linarith [hFirst.1, hFirst.2, hSecond.1, hSecond.2]

theorem terminalReduction_pos_iff
    (current expectedAfter : ℝ) :
    0 < terminalReduction current expectedAfter ↔
      expectedAfter < current := by
  unfold terminalReduction
  constructor <;> intro h <;> linarith

def UniformScoreApproximation
    {Action : Type*}
    (exact estimate : Action → ℝ)
    (eta : ℝ) : Prop :=
  ∀ action, |estimate action - exact action| ≤ eta

def PassesTwoEtaScoreGuard
    {Action : Type*}
    (estimate : Action → ℝ)
    (eta : ℝ)
    (baseline challenger : Action) : Prop :=
  estimate baseline + 2 * eta < estimate challenger

theorem two_eta_score_guard_implies_exact_improvement
    {Action : Type*}
    {exact estimate : Action → ℝ}
    {eta : ℝ}
    {baseline challenger : Action}
    (hUniform : UniformScoreApproximation exact estimate eta)
    (hGuard : PassesTwoEtaScoreGuard
      estimate eta baseline challenger) :
    exact baseline < exact challenger := by
  have hBaseline := abs_le.mp (hUniform baseline)
  have hChallenger := abs_le.mp (hUniform challenger)
  unfold PassesTwoEtaScoreGuard at hGuard
  linarith

def PassesConstrainedCertificateGuard
    {Action : Type*}
    (riskEstimate certificateEstimate : Action → ℝ)
    (riskEta certificateEta : ℝ)
    (baseline challenger : Action) : Prop :=
  PassesTwoEtaScoreGuard
      riskEstimate riskEta baseline challenger ∧
    PassesTwoEtaScoreGuard
      certificateEstimate certificateEta baseline challenger

theorem constrained_guard_improves_risk_and_certificate
    {Action : Type*}
    {exactRisk riskEstimate exactCertificate certificateEstimate :
      Action → ℝ}
    {riskEta certificateEta : ℝ}
    {baseline challenger : Action}
    (hRiskUniform : UniformScoreApproximation
      exactRisk riskEstimate riskEta)
    (hCertificateUniform : UniformScoreApproximation
      exactCertificate certificateEstimate certificateEta)
    (hGuard : PassesConstrainedCertificateGuard
      riskEstimate certificateEstimate
      riskEta certificateEta baseline challenger) :
    exactRisk baseline < exactRisk challenger ∧
      exactCertificate baseline < exactCertificate challenger := by
  exact ⟨
    two_eta_score_guard_implies_exact_improvement
      hRiskUniform hGuard.1,
    two_eta_score_guard_implies_exact_improvement
      hCertificateUniform hGuard.2,
  ⟩

noncomputable def positiveScoreNormalize (scale score : ℝ) : ℝ :=
  score / scale

theorem positive_score_normalize_lt_iff
    {scale left right : ℝ}
    (hScale : 0 < scale) :
    positiveScoreNormalize scale left <
        positiveScoreNormalize scale right ↔
      left < right := by
  simpa [positiveScoreNormalize] using
    (div_lt_div_iff_of_pos_right hScale : left / scale < right / scale ↔
      left < right)

theorem uniform_score_approximation_normalize
    {Action : Type*}
    {exact estimate : Action → ℝ}
    {eta scale : ℝ}
    (hScale : 0 < scale)
    (hUniform : UniformScoreApproximation exact estimate eta) :
    UniformScoreApproximation
      (fun action => positiveScoreNormalize scale (exact action))
      (fun action => positiveScoreNormalize scale (estimate action))
      (eta / scale) := by
  intro action
  unfold positiveScoreNormalize
  rw [← sub_div, abs_div, abs_of_pos hScale]
  exact (div_le_div_iff_of_pos_right hScale).2 (hUniform action)

theorem two_eta_score_guard_normalize_iff
    {Action : Type*}
    {estimate : Action → ℝ}
    {eta scale : ℝ}
    {baseline challenger : Action}
    (hScale : 0 < scale) :
    PassesTwoEtaScoreGuard
        (fun action => positiveScoreNormalize scale (estimate action))
        (eta / scale) baseline challenger ↔
      PassesTwoEtaScoreGuard
        estimate eta baseline challenger := by
  unfold PassesTwoEtaScoreGuard positiveScoreNormalize
  constructor
  · intro hGuard
    have hScaled :
        (estimate baseline + 2 * eta) / scale <
          estimate challenger / scale := by
      calc
        (estimate baseline + 2 * eta) / scale =
            estimate baseline / scale + 2 * (eta / scale) := by ring
        _ < estimate challenger / scale := hGuard
    exact (div_lt_div_iff_of_pos_right hScale).1 hScaled
  · intro hGuard
    have hScaled :
        (estimate baseline + 2 * eta) / scale <
          estimate challenger / scale :=
      (div_lt_div_iff_of_pos_right hScale).2 hGuard
    calc
      estimate baseline / scale + 2 * (eta / scale) =
          (estimate baseline + 2 * eta) / scale := by ring
      _ < estimate challenger / scale := hScaled

theorem constrained_guard_normalize_iff
    {Action : Type*}
    {riskEstimate certificateEstimate : Action → ℝ}
    {riskEta certificateEta riskScale certificateScale : ℝ}
    {baseline challenger : Action}
    (hRiskScale : 0 < riskScale)
    (hCertificateScale : 0 < certificateScale) :
    PassesConstrainedCertificateGuard
        (fun action =>
          positiveScoreNormalize riskScale (riskEstimate action))
        (fun action =>
          positiveScoreNormalize certificateScale
            (certificateEstimate action))
        (riskEta / riskScale)
        (certificateEta / certificateScale)
        baseline challenger ↔
      PassesConstrainedCertificateGuard
        riskEstimate certificateEstimate
        riskEta certificateEta baseline challenger := by
  unfold PassesConstrainedCertificateGuard
  rw [
    two_eta_score_guard_normalize_iff hRiskScale,
    two_eta_score_guard_normalize_iff hCertificateScale,
  ]

def PairDifferenceApproximation
    {Action : Type*}
    (exact estimate : Action → ℝ)
    (radius : ℝ)
    (baseline challenger : Action) : Prop :=
  |(estimate challenger - estimate baseline) -
      (exact challenger - exact baseline)| ≤ radius

def PassesPairDifferenceGuard
    {Action : Type*}
    (estimate : Action → ℝ)
    (radius : ℝ)
    (baseline challenger : Action) : Prop :=
  radius < estimate challenger - estimate baseline

theorem pair_difference_guard_implies_exact_improvement
    {Action : Type*}
    {exact estimate : Action → ℝ}
    {radius : ℝ}
    {baseline challenger : Action}
    (hPair : PairDifferenceApproximation
      exact estimate radius baseline challenger)
    (hGuard : PassesPairDifferenceGuard
      estimate radius baseline challenger) :
    exact baseline < exact challenger := by
  have hUpper := (abs_le.mp hPair).2
  unfold PassesPairDifferenceGuard at hGuard
  linarith

def nestedPairRadius
    {Action : Type*}
    (multiplier : ℝ)
    (prefixEstimate highEstimate : Action → ℝ)
    (baseline challenger : Action) : ℝ :=
  multiplier * abs (
    (highEstimate challenger - highEstimate baseline) -
      (prefixEstimate challenger - prefixEstimate baseline))

def NestedPairDifferenceError
    {Action : Type*}
    (exact prefixEstimate highEstimate : Action → ℝ)
    (multiplier : ℝ)
    (baseline challenger : Action) : Prop :=
  PairDifferenceApproximation
    exact highEstimate
    (nestedPairRadius multiplier prefixEstimate highEstimate
      baseline challenger)
    baseline challenger

theorem nested_pair_guard_implies_exact_improvement
    {Action : Type*}
    {exact prefixEstimate highEstimate : Action → ℝ}
    {multiplier : ℝ}
    {baseline challenger : Action}
    (hError : NestedPairDifferenceError
      exact prefixEstimate highEstimate multiplier baseline challenger)
    (hGuard : PassesPairDifferenceGuard
      highEstimate
      (nestedPairRadius multiplier prefixEstimate highEstimate
        baseline challenger)
      baseline challenger) :
    exact baseline < exact challenger := by
  exact pair_difference_guard_implies_exact_improvement hError hGuard


/-- Action-specific absolute score accuracy relative to the current terminal state. -/
def AbsoluteScoreApproximation
    {Action : Type*}
    (exact estimate : Action → ℝ)
    (radius : Action → ℝ) : Prop :=
  ∀ action, |estimate action - exact action| ≤ radius action

def scoreLowerConfidenceBound
    {Action : Type*}
    (estimate radius : Action → ℝ)
    (action : Action) : ℝ :=
  estimate action - radius action

theorem score_lower_confidence_bound_le_exact
    {Action : Type*}
    {exact estimate radius : Action → ℝ}
    (hApproximation : AbsoluteScoreApproximation exact estimate radius)
    (action : Action) :
    scoreLowerConfidenceBound estimate radius action ≤ exact action := by
  have hUpper := (abs_le.mp (hApproximation action)).2
  unfold scoreLowerConfidenceBound
  linarith

def currentRelativeJointLowerConfidenceBound
    {Action : Type*}
    (riskEstimate certificateEstimate : Action → ℝ)
    (riskRadius certificateRadius : Action → ℝ)
    (action : Action) : ℝ :=
  min
    (scoreLowerConfidenceBound riskEstimate riskRadius action)
    (scoreLowerConfidenceBound certificateEstimate certificateRadius action)

def PassesCurrentRelativeJointGuard
    {Action : Type*}
    (riskEstimate certificateEstimate : Action → ℝ)
    (riskRadius certificateRadius : Action → ℝ)
    (action : Action) : Prop :=
  0 < currentRelativeJointLowerConfidenceBound
    riskEstimate certificateEstimate riskRadius certificateRadius action

/--
The V55 maximin guard certifies positive current-relative reduction of both
terminal losses. It does not claim to dominate the V51 risk-maximizing action.
-/
theorem current_relative_joint_guard_improves_both_terminal_scores
    {Action : Type*}
    {exactRisk riskEstimate exactCertificate certificateEstimate :
      Action → ℝ}
    {riskRadius certificateRadius : Action → ℝ}
    {action : Action}
    (hRisk : AbsoluteScoreApproximation
      exactRisk riskEstimate riskRadius)
    (hCertificate : AbsoluteScoreApproximation
      exactCertificate certificateEstimate certificateRadius)
    (hGuard : PassesCurrentRelativeJointGuard
      riskEstimate certificateEstimate riskRadius certificateRadius action) :
    0 < exactRisk action ∧ 0 < exactCertificate action := by
  have hBounds :
      0 < scoreLowerConfidenceBound riskEstimate riskRadius action ∧
        0 < scoreLowerConfidenceBound
          certificateEstimate certificateRadius action := by
    exact lt_min_iff.mp hGuard
  exact ⟨
    lt_of_lt_of_le hBounds.1
      (score_lower_confidence_bound_le_exact hRisk action),
    lt_of_lt_of_le hBounds.2
      (score_lower_confidence_bound_le_exact hCertificate action),
  ⟩

theorem current_relative_joint_guard_decreases_both_terminal_costs
    {Action : Type*}
    {exactRisk exactCertificate : Action → ℝ}
    {currentRisk afterRisk currentCertificate afterCertificate : ℝ}
    {action : Action}
    (hPositive : 0 < exactRisk action ∧ 0 < exactCertificate action)
    (hRiskReduction :
      exactRisk action = terminalReduction currentRisk afterRisk)
    (hCertificateReduction :
      exactCertificate action =
        terminalReduction currentCertificate afterCertificate) :
    afterRisk < currentRisk ∧ afterCertificate < currentCertificate := by
  have hRiskPositive := hPositive.1
  have hCertificatePositive := hPositive.2
  rw [hRiskReduction] at hRiskPositive
  rw [hCertificateReduction] at hCertificatePositive
  exact ⟨
    (terminalReduction_pos_iff currentRisk afterRisk).mp hRiskPositive,
    (terminalReduction_pos_iff
      currentCertificate afterCertificate).mp hCertificatePositive,
  ⟩

def PassesPairedConstrainedGuard
    {Action : Type*}
    (riskEstimate certificateEstimate : Action → ℝ)
    (riskRadius certificateRadius : ℝ)
    (baseline challenger : Action) : Prop :=
  PassesPairDifferenceGuard
      riskEstimate riskRadius baseline challenger ∧
    PassesPairDifferenceGuard
      certificateEstimate certificateRadius baseline challenger

theorem paired_constrained_guard_improves_risk_and_certificate
    {Action : Type*}
    {exactRisk riskEstimate exactCertificate certificateEstimate :
      Action → ℝ}
    {riskRadius certificateRadius : ℝ}
    {baseline challenger : Action}
    (hRiskPair : PairDifferenceApproximation
      exactRisk riskEstimate riskRadius baseline challenger)
    (hCertificatePair : PairDifferenceApproximation
      exactCertificate certificateEstimate certificateRadius
      baseline challenger)
    (hGuard : PassesPairedConstrainedGuard
      riskEstimate certificateEstimate
      riskRadius certificateRadius baseline challenger) :
    exactRisk baseline < exactRisk challenger ∧
      exactCertificate baseline < exactCertificate challenger := by
  exact ⟨
    pair_difference_guard_implies_exact_improvement
      hRiskPair hGuard.1,
    pair_difference_guard_implies_exact_improvement
      hCertificatePair hGuard.2,
  ⟩

theorem paired_fallback_or_switch_joint_noninferiority
    {Action : Type*}
    {exactRisk riskEstimate exactCertificate certificateEstimate :
      Action → ℝ}
    {riskRadius certificateRadius : ℝ}
    {baseline challenger selected : Action}
    (hRiskPair : PairDifferenceApproximation
      exactRisk riskEstimate riskRadius baseline challenger)
    (hCertificatePair : PairDifferenceApproximation
      exactCertificate certificateEstimate certificateRadius
      baseline challenger)
    (hDecision :
      selected = baseline ∨
        (selected = challenger ∧
          PassesPairedConstrainedGuard
            riskEstimate certificateEstimate
            riskRadius certificateRadius baseline challenger)) :
    exactRisk baseline ≤ exactRisk selected ∧
      exactCertificate baseline ≤ exactCertificate selected := by
  rcases hDecision with rfl | ⟨rfl, hGuard⟩
  · exact ⟨le_rfl, le_rfl⟩
  · have hImprovement :=
      paired_constrained_guard_improves_risk_and_certificate
        hRiskPair hCertificatePair hGuard
    exact ⟨hImprovement.1.le, hImprovement.2.le⟩

theorem constrained_fallback_or_switch_joint_noninferiority
    {Action : Type*}
    {exactRisk riskEstimate exactCertificate certificateEstimate :
      Action → ℝ}
    {riskEta certificateEta : ℝ}
    {baseline challenger selected : Action}
    (hRiskUniform : UniformScoreApproximation
      exactRisk riskEstimate riskEta)
    (hCertificateUniform : UniformScoreApproximation
      exactCertificate certificateEstimate certificateEta)
    (hDecision :
      selected = baseline ∨
        (selected = challenger ∧
          PassesConstrainedCertificateGuard
            riskEstimate certificateEstimate
            riskEta certificateEta baseline challenger)) :
    exactRisk baseline ≤ exactRisk selected ∧
      exactCertificate baseline ≤ exactCertificate selected := by
  rcases hDecision with rfl | ⟨rfl, hGuard⟩
  · exact ⟨le_rfl, le_rfl⟩
  · have hImprovement :=
      constrained_guard_improves_risk_and_certificate
        hRiskUniform hCertificateUniform hGuard
    exact ⟨hImprovement.1.le, hImprovement.2.le⟩



/-- The expanded posterior-support action set retains every literal V51 action. -/
def LiteralActionSuperset
    {Action : Type*}
    (baseline expanded : Set Action) : Prop :=
  baseline ⊆ expanded

theorem literal_action_superset_preserves_fallback
    {Action : Type*}
    {baseline expanded : Set Action}
    {fallback : Action}
    (hSubset : LiteralActionSuperset baseline expanded)
    (hFallback : fallback ∈ baseline) :
    fallback ∈ expanded := by
  exact hSubset hFallback

/--
Adding oracle-free posterior support actions cannot invalidate the V51
fallback guarantee. A guarded switch remains jointly noninferior, while a
failed guard executes the retained baseline action.
-/
theorem paired_action_superset_policy_joint_noninferiority
    {Action : Type*}
    {baselineSet expandedSet : Set Action}
    {exactRisk riskEstimate exactCertificate certificateEstimate :
      Action → ℝ}
    {riskRadius certificateRadius : ℝ}
    {baseline challenger selected : Action}
    (hSubset : LiteralActionSuperset baselineSet expandedSet)
    (hBaseline : baseline ∈ baselineSet)
    (hChallenger : challenger ∈ expandedSet)
    (hRiskPair : PairDifferenceApproximation
      exactRisk riskEstimate riskRadius baseline challenger)
    (hCertificatePair : PairDifferenceApproximation
      exactCertificate certificateEstimate certificateRadius
      baseline challenger)
    (hDecision :
      selected = baseline ∨
        (selected = challenger ∧
          PassesPairedConstrainedGuard
            riskEstimate certificateEstimate
            riskRadius certificateRadius baseline challenger)) :
    selected ∈ expandedSet ∧
      exactRisk baseline ≤ exactRisk selected ∧
      exactCertificate baseline ≤ exactCertificate selected := by
  have hScores := paired_fallback_or_switch_joint_noninferiority
    hRiskPair hCertificatePair hDecision
  have hSelected : selected ∈ expandedSet := by
    rcases hDecision with hFallback | hSwitch
    · rw [hFallback]
      exact hSubset hBaseline
    · rw [hSwitch.1]
      exact hChallenger
  exact ⟨hSelected, hScores.1, hScores.2⟩

/--
Evaluating two terminal functionals after each shared fantasy update changes
only the execution schedule. It returns exactly the same two weighted gains as
two separate passes over the identical finite fantasy law.
-/
noncomputable def finiteWeightedTerminalGain
    {n : ℕ}
    (weight : Fin n → ℝ)
    (current : ℝ)
    (afterFantasy : Fin n → ℝ) : ℝ :=
  ∑ index, weight index * (current - afterFantasy index)

noncomputable def jointFiniteWeightedTerminalGain
    {n : ℕ}
    (weight : Fin n → ℝ)
    (riskCurrent certificateCurrent : ℝ)
    (riskAfter certificateAfter : Fin n → ℝ) : ℝ × ℝ :=
  (
    finiteWeightedTerminalGain weight riskCurrent riskAfter,
    finiteWeightedTerminalGain weight certificateCurrent certificateAfter
  )

theorem joint_terminal_head_reuse_eq_separate_passes
    {n : ℕ}
    (weight : Fin n → ℝ)
    (riskCurrent certificateCurrent : ℝ)
    (riskAfter certificateAfter : Fin n → ℝ) :
    jointFiniteWeightedTerminalGain
        weight riskCurrent certificateCurrent riskAfter certificateAfter =
      (
        finiteWeightedTerminalGain weight riskCurrent riskAfter,
        finiteWeightedTerminalGain
          weight certificateCurrent certificateAfter
      ) := by
  rfl

end SCOLHKG.Real

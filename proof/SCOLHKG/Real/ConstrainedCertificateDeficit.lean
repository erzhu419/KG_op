import Mathlib

namespace SCOLHKG.Real

/-!
V53 constrained certificate-deficit policy improvement.

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

The theorem is conditional on posterior-model and uniform-MC events. It does
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

end SCOLHKG.Real

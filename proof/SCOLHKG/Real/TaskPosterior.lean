import Mathlib
import SCOLHKG.Real.ConditionalVariance
import SCOLHKG.Real.ExactKGImplementation

namespace SCOLHKG.Real

open scoped BigOperators

/-!
Finite task-posterior layer used by `representation/task_posterior.py`.

The implementation maintains positive unnormalized generalized-Bayes weights,
normalizes them, aggregates expert moments by the law of total variance, and
uses an upper envelope over a KL ambiguity set for certification.  The KL-ball
construction is deliberately abstract here: the robust theorems apply to every
alternative normalized nonnegative weight vector admitted by that set.
-/

noncomputable def normalizeFiniteWeights
    {ι : Type*} [Fintype ι]
    (u : ι → ℝ) (i : ι) : ℝ :=
  u i / ∑ j, u j

noncomputable def generalizedBayesMass
    {ι : Type*}
    (prior score : ι → ℝ)
    (eta : ℝ)
    (i : ι) : ℝ :=
  prior i * Real.exp (eta * score i)

theorem normalizeFiniteWeights_sum_eq_one
    {ι : Type*} [Fintype ι]
    {u : ι → ℝ}
    (hSum : (∑ i, u i) ≠ 0) :
    ∑ i, normalizeFiniteWeights u i = 1 := by
  simp only [normalizeFiniteWeights]
  rw [← Finset.sum_div]
  exact div_self hSum

theorem normalizeFiniteWeights_nonneg
    {ι : Type*} [Fintype ι]
    {u : ι → ℝ}
    (hu : ∀ i, 0 ≤ u i)
    (hSum : 0 < ∑ i, u i)
    (i : ι) :
    0 ≤ normalizeFiniteWeights u i := by
  exact div_nonneg (hu i) (le_of_lt hSum)

theorem generalizedBayesMass_pos
    {ι : Type*}
    {prior score : ι → ℝ}
    {eta : ℝ}
    (hPrior : ∀ i, 0 < prior i)
    (i : ι) :
    0 < generalizedBayesMass prior score eta i := by
  unfold generalizedBayesMass
  exact mul_pos (hPrior i) (Real.exp_pos _)

theorem generalizedBayes_normalized_support
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {prior score : ι → ℝ}
    {eta : ℝ}
    (hPrior : ∀ i, 0 < prior i)
    (i : ι) :
    0 < normalizeFiniteWeights
      (generalizedBayesMass prior score eta) i := by
  have hEach : ∀ j, 0 < generalizedBayesMass prior score eta j :=
    generalizedBayesMass_pos hPrior
  have hSum : 0 < ∑ j, generalizedBayesMass prior score eta j := by
    exact Finset.sum_pos (fun j _hj => hEach j) Finset.univ_nonempty
  exact div_pos (hEach i) hSum

def finiteTaskProposalMass
    {ι : Type*}
    (prior posterior : ι → ℝ)
    (epsilon : ℝ)
    (i : ι) : ℝ :=
  (1 - epsilon) * posterior i + epsilon * prior i

theorem finite_task_proposal_normalized
    {ι : Type*} [Fintype ι]
    {prior posterior : ι → ℝ}
    {epsilon : ℝ}
    (hPrior : ∑ i, prior i = 1)
    (hPosterior : ∑ i, posterior i = 1) :
    ∑ i, finiteTaskProposalMass prior posterior epsilon i = 1 := by
  simp only [finiteTaskProposalMass]
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
  rw [hPrior, hPosterior]
  ring

theorem finite_task_proposal_preserves_prior_support
    {ι : Type*}
    {prior posterior : ι → ℝ}
    {epsilon : ℝ}
    (hEpsilonLeOne : epsilon ≤ 1)
    (hPosterior : ∀ i, 0 ≤ posterior i)
    (i : ι) :
    epsilon * prior i ≤
      finiteTaskProposalMass prior posterior epsilon i := by
  unfold finiteTaskProposalMass
  have hResidual : 0 ≤ (1 - epsilon) * posterior i :=
    mul_nonneg (sub_nonneg.mpr hEpsilonLeOne) (hPosterior i)
  linarith

theorem finite_task_proposal_positive_of_prior_positive
    {ι : Type*}
    {prior posterior : ι → ℝ}
    {epsilon : ℝ}
    (hEpsilon : 0 < epsilon)
    (hEpsilonLeOne : epsilon ≤ 1)
    (hPrior : ∀ i, 0 < prior i)
    (hPosterior : ∀ i, 0 ≤ posterior i)
    (i : ι) :
    0 < finiteTaskProposalMass prior posterior epsilon i := by
  have hLower := finite_task_proposal_preserves_prior_support
    (prior := prior)
    (posterior := posterior)
    (epsilon := epsilon)
    hEpsilonLeOne
    hPosterior
    i
  exact lt_of_lt_of_le (mul_pos hEpsilon (hPrior i)) hLower

def taskWithinEpistemic
    {ι : Type*} [Fintype ι]
    (q epistemic : ι → ℝ) : ℝ :=
  ∑ i, q i * epistemic i

def taskBetweenMean
    {ι : Type*} [Fintype ι]
    (q mean : ι → ℝ) : ℝ :=
  let μ := finiteMean q mean
  ∑ i, q i * (mean i - μ) ^ 2

def taskAleatoric
    {ι : Type*} [Fintype ι]
    (q aleatoric : ι → ℝ) : ℝ :=
  ∑ i, q i * aleatoric i

def taskTotalPredictiveVariance
    {ι : Type*} [Fintype ι]
    (q mean epistemic aleatoric : ι → ℝ) : ℝ :=
  taskWithinEpistemic q epistemic
    + taskBetweenMean q mean
    + taskAleatoric q aleatoric

theorem task_total_variance_is_within_between_aleatoric
    {ι : Type*} [Fintype ι]
    (q mean epistemic aleatoric : ι → ℝ) :
    taskTotalPredictiveVariance q mean epistemic aleatoric =
      finiteTotalVariance q mean epistemic
        + taskAleatoric q aleatoric := by
  rw [finite_law_total_variance]
  rfl

theorem weighted_expectation_le_envelope
    {ι : Type*} [Fintype ι]
    {q payoff : ι → ℝ}
    {upper : ℝ}
    (hq : ∀ i, 0 ≤ q i)
    (hNorm : ∑ i, q i = 1)
    (hUpper : ∀ i, payoff i ≤ upper) :
    ∑ i, q i * payoff i ≤ upper := by
  calc
    ∑ i, q i * payoff i ≤ ∑ i, q i * upper := by
      exact Finset.sum_le_sum (fun i _hi =>
        mul_le_mul_of_nonneg_left (hUpper i) (hq i))
    _ = (∑ i, q i) * upper := by
      rw [Finset.sum_mul]
    _ = upper := by rw [hNorm, one_mul]

noncomputable def finiteTaskKL
    {ι : Type*} [Fintype ι]
    (q p : ι → ℝ) : ℝ :=
  ∑ i, q i * Real.log (q i / p i)

theorem finiteTaskKL_nonnegative
    {ι : Type*} [Fintype ι]
    {q p : ι → ℝ}
    (hq : ∀ i, 0 < q i)
    (hp : ∀ i, 0 < p i)
    (hqNorm : ∑ i, q i = 1)
    (hpNorm : ∑ i, p i = 1) :
    0 ≤ finiteTaskKL q p := by
  have hTerm : ∀ i, q i * Real.log (p i / q i) ≤ p i - q i := by
    intro i
    have hLog := Real.log_le_sub_one_of_pos (div_pos (hp i) (hq i))
    calc
      q i * Real.log (p i / q i)
          ≤ q i * (p i / q i - 1) :=
            mul_le_mul_of_nonneg_left hLog (le_of_lt (hq i))
      _ = p i - q i := by
        field_simp [ne_of_gt (hq i)]
  have hSum : ∑ i, q i * Real.log (p i / q i) ≤ 0 := by
    calc
      ∑ i, q i * Real.log (p i / q i)
          ≤ ∑ i, (p i - q i) :=
            Finset.sum_le_sum (fun i _hi => hTerm i)
      _ = 0 := by
        rw [Finset.sum_sub_distrib, hpNorm, hqNorm, sub_self]
  have hRewrite :
      finiteTaskKL q p = -∑ i, q i * Real.log (p i / q i) := by
    unfold finiteTaskKL
    rw [← Finset.sum_neg_distrib]
    apply Finset.sum_congr rfl
    intro i _hi
    rw [Real.log_div (ne_of_gt (hq i)) (ne_of_gt (hp i))]
    rw [Real.log_div (ne_of_gt (hp i)) (ne_of_gt (hq i))]
    ring
  rw [hRewrite]
  exact neg_nonneg.mpr hSum

theorem finite_task_change_of_measure
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {q p payoff : ι → ℝ}
    (hq : ∀ i, 0 < q i)
    (hp : ∀ i, 0 < p i)
    (hqNorm : ∑ i, q i = 1)
    (_hpNorm : ∑ i, p i = 1) :
    (∑ i, q i * payoff i) ≤
      finiteTaskKL q p + Real.log (∑ i, p i * Real.exp (payoff i)) := by
  let z : ℝ := ∑ i, p i * Real.exp (payoff i)
  let tilted : ι → ℝ := fun i => p i * Real.exp (payoff i) / z
  have hPositiveTerm : ∀ i, 0 < p i * Real.exp (payoff i) := by
    intro i
    exact mul_pos (hp i) (Real.exp_pos _)
  have hz : 0 < z := by
    unfold z
    exact Finset.sum_pos (fun i _hi => hPositiveTerm i) Finset.univ_nonempty
  have hTilted : ∀ i, 0 < tilted i := by
    intro i
    exact div_pos (hPositiveTerm i) hz
  have hTiltedNorm : ∑ i, tilted i = 1 := by
    unfold tilted z
    rw [← Finset.sum_div]
    exact div_self (ne_of_gt hz)
  have hKL := finiteTaskKL_nonnegative hq hTilted hqNorm hTiltedNorm
  have hExpansion :
      finiteTaskKL q tilted =
        finiteTaskKL q p
          - (∑ i, q i * payoff i)
          + Real.log z := by
    unfold finiteTaskKL
    calc
      ∑ i, q i * Real.log (q i / tilted i) =
          ∑ i, (q i * Real.log (q i / p i)
            - q i * payoff i + q i * Real.log z) := by
        apply Finset.sum_congr rfl
        intro i _hi
        have hqNe := ne_of_gt (hq i)
        have hpNe := ne_of_gt (hp i)
        have hzNe := ne_of_gt hz
        have hExpNe : Real.exp (payoff i) ≠ 0 := ne_of_gt (Real.exp_pos _)
        unfold tilted
        rw [Real.log_div hqNe (div_ne_zero (mul_ne_zero hpNe hExpNe) hzNe)]
        rw [Real.log_div hqNe hpNe]
        rw [Real.log_div (mul_ne_zero hpNe hExpNe) hzNe]
        rw [Real.log_mul hpNe hExpNe, Real.log_exp]
        ring
      _ = (∑ i, q i * Real.log (q i / p i))
          - (∑ i, q i * payoff i) + Real.log z := by
        rw [Finset.sum_add_distrib, Finset.sum_sub_distrib]
        rw [← Finset.sum_mul, hqNorm, one_mul]
  rw [hExpansion] at hKL
  dsimp [z] at hKL
  linarith

theorem kl_ball_entropic_upper
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {q p payoff : ι → ℝ}
    {rho lambda : ℝ}
    (hq : ∀ i, 0 < q i)
    (hp : ∀ i, 0 < p i)
    (hqNorm : ∑ i, q i = 1)
    (hpNorm : ∑ i, p i = 1)
    (hKL : finiteTaskKL q p ≤ rho)
    (hLambda : 0 < lambda) :
    (∑ i, q i * payoff i) ≤
      lambda *
        (Real.log (∑ i, p i * Real.exp (payoff i / lambda)) + rho) := by
  have hChange := finite_task_change_of_measure
    (q := q)
    (p := p)
    (payoff := fun i => payoff i / lambda)
    hq hp hqNorm hpNorm
  have hScaled :
      (∑ i, q i * payoff i) / lambda =
        ∑ i, q i * (payoff i / lambda) := by
    rw [Finset.sum_div]
    apply Finset.sum_congr rfl
    intro i _hi
    ring
  rw [← hScaled] at hChange
  have hBound :
      (∑ i, q i * payoff i) / lambda ≤
        Real.log (∑ i, p i * Real.exp (payoff i / lambda)) + rho := by
    linarith
  simpa [mul_comm] using (div_le_iff₀ hLambda).mp hBound

theorem finite_pac_bayes_bound_on_moment_event
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {q p gap : ι → ℝ}
    {rho delta n : ℝ}
    (hq : ∀ i, 0 < q i)
    (hp : ∀ i, 0 < p i)
    (hqNorm : ∑ i, q i = 1)
    (hpNorm : ∑ i, p i = 1)
    (hKL : finiteTaskKL q p ≤ rho)
    (_hDelta : 0 < delta)
    (hN : 0 < n)
    (hMoment :
      (∑ i, p i * Real.exp (n * gap i)) ≤ 1 / delta) :
    (∑ i, q i * gap i) ≤
      (rho + Real.log (1 / delta)) / n := by
  have hChange := finite_task_change_of_measure
    (q := q)
    (p := p)
    (payoff := fun i => n * gap i)
    hq hp hqNorm hpNorm
  have hMomentPositive : 0 < ∑ i, p i * Real.exp (n * gap i) := by
    exact Finset.sum_pos
      (fun i _hi => mul_pos (hp i) (Real.exp_pos _))
      Finset.univ_nonempty
  have hLogMoment :
      Real.log (∑ i, p i * Real.exp (n * gap i))
        ≤ Real.log (1 / delta) :=
    Real.log_le_log hMomentPositive hMoment
  have hScaled :
      (∑ i, q i * (n * gap i)) = n * ∑ i, q i * gap i := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _hi
    ring
  rw [hScaled] at hChange
  have hNumerator :
      n * (∑ i, q i * gap i) ≤ rho + Real.log (1 / delta) := by
    linarith
  apply (le_div_iff₀ hN).mpr
  simpa [mul_comm, add_comm] using hNumerator

def InAmbiguitySet
    {ι : Type*}
    (admissible : (ι → ℝ) → Prop)
    (q : ι → ℝ) : Prop :=
  admissible q

def RobustEnvelope
    {ι : Type*} [Fintype ι]
    (admissible : (ι → ℝ) → Prop)
    (payoff : ι → ℝ)
    (upper : ℝ) : Prop :=
  ∀ q, InAmbiguitySet admissible q →
    (∀ i, 0 ≤ q i) →
    (∑ i, q i = 1) →
    ∑ i, q i * payoff i ≤ upper

theorem pointwise_upper_is_robust_envelope
    {ι : Type*} [Fintype ι]
    {admissible : (ι → ℝ) → Prop}
    {payoff : ι → ℝ}
    {upper : ℝ}
    (hUpper : ∀ i, payoff i ≤ upper) :
    RobustEnvelope admissible payoff upper := by
  intro q _hqSet hq hNorm
  exact weighted_expectation_le_envelope hq hNorm hUpper

theorem robust_certificate_holds_for_every_admissible_task_posterior
    {ι : Type*} [Fintype ι]
    {admissible : (ι → ℝ) → Prop}
    {meanPayoff epistemicPayoff aleatoricPayoff : ι → ℝ}
    {meanUpper epistemicUpper aleatoricUpper beta z tau : ℝ}
    (hMean : RobustEnvelope admissible meanPayoff meanUpper)
    (hEpistemic :
      RobustEnvelope admissible epistemicPayoff epistemicUpper)
    (hAleatoric :
      RobustEnvelope admissible aleatoricPayoff aleatoricUpper)
    (hCertificate :
      meanUpper + Real.sqrt beta * Real.sqrt epistemicUpper
        + z * Real.sqrt aleatoricUpper ≤ tau) :
    ∀ q, InAmbiguitySet admissible q →
      (∀ i, 0 ≤ q i) →
      (∑ i, q i = 1) →
      (∑ i, q i * meanPayoff i) ≤ meanUpper ∧
      (∑ i, q i * epistemicPayoff i) ≤ epistemicUpper ∧
      (∑ i, q i * aleatoricPayoff i) ≤ aleatoricUpper ∧
      meanUpper + Real.sqrt beta * Real.sqrt epistemicUpper
        + z * Real.sqrt aleatoricUpper ≤ tau := by
  intro q hqSet hq hNorm
  exact ⟨
    hMean q hqSet hq hNorm,
    hEpistemic q hqSet hq hNorm,
    hAleatoric q hqSet hq hNorm,
    hCertificate,
  ⟩

structure JointTaskBelief (Expert ModelState : Type*) where
  taskWeight : Expert → ℝ
  objectiveState : Expert → ModelState
  constraintState : Expert → ModelState
  varianceState : Expert → ModelState

def jointTaskExactGain
    {Expert ModelState Design Observation : Type*}
    (terminal : JointTaskBelief Expert ModelState → ℝ)
    (update : JointTaskBelief Expert ModelState →
      Design → Observation → JointTaskBelief Expert ModelState)
    (belief : JointTaskBelief Expert ModelState)
    (x : Design)
    (y : Observation) : ℝ :=
  terminal belief - terminal (update belief x y)

theorem joint_task_exact_mc_zero_error_is_one_step_optimal
    {Design : Type*}
    {exact estimate : Design → ℝ}
    {x : Design}
    (hEstimator : ExactMCEstimator exact estimate 0)
    (hMax : ∀ y, estimate y ≤ estimate x) :
    KGMaximizer { expectedTerminalGain := exact } x := by
  exact exact_mc_zero_error_recovers_exact_maximizer hEstimator hMax

end SCOLHKG.Real

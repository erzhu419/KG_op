import Mathlib
import SCOLHKG.Real.AdditiveApproxKG
import SCOLHKG.Real.KG

namespace SCOLHKG.Real

open scoped BigOperators

/-!
Implementation bridge for `acquisition_mode="exact_mc"`.

The Python path estimates exact posterior-update KG by cloning the GPR/HVD
state, simulating candidate observations, recomputing the certified terminal
value, and averaging the resulting gains.  The probabilistic concentration of
that estimator lives in the measure-theoretic files; this file records the
deterministic implication used by optimization: a uniformly accurate estimator
inherits exact-KG optimality up to `2 * eta`.
-/

noncomputable def finiteSampleAverage
    {Sample Design : Type*}
    (samples : Finset Sample)
    (gain : Sample → Design → ℝ)
    (x : Design) : ℝ :=
  (samples.card : ℝ)⁻¹ * ∑ s ∈ samples, gain s x

def ExactMCEstimator
    {Design : Type*}
    (exact estimate : Design → ℝ)
    (eta : ℝ) : Prop :=
  UniformKGApprox exact estimate eta

theorem exact_mc_estimator_maximizer_gap
    {Design : Type*}
    {exact estimate : Design → ℝ}
    {eta : ℝ}
    {x : Design}
    (hEstimator : ExactMCEstimator exact estimate eta)
    (hMax : ∀ y, estimate y ≤ estimate x) :
    ∀ y, exact y ≤ exact x + 2 * eta := by
  exact proxy_maximizer_exact_gap_le_two_eta
    (exact := exact)
    (proxy := estimate)
    (eta := eta)
    (x := x)
    hEstimator
    hMax

theorem exact_mc_zero_error_recovers_exact_maximizer
    {Design : Type*}
    {exact estimate : Design → ℝ}
    {x : Design}
    (hEstimator : ExactMCEstimator exact estimate 0)
    (hMax : ∀ y, estimate y ≤ estimate x) :
    KGMaximizer { expectedTerminalGain := exact } x := by
  intro y
  have hgap := exact_mc_estimator_maximizer_gap
    (exact := exact)
    (estimate := estimate)
    (eta := 0)
    (x := x)
    hEstimator
    hMax
    y
  linarith

theorem finiteSampleAverage_empty_is_zero
    {Sample Design : Type*}
    (gain : Sample → Design → ℝ)
    (x : Design) :
    finiteSampleAverage (∅ : Finset Sample) gain x = 0 := by
  simp [finiteSampleAverage]

end SCOLHKG.Real

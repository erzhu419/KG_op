import Mathlib

namespace SCOLHKG.Real

/-!
Exact SC-OLH-KG value theorem, at the deterministic expectation layer.

The exact acquisition is represented as the expected terminal value gain.  The
proof says a maximizer of that exact value dominates every other one-step
sampling decision.  A separate theorem records when an additive proxy is exact.
-/

universe u

structure ExactKG (Design : Type u) where
  expectedTerminalGain : Design → ℝ

def KGMaximizer {Design : Type u} (kg : ExactKG Design) (x : Design) : Prop :=
  ∀ y, kg.expectedTerminalGain y ≤ kg.expectedTerminalGain x

theorem exact_kg_maximizer_optimal
    {Design : Type u}
    (kg : ExactKG Design)
    (x y : Design)
    (hMax : KGMaximizer kg x) :
    kg.expectedTerminalGain y ≤ kg.expectedTerminalGain x := by
  exact hMax y

structure AdditiveKG (Design : Type u) where
  exact : Design → ℝ
  objective : Design → ℝ
  feasibility : Design → ℝ
  variance : Design → ℝ
  coupling : Design → ℝ

def additiveScore {Design : Type u} (kg : AdditiveKG Design) (x : Design) : ℝ :=
  kg.objective x + kg.feasibility x + kg.variance x + kg.coupling x

def AdditiveMatchesExact {Design : Type u} (kg : AdditiveKG Design) : Prop :=
  ∀ x, kg.exact x = additiveScore kg x

theorem additive_maximizer_is_exact_maximizer
    {Design : Type u}
    (kg : AdditiveKG Design)
    (x : Design)
    (hMatch : AdditiveMatchesExact kg)
    (hMax : ∀ y, additiveScore kg y ≤ additiveScore kg x) :
    ∀ y, kg.exact y ≤ kg.exact x := by
  intro y
  rw [hMatch y, hMatch x]
  exact hMax y

end SCOLHKG.Real


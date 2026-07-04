namespace SCOLHKG

/-!
Exact KG and additive-surrogate bookkeeping.

The code currently uses an additive acquisition proxy.  This file makes the
gap explicit: additive scoring is exact only when a decomposition theorem
identifies the exact one-step KG with the sum of the chosen components.
-/

universe u

structure ExactKG (Design : Type u) where
  expectedTerminalGain : Design → Nat

def KGMaximizer
    {Design : Type u}
    (kg : ExactKG Design)
    (x : Design) : Prop :=
  ∀ y, kg.expectedTerminalGain y ≤ kg.expectedTerminalGain x

theorem kgMaximizer_dominates
    {Design : Type u}
    (kg : ExactKG Design)
    (x y : Design)
    (h : KGMaximizer kg x) :
    kg.expectedTerminalGain y ≤ kg.expectedTerminalGain x := by
  exact h y

structure KGDecomposition (Design : Type u) where
  exact : Design → Nat
  objective : Design → Nat
  feasibility : Design → Nat
  variance : Design → Nat
  coupling : Design → Nat

def additiveScore
    {Design : Type u}
    (d : KGDecomposition Design)
    (x : Design) : Nat :=
  d.objective x + d.feasibility x + d.variance x + d.coupling x

def AdditiveMatchesExact
    {Design : Type u}
    (d : KGDecomposition Design) : Prop :=
  ∀ x, d.exact x = additiveScore d x

theorem additiveScore_exact_when_matches
    {Design : Type u}
    (d : KGDecomposition Design)
    (h : AdditiveMatchesExact d)
    (x : Design) :
    d.exact x = additiveScore d x := by
  exact h x

theorem additiveMaximizer_is_exactMaximizer
    {Design : Type u}
    (d : KGDecomposition Design)
    (x : Design)
    (hMatch : AdditiveMatchesExact d)
    (hMax : ∀ y, additiveScore d y ≤ additiveScore d x) :
    ∀ y, d.exact y ≤ d.exact x := by
  intro y
  rw [hMatch y, hMatch x]
  exact hMax y

end SCOLHKG


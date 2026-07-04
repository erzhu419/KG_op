namespace SCOLHKG

/-!
Scalarization lemmas inherited from the original bi-objective GPR-KG path.

The current paper direction is single-objective chance constrained, but the
bridge to the original bi-objective manuscript relies on weighted
scalarization.  This file proves the core order property: weak Pareto
dominance is preserved by nonnegative weighted sums.
-/

structure BiObjectivePoint where
  f1 : Nat
  f2 : Nat
deriving Repr, BEq

def scalar2 (w1 w2 f1 f2 : Nat) : Nat :=
  w1 * f1 + w2 * f2

def scalarized (w1 w2 : Nat) (p : BiObjectivePoint) : Nat :=
  scalar2 w1 w2 p.f1 p.f2

def weaklyDominates (p q : BiObjectivePoint) : Prop :=
  p.f1 ≤ q.f1 ∧ p.f2 ≤ q.f2

theorem scalar2_monotone
    {w1 w2 a1 a2 b1 b2 : Nat}
    (h1 : a1 ≤ b1)
    (h2 : a2 ≤ b2) :
    scalar2 w1 w2 a1 a2 ≤ scalar2 w1 w2 b1 b2 := by
  unfold scalar2
  exact Nat.add_le_add
    (Nat.mul_le_mul_left w1 h1)
    (Nat.mul_le_mul_left w2 h2)

theorem weakDominance_scalarized_le
    {w1 w2 : Nat}
    {p q : BiObjectivePoint}
    (h : weaklyDominates p q) :
    scalarized w1 w2 p ≤ scalarized w1 w2 q := by
  unfold scalarized
  exact scalar2_monotone h.left h.right

theorem scalarized_equal_when_objectives_equal
    {w1 w2 : Nat}
    {p q : BiObjectivePoint}
    (h1 : p.f1 = q.f1)
    (h2 : p.f2 = q.f2) :
    scalarized w1 w2 p = scalarized w1 w2 q := by
  unfold scalarized scalar2
  rw [h1, h2]

end SCOLHKG


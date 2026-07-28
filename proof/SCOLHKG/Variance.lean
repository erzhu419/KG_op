namespace SCOLHKG

/-!
Lean core formalization of the cumulative-risk algebra used by SC-OLH-KG.

The file deliberately uses `Nat` rather than real-valued probability objects so
that the first proof layer has no external dependency.  It proves the exact
algebraic skeleton behind

  A^T Lambda A + N^T B N + N^T omega + floor.

The later measure-theoretic version can replace `Nat` with `ℝ` after mathlib
probability dependencies are introduced.
-/

abbrev Exposure := List Nat
abbrev RiskMatrix := List (List Nat)

def dot : Exposure → Exposure → Nat
  | [], _ => 0
  | _, [] => 0
  | a :: as, b :: bs => a * b + dot as bs

def squares (xs : Exposure) : Exposure :=
  xs.map (fun x => x * x)

def diagQuad (a lambda : Exposure) : Nat :=
  dot (squares a) lambda

def sharedShock : Exposure → RiskMatrix → Nat
  | [], _ => 0
  | _, [] => 0
  | n :: ns, row :: rows => n * dot row (n :: ns) + sharedShock ns rows

structure CumulativeRisk where
  A : Exposure
  N : Exposure
  Lambda : Exposure
  B : RiskMatrix
  omega : Exposure
  floor : Nat
deriving Repr, BEq

def independentRisk (r : CumulativeRisk) : Nat :=
  diagQuad r.A r.Lambda

def sharedShockRisk (r : CumulativeRisk) : Nat :=
  sharedShock r.N r.B

def linearRisk (r : CumulativeRisk) : Nat :=
  dot r.N r.omega

def cumulativeVarianceNoFloor (r : CumulativeRisk) : Nat :=
  independentRisk r + sharedShockRisk r + linearRisk r

def totalVariance (r : CumulativeRisk) : Nat :=
  r.floor + cumulativeVarianceNoFloor r

@[simp]
theorem fixedTrajectoryVarianceDecomposition (r : CumulativeRisk) :
    totalVariance r =
      r.floor + independentRisk r + sharedShockRisk r + linearRisk r := by
  unfold totalVariance cumulativeVarianceNoFloor
  simp [Nat.add_assoc]

@[simp]
theorem cumulativeVarianceWithFloor (r : CumulativeRisk) :
    totalVariance r = r.floor + cumulativeVarianceNoFloor r := by
  rfl

theorem independentRisk_nonnegative (r : CumulativeRisk) :
    0 ≤ independentRisk r := by
  exact Nat.zero_le _

theorem sharedShockRisk_nonnegative (r : CumulativeRisk) :
    0 ≤ sharedShockRisk r := by
  exact Nat.zero_le _

theorem linearRisk_nonnegative (r : CumulativeRisk) :
    0 ≤ linearRisk r := by
  exact Nat.zero_le _

theorem cumulativeVarianceNoFloor_le_totalVariance (r : CumulativeRisk) :
    cumulativeVarianceNoFloor r ≤ totalVariance r := by
  unfold totalVariance
  exact Nat.le_add_left _ _

structure LowRankTruncation where
  fullSharedRisk : Nat
  keptSharedRisk : Nat
  tailSharedRisk : Nat
deriving Repr, BEq

namespace LowRankTruncation

def Valid (t : LowRankTruncation) : Prop :=
  t.fullSharedRisk = t.keptSharedRisk + t.tailSharedRisk

theorem kept_le_full (t : LowRankTruncation) (h : t.Valid) :
    t.keptSharedRisk ≤ t.fullSharedRisk := by
  rw [h]
  exact Nat.le_add_right _ _

theorem tail_le_full (t : LowRankTruncation) (h : t.Valid) :
    t.tailSharedRisk ≤ t.fullSharedRisk := by
  rw [h]
  exact Nat.le_add_left _ _

theorem full_eq_kept_plus_tail (t : LowRankTruncation) (h : t.Valid) :
    t.fullSharedRisk = t.keptSharedRisk + t.tailSharedRisk := by
  exact h

end LowRankTruncation

end SCOLHKG

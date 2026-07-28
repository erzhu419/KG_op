import Mathlib

namespace SCOLHKG.Real

/-!
Real-valued cumulative-risk algebra.

This upgrades the first Lean layer from `Nat` bookkeeping to `ℝ`, which is the
right ambient type for variances, Gaussian quantiles, and chance constraints.
-/

abbrev Exposure := List ℝ
abbrev RiskMatrix := List (List ℝ)

def dot : Exposure → Exposure → ℝ
  | [], _ => 0
  | _, [] => 0
  | a :: as, b :: bs => a * b + dot as bs

def squares (xs : Exposure) : Exposure :=
  xs.map (fun x => x ^ 2)

def diagQuad (a lambda : Exposure) : ℝ :=
  dot (squares a) lambda

def sharedShock : Exposure → RiskMatrix → ℝ
  | [], _ => 0
  | _, [] => 0
  | n :: ns, row :: rows => n * dot row (n :: ns) + sharedShock ns rows

structure CumulativeRisk where
  A : Exposure
  N : Exposure
  Lambda : Exposure
  B : RiskMatrix
  omega : Exposure
  floor : ℝ

def independentRisk (r : CumulativeRisk) : ℝ :=
  diagQuad r.A r.Lambda

def sharedShockRisk (r : CumulativeRisk) : ℝ :=
  sharedShock r.N r.B

def linearRisk (r : CumulativeRisk) : ℝ :=
  dot r.N r.omega

def cumulativeVarianceNoFloor (r : CumulativeRisk) : ℝ :=
  independentRisk r + sharedShockRisk r + linearRisk r

def totalVariance (r : CumulativeRisk) : ℝ :=
  r.floor + cumulativeVarianceNoFloor r

@[simp]
theorem fixedTrajectoryVarianceDecomposition (r : CumulativeRisk) :
    totalVariance r =
      r.floor + independentRisk r + sharedShockRisk r + linearRisk r := by
  unfold totalVariance cumulativeVarianceNoFloor
  ring

theorem cumulativeVarianceNoFloor_le_totalVariance
    (r : CumulativeRisk)
    (hfloor : 0 ≤ r.floor) :
    cumulativeVarianceNoFloor r ≤ totalVariance r := by
  unfold totalVariance
  linarith

structure LowRankTruncation where
  fullSharedRisk : ℝ
  keptSharedRisk : ℝ
  tailSharedRisk : ℝ

namespace LowRankTruncation

def Valid (t : LowRankTruncation) : Prop :=
  t.fullSharedRisk = t.keptSharedRisk + t.tailSharedRisk

theorem kept_le_full
    (t : LowRankTruncation)
    (h : t.Valid)
    (htail : 0 ≤ t.tailSharedRisk) :
    t.keptSharedRisk ≤ t.fullSharedRisk := by
  unfold Valid at h
  rw [h]
  linarith

theorem tail_le_full
    (t : LowRankTruncation)
    (h : t.Valid)
    (hkept : 0 ≤ t.keptSharedRisk) :
    t.tailSharedRisk ≤ t.fullSharedRisk := by
  unfold Valid at h
  rw [h]
  linarith

theorem truncation_error_nonnegative
    (t : LowRankTruncation)
    (htail : 0 ≤ t.tailSharedRisk) :
    0 ≤ t.tailSharedRisk := by
  exact htail

end LowRankTruncation

end SCOLHKG.Real


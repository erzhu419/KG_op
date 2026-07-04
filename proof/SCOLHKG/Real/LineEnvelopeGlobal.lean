import SCOLHKG.Real.LineEnvelopeAlgorithm

namespace SCOLHKG.Real

/-!
Global stack invariant for the line-envelope algorithm.

`LineEnvelopeStack.lean` proves that endpoint/tail checks imply a pointwise
certificate, and `LineEnvelopeAlgorithm.lean` proves the low-level pop/push
stack order invariants.  This file removes the Python runtime validator from
the theorem path: if the final stack carries the standard global dominance
invariant for all processed/original lines, then every atom is certified and
the `compute_h` envelope sum is exact.

The remaining implementation-level work after this file is to prove that the
particular intersection arithmetic in the imperative while-loop establishes
`FinalEnvelopeStackInvariant`.  Once that invariant is available, no runtime
validator assumption is needed.
-/

universe u v

variable {Line : Type u} {Atom : Type v}

structure FiniteEnvelopeCell
  (sys : LineSystem Line)
  (lines : Finset Line) where
  activeLine : Line
  leftCut : ℝ
  rightCut : ℝ
  active_mem : activeLine ∈ lines
  lo_le_hi : leftCut ≤ rightCut
  dominates_lo :
    ∀ line ∈ lines,
      lineValue sys line leftCut ≤ lineValue sys activeLine leftCut
  dominates_hi :
    ∀ line ∈ lines,
      lineValue sys line rightCut ≤ lineValue sys activeLine rightCut

structure LeftTailEnvelopeCell
    (sys : LineSystem Line)
    (lines : Finset Line) where
  activeLine : Line
  rightCut : ℝ
  active_mem : activeLine ∈ lines
  left_slope_min :
    ∀ line ∈ lines, sys.slope activeLine ≤ sys.slope line
  dominates_hi :
    ∀ line ∈ lines,
      lineValue sys line rightCut ≤ lineValue sys activeLine rightCut

structure RightTailEnvelopeCell
    (sys : LineSystem Line)
    (lines : Finset Line) where
  activeLine : Line
  leftCut : ℝ
  active_mem : activeLine ∈ lines
  right_slope_max :
    ∀ line ∈ lines, sys.slope line ≤ sys.slope activeLine
  dominates_lo :
    ∀ line ∈ lines,
      lineValue sys line leftCut ≤ lineValue sys activeLine leftCut

def FiniteEnvelopeCell.activeValue
    {sys : LineSystem Line}
    {lines : Finset Line}
    (cell : FiniteEnvelopeCell sys lines)
    (z : ℝ) : ℝ :=
  lineValue sys cell.activeLine z

def LeftTailEnvelopeCell.activeValue
    {sys : LineSystem Line}
    {lines : Finset Line}
    (cell : LeftTailEnvelopeCell sys lines)
    (z : ℝ) : ℝ :=
  lineValue sys cell.activeLine z

def RightTailEnvelopeCell.activeValue
    {sys : LineSystem Line}
    {lines : Finset Line}
    (cell : RightTailEnvelopeCell sys lines)
    (z : ℝ) : ℝ :=
  lineValue sys cell.activeLine z

theorem finiteEnvelopeCell_atom_certificate
    {sys : LineSystem Line}
    {lines : Finset Line}
    (cell : FiniteEnvelopeCell sys lines)
    {z : ℝ}
    (hzlo : cell.leftCut ≤ z)
    (hzhi : z ≤ cell.rightCut) :
    EnvelopeAtomCertificate sys lines z (cell.activeValue z) := by
  unfold FiniteEnvelopeCell.activeValue
  exact finite_interval_stack_atom_certificate
    sys
    lines
    cell.activeLine
    cell.active_mem
    hzlo
    hzhi
    cell.dominates_lo
    cell.dominates_hi

theorem leftTailEnvelopeCell_atom_certificate
    {sys : LineSystem Line}
    {lines : Finset Line}
    (cell : LeftTailEnvelopeCell sys lines)
    {z : ℝ}
    (hzhi : z ≤ cell.rightCut) :
    EnvelopeAtomCertificate sys lines z (cell.activeValue z) := by
  unfold LeftTailEnvelopeCell.activeValue
  exact left_tail_stack_atom_certificate
    sys
    lines
    cell.activeLine
    cell.active_mem
    hzhi
    cell.left_slope_min
    cell.dominates_hi

theorem rightTailEnvelopeCell_atom_certificate
    {sys : LineSystem Line}
    {lines : Finset Line}
    (cell : RightTailEnvelopeCell sys lines)
    {z : ℝ}
    (hzlo : cell.leftCut ≤ z) :
    EnvelopeAtomCertificate sys lines z (cell.activeValue z) := by
  unfold RightTailEnvelopeCell.activeValue
  exact right_tail_stack_atom_certificate
    sys
    lines
    cell.activeLine
    cell.active_mem
    hzlo
    cell.right_slope_max
    cell.dominates_lo

structure FinalEnvelopeStackInvariant
    (sys : LineSystem Line)
    (lines : Finset Line)
    (atoms : Finset Atom)
    (active : Atom → Line)
    (z lo hi : Atom → ℝ) : Prop where
  active_mem : ∀ atom ∈ atoms, active atom ∈ lines
  interval_contains : ∀ atom ∈ atoms, lo atom ≤ z atom ∧ z atom ≤ hi atom
  dominates_lo :
    ∀ atom ∈ atoms, ∀ line ∈ lines,
      lineValue sys line (lo atom) ≤ lineValue sys (active atom) (lo atom)
  dominates_hi :
    ∀ atom ∈ atoms, ∀ line ∈ lines,
      lineValue sys line (hi atom) ≤ lineValue sys (active atom) (hi atom)

theorem finalEnvelopeStackInvariant_atom_certificate
    {sys : LineSystem Line}
    {lines : Finset Line}
    {atoms : Finset Atom}
    {active : Atom → Line}
    {z lo hi : Atom → ℝ}
    (h : FinalEnvelopeStackInvariant sys lines atoms active z lo hi) :
    ∀ atom ∈ atoms,
      EnvelopeAtomCertificate sys lines (z atom)
        (lineValue sys (active atom) (z atom)) := by
  intro atom hatom
  let cell : FiniteEnvelopeCell sys lines :=
    { activeLine := active atom
      leftCut := lo atom
      rightCut := hi atom
      active_mem := h.active_mem atom hatom
      lo_le_hi := (h.interval_contains atom hatom).1.trans
        (h.interval_contains atom hatom).2
      dominates_lo := h.dominates_lo atom hatom
      dominates_hi := h.dominates_hi atom hatom }
  exact finiteEnvelopeCell_atom_certificate
    cell
    (h.interval_contains atom hatom).1
    (h.interval_contains atom hatom).2

theorem finalEnvelopeStackInvariant_lineEnvelopeKG_exact
    {sys : LineSystem Line}
    {lines : Finset Line}
    {atoms : Finset Atom}
    {probMass firstMoment z lo hi : Atom → ℝ}
    {active : Atom → Line}
    {baseline expectedMax : ℝ}
    (hStack :
      FinalEnvelopeStackInvariant sys lines atoms active z lo hi)
    (hMoment : ∀ atom ∈ atoms, firstMoment atom = probMass atom * z atom)
    (hExpected :
      expectedMax =
        finiteLineEnvelopeMaxExpectation atoms probMass
          (fun atom ↦ lineValue sys (active atom) (z atom))) :
    lineEnvelopeKGFormula sys atoms probMass firstMoment active baseline =
      expectedMax - baseline := by
  exact certified_lineEnvelopeKG_exact
    sys
    lines
    atoms
    probMass
    firstMoment
    z
    active
    baseline
    expectedMax
    hMoment
    (finalEnvelopeStackInvariant_atom_certificate hStack)
    hExpected

theorem finalEnvelopeStackInvariant_extend_lines
    [DecidableEq Line]
    {sys : LineSystem Line}
    {lines : Finset Line}
    {newLine : Line}
    {atoms : Finset Atom}
    {active : Atom → Line}
    {z lo hi : Atom → ℝ}
    (h :
      FinalEnvelopeStackInvariant sys lines atoms active z lo hi)
    (hNewLo :
      ∀ atom ∈ atoms,
        lineValue sys newLine (lo atom) ≤
          lineValue sys (active atom) (lo atom))
    (hNewHi :
      ∀ atom ∈ atoms,
        lineValue sys newLine (hi atom) ≤
          lineValue sys (active atom) (hi atom)) :
    FinalEnvelopeStackInvariant sys (insert newLine lines) atoms active z lo hi := by
  refine ⟨?_, h.interval_contains, ?_, ?_⟩
  · intro atom hatom
    exact Finset.mem_insert_of_mem (h.active_mem atom hatom)
  · intro atom hatom line hline
    rw [Finset.mem_insert] at hline
    rcases hline with hEq | hOld
    · subst line
      exact hNewLo atom hatom
    · exact h.dominates_lo atom hatom line hOld
  · intro atom hatom line hline
    rw [Finset.mem_insert] at hline
    rcases hline with hEq | hOld
    · subst line
      exact hNewHi atom hatom
    · exact h.dominates_hi atom hatom line hOld

end SCOLHKG.Real

import SCOLHKG.Real.LineEnvelopeGlobal

namespace SCOLHKG.Real

/-!
Concrete intersection arithmetic used by `core/kg.py::compute_h`.

For two active affine lines with strictly increasing slopes, the Python code
computes

```text
z = (a_old - a_new) / (b_new - b_old)
```

This file proves the exact arithmetic facts that justify the while-loop:
to the left of this intersection the old line dominates the new line, and to
the right the new line dominates the old line.
-/

universe u

variable {Line : Type u}

noncomputable def lineIntersection
    (sys : LineSystem Line)
    (oldLine newLine : Line) : ℝ :=
  (sys.intercept oldLine - sys.intercept newLine)
    / (sys.slope newLine - sys.slope oldLine)

theorem lineValue_difference_eq_slopeDiff_mul_intersection_offset
    (sys : LineSystem Line)
    (oldLine newLine : Line)
    {z : ℝ}
    (hSlopeNe : sys.slope newLine - sys.slope oldLine ≠ 0) :
    lineValue sys newLine z - lineValue sys oldLine z =
      (sys.slope newLine - sys.slope oldLine)
        * (z - lineIntersection sys oldLine newLine) := by
  unfold lineValue lineIntersection
  field_simp [hSlopeNe]
  ring

theorem oldLine_dominates_left_of_intersection
    (sys : LineSystem Line)
    (oldLine newLine : Line)
    {z : ℝ}
    (hSlope : sys.slope oldLine < sys.slope newLine)
    (hz : z ≤ lineIntersection sys oldLine newLine) :
    lineValue sys newLine z ≤ lineValue sys oldLine z := by
  have hSlopePos : 0 < sys.slope newLine - sys.slope oldLine :=
    sub_pos.mpr hSlope
  have hSlopeNe : sys.slope newLine - sys.slope oldLine ≠ 0 :=
    hSlopePos.ne'
  have hOffset : z - lineIntersection sys oldLine newLine ≤ 0 :=
    sub_nonpos.mpr hz
  have hProd :
      (sys.slope newLine - sys.slope oldLine)
        * (z - lineIntersection sys oldLine newLine) ≤ 0 := by
    exact mul_nonpos_of_nonneg_of_nonpos hSlopePos.le hOffset
  have hDiff :
      lineValue sys newLine z - lineValue sys oldLine z ≤ 0 := by
    rw [lineValue_difference_eq_slopeDiff_mul_intersection_offset
      (sys := sys)
      (oldLine := oldLine)
      (newLine := newLine)
      (z := z)
      hSlopeNe]
    exact hProd
  exact sub_nonpos.mp hDiff

theorem newLine_dominates_right_of_intersection
    (sys : LineSystem Line)
    (oldLine newLine : Line)
    {z : ℝ}
    (hSlope : sys.slope oldLine < sys.slope newLine)
    (hz : lineIntersection sys oldLine newLine ≤ z) :
    lineValue sys oldLine z ≤ lineValue sys newLine z := by
  have hSlopePos : 0 < sys.slope newLine - sys.slope oldLine :=
    sub_pos.mpr hSlope
  have hSlopeNe : sys.slope newLine - sys.slope oldLine ≠ 0 :=
    hSlopePos.ne'
  have hOffset : 0 ≤ z - lineIntersection sys oldLine newLine :=
    sub_nonneg.mpr hz
  have hProd :
      0 ≤
        (sys.slope newLine - sys.slope oldLine)
          * (z - lineIntersection sys oldLine newLine) := by
    exact mul_nonneg hSlopePos.le hOffset
  have hDiff :
      0 ≤ lineValue sys newLine z - lineValue sys oldLine z := by
    rw [lineValue_difference_eq_slopeDiff_mul_intersection_offset
      (sys := sys)
      (oldLine := oldLine)
      (newLine := newLine)
      (z := z)
      hSlopeNe]
    exact hProd
  exact sub_nonneg.mp hDiff

theorem oldLine_eq_newLine_at_intersection
    (sys : LineSystem Line)
    (oldLine newLine : Line)
    (hSlope : sys.slope oldLine < sys.slope newLine) :
    lineValue sys oldLine (lineIntersection sys oldLine newLine) =
      lineValue sys newLine (lineIntersection sys oldLine newLine) := by
  have hLeft := oldLine_dominates_left_of_intersection
    sys oldLine newLine hSlope (le_rfl)
  have hRight := newLine_dominates_right_of_intersection
    sys oldLine newLine hSlope (le_rfl)
  exact le_antisymm hRight hLeft

theorem newLine_dominates_popped_finite_cell_endpoints
    {sys : LineSystem Line}
    {lines : Finset Line}
    (cell : FiniteEnvelopeCell sys lines)
    {newLine : Line}
    (hSlope : sys.slope cell.activeLine < sys.slope newLine)
    (hCut : lineIntersection sys cell.activeLine newLine ≤ cell.leftCut) :
    lineValue sys cell.activeLine cell.leftCut ≤
        lineValue sys newLine cell.leftCut
      ∧
      lineValue sys cell.activeLine cell.rightCut ≤
        lineValue sys newLine cell.rightCut := by
  constructor
  · exact newLine_dominates_right_of_intersection
      sys cell.activeLine newLine hSlope hCut
  · exact newLine_dominates_right_of_intersection
      sys cell.activeLine newLine hSlope (hCut.trans cell.lo_le_hi)

theorem popped_finite_cell_dominance_transfers_to_newLine
    {sys : LineSystem Line}
    {lines : Finset Line}
    (cell : FiniteEnvelopeCell sys lines)
    {newLine : Line}
    (hSlope : sys.slope cell.activeLine < sys.slope newLine)
    (hCut : lineIntersection sys cell.activeLine newLine ≤ cell.leftCut) :
    (∀ line ∈ lines,
      lineValue sys line cell.leftCut ≤ lineValue sys newLine cell.leftCut)
      ∧
    (∀ line ∈ lines,
      lineValue sys line cell.rightCut ≤ lineValue sys newLine cell.rightCut) := by
  have hNew := newLine_dominates_popped_finite_cell_endpoints
    cell hSlope hCut
  constructor
  · intro line hline
    exact (cell.dominates_lo line hline).trans hNew.1
  · intro line hline
    exact (cell.dominates_hi line hline).trans hNew.2

theorem oldLine_dominates_newLine_on_truncated_left_endpoint
    {sys : LineSystem Line}
    {oldLine newLine : Line}
    {leftCut : ℝ}
    (hSlope : sys.slope oldLine < sys.slope newLine)
    (hLeft : leftCut ≤ lineIntersection sys oldLine newLine) :
    lineValue sys newLine leftCut ≤ lineValue sys oldLine leftCut := by
  exact oldLine_dominates_left_of_intersection
    sys oldLine newLine hSlope hLeft

theorem split_right_tail_old_cell_dominates_new_left_piece
    [DecidableEq Line]
    {sys : LineSystem Line}
    {lines : Finset Line}
    (cell : RightTailEnvelopeCell sys lines)
    {newLine : Line}
    (hSlope : sys.slope cell.activeLine < sys.slope newLine)
    (hCut : cell.leftCut ≤ lineIntersection sys cell.activeLine newLine) :
    ∀ line ∈ insert newLine lines,
      lineValue sys line cell.leftCut ≤ lineValue sys cell.activeLine cell.leftCut := by
  intro line hline
  rw [Finset.mem_insert] at hline
  rcases hline with hEq | hOld
  · subst line
    exact oldLine_dominates_newLine_on_truncated_left_endpoint
      hSlope hCut
  · exact cell.dominates_lo line hOld

theorem right_tail_old_active_dominates_processed_at_intersection
    {sys : LineSystem Line}
    {lines : Finset Line}
    (cell : RightTailEnvelopeCell sys lines)
    {newLine : Line}
    (_hSlope : sys.slope cell.activeLine < sys.slope newLine)
    (hCut : cell.leftCut ≤ lineIntersection sys cell.activeLine newLine) :
    ∀ line ∈ lines,
      lineValue sys line (lineIntersection sys cell.activeLine newLine) ≤
        lineValue sys cell.activeLine
          (lineIntersection sys cell.activeLine newLine) := by
  intro line hline
  exact right_tail_endpoint_slope_dominance
    sys
    line
    cell.activeLine
    hCut
    (cell.right_slope_max line hline)
    (cell.dominates_lo line hline)

theorem split_right_tail_newLine_left_endpoint_dominates_processed
    {sys : LineSystem Line}
    {lines : Finset Line}
    (cell : RightTailEnvelopeCell sys lines)
    {newLine : Line}
    (hSlope : sys.slope cell.activeLine < sys.slope newLine)
    (hCut : cell.leftCut ≤ lineIntersection sys cell.activeLine newLine) :
    ∀ line ∈ lines,
      lineValue sys line (lineIntersection sys cell.activeLine newLine) ≤
        lineValue sys newLine
          (lineIntersection sys cell.activeLine newLine) := by
  intro line hline
  have hOld := right_tail_old_active_dominates_processed_at_intersection
    cell hSlope hCut line hline
  have hEq := oldLine_eq_newLine_at_intersection
    sys cell.activeLine newLine hSlope
  exact hOld.trans hEq.le

theorem split_right_tail_newLine_is_right_slope_max
    [DecidableEq Line]
    {sys : LineSystem Line}
    {lines : Finset Line}
    (cell : RightTailEnvelopeCell sys lines)
    {newLine : Line}
    (hSlope : sys.slope cell.activeLine < sys.slope newLine) :
    ∀ line ∈ insert newLine lines, sys.slope line ≤ sys.slope newLine := by
  intro line hline
  rw [Finset.mem_insert] at hline
  rcases hline with hEq | hOld
  · subst line
    exact le_rfl
  · exact (cell.right_slope_max line hOld).trans hSlope.le

theorem split_right_tail_newLine_dominates_at_new_left_cut
    [DecidableEq Line]
    {sys : LineSystem Line}
    {lines : Finset Line}
    (cell : RightTailEnvelopeCell sys lines)
    {newLine : Line}
    (hSlope : sys.slope cell.activeLine < sys.slope newLine)
    (hCut : cell.leftCut ≤ lineIntersection sys cell.activeLine newLine) :
    ∀ line ∈ insert newLine lines,
      lineValue sys line (lineIntersection sys cell.activeLine newLine) ≤
        lineValue sys newLine
          (lineIntersection sys cell.activeLine newLine) := by
  intro line hline
  rw [Finset.mem_insert] at hline
  rcases hline with hEq | hOld
  · subst line
    exact le_rfl
  · exact split_right_tail_newLine_left_endpoint_dominates_processed
      cell hSlope hCut line hOld

def popped_finite_cell_as_newLine_cell
    [DecidableEq Line]
    {sys : LineSystem Line}
    {lines : Finset Line}
    (cell : FiniteEnvelopeCell sys lines)
    {newLine : Line}
    (hSlope : sys.slope cell.activeLine < sys.slope newLine)
    (hCut : lineIntersection sys cell.activeLine newLine ≤ cell.leftCut) :
    FiniteEnvelopeCell sys (insert newLine lines) := by
  have hTransfer :=
    popped_finite_cell_dominance_transfers_to_newLine
      cell hSlope hCut
  refine
    { activeLine := newLine
      leftCut := cell.leftCut
      rightCut := cell.rightCut
      active_mem := Finset.mem_insert_self newLine lines
      lo_le_hi := cell.lo_le_hi
      dominates_lo := ?_
      dominates_hi := ?_ }
  · intro line hline
    rw [Finset.mem_insert] at hline
    rcases hline with hEq | hOld
    · subst line
      exact le_rfl
    · exact hTransfer.1 line hOld
  · intro line hline
    rw [Finset.mem_insert] at hline
    rcases hline with hEq | hOld
    · subst line
      exact le_rfl
    · exact hTransfer.2 line hOld

theorem popped_finite_cell_newLine_atom_certificate
    [DecidableEq Line]
    {sys : LineSystem Line}
    {lines : Finset Line}
    (cell : FiniteEnvelopeCell sys lines)
    {newLine : Line}
    {z : ℝ}
    (hSlope : sys.slope cell.activeLine < sys.slope newLine)
    (hCut : lineIntersection sys cell.activeLine newLine ≤ cell.leftCut)
    (hzlo : cell.leftCut ≤ z)
    (hzhi : z ≤ cell.rightCut) :
    EnvelopeAtomCertificate sys (insert newLine lines) z
      (lineValue sys newLine z) := by
  let poppedCell :=
    popped_finite_cell_as_newLine_cell cell hSlope hCut
  simpa [FiniteEnvelopeCell.activeValue,
    popped_finite_cell_as_newLine_cell, poppedCell] using
    finiteEnvelopeCell_atom_certificate poppedCell hzlo hzhi

theorem popped_finite_cell_newLine_dominates_interval
    [DecidableEq Line]
    {sys : LineSystem Line}
    {lines : Finset Line}
    (cell : FiniteEnvelopeCell sys lines)
    {newLine : Line}
    {z : ℝ}
    (hSlope : sys.slope cell.activeLine < sys.slope newLine)
    (hCut : lineIntersection sys cell.activeLine newLine ≤ cell.leftCut)
    (hzlo : cell.leftCut ≤ z)
    (hzhi : z ≤ cell.rightCut) :
    ∀ line ∈ insert newLine lines,
      lineValue sys line z ≤ lineValue sys newLine z := by
  intro line hline
  exact
    (popped_finite_cell_newLine_atom_certificate
      cell hSlope hCut hzlo hzhi).upper line hline

noncomputable def split_right_tail_old_cell
    [DecidableEq Line]
    {sys : LineSystem Line}
    {lines : Finset Line}
    (cell : RightTailEnvelopeCell sys lines)
    {newLine : Line}
    (hSlope : sys.slope cell.activeLine < sys.slope newLine)
    (hCut : cell.leftCut ≤ lineIntersection sys cell.activeLine newLine) :
    FiniteEnvelopeCell sys (insert newLine lines) := by
  let cut := lineIntersection sys cell.activeLine newLine
  exact
    { activeLine := cell.activeLine
      leftCut := cell.leftCut
      rightCut := cut
      active_mem := Finset.mem_insert_of_mem cell.active_mem
      lo_le_hi := hCut
      dominates_lo :=
        split_right_tail_old_cell_dominates_new_left_piece
          cell hSlope hCut
      dominates_hi := by
        intro line hline
        rw [Finset.mem_insert] at hline
        rcases hline with hEq | hOld
        · subst line
          exact le_of_eq
            (oldLine_eq_newLine_at_intersection
              sys cell.activeLine newLine hSlope).symm
        · exact right_tail_old_active_dominates_processed_at_intersection
            cell hSlope hCut line hOld }

noncomputable def split_right_tail_new_cell
    [DecidableEq Line]
    {sys : LineSystem Line}
    {lines : Finset Line}
    (cell : RightTailEnvelopeCell sys lines)
    {newLine : Line}
    (hSlope : sys.slope cell.activeLine < sys.slope newLine)
    (hCut : cell.leftCut ≤ lineIntersection sys cell.activeLine newLine) :
    RightTailEnvelopeCell sys (insert newLine lines) := by
  let cut := lineIntersection sys cell.activeLine newLine
  exact
    { activeLine := newLine
      leftCut := cut
      active_mem := Finset.mem_insert_self newLine lines
      right_slope_max :=
        split_right_tail_newLine_is_right_slope_max cell hSlope
      dominates_lo :=
        split_right_tail_newLine_dominates_at_new_left_cut
          cell hSlope hCut }

theorem split_right_tail_constructs_finite_and_right_tail_cells
    [DecidableEq Line]
    {sys : LineSystem Line}
    {lines : Finset Line}
    (cell : RightTailEnvelopeCell sys lines)
    {newLine : Line}
    (hSlope : sys.slope cell.activeLine < sys.slope newLine)
    (hCut : cell.leftCut ≤ lineIntersection sys cell.activeLine newLine) :
    ∃ oldCell : FiniteEnvelopeCell sys (insert newLine lines),
      ∃ newCell : RightTailEnvelopeCell sys (insert newLine lines),
        oldCell.activeLine = cell.activeLine
          ∧ oldCell.leftCut = cell.leftCut
          ∧ oldCell.rightCut = lineIntersection sys cell.activeLine newLine
          ∧ newCell.activeLine = newLine
          ∧ newCell.leftCut = lineIntersection sys cell.activeLine newLine := by
  exact
    ⟨split_right_tail_old_cell cell hSlope hCut,
      split_right_tail_new_cell cell hSlope hCut,
      rfl, rfl, rfl, rfl, rfl⟩

theorem split_right_tail_old_piece_atom_certificate
    [DecidableEq Line]
    {sys : LineSystem Line}
    {lines : Finset Line}
    (cell : RightTailEnvelopeCell sys lines)
    {newLine : Line}
    {z : ℝ}
    (hSlope : sys.slope cell.activeLine < sys.slope newLine)
    (hCut : cell.leftCut ≤ lineIntersection sys cell.activeLine newLine)
    (hzlo : cell.leftCut ≤ z)
    (hzhi : z ≤ lineIntersection sys cell.activeLine newLine) :
    EnvelopeAtomCertificate sys (insert newLine lines) z
      (lineValue sys cell.activeLine z) := by
  let oldCell :=
    split_right_tail_old_cell cell hSlope hCut
  simpa [FiniteEnvelopeCell.activeValue,
    split_right_tail_old_cell, oldCell] using
    finiteEnvelopeCell_atom_certificate oldCell hzlo hzhi

theorem split_right_tail_old_piece_dominates_interval
    [DecidableEq Line]
    {sys : LineSystem Line}
    {lines : Finset Line}
    (cell : RightTailEnvelopeCell sys lines)
    {newLine : Line}
    {z : ℝ}
    (hSlope : sys.slope cell.activeLine < sys.slope newLine)
    (hCut : cell.leftCut ≤ lineIntersection sys cell.activeLine newLine)
    (hzlo : cell.leftCut ≤ z)
    (hzhi : z ≤ lineIntersection sys cell.activeLine newLine) :
    ∀ line ∈ insert newLine lines,
      lineValue sys line z ≤ lineValue sys cell.activeLine z := by
  intro line hline
  exact
    (split_right_tail_old_piece_atom_certificate
      cell hSlope hCut hzlo hzhi).upper line hline

theorem split_right_tail_new_piece_atom_certificate
    [DecidableEq Line]
    {sys : LineSystem Line}
    {lines : Finset Line}
    (cell : RightTailEnvelopeCell sys lines)
    {newLine : Line}
    {z : ℝ}
    (hSlope : sys.slope cell.activeLine < sys.slope newLine)
    (hCut : cell.leftCut ≤ lineIntersection sys cell.activeLine newLine)
    (hzlo : lineIntersection sys cell.activeLine newLine ≤ z) :
    EnvelopeAtomCertificate sys (insert newLine lines) z
      (lineValue sys newLine z) := by
  let newCell :=
    split_right_tail_new_cell cell hSlope hCut
  simpa [RightTailEnvelopeCell.activeValue,
    split_right_tail_new_cell, newCell] using
    rightTailEnvelopeCell_atom_certificate newCell hzlo

theorem split_right_tail_new_piece_dominates_right_tail
    [DecidableEq Line]
    {sys : LineSystem Line}
    {lines : Finset Line}
    (cell : RightTailEnvelopeCell sys lines)
    {newLine : Line}
    {z : ℝ}
    (hSlope : sys.slope cell.activeLine < sys.slope newLine)
    (hCut : cell.leftCut ≤ lineIntersection sys cell.activeLine newLine)
    (hzlo : lineIntersection sys cell.activeLine newLine ≤ z) :
    ∀ line ∈ insert newLine lines,
      lineValue sys line z ≤ lineValue sys newLine z := by
  intro line hline
  exact
    (split_right_tail_new_piece_atom_certificate
      cell hSlope hCut hzlo).upper line hline

end SCOLHKG.Real

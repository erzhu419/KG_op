import SCOLHKG.Real.LineEnvelopeIntersection

namespace SCOLHKG.Real

/-!
Full recursive fold proof for the `compute_h` active-line stack.

`LineEnvelopeIntersection.lean` proves the local branch facts.  This file
formalizes the whole sorted/collapsed line fold: inserting a new higher-slope
line either pushes it, or repeatedly pops the previous rightmost active line
when the Python while-loop condition

```text
intersection(last, new) <= intersection(prev, last)
```

fires.  The central invariant is semantic and non-vacuous: every original
input line is pointwise dominated, for every real `z`, by some final active
line in the output stack.  Therefore popped lines cannot silently disappear.
-/

universe u v

variable {Line : Type u} {Atom : Type v}

def LineListDominates
    (sys : LineSystem Line)
    (active input : List Line) : Prop :=
  ∀ line ∈ input, ∀ z : ℝ,
    ∃ activeLine ∈ active,
      lineValue sys line z ≤ lineValue sys activeLine z

inductive InsertLoop
    (sys : LineSystem Line)
    (newLine : Line) :
    List Line → List Line → Prop where
  | empty :
      InsertLoop sys newLine [] [newLine]
  | single
      (oldLine : Line)
      (hSlope : sys.slope oldLine < sys.slope newLine) :
      InsertLoop sys newLine [oldLine] [oldLine, newLine]
  | push
      (pref : List Line)
      (prev last : Line)
      (hPrevLast : sys.slope prev < sys.slope last)
      (hLastNew : sys.slope last < sys.slope newLine)
      (hBreak :
        lineIntersection sys prev last <
          lineIntersection sys last newLine) :
      InsertLoop sys newLine
        (pref ++ [prev, last])
        (pref ++ [prev, last, newLine])
  | pop
      (pref : List Line)
      (prev last : Line)
      (out : List Line)
      (hPrevLast : sys.slope prev < sys.slope last)
      (hLastNew : sys.slope last < sys.slope newLine)
      (hPop :
        lineIntersection sys last newLine ≤
          lineIntersection sys prev last)
      (tail :
        InsertLoop sys newLine (pref ++ [prev]) out) :
      InsertLoop sys newLine
        (pref ++ [prev, last])
        out

theorem insertLoop_newLine_mem
    {sys : LineSystem Line}
    {newLine : Line}
    {active out : List Line}
    (h : InsertLoop sys newLine active out) :
    newLine ∈ out := by
  induction h with
  | empty =>
      simp
  | single =>
      simp
  | push =>
      simp
  | pop _prefix _prev _last _out _hPrevLast _hLastNew _hPop _tail ih =>
      exact ih

theorem insertLoop_output_subset_active_append_new
    {sys : LineSystem Line}
    {newLine : Line}
    {active out : List Line}
    (h : InsertLoop sys newLine active out) :
    ∀ line ∈ out, line ∈ active ++ [newLine] := by
  induction h with
  | empty =>
      intro line hline
      simpa using hline
  | single oldLine _hSlope =>
      intro line hline
      simpa using hline
  | push pref prev last _hPrevLast _hLastNew _hBreak =>
      intro line hline
      simpa [List.append_assoc] using hline
  | pop pref prev last out _hPrevLast _hLastNew _hPop _tail ih =>
      intro line hline
      have hSmall : line ∈ (pref ++ [prev]) ++ [newLine] :=
        ih line hline
      rw [List.mem_append] at hSmall ⊢
      rcases hSmall with hPrefix | hNew
      · left
        rw [List.mem_append] at hPrefix ⊢
        rcases hPrefix with hPrefix | hPrev
        · exact Or.inl hPrefix
        · right
          simp at hPrev ⊢
          exact Or.inl hPrev
      · exact Or.inr hNew

theorem insertLoop_active_line_dominated
    {sys : LineSystem Line}
    {newLine : Line}
    {active out : List Line}
    (h : InsertLoop sys newLine active out) :
    ∀ line ∈ active, ∀ z : ℝ,
      ∃ activeLine ∈ out,
        lineValue sys line z ≤ lineValue sys activeLine z := by
  induction h with
  | empty =>
      intro line hline _z
      simp at hline
  | single oldLine _hSlope =>
      intro line hline _z
      simp at hline
      subst line
      exact ⟨oldLine, by simp, le_rfl⟩
  | push pref prev last _hPrevLast _hLastNew _hBreak =>
      intro line hline _z
      have hout : line ∈ pref ++ [prev, last, newLine] := by
        rw [List.mem_append] at hline ⊢
        rcases hline with hPrefix | hTail
        · exact Or.inl hPrefix
        · right
          simp at hTail ⊢
          rcases hTail with hPrev | hLast
          · exact Or.inl hPrev
          · exact Or.inr (Or.inl hLast)
      exact ⟨line, hout, le_rfl⟩
  | pop pref prev last out hPrevLast hLastNew hPop _tail ih =>
      intro line hline z
      rw [List.mem_append] at hline
      rcases hline with hPrefix | hTail
      · have hPrefixMem : line ∈ pref ++ [prev] := by
          rw [List.mem_append]
          exact Or.inl hPrefix
        exact ih line hPrefixMem z
      · simp at hTail
        rcases hTail with hPrev | hLast
        · subst line
          have hPrevMem : prev ∈ pref ++ [prev] := by
            simp
          exact ih prev hPrevMem z
        · subst line
          let cut := lineIntersection sys prev last
          by_cases hz : z ≤ cut
          · have hLastLePrev :
                lineValue sys last z ≤ lineValue sys prev z := by
              exact oldLine_dominates_left_of_intersection
                sys prev last hPrevLast hz
            have hPrevMem : prev ∈ pref ++ [prev] := by
              simp
            rcases ih prev hPrevMem z with
              ⟨activeLine, hActiveLine, hPrevLeActive⟩
            exact
              ⟨activeLine, hActiveLine,
                hLastLePrev.trans hPrevLeActive⟩
          · have hCutLeZ : cut ≤ z := le_of_lt (lt_of_not_ge hz)
            have hNewCutLeZ :
                lineIntersection sys last newLine ≤ z :=
              hPop.trans hCutLeZ
            have hLastLeNew :
                lineValue sys last z ≤ lineValue sys newLine z := by
              exact newLine_dominates_right_of_intersection
                sys last newLine hLastNew hNewCutLeZ
            exact
              ⟨newLine, insertLoop_newLine_mem _tail, hLastLeNew⟩

theorem insertLoop_preserves_list_domination
    {sys : LineSystem Line}
    {newLine : Line}
    {input active out : List Line}
    (hDom : LineListDominates sys active input)
    (hLoop : InsertLoop sys newLine active out) :
    LineListDominates sys out (input ++ [newLine]) := by
  intro line hline z
  rw [List.mem_append] at hline
  rcases hline with hOld | hNew
  · rcases hDom line hOld z with
      ⟨oldActive, hOldActive, hLineLeOldActive⟩
    rcases insertLoop_active_line_dominated hLoop oldActive hOldActive z with
      ⟨newActive, hNewActive, hOldActiveLeNewActive⟩
    exact
      ⟨newActive, hNewActive,
        hLineLeOldActive.trans hOldActiveLeNewActive⟩
  · simp at hNew
    subst line
    exact ⟨newLine, insertLoop_newLine_mem hLoop, le_rfl⟩

inductive FoldLoop
    (sys : LineSystem Line) :
    List Line → List Line → Prop where
  | nil :
      FoldLoop sys [] []
  | snoc
      {input active out : List Line}
      {newLine : Line}
      (prefixLoop : FoldLoop sys input active)
      (insertLoop : InsertLoop sys newLine active out) :
      FoldLoop sys (input ++ [newLine]) out

theorem foldLoop_dominates_input
    {sys : LineSystem Line}
    {input output : List Line}
    (h : FoldLoop sys input output) :
    LineListDominates sys output input := by
  induction h with
  | nil =>
      intro line hline _z
      simp at hline
  | snoc _prefixLoop insertLoop ih =>
      exact insertLoop_preserves_list_domination ih insertLoop

theorem foldLoop_output_subset_input
    {sys : LineSystem Line}
    {input output : List Line}
    (h : FoldLoop sys input output) :
    ∀ line ∈ output, line ∈ input := by
  induction h with
  | nil =>
      intro line hline
      simp at hline
  | snoc _prefixLoop insertLoop ih =>
      intro line hline
      have hSubset :=
        insertLoop_output_subset_active_append_new insertLoop line hline
      rw [List.mem_append] at hSubset ⊢
      rcases hSubset with hOldActive | hNew
      · exact Or.inl (ih line hOldActive)
      · exact Or.inr hNew

theorem foldLoop_toFinset_dominates_input
    [DecidableEq Line]
    {sys : LineSystem Line}
    {input output : List Line}
    (h : FoldLoop sys input output) :
    ∀ line ∈ input.toFinset, ∀ z : ℝ,
      ∃ activeLine ∈ output,
        lineValue sys line z ≤ lineValue sys activeLine z := by
  intro line hline z
  exact foldLoop_dominates_input h line (by simpa using hline) z

theorem foldLoop_output_endpoint_dominance_to_finalInvariant
    [DecidableEq Line]
    {sys : LineSystem Line}
    {input output : List Line}
    {atoms : Finset Atom}
    {active : Atom → Line}
    {z lo hi : Atom → ℝ}
    (hLoop : FoldLoop sys input output)
    (hActiveOutput :
      ∀ atom ∈ atoms, active atom ∈ output)
    (hInterval :
      ∀ atom ∈ atoms, lo atom ≤ z atom ∧ z atom ≤ hi atom)
    (hOutputLo :
      ∀ atom ∈ atoms, ∀ line ∈ output,
        lineValue sys line (lo atom) ≤
          lineValue sys (active atom) (lo atom))
    (hOutputHi :
      ∀ atom ∈ atoms, ∀ line ∈ output,
        lineValue sys line (hi atom) ≤
          lineValue sys (active atom) (hi atom)) :
    FinalEnvelopeStackInvariant
      sys input.toFinset atoms active z lo hi := by
  refine ⟨?_, hInterval, ?_, ?_⟩
  · intro atom hatom
    have hOut := hActiveOutput atom hatom
    have hIn := foldLoop_output_subset_input hLoop (active atom) hOut
    simpa using hIn
  · intro atom hatom line hline
    rcases foldLoop_toFinset_dominates_input hLoop line hline
        (lo atom) with
      ⟨outLine, hOutLine, hLineLeOut⟩
    exact hLineLeOut.trans (hOutputLo atom hatom outLine hOutLine)
  · intro atom hatom line hline
    rcases foldLoop_toFinset_dominates_input hLoop line hline
        (hi atom) with
      ⟨outLine, hOutLine, hLineLeOut⟩
    exact hLineLeOut.trans (hOutputHi atom hatom outLine hOutLine)

theorem foldLoop_lineEnvelopeKG_exact_from_output_endpoint_dominance
    [DecidableEq Line]
    {sys : LineSystem Line}
    {input output : List Line}
    {atoms : Finset Atom}
    {probMass firstMoment z lo hi : Atom → ℝ}
    {active : Atom → Line}
    {baseline expectedMax : ℝ}
    (hLoop : FoldLoop sys input output)
    (hActiveOutput :
      ∀ atom ∈ atoms, active atom ∈ output)
    (hInterval :
      ∀ atom ∈ atoms, lo atom ≤ z atom ∧ z atom ≤ hi atom)
    (hOutputLo :
      ∀ atom ∈ atoms, ∀ line ∈ output,
        lineValue sys line (lo atom) ≤
          lineValue sys (active atom) (lo atom))
    (hOutputHi :
      ∀ atom ∈ atoms, ∀ line ∈ output,
        lineValue sys line (hi atom) ≤
          lineValue sys (active atom) (hi atom))
    (hMoment :
      ∀ atom ∈ atoms, firstMoment atom = probMass atom * z atom)
    (hExpected :
      expectedMax =
        finiteLineEnvelopeMaxExpectation atoms probMass
          (fun atom ↦ lineValue sys (active atom) (z atom))) :
    lineEnvelopeKGFormula sys atoms probMass firstMoment active baseline =
      expectedMax - baseline := by
  exact finalEnvelopeStackInvariant_lineEnvelopeKG_exact
    (sys := sys)
    (lines := input.toFinset)
    (atoms := atoms)
    (probMass := probMass)
    (firstMoment := firstMoment)
    (z := z)
    (lo := lo)
    (hi := hi)
    (active := active)
    (baseline := baseline)
    (expectedMax := expectedMax)
    (foldLoop_output_endpoint_dominance_to_finalInvariant
      hLoop hActiveOutput hInterval hOutputLo hOutputHi)
    hMoment
    hExpected

end SCOLHKG.Real

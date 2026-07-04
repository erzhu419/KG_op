import SCOLHKG.Real.LineEnvelopeKG

namespace SCOLHKG.Real

/-!
Stack-hull certificate bridge for the imperative `compute_h` implementation.

`core/kg.py::validate_h_certificate` checks active-line dominance by endpoint
dominance on finite intervals and by slope dominance on the two Gaussian tails.
The lemmas below prove that these local checks imply the pointwise
`EnvelopeAtomCertificate` consumed by `LineEnvelopeKG.lean`.
-/

universe u

variable {Line : Type u}

theorem affine_difference_at_right_endpoint
    (sys : LineSystem Line)
    (line active : Line)
    (z hi : ℝ) :
    lineValue sys line z - lineValue sys active z =
      (lineValue sys line hi - lineValue sys active hi)
        + (sys.slope line - sys.slope active) * (z - hi) := by
  unfold lineValue
  ring

theorem affine_difference_at_left_endpoint
    (sys : LineSystem Line)
    (line active : Line)
    (lo z : ℝ) :
    lineValue sys line z - lineValue sys active z =
      (lineValue sys line lo - lineValue sys active lo)
        + (sys.slope line - sys.slope active) * (z - lo) := by
  unfold lineValue
  ring

theorem finite_interval_endpoint_dominance
    (sys : LineSystem Line)
    (line active : Line)
    {lo hi z : ℝ}
    (hlo : lo ≤ z)
    (hhi : z ≤ hi)
    (hDomLo : lineValue sys line lo ≤ lineValue sys active lo)
    (hDomHi : lineValue sys line hi ≤ lineValue sys active hi) :
    lineValue sys line z ≤ lineValue sys active z := by
  by_cases hSlope : 0 ≤ sys.slope line - sys.slope active
  · have hProd :
        (sys.slope line - sys.slope active) * (z - hi) ≤ 0 := by
      exact mul_nonpos_of_nonneg_of_nonpos hSlope (sub_nonpos.mpr hhi)
    have hEndpoint :
        lineValue sys line hi - lineValue sys active hi ≤ 0 := by
      exact sub_nonpos.mpr hDomHi
    have hDiff :
        lineValue sys line z - lineValue sys active z ≤ 0 := by
      rw [affine_difference_at_right_endpoint
        (sys := sys) (line := line) (active := active) (z := z) (hi := hi)]
      linarith
    exact sub_nonpos.mp hDiff
  · have hSlopeLe : sys.slope line - sys.slope active ≤ 0 :=
      le_of_not_ge hSlope
    have hProd :
        (sys.slope line - sys.slope active) * (z - lo) ≤ 0 := by
      exact mul_nonpos_of_nonpos_of_nonneg hSlopeLe (sub_nonneg.mpr hlo)
    have hEndpoint :
        lineValue sys line lo - lineValue sys active lo ≤ 0 := by
      exact sub_nonpos.mpr hDomLo
    have hDiff :
        lineValue sys line z - lineValue sys active z ≤ 0 := by
      rw [affine_difference_at_left_endpoint
        (sys := sys) (line := line) (active := active) (lo := lo) (z := z)]
      linarith
    exact sub_nonpos.mp hDiff

theorem left_tail_endpoint_slope_dominance
    (sys : LineSystem Line)
    (line active : Line)
    {z hi : ℝ}
    (hhi : z ≤ hi)
    (hSlope : sys.slope active ≤ sys.slope line)
    (hDomHi : lineValue sys line hi ≤ lineValue sys active hi) :
    lineValue sys line z ≤ lineValue sys active z := by
  have hSlopeDiff : 0 ≤ sys.slope line - sys.slope active := by
    exact sub_nonneg.mpr hSlope
  have hProd :
      (sys.slope line - sys.slope active) * (z - hi) ≤ 0 := by
    exact mul_nonpos_of_nonneg_of_nonpos hSlopeDiff (sub_nonpos.mpr hhi)
  have hEndpoint :
      lineValue sys line hi - lineValue sys active hi ≤ 0 := by
    exact sub_nonpos.mpr hDomHi
  have hDiff :
      lineValue sys line z - lineValue sys active z ≤ 0 := by
    rw [affine_difference_at_right_endpoint
      (sys := sys) (line := line) (active := active) (z := z) (hi := hi)]
    linarith
  exact sub_nonpos.mp hDiff

theorem right_tail_endpoint_slope_dominance
    (sys : LineSystem Line)
    (line active : Line)
    {lo z : ℝ}
    (hlo : lo ≤ z)
    (hSlope : sys.slope line ≤ sys.slope active)
    (hDomLo : lineValue sys line lo ≤ lineValue sys active lo) :
    lineValue sys line z ≤ lineValue sys active z := by
  have hSlopeDiff : sys.slope line - sys.slope active ≤ 0 := by
    exact sub_nonpos.mpr hSlope
  have hProd :
      (sys.slope line - sys.slope active) * (z - lo) ≤ 0 := by
    exact mul_nonpos_of_nonpos_of_nonneg hSlopeDiff (sub_nonneg.mpr hlo)
  have hEndpoint :
      lineValue sys line lo - lineValue sys active lo ≤ 0 := by
    exact sub_nonpos.mpr hDomLo
  have hDiff :
      lineValue sys line z - lineValue sys active z ≤ 0 := by
    rw [affine_difference_at_left_endpoint
      (sys := sys) (line := line) (active := active) (lo := lo) (z := z)]
    linarith
  exact sub_nonpos.mp hDiff

theorem finite_interval_stack_atom_certificate
    (sys : LineSystem Line)
    (lines : Finset Line)
    (active : Line)
    {lo hi z : ℝ}
    (hActive : active ∈ lines)
    (hlo : lo ≤ z)
    (hhi : z ≤ hi)
    (hDomLo :
      ∀ line ∈ lines,
        lineValue sys line lo ≤ lineValue sys active lo)
    (hDomHi :
      ∀ line ∈ lines,
        lineValue sys line hi ≤ lineValue sys active hi) :
    EnvelopeAtomCertificate sys lines z (lineValue sys active z) := by
  refine ⟨?_, ?_⟩
  · intro line hline
    exact finite_interval_endpoint_dominance
      sys line active hlo hhi (hDomLo line hline) (hDomHi line hline)
  · exact ⟨active, hActive, rfl⟩

theorem left_tail_stack_atom_certificate
    (sys : LineSystem Line)
    (lines : Finset Line)
    (active : Line)
    {hi z : ℝ}
    (hActive : active ∈ lines)
    (hhi : z ≤ hi)
    (hSlope :
      ∀ line ∈ lines, sys.slope active ≤ sys.slope line)
    (hDomHi :
      ∀ line ∈ lines,
        lineValue sys line hi ≤ lineValue sys active hi) :
    EnvelopeAtomCertificate sys lines z (lineValue sys active z) := by
  refine ⟨?_, ?_⟩
  · intro line hline
    exact left_tail_endpoint_slope_dominance
      sys line active hhi (hSlope line hline) (hDomHi line hline)
  · exact ⟨active, hActive, rfl⟩

theorem right_tail_stack_atom_certificate
    (sys : LineSystem Line)
    (lines : Finset Line)
    (active : Line)
    {lo z : ℝ}
    (hActive : active ∈ lines)
    (hlo : lo ≤ z)
    (hSlope :
      ∀ line ∈ lines, sys.slope line ≤ sys.slope active)
    (hDomLo :
      ∀ line ∈ lines,
        lineValue sys line lo ≤ lineValue sys active lo) :
    EnvelopeAtomCertificate sys lines z (lineValue sys active z) := by
  refine ⟨?_, ?_⟩
  · intro line hline
    exact right_tail_endpoint_slope_dominance
      sys line active hlo (hSlope line hline) (hDomLo line hline)
  · exact ⟨active, hActive, rfl⟩

end SCOLHKG.Real

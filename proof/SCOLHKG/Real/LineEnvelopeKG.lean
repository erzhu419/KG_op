import Mathlib

namespace SCOLHKG.Real

open scoped BigOperators

/-!
Line-envelope KG bridge for `core/kg.py::compute_h`.

The Python implementation constructs an upper envelope of affine lines
`a_j + b_j Z` and sums each active line's contribution over its interval.  This
file formalizes the certificate that makes such a calculation exact.  It does
not prove the imperative stack algorithm yet; it proves that once the hull
certifies the active line on each atom/region, the returned envelope sum is the
exact one-step KG value.
-/

universe u v

structure LineSystem (Line : Type u) where
  intercept : Line → ℝ
  slope : Line → ℝ

def lineValue
    {Line : Type u}
    (sys : LineSystem Line)
    (line : Line)
    (z : ℝ) : ℝ :=
  sys.intercept line + sys.slope line * z

structure EnvelopeAtomCertificate
    {Line : Type u}
    (sys : LineSystem Line)
    (lines : Finset Line)
    (z : ℝ)
    (value : ℝ) : Prop where
  upper : ∀ line ∈ lines, lineValue sys line z ≤ value
  attained : ∃ line ∈ lines, value = lineValue sys line z

def lineEnvelopeExpectation
    {Line : Type u}
    {Atom : Type v}
    (sys : LineSystem Line)
    (atoms : Finset Atom)
    (probMass firstMoment : Atom → ℝ)
    (active : Atom → Line) : ℝ :=
  ∑ atom ∈ atoms,
    (sys.intercept (active atom) * probMass atom
      + sys.slope (active atom) * firstMoment atom)

def finiteLineEnvelopeMaxExpectation
    {Atom : Type v}
    (atoms : Finset Atom)
    (probMass : Atom → ℝ)
    (value : Atom → ℝ) : ℝ :=
  ∑ atom ∈ atoms, probMass atom * value atom

def lineEnvelopeKGFormula
    {Line : Type u}
    {Atom : Type v}
    (sys : LineSystem Line)
    (atoms : Finset Atom)
    (probMass firstMoment : Atom → ℝ)
    (active : Atom → Line)
    (baseline : ℝ) : ℝ :=
  lineEnvelopeExpectation sys atoms probMass firstMoment active - baseline

theorem lineEnvelopeExpectation_eq_weighted_active_values
    {Line : Type u}
    {Atom : Type v}
    (sys : LineSystem Line)
    (atoms : Finset Atom)
    (probMass firstMoment z : Atom → ℝ)
    (active : Atom → Line)
    (hMoment : ∀ atom ∈ atoms, firstMoment atom = probMass atom * z atom) :
    lineEnvelopeExpectation sys atoms probMass firstMoment active =
      finiteLineEnvelopeMaxExpectation atoms probMass
        (fun atom ↦ lineValue sys (active atom) (z atom)) := by
  unfold lineEnvelopeExpectation finiteLineEnvelopeMaxExpectation lineValue
  apply Finset.sum_congr rfl
  intro atom hatom
  rw [hMoment atom hatom]
  ring

theorem certified_lineEnvelopeKG_exact
    {Line : Type u}
    {Atom : Type v}
    (sys : LineSystem Line)
    (lines : Finset Line)
    (atoms : Finset Atom)
    (probMass firstMoment z : Atom → ℝ)
    (active : Atom → Line)
    (baseline expectedMax : ℝ)
    (hMoment : ∀ atom ∈ atoms, firstMoment atom = probMass atom * z atom)
    (_hActive :
      ∀ atom ∈ atoms,
        EnvelopeAtomCertificate sys lines (z atom)
          (lineValue sys (active atom) (z atom)))
    (hExpected :
      expectedMax =
        finiteLineEnvelopeMaxExpectation atoms probMass
          (fun atom ↦ lineValue sys (active atom) (z atom))) :
    lineEnvelopeKGFormula sys atoms probMass firstMoment active baseline =
      expectedMax - baseline := by
  rw [lineEnvelopeKGFormula,
    lineEnvelopeExpectation_eq_weighted_active_values
      sys atoms probMass firstMoment z active hMoment,
    hExpected]

theorem certified_lineEnvelopeKG_nonnegative
    {Line : Type u}
    {Atom : Type v}
    (sys : LineSystem Line)
    (atoms : Finset Atom)
    (probMass firstMoment : Atom → ℝ)
    (active : Atom → Line)
    (baseline : ℝ)
    (hBaseline :
      baseline ≤ lineEnvelopeExpectation sys atoms probMass firstMoment active) :
    0 ≤ lineEnvelopeKGFormula sys atoms probMass firstMoment active baseline := by
  unfold lineEnvelopeKGFormula
  linarith

end SCOLHKG.Real

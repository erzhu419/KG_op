import Mathlib

namespace SCOLHKG.Real

/-!
Finite implementation bridge for the source-affine chance-boundary transfer
used by `ObservableConstraintMeanBasis(output_mode="source_affine")`.

Each source expert contributes one frozen signed-distance atom `h`. Source
episodes learn an offset and scale law `a + b h`; charged target observations
update the same two coefficients. The results below isolate the assumptions
needed by certification: positive scale preserves the safe side, coefficient
errors give an explicit uniform margin error, and adding that error to the
posterior upper margin yields a sound target certificate.
-/

def sourceAffineBoundary
    (offset scale atom : ℝ) : ℝ :=
  offset + scale * atom

theorem sourceAffineBoundary_error_identity
    (offset scale estimatedOffset estimatedScale atom : ℝ) :
    sourceAffineBoundary offset scale atom
        - sourceAffineBoundary estimatedOffset estimatedScale atom
      = (offset - estimatedOffset)
        + (scale - estimatedScale) * atom := by
  unfold sourceAffineBoundary
  ring

theorem sourceAffineBoundary_error_abs_le
    (offset scale estimatedOffset estimatedScale atom : ℝ) :
    abs
        (sourceAffineBoundary offset scale atom
          - sourceAffineBoundary estimatedOffset estimatedScale atom)
      ≤ abs (offset - estimatedOffset)
        + abs (scale - estimatedScale) * abs atom := by
  rw [sourceAffineBoundary_error_identity]
  calc
    abs
        ((offset - estimatedOffset)
          + (scale - estimatedScale) * atom)
      ≤ abs (offset - estimatedOffset)
          + abs ((scale - estimatedScale) * atom) := abs_add_le _ _
    _ = abs (offset - estimatedOffset)
          + abs (scale - estimatedScale) * abs atom := by
      rw [abs_mul]

theorem positive_source_scale_preserves_safe_side
    {scale atom : ℝ}
    (hScale : 0 < scale) :
    sourceAffineBoundary 0 scale atom ≤ 0 ↔ atom ≤ 0 := by
  unfold sourceAffineBoundary
  simp only [zero_add]
  constructor
  · intro hProduct
    by_contra hAtom
    have hAtomPositive : 0 < atom := lt_of_not_ge hAtom
    have hProductPositive : 0 < scale * atom :=
      mul_pos hScale hAtomPositive
    linarith
  · intro hAtom
    exact mul_nonpos_of_nonneg_of_nonpos (le_of_lt hScale) hAtom

theorem sourceAffineBoundary_upper_of_parameter_error
    {offset scale estimatedOffset estimatedScale atom
      offsetRadius scaleRadius atomRadius : ℝ}
    (hOffset : abs (offset - estimatedOffset) ≤ offsetRadius)
    (hScale : abs (scale - estimatedScale) ≤ scaleRadius)
    (hAtom : abs atom ≤ atomRadius)
    (hScaleRadius : 0 ≤ scaleRadius) :
    sourceAffineBoundary offset scale atom
      ≤ sourceAffineBoundary estimatedOffset estimatedScale atom
        + offsetRadius + scaleRadius * atomRadius := by
  have hProduct :
      abs (scale - estimatedScale) * abs atom
        ≤ scaleRadius * atomRadius := by
    exact mul_le_mul hScale hAtom (abs_nonneg atom) hScaleRadius
  have hAbsoluteError :
      abs
          (sourceAffineBoundary offset scale atom
            - sourceAffineBoundary estimatedOffset estimatedScale atom)
        ≤ offsetRadius + scaleRadius * atomRadius := by
    calc
      abs
          (sourceAffineBoundary offset scale atom
            - sourceAffineBoundary estimatedOffset estimatedScale atom)
        ≤ abs (offset - estimatedOffset)
            + abs (scale - estimatedScale) * abs atom :=
          sourceAffineBoundary_error_abs_le _ _ _ _ _
      _ ≤ offsetRadius + scaleRadius * atomRadius :=
        add_le_add hOffset hProduct
  have hDifference :
      sourceAffineBoundary offset scale atom
          - sourceAffineBoundary estimatedOffset estimatedScale atom
        ≤ offsetRadius + scaleRadius * atomRadius :=
    le_trans (le_abs_self _) hAbsoluteError
  linarith

theorem sourceAffineBoundary_certificate_sound
    {offset scale estimatedOffset estimatedScale atom
      offsetRadius scaleRadius atomRadius epistemicGuard aleatoricGuard : ℝ}
    (hOffset : abs (offset - estimatedOffset) ≤ offsetRadius)
    (hScale : abs (scale - estimatedScale) ≤ scaleRadius)
    (hAtom : abs atom ≤ atomRadius)
    (hScaleRadius : 0 ≤ scaleRadius)
    (hCertificate :
      sourceAffineBoundary estimatedOffset estimatedScale atom
          + offsetRadius + scaleRadius * atomRadius
          + epistemicGuard + aleatoricGuard
        ≤ 0) :
    sourceAffineBoundary offset scale atom
        + epistemicGuard + aleatoricGuard
      ≤ 0 := by
  have hTransferred := sourceAffineBoundary_upper_of_parameter_error
    hOffset hScale hAtom hScaleRadius
  linarith

theorem sourceAffineBoundary_certificate_with_threshold
    {offset scale estimatedOffset estimatedScale atom
      offsetRadius scaleRadius atomRadius epistemicGuard aleatoricGuard
      threshold : ℝ}
    (hOffset : abs (offset - estimatedOffset) ≤ offsetRadius)
    (hScale : abs (scale - estimatedScale) ≤ scaleRadius)
    (hAtom : abs atom ≤ atomRadius)
    (hScaleRadius : 0 ≤ scaleRadius)
    (hCertificate :
      sourceAffineBoundary estimatedOffset estimatedScale atom
          + offsetRadius + scaleRadius * atomRadius
          + epistemicGuard + aleatoricGuard
        ≤ threshold) :
    sourceAffineBoundary offset scale atom
        + epistemicGuard + aleatoricGuard
      ≤ threshold := by
  have hTransferred := sourceAffineBoundary_upper_of_parameter_error
    hOffset hScale hAtom hScaleRadius
  linarith

end SCOLHKG.Real

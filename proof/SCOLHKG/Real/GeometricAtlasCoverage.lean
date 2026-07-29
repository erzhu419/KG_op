import Mathlib

namespace SCOLHKG.Real

/-!
Geometric coverage of a held-out safe basin in the transferable coordinate.

The frozen proposal is a finite maximin atlas in `psi` space.  The source
support is covered by the atlas, a target-safe center lies near that support,
and a complete coordinate ball around the center is feasible.  If atlas
covering radius plus source/target support shift fits inside the safe radius,
the atlas must contain a feasible policy.  Nominal policy dimension does not
enter this implication.
-/

def CoordinateAtlasCovers
    {X Z : Type*} [DecidableEq X] [PseudoMetricSpace Z]
    (psi : X → Z) (atlas support : Finset X) (radius : ℝ) : Prop :=
  ∀ source ∈ support,
    ∃ candidate ∈ atlas, dist (psi candidate) (psi source) ≤ radius

def CoordinateSafeBall
    {X Z : Type*} [PseudoMetricSpace Z]
    (psi : X → Z) (feasible : X → Prop) (center : X) (radius : ℝ) : Prop :=
  ∀ candidate,
    dist (psi candidate) (psi center) ≤ radius → feasible candidate

def CoordinateMarginOneSidedLipschitz
    {X Z : Type*} [PseudoMetricSpace Z]
    (psi : X → Z) (margin : X → ℝ) (L : ℝ) : Prop :=
  ∀ x y, margin x ≤ margin y + L * dist (psi x) (psi y)

def UniformCoordinateApproximation
    {X Z : Type*} [PseudoMetricSpace Z]
    (learned truth : X → Z) (epsilon : ℝ) : Prop :=
  ∀ x, dist (learned x) (truth x) ≤ epsilon

def TruthCoordinateSupportProxy
    {X Z : Type*} [PseudoMetricSpace Z]
    (truth : X → Z) (support : Finset X) (center : X)
    (domainShift : ℝ) : Prop :=
  ∃ source ∈ support,
    dist (truth source) (truth center) ≤ domainShift

theorem learned_coordinate_support_shift
    {X Z : Type*} [PseudoMetricSpace Z]
    {learned truth : X → Z}
    {support : Finset X}
    {center : X}
    {epsilon domainShift : ℝ}
    (hApproximation :
      UniformCoordinateApproximation learned truth epsilon)
    (hTruthProxy :
      TruthCoordinateSupportProxy truth support center domainShift) :
    ∃ source ∈ support,
      dist (learned source) (learned center)
        ≤ domainShift + 2 * epsilon := by
  obtain ⟨source, hSource, hShift⟩ := hTruthProxy
  have hCenterApproximation :
      dist (truth center) (learned center) ≤ epsilon := by
    simpa [dist_comm] using hApproximation center
  refine ⟨source, hSource, ?_⟩
  calc
    dist (learned source) (learned center) ≤
        dist (learned source) (truth source)
          + dist (truth source) (learned center) := dist_triangle _ _ _
    _ ≤ dist (learned source) (truth source)
        + (dist (truth source) (truth center)
          + dist (truth center) (learned center)) := by
            gcongr
            exact dist_triangle _ _ _
    _ ≤ epsilon + (domainShift + epsilon) := by
          exact add_le_add
            (hApproximation source)
            (add_le_add hShift hCenterApproximation)
    _ = domainShift + 2 * epsilon := by ring

theorem geometric_atlas_hits_safe_basin
    {X Z : Type*} [DecidableEq X] [PseudoMetricSpace Z]
    {psi : X → Z}
    {atlas support : Finset X}
    {feasible : X → Prop}
    {center : X}
    {coverRadius supportShift safeRadius : ℝ}
    (hCover :
      CoordinateAtlasCovers psi atlas support coverRadius)
    (hSupportProxy :
      ∃ source ∈ support,
        dist (psi source) (psi center) ≤ supportShift)
    (hSafeBall :
      CoordinateSafeBall psi feasible center safeRadius)
    (hRadius : coverRadius + supportShift ≤ safeRadius) :
    ∃ candidate ∈ atlas, feasible candidate := by
  obtain ⟨source, hSource, hShift⟩ := hSupportProxy
  obtain ⟨candidate, hCandidate, hCoverDistance⟩ :=
    hCover source hSource
  have hCenterDistance :
      dist (psi candidate) (psi center) ≤ safeRadius := by
    calc
      dist (psi candidate) (psi center) ≤
          dist (psi candidate) (psi source)
            + dist (psi source) (psi center) := dist_triangle _ _ _
      _ ≤ coverRadius + supportShift := add_le_add
        hCoverDistance hShift
      _ ≤ safeRadius := hRadius
  exact ⟨candidate, hCandidate, hSafeBall candidate hCenterDistance⟩

theorem finite_geometric_atlas_coverage
    {X Z : Type*} [DecidableEq X] [PseudoMetricSpace Z]
    {psi : X → Z}
    {atlas support : Finset X}
    {feasible : X → Prop}
    {center : X}
    {coverRadius supportShift safeRadius : ℝ}
    {n0 : ℕ}
    (hAtlasSize : atlas.card ≤ n0)
    (hCover :
      CoordinateAtlasCovers psi atlas support coverRadius)
    (hSupportProxy :
      ∃ source ∈ support,
        dist (psi source) (psi center) ≤ supportShift)
    (hSafeBall :
      CoordinateSafeBall psi feasible center safeRadius)
    (hRadius : coverRadius + supportShift ≤ safeRadius) :
    atlas.card ≤ n0 ∧ ∃ candidate ∈ atlas, feasible candidate := by
  exact ⟨hAtlasSize, geometric_atlas_hits_safe_basin
    hCover hSupportProxy hSafeBall hRadius⟩

theorem geometric_atlas_hits_lipschitz_safe_basin
    {X Z : Type*} [DecidableEq X] [PseudoMetricSpace Z]
    {psi : X → Z}
    {atlas support : Finset X}
    {margin : X → ℝ}
    {center : X}
    {coverRadius supportShift L safeDepth : ℝ}
    (hCover :
      CoordinateAtlasCovers psi atlas support coverRadius)
    (hSupportProxy :
      ∃ source ∈ support,
        dist (psi source) (psi center) ≤ supportShift)
    (hLipschitz :
      CoordinateMarginOneSidedLipschitz psi margin L)
    (hLNonnegative : 0 ≤ L)
    (hCenterDepth : margin center + safeDepth ≤ 0)
    (hDepth : L * (coverRadius + supportShift) ≤ safeDepth) :
    ∃ candidate ∈ atlas, margin candidate ≤ 0 := by
  obtain ⟨source, hSource, hShift⟩ := hSupportProxy
  obtain ⟨candidate, hCandidate, hCoverDistance⟩ :=
    hCover source hSource
  have hDistance :
      dist (psi candidate) (psi center)
        ≤ coverRadius + supportShift := by
    calc
      dist (psi candidate) (psi center) ≤
          dist (psi candidate) (psi source)
            + dist (psi source) (psi center) := dist_triangle _ _ _
      _ ≤ coverRadius + supportShift := add_le_add
        hCoverDistance hShift
  have hScaledDistance :
      L * dist (psi candidate) (psi center)
        ≤ L * (coverRadius + supportShift) :=
    mul_le_mul_of_nonneg_left hDistance hLNonnegative
  have hCandidateMargin := hLipschitz candidate center
  refine ⟨candidate, hCandidate, ?_⟩
  linarith

theorem finite_geometric_lipschitz_atlas_coverage
    {X Z : Type*} [DecidableEq X] [PseudoMetricSpace Z]
    {psi : X → Z}
    {atlas support : Finset X}
    {margin : X → ℝ}
    {center : X}
    {coverRadius supportShift L safeDepth : ℝ}
    {n0 : ℕ}
    (hAtlasSize : atlas.card ≤ n0)
    (hCover :
      CoordinateAtlasCovers psi atlas support coverRadius)
    (hSupportProxy :
      ∃ source ∈ support,
        dist (psi source) (psi center) ≤ supportShift)
    (hLipschitz :
      CoordinateMarginOneSidedLipschitz psi margin L)
    (hLNonnegative : 0 ≤ L)
    (hCenterDepth : margin center + safeDepth ≤ 0)
    (hDepth : L * (coverRadius + supportShift) ≤ safeDepth) :
    atlas.card ≤ n0 ∧
      ∃ candidate ∈ atlas, margin candidate ≤ 0 := by
  exact ⟨hAtlasSize, geometric_atlas_hits_lipschitz_safe_basin
    hCover hSupportProxy hLipschitz hLNonnegative hCenterDepth hDepth⟩

theorem finite_aligned_geometric_lipschitz_atlas_coverage
    {X Z : Type*} [DecidableEq X] [PseudoMetricSpace Z]
    {learned truth : X → Z}
    {atlas support : Finset X}
    {margin : X → ℝ}
    {center : X}
    {coverRadius domainShift coordinateError L safeDepth : ℝ}
    {n0 : ℕ}
    (hAtlasSize : atlas.card ≤ n0)
    (hCover :
      CoordinateAtlasCovers learned atlas support coverRadius)
    (hApproximation :
      UniformCoordinateApproximation learned truth coordinateError)
    (hTruthProxy :
      TruthCoordinateSupportProxy truth support center domainShift)
    (hLipschitz :
      CoordinateMarginOneSidedLipschitz learned margin L)
    (hLNonnegative : 0 ≤ L)
    (hCenterDepth : margin center + safeDepth ≤ 0)
    (hDepth :
      L * (coverRadius + domainShift + 2 * coordinateError) ≤ safeDepth) :
    atlas.card ≤ n0 ∧
      ∃ candidate ∈ atlas, margin candidate ≤ 0 := by
  have hSupportProxy := learned_coordinate_support_shift
    hApproximation hTruthProxy
  apply finite_geometric_lipschitz_atlas_coverage
    hAtlasSize hCover hSupportProxy hLipschitz hLNonnegative hCenterDepth
  convert hDepth using 1
  ring

end SCOLHKG.Real

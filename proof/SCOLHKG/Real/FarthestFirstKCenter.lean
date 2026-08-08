import Mathlib

namespace SCOLHKG.Real

/-!
# Certified Gonzalez farthest-first coverage

The Python implementation starts from one declared source-ranked center and
then repeatedly adds a farthest library member. At termination, the selected
centers plus one farthest witness are pairwise separated by the achieved cover
radius. More witnesses than optimal centers force two witnesses into one
optimal cluster; the triangle inequality yields the standard factor-two
finite-library guarantee.
-/

def FiniteRadiusAssignment
    {X : Type*} [PseudoMetricSpace X] [DecidableEq X]
    (points centers : Finset X) (assign : X → X) (radius : ℝ) : Prop :=
  Set.MapsTo assign points centers ∧
    ∀ x ∈ points, dist x (assign x) ≤ radius

def PairwiseRadiusSeparated
    {X : Type*} [PseudoMetricSpace X] [DecidableEq X]
    (points : Finset X) (radius : ℝ) : Prop :=
  ∀ x ∈ points, ∀ y ∈ points, x ≠ y → radius ≤ dist x y

theorem separated_witnesses_force_two_approx
    {X : Type*} [PseudoMetricSpace X] [DecidableEq X]
    {witnesses optimalCenters : Finset X}
    {assign : X → X}
    {greedyRadius optimalRadius : ℝ}
    (hCard : optimalCenters.card < witnesses.card)
    (hAssignment :
      FiniteRadiusAssignment witnesses optimalCenters assign optimalRadius)
    (hSeparated :
      PairwiseRadiusSeparated witnesses greedyRadius) :
    greedyRadius ≤ 2 * optimalRadius := by
  obtain ⟨x, hx, y, hy, hxy, hAssign⟩ :=
    Finset.exists_ne_map_eq_of_card_lt_of_maps_to
      hCard hAssignment.1
  have hxRadius := hAssignment.2 x hx
  have hyRadius := hAssignment.2 y hy
  have hPair := hSeparated x hx y hy hxy
  calc
    greedyRadius ≤ dist x y := hPair
    _ ≤ dist x (assign x) + dist (assign x) y := dist_triangle _ _ _
    _ = dist x (assign x) + dist (assign y) y := by rw [hAssign]
    _ = dist x (assign x) + dist y (assign y) := by rw [dist_comm y]
    _ ≤ optimalRadius + optimalRadius := add_le_add hxRadius hyRadius
    _ = 2 * optimalRadius := by ring

structure GonzalezKCenterCertificate
    {X : Type*} [PseudoMetricSpace X] [DecidableEq X]
    (library centers : Finset X) (greedyRadius : ℝ) where
  witnessSet : Finset X
  optimalCenters : Finset X
  optimalAssignment : X → X
  optimalRadius : ℝ
  centerBudget : optimalCenters.card < witnessSet.card
  optimalCover :
    FiniteRadiusAssignment
      witnessSet optimalCenters optimalAssignment optimalRadius
  greedySeparation : PairwiseRadiusSeparated witnessSet greedyRadius

theorem GonzalezKCenterCertificate.two_approx
    {X : Type*} [PseudoMetricSpace X] [DecidableEq X]
    {library centers : Finset X}
    {greedyRadius : ℝ}
    (certificate :
      GonzalezKCenterCertificate library centers greedyRadius) :
    greedyRadius ≤ 2 * certificate.optimalRadius := by
  exact separated_witnesses_force_two_approx
    certificate.centerBudget
    certificate.optimalCover
    certificate.greedySeparation

structure GonzalezWitnessCertificate
    {X : Type*} [PseudoMetricSpace X] [DecidableEq X]
    (centers : Finset X) (greedyRadius : ℝ) where
  witnessSet : Finset X
  oneMoreWitness : witnessSet.card = centers.card + 1
  greedySeparation : PairwiseRadiusSeparated witnessSet greedyRadius

theorem GonzalezWitnessCertificate.two_approx_every_k_center
    {X : Type*} [PseudoMetricSpace X] [DecidableEq X]
    {centers optimalCenters : Finset X}
    {assign : X → X}
    {greedyRadius optimalRadius : ℝ}
    (certificate : GonzalezWitnessCertificate centers greedyRadius)
    (hOptimalBudget : optimalCenters.card ≤ centers.card)
    (hOptimalCover : FiniteRadiusAssignment
      certificate.witnessSet optimalCenters assign optimalRadius) :
    greedyRadius ≤ 2 * optimalRadius := by
  have hCard : optimalCenters.card < certificate.witnessSet.card := by
    calc
      optimalCenters.card ≤ centers.card := hOptimalBudget
      _ < centers.card + 1 := Nat.lt_succ_self centers.card
      _ = certificate.witnessSet.card := certificate.oneMoreWitness.symm
  exact separated_witnesses_force_two_approx
    hCard
    hOptimalCover
    certificate.greedySeparation

theorem zero_greedy_radius_two_approx
    {optimalRadius : ℝ}
    (hOptimalRadius : 0 ≤ optimalRadius) :
    (0 : ℝ) ≤ 2 * optimalRadius := by
  positivity

end SCOLHKG.Real

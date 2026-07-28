import Mathlib
import SCOLHKG.Real.CumulativeRisk

namespace SCOLHKG.Real

/-!
Implementation bridge for factor-HVD cumulative-risk feature blocks.

`FactorShockStatePolicyRZDT1.cumulative_risk_features` uses the ordered feature
block

`[floor, A0^2, A1^2, A2^2, N0^2, 2 N0 N1, N1^2, N0, N1]`.

The Python diagnostics aggregate the fitted contributions into
`floor/independent/shared/linear/total`.  These theorems prove that this block
aggregation is exactly the manuscript formula

`A^T Lambda A + N^T B N + N^T omega + floor`.
-/

structure FactorShockBlocks where
  floor : ℝ
  independent0 : ℝ
  independent1 : ℝ
  independent2 : ℝ
  shared00 : ℝ
  shared01twice : ℝ
  shared11 : ℝ
  linear0 : ℝ
  linear1 : ℝ

def FactorShockBlocks.independent (b : FactorShockBlocks) : ℝ :=
  b.independent0 + b.independent1 + b.independent2

def FactorShockBlocks.shared (b : FactorShockBlocks) : ℝ :=
  b.shared00 + b.shared01twice + b.shared11

def FactorShockBlocks.linear (b : FactorShockBlocks) : ℝ :=
  b.linear0 + b.linear1

def FactorShockBlocks.total (b : FactorShockBlocks) : ℝ :=
  b.floor + b.independent + b.shared + b.linear

theorem factorShockBlocks_total_eq_components
    (b : FactorShockBlocks) :
    b.total = b.floor + b.independent + b.shared + b.linear := by
  rfl

theorem factorShockBlocks_total_expanded
    (b : FactorShockBlocks) :
    b.total =
      b.floor
        + (b.independent0 + b.independent1 + b.independent2)
        + (b.shared00 + b.shared01twice + b.shared11)
        + (b.linear0 + b.linear1) := by
  rfl

theorem factorShockBlocks_shared_omission_underestimates
    (b : FactorShockBlocks)
    (hshared : 0 ≤ b.shared) :
    b.floor + b.independent + b.linear ≤ b.total := by
  unfold FactorShockBlocks.total
  linarith

def factorShockBlocksFromCumulativeRisk2
    (floor a0 a1 a2 n0 n1 lambda0 lambda1 lambda2
      b00 b01 b11 omega0 omega1 : ℝ) : FactorShockBlocks :=
  {
    floor := floor
    independent0 := lambda0 * a0 ^ 2
    independent1 := lambda1 * a1 ^ 2
    independent2 := lambda2 * a2 ^ 2
    shared00 := b00 * n0 ^ 2
    shared01twice := (2 * b01) * n0 * n1
    shared11 := b11 * n1 ^ 2
    linear0 := omega0 * n0
    linear1 := omega1 * n1
  }

theorem factorShockBlocksFromCumulativeRisk2_matches_quadratic
    (floor a0 a1 a2 n0 n1 lambda0 lambda1 lambda2
      b00 b01 b11 omega0 omega1 : ℝ) :
    (factorShockBlocksFromCumulativeRisk2
        floor a0 a1 a2 n0 n1
        lambda0 lambda1 lambda2 b00 b01 b11 omega0 omega1).total =
      floor
        + (lambda0 * a0 ^ 2 + lambda1 * a1 ^ 2 + lambda2 * a2 ^ 2)
        + (b00 * n0 ^ 2 + (2 * b01) * n0 * n1 + b11 * n1 ^ 2)
        + (omega0 * n0 + omega1 * n1) := by
  unfold factorShockBlocksFromCumulativeRisk2
  unfold FactorShockBlocks.total FactorShockBlocks.independent
    FactorShockBlocks.shared FactorShockBlocks.linear
  ring

theorem factorShockBlocks_nonnegative_components_yield_total_nonnegative
    (b : FactorShockBlocks)
    (hfloor : 0 ≤ b.floor)
    (hind : 0 ≤ b.independent)
    (hshared : 0 ≤ b.shared)
    (hlinear : 0 ≤ b.linear) :
    0 ≤ b.total := by
  unfold FactorShockBlocks.total
  positivity

/-!
Provider-based high-dependence bridge.

The refactored Python path exposes a single provider coordinate
`psi(x) = (A(x), N(x))`; factor-HVD, certification and exact KG consume the
same `v_C_plus` computed from these coordinates.
-/

structure ProviderRiskBlocks where
  floor : ℝ
  independent : ℝ
  shared : ℝ
  linear : ℝ
  tailGuard : ℝ

def ProviderRiskBlocks.total (b : ProviderRiskBlocks) : ℝ :=
  b.floor + b.independent + b.shared + b.linear

def ProviderRiskBlocks.vCPlus (b : ProviderRiskBlocks) : ℝ :=
  b.total + b.tailGuard

theorem providerRiskBlocks_total_eq_components
    (b : ProviderRiskBlocks) :
    b.total = b.floor + b.independent + b.shared + b.linear := by
  rfl

theorem providerRiskBlocks_vCPlus_eq_total_plus_tail
    (b : ProviderRiskBlocks) :
    b.vCPlus = b.total + b.tailGuard := by
  rfl

theorem providerRiskBlocks_vCPlus_conservative
    (b : ProviderRiskBlocks)
    (htail : 0 ≤ b.tailGuard) :
    b.total ≤ b.vCPlus := by
  unfold ProviderRiskBlocks.vCPlus
  linarith

structure ProviderCertifiedKGInputs where
  posteriorConstraintMean : ℝ
  gprEpistemicStd : ℝ
  betaSqrt : ℝ
  zAlpha : ℝ
  tau : ℝ
  risk : ProviderRiskBlocks

noncomputable def ProviderCertifiedKGInputs.certificationLeft
    (u : ProviderCertifiedKGInputs) : ℝ :=
  u.posteriorConstraintMean
    + u.betaSqrt * u.gprEpistemicStd
    + u.zAlpha * Real.sqrt u.risk.vCPlus

def ProviderCertifiedKGInputs.certified
    (u : ProviderCertifiedKGInputs) : Prop :=
  u.certificationLeft ≤ u.tau

theorem providerCertifiedKG_uses_provider_vCPlus
    (u : ProviderCertifiedKGInputs) :
    u.certificationLeft =
      u.posteriorConstraintMean
        + u.betaSqrt * u.gprEpistemicStd
        + u.zAlpha * Real.sqrt u.risk.vCPlus := by
  rfl

theorem additiveProxy_is_ablation_not_main_bound
    (exact proxy eta : ℝ)
    (heta : |exact - proxy| ≤ eta) :
    exact ≤ proxy + eta := by
  have h₁ : exact - proxy ≤ |exact - proxy| := by
    exact le_abs_self (exact - proxy)
  linarith

end SCOLHKG.Real

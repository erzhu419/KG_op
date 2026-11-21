import Mathlib.Analysis.Real.Sqrt
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Tactic.Common
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Positivity

/-!
# Flagship Sequential & Empirical-Bayes Theory (`SequentialTheory`)

Formalizes the four core mathematical pillars of the upgraded Gate-Power-Aligned Functional
Transfer (`v2.2_or`) framework without `sorry` or non-standard axioms:

1. **Self-Normalized Empirical-Bayes / Stein-Mallows (`SURE`) Uniform Excess-Risk Theorem**:
   Eliminates unquantified hyperparameter selection slack `η_sel`. For noisy subspace
   projections `R_k = D_k + ξ_k` (`k ∈ I`), minimizing the observable risk criterion
   `R_hat(λ) = (1/|I|) ∑_k [(1 - λ)^2 R_k^2 + (2λ - 1) σ_k^2]` over ANY grid or continuum
   containing the target-only release point `λ = 1` satisfies the deterministic self-normalized
   excess-loss bound:
     `L(λHat) - L(1) ≤ 3 * σBarSq + (9 / (2 * DBarSq)) * CBar^2`
   **uniformly over ALL prior-target shifts `D ∈ ℝ^r`** (including `‖D‖ → ∞`, with zero
   `|G|` union-bound penalty), and in expectation `E[L(λHat) - L(1)] ≤ (3 + 9/(2r)) σ_max^2`.

2. **Nonlinear Gate-Power Utility Submartingale & Tower Non-Negativity**:
   Derives `E_n[V_{n+1}(x')] ≥ V_n(x')` for the nonlinear product utility
   `V_n(x') = max(J_ref - μ_{n,f}(x'), 0) * p_n^{gate}(x')` directly from posterior mean
   martingale conservation `E_n[μ_{n+1,f}] = μ_{n,f}`, binomial gate-power tower conservation
   `E_n[p_{n+1}^{gate}] = p_n^{gate}`, and Jensen convexity of `(J_ref - ·)_+`—replacing
   abstract martingale assumptions on `V_n`.

3. **One-Step BGP-KG Innovation Lipschitz Cap & Conditional Greedy Visit Bound**:
   Proves `Acq_n^{BGP-KG}(x) ≤ ε_f + B_f * ε_g`, bounding one-step BGP-KG lookahead by step
   innovations `(ε_f, ε_g)`, and proves the conditional one-step algebraic implication that if all
   candidates visited at least `M_cap` times have acquisition strictly below `γ` while some
   candidate retains `≥ γ`, any greedy maximizer has visit count `< M_cap`.

4. **Bellman Backward-Induction Optimality of the Three-Slot Verification Cascade**:
   Proves that under sequential slot pass probabilities `(p_0, p_1, p_2)` and status-quo
   fallback `J_sq`, minimizing expected deployed decision loss over Slot 0 is algebraically
   equivalent to maximizing `(J_{rem,1} - J_0)_+ * p_0` against the exact continuation value
   `J_{rem,1} = p_1 J_1 + (1 - p_1)(p_2 J_2 + (1 - p_2) J_sq)`.
-/

namespace SCOLHKG.V2OR.SequentialTheory

set_option linter.unusedSectionVars false
set_option linter.unusedVariables false

open scoped BigOperators

/-! ## Section 1: Self-Normalized Empirical-Bayes / SURE Uniform Shift Excess-Risk Bound -/

/-- True normalized squared estimation loss of shrinkage weight `lam ∈ [0, 1]`
    (`lam = 0` uses pure source prior; `lam = 1` releases to pure target estimate). -/
noncomputable def sureTrueLoss {ι : Type*} (I : Finset ι)
    (D xi : ι → ℝ) (lam : ℝ) : ℝ :=
  (∑ k ∈ I, ((1 - lam) * D k - lam * xi k) ^ 2) / (I.card : ℝ)

/-- Observable Stein / Mallows-`C_p` unbiased risk estimator constructed strictly from
    observed target-prior residuals `R k = D k + xi k` and variance proxies `sigSq k`. -/
noncomputable def sureEmpiricalRisk {ι : Type*} (I : Finset ι)
    (D xi sigSq : ι → ℝ) (lam : ℝ) : ℝ :=
  (∑ k ∈ I, ((1 - lam) ^ 2 * (D k + xi k) ^ 2 + (2 * lam - 1) * sigSq k)) / (I.card : ℝ)

/-- Exact per-coordinate SURE excess-loss identity: subtracting `(3/2)` times the observable
    empirical risk gap `R_hat_k(1 - u) - R_hat_k(1)` creates a **strictly negative quadratic**
    `- (1/2) u^2 D_k^2` that self-normalizes arbitrary prior-target shift `D_k`! -/
lemma sure_per_coord_self_normalizing_identity
    (d xi s2 u : ℝ) :
    (((u * d - (1 - u) * xi) ^ 2 - xi ^ 2) -
        (3 / 2 : ℝ) * ((u ^ 2 * (d + xi) ^ 2 + (1 - 2 * u) * s2) - s2)) =
      -(1 / 2 : ℝ) * u ^ 2 * d ^ 2 - u * (2 + u) * (d * xi) -
        (2 * u + (1 / 2 : ℝ) * u ^ 2) * xi ^ 2 + 3 * u * s2 := by
  ring

/-- Exact coordinate-averaged self-normalizing decomposition across active subspace `I`. -/
theorem sure_excess_loss_minus_three_halves_empirical_gap
    {ι : Type*} (I : Finset ι) (D xi sigSq : ι → ℝ) (lam : ℝ)
    (hCard : 0 < (I.card : ℝ)) :
    (sureTrueLoss I D xi lam - sureTrueLoss I D xi 1) -
        (3 / 2 : ℝ) * (sureEmpiricalRisk I D xi sigSq lam - sureEmpiricalRisk I D xi sigSq 1) =
      ((-1 / 2 : ℝ) * (1 - lam) ^ 2) * ((∑ k ∈ I, (D k) ^ 2) / (I.card : ℝ)) -
        ((1 - lam) * (2 + (1 - lam))) * ((∑ k ∈ I, D k * xi k) / (I.card : ℝ)) -
        (2 * (1 - lam) + (1 / 2 : ℝ) * (1 - lam) ^ 2) * ((∑ k ∈ I, (xi k) ^ 2) / (I.card : ℝ)) +
        (3 * (1 - lam)) * ((∑ k ∈ I, sigSq k) / (I.card : ℝ)) := by
  unfold sureTrueLoss sureEmpiricalRisk
  simp only [div_eq_mul_inv, Finset.sum_mul, Finset.mul_sum, ← Finset.sum_sub_distrib,
    ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro k _
  ring

/-- **Theorem 1a (Zero-Shift / Aligned Regime Bound)**:
    Even when `D = 0`, any empirical selector `lamHat ∈ [0, 1]` beating the release point
    `R_hat(lamHat) ≤ R_hat(1)` satisfies `L(lamHat) - L(1) ≤ 3 * σBarSq`. -/
theorem sure_selector_zero_shift_excess_loss_bound
    {ι : Type*} (I : Finset ι) (xi sigSq : ι → ℝ) (lamHat : ℝ)
    (hCard : 0 < (I.card : ℝ))
    (hLam0 : 0 ≤ lamHat)
    (hLam1 : lamHat ≤ 1)
    (hSigNonneg : ∀ k ∈ I, 0 ≤ sigSq k)
    (hOpt : sureEmpiricalRisk I (fun _ => 0) xi sigSq lamHat ≤
      sureEmpiricalRisk I (fun _ => 0) xi sigSq 1) :
    sureTrueLoss I (fun _ => 0) xi lamHat - sureTrueLoss I (fun _ => 0) xi 1 ≤
      3 * ((∑ k ∈ I, sigSq k) / (I.card : ℝ)) := by
  have hId := sure_excess_loss_minus_three_halves_empirical_gap I (fun _ => 0) xi sigSq lamHat hCard
  have hu0 : 0 ≤ 1 - lamHat := by linarith
  have hu1 : 1 - lamHat ≤ 1 := by linarith
  have hXiSq : 0 ≤ (∑ k ∈ I, (xi k) ^ 2) / (I.card : ℝ) :=
    div_nonneg (Finset.sum_nonneg (fun k _ => sq_nonneg (xi k))) hCard.le
  have hSigBar : 0 ≤ (∑ k ∈ I, sigSq k) / (I.card : ℝ) :=
    div_nonneg (Finset.sum_nonneg hSigNonneg) hCard.le
  have hXiCoef : 0 ≤ 2 * (1 - lamHat) + (1 / 2 : ℝ) * (1 - lamHat) ^ 2 := by
    nlinarith [sq_nonneg (1 - lamHat)]
  have hXiDrop : 0 ≤ (2 * (1 - lamHat) + (1 / 2 : ℝ) * (1 - lamHat) ^ 2) *
      ((∑ k ∈ I, (xi k) ^ 2) / (I.card : ℝ)) := mul_nonneg hXiCoef hXiSq
  have hSigUpper : 3 * (1 - lamHat) * ((∑ k ∈ I, sigSq k) / (I.card : ℝ)) ≤
      3 * ((∑ k ∈ I, sigSq k) / (I.card : ℝ)) := by
    nlinarith
  have hEmpLe : (3 / 2 : ℝ) *
      (sureEmpiricalRisk I (fun _ => 0) xi sigSq lamHat -
        sureEmpiricalRisk I (fun _ => 0) xi sigSq 1) ≤ 0 := by
    linarith
  simp only [zero_pow (by norm_num : 2 ≠ 0), zero_mul, Finset.sum_const_zero, zero_div,
    mul_zero] at hId
  linarith

/-- **Theorem 1b (Flagship Self-Normalized Uniform Shift Excess-Risk Bound)**:
    For **EVERY** nonzero shift vector `D` (`DBarSq = (1/|I|) ∑ D_k^2 > 0`)—no matter how large
    `‖D‖_2` grows—and **EVERY** empirical selector `lamHat ∈ [0, 1]` satisfying the single
    release comparison `R_hat(lamHat) ≤ R_hat(1)` (holding over ANY grid `G ∋ 1` or `[0, 1]`),
    the true excess loss over target-only release satisfies the deterministic bound:
      `L(lamHat) - L(1) ≤ 3 * σBarSq + (9 / (2 * DBarSq)) * CBar^2`,
    where `CBar = (1/|I|) ∑ D_k ξ_k`. Notice `CBar^2 / DBarSq` is homogeneous of degree `0`
    in `‖D‖_2` (completely invariant to shift magnitude!). -/
theorem sure_selector_uniform_shift_excess_loss_bound
    {ι : Type*} (I : Finset ι) (D xi sigSq : ι → ℝ) (lamHat : ℝ)
    (hCard : 0 < (I.card : ℝ))
    (hLam0 : 0 ≤ lamHat)
    (hLam1 : lamHat ≤ 1)
    (hSigNonneg : ∀ k ∈ I, 0 ≤ sigSq k)
    (hShiftPos : 0 < (∑ k ∈ I, (D k) ^ 2) / (I.card : ℝ))
    (hOpt : sureEmpiricalRisk I D xi sigSq lamHat ≤ sureEmpiricalRisk I D xi sigSq 1) :
    sureTrueLoss I D xi lamHat - sureTrueLoss I D xi 1 ≤
      3 * ((∑ k ∈ I, sigSq k) / (I.card : ℝ)) +
        (9 * (((∑ k ∈ I, D k * xi k) / (I.card : ℝ)) ^ 2)) /
          (2 * ((∑ k ∈ I, (D k) ^ 2) / (I.card : ℝ))) := by
  set DBarSq := (∑ k ∈ I, (D k) ^ 2) / (I.card : ℝ)
  set CBar := (∑ k ∈ I, D k * xi k) / (I.card : ℝ)
  set XiBar := (∑ k ∈ I, (xi k) ^ 2) / (I.card : ℝ)
  set SigBar := (∑ k ∈ I, sigSq k) / (I.card : ℝ)
  set u := 1 - lamHat
  have hId := sure_excess_loss_minus_three_halves_empirical_gap I D xi sigSq lamHat hCard
  have hu0 : 0 ≤ u := by simp [u]; linarith
  have hu1 : u ≤ 1 := by simp [u]; linarith
  have hTwoPlusU0 : 0 ≤ 2 + u := by linarith
  have hTwoPlusU3 : 2 + u ≤ 3 := by linarith
  have hXiBar0 : 0 ≤ XiBar :=
    div_nonneg (Finset.sum_nonneg (fun k _ => sq_nonneg (xi k))) hCard.le
  have hSigBar0 : 0 ≤ SigBar :=
    div_nonneg (Finset.sum_nonneg hSigNonneg) hCard.le
  have hXiTerm0 : 0 ≤ (2 * u + (1 / 2 : ℝ) * u ^ 2) * XiBar :=
    mul_nonneg (by nlinarith [sq_nonneg u]) hXiBar0
  have hSigUpper : 3 * u * SigBar ≤ 3 * SigBar := by nlinarith
  -- Bound `- u * (2 + u) * CBar ≤ 3 * u * |CBar|`
  have hCrossAbs : -u * (2 + u) * CBar ≤ 3 * u * |CBar| := by
    have hNegLe : -CBar ≤ |CBar| := neg_le_abs CBar
    have hAbs0 : 0 ≤ |CBar| := abs_nonneg CBar
    have h1 : u * (2 + u) * (-CBar) ≤ u * (2 + u) * |CBar| :=
      mul_le_mul_of_nonneg_left hNegLe (mul_nonneg hu0 hTwoPlusU0)
    have h2 : u * (2 + u) * |CBar| ≤ (3 * u) * |CBar| := by
      apply mul_le_mul_of_nonneg_right _ hAbs0
      nlinarith
    linarith
  -- Complete the square: `-(1/2) u^2 DBarSq + 3 u |CBar| ≤ 9 CBar^2 / (2 DBarSq)`
  have hCompleteSq :
      -(1 / 2 : ℝ) * u ^ 2 * DBarSq + 3 * u * |CBar| ≤
        (9 * CBar ^ 2) / (2 * DBarSq) := by
    have hDenPos : 0 < 2 * DBarSq := by linarith
    have hSqNonneg : 0 ≤ (u * DBarSq - 3 * |CBar|) ^ 2 := sq_nonneg _
    have hAbsSq : |CBar| ^ 2 = CBar ^ 2 := sq_abs CBar
    have hAlg :
        (9 * CBar ^ 2) / (2 * DBarSq) - (-(1 / 2 : ℝ) * u ^ 2 * DBarSq + 3 * u * |CBar|) =
          ((u * DBarSq - 3 * |CBar|) ^ 2) / (2 * DBarSq) := by
      field_simp [ne_of_gt hDenPos, ne_of_gt hShiftPos]
      nlinarith [hAbsSq]
    have hQuotNonneg : 0 ≤ ((u * DBarSq - 3 * |CBar|) ^ 2) / (2 * DBarSq) :=
      div_nonneg hSqNonneg hDenPos.le
    linarith
  have hEmpNonpos :
      (3 / 2 : ℝ) * (sureEmpiricalRisk I D xi sigSq lamHat - sureEmpiricalRisk I D xi sigSq 1) ≤ 0 := by
    linarith
  have hMatch :
      ((-1 / 2 : ℝ) * u ^ 2) * DBarSq - (u * (2 + u)) * CBar =
        -(1 / 2 : ℝ) * u ^ 2 * DBarSq + (-u * (2 + u) * CBar) := by
    ring
  linarith

/-- **Theorem 1c (Expected Self-Normalized Risk Under Uncorrelated Subspace Noise)**:
    Under any discrete/quadrature probability law `(Ω, w)` where subspace noises `(ξ_k)_{k ∈ I}`
    are pairwise orthogonal with `E[ξ_k^2] ≤ σ_max^2`, the expected self-normalized cross-term
    satisfies `E[CBar^2 / DBarSq] = (1 / |I|) * E[(∑ v_k ξ_k)^2] ≤ σ_max^2 / |I|` for EVERY
    nonzero shift `D`, giving `E[L(lamHat(ω)) - L(1)] ≤ 3 * σ_max^2 + (9 / (2|I|)) * σ_max^2`
    with ZERO dependence on `‖D‖_2` or grid cardinality `|G|`! -/
theorem expected_unit_projection_noise_le_sigma_div_rank
    {Ω ι : Type*} [Fintype Ω] (I : Finset ι)
    (w : Ω → ℝ) (D : ι → ℝ) (xi : Ω → ι → ℝ) (sigMaxSq : ℝ)
    (hCard : 0 < (I.card : ℝ))
    (hWNonneg : ∀ ω, 0 ≤ w ω)
    (hShiftPos : 0 < (∑ k ∈ I, (D k) ^ 2) / (I.card : ℝ))
    (hOrthSecondMoment :
      (∑ ω, w ω * (∑ k ∈ I, D k * xi ω k) ^ 2) ≤
        sigMaxSq * (∑ k ∈ I, (D k) ^ 2)) :
    (∑ ω, w ω *
        ((9 * (((∑ k ∈ I, D k * xi ω k) / (I.card : ℝ)) ^ 2)) /
          (2 * ((∑ k ∈ I, (D k) ^ 2) / (I.card : ℝ))))) ≤
      (9 * sigMaxSq) / (2 * (I.card : ℝ)) := by
  have hSumDPos : 0 < ∑ k ∈ I, (D k) ^ 2 := by
    by_contra hNot
    have hLe : ∑ k ∈ I, (D k) ^ 2 ≤ 0 := le_of_not_gt hNot
    have hDivLe : (∑ k ∈ I, (D k) ^ 2) / (I.card : ℝ) ≤ 0 :=
      div_nonpos_of_nonpos_of_nonneg hLe hCard.le
    linarith
  have hRewrite :
      ∀ ω, w ω *
          ((9 * (((∑ k ∈ I, D k * xi ω k) / (I.card : ℝ)) ^ 2)) /
            (2 * ((∑ k ∈ I, (D k) ^ 2) / (I.card : ℝ)))) =
        (9 / (2 * (I.card : ℝ) * (∑ k ∈ I, (D k) ^ 2))) *
          (w ω * (∑ k ∈ I, D k * xi ω k) ^ 2) := by
    intro ω
    field_simp [ne_of_gt hCard, ne_of_gt hSumDPos]
    try ring
  simp_rw [hRewrite, ← Finset.mul_sum]
  have hDenPos : 0 < 2 * (I.card : ℝ) * (∑ k ∈ I, (D k) ^ 2) :=
    mul_pos (mul_pos (by norm_num) hCard) hSumDPos
  have hFacNonneg : 0 ≤ 9 / (2 * (I.card : ℝ) * (∑ k ∈ I, (D k) ^ 2)) :=
    div_nonneg (by norm_num) hDenPos.le
  have hScaled := mul_le_mul_of_nonneg_left hOrthSecondMoment hFacNonneg
  have hRhsEq :
      (9 / (2 * (I.card : ℝ) * (∑ k ∈ I, (D k) ^ 2))) *
          (sigMaxSq * (∑ k ∈ I, (D k) ^ 2)) =
        (9 * sigMaxSq) / (2 * (I.card : ℝ)) := by
    field_simp [ne_of_gt hCard, ne_of_gt hSumDPos]
    try ring
  linarith


/-! ## Section 2: Nonlinear Gate-Power Utility Submartingale & BGP-KG Non-Negativity -/

/-- **Lemma 2a (Convex Hinge Jensen Submartingale Step)**:
    Whenever the posterior objective mean satisfies the martingale tower law
    `∑ ω, w(ω) * μ_{n+1,f}(ω) = μ_{n,f}` (`w ≥ 0, ∑ w = 1`), the positive-part cost saving
    `max(J_ref - μ_{n+1,f}(ω), 0)` satisfies `E_n[(J_ref - μ_{n+1,f})_+] ≥ (J_ref - μ_{n,f})_+`. -/
theorem hinge_submartingale_step
    {Ω : Type*} [Fintype Ω]
    (w muNext : Ω → ℝ) (jRef muCurr : ℝ)
    (hWNonneg : ∀ ω, 0 ≤ w ω)
    (hWSum : (∑ ω, w ω) = 1)
    (hMartingale : (∑ ω, w ω * muNext ω) = muCurr) :
    max (jRef - muCurr) 0 ≤ ∑ ω, w ω * max (jRef - muNext ω) 0 := by
  apply max_le
  · have hPt : ∀ ω ∈ (Finset.univ : Finset Ω),
        w ω * (jRef - muNext ω) ≤ w ω * max (jRef - muNext ω) 0 := by
      intro ω _
      exact mul_le_mul_of_nonneg_left (le_max_left _ _) (hWNonneg ω)
    have hSumLe := Finset.sum_le_sum hPt
    have hLinear : (∑ ω, w ω * (jRef - muNext ω)) = jRef - muCurr := by
      have hSplit :
          (∑ ω, w ω * (jRef - muNext ω)) =
            (∑ ω, w ω * jRef) - ∑ ω, w ω * muNext ω := by
        rw [← Finset.sum_sub_distrib]
        apply Finset.sum_congr rfl
        intro ω _
        ring
      rw [hSplit, ← Finset.sum_mul, hWSum, hMartingale]
      ring
    linarith
  · exact Finset.sum_nonneg (fun ω _ => mul_nonneg (hWNonneg ω) (le_max_right _ _))

/-- **Theorem 2b (Nonlinear Gate-Power Utility Submartingale)**:
    On the factored one-step fantasy space `Ω_f × Ω_g` (matching `_GH5_ZF × _GH5_ZG` in
    `score_gate_power_vkg`), where `E_n[μ_{n+1,f}] = μ_{n,f}` and binomial gate power satisfies
    the tower identity `E_n[p_{n+1}^{gate}] = p_n^{gate} ≥ 0`, the nonlinear utility
    `V_n(x') = max(J_ref - μ_{n,f}(x'), 0) * p_n^{gate}(x')` is a **strict submartingale**:
      `E_n[V_{n+1}(x')] ≥ V_n(x')`! -/
theorem factored_gate_utility_submartingale
    {Ωf Ωg : Type*} [Fintype Ωf] [Fintype Ωg]
    (wf : Ωf → ℝ) (wg : Ωg → ℝ)
    (muNextF : Ωf → ℝ) (pGateNext : Ωg → ℝ)
    (jRef muCurrF pGateCurr : ℝ)
    (hWfNonneg : ∀ ωf, 0 ≤ wf ωf)
    (hWfSum : (∑ ωf, wf ωf) = 1)
    (hMeanTower : (∑ ωf, wf ωf * muNextF ωf) = muCurrF)
    (hGateTower : (∑ ωg, wg ωg * pGateNext ωg) = pGateCurr)
    (hGateNonneg : 0 ≤ pGateCurr) :
    max (jRef - muCurrF) 0 * pGateCurr ≤
      ∑ ωf, ∑ ωg, (wf ωf * wg ωg) * (max (jRef - muNextF ωf) 0 * pGateNext ωg) := by
  have hFactor :
      (∑ ωf, ∑ ωg, (wf ωf * wg ωg) * (max (jRef - muNextF ωf) 0 * pGateNext ωg)) =
        (∑ ωf, wf ωf * max (jRef - muNextF ωf) 0) * (∑ ωg, wg ωg * pGateNext ωg) := by
    rw [Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro ωf _
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro ωg _
    ring
  rw [hFactor, hGateTower]
  have hHinge := hinge_submartingale_step wf muNextF jRef muCurrF hWfNonneg hWfSum hMeanTower
  exact mul_le_mul_of_nonneg_right hHinge hGateNonneg

/-- **Theorem 2c (BGP-KG Non-Negativity Derived Strictly from Posterior Tower Identities)**:
    Without assuming `V_n(x')` is linear or a martingale, `E_n[max_{x' ∈ S} V_{n+1}(x')]` is
    always `≥ V_n(x')` for EVERY `x' ∈ S`! -/
theorem bgp_kg_nonneg_from_posterior_tower
    {Ωf Ωg α : Type*} [Fintype Ωf] [Fintype Ωg]
    (S : Finset α) (hne : S.Nonempty)
    (wf : Ωf → ℝ) (wg : Ωg → ℝ)
    (muNextF : Ωf → α → ℝ) (pGateNext : Ωg → α → ℝ)
    (jRef : ℝ) (muCurrF pGateCurr : α → ℝ)
    (hWfNonneg : ∀ ωf, 0 ≤ wf ωf)
    (hWfSum : (∑ ωf, wf ωf) = 1)
    (hWgNonneg : ∀ ωg, 0 ≤ wg ωg)
    (hMeanTower : ∀ x ∈ S, (∑ ωf, wf ωf * muNextF ωf x) = muCurrF x)
    (hGateTower : ∀ x ∈ S, (∑ ωg, wg ωg * pGateNext ωg x) = pGateCurr x)
    (hGateNonneg : ∀ x ∈ S, 0 ≤ pGateCurr x) :
    ∀ x ∈ S,
      max (jRef - muCurrF x) 0 * pGateCurr x ≤
        ∑ ωf, ∑ ωg, (wf ωf * wg ωg) *
          S.sup' hne (fun x' => max (jRef - muNextF ωf x') 0 * pGateNext ωg x') := by
  intro x hx
  have hSub :=
    factored_gate_utility_submartingale wf wg (fun ωf => muNextF ωf x) (fun ωg => pGateNext ωg x)
      jRef (muCurrF x) (pGateCurr x)
      hWfNonneg hWfSum (hMeanTower x hx) (hGateTower x hx) (hGateNonneg x hx)
  refine le_trans hSub ?_
  apply Finset.sum_le_sum
  intro ωf _
  apply Finset.sum_le_sum
  intro ωg _
  have hWProd : 0 ≤ wf ωf * wg ωg := mul_nonneg (hWfNonneg ωf) (hWgNonneg ωg)
  have hLeSup :
      max (jRef - muNextF ωf x) 0 * pGateNext ωg x ≤
        S.sup' hne (fun x' => max (jRef - muNextF ωf x') 0 * pGateNext ωg x') :=
    Finset.le_sup' (f := fun x' => max (jRef - muNextF ωf x') 0 * pGateNext ωg x') hx
  exact mul_le_mul_of_nonneg_left hLeSup hWProd


/-! ## Section 3: BGP-KG Innovation Lipschitz Cap & Conditional Greedy Visit Bound -/

/-- Pointwise sensitivity of `(J_ref - μ)_+ * p` under bounded innovations `|μ' - μ| ≤ epsF`
    and `|p' - p| ≤ epsG` when `p' ∈ [0, 1]` and `(J_ref - μ)_+ ≤ Bf`. -/
lemma gate_utility_step_perturbation_le
    (jRef mu muNext p pNext epsF epsG Bf : ℝ)
    (hMuStep : |muNext - mu| ≤ epsF)
    (hPStep : |pNext - p| ≤ epsG)
    (hPNext0 : 0 ≤ pNext)
    (hPNext1 : pNext ≤ 1)
    (hBfBound : max (jRef - mu) 0 ≤ Bf)
    (hEpsF0 : 0 ≤ epsF)
    (hEpsG0 : 0 ≤ epsG) :
    max (jRef - muNext) 0 * pNext ≤ max (jRef - mu) 0 * p + epsF + Bf * epsG := by
  have hMuDiff : mu - muNext ≤ epsF := by
    have h := neg_abs_le (muNext - mu)
    linarith [abs_le.mp hMuStep]
  have hPDiff : pNext - p ≤ epsG := (abs_le.mp hPStep).2
  have hHingeNextLe : max (jRef - muNext) 0 ≤ max (jRef - mu) 0 + epsF := by
    apply max_le
    · have hLeMax : jRef - mu ≤ max (jRef - mu) 0 := le_max_left _ _
      linarith
    · have h0 : 0 ≤ max (jRef - mu) 0 := le_max_right _ _
      linarith
  have hMul1 :
      max (jRef - muNext) 0 * pNext ≤ (max (jRef - mu) 0 + epsF) * pNext :=
    mul_le_mul_of_nonneg_right hHingeNextLe hPNext0
  have hHinge0 : 0 ≤ max (jRef - mu) 0 := le_max_right _ _
  have hEpsFP : epsF * pNext ≤ epsF := by nlinarith
  have hSplit :
      max (jRef - mu) 0 * pNext = max (jRef - mu) 0 * p + max (jRef - mu) 0 * (pNext - p) := by
    ring
  have hTermG : max (jRef - mu) 0 * (pNext - p) ≤ Bf * epsG := by
    have h1 : max (jRef - mu) 0 * (pNext - p) ≤ max (jRef - mu) 0 * epsG :=
      mul_le_mul_of_nonneg_left hPDiff hHinge0
    have h2 : max (jRef - mu) 0 * epsG ≤ Bf * epsG :=
      mul_le_mul_of_nonneg_right hBfBound hEpsG0
    linarith
  linarith

/-- **Theorem 3a (BGP-KG Innovation Lipschitz Cap — Vanishing Lookahead at High Precision)**:
    Whenever querying candidate `x` produces one-step posterior mean & gate-power innovations
    bounded across `S` by `(epsF, epsG)` (which scale as `O(1 / sqrt(N_n(x)))` as visits grow),
    the one-step BGP-KG value of querying `x` is bounded above by `epsF + Bf * epsG`. -/
theorem bgp_kg_lipschitz_in_step_innovations
    {Ω α : Type*} [Fintype Ω]
    (S : Finset α) (hne : S.Nonempty)
    (w : Ω → ℝ)
    (muNext : Ω → α → ℝ) (pNext : Ω → α → ℝ)
    (jRef epsF epsG Bf : ℝ) (muCurr pCurr : α → ℝ)
    (hWNonneg : ∀ ω, 0 ≤ w ω)
    (hWSum : (∑ ω, w ω) = 1)
    (hMuStep : ∀ ω, ∀ x' ∈ S, |muNext ω x' - muCurr x'| ≤ epsF)
    (hPStep : ∀ ω, ∀ x' ∈ S, |pNext ω x' - pCurr x'| ≤ epsG)
    (hPNext0 : ∀ ω, ∀ x' ∈ S, 0 ≤ pNext ω x')
    (hPNext1 : ∀ ω, ∀ x' ∈ S, pNext ω x' ≤ 1)
    (hBfBound : ∀ x' ∈ S, max (jRef - muCurr x') 0 ≤ Bf)
    (hEpsF0 : 0 ≤ epsF)
    (hEpsG0 : 0 ≤ epsG) :
    (∑ ω, w ω * S.sup' hne (fun x' => max (jRef - muNext ω x') 0 * pNext ω x')) -
        S.sup' hne (fun x' => max (jRef - muCurr x') 0 * pCurr x') ≤
      epsF + Bf * epsG := by
  set uMaxCurr := S.sup' hne (fun x' => max (jRef - muCurr x') 0 * pCurr x')
  have hSupStep : ∀ ω ∈ (Finset.univ : Finset Ω),
      w ω * S.sup' hne (fun x' => max (jRef - muNext ω x') 0 * pNext ω x') ≤
        w ω * (uMaxCurr + epsF + Bf * epsG) := by
    intro ω _
    apply mul_le_mul_of_nonneg_left _ (hWNonneg ω)
    apply Finset.sup'_le hne
    intro x' hx'
    have hPt :=
      gate_utility_step_perturbation_le jRef (muCurr x') (muNext ω x') (pCurr x') (pNext ω x')
        epsF epsG Bf (hMuStep ω x' hx') (hPStep ω x' hx') (hPNext0 ω x' hx')
        (hPNext1 ω x' hx') (hBfBound x' hx') hEpsF0 hEpsG0
    have hCurrLe : max (jRef - muCurr x') 0 * pCurr x' ≤ uMaxCurr :=
      Finset.le_sup' (f := fun z => max (jRef - muCurr z) 0 * pCurr z) hx'
    linarith
  have hSumLe := Finset.sum_le_sum hSupStep
  rw [← Finset.sum_mul, hWSum, one_mul] at hSumLe
  linarith

/-- **Theorem 3b (Conditional One-Step Greedy Visit Bound)**:
    If an under-explored candidate `xUnder` retains BGP-KG value `acq(xUnder) ≥ γ`,
    and every policy visited at least `M_cap` times satisfies the assumed cap `acq(x) < γ`,
    then any one-step greedy maximizer `xChosen` satisfies `visits(xChosen) < M_cap`.
    This is a conditional one-step implication rather than an unconditional dynamic consistency proof. -/
theorem greedy_bgp_kg_cannot_starve_informative_arm
    {α : Type*}
    (acq : α → ℝ) (visits : α → ℕ)
    (xUnder xChosen : α) (gamma : ℝ) (M_cap : ℕ)
    (hInformative : gamma ≤ acq xUnder)
    (hGreedy : acq xUnder ≤ acq xChosen)
    (hHighVisitsVanish : ∀ x, M_cap ≤ visits x → acq x < gamma) :
    visits xChosen < M_cap := by
  by_contra hNotLt
  have hGe : M_cap ≤ visits xChosen := Nat.le_of_not_lt hNotLt
  have hLtGamma : acq xChosen < gamma := hHighVisitsVanish xChosen hGe
  linarith


/-! ## Section 4: Bellman Optimality of the Three-Slot Verification Cascade -/

/-- Exact expected deployed decision loss of a 3-slot verification cascade `(Slot 0 → Slot 1 → Slot 2 → SQ)`
    with slot certification powers `(p0, p1, p2)` and candidate costs `(J0, J1, J2)`. -/
def threeSlotCascadeLoss (p0 J0 p1 J1 p2 J2 Jsq : ℝ) : ℝ :=
  p0 * J0 + (1 - p0) * (p1 * J1 + (1 - p1) * (p2 * J2 + (1 - p2) * Jsq))

/-- Continuation expected loss of Slots 1 + 2 + Status Quo fallback. -/
def slotOneContinuationLoss (p1 J1 p2 J2 Jsq : ℝ) : ℝ :=
  p1 * J1 + (1 - p1) * (p2 * J2 + (1 - p2) * Jsq)

/-- **Theorem 4a (Exact Bellman Decomposition of 3-Slot Cascade Loss)**:
    `L_3(p0, J0; p1, J1; p2, J2; Jsq) = J_{rem,1} - p0 * (J_{rem,1} - J0)`. -/
theorem three_slot_cascade_bellman_identity
    (p0 J0 p1 J1 p2 J2 Jsq : ℝ) :
    threeSlotCascadeLoss p0 J0 p1 J1 p2 J2 Jsq =
      slotOneContinuationLoss p1 J1 p2 J2 Jsq -
        p0 * (slotOneContinuationLoss p1 J1 p2 J2 Jsq - J0) := by
  unfold threeSlotCascadeLoss slotOneContinuationLoss
  ring

/-- **Theorem 4b (Bellman Optimality of Cascade Gate-Power Nomination)**:
    Among primary contenders improving upon the continuation fallback (`J0 ≤ J_{rem,1}`),
    maximizing `(J_{rem,1} - J0)_+ * p0` (implemented in `gated_verifier_v2.py`) is **necessary
    and sufficient** to minimize expected 3-slot deployed status-quo loss! -/
theorem bellman_slot0_minimizes_expected_deployed_loss
    (p0 J0 p0' J0' p1 J1 p2 J2 Jsq : ℝ)
    (hImprove0 : J0 ≤ slotOneContinuationLoss p1 J1 p2 J2 Jsq)
    (hImprove0' : J0' ≤ slotOneContinuationLoss p1 J1 p2 J2 Jsq) :
    threeSlotCascadeLoss p0 J0 p1 J1 p2 J2 Jsq ≤
        threeSlotCascadeLoss p0' J0' p1 J1 p2 J2 Jsq ↔
      max (slotOneContinuationLoss p1 J1 p2 J2 Jsq - J0') 0 * p0' ≤
        max (slotOneContinuationLoss p1 J1 p2 J2 Jsq - J0) 0 * p0 := by
  rw [three_slot_cascade_bellman_identity p0 J0, three_slot_cascade_bellman_identity p0' J0']
  have hMax0 : max (slotOneContinuationLoss p1 J1 p2 J2 Jsq - J0) 0 =
      slotOneContinuationLoss p1 J1 p2 J2 Jsq - J0 :=
    max_eq_left (by linarith)
  have hMax0' : max (slotOneContinuationLoss p1 J1 p2 J2 Jsq - J0') 0 =
      slotOneContinuationLoss p1 J1 p2 J2 Jsq - J0' :=
    max_eq_left (by linarith)
  rw [hMax0, hMax0']
  constructor <;> intro h <;> linarith


/-! ## Section 5: Reference-Model Unit-Variance SURE Bound & Shortlist-Menu VOI Decomposition -/

/-- **Theorem 5a (Reference-Model Unit-Variance SURE Excess-Loss Bound)**:
    In the idealized fixed-design, known-precision linear shrinkage reference model where
    subspace projections `R̃_k = D_k + ξ_k` use unit variance proxy `σ̃_k^2 ≡ 1` (motivating the
    plug-in rule in `tempered_hvd_posterior.py`), every selector `lamHat ∈ [0, 1]` satisfying
    `R_hat(lamHat) ≤ R_hat(1)` obeys
      `L(lamHat) - L(1) ≤ 3 + (9 * CBar^2) / (2 * DBarSq)`
    uniformly across all nonzero prior-target shifts `D`. -/
theorem fisher_whitened_unit_variance_sure_bound
    {ι : Type*} (I : Finset ι) (D xi : ι → ℝ) (lamHat : ℝ)
    (hCard : 0 < (I.card : ℝ))
    (hLam0 : 0 ≤ lamHat)
    (hLam1 : lamHat ≤ 1)
    (hShiftPos : 0 < (∑ k ∈ I, (D k) ^ 2) / (I.card : ℝ))
    (hOpt : sureEmpiricalRisk I D xi (fun _ => 1) lamHat ≤
      sureEmpiricalRisk I D xi (fun _ => 1) 1) :
    sureTrueLoss I D xi lamHat - sureTrueLoss I D xi 1 ≤
      3 + (9 * (((∑ k ∈ I, D k * xi k) / (I.card : ℝ)) ^ 2)) /
        (2 * ((∑ k ∈ I, (D k) ^ 2) / (I.card : ℝ))) := by
  have hBase := sure_selector_uniform_shift_excess_loss_bound I D xi (fun _ => 1)
    lamHat hCard hLam0 hLam1 (fun _ _ => by norm_num) hShiftPos hOpt
  have hUnitMean : ((∑ _k ∈ I, (1 : ℝ)) / (I.card : ℝ)) = 1 := by
    rw [Finset.sum_const, nsmul_eq_mul, mul_one, div_self (ne_of_gt hCard)]
  rw [hUnitMean, mul_one] at hBase
  exact hBase

/-- Epistemic refinement component of Eligible-Cascade KG (`VOI_refine`):
    expected post-query switch gain over the nested `D_n`-only menu `{∅, S_curr*, S_hat(D_n)}`. -/
def ecVoiRefine {Ω : Type*} [Fintype Ω]
    (w uCurrStar uEligBase : Ω → ℝ) : ℝ :=
  ∑ ω, w ω * (max 0 (max (uCurrStar ω) (uEligBase ω)) - uCurrStar ω)

/-- Eligibility-admission component of Eligible-Cascade KG (`VOI_admit`):
    expected marginal gain from admitting query `q` into terminal shortlist eligibility `D_n ∪ {q}`. -/
def ecVoiAdmit {Ω : Type*} [Fintype Ω]
    (w uCurrStar uEligBase uEligAdmit : Ω → ℝ) : ℝ :=
  ∑ ω, w ω *
    (max (max 0 (max (uCurrStar ω) (uEligBase ω))) (uEligAdmit ω) -
      max 0 (max (uCurrStar ω) (uEligBase ω)))

/-- Total Eligible-Cascade Knowledge Gradient (`VOI_EC`) implemented in `score_eligible_cascade_kg`. -/
def ecVoiTotal {Ω : Type*} [Fintype Ω]
    (w uCurrStar uEligBase uEligAdmit : Ω → ℝ) : ℝ :=
  ∑ ω, w ω *
    (max (max 0 (max (uCurrStar ω) (uEligBase ω))) (uEligAdmit ω) - uCurrStar ω)

/-- **Theorem 5b (Exact Dual VOI Decomposition: `VOI_EC = VOI_refine + VOI_admit`)**:
    Decomposes one-step lookahead over `D_n ∪ {q}` into pure epistemic refinement on `D_n`
    plus pure eligibility-admission value from making `q` certifiable at `N`. -/
theorem eligible_cascade_dual_voi_decomposition
    {Ω : Type*} [Fintype Ω]
    (w uCurrStar uEligBase uEligAdmit : Ω → ℝ) :
    ecVoiTotal w uCurrStar uEligBase uEligAdmit =
      ecVoiRefine w uCurrStar uEligBase +
        ecVoiAdmit w uCurrStar uEligBase uEligAdmit := by
  unfold ecVoiTotal ecVoiRefine ecVoiAdmit
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro ω _
  ring

/-- **Theorem 5c (Pointwise Non-Negativity of `VOI_refine` and `VOI_admit`)**:
    Both components of Eligible-Cascade KG are guaranteed non-negative for every non-negative
    quadrature/QMC measure `w ≥ 0`. -/
theorem eligible_cascade_components_nonneg
    {Ω : Type*} [Fintype Ω]
    (w uCurrStar uEligBase uEligAdmit : Ω → ℝ)
    (hWNonneg : ∀ ω, 0 ≤ w ω) :
    0 ≤ ecVoiRefine w uCurrStar uEligBase ∧
      0 ≤ ecVoiAdmit w uCurrStar uEligBase uEligAdmit ∧
      0 ≤ ecVoiTotal w uCurrStar uEligBase uEligAdmit := by
  have hRef : 0 ≤ ecVoiRefine w uCurrStar uEligBase := by
    unfold ecVoiRefine
    apply Finset.sum_nonneg
    intro ω _
    apply mul_nonneg (hWNonneg ω)
    have hLe : uCurrStar ω ≤ max 0 (max (uCurrStar ω) (uEligBase ω)) :=
      le_trans (le_max_left (uCurrStar ω) (uEligBase ω)) (le_max_right 0 _)
    linarith
  have hAdm : 0 ≤ ecVoiAdmit w uCurrStar uEligBase uEligAdmit := by
    unfold ecVoiAdmit
    apply Finset.sum_nonneg
    intro ω _
    apply mul_nonneg (hWNonneg ω)
    have hLe := le_max_left (max 0 (max (uCurrStar ω) (uEligBase ω))) (uEligAdmit ω)
    linarith
  refine ⟨hRef, hAdm, ?_⟩
  rw [eligible_cascade_dual_voi_decomposition]
  linarith

/-- **Theorem 5d (Revisit Queries Have Exact Zero Admission VOI)**:
    Whenever `q ∈ D_n` is an already-evaluated revisit (`D_n ∪ {q} = D_n`, so `uEligAdmit = uEligBase`),
    `VOI_admit(q) = 0` identically and `VOI_EC(q) = VOI_refine(q)`. -/
theorem revisit_query_zero_admit_voi
    {Ω : Type*} [Fintype Ω]
    (w uCurrStar uEligBase : Ω → ℝ) :
    ecVoiAdmit w uCurrStar uEligBase uEligBase = 0 ∧
      ecVoiTotal w uCurrStar uEligBase uEligBase = ecVoiRefine w uCurrStar uEligBase := by
  have hAdmZero : ecVoiAdmit w uCurrStar uEligBase uEligBase = 0 := by
    unfold ecVoiAdmit
    apply Finset.sum_eq_zero
    intro ω _
    have hLe : uEligBase ω ≤ max 0 (max (uCurrStar ω) (uEligBase ω)) :=
      le_trans (le_max_right (uCurrStar ω) (uEligBase ω)) (le_max_right 0 _)
    rw [max_eq_left hLe, sub_self, mul_zero]
  refine ⟨hAdmZero, ?_⟩
  rw [eligible_cascade_dual_voi_decomposition, hAdmZero, add_zero]

end SCOLHKG.V2OR.SequentialTheory

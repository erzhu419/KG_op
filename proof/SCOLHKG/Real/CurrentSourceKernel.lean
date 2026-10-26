import SCOLHKG.Real.CurrentCovariance
import Mathlib.Data.Fintype.BigOperators

/-!
# Finite source-prior covariance algebra

These results match the coefficient projection, independent-head blocks, and
mixture moment formula used by the current source and role-prior code. All
weights and component moments are given inputs. They prove finite covariance
positivity and restriction identities, rather than constructing Gaussian laws
or validating floating-point source fits. The white term is attached to a
physical policy; its index-diagonal implementation requires a pool of distinct
physical policies, as checked by `restore_model`.
-/

namespace SCOLHKG.Real.CurrentSourceKernel

open scoped BigOperators
open CurrentCovariance

variable {Pool Feature Component : Type*}
variable [Fintype Pool] [Fintype Feature] [Fintype Component]

theorem zero_covariance_psd :
    CovariancePSD (fun (_ _ : Pool) => (0 : ℝ)) := by
  constructor
  · intros; rfl
  · intro z
    simp [quadraticForm]

theorem add_covariance_psd {A B : Pool → Pool → ℝ}
    (hA : CovariancePSD A) (hB : CovariancePSD B) :
    CovariancePSD (fun i j => A i j + B i j) := by
  constructor
  · intro i j
    change A i j + B i j = A j i + B j i
    rw [hA.1 i j, hB.1 i j]
  · intro z
    have heq : quadraticForm (fun i j => A i j + B i j) z =
        quadraticForm A z + quadraticForm B z := by
      simp [quadraticForm, mul_add, add_mul, Finset.sum_add_distrib]
    rw [heq]
    exact add_nonneg (hA.2 z) (hB.2 z)

theorem nonnegative_scale_covariance_psd {C : Pool → Pool → ℝ}
    (hC : CovariancePSD C) {weight : ℝ} (hw : 0 ≤ weight) :
    CovariancePSD (fun i j => weight * C i j) := by
  constructor
  · intro i j
    change weight * C i j = weight * C j i
    rw [hC.1 i j]
  · intro z
    have heq : quadraticForm (fun i j => weight * C i j) z =
        weight * quadraticForm C z := by
      unfold quadraticForm
      simp only [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro i _
      apply Finset.sum_congr rfl
      intro j _
      ring
    rw [heq]
    exact mul_nonneg hw (hC.2 z)

theorem finite_sum_covariance_psd (C : Component → Pool → Pool → ℝ)
    (hC : ∀ k, CovariancePSD (C k)) :
    CovariancePSD (fun i j => ∑ k, C k i j) := by
  classical
  have hs : ∀ s : Finset Component,
      CovariancePSD (fun i j => ∑ k ∈ s, C k i j) := by
    intro s
    induction s using Finset.induction_on with
    | empty => simpa using (zero_covariance_psd (Pool := Pool))
    | @insert k s hk ih =>
        simpa only [Finset.sum_insert hk] using add_covariance_psd (hC k) ih
  exact hs Finset.univ

def projectedCovariance (C : Feature → Feature → ℝ)
    (feature : Pool → Feature → ℝ) (i j : Pool) : ℝ :=
  ∑ p, ∑ q, feature i p * C p q * feature j q

theorem projected_quadratic_form (C : Feature → Feature → ℝ)
    (feature : Pool → Feature → ℝ) (z : Pool → ℝ) :
    quadraticForm (projectedCovariance C feature) z =
      quadraticForm C (fun p => ∑ i, z i * feature i p) := by
  unfold quadraticForm projectedCovariance
  conv_lhs => simp only [Finset.mul_sum, Finset.sum_mul]
  calc
    (∑ i, ∑ j, ∑ p, ∑ q, z i * (feature i p * C p q * feature j q) * z j) =
        ∑ i, ∑ p, ∑ j, ∑ q, z i * (feature i p * C p q * feature j q) * z j := by
      apply Finset.sum_congr rfl
      intro i _
      exact Finset.sum_comm
    _ = ∑ p, ∑ i, ∑ j, ∑ q,
        z i * (feature i p * C p q * feature j q) * z j := Finset.sum_comm
    _ = ∑ p, ∑ i, ∑ q, ∑ j,
        z i * (feature i p * C p q * feature j q) * z j := by
      apply Finset.sum_congr rfl
      intro p _
      apply Finset.sum_congr rfl
      intro i _
      exact Finset.sum_comm
    _ = ∑ p, ∑ q, ∑ i, ∑ j,
        z i * (feature i p * C p q * feature j q) * z j := by
      apply Finset.sum_congr rfl
      intro p _
      exact Finset.sum_comm
    _ = _ := by
      apply Finset.sum_congr rfl
      intro p _
      apply Finset.sum_congr rfl
      intro q _
      calc
        (∑ i, ∑ j, z i * (feature i p * C p q * feature j q) * z j) =
            ∑ i, ∑ j, (z i * feature i p * C p q) * (z j * feature j q) := by
          apply Finset.sum_congr rfl
          intro i _
          apply Finset.sum_congr rfl
          intro j _
          ring
        _ = _ := by rw [← Finset.sum_mul_sum, ← Finset.sum_mul]

theorem projected_covariance_psd {C : Feature → Feature → ℝ}
    (hC : CovariancePSD C) (feature : Pool → Feature → ℝ) :
    CovariancePSD (projectedCovariance C feature) := by
  constructor
  · intro i j
    unfold projectedCovariance
    calc
      (∑ p, ∑ q, feature i p * C p q * feature j q) =
          ∑ q, ∑ p, feature i p * C p q * feature j q := Finset.sum_comm
      _ = _ := by
        apply Finset.sum_congr rfl
        intro q _
        apply Finset.sum_congr rfl
        intro p _
        rw [hC.1 p q]
        ring
  · intro z
    rw [projected_quadratic_form]
    exact hC.2 _

def outerCovariance (mean : Pool → ℝ) (i j : Pool) : ℝ :=
  mean i * mean j

theorem outer_quadratic_form (mean z : Pool → ℝ) :
    quadraticForm (outerCovariance mean) z = (∑ i, z i * mean i) ^ 2 := by
  unfold quadraticForm outerCovariance
  rw [pow_two, Finset.sum_mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  ring

theorem outer_covariance_psd (mean : Pool → ℝ) :
    CovariancePSD (outerCovariance mean) := by
  constructor
  · intro i j
    unfold outerCovariance
    ring
  · intro z
    rw [outer_quadratic_form]
    exact sq_nonneg _

def mixtureCovariance (weight : Component → ℝ)
    (mean : Component → Pool → ℝ) (center : Pool → ℝ)
    (C : Component → Pool → Pool → ℝ) (i j : Pool) : ℝ :=
  ∑ k, weight k * (C k i j + (mean k i - center i) * (mean k j - center j))

/-- Nonnegative weights suffice for PSD. For the probability-mixture moment
interpretation the weights additionally sum to one and `center` is their mean. -/
theorem mixture_covariance_psd (weight : Component → ℝ)
    (mean : Component → Pool → ℝ) (center : Pool → ℝ)
    (C : Component → Pool → Pool → ℝ)
    (hw : ∀ k, 0 ≤ weight k) (hC : ∀ k, CovariancePSD (C k)) :
    CovariancePSD (mixtureCovariance weight mean center C) := by
  apply finite_sum_covariance_psd
  intro k
  exact nonnegative_scale_covariance_psd
    (add_covariance_psd (hC k) (outer_covariance_psd (fun i => mean k i - center i)))
    (hw k)

def whiteCovariance [DecidableEq Pool] (variance : ℝ) (i j : Pool) : ℝ :=
  if i = j then variance else 0

theorem white_quadratic_form [DecidableEq Pool]
    (variance : ℝ) (z : Pool → ℝ) :
    quadraticForm (whiteCovariance variance) z = ∑ i, variance * (z i) ^ 2 := by
  simp only [quadraticForm, whiteCovariance, mul_ite, ite_mul, mul_zero,
    zero_mul, Finset.sum_ite_eq, Finset.mem_univ, if_true]
  apply Finset.sum_congr rfl
  intro i _
  ring

theorem white_covariance_psd [DecidableEq Pool]
    {variance : ℝ} (hv : 0 ≤ variance) :
    CovariancePSD (whiteCovariance (Pool := Pool) variance) := by
  constructor
  · intro i j
    by_cases hij : i = j
    · simp [whiteCovariance, hij]
    · simp [whiteCovariance, hij, Ne.symm hij]
  · intro z
    rw [white_quadratic_form]
    exact Finset.sum_nonneg (fun i _ => mul_nonneg hv (sq_nonneg (z i)))

def physicalWhiteCovariance {Policy : Type*} [DecidableEq Policy]
    (policy : Pool → Policy) (variance : ℝ) (i j : Pool) : ℝ :=
  if policy i = policy j then variance else 0

/-- The implementation uses `variance * I` only on a distinct-policy pool. -/
theorem physical_white_eq_index_white {Policy : Type*}
    [DecidableEq Policy] [DecidableEq Pool]
    (policy : Pool → Policy) (hpolicy : Function.Injective policy)
    (variance : ℝ) :
    physicalWhiteCovariance policy variance = whiteCovariance variance := by
  funext i j
  simp [physicalWhiteCovariance, whiteCovariance, hpolicy.eq_iff]

theorem physical_white_covariance_psd {Policy : Type*}
    [DecidableEq Policy] [DecidableEq Pool]
    (policy : Pool → Policy) (hpolicy : Function.Injective policy)
    {variance : ℝ} (hv : 0 ≤ variance) :
    CovariancePSD (physicalWhiteCovariance policy variance) := by
  rw [physical_white_eq_index_white policy hpolicy]
  exact white_covariance_psd hv

def sourceCovariance [DecidableEq Pool]
    (coefficient task : Feature → Feature → ℝ)
    (feature : Pool → Feature → ℝ) (white : ℝ) (i j : Pool) : ℝ :=
  projectedCovariance (fun p q => coefficient p q + task p q) feature i j +
    whiteCovariance white i j

theorem source_covariance_psd [DecidableEq Pool]
    {coefficient task : Feature → Feature → ℝ}
    (hc : CovariancePSD coefficient) (ht : CovariancePSD task)
    (feature : Pool → Feature → ℝ) {white : ℝ} (hw : 0 ≤ white) :
    CovariancePSD (sourceCovariance coefficient task feature white) :=
  add_covariance_psd (projected_covariance_psd (add_covariance_psd hc ht) feature)
    (white_covariance_psd hw)

def projectedMean (coefficientMean : Feature → ℝ)
    (feature : Pool → Feature → ℝ) (i : Pool) : ℝ :=
  ∑ p, coefficientMean p * feature i p

theorem projected_mean_restriction {Small : Type*}
    (select : Small → Pool) (coefficientMean : Feature → ℝ)
    (feature : Pool → Feature → ℝ) (i : Small) :
    projectedMean coefficientMean (fun a => feature (select a)) i =
      projectedMean coefficientMean feature (select i) := rfl

/-- Restricting the feature rows commutes exactly with coefficient projection. -/
theorem projected_covariance_restriction {Small : Type*}
    (select : Small → Pool) (C : Feature → Feature → ℝ)
    (feature : Pool → Feature → ℝ) (i j : Small) :
    projectedCovariance C (fun a => feature (select a)) i j =
      projectedCovariance C feature (select i) (select j) := rfl

/-- General finite kernel pullback, including repeated indices, preserves PSD. -/
theorem pullback_covariance_psd {Small : Type*} [Fintype Small]
    {C : Pool → Pool → ℝ} (hC : CovariancePSD C) (select : Small → Pool) :
    CovariancePSD (fun i j => C (select i) (select j)) := by
  classical
  have hp := projected_covariance_psd hC
    (fun (i : Small) (p : Pool) => if select i = p then (1 : ℝ) else 0)
  have heq : projectedCovariance C
      (fun (i : Small) (p : Pool) => if select i = p then (1 : ℝ) else 0) =
        fun i j => C (select i) (select j) := by
    funext i j
    simp [projectedCovariance, mul_ite, ite_mul]
  rw [heq] at hp
  exact hp

/-- The white discrepancy follows physical identity when a subpool is taken. -/
theorem source_covariance_restriction {Small : Type*}
    [DecidableEq Small] [DecidableEq Pool]
    (select : Small → Pool) (hselect : Function.Injective select)
    (coefficient task : Feature → Feature → ℝ)
    (feature : Pool → Feature → ℝ) (white : ℝ) (i j : Small) :
    sourceCovariance coefficient task (fun a => feature (select a)) white i j =
      sourceCovariance coefficient task feature white (select i) (select j) := by
  simp [sourceCovariance, projectedCovariance, whiteCovariance, hselect.eq_iff]

def blockCovariance {Left Right : Type*}
    (A : Left → Left → ℝ) (B : Right → Right → ℝ) :
    Left ⊕ Right → Left ⊕ Right → ℝ
  | .inl i, .inl j => A i j
  | .inr i, .inr j => B i j
  | _, _ => 0

theorem block_quadratic_form {Left Right : Type*}
    [Fintype Left] [Fintype Right]
    (A : Left → Left → ℝ) (B : Right → Right → ℝ)
    (z : Left ⊕ Right → ℝ) :
    quadraticForm (blockCovariance A B) z =
      quadraticForm A (fun i => z (.inl i)) +
        quadraticForm B (fun i => z (.inr i)) := by
  simp [quadraticForm, Fintype.sum_sum_type, blockCovariance]

theorem block_covariance_psd {Left Right : Type*}
    [Fintype Left] [Fintype Right]
    {A : Left → Left → ℝ} {B : Right → Right → ℝ}
    (hA : CovariancePSD A) (hB : CovariancePSD B) :
    CovariancePSD (blockCovariance A B) := by
  constructor
  · intro i j
    cases i <;> cases j
    · exact hA.1 _ _
    · rfl
    · rfl
    · exact hB.1 _ _
  · intro z
    rw [block_quadratic_form]
    exact add_nonneg (hA.2 _) (hB.2 _)

/-- Replacing one head's mean-field covariance while retaining its independent
eta block remains valid whenever the replacement and retained block are PSD. -/
theorem role_replacement_covariance_psd [DecidableEq Pool]
    {replacement : Feature → Feature → ℝ} {eta : Pool → Pool → ℝ}
    (hr : CovariancePSD replacement) (he : CovariancePSD eta)
    (design : Pool → Feature → ℝ) {white : ℝ} (hw : 0 ≤ white) :
    CovariancePSD (blockCovariance
      (sourceCovariance replacement (fun _ _ => 0) design white) eta) :=
  block_covariance_psd (source_covariance_psd hr zero_covariance_psd design hw) he

end SCOLHKG.Real.CurrentSourceKernel

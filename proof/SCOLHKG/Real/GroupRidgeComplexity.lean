import Mathlib

namespace SCOLHKG.Real

open scoped BigOperators

/-!
V27 replaces a universal hard rank cap by a finite, target-observation-only
group-ridge selector.  The first results below formalize spectral effective
degrees of freedom.  The final theorem is the deterministic core of the finite
nested-refit oracle inequality: a uniformly accurate empirical selector loses
at most twice the uniform deviation from the best finite penalty model.
-/

noncomputable def ridgeDirectionEffectiveDimension
    (information penalty : ℝ) : ℝ :=
  information / (information + penalty)

theorem ridgeDirectionEffectiveDimension_nonnegative
    (information penalty : ℝ)
    (hInformation : 0 ≤ information)
    (hPositive : 0 < information + penalty) :
    0 ≤ ridgeDirectionEffectiveDimension information penalty := by
  unfold ridgeDirectionEffectiveDimension
  exact div_nonneg hInformation (le_of_lt hPositive)

theorem ridgeDirectionEffectiveDimension_le_one
    (information penalty : ℝ)
    (hPenalty : 0 ≤ penalty)
    (hPositive : 0 < information + penalty) :
    ridgeDirectionEffectiveDimension information penalty ≤ 1 := by
  unfold ridgeDirectionEffectiveDimension
  exact (div_le_one hPositive).2 (by linarith)

noncomputable def finiteRidgeEffectiveDimension {n : ℕ}
    (information penalty : Fin n → ℝ) : ℝ :=
  ∑ i, ridgeDirectionEffectiveDimension (information i) (penalty i)

theorem finiteRidgeEffectiveDimension_nonnegative
    {n : ℕ}
    (information penalty : Fin n → ℝ)
    (hInformation : ∀ i, 0 ≤ information i)
    (hPositive : ∀ i, 0 < information i + penalty i) :
    0 ≤ finiteRidgeEffectiveDimension information penalty := by
  unfold finiteRidgeEffectiveDimension
  exact Finset.sum_nonneg fun i _ =>
    ridgeDirectionEffectiveDimension_nonnegative
      (information i) (penalty i)
      (hInformation i) (hPositive i)

theorem finiteRidgeEffectiveDimension_le_feature_count
    {n : ℕ}
    (information penalty : Fin n → ℝ)
    (hPenalty : ∀ i, 0 ≤ penalty i)
    (hPositive : ∀ i, 0 < information i + penalty i) :
    finiteRidgeEffectiveDimension information penalty ≤ (n : ℝ) := by
  unfold finiteRidgeEffectiveDimension
  calc
    (∑ i : Fin n,
        ridgeDirectionEffectiveDimension (information i) (penalty i))
        ≤ ∑ _i : Fin n, (1 : ℝ) := by
          apply Finset.sum_le_sum
          intro i _hi
          exact ridgeDirectionEffectiveDimension_le_one
            (information i) (penalty i) (hPenalty i) (hPositive i)
    _ = (n : ℝ) := by simp

theorem finite_nested_selector_oracle_bound
    {Model : Type*} [Fintype Model]
    (empiricalRisk trueRisk : Model → ℝ)
    (selected comparator : Model)
    (epsilon : ℝ)
    (hSelected : ∀ model, empiricalRisk selected ≤ empiricalRisk model)
    (hUniform : ∀ model,
      |empiricalRisk model - trueRisk model| ≤ epsilon) :
    trueRisk selected ≤ trueRisk comparator + 2 * epsilon := by
  have hSelectedDeviation := (abs_le.mp (hUniform selected)).1
  have hComparatorDeviation := (abs_le.mp (hUniform comparator)).2
  have hEmpirical := hSelected comparator
  linarith

end SCOLHKG.Real

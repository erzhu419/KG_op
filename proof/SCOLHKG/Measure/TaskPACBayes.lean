import Mathlib
import SCOLHKG.Real.TaskPosterior

namespace SCOLHKG.Measure

open MeasureTheory
open scoped BigOperators

/-!
Finite unseen-task PAC-Bayes concentration layer.

Each source expert supplies an exponential generalization-gap moment bounded
by one. A source-only prior mixture preserves that moment bound. Markov's
inequality then gives the high-probability moment event consumed by
`SCOLHKG.Real.finite_pac_bayes_bound_on_moment_event`.
-/

theorem finite_prior_exponential_moment_le_one
    {Ω ι : Type*}
    [MeasurableSpace Ω]
    [Fintype ι]
    {μ : Measure Ω}
    {prior : ι → ℝ}
    {moment : ι → Ω → ℝ}
    (hPrior : ∀ i, 0 ≤ prior i)
    (hPriorNorm : ∑ i, prior i = 1)
    (hIntegrable : ∀ i, Integrable (moment i) μ)
    (hMoment : ∀ i, (∫ ω, moment i ω ∂μ) ≤ 1) :
    Integrable (fun ω => ∑ i, prior i * moment i ω) μ ∧
      (∫ ω, ∑ i, prior i * moment i ω ∂μ) ≤ 1 := by
  have hWeighted : ∀ i, Integrable (fun ω => prior i * moment i ω) μ := by
    intro i
    exact (hIntegrable i).const_mul (prior i)
  have hSumIntegrable :
      Integrable (fun ω => ∑ i, prior i * moment i ω) μ := by
    exact integrable_finsetSum Finset.univ (fun i _hi => hWeighted i)
  refine ⟨hSumIntegrable, ?_⟩
  calc
    (∫ ω, ∑ i, prior i * moment i ω ∂μ) =
        ∑ i, ∫ ω, prior i * moment i ω ∂μ := by
      exact integral_finsetSum Finset.univ (fun i _hi => hWeighted i)
    _ = ∑ i, prior i * ∫ ω, moment i ω ∂μ := by
      apply Finset.sum_congr rfl
      intro i _hi
      exact integral_const_mul (prior i) (moment i)
    _ ≤ ∑ i, prior i * 1 := by
      exact Finset.sum_le_sum (fun i _hi =>
        mul_le_mul_of_nonneg_left (hMoment i) (hPrior i))
    _ = 1 := by simpa using hPriorNorm

theorem pac_bayes_moment_bad_event_le_delta
    {Ω : Type*}
    [MeasurableSpace Ω]
    {μ : Measure Ω}
    {aggregateMoment : Ω → ℝ}
    {delta : ℝ}
    (hNonnegative : 0 ≤ᵐ[μ] aggregateMoment)
    (hIntegrable : Integrable aggregateMoment μ)
    (hExpectation : (∫ ω, aggregateMoment ω ∂μ) ≤ 1)
    (hDelta : 0 < delta) :
    μ.real {ω | 1 / delta ≤ aggregateMoment ω} ≤ delta := by
  have hMarkov := mul_meas_ge_le_integral_of_nonneg
    hNonnegative hIntegrable (1 / delta)
  have hDiv :
      μ.real {ω | 1 / delta ≤ aggregateMoment ω} / delta ≤ 1 := by
    calc
      μ.real {ω | 1 / delta ≤ aggregateMoment ω} / delta =
          (1 / delta) * μ.real {ω | 1 / delta ≤ aggregateMoment ω} := by
        ring
      _ ≤ ∫ ω, aggregateMoment ω ∂μ := hMarkov
      _ ≤ 1 := hExpectation
  have hBound := (div_le_iff₀ hDelta).mp hDiv
  simpa using hBound

theorem finite_source_task_pac_bayes_bad_event_le_delta
    {Ω ι : Type*}
    [MeasurableSpace Ω]
    [Fintype ι]
    {μ : Measure Ω}
    {prior : ι → ℝ}
    {moment : ι → Ω → ℝ}
    {delta : ℝ}
    (hPrior : ∀ i, 0 ≤ prior i)
    (hPriorNorm : ∑ i, prior i = 1)
    (hMomentNonnegative : ∀ i, 0 ≤ᵐ[μ] moment i)
    (hIntegrable : ∀ i, Integrable (moment i) μ)
    (hMoment : ∀ i, (∫ ω, moment i ω ∂μ) ≤ 1)
    (hDelta : 0 < delta) :
    μ.real {
      ω | 1 / delta ≤ ∑ i, prior i * moment i ω
    } ≤ delta := by
  obtain ⟨hAggregateIntegrable, hAggregateExpectation⟩ :=
    finite_prior_exponential_moment_le_one
      hPrior hPriorNorm hIntegrable hMoment
  have hAggregateNonnegative :
      0 ≤ᵐ[μ] fun ω => ∑ i, prior i * moment i ω := by
    filter_upwards [ae_all_iff.mpr hMomentNonnegative] with ω hMomentOmega
    exact Finset.sum_nonneg (fun i _hi =>
      mul_nonneg (hPrior i) (hMomentOmega i))
  exact pac_bayes_moment_bad_event_le_delta
    hAggregateNonnegative
    hAggregateIntegrable
    hAggregateExpectation
    hDelta

end SCOLHKG.Measure

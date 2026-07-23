import Mathlib.Probability.Independence.Integration
import SCOLHKG.Measure.TaskPACBayes
import SCOLHKG.Real.ConstrainedCertificateDeficit

namespace SCOLHKG.Measure

open MeasureTheory ProbabilityTheory
open scoped BigOperators ENNReal

/-!
# Independent pilot/confirmation policy-improvement certificate

V56 selects one action with a pilot RQMC stream, freezes that action, and uses
a disjoint IID stream to test two null hypotheses: nonpositive Bayes-risk
reduction and nonpositive certificate-deficit reduction.  For a bounded gain
`X ∈ [-1,1]` and a fixed bet `lambda ∈ [0,1]`, `1 + lambda * X` is
nonnegative and has expectation at most one under either null. Independence
therefore makes its finite product an e-value. A source-frozen finite mixture
of such bets remains an e-value. Markov's inequality controls each fixed batch
look, and an explicit finite union bound pays for all permitted looks. The
final theorem exposes the finite two-head, finite-horizon, finite-look error
budget used by the Python implementation.

The action is arbitrary in these statements. Thus, after conditioning on the
independent pilot sigma-field, the same theorem applies to the pilot-selected
action without paying a union bound over unselected actions.
-/

def bettingFactor (lambda gain : ℝ) : ℝ :=
  1 + lambda * gain

def bettingWealth
    {ι Ω : Type*}
    [Fintype ι]
    (lambda : ℝ)
    (gain : ι → Ω → ℝ)
    (omega : Ω) : ℝ :=
  ∏ i, bettingFactor lambda (gain i omega)

theorem bettingFactor_nonnegative
    {lambda gain : ℝ}
    (hLambdaNonnegative : 0 ≤ lambda)
    (hLambdaAtMostOne : lambda ≤ 1)
    (hGainLower : -1 ≤ gain) :
    0 ≤ bettingFactor lambda gain := by
  have hProduct : 0 ≤ lambda * (gain + 1) :=
    mul_nonneg hLambdaNonnegative (by linarith)
  unfold bettingFactor
  nlinarith

theorem bettingFactor_le_two
    {lambda gain : ℝ}
    (hLambdaNonnegative : 0 ≤ lambda)
    (hLambdaAtMostOne : lambda ≤ 1)
    (hGainUpper : gain ≤ 1) :
    bettingFactor lambda gain ≤ 2 := by
  have hProduct : lambda * gain ≤ lambda :=
    by simpa using
      (mul_le_mul_of_nonneg_left hGainUpper hLambdaNonnegative)
  unfold bettingFactor
  linarith

theorem bettingFactor_integral_eq
    {Ω : Type*}
    [MeasurableSpace Ω]
    {mu : Measure Ω}
    [IsProbabilityMeasure mu]
    {gain : Ω → ℝ}
    (hGainIntegrable : Integrable gain mu)
    (lambda : ℝ) :
    (∫ omega, bettingFactor lambda (gain omega) ∂mu) =
      1 + lambda * (∫ omega, gain omega ∂mu) := by
  unfold bettingFactor
  rw [integral_add (integrable_const 1) (hGainIntegrable.const_mul lambda)]
  rw [integral_const_mul]
  simp

theorem independent_bettingWealth_integral_le_one
    {ι Ω : Type*}
    [Fintype ι]
    [MeasurableSpace Ω]
    {mu : Measure Ω}
    [IsProbabilityMeasure mu]
    {gain : ι → Ω → ℝ}
    {lambda : ℝ}
    (hIndependent : iIndepFun gain mu)
    (hGainMeasurable : ∀ i, Measurable (gain i))
    (hGainIntegrable : ∀ i, Integrable (gain i) mu)
    (hGainBounded : ∀ i omega, gain i omega ∈ Set.Icc (-1) 1)
    (hMeanNull : ∀ i, (∫ omega, gain i omega ∂mu) ≤ 0)
    (hLambdaNonnegative : 0 ≤ lambda)
    (hLambdaAtMostOne : lambda ≤ 1) :
    (∫ omega, bettingWealth lambda gain omega ∂mu) ≤ 1 := by
  have hFactorMeasurable : Measurable (fun x : ℝ =>
      bettingFactor lambda x) := by
    simpa [bettingFactor, Function.id_def] using
      (measurable_const.add (measurable_const.mul measurable_id))
  have hFactorStrong : ∀ i, AEStronglyMeasurable
      (fun x : ℝ => bettingFactor lambda x) (mu.map (gain i)) := by
    intro i
    exact hFactorMeasurable.aestronglyMeasurable
  change (∫ omega, ∏ i, bettingFactor lambda (gain i omega) ∂mu) ≤ 1
  rw [hIndependent.integral_fun_prod_comp
    (fun i => (hGainMeasurable i).aemeasurable)
    hFactorStrong]
  refine Finset.prod_le_one (fun i _hi => ?_) (fun i _hi => ?_)
  · apply integral_nonneg_of_ae
    exact Filter.Eventually.of_forall (fun omega =>
      bettingFactor_nonnegative
        hLambdaNonnegative hLambdaAtMostOne (hGainBounded i omega).1)
  · rw [bettingFactor_integral_eq (hGainIntegrable i) lambda]
    have hScaled :
        lambda * (∫ omega, gain i omega ∂mu) ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos hLambdaNonnegative (hMeanNull i)
    linarith

theorem independent_betting_mixture_false_acceptance_le
    {ι Lambda Ω : Type*}
    [Fintype ι]
    [Fintype Lambda]
    [MeasurableSpace Ω]
    {mu : Measure Ω}
    [IsProbabilityMeasure mu]
    {gain : ι → Ω → ℝ}
    {lambda : Lambda → ℝ}
    {prior : Lambda → ℝ}
    {delta : ℝ}
    (hIndependent : iIndepFun gain mu)
    (hGainMeasurable : ∀ i, Measurable (gain i))
    (hGainIntegrable : ∀ i, Integrable (gain i) mu)
    (hGainBounded : ∀ i omega, gain i omega ∈ Set.Icc (-1) 1)
    (hMeanNull : ∀ i, (∫ omega, gain i omega ∂mu) ≤ 0)
    (hLambda : ∀ ell, lambda ell ∈ Set.Icc 0 1)
    (hPrior : ∀ ell, 0 ≤ prior ell)
    (hPriorNorm : ∑ ell, prior ell = 1)
    (hWealthIntegrable : ∀ ell,
      Integrable (bettingWealth (lambda ell) gain) mu)
    (hDelta : 0 < delta) :
    mu.real {
      omega |
        1 / delta ≤
          ∑ ell, prior ell * bettingWealth (lambda ell) gain omega
    } ≤ delta := by
  apply finite_source_task_pac_bayes_bad_event_le_delta
      hPrior hPriorNorm
  · intro ell
    exact Filter.Eventually.of_forall (fun omega => by
      exact Finset.prod_nonneg (fun i _hi =>
        bettingFactor_nonnegative
          (hLambda ell).1 (hLambda ell).2 (hGainBounded i omega).1))
  · exact hWealthIntegrable
  · intro ell
    exact independent_bettingWealth_integral_le_one
      hIndependent hGainMeasurable hGainIntegrable hGainBounded hMeanNull
      (hLambda ell).1 (hLambda ell).2
  · exact hDelta

theorem finite_two_head_horizon_error_spending
    {Stage Head Omega : Type*}
    [MeasurableSpace Omega]
    {mu : Measure Omega}
    (stages : Finset Stage)
    (heads : Finset Head)
    (bad : Stage → Head → Set Omega)
    (alpha : Stage → Head → ℝ≥0∞)
    (delta : ℝ≥0∞)
    (hCell : ∀ stage ∈ stages, ∀ head ∈ heads,
      mu (bad stage head) ≤ alpha stage head)
    (hBudget :
      ∑ stage ∈ stages, ∑ head ∈ heads, alpha stage head ≤ delta) :
    mu (⋃ stage ∈ stages, ⋃ head ∈ heads, bad stage head) ≤ delta := by
  calc
    mu (⋃ stage ∈ stages, ⋃ head ∈ heads, bad stage head) ≤
        ∑ stage ∈ stages, mu (⋃ head ∈ heads, bad stage head) := by
      exact measure_biUnion_finset_le stages
        (fun stage => ⋃ head ∈ heads, bad stage head)
    _ ≤ ∑ stage ∈ stages, ∑ head ∈ heads,
        mu (bad stage head) := by
      exact Finset.sum_le_sum (fun stage _hStage =>
        measure_biUnion_finset_le heads (bad stage))
    _ ≤ ∑ stage ∈ stages, ∑ head ∈ heads,
        alpha stage head := by
      exact Finset.sum_le_sum (fun stage hStage =>
        Finset.sum_le_sum (fun head hHead =>
          hCell stage hStage head hHead))
    _ ≤ delta := hBudget

theorem finite_two_head_horizon_look_error_spending
    {Stage Head Look Omega : Type*}
    [MeasurableSpace Omega]
    {mu : Measure Omega}
    (stages : Finset Stage)
    (heads : Finset Head)
    (looks : Finset Look)
    (bad : Stage → Head → Look → Set Omega)
    (alpha : Stage → Head → Look → ℝ≥0∞)
    (delta : ℝ≥0∞)
    (hCell : ∀ stage ∈ stages, ∀ head ∈ heads, ∀ look ∈ looks,
      mu (bad stage head look) ≤ alpha stage head look)
    (hBudget :
      ∑ stage ∈ stages, ∑ head ∈ heads, ∑ look ∈ looks,
        alpha stage head look ≤ delta) :
    mu (⋃ stage ∈ stages, ⋃ head ∈ heads, ⋃ look ∈ looks,
      bad stage head look) ≤ delta := by
  calc
    mu (⋃ stage ∈ stages, ⋃ head ∈ heads, ⋃ look ∈ looks,
        bad stage head look) ≤
        ∑ stage ∈ stages,
          mu (⋃ head ∈ heads, ⋃ look ∈ looks,
            bad stage head look) := by
      exact measure_biUnion_finset_le stages
        (fun stage => ⋃ head ∈ heads, ⋃ look ∈ looks,
          bad stage head look)
    _ ≤ ∑ stage ∈ stages, ∑ head ∈ heads,
        mu (⋃ look ∈ looks, bad stage head look) := by
      exact Finset.sum_le_sum (fun stage _hStage =>
        measure_biUnion_finset_le heads
          (fun head => ⋃ look ∈ looks, bad stage head look))
    _ ≤ ∑ stage ∈ stages, ∑ head ∈ heads, ∑ look ∈ looks,
        mu (bad stage head look) := by
      exact Finset.sum_le_sum (fun stage _hStage =>
        Finset.sum_le_sum (fun head _hHead =>
          measure_biUnion_finset_le looks (bad stage head)))
    _ ≤ ∑ stage ∈ stages, ∑ head ∈ heads, ∑ look ∈ looks,
        alpha stage head look := by
      exact Finset.sum_le_sum (fun stage hStage =>
        Finset.sum_le_sum (fun head hHead =>
          Finset.sum_le_sum (fun look hLook =>
            hCell stage hStage head hHead look hLook)))
    _ ≤ delta := hBudget

end SCOLHKG.Measure

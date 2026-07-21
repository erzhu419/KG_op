import Mathlib
import SCOLHKG.Real.CumulativeRisk
import SCOLHKG.Real.MeanRiskCoordinateSeparation
import SCOLHKG.Real.PromotedV51Closure
import SCOLHKG.Real.RiskAlignedRepresentation

namespace SCOLHKG.Real

/-!
The single theorem spine used by the revised manuscript.

It composes four interfaces that used to be described as separate modules:
the source-only proposal, the separated observable mean/risk coordinate, the
cumulative heteroscedastic variance, and the promoted evaluate-or-replicate
posterior decision.  The result deliberately retains the statistical coverage
and finite-action approximation hypotheses; those are measured experiment
obligations, not hidden axioms.
-/

theorem paper_mainline_finite_closure
    {SourceRecords Proposal TargetLabels : Type}
    {X Eta Psi State Design Ω : Type*}
    [MeasurableSpace Ω]
    [DecidableEq Design]
    (fitProposal : SourceRecords → Proposal)
    (sourceRecords : SourceRecords)
    (leftTargetLabels rightTargetLabels : TargetLabels)
    (risk : CumulativeRisk)
    (model : SeparatedMeanRiskModel X Eta Psi)
    (x y : X)
    (hEta : model.eta x = model.eta y)
    (hPsi : model.psi x = model.psi y)
    (beta z tau : ℝ)
    (kg : SCOLHKG.Measure.PosteriorUpdateKG
      State (EvaluateOrReplicateAction Design) Ω)
    (μ : MeasureTheory.Measure Ω) (state : State)
    (full shortlist : Finset (EvaluateOrReplicateAction Design))
    (estimate : EvaluateOrReplicateAction Design → ℝ)
    (epsilon etaMC : ℝ)
    (selected : EvaluateOrReplicateAction Design)
    (hCover : ShortlistCoversVOI full shortlist
      (SCOLHKG.Measure.posteriorUpdateExpectedGain kg μ state) epsilon)
    (hUniform : UniformVOIApproximationOn shortlist
      (SCOLHKG.Measure.posteriorUpdateExpectedGain kg μ state)
      estimate etaMC)
    (hMax : MaximizesVOIOn shortlist estimate selected)
    {trueMean postMean epistemicVar trueSigma vC : ℝ}
    (hz : 0 ≤ z)
    (hMean : trueMean ≤ postMean
      + implementationEpistemicSlack beta epistemicVar)
    (hSigma : trueSigma ≤ implementationCertSigma vC)
    (hMargin : theoryCertificationMargin
      postMean beta epistemicVar z vC tau ≤ 0) :
    sourceOnlyProposal fitProposal sourceRecords leftTargetLabels =
        sourceOnlyProposal fitProposal sourceRecords rightTargetLabels
      ∧ totalVariance risk =
        risk.floor + independentRisk risk + sharedShockRisk risk
          + linearRisk risk
      ∧ separatedCertificationMargin model beta z tau x =
        separatedCertificationMargin model beta z tau y
      ∧ (∀ action ∈ full,
        SCOLHKG.Measure.posteriorUpdateExpectedGain kg μ state action
          ≤ SCOLHKG.Measure.posteriorUpdateExpectedGain kg μ state selected
            + epsilon + 2 * etaMC)
      ∧ trueMean + z * trueSigma ≤ tau := by
  refine ⟨sourceOnlyProposal_targetLabel_invariant
    fitProposal sourceRecords leftTargetLabels rightTargetLabels, ?_⟩
  refine ⟨fixedTrajectoryVarianceDecomposition risk, ?_⟩
  refine ⟨joint_coordinate_equivalence_preserves_margin
    model beta z tau hEta hPsi, ?_⟩
  exact promoted_v51_one_step_and_terminal_certificate
    kg μ state full shortlist estimate epsilon etaMC selected
    hCover hUniform hMax hz hMean hSigma hMargin

end SCOLHKG.Real

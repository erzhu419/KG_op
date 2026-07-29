import Mathlib
import SCOLHKG.Real.CumulativeRisk
import SCOLHKG.Real.GeometricAtlasCoverage
import SCOLHKG.Real.MeanRiskCoordinateSeparation
import SCOLHKG.Real.PromotedV51Closure
import SCOLHKG.Real.ProposalCoverage
import SCOLHKG.Real.RankAlignedAtlasCoverage
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

/-!
The final front-end theorem is backend independent.  It composes source-label
noninterference, effective-dimension PAC-Bayes transfer, IID initial-design
coverage, the observable coordinate quotient, cumulative HVD algebra, and the
terminal chance certificate.  SAASBO is one empirical backend consuming this
front end; it is not inserted as a novel theorem assumption.
-/

theorem paper_frontend_transfer_coverage_and_certificate
    {SourceRecords Proposal TargetLabels : Type}
    {X Eta Psi : Type*}
    {Expert : Type*} [Fintype Expert] [Nonempty Expert]
    (fitProposal : SourceRecords → Proposal)
    (sourceRecords : SourceRecords)
    (leftTargetLabels rightTargetLabels : TargetLabels)
    (risk : CumulativeRisk)
    (model : SeparatedMeanRiskModel X Eta Psi)
    (x y : X)
    (hEta : model.eta x = model.eta y)
    (hPsi : model.psi x = model.psi y)
    {posterior prior : Expert → ℝ}
    {sourceMiss targetMiss domainShift : Expert → ℝ}
    {effectiveDim logLibrary delta sourceSamples : ℝ}
    {n0 : ℕ}
    {hitProbability : ℝ}
    (hPosterior : ∀ i, 0 < posterior i)
    (hPrior : ∀ i, 0 < prior i)
    (hPosteriorNorm : ∑ i, posterior i = 1)
    (hPriorNorm : ∑ i, prior i = 1)
    (hKL :
      finiteTaskKL posterior prior ≤ effectiveDim * logLibrary)
    (hDelta : 0 < delta)
    (hSamples : 0 < sourceSamples)
    (hMoment :
      (∑ i, prior i * Real.exp
        (sourceSamples * sourceTargetGap
          sourceMiss targetMiss domainShift i)) ≤ 1 / delta)
    (hTargetMissNonnegative :
      0 ≤ ∑ i, posterior i * targetMiss i)
    (hTargetMissLeOne :
      (∑ i, posterior i * targetMiss i) ≤ 1)
    (hIID :
      hitProbability =
        iidProposalHitLowerBound
          (targetFeasibleMass (∑ i, posterior i * targetMiss i)) n0)
    {trueMean postMean epistemicVar trueSigma vC beta z tau : ℝ}
    (hz : 0 ≤ z)
    (hMean : trueMean ≤ postMean
      + implementationEpistemicSlack beta epistemicVar)
    (hSigma : trueSigma ≤ implementationCertSigma vC)
    (hMargin : theoryCertificationMargin
      postMean beta epistemicVar z vC tau ≤ 0) :
    let sourcePosteriorMiss := ∑ i, posterior i * sourceMiss i
    let posteriorShift := ∑ i, posterior i * domainShift i
    let radius := effectiveDimensionTransferRadius effectiveDim logLibrary
      (Real.log (1 / delta)) sourceSamples
    let pLower :=
      proposalFeasibleMassLower sourcePosteriorMiss posteriorShift radius
    sourceOnlyProposal fitProposal sourceRecords leftTargetLabels =
        sourceOnlyProposal fitProposal sourceRecords rightTargetLabels
      ∧ pLower ≤ targetFeasibleMass (∑ i, posterior i * targetMiss i)
      ∧ iidProposalHitLowerBound pLower n0 ≤ hitProbability
      ∧ separatedCertificationMargin model beta z tau x =
        separatedCertificationMargin model beta z tau y
      ∧ totalVariance risk =
        risk.floor + independentRisk risk + sharedShockRisk risk
          + linearRisk risk
      ∧ trueMean + z * trueSigma ≤ tau := by
  dsimp
  obtain ⟨hMass, hHit⟩ := finite_source_to_target_proposal_coverage
    hPosterior hPrior hPosteriorNorm hPriorNorm hKL hDelta hSamples hMoment
    hTargetMissNonnegative hTargetMissLeOne hIID
  refine ⟨sourceOnlyProposal_targetLabel_invariant
    fitProposal sourceRecords leftTargetLabels rightTargetLabels, ?_⟩
  refine ⟨hMass, hHit, ?_, fixedTrajectoryVarianceDecomposition risk, ?_⟩
  · exact joint_coordinate_equivalence_preserves_margin
      model beta z tau hEta hPsi
  · exact implementation_certifies_true_quantile
      hz hMean hSigma hMargin

/-!
Implementation-matched front-end closure for the deployed deterministic
`risk_objective_atlas`.  `Expert` is the finite atlas support and has at most
`n0` policies.  Unlike the randomized corollary above, no independence claim
is made between atlas members.  A strictly positive transferred feasible-mass
lower bound forces at least one feasible policy to exist in that support.
-/

theorem paper_frontend_atlas_coverage_and_certificate
    {SourceRecords Proposal TargetLabels : Type}
    {X Eta Psi : Type*}
    {Expert : Type*} [Fintype Expert] [Nonempty Expert]
    {feasible : Expert → Prop} [DecidablePred feasible]
    (fitProposal : SourceRecords → Proposal)
    (sourceRecords : SourceRecords)
    (leftTargetLabels rightTargetLabels : TargetLabels)
    (risk : CumulativeRisk)
    (model : SeparatedMeanRiskModel X Eta Psi)
    (x y : X)
    (hEta : model.eta x = model.eta y)
    (hPsi : model.psi x = model.psi y)
    {posterior prior : Expert → ℝ}
    {sourceMiss domainShift : Expert → ℝ}
    {effectiveDim logLibrary delta sourceSamples : ℝ}
    {n0 : ℕ}
    (hAtlasSize : Fintype.card Expert ≤ n0)
    (hPosterior : ∀ i, 0 < posterior i)
    (hPrior : ∀ i, 0 < prior i)
    (hPosteriorNorm : ∑ i, posterior i = 1)
    (hPriorNorm : ∑ i, prior i = 1)
    (hKL :
      finiteTaskKL posterior prior ≤ effectiveDim * logLibrary)
    (hDelta : 0 < delta)
    (hSamples : 0 < sourceSamples)
    (hMoment :
      (∑ i, prior i * Real.exp
        (sourceSamples * sourceTargetGap sourceMiss
          (targetMissIndicator feasible) domainShift i)) ≤ 1 / delta)
    (hTargetMissLeOne :
      (∑ i, posterior i * targetMissIndicator feasible i) ≤ 1)
    (hLowerPositive :
      0 < proposalFeasibleMassLower
        (∑ i, posterior i * sourceMiss i)
        (∑ i, posterior i * domainShift i)
        (effectiveDimensionTransferRadius effectiveDim logLibrary
          (Real.log (1 / delta)) sourceSamples))
    {trueMean postMean epistemicVar trueSigma vC beta z tau : ℝ}
    (hz : 0 ≤ z)
    (hMean : trueMean ≤ postMean
      + implementationEpistemicSlack beta epistemicVar)
    (hSigma : trueSigma ≤ implementationCertSigma vC)
    (hMargin : theoryCertificationMargin
      postMean beta epistemicVar z vC tau ≤ 0) :
    let sourcePosteriorMiss := ∑ i, posterior i * sourceMiss i
    let posteriorShift := ∑ i, posterior i * domainShift i
    let radius := effectiveDimensionTransferRadius effectiveDim logLibrary
      (Real.log (1 / delta)) sourceSamples
    let pLower :=
      proposalFeasibleMassLower sourcePosteriorMiss posteriorShift radius
    sourceOnlyProposal fitProposal sourceRecords leftTargetLabels =
        sourceOnlyProposal fitProposal sourceRecords rightTargetLabels
      ∧ Fintype.card Expert ≤ n0
      ∧ pLower ≤ finiteAtlasFeasibleMass posterior feasible
      ∧ (∃ i, feasible i)
      ∧ separatedCertificationMargin model beta z tau x =
        separatedCertificationMargin model beta z tau y
      ∧ totalVariance risk =
        risk.floor + independentRisk risk + sharedShockRisk risk
          + linearRisk risk
      ∧ trueMean + z * trueSigma ≤ tau := by
  dsimp
  obtain ⟨hMass, hExists⟩ := finite_source_to_target_atlas_coverage
    hPosterior hPrior hPosteriorNorm hPriorNorm hKL hDelta hSamples hMoment
    hTargetMissLeOne hLowerPositive
  refine ⟨sourceOnlyProposal_targetLabel_invariant
    fitProposal sourceRecords leftTargetLabels rightTargetLabels, ?_⟩
  refine ⟨hAtlasSize, hMass, hExists, ?_,
    fixedTrajectoryVarianceDecomposition risk, ?_⟩
  · exact joint_coordinate_equivalence_preserves_margin
      model beta z tau hEta hPsi
  · exact implementation_certifies_true_quantile
      hz hMean hSigma hMargin

/-!
Headline implementation contract for domains whose absolute chance thresholds
differ.  The source and target share a normalized risk rank up to `epsilon`;
the finite frozen atlas covers source rank up to `coverError`; and the held-out
task contains a safe policy deeper than the combined alignment/coverage
radius.  This is the rank-based bridge used by the paper experiments.
-/

theorem paper_frontend_rank_aligned_atlas_and_certificate
    {SourceRecords Proposal TargetLabels : Type}
    {X Eta Psi : Type*}
    {Expert : Type*} [DecidableEq Expert]
    (fitProposal : SourceRecords → Proposal)
    (sourceRecords : SourceRecords)
    (leftTargetLabels rightTargetLabels : TargetLabels)
    (risk : CumulativeRisk)
    (model : SeparatedMeanRiskModel X Eta Psi)
    (x y : X)
    (hEta : model.eta x = model.eta y)
    (hPsi : model.psi x = model.psi y)
    (atlas : Finset Expert)
    (sourceRank targetRank : Expert → ℝ)
    (feasible : Expert → Prop)
    (epsilon coverError threshold : ℝ)
    (n0 : ℕ)
    (hAtlasSize : atlas.card ≤ n0)
    (hAlignment :
      UniformRiskRankAlignment sourceRank targetRank epsilon)
    (hCover :
      OneSidedRiskRankAtlasCover atlas sourceRank coverError)
    (hSafe :
      RiskRankImpliesFeasible targetRank feasible threshold)
    (hInterior :
      ∃ candidate,
        targetRank candidate + 2 * epsilon + coverError ≤ threshold)
    {trueMean postMean epistemicVar trueSigma vC beta z tau : ℝ}
    (hz : 0 ≤ z)
    (hMean : trueMean ≤ postMean
      + implementationEpistemicSlack beta epistemicVar)
    (hSigma : trueSigma ≤ implementationCertSigma vC)
    (hMargin : theoryCertificationMargin
      postMean beta epistemicVar z vC tau ≤ 0) :
    sourceOnlyProposal fitProposal sourceRecords leftTargetLabels =
        sourceOnlyProposal fitProposal sourceRecords rightTargetLabels
      ∧ atlas.card ≤ n0
      ∧ (∃ candidate ∈ atlas, feasible candidate)
      ∧ separatedCertificationMargin model beta z tau x =
        separatedCertificationMargin model beta z tau y
      ∧ totalVariance risk =
        risk.floor + independentRisk risk + sharedShockRisk risk
          + linearRisk risk
      ∧ trueMean + z * trueSigma ≤ tau := by
  obtain ⟨hCard, hExists⟩ := finite_rank_aligned_atlas_coverage
    hAtlasSize hAlignment hCover hSafe hInterior
  refine ⟨sourceOnlyProposal_targetLabel_invariant
    fitProposal sourceRecords leftTargetLabels rightTargetLabels,
    hCard, hExists, ?_, fixedTrajectoryVarianceDecomposition risk, ?_⟩
  · exact joint_coordinate_equivalence_preserves_margin
      model beta z tau hEta hPsi
  · exact implementation_certifies_true_quantile
      hz hMean hSigma hMargin

/-!
Headline geometric contract for the deployed maximin atlas.  The nominal
policy dimension is absent: coverage is measured only in the transferable
coordinate `psiAtlas`.  A source-supported proxy for a target-safe center and
a complete safe coordinate ball turn the finite covering-radius inequality
into an actual feasible atlas member.
-/

theorem paper_frontend_geometric_atlas_and_certificate
    {SourceRecords Proposal TargetLabels : Type}
    {X Eta Psi : Type*}
    {Expert Z : Type*} [DecidableEq Expert] [PseudoMetricSpace Z]
    (fitProposal : SourceRecords → Proposal)
    (sourceRecords : SourceRecords)
    (leftTargetLabels rightTargetLabels : TargetLabels)
    (risk : CumulativeRisk)
    (model : SeparatedMeanRiskModel X Eta Psi)
    (x y : X)
    (hEta : model.eta x = model.eta y)
    (hPsi : model.psi x = model.psi y)
    (psiAtlas : Expert → Z)
    (atlas support : Finset Expert)
    (feasible : Expert → Prop)
    (center : Expert)
    (coverRadius supportShift safeRadius : ℝ)
    (n0 : ℕ)
    (hAtlasSize : atlas.card ≤ n0)
    (hCover :
      CoordinateAtlasCovers psiAtlas atlas support coverRadius)
    (hSupportProxy :
      ∃ source ∈ support,
        dist (psiAtlas source) (psiAtlas center) ≤ supportShift)
    (hSafeBall :
      CoordinateSafeBall psiAtlas feasible center safeRadius)
    (hRadius : coverRadius + supportShift ≤ safeRadius)
    {trueMean postMean epistemicVar trueSigma vC beta z tau : ℝ}
    (hz : 0 ≤ z)
    (hMean : trueMean ≤ postMean
      + implementationEpistemicSlack beta epistemicVar)
    (hSigma : trueSigma ≤ implementationCertSigma vC)
    (hMargin : theoryCertificationMargin
      postMean beta epistemicVar z vC tau ≤ 0) :
    sourceOnlyProposal fitProposal sourceRecords leftTargetLabels =
        sourceOnlyProposal fitProposal sourceRecords rightTargetLabels
      ∧ atlas.card ≤ n0
      ∧ (∃ candidate ∈ atlas, feasible candidate)
      ∧ separatedCertificationMargin model beta z tau x =
        separatedCertificationMargin model beta z tau y
      ∧ totalVariance risk =
        risk.floor + independentRisk risk + sharedShockRisk risk
          + linearRisk risk
      ∧ trueMean + z * trueSigma ≤ tau := by
  obtain ⟨hCard, hExists⟩ := finite_geometric_atlas_coverage
    hAtlasSize hCover hSupportProxy hSafeBall hRadius
  refine ⟨sourceOnlyProposal_targetLabel_invariant
    fitProposal sourceRecords leftTargetLabels rightTargetLabels,
    hCard, hExists, ?_, fixedTrajectoryVarianceDecomposition risk, ?_⟩
  · exact joint_coordinate_equivalence_preserves_margin
      model beta z tau hEta hPsi
  · exact implementation_certifies_true_quantile
      hz hMean hSigma hMargin

theorem paper_frontend_lipschitz_geometric_atlas_and_certificate
    {SourceRecords Proposal TargetLabels : Type}
    {X Eta Psi : Type*}
    {Expert Z : Type*} [DecidableEq Expert] [PseudoMetricSpace Z]
    (fitProposal : SourceRecords → Proposal)
    (sourceRecords : SourceRecords)
    (leftTargetLabels rightTargetLabels : TargetLabels)
    (risk : CumulativeRisk)
    (model : SeparatedMeanRiskModel X Eta Psi)
    (x y : X)
    (hEta : model.eta x = model.eta y)
    (hPsi : model.psi x = model.psi y)
    (psiAtlas : Expert → Z)
    (atlas support : Finset Expert)
    (margin : Expert → ℝ)
    (center : Expert)
    (coverRadius supportShift L safeDepth : ℝ)
    (n0 : ℕ)
    (hAtlasSize : atlas.card ≤ n0)
    (hCover :
      CoordinateAtlasCovers psiAtlas atlas support coverRadius)
    (hSupportProxy :
      ∃ source ∈ support,
        dist (psiAtlas source) (psiAtlas center) ≤ supportShift)
    (hLipschitz :
      CoordinateMarginOneSidedLipschitz psiAtlas margin L)
    (hLNonnegative : 0 ≤ L)
    (hCenterDepth : margin center + safeDepth ≤ 0)
    (hDepth : L * (coverRadius + supportShift) ≤ safeDepth)
    {trueMean postMean epistemicVar trueSigma vC beta z tau : ℝ}
    (hz : 0 ≤ z)
    (hMean : trueMean ≤ postMean
      + implementationEpistemicSlack beta epistemicVar)
    (hSigma : trueSigma ≤ implementationCertSigma vC)
    (hMargin : theoryCertificationMargin
      postMean beta epistemicVar z vC tau ≤ 0) :
    sourceOnlyProposal fitProposal sourceRecords leftTargetLabels =
        sourceOnlyProposal fitProposal sourceRecords rightTargetLabels
      ∧ atlas.card ≤ n0
      ∧ (∃ candidate ∈ atlas, margin candidate ≤ 0)
      ∧ separatedCertificationMargin model beta z tau x =
        separatedCertificationMargin model beta z tau y
      ∧ totalVariance risk =
        risk.floor + independentRisk risk + sharedShockRisk risk
          + linearRisk risk
      ∧ trueMean + z * trueSigma ≤ tau := by
  obtain ⟨hCard, hExists⟩ := finite_geometric_lipschitz_atlas_coverage
    hAtlasSize hCover hSupportProxy hLipschitz hLNonnegative
    hCenterDepth hDepth
  refine ⟨sourceOnlyProposal_targetLabel_invariant
    fitProposal sourceRecords leftTargetLabels rightTargetLabels,
    hCard, hExists, ?_, fixedTrajectoryVarianceDecomposition risk, ?_⟩
  · exact joint_coordinate_equivalence_preserves_margin
      model beta z tau hEta hPsi
  · exact implementation_certifies_true_quantile
      hz hMean hSigma hMargin

end SCOLHKG.Real

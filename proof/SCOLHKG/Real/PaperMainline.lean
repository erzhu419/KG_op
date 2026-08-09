import Mathlib
import SCOLHKG.Real.CumulativeRisk
import SCOLHKG.Real.GeometricAtlasCoverage
import SCOLHKG.Real.MeanRiskCoordinateSeparation
import SCOLHKG.Real.PromotedV51Closure
import SCOLHKG.Real.ProposalCoverage
import SCOLHKG.Real.RankAlignedAtlasCoverage
import SCOLHKG.Real.RiskAlignedRepresentation
import SCOLHKG.Real.SourceMonotoneEnvelope

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

/-!
The historical theorems below retain the SC-OLH-KG acquisition and HVD
ablations.  The final Operations Research method identity is narrower: a
source-frozen proposal atlas, a replaceable backend, and method-independent
terminal verification.  Its V3 source-monotone endpoint is governed by
`SourceMonotoneEnvelope.lean`; it fails closed when source directions disagree
and is safe only under an explicit transferred monotonicity condition.
-/

theorem paper_final_v3_fail_closed_contract
    {SourceRecords Descriptor TargetLabels X : Type*}
    (select : SourceRecords -> Descriptor -> Option X)
    (source : SourceRecords) (descriptor : Descriptor)
    (leftLabels rightLabels : TargetLabels)
    (baseline : List X) (endpoint : X) :
    sourceEnvelopeFromFrozenRecords
        select source descriptor leftLabels =
      sourceEnvelopeFromFrozenRecords
        select source descriptor rightLabels
      ∧ failClosedEnvelopeProposal baseline endpoint false = baseline
      ∧ (failClosedEnvelopeProposal baseline endpoint false).length =
        baseline.length := by
  exact ⟨
    source_envelope_target_label_noninterference
      select source descriptor leftLabels rightLabels,
    rejected_envelope_preserves_baseline baseline endpoint,
    fail_closed_envelope_preserves_budget baseline endpoint false⟩

theorem paper_final_v3_admitted_endpoint_contract
    {Source X : Type*} [Fintype Source]
    {correlation : Source -> Real} {threshold : Real}
    {coordinate margin : X -> Real} {lower upper : X}
    (hAdmitted :
      SourceMonotoneEnvelopeAdmitted correlation threshold)
    (hUpper : forall x, coordinate x <= coordinate upper)
    (hLower : forall x, coordinate lower <= coordinate x)
    (hNegativeTransfer :
      SourceNegativeAgreement correlation threshold ->
        CoordinateNonincreasingMargin coordinate margin)
    (hPositiveTransfer :
      SourcePositiveAgreement correlation threshold ->
        CoordinateNondecreasingMargin coordinate margin)
    (hSafeWitness : exists x, margin x <= 0) :
    (SourceNegativeAgreement correlation threshold ∧ margin upper <= 0)
      ∨
    (SourcePositiveAgreement correlation threshold ∧ margin lower <= 0) := by
  exact admitted_source_envelope_safe_under_transferred_direction
    hAdmitted hUpper hLower hNegativeTransfer hPositiveTransfer hSafeWitness

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
Audited rank-transfer alternative for domains whose absolute chance thresholds
differ.  Its source-only finite-sample calibration was vacuous in the paper
domains, so this theorem is retained as a negative-control contract rather
than the headline empirical bridge.
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

/-!
Fully decomposed headline contract. The learned coordinate approximates an
ideal transferable coordinate within `coordinateError`; source and target safe
support differ by `domainShift`; and the deterministic atlas covers learned
source support within `coverRadius`. The factor two is the representation error
at the source proxy and target-safe center. Nominal policy dimension remains
absent.
-/

theorem paper_frontend_aligned_geometric_atlas_and_certificate
    {SourceRecords Proposal TargetLabels : Type}
    {X Eta Psi : Type*}
    {Expert Augmented Structural : Type*} [DecidableEq Expert]
    [PseudoMetricSpace Augmented] [PseudoMetricSpace Structural]
    (fitProposal : SourceRecords → Proposal)
    (sourceRecords : SourceRecords)
    (leftTargetLabels rightTargetLabels : TargetLabels)
    (risk : CumulativeRisk)
    (model : SeparatedMeanRiskModel X Eta Psi)
    (x y : X)
    (hEta : model.eta x = model.eta y)
    (hPsi : model.psi x = model.psi y)
    (augmentedCoordinate : Expert → Augmented)
    (structuralCoordinate truthCoordinate : Expert → Structural)
    (project : Augmented → Structural)
    (atlas support : Finset Expert)
    (margin : Expert → ℝ)
    (center : Expert)
    (coverRadius domainShift coordinateError L safeDepth : ℝ)
    (n0 : ℕ)
    (hAtlasSize : atlas.card ≤ n0)
    (hCover :
      CoordinateAtlasCovers augmentedCoordinate atlas support coverRadius)
    (hProjection :
      CoordinateProjectionCompatible
        augmentedCoordinate structuralCoordinate project)
    (hProjectionNonexpansive : CoordinateProjectionNonexpansive project)
    (hApproximation :
      UniformCoordinateApproximation
        structuralCoordinate truthCoordinate coordinateError)
    (hTruthProxy :
      TruthCoordinateSupportProxy
        truthCoordinate support center domainShift)
    (hLipschitz :
      CoordinateMarginOneSidedLipschitz truthCoordinate margin L)
    (hLNonnegative : 0 ≤ L)
    (hCenterDepth : margin center + safeDepth ≤ 0)
    (hDepth :
      L * (coverRadius + domainShift + 2 * coordinateError) ≤ safeDepth)
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
  obtain ⟨hCard, hExists⟩ :=
    finite_projected_aligned_geometric_lipschitz_atlas_coverage
      hAtlasSize hCover hProjection hProjectionNonexpansive hApproximation
      hTruthProxy hLipschitz hLNonnegative hCenterDepth hDepth
  refine ⟨sourceOnlyProposal_targetLabel_invariant
    fitProposal sourceRecords leftTargetLabels rightTargetLabels,
    hCard, hExists, ?_, fixedTrajectoryVarianceDecomposition risk, ?_⟩
  · exact joint_coordinate_equivalence_preserves_margin
      model beta z tau hEta hPsi
  · exact implementation_certifies_true_quantile
      hz hMean hSigma hMargin

end SCOLHKG.Real

import Mathlib
import SCOLHKG.Real.CertificationImplementation
import SCOLHKG.Real.SourceConstraintMeanPosterior

namespace SCOLHKG.Real

/-!
Formal contract for the source-aligned chance-boundary coordinate `phi`.

The implementation audits four margins after optimization:

* the fitted constraint mean and fitted cumulative variance;
* fitted mean with oracle variance;
* oracle mean with fitted variance;
* oracle mean and oracle variance, retaining only epistemic safety radius.

The identities below justify that diagnostic decomposition exactly. They do
not assume that source alignment succeeds on a held-out domain. Coordinate
sufficiency is stated explicitly as an assumption and tested empirically by a
paired source-only gate.
-/

def chanceMargin
    (constraintMean z constraintSigma tau : ℝ) : ℝ :=
  constraintMean + z * constraintSigma - tau

def guardedChanceMargin
    (constraintMean epistemicRadius z constraintSigma tau : ℝ) : ℝ :=
  constraintMean + epistemicRadius + z * constraintSigma - tau

theorem fitted_minus_oracle_variance_margin
    (predictedMean epistemicRadius z predictedSigma trueSigma tau : ℝ) :
    guardedChanceMargin
        predictedMean epistemicRadius z predictedSigma tau
      - guardedChanceMargin predictedMean epistemicRadius z trueSigma tau
      = z * (predictedSigma - trueSigma) := by
  unfold guardedChanceMargin
  ring

theorem fitted_minus_oracle_mean_margin
    (predictedMean trueMean epistemicRadius z predictedSigma tau : ℝ) :
    guardedChanceMargin
        predictedMean epistemicRadius z predictedSigma tau
      - guardedChanceMargin trueMean epistemicRadius z predictedSigma tau
      = predictedMean - trueMean := by
  unfold guardedChanceMargin
  ring

theorem oracle_both_minus_true_margin
    (trueMean epistemicRadius z trueSigma tau : ℝ) :
    guardedChanceMargin trueMean epistemicRadius z trueSigma tau
      - chanceMargin trueMean z trueSigma tau
      = epistemicRadius := by
  unfold guardedChanceMargin chanceMargin
  ring

theorem oracle_both_certifiable_iff_radius_le_safety_depth
    (trueMean epistemicRadius z trueSigma tau : ℝ) :
    guardedChanceMargin trueMean epistemicRadius z trueSigma tau ≤ 0
      ↔ epistemicRadius ≤ -chanceMargin trueMean z trueSigma tau := by
  unfold guardedChanceMargin chanceMargin
  constructor <;> intro h <;> linarith

def CoordinateSufficient
    {X Phi : Type*} (phi : X → Phi) (quantity : X → ℝ) : Prop :=
  ∃ decoder : Phi → ℝ, quantity = decoder ∘ phi

def FactorsThrough
    {X Exposure Quantity : Type*}
    (exposure : X → Exposure) (quantity : X → Quantity) : Prop :=
  ∃ head : Exposure → Quantity, quantity = head ∘ exposure

theorem shared_observable_exposure_preserves_both_heads
    {X Exposure : Type*}
    {exposure : X → Exposure}
    {constraintMean cumulativeVariance : X → ℝ}
    (hMean : FactorsThrough exposure constraintMean)
    (hVariance : FactorsThrough exposure cumulativeVariance)
    {x y : X}
    (hObservable : exposure x = exposure y) :
    constraintMean x = constraintMean y
      ∧ cumulativeVariance x = cumulativeVariance y := by
  rcases hMean with ⟨meanHead, hMeanHead⟩
  rcases hVariance with ⟨varianceHead, hVarianceHead⟩
  constructor
  · rw [hMeanHead]
    exact congrArg meanHead hObservable
  · rw [hVarianceHead]
    exact congrArg varianceHead hObservable

theorem coordinate_heads_factor_through_shared_exposure
    {X Exposure Phi Psi : Type*}
    (exposure : X → Exposure)
    (meanCoordinate : Exposure → Phi)
    (riskCoordinate : Exposure → Psi)
    (meanHead : Phi → ℝ)
    (varianceHead : Psi → ℝ) :
    FactorsThrough exposure (meanHead ∘ meanCoordinate ∘ exposure)
      ∧ FactorsThrough exposure
        (varianceHead ∘ riskCoordinate ∘ exposure) := by
  constructor
  · exact ⟨meanHead ∘ meanCoordinate, rfl⟩
  · exact ⟨varianceHead ∘ riskCoordinate, rfl⟩

theorem coordinate_equivalence_preserves_sufficient_quantity
    {X Phi : Type*}
    {phi : X → Phi} {quantity : X → ℝ}
    (hSufficient : CoordinateSufficient phi quantity)
    {x y : X}
    (hCoordinate : phi x = phi y) :
    quantity x = quantity y := by
  rcases hSufficient with ⟨decoder, hDecoder⟩
  rw [hDecoder]
  exact congrArg decoder hCoordinate

theorem aligned_coordinate_certificate_sound
    {X Phi : Type*}
    {phi : X → Phi}
    {predictedMean epistemicVar certificationVariance : Phi → ℝ}
    {trueMean trueSigma : X → ℝ}
    {beta z tau : ℝ}
    {x : X}
    (hz : 0 ≤ z)
    (hMean :
      trueMean x ≤ predictedMean (phi x)
        + implementationEpistemicSlack beta (epistemicVar (phi x)))
    (hSigma :
      trueSigma x ≤ implementationCertSigma
        (certificationVariance (phi x)))
    (hCertificate :
      theoryCertificationMargin
        (predictedMean (phi x))
        beta
        (epistemicVar (phi x))
        z
        (certificationVariance (phi x))
        tau ≤ 0) :
    trueMean x + z * trueSigma x ≤ tau := by
  exact implementation_certifies_true_quantile
    hz hMean hSigma hCertificate

noncomputable def separatedCoordinateCertificationMargin
    {Phi Psi : Type*}
    (predictedMean epistemicVar : Phi → ℝ)
    (certificationVariance : Psi → ℝ)
    (phi : Phi) (psi : Psi) (beta z tau : ℝ) : ℝ :=
  theoryCertificationMargin
    (predictedMean phi) beta (epistemicVar phi) z
    (certificationVariance psi) tau

theorem shared_observable_exposure_preserves_separated_margin
    {X Exposure Phi Psi : Type*}
    (exposure : X → Exposure)
    (meanCoordinate : Exposure → Phi)
    (riskCoordinate : Exposure → Psi)
    (predictedMean epistemicVar : Phi → ℝ)
    (certificationVariance : Psi → ℝ)
    (beta z tau : ℝ)
    {x y : X}
    (hObservable : exposure x = exposure y) :
    separatedCoordinateCertificationMargin
        predictedMean epistemicVar certificationVariance
        (meanCoordinate (exposure x)) (riskCoordinate (exposure x))
        beta z tau
      = separatedCoordinateCertificationMargin
        predictedMean epistemicVar certificationVariance
        (meanCoordinate (exposure y)) (riskCoordinate (exposure y))
        beta z tau := by
  rw [hObservable]

theorem separated_coordinate_certificate_sound
    {X Phi Psi : Type*}
    {phi : X → Phi} {psi : X → Psi}
    {predictedMean epistemicVar : Phi → ℝ}
    {certificationVariance : Psi → ℝ}
    {trueMean trueSigma : X → ℝ}
    {beta z tau : ℝ}
    {x : X}
    (hz : 0 ≤ z)
    (hMean :
      trueMean x ≤ predictedMean (phi x)
        + implementationEpistemicSlack beta (epistemicVar (phi x)))
    (hSigma :
      trueSigma x ≤ implementationCertSigma
        (certificationVariance (psi x)))
    (hCertificate :
      separatedCoordinateCertificationMargin
        predictedMean epistemicVar certificationVariance
        (phi x) (psi x) beta z tau ≤ 0) :
    trueMean x + z * trueSigma x ≤ tau := by
  exact implementation_certifies_true_quantile
    hz hMean hSigma hCertificate

theorem shared_observable_exposure_separate_heads_sound
    {X Exposure Phi Psi : Type*}
    {exposure : X → Exposure}
    {meanCoordinate : Exposure → Phi}
    {riskCoordinate : Exposure → Psi}
    {predictedMean epistemicVar : Phi → ℝ}
    {certificationVariance : Psi → ℝ}
    {trueMean trueSigma : X → ℝ}
    {beta z tau : ℝ}
    {x : X}
    (hz : 0 ≤ z)
    (hMean :
      trueMean x ≤ predictedMean (meanCoordinate (exposure x))
        + implementationEpistemicSlack beta
          (epistemicVar (meanCoordinate (exposure x))))
    (hSigma :
      trueSigma x ≤ implementationCertSigma
        (certificationVariance (riskCoordinate (exposure x))))
    (hCertificate :
      separatedCoordinateCertificationMargin
        predictedMean epistemicVar certificationVariance
        (meanCoordinate (exposure x))
        (riskCoordinate (exposure x)) beta z tau ≤ 0) :
    trueMean x + z * trueSigma x ≤ tau := by
  apply implementation_certifies_true_quantile hz hMean hSigma
  simpa [separatedCoordinateCertificationMargin] using hCertificate

theorem mean_head_difference_does_not_change_risk_head
    {Phi Psi : Type*}
    (firstMean secondMean epistemicVar : Phi → ℝ)
    (certificationVariance : Psi → ℝ)
    (phi : Phi) (psi : Psi) (beta z tau : ℝ) :
    separatedCoordinateCertificationMargin
        firstMean epistemicVar certificationVariance
        phi psi beta z tau
      - separatedCoordinateCertificationMargin
        secondMean epistemicVar certificationVariance
        phi psi beta z tau
      = firstMean phi - secondMean phi := by
  unfold separatedCoordinateCertificationMargin theoryCertificationMargin
  ring

theorem risk_head_difference_does_not_change_mean_head
    {Phi Psi : Type*}
    (predictedMean epistemicVar : Phi → ℝ)
    (firstVariance secondVariance : Psi → ℝ)
    (phi : Phi) (psi : Psi) (beta z tau : ℝ) :
    separatedCoordinateCertificationMargin
        predictedMean epistemicVar firstVariance
        phi psi beta z tau
      - separatedCoordinateCertificationMargin
        predictedMean epistemicVar secondVariance
        phi psi beta z tau
      = z * (
          implementationCertSigma (firstVariance psi)
          - implementationCertSigma (secondVariance psi)) := by
  unfold separatedCoordinateCertificationMargin theoryCertificationMargin
    implementationCertSigma
  ring

theorem selected_infeasible_of_candidate_support_failure
    {X : Type*}
    {candidates : Set X}
    {trueMargin : X → ℝ}
    {selected : X}
    (hSelected : selected ∈ candidates)
    (hNoFeasible : ∀ x ∈ candidates, 0 < trueMargin x) :
    0 < trueMargin selected := by
  exact hNoFeasible selected hSelected

theorem proposal_restores_candidate_support
    {X : Type*}
    {base proposal : Set X}
    {trueMargin : X → ℝ}
    {x : X}
    (hProposal : x ∈ proposal)
    (hFeasible : trueMargin x ≤ 0) :
    ∃ y ∈ base ∪ proposal, trueMargin y ≤ 0 := by
  exact ⟨x, Set.mem_union_right base hProposal, hFeasible⟩

theorem nonnegative_transfer_guard_cannot_relax_margin
    (mean epistemicRadius z sigma tau transferGuard : ℝ)
    (hGuard : 0 ≤ transferGuard) :
    guardedChanceMargin mean epistemicRadius z sigma tau
      ≤ guardedChanceMargin
          mean (epistemicRadius + transferGuard) z sigma tau := by
  unfold guardedChanceMargin
  linarith

/-! ## Source discrepancy in the transferable mean coordinate

The runtime V4 bridge decomposes a source residual variance `r`, estimated from
`m` source records, into the finite floor `r / m` and an isotropic coefficient
covariance whose trace is `r - r / m`.  With `p` coefficient features, the
per-coordinate inflation is `(r - r / m) / p`.  Thus the reference feature
energy `p` preserves the original uncertainty exactly; only its geometry has
changed from raw-policy independent to coordinate shared.
-/

noncomputable def sourceResidualFloor (r : ℝ) (m : ℕ) : ℝ :=
  r / (m : ℝ)

noncomputable def sourceLatentDiscrepancyMass (r : ℝ) (m : ℕ) : ℝ :=
  r - sourceResidualFloor r m

noncomputable def sourceLatentVariancePerCoefficient
    (r : ℝ) (m p : ℕ) : ℝ :=
  sourceLatentDiscrepancyMass r m / (p : ℝ)

theorem source_discrepancy_reference_variance_preserved
    (r : ℝ) (m p : ℕ) (hp : 0 < p) :
    (p : ℝ) * sourceLatentVariancePerCoefficient r m p
        + sourceResidualFloor r m = r := by
  have hpReal : (p : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hp
  unfold sourceLatentVariancePerCoefficient
    sourceLatentDiscrepancyMass
  rw [mul_div_cancel₀ _ hpReal]
  ring

theorem source_residual_floor_nonnegative
    (r : ℝ) (m : ℕ) (hr : 0 ≤ r) :
    0 ≤ sourceResidualFloor r m := by
  unfold sourceResidualFloor
  positivity

theorem source_latent_discrepancy_mass_nonnegative
    (r : ℝ) (m : ℕ) (hr : 0 ≤ r) (hm : 1 ≤ m) :
    0 ≤ sourceLatentDiscrepancyMass r m := by
  have hmReal : (1 : ℝ) ≤ (m : ℝ) := by exact_mod_cast hm
  unfold sourceLatentDiscrepancyMass sourceResidualFloor
  have hDiv : r / (m : ℝ) ≤ r := by
    exact (div_le_iff₀ (by positivity : (0 : ℝ) < (m : ℝ))).2
      (by nlinarith)
  linarith

theorem latent_discrepancy_contraction_reduces_epistemic_variance
    (posteriorCoefficient priorCoefficient residualFloor : ℝ)
    (hContract : posteriorCoefficient ≤ priorCoefficient) :
    posteriorCoefficient + residualFloor
      ≤ priorCoefficient + residualFloor := by
  linarith

/-! ## V5 equivariant role matching

The implementation learns role prototypes on source domains and transports a
held-out target channel into those roles using unlabeled observable exposure.
The theorem below isolates the algebraic equivariance contract: if both channel
values and the inverse role assignment are transported by the same channel
permutation, every role receives exactly the same value. Any downstream
descriptor therefore remains unchanged.
-/

def alignedRoleValues
    {Channel Role Value : Type*}
    (channelValue : Channel → Value)
    (roleChannel : Role → Channel) : Role → Value :=
  fun role => channelValue (roleChannel role)

theorem aligned_role_values_equivariant
    {Channel Role Value : Type*}
    (channelValue : Channel → Value)
    (roleChannel : Role → Channel)
    (permutation : Channel ≃ Channel) :
    alignedRoleValues
        (fun channel => channelValue (permutation.symm channel))
        (fun role => permutation (roleChannel role))
      = alignedRoleValues channelValue roleChannel := by
  funext role
  simp [alignedRoleValues]

theorem equivariant_role_matching_preserves_descriptor
    {Channel Role Value Descriptor : Type*}
    (channelValue : Channel → Value)
    (roleChannel : Role → Channel)
    (permutation : Channel ≃ Channel)
    (descriptor : (Role → Value) → Descriptor) :
    descriptor
        (alignedRoleValues
          (fun channel => channelValue (permutation.symm channel))
          (fun role => permutation (roleChannel role)))
      = descriptor (alignedRoleValues channelValue roleChannel) := by
  rw [aligned_role_values_equivariant]

/-! ## Conservative source-mean misspecification posterior

The runtime uses a scale posterior
`max 1 ((nu + q) / (nu + n))` and optionally adds a PSD directional
coefficient covariance. These finite real theorems bridge the two properties
required by certification: the scale is never below one, and a nonnegative
quadratic form plus a nonnegative directional form can only increase.
-/

noncomputable def conservativeMisspecificationScale
    (priorDf mahalanobis targetCount : ℝ) : ℝ :=
  max 1 ((priorDf + mahalanobis) / (priorDf + targetCount))

theorem one_le_conservative_misspecification_scale
    (priorDf mahalanobis targetCount : ℝ) :
    1 ≤ conservativeMisspecificationScale
      priorDf mahalanobis targetCount := by
  unfold conservativeMisspecificationScale
  exact le_max_left _ _

theorem psd_directional_misspecification_can_only_increase
    (priorQuadratic directionalQuadratic scale : ℝ)
    (hPrior : 0 ≤ priorQuadratic)
    (hDirectional : 0 ≤ directionalQuadratic)
    (hScale : 1 ≤ scale) :
    priorQuadratic
      ≤ scale * priorQuadratic + directionalQuadratic := by
  have hScaleNonnegative : 0 ≤ scale - 1 := by linarith
  have hInflation : 0 ≤ (scale - 1) * priorQuadratic :=
    mul_nonneg hScaleNonnegative hPrior
  nlinarith

theorem misspecification_floor_can_only_increase
    (residualFloor scale : ℝ)
    (hFloor : 0 ≤ residualFloor)
    (hScale : 1 ≤ scale) :
    residualFloor ≤ scale * residualFloor := by
  nlinarith [mul_nonneg (sub_nonneg.mpr hScale) hFloor]

theorem total_misspecification_uncertainty_can_only_increase
    (priorQuadratic directionalQuadratic residualFloor scale : ℝ)
    (hPrior : 0 ≤ priorQuadratic)
    (hDirectional : 0 ≤ directionalQuadratic)
    (hFloor : 0 ≤ residualFloor)
    (hScale : 1 ≤ scale) :
    priorQuadratic + residualFloor
      ≤ scale * priorQuadratic + directionalQuadratic
        + scale * residualFloor := by
  have hCoefficient := psd_directional_misspecification_can_only_increase
    priorQuadratic directionalQuadratic scale hPrior hDirectional hScale
  have hResidual := misspecification_floor_can_only_increase
    residualFloor scale hFloor hScale
  linarith

/-! ## V6 online hierarchical misspecification

V6 retains a frozen source law and accumulates one nonnegative standardized
innovation square per charged target observation.  At every step the runtime
recomputes the scale from the complete sufficient statistic and refits from
that frozen law; it never compounds a scale onto an already-scaled posterior.
The scale itself need not be monotone as new evidence arrives.  What remains
conservative at every step is the comparison with the same frozen source law.
-/

def updateMisspecificationStatistic
    (mahalanobis innovationSquare : ℝ) : ℝ :=
  mahalanobis + innovationSquare

theorem online_misspecification_statistic_nonnegative
    (mahalanobis innovationSquare : ℝ)
    (hMahalanobis : 0 ≤ mahalanobis)
    (hInnovation : 0 ≤ innovationSquare) :
    0 ≤ updateMisspecificationStatistic mahalanobis innovationSquare := by
  unfold updateMisspecificationStatistic
  linarith

theorem online_misspecification_statistic_accumulates
    (mahalanobis innovationSquare : ℝ)
    (hInnovation : 0 ≤ innovationSquare) :
    mahalanobis
      ≤ updateMisspecificationStatistic mahalanobis innovationSquare := by
  unfold updateMisspecificationStatistic
  linarith

noncomputable def onlineHierarchicalScale
    (priorDf accumulatedSquare targetCount : ℝ) : ℝ :=
  conservativeMisspecificationScale
    priorDf accumulatedSquare targetCount

theorem one_le_online_hierarchical_scale
    (priorDf accumulatedSquare targetCount : ℝ) :
    1 ≤ onlineHierarchicalScale
      priorDf accumulatedSquare targetCount := by
  unfold onlineHierarchicalScale
  exact one_le_conservative_misspecification_scale
    priorDf accumulatedSquare targetCount

def refitVarianceFromFrozenLaw
    (frozenVariance scale : ℝ) : ℝ :=
  scale * frozenVariance

theorem online_refit_never_understates_frozen_variance
    (frozenVariance scale : ℝ)
    (hFrozen : 0 ≤ frozenVariance)
    (hScale : 1 ≤ scale) :
    frozenVariance ≤ refitVarianceFromFrozenLaw frozenVariance scale := by
  unfold refitVarianceFromFrozenLaw
  nlinarith [mul_nonneg (sub_nonneg.mpr hScale) hFrozen]

theorem online_refit_total_uncertainty_never_understates_frozen_law
    (frozenCoefficient frozenResidual scale : ℝ)
    (hCoefficient : 0 ≤ frozenCoefficient)
    (hResidual : 0 ≤ frozenResidual)
    (hScale : 1 ≤ scale) :
    frozenCoefficient + frozenResidual
      ≤ refitVarianceFromFrozenLaw frozenCoefficient scale
        + refitVarianceFromFrozenLaw frozenResidual scale := by
  have hCoefficientBound := online_refit_never_understates_frozen_variance
    frozenCoefficient scale hCoefficient hScale
  have hResidualBound := online_refit_never_understates_frozen_variance
    frozenResidual scale hResidual hScale
  linarith

theorem target_null_unit_scale_is_unchanged
    (coefficient residual : ℝ) :
    refitVarianceFromFrozenLaw coefficient 1
        + refitVarianceFromFrozenLaw residual 1
      = coefficient + residual := by
  simp [refitVarianceFromFrozenLaw]

/-! ## V7 role support and source-contrast calibration

The role matcher standardizes every signature coordinate with source-only
moments.  Runtime trust is `exp (-max (lossPerCoordinate - 1) 0 / 2)`, hence it
is a valid probability mass and can only remove source mass.  Separately, a
source-contrast random effect contributes a nonnegative projected variance.
The two mechanisms calibrate semantic support and coefficient misspecification
without consulting target truth.
-/

noncomputable def roleSupportTrust (lossPerCoordinate : ℝ) : ℝ :=
  Real.exp (-max (lossPerCoordinate - 1) 0 / 2)

theorem role_support_trust_positive (lossPerCoordinate : ℝ) :
    0 < roleSupportTrust lossPerCoordinate := by
  exact Real.exp_pos _

theorem role_support_trust_le_one (lossPerCoordinate : ℝ) :
    roleSupportTrust lossPerCoordinate ≤ 1 := by
  have hMax : 0 ≤ max (lossPerCoordinate - 1) 0 := le_max_right _ _
  have hExponent : -max (lossPerCoordinate - 1) 0 / 2 ≤ 0 := by
    linarith
  calc
    roleSupportTrust lossPerCoordinate
        = Real.exp (-max (lossPerCoordinate - 1) 0 / 2) := rfl
    _ ≤ Real.exp 0 := Real.exp_le_exp.mpr hExponent
    _ = 1 := Real.exp_zero

theorem role_support_cannot_increase_source_mass
    (sourceMass lossPerCoordinate : ℝ)
    (hSource : 0 ≤ sourceMass) :
    sourceMass * roleSupportTrust lossPerCoordinate ≤ sourceMass := by
  have hTrust := role_support_trust_le_one lossPerCoordinate
  nlinarith [role_support_trust_positive lossPerCoordinate]

def sourceContrastPredictiveGain
    (contrastVariance projectedFeature : ℝ) : ℝ :=
  contrastVariance * projectedFeature ^ 2

theorem source_contrast_predictive_gain_nonnegative
    (contrastVariance projectedFeature : ℝ)
    (hVariance : 0 ≤ contrastVariance) :
    0 ≤ sourceContrastPredictiveGain contrastVariance projectedFeature := by
  unfold sourceContrastPredictiveGain
  positivity

theorem source_contrast_can_only_increase_epistemic_variance
    (priorVariance contrastVariance projectedFeature : ℝ)
    (hContrast : 0 ≤ contrastVariance) :
    priorVariance ≤ priorVariance
      + sourceContrastPredictiveGain contrastVariance projectedFeature := by
  have hGain := source_contrast_predictive_gain_nonnegative
    contrastVariance projectedFeature hContrast
  linarith

/-! ## V8 source-support adaptive coordinate

The runtime fits the role-aligned coordinate and its fallback independently
from the frozen source archive.  A selector that depends only on observable
channel cardinalities chooses the role coordinate when that cardinality was
represented in the source archive and the fallback otherwise.  The finite
result below deliberately does not claim that either coordinate is sufficient:
it says that selecting between two already-sound certification bridges
preserves soundness.
-/

def supportAdaptiveCoordinate
    {Coordinate : Type*}
    (supported : Bool) (role fallback : Coordinate) : Coordinate :=
  if supported then role else fallback

@[simp] theorem support_adaptive_coordinate_of_supported
    {Coordinate : Type*} (role fallback : Coordinate) :
    supportAdaptiveCoordinate true role fallback = role := by
  rfl

@[simp] theorem support_adaptive_coordinate_of_unsupported
    {Coordinate : Type*} (role fallback : Coordinate) :
    supportAdaptiveCoordinate false role fallback = fallback := by
  rfl

theorem support_adaptive_coordinate_preserves_property
    {Coordinate : Type*}
    (supported : Bool) (role fallback : Coordinate)
    (property : Coordinate → Prop)
    (hRole : property role)
    (hFallback : property fallback) :
    property (supportAdaptiveCoordinate supported role fallback) := by
  cases supported <;> simp [supportAdaptiveCoordinate, hRole, hFallback]

theorem support_adaptive_certificate_sound
    (supported : Bool)
    (roleMargin fallbackMargin trueQuantile tau : ℝ)
    (hRoleSound : roleMargin ≤ 0 → trueQuantile ≤ tau)
    (hFallbackSound : fallbackMargin ≤ 0 → trueQuantile ≤ tau)
    (hSelectedCertificate :
      supportAdaptiveCoordinate supported roleMargin fallbackMargin ≤ 0) :
    trueQuantile ≤ tau := by
  cases supported
  · exact hFallbackSound (by simpa using hSelectedCertificate)
  · exact hRoleSound (by simpa using hSelectedCertificate)

/-! ## V12 bounded alignment and target feature geometry

The source-selected temperature changes the argument of `tanh` but not its
universal range.  Hence every transformed latent channel is strictly bounded
without reading a target response.  The target-null geometry rescales a PSD
inverse-Gram quadratic form so that its average predictive variance equals the
previous isotropic reference.  The finite scalar bridge below is the exact
normalization used after the matrix quadratic forms have been evaluated.
-/

noncomputable def sourceTanhCoordinate
    (latent temperature : ℝ) : ℝ :=
  Real.tanh (latent / temperature)

theorem source_tanh_coordinate_abs_lt_one
    (latent temperature : ℝ) :
    |sourceTanhCoordinate latent temperature| < 1 := by
  unfold sourceTanhCoordinate
  exact Real.abs_tanh_lt_one _

noncomputable def predictiveGeometryScale
    (referenceAverage rawGeometryAverage : ℝ) : ℝ :=
  referenceAverage / rawGeometryAverage

theorem target_geometry_preserves_average_predictive_variance
    (referenceAverage rawGeometryAverage : ℝ)
    (hRaw : rawGeometryAverage ≠ 0) :
    predictiveGeometryScale referenceAverage rawGeometryAverage
        * rawGeometryAverage
      = referenceAverage := by
  unfold predictiveGeometryScale
  exact div_mul_cancel₀ referenceAverage hRaw

theorem target_geometry_scale_nonnegative
    (referenceAverage rawGeometryAverage : ℝ)
    (hReference : 0 ≤ referenceAverage)
    (hRaw : 0 < rawGeometryAverage) :
    0 ≤ predictiveGeometryScale referenceAverage rawGeometryAverage := by
  unfold predictiveGeometryScale
  exact div_nonneg hReference (le_of_lt hRaw)

theorem scaled_target_geometry_quadratic_nonnegative
    (referenceAverage rawGeometryAverage quadratic : ℝ)
    (hReference : 0 ≤ referenceAverage)
    (hRaw : 0 < rawGeometryAverage)
    (hQuadratic : 0 ≤ quadratic) :
    0 ≤ predictiveGeometryScale referenceAverage rawGeometryAverage
      * quadratic := by
  exact mul_nonneg
    (target_geometry_scale_nonnegative
      referenceAverage rawGeometryAverage hReference hRaw)
    hQuadratic

/-! ## V13 source-support projection and discrepancy

Unlike the V12 global `tanh`, the V13 coordinate is exactly the identity on
the source-supported interval and clips only outside it. The optional residual
channel is zero on support and universally bounded off support. It can
therefore receive an independent linear-Gaussian coefficient without making
the target mean unbounded.
-/

def sourceSupportClip (latent bound : ℝ) : ℝ :=
  max (-bound) (min bound latent)

theorem source_support_clip_lower_bound
    (latent bound : ℝ) :
    -bound ≤ sourceSupportClip latent bound := by
  unfold sourceSupportClip
  exact le_max_left _ _

theorem source_support_clip_upper_bound
    (latent bound : ℝ) (hBound : 0 ≤ bound) :
    sourceSupportClip latent bound ≤ bound := by
  unfold sourceSupportClip
  apply max_le
  · linarith
  · exact min_le_left _ _

theorem source_support_clip_eq_self
    (latent bound : ℝ)
    (hLower : -bound ≤ latent)
    (hUpper : latent ≤ bound) :
    sourceSupportClip latent bound = latent := by
  simp [sourceSupportClip, min_eq_right hUpper, max_eq_right hLower]

def sourceSupportOverflow (latent bound : ℝ) : ℝ :=
  max (|latent| - bound) 0

theorem source_support_overflow_nonnegative
    (latent bound : ℝ) :
    0 ≤ sourceSupportOverflow latent bound := by
  unfold sourceSupportOverflow
  exact le_max_right _ _

noncomputable def sourceSupportResidual (latent bound : ℝ) : ℝ :=
  |Real.tanh (sourceSupportOverflow latent bound)|

theorem source_support_residual_nonnegative
    (latent bound : ℝ) :
    0 ≤ sourceSupportResidual latent bound := by
  unfold sourceSupportResidual
  exact abs_nonneg _

theorem source_support_residual_lt_one
    (latent bound : ℝ) :
    sourceSupportResidual latent bound < 1 := by
  unfold sourceSupportResidual
  exact Real.abs_tanh_lt_one _

theorem source_support_residual_eq_zero_on_support
    (latent bound : ℝ)
    (hSupport : |latent| ≤ bound) :
    sourceSupportResidual latent bound = 0 := by
  have hOverflow : sourceSupportOverflow latent bound = 0 := by
    unfold sourceSupportOverflow
    rw [max_eq_right]
    linarith
  simp [sourceSupportResidual, hOverflow]

/-! ## V16 partial role transport and epistemic calibration

The implementation learns a rectangular, outcome-free channel-to-role
transport on source-domain dropout episodes.  Each transport weight and role
mass is nonnegative.  A source/target semantic mismatch is represented only by
an epistemic covariance multiplier at least one; it never shifts the posterior
mean or decreases uncertainty.
-/

def partialRoleMass (weights : List ℝ) : ℝ :=
  weights.sum

theorem partial_role_mass_nonnegative
    (weights : List ℝ)
    (hWeights : ∀ weight ∈ weights, 0 ≤ weight) :
    0 ≤ partialRoleMass weights := by
  unfold partialRoleMass
  exact List.sum_nonneg hWeights

def roleMismatchEpistemicScale
    (excessLoss assignmentEntropy cardinalityGap : ℝ) : ℝ :=
  1 + max excessLoss 0 + max assignmentEntropy 0 + max cardinalityGap 0

theorem role_mismatch_epistemic_scale_ge_one
    (excessLoss assignmentEntropy cardinalityGap : ℝ) :
    1 ≤ roleMismatchEpistemicScale
      excessLoss assignmentEntropy cardinalityGap := by
  unfold roleMismatchEpistemicScale
  have hLoss : 0 ≤ max excessLoss 0 := le_max_right _ _
  have hEntropy : 0 ≤ max assignmentEntropy 0 := le_max_right _ _
  have hGap : 0 ≤ max cardinalityGap 0 := le_max_right _ _
  linarith

def roleMismatchInflatedVariance
    (priorVariance excessLoss assignmentEntropy cardinalityGap : ℝ) : ℝ :=
  roleMismatchEpistemicScale excessLoss assignmentEntropy cardinalityGap
    * priorVariance

theorem role_mismatch_inflation_cannot_decrease_variance
    (priorVariance excessLoss assignmentEntropy cardinalityGap : ℝ)
    (hPrior : 0 ≤ priorVariance) :
    priorVariance ≤ roleMismatchInflatedVariance priorVariance
      excessLoss assignmentEntropy cardinalityGap := by
  unfold roleMismatchInflatedVariance
  have hScale := role_mismatch_epistemic_scale_ge_one
    excessLoss assignmentEntropy cardinalityGap
  nlinarith

theorem role_mismatch_inflation_preserves_psd_scalar
    (priorVariance excessLoss assignmentEntropy cardinalityGap : ℝ)
    (hPrior : 0 ≤ priorVariance) :
    0 ≤ roleMismatchInflatedVariance priorVariance
      excessLoss assignmentEntropy cardinalityGap := by
  exact hPrior.trans
    (role_mismatch_inflation_cannot_decrease_variance
      priorVariance excessLoss assignmentEntropy cardinalityGap hPrior)

/-! ## V17 intervention-response roles

The target channel may span several source roles.  Its aligned response is a
barycenter of source-role responses.  Nonnegative weights with unit mass keep
every scalar response inside the source-role convex hull.  This is the finite
contract used by the intervention-response transport; it does not assume that
the learned barycenter is sufficient for a held-out chance boundary.
-/

noncomputable def barycentricRoleResponse {n : ℕ}
    (weights responses : Fin n → ℝ) : ℝ :=
  ∑ index, weights index * responses index

theorem barycentric_role_response_in_convex_hull {n : ℕ}
    (weights responses : Fin n → ℝ)
    (lower upper : ℝ)
    (hWeight : ∀ index, 0 ≤ weights index)
    (hMass : ∑ index, weights index = 1)
    (hLower : ∀ index, lower ≤ responses index)
    (hUpper : ∀ index, responses index ≤ upper) :
    lower ≤ barycentricRoleResponse weights responses ∧
      barycentricRoleResponse weights responses ≤ upper := by
  have hLowerSum :
      ∑ index, weights index * lower ≤
        ∑ index, weights index * responses index := by
    exact Finset.sum_le_sum fun index _ =>
      mul_le_mul_of_nonneg_left (hLower index) (hWeight index)
  have hUpperSum :
      ∑ index, weights index * responses index ≤
        ∑ index, weights index * upper := by
    exact Finset.sum_le_sum fun index _ =>
      mul_le_mul_of_nonneg_left (hUpper index) (hWeight index)
  constructor
  · calc
      lower = (∑ index, weights index) * lower := by rw [hMass]; simp
      _ = ∑ index, weights index * lower := by
        rw [Finset.sum_mul]
      _ ≤ barycentricRoleResponse weights responses := by
        simpa [barycentricRoleResponse] using hLowerSum
  · calc
      barycentricRoleResponse weights responses
          ≤ ∑ index, weights index * upper := by
            simpa [barycentricRoleResponse] using hUpperSum
      _ = (∑ index, weights index) * upper := by
        rw [Finset.sum_mul]
      _ = upper := by rw [hMass]; simp

def hierarchicalMeanVariance
    (isSource : Bool) (scale priorVariance : ℝ) : ℝ :=
  if isSource then max scale 1 * priorVariance else priorVariance

@[simp] theorem target_null_mean_variance_is_not_inflated
    (scale priorVariance : ℝ) :
    hierarchicalMeanVariance false scale priorVariance = priorVariance := by
  rfl

theorem source_hierarchical_mean_variance_cannot_decrease
    (scale priorVariance : ℝ)
    (hPrior : 0 ≤ priorVariance) :
    priorVariance ≤ hierarchicalMeanVariance true scale priorVariance := by
  unfold hierarchicalMeanVariance
  simp only [↓reduceIte]
  have hScale : 1 ≤ max scale 1 := le_max_right _ _
  nlinarith

/-! ## V18 target-orthogonal residual mean coordinate

The source mean span remains frozen. An outcome-free target design defines a
finite residual coordinate orthogonal to that span. Its coefficient law has
zero mean and an independent PSD covariance block, so adding the residual head
does not shift the source prior mean and cannot reduce predictive uncertainty.
Only charged target observations subsequently condition its coefficients.
-/

def directSumPriorMean (sourceMean residualMean : ℝ) : ℝ :=
  sourceMean + residualMean

@[simp] theorem zero_residual_prior_preserves_source_mean
    (sourceMean : ℝ) :
    directSumPriorMean sourceMean 0 = sourceMean := by
  simp [directSumPriorMean]

def directSumPredictiveVariance
    (sourceVariance residualVariance : ℝ) : ℝ :=
  sourceVariance + residualVariance

theorem independent_residual_cannot_reduce_predictive_variance
    (sourceVariance residualVariance : ℝ)
    (hResidual : 0 ≤ residualVariance) :
    sourceVariance ≤ directSumPredictiveVariance
      sourceVariance residualVariance := by
  unfold directSumPredictiveVariance
  linarith

theorem independent_residual_preserves_nonnegative_variance
    (sourceVariance residualVariance : ℝ)
    (hSource : 0 ≤ sourceVariance)
    (hResidual : 0 ≤ residualVariance) :
    0 ≤ directSumPredictiveVariance sourceVariance residualVariance := by
  unfold directSumPredictiveVariance
  linarith

theorem orthogonal_residual_energy_decomposition {n : ℕ}
    (source residual : Fin n → ℝ)
    (hOrthogonal : ∑ index, source index * residual index = 0) :
    (∑ index, (source index + residual index) ^ 2) =
      (∑ index, (source index) ^ 2) +
        (∑ index, (residual index) ^ 2) := by
  calc
    (∑ index, (source index + residual index) ^ 2) =
        ∑ index,
          ((source index) ^ 2
            + 2 * (source index * residual index)
            + (residual index) ^ 2) := by
      apply Finset.sum_congr rfl
      intro index _
      ring
    _ = (∑ index, (source index) ^ 2)
          + 2 * (∑ index, source index * residual index)
          + (∑ index, (residual index) ^ 2) := by
      rw [Finset.sum_add_distrib, Finset.sum_add_distrib]
      rw [← Finset.mul_sum]
    _ = (∑ index, (source index) ^ 2)
          + (∑ index, (residual index) ^ 2) := by
      rw [hOrthogonal]
      ring

/-! ## V19 Bayesian residual-rank structure

All rank atoms use one maximum-rank feature map. Inactive coefficients retain
only a nonnegative numerical variance, while active coefficients use their
source-scaled Gaussian variance. Ordinary target likelihoods update the finite
rank mixture. Moment matching retains both within-rank covariance and
between-rank disagreement.
-/

def nestedResidualCoefficientVariance
    (activeRank coefficient : ℕ)
    (activeVariance inactiveVariance : ℝ) : ℝ :=
  if coefficient < activeRank then activeVariance else inactiveVariance

theorem nested_residual_coefficient_variance_nonnegative
    (activeRank coefficient : ℕ)
    (activeVariance inactiveVariance : ℝ)
    (hActive : 0 ≤ activeVariance)
    (hInactive : 0 ≤ inactiveVariance) :
    0 ≤ nestedResidualCoefficientVariance
      activeRank coefficient activeVariance inactiveVariance := by
  unfold nestedResidualCoefficientVariance
  split_ifs <;> assumption

@[simp] theorem rank_zero_uses_inactive_residual_variance
    (coefficient : ℕ) (activeVariance inactiveVariance : ℝ) :
    nestedResidualCoefficientVariance
      0 coefficient activeVariance inactiveVariance = inactiveVariance := by
  simp [nestedResidualCoefficientVariance]

theorem residual_rank_mixture_disagreement_cannot_reduce_variance {k : ℕ}
    (weight mean withinVariance : Fin k → ℝ)
    (hWeight : ∀ index, 0 ≤ weight index) :
    (∑ index, weight index * withinVariance index)
      ≤ finiteMixtureDirectionalVariance weight mean withinVariance := by
  exact mixture_disagreement_cannot_reduce_directional_variance
    weight mean withinVariance hWeight

theorem target_evidence_rank_weights_are_probabilities {k : ℕ}
    (priorWeight likelihood : Fin k → ℝ)
    (hWeight : ∀ index, 0 ≤ priorWeight index)
    (hLikelihood : ∀ index, 0 ≤ likelihood index)
    (hMass : 0 < sequentialMixtureEvidenceMass priorWeight likelihood) :
    (∀ index, 0 ≤ sequentialMixtureWeight
      priorWeight likelihood index) ∧
      (∑ index, sequentialMixtureWeight
        priorWeight likelihood index = 1) := by
  constructor
  · exact sequentialMixtureWeight_nonnegative
      priorWeight likelihood hWeight hLikelihood hMass
  · exact sequentialMixtureWeight_sum_one
      priorWeight likelihood (ne_of_gt hMass)

/-! ## V20 finite channel-role assignment posterior

Every admissible injection of target channels into source-fitted canonical
roles is one atom of a finite structure posterior. The atom family is fixed
without target responses. Charged target likelihoods update only its weights.
Relabeling channels permutes the atom enumeration, so the mixture mean is
unchanged. Moment matching retains a nonnegative between-assignment variance.
-/

noncomputable def uniformAssignmentWeight (k : ℕ) : Fin k → ℝ :=
  fun _ => 1 / (k : ℝ)

theorem uniform_assignment_weights_sum_one
    (k : ℕ) (hk : 0 < k) :
    ∑ index : Fin k, uniformAssignmentWeight k index = 1 := by
  simp [uniformAssignmentWeight, hk.ne']

noncomputable def assignmentMixtureMean {k : ℕ}
    (weight atomMean : Fin k → ℝ) : ℝ :=
  ∑ index, weight index * atomMean index

theorem assignment_mixture_mean_permutation_invariant {k : ℕ}
    (weight atomMean : Fin k → ℝ)
    (permutation : Fin k ≃ Fin k) :
    assignmentMixtureMean
        (fun index => weight (permutation index))
        (fun index => atomMean (permutation index))
      = assignmentMixtureMean weight atomMean := by
  unfold assignmentMixtureMean
  exact permutation.sum_comp
    (fun index => weight index * atomMean index)

theorem assignment_mixture_disagreement_cannot_reduce_variance {k : ℕ}
    (weight atomMean withinVariance : Fin k → ℝ)
    (hWeight : ∀ index, 0 ≤ weight index) :
    (∑ index, weight index * withinVariance index)
      ≤ finiteMixtureDirectionalVariance weight atomMean withinVariance := by
  exact mixture_disagreement_cannot_reduce_directional_variance
    weight atomMean withinVariance hWeight

theorem target_evidence_assignment_weights_are_probabilities {k : ℕ}
    (priorWeight likelihood : Fin k → ℝ)
    (hWeight : ∀ index, 0 ≤ priorWeight index)
    (hLikelihood : ∀ index, 0 ≤ likelihood index)
    (hMass : 0 < sequentialMixtureEvidenceMass priorWeight likelihood) :
    (∀ index, 0 ≤ sequentialMixtureWeight
      priorWeight likelihood index) ∧
      (∑ index, sequentialMixtureWeight
        priorWeight likelihood index = 1) := by
  constructor
  · exact sequentialMixtureWeight_nonnegative
      priorWeight likelihood hWeight hLikelihood hMass
  · exact sequentialMixtureWeight_sum_one
      priorWeight likelihood (ne_of_gt hMass)

/-! ## V21 cross-fitted assignment evidence

Each assignment atom is scored only by predictions of charged target responses
that omit the response being scored.  Exponentiating a finite LOO score gives
a strictly positive generalized likelihood.  Consequently any finite prior
with at least one positive atom has positive evidence mass, and normalization
produces a probability distribution.  The role hypotheses remain fixed; only
their posterior mass changes.
-/

noncomputable def crossFittedScoreLikelihood {k : ℕ}
    (score : Fin k → ℝ) (temperature : ℝ) : Fin k → ℝ :=
  fun index => Real.exp (score index / temperature)

theorem cross_fitted_score_likelihood_positive {k : ℕ}
    (score : Fin k → ℝ) (temperature : ℝ) (index : Fin k) :
    0 < crossFittedScoreLikelihood score temperature index := by
  exact Real.exp_pos _

theorem cross_fitted_evidence_mass_positive {k : ℕ}
    (priorWeight score : Fin k → ℝ)
    (temperature : ℝ)
    (hWeight : ∀ index, 0 ≤ priorWeight index)
    (hPositive : ∃ index, 0 < priorWeight index) :
    0 < sequentialMixtureEvidenceMass priorWeight
      (crossFittedScoreLikelihood score temperature) := by
  unfold sequentialMixtureEvidenceMass
  apply Finset.sum_pos'
  · intro index _
    exact mul_nonneg (hWeight index)
      (le_of_lt (cross_fitted_score_likelihood_positive
        score temperature index))
  · rcases hPositive with ⟨index, hIndex⟩
    exact ⟨index, Finset.mem_univ index,
      mul_pos hIndex (cross_fitted_score_likelihood_positive
        score temperature index)⟩

theorem cross_fitted_assignment_weights_are_probabilities {k : ℕ}
    (priorWeight score : Fin k → ℝ)
    (temperature : ℝ)
    (hWeight : ∀ index, 0 ≤ priorWeight index)
    (hPositive : ∃ index, 0 < priorWeight index) :
    (∀ index, 0 ≤ sequentialMixtureWeight priorWeight
      (crossFittedScoreLikelihood score temperature) index) ∧
      (∑ index, sequentialMixtureWeight priorWeight
        (crossFittedScoreLikelihood score temperature) index = 1) := by
  have hMass := cross_fitted_evidence_mass_positive
    priorWeight score temperature hWeight hPositive
  constructor
  · exact sequentialMixtureWeight_nonnegative
      priorWeight (crossFittedScoreLikelihood score temperature)
      hWeight
      (fun index => le_of_lt
        (cross_fitted_score_likelihood_positive score temperature index))
      hMass
  · exact sequentialMixtureWeight_sum_one
      priorWeight (crossFittedScoreLikelihood score temperature)
      (ne_of_gt hMass)

@[simp] theorem cross_fitted_score_likelihood_relabel {k : ℕ}
    (score : Fin k → ℝ) (temperature : ℝ)
    (permutation : Fin k ≃ Fin k) (index : Fin k) :
    crossFittedScoreLikelihood (fun atom => score (permutation atom))
        temperature index
      = crossFittedScoreLikelihood score temperature (permutation index) := by
  rfl

/-! ## V22 source-geometry assignment prior

The source role atlas induces a nonnegative matching cost for every assignment.
The held-out target contributes only an unlabeled exposure distribution.  A
positive source-calibrated temperature maps negative costs to positive prior
likelihoods. Lower-cost assignments receive no less prior mass before charged
target evidence is observed.
-/

noncomputable def geometryAssignmentLikelihood {k : ℕ}
    (cost : Fin k → ℝ) (temperature : ℝ) : Fin k → ℝ :=
  fun index => Real.exp (-cost index / temperature)

theorem geometry_assignment_likelihood_positive {k : ℕ}
    (cost : Fin k → ℝ) (temperature : ℝ) (index : Fin k) :
    0 < geometryAssignmentLikelihood cost temperature index := by
  exact Real.exp_pos _

theorem lower_geometry_cost_has_no_less_likelihood {k : ℕ}
    (cost : Fin k → ℝ) (temperature : ℝ)
    (best other : Fin k)
    (hTemperature : 0 < temperature)
    (hCost : cost best ≤ cost other) :
    geometryAssignmentLikelihood cost temperature other
      ≤ geometryAssignmentLikelihood cost temperature best := by
  unfold geometryAssignmentLikelihood
  apply Real.exp_le_exp.mpr
  exact div_le_div_of_nonneg_right
    (neg_le_neg hCost) (le_of_lt hTemperature)

theorem geometry_assignment_prior_is_probability {k : ℕ}
    (priorWeight cost : Fin k → ℝ)
    (temperature : ℝ)
    (hWeight : ∀ index, 0 ≤ priorWeight index)
    (hPositive : ∃ index, 0 < priorWeight index) :
    (∀ index, 0 ≤ sequentialMixtureWeight priorWeight
      (geometryAssignmentLikelihood cost temperature) index) ∧
      (∑ index, sequentialMixtureWeight priorWeight
        (geometryAssignmentLikelihood cost temperature) index = 1) := by
  have hMass : 0 < sequentialMixtureEvidenceMass priorWeight
      (geometryAssignmentLikelihood cost temperature) := by
    unfold sequentialMixtureEvidenceMass
    apply Finset.sum_pos'
    · intro index _
      exact mul_nonneg (hWeight index)
        (le_of_lt (geometry_assignment_likelihood_positive
          cost temperature index))
    · rcases hPositive with ⟨index, hIndex⟩
      exact ⟨index, Finset.mem_univ index,
        mul_pos hIndex (geometry_assignment_likelihood_positive
          cost temperature index)⟩
  constructor
  · exact sequentialMixtureWeight_nonnegative
      priorWeight (geometryAssignmentLikelihood cost temperature)
      hWeight
      (fun index => le_of_lt
        (geometry_assignment_likelihood_positive cost temperature index))
      hMass
  · exact sequentialMixtureWeight_sum_one
      priorWeight (geometryAssignmentLikelihood cost temperature)
      (ne_of_gt hMass)

/-! ## V23 factorized assignment and misspecification posterior

The assignment marginal is learned from source geometry and unlabeled target
exposure. Charged target outcomes update only the conditional source/null
expert law inside each assignment. The joint law is therefore a hierarchical
product. Normalizing every conditional expert law preserves both total mass and
the assignment marginal, regardless of the target evidence used to obtain that
conditional law.
-/

def factorizedAssignmentExpertWeight {a e : ℕ}
    (assignmentMass : Fin a → ℝ)
    (conditionalExpertMass : Fin a → Fin e → ℝ)
    (assignment : Fin a) (expert : Fin e) : ℝ :=
  assignmentMass assignment * conditionalExpertMass assignment expert

theorem factorized_assignment_expert_weight_nonnegative {a e : ℕ}
    (assignmentMass : Fin a → ℝ)
    (conditionalExpertMass : Fin a → Fin e → ℝ)
    (hAssignment : ∀ assignment, 0 ≤ assignmentMass assignment)
    (hConditional : ∀ assignment expert,
      0 ≤ conditionalExpertMass assignment expert)
    (assignment : Fin a) (expert : Fin e) :
    0 ≤ factorizedAssignmentExpertWeight
      assignmentMass conditionalExpertMass assignment expert := by
  exact mul_nonneg (hAssignment assignment) (hConditional assignment expert)

theorem factorized_assignment_marginal_is_fixed {a e : ℕ}
    (assignmentMass : Fin a → ℝ)
    (conditionalExpertMass : Fin a → Fin e → ℝ)
    (assignment : Fin a)
    (hConditional : ∑ expert, conditionalExpertMass assignment expert = 1) :
    (∑ expert, factorizedAssignmentExpertWeight
      assignmentMass conditionalExpertMass assignment expert)
      = assignmentMass assignment := by
  simp only [factorizedAssignmentExpertWeight]
  rw [← Finset.mul_sum, hConditional, mul_one]

theorem factorized_assignment_expert_weight_sum_one {a e : ℕ}
    (assignmentMass : Fin a → ℝ)
    (conditionalExpertMass : Fin a → Fin e → ℝ)
    (hAssignment : ∑ assignment, assignmentMass assignment = 1)
    (hConditional : ∀ assignment,
      ∑ expert, conditionalExpertMass assignment expert = 1) :
    (∑ assignment, ∑ expert, factorizedAssignmentExpertWeight
      assignmentMass conditionalExpertMass assignment expert) = 1 := by
  calc
    (∑ assignment, ∑ expert, factorizedAssignmentExpertWeight
      assignmentMass conditionalExpertMass assignment expert)
        = ∑ assignment, assignmentMass assignment := by
          apply Finset.sum_congr rfl
          intro assignment _
          exact factorized_assignment_marginal_is_fixed
            assignmentMass conditionalExpertMass assignment
            (hConditional assignment)
    _ = 1 := hAssignment

theorem conditional_expert_evidence_cannot_change_assignment_marginal
    {a e : ℕ}
    (assignmentMass : Fin a → ℝ)
    (firstConditional secondConditional : Fin a → Fin e → ℝ)
    (assignment : Fin a)
    (hFirst : ∑ expert, firstConditional assignment expert = 1)
    (hSecond : ∑ expert, secondConditional assignment expert = 1) :
    (∑ expert, factorizedAssignmentExpertWeight
      assignmentMass firstConditional assignment expert)
      = ∑ expert, factorizedAssignmentExpertWeight
        assignmentMass secondConditional assignment expert := by
  rw [factorized_assignment_marginal_is_fixed
        assignmentMass firstConditional assignment hFirst]
  rw [factorized_assignment_marginal_is_fixed
        assignmentMass secondConditional assignment hSecond]

theorem hierarchical_misspecification_refit_preserves_assignment_marginal
    {a e : ℕ}
    (assignmentMass : Fin a → ℝ)
    (beforeScale afterScale : Fin a → Fin e → ℝ)
    (assignment : Fin a)
    (hBefore : ∑ expert, beforeScale assignment expert = 1)
    (hAfter : ∑ expert, afterScale assignment expert = 1) :
    (∑ expert, factorizedAssignmentExpertWeight
      assignmentMass beforeScale assignment expert)
      = ∑ expert, factorizedAssignmentExpertWeight
        assignmentMass afterScale assignment expert := by
  exact conditional_expert_evidence_cannot_change_assignment_marginal
    assignmentMass beforeScale afterScale assignment hBefore hAfter

/-! ## V25 charged-pilot boundary-role posterior

The source archive fixes a Gaussian law for the Fisher-transformed association
between each canonical role and the chance margin.  The charged target pilot
provides the corresponding noisy channel associations.  Every admissible
channel-to-role injection receives its Gaussian log likelihood. Exponentiation
therefore gives strictly positive assignment evidence, and normalizing against
the source-geometry prior gives a probability law. Relabeling target channels
and the injection together leaves the likelihood unchanged.

The resulting assignment law is frozen after the pilot. Source-mean contrast
covariance is then constructed independently inside each assignment block;
an inactive block has zero projected contrast gain.
-/

noncomputable def boundaryRoleLogLikelihood {c r : ℕ}
    (targetMean targetVariance : Fin c → ℝ)
    (roleMean roleVariance : Fin r → ℝ)
    (assignment : Fin c → Fin r) : ℝ :=
  -1 / 2 * ∑ channel,
    ((targetMean channel - roleMean (assignment channel)) ^ 2 /
        (targetVariance channel + roleVariance (assignment channel))
      + Real.log
        (targetVariance channel + roleVariance (assignment channel)))

noncomputable def boundaryRoleLikelihood {c r : ℕ}
    (targetMean targetVariance : Fin c → ℝ)
    (roleMean roleVariance : Fin r → ℝ)
    (assignment : Fin c → Fin r)
    (temperature : ℝ) : ℝ :=
  Real.exp (
    boundaryRoleLogLikelihood targetMean targetVariance
      roleMean roleVariance assignment / temperature)

theorem boundary_role_likelihood_positive {c r : ℕ}
    (targetMean targetVariance : Fin c → ℝ)
    (roleMean roleVariance : Fin r → ℝ)
    (assignment : Fin c → Fin r)
    (temperature : ℝ) :
    0 < boundaryRoleLikelihood targetMean targetVariance
      roleMean roleVariance assignment temperature := by
  exact Real.exp_pos _

theorem boundary_role_log_likelihood_channel_relabel {c r : ℕ}
    (targetMean targetVariance : Fin c → ℝ)
    (roleMean roleVariance : Fin r → ℝ)
    (assignment : Fin c → Fin r)
    (permutation : Fin c ≃ Fin c) :
    boundaryRoleLogLikelihood
        (fun channel => targetMean (permutation channel))
        (fun channel => targetVariance (permutation channel))
        roleMean roleVariance
        (fun channel => assignment (permutation channel))
      = boundaryRoleLogLikelihood targetMean targetVariance
        roleMean roleVariance assignment := by
  unfold boundaryRoleLogLikelihood
  congr 1
  exact permutation.sum_comp (fun channel =>
    ((targetMean channel - roleMean (assignment channel)) ^ 2 /
        (targetVariance channel + roleVariance (assignment channel))
      + Real.log
        (targetVariance channel + roleVariance (assignment channel))))

theorem boundary_role_likelihood_channel_relabel {c r : ℕ}
    (targetMean targetVariance : Fin c → ℝ)
    (roleMean roleVariance : Fin r → ℝ)
    (assignment : Fin c → Fin r)
    (temperature : ℝ)
    (permutation : Fin c ≃ Fin c) :
    boundaryRoleLikelihood
        (fun channel => targetMean (permutation channel))
        (fun channel => targetVariance (permutation channel))
        roleMean roleVariance
        (fun channel => assignment (permutation channel))
        temperature
      = boundaryRoleLikelihood targetMean targetVariance
        roleMean roleVariance assignment temperature := by
  unfold boundaryRoleLikelihood
  rw [boundary_role_log_likelihood_channel_relabel]

theorem boundary_role_assignment_weights_are_probabilities {k : ℕ}
    (priorWeight logLikelihood : Fin k → ℝ)
    (temperature : ℝ)
    (hWeight : ∀ index, 0 ≤ priorWeight index)
    (hPositive : ∃ index, 0 < priorWeight index) :
    (∀ index, 0 ≤ sequentialMixtureWeight priorWeight
      (fun atom => Real.exp (logLikelihood atom / temperature)) index) ∧
      (∑ index, sequentialMixtureWeight priorWeight
        (fun atom => Real.exp (logLikelihood atom / temperature)) index = 1) := by
  have hMass : 0 < sequentialMixtureEvidenceMass priorWeight
      (fun atom => Real.exp (logLikelihood atom / temperature)) := by
    unfold sequentialMixtureEvidenceMass
    apply Finset.sum_pos'
    · intro index _
      exact mul_nonneg (hWeight index) (le_of_lt (Real.exp_pos _))
    · rcases hPositive with ⟨index, hIndex⟩
      exact ⟨index, Finset.mem_univ index,
        mul_pos hIndex (Real.exp_pos _)⟩
  constructor
  · exact sequentialMixtureWeight_nonnegative
      priorWeight
      (fun atom => Real.exp (logLikelihood atom / temperature))
      hWeight (fun index => le_of_lt (Real.exp_pos _)) hMass
  · exact sequentialMixtureWeight_sum_one
      priorWeight
      (fun atom => Real.exp (logLikelihood atom / temperature))
      (ne_of_gt hMass)

theorem assignment_conditional_contrast_zero_outside_active_block
    (contrastVariance projectedFeature : ℝ)
    (hInactive : projectedFeature = 0) :
    sourceContrastPredictiveGain contrastVariance projectedFeature = 0 := by
  simp [sourceContrastPredictiveGain, hInactive]

/-! ## V26 exchangeable target-linear mean coordinate

The source archive no longer transfers a discrete channel-role identity.  It
transfers one common coefficient-block law, copied to every observable target
channel.  Paid target observations update the individual coefficient blocks.
Relabeling channels therefore relabels both features and coefficients while
leaving the scalar constraint mean and the exchangeable source law unchanged.
-/

noncomputable def exchangeableChannelLinearMean {c q : ℕ}
    (intercept : ℝ)
    (coefficient feature : Fin c → Fin q → ℝ) : ℝ :=
  intercept + ∑ channel, ∑ coordinate,
    coefficient channel coordinate * feature channel coordinate

theorem exchangeable_channel_linear_mean_relabel {c q : ℕ}
    (intercept : ℝ)
    (coefficient feature : Fin c → Fin q → ℝ)
    (permutation : Fin c ≃ Fin c) :
    exchangeableChannelLinearMean intercept
        (fun channel => coefficient (permutation channel))
        (fun channel => feature (permutation channel))
      = exchangeableChannelLinearMean intercept coefficient feature := by
  unfold exchangeableChannelLinearMean
  congr 1
  exact permutation.sum_comp (fun channel => ∑ coordinate,
    coefficient channel coordinate * feature channel coordinate)

noncomputable def exchangeableChannelLogScore {c q : ℕ}
    (blockScore : (Fin q → ℝ) → ℝ)
    (coefficient : Fin c → Fin q → ℝ) : ℝ :=
  ∑ channel, blockScore (coefficient channel)

theorem exchangeable_channel_log_score_relabel {c q : ℕ}
    (blockScore : (Fin q → ℝ) → ℝ)
    (coefficient : Fin c → Fin q → ℝ)
    (permutation : Fin c ≃ Fin c) :
    exchangeableChannelLogScore blockScore
        (fun channel => coefficient (permutation channel))
      = exchangeableChannelLogScore blockScore coefficient := by
  unfold exchangeableChannelLogScore
  exact permutation.sum_comp (fun channel => blockScore (coefficient channel))

noncomputable def exchangeableChannelPriorScore {c q : ℕ}
    (blockScore : (Fin q → ℝ) → ℝ)
    (coefficient : Fin c → Fin q → ℝ) : ℝ :=
  Real.exp (exchangeableChannelLogScore blockScore coefficient)

theorem exchangeable_channel_prior_score_relabel {c q : ℕ}
    (blockScore : (Fin q → ℝ) → ℝ)
    (coefficient : Fin c → Fin q → ℝ)
    (permutation : Fin c ≃ Fin c) :
    exchangeableChannelPriorScore blockScore
        (fun channel => coefficient (permutation channel))
      = exchangeableChannelPriorScore blockScore coefficient := by
  unfold exchangeableChannelPriorScore
  rw [exchangeable_channel_log_score_relabel]

/-! ## V27 single exchangeable empirical-Bayes hyperlaw

Source-domain identity is marginalized before target adaptation. Along every
coefficient-space direction, moment matching retains both the within-source
projected covariance and the squared between-source displacement. The target
posterior contains one aggregate Gaussian atom, so posterior conditioning does
not introduce mixture disagreement or a source-domain selector.
-/

noncomputable def empiricalBayesProjectedMean {k : ℕ}
    (weight projectedMean : Fin k → ℝ) : ℝ :=
  ∑ atom, weight atom * projectedMean atom

noncomputable def empiricalBayesProjectedVariance {k : ℕ}
    (weight withinVariance projectedMean : Fin k → ℝ)
    (aggregateMean : ℝ) : ℝ :=
  ∑ atom, weight atom *
    (withinVariance atom + (projectedMean atom - aggregateMean) ^ 2)

theorem empirical_bayes_projected_variance_nonnegative {k : ℕ}
    (weight withinVariance projectedMean : Fin k → ℝ)
    (aggregateMean : ℝ)
    (hWeight : ∀ atom, 0 ≤ weight atom)
    (hWithin : ∀ atom, 0 ≤ withinVariance atom) :
    0 ≤ empiricalBayesProjectedVariance
      weight withinVariance projectedMean aggregateMean := by
  unfold empiricalBayesProjectedVariance
  exact Finset.sum_nonneg fun atom _ =>
    mul_nonneg (hWeight atom)
      (add_nonneg (hWithin atom) (sq_nonneg _))

theorem singleton_empirical_bayes_mean
    (projectedMean : Fin 1 → ℝ) :
    empiricalBayesProjectedMean (fun _ => 1) projectedMean
      = projectedMean 0 := by
  simp [empiricalBayesProjectedMean]

theorem singleton_empirical_bayes_variance
    (withinVariance : Fin 1 → ℝ) (mean : ℝ) :
    empiricalBayesProjectedVariance
        (fun _ => 1) withinVariance (fun _ => mean) mean
      = withinVariance 0 := by
  simp [empiricalBayesProjectedVariance]

theorem append_one_target_observation_count {Observation : Type}
    (history : List Observation) (observation : Observation) :
    (history ++ [observation]).length = history.length + 1 := by
  simp

/-! ## V28 constraint-head authority separation

The aggregate target GPR is authoritative for the constraint mean and its
epistemic variance. Exactly one HVD head is authoritative for aleatoric risk.
Legacy task-joint mean and epistemic moments remain explicit arguments only to
state, and prove, that neither can affect the separated certificate or terminal
Bayes-margin mean.
-/

inductive HvdHeadAuthority where
  | taskRobust
  | directCumulative
  deriving DecidableEq

def authoritativeHvdVariance
    (authority : HvdHeadAuthority)
    (taskAleatoric cumulativeAleatoric : ℝ) : ℝ :=
  match authority with
  | .taskRobust => taskAleatoric
  | .directCumulative => cumulativeAleatoric

noncomputable def separatedConstraintUpper
    (authority : HvdHeadAuthority)
    (aggregateMean aggregateEpistemic : ℝ)
    (taskAleatoric cumulativeAleatoric : ℝ)
    (_legacyTaskMean _legacyTaskEpistemic : ℝ)
  (beta zAlpha tau : ℝ) : ℝ :=
  aggregateMean
    + Real.sqrt beta * Real.sqrt aggregateEpistemic
    + zAlpha * Real.sqrt
      (authoritativeHvdVariance authority
        taskAleatoric cumulativeAleatoric)
    - tau

noncomputable def separatedBayesMarginMean
    (authority : HvdHeadAuthority)
    (aggregateMean : ℝ)
    (taskAleatoric cumulativeAleatoric : ℝ)
    (_legacyTaskMean _legacyTaskEpistemic : ℝ)
    (zAlpha tau : ℝ) : ℝ :=
  aggregateMean
    + zAlpha * Real.sqrt
      (authoritativeHvdVariance authority
        taskAleatoric cumulativeAleatoric)
    - tau

theorem separated_constraint_upper_ignores_legacy_task_heads
    (authority : HvdHeadAuthority)
    (aggregateMean aggregateEpistemic : ℝ)
    (taskAleatoric cumulativeAleatoric : ℝ)
    (legacyMean₁ legacyEpistemic₁ legacyMean₂ legacyEpistemic₂ : ℝ)
    (beta zAlpha tau : ℝ) :
    separatedConstraintUpper authority aggregateMean aggregateEpistemic
        taskAleatoric cumulativeAleatoric legacyMean₁ legacyEpistemic₁
        beta zAlpha tau
      = separatedConstraintUpper authority aggregateMean aggregateEpistemic
        taskAleatoric cumulativeAleatoric legacyMean₂ legacyEpistemic₂
        beta zAlpha tau := by
  rfl

theorem direct_cumulative_authority_selects_only_cumulative_hvd
    (aggregateMean aggregateEpistemic taskAleatoric cumulativeAleatoric : ℝ)
    (legacyTaskMean legacyTaskEpistemic beta zAlpha tau : ℝ) :
    separatedConstraintUpper .directCumulative
        aggregateMean aggregateEpistemic
        taskAleatoric cumulativeAleatoric
        legacyTaskMean legacyTaskEpistemic beta zAlpha tau
      = aggregateMean
        + Real.sqrt beta * Real.sqrt aggregateEpistemic
        + zAlpha * Real.sqrt cumulativeAleatoric
        - tau := by
  rfl

theorem task_hvd_authority_selects_only_task_aleatoric
    (aggregateMean aggregateEpistemic taskAleatoric cumulativeAleatoric : ℝ)
    (legacyTaskMean legacyTaskEpistemic beta zAlpha tau : ℝ) :
    separatedConstraintUpper .taskRobust
        aggregateMean aggregateEpistemic
        taskAleatoric cumulativeAleatoric
        legacyTaskMean legacyTaskEpistemic beta zAlpha tau
      = aggregateMean
        + Real.sqrt beta * Real.sqrt aggregateEpistemic
        + zAlpha * Real.sqrt taskAleatoric
        - tau := by
  rfl

theorem separated_bayes_margin_ignores_legacy_task_heads
    (authority : HvdHeadAuthority)
    (aggregateMean taskAleatoric cumulativeAleatoric : ℝ)
    (legacyMean₁ legacyEpistemic₁ legacyMean₂ legacyEpistemic₂ : ℝ)
    (zAlpha tau : ℝ) :
    separatedBayesMarginMean authority aggregateMean
        taskAleatoric cumulativeAleatoric legacyMean₁ legacyEpistemic₁
        zAlpha tau
      = separatedBayesMarginMean authority aggregateMean
        taskAleatoric cumulativeAleatoric legacyMean₂ legacyEpistemic₂
        zAlpha tau := by
  rfl

/-! ## V29 posterior-dominance incumbent preservation

The runtime does not assume that incumbent and challenger posterior losses are
independent.  It upper-bounds the variance of their difference by the square
of the sum of posterior standard deviations.  Given the one-sided Cantelli
bound for that difference, accepting only when the corresponding improvement
lower bound is at least `1 - delta` controls the posterior false-switch
probability by `delta`.
-/

noncomputable def covarianceFreeDifferenceVarianceUpper
    (incumbentVariance challengerVariance : ℝ) : ℝ :=
  (Real.sqrt (max incumbentVariance 0)
    + Real.sqrt (max challengerVariance 0)) ^ 2

noncomputable def cantelliImprovementLowerBound
    (meanGain varianceUpper : ℝ) : ℝ :=
  meanGain ^ 2 / (meanGain ^ 2 + varianceUpper)

noncomputable def cantelliFalseSwitchUpperBound
    (meanGain varianceUpper : ℝ) : ℝ :=
  varianceUpper / (meanGain ^ 2 + varianceUpper)

theorem covariance_free_difference_variance_upper
    (incumbentVariance challengerVariance covariance : ℝ)
    (hIncumbent : 0 ≤ incumbentVariance)
    (hChallenger : 0 ≤ challengerVariance)
    (hCovariance :
      -Real.sqrt incumbentVariance * Real.sqrt challengerVariance
        ≤ covariance) :
    incumbentVariance + challengerVariance - 2 * covariance
      ≤ covarianceFreeDifferenceVarianceUpper
        incumbentVariance challengerVariance := by
  have hIncumbentSqrt :
      (Real.sqrt incumbentVariance) ^ 2 = incumbentVariance :=
    Real.sq_sqrt hIncumbent
  have hChallengerSqrt :
      (Real.sqrt challengerVariance) ^ 2 = challengerVariance :=
    Real.sq_sqrt hChallenger
  unfold covarianceFreeDifferenceVarianceUpper
  rw [max_eq_left hIncumbent, max_eq_left hChallenger]
  nlinarith

theorem cantelli_improvement_and_false_switch_sum_one
    (meanGain varianceUpper : ℝ)
    (hGain : 0 < meanGain)
    (hVariance : 0 ≤ varianceUpper) :
    cantelliImprovementLowerBound meanGain varianceUpper
      + cantelliFalseSwitchUpperBound meanGain varianceUpper = 1 := by
  have hDenominator : meanGain ^ 2 + varianceUpper ≠ 0 := by
    nlinarith [sq_pos_of_pos hGain]
  unfold cantelliImprovementLowerBound cantelliFalseSwitchUpperBound
  field_simp [hDenominator]

theorem accepted_cantelli_switch_controls_posterior_error
    (meanGain varianceUpper posteriorFalseSwitch delta : ℝ)
    (hGain : 0 < meanGain)
    (hVariance : 0 ≤ varianceUpper)
    (hCantelli :
      posteriorFalseSwitch
        ≤ cantelliFalseSwitchUpperBound meanGain varianceUpper)
    (hAccepted :
      1 - delta
        ≤ cantelliImprovementLowerBound meanGain varianceUpper) :
    posteriorFalseSwitch ≤ delta := by
  have hPartition := cantelli_improvement_and_false_switch_sum_one
    meanGain varianceUpper hGain hVariance
  linarith

/-! ## V30--V34 finite-sample mean-misspecification calibration

V30 multiplies a conditioned covariance by an upper scale that is clipped
below at one. V31 adds a local HC3 sandwich covariance. V34 first relaxes the
source hyperlaw before target conditioning and then adds the HC3 correction to
the resulting target posterior. The scalar projection lemmas below are the
implementation bridge needed by certification: neither operation can reduce
posterior uncertainty, and the post-conditioning sandwich operation does not
alter the already-conditioned posterior mean.
-/

def posteriorScaleVariance (baseVariance scale : ℝ) : ℝ :=
  scale * baseVariance

theorem posterior_scale_at_least_one_cannot_reduce_variance
    (baseVariance scale : ℝ)
    (hVariance : 0 ≤ baseVariance)
    (hScale : 1 ≤ scale) :
    baseVariance ≤ posteriorScaleVariance baseVariance scale := by
  unfold posteriorScaleVariance
  nlinarith

noncomputable def hc3ProjectedCorrection {n : ℕ}
    (weight projectedScore : Fin n → ℝ) : ℝ :=
  ∑ index, weight index * (projectedScore index) ^ 2

theorem hc3_projected_correction_nonnegative {n : ℕ}
    (weight projectedScore : Fin n → ℝ)
    (hWeight : ∀ index, 0 ≤ weight index) :
    0 ≤ hc3ProjectedCorrection weight projectedScore := by
  unfold hc3ProjectedCorrection
  exact Finset.sum_nonneg fun index _ =>
    mul_nonneg (hWeight index) (sq_nonneg _)

def sandwichCorrectedVariance
    (conditionedVariance correction : ℝ) : ℝ :=
  conditionedVariance + correction

theorem sandwich_correction_cannot_reduce_variance
    (conditionedVariance correction : ℝ)
    (hCorrection : 0 ≤ correction) :
    conditionedVariance ≤
      sandwichCorrectedVariance conditionedVariance correction := by
  unfold sandwichCorrectedVariance
  linarith

def sandwichCorrectedMean
    (conditionedMean _correction : ℝ) : ℝ :=
  conditionedMean

theorem sandwich_correction_preserves_conditioned_mean
    (conditionedMean correction : ℝ) :
    sandwichCorrectedMean conditionedMean correction = conditionedMean := by
  rfl

def robustEmpiricalBayesProjectedVariance
    (baseVariance priorScale sandwichCorrection : ℝ) : ℝ :=
  posteriorScaleVariance baseVariance priorScale + sandwichCorrection

theorem robust_empirical_bayes_two_layer_variance_nonshrinking
    (baseVariance priorScale sandwichCorrection : ℝ)
    (hVariance : 0 ≤ baseVariance)
    (hScale : 1 ≤ priorScale)
    (hCorrection : 0 ≤ sandwichCorrection) :
    baseVariance ≤ robustEmpiricalBayesProjectedVariance
      baseVariance priorScale sandwichCorrection := by
  unfold robustEmpiricalBayesProjectedVariance
  have hScaled := posterior_scale_at_least_one_cannot_reduce_variance
    baseVariance priorScale hVariance hScale
  linarith

def centralPredictiveVariance
    (baseVariance priorScale : ℝ) : ℝ :=
  posteriorScaleVariance baseVariance priorScale

def robustConfidenceVariance
    (baseVariance priorScale sandwichCorrection : ℝ) : ℝ :=
  robustEmpiricalBayesProjectedVariance
    baseVariance priorScale sandwichCorrection

theorem robust_confidence_variance_dominates_central_prediction
    (baseVariance priorScale sandwichCorrection : ℝ)
    (hCorrection : 0 ≤ sandwichCorrection) :
    centralPredictiveVariance baseVariance priorScale
      ≤ robustConfidenceVariance
        baseVariance priorScale sandwichCorrection := by
  unfold centralPredictiveVariance robustConfidenceVariance
  unfold robustEmpiricalBayesProjectedVariance
  linarith

def confidenceOnlyBayesRiskVariance
    (centralVariance _robustConfidenceVariance : ℝ) : ℝ :=
  centralVariance

theorem confidence_only_correction_does_not_change_bayes_ranking_variance
    (centralVariance robustVariance : ℝ) :
    confidenceOnlyBayesRiskVariance centralVariance robustVariance
      = centralVariance := by
  rfl

noncomputable def certifiedInitialIndices {n : ℕ}
    (upperMargin : Fin n → ℝ) : Finset (Fin n) :=
  Finset.univ.filter fun index => upperMargin index ≤ 0

theorem mem_certified_initial_indices_iff {n : ℕ}
    (upperMargin : Fin n → ℝ) (index : Fin n) :
    index ∈ certifiedInitialIndices upperMargin
      ↔ upperMargin index ≤ 0 := by
  simp [certifiedInitialIndices]

theorem certified_initializer_is_safe_when_certificate_nonempty {n : ℕ}
    (upperMargin : Fin n → ℝ)
    (chosen : Fin n)
    (hChosen : chosen ∈ certifiedInitialIndices upperMargin) :
    upperMargin chosen ≤ 0 := by
  exact (mem_certified_initial_indices_iff upperMargin chosen).mp hChosen

theorem empty_certificate_has_no_certified_initializer {n : ℕ}
    (upperMargin : Fin n → ℝ)
    (hEmpty : certifiedInitialIndices upperMargin = ∅) :
    ∀ chosen, ¬ upperMargin chosen ≤ 0 := by
  intro chosen hMargin
  have hMember : chosen ∈ certifiedInitialIndices upperMargin :=
    (mem_certified_initial_indices_iff upperMargin chosen).mpr hMargin
  rw [hEmpty] at hMember
  simp at hMember

def CertifiedOnlyInitializerContract {n : ℕ}
    (upperMargin : Fin n → ℝ) (chosen : Option (Fin n)) : Prop :=
  match chosen with
  | none => certifiedInitialIndices upperMargin = ∅
  | some index => index ∈ certifiedInitialIndices upperMargin

theorem certified_only_initializer_some_is_safe {n : ℕ}
    (upperMargin : Fin n → ℝ) (chosen : Fin n)
    (hContract :
      CertifiedOnlyInitializerContract upperMargin (some chosen)) :
    upperMargin chosen ≤ 0 := by
  apply certified_initializer_is_safe_when_certificate_nonempty upperMargin chosen
  simpa [CertifiedOnlyInitializerContract] using hContract

theorem certified_only_initializer_none_requires_empty {n : ℕ}
    (upperMargin : Fin n → ℝ)
    (hContract : CertifiedOnlyInitializerContract upperMargin none) :
    certifiedInitialIndices upperMargin = ∅ := by
  simpa [CertifiedOnlyInitializerContract] using hContract

theorem certified_only_initializer_cannot_fabricate_incumbent {n : ℕ}
    (upperMargin : Fin n → ℝ)
    (hEmpty : certifiedInitialIndices upperMargin = ∅) (chosen : Fin n) :
    ¬ CertifiedOnlyInitializerContract upperMargin (some chosen) := by
  intro hContract
  have hMember : chosen ∈ certifiedInitialIndices upperMargin := by
    simpa [CertifiedOnlyInitializerContract] using hContract
  rw [hEmpty] at hMember
  simp at hMember

noncomputable def posteriorCentralAleatoricMargin
    (mean z centralVariance threshold : ℝ) : ℝ :=
  mean + z * Real.sqrt centralVariance - threshold

noncomputable def certificationUpperAleatoricMargin
    (mean epistemicRadius z upperVariance threshold : ℝ) : ℝ :=
  mean + epistemicRadius + z * Real.sqrt upperVariance - threshold

theorem certification_upper_margin_independent_of_decision_variance
    (mean epistemicRadius z upperVariance threshold
      _centralVariance₁ _centralVariance₂ : ℝ) :
    certificationUpperAleatoricMargin
        mean epistemicRadius z upperVariance threshold
      = certificationUpperAleatoricMargin
        mean epistemicRadius z upperVariance threshold := by
  rfl

theorem central_decision_margin_cannot_relax_upper_certificate
    (mean epistemicRadius z centralVariance upperVariance threshold : ℝ)
    (hEpistemic : 0 ≤ epistemicRadius) (hZ : 0 ≤ z)
    (hVariance : centralVariance ≤ upperVariance) :
    posteriorCentralAleatoricMargin mean z centralVariance threshold
      ≤ certificationUpperAleatoricMargin
        mean epistemicRadius z upperVariance threshold := by
  have hSqrt : Real.sqrt centralVariance ≤ Real.sqrt upperVariance :=
    Real.sqrt_le_sqrt hVariance
  have hScaled :
      z * Real.sqrt centralVariance ≤ z * Real.sqrt upperVariance :=
    mul_le_mul_of_nonneg_left hSqrt hZ
  unfold posteriorCentralAleatoricMargin
  unfold certificationUpperAleatoricMargin
  linarith

def posteriorBinaryChanceRisk
    (objective penalty failureProbability : ℝ) : ℝ :=
  objective + penalty * failureProbability

theorem posterior_binary_chance_risk_monotone
    (objective penalty p q : ℝ) (hPenalty : 0 ≤ penalty) (hPQ : p ≤ q) :
    posteriorBinaryChanceRisk objective penalty p
      ≤ posteriorBinaryChanceRisk objective penalty q := by
  unfold posteriorBinaryChanceRisk
  linarith [mul_le_mul_of_nonneg_left hPQ hPenalty]

theorem posterior_binary_chance_risk_excess_nonnegative
    (objective penalty p : ℝ) (hPenalty : 0 ≤ penalty) (hP : 0 ≤ p) :
    objective ≤ posteriorBinaryChanceRisk objective penalty p := by
  unfold posteriorBinaryChanceRisk
  linarith [mul_nonneg hPenalty hP]

def posteriorNominalDecisionRisk
    (objective penalty nominalViolation : ℝ) : ℝ :=
  objective + penalty * nominalViolation

def klRobustDecisionRisk
    (objective penalty nominalViolation ambiguityPremium : ℝ) : ℝ :=
  objective + penalty * (nominalViolation + ambiguityPremium)

theorem kl_robust_decision_risk_decomposition
    (objective penalty nominalViolation ambiguityPremium : ℝ) :
    klRobustDecisionRisk
        objective penalty nominalViolation ambiguityPremium
      = posteriorNominalDecisionRisk objective penalty nominalViolation
        + penalty * ambiguityPremium := by
  unfold klRobustDecisionRisk posteriorNominalDecisionRisk
  ring

theorem posterior_nominal_decision_risk_le_kl_robust
    (objective penalty nominalViolation ambiguityPremium : ℝ)
    (hPenalty : 0 ≤ penalty) (hPremium : 0 ≤ ambiguityPremium) :
    posteriorNominalDecisionRisk objective penalty nominalViolation
      ≤ klRobustDecisionRisk
        objective penalty nominalViolation ambiguityPremium := by
  rw [kl_robust_decision_risk_decomposition]
  exact le_add_of_nonneg_right (mul_nonneg hPenalty hPremium)

theorem certification_upper_margin_independent_of_decision_ambiguity
    (mean epistemicRadius z upperVariance threshold
      _nominalAmbiguity _robustAmbiguity : ℝ) :
    certificationUpperAleatoricMargin
        mean epistemicRadius z upperVariance threshold
      = certificationUpperAleatoricMargin
        mean epistemicRadius z upperVariance threshold := by
  rfl

theorem finite_action_set_expansion_cannot_reduce_best_voi
    {Action : Type*} [DecidableEq Action]
    (small large : Finset Action) (voi : Action → ℝ)
    (bestSmall bestLarge : Action)
    (hBestSmallMem : bestSmall ∈ small)
    (hSubset : small ⊆ large)
    (hBestLarge : ∀ action ∈ large, voi action ≤ voi bestLarge) :
    voi bestSmall ≤ voi bestLarge := by
  exact hBestLarge bestSmall (hSubset hBestSmallMem)

end SCOLHKG.Real

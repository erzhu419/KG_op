import Mathlib
import SCOLHKG.Real.CertificationImplementation

namespace SCOLHKG.Real

/-!
The oracle-free implementation uses two frozen source-learned coordinates:

* `eta` parameterizes the conditional constraint mean and its epistemic error;
* `psi = (A,N)` parameterizes cumulative aleatoric risk.

They meet only in the certified chance margin.  This rules out the earlier
implicit assumption that one coordinate must be sufficient for both objects.
-/

structure SeparatedMeanRiskModel (X Eta Psi : Type*) where
  eta : X → Eta
  psi : X → Psi
  mean : Eta → ℝ
  epistemicVar : Eta → ℝ
  certificationVariance : Psi → ℝ

noncomputable def separatedCertificationMargin
    {X Eta Psi : Type*}
    (model : SeparatedMeanRiskModel X Eta Psi)
    (beta z tau : ℝ)
    (x : X) : ℝ :=
  theoryCertificationMargin
    (model.mean (model.eta x))
    beta
    (model.epistemicVar (model.eta x))
    z
    (model.certificationVariance (model.psi x))
    tau

theorem eta_equivalence_preserves_mean
    {X Eta Psi : Type*}
    (model : SeparatedMeanRiskModel X Eta Psi)
    {x y : X}
    (hEta : model.eta x = model.eta y) :
    model.mean (model.eta x) = model.mean (model.eta y) := by
  exact congrArg model.mean hEta

theorem eta_equivalence_preserves_epistemic_variance
    {X Eta Psi : Type*}
    (model : SeparatedMeanRiskModel X Eta Psi)
    {x y : X}
    (hEta : model.eta x = model.eta y) :
    model.epistemicVar (model.eta x)
      = model.epistemicVar (model.eta y) := by
  exact congrArg model.epistemicVar hEta

theorem psi_equivalence_preserves_certification_variance
    {X Eta Psi : Type*}
    (model : SeparatedMeanRiskModel X Eta Psi)
    {x y : X}
    (hPsi : model.psi x = model.psi y) :
    model.certificationVariance (model.psi x)
      = model.certificationVariance (model.psi y) := by
  exact congrArg model.certificationVariance hPsi

theorem joint_coordinate_equivalence_preserves_margin
    {X Eta Psi : Type*}
    (model : SeparatedMeanRiskModel X Eta Psi)
    (beta z tau : ℝ)
    {x y : X}
    (hEta : model.eta x = model.eta y)
    (hPsi : model.psi x = model.psi y) :
    separatedCertificationMargin model beta z tau x
      = separatedCertificationMargin model beta z tau y := by
  unfold separatedCertificationMargin
  rw [hEta, hPsi]

theorem separated_certificate_sound
    {X Eta Psi : Type*}
    (model : SeparatedMeanRiskModel X Eta Psi)
    {trueMean trueSigma : X → ℝ}
    {beta z tau : ℝ}
    {x : X}
    (hz : 0 ≤ z)
    (hMean :
      trueMean x ≤
        model.mean (model.eta x)
          + implementationEpistemicSlack beta
              (model.epistemicVar (model.eta x)))
    (hSigma :
      trueSigma x ≤ implementationCertSigma
        (model.certificationVariance (model.psi x)))
    (hMargin :
      separatedCertificationMargin model beta z tau x ≤ 0) :
    trueMean x + z * trueSigma x ≤ tau := by
  exact implementation_certifies_true_quantile
    hz hMean hSigma hMargin

end SCOLHKG.Real

import Mathlib.Probability.Distributions.Gaussian.Multivariate
import SCOLHKG.Measure.PosteriorCoefficientSampler

namespace SCOLHKG.Measure

open MeasureTheory ProbabilityTheory
open scoped RealInnerProductSpace Matrix

/-!
Multivariate-normal posterior coefficient sampler.

This file closes the distribution-level gap for
`core/candidates.py::posterior_sample_candidates`: sampled coefficient vectors
are represented by a random variable whose law is mathlib's
`multivariateGaussian mean covariance`.  The selector/envelope result remains
in `PosteriorCoefficientSampler.lean`; this file proves the actual Gaussian
law, mean, covariance, and linear-score Gaussianity facts used by that bridge.
-/

variable {Ω ι : Type*} [MeasurableSpace Ω] [Fintype ι] [DecidableEq ι]
variable {μΩ : Measure Ω}

abbrev CoeffSpace (ι : Type*) [Fintype ι] := EuclideanSpace ℝ ι

structure MultivariateGaussianCoefficientSampler
    (Ω ι : Type*)
    [MeasurableSpace Ω] [Fintype ι] [DecidableEq ι]
    (μΩ : Measure Ω) where
  mean : CoeffSpace ι
  covariance : Matrix ι ι ℝ
  covariance_psd : covariance.PosSemidef
  draw : Ω → CoeffSpace ι
  hasLaw :
    HasLaw draw (multivariateGaussian mean covariance) μΩ

theorem mvn_sampler_hasGaussianLaw
    (sampler :
      MultivariateGaussianCoefficientSampler Ω ι μΩ) :
    HasGaussianLaw sampler.draw μΩ := by
  exact HasLaw.hasGaussianLaw sampler.hasLaw

theorem mvn_sampler_map_eq_multivariateGaussian
    (sampler :
      MultivariateGaussianCoefficientSampler Ω ι μΩ) :
    μΩ.map sampler.draw =
      multivariateGaussian sampler.mean sampler.covariance := by
  exact sampler.hasLaw.map_eq

theorem mvn_sampler_integral_eq_mean
    (sampler :
      MultivariateGaussianCoefficientSampler Ω ι μΩ) :
    μΩ[sampler.draw] = sampler.mean := by
  rw [sampler.hasLaw.integral_eq]
  exact integral_id_multivariateGaussian

theorem mvn_sampler_covariance_eval
    (sampler :
      MultivariateGaussianCoefficientSampler Ω ι μΩ)
    (i j : ι) :
    cov[fun ω ↦ sampler.draw ω i,
        fun ω ↦ sampler.draw ω j; μΩ]
      = sampler.covariance i j := by
  change
    cov[(fun x : CoeffSpace ι ↦ x i) ∘ sampler.draw,
        (fun x : CoeffSpace ι ↦ x j) ∘ sampler.draw; μΩ]
      = sampler.covariance i j
  rw [sampler.hasLaw.covariance_comp (by fun_prop) (by fun_prop)]
  exact covariance_eval_multivariateGaussian sampler.covariance_psd i j

theorem mvn_sampler_linear_score_hasGaussianLaw
    (sampler :
      MultivariateGaussianCoefficientSampler Ω ι μΩ)
    (score : CoeffSpace ι →L[ℝ] ℝ) :
    HasGaussianLaw (fun ω ↦ score (sampler.draw ω)) μΩ := by
  exact (mvn_sampler_hasGaussianLaw sampler).map_fun score

def mvn_sampler_as_posterior_coefficient_sampler
    (sampler :
      MultivariateGaussianCoefficientSampler Ω ι μΩ) :
    PosteriorCoefficientSampler Ω (CoeffSpace ι) :=
  {
    draw := sampler.draw
  }

end SCOLHKG.Measure

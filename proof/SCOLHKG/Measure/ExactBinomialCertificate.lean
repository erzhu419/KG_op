import Mathlib.Probability.Distributions.Binomial

namespace SCOLHKG.Measure

open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal ProbabilityTheory unitInterval

variable {Omega : Type*} {mOmega : MeasurableSpace Omega}
  {mu : Measure Omega}

/-!
# Exact binomial terminal certificates

The external reliability experiment freezes a policy and evaluates independent
verification windows.  Its success count has a binomial law.  The registered
80-replication first-stage certificate can pass only after 80 successes, so
its false-certificate probability is exactly controlled by the all-success
binomial mass.  The generic ordered-shortlist result then follows by ordinary
finite error spending.
-/

theorem binomial_all_success_probability
    (n : Nat) (p : unitInterval) :
    Bin(n, p).real {n} = (p : Real) ^ n := by
  exact ProbabilityTheory.binomial_real_self n p

theorem binomial_all_success_probability_mono
    (n : Nat) (p q : unitInterval) (hpq : p <= q) :
    Bin(n, p).real {n} <= (q : Real) ^ n := by
  rw [binomial_all_success_probability]
  have hpNonnegative : 0 <= (p : Real) := p.2.1
  gcongr

theorem hasLaw_binomial_all_success_probability
    (successCount : Omega -> Nat) (n : Nat) (p : unitInterval)
    (hLaw : HasLaw successCount Bin(n, p) mu) :
    mu.real {omega | successCount omega = n} = (p : Real) ^ n := by
  have hCount := hLaw.measureReal_eq
    (p := fun count => count = n) (by measurability)
  simpa using hCount.trans (binomial_all_success_probability n p)

theorem unsafe_all_success_certificate_probability_le
    (successCount : Omega -> Nat) (n : Nat)
    (trueProbability requiredProbability : unitInterval)
    (hLaw : HasLaw successCount Bin(n, trueProbability) mu)
    (hUnsafe : trueProbability <= requiredProbability) :
    mu.real {omega | successCount omega = n} <=
      (requiredProbability : Real) ^ n := by
  rw [hasLaw_binomial_all_success_probability successCount n
    trueProbability hLaw]
  have hNonnegative : 0 <= (trueProbability : Real) :=
    trueProbability.2.1
  gcongr

theorem finite_all_success_certificates_familywise
    {Candidate : Type*} [DecidableEq Candidate]
    [IsFiniteMeasure mu]
    (shortlist : Finset Candidate)
    (successCount : Candidate -> Omega -> Nat)
    (replications : Candidate -> Nat)
    (trueProbability : Candidate -> unitInterval)
    (requiredProbability : unitInterval)
    (delta : Candidate -> Real)
    (familywiseDelta : Real)
    (hLaw : ∀ candidate ∈ shortlist,
      HasLaw (successCount candidate)
        Bin(replications candidate, trueProbability candidate) mu)
    (hUnsafe : ∀ candidate ∈ shortlist,
      trueProbability candidate <= requiredProbability)
    (hCandidateSpend : ∀ candidate ∈ shortlist,
      (requiredProbability : Real) ^ replications candidate
        <= delta candidate)
    (hFamilySpend :
      ∑ candidate ∈ shortlist, delta candidate <= familywiseDelta) :
    mu.real (⋃ candidate ∈ shortlist,
      {omega | successCount candidate omega = replications candidate})
      <= familywiseDelta := by
  calc
    mu.real (⋃ candidate ∈ shortlist,
        {omega | successCount candidate omega = replications candidate})
      <= ∑ candidate ∈ shortlist,
          mu.real {omega |
            successCount candidate omega = replications candidate} := by
        exact measureReal_biUnion_finset_le (μ := mu) shortlist
          (fun candidate =>
            {omega |
              successCount candidate omega = replications candidate})
    _ <= ∑ candidate ∈ shortlist, delta candidate := by
      exact Finset.sum_le_sum (fun candidate hCandidate =>
        (unsafe_all_success_certificate_probability_le
          (successCount candidate)
          (replications candidate)
          (trueProbability candidate)
          requiredProbability
          (hLaw candidate hCandidate)
          (hUnsafe candidate hCandidate)).trans
            (hCandidateSpend candidate hCandidate))
    _ <= familywiseDelta := hFamilySpend

theorem three_all_success_certificates_familywise
    [IsFiniteMeasure mu]
    (firstCount secondCount thirdCount : Omega -> Nat)
    (firstN secondN thirdN : Nat)
    (firstP secondP thirdP requiredP : unitInterval)
    (firstDelta secondDelta thirdDelta familywiseDelta : Real)
    (hFirstLaw : HasLaw firstCount Bin(firstN, firstP) mu)
    (hSecondLaw : HasLaw secondCount Bin(secondN, secondP) mu)
    (hThirdLaw : HasLaw thirdCount Bin(thirdN, thirdP) mu)
    (hFirstUnsafe : firstP <= requiredP)
    (hSecondUnsafe : secondP <= requiredP)
    (hThirdUnsafe : thirdP <= requiredP)
    (hFirstSpend : (requiredP : Real) ^ firstN <= firstDelta)
    (hSecondSpend : (requiredP : Real) ^ secondN <= secondDelta)
    (hThirdSpend : (requiredP : Real) ^ thirdN <= thirdDelta)
    (hFamilySpend :
      firstDelta + secondDelta + thirdDelta <= familywiseDelta) :
    mu.real (
        {omega | firstCount omega = firstN} ∪
        ({omega | secondCount omega = secondN} ∪
          {omega | thirdCount omega = thirdN}))
      <= familywiseDelta := by
  have hFirst :
      mu.real {omega | firstCount omega = firstN} <= firstDelta :=
    (unsafe_all_success_certificate_probability_le
      firstCount firstN firstP requiredP hFirstLaw hFirstUnsafe).trans
        hFirstSpend
  have hSecond :
      mu.real {omega | secondCount omega = secondN} <= secondDelta :=
    (unsafe_all_success_certificate_probability_le
      secondCount secondN secondP requiredP hSecondLaw hSecondUnsafe).trans
        hSecondSpend
  have hThird :
      mu.real {omega | thirdCount omega = thirdN} <= thirdDelta :=
    (unsafe_all_success_certificate_probability_le
      thirdCount thirdN thirdP requiredP hThirdLaw hThirdUnsafe).trans
        hThirdSpend
  calc
    mu.real (
        {omega | firstCount omega = firstN} ∪
        ({omega | secondCount omega = secondN} ∪
          {omega | thirdCount omega = thirdN}))
      <= mu.real {omega | firstCount omega = firstN}
          + mu.real (
            {omega | secondCount omega = secondN} ∪
              {omega | thirdCount omega = thirdN}) :=
        measureReal_union_le _ _
    _ <= mu.real {omega | firstCount omega = firstN}
          + (mu.real {omega | secondCount omega = secondN}
            + mu.real {omega | thirdCount omega = thirdN}) := by
        gcongr
        exact measureReal_union_le _ _
    _ <= firstDelta + secondDelta + thirdDelta := by linarith
    _ <= familywiseDelta := hFamilySpend

end SCOLHKG.Measure

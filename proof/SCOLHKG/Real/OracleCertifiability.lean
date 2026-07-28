import Mathlib

namespace SCOLHKG.Real

/-!
Noise-limited oracle certifiability.

The oracle knows the true observation scale and only estimates the constraint
mean from independent repetitions.  Its radius is therefore an optimistic
lower bound on the uncertainty faced by any implementable optimizer.
-/

noncomputable def oracleMeanRadius
    (quantile sigma replicates : ℝ) : ℝ :=
  quantile * sigma / Real.sqrt replicates

noncomputable def oracleMarginUpper
    (margin quantile sigma replicates : ℝ) : ℝ :=
  margin + oracleMeanRadius quantile sigma replicates

theorem oracleMeanRadius_nonnegative
    {quantile sigma replicates : ℝ}
    (hQuantile : 0 ≤ quantile)
    (hSigma : 0 ≤ sigma) :
    0 ≤ oracleMeanRadius quantile sigma replicates := by
  unfold oracleMeanRadius
  exact div_nonneg
    (mul_nonneg hQuantile hSigma)
    (Real.sqrt_nonneg replicates)

theorem oracleMeanRadius_antitone_in_replicates
    {quantile sigma firstReplicates secondReplicates : ℝ}
    (hQuantile : 0 ≤ quantile)
    (hSigma : 0 ≤ sigma)
    (hFirst : 0 < firstReplicates)
    (hOrder : firstReplicates ≤ secondReplicates) :
    oracleMeanRadius quantile sigma secondReplicates ≤
      oracleMeanRadius quantile sigma firstReplicates := by
  unfold oracleMeanRadius
  have hNumerator : 0 ≤ quantile * sigma :=
    mul_nonneg hQuantile hSigma
  have hSqrtFirst : 0 < Real.sqrt firstReplicates :=
    Real.sqrt_pos.2 hFirst
  have hSqrtOrder :
      Real.sqrt firstReplicates ≤ Real.sqrt secondReplicates :=
    Real.sqrt_le_sqrt hOrder
  exact div_le_div_of_nonneg_left
    hNumerator hSqrtFirst hSqrtOrder

theorem oracleMarginUpper_antitone_in_replicates
    {margin quantile sigma firstReplicates secondReplicates : ℝ}
    (hQuantile : 0 ≤ quantile)
    (hSigma : 0 ≤ sigma)
    (hFirst : 0 < firstReplicates)
    (hOrder : firstReplicates ≤ secondReplicates) :
    oracleMarginUpper margin quantile sigma secondReplicates ≤
      oracleMarginUpper margin quantile sigma firstReplicates := by
  unfold oracleMarginUpper
  simpa [add_comm] using
    (add_le_add_left
      (oracleMeanRadius_antitone_in_replicates
        hQuantile hSigma hFirst hOrder)
      margin)

theorem oracle_certified_of_product_budget
    {margin quantile sigma replicates : ℝ}
    (hReplicates : 0 < replicates)
    (hBudget :
      quantile * sigma ≤ -margin * Real.sqrt replicates) :
    oracleMarginUpper margin quantile sigma replicates ≤ 0 := by
  unfold oracleMarginUpper oracleMeanRadius
  have hRadius : quantile * sigma / Real.sqrt replicates ≤ -margin := by
    exact (div_le_iff₀ (Real.sqrt_pos.2 hReplicates)).2 hBudget
  linarith

theorem oracle_certified_of_squared_budget
    {margin quantile sigma replicates : ℝ}
    (hMargin : margin < 0)
    (hQuantile : 0 ≤ quantile)
    (hSigma : 0 ≤ sigma)
    (hReplicates : 0 < replicates)
    (hBudget :
      (quantile * sigma) ^ 2 ≤ (-margin) ^ 2 * replicates) :
    oracleMarginUpper margin quantile sigma replicates ≤ 0 := by
  have hSqrtSquare :
      (Real.sqrt replicates) ^ 2 = replicates :=
    Real.sq_sqrt (le_of_lt hReplicates)
  have hLeft : 0 ≤ quantile * sigma :=
    mul_nonneg hQuantile hSigma
  have hRight : 0 ≤ -margin * Real.sqrt replicates :=
    mul_nonneg (by linarith) (Real.sqrt_nonneg replicates)
  have hSquare :
      (quantile * sigma) ^ 2 ≤
        (-margin * Real.sqrt replicates) ^ 2 := by
    nlinarith
  have hProduct :
      quantile * sigma ≤ -margin * Real.sqrt replicates := by
    nlinarith
  exact oracle_certified_of_product_budget hReplicates hProduct

theorem oracle_certificate_persists_with_more_replicates
    {margin quantile sigma firstReplicates secondReplicates : ℝ}
    (hQuantile : 0 ≤ quantile)
    (hSigma : 0 ≤ sigma)
    (hFirst : 0 < firstReplicates)
    (hOrder : firstReplicates ≤ secondReplicates)
    (hCertified :
      oracleMarginUpper margin quantile sigma firstReplicates ≤ 0) :
    oracleMarginUpper margin quantile sigma secondReplicates ≤ 0 := by
  exact (oracleMarginUpper_antitone_in_replicates
    hQuantile hSigma hFirst hOrder).trans hCertified

end SCOLHKG.Real

namespace SCOLHKG

/-!
Algebraic core of the information-refinement proposition.

The probabilistic statement is the law of total variance.  In Lean core we
formalize the deterministic consequence used by the paper: if a coarse
variance decomposes as refined variance plus an explained nonnegative term,
then the refined apparent variance cannot exceed the coarse one.
-/

structure VarianceRefinement where
  coarse : Nat
  refined : Nat
  explained : Nat
deriving Repr, BEq

namespace VarianceRefinement

def Valid (r : VarianceRefinement) : Prop :=
  r.coarse = r.refined + r.explained

theorem refined_le_coarse (r : VarianceRefinement) (h : r.Valid) :
    r.refined ≤ r.coarse := by
  rw [h]
  exact Nat.le_add_right _ _

theorem explained_le_coarse (r : VarianceRefinement) (h : r.Valid) :
    r.explained ≤ r.coarse := by
  rw [h]
  exact Nat.le_add_left _ _

theorem no_explained_information_preserves_variance
    (r : VarianceRefinement) (h : r.Valid) (hzero : r.explained = 0) :
    r.coarse = r.refined := by
  rw [h, hzero, Nat.add_zero]

end VarianceRefinement

end SCOLHKG


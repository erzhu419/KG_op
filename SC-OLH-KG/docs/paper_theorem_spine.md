# Revised Paper Theorem Spine

## One object, not a module sum

The revised method is described by one source-to-target decision model:

```text
source-only archive
  -> frozen transferable proposal q0 and low-frequency coordinate
  -> separated observable heads eta(x) and psi(x)=(A(x),N(x))
  -> joint mean/cumulative-HVD target posterior
  -> unit-cost evaluate-or-replicate posterior VOI
  -> observed-action posterior Bayes decision
  -> separate conservative chance certificate.
```

No weighted `KG_obj + KG_feas + KG_var + KG_coupling` expression is a main
paper object. The legacy additive score remains an ablation only.

## Supported structural claim

The completed causal experiments reject the claim that low frequency,
orthogonality, sparsity, and additivity are four coequal empirical causes. The
supported front-end claim is narrower and stronger:

1. a universal low-frequency policy-support library is fixed before seeing the
   held-out target;
2. ordinary source observations rank policies in a dimension-equivariant
   low-frequency risk/objective coordinate;
3. the resulting target `n0` proposal is frozen and target-label invariant.

Orthogonality, adaptive sparsity, and additive groups define the larger
hypothesis class and remain preregistered causal ablations. They become main
claims only if their independently retrained rows show an incremental effect.

## Assumptions

- **A1 Source noninterference.** Source records and proposal hyperparameters
  are frozen before held-out target observations; target truth is never used.
- **A2 Observable sufficiency.** Constraint mean and epistemic error factor
  through `eta(x)`; cumulative aleatoric risk factors through
  `psi(x)=(A(x),N(x))`. This is a falsifiable held-out-domain assumption.
- **A3 Cumulative risk law.** Conditional trajectory variance is
  `floor + A^T Lambda A + N^T B N + N^T omega`, with the shared-shock block
  projected to PSD and a conservative residual upper radius.
- **A4 Coverage event.** The GP mean confidence event and HVD variance-upper
  event hold at the reported level. Source-task slack and numeric constants are
  fixed on source-only holdouts.
- **A5 Finite decision audit.** The posterior-only shortlist has exact-VOI
  coverage error at most `epsilon_shortlist`; uniform MC error is at most
  `eta_MC` on that shortlist.

## Main results

1. **Source invariance.** A source-only proposal is unchanged under every
   change of held-out target labels.
2. **Coordinate quotient.** Policies sharing both `eta` and `psi` have the
   same modeled mean, epistemic variance, cumulative certification variance,
   and chance margin. Complexity is therefore tied to the observable
   coordinate rather than nominal policy dimension under A2.
3. **Cumulative HVD decomposition.** The total risk separates exactly into
   floor, independent exposure, PSD shared shock, and linear residual terms;
   low-rank truncation error is the nonnegative omitted risk tail.
4. **Certificate soundness.** On A4,
   `m_g + sqrt(beta_g)s_g + z sqrt(v_C_plus) <= tau` implies true chance
   feasibility. Bayes-ranking variance cannot relax the certificate.
5. **Approximate one-step Bayes optimality.** The selected evaluate-or-
   replicate action is within `epsilon_shortlist + 2 eta_MC` of every action
   in the declared finite audit pool.
6. **Finite-budget accounting.** One-step posterior value reductions telescope
   over the charged target budget, with the finite-action errors entering
   additively.
7. **Safe regret.** A certified terminal action with objective error `epsilon`
   is truly feasible and has safe simple regret at most `epsilon` on the joint
   confidence event.

The umbrella Lean theorem is
`SCOLHKG.Real.paper_mainline_finite_closure` in
`proof/SCOLHKG/Real/PaperMainline.lean`. It composes source noninterference,
cumulative HVD, coordinate quotienting, finite-action VOI, and terminal
certification. The supporting probability, estimation, information-gain, and
safe-regret results remain in their specialized files.

## Empirical obligations

The theorem package is conditionally complete, not an unconditional statement
about every simulator. The paper must still measure:

- source-only calibration of the PAC-Bayes/exponential-moment slack;
- held-out sufficiency of `eta` and `psi`;
- certificate coverage as well as false certification;
- `epsilon_shortlist` and `eta_MC` sensitivity;
- HVD value under controlled shared-shock and replication sweeps;
- fresh-seed SUMO out-of-sample certification.

Failure of any obligation narrows the empirical claim; it does not license a
new heuristic gate. This is the line that keeps the revised paper from becoming
an engineering stack.

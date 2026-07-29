# SC-OLH-KG Lean4 Proof Workspace

This directory is the local proof workspace for the current `KG_op` project.
It is intentionally kept at the project root instead of being moved into any
external proof directory.

Lean4 is the source of truth for formal proofs.  Markdown files are only the
human-readable roadmap and code-to-theory map.

## Build

```bash
cd proof
lake build
```

The project now depends on mathlib `v4.31.0`.  The first build may download
mathlib and its cache; later builds should be fast.

## Core Files

- `SCOLHKG/Variance.lean`: formal cumulative-risk algebra, including the
  `A^T Lambda A + N^T B N + N^T omega + floor` decomposition and truncation
  bookkeeping lemmas.
- `SCOLHKG/Information.lean`: formal algebraic core of the information
  refinement proposition.
- `SCOLHKG/Certification.lean`: formal ordered-arithmetic core of conservative
  chance certification.
- `SCOLHKG/Scalarization.lean`: inherited bi-objective-to-scalar bridge:
  weak Pareto dominance is preserved by nonnegative weighted scalarization.
- `SCOLHKG/HVD.lean`: deterministic oracle-inequality and conservative
  variance-estimation skeleton for HVD.
- `SCOLHKG/Optimization.lean`: safe recommendation and simple-regret
  consequences once mean confidence and variance upper bounds hold.
- `SCOLHKG/KG.lean`: exact KG maximizer bookkeeping and the condition under
  which an additive KG proxy is exact.
- `SCOLHKG/Real/CumulativeRisk.lean`: real-valued cumulative-risk algebra.
- `SCOLHKG/Real/ConditionalVariance.lean`: finite-partition law of total
  variance over real numbers, proved by algebraic expansion.
- `SCOLHKG/Real/GPRUpdate.lean`: code-level rank-one GPR update bridge, showing
  that PSD-repaired quadratic variance yields a strictly positive noisy
  Kalman denominator, and
  the KG slope used in `core/kg.py` is the standard-shock posterior mean
  response.
- `SCOLHKG/Real/OccupancyDecomposition.lean`: policy trajectory mixture risk
  decomposition into occupancy cumulative risk, occupancy remainder, and
  between-trajectory explained variance.
- `SCOLHKG/Real/Certification.lean`: real-valued GP-confidence plus variance
  upper-bound certification implication.
- `SCOLHKG/Real/CertificationImplementation.lean`: code-level bridge proving
  the implemented `mu + sqrt(beta)s + z sqrt(v_C^+) <= tau` margin is the
  Lean certification predicate and is more conservative than legacy mode. It
  also proves the nonvacuity necessity that epistemic radius cannot exceed
  true safety depth, equivalently
  `beta * epistemicVariance <= safetyDepth^2` under oracle mean.
- `SCOLHKG/Real/FiniteSampleHVD.lean`: identifies the active source-shape
  calibration law under exposure excitation, derives a finite-sample ridge
  parameter bound from replicated variance targets, and propagates active
  parameter and coordinate misspecification errors to cumulative variance.
- `SCOLHKG/Real/CertificateNonvacuity.lean`: gives an explicit finite mean and
  replication budget under positive true safety depth that makes the
  implementation certificate nonempty.
- `SCOLHKG/Real/EndToEndSafeRegret.lean`: truth-relative finite-pool safe regret
  with separate representation, HVD, transfer, pool, shortlist, MC, and
  sequential terms.
- `SCOLHKG/Real/SafeguardedPolicyImprovement.lean`: V52 posterior-value
  noninferiority for action-superset and rollout challengers that retain V51
  as a fallback and switch only beyond a `2 * eta` uniform-error guard.
- `SCOLHKG/Real/ConstrainedCertificateDeficit.lean`: V53 keeps the V51
  Bayes-risk fallback, removes rollout, and proves joint risk/certificate
  noninferiority when separate estimated score gaps exceed their calibrated
  `2 * eta` guards. V53-v2 additionally proves that positive current-terminal
  normalization preserves score order, uniform-error control, and both guard
  decisions exactly. V53-v3 proves that clipping every normalized fantasy gain
  before integration gives a score in `[-1,1]` with pairwise range at most two;
  the literal V51 fallback and terminal recommendation functional are unchanged.
  V54 replaces the global worst-action radius with a nested-common-random-number
  radius for each challenger/fallback difference and proves guarded joint
  improvement conditional on that action-specific radius covering exact error.
- `SCOLHKG/Real/GuardDecompositionPolicy.lean`: V58 exactly decomposes the
  robust chance margin into mean, epistemic, joint-epistemic, cumulative
  aleatoric, and favorable-coupling terms; proves that the dynamic action
  support retains every V51 action; and proves joint posterior Bayes-risk and
  certificate-deficit noninferiority for fallback-or-confirmed selection.
- `SCOLHKG/Measure/StatisticalClosure.lean`: combines the component bad events
  into a high-probability end-to-end safe-regret statement.
- `SCOLHKG/Measure/FiniteSampleHVDConcentration.lean`: converts a declared
  target replication schedule and sub-Gaussian variance-target errors into the
  uniform active-HVD event used by the finite-sample oracle inequality.
- `SCOLHKG/Real/TransferGeneralization.lean`: source-task PAC-Bayes target-risk
  bound with an explicit held-out-domain discrepancy term.
- `SCOLHKG/Real/ProposalCoverage.lean`: composes the source-to-target
  PAC-Bayes miss-risk bound and effective structural dimension with two
  distinct proposal contracts.  The deployed deterministic finite atlas uses
  positive feasible mass to prove existence of a feasible support member;
  a separate randomized backend may use the explicit
  `1-(1-p_lower)^n0` IID hit bound.
- `SCOLHKG/Real/ProposalNoFreeLunch.lean`: proves that every proper finite
  target-label-free atlas misses some nonempty target feasible set. A
  source-to-target structural/discrepancy assumption is therefore necessary,
  not an artifact of the proof technique.
- `SCOLHKG/Real/RankAlignedAtlasCoverage.lean`: the headline deterministic
  rank-transfer candidate. Uniform normalized risk-rank alignment, one-sided
  source-rank atlas coverage, and target safe-rank interior depth imply a
  feasible atlas member. Its source-only finite-sample audit was vacuous, so
  it is not the headline empirical bridge.
- `SCOLHKG/Real/GeometricAtlasCoverage.lean`: the deployed maximin-atlas
  contract. Source-support covering radius plus source/target support shift
  must fit inside either a complete safe coordinate ball or a Lipschitz
  chance-margin depth. The fully decomposed version adds twice the learned
  coordinate approximation error to the ideal-coordinate domain shift.
  Nominal policy dimension is absent.
- `SCOLHKG/Measure/ProposalCoverage.lean`: proves the exact finite-product
  probability of missing a measurable feasible set in all IID proposal draws.
- `SCOLHKG/Real/MeanRiskCoordinateSeparation.lean`: formal separation of the
  source-learned constraint-mean coordinate `eta` from cumulative-risk
  `psi=(A,N)`, plus joint-margin invariance and inherited certificate
  soundness.
- `SCOLHKG/Real/BoundaryCoordinateSufficiency.lean`: exact post-run
  mean/variance oracle-substitution identities, certificate soundness for
  independent mean and cumulative-variance heads derived from a common
  observable state/trajectory exposure, factorization of both heads through
  that exposure, invariance of the complete separated margin under equal
  observable exposures, head noninterference, candidate-support necessity, and
  the V4 source-residual split into contractible latent coefficient covariance
  plus a finite floor, including exact reference-variance preservation and
  monotonic contraction of epistemic variance. V5 adds channel-permutation
  equivariance for learned canonical role assignments and proves that its
  conservative source-mean scale plus PSD directional covariance can only
  increase coefficient and residual-floor uncertainty. V6 adds the online
  sufficient-statistic update, unit lower bound for every recomputed scale,
  conservative full-history refit relative to a frozen source law, and exact
  invariance of the target-null law. V7 adds a source-only semantic-support
  trust in `(0,1]`, proves that it cannot increase transferred source mass,
  and proves that the PSD source-contrast random effect can only increase
  epistemic variance. V8 adds the finite source-support selector: choosing
  between independently fitted role and fallback coordinates preserves a
  certification property whenever both branch-specific bridges are sound.
  V12 proves strict boundedness of the source-tanh latent coordinate and the
  nonnegative normalization that preserves average predictive variance when
  `target:null` uses an unlabeled target-feature geometry. V13 proves that
  source-support clipping is exactly the identity on support and bounded off
  support, while its optional discrepancy channel is zero on support,
  nonnegative, and strictly below one. V16 proves nonnegative partial-role
  mass and that source/target matching loss, transport entropy, and cardinality
  gap define an epistemic covariance scale at least one which cannot decrease
  predictive variance. V17 replaces pairwise role matching by a barycentric
  intervention-response map. Lean proves that nonnegative unit-mass role
  weights keep every scalar response inside the source-role convex hull. It
  also proves that hierarchical misspecification scaling cannot reduce a
  source mean variance while the target-null variance remains exactly
  unchanged. V18 adds an outcome-free target residual coordinate orthogonal to
  the frozen source mean span. Lean proves exact energy decomposition under
  orthogonality, exact preservation of the source prior mean under a zero-mean
  residual law, and that an independent PSD residual covariance cannot reduce
  predictive variance. V19 makes residual rank a finite nested Bayesian
  structure variable. Lean proves nonnegative active/inactive coefficient
  variance, normalized nonnegative target-evidence weights, and that
  between-rank disagreement cannot reduce predictive variance. V20 instead
  treats channel-to-role assignment as a finite Bayesian structure variable.
  Lean proves that its uniform finite prior has unit mass, relabeling the
  assignment atoms leaves the mixture mean invariant, between-assignment
  disagreement cannot reduce predictive variance, and ordinary target-
  likelihood updates remain normalized nonnegative probabilities. V21 replaces
  in-sample structure evidence by exact leave-one-out predictive scores. Lean
  proves that exponentiated finite cross-fitted scores are strictly positive,
  that any nontrivial nonnegative structure prior has positive evidence mass,
  and that the resulting generalized-Bayes assignment posterior is normalized
  and nonnegative. Relabeling commutes exactly with the score likelihood.
  V22 uses the source role atlas and an unlabeled target exposure distribution
  to define assignment costs. Lean proves positivity of the resulting geometry
  likelihood, that lower matching cost receives no less prior likelihood at a
  positive temperature, and normalization/nonnegativity of the finite geometry
  prior. V23 factorizes that frozen assignment marginal from a target-updated
  conditional source/null expert law. Lean proves nonnegativity and unit mass
  of the joint law, exact recovery of each frozen assignment marginal, and
  invariance of that marginal to any normalized conditional expert update.
  V24 applies the same result to full-history hierarchical misspecification
  refits: scale learning can change expert conditionals but not role semantics.
  V25 uses the charged pilot only through Fisher-transformed channel/chance-
  margin associations. Lean proves positivity and normalization of the finite
  boundary-role posterior and exact invariance of its likelihood under a
  simultaneous channel/assignment relabeling. It also proves that an
  assignment-conditional source-contrast covariance has zero predictive gain
  outside its active block. V26 removes transferred discrete role identities.
  It copies one exchangeable source coefficient-block law to every observable
  target channel, then lets only charged target observations differentiate
  those channel blocks. Lean proves exact scalar-mean equivariance under a
  simultaneous channel relabeling and exact invariance of the exchangeable
  source prior score. The Python bridge separately audits that the frozen
  source blocks are identical, charged target updating differentiates them,
  source-mean misspecification cannot contract uncertainty, and the cumulative
  HVD state remains an independent variance head. V27 then marginalizes
  source-domain identity into one exchangeable empirical-Bayes Gaussian
  hyperlaw. Lean proves nonnegative within-plus-between projected covariance,
  exact identity of a one-atom aggregate projection, and exact one-observation
  growth of the charged target history. Runtime contracts check that every
  refit starts from the frozen hyperlaw, uses all target observations, retains
  no source selector or target-null atom, and leaves HVD independent.
  V28 then makes head authority explicit: the aggregate target GPR alone owns
  constraint mean and epistemic variance, while exactly one task-robust or
  direct cumulative HVD owns aleatoric variance. Lean proves that changing the
  discarded legacy task mean/epistemic heads cannot change either the theory
  certificate or posterior Bayes-margin mean, and proves exact reduction of
  both HVD authority choices. Python audits the same routing in recommendation,
  truth-pool certification, and cloned terminal posterior updates.
  V29 then maintains one posterior Bayes-risk incumbent and permits a switch
  only under a covariance-free Cantelli dominance bound. V30--V31 calibrate
  source-mean misspecification uncertainty with a scale clipped below at one
  and a local HC3 sandwich covariance. Lean proves that both operations are
  noncontractive in every scalar projection and that sandwich correction
  preserves the conditioned mean. V32 proves only the valid nonempty-set
  statement: an initializer selected from the canonical certified set is
  certified; when that set is empty, no certified initializer exists. V34
  composes source-hyperlaw scaling and HC3 correction in one robust
  empirical-Bayes posterior, whose projected variance cannot be smaller than
  its uncalibrated base variance.
  It does not claim that the learned scale
  is temporally monotone, nor does it assume that
  alignment succeeds on every held-out domain; that premise is tested by the
  paired empirical gate. The Python bridge uses independently fitted
  `SourceAlignedBoundaryCoordinate` and
  `SourceAlignedVarianceRiskCoordinate`; the latter is supervised only by
  ordinary replicated source variance.
- `SCOLHKG/Real/SharedLowRankSourceHyperlaw.lean`: V42 separates source-fit
  uncertainty in the weighted shared coefficient from genuine target-domain
  discrepancy. Lean proves nonnegative shared-estimation, between-domain, and
  factor-projected variance; proves the weighted shared-estimation variance is
  no larger than the legacy transfer when source weights lie in `[0,1]`; and
  encodes the domain factor with at most `S-1` coordinates, including exact
  zero discrepancy for one source. V43 additionally proves that the
  finite-source predictive multiplier `(1+c)/(1-c)` is nonnegative and at
  least one for `0 <= c < 1`, that its projected discrepancy is nonnegative
  and no smaller than the population block, and that scaling keeps the same
  `Fin (S-1)` factorization and one-source zero. Python additionally audits
  the exact multiplier, rank, permutation, and oracle-exclusion contracts.
  V45 proves that splitting a fixed per-base-domain source budget equally
  over exchangeable task episodes preserves the exact simulator-call budget,
  that the maximum `S-1` discrepancy-factor capacity is monotone in episode
  count, and that the resulting finite factor projection remains
  nonnegative. These are capacity and accounting theorems, not a claim that
  the observed episode coefficient matrix attains the enlarged rank. V46
  groups episodes by their frozen base-domain label. Lean proves nonnegativity
  of the centered within-base projected variance, exact invariance of that
  contrast under a common base-domain offset, exact separation of shared
  estimation from role/between/within variation, and the grouped
  `(B-1)+B(E-1)` capacity (seven for two bases and four episodes). V47 models
  the scalar projection of PSD random-effects deconvolution as
  `max(observed-fitNoise,0)`. Lean proves that the corrected variance is
  nonnegative, cannot exceed a nonnegative observed variance when fit noise is
  nonnegative, exactly recovers a nonnegative latent variance from
  `observed=latent+fitNoise`, and removes a noise-dominated direction.
- `SCOLHKG/Real/SourceConstraintMeanPosterior.lean`: hierarchical source
  coefficient uncertainty in `eta`, exact source-to-target conjugate mean
  conditioning, target-information variance contraction, nonnegative finite
  mixture moment projection with retained source disagreement, normalized
  sequential predictive-likelihood reweighting, and inherited chance-
  certificate soundness.
- `SCOLHKG/Real/HVD.lean`: real-valued residual-square concentration event to
  HVD oracle-inequality implication.
- `SCOLHKG/Real/HVDImplementation.lean`: code-level HVD bridges for residual
  square records, nonnegative within-policy sample variance, nonnegative
  cumulative beta predictions, clipping, and certification variance. It also
  proves replication-degree monotonicity and feasibility/objective descent for
  every accepted projected-IRLS iteration, plus V15 mean/HVD task-posterior
  noninterference and replication-only variance-score independence.
- `SCOLHKG/Real/CumulativeRiskImplementation.lean`: factor-HVD feature-block
  bridge for `floor/independent/shared/linear/total`, the provider-based
  `psi=(A,N)` bridge, `v_C^+ = total + tail_guard`, and the shared-shock
  omission underestimation lemma.
- `SCOLHKG/Real/PosteriorRecommendation.lean`: robust posterior recommendation
  logic used in `SingleOLHKGAlgorithm._solve_posterior_recommendation`.
- `SCOLHKG/Real/RidgeHVD.lean`: concrete ridge empirical minimizer plus
  residual-square uniform concentration to HVD oracle inequality, including
  nonnegative prior-centered penalties, hierarchical source/target scale, and
  the optimization-slack oracle inequality used by a finite projected solver.
- `SCOLHKG/Real/SourceShapeMixtureHVD.lean`: nonnegative source-domain risk
  mixtures, shared mean/HVD latent-task weighting, a nonnegative target-pooled
  null shape, and monotone contraction of the posterior shape radius under
  added target Fisher information. It also proves that a prediction frozen
  before an ordinary target observation has squared-innovation expectation
  equal to true noise second moment plus squared mean bias, which is the
  conservative-moment bridge for optional `prequential_upper` evidence. The
  same file proves nonnegativity of square-root radius reduction, used to put
  GPR and HVD information gains in one chance-margin response unit.
- `SCOLHKG/Real/JointKLChanceCertificate.lean`: one shared task law for the
  complete chance margin, the square-root tangent envelope, centered-second-
  moment domination of mixture epistemic variance, and validity of minimizing
  over a finite grid of robust tangent bounds.
- `SCOLHKG/Real/KG.lean`: real-valued exact KG and additive-proxy relation.
- `SCOLHKG/Real/LineEnvelopeKG.lean`: certificate-level line-envelope KG
  theorem for the `compute_h` calculation once active hull intervals are
  certified.
- `SCOLHKG/Real/LineEnvelopeStack.lean`: endpoint and tail-slope bridge showing
  that the Python `validate_h_certificate` checks imply active-line
  certificates.
- `SCOLHKG/Real/LineEnvelopeAlgorithm.lean`: step-level stack-loop
  formalization for the Python `cuts.pop()` and `cuts.append(z)` mutations.
- `SCOLHKG/Real/LineEnvelopeGlobal.lean`: final global stack dominance
  invariant implies atom certificates and exact line-envelope KG, without a
  Python runtime-validator assumption.
- `SCOLHKG/Real/LineEnvelopeIntersection.lean`: concrete `compute_h`
  intersection arithmetic, popped-cell takeover, and right-tail split
  certificates for the stack loop.
- `SCOLHKG/Real/LineEnvelopeFold.lean`: full sorted-line recursive fold for
  the `compute_h` active stack; popped lines are proved pointwise dominated by
  the final output stack, and output endpoint dominance lifts to the original
  input `FinalEnvelopeStackInvariant`.
- `SCOLHKG/Real/AdditiveApproxKG.lean`: uniform additive-acquisition
  approximation implies a `2 eta` exact-KG optimality gap.
- `SCOLHKG/Real/ExactKGImplementation.lean`: deterministic bridge from a
  uniformly accurate exact-MC estimator to the same `2 eta` exact-KG gap.
- `SCOLHKG/Real/InformationGainRegret.lean`: information-gain radius and
  finite-budget regret accounting.
- `SCOLHKG/Real/FiniteKernelInformationGain.lean`: finite-kernel scalar
  information-gain accumulation and uniform-cap bound.
- `SCOLHKG/Real/KernelDeterminantBridge.lean`: determinant-ratio cap bridge
  from finite product-ratio information gain into safe-regret accounting.
- `SCOLHKG/Real/FeatureKernelDeterminantCap.lean`: concrete finite
  feature/kernel ratio caps that imply determinant/log-product information-gain
  bounds, including feature-norm/coefficient-variance/noise-floor caps.
- `SCOLHKG/Real/SafeRegret.lean`: real-valued finite-budget safe simple-regret
  implication.
- `SCOLHKG/Measure/ProbabilityEvents.lean`: mathlib
  `ProbabilityTheory` layer connecting conditional variance, Chebyshev, and
  finite union bounds to GP/residual concentration events.
- `SCOLHKG/Measure/GPKernelConfidence.lean`: finite-kernel posterior error as
  a weighted sum of independent sub-Gaussian noise, with explicit
  `sum_i w_i^2 c_i` parameter and finite/adaptive confidence.
- `SCOLHKG/Measure/ExactMCConcentration.lean`: finite candidate-pool
  concentration for exact-MC KG estimator errors, feeding the deterministic
  exact-KG gap bridge.
- `SCOLHKG/Measure/SubGaussianConfidence.lean`: sub-Gaussian one-sided and
  centered confidence events over finite and adaptive candidate sets.
- `SCOLHKG/Measure/ResidualSquareConcentration.lean`: bounded residual-square
  distribution constants and finite concentration events for HVD.
- `SCOLHKG/Measure/ResidualSquareTail.lean`: generic sub-exponential or sharper
  residual-square tail interface, closed-form default radius inversion, and
  finite HVD concentration.
- `SCOLHKG/Measure/PosteriorKG.lean`: posterior expected terminal gain defined
  as a Bochner integral.
- `SCOLHKG/Measure/PosteriorUpdateKG.lean`: exact posterior-update SC-OLH-KG
  value as an integral over updated terminal certified value.
- `SCOLHKG/Measure/SharedTerminalPoolKG.lean`: current, hypothetical-update,
  and realized terminal values share the pre-observation action pool; adding
  posterior frontier actions preserves both frontier and original experiment
  sets by finite-set inclusion.
- `SCOLHKG/Measure/PosteriorSamplingCandidates.lean`: random posterior-sampled
  candidate sets controlled by deterministic finite envelope pools.
- `SCOLHKG/Measure/PosteriorCoefficientSampler.lean`: code-facing
  posterior-coefficient sampler bridge; sampled-score selected candidates stay
  inside deterministic finite pools.
- `SCOLHKG/Measure/PosteriorMultivariateGaussian.lean`: mathlib
  `multivariateGaussian` posterior coefficient sampler law, with mean,
  covariance, and linear-score Gaussianity facts.
- `SCOLHKG/Measure/SafeRegretEvent.lean`: high-probability transfer from bad
  events to safe-regret failure events.
- `SCOLHKG/Measure/TaskPACBayes.lean`: finite source-prior exponential-moment
  aggregation and Markov bad-event bound feeding the task-posterior PAC-Bayes
  radius.
- `SCOLHKG/Real/TrafficTrajectoryModel.lean`: finite fresh-seed traffic
  state-action occupancy, demand-shock risk decomposition, and schema-row
  field semantics for the fresh CSV contract.
- `SCOLHKG/Real/TaskPosterior.lean`: finite generalized-Bayes task-weight
  normalization and positive-support preservation, prior-supported proposal
  mixture normalization/support lower bounds, hierarchical
  within/between/aleatoric variance, robust-envelope certification for every
  admissible posterior, and the joint task-state exact-MC optimizer bridge.
- `SCOLHKG/Real/SafeGeneralizedTaskPosterior.lean`: separate normalized
  predictive and safe-decision generalized-Bayes masses, finite clipped
  Bernoulli/pairwise log-loss radius, bounded composite-loss propagation,
  PAC-Bayes specialization centred on the safe posterior, and the dual-weight
  joint-state exact-KG implementation bridge.
- `SCOLHKG/Real/JointTaskLatentPosterior.lean`: joint finite posterior over
  structural experts and sensitivity classes, including product-prior
  normalization, positive generalized-Bayes support, normalized nonnegative
  marginals, signed scalar or source-frozen functional
  decision-bias/certificate separation, the shadow
  sensitivity-independence bridge, and the authoritative scale-floor theorem
  proving sensitivity cannot relax the theory margin. It also proves that an
  expert-conditional conjugate precision update gains information and that
  adaptive calibration covariance can only enlarge the theory margin while
  its posterior mean remains decision-only.
- `SCOLHKG/Real/StratifiedExpertKG.lean`: exact finite-expert posterior
  summation and a weighted error theorem showing that categorical expert
  sampling error is zero; only within-expert Gaussian approximation error
  remains.
- `SCOLHKG/Real/OrderedCumulativeExposure.lean`: finite positional exposure
  transform, aggregate zero-frequency special case, selected-frequency
  support, and bridge back to the same cumulative-risk decomposition.
- `SCOLHKG/Real/GroupSharedShrinkage.lean`: isotropic semantic-group
  spike/slab penalty, invariance under every within-group linear isometry, and
  shared-PIP effective-dimension accounting.
- `SCOLHKG/Real/GroupRidgeComplexity.lean`: spectral ridge effective degrees
  of freedom and the finite nested-refit selector's `2 epsilon` oracle bound.
- `SCOLHKG/Real/FinalistReplication.lean`: budgeted frozen and adaptive
  finalist ranking-and-selection, replicated chance-upper-bound soundness on
  the joint mean/variance confidence event, adaptive archive support and size,
  fixed-universe subset/cardinality invariants, completed-candidate filtering,
  finite bad-event union bounds,
  replicate-deficit decrease, and reserved suffix budget accounting.
- `SCOLHKG/Real/TwoStageDecision.lean`: the deployed two-stage budget split,
  disjoint search/verification stages, certified versus fallback terminal
  reports, strict-margin and objective uniform-error lemmas, fallback
  relative-risk control, and the deterministic three-term regret theorem.
- `SCOLHKG/Measure/TwoStageDecision.lean`: simultaneous finite-finalist margin
  and objective concentration, uniform-error extraction, search/proposal/
  verification bad-event composition, and high-probability safe-regret
  transfer without an independence assumption.
- `SCOLHKG/Real/GaussianReplicationCertificate.lean`: frozen-policy terminal
  verification margin, deterministic safety implication, safety-depth versus
  bound-excess nonvacuity condition, policy immutability, exact
  search-plus-verification budget accounting, frozen finite shortlists,
  first-certified selection soundness, and the sequential verification budget
  upper bound, including V61's unequal rank-1/rank-2 fixed budgets. V62 adds
  the direct Gaussian quantile-tolerance margin, its deterministic safety
  implication, and the exact 64/96 budget cap. V63 adds a frozen
  cumulative-risk safe-interior shortlist contract and proves that
  verification cannot alter its primary/support pair. V64 retains that
  selector and instantiates the asymmetric budget theorem at 80/96 calls.
  The controlled-heteroscedastic V9 extension freezes an objective challenger,
  the posterior-feasible primary, and a cumulative-risk support before any
  verification sample, and proves the exact 80/128/128 budget cap.
- `SCOLHKG/Measure/GaussianReplicationCertificate.lean`: false terminal
  certification is contained in the union of mean- and scale-coverage
  failures, yielding the declared per-policy error bound by a finite-measure
  union bound. The deployed Student-t and chi-square quantiles instantiate
  those two classical one-sided coverage events under iid Gaussian
  replications. The same module proves that ordered two-policy fallback has
  false-deployment probability at most the sum of its two preallocated errors;
  this family-wise result does not assume independence between certification
  events. The V62 direct-quantile variant instead contains every false
  certificate in one quantile-coverage failure. V63 specializes the
  two-policy family-wise theorem to any frozen safe-interior selector, so its
  data-dependent posterior choice does not consume the independent
  verification error budget. The V9 theorem extends the same finite union
  argument to three ordered frozen policies and proves family-wise false
  deployment at most `delta_1 + delta_2 + delta_3`, without independence
  between certificate events.
- `SCOLHKG/Real/HierarchicalBoundaryCertificate.lean`: TCB-V2 positive
  location/log-scale adaptation, planar-rotation norm preservation,
  nonnegative Cholesky/rotation/orthogonal-residual predictive variance,
  covariance non-relaxation, coverage-reserved frontier order, one shared
  frontier/terminal/final certificate, and lexicographic terminal dominance
  that prevents objective value from overriding certification status or
  positive upper margin.
- `SCOLHKG/Real/BoundaryFamilyMixtureCertificate.lean`: TCB-V3 finite
  source-frozen family posterior, credible-family mass and envelope,
  nonnegative family guard, target-name noninterference, and the combined
  family-containment plus within-family coverage failure bound.
- `SCOLHKG/Real/BoundaryFamilySynthesisCertificate.lean`: TCB-V4
  source-frozen signed-distance dictionary, monotonicity under nonnegative
  synthesis coefficients, nonnegative coefficient/residual predictive
  variance, safe recommendation under upper coverage, and target-name
  noninterference.
- `SCOLHKG/Real/BoundaryFamilySemiparametricCertificate.lean`: TCB-V5 direct
  sum of nonnegative family synthesis and a frozen nullspace-projected local
  kernel residual, nonnegative covariance/noise radius, and recommendation
  safety under upper coverage.
- `SCOLHKG/Real/OracleCertifiability.lean`: optimistic direct-replication
  oracle radius, monotone contraction with replication count, squared-budget
  sufficiency, and persistence of a certificate at larger budgets.
- `SCOLHKG/Real/SourceConsensusCommit.lean`: source-frozen rank consensus,
  strictly monotone scale invariance, safety-objective Pareto monotonicity,
  target-name noninterference, rank-spanning coverage, bounded-error two-arm
  ordering, and committed completion of a protected finalist shortlist.
- `theory.md`: theorem statements, assumptions, and proof sketches for the
  manuscript-level theory.
- `code_map.md`: mapping from mathematical objects to the current
  `SC-OLH-KG/` implementation.

## Formalization Status

Implemented in Lean4 without `sorry`, `admit`, or `axiom`:

1. Fixed-trajectory cumulative variance decomposition algebra.
2. Information-refinement reduction of apparent variance as algebraic lemma.
3. Low-rank/effective-risk truncation bookkeeping lemmas.
4. Conservative chance-feasibility certification arithmetic.
5. Weighted scalarization monotonicity for the inherited bi-objective bridge.
6. Deterministic HVD oracle-inequality skeleton.
7. Safe recommendation and safe simple-regret implication.
8. Exact KG/additive-surrogate bookkeeping.
9. Real-valued cumulative-risk decomposition and truncation lemmas.
10. Two-cell and arbitrary finite-partition laws of total variance, including
    the information-refinement variance-reduction corollary.
11. Real-valued chance certification from GP confidence plus conservative
    standard-deviation upper bound.
12. Real-valued residual-square concentration event to HVD oracle bound.
13. Real-valued exact KG/additive-proxy theorem.
14. Real-valued finite-budget safe simple-regret implication.
15. General mathlib law of total variance via `condVar`.
16. Chebyshev-derived GP confidence bad-event probability.
17. Finite-candidate simultaneous confidence via union bound.
18. Chebyshev-derived residual-square concentration bad-event probability.
19. Posterior exact KG expected gain as an integral.
20. High-probability safe-regret event transfer.
21. Sub-Gaussian one-sided tail confidence from mathlib Chernoff bound.
22. Two-sided centered sub-Gaussian GP-confidence events.
23. Finite and adaptive candidate-set sub-Gaussian union bounds.
24. Ridge-HVD residual-square oracle inequality from a concrete ridge
    minimizer and uniform concentration event.
25. Additive acquisition to exact KG approximation gap (`2 eta` theorem).
26. Information-gain radius to finite-budget safe-regret accounting.
27. Finite-kernel posterior error sub-Gaussian parameter
    `sum_i w_i(x)^2 c_i`.
28. Finite/adaptive candidate confidence for that finite-kernel GP posterior
    error model.
29. Bounded residual-square HVD concentration constants via Hoeffding's lemma.
30. Policy trajectory occupancy-risk decomposition with explicit occupancy
    remainder.
31. Exact posterior-update SC-OLH-KG expected value theorem and maximizer
    optimality.
32. Code-level rank-one GPR update/KG-slope identity.
33. Code-level HVD residual-square, replication degrees of freedom,
    nonnegative/PSD projected-iteration feasibility, weighted-objective
    descent, clipping, and certification-variance guards.
34. Robust posterior recommendation implication.
35. Finite-kernel scalar information-gain accumulation and cap bound.
36. Certificate-level line-envelope KG exactness for `compute_h`.
37. Random posterior-sampled candidate events controlled by deterministic
    envelope pools.
38. Generic sub-exponential/sharper residual-square tail interface.
39. Default sub-exponential residual-square radius wrapper.
40. Stack-hull endpoint/tail certificate bridge for `compute_h`.
41. Stack-loop pop/push cut-order preservation for `compute_h`.
42. Closed-form sub-exponential default radius inversion.
43. Final global stack dominance invariant to exact line-envelope KG.
44. Concrete `compute_h` intersection arithmetic:
    `z=(a_old-a_new)/(b_new-b_old)` gives old-line dominance on the left and
    new-line dominance on the right.
45. Popped finite envelope cells are certificate-preservingly taken over by
    the new line under the Python pop branch condition, with all processed
    lines dominated at every point of the popped interval.
46. Right-tail split branch constructs certified old finite and new right-tail
    cells under the Python break/push branch condition, with all processed
    lines dominated on the finite left piece and the whole right tail.
47. Posterior-score candidate selection from sampled coefficients is contained
    in the deterministic finite candidate pool, so its bad event inherits the
    adaptive sub-Gaussian envelope bound.
48. Finite-kernel information gain equals a determinant/log-product-style cap
    for the finite product ratio, and this cap feeds the regret accounting.
49. Full recursive sorted-line `compute_h` stack fold:
    every input line is pointwise dominated by some final output active line
    after all while-pop/push insertions.
50. Final output endpoint dominance over output active lines lifts to
    `FinalEnvelopeStackInvariant` over all original input lines, closing the
    list-output gap without a Python runtime validator.
51. Code-level theory certification margin equals the Lean chance certificate,
    and theory mode is never less conservative than legacy aleatoric-only mode.
52. Factor-HVD cumulative feature blocks aggregate exactly into
    `floor + independent + shared + linear`, and omitting nonnegative shared
    shock underestimates total risk.
53. Uniformly accurate exact-MC posterior-update KG estimators inherit the
    exact-KG maximizer gap bound.
54. Posterior coefficient sampler selection inherits finite/adaptive
    sub-Gaussian envelope bounds because selected candidates remain inside the
    deterministic raw pool.
55. Finite product-ratio information gain is bridged to a determinant-ratio cap
    and then into the safe-regret budget theorem.
56. Posterior coefficient draws with mathlib's `multivariateGaussian` law have
    the specified mean, covariance, and Gaussian linear scores.
57. Finite feature/kernel ratio caps imply both scalar-log and determinant
    information-gain caps.
58. Fresh-seed traffic state-action occupancy risk decomposes into local
    queue/wait/flow risk, shared demand-shock risk, linear shock risk, and
    floor; omitting nonnegative shared shock underestimates risk.
59. Feature-map norm, coefficient-variance, and observation-noise-floor bounds
    imply the concrete finite-kernel information-gain cap used by the regret
    theorem.
60. Exact-MC finite candidate pools inherit uniform-error probability bounds
    from centered sub-Gaussian estimator errors, then feed the exact-KG
    maximizer gap bridge.
61. Traffic fresh-log schema rows expose the exact policy/state/action and
    queue/wait/flow/demand-shock fields consumed by the encoder contract.
62. The ingolstadt21 feature map has an explicit conservative numeric
    information-gain cap with feature-norm bound `10`, coefficient variance
    cap `10`, and observation-noise floor `1e-8`.
63. The exact-MC concentration layer includes the final MC schedule theorem:
    `M` posterior-update samples reduce the per-candidate variance proxy by
    `1/M`, and the finite candidate pool is controlled by a pool-level delta.
64. The traffic schema bridge includes simulator snapshot constructors proving
    that rows emitted by the SUMO logger expose the required policy, state,
    action, cell-key, and demand-shock fields.
65. State/risk subspace projectors are invariant to any orthogonal rotation of
    the retained basis, so domain-dependent axis names do not change the
    transferred subspace object.
66. Rank-truncated whitening is orthonormal on exactly the retained,
    identifiable eigendirections; numerically null directions are not inflated
    into surrogate features.
67. A simplex mixture of source risk experts remains inside the pointwise
    expert prediction envelope.
68. Nested leave-one-out representation refitting is noninterfering with
    changes to the held-out label.
69. Strong heredity survives interaction filtering: every retained interaction
    keeps both parent main effects.
70. An additive group below the source-domain support floor is an exact
    fallback to the pre-group model.
71. A target-adapted risk representation or frequency band with observations
    on only one side of the chance boundary is an exact fallback to the frozen
    Stage-1 model.
72. Frozen source-boundary episodes may replace unstable target gain evidence,
    but cannot bypass two-sided target support or target safety; unsafe and
    one-sided cases are exact Stage-1 fallback.
73. A source-only proposal builder is invariant to every change in held-out
    target labels.
74. A representation switch is transactional: rejected proposals preserve the
    old posterior exactly, while admitted proposals commit a posterior rebuilt
    by replaying the recorded updates in their original order.
75. Positive finite generalized-Bayes expert masses normalize to a simplex and
    preserve positive support.
76. Task-posterior predictive variance is exactly expert-within epistemic plus
    between-expert mean variance plus expert-averaged aleatoric risk.
77. A pointwise upper envelope controls every normalized nonnegative posterior
    in an abstract ambiguity set, and therefore supports robust certification.
78. A zero-error MC estimator over the joint task/GPR/HVD update state recovers
    the one-step exact-KG maximizer.
79. Finite task KL is nonnegative, and every posterior satisfying
    `KL(q||p) <= rho` obeys the entropic robust upper bound minimized by the
    Python `kl_robust_expectation` dual.
80. Source-expert exponential moments bounded by one remain bounded after a
    source-prior mixture; Markov gives the target-task moment bad event
    probability at most `delta`.
81. On that event, every finite hyper-posterior obeys the explicit
    `(KL + log(1/delta))/n` PAC-Bayes generalization-gap bound.
82. Enumerating every finite expert under its normalized posterior mass is the
    exact categorical expectation, with no expert-identity sampling error.
83. If every within-expert Gaussian estimator has absolute error at most
    `epsilon`, the posterior-weighted stratified estimator also has absolute
    error at most `epsilon`.
84. The constant positional basis reproduces aggregate occupancy exactly, so
    the previous state-coupled coordinate is the zero-frequency special case
    of the ordered coordinate.
85. Selected positional frequencies preserve the finite linear exposure map,
    and unselected frequencies contribute exactly zero.
86. Replacing local exposure by the ordered finite basis leaves the same
    `floor + independent + shared + linear` cumulative-risk decomposition.
87. The adaptive spike-and-slab budget is a bound on total effective
    dimension, including the always-active prefix; an optional coefficient
    budget of `rho * N - fixed` therefore implies total dimension at most
    `rho * N`.
88. A bounded local-kernel coefficient vector in the nullspace of the frozen
    ordered/kernel cross matrix makes every resulting residual feature
    orthogonal to the ordered feature span, establishing the corrected V24
    semiparametric direct-sum bridge.
89. Uniformly bounded kernel entries and bounded projection coefficients give
    every V24 residual feature a candidate-independent finite amplitude bound;
    V23's pool-external polynomial subtraction is not used by the bridge.
90. Sharing one inclusion probability and isotropic spike/slab precision over
    a semantic coefficient group makes its prior penalty invariant to every
    orthogonal rotation of that group's coordinates.
91. A group with `m` coordinates and shared inclusion probability `q` consumes
    exactly `m q` effective dimensions, so group selection does not evade the
    total coefficient budget.
92. When the aggregate minimum-PIP floor fits the optional budget, it provides
    a feasible cardinality-projection endpoint; convex damping of two
    budget-feasible PIP vectors remains budget feasible.
93. Ridge effective dimension is a sum of spectral fractions in `[0,1]`, so it
    is nonnegative and never exceeds the feature count without imposing one
    universal rank on every task.
94. On a uniform finite-model risk-deviation event, the nested-refit group
    penalty minimizer has true risk at most the best finite penalty model plus
    `2 epsilon`.
95. Exponentiating the held-out target log-scale gives a strictly positive
    boundary scale for every finite source prior and target update.
96. Cholesky parameter uncertainty plus squared residual scale is nonnegative,
    so the TCB-V2 posterior covariance cannot reduce its upper margin.
97. Reusing one upper-margin function for frontier nomination, terminal value,
    and final recommendation makes the three decision layers extensionally
    equal; upper-margin coverage then implies final recommendation safety.
98. A certified terminal tuple lexicographically dominates every uncertified
    tuple regardless of objective value.
99. Among uncertified terminal tuples, positive upper margin is minimized
    before objective; objective breaks ties only after both certificate
    components agree.
100. A planar target/source coordinate rotation preserves squared
     risk-coordinate norm exactly.
101. Rotation and orthogonal-residual posterior covariance can only increase,
     never relax, the TCB-V2 upper chance margin.
102. The coverage-reserved finalist list places Bayes risk, authoritative
     certificate margin, robust violation, and nominal violation before every
     expert nomination.
103. A credible-family envelope controls the true margin whenever the true
     family remains in the posterior credible set and that family's own upper
     bound covers the truth.
104. A nonnegative between-family guard cannot relax this envelope; an action
     certified by the guarded envelope is therefore safe on the same event.
105. If family-containment failure has probability at most `delta_family` and
     within-family coverage failure has probability at most `alpha`, the
     TCB-V3 certificate failure probability is at most their sum.
106. A nonnegative synthesis of source boundary atoms is monotone in every
     atom; coefficient-posterior and residual uncertainty enter its upper
     margin through a nonnegative radius.
107. Any TCB-V4 recommendation whose synthesis upper margin is nonpositive is
     safe on the synthesis coverage event, and changing an unused target name
     cannot change its source-frozen coefficient update.
108. A source-frozen kernel coefficient vector in the family-design cross
     nullspace produces a local residual orthogonal to every family feature on
     the frozen design.
109. Synthesis covariance, local-residual covariance, and remaining noise
     scale form a nonnegative TCB-V5 predictive variance.
110. A TCB-V5 recommendation with nonpositive semiparametric upper margin is
     safe on the corresponding upper-coverage event.
111. The known-variance oracle mean radius is nonnegative and nonincreasing in
     the number of direct replications.
112. If `(q sigma)^2 <= (-m)^2 R` for a strictly feasible margin `m`, then the
     optimistic replicated oracle upper margin is nonpositive.
113. Once an oracle certificate holds, increasing the replication budget
     cannot revoke it.
114. Source-consensus rank order is invariant under positive affine source
     scaling, and a source-frozen proposal is target-name noninterfering.
115. Commit-before-switch completes an observed finalist exactly when the
     reserved suffix contains its remaining replication deficit.
116. Initial design, adaptive search, and verification partition the charged
     target budget as `min(n0,N-R) + (N-R-min(n0,N-R)) + R = N`, and all three
     stage sets are disjoint.
117. A certified terminal report is safe on its upper-coverage event, whereas
     a fallback report provably does not claim certification.
118. Uniform finalist objective error gives a `2 epsilon` selection bound; a
     strictly safe comparator remains empirically certified when its safety
     buffer exceeds the margin error.
119. A least-upper-risk fallback has true margin at most any completed
     comparator's true margin plus twice the uniform margin error.
120. Two-stage safe regret decomposes exactly into search, proposal-retention,
     and verification terms.
121. Finite-finalist sub-Gaussian margin/objective failures and the three
     stage-level bad events are controlled by finite union bounds without
     assuming independence.
122. Mixture epistemic variance is bounded by the source-center second moment,
     so one common task law suffices for the mean, epistemic, and aleatoric
     terms of the complete chance margin.
123. Every positive square-root tangent produces a valid joint robust chance
     upper bound, and taking the minimum of a finite family of such bounds
     preserves validity.
124. A source-affine boundary expert has an exact offset/scale error identity;
     bounded coefficient error yields an explicit uniform margin radius.
125. Positive source scale preserves the source atom's safe side, and a target
     certificate that charges the affine-transfer radius remains sound.
126. A normalized nonnegative interpolation of source percentile ranks remains
     in `[0,1]`; strictly increasing source-margin rescaling leaves its order
     unchanged, and the implemented consensus score is at most `3/2`.
127. Constraint-head authority separation makes the aggregate target GPR the
     sole source of constraint mean/epistemic uncertainty and exactly one HVD
     head the sole source of aleatoric uncertainty; discarded legacy task
     moments cannot affect certification or terminal Bayes risk.
128. Without assuming posterior-loss independence, the variance of an
     incumbent/challenger loss difference is bounded by the square of the sum
     of their posterior standard deviations whenever covariance obeys the
     Cauchy lower bound.
129. Conditional on the one-sided Cantelli probability inequality, accepting
     a switch only when its Cantelli improvement lower bound is at least
     `1-delta` implies posterior false-switch probability at most `delta`.
130. Multiplying a nonnegative posterior variance by a scale at least one
     cannot reduce that variance.
131. An HC3 projected correction written as a finite sum of nonnegative
     weights times squared projected scores is nonnegative, so adding it
     cannot reduce posterior variance.
132. Post-conditioning sandwich covariance correction leaves the conditioned
     posterior mean exactly unchanged.
133. Scaling a nonnegative base variance by at least one and then adding a
     nonnegative sandwich correction cannot reduce the base variance.
134. Any initializer selected from a nonempty canonical certified set has
     nonpositive theory upper margin.
135. If the canonical certified set is empty, no initializer can satisfy that
     certificate; a fallback ranking cannot inherit a certification claim.
136. A nonnegative sandwich correction makes the robust confidence variance
     no smaller than the central predictive variance.
137. Treating sandwich covariance as confidence-only leaves the central
     Bayes-ranking variance exactly unchanged; certification and Cantelli
     switching may still use the dominating robust covariance.
138. A certified-only optional initializer may return `some x` only with
     certified-set membership; if the certified set is empty it must return
     `none`, so no protected incumbent or safety claim is fabricated.
139. Replacing the cumulative-HVD certification upper bound by its posterior
     central value in a Bayes action cannot alter or relax the separately
     computed theory certificate.
140. If the central variance is no larger than the certification upper
     variance, and the epistemic radius and chance quantile are nonnegative,
     the central decision margin is no larger than the certificate margin.
141. The posterior expected binary chance-failure loss
     `objective + penalty * P(failure)` is monotone in failure probability and
     adds a nonnegative excess whenever penalty and probability are nonnegative.
142. Evaluate and replicate are one disjoint action type with identical unit
     target cost; every selected action's design belongs to the post-update
     observed terminal universe.
143. An admissible replication preserves the observed terminal universe, while
     a new evaluation strictly expands it.
144. If a posterior-only shortlist covers the full finite action pool within
     `epsilon_shortlist` in exact VOI and MC error is uniformly at most
     `eta_MC` on the shortlist, its MC maximizer is within
     `epsilon_shortlist + 2 eta_MC` of every full-pool action.
145. The preceding result applies directly to the explicit posterior-update
     integral over the joint GPR/task/HVD state.
146. Consistent one-step posterior value reductions telescope over a finite
     target budget; shortlist and MC approximation errors enter additively.
147. The promoted one-step VOI bound and conservative terminal chance
     certificate hold jointly without identifying Bayes-risk ranking variance
     with the separate certification upper variance.
148. The revised-paper umbrella theorem composes source-label invariance,
     the exact cumulative-risk decomposition, separated mean/risk coordinate
     equivalence, the full finite-action `epsilon_shortlist + 2 eta_MC` VOI
     gap, and conservative terminal chance feasibility in one statement.
149. The V53 certificate deficit is nonnegative and vanishes exactly when the
     observed terminal universe contains a theory-certified action. On
     separate uniform MC-error events, both accepted two-eta score gaps imply
     strict exact Bayes-risk-reduction and certificate-deficit-reduction
     improvements; fallback or switch is jointly noninferior.
150. The V54 pair-difference guard needs only an error radius for the selected
     challenger relative to the literal V51 fallback. If the nested-CRN
     prefix/high discrepancy radius dominates that exact pair-difference error,
     every accepted switch strictly improves both posterior risk reduction and
     certificate-deficit reduction; fallback or switch is jointly noninferior.
151. Joint terminal-head reuse is behavior preserving: applying the Bayes-risk
     and certificate-deficit functionals to each identical fantasy update and
     accumulating both scores in one pass equals two separate finite weighted
     passes definitionally.
152. Posterior Pareto action support is a literal V51 action superset.
     Every failed guard can execute the retained fallback, while every accepted
     action-specific pair guard inherits joint posterior noninferiority. The
     finite support's empirical usefulness remains a separately reported gate.
153. The V55 current-relative joint guard gives every action separate nested-CRN
     lower confidence bounds for Bayes-risk and certificate-deficit reduction.
     Positivity of both bounds implies strict decrease of both exact current
     terminal costs on the declared approximation events; an empty admissible
     set retains the literal V51 fallback without claiming two-head dominance.
154. The V56 pilot/confirmation split freezes one pilot-selected action before
     drawing an independent IID fantasy stream. For bounded gains, every
     fixed betting factor is nonnegative under `lambda in [0,1]`; independence
     makes its finite product an e-value under the nonpositive-mean null, and
     a finite frozen lambda mixture preserves the expectation bound. Markov's
     inequality controls each fixed batch look. Explicit two-head,
     finite-horizon, finite-look error spending then controls any false
     run-level admission; a failed confirmation retains V51.
155. A finite switch horizon spends a declared run-level terminal budget
     across charged online stages.
156. V56 confirmation failures and V57 terminal-switch failures compose by a
     finite union bound without an independence assumption.
157. The V58 signed guard decomposition reconstructs the exact robust chance
     margin; positive joint source/task correction is assigned to epistemic
     uncertainty and favorable negative coupling remains explicit.
158. The largest positive mean, epistemic, or aleatoric obstruction selects a
     finite action-support family without target truth.
159. Every guard-decomposed support retains the complete promoted V51
     evaluate-or-replicate action set, so its best posterior value cannot be
     below the retained fallback value.
160. Conditional on independent two-head confirmation, the selected V58 action
     cannot increase either posterior Bayes risk or certificate deficit
     relative to the literal V51 fallback.
161. A finite-task PAC-Bayes source miss-risk bound with explicit
     source-to-target discrepancy and effective structural dimension yields a
     conservative held-out feasible-mass lower bound.
162. If chance margin is `L`-Lipschitz in the learned coordinate and
     `L*(atlas_cover_radius+domain_shift+2*coordinate_error) <= safe_depth`,
     the deterministic atlas contains a feasible policy; nominal policy
     dimension is irrelevant.
163. If source and target normalized risk ranks align within `epsilon`, the
     deterministic atlas covers source rank within `coverError`, and a target
     safe policy is deeper than `2*epsilon+coverError`, the atlas contains a
     feasible policy. This alternative was empirically vacuous.
164. If a deterministic frozen atlas has at most `n0` members and its raw
     transferred feasible-mass lower bound is strictly positive, at least one
     atlas support member is feasible; this is a special case, not the
     headline cross-threshold contract.
165. Under a separate finite IID randomized proposal law, `n0` draws hit the
     feasible basin with probability at least `1-(1-p_lower)^n0`.
166. The backend-independent paper theorem composes source-label invariance,
     proposal coverage, the observable coordinate quotient, cumulative HVD,
     and conservative terminal certification.
167. Every deterministic atlas with budget `n0` below the policy-space
     cardinality misses some nonempty target feasible set, so unconditional
     cross-domain finite-budget coverage is impossible.

Remaining work is empirical/binding and assumption validation:

1. Validate or conservatively upper-bound the source-task exponential-moment
   slack used by the proved finite PAC-Bayes radius; sharp domain-specific
   constants are empirical/model assumptions, not free theorems.
2. Generate and archive the real fresh-seed trajectory CSV logs with the SUMO
   logger now implemented in `sumo_sim.py`.
3. Separate model-based certificate nonvacuity from independent deployment
   certification. The internal posterior theory certificate remained empty in
   the earlier 60-run V51 audit, but V64's frozen independent 80/96
   noncentral-t protocol certified all 60 fresh deployments with zero false
   certificates. Paper tables must continue to report both quantities rather
   than using terminal replication to conceal a vacuous model certificate.
4. Freeze the final feature-map numeric constants using source-only held-out
   episodes before target evaluation.
5. Validate the two-antithetic-sample exact-MC schedule and shortlist coverage
   separately. The numerical path is an estimator with the proved
   `epsilon_shortlist + 2 eta_MC` bound, not an exact integral evaluation.
6. If the final manuscript chooses a less conservative traffic feature map
   than the current ingolstadt21 cap, add that sharper numeric cap.

The observed-terminal behavior repair itself is now revalidated: all 60 paired
runs completed, retained `60/60` true-feasible recommendations and zero
adaptive losses, improved 17 runs while losing 7 and tying 36, and increased
adaptive improvements from 19 to 26. It is the promoted V51 baseline.

## Current Math-Depth Assessment

The current package is paper-grade at the finite-model theorem/interface layer:
cumulative variance decomposition, factor-HVD block aggregation, conservative
theory certification, ridge/HVD oracle steps, exact/additive/MC KG bridges,
posterior candidate envelopes, mathlib multivariate-Gaussian coefficient
sampling, residual-square tails, line-envelope KG correctness,
feature/kernel information-gain caps, traffic occupancy-risk decomposition,
fresh-log schema semantics, exact-MC concentration, and safe-regret accounting
all build in Lean without `sorry`. `PromotedV51Closure.lean` now adds the
missing end-to-end decision theorem: one observed-action terminal Bayes risk,
one joint posterior update, a unified evaluate-or-replicate action, the full
finite-action error `epsilon_shortlist + 2 eta_MC`, and finite-budget
telescoping. The Python implementation now uses the same terminal action
universe and risk penalty for the current state, every fantasy, and the final
recommendation. This removes the previous mathematical mismatch in which
fantasies could optimize over an unobserved action that the final rule could
not return.

This is not an unconditional model-based certification result. Transfer
concentration still depends on a source-task exponential-moment assumption,
exact-MC and shortlist errors require numerical calibration, and the internal
posterior certificate was vacuous in the earlier 60 promoted-control runs.
V64 separately supplies a nonvacuous independently replicated deployment
certificate: 60/60 fresh policies certified with zero false certificates under
the precommitted 80/96 protocol. These two certificate notions remain separate
in every table. The historical TCB-V2--V5 failures remain useful negative
evidence but are not part of the promoted algorithm.

V49 additionally formalizes the separation between posterior-central
aleatoric variance used by a Bayes action and conservative upper variance used
by certification. The certificate is independent of the decision-only
variance, and the upper margin dominates the central margin under the stated
nonnegativity and variance-order assumptions. The binary chance-failure
terminal loss is monotone in posterior failure probability. These theorems do
not assert empirical calibration or promotion; the paired V49 gate remains
the required implementation and performance check.

V50 further separates the posterior-nominal Bayes action from the KL-robust
decision envelope. Lean proves the exact ambiguity-premium decomposition and
its nonnegative ordering while leaving the certification upper margin
unchanged. This is a conditional implementation bridge, not evidence that the
nominal action is empirically calibrated; the paired three-domain gate decides
that question.

V51 now formalizes the complete finite promoted decision contract rather than
only nested-set monotonicity. The theorem is deliberately conditional on an
auditable shortlist coverage error and a uniform MC error; it does not claim
that four new arms approximate an arbitrary continuous action space for free.
The paired three-domain closure gate decides whether the implementation repair
retains the promoted empirical advantage.

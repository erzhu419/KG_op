# V5: Equivariant Mean Alignment and Source Misspecification

## Why V4 Did Not Close Certification

V4 established that separating the constraint-mean coordinate from the HVD
variance coordinate is necessary. The isolated latent-discrepancy arm reduced
the audit-pool false-certificate count from 261 to 174 and improved Inventory
mean MAE from 0.760 to 0.609. It still failed the gate for two distinct reasons:

1. ordered channel semantics do not transfer when the same observable role is
   rotated or permuted across domains;
2. a wrong source mean expert can retain an overconfident coefficient law even
   after its target residuals reveal misspecification.

The set-invariant V4 arm removed channel order entirely. That also removed
useful role information and therefore did not solve the first problem.

## V5 Statistical Object

V5 retains the isolated dual-head model

\[
  m_g(x)=\beta_0+\beta^\top\phi_\mu(x),\qquad
  v_C^+(x)=h_v(\psi_v(x)),
\]

and leaves the cumulative HVD head `h_v` unchanged. It changes only the
constraint-mean head.

### Equivariant channel-role matching

For source domain `e`, each observable channel has an unlabeled distributional
signature `s_{e,j}`. Source domains jointly learn canonical role prototypes
`r_k` by alternating Hungarian assignment and prototype averaging:

\[
  \pi_e=\arg\min_\pi\sum_j\|s_{e,j}-r_{\pi(j)}\|_2^2.
\]

Source chance margins orient the otherwise arbitrary role numbering. On a
held-out target, `pi_target` is fitted only from a deterministic unlabeled
policy/exposure pool. No target objective, constraint, noise, risk provider, or
post-run truth audit may be read. The descriptor is equivariant: simultaneous
permutation of channels and their assignment leaves `phi_mu` unchanged.

### Source-mean misspecification posterior

For source expert `e`, target pilot residuals are evaluated under its prior
predictive covariance `K_e`. Let

\[
  q_e=r_e^\top K_e^{-1}r_e,\qquad
  \kappa_e=\max\left\{1,
  \frac{\nu+q_e}{\nu+n_0}\right\}.
\]

The scalar variant replaces `(C_e, sigma_e^2)` with
`(kappa_e C_e, kappa_e sigma_e^2)`. The directional variant additionally adds
a PSD rank-one covariance along the ridge direction that explains the observed
mean residual. Both variants can only increase source uncertainty. Ordinary
target posterior conditioning, rather than an oracle correction, learns the
corresponding mean shift.

## Frozen Experimental Contract

- HVD variance coordinate and fitting code are unchanged from V4.
- Source archive, source-informed `n0=10`, target seeds, target dimension
  `d=1000`, proposal, Sobol diagnostic backend, and audit pool are identical.
- The run is offline-only (`N=n0=10`); target truth is available only after the
  model decision for diagnostics.
- No checkpoints or model artifacts are written; each shard writes one JSON.
- All shards run on `node001-node006`.

The six arms are:

1. `v4_latent_control`: ordered roles, no misspecification scaling.
2. `role_match_only`: learned canonical roles only.
3. `misspec_scalar`: ordered roles plus scalar scale posterior.
4. `misspec_directional`: ordered roles plus scalar/directional posterior.
5. `v5_role_scalar`: canonical roles plus scalar posterior.
6. `v5_role_directional`: canonical roles plus scalar/directional posterior.

Each arm covers FactorShock at shock scales 0 and 4, Inventory, and Queue, with
five target seeds per scenario: `6 x 4 x 5 = 120` independent shards.

## Pre-registered Gate

A challenger advances to a sequential gate only if all conditions hold:

- every shard completes and passes the oracle-free audit;
- mean and variance heads share only the observable exposure input and retain
  independent parameters;
- target role matching uses no target labels or oracle;
- misspecification covariance and residual floors never decrease;
- FactorShock scale 4 has true-feasible candidate support in at least four of
  five seeds;
- mean rank correlation is non-worse in at least three of four scenarios;
- mean MAE is non-worse in at least three of four scenarios;
- the frozen HVD variance RMSE is non-worse in at least three of four scenarios;
- total audit-pool false certifications are strictly below V4 latent control.

Passing this gate does not by itself promote V5. It permits a sequential
`N>n0` comparison. Promotion and push require that the sequential comparison
also improves recommendation feasibility without adaptive loss.

## Offline Gate Result

Run `scolh_mean_alignment_v5_offline_s5_20260718_v5` completed all 120
pre-registered shards without failure.

- `misspec_scalar` was the only passing challenger. It reduced total audit-pool
  false certifications from 174 to 148 while retaining all other gate checks.
- FactorShock false certifications fell from 5 to 0 at shock scale 0 and from
  20 to 0 at shock scale 4.
- Inventory changed from 148 to 142 false certifications; Queue changed from 1
  to 6. Therefore the gain is real but not yet a complete certification fix.
- `role_match_only` improved Inventory mean MAE from 0.609 to 0.364 and Queue
  mean MAE from 0.265 to 0.246. It simultaneously increased FactorShock false
  certifications to 207 and 232. The role coordinate is useful in selected
  domains but is not cross-domain identifiable under the current unlabeled
  matching model.
- Combining role matching with either misspecification posterior did not repair
  FactorShock and failed the gate. Those arms remain negative ablations.
- The directional covariance arm did not improve the scalar arm. Its learned
  directional mass was zero in FactorShock and nonzero mainly in Inventory and
  Queue, where it increased false certification or variance error.

Only `misspec_scalar` advances to the paired `N=20,n0=10` sequential gate. No
role-aligned arm is eligible for promotion.

## Sequential Gate Result

Run `scolh_mean_alignment_v5_sequential_n20_s5_20260718` completed all 40
paired shards. The static scalar challenger did not pass promotion:

- both arms returned 14/20 true-feasible recommendations and one adaptive
  loss;
- both arms produced zero evaluated false certificates, but every posterior
  certificate was vacuous;
- audit-pool false certifications increased from 107 under V4 control to 227
  under the static scalar law;
- the largest regressions were Inventory (73 to 179) and Queue (20 to 31).

The failure is temporal rather than an offline arithmetic error. The scalar
law is calibrated once from `n0=10`; subsequent target conditioning contracts
the source components, but the misspecification scale is not recomputed.
Consequently the initial conservative guard can disappear while sequential
mixture evidence keeps updating.

## V6 Online Hierarchical Repair

V6 retains each frozen source component `(m_e,C_e,lambda_e)` and the complete
charged target history. At target step `t`, it recomputes

\[
 q_{e,t}=r_{e,t}^{\mathsf T}K_{e,t}^{-1}r_{e,t},\qquad
 \kappa_{e,t}=\max\left\{1,
 \frac{\nu+q_{e,t}}{\nu+n_t}\right\},
\]

then refits the component from the frozen law
`(m_e,kappa_e C_e,kappa_e lambda_e)` using all charged target observations.
Mixture evidence is recomputed on the same full history. This has three
important consequences:

1. scale is never multiplied recursively onto an already-scaled posterior;
2. ordinary posterior contraction cannot silently erase the source
   misspecification law;
3. constraint-mean mixture weights and cumulative-HVD source-task weights are
   refreshed from the same posterior after every target call.

`target:null` is never scaled. Exact-KG fantasies deep-clone the frozen laws,
target sufficient statistics, and scale trajectories, so a fantasy update
cannot mutate the live posterior. Scale is not claimed to be monotone over
time; only its conservative relation to the same frozen source law is proved.

The pre-registered V6 sequential matrix compares V4 control, the failed V5
static scalar law, and V6 hierarchical scalar calibration on the same source
archive, source-informed `n0`, seeds, proposal, Sobol backend, HVD model, and
post-run audit pools. Promotion requires recommendation quality to be no worse
than both references and audit-pool false certification to improve over the
static law without exceeding V4 control.

## V6 Sequential Result

Run `scolh_mean_alignment_v6_sequential_n20_s5_20260718_v6` completed all 60
pre-registered shards. V6 was not promoted:

- V4, V5, and V6 each returned 14/20 true-feasible recommendations and one
  adaptive loss;
- all three arms had zero evaluated false certificates, but all 20 posterior
  certificates in every arm were vacuous;
- audit-pool false certifications were 107 for V4, 227 for V5, and 182 for
  V6;
- V6 eliminated audit false certification in both FactorShock strata, but
  retained 150 false certifications in Inventory and 32 in Queue.

The online scale repair therefore recovered part of V5's regression but did
not solve source-mean transfer. In Inventory the learned source scale grew by
roughly 7--12 times while sequential evidence still assigned almost all mass
to one source component. A scalar predictive-error magnitude can detect that
a source law is unreliable, but it cannot identify the direction of its mean
bias or whether the target channels have the same semantics.

## V7 Semantic-Support And Contrast Posterior

V7 keeps the isolated linear mean and cumulative-variance heads. It replaces
unconditional role transfer with two source-only uncertainty mechanisms.

First, the channel-role matcher reports a source-standardized unlabeled target
matching loss. Its source-support trust is

\[
  w_{\mathrm{role}}
  = \exp\left[-\frac12\max\left(
      \frac{L_{\mathrm{match}}}{d_{\mathrm{match}}}-1,0
    \right)\right].
\]

The requested source prior mass is multiplied by this trust; removed mass is
assigned to the domain-independent `target:null` component. This is an
equivariant semantic-support diagnostic, not a target-label gate: it reads
only source role signatures and an unlabeled held-out policy/exposure pool.

Second, `source_contrast` computes the weighted covariance of frozen source
coefficient means. After PSD projection it adds the same low-rank random-effect
covariance to every source component. Its rank is at most the number of source
domains minus one. It is learned from source laws only, while ordinary charged
target responses update the posterior mean and component evidence. Thus role
trust answers whether a source coordinate is supported, while source contrast
represents the directions in which supported source means disagree.

The V7 offline gate compares V4, raw role matching, role matching with
epistemic trust, ordered features with source contrast, and role matching with
source contrast. Every arm uses the same frozen archive, source-informed
`n0=10`, target policy pool, post-run audit pool, and no target oracle. An
offline pass only permits a paired sequential gate; promotion still requires
improved recommendation feasibility and no adaptive loss.

## V7 Result And Finite-Mixture Repair

The first V7 gate completed 100/100 shards but promoted no challenger. Raw role
matching produced 449 audit-pool false certificates. Role trust and role
contrast reduced this only to 378 and 401, respectively, while ordered source
contrast reached 153 but failed the variance-RMSE gate.

Inspection found a separate numerical error in the finite source mixture. A
FactorShock role-support trust of about `1.7e-59` was floored to `1e-12` before
the log-domain evidence update. Target evidence could therefore resurrect a
source component that the source-only support audit had effectively rejected.
V7b removed that probability floor while retaining stable log normalization.
All mixture updates now preserve exact zero support and arbitrarily small
positive support. Unit tests cover initial, sequential, and hierarchical
mixtures.

The repaired V7b gate again promoted no challenger. Audit false certificates
were 174 for V4, 153 for ordered source contrast, 393 for role trust, and 416
for role contrast. This identified a representation failure rather than a
remaining mixture-calibration failure: removing the source coefficient mass
does not repair a mismatched role feature map because `target:null` still uses
that same map.

## V8 Source-Support Adaptive Coordinate

V8 fits two constraint-mean coordinates independently from the same frozen
source archive:

1. a learned equivariant role coordinate;
2. an ordered or set-invariant fallback coordinate.

An outcome-free support audit compares only source and unlabeled target
exposure-channel cardinalities. A cardinality represented in the source
archive uses the role coordinate; an unseen cardinality uses the independently
fitted fallback. The selector does not share coefficients across the two
feature maps and does not read target responses, true margins, or oracle
feasibility. The cumulative HVD variance head remains isolated and unchanged.

Run `scolh_mean_alignment_v8_offline_s5_20260718_support_switch` completed all
120 paired result-only shards. Exactly one challenger passed every
pre-registered offline check:

- V4: 14/20 true feasible and 174 audit false certificates;
- raw role: 15/20 and 449;
- adaptive role + ordered fallback: 15/20 and 35;
- adaptive role + set fallback: 15/20 and 12, but failed the cross-domain mean
  MAE gate;
- ordered and set adaptive coordinates with source contrast: 15/20 and 65/43,
  but failed variance-RMSE and, for set features, mean-MAE gates.

Thus only `adaptive_role_ordered` advances to the paired `N=20,n0=10`
sequential gate. Promotion additionally requires a strict final recommendation
gain, no increase in adaptive loss, lower audit false certification, and
nonworse feasible regret, mean error, variance error, and certificate vacuity.

## V8 Sequential Result

Run `scolh_mean_alignment_v8_sequential_n20_s5_20260718_support_switch`
completed 40/40 paired shards and did not promote V8. Both V4 and V8 returned
14/20 true-feasible recommendations, one adaptive loss, no adaptive rescue,
zero evaluated false certificates, and 20/20 vacuous posterior certificates.
V8 lowered audit false certificates from 107 to 46 and improved Inventory mean
MAE from 0.487 to 0.279, but made only three initial-regret improvements versus
five for V4. Its median Inventory feasible regret was 0.01090 versus 0.01082.

The paired rows expose the remaining break. In Inventory the V8 finite mixture
assigned essentially all posterior mass to `target:null`; the source-aligned
coordinate survived, but the source coefficient shape did not. V8 then
estimated truly safe boundary recommendations as systematically dangerous and
carried substantially larger epistemic variance. Better global mean fit thus
became safer audit behavior without better online ranking. Queue retained the
same one adaptive loss, and the unsupported FactorShock branch exactly matched
its ordered fallback control.

## V9 Hierarchical Source-Residual Posterior

V9 tests a continuous source-to-target posterior instead of the discrete
`source expert` versus `target:null` choice. The frozen source archive defines
a Gaussian coefficient law whose covariance includes within-source uncertainty
and between-source disagreement. Charged target observations condition that
law directly, which is equivalent to marginalizing an additive target
coefficient residual. Scalar and directional empirical-Bayes variants may only
increase the source covariance and residual floor before the same conjugate
target update.

The V9 offline gate keeps the V8 support-adaptive coordinate fixed and compares
V4 mixture, V8 mixture, unscaled hierarchical residual, scalar residual scale,
and directional residual scale. Promotion requires the source law to remain
source-only, all target fine-tuning to use charged observations, uncertainty
inflation to be noncontracting, and recommendation/audit quality to beat both
controls before a sequential run is allowed.

## V9--V11 Results

V9 completed 100/100 offline shards after separating HVD source-task weights
from the constraint-mean mixture. None of its continuous hierarchical-residual
arms passed: the unscaled arm returned 12/20 true-feasible recommendations and
the scalar/directional arms returned 13/20, versus 15/20 for both controls.
Queue was the clearest regression. An unconditional source coefficient shape
is therefore not a transferable mean law merely because the role coordinate is
shared.

V10 replaced that unconditional law by a finite transferability latent between
`source:aggregate` and `target:null`. The aggregate includes both within-source
uncertainty and between-source disagreement, while target observations update
only its posterior mass. It retained 15/20 true-feasible recommendations, but
produced 64 audit false certificates versus 54 for V8. All ten extra errors
came from the unsupported FactorShock coordinate.

V11 therefore used the same outcome-free channel-cardinality selector for both
the coordinate and the coefficient posterior. Unsupported FactorShock used the
ordinary domain mixture; supported Inventory and Queue used the aggregate
latent. The 60/60 offline gate passed every registered check with 15/20 true
feasible and 54 audit false certificates, so it advanced.

The paired V11 sequential gate also completed 60/60 shards, but did not
promote. V4, V8, and V11 each returned 14/20 final true-feasible
recommendations, one adaptive loss, no adaptive rescue, and 20/20 vacuous
certificates. Audit false certificates were 110, 14, and 13 respectively.
V11 thus retained V8's calibration gain without converting it into online
recommendation value. In supported Inventory cells, target evidence assigned
essentially all mass to `target:null`; the source aggregate was correctly
rejected but the remaining isotropic coefficient prior extrapolated poorly on
the online terminal candidate distribution.

## V12 Bounded Coordinate And Function-Space Null Geometry

V12 leaves the cumulative-HVD head unchanged and modifies only the isolated
linear constraint-mean head. It addresses the distribution shift exposed by
V11 in two independently ablated steps.

First, a source-only leave-one-domain-out criterion selects a temperature from
`{0.5,1,2,4}` and maps the aligned latent coordinate through
`tanh(phi / temperature)`. The downstream mean head remains linear. This
prevents an unseen target policy from turning a moderate source-coordinate
shift into an unbounded mean prediction, while preserving channel-role
equivariance.

Second, `target:null` may use a function-space prior defined by the same
deterministic unlabeled target policy pool used for role matching. If `F` is
its design matrix, the covariance is proportional to

\[
  \left(F^\mathsf{T}F/m + \lambda I\right)^{-1}.
\]

The proportionality constant preserves the average prior predictive variance
of the previous isotropic law on that pool. This changes geometry, not total
uncertainty, and reads no target outcome or oracle quantity. V12 compares each
change alone, their combination, and the combination with V11's
support-adaptive aggregate posterior before any sequential promotion.

## V12 Offline Result

Run `scolh_mean_alignment_v12_offline_s5_20260718_bounded_geometry`
completed all 140 paired shards. No challenger passed the pre-registered
offline gate.

- The source-only `tanh` coordinate was the useful intervention. It retained
  15/20 true-feasible recommendations, reduced audit-pool false certificates
  from 213 under V4 and 54 under V8/V11 to 12, and reduced median maximum
  posterior-mean magnitude from 12.99 to 0.43 in Inventory and from 3.77 to
  0.31 in Queue.
- That global compression changed the residual geometry seen by the otherwise
  frozen HVD update. Inventory variance log-RMSE increased from 0.77 to 2.01;
  only one of four scenarios met the registered non-regression check. The
  bounded arm therefore did not advance despite its large certification gain.
- Inverse-Gram `target:null` geometry was harmful. It preserved average prior
  predictive variance and remained PSD/oracle-free, but redistributed
  uncertainty downward in some FactorShock directions. False certificates
  rose to 74 for bounded+geometry and 98 for geometry alone.

The result rejects target-pool inverse geometry, not bounded cross-domain mean
coordinates. V13 must preserve the source-supported linear geometry while
bounding only out-of-support extrapolation. A separate bounded support-overflow
feature may represent target mean discrepancy and be updated by charged target
observations through the same linear posterior.

## V13 Offline Result And Head-Leakage Audit

Run `scolh_mean_alignment_v13_offline_s5_20260718_support_projection`
completed all 160 paired shards. No support-projected arm passed.

- Source-support clipping retained 15/20 true-feasible recommendations and
  bounded posterior extrapolation, but produced 147 audit false certificates.
  All errors came from FactorShock, where source LODO selected the full support
  quantile and retained a source slope whose semantics did not transfer.
- Adding one bounded support-overflow discrepancy feature increased false
  certificates to 164. With only ten target calls, this extra coefficient was
  not identifiable enough to repair the wrong source slope.
- The support arms also failed mean-rank and variance-RMSE checks. They do not
  advance to sequential evaluation.

The variance failure exposed a deeper implementation leak. Although the mean
and cumulative-risk feature maps and coefficients are separate, HVD
initialization still used in-sample residuals from the fitted constraint GPR.
Changing `phi_mu` therefore changed `global_var` and class floors in the HVD
head. V14 closes this leak: singleton target policies receive a prediction from
the frozen replicated-source HVD prior, while only within-policy target
replication may replace that prediction with target variance evidence.

## V14 Result And Remaining Mixture Leak

Run `scolh_mean_alignment_v14_offline_s5_20260718_singleton_isolation`
completed 120/120 paired result-only shards. The source-prior singleton rule
removed V12's variance-RMSE regression, but V14 was not promoted. Bounded
coordinates reduced audit false certificates from 54 to 46 after isolation,
whereas the old residual-coupled bounded control reported 12. More
importantly, changing only the mean coordinate still changed the final
variance calibration metrics.

The remaining path was outside each HVD: the ensemble mixed expert HVD shapes
with the same task weights that were updated by constraint-mean predictive
scores. Hence every expert variance head was isolated, but their mixture was
not. The old count of 12 was partly obtained by allowing mean misspecification
to move the variance mixture and cannot serve as evidence for a separated
model.

## V15 Product Posterior

V15 represents task uncertainty by two posteriors:

\[
  Q_t^{\mu}(z_\mu)
  \quad\text{and}\quad
  Q_t^{v}(z_v),
  \qquad
  Q_t(z_\mu,z_v)=Q_t^{\mu}(z_\mu)Q_t^{v}(z_v).
\]

`Q_t^mu` receives the ordinary charged target response and chance-boundary
scores. `Q_t^v` begins at the frozen source-task prior and receives only the
Gaussian sample-variance likelihood from a within-policy replication. A
singleton cannot update it. Repeated sample variances at the same policy
replace one sufficient-statistic record rather than being double counted.

Prediction uses `Q_t^mu` for the constraint mean and epistemic variance and
`Q_t^v` for cumulative HVD variance. Certification separately robustifies the
two heads before applying

\[
  \mu_g^+ + \sqrt{\beta_g}s_g^+
    + z_\alpha\sqrt{v_C^+} \le \tau.
\]

Exact-KG fantasy sampling uses the Cartesian-product expert law and clones and
updates both posteriors. The V15 gate returns to a linear downstream mean head,
retains learned equivariant channel-role matching, and independently ablates
source-contrast misspecification uncertainty and the source-only bounded
coordinate. Promotion requires exact per-seed invariance of all variance
metrics across mean-coordinate variants; conservative behavior caused by a
mean-to-HVD leak is no longer credited.

## V15 Result

Run `scolh_mean_alignment_v15_offline_s5_20260719_product_posterior`
completed all 140 result-only shards. No challenger advanced.

- Every isolated arm satisfied exact per-seed HVD invariance. Changing the
  mean coordinate no longer changes variance RMSE, coverage, or any HVD
  diagnostic.
- The bounded `source_tanh` mean coordinate improved mean MAE over the V8
  shared control in all four scenarios and reduced false certificates from 54
  to 33 while retaining 12 true certificates.
- The replication-only variance task posterior had no target replication in
  this `N=n0=10` gate and remained at its frozen source prior. Its FactorShock
  scale-4 variance log-RMSE was 1.602, versus 1.315 for the old response-updated
  shared posterior. The old improvement is inadmissible evidence for separated
  heads because target mean scores selected the HVD expert.
- Source contrast did not provide a stable extra gain. The valid next target is
  the constraint-mean coordinate and its epistemic calibration, not another
  residual-based HVD update.

## V16 Partial Role Transport

V16 replaces the hard cardinality gate by a source-trained partial transport
between observable channels and canonical roles. For a source atlas with `K`
roles and a target with `J <= K` channels, the transport has unit row mass and
role capacity at most one. Missing role mass remains visible in the descriptor.
Its entropy temperature is selected only by leave-one-source-domain
channel-dropout replay against source chance margins.

This directly targets the unresolved V15 case: held-out FactorShock has two
observable channels while its Inventory/Queue source atlas has three. The old
adaptive coordinate discarded role semantics and used an ordered fallback;
raw hard matching forced two channels into three unsupported source roles.

The same unlabeled target match supplies a misspecification scale
`1 + excess_matching_loss + assignment_entropy + cardinality_gap >= 1`.
`matching_uncertainty` multiplies only source coefficient covariance by this
scale. It cannot shift the mean, reduce epistemic uncertainty, update HVD, or
read a target response/oracle. V16 keeps the linear downstream head, the
source-selected bounded `tanh` coordinate, and the V15 product posterior.
Promotion requires fewer false certificates without losing V15's 12 true
certificates, exact HVD invariance, and non-worse mean/rank/regret behavior.

## V16 Result

Run `scolh_mean_alignment_v16_offline_s5_20260719_partial_transport`
completed all 140 result-only shards. No arm advanced.

- V15 retained 12 true and 33 false audit-pool certificates. Every partial-
  transport arm returned zero certificates: the apparent false-certificate
  improvement was entirely vacuous and lost all 12 true certificates.
- On FactorShock, hard and partial role matching changed median constraint-mean
  rank correlation from about `0.39` to `-0.38`. The target's two channels were
  force-matched to a three-role source atlas by marginal channel shape; that
  shape does not identify which source control roles a target channel spans.
- Outcome-free matching uncertainty did not repair the coordinate. Depending
  on the coordinate/mixture branch it either emptied the certificate set or
  increased false certificates from 33 to 221. A global matching-loss scalar
  is therefore not a calibrated source-mean misspecification posterior.
- All variants again had exactly identical replication-only HVD diagnostics.
  The negative result belongs to the mean coordinate, not to variance-head
  leakage.

V17 replaces marginal channel-shape matching by an observable intervention-
response coordinate. Each channel is represented by its finite-dimensional
response to a shared normalized low-frequency policy intervention library.
When the target has fewer channels than the source atlas, entropic barycentric
transport may represent one target channel as a convex combination of source
roles. Source signatures use deterministic unlabeled intervention pools; role
orientation and smoothing selection use source outcomes only, and target
matching remains outcome-free. V17 separately ablates the online hierarchical
predictive misspecification posterior on top of both the V15 ordered coordinate
and the intervention-response coordinate. The HVD product posterior remains
unchanged.

## V17 Result And Identifiability Split

Run `scolh_mean_alignment_v17_offline_s5_20260719_intervention_response`
completed all 80 result-only shards. No challenger advanced.

- V15 remained the strongest arm with 12 true and 33 false audit-pool
  certificates and 15/20 true-feasible recommendations.
- Intervention-response transport preserved 15/20 true-feasible
  recommendations, but reduced FactorShock mean-rank correlation from about
  `0.39` to `0.14-0.22`. The plain transport arm returned 1 true and 35 false
  certificates. Its hierarchical variant returned 10 true and 34 false
  certificates, so neither improved V15.
- Global hierarchical mean misspecification on the ordered coordinate removed
  every false certificate by also removing every true certificate. It reduced
  true-feasible recommendations to 13/20 and is therefore vacuous rather than
  calibrated.
- Every arm again had an exactly invariant replication-only HVD head. The
  failure is not caused by mean-to-variance leakage.

The gate also exposes a necessary identifiability split. FactorShock scale 0
and scale 4 have the same constraint mean but different aleatoric variance.
At `N=n0=10` there are no target replications, so a separated mean posterior
cannot infer the shock scale. Inflating mean uncertainty to compensate either
leaves the scale-4 false certificates or empties the certificate set. The next
stage therefore tests two independent claims:

1. A target-specific low-rank residual mean coordinate may adapt source
   semantics using only unlabeled target geometry plus charged target outcomes.
2. Within-policy target replication must update the independent HVD posterior
   before different shock scales can be certified.

Neither claim is allowed to borrow evidence from the other. The mean gate holds
the variance head fixed and excludes the scale-4 variance-identifiability case;
the replication gate holds the mean coordinate fixed and measures variance
recovery directly.

## V18 Result: Fixed Residual Rank Is Not Transferable

Run `scolh_mean_alignment_v18_offline_s5_20260719_target_residual`
completed all 75 result-only shards. No fixed-rank challenger passed.

- The V15 control produced 9 true and 5 false audit-pool certificates over
  zero-shock FactorShock, Inventory, and Queue, with 15/15 true-feasible
  recommendations.
- Rank-one residual coordinates retained all 15 true-feasible recommendations
  but increased false certificates to 95 and 122. Their new direction fitted
  charged pilot noise as transferable mean structure.
- Rank-two residual coordinates reduced false certificates to 2 and improved
  FactorShock mean rank correlation from about 0.386 to 0.741, but retained
  only 3 true certificates. Inventory and Queue remained vacuous. The fixed
  rank therefore over-inflated epistemic uncertainty where the extra target
  directions were unsupported.
- Every arm preserved exact HVD invariance. The failure belongs entirely to
  constraint-mean structure selection.

Orthogonality to the source mean span prevents duplicate coordinates but does
not identify which residual directions explain the target chance boundary.
The useful FactorShock rank-two signal and the harmful Queue rank-two signal
also rule out selecting one rank globally or by a hand-coded domain label.

## V19 Registered Residual-Structure Posterior

V19 keeps one maximum-rank-two observable basis and introduces a discrete
latent structure variable `K in {0,1,2}`. In rank `K`, only the first `K`
nested orthogonal residual coefficients have non-negligible prior variance.
Source expert and target-null laws are both expanded over the same `K`; charged
target observations update the joint expert/rank mass by Gaussian marginal
likelihood. Prediction is the exact first two moments of this finite mixture,
including between-rank disagreement as epistemic covariance.

This is not a domain gate. The coordinate and complexity prior are fixed before
target outcomes; ordinary target observations perform Bayesian structure
adaptation, and no target oracle enters either the basis or the update. The
offline gate compares complexity, mild, and uniform rank priors against V15 and
the fixed V18 rank-two diagnostic. Promotion requires nonvacuous true-
certificate support, a strict false-certificate reduction, exact independent
HVD invariance, and non-regression of mean error, rank, regret, and true-
feasible recommendations.

## V19 Result: Rank Was Not The Missing Transfer Variable

Run `scolh_mean_alignment_v19_offline_s5_20260718_rank_posterior`
completed all 75 result-only shards. No rank-mixture arm passed.

- V15 retained 15/15 true-feasible recommendations, 9 true audit-pool
  certificates, and 5 false certificates. The fixed V18 rank-two diagnostic
  retained only 3 true certificates and 2 false certificates.
- Complexity, mild, and uniform rank priors all converged to the same empirical
  result as fixed rank two: 15/15 true-feasible recommendations, 3 true
  certificates, and 2 false certificates. They reduced false certification but
  failed the registered nonvacuous-support check.
- Target evidence did update the rank probabilities. FactorShock selected rank
  two in every seed, Inventory mostly selected rank two, and Queue selected rank
  one. Hence the negative result is not a dormant structure posterior.
- The source structured mass collapsed almost completely to `target:null`.
  In a representative FactorShock seed it was `2.31e-21`; source marginal log
  evidence ranged roughly from `-108` to `-245`, versus `-5.49` for the target
  null. Rank selection cannot repair a source coordinate whose role semantics
  do not transfer.
- Every variant again preserved the replication-only HVD head exactly. The
  remaining failure is source-to-target constraint-mean semantic alignment.

## V20 Finite Channel-Role Assignment Posterior

V20 replaces one source-selected hard/soft role match by a finite posterior
over every admissible injection of target channels into source-fitted canonical
roles. For `J` observable target channels and `K` source roles,

\[
  p(g_t\mid D_t)
  =\sum_{\pi\in\operatorname{Inj}(J,K)}
    q_t(\pi)\,p(g_t\mid D_t,\pi).
\]

The source archive learns the role coordinate and the linear mean head. The
finite set of assignments and its uniform prior use only observable target
exposure and contain no target response or oracle label. The normal charged
`n0` target observations update `q_t(pi)` by Gaussian marginal likelihood.
Source experts and `target:null` are expanded over the same assignment orbit,
so role uncertainty cannot be confused with source-versus-target
misspecification.

All assignment blocks are stored in one deterministic basis for numerical
compatibility, but each mixture atom activates only one low-dimensional role
block plus the intercept. Moment matching retains within-assignment covariance
and between-assignment disagreement. Simultaneously permuting target channels
and the assignment produces the same role descriptor; relabeling the finite
atom enumeration therefore leaves the mixture prediction invariant.

The V15 product-posterior split remains mandatory. Assignment weights affect
only the constraint-mean GPR. Cumulative factor-HVD uses the independent frozen
source variance posterior and replication-only target evidence. A post-run
oracle expressivity audit refits every registered assignment against the shared
truth pool only after the recommendation is frozen; it is explicitly marked
`target_oracle_used_for_decision=false`. This audit separates two failure
modes: no assignment can represent the held-out boundary, or the admissible
family is expressive but ten charged observations cannot identify its useful
atom.

The registered offline gate compares V15, V17 hard intervention transport,
V18 fixed rank two, plain assignment Bayes, and assignment Bayes with the
hierarchical source-mean misspecification posterior. Promotion requires an
actual target-likelihood update of assignment mass, identical source/null
assignment orbits, exact HVD invariance, nonworse true-feasible and true-
certificate support, strictly fewer false certificates, and non-regression of
mean error, rank, and feasible regret.

## V20 Result: The Family Is Expressive, The Structure Score Is Wrong

Run `scolh_mean_alignment_v20_offline_s5_20260718_role_assignment`
completed all 75 result-only shards. No assignment arm passed.

- All five variants returned 15/15 true-feasible recommendations and every
  V20 contract passed: assignment hypotheses were outcome-free, charged target
  observations changed their mass, source and target-null used the same orbit,
  and HVD metrics were exactly invariant to the mean structure.
- Plain assignment Bayes produced 5 true and 172 false audit-pool
  certificates. All 172 false certificates were in Inventory. Its ordinary
  marginal evidence collapsed to one assignment with effective assignment
  count near one.
- Hierarchical assignment Bayes produced zero false certificates by producing
  zero certificates of either kind. Mean MAE and rank also regressed in at
  least two domains. This is conservative misspecification without useful
  certification, not an improvement.
- Across 30 challenger runs, the maximum-posterior assignment matched neither
  the oracle-pool best-MAE nor best-rank assignment once. The mismatch was
  systematic: FactorShock selected `2-1` while `0-2` was best; Inventory
  selected `1-0-2` while identity/reverse assignments were best; Queue selected
  `1-0-2` or `0-2-1` while `2-1-0` was usually best.
- The assignment family itself was not empty. On the ten-point initial design,
  an oracle-only refit could rank Queue margins at about `0.94`; on the shared
  512-point audit pool its best assignment reached roughly `0.60-0.68` rank
  correlation across the three domains. The current source-prior marginal
  likelihood did not identify that assignment from the same charged pilot.

V21 therefore keeps the V20 hypothesis class but changes the structure update.
Each assignment is refit from its frozen source law on all but one charged
target observation and scored on the omitted chance margin. Summed exact
leave-one-out Gaussian predictive log scores define a generalized finite Bayes
posterior. This score uses no target oracle, measures held-out rather than
in-sample support, and retains assignment disagreement in epistemic covariance.
The complete component bank is refit from its frozen laws after every new
target call so the sequential implementation has the same semantics as the
offline gate.

The registered V21 offline gate compares the V15 linear mean control, the V20
marginal-likelihood assignment posterior, and LOO predictive posteriors at
temperatures `0.5`, `1.0`, and `2.0`. All arms use the same frozen source
archive, source-informed `n0=10`, post-run truth pool, and independent
replication-only HVD posterior. Promotion additionally requires a strict
increase over V20 in post-run assignment/oracle agreement; oracle agreement is
an audit metric and never enters fitting, candidate selection, certification,
or recommendation.

## V21 Result: Cross-Fitting Learns The Pilot, Not The Target Pool

Run `scolh_mean_alignment_v21_offline_s5_20260718_loo_predictive`
completed all 75 result-only shards. No LOO arm passed.

- The V15 control retained 15/15 true-feasible recommendations, 9 true audit-
  pool certificates, and 5 false certificates.
- LOO temperatures `0.5` and `1.0` retained 10 true certificates but produced
  322 and 294 false certificates, respectively, and only 11/15 true-feasible
  recommendations. Temperature `2.0` reduced false certificates to 3 by also
  eliminating every true certificate and retained only 12/15 true-feasible
  recommendations.
- All software and statistical contracts passed: exact LOO scores, refitting
  from frozen source laws, charged-target-only updates, source/null assignment
  orbit equality, and exact HVD invariance.
- Posterior assignment agreement with the post-run pool oracle improved from
  V20's 0/30 to only 2/30. In Inventory, all LOO variants selected `1-0-2`,
  the best local-MAE assignment on the ten source-informed pilot points, while
  `2-1-0` was the best full-pool rank assignment. V15's deterministic
  source-learned, target-unlabeled role matcher had already selected `2-1-0`.

The failure is therefore covariate-shifted structure selection. Uniformizing
the assignment prior discards a useful transferable geometric match; ten
charged pilot labels then identify the best local interpolant, not the best
global boundary coordinate.

## V22 Source-Geometry Assignment Prior

V22 restores the information V20 removed. For every admissible assignment
`pi`, the frozen source role atlas and deterministic unlabeled target exposure
pool define a normalized matching cost `L_t(pi)`. Source-domain best/second-
best matching gaps calibrate a temperature `T_src`, and

\[
  q_0(\pi\mid E_t)
  \propto \exp\{-L_t(\pi)/(s T_{\rm src})\}.
\]

This prior uses no target outcome or oracle. Relabeling target channels
relabels assignments and their costs, leaving aligned predictions unchanged.
Both source experts and `target:null` receive exactly the same geometric
assignment prior. Charged target evidence may subsequently adapt the weights;
between-assignment disagreement remains epistemic uncertainty and HVD remains
an independent product posterior.

The registered gate compares V15, V21 uniform LOO at temperature 2, a strong
geometry prior with ordinary marginal evidence, and strong/weak geometry
priors with LOO evidence. The two prior scales are preregistered source-
calibration multipliers, not selected from held-out truth.

## V22 Result: Joint Updating Confounds Roles And Mean Misspecification

Run `scolh_mean_alignment_v22_offline_s5_20260718_geometry_prior` completed all
75 result-only shards. No geometry-prior challenger passed.

- V15 retained 15/15 true-feasible recommendations, 9 true audit-pool
  certificates, and 5 false certificates.
- Geometry marginal likelihood retained 14/15 true-feasible recommendations
  but produced 324 false certificates. Geometry plus LOO retained only 11-12
  true-feasible recommendations; the strongest two variants produced 71 and 89
  false certificates and only 3 true certificates in total.
- The source-geometry prior itself was often sharp, but the joint posterior
  retained its hard role assignment in only 4/15 runs. Charged pilot outcomes
  could improve their local likelihood by moving both the role assignment and
  the source/null expert identity at once.
- Every arm preserved exact replication-only HVD invariance. The failure is a
  probability-graph error in the constraint-mean posterior, not variance-head
  leakage.

## V23 Factorized Role And Misspecification Posterior

V23 separates two latent variables that V20-V22 updated jointly. Let `pi` be
the equivariant channel-role assignment learned from the frozen source atlas
and unlabeled target exposure, and let `e` select a source-domain or target-null
constraint-mean expert. The target posterior is

\[
  q(\pi,e\mid D_t,E_t)
  = q_0(\pi\mid E_t)\,q(e\mid \pi,D_t).
\]

The top-level assignment law `q_0` is frozen after observing `E_t`; charged
target outcomes cannot alter it. Within every assignment, those outcomes do
update the conditional source/null expert law. This lets target data diagnose
source-mean misspecification without redefining observable channel semantics.
Moment matching retains disagreement across both assignments and experts as
epistemic covariance. The independently updated cumulative HVD posterior is
unchanged.

The registered offline gate compares V15, the failed V22 joint update, and
factorized posteriors at geometry-temperature scales `0.05`, `0.25`, and
`1.0`. Promotion requires exact preservation of every assignment marginal,
nontrivial target updating of conditional expert mass, no target-label or
oracle use in the assignment marginal, exact HVD isolation, all 15 selections
retaining the geometry-hard role, nonworse feasible/certificate support, a
strict false-certificate reduction, and non-regression of mean error, rank,
regret, and true-feasible recommendations.

## V23 Result: Correct Factorization, Vacuous N=10 Certification

Run `scolh_mean_alignment_v23_offline_s5_20260718_factorized` completed all 75
result-only shards with no failures. No challenger advanced.

- Every factorized arm retained the geometry-hard assignment in 15/15 runs,
  versus 4/15 for the V22 joint posterior. All assignment marginals remained
  numerically fixed while charged target outcomes changed source/null expert
  conditionals. HVD remained exactly isolated.
- All three geometry temperatures returned 15/15 true-feasible
  recommendations and zero false certificates. They also returned zero true
  certificates. V15 returned 9 true and 5 false certificates, while V22
  returned 3 true and 89 false certificates.
- The factorized arms reduced feasible regret nontrivially in no domain because
  all methods used the same frozen `n0` proposal and `N=n0=10`. Their mean MAE
  and rank regressed relative to V15 in at least two domains.
- Most importantly, post-run oracle substitution found zero certifiable pool
  points in every domain even after substituting both the true constraint mean
  and true aleatoric variance. Median maximum true safety depths were only
  about `0.060`, `0.054`, and `0.036` for FactorShock, Inventory, and Queue,
  while the factorized posterior's best feasible epistemic radii were about
  `0.091`, `0.271`, and `0.298`. At ten observations the registered theory
  certificate is therefore structurally vacuous, not merely poorly ranked.

V24 keeps the mathematically correct factorization and moves to a sequential
`N=20,n0=10` gate. Within each frozen assignment it adds the existing
full-history hierarchical predictive-scale posterior to every source expert.
Each online observation recomputes a noncontractive source-misspecification
scale from the frozen law, updates source/null expert conditionals, and refits
all components. Assignment marginals and the HVD factor remain fixed. The
registered comparison is V15, factorized-without-scale, and hierarchical
factorized variants with prior degrees of freedom `4` and `16`; the latter is
a preregistered robustness check rather than target-selected tuning.

## V24 Result: Scale Inflation Rejects Confidence, Not Mean Bias

Run `scolh_mean_alignment_v24_sequential_s5_20260718_hierarchical` completed
all 60 result-only shards. Neither hierarchical arm passed.

- Both degrees-of-freedom settings preserved the geometry assignment in all
  15 runs, used all ten charged online observations, refit every scale from its
  frozen source law, and left the replication-only HVD posterior independent.
- Both hierarchical arms returned zero true and zero false certificates and
  retained only 14/15 true-feasible recommendations. V15 retained 15/15; the
  factorized no-scale control retained 13/15.
- Mean MAE and boundary rank regressed in at least two of three domains. The
  learned source scales frequently reached tens or the registered cap of 100.
  Those broad source laws then dominated the target-null marginal evidence,
  even though their posterior means remained misspecified. Plug-in scale
  inflation therefore rewarded a diffuse wrong expert instead of learning a
  target mean correction.
- Oracle substitution showed that the hierarchical arms had no certifiable
  audit-pool point even with true mean and true aleatoric variance. The
  no-scale factorized control had five such points, confirming that the added
  epistemic radius itself destroyed the remaining certification depth.
- The fixed geometry role was also semantically wrong where transfer matters:
  it matched neither the post-run best-MAE nor best-rank assignment in any
  FactorShock seed, and matched the best-rank Queue assignment in zero of five
  seeds. Unlabeled marginal channel geometry is not a sufficient
  chance-boundary coordinate.

V25 therefore removes global source-scale inflation. It keeps the independent
mean/HVD product posterior and learns a low-dimensional, permutation-
equivariant role likelihood from the ordinary charged pilot observations. The
source archive supplies a distribution over each canonical role's association
with the chance margin; the target pilot supplies noisy channel-margin
associations. Their uncertainty-aware finite Bayes update changes assignment
mass without using target oracle values or a full-pool interpolant. Source
mean discrepancy is then learned separately inside each assignment from the
source-domain coefficient contrasts, so role uncertainty and mean bias cannot
create covariance across mutually exclusive assignment blocks.

## V25 Result: Signed Source Roles Do Not Transfer

Run `scolh_mean_alignment_v25_sequential_s5_20260718_boundary_role` completed
all 90 result-only shards. No V25 arm passed. The analyzer now distinguishes
the charged-pilot boundary posterior from the original source-geometry prior;
under that corrected contract every V25 arm implemented its registered finite
posterior correctly, but still failed empirically.

- V15 retained 15/15 true-feasible recommendations and one oracle-both
  certifiable pool point. V23 retained 13/15 and five, respectively.
- Every V25 arm retained 14/15 true-feasible recommendations, zero true or
  false certificates, and zero oracle-both certifiable pool points.
- Every V25 arm changed the assignment in exactly the five Queue runs, and all
  five changes selected a role ordering with worse post-run boundary rank.
- Source/target signed channel-margin correlations reversed systematically.
  Inventory and Queue target channels were strongly negatively correlated with
  the margin, while the corresponding source role means were mostly positive.
  A more strongly concentrated likelihood therefore made transfer worse.
- The first four charged source-informed target points were nearly collinear.
  Reordering the same ten points by a D-optimal criterion improved local
  excitation, but did not recover the full-pool role ordering because the
  source role signs and semantics themselves were not invariant.

The discrete source role is therefore the wrong transferable random variable.
Its failure is not HVD error and cannot be repaired by another assignment
temperature or by globally inflating source uncertainty.

## V26 Exchangeable Target-Linear Mean Posterior

V26 transfers only an exchangeable source coefficient law. For observable
channel `j`, the linear block is

\[
  b_j^\top (\bar e_j,\bar e_j^2,s_j^2),
\]

with one separate global dispersion feature. Source domains fit their own
channel blocks, but the archive retains only the within-domain exchangeable
block mean and covariance. The same law is copied to every held-out target
channel. Consequently, source data transfer shrinkage and effect scale but no
channel identity, sign, or ordering. Ordinary paid target observations update
the channel-specific `b_j`, so role sign and magnitude are target posterior
quantities rather than source labels.

The coordinate has 13 fixed features for at most four observable channels and
is equivariant to channel relabeling. The source prior is invariant under that
relabeling. The scalar mean is unchanged when feature and coefficient blocks
are relabeled together. Source-domain components remain separate and compete
with `target:null`; the hierarchical variants can only increase each source
component's predictive uncertainty when charged target residuals reveal mean
misspecification. Cumulative HVD remains a separate posterior head over the
same observable exposure.

The registered sequential gate uses identical frozen source archives,
source-informed `n0=10`, `d=1000`, `N=20`, and five seeds in FactorShock,
Inventory, and Queue. Controls are V15, V23, and V25. Challengers are the plain
exchangeable posterior and hierarchical predictive-scale variants with source
degrees of freedom 4 and 16. Promotion requires 15/15 true-feasible results,
at least V23's oracle certifiability, no false-certificate regression, mean and
regret non-regression in at least two domains, exact HVD isolation, and runtime
evidence that initially identical source channel blocks became distinct only
after target posterior updating.

## V26 Result: Ranking Transfers, Absolute Certification Does Not

Run `scolh_mean_alignment_v26_sequential_s5_20260718_exchangeable` completed
all 90 result-only shards without a scheduler failure. No challenger advanced.

- The exchangeable coordinate learned useful target ranking. Median
  constraint-mean rank correlation increased from about `0.43` under V15 to
  `0.87`, `0.97`, and `0.96` in FactorShock, Inventory, and Queue. Charged
  target observations differentiated all target channel blocks, while every
  source block remained identical and no source role identity was transferred.
- Every V26 arm retained 14/15 true-feasible recommendations, compared with
  15/15 for V15. All returned zero true and zero false theory certificates and
  zero points certifiable after oracle mean-and-variance substitution. V23
  retained five oracle-substitution-certifiable points.
- The two hierarchical source-scale variants made no decision-level change.
  Their finite source-expert mixtures still carried between-component
  disagreement as epistemic covariance; increasing component scale could only
  make the already-vacuous certificate more conservative.
- Queue seed 2 exposed the decision failure. A true-feasible incumbent already
  existed in `n0`, but the empty certificate caused the diagnostic fallback to
  prefer a lower-objective, truly infeasible point. The true safe depth was
  about `0.036`, while even the smallest audited epistemic radius was about
  `0.255`; this is a finite-information failure, not an HVD tuning failure.
- Source-domain scalar scores did not transfer target rank. V26's improvement
  came from target-learned channel-specific coefficients under exchangeable
  shrinkage, not from selecting or affinely recalibrating a frozen source
  direction.

The original exact variance-output comparator was also too strong for a
representation-dependent `sobol_new` candidate pool: different mean
coordinates induced different charged evaluation paths, so their final HVD
states need not be numerically identical even when the variance head is
structurally independent. Future gates test structural head isolation. Exact
state equality is required only under a fixed shared evaluation path.

## V27 Single Exchangeable Empirical-Bayes Hyperlaw

V27 removes the discrete source-domain identity from the target posterior.
The source archive estimates one exchangeable Gaussian hyperlaw over channel
coefficient blocks. Within-source estimation covariance and between-source
coefficient variation are moment matched once, after which the held-out target
is treated as a new draw from that law. Charged target observations condition
this single Gaussian posterior directly:

\[
  \theta_s\mid\mathcal D_s \leadsto
  (\widehat\mu_\theta,\widehat\Sigma_\theta),\qquad
  \theta_t\sim\mathcal N(
    \widehat\mu_\theta,\widehat\Sigma_\theta),\qquad
  y_t\mid\theta_t\sim\mathcal N(\Phi_t\theta_t,\Sigma_{y,t}).
\]

There is no source-expert selector and no target-null mixture. Plain V27 uses
the conjugate aggregate posterior. Conservative challengers recompute a scalar
or rank-one directional misspecification inflation from the complete charged
target history before conditioning; every update refits from the same frozen
hyperlaw, so inflation cannot compound recursively or make a rejected source
law more confident. The cumulative HVD posterior stays an independent head
over the same observable exposure.

The V27 gate keeps the V15, V23, and V26 controls and compares aggregate
`none`, `predictive_scale`, and `predictive_scale_directional` variants at the
same `d=1000,N=20,n0=10`, three domains, and five seeds. Promotion requires
15/15 true-feasible recommendations, no false-certificate regression, the V26
rank gain in at least two domains, one actual aggregate Gaussian target
posterior with charged-data updating, no retained source-domain identity, and
structural independence of the HVD head. Certification is reported separately
from optimization because a correct N=20 posterior may remain uncertifiable.

### V27 Sequential-Update Audit

The first V27 preflight exposed an implementation mismatch: the constraint GPR
conditioned on all target observations, but the misspecification diagnostic was
calibrated only once at the ten-point pilot and then remained frozen. The
corresponding 90-task gate was cancelled and is not admissible evidence. The
corrected implementation represents the empirical-Bayes hyperlaw as a one-atom
Gaussian posterior named `source:aggregate`. Every paid online observation is
appended to the same target history, after which scalar and directional
misspecification are recomputed from the frozen source hyperlaw and the full
history before conjugate conditioning. No inflated posterior is recursively
reused as a prior.

Run `scolh_mean_alignment_v27_seqfix_preflight_s1_20260718` verified this
contract in FactorShock, Inventory, and Queue. Each result contained one
aggregate component, no target-null component, no source-domain selector, an
11-entry scale trajectory from target counts 10 through 20, and exactly ten
online refits. The exact-KG clone and checkpoint state preserve the same mode,
ridge, frozen hyperlaw, and charged target history. All three preflight
recommendations were truly feasible and none used target-oracle information.
As expected from the V23--V26 information audit, all three theory certificates
remained empty at `N=20`; optimization quality and certification coverage are
therefore evaluated as separate gate outcomes.

## V27 Result: One Hyperlaw Restores Mean Transfer

Run `scolh_mean_alignment_v27_seqfix_s5_20260719_single_hyperlaw` completed all
90 result-only shards with no scheduler failure. All three aggregate arms passed
the registered gate, and `exchangeable_aggregate_none` is promoted as the new
mean-coordinate baseline. The post-gate tie-break is explicit: among eligible
arms, maximize oracle-substitution certifiability and then prefer the least
misspecification complexity.

- Every aggregate arm returned 15/15 truly feasible recommendations and zero
  adaptive losses. V26 returned 14/15 and V23 returned 13/15. Every aggregate
  posterior used one `source:aggregate` Gaussian, all ten paid online updates,
  no source-domain selector, no target-null atom, no target oracle, and an
  independent replication-only HVD head.
- The plain aggregate law improved median constraint-mean rank from V15's
  approximately `0.43` to `0.983`, `0.966`, and `0.941` in FactorShock,
  Inventory, and Queue. Its median mean absolute errors were `0.060`, `0.172`,
  and `0.177`, versus V26's `0.209`, `0.227`, and `0.204`.
- The plain law recovered 13 oracle-mean-and-variance certifiable audit points
  in FactorShock. Scalar and directional inflation recovered none. Inflation
  slightly improved selected rank metrics but widened epistemic uncertainty
  enough to erase all recovered certification depth, so it remains a negative
  ablation rather than part of the promoted model.
- Actual theory certificates remained empty in every arm. This is not a false
  certificate failure: all arms had zero false certificates. It confirms the
  earlier finite-information diagnosis that certification at `N=20` needs
  either deeper safe points, repeated observations, or a sharper valid
  epistemic bound. Mean-coordinate repair alone cannot create safety depth.
- Median feasible regret was unchanged from the frozen source-informed initial
  design in all three domains. The diagnostic `sobol_new` backend produced only
  two strict improvements among 15 aggregate runs. V27 therefore promotes the
  transferable mean posterior, not a claim that online optimization is solved.

## V28 Constraint-Head Authority Separation

The V27 post-gate audit found a second posterior downstream of the promoted
aggregate GPR. Recommendation and Bayes-risk terminal values still called the
legacy finite-task robust hierarchy, allowing that hierarchy to replace the
constraint mean, epistemic variance, and cumulative aleatoric variance
together. The resulting certificate therefore counted source disagreement a
second time and violated the intended isolated-head model:

\[
  \underbrace{(m_g,s_g^2)}_{\text{aggregate target GPR}}
  \quad\text{and}\quad
  \underbrace{v_C^+}_{\text{cumulative HVD}}
\]

must be the only two inputs to

\[
  m_g(x)+\sqrt{\beta_g}s_g(x)
  +z_\alpha\sqrt{v_C^+(x)}\leq\tau.
\]

V28 makes this statistical authority explicit. `task_joint` is the exact V27
control. `split_gpr_task_hvd` takes mean and epistemic uncertainty only from
the single aggregate GPR while retaining the legacy robust task-HVD variance;
it isolates the duplicated mean head. `split_gpr_cumulative_hvd` additionally
takes aleatoric risk only from the direct provider-based cumulative HVD; it is
the fully separated model. In both split variants, caller-supplied legacy task
means and epistemic variances are ignored by construction. Joint-tangent
certification is unavailable because it would reintroduce the discarded joint
task posterior.

The same authority contract is used by candidate filtering, recommendation,
truth-pool audits, hard-certified terminal values, and posterior Bayes-risk
terminal values after every cloned posterior update. Legacy task-joint moments
remain visible only as counterfactual diagnostics. The gate fixes the V27
aggregate hyperlaw, frozen source archive, source-informed `n0=10`, Sobol
diagnostic backend, `d=1000`, `N=20`, and five target seeds. It varies only the
head authority over FactorShock, Inventory, and Queue. Promotion requires the
same 15/15 true-feasible performance, no false-certificate or adaptive-loss
regression, preserved mean accuracy, and a strict gain in actual certificate
coverage or oracle-substitution certifiability. Direct cumulative-HVD authority
is preferred only after those conditions hold.

### V28 Result: Correct Authority, Unsafe Mean Calibration

Run `scolh_mean_alignment_v28_authority_s5_20260719` completed all 45
checkpoint-free shards. Neither split-head challenger was promoted. The V27
`task_joint` control retained 15/15 truly feasible recommendations, zero
adaptive losses, and zero actual certificates. Both split variants returned
14/15 truly feasible recommendations and lost the same initially feasible
Queue seed 3 incumbent.

The two split variants identify different bottlenecks. Retaining the task HVD
head produced no actual certificates despite increasing oracle-substitution
certifiability from 13 to 36 points. Direct cumulative-HVD authority released
90 true audit-pool certificates but also 13 false certificates. No selected or
evaluated point was falsely certified; the errors occurred only in the fixed
post-run raw audit pool. Substituting oracle variance did not remove them,
whereas substituting oracle mean removed most of them. The direct-head failure
is therefore local constraint-mean overconfidence, not evidence that the
cumulative HVD variance is too small.

This is a useful negative result. V28 closes the implementation mismatch and
exposes the next statistical error instead of hiding it behind the legacy
task-joint variance inflation. V27 remains the promoted baseline.

## V29 Mean Calibration and Posterior Incumbent Preservation

V29 separates the two V28 failures. The existing empirical-Bayes
misspecification posterior is evaluated as `none`, `predictive_scale`, and
`predictive_scale_directional` under direct cumulative-HVD authority. A
sequential posterior-dominance rule independently protects the terminal
incumbent: a challenger replaces it only when the covariance-free Cantelli
lower bound for Bayes-risk improvement reaches `1-delta_switch`, with
`delta_switch=0.05`. Target truth is absent from initialization, updates, and
terminal selection.

The preregistered gate requires all 15 initially safe recommendations to
remain truly feasible, zero adaptive losses, zero false audit-pool
certificates, at least one true audit-pool certificate, preservation of mean
rank/MAE in at least two domains, and complete oracle-free/paired-data
contracts. Among passing calibrations, the gate prefers more true certificate
coverage and then lower model complexity.

Preflight run
`scolh_mean_alignment_v29_dominance_preflight_seed3_20260719` targeted the
known Queue seed 3 loss and the matching seed in FactorShock and Inventory.
All three terminal recommendations exactly retained their source-informed
`n0-best` true-feasible incumbent; Queue recovered from the V28 false-infeasible
terminal recommendation to true chance margin `-0.0243`. Each run performed
ten posterior updates, accepted zero unsupported switches, used the maintained
incumbent at termination, and recorded no target-oracle use. This establishes
mechanism viability only. The full 60-shard calibration gate remains the
promotion test.

### V29 Result: Incumbent Preservation Works, Calibration Does Not Close

Run `scolh_mean_alignment_v29_calibration_s5_20260719` completed all 60
result-only shards. No challenger was promoted. All three dominance variants
retained 15/15 truly feasible initial incumbents with zero adaptive losses,
whereas the V28 control retained 14/15. No dominance variant accepted an
unsupported online switch. This verifies the posterior switch mechanism, but
not the safety of its initial incumbent.

The mean-calibration arms remained miscalibrated in the fixed post-run raw
pool. Plain dominance retained 90 true and 13 false certificates. Scalar
predictive scaling reduced this to 40 true and eight false certificates;
directional scaling retained 55 true and nine false certificates. Because the
registered gate requires zero false certificates and nonzero true coverage,
none is admissible. V27 remains the promoted baseline.

## V30--V32 Finite-Sample Calibration Sentinels

V30 replaced plug-in scale inflation with finite-target inverse-chi-square and
finite-source-task Student-t upper scales. On the paired seeds 1--2 sentinel,
both upper-scale variants retained 6/6 feasible recommendations and reduced
raw-pool errors to three false certificates with 15 true certificates. The
remaining errors showed that a single global scale could not represent local
mean misspecification.

V31 added a posterior-only local HC3 sandwich covariance in coefficient space.
It eliminated all raw-pool false certificates on the same six paired cells and
retained one true certificate. However, each HC3 variant returned only 5/6
feasible recommendations. Queue seed 1 exposed a logically prior failure: the
Cantelli switch rule made no switch, but its initial Bayes-risk incumbent was
already truly infeasible even though the charged initial design contained one
true-feasible point. V31 therefore fixes certification precision but cannot be
promoted as a decision rule.

V32 tested whether the canonical theory certificate could initialize the
incumbent. Run
`scolh_mean_alignment_v32_safe_init_preflight_s1_20260719` completed all nine
paired shards. Every HC3 certificate set was empty. Lexicographically minimizing
the theory upper margin then selected truly infeasible initial incumbents in
both FactorShock and Queue; the two V32 variants retained only 1/3 feasible
recommendations, compared with 2/3 for the HC3 Bayes-risk control. This is an
expected impossibility boundary, not a coding failure: when the valid
certificate is empty, membership in a certified-safe set cannot define an
incumbent. V32 is a negative ablation and is not expanded.

## V34 Robust Empirical-Bayes Posterior

V34 combines the two empirically identified requirements in one posterior.
Charged target residuals first scale the frozen source hyperlaw before
conditioning, allowing the posterior mean to move away from an incompatible
source shrinkage direction. A local HC3 sandwich covariance is then added to
that same conditioned posterior, without changing its conditioned mean. Thus
mean adaptation and certification uncertainty use one target history and one
posterior rather than a terminal selection patch.

The paired gate compares scalar pre-conditioning alone, posterior-only HC3,
and the combined robust posterior with and without the finite-source-task
multiplier. All arms keep risk initialization so V32's invalid empty-certificate
fallback cannot contaminate the comparison. Promotion requires complete paired
execution, 15/15 true-feasible recommendations, zero adaptive losses, zero
false raw-pool certificates, at least one true raw-pool certificate, preserved
mean rank and MAE in at least two domains, and an oracle-free sequential update
contract. The simpler combined posterior wins ties.

### V34 Result: Correct Components, Incorrect Statistical Authority

Run `scolh_mean_alignment_v34_robust_eb_preflight_s1_20260719` completed all
12 paired shards. The combined posterior passed every implementation contract
and preserved mean rank relative to scalar scaling in at least two domains.
Nevertheless, both combined variants retained only 2/3 feasible terminal
recommendations and lost Queue seed 1. They eliminated all false raw-pool
certificates by also eliminating every true certificate. The scalar-scale
control retained 3/3 feasible recommendations and 16 true certificates, but
four Inventory certificates were false.

The paired mechanism is decisive. V34's central posterior mean is adequate;
putting the full HC3 coefficient covariance into the generative positive-part
loss changes Bayes ranking and selects the same unsafe Queue incumbent as V31.
An HC3 sandwich covariance estimates uncertainty of an M-estimator. Treating it
as additional generative constraint noise is therefore too conservative and
changes the wrong statistical object. V34 is not expanded.

## V35 Predictive/Confidence Covariance Separation

V35 retains one charged-data estimator but gives its two covariance views
their standard roles. The scaled source hyperlaw defines the central posterior
used by posterior expected-violation ranking. The same fit's HC3 sandwich
matrix defines a dominating robust confidence covariance used by the theory
certificate and by the covariance-free Cantelli switch bound. The posterior
mean, target history, HVD head, and candidate path remain shared.

This is not a fallback or a second learned mean head. It is the distinction
between a model-based covariance for prediction and a sandwich covariance for
coverage. The V35 preflight compares scalar scaling, V34 joint use, and
confidence-only HC3 with and without the finite-source-task multiplier on the
same source archive, initial design, domains, and seed. Expansion still
requires terminal feasibility and zero false certification; certificate
nonvacuity is assessed separately because Queue seed 1 is not certifiable even
after oracle mean-and-variance substitution at `N=20`.

### V35 Result: Authority Separation Is Sound but Vacuous

The final paired run
`scolh_mean_alignment_v35c_canonical_sobol_s5_20260719` completed all 30
result-only shards. It used a canonical Sobol continuation that is independent
of every fitted posterior. For every domain and seed, V29 and V35 therefore had
identical initial designs, identical ten-step online action sequences, and
identical noisy responses. Simulation noise and every proposal component used
independent deterministic random streams. This closes the earlier diagnostic
confound in which `sobol_new` ranked a model-dependent candidate pool.

V35 is not promoted. The confidence-only HC3 arm and scalar-scale control both
returned 14/15 truly feasible recommendations and both lost Queue seed 3. They
had identical mean MAE and rank because the robust covariance correctly had no
authority over posterior Bayes-risk ranking. Neither arm produced a true or
false certificate on the common 512-point raw audit pool. Oracle substitution
showed 20 certifiable FactorShock points under the control but only one under
the robust confidence covariance. The HC3 correction therefore preserved
soundness by making an already empty certificate still more conservative; it
did not create information.

This failure is structural. Any confidence covariance constrained to dominate
the scalar-scale covariance cannot make a certificate nonvacuous on a fixed
data set when the scalar-scale certificate is already empty. Tuning the HC3
multiplier would only trade coverage for apparent sharpness. The next test must
change the information collected, not weaken the registered confidence bound.

## V36 Evaluate-or-Replicate Information Gate

V36 treats a repeated evaluation as an ordinary acquisition action. At each
paid target step, the joint information backend compares the next canonical
Sobol point with observed candidates whose replication can reduce constraint-
mean or cumulative-HVD uncertainty. All actions have the same unit simulator
cost. No fixed recheck, truth-based rescue, or terminal empirical override is
allowed.

The diagnostic matrix keeps V35's split predictive/confidence posterior and
compares canonical Sobol continuation, joint-information new-point only, and
joint evaluate-or-replicate with four or eight replication candidates. The
one-seed preflight is only a mechanism test. A full promotion requires 15/15
true-feasible recommendations, zero adaptive losses, zero false certificates,
nonzero true certificate coverage, an actually selected replication action,
and improvement over the paired new-point control. V35's failed result is not
retroactively reclassified if V36 succeeds.

### V36--V37 Result: Replication Learns Variance but the Approximate VOI Lies

The paired V37 preflight completed all 12 result-only shards for one seed in
FactorShock, Inventory, and Queue. Both replication arms retained 3/3 truly
feasible recommendations with zero adaptive losses, but neither produced a
true or false posterior certificate. The approximate joint-VOI arms spent
20/30 online actions on replication: eight of ten actions in FactorShock and
six of ten in each of Inventory and Queue.

Replication did provide real statistical information. Median absolute
log-variance error fell from roughly `3.85--4.18` under new-point continuation
to `0.53--1.40` under clustered replication. It did not improve feasible
regret or certification. Collapsing duplicate policies into precision-weighted
HC3 clusters fixed the duplicate-row arithmetic, but left only 12--14 unique
target clusters for a roughly 13-dimensional mean head. Maximum leverage rose
to `0.98--1.00`, and the robust coefficient covariance expanded. In Queue,
even oracle mean and oracle aleatoric variance could not produce a certificate
under the fitted epistemic radius.

The V36 action value is therefore misspecified. Its rank-one covariance
formula treats the residual- and leverage-dependent HC3 covariance as if it
were an ordinary fixed Bayesian covariance. It predicts that a repeat shrinks
mean uncertainty without recomputing the loss of unique support, while the
actual post-action fit can increase the HC3 correction. V37 is not promoted.

## V38 Exact-Refit Evaluate-or-Replicate VOI

V38 keeps the same source archive, source-informed `n0=10`, V35 central mean,
clustered HC3 confidence view, cumulative HVD, terminal Bayes-risk decision,
and posterior-dominance incumbent. It changes only the action-value operator.
The active action set is exactly one canonical Sobol new point plus eligible
observed replication points. For each action and each common antithetic
predictive innovation, V38 clones and updates the same GPR/task posterior and
HVD used by the online algorithm. The resulting HC3 cluster count, leverage,
robust covariance, HVD posterior, and terminal decision are all recomputed.

The pre-registered gate compares V35 Sobol continuation, V37 approximate
replication, and V38 exact-refit actions with four or eight replication
candidates. The preflight restricts exact work to the active actions, uses one
common antithetic pair, and is bounded to 12 fork workers. A promoted full gate
must repeat the comparison with four common antithetic samples. Diagnostic eligibility
requires a paired oracle-free run, zero false certification, no feasibility
regression, and fewer replication actions than V37 with more new target
support. Promotion additionally requires complete five-seed execution, 15/15
true-feasible recommendations, zero adaptive losses, and either nonvacuous
certification or a strict adaptive improvement. A mere change in action mix is
not sufficient for promotion.

### V38 Result: Runtime Is Acceptable, Exact Refit Still Over-Replicates

The repaired one-seed sentinel completed on all three domains. The scheduler's
initial `2.3 h` estimate was inherited from the older unbatched exact-KG runtime
history and was not representative of V38. With active-action pruning, one
antithetic pair, and 12-way process-fork fantasy evaluation, measured wall
times were `14.1 min` for FactorShock, `21.3 min` for Inventory, and `17.9 min`
for Queue. Relative to the paired V37 runs, exact refitting added approximately
`3.9--6.1 min` per shard rather than two hours.

V38 is not promoted. All three recommendations remained truly feasible and no
false certificate was emitted, but certification remained vacuous and no run
improved its initial feasible incumbent. More importantly, exact refitting did
not correct the action mix: across the three domains V37 selected 20 repeats
and 10 new points, whereas V38 selected 23 repeats and only 7 new points. This
rules out the proposed explanation that HC3's approximate rank-one update was
the sole source of over-replication. The result points instead to the terminal
Bayes-risk/HVD information objective itself valuing variance calibration that
does not translate into boundary discovery or certification. The four-sample
and five-seed gates are therefore not launched.

## V39 Signed Common-Random Exact VOI

V38's failure does not yet identify a defect in posterior Bayes risk. Across
the three sentinel runs, `58%--74%` of active-action Monte Carlo gain estimates
were negative. V38 replaced every negative estimate by zero before ranking.
That transformation preserves nonnegativity but not the ordering of finite-MC
estimators: multiple actions become tied at zero, and the lowest candidate
index wins. Because previously observed replication candidates generally occur
earlier in the candidate list, clipping creates a systematic replication bias.
Inventory even selected a negative raw gain on average after clipping.

V39 keeps the terminal loss, fantasy posterior, action set, HVD update, and
common antithetic innovations unchanged. It ranks actions by their signed
estimate

`current terminal risk - expected post-action terminal risk`.

The current value is common to all actions, so this is exactly equivalent to
minimizing expected post-action terminal risk. Negative finite-MC values remain
valid for pairwise ranking instead of being collapsed into an order-dependent
tie. This is an estimator correction, not an exploration bonus, gate, or
replication penalty.

The one-seed gate compares canonical Sobol continuation, clipped V38 with two
antithetic samples, signed V39 with the same two samples, and signed V39 with
four samples. Diagnostic success requires paired initial data, oracle-free
actions, zero false certification, no feasibility regression, fewer repeats
and more new points than clipped V38. Promotion retains the earlier stronger
requirement of a nonvacuous certificate or strict adaptive improvement. If
signed ranking does not correct the action mix, the terminal loss itself rather
than its finite-MC ranking becomes the next object to replace.

### V39 Result: Signed Ranking Rejects the Clipping Hypothesis

All 12 one-seed shards completed. Signed MC2 and MC4 retained 3/3 truly
feasible recommendations and zero false certificates, but neither produced a
certificate or improved the initial incumbent. Clipped V38 selected 23 repeats
and 7 new points. Both signed variants selected 25 repeats and only 5 new
points. Increasing from two to four samples did not change the aggregate action
mix or median feasible regret. V39 is not promoted.

The new diagnostics also falsify the proposed order-tie mechanism. V38 selected
five nonpositive actions, but none was a replication action. Under signed MC4,
all nine selected nonpositive actions were repeats. Exact posterior refitting
therefore genuinely ranks replication above new support under the current
terminal loss; clipping merely hid part of that preference.

The mismatch is now explicit. Posterior Bayes risk integrates violation under
the central decision covariance, which repeated observations can reduce
quickly. The final theory certificate uses the robust HC3 mean covariance and
cumulative-HVD upper variance. Those robust quantities remained vacuous even
after repetition. The acquisition was paying for central certainty that the
terminal certificate could not use.

## V40 Robust-Certificate Lexicographic Terminal Value

V40 replaces that mismatched scalar terminal loss with the exact final decision
object

`(uncertified, positive robust theory margin, posterior objective)`.

The tuple is minimized lexicographically. When no point is certified, the
second component supplies a nonvacuous direction toward the robust chance
boundary. Once a certificate exists, objective quality becomes decisive. The
same function is used inside every GPR/HC3/HVD fantasy and for the final
observed-only recommendation. Posterior-dominance Bayes-risk switching is
disabled in this gate because it would reintroduce a different terminal loss;
no empirical override or forced replication is enabled.

The one-seed matrix compares canonical Sobol, signed Bayes-risk MC4, and the
robust lexicographic value with two or four common antithetic samples. A valid
V40 run must report a mathematically closed decision contract. Diagnostic
success requires fewer repeats and more new support than signed Bayes risk with
no feasibility or false-certification regression. Promotion still requires a
true certificate or strict adaptive improvement, not only a different action
sequence.

### V40 Result: Closing the Terminal Contract Does Not Fix Vacuity

All V40 sentinel contracts passed, so this was a valid test of a single robust
terminal object rather than a fallback heuristic. The V35 Sobol control and
signed V39 each retained `3/3` truly feasible recommendations. Robust V40 MC2
retained `2/3`, and MC4 retained only `1/3`. V39 selected 25 replications and
five new points; V40 MC2 made the same aggregate choice, while V40 MC4 moved
further toward replication with 27 repeats and three new points. No arm emitted
a posterior certificate or improved the initial incumbent. V40 is not
promoted.

The failure is more fundamental than terminal-value mismatch. In the Queue
sentinel, the selected policy's true chance margin was `-0.0243`, but the
posterior theory margin was `0.5383`. The epistemic radius was `0.1925` and the
cumulative-HVD aleatoric radius was `0.4200`. A post-run oracle audit then
showed that even oracle mean and oracle aleatoric variance produced essentially
no certifiable support under the target-only HC3 coefficient radius at
`N=20`. Therefore no acquisition objective can discover a nonempty certified
set while retaining that confidence law. The next intervention must change
the transferable coefficient confidence model, not the terminal ranking or
the HVD variance head.

## V41 Source-Conditioned Low-Rank Confidence Sequence

V41 separates three uncertainty objects that V35's robust covariance had
combined:

1. Source-conditioned coefficient uncertainty is represented by the ordinary
   low-rank posterior covariance `C_t` before HC3 inflation.
2. Adaptive finite-feature uncertainty uses the determinant-ratio information
   term `I_t = 0.5 (log det C_0 - log det C_t)`.
3. Source-target misspecification remains a separate Student-t residual guard;
   it can only increase the radius and is never absorbed into the HVD
   aleatoric term or a solution-specific deviation.

For candidate feature `phi(x)`, the self-normalized confidence radius is

`sqrt(beta_t * phi(x)^T C_t phi(x)) + t_delta * sqrt(r_transfer)`,

where

`beta_t = max(beta_g, 2 (log(1/delta) + I_t))`.

The implementation returns the exact equivalent variance obtained by squaring
this radius and dividing by `beta_g`, so every existing theory-certification
call continues to evaluate

`mu_g + sqrt(beta_g) s_g + z_alpha sqrt(v_C_plus) <= tau`

without changing its public API. Replication ranking continues to use only the
central, reducible decision covariance; an irreducible transfer guard cannot
make a repeated point look artificially informative. Hierarchical online
refits rebuild the confidence state after every charged target observation,
and exact-fantasy clones keep independent copies.

The pre-registered gate uses the same frozen source archive, source-informed
`n0=10`, canonical Sobol continuation, common target responses, cumulative
HVD, and final decision rule in all arms. It compares the V35 model covariance,
a one-sided source-Bayes radius, and the adaptive source self-normalized radius
over FactorShock, Inventory, and Queue with five common seeds. Promotion is
reserved for the self-normalized arm and requires `15/15` true-feasible final
recommendations, zero adaptive losses or false certificates, a non-wider
best-feasible epistemic radius than V35, and either nonvacuous certification or
a strict adaptive improvement. A narrower but still vacuous confidence band is
diagnostic evidence only.

### V41 Result: Narrower Bayes Radius, Still No Certifiable Support

All 45 paired runs completed. The source-Bayes arm reduced the median
best-feasible epistemic radius from `0.0865` to `0.0747`, while the
self-normalized arm increased it to `0.1955`. All three arms nevertheless
ended with the same `14/15` true-feasible recommendations, one adaptive loss,
zero adaptive improvements, and zero posterior certificates. V41 is therefore
not promoted.

The determinant sequence was not the only source of vacuity. In the Bayes arm,
the minimum best-feasible epistemic radii remained approximately `0.0430`,
`0.0465`, and `0.0411` in FactorShock, Inventory, and Queue. The corresponding
oracle-mean/oracle-variance margins remained positive. Inspection of the
frozen source hyperlaw revealed the structural reason: each source-domain
coefficient covariance was added directly to the target-task covariance. That
conflates uncertainty in estimating a source coefficient with genuine
variation of a newly drawn target task.

With two source domains the identifiable between-domain discrepancy has rank
at most one, but the old single-Gaussian law had effective rank `4.68` in the
Queue audit. Its prior covariance trace was `0.578`, even though its
decision-covariance trace after target conditioning was `0.239`. More
confidence tuning cannot distinguish estimation error from task variation.

## V42 Shared-Mean Plus Low-Rank Domain Discrepancy

V42 replaces the source-to-target law, not the decision backend. For source
domains `s=1,...,S`, let `theta_s` denote the exchangeable coefficient estimate
and `C_s` its source-fit covariance. The target coefficient law is

`theta_target = theta_shared + U z_target + epsilon_role`,

where the uncertainty in `theta_shared` is

`C_shared = sum_s w_s^2 C_s`,

the domain discrepancy is the reliability-weighted between-source covariance

`U U^T = sum_s w_s (theta_s-theta_shared)(theta_s-theta_shared)^T`,

and therefore has rank at most `S-1`. Observable channel-role covariance is
retained as a separate exchangeable block. In particular, `C_s` contributes to
uncertainty in the shared mean only; it is not reinterpreted as target-task
variation.

The gate has four paired arms:

1. V35 model confidence with the legacy single-Gaussian hyperlaw.
2. V41 source-Bayes confidence with the legacy hyperlaw.
3. V42 low-rank hyperlaw with model confidence.
4. V42 low-rank hyperlaw with source-Bayes confidence.

All arms use the same frozen source archive, source-informed `n0=10`, canonical
Sobol continuation, target seeds, simulator responses, cumulative HVD, and
observed-only terminal rule. This isolates the hyperlaw from acquisition and
candidate-generation effects. The V42 analyzer rejects any run that uses
target oracle information, treats source estimation covariance as target
variation, loses channel-role covariance, or reports discrepancy rank greater
than `S-1`.

The main challenger is the low-rank source-Bayes arm. Promotion requires
`15/15` true-feasible recommendations, zero adaptive losses and false
certificates, a nonvacuous certificate or strict adaptive improvement, and a
strict outcome or certification improvement over both V35 and V41. Merely
shrinking a covariance trace is diagnostic evidence and cannot trigger
promotion.

### V42 Result: Population Low Rank Is Not a New-Task Predictive Law

All 60 paired runs completed. V35 and V41 each retained `14/15` truly
feasible recommendations with one adaptive loss. The V42 model- and
source-Bayes arms retained only `13/15`, with two losses. All arms produced
zero posterior certificates and zero adaptive improvements. The low-rank
factor reduced the median prior covariance trace from `0.578` to `0.511`, but
that contraction was not a valid performance gain and V42 is not promoted.

The paired Queue audit localized both losses to initial incumbent selection,
not an online switch. Queue seed 1 was feasible under V41 but V42 selected an
unsafe all-55 policy; Queue seed 3 remained unsafe under every arm. The V42
central law became more confident around a misspecified shared coefficient.
This reveals a finite-source distinction: the weighted between-source matrix
is a population-covariance estimator, not the predictive covariance of a new
held-out domain.

## V43 Finite-Source Predictive Low-Rank Hyperlaw

Let normalized source reliabilities be `w_s` and define

`c = sum_s w_s^2`.

For exchangeable source-task coefficients with population covariance `Sigma`,
the weighted between-source matrix has expectation `(1-c) Sigma`. A new target
coefficient differs from the weighted source mean with covariance
`(1+c) Sigma`. V43 therefore replaces only the discrepancy block by

`B_predictive = ((1+c)/(1-c)) B_population`.

The correction is available only when at least two source domains provide
`c < 1`; it is never silently capped. It preserves the same discrepancy
factor rank at most `S-1`, leaves source-estimation covariance in the shared
mean block, retains channel-role covariance, and uses no target label or
oracle quantity.

The first gate is deliberately limited to the two observed Queue failures,
seeds 1 and 3. Four paired arms compare V41 source-Bayes, V42 population
low-rank source-Bayes, V43 predictive low-rank model confidence, and V43
predictive low-rank source-Bayes. All use the same frozen archive,
source-informed `n0=10`, canonical Sobol actions, common simulator responses,
cumulative HVD, and observed-only terminal decision. Promotion requires the
V43 source-Bayes arm to recover `2/2` true-feasible recommendations with zero
adaptive losses and false certificates, satisfy the exact finite-source
multiplier and rank contracts, and strictly improve both V41 and V42. Only a
passing sentinel can expand to the full three-domain, five-seed gate.

### V43 Result: Correct Predictive Scale, Unchanged Wrong Direction

All eight Queue sentinel runs completed and every multiplier, rank,
pairing, and oracle-exclusion contract passed. V41 retained one of two
truly feasible incumbents. V42 and both V43 arms retained zero of two,
with two adaptive losses and no posterior certificates or improvements.
V43 is not promoted and the full gate is not launched.

The observed source-weight concentration implied only `1.203` effective
source domains and a predictive multiplier of `10.86`, not the equal-source
value three. The between-domain trace increased from `0.111` to `1.204`, and
the full prior trace increased from `0.511` to `1.605`. After ten charged
target observations, however, the central decision trace returned to about
`0.243`; both V43 recommendations exactly matched V42. Scaling the rank-one
source discrepancy therefore did not add a missing target direction.

The post-run pool audit showed that the coordinate still ranked Queue's
constraint mean well (`0.928--0.953` Spearman correlation), but its median
absolute error was `0.144--0.146`, versus `0.073--0.098` for V41. Each seed
contained seven or eight truly feasible pool points but zero posterior
certificates. The remaining mechanism is not HVD variance scale. It is the
interaction between constraint-mean calibration and the terminal constrained
decision loss.

## V44 Terminal Bayes-Risk Penalty Diagnostic

The current observed-only terminal action minimizes

`E[f(x) | D] + rho E[(G(x))_+ | D]`

with the fixed dimensionless value `rho=5`. V44 asks only whether this
arbitrary loss tradeoff explains the Queue failures. It freezes the better V41
source-Bayes mean law, cumulative HVD, source archive, source-informed initial
design, canonical Sobol actions, common simulator responses, and every target
seed. The paired arms use `rho` in `{5,20,80}` on Queue seeds 1 and 3.

This is a mechanism diagnostic, not a hyperparameter search eligible for
promotion. Even if a larger fixed value recovers both seeds, it cannot enter
the method or main table. Such a result only authorizes the next principled
step: learn a normalized dual-penalty distribution from source tasks, freeze
it before seeing the held-out target, and update it only with charged target
observations. If no larger penalty helps, the terminal loss is exonerated and
the next intervention must enrich the source-task discrepancy support.

### V44 Result: Terminal Penalty Exonerated

All six paired Queue runs completed. Penalties `rho=5`, `20`, and `80`
produced exactly the same recommendations, true feasibility (`1/2`), one
adaptive loss, zero adaptive improvements, zero certificates, and the same
conditional median feasible regret. A source-learned dual is therefore not
warranted. Together with V38--V40, this rules out the local terminal estimator,
negative-gain clipping, lexicographic terminal tuple, and scalar risk penalty
as the cause of the failed recommendations.

## V45 Fixed-Budget Source-Task Episodes

The remaining identifiable limitation is source discrepancy support. Under
LODO, Queue has only two source tasks, FactorShock and Inventory. Any
between-task coefficient factor then has rank at most one. V43 changed the
scale of that direction but could not create a direction absent from the two
source coefficients.

V45 redistributes the same offline simulator budget across four independently
drawn task episodes per base source domain. Every episode is generated before
the held-out target is observed from one frozen, domain-independent context
law: three normalized boundary-center shifts and one log boundary-radius
shift. The same abstract context is implemented by FactorShock, Inventory,
and Queue through their ordinary observable state summaries. It never reads a
target name, target observation, target oracle, or target-specific anchor.

Budget equality is exact. The old design uses
`2 base domains * 64 records * 3 replicates = 384` simulator calls. V45 uses
`2 * 4 episodes * 16 records * 3 replicates = 384` calls. The source-informed
target initial design, all ten target continuation actions, simulator
responses, HVD law, and Sobol backend remain paired. Hence V45 tests task
diversity rather than more source data or a different online optimizer.

The V41 baseline retains the ordinary exact archive-fingerprint contract.
Episode arms explicitly use `paired_frozen_control`: their posterior-training
archive changes by construction, while the historical source-only proposal is
frozen so that all target initial points remain byte-identical. This exception
is not an algorithm default. It is accepted only for a cost-matched,
target-free mechanism intervention, records both archive fingerprints, and is
rejected unless every arm shares the same frozen proposal and paired target
action/response trace.

The Queue seeds 1 and 3 sentinel contains four preregistered arms:

1. V41 with two nominal source tasks.
2. Eight episode labels with no geometry perturbation, the negative control.
3. Eight geometry episodes with the V41 source-Bayes coefficient law, the
   primary challenger.
4. The same geometry episodes with V43 finite-source predictive low-rank
   scaling, a mechanism ablation.

All nuisance-only sigma, alpha, and scalarization-weight jitters are zero.
Promotion requires the primary challenger to retain both truly feasible
incumbents, incur no adaptive loss or false certificate, and strictly improve
both the two-task baseline and the label-only control. Every arm must report
the same 384 source calls, frozen source episode fingerprints across target
seeds, identical target actions/responses, and no target-oracle use. Only a
passing sentinel expands to three domains and five seeds.

The theorem layer deliberately claims only:

- equal episode allocation preserves the source simulator budget;
- increasing episode count cannot reduce the maximum `S-1` discrepancy-factor
  capacity;
- the realized discrepancy remains a nonnegative finite sum of squared
  factor projections.

It does not claim that more episode labels automatically attain higher rank.
That mechanism remains subject to the empirical rank/calibration audit and
the strict Queue sentinel.

### V45 Result: Better Mean Direction, Inflated Task Uncertainty

All eight Queue sentinel runs completed and passed the fixed 384-call source
budget, frozen-proposal, paired-target-action, target-oracle exclusion, and
episode-geometry contracts. V41 retained one of two truly feasible
recommendations with one adaptive loss. The label-only episode control retained
zero of two, with two losses. Both geometry arms retained one of two, with one
loss. No arm produced a posterior certificate or adaptive improvement. V45 is
not promoted.

The geometry intervention nevertheless identified a useful direction. On
Queue seed 3, the raw constraint-mean error fell from about `0.0983` under V41
to `0.0463` under the geometry population law and `0.0406` under the geometry
predictive law. The failure came from covariance, not mean fit. The prior trace
grew from `0.578` to `1.232` and `1.651`, while the minimum epistemic radius at
a truly feasible pool point grew from about `0.0411` to `0.0449--0.0453`.
Every pool still contained seven or eight truly feasible points but zero
posterior-certified points. Even replacing posterior mean and aleatoric
variance by their oracle values left certification empty because the
epistemic radius dominated.

The cause is hierarchical: four 16-record fits from one base domain were
treated as four independent source domains. Their short-episode estimation
error was therefore reinterpreted as target-task variation. V45 showed that
source task geometry carries useful chance-boundary information, but its flat
hyperlaw made the terminal Bayes-risk objective overvalue uncertainty
reduction and, consequently, repeated evaluation.

## V46 Grouped Within-Base Source-Task Hyperlaw

V46 keeps V45's fixed source contexts but changes their probability model.
For base domain `b` and source episode `e`, let `theta_be` be the fitted
coefficient and `C_be` its fit covariance. It first forms

`theta_b = (1/m_b) sum_e theta_be`

and separates four covariance blocks:

`C_shared = (1/B^2) sum_b (1/m_b^2) sum_e C_be`,

`C_between = (1/B) sum_b (theta_b-theta_bar)(theta_b-theta_bar)^T`,

`C_within = (1/B) sum_b (1/m_b) sum_e
  (theta_be-theta_b)(theta_be-theta_b)^T`,

plus the exchangeable channel-role block learned inside each episode. Thus
source-fit uncertainty contracts only the shared base mean. Centered episode
contrasts estimate within-base task variation and are invariant to any common
coefficient offset within a base domain. The population target covariance is

`C_shared + C_role + C_between + C_within`.

The predictive ablation multiplies the between-base block by
`(B+1)/(B-1)` and each within-base block by `(m_b+1)/(m_b-1)`. Both variants
use PSD projection only after the semantically distinct blocks are formed.
With two base domains and four episodes each, the between-base discrepancy
rank is at most one, the within-base rank is at most six, and their combined
capacity is at most seven.

The first gate again uses Queue seeds 1 and 3:

1. V41 two-task source-Bayes control.
2. V45 flat geometry-episode control.
3. V46 grouped population hyperlaw, the preregistered challenger.
4. V46 grouped finite-task predictive hyperlaw.

All arms use 384 source simulator calls and the same frozen source-informed
proposal, target initial points, Sobol continuation actions, simulator
responses, cumulative HVD, and terminal rule. Promotion requires the grouped
population arm to recover both truly feasible recommendations, incur no
adaptive loss or false certificate, pass all grouping/rank/oracle contracts,
and strictly improve both controls. The predictive arm is diagnostic and
cannot replace the preregistered population challenger after seeing results.

### V46 Result: Correct Grouping, Uncorrected Fit Noise

All eight paired Queue runs passed the fixed-source-budget, grouping, rank,
target-oracle exclusion, frozen-proposal, and target action/response contracts.
The four arms nevertheless had the same outcome: one of two truly feasible
recommendations, one adaptive loss, no adaptive improvement, and no posterior
certificate. V46 is not promoted.

Grouping did not contract uncertainty. The V41, V45, V46 population, and V46
predictive prior traces were respectively `0.578`, `1.232`, `3.040`, and
`5.311`; their median epistemic radii at the best truly feasible pool point
were `0.0774`, `0.0715`, `0.0914`, and `0.0918`. In V46 population, the shared
mean, between-base, and within-base traces were `0.225`, `0.536`, and `1.797`.
The dominant within-base term still contains the sampling covariance of each
short 16-record coefficient fit. Group labels stopped treating episodes as
independent domains, but centered episode contrasts alone did not distinguish
latent task variation from coefficient-estimation noise.

## V47 Random-Effects Noise Deconvolution

V47 keeps V46 unchanged as a control and corrects the three random-effect
blocks before PSD projection. For `m_b` episode estimates in base domain `b`,
with fit covariances `C_be`, the expected fit-noise contribution to the
divisor-`m_b` centered covariance is

`C_within,noise,b = ((m_b-1)/m_b^2) sum_e C_be`.

For `B` fitted base means with covariances `C_b`, the corresponding term is

`C_between,noise = ((B-1)/B^2) sum_b C_b`.

The exchangeable channel-role contrast subtracts `(c-1)/c` times its fitted
block covariance for a source with `c` channels. Each corrected block is
`Proj_PSD(C_observed-C_noise)`. Source-fit covariance still contributes
positively to uncertainty in the shared coefficient mean; its subtractive use
here only prevents the same fit noise from being counted again as transferable
random-effect variation.

The Queue seeds 1 and 3 gate contains V41, V45, and V46 controls plus V47
population and predictive variants. All five arms retain the same 384 source
calls, frozen proposal, target points, Sobol actions, and simulator responses.
The population variant is the only preregistered challenger. Promotion
requires every deconvolution trace to be finite and nonzero where expected,
each corrected trace not to exceed its observed counterpart, a strict
epistemic-radius contraction relative to V46, two of two truly feasible final
recommendations, no adaptive loss or false certificate, and strict outcome
improvement over all three controls.

### V47 Result: Variance Contracts, Terminal Decision Does Not Move

All ten paired Queue shards passed the source-budget, episode, deconvolution,
rank, oracle-exclusion, frozen-proposal, and target action/response contracts.
V47 population reduced the V46 prior trace from `3.040` to `2.622` and the
median best-feasible epistemic radius from `0.09142` to `0.09099`. The
channel-role, between-base, and within-base corrected traces were `0.253`,
`0.517`, and `1.626`, versus observed traces `0.482`, `0.536`, and `1.797`.
Separate PSD projection explains why corrected matrix traces need not equal
the scalar difference between observed and propagated-noise traces.

The correction did not change a single terminal outcome. V41, V45, V46, V47
population, and V47 predictive each returned one of two truly feasible
recommendations, one adaptive loss, no adaptive improvement, and no posterior
certificate. V47 is not promoted.

The paired seed audit localized the loss to Queue seed 3. Its `n0=10` design
contained one truly feasible point, but the certified set was empty. Risk
initialization selected an uncertified incumbent, and the subsequent Cantelli
rule made zero switches. The terminal recommendation therefore protected an
initial model error; deconvolving the source covariance could not change the
decision. This also exposes a semantic gap: a Cantelli switch bound controls
switch error conditional on the incumbent, but does not make an uncertified
initial incumbent safe.

## V48 Certified-Only Incumbent Preservation

V48 closes that gap without weakening the certificate. A posterior-dominance
incumbent is created only from the canonical certified set. If the set is
empty, the state remains explicitly uninitialized and the final decision
falls through to the same posterior Bayes action over all charged target
evaluations. Later charged observations may initialize preservation if and
only if they create a certificate. No empirical margin, target truth, or
problem-specific fallback enters this rule.

Every terminal decision now also emits a post-decision audit. Posterior risks
and their ranking are frozen first; only then may truth diagnostics join the
selected and counterfactual Bayes actions. Point identities are stored as
compact fingerprints, and all truth fields are marked inadmissible as decision
inputs.

The Queue seeds 1 and 3 sentinel compares five paired arms: promoted V27 with
no preservation, V27 with certified-only preservation, V47 with legacy risk
preservation, V47 with no preservation, and V47 with certified-only
preservation. The two certified-only arms must exactly match their
no-preservation counterparts whenever every certificate set is empty. The V47
challenger must remove the seed-3 adaptive loss and cannot regress relative to
promoted V27. A code-level contract improvement alone does not establish a
new performance baseline; push eligibility still requires a strict empirical
improvement over V27.

### V48 Result: Incumbent Preservation Was Not the Active Failure

All ten Queue sentinel shards completed normally. Whenever the certified set
was empty, `certified_only` preservation exactly matched the corresponding
no-preservation arm, so the option-valued initializer contract worked as
intended. It did not improve the outcome: every arm returned one of two truly
feasible recommendations, incurred one adaptive loss, produced no adaptive
improvement, and certified no point.

The shared failure is Queue seed 3. Every arm selected the same observed point
with true chance margin `0.0189237`; its posterior theory margin remained
between `0.585` and `0.854`. Queue seed 1 selected the same genuinely safe
point with true margin `-0.0243022`. Thus V48 falsifies incumbent preservation
as the active cause. With an empty certificate set, the ordinary terminal
posterior Bayes-risk ranking itself prefers the unsafe seed-3 point. V48 is not
promoted.

## V49 Central-HVD Decision Loss Gate

V49 separates two quantities that the legacy decision path conflated. Theory
certification always keeps the conservative cumulative-HVD upper variance
`v_C_plus`. Bayes actions may instead use the posterior-central cumulative-HVD
variance, which removes tail and model guards that are useful for guarantees
but are not aleatoric posterior means. Independently, the violation loss may be
either the legacy expected positive part or the expected binary chance-failure
loss `P(G > 0)`.

The sentinel is a two-by-two causal design under both static Sobol continuation
and exact evaluate-or-replicate VOI:

- certification-upper versus posterior-central decision variance;
- positive-part versus failure-probability violation loss;
- FactorShock seed 0, Inventory seed 0, and the known Queue seed-3 loss.

Static arms isolate terminal ranking from acquisition. Exact arms test whether
the same loss reduces the previous excessive replication preference. A full
gate is warranted only if the combined central/probability variant reduces
repeats, increases new-point evaluations, does not regress true feasibility,
and improves the paired static terminal decision. The sentinel itself cannot
promote a baseline.

### V49 Result: Replication Mix Moves, Static Ranking Does Not

The corrected run completed all 24 shards with the fixed 384-call source
archive and every source, mode, action-pairing, and oracle-exclusion contract
passing. The first submission used `96` instead of the frozen `64` source
records per domain; exact archive hashing rejected all shards before target
evaluation. That submission is an infrastructure failure and is excluded.

Under exact evaluate-or-replicate VOI, the legacy upper/positive arm selected
21 replications and 9 new points across the three sentinels. The
central/probability arm selected 19 replications and 11 new points. Both made
three of three truly feasible recommendations, incurred no adaptive loss or
false certificate, and had median feasible regret `0.00825`. Neither improved
over the initial best target point.

The static arms falsified the proposed terminal repair. Upper/positive,
central/positive, and central/probability all retained the Queue seed-3 loss;
upper/probability additionally lost Inventory. In Queue, the true feasible
point remained posterior rank 5. Under the certification upper variance every
candidate had posterior failure probability `1.0`; the binary loss therefore
collapsed to objective ranking. Central variance reduced the selected point's
failure probability to about `0.87`, but did not correct the order. V49 is not
promoted and does not warrant its preregistered full gate.

## V50 Posterior-Nominal Bayes Gate

V49 exposed one remaining conflation. The terminal code called its objective a
Bayes risk while using a KL-robust expectation over source experts. V50 keeps
the KL-robust chance margin exclusively for certification and compares it with
the posterior-nominal expert mixture exclusively for decision. Algebraically,
the robust decision risk is the nominal risk plus a nonnegative ambiguity
premium; removing that premium cannot alter the certificate.

The three-domain sentinel fixes posterior-central HVD and crosses:

- KL-robust versus posterior-nominal decision aggregation;
- positive-part versus failure-probability loss;
- static Sobol versus exact evaluate-or-replicate actions.

The primary challenger is posterior-nominal positive-part loss. A larger gate
is warranted only if paired static actions and observations remain identical,
all robust loss components dominate their nominal counterparts, the true-safe
posterior rank or outcome improves under the static challenger, and the exact
challenger does not regress safety while improving either action mix or final
outcome. Sentinel success still cannot promote a baseline.

### V50 Result: Expert Ambiguity Is Not the Active Cause

All 24 shards completed with the fixed 384-call source archive. Source,
decision-mode, static-pairing, oracle-exclusion, and robust-greater-than-nominal
loss contracts all passed. None of the four static decision combinations moved
an outcome: FactorShock and Inventory remained feasible, Queue seed 3 remained
an adaptive loss, and its truly feasible point remained posterior rank 5.

All four exact arms returned three of three feasible recommendations with no
adaptive loss, adaptive improvement, false certificate, or regret difference.
Positive-part arms selected 20 replications and 10 new points; probability-loss
arms selected 19 and 11. Posterior-nominal and KL-robust aggregation produced
the same action sequences within each loss family. V50 is not promoted and a
larger ambiguity gate is not warranted.

The action traces reveal a different asymmetry. Each exact step compared four
eligible replications with only one canonical Sobol new point. New actions,
when selected, had larger median exact gains than replications in every domain;
they were simply absent from most finite action sets. Thus the observed repeat
rate is not evidence that HVD information intrinsically dominates exploration.
It may be an artifact of an impoverished one-new-point discretization.

## V51 Balanced Evaluate-or-Replicate Action Set

V51 retains posterior-central, posterior-nominal positive-part Bayes risk and
changes only the finite action approximation. The new action set contains the
canonical Sobol continuation plus the lowest posterior-risk unobserved points
from the already generated candidate pool. Exact fantasy refits, rather than
the shortlist score, still make the final evaluate-or-replicate decision. No
target truth enters either stage.

The three preregistered variants are four new actions plus eligible
replications, eight new actions plus eligible replications, and eight new
actions with replication disabled. They run on the same three sentinels and
source archive as V50. Every iteration records all active-arm exact gains,
new/replication labels, and the best-new minus best-replication gain. A larger
gate requires a strict final-outcome or regret improvement over V50 exact
posterior-nominal positive-part with no safety regression; merely selecting
fewer replications is diagnostic evidence, not promotion evidence.

### V51 Sentinel Result: The One-New-Point Action Set Was the Bottleneck

All nine shards passed the fixed-source, decision-mode, oracle-exclusion, and
complete active-arm logging contracts. Enlarging the action set changed the
decision for the intended reason. With four new actions, best-new exact VOI
exceeded best-replication VOI in 16 of 30 online rounds; the median signed gap
was `0.00427`. The selected mix moved from V50's 10 new and 20 replication
actions to 16 new and 14 replication actions.

Balanced-four kept all three final recommendations truly feasible, incurred no
adaptive loss or false certificate, and improved over `n0` on both Inventory
and Queue. Inventory regret changed from `0.011557` to `0.010904`; Queue changed
from `0.0028765` to `0.0023553`; FactorShock remained `0.00825`. This is the
first recent challenger to produce cross-domain adaptive improvements rather
than merely preserve the initial proposal.

Eight balanced new actions and eight-new-only also improved Queue, but worsened
Inventory regret to `0.03231` and `0.02688`, respectively. They are rejected:
more exploration is not monotonically better in realized regret even though
the best available finite-set VOI is monotone. Balanced-four alone advances to
the five-seed gate. It is not yet promoted.

### V51 Full Result: Balanced-Four Is Promoted

The five-seed gate compared balanced-four with both promoted V27 and V50's
exact canonical one-new-action control under paired source archives, initial
designs, target seeds, and target budgets. Balanced-four was 7/0/8 in paired
win/loss/tie outcome against V27 and 5/1/9 against exact canonical. It retained
15/15 true feasibility, removed V27's Queue adaptive loss, and generated five
real improvements over `n0-best`. The gate therefore advanced to 20 seeds.

The original frozen proposal contained only seeds 0--4. Before extending the
gate, the same 384-call oracle-free source archives were used to materialize
seeds 0--19. For every domain, the new file matched the old archive
fingerprint, reproduced seeds 0--4 point-for-point, and recorded no target
labels, target oracle, or source oracle assistance. The first invalid
submission that referenced missing seeds was cancelled and contributes no
experimental row.

Across the final 60 paired domain/seed cases, balanced-four returned 60/60
truly feasible recommendations, zero adaptive losses, zero false certificates,
and 19 improvements over the initial best target point. It selected 311 new
points and 289 replications. Against V27 it achieved 26 wins, 1 loss, and 33
ties; against exact canonical it achieved 20 wins, 2 losses, and 38 ties.
Every domain passed safety, median-regret, and paired noninferiority checks.

V51 `balanced4` is therefore the new promoted baseline. This result changes
the interpretation of the earlier repeat pathology: exact HVD-aware VOI was
not intrinsically overvaluing replication. The finite acquisition problem
gave it four meaningful repeat actions but only one new action. Once the two
action classes received a balanced finite approximation, a single Bayes-risk
objective produced both safe replication and productive exploration without a
manual gate.

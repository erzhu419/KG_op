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

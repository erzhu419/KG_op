# Unified Task-Latent Meta Model

> Retrospective status (2026-07-15): references below to the "promoted V32"
> describe the historical 2026-07-12 checkpoint. V32 was later classified as a
> privileged source-oracle upper bound. The current promoted policy is the
> oracle-free source-consensus successor recorded in `lodo_current.json`.

## Purpose

The current system transfers useful structures, but alignment, basis support,
cumulative HVD, sensitivity, and the terminal expert race are not yet inferred
as one task-level object.  The paper-grade invariant is:

> One latent task state must jointly explain what is good, what is safe, and
> how risk accumulates.

For a held-out domain `d`, write

```text
z_d = (R_d, S_d, theta_v,d, c_d)
```

where `R` is risk-coordinate alignment, `S` is basis/frequency/additive
support, `theta_v=(Lambda,B,omega,floor)` is cumulative HVD structure, and `c`
is false-feasibility sensitivity.  Source domains learn `Pi(z)`.  Only charged
held-out observations may update `Q_t(z)`.

## Non-Negotiable Constraints

- No target domain name, oracle value, optimum, hidden boundary, or uncharged
  simulator call enters inference.
- Representation, GPR, HVD, certification, proposal, and exact KG must consume
  the same posterior state rather than independent tuned gates.
- Sensitivity can alter Bayes decision loss, but never relax the theory
  certificate.
- Queue is an untouched held-out falsification domain for V32.  No Queue result
  may be used to tune the frozen Gate-2 run.
- A challenger changes decisions only after a shadow audit demonstrates stable
  cross-domain calibration.

## Stage 0: Frozen Queue Gate 2

The then-promoted V32 configuration was frozen in the historical run snapshot
below. It must not be reconstructed from the later `lodo_current.json`, which
now records the oracle-free successor. The only experimental change in the
historical test was `heldout=QueueResourceControl`, leaving FactorShock and
Inventory as source domains.

Run ID:

```text
lodo_v32_queue_n20_gate2_frozen_20260712
```

Scheduler tasks `t29249` through `t29255` use one seed per task, only
`node001-node006`, `N=20`, and seeds `0..6`.

Status: completed. Queue reached `3/7` true feasibility with zero false
certificates, median feasible regret `0.00455`, mean violation `0.05767`, and
median true margin `+0.01829`. All seven terminal pools contained a true
feasible action, while none contained a posterior-certified action. The result
localizes the cross-domain gap to model/decision calibration and rejects a
Queue-specific candidate-coverage patch.

## Stage 1: Shadow Joint Posterior

Implemented in `representation/task_posterior.py` as
`FiniteTaskLatentPosterior`.

- A finite structural expert bundles alignment/basis, GPR, and cumulative HVD.
- The joint state is `(structural_expert, sensitivity_class)`.
- The source prior starts as the product of the two source marginals.
- The shadow sensitivity marginal always uses the source-trained
  `stable/balanced/sensitive` prior, even when the frozen V32 decision path uses
  its legacy one-class fixed penalty.  This separation makes coupling
  identifiable without changing baseline behavior.
- A proper prequential Gaussian and chance-boundary score updates every joint
  hypothesis from charged observations.
- Posterior mutual information measures whether target evidence learns a real
  structure-sensitivity compatibility relation.
- The shadow posterior is cloned and updated inside exact posterior-update KG,
  but it is not used for selection or certification in this stage.

This is a finite approximation to the full `Q_t(R,S,theta_v,c)`.  It removes
the conceptual separation between structure and sensitivity without silently
changing the promoted baseline.

Three-domain replay run
`lodo_joint_shadow_v1_n20_3domain_20260712` contains 21 one-seed scheduler
tasks (`t29259..t29279`) on `node001-node006`. It must reproduce the frozen V32
recommendations before its diagnostics can justify an authoritative
challenger.

Status: completed. All 21 recommendations and all truth metrics are identical
to frozen V32 at machine precision. Median structure-sensitivity mutual
information is approximately `0.00003` on FactorShock, `0.0786` on Inventory,
and `0.1473` on Queue. The joint and legacy robust reference decisions agree
on `6/7`, `3/7`, and `0/7` seeds respectively. Provider-backed cumulative HVD
mass remains high (`1.000`, `0.982`, `0.959` medians). Thus the joint state is
inactive when FactorShock is structurally unambiguous and becomes relevant on
the two transfer-sensitive domains. This is evidence of a missing meta layer,
not yet evidence that its decisions improve optimization quality.

The same audit also exposes the remaining failure mechanism: median
expert-feasible mass at the selected action is zero in all three domains. On
Queue, every terminal pool contains a true-feasible action, but every expert
still assigns positive theory margin. The authoritative challenger must
therefore improve posterior Bayes ranking when no point is certified; it may
not solve the problem by relaxing the certificate.

## Meta-Coherence Audit

Every task-posterior result now exposes `task_meta_coherence` on the terminal
candidate universe.  The audit reports:

- posterior mass by structural family and risk coordinate;
- mass supporting the algorithm's selected candidate;
- posterior mass that certifies that candidate;
- chance-margin sign agreement and normalized margin disagreement;
- posterior mass with provider-backed cumulative HVD active;
- whether all experts use the same frozen source-domain set;
- whether the shadow joint Bayes-risk decision agrees with the current robust
  finite-posterior reference decision;
- total-variation gaps between joint marginals and the legacy independent
  updates.

The audit consumes surrogate moments only and records
`target_oracle_used=False`, `used_for_decision=False`.

## Stage 2: Authoritative Joint Inference

Implemented behind `task_latent_inference_mode=authoritative`; `shadow` remains
the default until the paired gate passes. The authoritative mode uses the same
joint `Q_t(structure,sensitivity)` for:

- structural proposal allocation and predictive sampling;
- objective and constraint mixture weights;
- conservative constraint epistemic scaling, with
  `max(1, sensitivity_scale)^2` so sensitivity can never relax certification;
- posterior expected positive chance-margin loss in terminal recommendation;
- exact posterior-update KG clones and terminal values;
- checkpoint save/restore of both joint and legacy posterior state.

The first paired run is
`lodo_joint_authoritative_v1_n20_3domain_20260712`, scheduler tasks
`t29282..t29302`. It changes only the inference mode relative to the frozen
V32 configuration.

Status: rejected by the paired gate. FactorShock remains exactly `7/7`, but
Inventory falls from `5/7` to `3/7` and Queue from `3/7` to `2/7`. Queue mean
violation improves from `0.0577` to `0.0360`, so the joint loss moves toward
the boundary but does not rank it reliably. The causal diagnosis is that the
latent sensitivity state contains only a residual scale. It can learn that a
domain is difficult, but cannot represent whether the transferred constraint
mean is systematically conservative or optimistic. It also changes the search
trajectory, causing some V32 safe finalist actions to disappear.

The next isolated model extension is a source-LODO signed calibration state
`c=(bias,scale,loss)`. Bias centers are learned as source-domain LODO residual
quantiles after normalization by source LODO mean RMSE; they are not selected
from held-out truth. Signed bias enters predictive likelihood and Bayes loss,
while the theory certificate ignores bias and retains the non-relaxing scale
floor. This is a missing coordinate in the same task latent, not an additional
decision gate.

### V2 scalar-bias causal smoke

Run `lodo_joint_authoritative_v2_signed_bias_smoke_20260712` changed only the
signed sensitivity state and evaluated four pre-registered cases. It preserved
FactorShock seed 0 and recovered Inventory seed 4 from V1's infeasible action
to margin `-0.0509` with regret `0.00564`. Queue seed 0 improved from V1 margin
`+0.0468` to `+0.0167` but remained infeasible, while Queue seed 6 regressed
from V32's feasible `-0.0086` to `+0.0328`. V2 is therefore rejected. A single
task-wide signed offset cannot express that transferred mean error changes
over the chance boundary in cumulative-risk coordinates.

### V3 functional signed calibration

V3 keeps the same finite latent state but replaces the scalar offset by a
source-only low-rank function

```text
b_j(psi) = theta_j^T [1, A, Helmert(N)].
```

Each source-domain LODO constraint residual is normalized by its own LODO
RMSE before fitting. The shared exposure uses orthonormal simplex contrasts,
so the intercept is not collinear with `sum(N)=1`. Consequently `b_j` is a
dimensionless transferable error shape; at a target point it is multiplied by
that expert's current predictive standard deviation. Source-domain scoring
sets a full-support frozen prior over the null and source profiles. Held-out
observations may only reweight these profiles. The function enters predictive
likelihood, exact-KG samples, and terminal Bayes loss, but remains absent from
the conservative theory certificate.

Before scheduler submission, the three held-out constructions passed a
source-leakage and amplitude audit: all priors have positive support and random
target-pool profile outputs remain within approximately `[-1.4, 1.0]`
predictive standard deviations. V3 is evaluated first on exactly four causal
smokes: FactorShock seed 0, Inventory seed 4, and Queue seeds 0 and 6. It does
not expand to the 21-seed matrix unless FactorShock is preserved, Inventory
seed 4 is feasible, Queue seed 6 remains feasible, and Queue seed 0 is no
worse than V32 while moving toward feasibility.

Status: rejected; no 21-seed expansion. FactorShock seed 0 is preserved
exactly. Inventory seed 4 returns V1's infeasible action at margin `+0.01313`
instead of V2's feasible `-0.05090`. Queue seed 0 reaches only `+0.03646`
(slightly better than V32's `+0.03834`, but worse than V2's `+0.01665`), and
Queue seed 6 remains at V2's infeasible `+0.03275` instead of V32's feasible
`-0.00860`. All four runs have zero false certificates.

The rejection identifies a hierarchy error. V3 fits each `b_j(psi)` against
one common source-LODO mean and then shares that function across every
structural expert. The target posterior can learn compatibility weights, but
it cannot adapt the calibration function itself to an expert's distinct
prequential residuals. On Inventory and Queue, posterior mass consequently
returns mainly to null profiles; true-feasible points remain in the terminal
pool but receive larger Bayes/chance margins than the selected action.

The next admissible model is therefore not another profile dictionary or
gate. It is an expert-conditional hierarchical calibration posterior:

```text
theta_xi ~ source prior,
r_t,xi / predictive_sd_t,xi = phi(psi_t)^T theta_xi + epsilon_t,xi,
Q_t(xi, theta_xi, scale, loss) updated from charged target observations.
```

The source domains determine the coefficient prior and boundary-weighted
regularization. Each held-out target observation updates one conjugate
low-rank calibration posterior per structural expert. Coefficient uncertainty
may only inflate certification, while coefficient means affect Bayes ranking;
exact KG must clone and update the same posterior. This repairs the missing
expert/calibration self-consistency rather than adding a domain classifier.

## Stage 3: Expert-Conditional Hierarchical Calibration (V4)

Implemented behind
`task_latent_calibration_mode=expert_ridge`; the rejected V3 path remains
available as `source_profiles`. Source domains fit a boundary-weighted Gaussian
prior in the dimensionless Helmert risk coordinate:

```text
theta_xi ~ Normal(m_0, P_0^{-1})
r_t,xi = (y_t - mu_t,xi) / sqrt(s_t,xi^2 + v_t,xi)
P_t,xi = P_{t-1,xi} + w_t,xi phi_t phi_t^T
h_t,xi = h_{t-1,xi} + w_t,xi phi_t r_t,xi
m_t,xi = P_t,xi^{-1} h_t,xi.
```

`w_t,xi` emphasizes the observed chance boundary
`y_t + z_alpha sqrt(v_t,xi) - tau` and uses only the charged target
observation. The source covariance has a fixed dimensionless spectrum in
`[0.25,4]`; this prevents source-near-null coordinates from creating an
improperly diffuse target prior. The held-out target supplies no labels to the
prior fit.

Bayes ranking uses
`mu_t,xi + predictive_sd_t,xi phi_t^T m_t,xi`. Theory certification ignores
this signed mean and replaces expert epistemic variance by

```text
s_t,xi^2 max(1,c_scale)^2
  + predictive_sd_t,xi^2 phi_t^T P_t,xi^{-1} phi_t.
```

The second term is nonnegative, so adaptation cannot relax a theory
certificate. The continuous Gaussian KL is included in the finite task
ambiguity radius. Candidate filtering, terminal Bayes loss, expert finalist
nomination, predictive sampling, checkpointing, and exact-KG fantasy clones
all consume the same expert-conditional posterior.

V4 first reuses the four V3 causal cases with no changed budget or candidate
configuration. It expands only if FactorShock seed 0 is preserved, Inventory
seed 4 is feasible, Queue seed 6 remains feasible, and Queue seed 0 improves
toward feasibility without a false certificate.

Status: rejected; no 21-seed expansion and no baseline promotion. All four
scheduler tasks completed normally with `expert_ridge`, and no checkpoint was
pulled back. The predeclared outcomes are:

| Domain / seed | V32 true margin | V4 true margin | V4 feasible regret |
|---|---:|---:|---:|
| FactorShock / 0 | `-0.03320` | `-0.03320` | `0.00825` |
| Inventory / 4 | `-0.00306` | `+0.01313` | infeasible |
| Queue / 0 | `+0.03834` | `+0.06498` | infeasible |
| Queue / 6 | `-0.00860` | `+0.03275` | infeasible |

All four recommendations have `false_feasible=False`; V4 never relaxes the
theory certificate. Every final recommendation pool contains a truly feasible
point, so the rejection is not explained by an absent terminal candidate.
There are instead two failures. On Queue seed 0, the fraction of sequential
pools containing a true-feasible action falls from V32's `0.9` to `0.4`:
updating the same calibration posterior inside exact KG changes the search
trajectory in the wrong direction. On Inventory seed 4 and Queue seed 6, a
true-feasible terminal action is available but the calibrated Bayes loss ranks
it behind the selected infeasible action. V4 therefore fails both proposal
value learning and terminal ranking, not merely certification.

The result also rules out the tempting variance-guard explanation. Posterior
coefficient covariance does enlarge theory epistemic variance as designed,
but neither V32 nor V4 has a certified terminal point in these cases. The
behavioral difference is produced by the adaptive calibration mean inside
Bayes ranking and exact-KG fantasy updates. Tuning its ridge, boundary weight,
or ambiguity penalty after observing these four outcomes is disallowed. The
expert-conditional posterior remains a mechanically complete ablation, while
V32 remains the promoted baseline.

Promotion requires all of the following on FactorShock, Inventory, and Queue:

1. no increase in false-feasible recommendations;
2. non-worse true-feasible count and feasible regret;
3. joint posterior calibration better than the independent marginals under
   prequential log score;
4. stable structure-sensitivity mutual information across seeds rather than a
   single-seed collapse;
5. cumulative-HVD active posterior mass remains high on selected candidates;
6. exact KG and certification both integrate the same authoritative `Q_t(z)`.

If these gates pass, the joint structure marginal replaces the legacy expert
weights and the joint expected sensitivity loss replaces the separate penalty.
If they fail, the shadow model remains a diagnostic and no extra gate is added.

## Required Ablations

- learned source prior vs uniform source prior;
- joint posterior vs independent structure and sensitivity posteriors;
- alignment coordinate vs identity coordinate inside the same joint model;
- cumulative HVD vs pointwise variance inside the same joint model;
- source prior frozen vs target-only adaptation;
- fixed-universe V32 race with and without joint posterior decisions.

These ablations distinguish genuine cross-domain transfer from a flexible
target-only optimizer that happens to contain several useful experts.

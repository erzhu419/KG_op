# Transferable Chance-Boundary Coordinate: V4 Analysis and Experiment Plan

Status: system analysis complete; no scheduler jobs submitted.

Date: 2026-07-12

## 1. Question

Can the rejected V4 expert-conditional calibration posterior be developed into
a model that learns a transferable chance-boundary coordinate from historical
domains and a small number of charged target observations?

Short answer:

```text
Yes as a redesigned research branch.
No as another round of ridge/weight tuning on the current expert_ridge model.
```

V4 has useful probabilistic infrastructure, but its current statistical object
is not a chance-boundary coordinate. It assumes that a frozen source-learned
coordinate is already correct and fits a linear residual correction inside
each structural expert. The Queue results reject that assumption.

## 2. What a transferable chance-boundary coordinate means

### 2.1 The boundary is a level set, not merely a feature vector

For task/domain `d`, define the signed chance margin

```text
M_d(x) = m_{g,d}(x)
         + z_alpha sqrt(v_{C,d}(x))
         - tau_d.
```

The chance boundary is

```text
B_d = {x : M_d(x) = 0}.
```

A vector representation `z_d(x)` is useful only if a low-complexity function
`h_d` makes

```text
M_d(x) approximately h_d(z_d(x)).
```

Thus `z` is a canonical risk representation; the scalar signed coordinate
`rho=h(z)` measures location relative to the boundary. Calling raw `A,N` the
boundary coordinate skips the essential geometry `h`.

### 2.2 The transferable object is not the exact target boundary

Different domains need not share an identical threshold, center, or unit.
What may transfer is:

- which observable statistics indicate buffer, imbalance, overload, local
  exposure, and shared shock;
- which directions move toward or away from danger;
- whether the boundary is approximately linear, monotone, convex, or
  low-rank quadratic in a canonical coordinate;
- prior distributions over task-specific shifts, rotations, scales, and HVD
  parameters.

The target task must still learn its own location and scale from charged
observations. This is the same distinction as learning a common metric and
then adapting a task-specific decision boundary.

### 2.3 Interpreting the user's intuition

The sports example expresses a valid meta-principle: distance to a scoring
region, number of defenders, and relative ability may determine danger in
many games. The exact field, units, orientation, and scoring rule differ, but
the relationship between buffer, opposition, mismatch, and opportunity can
be historically learned.

For this project the analogous invariant is not a hand-written statement
about one benchmark. It is a source-trained relationship among:

```text
observable state statistics
    -> local/shared risk exposure
    -> signed chance margin and variance
    -> terminal decision value.
```

Historical data is therefore useful and, for strong cross-domain transfer,
probably necessary. The critical requirement is that historical labels come
from source tasks and held-out target labels enter only through budgeted
evaluations.

## 3. Explicit space, latent space, or both?

### 3.1 Pure explicit coordinate

Examples:

- queue length, waiting time, utilization;
- backlog, inventory gap, stockout exposure;
- state occupancy, flow imbalance, demand shock exposure.

Advantages:

- interpretable HVD decomposition;
- straightforward nonnegativity and PSD constraints;
- easier theory and audit;
- less representation uncertainty.

Limitations:

- meanings, units, ordering, and dimension differ across domains;
- a frozen semantic slot can rotate or permute in a new task;
- handcrafted target statistics risk becoming a structural oracle.

### 3.2 Pure latent coordinate

Learn `z=phi(u)` from trajectory/policy histories using margin-conditioned
alignment, contrastive loss, or a task posterior.

Advantages:

- can align rotated, permuted, or differently named statistics;
- can learn interactions not present in a fixed exposure dictionary;
- naturally supports historical multi-task data.

Limitations:

- latent coordinates are identifiable only up to transformations;
- marginal domain alignment can collapse safe and unsafe states together;
- a large encoder recreates the few-shot identifiability problem;
- black-box representation learning weakens the OR theory narrative.

### 3.3 Recommended hybrid coordinate

Use an explicit risk backbone plus a small orthogonal latent residual:

```text
psi_exp(x) = [whitened local exposure A,
              whitened shared exposure N,
              generic buffer/imbalance statistics]

z_res(x) = phi_perp(observable trajectory statistics)

z(x) = [psi_exp(x), z_res(x)]
```

with

```text
E_source[psi_exp z_res^T] approximately 0.
```

The explicit block carries the cumulative HVD theorem. The latent residual may
explain transferable structure omitted by the exposure ontology, while its
norm and rank enter a representation-error guard. This avoids replacing the
main theory with a Transformer.

For an unobserved policy, trajectory statistics may themselves be uncertain.
A source-trained predictor should return a distribution over `z`, not a point
estimate. After a charged simulation, the observed trajectory updates that
distribution. Representation uncertainty must enter certification and exact
KG.

## 4. Alignment must be conditional on boundary semantics

Aligning marginal `p(z)` across domains is insufficient: domains can have
different safe-region prevalence, and marginal alignment may mix opposite
sides of the boundary.

The correct source objective is closer to

```text
p(z | normalized chance-margin bin, exposure type, task family)
```

being invariant across source domains. Useful losses include:

- supervised contrastive loss on source chance-margin bins;
- conditional MMD or domain-adversarial loss within each bin;
- pairwise ordering loss for safer versus more dangerous policies;
- orthogonality/whitening of canonical axes;
- reconstruction or next-risk prediction only as an auxiliary loss.

The target adaptation should be low-dimensional:

```text
z_d(x) = Q_d z_0(x) + t_d,
```

where `Q_d` is orthogonal or selected from a finite source-trained subspace
dictionary and `t_d` is a translation. A task-specific boundary intercept and
scale are updated from target observations. This handles rotations,
permutations, and shifted danger thresholds without relearning the encoder.

## 5. What V4 currently does

V4 uses six structural experts. For each expert it fits

```text
theta_xi ~ Normal(m0, P0^{-1})

(Y_t - mu_{t,xi}) / predictive_sd_{t,xi}
    = phi(A_t,N_t)^T theta_xi + epsilon_t,
```

where

```text
phi(A,N) = [1, A_1, A_2, A_3, Helmert(N)_1, Helmert(N)_2].
```

The `A,N` values are already latent: they are produced by the frozen
source-trained `LearnedMetaPrior`, not by the target's hidden synthetic risk
provider. V4 therefore does not lack a latent feature input.

It lacks the following:

1. target adaptation of coordinate orientation or translation;
2. direct modeling of the chance margin or boundary level set;
3. nonlinear/convex geometry needed for compact feasible pockets;
4. a shared global boundary mechanism with small expert deviations;
5. enough source-domain diversity to identify invariance;
6. an acquisition-safe way to prevent an uncertain calibration posterior from
   steering exact KG into unsupported regions.

The current target state has six coefficients for each of six experts. Each
conditional posterior receives sixteen target updates, but the joint latent
object still contains 36 calibration coefficients plus discrete structure and
sensitivity variables. That is too much weakly identified flexibility for the
purpose of locating a new boundary with `N=20`.

## 6. What the V4 smoke actually rejects

| Domain / seed | V32 true margin | V4 true margin | Result |
|---|---:|---:|---|
| FactorShock / 0 | `-0.03320` | `-0.03320` | preserved |
| Inventory / 4 | `-0.00306` | `+0.01313` | lost feasibility |
| Queue / 0 | `+0.03834` | `+0.06498` | worsened |
| Queue / 6 | `-0.00860` | `+0.03275` | lost feasibility |

The V32 finalist mechanism partly confounds this comparison. Three of four V4
runs finish with an uncertified empirical finalist override. However, the
truth-only frozen Bayes-incumbent margins under V4 are still:

| Domain / seed | V4 frozen Bayes margin |
|---|---:|
| FactorShock / 0 | `-0.03320` |
| Inventory / 4 | `+0.01221` |
| Queue / 0 | `+0.04480` |
| Queue / 6 | `+0.03275` |

Fixing the finalist race is necessary for a fair V4 evaluation, but it cannot
rescue the current model by itself.

There are two independent V4 failures:

- Queue seed 0 sequential pool feasibility coverage falls from V32's `0.9`
  to `0.4`, so exact KG is steered in the wrong direction;
- Inventory seed 4 and Queue seed 6 retain feasible terminal candidates but
  rank them behind unsafe actions.

The experiment rejects the hypothesis that expert-conditional linear
residual calibration in frozen `A,N` is enough. It does not reject historical
boundary representation learning in general.

## 7. Is V4 worth continuing relative to V32?

### 7.1 V32 advantages

- passed the predeclared FactorShock/Inventory promotion gate;
- has broad candidate support, including `7/7` safe Queue terminal pools;
- fixed-universe state is simple and reproducible;
- currently stronger empirical baseline.

V32 limitations:

- final suffix uses an incoherent heuristic race;
- every empirical override is uncertified;
- Queue has no posterior-certified action;
- expert decision and predictive weights can disagree strongly;
- the cumulative expert carries little final decision mass;
- it adapts among existing experts but does not learn a transferable boundary
  geometry.

### 7.2 V4 advantages

- one probabilistic state links structure, sensitivity, and calibration;
- coefficient uncertainty is explicit;
- covariance cannot relax the theory certificate;
- Gaussian KL enters the ambiguity radius;
- exact KG clones and updates the same posterior;
- checkpointing and diagnostics already support continuous latent state.

V4 limitations:

- worse empirical quality on the controlled smoke;
- frozen coordinate and linear residual head;
- high target-adaptation dimension;
- calibration uncertainty makes exact KG vulnerable to negative exploration;
- no evidence yet that its learned state transfers across domains.

### 7.3 Decision

Continue V4 only as infrastructure for a new transferable boundary posterior.
Do not tune `expert_ridge` ridge strength, boundary weight, temperature, or
penalty using the four revealed seeds.

The future comparison should be:

```text
V33 = V32 with coherent terminal replication KG

TCB-V1 = V33 + transferable chance-boundary posterior
```

V33 is the low-risk baseline repair. TCB-V1 is the higher-risk, higher-novelty
research branch. They solve different problems and can be developed in
parallel. TCB-V1 may replace V33 only after a paired multi-domain gate.

## 8. Proposed transferable chance-boundary posterior

Let `u_d(x)` be generic observable trajectory/policy statistics. Define

```text
z_0 = phi(u_d(x))
z_d = Q_d z_0 + t_d
```

and a shared low-rank PSD boundary geometry

```text
m_{g,d}(x)
    = a_d + b^T z_d + z_d^T H z_d + r_d(z_d),

H = L L^T, rank(H) <= r_H.
```

The cumulative variance uses the same coordinate:

```text
v_{C,d}(x)
    = A(z_d)^T Lambda_d A(z_d)
      + N(z_d)^T B_d N(z_d)
      + N(z_d)^T omega_d
      + floor_d
      + residual_perp.
```

The boundary is

```text
a_d + b^T z_d + z_d^T H z_d
+ z_alpha sqrt(v_{C,d}^+(x)) - tau_d = 0.
```

Source history learns priors over `phi,b,H,Q_d,t_d,a_d` and HVD parameters.
At the held-out target:

- `phi,b,H` are frozen or strongly regularized;
- only low-dimensional `Q_d,t_d,a_d` and HVD scale are updated;
- the posterior starts with full support over a finite/low-rank alignment
  family;
- target updates use only budgeted observations and trajectories;
- signed posterior means affect Bayes ranking;
- alignment/parameter covariance only enlarges certification uncertainty;
- exact KG clones and updates the full target posterior.

For a first Lean-friendly implementation, replace continuous `Q_d` with a
finite source-trained dictionary of orthogonal subspaces plus a continuous
translation/scale posterior. Continuous Stiefel inference can remain a later
extension.

## 9. Historical data requirements

The current Queue LODO prior is trained from only FactorShock and Inventory.
More records within two domains estimate those domains better, but do not
identify a cross-domain invariant mechanism. Domain diversity matters more
than raw record count.

Each source episode should contain:

- policy/design variables;
- observable trajectory/state statistics;
- repeated simulation outputs or a defensible variance estimate;
- objective and constraint response;
- normalized chance-margin estimate with uncertainty;
- task/domain identity;
- provenance showing that target test data is absent.

Source episodes must cover both sides of the chance boundary. A source task
with only safe or only unsafe histories contributes weak information about
boundary orientation.

Paper-grade meta-training should contain at least five genuinely distinct
task families, for example traffic, inventory, queue/resource control,
factor-shock/state-policy control, and energy/reliability control. Multiple
parameterized tasks inside each family can improve estimation, but the outer
LODO split must hold out complete families.

## 10. Offline representation screen

Large parallelism is useful here because evaluation is offline and does not
consume KG simulations. It is not useful to run 100 full KG variants on the
same seven Queue seeds.

Predeclare a `4 x 4 x 3 x 2 = 96` configuration screen:

### Coordinate family (4)

1. explicit whitened `A,N`;
2. current frozen learned `psi=(A,N)`;
3. boundary-conditioned latent alignment;
4. hybrid explicit coordinate plus orthogonal latent residual.

### Boundary geometry (4)

1. monotone linear;
2. diagonal PSD quadratic;
3. low-rank PSD quadratic;
4. monotone spline/RBF ablation.

### Target adaptation family (3)

1. frozen source head;
2. task shift and scale;
3. finite orthogonal alignment plus task shift.

### Effective rank (2)

```text
r in {2,4}.
```

All hyperparameter selection must use nested source-domain splits. The final
held-out domain is untouched until one or two variants have been frozen.

Offline metrics:

- held-out pairwise chance-margin rank loss;
- false-safe rate and dangerous-point recall;
- proper predictive log score/Brier score;
- calibration by normalized margin bin;
- conditional domain discrepancy after alignment;
- effective rank and parameter count;
- performance when the pilot target set contains no feasible point;
- representation uncertainty on out-of-support candidates.

Promotion to online KG requires:

- non-worse false-safe rate than current learned `psi`;
- lower rank loss in at least four of five outer held-out task families;
- no single-domain collapse;
- effective target-adaptation dimension below `0.35 n0`;
- no target-oracle field in the feature or loss audit.

## 11. Online causal experiment sequence

### Track A: repair V32 first

Run the five suffix-policy variants specified in
`v32_queue_system_improvement_plan.md`:

```text
5 variants x 3 domains x 7 seeds = 105 tasks.
```

The predeclared primary challenger is one-step terminal replication KG.

### Track B: boundary-coordinate smoke

After offline screening, retain at most two coordinate variants. On the V33
base run:

```text
2 coordinates
x 2 integration modes (terminal-only, acquisition+terminal)
x 4 sentinel cases
= 16 tasks.
```

Sentinel cases remain:

- FactorShock seed 0;
- Inventory seed 4;
- Queue seeds 0 and 6.

The boundary branch expands only if it preserves FactorShock, repairs or
preserves Inventory, and does not lose Queue seed 6 while improving Queue seed
0. A failure is not repaired by tuning after the four outcomes are revealed.

### Track C: full paired gate

For the surviving coordinate:

```text
V33 vs TCB-V1
x 3 domains
x 7 seeds
x N=20
```

Then, and only then:

```text
N in {40,80}, seeds=20,
outer LODO over at least five task families,
SOTA comparison at identical evaluation budgets.
```

The online primary claim is not selected from 96 variants. The 96-way screen
selects a representation on source validation; the held-out KG gate evaluates
one frozen method.

## 12. Theory targets

The redesigned branch can support a coherent theory package:

1. boundary-coordinate identifiability up to an orthogonal transform;
2. conditional-alignment generalization across source domains;
3. PSD low-rank boundary geometry and effective-dimension bound;
4. hierarchical target posterior contraction under boundary excitation;
5. cumulative HVD decomposition in the same canonical coordinate;
6. nonrelaxing certification under representation uncertainty;
7. exact posterior-update KG over structure, boundary, and HVD state;
8. finite-horizon terminal replication-KG Bayes optimality on fixed `U`.

This is materially deeper than adding a learned feature encoder to V32. The
representation, HVD, certification, and KG all operate on one posterior
object.

## 13. Final recommendation

V4 is not yet a transferable chance-boundary model, but its failure does not
close that direction. The most defensible route is:

1. keep V32 as the empirical baseline;
2. replace its finalist heuristic with terminal replication KG to obtain V33;
3. reuse V4's joint-posterior infrastructure for a hybrid explicit/latent
   boundary-coordinate posterior;
4. screen representation families offline with nested source splits;
5. run only one or two frozen boundary challengers online;
6. merge the boundary posterior into the mainline only after it beats V33 on
   paired multi-domain gates.

No jobs should be submitted until the terminal-KG primary variant and the
offline boundary-screen split protocol are frozen.

## 14. Frozen offline screen implementation

The boundary screen now instantiates the full preregistered factorial:

```text
4 coordinate families
x 4 boundary geometries
x 3 target-pilot adaptations
x 2 coordinate ranks
= 96 independent configurations.
```

The coordinate families are `explicit_stable`, `learned_psi`,
`boundary_latent`, and `hybrid_explicit_latent`. The geometries are monotone
linear, diagonal PSD quadratic, low-rank PSD quadratic, and RBF. Adaptation is
frozen, shift/scale, or boundary-bin-conditioned orthogonal Procrustes plus a
two-parameter output calibration. Rank is two or four.

Selection is source-only. The development source bank is:

```text
RZDT1, RZDT2, RegimeRZDT1,
StatePolicyRZDT1, HighDimStatePolicyRZDT1.
```

Every configuration is evaluated by nested leave-one-source-domain-out
replay. Held-out pilot policies are selected randomly before their outcomes
are read. Evaluation rows are disjoint, never passed to fitting, and used only
for false-safe, rank-loss, boundary-RMSE, and upper-coverage metrics. The three
online sentinel domains `FactorShockStatePolicyRZDT1`,
`InventorySupplyChain`, and `QueueResourceControl` are absent from the 96-way
selection. At most two configurations may enter the subsequent 16-task
sentinel smoke.

This development bank is leakage-safe but is not yet the paper-grade claim of
five genuinely distinct application families: RZDT1/RZDT2 and the two
state-policy tasks remain related synthetic families. A future manuscript
gate still requires traffic, inventory, queue/resource, factor-shock control,
and an independent energy/reliability family. The current screen chooses what
is worth testing online; it does not establish that final external-domain
generalization claim.

## 15. V1 source-only outcome and V2 calibration stage

The completed V1 matrix contained all 96 preregistered cells. Eighty were
dimension-admissible, but none passed the full source gate. The failure was
not primarily representation ranking:

```text
52/79 challengers improved rank loss on at least four of five folds;
17/79 avoided single-domain collapse;
12 challengers satisfied both conditions;
0 challengers also matched the fixed baseline's zero false-safe rate.
```

The fixed learned-psi baseline achieved zero false-safe errors by certifying
essentially no safe policies. It is therefore a vacuous safety reference, not
a useful operational certificate. The closest challenger, hybrid explicit
plus latent RBF with shift/scale adaptation and rank two, missed two dangerous
rows out of 48 in one fold while improving rank and avoiding collapse.

Inspection exposed a specific statistical omission. The target adapter fit a
two-coefficient output calibration but discarded its coefficient covariance
and reused the source residual constant after shifting and scaling the target
head. V2 replaces that point estimate with a source-regularized Bayesian
linear calibration posterior. Its coefficient covariance and posterior
residual scale can only widen the predictive upper bound. No manually tuned
guard is introduced.

The V2 gate is frozen before rerunning results. In addition to the V1
conditions, it requires non-worse dangerous-point recall, a strict increase in
mean certified-safe coverage over the fixed baseline, and nonempty certified
safe sets in at least two held-out folds that contain a nontrivial safe region.
This prevents an all-unsafe classifier from passing merely because its
false-safe count is zero.

Only the fixed baseline and the 12 V1 cells that already satisfy the
mean-ranking and no-collapse conditions are rerun. Predictive calibration does
not alter either quantity, so the other 83 cells cannot pass V2 regardless of
their upper uncertainty. The exact source-only selection and parameters are
recorded in `performance/manifests/tcb_v2_calibration_stage.json`.

## 16. V2 outcome and canonical-residual V3

V2 removed all false-safe errors for the 12 eligible challengers, but every
predictive upper bound remained vacuous: no method certified a safe row in any
of the source LODO folds. Posterior residual scales ranged from roughly 1.5 to
12.3 normalized margin units, so a 99% Student-t bound could not cross below
zero. This is not repaired by lowering the confidence level or hand-tuning a
guard. It shows that an intercept and source-score scale do not transfer the
target boundary shape accurately enough.

V3 adds one source-defined canonical residual direction to the target output
head:

```text
target margin = intercept
              + scale * source boundary score
              + gamma * first canonical boundary coordinate.
```

Only the three coefficients above use target labels. Their Bayesian posterior
covariance still widens certification, and the effective target dimension is
three, below the frozen `0.35 n0 = 3.5` cap. The source residual prior is made
weakly informative with one prior degree of freedom; the 99% upper level and
all promotion gates remain unchanged.

Because the added residual coefficient can alter mean rank and RMSE, V3 does
not reuse V1's rank-based pruning. It reruns all 32 shift/scale cells across
coordinate, geometry, and rank, plus the fixed frozen baseline. Orthogonal
adaptation is excluded because its rotation plus the three output parameters
exceeds the target-dimension cap. The exact 33-cell source-only manifest is
`performance/manifests/tcb_v3_residual_coordinate_stage.json`.

## 17. V3 outcome and weak-shift hierarchy

V3 improved mean boundary fits in several cells but again certified no safe
source-validation row. The diagnostic residual scale increased in domains
with a large boundary offset. This traced to a hierarchy mismatch: the output
calibration assigned the same precision to the task-specific intercept and to
the supposedly transferable source-score and canonical-direction
coefficients. The posterior scale update consequently charged a legitimate
domain intercept shift as unexplained residual noise.

The next source-only stage changes one quantity. Intercept prior precision is
`0.001` times the shape-coefficient precision. The intercept remains in the
Bayesian posterior, so its covariance still enlarges predictive uncertainty;
it is not treated as known. Source score, residual direction, upper confidence
level, target dimension, and all gates stay fixed. This implements the model's
stated hierarchy: boundary location is task-specific, while orientation and
shape are the transferable objects. The frozen 33-cell protocol is recorded
in `performance/manifests/tcb_v4_weak_shift_hierarchy.json`.

## 18. Weak-shift outcome and source-only boundary excitation

The weak-shift hierarchy preserved zero false-safe errors and produced the
first nonvacuous certificate: hybrid explicit/latent RBF with rank two
certified one safe RZDT1 row. It still failed the requirement of nonempty safe
sets in two held-out folds. The next missing condition is boundary excitation,
not another posterior coefficient.

The original ten target pilots were sampled uniformly before outcomes were
read. This is leakage-safe but ignores the source model's most useful role in
few-shot optimization: deciding which unlabeled target policies are jointly
informative about boundary location and orientation. The next stage freezes
the weak-shift posterior and selects pilots from an unlabeled random policy
pool using only source predictions and canonical coordinates. It seeds the
minimum predicted margin, nearest predicted boundary, and maximum predicted
margin, then greedily maximizes log-determinant information gain plus a
boundary proximity bonus.

Target outcomes and true margins are inaccessible to the selector. Only after
the ten indices are fixed are their charged outcomes exposed to the target
posterior. The fixed 33-cell source-only protocol is recorded in
`performance/manifests/tcb_v5_boundary_excitation.json`. Random-pilot V4 is
retained as the negative-control ablation.

## 19. Excitation outcome and robust support control

Source-only D-optimal excitation produced pilots on both true sides of the
boundary in the three folds with substantial feasible regions, but it did not
pass. The selected design was dominated by unsupported source-score leverage;
for example, one RZDT2 pilot had a frozen source margin prediction above 81.
That point is algebraically informative for a linear design matrix but poorly
supported for transfer calibration.

The robust excitation stage changes only pilot selection. Candidate policies
outside the 5th--95th percentile of frozen source scores are excluded. The
initial three points target the 10th percentile, nearest predicted boundary,
and 90th percentile. Source scores and canonical coordinates use robust
central-range scaling and are clipped to three standardized units before
log-determinant selection. Target outcomes remain unavailable until all ten
indices are fixed. The protocol is recorded in
`performance/manifests/tcb_v6_robust_excitation.json`.

## 20. Robust-excitation outcome and stop decision

Robust excitation did not pass the frozen source gate. Across the 32
challengers, all matched the fixed baseline's zero false-safe rate, 19
improved rank loss on at least four of five folds, and 23 avoided
single-domain collapse. Only one challenger certified any safe point, and it
did so in one fold rather than the required two. The strongest rank variants
still had target posterior residual scales of roughly 1.5--9.3 normalized
margin units, making a 99% upper certificate empty.

The sequence isolates what worked and what did not:

```text
conditional coordinates improved ordering;
Bayesian output covariance removed unsafe overconfidence;
weak task-intercept shrinkage corrected a hierarchy mismatch;
source-only pilot design reached both true boundary sides;
robust support control removed the most obvious leverage pathology;
none made the cross-domain absolute chance margin identifiable at n0=10.
```

The Boundary Track therefore stops before online sentinel evaluation. No
candidate is promoted, and no sentinel outcome is used to tune this branch.
The next defensible attempt would require a new statistical object and fresh
source validation families: for example a multi-family hierarchical
signed-distance likelihood whose domain scale is learned from repeated source
trajectories, not another support quantile or ridge sweep on the current RZDT
development bank. Until then, TCB remains a documented negative result and
V32 remains the active baseline.

## 21. V33 Inventory diagnosis and TCB-V1 disposition

The completed V33 matrix confirms that TCB-V1 stopped at the right gate. On
Inventory, the one-step terminal policy has the following post-hoc audit:

```text
mean candidate-pool-has-true-feasible rate       0.943
best-true-feasible posterior-feasible rate       0.000
mean missed-true-feasible rate                   0.843
true-feasible terminal targets                   5 / 28
seeds whose four-arm terminal set contains one   4 / 7
true-feasible terminal actions selected          2 / 21
seeds recovered by minimum empirical upper arm   3 / 7
upper margin on measured true-feasible targets
    min / median / max                 0.269 / 0.463 / 0.566
```

Truth is used only after each run for this audit. It was unavailable to
candidate generation, posterior fitting, KG, and recommendation. The reusable
audit is implemented in
`performance/diagnose_v33_inventory_terminal.py`.

This distinguishes three mechanisms:

1. Candidate support is not the primary failure. Safe points are usually in
   the pool.
2. The absolute certificate is vacuous. Every best true-feasible pool point
   is classified as posterior-unsafe.
3. Four-arm frontier compression drops all true-feasible arms in three of
   seven seeds even though the larger candidate pool usually contains them.
4. Terminal Bayes ranking compounds the issue. It selects a true-feasible arm
   in only two of 21 decisions because the terminal loss sees a poorly
   calibrated absolute margin and a large uncertainty penalty.

The terminal loss and final safety contract are also different mathematical
objects. The former is `posterior objective + 5 * robust expected positive
margin`; the latter requires a nonpositive upper confidence margin. When all
upper margins are positive, a rollout can improve smooth Bayes risk without
making any arm eligible for the final certified decision. Choosing the arm
with the smallest empirical upper margin after the existing repetitions would
recover only three of seven seeds, so changing the terminal tie-break alone is
not enough.

The V32-versus-V33 gap is also not evidence that V32 has a valid certificate.
V32 loses no false-feasible seeds in this seven-seed sample, but its feasible
recommendations in Inventory seeds 0 and 1 are rescued by the legacy
uncertified empirical override. The V33 contract correctly prohibits that
override and therefore exposes the calibration failure instead of hiding it.

TCB-V1 was never an online result. It was the intended label for a source-gate
winner integrated into V33. The 96-cell screen and five source-only repairs
produced no winner, so the preregistered 16 sentinel runs and the full paired
TCB-V1 gate were deliberately not launched. V33 subsequently failed its own
primary gate, removing the second prerequisite for Track C.

The offline development bank also contained five RZDT variants rather than
five genuinely different application families: RZDT1, RZDT2, RegimeRZDT1,
StatePolicyRZDT1, and HighDimStatePolicyRZDT1. It was appropriate for rejecting
an unstable coordinate, but it cannot support a paper claim of transfer to
Inventory, Queue, or Traffic.

TCB-V1 should not be revived by choosing the best rank-only configuration
post hoc. Its relative coordinate learned useful ordering, but its target
residual scale remained roughly 1.5--9.3 normalized margin units and made the
99% certificate empty. That is the same mechanism now observed directly in
Inventory.

## 22. Required statistical replacement: TCB-V2

Any continuation must replace residual regression with an identifiable
hierarchical signed-distance model. A minimal model is

```text
rho_d(x) = a_d + exp(s_d) h_theta(z_d(x)),
z_d(x)   = Q_d phi(u_d(x)) + t_d,

observed normalized margin = rho_d(x) + epsilon,
epsilon ~ replicate-aware source/target likelihood.
```

The shared `h_theta` learns transferable boundary shape. Domain location
`a_d`, positive scale `exp(s_d)`, and finite orthogonal alignment `Q_d` are
random effects with a source-trained hierarchical prior. Target observations
update only these low-dimensional effects. Their covariance can enlarge, but
never reduce, certification uncertainty.

The same upper margin must define the terminal decision. On a finite frozen
arm set, use the lexicographic terminal value

```text
V_cert(D) = min_lex,x ( 1{U_g(x;D) > 0},
                        max(U_g(x;D), 0),
                        posterior objective(x;D) ).
```

Thus a certified arm always dominates an uncertified arm; if no certificate
exists, KG values reduction of the smallest upper margin before objective
improvement. This removes the current arbitrary violation-penalty mismatch
without weakening the safety bound.

This changes the estimand from "predict a residual in a possibly rotated
coordinate" to "estimate a canonical signed distance plus an identifiable
domain location and scale." Repeated source and target trajectories enter the
likelihood directly, separating boundary-location uncertainty from
heteroscedastic simulation noise.

TCB-V2 must pass a source-only gate before any KG sentinel:

- zero or non-worse false-safe rate;
- rank-loss improvement in at least four of five source outer folds;
- nonempty certified-safe set in at least two folds;
- empirical coverage of the declared upper margin;
- finite target adaptation dimension below `0.35 n0`;
- no target truth in fitting, selection, or hyperparameter choice.

Until that gate passes, no online TCB experiment is admissible. The immediate
mainline task is therefore not deeper terminal rollout. It is to construct and
validate the hierarchical signed-distance likelihood on fresh source-family
splits.

## 23. TCB-V2 and V33 three-layer implementation contract

TCB-V2 is now implemented as `HierarchicalSignedDistancePosterior`. Source
domains jointly fit a canonical boundary shape and one location/positive-scale
effect per domain. A held-out target updates only `(location, log scale)` from
noisy replicated observations. The adapter covariance and residual scale enter
the predictive upper margin; they cannot lower it. Target oracle values are not
accepted by the fitting API.

The V33 repair is deliberately coupled to that posterior in three places:

1. Finalist nomination reserves actions with minimum TCB upper margin and
   closest TCB boundary distance.
2. Terminal rollout uses the vector value
   `(uncertified probability, positive upper margin, objective)` and compares
   it lexicographically. It no longer uses an arbitrary violation penalty.
3. `tcb_v2_mode=certified` makes the same upper margin authoritative at final
   recommendation and disables empirical, calibration, source-prior, and
   replicated-finalist overrides that would change the certificate.

Every fantasy observation rebuilds the two-dimensional target adapter from the
fantasy observation dictionary, so terminal KG values information about the
same boundary posterior that will make the final decision. `shadow`,
`frontier`, and `certified` modes provide explicit ablations of the three
layers.

The source-only gate is implemented by
`performance/benchmark_tcb_v2_source_gate.py` and
`performance/summarize_tcb_v2_source_gate.py`. It uses genuinely different
RZDT, state-policy, factor-shock, inventory, and queue families. Target truth
is evaluated only after fitting. Hyperparameters are selected separately for
each outer held-out domain by an inner LODO over the remaining source domains;
the outer target is absent from both inner training and selection. The online
V33/TCB-V2 submitter rejects non-nested summaries and refuses to launch unless
the saved gate contains a passing source-selected configuration for every
held-out domain. No empirical TCB-V2 claim is made until that gate passes.

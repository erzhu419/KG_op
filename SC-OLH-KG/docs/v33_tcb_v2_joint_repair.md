# V33 Three-Layer Repair and TCB-V2 Joint Gate

Status: implementation and both gates complete. Focused TCB-V2 source gate
failed, V33 coherent-frontier repair failed, and no challenger was promoted.

Date: 2026-07-12

## 1. Evidence that triggered the repair

The completed preregistered V33 matrix contains 105 runs over FactorShock,
Inventory, and Queue with seven seeds per domain and five suffix policies.
The one-step terminal KG result was:

| Domain | V32 feasible | V33 one-step feasible |
|---|---:|---:|
| FactorShock | 7/7 | 7/7 |
| Inventory | 5/7 | 2/7 |
| Queue | 3/7 | 5/7 |

Depth three retained the same 7/7, 2/7, and 5/7 outcome while increasing
runtime. Rollout depth was therefore not the missing mechanism.

The Inventory truth-only audit, performed after every decision was frozen,
showed:

- the terminal pool was approximately 94.3% truly feasible;
- only 4/7 frozen four-arm finalist sets contained a truly feasible arm;
- only 2/21 terminal actions selected a truly feasible arm;
- no recommendation was posterior-certified;
- the old empirical override rescued seeds 0 and 1, while the new scalar
  terminal rule removed that rescue without replacing it with a coherent
  certificate-aware decision.

Truth was not used by the optimizer. These facts identify three consecutive
implementation distortions rather than a candidate-pool oracle failure.

## 2. The three distortions

### 2.1 Frontier distortion

The old four-arm frontier inserted the minimum Bayes-risk arm and then let
expert nominations consume the remaining slots. The nominal and robust safety
axes were evaluated only after the expert loop, so they were frequently absent.

### 2.2 Terminal-value distortion

The main exact KG, the suffix replication KG, and the final recommendation
could use three different terminal objects: scalar Bayes risk, hard theory
certification, and empirical/calibrated fallback. Improving one scalar did not
necessarily improve the final certified decision.

### 2.3 Recommendation distortion

Even after a certificate-aware suffix, empirical calibration, observed-point
fallback, source-prior fallback, or an uncertified finalist could replace the
posterior action. The final layer could therefore undo the value optimized by
the preceding layer.

## 3. Implemented V33 repair

`decision_contract_mode=certified_lexicographic` is one authoritative switch.
It forces all three layers to minimize the same vector:

```text
(probability of no certified action,
 expected positive upper chance margin,
 posterior objective)
```

The contract has the following consequences:

- main exact KG uses the fantasy-updated vector terminal value;
- suffix replication KG uses the same vector after cloning and updating the
  task posterior, GPR, HVD, observations, and TCB target adapter;
- final recommendation first selects a certified action, otherwise the
  smallest upper margin, and uses objective only as the final tie-breaker;
- empirical reranking, observed fallback, source fallback, and uncertified
  finalist override are disabled inside this contract;
- TCB shadow/frontier modes may add candidates but cannot silently replace
  the theory-HVD terminal certificate.

`finalist_frontier_policy=coverage_reserved` reserves four directions before
any expert nomination:

1. minimum posterior Bayes risk;
2. minimum authoritative certificate margin;
3. minimum robust expected violation;
4. minimum nominal expected violation.

Duplicate minima advance to the next action on that axis. Remaining slots are
filled by expert nominations, supported disagreement, posterior-signature
diversity, and Bayes-risk order. No target truth enters this construction.

## 4. TCB-V2 status

TCB-V1 refers to the earlier expert-conditional linear calibration posterior.
It updated a separate residual coefficient vector inside each structural
expert but did not model a shared signed boundary level set. With six
coefficients for each of six experts it exposed 36 weakly identified target
parameters at `N=20`; the FactorShock/Inventory/Queue sentinel showed negative
transfer. TCB-V1 is therefore retained only as a failed ablation. V2 is the
lower-dimensional hierarchical replacement, not a tuned alias of V1.

TCB-V2 models a source-shared signed chance-boundary shape with a small target
posterior over location, positive scale, optional planar rotation, and an
orthogonal low-rank residual. Parameter covariance and residual uncertainty
can only increase the upper margin.

The earlier nested source-only matrix had 720 result files. Its selected outer
evaluation had mean Spearman 0.2284, predicted-safe count zero, and
nonvacuous-rate zero. The gate correctly failed. Subsequent one-seed screens
showed that canonical provider-risk coordinates improve rank correlation, but
those coordinates use target problem structure and are therefore an explicit
structure-aware upper bound, not strict LODO evidence.

Synthetic unit checks show that planar source alignment and a rank-one
orthogonal residual recover data generated by those assumptions. This proves
the implementation can represent the intended model; it does not show that
the five empirical domains share that model.

The focused TCB-V2 rerun is restricted to five strict configurations:

- raw rigid;
- raw plus planar rotation and rank-one residual;
- learned-risk rigid;
- learned-risk plus planar rotation and rank-one residual;
- raw plus learned-risk with planar rotation and rank-one residual.

Provider descriptors are now ineligible for strict LODO promotion even if
their numerical gate metrics pass.

## 5. Preregistered gates

### Gate A: V33 repair

Run four variants over three domains and seven seeds, 84 independent tasks:

```text
v32
v33_legacy_4
v33_coherent_coverage_4
v33_coherent_coverage_8
```

The challenger must preserve zero false certificates, retain FactorShock
7/7, recover Inventory to at least the V32 5/7 level, and retain the V33 Queue
gain of at least 5/7. Runtime is secondary within the already accepted
approximately 9-12 minute range.

The 84-task matrix completed with no failed runs:

| Variant | FactorShock | Inventory | Queue |
|---|---:|---:|---:|
| V32 | 7/7 | 5/7 | 3/7 |
| V33 legacy four-arm | 7/7 | 2/7 | 5/7 |
| Coherent coverage four-arm | 7/7 | 1/7 | 3/7 |
| Coherent coverage eight-arm | 7/7 | 1/7 | 4/7 |

All variants had zero false certificates and zero target-oracle use. The
coherent variants passed every implementation audit: all four reserved axes
were present in every seed, all 63 suffix decisions used the lexicographic
terminal value, and no empirical finalist override occurred. The primary
eight-arm challenger nevertheless failed both target feasibility and
per-domain nonworsening, so it was not promoted.

The post-hoc truth audit explains why. For coherent eight-arm Inventory, only
4/7 finalist sets contained a truly feasible arm, only 3/21 suffix actions
selected one, and the final recommendation was feasible in 1/7. It had three
support failures and three final-ranking failures. Queue had a truly feasible
arm in 6/7 sets but only 4/21 selected suffix actions and 4/7 final
recommendations were feasible. Thus the larger frontier improved support but
did not repair posterior ordering.

Most importantly, every recommendation in every domain remained uncertified.
For coherent eight-arm runs the median theory upper margin was `0.070` on
FactorShock, `0.301` on Inventory, and `0.637` on Queue. V32's empirical
override had hidden this vacuity in some seeds. The coherent contract did not
create the defect; it exposed that the current certificate cannot recognize
the safe actions on which empirical performance depends.

### Gate B: TCB-V2 source-only

Run five configurations over five outer domains and three seeds, 75 offline
tasks. Promotion requires complete nested source LODO, no target oracle,
strict learned/raw descriptors, coverage at least 0.95, false-safe rate no
worse than frozen, mean Spearman at least 0.35, and a nonvacuous safe set in at
least half of folds.

The focused 75-task gate completed on 2026-07-12. Nested outer evaluation
obtained coverage `0.9562`, zero false-safe predictions, mean Spearman
`0.3458`, and predicted-safe count `0`. It therefore missed the preregistered
rank threshold narrowly but failed the substantive nonvacuity requirement
completely. No configuration passed and no TCB-V2 model was promoted. This is
not repaired by rounding `0.3458` to `0.35`: a certificate that never admits
an action cannot be the online main method.

### Gate C: joint online KG

The V33/TCB-V2 joint matrix remains mechanically blocked unless Gate B emits a
source-selected configuration for every held-out domain. If Gate B fails,
TCB-V2 remains a negative result and V33 uses the theory-HVD certificate.
Both antecedent gates failed, so no joint online task was submitted.

## 6. Audit and proof map

- V33 algorithm: `algorithms/single_olhkg.py`
- V33 repair scheduler: `scripts/submit_scolhkg_v33_frontier_repair_scheduler.py`
- TCB source gate: `performance/benchmark_tcb_v2_source_gate.py`
- TCB gate summary: `performance/summarize_tcb_v2_source_gate.py`
- joint scheduler: `scripts/submit_scolhkg_tcb_v2_v33_scheduler.py`
- Lean bridge: `proof/SCOLHKG/Real/HierarchicalBoundaryCertificate.lean`

The Lean bridge covers positive scale, nonnegative covariance radius, planar
rotation norm preservation, monotonicity of the upper margin under added
rotation/residual uncertainty, reserved frontier order, coherent three-layer
margins, and certified lexicographic dominance. It does not claim that an
empirical source gate must pass.

## 7. TCB-V3 statistical replacement

TCB-V2's failure identifies a model error rather than a tuning error: one
source-shared boundary shape cannot simultaneously represent the five source
domains without inflating residual uncertainty until every safe set is empty.
TCB-V3 therefore keeps a finite source-frozen library instead of averaging the
source domains into one shape.

For each outer LODO split, the library contains one pooled family and one
leave-one-source-domain-out family per available source domain. The held-out
target name and formula are absent from the fitting API. Ordinary budgeted
pilot margins update family mass using leave-one-pilot-out generalized-Bayes
predictive evidence. Family parameters remain frozen.

The Bayes ranking mean is the posterior-weighted family mean. The certificate
is deliberately different: it is the pointwise maximum upper margin over the
smallest family set carrying at least `1 - delta_family` posterior mass, plus
an optional nonnegative between-family guard. Thus an optimistic low-weight
family cannot average away a dangerous family that remains credible.

Implementation and audit paths are:

- `representation/transferable_boundary.py`: `BoundaryFamilyMixturePosterior`;
- `performance/benchmark_tcb_v3_family_gate.py`: identical V2 data and budget,
  with only the model builder replaced;
- `performance/summarize_tcb_v3_family_gate.py`: nested source-only selection
  plus family-simplex, credible-mass, frozen-parameter, and target-label audits;
- `scripts/submit_scolhkg_tcb_v3_family_gate_scheduler.py`: six preregistered
  configurations, `6 x 5 x 3 = 90` offline tasks on `node001-node006`;
- `proof/SCOLHKG/Real/BoundaryFamilyMixtureCertificate.lean`: credible-family
  envelope safety and the `delta_family + alpha` failure bound.

Online V33/KG integration remains blocked until this source-only gate passes
coverage, false-safe, ordering, and nonvacuity checks. This prevents another
online search from concealing a vacuous boundary model.

The first 90-task V3 gate used pooled and leave-one-source-out families. It
passed every leakage/family-posterior audit but failed statistically: nested
outer Spearman was `0.2943`, the posterior retained `4.21` effective families
on average, and no evaluation fold had a nonempty certified set. Leaving out
one domain still pooled three heterogeneous source boundaries, so these were
not genuine boundary atoms.

TCB-V3.1 replaced those broad members by one frozen atom per source domain.
The complete 90-task gate again passed every audit. It improved nested outer
Spearman to `0.7103`, beat the frozen ranking in `93.3%` of folds, and reduced
the effective/credible family counts to `1.38/1.82`. This is strong evidence
that target pilots can identify a transferable *boundary family* without the
target label. It did not pass certification: conservative configurations
remained empty, while the narrower configurations issued 20 safe predictions
and all 20 were false-safe. V3.1 is therefore a successful representation
diagnostic but not a certifier and is not promoted.

Code inspection refines the failure diagnosis. Each atom already fits a
target-specific intercept and positive scale from the ordinary pilot data;
the missing degree of freedom is family-conditional shape correction. A
single global Student-t residual scale absorbs remaining shape error and
widens the chance boundary everywhere. TCB-V3.2 therefore keeps the discrete
family posterior and adds one or two source-learned orthogonal residual
coordinates inside each family. The residual coefficient posterior is fitted
only from the same target pilots, and its covariance can only increase the
family upper margin. The V3.1 rank-zero model remains in the gate as a control.
The joint online KG gate remains blocked until this correction is nonvacuous
without false-safe transfer.

The V3.2 correction also failed its 90-task gate. Nested selection produced
33 certified points, but 32 were false-safe; coverage fell to `0.9376` and
Spearman to `0.6602`. Rank-one and rank-two residual coordinates therefore
fit target-pilot accidents at `n0=10` instead of a transferable deformation.
The rank-zero V3.1 model remains the better family representation, but neither
version is eligible for online integration.

TCB-V4 changes the model class instead of adding another residual coordinate.
Every source domain contributes one frozen canonical signed-distance atom
`s_k(x)`. The held-out target posterior is

```text
m(x) = c + sum_k gamma_k s_k(x),  gamma_k >= 0.
```

The source-domain coefficient vectors define the prior over `(c, gamma)`;
ordinary target pilots update that posterior. The nonnegative cone preserves
the orientation of every source risk coordinate while allowing a held-out
boundary to lie between source families. The unconstrained covariance is kept
at active nonnegativity constraints, so truncation is not used to make the
certificate narrower. The predictive residual and coefficient covariance
both enter the Student-t upper margin. V4 has its own leakage/PSD audit and
Lean contract in
`proof/SCOLHKG/Real/BoundaryFamilySynthesisCertificate.lean`; it does not
reuse the discrete-family envelope theorem as if the assumptions were equal.

The first 90-task V4 gate passed every implementation audit and obtained
coverage `0.9508`, zero false-safe predictions, nested Spearman `0.7108`, and
a `93.3%` rank-win rate. One fixed configuration certified two genuinely safe
points, but source-only nested selection still chose configurations with an
empty safe set. V4 is therefore materially safer than V3.2 and as accurate in
ranking as V3.1, but it has not passed the preregistered nonvacuity gate.

The remaining width is primarily posterior residual scale rather than
coefficient covariance. This matters because Inventory and Queue have minimum
true margins near `-0.03`, below the approximate one-point 99% noise radius at
three replicates, while FactorShock has no feasible point in the fixed random
evaluation pools. A follow-up scaling diagnostic freezes the two V4 models
selected before inspection and evaluates `n0=20` and `n0=40`. It does not
retroactively change the V4 gate. Its purpose is to distinguish a posterior
that contracts with information from irreducible dictionary misspecification.

The scaling diagnostic resolved that distinction. From `n0=10` to `20` and
`40`, rank correlation improved from roughly `0.72` to `0.79`, and parameter
uncertainty contracted, but posterior residual scale stayed essentially flat.
No `n0=20/40` configuration certified a point. The V4 source cone is therefore
misspecified for the held-out local boundary geometry; more observations only
estimate that misspecified cone more precisely.

TCB-V5 is the preregistered functional-class repair:

```text
m_T(x) = c_T + gamma_T' s_S(x) + r_T(s_S(x)),
gamma_T >= 0,
r_T orthogonal to span{1, s_S} on the frozen source design.
```

`r_T` uses a bounded RBF-center dictionary in source-family score space. Its
coefficient projection is computed from source descriptors without target
labels and frozen before target adaptation. Nullspace orthogonality makes the
transferable synthesis and target-local residual identifiable on that design;
the residual cannot silently rewrite the source-family coefficients. Only one
or two residual coordinates are allowed, keeping the outer adapter dimension
at six or seven. Coefficient covariance and remaining residual scale both
enter the upper margin. TCB-V5 has a separate audit and Lean bridge and remains
blocked from online KG until its nested source-only gate passes.

The complete V5 gate contained 90 tasks and 180 outer rows. Every leakage,
nonnegativity, covariance-PSD, dimension, source-projection orthogonality, and
frozen-dictionary audit passed. Nested evaluation nevertheless obtained
coverage `0.9486`, Spearman `0.7010`, zero false-safe predictions, one true
certified point, and nonvacuity rate `0.0333`. It therefore failed both the
coverage and nonvacuity requirements and was not promoted.

The sequence now separates the evidence cleanly:

1. V3.1 shows that source boundary atoms transfer ranking information.
2. V4 shows that nonnegative continuous family synthesis preserves that rank
   signal and avoids false-safe transfer.
3. `n0=10/20/40` shows that the remaining global residual is structural, not
   parameter uncertainty.
4. V5 shows that one or two source-frozen orthogonal RBF coordinates do not
   close that structural gap under strict LODO.

No TCB challenger is connected to V33 or online KG. A further attempt must
change the *observable cross-domain boundary coordinate or the benchmark's
certifiability design*, not add another small residual basis or relax the
confidence level after seeing these results.

## 10. Certifiability and coordinate-sufficiency audit

The next diagnostic separates two explanations that V3-V5 could not
distinguish:

1. the benchmark may contain feasible points that are too close to the chance
   boundary to certify at the stated confidence with the available
   replications;
2. the benchmark may be certifiable in principle, but the source-frozen
   observable coordinate discards information needed to recover its absolute
   chance margin.

For a point with true chance margin `m(x)` and true constraint standard
deviation `sigma_g(x)`, the optimistic known-variance oracle uses

```text
U_R^oracle(x) = m(x) + z_(1-delta) sigma_g(x) / sqrt(R).
```

Thus a strictly feasible point requires at least
`ceil((z_(1-delta) sigma_g(x) / -m(x))^2)` direct replications even when the
functional form and variance are known. The audit reports both this optimistic
normal radius and the finite-replication Student-t radius for
`R in {1,3,5,10,20,50,100}` on a uniform random pool and a separately labelled
domain-augmented oracle pool.

Coordinate sufficiency is tested with disjoint target evaluation points and
three explicit information strata:

- `strict_source_frozen`: the untouched V4 source prior;
- `target_oracle_diagnostic`: target truth may fit ridge/RBF ceilings in raw,
  source-learned, and source-family-score coordinates;
- `domain_tuned_oracle_upper_bound`: declared provider `A,N` coordinates,
  alone and concatenated with raw descriptors, available only as a diagnostic
  upper bound. The concatenation distinguishes missing risk exposure from
  missing chance-mean geometry.

Every row is marked `promotion_eligible=false`. If a flexible target-oracle
regressor cannot recover the boundary in a source-frozen coordinate, further
adapter tuning is logically incapable of repairing that coordinate. If the
provider closes the gap, the next model must learn a transferable observable
exposure coordinate rather than importing the target provider. If even the
noise oracle cannot certify feasible support, the benchmark replication or
safe-support design must be repaired before another boundary posterior is
tested.

Within the target-oracle stratum, `random` training draws only from a disjoint
uniform-random candidate prefix. The `oracle_boundary_stratified` protocol is
the only one allowed to select from domain-declared hooks, and its row is
labelled `training_candidate_pool=domain_augmented_oracle_pool`.

Implementation:

- `core/oracle_certification.py`
- `performance/benchmark_certifiability_coordinate_audit.py`
- `performance/summarize_certifiability_coordinate_audit.py`
- `scripts/submit_scolhkg_certifiability_coordinate_audit_scheduler.py`

The formal 50-task matrix is registered as
`certifiability_coordinate_20260712_v1` (`t33088` through `t33137`), one
domain/seed per 12-core task on `node001-node006`. All 50 tasks completed after
the CPU pool recovered. Every held-out domain has all ten seeds, all result
contracts are valid, and the associated full test run passed 295 tests with no
failure.

### 10.1 Completed audit result

The domain-augmented oracle pools contain nontrivial truly feasible support:

| Held-out domain | True-feasible pool rate | Oracle feasible recall at R=20 | Oracle feasible recall at R=100 |
|---|---:|---:|---:|
| FactorShock | `0.0577` | `0.6113` | `0.8741` |
| Inventory | `0.1940` | `0.7504` | `0.8934` |
| Queue | `0.2167` | `0.6914` | `0.8601` |
| RZDT1 | `0.3646` | `0.9210` | `0.9728` |
| StatePolicyRZDT1 | `0.1212` | `0.7939` | `0.8827` |

These are optimistic known-variance lower bounds on the required replication
burden. They reject the explanation that the benchmarks contain no points with
meaningful safety depth. They also show why a total budget of 20 cannot certify
many distinct candidates by direct replication alone.

The first summary contained a selection bug: it hard-coded the RBF diagnostic
and ignored the preregistered ridge regressor. After correction, the declared
provider upper bounds do close the missing-coordinate gap: FactorShock reaches
Spearman `0.8858` and normalized boundary MAE `0.0742`; Inventory reaches
`0.7876/0.2990`; Queue reaches `0.8517/0.2660`. These are still target-oracle,
domain-provider ceilings and remain ineligible for promotion.

The existing source-frozen coordinate is already sufficient for FactorShock
(`0.9156/0.2718`), StatePolicy (`0.9474/0.1262`), and RZDT1
(`0.9999/0.0012`), but not Inventory or Queue. A formula-free multiscale
observable dictionary then showed that Queue's mean geometry is recoverable
(`0.7501/0.3997` in the 97-feature audit), while Inventory remained marginal
(`0.6627/0.5553`). A compact 55-feature quadratic/low-frequency dictionary
raised Inventory rank correlation to `0.8651`, but its absolute boundary MAE
remained high because the diagnostic split fits only 48 rows before validation
and calibration. This is an estimation-dimension failure, not evidence that
the quadratic exposure pocket is absent; a unit-level exact-span test confirms
the dictionary contains that family.

The registered decision remains `rebuild_transferable_observable_coordinate`,
but the implementation consequence is now precise: source domains fit the
larger observable dictionary, then expose only a 4-6 dimensional frozen atom
summary to the held-out target. Another boundary adapter, wider confidence
envelope, or residual basis on the old shared coordinate is ruled out. The new
model distinguishes a mean coordinate `eta(x)` from the cumulative-risk
coordinate `psi(x)=(A(x),N(x))`, while joining them in the single chance-margin
object

```text
rho(x) = m_g(eta(x)) + z_alpha sqrt(v_C(psi(x))) - tau.
```

Both coordinates must remain observable, source-learned, frozen before the
held-out target, and updated only through paid target evaluations. Target
oracle regressors in this audit remain diagnostic ceilings and are never
promotion eligible.

### 10.2 Oracle-free online challenger

Retrospective audit classifies V32 as a privileged source-oracle upper bound:
its held-out target is clean, but `lodo_teacher` uses source `true_sigma`,
source `true_outputs`, and source structural hooks. New promotion gates reject
such rows. The oracle-free challenger instead uses three ordinary simulator
replicates for each of 64 records on each of two source domains (`384` source
calls), no teacher records, and `N=20` charged target calls.

The constraint-mean GPR uses frozen `eta`; every finite expert retains its own
`psi=(A,N)` cumulative HVD, theory certificate, and exact-KG clone/update. The
source training target is the replicated constraint mean with chance-boundary
sample weights, so variance cannot leak into `eta`. The first margin-trained
`eta` run is retained as an ablation. Oracle-free V32 control, aggregate-eta,
margin-latent-eta, and mean-latent-eta three-domain smoke gates are registered
on `node001-node006`; none is promoted before all three domains are complete.

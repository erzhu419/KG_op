# SC-OLH-KG V5-V24 Failure Analysis

> Retrospective status (2026-07-15): this is a chronological mechanism log.
> Statements that V32 was the promoted baseline are true only at their recorded
> experiment date. The information audit later reclassified V32 as a privileged
> upper bound and promoted the oracle-free source-consensus successor.

## Scope

This note records the failed and partially successful challengers that followed
the frozen V5 task-posterior baseline. It separates empirical facts from
interpretations so that later changes do not retrofit a favorable story.

All runs below used the same held-out protocol, no target oracle in decisions,
`d=50`, `N=20`, `n0=10`, seven seeds, factor HVD, theory certification, and
exact-MC KG. Truth was used only for post-run audit metrics.

## Result Trail

| Version | Main change | FactorShock true feasible | Inventory true feasible | FactorShock median regret | Inventory median regret | Inventory mean violation |
|---|---|---:|---:|---:|---:|---:|
| V5 | robust task posterior | 4/7 | 1/7 | 0.03065 | 0.02681 | 0.09896 |
| V6 lex | cumulative HVD for every non-null expert, minimum-margin fallback | 6/7 | 0/7 | 0.00825 | 0.03663 | 0.08253 |
| V6 adaptive | latent sensitivity penalty | 7/7 | 0/7 | 0.00825 | 0.02825 | 0.07823 |
| V7 | empirical boundary expected loss replaces robust fallback | 2/7 | 1/7 | 0.27418 | 0.01903 | 0.02267 |
| V8 | class-conditioned robust/empirical model averaging | 7/7 | 2/7 | 0.00825 | 0.02294 | 0.03262 |
| V9 | robust cumulative HVD variance also used in empirical branch | 7/7 | 1/7 | 0.00825 | 0.02294 | 0.03917 |
| V10 | feature standardization and nominal-HVD empirical risk with LOO ridge selection | 7/7 | 1/7 | 0.00825 | 0.02294 | 0.03917 |

None of V6-V10 produced a posterior-certified point in either held-out domain.
Their true-feasible counts therefore measure recommendation quality, not a
successful theory certificate.

## Current Root-Cause Hierarchy

The accumulated V5-V21 evidence separates the failure into five levels.  They
must be repaired in this order because a later decision rule cannot recover
information or structure missing at an earlier level.

| Level | Root cause | Evidence | Current status |
|---|---|---|---|
| 1. Information | Inventory feasibility is rare and early target samples often lie on one side of the chance boundary | only 1.35% of the frozen oracle-audit pool is feasible; V21a has an initial feasible point in 6/7 seeds but still ends at 0/7, so coverage is necessary but not sufficient | partially open |
| 2. Representation | aggregate/aligned `psi` has cross-boundary collisions and weak held-out semantics | compact aligned coordinate has audit AUC 0.444; source-only ordered diagonal coordinate reaches AUC 0.985 | ordered coordinate fixes capacity in audit |
| 3. Identification | the useful coordinate is too dense for `N=20` | V21 full quadratic uses 17 coefficients, gets only 0.0124 median posterior mass on Inventory, and falls to 0/7 despite oracle AUC 0.992 | primary open cause; V22 targets it |
| 4. KG estimation | few Monte Carlo draws add ranking noise | antithetic MC8 improves median safe rank from 11 to 7, but exact categorical stratification still selects 0/6 safe actions | secondary, not sufficient |
| 5. Certification | the theory bound rejects empirically safe recommendations | even 7/7 safe FactorShock recommendations remain uncertified | open after mean/risk identification |

The causal diagnosis is therefore not simply "the gate is conservative".
The main current failure is that a transferable coordinate with enough oracle
capacity cannot yet be identified from the charged target budget.  V22 makes
one structural change only: replace the local-kernel expert by an 11-feature
ordered diagonal model and constrain its complete posterior effective
dimension to approximately `0.35 N`.  Threshold relaxation, target-specific
anchors, and additional experts are excluded from this challenger.

## Failure Chain

### 1. The target observations often do not bracket the chance boundary

The final Inventory recommendation pool contains true-feasible low-regret
points in every seed, so the optimizer is not failing because the entire pool
misses the feasible region. However, many sequential runs observe no
true-feasible point among the first 15-20 evaluations. The boundary model is
then trained from one side and must extrapolate to choose a final point.

This distinguishes two events:

- `pool coverage`: a true-feasible point exists in the diagnostic candidate
  pool;
- `information coverage`: a feasible and an infeasible point have both been
  evaluated within the optimization budget.

V5-V10 generally achieve the first and fail the second. More recommendation
weighting cannot recover information that was never collected.

### 2. The empirical boundary fit is not identifiable at N=20

The recommendation calibration uses roughly 40 or more frozen features with
at most 20 distinct target observations. Before V10 it used a fixed ridge of
`1e-6`. This produced nearly zero in-sample residuals but extreme candidate
predictions: audited calibrated margins for good feasible points ranged from
about `-1.6` to `+2.2`.

V10 allowed LOO ridge selection, but five of seven Inventory seeds still chose
`ridge=1e-6`; their effective ranks were 14-20. Hat-matrix LOO is degenerate in
this interpolation regime because both the residual and `1-h_ii` approach
zero. Clipping the denominator does not turn the resulting score into an
honest out-of-sample estimate.

The missing condition is therefore structural, not numerical:

`effective_unknowns << n_target_observations`.

The next fit must impose this condition before comparing predictive scores.

### 3. Source alignment does not yet guarantee target boundary semantics

The source-only alignment and spectral basis are admissible, but their
coordinate meaning can change on the held-out domain. Inventory diagnostics
frequently reject the aligned challenger and retain the universal coordinate
basis. Replayed source-safe profiles can consequently remain target-unsafe.

The representation is useful as a proposal coordinate, but source feasibility
must not be interpreted as target feasibility. Its defensible use is to form a
stratified bracket along a frozen source risk score, followed by target-budget
updates.

### 4. Latent task sensitivity is a protection layer, not an identifiability model

The sensitivity posterior is not useless: V6 adaptive and V8-V10 preserve
FactorShock at 7/7 feasible recommendations. V7 shows why this matters: fully
trusting an empirical boundary collapses FactorShock to 2/7.

It does not solve Inventory. A posterior over the cost or scale of prediction
errors can decide how cautiously to act, but it cannot identify a high-rank
boundary from one-sided data. It remains an optional decision layer and must
earn its place through an ablation after the low-rank model is fixed.

### 5. Robust and empirical uncertainty were double-counted in V9

V9 inserted the KL-robust upper cumulative HVD variance into an empirical
branch that was already mixed with the robust loss. This increased predicted
violation probabilities without correcting the mean ranking and reduced
Inventory from 2/7 to 1/7.

The correct separation is:

- nominal cumulative HVD for the empirical posterior predictive model;
- KL/PAC-Bayes upper moments for the robust branch and formal certificate.

Both use the same `psi=(A,N)` risk coordinates, but they represent different
levels of epistemic protection.

### 6. The theory certificate remains too conservative to validate the empirical wins

Even the consistently true-feasible FactorShock recommendation has a positive
posterior theory margin in all seven seeds. The present experiments therefore
show that the optimizer can locate a good point, but not that the certificate
can recognize it. This gap must be reported explicitly and addressed after
the mean model becomes identifiable; relaxing `beta_g` or `v_C_plus` now would
hide rather than solve the problem.

## Discarded Explanations

- `The search pool has no feasible point`: false for the audited Inventory
  recommendation pools.
- `HVD alone is the Inventory bottleneck`: unsupported; changing empirical
  HVD use did not repair the mean ranking.
- `A task label should select a domain-specific rule`: rejected as leakage and
  unnecessary. The latent posterior uses source data and budgeted target
  observations only.
- `More penalty tuning will solve the issue`: contradicted by V6-V10. Penalty
  changes move recommendations but do not create boundary information.
- `LOO always protects against overfitting`: false when the candidate model is
  allowed to interpolate with effective rank near the sample count.

## Next Falsifiable Plan

1. **Identifiability-constrained calibration**
   - cap effective rank at a configurable fraction of distinct target points;
   - default to the existing structural prior `rank <= 0.35 n`;
   - rebuild preprocessing and the fit inside every validation fold;
   - reject all ridge/basis candidates that violate the rank cap;
   - score one-sided margin prediction and chance-boundary ordering in addition
     to squared error.

2. **Source-only boundary-coordinate design**
   - fit a frozen source margin score in the invariant/aligned coordinate;
   - generate a large unlabeled held-out pool;
   - select source-safe, source-boundary, source-unsafe, and maximin-diverse
     strata rather than treating source-safe as target-safe;
   - reserve part of `n0` for this bracket before expert-weighted proposals;
   - audit evaluated target feasibility coverage without using it in proposal
     decisions.

3. **Ablation after the structural fix**
   - low-rank model without latent sensitivity;
   - the same model with latent sensitivity;
   - the same model without source boundary bracketing;
   - current domain-tuned upper bound only as a non-mainline reference.

4. **Promotion gate**
   - FactorShock must remain at least 7/7 true feasible with zero violation;
   - Inventory must reach at least 4/7 true feasible, at most 1/7 false
     feasible, and improve mean violation over V5;
   - only then run Queue as a previously untouched held-out confirmation;
   - no challenger is committed or pushed before passing this gate.

## V11 Intervention And Discriminating Outcomes

V11 implements the first three items above without changing the simulation
budget or using target truth in any decision:

- recommendation calibration rebuilds scaling and ridge coefficients inside
  every held-out fold;
- a candidate is admissible only when its effective rank is at most
  `floor(0.35 n)` (with a minimum cap of two for tiny pilots);
- ridge selection scores constraint NMSE, dangerous underprediction, and
  pairwise boundary ordering;
- a source-domain leave-one-domain-out boundary score stratifies an unlabeled
  target pool across five risk quantiles;
- universal constants, ramps, profile perturbations, and random points expand
  that pool without target-specific structural hooks;
- a fixed one-class sensitivity model provides a clean no-latent ablation
  while retaining exactly the same calibration and recommendation loss.

An offline coverage audit on held-out Inventory (`d=12`, five selected points,
256 unlabeled candidates) produced two true-feasible and three infeasible
points. Truth was computed after selection and did not affect the candidates.
This validates the mechanism, not optimization performance.

The scheduled Gate-1 matrix uses the same `d=50`, `N=20`, `n0=10`, seven-seed
budget as V5-V10 and contains four cells:

| Cell | Rank-constrained nested fit | Source boundary bracket | Latent class |
|---|---:|---:|---:|
| fixed / no bracket | yes | no | no (one fixed class) |
| fixed / bracket | yes | yes | no (one fixed class) |
| latent / no bracket | yes | no | yes |
| latent / bracket | yes | yes | yes |

The outcomes have predeclared interpretations:

- bracket improves initial two-sided coverage and final feasibility: the main
  failure was information coverage;
- bracket improves initial coverage but not final feasibility: selection or
  certification remains the bottleneck;
- fixed and latent perform alike: remove the latent class from the mainline;
- latent improves both held-out tasks on the same low-rank model: retain it as
  a learned decision layer;
- neither low rank nor bracket improves Inventory: the frozen source alignment
  lacks transferable boundary ordering, so the next admissible method is a
  sequential bracket updated from budgeted target pilot evaluations;
- FactorShock drops below 7/7 or incurs violation: reject the challenger even
  if Inventory improves.

## V11 Gate-1 Results

All 56 scheduled runs completed (`4 cells x 2 held-out domains x 7 seeds`).
The gate was not passed.

| Cell | FactorShock feasible | Inventory feasible | FactorShock mean violation | Inventory mean violation |
|---|---:|---:|---:|---:|
| fixed / no bracket | 6/7 | 2/7 | 0.00162 | 0.04856 |
| fixed / bracket | 3/7 | 3/7 | 0.49825 | 0.06066 |
| latent / no bracket | 5/7 | 0/7 | 0.12844 | 0.07461 |
| latent / bracket | 3/7 | 2/7 | 0.35410 | 0.08178 |

Every final nested calibration fit satisfied its rank cap. Inventory initial
designs containing at least one true-feasible point increased from `2/7`
without bracket to `5/7` with bracket, so the bracket did improve information
coverage. It did not meet the final recommendation gate and severely damaged
FactorShock. The latent sensitivity class was a negative ablation in both
domains and is removed from the next challenger.

The FactorShock damage has a concrete support-coverage cause. With ten expert
proposals, the universal-coordinate expert receives two points; the second
generic low-frequency shape is the sole feasible pilot in all seven seeds.
Allocating five slots to the bracket left only one slot per expert and removed
that point. This is an invalid comparison of bracket value because adding the
bracket simultaneously deleted existing prior support.

V12 therefore reserves two source-only universal-coordinate support points,
then adds the bracket, and allocates only the remaining budget among other
experts. The rule is identical across held-out domains and uses no target
labels. Before scheduling, its truth-only initial-design audit found a feasible
pilot in `7/7` FactorShock and `6/7` Inventory seeds. V12 remains a challenger
until its final recommendation results pass the original gate.

## V12 Gate-1 Results

All 14 scheduled runs completed. V12 restored FactorShock but failed
Inventory after the initial design:

| Domain | Initial design contains feasible point | Final true feasible | False feasible | Median feasible regret | Mean violation |
|---|---:|---:|---:|---:|---:|
| FactorShock | 7/7 | 7/7 | 0/7 | 0.00825 | 0.00000 |
| Inventory | 6/7 | 0/7 | 0/7 | n/a | 0.06347 |

Thus source support and bracketing solved candidate coverage but not posterior
recognition. In every Inventory seed the final recommendation pool contained
the same low-regret true-feasible segmented profile, while the theory and
calibrated posterior margins rejected it. Seed 0 is especially diagnostic:
the initial design had already evaluated a nearby feasible segmented support
once, but one noisy observation was insufficient to distinguish it from a
nearby unsafe point. Exact KG did not return to that surprise observation.

This rules out three further explanations:

- the held-out pool lacks a feasible candidate;
- the source-only proposal cannot reach the relevant policy family;
- latent task classification is needed before the method can act safely.

The remaining failure is a budget-allocation problem under observation noise:
the method spends all later evaluations on new KG candidates while asking a
single observation to establish both local mean and heteroscedastic variance.

## V13 Budgeted Certification Recheck

V13 freezes up to three initial observations nearest the empirical chance
boundary and replicates each until it has three observations. Selection uses
only the budgeted noisy objective/constraint values and the common noise floor;
target truth, target labels, and extra simulator calls are forbidden. The six
possible rechecks consume the existing `N-n0=10` sequential evaluations.

Replicates have two distinct roles:

1. their sample variance enters the existing cumulative HVD update;
2. an explicitly uncertified observed-incumbent fallback may use a
   prior-shrunk replicate variance instead of the global noise floor.

The theory certificate remains
`mu_g + sqrt(beta_g) s_g + z_alpha sqrt(v_C_plus) <= tau`; V13 does not relax
`beta_g`, remove the HVD tail guard, or label an empirical fallback as
posterior-certified. Checkpoints store the frozen recheck targets, and a
resumed run never repeats a target after its prescribed replicate count.

The predeclared Gate-1 criterion is unchanged: FactorShock must remain `7/7`
feasible with zero violation, Inventory must reach at least `4/7` feasible
with at most one false-feasible recommendation, and only then may Queue be
opened as an untouched confirmation domain.

## V13 Gate-1 Results And Failure Localization

All 14 runs completed with exactly 20 charged simulations and no scheduler
failure. V13 did not pass:

| Domain | Final true feasible | Posterior certified | False feasible | Median feasible regret | Mean violation |
|---|---:|---:|---:|---:|---:|
| FactorShock | 6/7 | 0/7 | 0/7 | 0.00825 | 0.05338 |
| Inventory | 1/7 | 0/7 | 0/7 | 0.00564 | 0.07572 |

The one Inventory success is informative. Seed 3 rechecked the low-regret
segmented profile three times, obtained a prior-shrunk empirical margin of
`-0.022121`, passed the fixed fallback threshold `-0.020000`, and recommended
the true-feasible point with regret `0.00564`. In seed 2 another genuinely safe
point obtained margin `-0.019842`; it missed the same threshold by only
`0.000158` and was rejected. Seed 4 rechecked a safe point with true margin
`-0.05090`, but its empirical margin was `-0.00611` and it was also rejected.

The paired evidence identifies six defects rather than one tuning error:

1. **Winner's-curse target selection.** Closeness to a boundary after one
   noisy observation selects observations with unusually favorable noise.
2. **Non-adaptive allocation.** Every frozen target receives the same number
   of replicates, including candidates already trending unsafe, while a
   promising candidate cannot request a fourth observation.
3. **Insufficient joint mean/variance evidence.** Three observations can
   expose within-policy variance but rarely establish a chance margin at the
   desired confidence.
4. **Discontinuous fallback.** A `0.000158` numerical difference changes the
   final decision even though the underlying posterior evidence is nearly
   identical. This is a gate, not a coherent posterior decision rule.
5. **Exploration displacement.** Forced rechecks consume up to six of ten
   sequential evaluations. FactorShock seed 0 lost its formerly reliable
   recommendation and ended with margin `0.37364` despite rechecking a truly
   safe support point.
6. **Acquisition/decision mismatch.** Exact KG uses a hard certified terminal
   set and a separately normalized infeasibility penalty. The final fallback
   uses a different hard empirical threshold. Reduction of boundary
   uncertainty has almost no value until one of these thresholds is crossed.

The last defect is the theoretical bottleneck. Per-hypothetical-state min/max
normalization is not a fixed terminal loss, and clipping negative MC gains
does not restore Bayes coherence. V14 therefore removes forced replication
from the challenger and defines a fixed smooth terminal Bayes risk:

\[
R_t(x)=\mathbb E_{Q_t}[m_f(x)]
+\lambda\sup_{Q':\mathrm{KL}(Q'\Vert Q_t)\le\rho_t}
\mathbb E_{Q'}\!\left[\mathbb E[(M_x)_+]\right],
\]

where, within each task expert,

\[
M_x\sim\mathcal N\!\left(
m_g(x)+z_\alpha\sqrt{v_C^+(x)}-\tau,\ s_g(x)
\right).
\]

The Gaussian positive-part expectation is closed form. It is continuous at
the boundary, assigns value to epistemic reduction before certification, and
uses the same task posterior and cumulative HVD as the main theory. Existing
replication candidates remain in the candidate set, but exact KG must choose
them by value of information; no evaluation is forced. The final
uncertified fallback uses the same Bayes-risk ordering, while the formal
theory certificate remains unchanged and separately reported.

## V14 Bayes-Risk Gate Results

V14 completed all 14 runs but failed more sharply than V13:

| Domain | Final true feasible | Posterior certified | Median feasible regret | Mean violation |
|---|---:|---:|---:|---:|
| FactorShock | 2/7 | 0/7 | 0.00825 | 0.53754 |
| Inventory | 0/7 | 0/7 | n/a | 0.04225 |

The smooth terminal objective itself worked numerically: exact gains were
positive, decayed as information accumulated, and selected replication in
multiple seeds without a forced quota. The quality failure was predictive
misspecification. For FactorShock seed 0, the selected point had true margin
`1.39037` but posterior expected violation only `0.07022`.

The finite task posterior had collapsed to `risk_aligned_spectral` with weight
`0.9999976`. At the selected unsafe point that expert predicted margin
`-0.14479`, while the other four experts predicted margins from `1.09320` to
`1.90942`. At the true-feasible support all five experts predicted lower
positive-part losses than at the unsafe point. Thus the KL ambiguity set did
not fail to optimize its stated center; its center had deleted the models that
contained the relevant warning.

This distinguishes two remaining failures:

1. **Posterior support collapse (FactorShock).** Prequential Gaussian scores
   can accumulate differences of hundreds of nats under misspecification.
   A forward-KL ball around the resulting near-point-mass cannot recover an
   expert assigned probability near zero.
2. **Expert-family misspecification (Inventory).** Even before collapse, all
   transferred experts rank some unsafe profiles below the repeated safe
   segmented profile; only the null expert gives the safe point a lower
   violation loss. Prior protection alone is therefore necessary but not
   sufficient for Inventory.

V15 isolates the first defect. Decisions use

\[
\widetilde Q_t=(1-\epsilon_t)Q_t+\epsilon_t\Pi,
\qquad
\epsilon_t=\min\{1/2,\,1/\sqrt{n_t}\},
\]

for mixture moments, KL-robust risk, certification, and hypothetical expert
sampling. Raw generalized-Bayes weights remain separately logged and updated.
The prior component vanishes with evidence and has the already formalized
support guarantee
`widetilde_Q_t(k) >= epsilon_t Pi(k)`. V15 is tested on FactorShock only; it
must restore `7/7` before a second model-discrepancy component is allowed to
address Inventory.

## V15 Prior-Protection Gate Results

V15 protected decision support as designed but did not improve FactorShock:
`2/7` true-feasible, `0/7` certified, mean violation `0.34033`. Raw posteriors
still concentrated on one expert, while protected decision distributions kept
effective expert counts between about `1.5` and `2.4`. The mechanism worked;
the quality hypothesis did not.

Post-run recommendation counterfactuals used the identical posterior and pool,
without new simulator calls. `task_adaptive` recovered `6/7`, while pure
`min_margin` recovered only `3/7`. No single existing rule recovered all seven
seeds. In seed 0, the protected model assigned the unsafe selected point lower
robust expected violation (`0.216`) than the true-feasible support (`0.240`).
Thus posterior collapse amplified error, but the surviving expert family still
misordered target risk.

The model-class gap is concrete in the implementation. Every unvisited policy
uses only its global parametric basis. The solution-specific deviation is a
Kronecker delta and transfers no residual information to a nearby point in
state-risk coordinates. V16 therefore returns to the stable V12 decision path
and adds exactly one finite expert:

\[
g(x)=\phi(x)^\top\beta+r(\psi(x)),\qquad
r(\psi)=\sum_{j=1}^{6}w_j
\exp\{-\|\psi-c_j\|^2/(2\ell^2)\}.
\]

The six landmarks are selected by deterministic farthest-point coverage from
source-profile and universal-shape candidates. Their scale and lengthscale use
only source records and unlabeled held-out coordinates; target labels, truth,
and the target simulator are never queried. Target-budget observations update
the finite RBF coefficients. This is a null/model-discrepancy expert inside the
existing finite posterior, not a new target-specific anchor.

The V16 controlled comparison is V12 plus this expert only: hard-certified
exact KG, task-adaptive recommendation, no forced recheck, and no posterior
prior protection. FactorShock must remain `7/7`; Inventory must reach `4/7`
before Queue is opened.

## V16 Local-Discrepancy Gate Results

All 14 runs completed without failure or retry. V16 preserved the stable
FactorShock result but did not cross the Inventory feasibility boundary:

| Domain | Final true feasible | Posterior certified | Median feasible regret | Mean violation |
|---|---:|---:|---:|---:|
| FactorShock | 7/7 | 0/7 | 0.00825 | 0.00000 |
| Inventory | 0/7 | 0/7 | n/a | 0.01696 |

The Inventory mean violation is 73.3% lower than V12 (`0.06347`), so the local
expert is useful, but it does not pass the predeclared `4/7` promotion gate.
Its behavior is data-dependent rather than a hidden domain switch. The final
Inventory decision weight on `local_risk_kernel` ranges from about `0.032` to
`0.920`; on FactorShock it converges to one in every seed while preserving all
seven feasible recommendations.

Truth-only post-run audits localize the remaining gap. Every Inventory pool
contains the same low-regret feasible segmented profile with true chance
margin `-0.05090`. Its model decision margin remains between `+0.52993` and
`+0.93860`. The selected policies have true margins from `+0.00882` to
`+0.03084` and lower model decision margins. Thus V16 substantially improves
the unsafe ranking but neither recognizes the safe side nor produces a formal
certificate. This is not a candidate-coverage failure, and the absence of any
certified point means that simply changing the final hard threshold would
again be an uncertified gate.

The next discriminating experiment is the missing factorial combination:

- V14 tested smooth Bayes-risk exact KG with the misspecified global expert
  family and failed;
- V16 tested the enlarged local-discrepancy family with a hard terminal value
  and improved Inventory violation;
- V17 combines only these two existing changes, with no forced rechecks and no
  prior-protection mixture.

If V17 improves both domains, the failure was the interaction between model
support and a discontinuous terminal value. If it damages FactorShock again,
the smooth risk is rejected even with the repaired family. If it preserves
FactorShock but leaves Inventory infeasible, the remaining issue is target
boundary identification and requires an adaptive local experiment design,
not another recommendation penalty.

## V17 Smooth-Risk Plus Local-Discrepancy Results

V17 completed all 14 runs without failure. The combination was complementary
but missed the promotion threshold by one Inventory seed:

| Domain | Final true feasible | Posterior certified | Median feasible regret | Mean violation |
|---|---:|---:|---:|---:|
| FactorShock | 7/7 | 0/7 | 0.00825 | 0.00000 |
| Inventory | 3/7 | 0/7 | 0.01377 | 0.02870 |

This is the first smooth Bayes-risk challenger to retain the complete
FactorShock result, and it raises Inventory from V16's `0/7` to `3/7`.
Therefore model support and continuous terminal loss are both necessary in
this gate. V17 is still not promoted because the criterion is `>=4/7`.

The four Inventory failures separate into two empirical classes. Seed 1 is
the only seed whose initial design contains no true-feasible point and has the
large final violation `0.15815`. Seeds 4--6 begin with a feasible point but
finish with small positive margins `0.01052--0.01909`. Exact KG does choose
replication without a forced quota: the seven final runs contain between 11
and 20 distinct solutions out of 20 charged evaluations. It nevertheless
does not concentrate those repeats on the safe support reliably.

Candidate truth audits show a feasible acquisition candidate in every
iteration for seeds 0 and 2--6, and in half of the iterations for seed 1.
Selection remains sparse: a true-feasible candidate is chosen in only zero to
two of the ten sequential steps. Thus the next failure is neither pool
coverage nor the absence of replication actions. It is experimental ranking
within an available candidate set.

Code inspection exposes an implementation-level mismatch with the exact-KG
mathematical object. Before V18, `_solve_posterior_recommendation` generated a
terminal pool, `_exact_posterior_update_scores` immediately generated a second
pool after advancing `rec_rng`, and the realized recommendation generated a
third. The exact update was internally consistent on its private pool, but it
did not value information for the same action set used before and after the
real observation. Moreover, model-risk frontier actions from the terminal set
were not guaranteed to be available as experiments.

V18 repairs this without target truth or a hard safety gate. At each history
it constructs one terminal set, unions in the ordinary experiment candidates,
and uses that same set for the current value, every hypothetical update, and
the realized post-update action. An optional frontier closure adds the current
Bayes action, minimum theory-margin action, minimum expected-violation action,
and a supported high-disagreement action to the experiment set. V18a tests
shared-pool consistency alone; V18b additionally tests frontier closure. The
two cells use identical seeds and budgets.

## V18 Shared-Pool And Frontier Results

Both V18 cells completed all 14 runs. They preserve FactorShock but do not
improve on V17:

| Cell | FactorShock feasible | Inventory feasible | Inventory mean violation |
|---|---:|---:|---:|
| V17: local expert + smooth risk | 7/7 | 3/7 | 0.02870 |
| V18a: one shared terminal pool | 7/7 | 2/7 | 0.01998 |
| V18b: shared pool + four frontier actions | 7/7 | 2/7 | 0.01877 |

The implementation repair is still retained: exact KG must value the same
history-measurable action set before a hypothetical observation, after every
hypothetical update, and after the realized update. Its Lean bridge is in
`SharedTerminalPoolKG.lean`. Performance, however, rejects frontier closure as
the missing ingredient. Frontier actions were actually selected in 10--40%
of Inventory iterations and 30--60% of FactorShock iterations, so the null
result is not an inactive-code artifact.

Truth is used only after scoring to audit ranks. In V18b every Inventory run
contains a true-feasible action in the acquisition pool, yet the
highest-scored true-feasible action has mean rank about 7.5--17.3 across
seeds. The selected score exceeds the best feasible score by 0.109--0.387.
V18a shows the same pattern. Thus the failure is no longer attributable to
candidate absence, separate terminal pools, or failure to expose posterior
frontier actions as experiments. The exact-KG score itself systematically
ranks the available safe actions too low.

## Systematic Failure Decomposition Through V18

The negative results now support a narrower hierarchy of causes:

1. **Not target leakage.** The mainline audit disables target-specific axes,
   initial designs, refinement formulas, state anchors, truth, and simulator
   queries outside the charged budget. Source records and the frozen LODO
   prior are the only transferred labels.
2. **Not candidate coverage.** V17--V18 pools contain feasible and low-regret
   actions in nearly every iteration. The optimizer sees them but scores them
   below unsafe actions.
3. **Not a missing final gate.** Hard certification, empirical replicate
   thresholds, smooth Bayes risk, task-adaptive fallback, and guarded
   calibration have all been isolated. Gates can exchange false negatives for
   false positives but do not repair the posterior ordering.
4. **Not posterior collapse alone.** Prior-protected V15 kept multiple experts
   alive and still failed. The surviving expert family can jointly misorder
   the target boundary.
5. **Model discrepancy matters but is incomplete.** The source-admissible
   local-risk kernel reduces Inventory mean violation by 73.3% under the hard
   path and becomes complementary with smooth risk in V17. Six frozen RBF
   landmarks nevertheless do not yet recover the safe side reliably. A
   post-score representation audit rejects missing landmark coverage as the
   explanation: the recurrent safe segmented profile has nearest-center
   distance `1.067` and maximum activation `0.978`. Instead the frozen median
   lengthscale is `5.033`, making safe and unsafe profiles almost
   indistinguishable. The safe profile and seed-1 unsafe segmented profile
   have feature cosine `0.99923` and Euclidean feature distance `0.0712`.
   Across 367 unlabeled Inventory policies the six centered RBF features have
   effective rank only `2.57` and condition number `53.9`. The discrepancy
   expert is therefore over-smoothed, not under-covered.
6. **One unresolved estimator/model fork remains.** Exact KG currently uses
   two common Monte Carlo draws and clips negative estimated gains. It also
   samples expert identity from the nominal posterior although the terminal
   loss is KL-robust. The V18 score gap can therefore arise from categorical
   and Gaussian Monte Carlo noise, or from a stable but wrong posterior risk
   model. Those explanations require different repairs.

## V19 Exact-KG Estimator Audit

V19 does not propose a new algorithm. It replays three representative V18b
Inventory seeds: seed 0 (successful), seed 1 (large false-feasible error), and
seed 4 (small boundary error). Each replay creates the stage-19 checkpoint and
immediately rescores the same next-step candidates on the same compute node
under four schedules: IID MC2, IID MC8, antithetic MC2, and antithetic MC8.
Raw gains are retained before clipping. Synthetic truth is joined only after
all scores are fixed.

The diagnostic decision rule is predeclared:

- If MC8 or antithetic sampling consistently moves the best feasible action
  near the top and changes the selected action, estimator variance is the next
  repair. The finite expert identity should then be integrated exactly or
  stratified instead of sampled twice.
- If all four schedules preserve the low feasible rank and positive score
  gap, more MC is rejected as the main fix. The next challenger must enlarge
  or recalibrate the latent discrepancy model, with special attention to
  landmark coverage of the Inventory safe profile.
- No V19 replay is eligible for promotion because it adds no decision rule;
  it is a causal diagnostic only.

Runtime checkpoints are now registered as scheduler `ckpt_dir` inputs. The
benchmark and post-run audit execute within one node-local shard, while only
JSON/CSV results are synchronized. This prevents code staging from deleting
unregistered checkpoint directories and keeps checkpoint payloads off the
local machine.

## Oracle-Only Identifiability Audit Of Inventory Coordinates

An additional post-score audit asks whether the current six-dimensional risk
coordinate is expressive enough even with abundant target truth. It is not an
algorithm input and consumes no benchmark evaluation: target mean and variance
oracles are called only after the V18 decisions are frozen.

Among 879 random and universal-shape Inventory policies, only 1.25% are truly
chance-feasible. The standardized coordinate matrix has rank five because one
occupancy block is compositional. With a 70/30 oracle train/test split, linear
ridge reaches test `R^2=0.51`; isotropic RBF kernel ridge reaches at most
`R^2=0.70`; and a random forest reaches `R^2=0.74`. Feasible-vs-infeasible AUC
ranges from `0.76` to `0.93`, but the highly imbalanced boundary remains hard:
the random forest predicts no feasible test point.

More importantly, the coordinate map contains cross-boundary collisions.
There are feasible and infeasible policies only `0.09` apart in standardized
coordinate distance whose true chance margins differ by about `0.057--0.103`.
Several pairs contain nearly identical control blocks assigned to different
ordered coordinate roles. In this synthetic domain the first, second, and
third policy blocks mean stock, reorder, and safety rather than literal time;
in trajectory domains the same information loss becomes temporal order.
Aggregated occupancy and the three-dimensional PCA exposure retain how much
of each regime is present but can discard which role or path position carries
it.

This rejects a scalar-bandwidth-only repair. The general replacement, if V19
also rejects Monte Carlo variance as the primary cause, is an ordered
cumulative-risk coordinate:

\[
\psi_R(x)=\{A_r(x),N_r(x)\}_{r=0}^{R},\qquad
A_r(x)=\sum_t b_r(t)\,a(s_t,x_t),
\]

where `b_r` are frozen orthogonal positional/temporal functions or finite
path-signature blocks. The posterior learns the active ordered rank, frequencies, metric,
and coefficient shrinkage from source domains plus charged target
observations. Aggregate occupancy is the `r=0` special case. This is a
domain-invariant assumption shared by traffic phases, inventory trajectories,
and queue bursts; it does not require a target label such as "Inventory".
The same ordered exposure must drive the GPR basis, cumulative HVD,
certification, candidate inversion, and exact KG for the high-dependence claim
to remain intact.

## Matched Representation Ablation

A second truth-only audit keeps the policy pool, train/test split, and
ExtraTrees regressor fixed while changing only the representation.  It is a
model-class diagnostic, not a reported optimization result:

| Representation | Test margin R2 | Feasibility AUC |
|---|---:|---:|
| current six-dimensional aggregate `psi` | 0.773 | 0.809 |
| full 28-dimensional generic descriptor | 0.863 | 0.975 |
| eight ordered DCT moments only | 0.420 | 0.589 |
| global mean/std plus eight ordered DCT moments | 0.847 | 0.924 |

The full descriptor already contains segment means and variances, so the
failure is not the total absence of ordered information.  The current mainline
compresses that descriptor into three PCA local exposures or four alignment
semantics before HVD and certification.  That compression removes
role-specific information that the full descriptor can use.  Conversely, DCT
moments alone remove the global risk level and also fail.  The admissible next
model is therefore the joint coordinate

`global cumulative exposure + low-frequency ordered exposure`,

with source-learned frequency/rank and target-budget posterior shrinkage.  It
is not a target-domain formula: the same positional basis is frozen for every
held-out domain, and only charged observations update its coefficients.

Only 1.25% of the audited Inventory pool is truly chance-feasible.  Despite
high AUC, none of the ordinary regressors predicts a feasible point at its
default threshold.  This shows why global squared-error or AUC alone is an
insufficient training objective.  The sequential posterior must explicitly
weight the chance boundary and control effective rank below the target sample
count.

## Failure Status Before The V19 Fork

| Candidate cause | Status | Discriminating evidence |
|---|---|---|
| no feasible candidate is generated | rejected | every V18 Inventory acquisition pool contains a true-feasible action |
| exact KG values a different terminal action set | rejected | V18 uses one shared measurable pool before and after every update |
| frontier actions cannot be sampled | rejected | V18b adds them and selects them in 10--40% of Inventory rounds |
| final hard gate alone causes failure | rejected | hard, smooth-risk, calibrated, and replicated variants all preserve misranking |
| posterior collapse is the sole cause | rejected | V15 protects expert mass and still fails |
| local model discrepancy is irrelevant | rejected | V16 lowers Inventory mean violation by 73.3% |
| current aggregate risk coordinate is sufficient | rejected | cross-boundary collisions and the matched representation ablation remain with oracle labels |
| finite-expert/normal MC noise is the primary cause | open until V19 | same checkpoint is being rescored with IID/antithetic MC2/MC8 |
| theory certificate recognizes empirical safe points | rejected for current model | even all seven safe FactorShock recommendations remain uncertified |

This ordering matters.  A gate or penalty can only change a decision after the
model has ranked risk correctly.  If V19 rejects estimator noise, the next
challenger must repair the common ordered cumulative coordinate used by the
GPR, HVD, certificate, proposal, and KG; adding a separate recommendation
feature would weaken the high-dependence claim and is not an admissible fix.

## V19 Exact-KG Estimator Results

All three checkpoint replays completed without simulator calls.  The table
reports the median highest-scored feasible rank and score gap across Inventory
seeds 0, 1, and 4.  Rank correlation is against the IID-MC2 score vector on
the identical candidate set.

| Sampling schedule | Selected feasible | Median feasible rank | Median score gap | Median Spearman vs IID-MC2 |
|---|---:|---:|---:|---:|
| IID MC2 | 1/3 | 11 | 0.04534 | 1.000 |
| IID MC8 | 0/3 | 17 | 0.04839 | 0.430 |
| antithetic MC2 | 0/3 | 13 | 0.04549 | 0.483 |
| antithetic MC8 | 0/3 | 7 | 0.01246 | 0.547 |

The predeclared outcome is `mixed_estimator_and_representation`.
Antithetic MC8 materially improves the median safe rank and reduces the score
gap by 72.5%, so estimator noise is real.  In seed 1 it moves the safe action
to rank two with a gap of only `0.000002`.  It still selects no safe action in
all three seeds, and the median score/safety rank correlation is only `0.294`.
Increasing IID draws alone is not monotone and makes the median safe rank
worse.  Therefore neither "MC noise only" nor "representation only" is
consistent with the audit.

Between 24.6% and 30.2% of the raw gains are negative depending on schedule,
but every winner has a positive raw gain.  Clipping negative gains does not
change these winners and is rejected as the direct cause.  The finite task
posterior has effective expert count about `2.3--2.8`, while two categorical
draws cover only about `1.6` distinct experts in expectation.  V20 therefore
makes one estimator-only change: enumerate every finite expert exactly under
its posterior decision weight and use common antithetic Gaussian draws within
each expert.  This Rao-Blackwellized replay remains offline and uses the same
stage-19 checkpoints.  The ordered cumulative-coordinate challenger remains
blocked until V20 quantifies how much misranking survives after categorical
noise is removed.

## V20 Stratified Finite-Expert Results

V20 evaluated two independent antithetic Gaussian repeats per checkpoint while
enumerating all finite experts exactly.  It used the same candidate sets and
made zero simulator calls.

| Estimator | Rows selecting feasible | Median feasible rank | Median score gap | Median score/safety Spearman |
|---|---:|---:|---:|---:|
| stratified expert, conditional MC2 | 0/6 | 13 | 0.03161 | 0.0877 |

Exact categorical integration does not recover the safe action.  It lowers
the median gap relative to IID MC2, but is worse than antithetic MC8 and never
moves a safe action into the near-top set.  Different Gaussian repeats still
change winners and move feasible ranks between 6 and 14.  The finite-expert
sampling defect is therefore removed as a primary explanation; conditional
Gaussian noise remains, but spending more draws cannot repair the low
score/safety alignment already visible under MC8.

V21 consequently targets the representation failure.  It adds a source-only,
dimension-invariant ordered cumulative coordinate consisting of global policy
mean/std plus a source-selected low-frequency cosine basis.  No target-domain
block formula or target truth selects the frequencies.  The ordered local
exposure and the existing shared-shock occupancy jointly define one
`psi_ordered=(A_ordered,N)` consumed by the expert GPR quadratic basis,
factor-HVD, theory certification, and exact KG.  A manifest-shard runner fixes
all other V18b settings so the IID cell changes representation only; a second
cell adds the now-proved stratified estimator separately.

## V21 Oracle-Only Coordinate Check

Before reading V21 optimization outcomes, a frozen-candidate oracle audit was
run on held-out Inventory.  It trained no algorithm and supplied no truth to a
decision: 1,262 source-only random/profile/proposal candidates and all feature
maps were fixed first, then target truth was joined for a 60/20/20 ridge
identifiability split.  The true feasible rate was 1.35%.

| Coordinate | Feature dimension | Fit rank | Test R2 | Feasibility AUC | Predicted feasible |
|---|---:|---:|---:|---:|---:|
| current aligned compact coordinate | 4 | 4 | 0.190 | 0.444 | 0 |
| ordered linear coordinate | 7 | 7 | 0.586 | 0.520 | 2 |
| ordered diagonal-quadratic coordinate | 11 | 11 | 0.871 | 0.985 | 1 |
| ordered cumulative quadratic coordinate | 17 | 17 | 0.885 | 0.992 | 3 |
| full generic descriptor | 28 | 25 | 0.911 | 0.981 | 3 |
| uncompressed aggregate coordinate basis | 52 | 34 | 0.916 | 0.979 | 2 |

The source-only frequency learner selected frequencies 4 and 1 from two
source domains.  This supports the V21 hypothesis that ordered exposure fixes
a real representation collision: it nearly matches much larger coordinates
while retaining only 17 quadratic features.  It also predeclares the next
failure mode.  Rank 17 is still high relative to `N=20` and much higher than
the preferred `0.35 N` effective-rank target.  If V21 fails despite the oracle
capacity result, the next admissible repair is source-learned hierarchical
interaction/square selection or posterior sparsity inside the ordered basis,
not another chance threshold and not a target-specific Inventory block map.
The low-rank ablation sharpens that repair: seven linear features lose the
boundary (`AUC=0.520`), whereas retaining only main effects and coordinate
squares uses 11 features and recovers `AUC=0.985`.  Full pairwise interactions
are unnecessary.  A diagonal-quadratic ordered basis with posterior sparsity
capped near effective rank seven is therefore the predeclared V22 model if the
currently running V21 full-quadratic gate does not pass.

V21 is explicitly an admission test, not a promotable final architecture.  It
adds the ordered expert to the existing six-expert posterior, so it also
changes prior normalization and proposal allocation.  If it shows a positive
signal, the required causal follow-up replaces `local_risk_kernel` with
`ordered_cumulative` while holding expert count, initial-design slots, and
simulation budget fixed.  Keeping both merely because the larger ensemble
wins would recreate the engineering-stacking failure mode and is not allowed
as the paper mainline.

## V21a Ordered-Expert IID Gate Results

All 14 IID-MC2 shards completed.  V21a preserves the stable domain but fails
the held-out Inventory gate:

| Domain | Initial design has feasible | Final true feasible | Mean violation | Median feasible regret | Ordered expert median weight |
|---|---:|---:|---:|---:|---:|
| FactorShock | 7/7 | 7/7 | 0.00000 | 0.00825 | `1.7e-9` |
| Inventory | 6/7 | 0/7 | 0.02296 | n/a | 0.0124 |

No run is posterior-certified.  Inventory is worse than V18b's `2/7`
feasible recommendations and `0.01877` mean violation even though six initial
designs already contain a true-feasible point.  The source-only frequency
learner consistently chooses frequencies `(1,2)` when FactorShock is held out
and `(4,1)` when Inventory is held out, as expected under different LODO
source sets.  The ordered expert then loses cumulative predictive log score by
large margins and receives negligible posterior mass.

V21a therefore rejects adding a dense 17-feature ordered expert.  Together
with the oracle audit, it isolates an identification failure rather than a
capacity or candidate-coverage failure.  V22 is predeclared as a structural
replacement: diagonal quadratic ordered features (11 instead of 17), source
PIP plus target spike-and-slab with total effective dimension capped near
`0.35 N`, and replacement of the local-kernel expert so the ensemble remains
six experts.  V21b remains in progress solely to quantify the stratified
estimator interaction; it cannot retroactively promote V21a.

V22 deliberately does not transfer an ordered-coordinate HVD coefficient
prior.  This isolates low-rank mean/boundary identification from variance
transfer.  Its result JSON records both ordered GPR effective dimension and
its cap.  The next intervention is predeclared as follows:

- if the cap is violated, reject the implementation before interpreting any
  optimization outcome;
- if the cap holds but ordered mean/boundary ranking remains poor, repair
  representation or two-sided information coverage, not certification;
- only if mean ranking improves while cumulative variance or certification
  remains the limiting error may a source-only ordered HVD prior be tested;
- any such HVD-prior cell must be paired with V22 under the same manifest and
  cannot be folded into V22 after seeing its seeds.

## V21b Stratified-Estimator Gate Results

All 14 stratified-expert shards completed without failure.  This cell changes
the exact-KG estimator only relative to V21a: it sums every finite expert under
its posterior mass and uses two conditional Gaussian draws per expert.

| Domain | Initial design has feasible | Final true feasible | Mean violation | Median feasible regret | Ordered expert median weight | Median runtime |
|---|---:|---:|---:|---:|---:|---:|
| FactorShock | 7/7 | 6/7 | 0.01690 | 0.00825 | `1.9e-10` | 78.4 min |
| Inventory | 6/7 | 1/7 | 0.01733 | 0.00569 (one feasible seed) | 0.0161 | 50.5 min |

No recommendation is posterior-certified and neither domain has a recorded
false-feasible certificate.  Relative to V21a, stratification lowers Inventory
mean violation and recovers one truly feasible recommendation, but it costs
about 2.4--2.7 times the runtime.  It also breaks the formerly stable
FactorShock seed 1, reducing that domain from 7/7 to 6/7.  Thus exact expert
identity integration exposes a real estimator effect but fails both promotion
requirements and is rejected as the mainline schedule.

The FactorShock seed-1 recommendation pool still contains the known safe
anchor `[25,75,...,75]`.  Under the stratified sequential history its decision
margin is `0.961`, so the algorithm instead recommends a wave-profile policy
with true chance margin `+0.1183`.  This is a risk-ranking failure induced by
the changed exploration history, not a missing-candidate failure.  V22 keeps
IID MC2 and changes only ordered-coordinate identifiability; conditional
stratification remains an offline diagnostic.

## V22 Sparse Ordered-Replacement Gate Results

V22 replaces the local-kernel expert by one diagonal-quadratic ordered expert.
The ordered expert uses source PIPs, target-updated spike-and-slab coefficients,
and a total effective-dimension cap of `0.35 N = 7`.  All 14 IID-MC2 shards
completed without failure.

| Domain | Final true feasible | Mean violation | Median feasible regret | Ordered weight | Effective dimension / cap | Median runtime |
|---|---:|---:|---:|---:|---:|---:|
| FactorShock | 5/7 | 0.14427 | 0.00825 | 1.0000 | 7.000 / 7 | 18.3 min |
| Inventory | 2/7 | 0.00585 | 0.02026 | 0.0011 | 5.418 / 7 | 15.9 min |

The cap is therefore implemented and active; over-rank fitting is not an
explanation for this failure.  Inventory improves materially in violation and
recovers two feasible seeds.  Its five infeasible margins lie only between
`+0.00055` and `+0.01053`, so the ordered sparse model moves recommendations
toward the true boundary even when it does not cross it.  FactorShock regresses
in seeds 3 and 5 with margins about `+0.50`.  No run is posterior-certified.

Every final recommendation pool still contains a truly feasible point.  The
remaining error is ranking, not candidate coverage.  Two structural causes
are now separated:

1. V21a assigns essentially all FactorShock mass to `local_risk_kernel` and is
   7/7 feasible.  Removing that residual model in V22 forces the ordered expert
   to carry nearly all mass and loses two seeds.  Local adaptation is not a
   redundant ensemble decoration.
2. Coordinate-wise source PIPs do not transfer reliably.  With Inventory held
   out, the source prior supports one ordered square while target evidence
   promotes a different square and shrinks most remaining quadratic/shared
   terms.  The result uses only about 5.4 effective dimensions despite the
   oracle audit showing that quadratic ordered structure is necessary.

The next admissible model must not add a seventh expert or relax the chance
bound.  It combines ordered structure and local adaptation inside one
orthogonal semiparametric expert, with an unlabeled fixed-pool projection that
removes the ordered span from local kernel features.  This preserves a single
identifiable direct-sum model while keeping the effective-dimension cap.

## V23 Orthogonal Semiparametric Gate Results

V23 combines the V22 ordered basis and six local RBF residual directions in a
single expert.  It keeps six experts globally and the same total effective
dimension cap.  All 14 IID-MC2 shards completed without failure.

| Domain | Final true feasible | False feasible | Mean violation | Ordered weight | Local inclusion mass | Effective dimension / cap | Median runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| FactorShock | 5/7 | 1/7 | 0.22943 | 1.0000 | 0.850 | 7.000 / 7 | 20.4 min |
| Inventory | 2/7 | 0/7 | 0.01531 | 0.0058 | 0.300 | 5.590 / 7 | 18.4 min |

The finite-pool projection is numerically correct: median relative ordered
cross error is below `3.1e-14`, and every fit obeys its rank cap.  Those facts
do not make the V23 feature extension safe away from the projection pool.
V23 forms each residual as `k(x)-phi(x)^T C`.  The RBF term is bounded, but the
subtracted ordered polynomial is not controlled by the finite-pool normal
equations at a new candidate.

This produces a concrete false certificate in FactorShock seed 1.  The chosen
candidate has recommendation leverage `3.18e7` relative to approximately one
for the known safe candidate.  Its constraint mean is extrapolated to `-2.02`,
the theory margin becomes `-0.593`, and the method certifies a point whose true
chance margin is `+1.232`.  FactorShock seed 3 has leverage `2.21e6` and true
margin `+0.374`.  Thus V23 fails at mean-model support extrapolation, not HVD,
candidate coverage, numerical orthogonality, or total rank.

Inventory gives the complementary result.  All six local residual PIPs remain
at their lower bound `0.05`, for total local inclusion mass `0.30`.  The new
component is effectively inactive, leaving the same two feasible seeds as
V22.  V23 therefore does not test or solve source-to-target ordered-group
transfer for Inventory.

V24 changes only the invalid residual construction.  It uses more bounded RBF
centers and projects their coefficient vectors into the nullspace of
`Phi^T K`.  A residual is then `k(x)^T p`, not `k(x)-phi(x)^T C`; it remains
orthogonal on the frozen unlabeled pool and has a global finite norm bound at
every candidate.  The expert count, PIP update, total rank cap, certification,
candidate pool, MC schedule, and budget remain fixed.  Only after this repair
restores FactorShock may ordered group/subspace transfer be challenged for
Inventory.

## V24 Bounded-Nullspace Gate Results

V24 replaces V23's pool-external polynomial subtraction by a coefficient
nullspace projection.  Every residual is a bounded linear combination of RBF
features.  All 14 shards completed, every projection error is below `1.3e-14`,
and every adaptive fit obeys its total dimension cap.

| Domain | Final true feasible | False feasible | Mean violation | Ordered weight | Local inclusion mass | Effective dimension / cap | Median runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| FactorShock | 4/7 | 0/7 | 0.22458 | 1.0000 | 0.443 | 7.000 / 7 | 20.7 min |
| Inventory | 2/7 | 0/7 | 0.03348 | 0.0246 | 0.300 | 5.592 / 7 | 17.9 min |

The V23 safety bug is fixed: no V24 recommendation is posterior-certified,
and FactorShock seed 1 no longer has an extreme negative constraint mean or a
false certificate.  This does not restore optimization quality.  FactorShock
fails in seeds 0, 1, and 5; Inventory retains exactly two feasible seeds and
all six local residual PIPs remain at their minimum `0.05`.

Together V23 and V24 reject the direct-sum hypothesis.  Local residual feature
activation is neither necessary nor sufficient for a safe FactorShock
recommendation, and it is inactive on Inventory.  Even minimum nonzero
inclusion changes posterior geometry and sequential histories enough to make
V24 worse than V22.  The residual should not be rescued by a leverage guard,
different center count, or another PIP threshold.

The remaining admissible structural hypothesis is model uncertainty rather
than feature stacking.  A latent structure variable should choose between a
local model and an ordered cumulative model using only source data plus
budgeted target observations.  Inside the ordered branch, source transfer
must operate on semantic groups or low-rank subspaces rather than individual
coordinate PIPs, because V22 already showed source and target square supports
do not coincide.  The next truth-only audit tests whether rotation/permutation
invariant group summaries retain the held-out boundary before any new KG gate
is submitted.

## V25 Latent-Structure Gate Results

V25 treats local and ordered cumulative models as mutually exclusive finite
task hypotheses instead of concatenating their features.  It keeps exactly six
experts and all V22 budgets.  All 14 shards completed.

| Domain | Final true feasible | False feasible | Mean violation | Median feasible regret | Median runtime |
|---|---:|---:|---:|---:|---:|
| FactorShock | 7/7 | 0/7 | 0.00000 | 0.00825 | 17.0 min |
| Inventory | 3/7 | 0/7 | 0.01016 | 0.00977 | 15.1 min |

FactorShock returns to the stable V12 result.  Its local-expert posterior
weight is `0.9999--1.0` in every seed, so the task posterior learns a genuine
domain-level structural distinction rather than averaging incompatible bases.
Inventory improves by one feasible seed over V22 and remains free of false
certificates, but misses the 4/7 gate.

The remaining Inventory failure is inside the ordered branch.  A truth-only
frozen-pool audit shows that pooling all ordered features destroys useful
directional information: fully invariant ordered summaries have chance-boundary
AUC `0.611`, while the `A^2` block alone must retain a learned target direction.
In contrast, grouping only shared exposure `N` retains AUC `0.992`; diagonal
ordered mean plus cumulative factor variance reaches AUC `0.984`, whereas a
linear mean plus the same variance reaches only `0.511`.  Thus quadratic local
exposure is needed in the mean model as well as HVD.

V26 consequently transfers only block strength: one inclusion probability and
one isotropic slab scale for the `A^2` block, and another pair for the shared
`N` block.  Target observations still learn every coefficient direction.  It
keeps V25's latent local/ordered structure, six experts, total rank cap,
candidate pool, exact-MC schedule, and certification unchanged.

## V26 Attempt-A Implementation Rejection

The first semantic group-shrinkage gate returns FactorShock 7/7 and Inventory
3/7 with no false-feasible certificate.  These quality counts are not eligible
for promotion because the rank invariant fails in 5/7 FactorShock and 3/7
Inventory seeds.  A fixed logit-shift bracket of `50` cannot project joint
group Bayes factors near `95`; the optional inclusion mass becomes `3.05`
against a cap of `3.0`.

The bug is repaired before any interpretation: cardinality projection now
constructs its upper bracket from the maximum finite posterior logit, tests
include a group Bayes factor above 50, and the gate requires zero per-seed
dimension violations.  V26b reruns the same paired matrix.

Even before V26b, Attempt A identifies the likely next statistical failure.
Inventory's curvature group is almost always assigned its lower-bound PIP,
while the shared-exposure group saturates.  This reproduces V25's 3/7 outcome
and makes the ordered expert's median task weight only about `1e-4`.  If the
cap-correct run confirms this pattern, a finite hierarchical posterior over
semantic group structures is the next admissible model; another threshold or
feasibility guard is not.

## V26b Corrected Group-Shrinkage Result

V26b enforces the rank invariant in every seed.  It returns FactorShock 7/7
and Inventory 3/7, with no false-feasible certificate in either domain.  The
quality outcome is therefore unchanged from V25 after repairing the V26
implementation defect.

An offline checkpoint audit uses only each run's already charged target
observations for nested-LOO selection; analytic truth is joined only after the
candidate pool and model family freeze.  It shows a reproducible task
difference:

| Held-out domain | Median selected group-ridge effective df | Full diagonal chance AUC | Linear/shared chance AUC |
|---|---:|---:|---:|
| FactorShock | 3.35 | 0.984 | 0.986 |
| Inventory | 8.03 | 0.928 | 0.476 |

Inventory needs directional curvature, while FactorShock does not need a
larger ordered model.  A universal hard cap of seven total effective
dimensions leaves only three optional dimensions after the four-feature fixed
prefix; the three-dimensional `N` block then excludes `A^2`.  In seed 5 this
happens even though the curvature group Bayes factor is positive (`+11`),
because the shared group Bayes factor is about `+200`.

V27 replaces the hard cap and group inclusion competition with target-updated
continuous group ridge penalties selected by full nested refits.  It retains
the six-expert latent task model and does not alter certification.

## Systematic Failure Chain Through V27

The failed variants are not independent tuning accidents.  They expose four
separate identifiability failures in sequence:

1. A shared coordinate prior can predict source observations while assigning
   the wrong semantics to a held-out chance boundary.
2. Concatenating local and ordered dictionaries lets one incompatible basis
   contaminate every posterior; V25 therefore represents them as competing
   latent task hypotheses.
3. Coordinate-wise or group spike/slab support is too discontinuous at
   `n0=10`: a strong shared-exposure Bayes factor can exclude a weaker but
   necessary curvature block.
4. A universal hard effective-rank cap is not transferable.  The charged-data
   audit selects median df `3.35` on FactorShock but `8.03` on Inventory.

V27 addresses only the fourth failure while preserving the controls that fixed
the first three.  It learns continuous block penalties from full nested target
refits, then lets the finite latent task posterior decide whether that ordered
model should influence the decision.  A target-domain label, analytic truth,
or post-hoc success gate is never an input to either update.

## V27 Full-Gate Failure: Predictive Fit Is Not Safe Ranking

V27 passes every implementation and leakage check but returns the same
Inventory feasibility count as V26b: 3/7.  FactorShock remains 7/7.  The
candidate audit excludes proposal support as the cause: every Inventory pool
contains truly feasible and low-regret points.

The terminal-ranking audit freezes each final posterior and pool before truth
is joined.  Its main findings are:

- the ordered cumulative expert has median true-feasibility AUC `0.882`;
- universal and source-spectral experts also have median AUC `0.925` and
  `0.862`, but their quality changes sharply by seed;
- the task posterior is trained mainly by joint predictive likelihood, so it
  can put high mass on the wrong boundary-ranking expert;
- every current Bayes-risk action is already observed, often after only one or
  two noisy evaluations, while the better feasible action is frequently
  unobserved;
- all final theory margins remain positive, so the action is selected by the
  smooth Bayes-risk fallback rather than by a certified feasible set.

This creates two coupled biases.  First, predictive density fit and
chance-boundary ranking are different tasks.  Second, selecting among noisy
visited points reuses favorable noise and creates a winner's-curse effect.
Changing only the terminal penalty cannot repair the sequential data path.

The frozen-posterior policy audit confirms that conclusion.  Penalties from 5
to 100, minimum robust expected violation, minimum theory margin,
prior/posterior averages, median/trimmed expert aggregation, and rank consensus
all remain at at most 3/7.  These alternatives are diagnostic upper-level
hypotheses only and are not promoted.

## V28 Seed-0 Rejection: Safe Ranking Is Not Safe Selection

V28 separates the multichannel predictive posterior `Q_pred` from a
constraint/boundary-ranking posterior `Q_safe`. Python and Lean verification
pass, and the safe update is active rather than cosmetic. On Inventory seed 0,
`TV(Q_pred,Q_safe)=0.784`; `Q_safe` assigns `0.822` mass to the universal
coordinate expert instead of the source-spectral expert preferred by
predictive likelihood.

That correction does not pass the controlled smoke. FactorShock remains
feasible with regret `0.00825`, but Inventory changes from V27's structured
feasible anchor (true margin `-0.00142`) to a better-objective, nonstructured
observed action with true margin `+0.000551`. The action is not falsely
certified: every theory margin is positive, so the Bayes-risk fallback makes
the decision.

A second frozen checkpoint audit rules out candidate coverage. The Inventory
terminal pool contains 11 truly feasible policies and one has true margin
`-0.0509`. The selected policy has only one charged observation. The
safe-posterior nominal expected-violation minimizer is itself truly feasible
with margin `-0.0173`, but is unobserved. Expert AUCs also show that the
ordered cumulative expert ranks feasibility well (`0.929`) despite receiving
negligible final mass, while the dominant universal expert reaches `0.849`.

Thus V28 repairs expert-task alignment but exposes the next layer: finite
candidate selection under noisy labels. A static penalty cannot distinguish a
one-sample boundary winner from a genuinely safe finalist. V29 therefore
reserves part of the unchanged evaluation budget for posterior-frozen finalist
replication and uses an explicit mean-estimation confidence radius in the
fallback decision. It does not add another representation, expert, or
domain-specific rule.

## V29 Seed-0 Rejection: A Safe Race Needs A Safe Finalist

V29 passes all 201 Python tests, all 8618 Lean build jobs, checkpoint resume,
and strict budget accounting. Its last three stages reserve evaluations inside
the original `N=20`; forced stages skip unused exact-KG computation. The
FactorShock result remains feasible with regret `0.00825`.

Inventory remains infeasible at the same true margin `+0.000551`. This is not
caused by the replicated safety rule selecting the wrong member of a valid
set. At stage 17, the frozen finalists are:

| Posterior-only label | True margin after freeze | Replicates | Final empirical upper margin |
|---|---:|---:|---:|
| minimum Bayes risk | `+0.000551` | 2 | `0.3523` |
| minimum nominal `Q_safe` violation | `+0.05509` | 2 | `0.3610` |

Thus the finalist set contains no truly feasible action. The safety-first rule
correctly chooses the smaller empirical upper margin, but replication cannot
repair `F_t intersect Safe = empty`. In contrast, the final terminal pool
contains feasible points after later posterior/proposal updates. The rejected
assumption is one-time early freezing, not the value of replication itself.

The next admissible hypothesis is a history-measurable adaptive race: each
charged observation may update `F_t`, and frozen source-prior support should
nominate at least one safety candidate from each structural expert before
successive elimination. A family-wise confidence rule can then control false
selection over the evolving finite set. This is a new algorithmic choice and
is not silently introduced as a V29 parameter tweak.

## V30 Seed-0 Pass: Preserve Structural Support In The Decision Set

A stage-18 frozen audit sharpens the V29 diagnosis. The terminal pool contains
eight truly feasible Inventory candidates. The dominant universal expert
nominates an unsafe action with true margin `+0.05509`; the ordered cumulative
expert, despite posterior mass near `3.3e-5`, nominates a feasible action with
true margin `-0.03612` and the lowest expert-specific predicted violation.
The useful safe direction was learned, but mixture averaging deleted it before
the finite finalist race.

V30 therefore changes only finalist support: every finite source-supported
expert may nominate one action, and nomination eligibility is independent of
posterior mass. In the controlled seed-0 smoke, Inventory selects the ordered
nomination and becomes feasible with regret `0.00569`; FactorShock stays
feasible with regret `0.00825`. Neither run claims a false certificate. This
supports the mechanism but is not promotion evidence; the predeclared 7+7
gate remains necessary.

## V30 Full-Gate Rejection: One-Time Nomination Is Another Hard Compression

V30 completes all 14 runs without retry. FactorShock remains `7/7` feasible
with zero violation and median regret `0.00825`. Inventory reaches only `3/7`
feasible, with zero false certificates and median feasible regret `0.00569`.
It therefore misses the fixed `4/7` promotion threshold and is not promoted.

The failure is not candidate coverage, complexity selection, or a return of
posterior false certification. Every failed Inventory terminal pool contains
five to eight truly feasible actions, and all nested group-ridge selections
are valid. The decisive compression occurs after those candidates exist:

1. every expert is reduced to its single minimum predicted violation action;
2. expert actions are compared by raw violation values whose scales are not
   calibrated across experts;
3. only one expert action enters a two-finalist set fixed before any reserved
   observation;
4. later labels may change the best expert nomination, but the set cannot
   change;
5. one incomplete finalist makes the whole replicated fallback unavailable.

Truth is joined only after the posterior and pool are frozen. That audit finds
the first feasible expert rank shown below:

| Seed | Best expert carrying safe support | First feasible rank |
|---:|---|---:|
| 1 | null universal | 1 |
| 2 | local risk kernel | 2 |
| 4 | local risk kernel | 2 |
| 5 | ordered cumulative | 1 |

The raw cross-expert score is actively misleading in seeds 2 and 4: an
ordered expert assigns near-zero predicted violation to a policy with true
margin about `+0.311`, while the local expert's second action has true margin
`-0.044` or `-0.051` but a larger numerical score. This rejects globally
sorting one proposal per expert by an uncalibrated score.

V31 is therefore restricted to a history-measurable adaptive race over the
existing expert-supported pool. It may refresh the challenger after each
charged observation and archives all directly tested actions. Recommendation
uses completed candidates only; no target truth, task label, static gate, or
new representation enters the decision.

## V31 Smoke Rejection: Adaptive Scores Need A Fixed Action Universe

The V31 state machine and its conservative fallback work as specified, but
the controlled smoke rejects the performance hypothesis. FactorShock seed 0
remains feasible at margin `-0.03320`; Inventory seed 0 remains feasible at
`-0.03612`; Inventory seed 1 remains infeasible at `+0.03731`.

In the failing seed, stages 17, 18, and 19 each regenerate the terminal pool
and nominate a different ordered action. Every challenger receives one paid
observation, none reaches two replicates, and the algorithm truthfully reports
`no_completed_finalist`. FactorShock exhibits the same churn with three unsafe
challengers, although its already completed Bayes action protects the final
recommendation.

The discrepancy with the earlier stage-18 audit is informative rather than a
contradiction. The audit reranks the `last_terminal_pool` saved before the
stage-17 observation and finds a safe null nomination. V31 reranks a newly
sampled stage-18 pool, where that action is absent. The next isolated change is
therefore to freeze the finite race universe at suffix entry and update only
its scores. V32 adds no model, prior, score, or evaluation.

## V32 Controlled Smoke Pass: Fixed Support Makes Refresh Effective

V32 freezes the stage-17 terminal universe and changes no other V31 component.
FactorShock seed 0 and Inventory seed 0 retain their feasible recommendations.
Inventory seed 1 changes from true margin `+0.03731` under V31 to `-0.05090`
with feasible regret `0.00564`.

The stored refresh history supplies the causal trace: an unsafe ordered action
is evaluated at stage 17; the updated models nominate the safe null action
from the same fixed universe at stage 18; stage 19 repeats it to the declared
minimum; the final completed-only race selects it. No target truth or
additional simulator call enters nomination. This passes the controlled smoke
and permits the unchanged 7+7 gate, but is not itself promotion evidence.

## V32 Full-Gate Pass And Promotion

All 14 V32 Gate-1 tasks complete without retry or failure. FactorShock is
`7/7` truly feasible with zero mean violation and median regret `0.00825`.
Inventory is `5/7` truly feasible, with zero false certificates, median
feasible regret `0.00569`, and mean violation `0.001582`. Every nested
group-ridge complexity selection is valid.

Against the V30 frozen-support challenger, V32 preserves FactorShock and raises
Inventory feasibility from `3/7` to `5/7`; Inventory mean violation drops from
`0.05306` to `0.001582`. Median wall time is non-worse in both domains. The
fixed-universe adaptive race therefore passes the predeclared promotion rule
and becomes the tracked LODO baseline. Inventory seeds 2 and 5 remain
uncertified infeasible outcomes and are retained explicitly for the next
failure analysis; the promotion does not claim universal convergence.

## V32 Queue Gate-2 Rejection: Target Adaptation Is Not Yet Meta-Generalization

The frozen Queue held-out run changes no V32 option and uses only FactorShock
and Inventory as source domains. It returns `3/7` truly feasible
recommendations, zero false certificates, median feasible regret `0.00455`,
mean violation `0.05767`, and median true chance margin `+0.01829`.

This failure is not proposal support. Every one of the seven terminal pools
contains a truly feasible action, yet no run produces a posterior-certified
action. Seeds 0, 1, 3, and 5 select infeasible actions with margins `+0.03834`,
`+0.04047`, `+0.01829`, and `+0.30658`. The dominant safe-posterior expert also
changes across failures (`local_risk_kernel` in seed 0 and
`universal_coordinate` in the others), so one expert-specific correction is
not supported.

The Gate-2 result separates the current claims:

1. V32 is an effective target-adaptive finite expert race on FactorShock and
   much of Inventory;
2. it does not yet establish a domain-general source prior;
3. adding a Queue-specific anchor, gate, or score would recreate the
   engineering-stacking failure;
4. the next admissible change is a shadow joint posterior over structural
   expert and source-trained sensitivity class, followed by a three-domain
   coherence audit before any decision rule changes.

## Joint Shadow Audit: The Missing Layer Is Task Coherence

Run `lodo_joint_shadow_v1_n20_3domain_20260712` reproduces all 21 frozen V32
recommendations and truth metrics exactly, so the added posterior has no
behavioral side effect. Its median structure-sensitivity mutual information is
near zero on FactorShock (`0.00003`), moderate on Inventory (`0.0786`), and
largest on Queue (`0.1473`). Joint and legacy robust reference actions agree
on `6/7`, `3/7`, and `0/7` seeds respectively.

This pattern supports a task-level coherence failure rather than a universal
need for more model capacity: FactorShock already identifies one coherent
structure, whereas Queue observations couple structural validity and error
sensitivity. At the same time, median expert-feasible mass at the chosen point
is zero in every domain. The joint posterior therefore cannot be promoted by
merely replacing expert weights. It must define the no-certificate Bayes loss
used by recommendation and exact KG while retaining a non-relaxing theory
certificate.

The isolated authoritative challenger implements exactly that change behind
`task_latent_inference_mode=authoritative`. Its first paired run is
`lodo_joint_authoritative_v1_n20_3domain_20260712` (`t29282..t29302`). It is
not a promoted baseline until FactorShock, Inventory, and Queue jointly pass.

## Authoritative V1 Rejection: Scale Is Not Signed Calibration

The predeclared paired gate rejects authoritative V1. FactorShock is unchanged
at `7/7`; Inventory drops from `5/7` to `3/7`; Queue drops from `3/7` to `2/7`.
Queue mean violation improves from `0.05767` to `0.03604`, but its feasible
regret worsens. No false certificate is introduced because theory
certification remains conservative.

Truth-only post-run audit shows that Queue's joint action is often much safer
than the legacy robust action and is truly feasible in seed 0, but the
replicated finalist can still replace it. Inventory and Queue also lose some
safe V32 actions after joint proposal/predictive sampling changes the search
trajectory. Most importantly, the latent class can only scale epistemic error;
it cannot distinguish a systematically conservative constraint mean from an
optimistic one. Increasing or tuning the class penalty cannot repair that
identifiability defect.

V2 therefore extends the same latent class with a signed standardized bias.
Its centers and prior weights are learned from source-domain LODO mean
residuals. The signed correction enters posterior likelihood and Bayes
decision loss only; the theory certificate ignores it and still floors
epistemic scaling at one. No domain-specific rule or held-out oracle enters
the extension.

## Authoritative V2 Rejection: Signed Error Is State Dependent

The four-case causal smoke preserves FactorShock seed 0 and repairs Inventory
seed 4, but fails the two Queue controls. Queue seed 0 remains infeasible at
margin `+0.01665`, and seed 6 changes from V32's feasible `-0.00860` to
`+0.03275`. Post-run calibration audit ranks the truly feasible terminal-pool
action safer than the selected action after state-dependent calibration. The
missing variable is therefore not another task-wide sensitivity constant: it
is the location of transferred error in the cumulative-risk state.

V3 represents that location by source-frozen profiles
`b_j(psi)=theta_j^T[1,A,Helmert(N)]`. Source LODO residuals are normalized by
domain RMSE, and the profile is applied in target predictive-standard-
deviation units. Helmert coordinates remove the exact intercept/simplex
collinearity that made an initial raw `[1,A,N]` implementation numerically
unstable. This is still one joint task latent, not a domain classifier or a
Queue-specific gate. Its predeclared four-seed smoke must pass before any full
matrix is run.

## Authoritative V3 Rejection: Calibration Must Be Expert Conditional

V3 passes all leakage, bounded-amplitude, Python, and Lean checks but fails the
four-case causal gate. It preserves FactorShock seed 0, loses Inventory seed
4, barely improves Queue seed 0 relative to V32, and fails to preserve Queue
seed 6. It is therefore not expanded or promoted.

This is not a candidate-support failure: every failed terminal pool contains a
true-feasible point. Nor is it a missing function-capacity failure: the
non-null profiles span positive and negative corrections of roughly one
predictive standard deviation. The defect is that one profile dictionary was
fit to a common source mean and then reused by six structurally different
GPR/HVD experts. Final target evidence mostly selects the null profile, while
the true-feasible pool action remains ranked more dangerous than the selected
action.

The next theoretical object is a hierarchical linear calibration posterior
per expert, with a common source-learned prior and target updates from charged
prequential residuals. This makes structural validity and calibration
function genuinely conditional on one another. A coefficient mean may change
Bayes ranking; its posterior covariance may only enlarge epistemic
certification uncertainty. The object must be cloned and updated inside exact
KG. Tuning V3's profile prior, temperature, or finalist score is explicitly
disallowed because it does not repair this hierarchy error.

## V4 Implementation Contract

V4 implements that object as `expert_ridge`, while retaining V3 as the
`source_profiles` ablation. One boundary-weighted source Gaussian prior is
copied into every structural expert. Paid target residuals update separate
precision/information pairs for each expert after the current prequential
score is recorded. The resulting means can differ because the expert GPR
means differ, even though all experts share the same source prior and risk
features.

The continuous posterior is not a hidden certificate relaxation. Its mean is
absent from the theory margin; its covariance contributes the nonnegative
term `predictive_sd^2 phi^T P^{-1} phi`. Gaussian KL is included in the task
ambiguity radius, and exact-KG fantasies clone/update `P`, `h`, GPR, and HVD
together. Fixed-universe expert nominations also use the same calibrated
decision moments. The next experiment is restricted to the same four causal
seeds used to reject V3.

## Authoritative V4 Rejection: Conditional Calibration Is Not Yet Transfer

The four-case causal smoke completes without runtime failures, false
certificates, or target leakage, but rejects V4 before the 21-seed matrix.
FactorShock seed 0 is unchanged and feasible at margin `-0.03320`. Inventory
seed 4 returns the V1/V3 infeasible action at `+0.01313`. Queue seed 0 worsens
from V32's `+0.03834` to `+0.06498`, and Queue seed 6 loses V32's feasible
action, moving from `-0.00860` to `+0.03275`.

This run separates search and recommendation failures. Queue seed 0's
sequential pool true-feasible coverage falls from `0.9` under V32 to `0.4`
under V4, despite a terminal true-feasible candidate. The expert-conditional
posterior therefore changes exact-KG fantasies enough to reduce useful search
coverage. Inventory seed 4 and Queue seed 6 retain true-feasible candidates in
every terminal recommendation pool, but the positive-margin Bayes loss ranks
them behind unsafe actions. Calibration thus also fails at terminal ranking.

The nonrelaxing certificate contract remains intact: coefficient means never
enter the theory margin, coefficient covariance is nonnegative, and all four
selected points are posterior-infeasible rather than falsely certified. The
failure is consequently not repaired by weakening the covariance guard. It is
evidence that six target-conditioned residual regressions with one shared
source Gaussian prior do not by themselves identify a transferable chance-
boundary coordinate from sixteen adaptive observations. Post-hoc tuning of
ridge strength, boundary weights, or expert penalties is prohibited. V4 stays
available as `task_latent_calibration_mode=expert_ridge` for ablation and
mechanism tests; `source_profiles` remains the default and V32 remains the
promoted performance baseline.

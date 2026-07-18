# Cumulative-HVD decision closure gate

## Objective

This experiment asks one narrow causal question: after freezing the latest
dimension-equivariant source proposal and using a neutral Sobol continuation,
does corrected cumulative factor-HVD improve calibration, certification, or
online decisions beyond pooled variance?

No target truth is available to proposal generation, posterior updates,
evaluate-or-replicate decisions, or terminal recommendation. Target oracle
quantities are computed only after each run for audit metrics.

## Frozen front end

- Proposal: `risk_objective_atlas/low_frequency_only`.
- Source dimension: 50.
- Target dimension: 1000.
- Source archive: 384 ordinary replicated simulator calls from the two
  non-target domains.
- Target initial design: the same frozen `n0=10` points in every matched cell.
- Target budget: `N=20`, so every replication consumes one of the same ten
  post-initialization calls as a new point.

## Causal factors

The end-to-end gate crosses three binary factors:

1. HVD: `pooled` versus corrected `factor_cumulative`.
2. Source adaptation: frozen source experts versus online source-discrepancy
   reweighting from budgeted target observations.
3. Action space: `sobol_new` versus `sobol_hvd_voi`.

`sobol_new` selects the nearest unevaluated point to a deterministic Sobol
stream. `sobol_hvd_voi` keeps exactly that same point as its sole new-point
action and compares it with eligible observed points. The comparison value is
boundary-weighted integrated reduction in HVD parameter uncertainty per unit
evaluation cost. Fresh residual squares are reliability-weighted by the
current aleatoric share; within-policy replications receive unit reliability.

Posterior dominance, certification rechecks, finalist replication, empirical
override, exact KG, TS, and target-specific refinement are disabled.

## Matrix

- FactorShock: shared-shock scales `0, 0.25, 1, 4`.
- Inventory and Queue: their registered cumulative-risk models.
- Five gate seeds per cell.
- Total: `2 x 2 x 2 x (4 + 1 + 1) x 5 = 240` target runs.

The existing isolated HVD-identifiability experiment supplies the replication
intensity sweep `r in {2,4,8,16}`. The end-to-end action gate instead lets HVD
choose replication adaptively, with a maximum of eight evaluations per policy.

## Reported outcomes

- true and posterior feasible counts;
- false-feasible and false-certificate counts;
- certificate nonvacuity, precision, and recall;
- held-out variance log-RMSE, certified upper coverage, and variance ratios;
- `n0-best -> final` regret change;
- adaptive rescue, preservation, improvement, and loss;
- selected new-point and replication counts;
- paired HVD, discrepancy, and action effects.

## Promotion rule

The 20-seed extension is submitted only if the registered primary cell
`factor_cumulative + adaptive discrepancy + HVD-VOI` satisfies all five rules:

1. all 240 gate results parse successfully;
2. zero false certificates;
3. at least one nonvacuous posterior certificate;
4. median post-run variance upper coverage is at least 0.90;
5. adaptive rescues or improvements outnumber adaptive losses.

Failure means that cumulative HVD remains a statistically identifiable
variance model or appendix result, not a promoted end-to-end optimization
contribution. Passing advances seeds 5 through 19 under the same frozen source
archive and protocol; it does not authorize a `d=10000` claim by itself.

## Five-seed result and registered repair

The complete `240/240` end-to-end matrix did not pass promotion. The primary
cell produced no nonvacuous certificates, no online improvements, and two
adaptive losses. HVD-VOI spent 299 of 300 post-initialization evaluations on
replication. Inventory and Queue nominal variance was overestimated by roughly
9x and 13x at the median, respectively.

The separate controlled matrix completed `160/160` runs and isolates the
failure. With ordinary replicated target designs, cumulative factor-HVD beat
pooled HVD by `-0.276` median log-variance RMSE. At 16 replicates it achieved
variance rank correlation above `0.95`, full upper coverage, and zero false
certificates at every registered shared-shock strength. Thus the decomposition
is identifiable; the failed component is its transfer/action integration.

Two repairs are registered before any 20-seed expansion:

1. When a normalized source HVD shape is present, target replications update
   only one positive scale parameter. They cannot refit the full cumulative
   coefficient vector from four or five unique policies.
2. `sobol_joint_voi` compares the same one Sobol new point with eligible
   replications using expected reduction in chance-margin uncertainty. It adds
   exact finite-feature GPR constraint-variance reduction to the delta-method
   HVD contribution `z_alpha^2 Var(v_C) / (4 v_C)`. The legacy HVD-only action
   remains unchanged as an ablation.

The repaired gate first compares `sobol_new` with `sobol_joint_voi` under
`factor_cumulative + adaptive discrepancy` for the same six strata and five
seeds (`60` runs). It advances to 20 seeds only if certificates become
nonvacuous, false certificates remain zero, and online gains exceed losses.

The first repaired gate completed `60/60` but also failed promotion. Joint VOI
produced zero nonvacuous certificates, zero online gains, and three adaptive
losses; it chose 296 replications and four new points. Compared with new-only,
it had one feasibility win, two feasibility losses, and a `+0.361` median
log-variance RMSE change. Inventory and Queue nominal variance remained about
10x and 12x too large, so scalar calibration alone does not repair a
cross-domain shape mismatch.

An implementation audit then found that the GPR component of the first joint
VOI gate used the conservative certification variance in its observation-noise
denominator, whereas the actual GPR rank-one update uses nominal HVD variance.
This systematically undervalues new policies. That run is retained as the
`certification-noise` conservative diagnostic, not as the final joint-VOI
test. The exact-update version uses nominal observation variance, records the
selected GPR and HVD information contributions separately, and must repeat the
same 60-cell gate before any 20-seed decision.

The nominal-noise gate also completed `60/60` without passing promotion. It
selected 294 replications and six new policies, produced no nonvacuous
certificates or online gains, and incurred three adaptive losses. Its component
diagnostics exposed a second unit mismatch: the median selected GPR posterior
variance reduction was `0.00789`, while the median HVD term was `62.8`. The HVD
term had used a median-normalized IRLS information matrix. That matrix is a
stable numerical geometry for fitting the coefficient vector, but it is not a
posterior covariance in physical variance units and therefore cannot be added
to GPR variance reduction.

For the frozen normalized source-shape model `v(x)=s h(x)`, the final registered
repair replaces that quantity with the scaled-chi-square Fisher covariance of
the single target scale. With source pseudo degrees of freedom `nu_0`, target
replication degrees of freedom `nu_t`, and action reliability `r`,

`P_s=(nu_0+nu_t)/(2s^2)`, `Delta P_s=r/(2s^2)`.

The integrated HVD variance reduction is then
`Delta P_s/[P_s(P_s+Delta P_s)] E_w[h(x)^2]`, followed by the same delta-method
chance-margin conversion. This expression is unit-tested against its closed
form and is in variance-squared units. A third 60-run gate repeats exactly the
same strata, seeds, proposal, initial designs, and target budgets. The 20-seed
extension remains conditional on the unchanged promotion rule.

The Fisher-unit gate completed `60/60` but again did not pass promotion. It
produced no nonvacuous certificates or online gains, incurred three adaptive
losses, and selected 294 replications versus six new policies. Relative to
new-only it had one feasibility win, two losses, no regret wins, four regret
losses, and a `+0.387` median log-variance RMSE change. The new diagnostics
showed that the unit correction worked for Inventory, but exposed one remaining
forward-model inconsistency. For example, the selected HVD margin information
in seed 0 was about `0.111` for Inventory yet `252` for FactorShock.

The cause is the residual-variance cap. Prediction uses
`v(x)=clip(s h(x), floor, cap)`, while the Fisher VOI still differentiated the
unclipped `s h(x)`. Source shapes that saturate the cap therefore received a
large fictitious derivative. The cap-consistent repair uses derivative `h(x)`
only when `floor < s h(x) < cap` and zero derivative otherwise, for both the
action likelihood and the reference-pool terminal margin. A mixed saturated
and unsaturated unit test verifies the resulting closed form and zero VOI for
saturated actions. One final 60-run consistency gate is allowed before the
pre-registered decision; failure still blocks the 20-seed extension.

## Final decision

The cap-consistent gate completed `60/60` with no task failures and did not
pass. It again produced zero nonvacuous certificates, zero online gains, and
three adaptive losses. The paired action effect was one feasibility win versus
two losses, zero regret wins versus four losses, and `+0.387` median
log-variance RMSE. It selected 294 replications and six new policies. The cap
repair left these decisions effectively unchanged, so saturation was not the
dominant end-to-end failure and that diagnostic hypothesis is rejected.

The final component audit makes the remaining failure concrete. Median
selected GPR/HVD chance-margin information was approximately `0.013/114` at
FactorShock scale 0, `0.0056/98.1` at scale 1, `0.0336/0.0956` for Inventory,
and `0.000238/1.20` for Queue. The transferred variance-shape posterior still
dominates the decision geometry in FactorShock and Queue, while producing no
posterior certificate or final regret gain. Expanding this behavior to 20 seeds
would add precision to a failed mechanism rather than test a promising one.

Accordingly, the 20-seed extension is not submitted. The causal program is
complete under its registered gate:

- cumulative factor-HVD is retained as an identifiable replicated-data model;
- its controlled factor-versus-pooled median log-RMSE improvement is `-0.276`;
- adaptive source discrepancy remains independently reportable from the full
  factorial matrix;
- adaptive evaluate-or-replicate VOI is not promoted as an optimization
  contribution;
- end-to-end certification remains vacuous in this `d=1000, N=20` transfer
  regime and must not be presented as successful certification.

## Variance-shape transfer follow-up

The next gate leaves the failed action mechanism disabled and fixes
`source-informed n0 + adaptive discrepancy + sobol_new`. It first records a
candidate-level margin decomposition at the nominal expert, expert-certified,
KL-robust task-mixture, and final certificate layers. This distinguishes mean,
GPR epistemic, HVD transfer, and task-robust inflation without changing any
decision.

The paired challenger replaces the permanently frozen source multiplier by a
hierarchical low-rank shape posterior. Each source domain contributes one PSD
cumulative-risk shape; nonnegative target weights preserve PSD and are updated
only by within-policy target sample variances. The certificate adds the
pointwise posterior shape radius, which contracts with target Fisher
information. The legacy scalar/static path remains the control. No 20-seed
extension is allowed unless this model yields nonvacuous certificates without
introducing false certificates or adaptive losses.

The five-seed hierarchical source-shape follow-up also failed promotion. It
produced no posterior certificates and was slightly worse than the static
shape control. Because `sobol_new` generated no target replications, its shape
posterior could not contract. More importantly, the margin decomposition found
that Inventory had candidates that were safe under every nominal expert but
became unsafe only after task robustness. The legacy task certificate computes
three separate KL suprema for mean, epistemic variance, and aleatoric variance;
those suprema may correspond to three incompatible task laws.

The next registered challenger therefore changes the certificate rather than
the action rule. `joint_tangent` uses one shared KL-admissible task law for the
complete chance margin. For positive tangent scales it applies
`sqrt(v) <= v/(2s) + s/2`, robustifies the resulting combined expert payoff
once, and minimizes over a fixed finite tangent grid. The legacy `separable`
bound remains the paired control. Both variants keep the frozen proposal,
adaptive discrepancy, cumulative factor-HVD, and `sobol_new` action sequence.
Promotion requires nonzero certificate coverage, zero added false
certificates, and no feasibility/adaptive-loss regression.

The paired joint-certificate gate completed `40/40` runs without task failure.
`joint_tangent` tightened the minimum robust margin by about `0.021-0.029` in
FactorShock, but every registered cell still had zero posterior certificates.
It gave no tightening in Inventory or Queue because the valid separable upper
bound was already smaller on their selected policies. The decision outcomes
were unchanged in FactorShock and Inventory; Queue retained the same four of
five feasible recommendations in both registered certificate modes. The
joint-KL theorem and implementation are retained as a strictly tighter valid
certificate option, but the challenger is not promoted as the solution to
vacuity.

The decomposition now identifies two different remaining causes. FactorShock
is primarily blocked by constraint-mean epistemic radius. Inventory and Queue
also overpredict held-out aleatoric variance by roughly `4.7x` and `5.2x` at
the median. A scalar target multiplier cannot correct this cross-domain shape
mismatch. The next registered gate therefore combines the nonnegative
source-shape mixture with budgeted target replications. It compares
`factor_cumulative` with `factor_hierarchical` and `sobol_new` with
`sobol_joint_voi`, while freezing the proposal, adaptive source discrepancy,
joint certificate, target budget, and seeds. This is the first end-to-end test
in which the hierarchical shape posterior receives target Fisher information;
the earlier `sobol_new` hierarchy gate could not update it.

The replication gate completed all `120/120` registered runs and failed. The
hierarchical HVD had `0/2` paired feasibility wins/losses and essentially zero
median global log-RMSE gain (`-0.00058`). `sobol_joint_voi` over-replicated:
most FactorShock and Queue cells spent all ten adaptive calls on existing
policies. Relative to `sobol_new`, it had `2/4` feasibility wins/losses,
`0/7` regret wins/losses, and worsened median log-RMSE by `0.226`. Inventory
and Queue variance ratios improved from roughly `10.5/11.6` under scalar
transfer to `6.4/6.7` under the source-shape mixture, but remained strongly
overconservative. No registered cell produced a posterior certificate.

A post-run oracle-variance audit then replaced only `v_C_plus` by the held-out
target's true variance at each run's best true-feasible candidate. This audit
is explicitly excluded from every decision. Even with oracle variance, every
cell remained uncertified. Removing the GPR epistemic term in the same audit
made all FactorShock `shock0` and Inventory candidates certifiable, and three
of five FactorShock `shock1` candidates certifiable. Queue and FactorShock
`sobol_new` still showed constraint-mean bias. Therefore variance-shape
transfer is not the sole, or currently the dominant, cause of vacuity.

The next mechanism is a source-informed low-rank constraint-mean posterior.
The frozen observable coordinate `eta` estimates a hierarchical source
coefficient mean and covariance from ordinary source archive observations.
The held-out constraint GPR starts from that coefficient law and conditions it
on charged target calls; source disagreement contributes prior covariance and
can only increase uncertainty before target conditioning. This replaces the
old empirical initialization in which `n0` target least squares silently
overwrote the source coefficient law. The legacy initialization remains the
paired control. Promotion requires reduced oracle-variance margins and at
least one nonvacuous certificate without a false certificate, feasibility
loss, or target-oracle access.

The `N=n0=10` offline gate completed `90/90` runs. The frozen source
coefficient law reduced the FactorShock epistemic contribution from about
`0.19` to `0.10` and the final margin from about `0.55` to `0.45`. The
observable `eta` coordinate also changed Queue's terminal recommendation from
`0/5` to `5/5` true-feasible seeds. However, every cell still had zero
posterior certificates, and Inventory's oracle-variance margin worsened from
about `0.25` under the raw-coordinate control to `0.60` under the frozen
source law. The fixed source average is therefore not promoted.

The registered repair is a target-evidence source mixture, not another
boundary adapter. Each source domain supplies one frozen Gaussian coefficient
law in `eta`; a domain-independent null component permits no-transfer. The
same charged target pilot conditions every component and updates its model
mass by the exact linear-Gaussian marginal likelihood. The resulting finite
mixture is projected to one GPR by matching its mean and covariance, retaining
both within-component uncertainty and between-component disagreement. Target
truth and target oracle values remain post-run audit quantities only. The
proposal, Sobol continuation, adaptive task discrepancy, HVD, initial points,
and seeds stay fixed in the paired re-gate.

The target-evidence mixture re-gate completed `60/60` runs. It lowered the
minimum final margin in every paired domain/scale cell and preserved zero false
certificates and zero adaptive losses, but all cells still had zero posterior
certificate coverage. The learned task law was strongly nontrivial: all five
Queue seeds selected `target:null`; FactorShock selected the Inventory source
component; Inventory retained mixed FactorShock/null mass and selected the null
component in one seed. Median final margins nevertheless remained about
`0.42-0.44` for FactorShock, `0.97` for Inventory, and `0.87` for Queue. This is
not promoted to `N=20`: the mean posterior improved, but the independently
averaged source HVD shape continued to encode a different latent task law.

The next paired challenger shares the exact target-pilot source-component
posterior between the constraint-mean mixture and cumulative HVD. Source-domain
mass weights the corresponding PSD HVD shapes; `target:null` mass maps to a
constant target-pooled nonnegative shape rather than being redistributed over
incompatible source domains. Component covariance is allocated in proportion
to posterior mass, so a rejected component cannot retain a large artificial
shape guard. The independent HVD mixture is the frozen control. The proposal,
Sobol action sequence, target calls, task-discrepancy update, certificate,
mean posterior, and seeds remain identical.

The shared-task `N=n0=10` gate completed `60/60` runs. It materially repaired
cross-domain variance-shape transfer without changing the evaluated policies:
Queue median held-out variance log-RMSE fell from `1.698` to `0.574` and its
predicted-to-true variance ratio fell from `5.47` to `0.564`; Inventory
log-RMSE fell from `1.966` to `1.743` and its ratio from `7.13` to `5.71`.
FactorShock was approximately unchanged. All cells retained zero false
certificates, but all also retained zero posterior certificates. The model is
therefore accepted as a variance-transfer repair, not as an end-to-end
certification success.

A second post-run audit separates true chance depth, constraint-mean bias, and
GPR epistemic radius at the best true-feasible candidate. The frozen initial
design contains only shallow safe anchors: the median best true chance margin
is approximately `-0.033` for FactorShock at unit shock, `-0.020` for
Inventory, and `-0.024` for Queue. At `N=n0=10`, posterior mean bias is about
`0.116`, `0.272`, and `0.264`, respectively. Even replacing the posterior mean
by the held-out true mean leaves the epistemic certificate margin positive.
Thus the original five-seed certificate gate is statistically unattainable at
this sample size; changing HVD shape or weakening the confidence level cannot
legitimately make it pass.

The registered follow-up is consequently an information-contraction test at
`N=20`, not an automatic promotion or a larger-seed confirmation. It keeps the
same frozen proposal, `n0=10`, Sobol new-point continuation, adaptive source
discrepancy, source-evidence mean posterior, joint-tangent certificate, domains,
shock strata, and five seeds. The only paired factor remains independent versus
shared mean/HVD task weights. The additional ten ordinary target calls must
reduce mean bias and epistemic radius enough to produce a nonvacuous posterior
certificate while preserving zero false certificates. Failure means that the
next model must change the transferable constraint-mean geometry or the frozen
source proposal's boundary excitation; it does not authorize a smaller
`beta_g`, target-specific anchor, or oracle-informed repair.

The information-contraction gate completed `60/60` runs with no failures and
again produced zero posterior certificates and zero false certificates. The
additional ten target observations did reduce uncertainty: shared-task final
margins fell from about `0.437` to `0.400` for unit-shock FactorShock, `0.926`
to `0.746` for Inventory, and `0.647` to `0.529` for Queue. Inventory and Queue
constraint-mean bias fell from approximately `0.272/0.264` to `0.185/0.190`.
Shared-task HVD remained the better variance-transfer model, reaching median
log-variance RMSE `0.417` on Queue versus `1.497` for independent source-shape
weights. This is a real transfer improvement but still not a certification
result.

The safety-depth audit explains why ordinary contraction was insufficient.
The ten Sobol continuation points did not discover a deeper true-safe policy:
median true chance margins remained about `-0.033`, `-0.020`, and `-0.024`.
Even under oracle mean, the remaining epistemic variance would have to shrink
by approximately `9.2x`, `204x`, and `121x`, respectively, before those points
could be certified. The Lean implementation bridge now records the necessary
condition `beta_g * v_epi <= safety_depth^2`; lowering the confidence radius
would evade rather than solve this condition.

The next registered mean-geometry challenger uses the already source-trained
`consensus` observable coordinate. It replaces the seven-parameter latent
mean basis by a three-parameter affine boundary family: intercept, normalized
source consensus signed distance, and source disagreement. Everything else is
frozen at the successful shared-task HVD diagnostic configuration, initially
at `N=n0=10`. This isolates whether excessive mean-coordinate dimension causes
vacuity. Only if that posterior contracts without unacceptable mean bias will
a later gate use the calibrated boundary coordinate to propose deeper-safe
policies; candidate adaptation is not confounded with this first test.

The consensus-coordinate gate completed `30/30` runs and was rejected. Reducing
the constraint-mean family from seven coefficients to three did not materially
contract epistemic uncertainty, while it discarded boundary shape needed for
transfer. Relative to the latent coordinate, median mean bias worsened in the
main held-out cells. The final margins remained approximately `0.43` for
unit-shock FactorShock, `0.97` for Inventory, and `0.61` for Queue, with zero
certificates. The failure rules out the simple explanation that the latent
posterior is vacuous only because it has too many coefficients. The accepted
diagnostic baseline remains latent constraint mean plus shared-task cumulative
HVD.

The next registered gate changes only how the ten post-`n0` evaluations are
chosen. `certificate_depth_new` ranks unseen candidates by the negative of the
same joint-tangent theory margin used at terminal certification. It contains no
objective term, target truth, target-specific anchor, or relaxed confidence
constant. A second `certificate_depth_search` cell adds twelve candidates from
a domain-generic random pool of 512 by minimizing that same posterior margin;
this separates ranking failure in the existing pool from insufficient
candidate coverage. `sobol_new` is rerun as the paired neutral control. The
registered gate uses `d=1000`, `N=20`, `n0=10`, five seeds, FactorShock scales
`0` and `4`, Inventory, and Queue. Promotion requires a more negative selected
true margin, reduced posterior minimum margin, nonzero sound certificate
coverage, and no increase in false certificates or adaptive loss.

The certificate-depth gate completed `60/60` runs with zero task failures and
was rejected. Every cell still had zero posterior certificates. In Inventory
and Queue the generated pool already contained true-safe candidates with mean
minimum true margins around `-0.026`, but their mean minimum posterior margins
remained approximately `0.83` and `0.58-0.60`. Both challengers therefore
ranked a truly unsafe policy above an available safe one. The generic 512-point
search did not repair the inversion and caused one Inventory adaptive loss.
For FactorShock scale four even the pool minimum true margin remained positive
(`0.009`), exposing a separate coverage failure. Thus Sobol is not the binding
cause of certificate vacuity: the learned cross-domain chance-boundary ranking
is wrong before the action rule is applied.

The registered repair preserves, rather than averages, source boundary shape.
For each frozen source atom `h_s`, ordinary source episodes fit the transferable
two-parameter family `a_s + b_s h_s(x)` across every other source domain. The
source-only distribution of offset and scale becomes one evidence-mixture
component; all coefficients belonging to other atoms are fixed near zero. The
charged target pilot updates only this affine calibration and component mass.
This differs from the failed three-parameter consensus coordinate: no source
shape is collapsed before target evidence selects and calibrates it. The first
gate compares `latent` with `source_affine` under identical frozen proposal,
shared cumulative HVD, adaptive discrepancy, target calls, and seeds.

The paired submission is statically registered as 80 independent tasks:
`latent/source_affine` x `new_only/certificate_depth_new` x four target
scenarios (FactorShock scales 0 and 4, Inventory, Queue) x five seeds. Both
coordinates use `factor_hierarchical`, joint-tangent certification,
constraint-mean task weights, `d=1000`, `N=20`, and `n0=10`; only
`node001-node006` are admissible. Runtime checkpoints remain server-side and
only result JSON is eligible for later synchronization. Submission is pending
because the current Codex session cannot update the external scheduler deploy
tree without renewed external-write authorization.

The source-affine gate subsequently completed all `80/80` tasks without a
failure or retry and was rejected. Under the neutral `new_only` backend it
reduced the median final margin from `0.7463` to `0.5829` in Inventory, from
`0.5294` to `0.5271` in Queue, and from `0.4018` to `0.3882` in zero-shock
FactorShock, but every cell still had zero certificates. Direct
certificate-depth ranking caused one Inventory and four Queue adaptive losses.
Evidence selected `target:null` with median mass `0.808` in Inventory and
approximately one in Queue, while zero-shock FactorShock assigned nearly all
mass to one biased source atom. Source affine calibration therefore improves
some neutral-backend diagnostics but is not a stable transferable safety
coordinate.

The next registered coordinate is `source_rank`. It retains only within-source
percentile order of ordinary observed chance margins, interpolates each frozen
source rank atlas in a fixed observable-policy library, and exposes consensus
rank plus cross-source disagreement. This is invariant to every strictly
increasing source-margin rescaling and directly matches the information that
made the frozen source proposal effective. Its first gate keeps `new_only`,
the frozen proposal, hierarchical factor-HVD, adaptive discrepancy, and all
budgets fixed; the rejected certificate-depth action is not reused.

The source-rank gate completed `20/20` runs without a task failure and was
rejected. It slightly reduced the final margin for FactorShock at shock scales
zero and four (`0.3891` and `0.3788`), but worsened Inventory to `0.824` and
Queue to `0.8385`. Queue retained only four of five feasible runs and incurred
one adaptive loss. Every cell again had zero posterior certificates. The rank
coordinate transfers source ordering but cannot recover an absolute held-out
chance-margin scale; it is therefore not promoted over the latent coordinate.

The variance diagnostics exposed a separate identifiability defect shared by
all preceding `new_only` cells. The source-mixture HVD target shape degrees of
freedom were exactly zero: ordinary online evaluations updated the constraint
mean GPR, but only within-policy replications were admitted as target variance
shape evidence. Consequently the cumulative HVD remained a source-only
extrapolation after all ten online calls.

The next registered gate isolates that defect with
`hvd_cumulative_target_evidence_mode=prequential_upper`. For an online policy
that has not previously been evaluated, its prediction is frozen before the
new response and the raw squared innovation is retained. In expectation this
equals the true aleatoric variance plus squared mean-prediction bias, so mean
error can only make the moment conservative. Initial in-sample residuals are
excluded, while an actual replicate continues to replace singleton evidence
with its within-policy sample variance. The ordinary residual-square tail and
source-shape posterior guards remain active; no confidence constant is
relaxed.

The paired gate contains 40 independent tasks:
`replication_only/prequential_upper` x four target scenarios (FactorShock
scales zero and four, Inventory, Queue) x five seeds. It fixes the accepted
latent mean coordinate, frozen source-informed initial design,
`factor_hierarchical`, adaptive source discrepancy, `new_only`, joint-tangent
certification, constraint-mean task weights, `d=1000`, `N=20`, and `n0=10`.
Promotion first requires the challenger to produce positive target shape DOF
and improve variance calibration without false certificates or adaptive loss.
Certificate coverage and final margin are secondary outcomes because the
previous oracle audit also identified mean epistemic radius as an independent
binding layer.

The prequential gate completed `40/40` tasks without failure, retry, false
certificate, or adaptive loss. It passed the plumbing condition exactly:
constraint-output target shape DOF was `10` in every challenger seed and zero
in every control. It also reduced the final theory margin in all five seeds for
zero-shock FactorShock, Inventory, and Queue, and in four of five scale-four
FactorShock seeds. Median margins changed from `0.4018` to `0.3470`, `0.4224`
to `0.3721`, `0.7463` to `0.5597`, and `0.5294` to `0.4366`, respectively.

Nevertheless, `prequential_upper` is not promoted. Every paired run selected
the same recommendation and every cell still had zero certificates. More
importantly, variance calibration improved seedwise `5/5` only for Inventory;
zero-shock FactorShock improved `1/5`, scale-four FactorShock `3/5`, and Queue
`2/5` by log-variance RMSE. A single squared innovation is valid
conservative-moment evidence but remains a high-variance observation, and its
mean-prediction bias is not an identified aleatoric component. The smaller
theory margins therefore do not yet establish a sound calibration gain.

The next diagnostic asks whether the remaining vacuity is a low-budget
identifiability limit rather than another representation defect. At `N=40`
and `n0=10`, it compares neutral `new_only` with `joint_voi`, where unseen and
previously observed policies share one action space and the latter can buy
direct within-policy variance evidence. The 40-task matrix is
`new_only/joint_voi` x four target scenarios x five seeds. All other choices
remain fixed at latent constraint mean, hierarchical cumulative factor-HVD,
adaptive source discrepancy, replication-only shape evidence, joint-tangent
certification, constraint-mean task weights, and `d=1000`. This is a
certifiability-budget audit, not automatic promotion of the previously failed
`N=20` replication backend. A positive result requires nonzero sound
certificate coverage, no adaptive loss, and an interpretable new/replicate
action mix; a second failure means the current transferable mean/variance
posterior cannot support theory certification at these budgets.

The `N=40` audit completed `40/40` tasks without failure, false certificate,
or adaptive loss, but it failed the gate. `joint_voi` acquired real target HVD
information: median constraint shape DOF reached `30` for zero-shock
FactorShock, Inventory, and Queue, and `19` for scale-four FactorShock. Relative
to `new_only`, the overall paired median log-variance RMSE delta was `-0.455`;
variance upper coverage was one in the first three primary cells. Nevertheless,
posterior certificate coverage remained exactly zero.

The action trace explains why the extra budget did not close the certificate.
For zero-shock FactorShock, Inventory, and Queue, `joint_voi` selected
`150/150` replications and no new policies across five seeds. Scale-four
FactorShock selected `89` replications and `61` new policies, but still found no
true-feasible recommendation. No primary run improved its initial feasible
regret, and the paired action comparison produced one regret win versus six
losses. Direct replication can therefore identify much of the cumulative HVD,
but the current action rule spends the entire budget on that layer while the
constraint-mean bias/epistemic radius and candidate coverage remain binding.

This collapse exposed a unit error in the old `joint_voi` implementation. Its
GPR term was an integrated reduction of constraint posterior variance, while
its HVD term was an integrated reduction of variance-parameter predictive
variance followed by a duplicated delta-method scale. Adding those quantities
does not define a joint value of information. The repaired implementation maps
the GPR update through
`sqrt(beta_g) * (sqrt(q_old) - sqrt(q_new))` and maps the source-shape
covariance update through the induced decrease of
`z_alpha * sqrt(v_C_plus)`. Both terms are now expressed as a reduction of the
same chance margin in response units and are labeled
`sqrt_radius_reduction_v2` in every result. The completed `N=40` runs remain
evidence for the rejected v1 score; the corrected v2 action requires a new
paired gate and must not inherit those results.

The corrected `N=20` v2 gate then completed `40/40` tasks without failure,
retry, false certificate, or adaptive loss. The unit repair eliminated the
action collapse: across five seeds, replication/new-point totals were `14/36`
for zero-shock FactorShock, `27/23` for scale-four FactorShock, `16/34` for
Inventory, and `13/37` for Queue. Median final theory margins decreased from
`0.4018` to `0.3383`, `0.4224` to `0.3574`, `0.7463` to `0.4855`, and `0.5294`
to `0.4588`, respectively. The overall paired median log-variance RMSE change
was `-0.184`.

V2 is nevertheless not promoted. Posterior certificate coverage remained
`0/20`, no run improved its initial feasible regret, and one conditional
regret loss occurred. The oracle decomposition localizes the remaining
failure. In zero-shock FactorShock, replacing only the variance model by the
oracle still leaves median margin `0.131`; Inventory and Queue retain oracle-
variance margins `0.497` and `0.434`. Their posterior mean biases are `0.261`
and `0.192`, while epistemic radii are `0.291` and `0.266`. Scale-four
FactorShock additionally has no true-safe policy in the generated pool. The
registered conclusion is therefore that direct target replications can learn
the HVD layer, but variance identification is no longer the binding cause of
vacuous certification at this budget.

Inspection exposed the next posterior-level defect. The existing
`evidence_mixture` computes source-domain/null component mass from the charged
`n0` pilot and then projects the mixture to one Gaussian. Later target calls
condition that projected Gaussian but do not update component mass; cumulative
HVD also keeps the pilot-only task law. The next isolated challenger is
`sequential_evidence_mixture`: every charged target response updates each
Gaussian component and its predictive-likelihood weight before moment
matching, and the same current component law is passed to cumulative HVD and
exact posterior clones. The frozen source archive and coordinate remain
unchanged. Its first gate compares collapsed versus sequential mixture under
the neutral `new_only` backend, so any gain is attributable to a coherent
online posterior rather than a new acquisition rule.

The sequential-mixture gate completed all `40/40` tasks without failure,
retry, false certificate, or adaptive loss, but it also failed promotion.
Every constraint model performed exactly ten online mixture updates and the
same final component law reached cumulative HVD, so this is a mechanism result
rather than a plumbing failure. Certificate coverage remained `0/20`, no run
improved its `n0` incumbent, and the overall paired median log-variance RMSE
change was `+0.0083`.

The domainwise effect was heterogeneous. Inventory log-variance RMSE improved
by `-0.630`, but Queue worsened by `+0.236`; zero-shock FactorShock also
worsened by `+0.078`. Sequential evidence drove the target-null mass to about
one for Inventory and Queue and to essentially zero for FactorShock. This
reduced the median final margin from `0.746` to `0.595` in Inventory and from
`0.529` to `0.461` in Queue by shrinking the aleatoric term, but posterior mean
bias increased from `0.185` to `0.340` and from `0.190` to `0.205`,
respectively. FactorShock margins increased. Exact sequential Bayes updating
therefore does not repair the transferable constraint-mean coordinate and is
not promoted over the pilot-conditioned mixture.

The registered diagnosis is now narrower. Corrected joint VOI can acquire
target variance information, and shared-task HVD can transfer variance shape,
but the frozen source proposal exposes only shallow safe points and the latent
constraint-mean coordinate ranks those points with persistent bias and
epistemic radius. Further HVD weighting or mixture-temperature tuning is not
supported by this gate. The next experiment must change boundary excitation
or the observable mean coordinate while keeping the accepted HVD diagnostic
and neutral backend fixed.

Before changing the coordinate again, one final budget-identifiability curve
is registered. It restores the accepted pilot-conditioned
`evidence_mixture`, fixes latent constraint mean, hierarchical cumulative HVD,
adaptive source discrepancy, joint-tangent certification, and the frozen
source proposal, then compares `new_only` with corrected `joint_voi v2` at
`N=40` and `N=80`. FactorShock scales zero and four, Inventory, Queue, and five
seeds give `2 x 2 x 4 x 5 = 80` independent target runs. This is not a larger-
seed confirmation: it estimates whether sound certification appears as target
information increases while the coordinate is fixed.

The action is promoted only if it produces a nonzero sound certificate count,
no false certificate, no net adaptive loss, and a monotone improvement from
`N=40` to `N=80`. If both budgets remain vacuous, the latent coordinate and
frozen proposal fail a direct information-sufficiency test; further HVD or VOI
tuning is then prohibited until boundary excitation or the observable mean
coordinate changes.

The complete budget curve contains `80/80` successful runs and rejects the
action. At `N=80`, corrected `joint_voi` still produced `0/20` nonvacuous
certificates and no false certificates. It improved conditional regret in one
paired seed and worsened it in five; relative to `new_only`, its median
log-variance RMSE improved by `-0.462`. Increasing `N=40` to `N=80` reduced the
median final margin by only `0.013` under `joint_voi`, without changing the
certificate count. The largest-budget oracle decomposition also remained
vacuous when only the variance was replaced by truth. Scale-four FactorShock
had no true-feasible policy in the generated pool. The registered conclusion
is therefore information insufficiency in the transferable constraint-mean
coordinate and boundary proposal, not unresolved cumulative-variance fitting.

A paired runtime probe additionally invalidated the initial attribution of the
large untracked iteration time to checkpoint I/O. With identical `N=20`, seed,
and evaluation cadence, checkpoint intervals one and five took `1666.2s` and
`1666.4s`; the checkpoint timers summed to only `3.28s` over ten online
iterations. Reducing diagnostic recommendation evaluation from every five
iterations to the final iteration reduced wall time to `1587.9s`. Remote
inspection of the retained timing state, without copying its pickle locally,
showed `939.4s` in the ten pre-action posterior recommendation solves,
`285.2s` in three post-action recommendation evaluations, and only `3.28s` in
checkpoint writes. The posterior solve repeatedly recomputed the same expert
GPR/HVD moments for nominal, robust, joint-margin, and Bayes-risk summaries.
The implementation now reuses those moments within one solve and records
stage-time summaries in the final JSON; synthetic decision-gate runs disable
runtime checkpoints with interval zero.

The same-seed cached-moment challenger passed its performance and semantic
regression. Wall time decreased from `1587.9s` to `851.3s` (`1.865x`) and
algorithm time from `1579.0s` to `843.7s` (`1.871x`). Pre-action posterior
solve time decreased from `939.4s` to `401.6s`. After removing only runtime
fields, experiment identifiers, and checkpoint paths, the two result JSON
objects were exactly equal: recommendation, action-source counts, feasible
regret, chance margins, variance calibration, and all posterior diagnostics
were unchanged. This cached implementation is therefore promoted as the new
performance baseline; it does not alter the rejected `N=80` statistical gate.

## Entrypoints

- `scripts/submit_scolhkg_hvd_decision_gate_scheduler.py`
- `SC-OLH-KG/performance/analyze_hvd_decision_gate.py`
- `scripts/submit_scolhkg_hvd_identifiability_scheduler.py`

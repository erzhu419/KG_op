# Source-aligned chance-boundary coordinate gate

## Registered question

The cumulative-HVD gates showed that better variance RMSE did not improve
feasibility or regret. Post-run oracle substitution localized the remaining
failure to two layers: the generated pool sometimes lacked a truly feasible
policy, and the constraint-mean posterior ranked available policies with
persistent bias and epistemic radius. This gate changes only those layers.

It tests whether a source-learned chance-boundary coordinate can improve the
constraint-mean posterior and whether proposals selected in that same
coordinate restore candidate support. HVD constants, confidence constants,
the source archive, target initial points, online budget, and neutral Sobol
backend remain fixed.

## Two-coordinate contract

The model deliberately uses two coordinates.

- `phi(x)` is learned from source chance-margin strata and parameterizes the
  target constraint mean and its epistemic uncertainty.
- `psi(x)=(A(x),N(x))` remains the cumulative-risk coordinate used only by
  factor-HVD and `v_C_plus`.
- They meet only in
  `mu_g(phi) + sqrt(beta_g) s_g(phi) + z_alpha sqrt(v_C_plus(psi)) - tau`.

The source labels used for the two parts of `phi` are also separated. Signed
chance-margin bins learn the representation. Independent source
constraint-mean responses learn the Gaussian coefficient prior. Aleatoric
variance therefore does not become a hidden constraint-mean target.

## Alignment and target adaptation

For every source record, the observable policy-profile library is standardized.
The alignment solves a generalized eigenproblem that maximizes scatter between
signed margin bins while penalizing source-domain shifts within the same bin
and within-bin scatter. Only the leading two directions are frozen as `phi`.

The held-out target never supplies a true margin, true mean, true variance,
problem-specific anchor, or analytic risk formula to this fit. Its charged
ordinary observations update only the low-dimensional coefficient posterior
and source-component evidence weights.

## Candidate proposal

The proposal pool contains only dimension-equivariant universal profiles,
source-frozen boundary-stratum templates, and random low-frequency
perturbations. The current target posterior divides selected proposals into
three roles:

1. posterior-safe candidates;
2. candidates near the estimated chance boundary;
3. under-covered regions in `phi`.

The default allocation is `30/40/30` percent. All three roles share one pool
and one target-conditioned posterior; no role reads target truth.

## Causal matrix

Three paired variants isolate representation and proposal effects:

- `latent_control`: previous latent observable mean coordinate, no `phi`
  candidates;
- `phi_mean_only`: source-aligned `phi`, no `phi` candidates;
- `phi_mean_proposal`: source-aligned `phi` plus 12 selected candidates from a
  512-policy pool per online iteration.

The four scenarios are FactorShock at shared-shock scales zero and four,
Inventory, and Queue. Every cell uses five seeds, `d=1000`, `N=20`, `n0=10`,
the same frozen source-informed initial design, source dimension 50, two source
domains with 64 policies and three replications each, hierarchical cumulative
factor-HVD, adaptive source discrepancy, joint-tangent certification, and
`sobol_new`. The total is `3 x 4 x 5 = 60` independent scheduler tasks.

## Post-run oracle audit

Target truth is evaluated only after each decision. Every generated pool is
audited under four margins: fitted mean/fitted variance, fitted mean/oracle
variance, oracle mean/fitted variance, and oracle mean/oracle variance with the
same epistemic radius. Exact algebra then assigns each failed iteration to
candidate support, constraint mean, cumulative variance, epistemic safety
depth, or closed certificate. Per-source support reveals whether the new
`phi` pool, rather than an unchanged source, supplied feasible points.

## Promotion rule

No 20-seed or larger-budget experiment is allowed unless all 60 results and all
12 registered cells are complete with five unique seeds and the joint variant
satisfies all of the following:

1. the scale-four FactorShock `phi` pool contains a truly feasible policy in at
   least one audited iteration;
2. at least one sound posterior certificate is produced;
3. zero false certificates;
4. zero adaptive losses;
5. mean-rank wins across scenarios are at least as frequent as losses relative
   to the latent control.

Failure blocks further HVD/VOI tuning. It means the observable source archive
and profile library are not sufficient to transfer the held-out chance
boundary at this budget.

## Registered result

Run `scolh_boundary_coordinate_gate_s5_20260718_v1` completed all 60 tasks
without failure or retry. The challenger did not pass the promotion rule.

| Variant | FactorShock 0 | FactorShock 4 | Inventory | Queue | Certificates | Adaptive losses |
|---|---:|---:|---:|---:|---:|---:|
| `latent_control` | 5/5 | 0/5 | 5/5 | 5/5 | 0 | 0 |
| `phi_mean_only` | 5/5 | 0/5 | 5/5 | 5/5 | 0 | 0 |
| `phi_mean_proposal` | 5/5 | 0/5 | 5/5 | 3/5 | 0 | 2 |

The aligned mean coordinate increased median constraint-mean absolute error
from `0.291` to `6.457` on Inventory and from `0.241` to `3.608` on Queue.
Its mean-rank correlation beat the latent control in one scenario and lost in
three. Selected `phi` proposals never supplied a truly feasible candidate on
FactorShock at shared-shock scale four. All zero false-certificate counts are
therefore vacuous because no variant produced a certificate.

This rejects the raw policy-profile alignment, not cumulative HVD. The next
coordinate must be built from a common observable state/trajectory exposure
and use separate mean and variance heads. A random or low-frequency raw-policy
pool is not a valid inverse map from a boundary coordinate to a thin feasible
policy manifold.

The run also exposed that the inherited FactorShock scalarized oracle used the
old pointwise noise law. The oracle now calls `true_sigma`, so shared-shock
scale changes are included. FactorShock regret values from this run are not
used for promotion; feasibility, certificate, rank, and calibration findings
above are unaffected.

## Entrypoints

- `representation/boundary_coordinate.py`
- `performance/analyze_boundary_coordinate_gate.py`
- `scripts/submit_scolhkg_boundary_coordinate_gate_scheduler.py`
- `proof/SCOLHKG/Real/BoundaryCoordinateSufficiency.lean`

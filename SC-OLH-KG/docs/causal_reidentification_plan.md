# Causal Re-identification Plan

## Why this reset is necessary

The completed matrix shows that the frozen source-informed `n0` proposal is
the dominant observed effect. Exact KG and risk-aware Thompson sampling do not
beat Sobol continuation, cumulative HVD does not beat pooled variance, and the
theory certificate is empty in every completed structural run.

The old structural-prior table is not causal. It reused one full-prior
proposal and changed only spectral posterior flags; the ordered implementation
still retained low-frequency weighting, orthogonal coordinates, sparse model
selection, group ridge, and diagonal-additive blocks.

## Corrected estimands

All structural experiments use Sobol continuation, pooled variance, adaptive
source-discrepancy updating, identical source-call budgets, and one seed per
scheduler task.

1. `proposal_only`: profile-specific source proposal plus a no-four-prior
   target posterior.
2. `posterior_only`: common Sobol `n0` plus a profile-specific posterior.
3. `joint`: profile-specific source proposal and the matching posterior.

Every profile is retrained from the same immutable source archive and emits a
new proposal fingerprint. Both spectral and ordered realizations of each
assumption are switched together. If fingerprints remain identical, the
proposal gain belongs to source-consensus ranking rather than that prior.

## Dimension transfer

The `risk_coordinate_atlas` proposal represents source policies with
dimensionless ordered moments and cosine coordinates, selects source-ranked
maximin coverage, and synthesizes policies directly at the target dimension.
It is compared with legacy rank-spanning interpolation.

Domain and dimension are held out simultaneously: for example, source domains
are observed only at `d_source=50`, while a held-out target domain is optimized
at `d_target=200` or `1000`. Source and target dimensions, archive
fingerprints, and proposal fingerprints are recorded separately. No `d=10000`
experiment is admissible until the `d=1000,N=20` gate succeeds.

## HVD identification

HVD is tested outside objective search so candidate quality cannot hide its
statistical value. The FactorShock benchmark sweeps:

- shared-shock scale: `0, 0.25, 1, 4`;
- replications per policy: `2, 4, 8, 16`;
- pooled, class, orthogonal-pointwise, factor-pointwise, and cumulative-factor
  variance models.

Fit data are ordinary replicated simulator observations. Oracle variance is
used only after fitting to report log-variance RMSE, risk ordering,
upper-variance coverage, false feasibility, missed feasibility, and whether
the certificate is non-vacuous.

The first 400-cell gate identified two implementation defects rather than a
failed risk coordinate: sample-variance replication degrees of freedom were
discarded by the tail guard, and a one-shot PSD/nonnegative projection inflated
the unconstrained ridge scale. The promoted estimator uses replication-aware
projected IRLS. In the corrected five-seed gate, cumulative factor-HVD ranks
variance best in all 16 shock/replication cells and has the lowest log-variance
RMSE in 14, with zero false-certified fraction. Certification remains
deliberately conservative at the highest shock levels, so recall and precision
are reported separately rather than collapsed into population error rates.

## Gates

1. Unit tests and local tiny smoke.
2. Corrected causal matrix at `d=50,N=20,n0=10`, 5 seeds.
3. HVD identification, 5 seeds per cell.
4. Dimension holdout at `d_target=200,N=20`, source `d=50`.
5. Only promoted proposal variants proceed to `d_target=1000,N=20`.
6. Increase seeds only after a positive paired effect; do not use `d=10000`
   to obscure a failed `d=1000` gate.

## Completed gate evidence

The corrected `d=50,N=20,n0=10` matrix
`scolh_causal_prior_v2_gate_s5_20260716` completed all 450 optimization cells
with no parse failures. Every comparison retrained its source proposal and used
the same frozen source archive and paired target seeds.

- `full` is rejected as the promoted profile: joint feasibility was `11/15`.
- `additivity_only` and `orthogonality_only` each achieved `15/15` joint
  feasibility across FactorShock, Inventory, and Queue.
- `leave_out_sparsity` also achieved `15/15`, while the no-four-prior control
  achieved `13/15`.
- On Queue, the full and sparsity-only proposals had `0/5` initially feasible
  runs, versus `5/5` for the no-four-prior, additivity-only, and
  orthogonality-only proposals. Thus the failure is already present in the
  frozen proposal and cannot be attributed to the online backend.
- With common Sobol initialization, every FactorShock posterior profile was
  `0/5`; the transferable proposal is necessary in that domain.

The next registered gate holds source dimension at 50 and target dimension at
200, comparing rank-spanning interpolation against the dimension-equivariant
risk-coordinate atlas for `none`, rejected `full`, `additivity_only`, and
`orthogonality_only`.

### Pre-registered d=200 promotion rule

This rule was fixed before inspecting any `d=200` optimization result. A
structural profile can proceed to `d=1000,N=20` only when all of the following
hold:

1. all 60 cells for that profile are present (`2` causal modes, `2` proposal
   modes, `3` held-out domains, and `5` seeds), with no failed or unparsable
   result;
2. all paired proposal comparisons use the same frozen source-archive
   fingerprint, while the two proposal modes retain separately recorded design
   fingerprints and use no target labels or target oracle during fitting;
3. `joint/risk_coordinate_atlas` is truly feasible in at least `12/15` runs
   and at least `3/5` runs in every held-out domain;
4. among initially truly feasible atlas runs, at most one becomes infeasible
   after online continuation;
5. relative to `rank_spanning` on identical domain/seed/profile cells, atlas
   has nonnegative overall paired final-feasibility net and loses at most one
   feasibility pair in any domain;
6. conditional on both recommendations being feasible, atlas has at least as
   many regret wins as losses overall. Feasibility takes precedence over this
   conditional-regret criterion;
7. relative to the `none` profile under the atlas, the structural profile has
   a positive paired effect: either positive final-feasibility net, or tied
   feasibility with more regret wins than losses. Proposal-only and joint
   effects are reported separately so an offline proposal effect is not
   attributed to the posterior.

If no profile passes, `d=1000` is not launched. A failed gate triggers a model
diagnosis at `d=200`, not a larger-dimensional search for a favorable result.

### d=200 gate outcome

Run `scolh_causal_dimholdout_d200_s5_20260716` completed all 240 optimization
cells with zero failed or unparsable results. Every proposal-mode pair matched
the frozen archive fingerprint, and all paired design fingerprints were
separately recorded.

- The risk-coordinate atlas gave `additivity_only` and
  `orthogonality_only` `15/15` final feasibility with zero adaptive losses.
  The rejected full profile reached only `11/15` because its Queue proposal
  remained poorly transferable.
- Relative to rank-spanning interpolation, atlas feasibility net was `+2` for
  additivity and `+1` for orthogonality. It nevertheless lost the conditional
  regret comparison (`2/6` wins/losses for additivity and `1/8` for
  orthogonality), so neither profile passes the pre-registered promotion rule.
- Against the no-four-prior atlas, orthogonality tied feasibility and won
  conditional regret `3/0`; additivity won `3/2`. These effects already appear
  in `proposal_only`. The joint posterior produced almost no additional gain.

No `d=1000` task is launched from this gate. The diagnosed defect is specific:
the original atlas ranks source templates only by chance margin and then
maximizes coordinate coverage, so safety improves while source objective
quality is incidental. The registered challenger is a source-only
`risk_objective_atlas`: it converts chance margins and objectives to separate
within-domain percentile ranks, retains their Pareto elites, and uses maximin
risk-coordinate filling only for the remaining slots. The old atlas remains
an explicit ablation.

The `risk_objective_atlas` repair is evaluated on the same source archive,
domains, target seeds, `d=200`, `N=20`, and `n0=10`. Its promotion rule was
fixed before any repair run completed:

1. all 90 repair cells must complete without parse failures;
2. each profile must satisfy the original safety floor (`12/15` overall,
   `3/5` per domain, and at most one adaptive loss);
3. against the old risk-coordinate atlas, feasibility is compared
   lexicographically before regret: the repair may not lose feasibility in any
   domain; when feasibility ties, paired conditional-regret wins must be at
   least losses;
4. the same lexicographic comparison is repeated against rank-spanning, so the
   repair must retain the atlas safety gain without giving back its objective
   correction;
5. a structural profile must still beat the repaired `none` proposal by
   paired feasibility, or by paired regret when feasibility ties.

Only profiles satisfying all five conditions can proceed to `d=1000`. The
repair is not allowed to pass merely because its source objective rank is
available in diagnostics.

### Strict archive-level control

The repair gate exposed a deeper control issue. The nominal `none` profile
closed spectral and ordered posterior switches, but all source observations
still came from `universal_low_frequency` policies and the target atlas still
used truncated cosine reconstruction. This is not target leakage, but it means
the row was not a true no-low-frequency control.

The strict control therefore changes the common source archive itself:

- `shared_uniform` freezes one unrestricted normalized random policy library
  and evaluates byte-identical policies in every source domain;
- within-domain chance-margin and objective percentiles are computed on those
  paired policies, preserving the scale-invariant LODO comparison;
- when low frequency is disabled, proposal distance uses the complete
  64-point canonical policy and target synthesis uses direct interpolation;
- cosine truncation is available only to profiles that explicitly enable the
  low-frequency assumption;
- the generic low-frequency sentinel is removed from the strict proposal. It
  is used only as a recorded fallback when fewer than `n0` source templates
  survive deduplication.

This strict archive is a new experiment and is never merged with the legacy
universal-shape archive. It first repeats the full `d=50,N=20,n0=10`, five-seed
causal matrix. A profile must beat strict `none` before any dimension holdout
is repeated.

The strict `d=50` decision rule was fixed before inspecting its optimization
results:

1. all 450 optimization cells must complete and parse;
2. a promotable joint profile must be feasible in at least `12/15` runs, at
   least `3/5` in each domain, and lose at most one initially feasible run;
3. effects are lexicographic: a profile must have nonnegative paired
   feasibility net in every domain and positive net overall; when overall
   feasibility ties, conditional-regret wins must exceed losses;
4. standalone evidence is `component_only` versus strict `none`, reported
   separately for proposal-only, posterior-only, and joint paths;
5. necessity evidence is `full` versus `leave_out_component`; it is not
   substituted for standalone evidence and may reveal interactions;
6. proposal fingerprints and overlap are reported. Identical proposals are a
   valid zero effect, not grounds to relabel posterior evidence as proposal
   evidence.

If strict `none` is noninferior to every structural profile, the claim that the
four priors cause the proposal advantage is rejected. The source
safety-objective ranker may remain as a separate learned-transfer mechanism,
but it cannot be attributed to the four switches.

### Strict d=50 outcome and support-prior audit

Run `scolh_causal_strict_shareduniform_d50_s5_20260716` completed all 450
optimization cells with zero failures or parse errors. No profile passed the
pre-registered safety floor:

- joint strict `none` reached `6/15` feasibility;
- the best single profiles, additivity and orthogonality, reached only `7/15`;
- full reached `2/15`;
- every source-informed profile had `0/5` initial and final feasibility on
  Inventory;
- no joint profile improved its initial best feasible regret, and all profiles
  lost initially feasible Queue recommendations.

Low-frequency-only proposal selection improved Queue feasibility by two paired
seeds relative to strict `none`, but did not create Inventory support and
remained below the global safety floor. Thus a low-frequency representation
cannot recover policies absent from the source design.

The next audit isolates the policy-support assumption itself. It compares
`shared_uniform` against `universal_mixture` with the same `384` ordinary
source simulator calls, held-out domains, target seeds, `d=50`, `N=20`,
`n0=10`, risk-objective ranker, and Sobol backend. Only the source policy
support changes. The primary contrast is `none` versus `none`; a secondary
low-frequency-only row measures interaction with low-frequency coordinates.
The universal support prior is supported only if its paired feasibility net is
positive without a negative domain net; conditional regret is consulted only
when feasibility ties.

Scheduler entrypoints:

- `scripts/submit_scolhkg_causal_prior_matrix_scheduler.py`
- `scripts/submit_scolhkg_hvd_identifiability_scheduler.py`

Runtime checkpoints stay in remote checkpoint directories. Result collection
accepts only `result.json`; pickle files, model weights, NumPy arrays, and
checkpoint trees are never synchronized for aggregation.

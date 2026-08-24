# Shadow Interval/Multi-Q Theory-Operation Competition V2

## Boundary

This strictly additive V2 exact-replays the eight preceding shadow slices and
consumes one lossless finite-interval/two-Q recompetition seed.  It implements
one bounded, validation-selected, stress-confirmed shadow competition among
three representation-level operation families.  It does not materialize the
selected candidate as a new theory state, execute rollback, expand the
language, decide adoption, promote a theory, or write a current pointer.

The slice consists only of:

- `performance/shadow_interval_multi_q_theory_operation_competition.py`;
- `performance/manifests/shadow_interval_multi_q_theory_operation_competition_v2.json`;
- `runners/run_shadow_interval_multi_q_theory_operation_competition.py`;
- `tests/test_shadow_interval_multi_q_theory_operation_competition.py`;
- this document.

The pure core writes nothing.  The runner's explicit absolute `--out` is the
only optional write surface.  No benchmark, `run_one`, scheduler, network
service, external probe acquisition, or language-invention mechanism is
called.  The frozen Operations Research worktree, manuscript, baselines, and
claims are outside this slice and remain untouched.

## Exact replay and typed handoff

The contract identifier is
`shadow_interval_multi_q_theory_operation_competition_v2`.  Its contract,
input, report, and candidate schemas are all revision 2.  The source adapter
contract digest is frozen as:

```text
sha256:16d2a30873e3f8b2e56fe5d7ac272140eb83dbcb441d8d80a892c4028f28f029
```

The canonical V2 contract digest is:

```text
sha256:4c30c0b1a2cdec92ab1676e98677b620907bb9652bff1ce71865fce9d45ccd1e
```

The core exact-replays the complete shadow chain through the adapter report and
requires a canonical, emitted `finite_interval_table` seed with exactly the
two registered probes:

```text
absolute_error_point_prediction
normalized_signed_interval_boundary_margin
```

An adapter route that emitted no seed cannot be converted into a candidate.
Its needs-evidence and incomparable-epoch states map to distinct blocked V2
dispositions.  Rehashed upstream tampering, state substitution, Q/V drift,
lineage drift, or a forged adapter interface fails closed.

The public builder takes exactly twenty-four positional JSON objects:

```text
competition V1 input, contract, report
transition contract, report
qualification input, contract, report
external-review contract, report
failure-boundary input, contract, report
restriction input, contract, report
post-restriction adjudication input, contract, report
interval/multi-Q adapter input, contract, report
interval/multi-Q competition V2 input, contract
```

The runner requires all twenty-four paths to be absolute, existing,
non-symlink regular files with distinct resolved paths and distinct inodes.  It
rejects all 276 possible input-pair aliases, including hard links.  Duplicate
JSON keys, non-finite JSON constants, non-object roots, input/output aliases,
and forged artifact metadata fail closed.  Nested receipts preserve the exact
2/4/7/9/12/15/18/21/24-file replay boundaries.

## Public API

```python
validate_shadow_interval_multi_q_theory_operation_competition_contract(contract)

derive_shadow_interval_multi_q_theory_operation_competition_id(
    *, adapter_contract_digest, adapter_report_digest,
    recompetition_seed_digest, seed_theory_state_digest,
    interval_competition_contract,
)

derive_shadow_interval_multi_q_theory_operation_competition_epoch(
    *, adapter_contract_digest, adapter_report_digest,
    recompetition_seed_digest, seed_theory_state_digest,
    fixed_anchor, interval_competition_contract,
)

synthesize_shadow_interval_multi_q_theory_operation_candidates(
    recompetition_seed, discovery_rows, evaluator, contract,
)

run_shadow_interval_multi_q_theory_operation_competition(
    # the twenty-four positional objects listed above,
    ...,
    # twenty-three independent expected digests and nested receipts,
    input_artifacts=None,
)

verify_shadow_interval_multi_q_theory_operation_competition(
    # the same twenty-four objects, then the V2 report,
    ...,
    expected_interval_competition_report_digest,
    expected_interval_competition_input_artifacts,
)
```

The four-argument synthesis helper is an untrusted, discovery-only pure
constructor.  It requires exactly two unique discovery rows per registered
scope/context cell, but its output is neither a V2 report nor a verified
competition commitment.  By itself it does not establish the canonically
derived evaluator epoch, five-generation prior-ID exclusion, or adapter exact
replay; only the full runner and public verifier bind those authorities.

The result exposes only its bounded disposition, report digest,
`candidate_selected`, selected candidate identifier, selected operation kind,
and `to_dict()`.  It has no materialization,
adoption, promotion, current-pointer, language-expansion, rollback-execution,
or ambient-state mutation capability.  The verifier exact-replays the report
and compares canonical JSON.

## Fresh evidence and stable evaluator epoch

The V2 input contains only source-adapter commitments, one evaluator binding,
the inherited five-class prior-record exclusion ledger, and three fresh
evidence splits.  Every row has exactly:

```text
observation_id, evaluator_epoch, fixed_anchor,
scope_id, context, observed_value
```

For every registered scope/context cell, discovery must contain exactly two
rows, validation exactly one, and stress exactly one.  Missing or extra rows in
registered cells produce
`INTERVAL_MULTI_Q_COMPETITION_NEEDS_EXACT_FRESH_EVIDENCE` before any numeric
candidate evaluation.  An unregistered cell, non-finite value, duplicate
observation identifier, or reuse of any identifier from a prior record class
is instead a hard invalid input.

The evaluator epoch is content-addressed from the adapter contract and report,
the emitted seed and its theory-state digest, the fixed anchor, and this V2
contract.  It deliberately excludes observation identifiers, values, and row
order.  All rows must bind the same derived epoch and anchor.  A different
epoch maps to the incomparable-epoch disposition; cross-epoch pooling is never
allowed.

The bounds are finite and explicit: at most 64 contexts, 16 scopes, 8 removable
features, and 260 raw candidates.  Bounded synthetic known-answer tests are
mechanical checks, not external data or evaluator attestation.

## Ordered diagnostics

Discovery diagnostics are applied in the contract's frozen stage order:

```text
reestimate, noise, scope, mixture,
simplify, robustify, new_probe, language_last
```

The first four are diagnosis gates, not candidate families.  `reestimate`
distinguishes no raw boundary violation, exclusion, and a viable explanation;
`noise` distinguishes zero signed-exceedance variance, exclusion, and a viable
explanation; `scope` also handles a single-scope or no-scope-exceedance case;
and `mixture` is not applicable with fewer than four raw boundary violations or
zero violating signed-exceedance variance.  A viable early explanation is
blocking.  Later stages remain unevaluated, all candidate arrays and selection
fields are null or empty, and the disposition is
`INTERVAL_MULTI_Q_COMPETITION_EARLY_DIAGNOSTIC_UNRESOLVED`.

The manifest freezes the four diagnostic formulas rather than leaving their
statistics implicit:

- `reestimate` uses `raw_observed_minus_center`.  It divides raw residuals by
  their global maximum absolute magnitude, computes the pre-rounding center
  shift, represents that shift back in source prediction units, and recomputes
  the effective scaled shift from the represented float.  Both source and
  shifted fit values use `max_scaled_mean_absolute_raw_residual`; the shifted
  statistic is the represented shifted raw residual divided by the raw
  residual scale.  Both are dimensionless max-scaled quantities; the
  fractional gain divides by the source scaled MAE without a numeric epsilon.
  Viability requires both the frozen fractional-MAE gain and a nonincreasing
  raw boundary violation rate.  This represented-shift recomputation prevents
  an ideal but unrepresentable minimum-subnormal beta from creating false gain.
  When discovery has no raw boundary violation, `reestimate` short-circuits as
  not applicable before beta or shifted-residual arithmetic; all eight of its
  numeric report metrics are null, including at extreme finite exact-boundary
  values.
- `noise` uses
  `sign_residual_times_max_zero_abs_residual_minus_radius`, groups by the exact
  scope/context pair, divides signed exceedances by their global maximum
  absolute magnitude, and measures dimensionless total and within-pair
  max-scaled signed-exceedance SSE and their fraction.
- `scope` uses `positive_raw_boundary_exceedance` in
  `source_prediction_units`, divides each value by the global maximum positive
  raw exceedance, aggregates `sum_max_scaled_exceedance_per_scope`, and combines
  its `max_minus_min_divided_by_max_without_numeric_epsilon` ratio with the raw
  boundary-violation-rate spread by taking their maximum.
- `mixture` considers `raw_boundary_violations_only`, uses
  `signed_raw_boundary_exceedance`, divides by the global maximum absolute
  signed exceedance, and chooses the deterministic minimum two-cluster split
  using dimensionless max-scaled SSE, minimum cluster size two,
  lowest-split-index tie breaking, and fractional SSE reduction as its score.

The scope diagnostic is interval-native.  It first divides every positive raw
boundary exceedance by the single global maximum raw exceedance, sums those
dimensionless terms within each scope, and computes
`(maximum scope sum - minimum scope sum) / maximum scope sum` without an
epsilon.  Its structure ratio is the maximum of that quantity and the spread
in raw boundary-violation rates.  This maximum-scaling keeps a represented
minimum-subnormal violation visible while avoiding raw-sum overflow.  The
diagnostic does not use raw residual magnitude or a per-scope normalized mean,
so a scope with a larger registered radius is not treated as structurally worse
merely because it contains larger in-bound residuals.

Raw boundary violation is the strict predicate

```text
abs(observed_value - center_prediction) > radius
```

No normalized statistic is used to decide this raw predicate.  Tail membership
uses the source-defined kth-largest `raw_boundary_exceedance`, measured in
`source_prediction_units`, and every tie at or above the cutoff is included.
Membership is invariant to observation identifiers and row order.  A positive
raw exceedance remains a tail violation even if division by a very large
prediction scale would underflow its normalized exceedance to zero; it is not
tied with a zero or nonviolating row.

## Three bounded candidate families

Only three separately registered families can synthesize candidates.  The
scope diagnostic never becomes an executable scope-restriction family here.

### Interval expansion (`interval_robustify`)

For every registered source radius group, discovery sets

```text
new radius = max(source radius,
                 maximum discovery absolute error in that group)
```

Centers, grouping, group keys, Q/V registries, scopes, and object-space bytes
are preserved.  A candidate exists only when at least one radius strictly
increases; expansion can never shrink a source interval.

### Uniform restriction (`interval_restrict`)

The fixed multiplier registry is `1/4`, `1/2`, `3/4`, and `9/10`.  Every
stored radius is the actual floating-point product of its finite non-negative
source radius and the multiplier.  A candidate is retained only when the
represented model is a strict subset.  All-zero and subnormal-underflow cases
that fail to change the represented radii are excluded rather than mislabeled
as strict restrictions.

### Conservative quotient envelope (`interval_quotient`)

For every non-empty subset of registered removable features, contexts are
projected onto the retained coordinates.  Within each quotient fiber and
across every registered scope, the construction computes

```text
L = min(parent center - parent radius)
U = max(parent center + parent radius)
child center = L/2 + U/2
child radius = max(child center - L, U - child center)
```

The split midpoint avoids the avoidable overflow of `(L + U) / 2`.  Each child
uses exact per-context quotient keys.  Its certificate stores a canonical
`fiber_envelope_table`, the source-seed digest, and the restore method needed to
recover the exact verified parent state; every parent interval must be contained
in its recorded fiber envelope.  A quotient whose source endpoints, hull,
midpoint, radius, or reconstructed stored-float containment are not finitely
representable is omitted; that optional quotient failure does not invalidate
the finite source seed or discard unrelated strict restriction candidates.

Every candidate has the exact revision-2 candidate surface: schema and ID,
family and operation kind, source-state identity, object and model bytes,
model and semantic digests, scopes, removable features, probes and violation
functionals, construction, certificate, discovery metrics and admissibility,
and a validation-evaluation slot.  Semantic deduplication hashes only the
candidate's represented object/model/scope/removable/Q/V semantics.  It is
independent of candidate labels.  Survivors are ordered by family
`robustify`, `restrict`, `quotient`, then ascending candidate ID within a
family; ascending candidate ID is also the representative tie-break within one
family.

The overall candidate commitment binds, in order, the recompetition seed
digest and ID, source theory-state digest, V2 contract digest, discovery
evidence digest, sorted raw candidate bodies without validation, the semantic
deduplication result, and retained candidate IDs.  The
discovery-derived expansion ID binds its synthesis evidence.  Intrinsic
restriction and quotient IDs may remain stable when discovery changes but their
represented structural body does not.  Validation or stress changes cannot
change candidate IDs, represented models, or the overall commitment.

## Validation selection and stress confirmation

Discovery constructs candidates; validation alone selects at most one
provisional winner.  Every weighted component and the resulting score are
dimensionless.  Both every candidate's `validation_evaluation` (when present)
and the top-level `validation_selection`, including its not-performed
placeholder, declare `validation_score_units: dimensionless` consistently with
the contract.  The frozen score is:

```text
1.00 * normalized_center_mae_gain
+ 0.75 * raw_boundary_coverage_gain
+ 0.75 * source_tail_coverage_gain
+ 1.00 * context_reduction_fraction
+ 0.50 * uniform_contraction_fraction
+ 0.50 * normalized_radius_reduction
- 0.75 * max_probe_divergence
- 0.50 * normalized_radius_expansion
```

All candidates must retain at least 0.75 validation coverage and source-tail
coverage, increase MAE by no more than 0.05, keep the two registered Q
divergences within 0.2 and 1.0 respectively, and have non-negative score.
The source-tail cutoff itself is in source prediction units, but source-tail
coverage and its coverage-gain score component are dimensionless fractions.
Expansion additionally requires at least 0.05 coverage gain with bounded
radius cost.  Restriction requires zero discovery and validation boundary
violations.  Quotient requires at least 0.2 representation reduction and no
coverage or tail loss.

Family winners are ranked by descending score and then candidate ID.  If the
top-two score margin is below 0.02, there is no provisional winner.
Stress is evaluated only for the single provisional winner, never for a runner-up.  It
rechecks the common gates and the winning family's additional gates.  Stress
does not change the provisional candidate ID or its validation score.  A
stress failure produces the explicit nonconfirmation disposition with no
fallback, silent rerank, or second stress trial.

There is no fallback after stress nonconfirmation.

## Ten dispositions and null routes

The top-level contract registry maps exactly ten route keys to exactly ten
dispositions:

```text
select_interval_expansion -> SELECT_SHADOW_INTERVAL_EXPANSION_CANDIDATE
select_uniform_restriction -> SELECT_SHADOW_UNIFORM_RESTRICTION_CANDIDATE
select_conservative_quotient -> SELECT_SHADOW_CONSERVATIVE_QUOTIENT_ENVELOPE_CANDIDATE
needs_exact_fresh_evidence -> INTERVAL_MULTI_Q_COMPETITION_NEEDS_EXACT_FRESH_EVIDENCE
incomparable_evaluator_epoch -> INTERVAL_MULTI_Q_COMPETITION_INCOMPARABLE_EVALUATOR_EPOCH
early_diagnostic_unresolved -> INTERVAL_MULTI_Q_COMPETITION_EARLY_DIAGNOSTIC_UNRESOLVED
no_validation_winner -> INTERVAL_MULTI_Q_COMPETITION_NO_VALIDATION_WINNER
provisional_winner_failed_stress_confirmation -> INTERVAL_MULTI_Q_COMPETITION_PROVISIONAL_WINNER_FAILED_STRESS_CONFIRMATION
blocked_adapter_needs_post_restriction_evidence -> INTERVAL_MULTI_Q_COMPETITION_BLOCKED_ADAPTER_NEEDS_POST_RESTRICTION_EVIDENCE
blocked_adapter_incomparable_post_restriction_epoch -> INTERVAL_MULTI_Q_COMPETITION_BLOCKED_ADAPTER_INCOMPARABLE_POST_RESTRICTION_EPOCH
```

The first three set `selection_status` to
`SELECTED_SHADOW_PROPOSAL_NOT_MATERIALIZED` and select a candidate record only.
They do not create an adopted or current theory.  Every blocked, incomplete,
incomparable, diagnostically unresolved, no-winner, or stress-failed route has
an explicit null selection surface with
`selection_status: NO_SELECTED_SHADOW_PROPOSAL`.  `new_probe` may be marked
required after unresolved competition, but this slice does not design or
acquire a probe.  `language_last` remains deferred after that future probe
route and is never executed here.

## Lifecycle, authority, and nonclaims

Competition, qualification, failure-boundary, restriction, adjudication, and
adapter records are audit-only and excluded from V2 scoring.  Discovery rows
are consumed only for diagnostics and construction; validation rows are
consumed only for provisional selection; stress rows are consumed only for the
single provisional winner's confirmation.  All V2 rows become ineligible for
future scoring under the local logical selective-erasure policy.  No physical
deletion or retention-system attestation occurs.

The report records `NOT_ADOPTED_SHADOW_ONLY`, `NOT_PROMOTED`, and `NOT_CURRENT`.
Selection is not materialization, qualification, adoption eligibility,
adoption, promotion, rollback execution, or an authorized
\(H_t\rightarrow H_{t+1}\) transition.  The finite interval hull and frozen
metric checks are representation-level mechanics over caller-supplied static
rows.  They do not establish scientific validity, generalization, generic
quotient correctness, or external data/evaluator authority.

The report's four dynamic authority facts describe work actually performed.
`candidate_synthesis_performed` is true exactly when candidate commitments
exist.  `candidate_evaluation_performed` is true exactly when at least one
candidate has a non-null `validation_evaluation`.
`validation_selection_performed` records entry into the synthesis/commitment
and validation-selection phase, including the zero-candidate case.
`stress_confirmation_performed` is true only when the provisional candidate
was actually stress-evaluated.  No-seed, inexact, incomparable, and early
diagnostic routes set all four false; a validation no-winner leaves stress
false; selected and stress-failed routes set all four true.  Materialization,
adoption, promotion, language expansion, and current-pointer authority remain
false on every route.

## CLI and verification boundary

The runner emits one canonical JSON line to stdout.  With an explicit absolute
`--out`, it writes exactly the same bytes atomically.  Its twenty-four artifact
receipts bind the exact bytes, absolute resolved path, and SHA-256 digest of
every input without modifying them.

Known-answer tests cover exact split cardinality, all ten routes, ordered
diagnostic blocking, stable epochs and commitments, raw-boundary math, tail
cutoff ties, underflow and overflow edges, quotient hull containment, semantic
deduplication, finite bounds, all 276 path and hard-link alias pairs, artifact
tampering, rehashed report tampering, and the no-fallback stress boundary.
Green tests establish only the frozen local mechanics and fail-closed behavior.
This bounded slice is not a complete autonomous theory-evolution loop.

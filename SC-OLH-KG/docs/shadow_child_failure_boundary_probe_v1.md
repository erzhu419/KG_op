# Shadow Child Failure-Boundary Probe V1

## Boundary

This strictly additive V1 exact-replays one locally review-ready shadow-child
packet, deterministically compiles one transition-specific failure-boundary
probe, and evaluates that probe only on caller-supplied static JSON rows.  It
materializes a new content-addressed, probe-expanded shadow state;
it never mutates the source child.

Probe expansion is not external probe acquisition, external review, child
invalidation, rollback, restriction, adoption, promotion, or an
\(H_t\rightarrow H_{t+1}\) transition.  The evaluator epoch is a local content
identity, not an external attestation.  A boundary counterexample is a bounded
verdict on the supplied finite rows, not scientific falsification.  Absence of
a counterexample is not global preservation or a domain-safety claim.

The slice consists only of:

- `performance/shadow_child_failure_boundary_probe.py`;
- `performance/manifests/shadow_child_failure_boundary_probe_v1.json`;
- `runners/run_shadow_child_failure_boundary_probe.py`;
- `tests/test_shadow_child_failure_boundary_probe.py`;
- this document.

The pure core writes nothing.  The runner's explicit `--out` is the only
optional write surface.  The slice does not call a benchmark, `run_one`, a
scheduler, a network service, or an environment/sensor acquisition path.  It
does not change the frozen Operations Research baseline or any paper claim.

## Why this is the next bounded mechanism step

The preceding additive chain compares two registered theory operations,
materializes the selected robustification or quotient child, qualifies its
fixed operational probes under a new local epoch, and binds a non-authoritative
review packet plus record lifecycle.  That packet requires every later score
to use new, unconsumed evidence and forbids pooling old epochs.

This V1 implements the previously missing observation-expansion mechanism in a
bounded form: \(Q\rightarrow Q\cup\{q_{\mathrm{new}}\}\).  Only a packet with
disposition `READY_FOR_EXTERNAL_REVIEW_PACKET_ONLY` can produce a compiled
probe-expanded state.  Packet readiness remains distinct from an external
review outcome and from adoption eligibility.

## Public API

```python
validate_shadow_child_failure_boundary_probe_contract(contract) -> dict

derive_shadow_child_failure_boundary_probe_epoch(
    *,
    review_contract_digest,
    review_report_digest,
    child_theory_state_digest,
    transition_kind,
    fixed_anchor,
    probe_contract,
) -> str

expand_and_evaluate_shadow_child_failure_boundary_probe(
    competition_input,
    competition_contract,
    competition_report,
    transition_contract,
    transition_report,
    qualification_input,
    qualification_contract,
    qualification_report,
    review_contract,
    review_report,
    probe_input,
    probe_contract,
    *,
    expected_competition_contract_digest,
    expected_competition_report_digest,
    expected_competition_input_artifacts,
    expected_transition_contract_digest,
    expected_transition_report_digest,
    expected_transition_input_artifacts,
    expected_qualification_input_digest,
    expected_qualification_contract_digest,
    expected_qualification_report_digest,
    expected_qualification_input_artifacts,
    expected_review_contract_digest,
    expected_review_report_digest,
    expected_review_input_artifacts,
    expected_probe_input_digest,
    expected_probe_contract_digest,
    input_artifacts=None,
)

verify_shadow_child_failure_boundary_probe(
    # the same twelve source inputs, then probe_report
    ...,
    expected_probe_report_digest,
    expected_probe_input_artifacts,
) -> dict
```

The result exposes only `disposition`, `report_digest`, `probe_expanded`,
`boundary_counterexample_found`, and `to_dict()`.  It deliberately has no
`eligible`, `adopt`, `promote`, or `make_current` property.

The builder public-verifies the complete competition, transition,
qualification, and review chain before compiling a probe.  Each upstream
contract, input, report, and artifact map remains independently anchored.  The
public verifier exact-replays the complete report and compares canonical JSON.

## Frozen contract and input

The contract ID is `shadow_child_failure_boundary_probe_v1`, with schema:

```text
sc-olh-kg.shadow-child-failure-boundary-probe-contract/1
```

It pins `shadow_child_external_review_packet_v1` at canonical digest:

```text
sha256:7b438072804c95eee26be901c0839bfea0b65b31824a708940b055ab61f858f1
```

The probe-input schema is:

```text
sc-olh-kg.shadow-child-failure-boundary-probe-input/1
```

Its exact top-level fields bind the probe expansion ID, source review packet,
evaluator, prior-record exclusion commitments, and `holdout`/`stress`
evidence.  Every evidence row contains exactly:

```text
observation_id
evaluator_epoch
fixed_anchor
scope_id
context
observed_value
```

Free-form probe definitions are rejected.  New observation IDs must be unique
and disjoint from both competition and consumed qualification IDs.  Every row
must use the exact derived epoch and inherited fixed anchor.  The epoch must
differ from both previous epochs.

Each split must satisfy
`require_complete_parent_context_scope_pairs_per_split = true`: it covers the
complete Cartesian product of every registered scope and every frozen parent
context.  Merely observing every scope somewhere and every context somewhere
is insufficient, because a sparse cross can conceal a missing within-scope
fiber.  The report binds required, covered, and missing context/scope pairs,
per-split row counts and minimum-row checks, and per-split complete-coverage
booleans.  Incomplete Cartesian coverage yields `NEEDS`, with no numerical
probe result.

The new epoch is a canonical commitment to:

```text
review contract digest
review report digest
source child state digest
transition kind
fixed anchor
probe contract digest
expanded fixed probe registry
```

with prefix `shadow-failure-boundary-probe-epoch:`.  Evidence values and row
order cannot influence the epoch, compiled probe definition, or expanded-state
digest.

## Fixed transition-specific probes

The contract contains exactly two registered transition/probe pairs.

### Robust interval expansion

`ROBUST_INTERVAL_EXPANSION` adds only:

```text
normalized_signed_interval_boundary_margin
```

For each row its functional is:

```text
(radius - abs(observed_value - center)) / prediction_scale
```

A normalized margin below zero is a boundary counterexample.  Split results
bind the row count, minimum signed margin, boundary-violation rate, mean
normalized exceedance, and maximum normalized exceedance.

### Quotient idealization

`QUOTIENT_IDEALIZATION` adds only:

```text
deleted_feature_conditional_response_spread

functional =
  (max_within_scope_fiber_context_mean
   - min_within_scope_fiber_context_mean) / prediction_scale

aggregation =
  maximum_normalized_nontrivial_scope_fiber_spread_by_fresh_split
```

Within each registered scope and nontrivial quotient fiber it computes the
maximum parent-context mean minus the minimum parent-context mean, normalized
by `prediction_scale`.  Scope is part of the fiber identity; effects from
different scopes are never pooled before this spread is calculated.  A
normalized spread above `0.20` is a boundary counterexample.  Split results
bind the row count, evaluated nontrivial-fiber count, maximum normalized fiber
response spread, offending-fiber count, and content digests of offending
scope/fiber pairs.

Both probes use the observation-independent frozen normalization:

```text
prediction_scale = max(
    1e-12,
    mean(abs(parent_point_predictions))
    + parent_absolute_error_threshold,
)
```

The scale depends only on the frozen parent theory and contract, not the new
observed values.

## Probe-expanded shadow state

Only a ready source packet materializes a state with schema:

```text
sc-olh-kg.shadow-probe-expanded-theory-state/1
```

The state copies the source child semantics and appends exactly one registered,
transition-specific probe ID.  Its `violation_functionals` remains canonical
byte-equal to the source child's frozen \(V\): this slice expands \(Q\) only.
The new probe semantics are bound by the frozen contract, report
`probe_definition`, and probe-expansion lineage, not by mutating \(V\).  The
state also binds the source child digest, source review packet digest, local
evaluator epoch and anchor, source scope/removable-feature registries,
evidence-reuse policy, and lineage.  Its fixed authority fields are:

```text
evaluator_status = LOCAL_FAILURE_BOUNDARY_PROBE_EPOCH_UNATTESTED
operational_probe_status = FAILURE_BOUNDARY_PROBE_COMPILED_SHADOW_ONLY
adoption_status = NOT_ADOPTED_SHADOW_ONLY
current_status = NOT_CURRENT
```

`theory_id` is content-addressed from the canonical state payload excluding
`theory_id`.  The original child and parent remain unchanged.

## Dispositions and precedence

The disposition is exactly one of:

```text
EXPANDED_PROBE_NO_BOUNDARY_COUNTEREXAMPLE_ON_SUPPLIED_EVIDENCE
EXPANDED_PROBE_BOUNDARY_COUNTEREXAMPLE_FOUND_SHADOW_ONLY
EXPANDED_PROBE_NEEDS_NEW_EVIDENCE
EXPANDED_PROBE_INCOMPARABLE_EVALUATOR_EPOCH
PROBE_EXPANSION_BLOCKED_SOURCE_PACKET_NOT_READY
```

Precedence is fixed:

1. A non-ready source packet is `BLOCKED`; probe definition, expanded state,
   and results are all `null`.
2. A ready packet with a mismatched epoch or anchor is `INCOMPARABLE`; the
   probe is compiled, but results are `null`.
3. Comparable but insufficient rows or incomplete registered-scope by
   parent-context Cartesian coverage are `NEEDS`; the probe is compiled, but
   results are `null`.
4. Complete comparable evidence reports whether a boundary counterexample was
   found on the supplied finite rows.

No disposition authorizes invalidation, rollback, restriction, adoption,
promotion, current-pointer writes, or parent/child writes.

## Record lifecycle extension

The report extends the frozen logical lifecycle as follows:

```text
competition records
  = AUDIT_ONLY_SCORING_EXCLUDED
qualification records
  = CONSUMED_AUDIT_ONLY_SCORING_EXCLUDED
new probe records
  = CONSUMED_FAILURE_BOUNDARY_EVIDENCE_AUDIT_ONLY
```

All three classes have `eligible_for_future_scoring = false`.  Any future score
requires new, unconsumed evidence; cross-epoch pooling remains false.  Logical
selective erasure is applied, while:

```text
physical_erasure = NOT_PERFORMED
physical_retention_attestation = REQUIRED_NOT_PRESENT
```

The attestation boundary always also records:

```text
external_data_attestation = REQUIRED_NOT_PRESENT
external_evaluator_attestation = REQUIRED_NOT_PRESENT
external_adoption_authority = REQUIRED_NOT_PRESENT
```

## Exact report and audit

The report schema is:

```text
sc-olh-kg.shadow-child-failure-boundary-probe-report/1
```

It binds the full source packet, source-child digest and transition kind;
compiled probe definition and expanded-state digest; evaluator and evidence
bindings; lifecycle extension; split/aggregate probe results; disposition and
boundary assessment; absent attestations; withheld adoption/promotion/current
statuses; input artifacts; canonical audit chain; and report digest.

On a complete ready path the audit event types are exactly:

```text
SOURCE_REVIEW_PACKET_VERIFIED
FAILURE_BOUNDARY_PROBE_COMPILED
NEW_UNCONSUMED_EVIDENCE_ISOLATION_BOUND
FAILURE_BOUNDARY_PROBE_ASSESSED_AND_AUTHORITY_WITHHELD
```

Every action bit in the boundary assessment is false:

```text
source_child_invalidated
rollback_execution_allowed
restriction_execution_allowed
adoption_decision_allowed
promotion_decision_allowed
current_pointer_write_allowed
parent_or_child_state_write_allowed
```

## CLI

```bash
python3 runners/run_shadow_child_failure_boundary_probe.py \
  --competition-input /absolute/competition_input.json \
  --competition-contract /absolute/theory_operation_competition_v1.json \
  --competition-report /absolute/competition_report.json \
  --transition-contract /absolute/shadow_theory_transition_v1.json \
  --transition-report /absolute/transition_report.json \
  --qualification-input /absolute/qualification_input.json \
  --qualification-contract /absolute/shadow_child_probe_qualification_v1.json \
  --qualification-report /absolute/qualification_report.json \
  --review-contract /absolute/shadow_child_external_review_packet_v1.json \
  --review-report /absolute/review_report.json \
  --probe-input /absolute/probe_input.json \
  --probe-contract /absolute/shadow_child_failure_boundary_probe_v1.json \
  --expected-competition-contract-digest sha256:... \
  --expected-competition-report-digest sha256:... \
  --expected-transition-contract-digest sha256:... \
  --expected-transition-report-digest sha256:... \
  --expected-qualification-input-digest sha256:... \
  --expected-qualification-contract-digest sha256:... \
  --expected-qualification-report-digest sha256:... \
  --expected-review-contract-digest sha256:... \
  --expected-review-report-digest sha256:... \
  --expected-probe-input-digest sha256:... \
  --expected-probe-contract-digest sha256:... \
  --out /absolute/probe_report.json
```

The builder consumes exactly twelve absolute, existing regular, non-symlink
JSON files.  Their resolved paths and device/inode pairs must all differ, so
all 66 input pairs are protected from path and hard-link aliases.  `--out` may
not alias any input.  The runner reconstructs the exact prior 2-, 4-, 7-, and
9-artifact maps, plus the exact 12-artifact probe map.

Duplicate JSON keys, `NaN`, `Infinity`, non-object roots, and digest mismatches
fail with exit status 2 and no report on stdout.  Success emits canonical JSON
plus one newline.  `--out`, when explicit, receives exactly the same bytes via
an atomic replacement.

## What may and may not be concluded

The artifact supports only these local claims: the upstream chain and ready
packet exact-replayed; exactly one fixed transition-specific probe was appended
to an immutable shadow child; new rows were isolated from all consumed records
and evaluated under a distinct local epoch; and those finite rows did or did
not cross the declared boundary.

It does not establish external provenance or evaluator authority, an external
review outcome, physical retention or erasure, global preservation, scientific
falsification, domain safety, generalization, rollback/restriction execution,
child invalidation, adoption eligibility, adoption, promotion, current state,
or an \(H_t\rightarrow H_{t+1}\) acceptance.

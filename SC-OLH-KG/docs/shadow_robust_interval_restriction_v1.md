# Shadow Robust Interval Restriction V1

## Boundary

This strictly additive V1 exact-replays the five preceding shadow slices and,
only for one qualifying robust source, competes a frozen finite registry of
uniform interval contractions on fresh caller-supplied static rows.  If a
candidate passes the bounded gates, the core materializes a new
content-addressed shadow object whose model class is a mechanically verified
strict subset of the source finite interval table:

\[
\mathcal M_{\alpha}\subsetneq\mathcal M_{\rm source}.
\]

The source probe-expanded state remains byte-unchanged and valid as the exact
rollback target.  Materialization is not adoption, promotion, current-pointer
movement, external review, external attestation, physical erasure, or an
\(H_t\rightarrow H_{t+1}\) transition.  The new object requires another fresh
evaluator epoch before any later scoring.

The slice consists only of:

- `performance/shadow_robust_interval_restriction.py`;
- `performance/manifests/shadow_robust_interval_restriction_v1.json`;
- `runners/run_shadow_robust_interval_restriction.py`;
- `tests/test_shadow_robust_interval_restriction.py`;
- this document.

The pure core writes nothing.  The runner's explicit `--out` is the only
optional write surface.  The slice does not call a benchmark, `run_one`, a
scheduler, a network service, or an acquisition path.  It does not edit the
frozen Operations Research worktree, its manuscript, or any prior KG-op file
or claim.

## Applicability and source gate

The frozen contract is `shadow_robust_interval_restriction_v1`, with schema:

```text
sc-olh-kg.shadow-robust-interval-restriction-contract/1
```

It pins `shadow_child_failure_boundary_probe_v1` at canonical digest:

```text
sha256:fdc92e276f7d8cb0c1ab6fd097242932851da04e1f97888d3f9597bfb0f726e0
```

Restriction competition is applicable only when exact replay establishes all
of the following:

```text
transition_kind = ROBUST_INTERVAL_EXPANSION
source probe disposition =
  EXPANDED_PROBE_NO_BOUNDARY_COUNTEREXAMPLE_ON_SUPPLIED_EVIDENCE
probe-expanded shadow state is materialized
model_class.kind = finite_interval_table
```

The fifth slice's failure to find a counterexample is only an entry condition.
It does not justify contraction.  This slice therefore requires an independent
new local epoch and three new evidence splits.  Quotient sources are explicitly
not applicable.  A source counterexample blocks restriction, and every
unresolved fifth-slice outcome remains unresolved rather than being reinterpreted
as support for restriction.

## Public API

```python
validate_shadow_robust_interval_restriction_contract(contract) -> dict

derive_shadow_robust_interval_restriction_epoch(
    *,
    probe_contract_digest,
    probe_report_digest,
    probe_expanded_shadow_theory_state_digest,
    fixed_anchor,
    restriction_contract,
) -> str

compete_and_materialize_shadow_robust_interval_restriction(
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
    probe_report,
    restriction_input,
    restriction_contract,
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
    expected_probe_report_digest,
    expected_probe_input_artifacts,
    expected_restriction_input_digest,
    expected_restriction_contract_digest,
    input_artifacts=None,
)

verify_shadow_robust_interval_restriction(
    # the same fifteen source inputs, then restriction_report
    ...,
    expected_restriction_report_digest,
    expected_restriction_input_artifacts,
) -> dict
```

The result exposes only `disposition`, `report_digest`,
`restriction_materialized`, `selected_radius_multiplier`, and `to_dict()`.
There is no `eligible`, `adopt`, `promote`, or `make_current` capability.  The
public verifier exact-replays the full report and compares canonical JSON.

## Frozen input and local epoch

The input schema is:

```text
sc-olh-kg.shadow-robust-interval-restriction-input/1
```

Its exact top-level fields are:

```text
schema_version
restriction_id
source_failure_boundary_probe
evaluator
prior_record_exclusion
evidence
```

`source_failure_boundary_probe` binds the probe contract/report digests, probe
expansion ID, and probe-expanded state digest.  `prior_record_exclusion` binds
the observation-ID digests for competition, qualification, and failure-boundary
records.  `evidence` contains exactly `calibration`, `holdout`, and `stress`.
Every row contains exactly:

```text
observation_id
evaluator_epoch
fixed_anchor
scope_id
context
observed_value
```

Each split covers the complete Cartesian product of every registered scope and
every frozen parent context.  Every new observation ID is unique across all
three splits and disjoint from all three prior generations.  Incomplete
coverage yields a non-numerical `NEEDS` report.  Epoch or anchor mismatch yields
a non-numerical `INCOMPARABLE` report.  Duplicate keys, non-finite values,
unregistered pairs, ID reuse, or forged upstream commitments fail closed.

The local epoch is content-addressed from the probe contract digest, exact
probe report digest, probe-expanded state digest, fixed anchor, restriction
contract digest, and frozen multiplier registry.  It does not bind observed
values or row order.  It is a local selection identity, not external evaluator
attestation.

## Fixed grammar and competition

V1 admits exactly four evidence-independent uniform contractions:

```text
candidate                  exact multiplier
uniform_radius_1_over_4    1/4
uniform_radius_1_over_2    1/2
uniform_radius_3_over_4    3/4
uniform_radius_9_over_10   9/10
```

Callers cannot provide a multiplier or constraint.  For every radius group
\(g\), a candidate retains the exact source center, grouping, and group key and
sets

\[
r'_g=\alpha r_g,\qquad 0<\alpha<1.
\]

The observation-independent scale is frozen as

\[
s=\max\!\left(10^{-12},
\operatorname{mean}|c(x)|+
\text{parent absolute-error threshold}\right).
\]

For row \(i\), the report binds both source and candidate margins:

\[
m_i^{\rm src}=\frac{r_{g(i)}-|y_i-c(x_i)|}{s},\qquad
m_{i,\alpha}=\frac{\alpha r_{g(i)}-|y_i-c(x_i)|}{s}.
\]

Calibration determines only which frozen candidates have minimum candidate
margin at least zero.  Holdout and stress then compete only those admissible
candidates; both must have zero boundary-violation rate.  The smallest passing
multiplier is selected.  A failure never causes multiplier editing, candidate
creation, or data-driven geometry.

Candidate payloads, IDs, restricted model-class digests, and commitments are
constructed solely from the source state, contract, and exact fixed multiplier.
Changing observed values can change admissibility, selection, and disposition,
but cannot change the candidate registry.

## Strict-subset certificate

For the complete context/scope product, the certificate checks:

```text
centers byte-equal
radius grouping and group keys byte-equal
all restricted radii finite and nonnegative
every restricted radius <= its source radius
at least one source radius strictly reduced
```

At least one source radius must be strictly positive.  Together with
\(0<\alpha<1\), these checks mechanically establish a strict subset for the
bound finite product of scalar intervals.  They do not establish nominal
utility, domain safety, global preservation, or any restriction theorem beyond
this frozen `finite_interval_table` representation.

The certificate kind is:

```text
STRICT_FINITE_INTERVAL_SUBSET_BY_UNIFORM_RADIUS_CONTRACTION
```

The implementation checks `global`, `per_scope`, and `per_context` radius
grouping without changing the frozen grouping semantics.  The full fifth-slice
probe registry \(Q\) and violation-functional registry \(V\) are copied
canonical-byte-equal.  This slice creates neither a new probe nor a new
violation functional.

## Restricted shadow object and rollback

A successful report materializes schema:

```text
sc-olh-kg.shadow-robust-interval-restricted-theory-state/1
```

The object copies the source object space, scopes, removable features, \(Q\),
and \(V\), and binds the selected restricted model class and complete
restriction lineage.  Its authority fields remain:

```text
operation_kind = restrict
restriction_kind = UNIFORM_RADIUS_CONTRACTION
evaluator_epoch = null
evaluator_status = POST_RESTRICTION_FRESH_EVALUATOR_REQUIRED
operational_probe_status =
  REQUALIFICATION_WITH_NEW_EPOCH_REQUIRED_AFTER_MODEL_RESTRICTION
adoption_status = NOT_ADOPTED_SHADOW_ONLY
current_status = NOT_CURRENT
```

All calibration, holdout, and stress rows are consumed by this report and are
ineligible for future scoring.  The restricted state must receive new,
unconsumed evidence under another fresh epoch.

Rollback is bound to restoring the exact frozen unrestricted probe-expanded
state from the verified fifth-slice report:

```text
RESTORE_FROZEN_PROBE_EXPANDED_SOURCE_STATE_FROM_VERIFIED_PROBE_REPORT
```

The rollback boundary also binds the materialized child, original parent,
source/restricted model-class digests, and `NOT_PERFORMED`.  It never treats
division by the multiplier as a reliable recovery mechanism and never executes
rollback.

## Dispositions and precedence

The eight exact dispositions are:

```text
MATERIALIZED_SHADOW_ROBUST_INTERVAL_RESTRICTION
NO_CALIBRATION_ADMISSIBLE_STRICT_INTERVAL_RESTRICTION
NO_VALIDATED_STRICT_INTERVAL_RESTRICTION
RESTRICTION_NEEDS_NEW_EVIDENCE
RESTRICTION_INCOMPARABLE_EVALUATOR_EPOCH
RESTRICTION_BLOCKED_SOURCE_BOUNDARY_COUNTEREXAMPLE
RESTRICTION_BLOCKED_SOURCE_PROBE_UNRESOLVED
RESTRICTION_NOT_APPLICABLE_NON_ROBUST_SOURCE
```

Precedence is: non-robust source; source counterexample; other unresolved source
probe outcomes; epoch/anchor mismatch; insufficient row count or Cartesian
coverage; no calibration-admissible candidate; no holdout/stress-validated
candidate; then materialization of the smallest passing multiplier.  Structurally
invalid input, forged upstream state, or digest/artifact mismatch raises an
error rather than manufacturing a disposition.

## Runner containment

The runner reads exactly fifteen absolute, existing, non-symlink JSON inputs.
All paths and inodes must be distinct, so all
\(\binom{15}{2}=105\) input-alias pairs are rejected.  It builds independent
2/4/7/9/12/15-artifact maps for exact upstream replay.  Duplicate JSON keys,
non-object JSON, non-finite constants, relative paths, symlinks, hard links,
and input/output aliases fail before the core runs.  Canonical report JSON is
written to standard output; `--out` optionally copies those exact bytes by an
atomic replacement.

## Explicit nonclaims

The report freezes the following nonclaims:

```text
shadow_only
robust_finite_interval_restriction_only
no_generic_restriction_engine
no_rigid_body_markov_or_independence_assumption
no_scope_restriction
no_quotient_restriction
no_new_probe
q_registry_copied_not_requalified
v_registry_unchanged
no_language_or_predicate_invention
no_external_probe_acquisition
caller_supplied_static_rows_only
local_epoch_is_not_external_attestation
fresh_evidence_pass_is_not_global_preservation
interval_width_reduction_is_not_nominal_utility_or_domain_safety
no_source_child_invalidation
no_rollback_execution
no_adoption_eligibility_determination
no_adoption
no_promotion
no_current_pointer_write
no_parent_source_or_ambient_state_write
no_h_t_to_h_t_plus_1_acceptance
no_cross_epoch_pooling
no_physical_erasure
no_external_data_or_evaluator_attestation
no_run_one
no_benchmark_execution
no_scheduler_or_network_access
no_operations_research_baseline_or_claim_change
no_paper_promotion
explicit_cli_out_is_only_optional_write
input_artifacts_require_independent_expected_values
report_digest_is_not_a_signature
no_scientific_validity_or_generalization_claim
```

This is the first narrow executable `restrict` family in the additive Meta-prior
chain.  It is not a generic restriction engine or a completed recursive loop.
Post-restriction requalification, routing, active probe acquisition,
language/predicate invention, external attestations, review outcomes, and a
real adoption authority remain absent.

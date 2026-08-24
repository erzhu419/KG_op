# Shadow Child Probe Qualification V1

## Boundary

This additive V1 qualifies one exact-replayed, materialized shadow child with
new holdout and stress records under a content-derived evaluator epoch.  It is
the evaluator/probe gate deliberately left open by Shadow Theory Transition
V1.  It does not adopt the child, mutate either parent or child, execute or
acquire a probe, reuse source competition evidence for child scoring, run a
benchmark, contact a scheduler or network service, or change an Operations
Research result or paper claim.  The pure core writes nothing.  The runner's
explicit `--out` is the only optional write surface.

The slice consists only of:

- `performance/shadow_child_probe_qualification.py`;
- `performance/manifests/shadow_child_probe_qualification_v1.json`;
- `runners/run_shadow_child_probe_qualification.py`;
- `tests/test_shadow_child_probe_qualification.py`;
- this document.

It consumes only the two earlier additive slices.  It neither imports nor
calls the historical benchmark, `run_one`, a scheduler, or an environment.

## Public API

```python
validate_shadow_child_probe_qualification_contract(contract) -> dict

derive_shadow_child_evaluator_epoch(
    *,
    transition_contract_digest,
    transition_report_digest,
    child_theory_state_digest,
    transition_kind,
    fixed_anchor,
    qualification_contract,
) -> str

qualify_shadow_child_operational_probes(
    competition_input,
    competition_contract,
    competition_report,
    transition_contract,
    transition_report,
    qualification_input,
    qualification_contract,
    *,
    expected_competition_contract_digest,
    expected_competition_report_digest,
    expected_competition_input_artifacts,
    expected_transition_contract_digest,
    expected_transition_report_digest,
    expected_transition_input_artifacts,
    expected_qualification_input_digest,
    expected_qualification_contract_digest,
    input_artifacts=None,
)

verify_shadow_child_probe_qualification(
    competition_input,
    competition_contract,
    competition_report,
    transition_contract,
    transition_report,
    qualification_input,
    qualification_contract,
    qualification_report,
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
) -> dict
```

Qualification starts by replay-verifying the source competition and source
transition against independently retained digests and artifact maps.  The
qualification verifier additionally binds the qualification input, contract,
report, and artifact map and then replays the full computation.  A report,
transition, evaluator epoch, evidence-exclusion declaration, or digest is not
self-authenticating.

## Exact qualification input

The input schema is
`sc-olh-kg.shadow-child-probe-qualification-input/1`.  Its top-level keys are
exactly:

```text
schema_version
qualification_id
source_transition
evaluator
source_evidence_exclusion
evidence
```

`source_transition` has exactly the verified transition contract digest,
transition report digest, and child-theory-state digest.  `evaluator` has the
declared new evaluator epoch and inherited fixed anchor.
`source_evidence_exclusion` has exactly:

```text
policy = EXCLUDE_ALL_SOURCE_COMPETITION_RECORDS
source_evidence_digests
```

The digest mapping must exactly match the verified source competition's
`discovery`, `validation`, and `stress` evidence digests.  It is an exclusion
binding, not a permission to score the child with those records.

`evidence` has exactly `holdout` and `stress`.  Every row has exactly:

```text
observation_id
evaluator_epoch
fixed_anchor
scope_id
context
observed_value
```

Values and contexts are finite and drawn from the registered parent finite
object space and scopes.  Each split must contain at least four rows and cover
every parent context/scope pair.  New IDs must be unique across both splits
and disjoint from all source competition observation IDs.

## Fresh evaluator epoch

The new epoch is a deterministic content commitment to:

- the transition contract digest;
- the transition report digest;
- the materialized child-theory-state digest;
- the transition kind;
- the inherited fixed anchor;
- the qualification contract digest;
- the fixed operational-probe registry for that transition kind.

Every supplied holdout and stress row must carry that exact derived epoch and
the exact inherited fixed anchor.  The epoch must differ from the parent
source epoch.  The binding records all declared and observed epochs and
anchors, requires comparability, and always records
`old_new_records_pooled = false`.

The epoch is content-derived local identity.  It is not an external evaluator
attestation.  Likewise, fixed-anchor equality is a replay binding and not
external authority.

## Fixed probe registries

V1 accepts only the contract's transition-specific registry.  It cannot take
free-form probe definitions from the qualification input and does not mutate
the child's declared probe registry.

For `ROBUST_INTERVAL_EXPANSION`, the fixed probes are:

```text
absolute_error_point_prediction
interval_containment
tail_interval_containment
normalized_interval_radius
```

The core recomputes parent-center and child-center mean absolute error,
nominal and interval coverage, coverage gain, radius statistics, tail interval
coverage, scalar safety-rate containment, prediction scale, and normalized
radius on the new evidence.  Qualification requires:

```text
holdout coverage gain >= 0.05
holdout interval coverage >= 0.75
stress tail interval coverage >= 0.75
stress safety rate >= 0.75
holdout and stress nominal MAE increase <= 0.05
normalized radius <= 1.0
```

Interval and tail containment are finite scalar diagnostics.  They are not
domain safety, CVaR, or worst-case safety claims.

For `QUOTIENT_IDEALIZATION`, the fixed probes are:

```text
absolute_error_point_prediction
parent_child_point_divergence
```

The child prediction is evaluated after the verified quotient projection.
The core recomputes parent and child MAE and complete finite parent/child point
divergence.  Qualification requires both split MAE increases to be at most
`0.05` and aggregate divergence to be at most `0.20`.  This is task- and
finite-evidence-scoped; it does not turn the earlier proxy certificate into a
global preservation theorem.  Exact quotient recovery still requires the
frozen parent snapshot from the transition.

Every gate record has exactly:

```text
gate_id
metric_path
operator
threshold
actual
passed
```

Row-local interval or idealization counterexamples are bound when present.
Aggregate-only failures remain fully bound by their explicit `gate_checks` and
need not identify an individual observation as a counterexample.

## Evidence isolation and selective erasure

The report binds source and new observation-ID sets and digests separately.
The source evidence is used only to replay the upstream decisions and prove
the exclusion set.  It is never included in a child metric.  The selective
erasure receipt records:

```text
source_evidence_used_for_child_scoring = false
old_new_records_pooled = false
logical_selective_erasure_applied = true
physical_records_deleted = false
```

This is logical selective erasure at the evaluation boundary.  The slice does
not delete any physical record.

## Dispositions

The public disposition is exactly one of:

```text
QUALIFIED_NEW_EVALUATOR_EPOCH
NEEDS_NEW_EVALUATOR_EVIDENCE
INCOMPARABLE_NEW_EVALUATOR_EPOCH
FAILED_OPERATIONAL_PROBE_QUALIFICATION
```

An epoch or anchor mismatch takes precedence over insufficient evidence and
produces `INCOMPARABLE_NEW_EVALUATOR_EPOCH`.  Insufficient row count, context
or scope coverage produces `NEEDS_NEW_EVALUATOR_EVIDENCE`.  Reusing a source
observation ID or duplicating a new ID is a structurally invalid input and
fails closed rather than producing a report.  The incomparable and needs
states both have `probe_results = null`, so
numbers from incomparable or incomplete records are not presented as a probe
verdict.  Complete comparable evidence yields either qualified or failed
according to the fixed gates.

`qualification_binding` is non-null for all four dispositions.  Its status
mapping is:

| disposition | evaluator status | operational-probe status |
|---|---|---|
| `QUALIFIED_NEW_EVALUATOR_EPOCH` | `OPERATIONAL_PROBE_QUALIFIED_SHADOW_ONLY` | `QUALIFIED_ON_NEW_HOLDOUT_AND_STRESS` |
| `NEEDS_NEW_EVALUATOR_EVIDENCE` | `NEW_EVALUATOR_EVIDENCE_INCOMPLETE` | `OPERATIONAL_PROBE_EVIDENCE_REQUIRED` |
| `INCOMPARABLE_NEW_EVALUATOR_EPOCH` | `NEW_EVALUATOR_EVIDENCE_INCOMPARABLE` | `OPERATIONAL_PROBE_EVIDENCE_INCOMPARABLE` |
| `FAILED_OPERATIONAL_PROBE_QUALIFICATION` | `OPERATIONAL_PROBE_QUALIFICATION_FAILED` | `OPERATIONAL_PROBE_FAILED_ON_NEW_HOLDOUT_OR_STRESS` |

The bound evaluator epoch remains the derived epoch, the anchor remains the
inherited anchor, and qualification status is the disposition.  Adoption is
always `NOT_ADOPTED_SHADOW_ONLY`.

## Exact report

The report schema is
`sc-olh-kg.shadow-child-probe-qualification-report/1` and has exactly:

```text
schema_version
contract_id
contract_digest
qualification_input_digest
source_transition
qualification_id
child_theory_state_digest
transition_kind
evaluator_definition
evaluator_binding
evidence_binding
selective_erasure_receipt
probe_results
disposition
qualification_binding
adoption_status
nonclaims
input_artifacts
audit_events
audit_head
report_digest
```

`source_transition` binds the upstream verification status, identities,
digests, selected candidate, transition kind, and non-adoption status.
`evaluator_definition` and `evaluator_binding` bind the fixed registry and new
epoch.  `evidence_binding` binds row counts, complete context/scope coverage,
source/new ID commitments, and sufficiency.  The erasure receipt makes source
exclusion explicit.  The qualification binding states the child digest,
epoch, anchor, status, lack of mutation, score-reuse prohibition, and logical
erasure.  Audit and report digests commit the canonical report body; they are
not signatures.

## CLI

```bash
python3 runners/run_shadow_child_probe_qualification.py \
  --competition-input /absolute/competition_input.json \
  --competition-contract /absolute/theory_operation_competition_v1.json \
  --competition-report /absolute/competition_report.json \
  --transition-contract /absolute/shadow_theory_transition_v1.json \
  --transition-report /absolute/transition_report.json \
  --qualification-input /absolute/qualification_input.json \
  --qualification-contract /absolute/shadow_child_probe_qualification_v1.json \
  --expected-competition-contract-digest sha256:... \
  --expected-competition-report-digest sha256:... \
  --expected-transition-contract-digest sha256:... \
  --expected-transition-report-digest sha256:... \
  --expected-qualification-input-digest sha256:... \
  --expected-qualification-contract-digest sha256:... \
  --out /absolute/qualification_report.json
```

All seven inputs must be absolute, existing, non-symlink files.  The runner
reconstructs the exact upstream competition and transition artifact maps and
binds all seven files into the qualification artifact map.  It rejects
duplicate JSON keys, non-finite constants, non-object roots, relative or
symlinked inputs, any two inputs that share a resolved path or inode, and an
output that aliases, overwrites, or is hard-linked to any input.  Input
identity is checked before any JSON is read or any core is imported.  Success
emits one compact, key-sorted canonical JSON object plus a
newline.  Optional `--out` is an identical atomic copy.  Failure returns 2 and
emits no report on stdout.

## Claim boundary and next gate

The KATs prove deterministic replay, evidence isolation, finite probe
mechanics, and CLI immutability for bounded robust and quotient fixtures.  They
do not prove autonomous probe invention, acquisition, external evaluator
authority, provenance beyond the bound artifacts, domain safety, scientific
validity, or generalization.

A qualified receipt closes only the fresh-evaluator/operational-probe gate for
this shadow child and this finite evidence contract.  It is not adoption and
does not establish an `H_t -> H_{t+1}` acceptance decision.  Any later
acceptance layer must remain a separate additive slice with its own authority
and claim boundary.

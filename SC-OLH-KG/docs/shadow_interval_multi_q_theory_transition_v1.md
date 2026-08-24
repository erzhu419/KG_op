# Shadow Interval/Multi-Q Theory Transition V1

## Boundary

This strictly additive V1 exact-replays the complete interval/multi-Q
competition V2 chain and materializes only its verified, validation-selected,
stress-confirmed proposal as one detached shadow child. It does not rerank a
candidate, evaluate a runner-up, use fallback, rerun validation or stress,
create a fresh evaluator epoch, perform fresh post-transition qualification,
decide adoption eligibility, adopt, promote, or write a current pointer. The
pure core writes nothing. The runner's explicit `--out` is the only optional
write surface.

The slice consists only of:

- `performance/shadow_interval_multi_q_theory_transition.py`;
- `performance/manifests/shadow_interval_multi_q_theory_transition_v1.json`;
- `runners/run_shadow_interval_multi_q_theory_transition.py`;
- `tests/test_shadow_interval_multi_q_theory_transition.py`;
- this document.

It is a bypass-style extension over the additive Meta-prior chain. It does not
modify any earlier slice, benchmark, historical result, or the separate
Operations Research paper baseline or claim surface.

## Public API

```python
validate_shadow_interval_multi_q_theory_transition_contract(contract) -> dict

materialize_shadow_interval_multi_q_theory_transition(
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
    restriction_report,
    adjudication_input,
    adjudication_contract,
    adjudication_report,
    adapter_input,
    adapter_contract,
    adapter_report,
    interval_competition_input,
    interval_competition_contract,
    interval_competition_report,
    interval_transition_contract,
    *,
    # twenty-five independent digest anchors,
    # prior-layer expected artifact maps,
    input_artifacts=None,
) -> ShadowIntervalMultiQTheoryTransitionResult

verify_shadow_interval_multi_q_theory_transition(
    # the same twenty-six inputs,
    interval_transition_report,
    *,
    # the same twenty-five source anchors and artifact maps,
    expected_interval_transition_report_digest,
    expected_interval_transition_input_artifacts,
) -> dict
```

The builder has exactly twenty-six positional inputs: the twenty-four V2
competition inputs, its report, and this transition contract. The verifier has
the same inputs plus the transition report, for twenty-seven positional
objects. The result wrapper exposes only `disposition`, `report_digest`, and
`to_dict`.

The core first invokes the public interval/multi-Q competition V2 verifier.
Every source contract, input, report, and prior artifact map is therefore
checked by exact replay before the selected candidate is read. A source report
whose body and self-hash were forged together is not admissible.

## Exact report and child surfaces

The transition report has exactly:

```text
schema_version
contract_id
contract_digest
source_interval_competition
parent_theory_state
parent_theory_state_digest
disposition
operation_kind
transition_kind
selected_candidate_id
selected_candidate_family
selected_candidate_binding
child_theory_state
child_theory_state_digest
materialization_certificate
preservation_certificate
rollback_boundary
evaluator_gate
record_lifecycle_extension
authority_boundary
adoption_eligibility
adoption_status
promotion_status
current_status
nonclaims
input_artifacts
audit_events
audit_head
report_digest
```

A materialized child has exactly:

```text
schema_version
theory_id
task_id
evaluator_epoch
evaluator_status
fixed_anchor
object_space
model_class
model_class_digest
semantic_model_digest
probe_ids
violation_functionals
scope_ids
removable_feature_ids
evidence_reuse_policy
operational_probe_status
transition_lineage
```

The report digest commits the complete canonical report body. It is neither a
signature nor external attestation.

## Exact provenance closure

The twenty-five independent digest anchors cover every contract, input, and
report after the initial competition input:

```text
competition contract/report
transition contract/report
qualification input/contract/report
review contract/report
probe input/contract/report
restriction input/contract/report
adjudication input/contract/report
adapter input/contract/report
interval competition input/contract/report
interval transition contract
```

The exact cumulative artifact maps occur at layers
`2/4/7/9/12/15/18/21/24/26`. The transition runner reconstructs each map from
the bytes it actually reads. Every final receipt contains only `bytes`,
absolute `path`, and raw-file `sha256`; observation values are never embedded
in artifact metadata.

The source competition contract is exactly
`shadow_interval_multi_q_theory_operation_competition_v2`, with canonical
digest
`sha256:4c30c0b1a2cdec92ab1676e98677b620907bb9652bff1ce71865fce9d45ccd1e`.
The source adapter remains pinned to
`sha256:16d2a30873e3f8b2e56fe5d7ac272140eb83dbcb441d8d80a892c4028f28f029`.
The canonical transition V1 contract digest is
`sha256:b1a5f1761c2cafcae24f37f22810178074b9fc7800b6d73bdfd631be3b1df86d`.

## Total source-to-transition routing

Exactly three selected source dispositions can materialize:

```text
SELECT_SHADOW_INTERVAL_EXPANSION_CANDIDATE
  -> MATERIALIZED_SHADOW_INTERVAL_EXPANSION

SELECT_SHADOW_UNIFORM_RESTRICTION_CANDIDATE
  -> MATERIALIZED_SHADOW_UNIFORM_INTERVAL_RESTRICTION

SELECT_SHADOW_CONSERVATIVE_QUOTIENT_ENVELOPE_CANDIDATE
  -> MATERIALIZED_SHADOW_CONSERVATIVE_QUOTIENT_ENVELOPE
```

The selected candidate must be the exact candidate bytes in the verified V2
report, must occur exactly once in its frozen family array, and must remain
bound to the candidate commitments, validation selection, and stress
confirmation. The transition does not copy `validation_evaluation` into the
child. Candidate reranking and fallback are forbidden.

Exactly seven nonselected routes create no child:

```text
INTERVAL_MULTI_Q_COMPETITION_NEEDS_EXACT_FRESH_EVIDENCE
  -> NOT_MATERIALIZED_NEEDS_EXACT_FRESH_EVIDENCE

INTERVAL_MULTI_Q_COMPETITION_INCOMPARABLE_EVALUATOR_EPOCH
  -> NOT_MATERIALIZED_INCOMPARABLE_EVALUATOR_EPOCH

INTERVAL_MULTI_Q_COMPETITION_EARLY_DIAGNOSTIC_UNRESOLVED
  -> NOT_MATERIALIZED_EARLY_DIAGNOSTIC_UNRESOLVED

INTERVAL_MULTI_Q_COMPETITION_NO_VALIDATION_WINNER
  -> NOT_MATERIALIZED_NO_VALIDATION_WINNER

INTERVAL_MULTI_Q_COMPETITION_PROVISIONAL_WINNER_FAILED_STRESS_CONFIRMATION
  -> NOT_MATERIALIZED_PROVISIONAL_WINNER_FAILED_STRESS_CONFIRMATION

INTERVAL_MULTI_Q_COMPETITION_BLOCKED_ADAPTER_NEEDS_POST_RESTRICTION_EVIDENCE
  -> NOT_MATERIALIZED_ADAPTER_NEEDS_POST_RESTRICTION_EVIDENCE

INTERVAL_MULTI_Q_COMPETITION_BLOCKED_ADAPTER_INCOMPARABLE_POST_RESTRICTION_EPOCH
  -> NOT_MATERIALIZED_ADAPTER_INCOMPARABLE_POST_RESTRICTION_EPOCH
```

For all seven routes, operation, transition, selected-candidate, child,
materialization-certificate, preservation-certificate, rollback-boundary, and
evaluator-gate surfaces are null. A malformed, tampered,
self-consistent-but-forged, or unsupported source is a hard error; it is not
converted into a nonmaterialization report.

## Detached child

A materialized child is a finite interval/multi-Q state whose object space,
model class, scope IDs, removable-feature IDs, fixed two-Q probe registry, and
violation-function registry are copied from the exact selected candidate.
Candidate evaluation and old evidence do not become child state.

The child ID is deterministic over the canonical child payload with only its
ID omitted. Lineage binds the verified source adapter seed, V2 competition
contract and report, candidate commitment, selected candidate, source theory
state, operation, and transition. The verified source seed is never mutated.

Every child freezes this qualification boundary:

```text
child_evaluator_epoch = null
child_evaluator_status = UNASSIGNED_FRESH_POST_TRANSITION_EVALUATOR_REQUIRED
operational_probe_status = FIXED_TWO_Q_FRESH_QUALIFICATION_REQUIRED
source_evidence_allowed_for_child_scoring = false
old_new_records_pooled = false
fresh_post_transition_evaluator_created = false
fresh_post_transition_qualification_performed = false
```

The fixed anchor and two-Q registry are declarations carried into the detached
child. Their equality is not external attestation and does not claim equality
of future probe values.

## Operation-specific materialization

### Interval expansion

`MATERIALIZED_SHADOW_INTERVAL_EXPANSION` uses operation `expand` and transition
`INTERVAL_EXPANSION`. The exact selected `finite_interval_table` geometry is
copied. Center table and radius grouping remain byte-equal to the verified
source seed; every child radius is at least its source radius and at least one
is strictly larger. Complete object-space and scope/radius-group coverage is
checked with finite stored floats.

The preservation certificate is an exact-center conservative interval
extension. It proves finite-geometry containment over the registered finite
domain, not domain safety, adoption, or future operational-probe behavior.

### Uniform interval restriction

`MATERIALIZED_SHADOW_UNIFORM_INTERVAL_RESTRICTION` uses operation `restrict`
and transition `UNIFORM_INTERVAL_RESTRICTION`. The exact selected interval
geometry is copied. Centers and grouping remain byte-equal to the source;
every child interval is contained in its source interval and at least one
radius is strictly smaller. The source model is recoverable only through the
frozen verified parent snapshot; restriction is not silently treated as an
identity operation or as qualification.

### Conservative quotient envelope

`MATERIALIZED_SHADOW_CONSERVATIVE_QUOTIENT_ENVELOPE` uses operation `quotient`
and transition `CONSERVATIVE_INTERVAL_QUOTIENT_ENVELOPE`. It requires an exact
feature projection, `per_context` child radius grouping, and one stored-float
hull for each quotient fiber. Every hull is recomputed across all parent
contexts in the fiber and all registered scopes. The reconstructed stored
child endpoints must contain every verified parent endpoint.

The certificate preserves conservative interval containment, not point
predictions. Exact parent recovery requires the verified parent snapshot. A
noncontained or nonrepresentable quotient is rejected rather than weakened.

## Record lifecycle and authority

All source evidence across the five prior generations and all V2 discovery,
validation, and stress rows remains audit evidence only and is excluded from
future child scoring. This is logical selective erasure: the report records
`logical_selective_erasure_applied = true`, while physical erasure is
`NOT_PERFORMED`. Cross-epoch pooling is forbidden.

Materialization yields status
`DETACHED_SHADOW_CHILD_MATERIALIZED_FRESH_QUALIFICATION_REQUIRED`. It always
retains:

```text
adoption_eligibility = NOT_DETERMINED_EXTERNAL_AUTHORITY_REQUIRED
adoption_status = NOT_ADOPTED_SHADOW_ONLY
promotion_status = NOT_PROMOTED
current_status = NOT_CURRENT
```

The core has no fresh-qualification, adoption-eligibility, adoption,
promotion, current-pointer, rollback-execution, probe-execution,
language-expansion, or parent/ambient-write authority. Materialization is not
`H_t -> H_{t+1}` acceptance.

## CLI

```bash
python3 runners/run_shadow_interval_multi_q_theory_transition.py \
  --competition-input /absolute/00-competition-input.json \
  --competition-contract /absolute/01-competition-contract.json \
  --competition-report /absolute/02-competition-report.json \
  --transition-contract /absolute/03-transition-contract.json \
  --transition-report /absolute/04-transition-report.json \
  --qualification-input /absolute/05-qualification-input.json \
  --qualification-contract /absolute/06-qualification-contract.json \
  --qualification-report /absolute/07-qualification-report.json \
  --review-contract /absolute/08-review-contract.json \
  --review-report /absolute/09-review-report.json \
  --probe-input /absolute/10-probe-input.json \
  --probe-contract /absolute/11-probe-contract.json \
  --probe-report /absolute/12-probe-report.json \
  --restriction-input /absolute/13-restriction-input.json \
  --restriction-contract /absolute/14-restriction-contract.json \
  --restriction-report /absolute/15-restriction-report.json \
  --adjudication-input /absolute/16-adjudication-input.json \
  --adjudication-contract /absolute/17-adjudication-contract.json \
  --adjudication-report /absolute/18-adjudication-report.json \
  --adapter-input /absolute/19-adapter-input.json \
  --adapter-contract /absolute/20-adapter-contract.json \
  --adapter-report /absolute/21-adapter-report.json \
  --interval-competition-input /absolute/22-interval-input.json \
  --interval-competition-contract /absolute/23-interval-contract.json \
  --interval-competition-report /absolute/24-interval-report.json \
  --interval-transition-contract /absolute/25-transition-contract.json \
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
  --expected-probe-report-digest sha256:... \
  --expected-restriction-input-digest sha256:... \
  --expected-restriction-contract-digest sha256:... \
  --expected-restriction-report-digest sha256:... \
  --expected-adjudication-input-digest sha256:... \
  --expected-adjudication-contract-digest sha256:... \
  --expected-adjudication-report-digest sha256:... \
  --expected-adapter-input-digest sha256:... \
  --expected-adapter-contract-digest sha256:... \
  --expected-adapter-report-digest sha256:... \
  --expected-interval-competition-input-digest sha256:... \
  --expected-interval-competition-contract-digest sha256:... \
  --expected-interval-competition-report-digest sha256:... \
  --expected-interval-transition-contract-digest sha256:... \
  --out /absolute/interval-transition-report.json
```

All twenty-six inputs must be absolute existing non-symlink JSON object files
with distinct normalized paths and distinct inodes. There are 325 unordered
input pairs, and both same-path and hard-link aliases are rejected for every
pair. Duplicate JSON keys, non-finite JSON constants, non-object roots, any
digest or artifact mismatch, a symlink, and output aliasing fail with exit 2
and no report on stdout. Success emits canonical JSON plus one newline;
optional `--out` receives exactly the same bytes via atomic replacement.

## Nonclaims and next gate

This slice is not a complete autonomous theory-evolution loop. It materializes
one exact selected finite interval/multi-Q proposal locally, or records one of
seven exact nonmaterialization routes. It performs no new evidence collection,
benchmark, scheduler or network access, `run_one`, probe execution, language
or predicate invention, physical record erasure, rollback, external
attestation, scientific generalization, or paper promotion.

The next sound gate is a separate fresh post-transition qualification slice.
That successor must bind this exact transition contract/report and detached
child, assign a new evaluator epoch under its own authority, collect fresh
fixed-two-Q records, exclude all five prior generations and V2 competition
rows from scoring, and verify exact cell coverage before any adoption-
eligibility decision. This transition cannot silently absorb that gate.

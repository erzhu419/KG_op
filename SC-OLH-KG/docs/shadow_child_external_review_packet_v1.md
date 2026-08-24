# Shadow Child External Review Packet V1

## Boundary

This strictly additive V1 turns one exact-replayed shadow-child qualification
into a local packet that is complete enough to hand to a separate external
review process.  Packet readiness is not adoption eligibility.  The slice has
no adoption authority and never adopts or promotes a child, makes it current,
changes a parent or child, acquires evidence, executes a probe or benchmark,
contacts a scheduler or network service, retains or deletes physical records,
or changes an Operations Research result or paper claim.

The slice consists only of:

- `performance/shadow_child_external_review_packet.py`;
- `performance/manifests/shadow_child_external_review_packet_v1.json`;
- `runners/run_shadow_child_external_review_packet.py`;
- `tests/test_shadow_child_external_review_packet.py`;
- this document.

The pure core writes nothing.  The runner's explicit `--out` is the only
optional write surface.

## Why this precedes another theory operation

The prior slices already compare robustification with quotient idealization,
materialize the selected child as an immutable shadow, and qualify supplied
fixed operational-probe records from new holdout and stress sets under a fresh
evaluator epoch.  They do not acquire or execute those probes.  The resulting
qualification still explicitly withholds adoption.

Before broadening the theory-operation grammar with restriction or observation
expansion, the current child needs a bounded lifecycle handoff: which snapshots
are required for rollback, which evidence commitments remain audit-only, which
records are consumed and unavailable for later scoring, and which external
attestations and authority are still absent.  V1 binds that handoff without
claiming that any external review has occurred.

## Public API

```python
validate_shadow_child_external_review_packet_contract(contract) -> dict

build_shadow_child_external_review_packet(
    competition_input,
    competition_contract,
    competition_report,
    transition_contract,
    transition_report,
    qualification_input,
    qualification_contract,
    qualification_report,
    review_contract,
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
    input_artifacts=None,
)

verify_shadow_child_external_review_packet(
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
) -> dict
```

The result exposes `disposition`, `report_digest`,
`ready_for_external_review`, and `to_dict()`.  It deliberately has no
`eligible`, `adopt`, `promote`, or `make_current` property.

The builder first invokes the public qualification verifier, which in turn
exact-replays the transition and competition.  Every upstream contract,
input, report, and artifact map remains independently anchored.  The review
verifier exact-replays the entire packet and byte-compares canonical JSON.

## Frozen contract and source

The contract is `shadow_child_external_review_packet_v1`.  It pins the source
qualification contract `shadow_child_probe_qualification_v1` at canonical
digest:

```text
sha256:593b8727f7f985cea82ae86e0758a67018c8d6a0a7c1dc0d518bcc3615f7d4ee
```

The contract fixes the qualification-to-packet mapping, local review checks,
record lifecycle roles, logical selective-erasure boundary, missing external
attestations, and authority boundary.  There is no free-form review policy or
new evidence input.

## Dispositions

The packet disposition is exactly one of:

```text
READY_FOR_EXTERNAL_REVIEW_PACKET_ONLY
REVIEW_PACKET_PENDING_NEW_EVIDENCE
REVIEW_PACKET_BLOCKED_INCOMPARABLE_EPOCH
REVIEW_PACKET_BLOCKED_PROBE_FAILURE
```

They map exactly from the four source qualification dispositions.  `READY`
means only that the local packet and its replayable commitments are complete.
It does not mean that the child is eligible for adoption.  In every case:

```text
adoption_eligibility = NOT_DETERMINED_EXTERNAL_AUTHORITY_REQUIRED
adoption_status = NOT_ADOPTED_SHADOW_ONLY
promotion_status = NOT_PROMOTED
current_status = NOT_CURRENT
```

An invalid contract, source chain, rollback binding, retention boundary, or
report fails closed without producing a packet.

## Exact report

The report schema is
`sc-olh-kg.shadow-child-external-review-packet-report/1`.  Its top-level keys
are exactly:

```text
schema_version
contract_id
contract_digest
packet_id
source_qualification
child_theory_state_digest
parent_theory_state_digest
transition_kind
evaluator_epoch
review_checks
record_lifecycle_boundary
rollback_boundary
selective_erasure_boundary
attestation_boundary
disposition
review_boundary
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

`packet_id` is content-addressed by the review contract digest, qualification
report digest, child-theory-state digest, and evaluator epoch.

`source_qualification` binds the public verification receipt, qualification
contract and report identities, qualification input digest and disposition,
source transition report digest, child digest, evaluator epoch, and upstream
non-adoption status.

## Local review checks

The packet records exact booleans for:

```text
source qualification exact replay
materialized source transition
child digest binding
fresh evaluator qualification
epoch and anchor comparability
complete and sufficient qualification evidence
presence and success of fixed operational probes
source-evidence scoring exclusion
old/new pooling prohibition
logical selective erasure
absence of physical deletion
parent rollback binding
original child immutability
upstream non-adoption
record-lifecycle boundary binding
local packet completeness
```

`local_packet_complete = true` for every valid packet: it means that the local
replay and boundary record were constructed completely, including when the
source qualification is pending, incomparable, or failed.  Only the qualified
source disposition makes `review_boundary.packet_ready = true` and produces
`READY_FOR_EXTERNAL_REVIEW_PACKET_ONLY`.  External attestations do not enter
local packet completeness or readiness because they are explicitly absent and
must be supplied to a future, separately authorized process.

No score is recomputed here.  Source competition evidence and consumed
qualification evidence never enter a new numerical comparison.

## Record lifecycle boundary

The boundary schema literal is exactly:

```text
sc-olh-kg.shadow-child-record-lifecycle-boundary/1
```

The parent snapshot has role `ROLLBACK_REQUIRED`; it is a required commitment
and is ineligible for scoring.  The child snapshot has role
`SHADOW_REVIEW_CANDIDATE`; it remains unadopted and non-current.

Source competition evidence has role:

```text
AUDIT_ONLY_SOURCE_SCORING_EXCLUDED
```

Consumed qualification holdout and stress evidence has role:

```text
CONSUMED_QUALIFICATION_EVIDENCE_AUDIT_ONLY
```

For both record classes, the packet binds evidence digests, observation-ID
digest and count, evaluator epoch, audit-retention requirement, and
`eligible_for_future_scoring = false`.  Future scoring requires new,
unconsumed evidence and may not pool evaluator epochs.

The boundary binds the competition, transition, qualification, and review
contract/report commitment chain.  It states that retention requirements are
bound, while physical retention attestation is
`REQUIRED_NOT_PRESENT`.  It does not claim that an external archive exists.

## Rollback boundary

The rollback boundary binds the parent and child digests, transition kind,
reduction-certificate digest, parent snapshot digest, and transition-specific
method:

```text
ROBUST_INTERVAL_EXPANSION
  -> COLLAPSE_INTERVAL_AT_RADIUS_MULTIPLIER_ZERO

QUOTIENT_IDEALIZATION
  -> RESTORE_FROZEN_PARENT_POINT_TABLE
```

`rollback_binding_verified = true` states only that the exact-replayed
transition carries the required reduction or frozen-parent recovery binding.
`rollback_execution_status = NOT_PERFORMED`; no state is rolled back.

## Selective erasure and attestations

The selective-erasure mode is
`LOGICAL_ACTIVE_SCORING_VIEW_ONLY`.  Source records are excluded from active
child scoring.  Consumed qualification records are excluded from future
rescoring.  New evidence is required for any future scoring and cross-epoch
pooling remains forbidden.

Logical exclusion retains commitments for audit; it does not delete files or
records.  The packet therefore fixes:

```text
physical_erasure = NOT_PERFORMED
```

The attestation boundary is also explicit:

```text
external_data_attestation = REQUIRED_NOT_PRESENT
external_evaluator_attestation = REQUIRED_NOT_PRESENT
physical_retention_attestation = REQUIRED_NOT_PRESENT
physical_erasure = NOT_PERFORMED
external_adoption_authority = REQUIRED_NOT_PRESENT
```

Changing any of these statuses in a supplied report breaks exact replay.  A
future authority may create a new additive artifact; this V1 cannot accept an
attestation or decision as input.

## Review and authority boundary

The scope is `LOCAL_EXTERNAL_REVIEW_PACKET_ONLY`.  The packet records whether
its local contents are ready for external review, but always requires external
review.  It fixes all of the following to false:

```text
adoption_decision_allowed
promotion_decision_allowed
current_pointer_write_allowed
parent_or_child_state_write_allowed
```

No packet field, digest, audit chain, fixed-anchor equality, or derived
evaluator epoch is an external signature or authority.

## CLI

```bash
python3 runners/run_shadow_child_external_review_packet.py \
  --competition-input /absolute/competition_input.json \
  --competition-contract /absolute/theory_operation_competition_v1.json \
  --competition-report /absolute/competition_report.json \
  --transition-contract /absolute/shadow_theory_transition_v1.json \
  --transition-report /absolute/transition_report.json \
  --qualification-input /absolute/qualification_input.json \
  --qualification-contract /absolute/shadow_child_probe_qualification_v1.json \
  --qualification-report /absolute/qualification_report.json \
  --review-contract /absolute/shadow_child_external_review_packet_v1.json \
  --expected-competition-contract-digest sha256:... \
  --expected-competition-report-digest sha256:... \
  --expected-transition-contract-digest sha256:... \
  --expected-transition-report-digest sha256:... \
  --expected-qualification-input-digest sha256:... \
  --expected-qualification-contract-digest sha256:... \
  --expected-qualification-report-digest sha256:... \
  --expected-review-contract-digest sha256:... \
  --out /absolute/review_packet.json
```

All nine inputs must be absolute, existing, non-symlink regular files with
distinct resolved paths and inodes.  The runner reconstructs the exact prior
two-, four-, and seven-artifact maps and binds all nine files into the packet's
artifact map.  It rejects duplicate JSON keys, non-finite constants,
non-object roots, relative paths, symlinks, any input path or hard-link alias,
and any output alias.  Success emits one compact key-sorted canonical JSON
object plus a newline; optional `--out` is an identical atomic copy.  Failure
returns 2 and emits no packet on stdout.

## Claim boundary

V1 may claim only that the full local chain was exact-replayed, packet
identities and local rollback/lifecycle commitments are bound, source and
consumed qualification records are logically excluded from future active
scoring, and the packet is ready or not ready for a separate external review
step.

It does not establish adoption eligibility, an adoption or promotion decision,
external data provenance, external evaluator authority, physical retention or
erasure, rollback execution, a review outcome, scientific validity,
generalization, or an `H_t -> H_{t+1}` acceptance decision.

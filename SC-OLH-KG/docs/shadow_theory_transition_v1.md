# Shadow Theory Transition V1

## Boundary

This additive V1 materializes an exact-replayed shadow competition result as a
deterministic child theory state.  It does not adopt the child, mutate the
parent, reuse source evidence to score the child, assign a child evaluator
epoch, execute a probe, run a benchmark, or promote a scientific or paper
claim.  The pure core writes nothing; the runner's explicit `--out` is its only
optional write surface.

The slice consists only of:

- `performance/shadow_theory_transition.py`;
- `performance/manifests/shadow_theory_transition_v1.json`;
- `runners/run_shadow_theory_transition.py`;
- `tests/test_shadow_theory_transition.py`;
- this document.

It depends only on the additive theory-operation competition V1.  It never
imports the historical benchmark or structural campaign.

## Public API

```python
validate_shadow_transition_contract(contract) -> dict

materialize_shadow_theory_transition(
    competition_input,
    competition_contract,
    competition_report,
    transition_contract,
    *,
    expected_competition_contract_digest,
    expected_competition_report_digest,
    expected_competition_input_artifacts,
    expected_transition_contract_digest,
    input_artifacts=None,
) -> ShadowTransitionResult

verify_shadow_theory_transition(
    competition_input,
    competition_contract,
    competition_report,
    transition_contract,
    transition_report,
    *,
    expected_competition_contract_digest,
    expected_competition_report_digest,
    expected_competition_input_artifacts,
    expected_transition_contract_digest,
    expected_transition_report_digest,
    expected_transition_input_artifacts,
) -> dict
```

The core exports `ShadowTransitionValidationError`, `ShadowTransitionKind`,
`ShadowTransitionDisposition`, `ShadowTransitionResult`, and
`canonical_json_bytes`.  It first calls the public competition verifier with
the independently retained upstream contract/report/artifact anchors.  A
self-consistent but forged source report is therefore not an admissible input.
The transition verifier independently binds both contract digests, both report
digests, and both artifact maps before exact replay.

## Dispositions

The only transition dispositions are:

```text
MATERIALIZED_SHADOW_ROBUSTIFICATION
MATERIALIZED_SHADOW_IDEALIZATION
NOT_MATERIALIZED_NEEDS_EVIDENCE
NOT_MATERIALIZED_INCOMPARABLE_EVALUATOR_EPOCH
```

The first two require a verified `SELECT_ROBUSTIFICATION` or
`SELECT_IDEALIZATION` source.  A verified `NEEDS_EVIDENCE` or
`INCOMPARABLE_EVALUATOR_EPOCH` source produces a replayable non-materialization
report whose operation, transition, candidate, child, certificates, and
evaluator gate are all `null`.  Invalid or tampered source material raises an
error and produces no report.  `adoption_status` is always
`NOT_ADOPTED_SHADOW_ONLY`.

## Exact report

The report has exactly:

```text
schema_version
contract_id
contract_digest
source_competition
parent_theory_state
parent_theory_state_digest
disposition
operation_kind
transition_kind
selected_candidate_id
child_theory_state
child_theory_state_digest
preservation_certificate
reduction_certificate
evaluator_gate
adoption_status
nonclaims
input_artifacts
audit_events
audit_head
report_digest
```

`source_competition` binds the verifier status, contract identity/digest,
report schema/digest, case and source disposition, promotion status, candidate
commitment and selected candidate IDs, theory-state digest, evidence digests,
and independently expected input artifacts.  The report digest commits the
entire canonical body.  A digest is not a signature or external attestation.

## Child theory state

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
probe_ids
violation_functionals
scope_ids
removable_feature_ids
evidence_reuse_policy
operational_probe_status
transition_lineage
```

The ID is `shadow-theory:` followed by the SHA-256 of the canonical child
payload with only `theory_id` omitted.  Lineage binds the parent theory and
source competition contract/report/candidate commitment, candidate,
operation, and transition.

The fixed anchor is copied exactly, but it is not external attestation.  The
new child deliberately has:

```text
evaluator_epoch = null
evaluator_status = UNASSIGNED_NEW_EVALUATOR_REQUIRED
source_evidence_role = QUALIFICATION_ONLY
source_evidence_allowed_for_child_scoring = false
old_new_records_pooled = false
operational_probe_status = OPERATIONAL_PROBE_AND_FRESH_EPOCH_REQUIRED
```

Copied `probe_ids` are declarations only.  The transition does not certify
that an operational probe was preserved or executed.  A fresh evaluator epoch
and operational probe are required before the child can be scored or adopted.

## Robust interval expansion

`MATERIALIZED_SHADOW_ROBUSTIFICATION` yields operation `expand`, transition
`ROBUST_INTERVAL_EXPANSION`, and a `finite_interval_table`.  Its center table
is the exact parent point table; its grouping and radii are the exact verified
selected candidate.  The full finite parent object space and every required
context/scope radius lookup are checked.  Every radius must be finite and
non-negative.

The preservation certificate is
`EXACT_CENTER_CONSERVATIVE_EXTENSION`.  It certifies center equality and
containment of the parent point prediction, not domain safety and not
operational probe preservation.  The reduction certificate is
`COLLAPSE_INTERVAL_AT_RADIUS_MULTIPLIER_ZERO`: multiplying every radius by
zero and collapsing the interval representation must reproduce the exact
parent point-model digest.

## Quotient idealization

`MATERIALIZED_SHADOW_IDEALIZATION` yields operation `quotient`, transition
`QUOTIENT_IDEALIZATION`, and a child `finite_point_table` on retained features
and deduplicated quotient contexts.  Every quotient prediction is recomputed
as the mean of the complete parent fiber; candidate-authored values are not
trusted.

The preservation certificate is intentionally named
`FINITE_PREDICTION_PROXY_BOUND`, not probe preservation.  It recomputes the
complete finite parent-space divergence, source-contract bound, state-count
reduction, and within-bound result.  Its operational-probe certification is
always false.

The reduction certificate is
`QUOTIENT_PROJECTION_WITH_FROZEN_PARENT_SNAPSHOT`.  The fiber map must cover
every parent context once.  Exact recovery is verified from the frozen parent
snapshot and prediction snapshot.  A lossy quotient alone does not recover
the deleted degrees of freedom.

## CLI

```bash
python3 runners/run_shadow_theory_transition.py \
  --competition-input /absolute/input.json \
  --competition-contract /absolute/theory_operation_competition_v1.json \
  --competition-report /absolute/competition_report.json \
  --transition-contract /absolute/shadow_theory_transition_v1.json \
  --expected-competition-contract-digest sha256:... \
  --expected-competition-report-digest sha256:... \
  --expected-transition-contract-digest sha256:... \
  --out /absolute/transition_report.json
```

The runner reconstructs the exact upstream `contract_json` and
`evidence_json` artifact map and binds all four input files into the transition
artifact map.  All inputs must be absolute existing non-symlink JSON files.
Duplicate keys, non-finite constants, aliasing or hard-linking an input as the
output, and every digest mismatch fail with exit 2 and no report on stdout.
Success always writes canonical stdout plus one newline; optional `--out` is
the identical atomic byte sequence.

## Nonclaims and next gate

This slice materializes only finite scalar interval expansion and finite
quotient idealization.  It implements no model restriction, probe expansion,
language/predicate invention, selective erasure, evaluator creation,
acceptance, adoption, or general scientific validation.  Source evidence is
qualification evidence only.  No old/new scores are pooled, and no
`H_t -> H_{t+1}` acceptance is claimed.

The next sound boundary is a separate operational-probe and fresh-evaluator
epoch qualification slice.  It must evaluate the materialized child with new
records before any adoption decision; this transition must not silently absorb
that authority.

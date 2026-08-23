# Structural-hypothesis recursive successor materializer V1

This stage converts one fully verified immutable recursive report advance into
one deterministic, local successor bundle.  For the frozen V1 instance the
transition is:

```text
verified advance: 1,352 typed rows; 28 pending cells; NOT_CURRENT
  -> exact ordered materialization of all 28 pending cells
  -> first task: FactorShockStatePolicyRZDT1 / full / seed 2
  -> 28 native task dictionaries, each with exactly 438 run_one arguments
```

The committed status is
`RECURSIVE_SUCCESSOR_MATERIALIZED_FROM_VERIFIED_ADVANCE_NOT_AUTHORIZED`.
Successful verification reports the same status with the `VERIFIED_` prefix.
The four state fields remain fixed:

```text
current_status=NOT_CURRENT
authorization_status=NOT_AUTHORIZED
attempt_status=NOT_PREPARED
execution_status=NOT_EXECUTED
```

Materialization is not authorization, preparation, execution, currentness,
runtime readiness, scientific evidence, external authority, or paper
promotion.  It does not change the historical meaning of the source advance.

## Entry points and exact API

- Contract:
  `performance/manifests/structural_hypothesis_recursive_successor_materializer_v1.json`
- Core:
  `performance/structural_hypothesis_recursive_successor_materializer.py`
- CLI:
  `runners/run_structural_hypothesis_recursive_successor_materializer.py`

The contract schema is
`sc-olh-kg.structural-hypothesis-recursive-successor-materializer/1`, the
capsule schema is
`sc-olh-kg.structural-hypothesis-recursive-successor-materialization/1`, and
the contract ID is
`structural_hypothesis_recursive_successor_materializer_v1`.

The core exposes:

```python
materialize_recursive_successor(...)
verify_recursive_successor(...)
```

Both functions take these 21 positional arguments in this exact order:

```text
publication_root
adoption_contract_path
adoption_root
source_successor_contract_path
source_successor_root
base_evidence_csv
source_attempt_root
hypothesis_contract_path
executor_contract_path
runtime_contract_path
publisher_contract_path
materializer_contract_path
bridge_contract_path
base_manifest_path
asset_root
completed_attempt_root
advance_contract_path
advance_root
recursive_successor_contract_path
recursive_successor_root
future_attempt_root
```

The common keyword-only surface is:

```text
adoption_id
source_successor_id
advance_id
recursive_successor_id
expected_adoption_digest
expected_source_pending_evidence_digest
expected_source_first_pending_projection_digest
expected_source_successor_digest
expected_source_bundle_digest
expected_source_plan_digest
completed_task_id
expected_completed_task_digest
expected_source_provenance_binding_digest
expected_source_authorization_digest
expected_source_attempt_digest
expected_source_execution_receipt_digest
expected_source_execution_journal_head_digest
expected_advance_digest
expected_advance_reingestion_digest
expected_advance_output_report_body_digest
expected_advance_output_audit_head
expected_advance_output_evidence_digest
expected_next_pending_evidence_digest
expected_next_first_pending_projection_digest
```

Materialization additionally requires
`confirm_recursive_successor_materialization=True`. Verification has no
confirmation argument and additionally requires:

```text
expected_recursive_successor_digest
expected_next_bundle_digest
expected_next_plan_digest
```

The exact return object has 18 fields:

```text
status
recursive_successor_root
recursive_successor_digest
advance_digest
advance_output_evidence_digest
pending_evidence_digest
first_pending_projection_digest
bundle_digest
plan_digest
first_task_id
first_task_digest
task_count
future_attempt_root
checkpoint_root
current_status
authorization_status
attempt_status
execution_status
```

The CLI prints that compact, key-sorted JSON object to stdout only after a
successful `materialize`. Human-readable status is written to stderr.
Successful `verify` writes nothing to stdout and does not mutate any source or
output tree.

## Full admission and anchor boundary

V1 does not trust a detached `bundle.json` or hashes read only from the new
output tree. It captures the exact advance marker and artifact map through
held directory descriptors, verifies the complete original adoption,
successor, completed-attempt, and advance chain, replays the captured
`combined_rows.json`, and strongly verifies the task materializer output.

The caller must retain 19 independent input anchors outside the new recursive
successor tree:

1. Twelve source-chain anchors: adoption; source pending list; source first
   pending projection; source successor; source bundle; source plan; completed
   task; provenance binding; authorization; attempt; execution receipt; and
   execution journal head.
2. Five advance-output anchors: advance commit; recursive reingestion; output
   report body; output audit head; and output evidence.
3. Two next-round observations: the exact ordered 28-cell pending-list digest
   and the exact first-pending seven-field projection digest.

Read-only verification additionally requires independently retained recursive
successor, next-bundle, and next-plan digests. Every anchor is mandatory. A
matching basename, plausible local hash, or detached artifact is insufficient.
Values derived transitively from an already anchored marker are not described
as independent observations.

The first-pending projection contains exactly `profile`, `domain`, `line`,
`seed`, `d`, `N`, and `n0`. Pending order is replay order, not a new ranking or
scheduler decision. V1 accepts the frozen 1,352-row / 28-pending transition and
materializes exactly 28 tasks.

## CLI

All paths must be absolute. The recursive successor and future attempt paths
must be direct children of their frozen XDG state prefixes and must share the
same basename. Both are required absent at admission; the future attempt is
observed absent again immediately before the prepared commit marker is linked
into place. That point-in-time observation is not a reservation or a lock.

```bash
python3 runners/run_structural_hypothesis_recursive_successor_materializer.py materialize \
  --publication-root /absolute/state/structural-hypothesis-reingestion/v1/source \
  --adoption-root /absolute/state/structural-hypothesis-report-adoption/v1/source \
  --adoption-id source \
  --source-successor-root /absolute/state/structural-hypothesis-adopted-successor/v1/seed1 \
  --source-successor-id seed1 \
  --base-evidence-csv /absolute/input/base.csv \
  --source-attempt-root /absolute/state/structural-hypothesis-execution/v1/seed0 \
  --completed-attempt-root /absolute/state/structural-hypothesis-execution/v1/seed1 \
  --advance-root /absolute/state/structural-hypothesis-recursive-report-advance/v1/seed1 \
  --advance-id seed1 \
  --recursive-successor-root /absolute/state/structural-hypothesis-recursive-successor/v1/seed2 \
  --recursive-successor-id seed2 \
  --future-attempt-root /absolute/state/structural-hypothesis-execution/v1/seed2 \
  --completed-task-id task:000000000000000000000000 \
  --expected-adoption-digest sha256:... \
  --expected-source-pending-evidence-digest sha256:... \
  --expected-source-first-pending-projection-digest sha256:... \
  --expected-source-successor-digest sha256:... \
  --expected-source-bundle-digest sha256:... \
  --expected-source-plan-digest sha256:... \
  --expected-completed-task-digest sha256:... \
  --expected-source-provenance-binding-digest sha256:... \
  --expected-source-authorization-digest sha256:... \
  --expected-source-attempt-digest sha256:... \
  --expected-source-execution-receipt-digest sha256:... \
  --expected-source-execution-journal-head-digest sha256:... \
  --expected-advance-digest sha256:... \
  --expected-advance-reingestion-digest sha256:... \
  --expected-advance-output-report-body-digest sha256:... \
  --expected-advance-output-audit-head sha256:... \
  --expected-advance-output-evidence-digest sha256:... \
  --expected-next-pending-evidence-digest sha256:... \
  --expected-next-first-pending-projection-digest sha256:... \
  --confirm-recursive-successor-materialization
```

The checked-in contracts, base manifest, and task assets are defaults. They
may also be passed explicitly with `--adoption-contract`,
`--source-successor-contract`, `--hypothesis-contract`,
`--executor-contract`, `--runtime-contract`, `--publisher-contract`,
`--materializer-contract`, `--bridge-contract`, `--base-manifest`,
`--asset-root`, `--advance-contract`, and
`--recursive-successor-contract`.

The `verify` action takes the same common arguments, omits the confirmation,
and adds:

```text
--expected-recursive-successor-digest sha256:...
--expected-next-bundle-digest sha256:...
--expected-next-plan-digest sha256:...
```

Any path, layout, mode, ownership, alias, generation, replay, task, state, or
digest mismatch returns two and leaves stdout empty.

## Capsule, path binding, and commit protocol

The root is a fresh direct child of
`$XDG_STATE_HOME/kg-op/structural-hypothesis-recursive-successor/v1` and has
this exact layout:

```text
<recursive-successor-id>/
  recursive_successor_contract.json
  bundle.json
  successor.json
```

The directory mode is `0700`; every leaf is `0600`. Symlinks, FIFOs, hard
links, unexpected names, unsafe state-parent modes, generation changes, and
attempted overwrite are rejected. Writes are no-clobber and fsynced. The
marker temp file is fully written and fsynced before the final point-in-time
future-attempt observation; only then is it hard-linked as `successor.json`.
That file is the sole commit marker and is published last. A failure before
that link leaves `INCOMPLETE_NOT_MATERIALIZED`; the incomplete root is never
reused.

The future attempt and `future_attempt_root/checkpoints` are string bindings
only. Neither path is created. Later existence is allowed during read-only
successor verification, but is not evidence of authorization, preparation,
readiness, or execution. No `current.json`, authorization, attempt, receipt,
journal, checkpoint, retry, or result artifact is written.

## Audited local seed-2 candidate

The source chain currently retained on the audited machine culminates in:

```text
advance root:
/home/erzhu419/.local/state/kg-op/structural-hypothesis-recursive-report-advance/v1/ca54f50-factor-seed1-v1

candidate recursive successor root (must remain absent before materialize):
/home/erzhu419/.local/state/kg-op/structural-hypothesis-recursive-successor/v1/ca54f50-factor-seed2-v1

candidate future attempt root (must remain absent before materialize):
/home/erzhu419/.local/state/kg-op/structural-hypothesis-execution/v1/ca54f50-factor-seed2-v1
```

The independently retained source-chain anchors are:

```text
adoption
sha256:b3a858c481ccc9ffd6bca6033e0ff2bbd410500bcfe6c4690afce85db40a45e5
source pending list
sha256:7006a17aaac206503977cb090a52dea3838aa06b378b4e0278e7cbebfe46041e
source first projection
sha256:e26ba673070e7a0f59edb47c2c7dba5eeaba19546ad51c741e37cdc2b15cf642
source successor
sha256:410ff24f59d43dae48f9e2fab01c7ff59e3e877d2c823470fad5a270b6d3badb
source bundle
sha256:ca0638a98e6d6712e3467a57d784a73a68144328b11500c44e43a296245a5b93
source plan
sha256:b6e0094d19067f936e5f0c1cb215584f08d423b21643258d3d7becdea23ad5d4
completed task ID
task:95ff940d4b1317f8564c161b
completed task
sha256:3063f34add53674166b489cf5892901d6d4cecd8c0621eaf5b76e3e1c930bf0c
source provenance binding
sha256:02b983740161fc5aa2aeb40e0db52677bd99e59178c6a05731cbfdd77432eab2
source authorization
sha256:9baa5f1d2fe967ea7b0a374883ab04018175973f9289d5658cea2d576ec2f284
source attempt
sha256:008800fbe323bea079a225420c3b05a96927dbcb86786f044bf7f1683c82f367
source execution receipt
sha256:f835b84aeb6abbb2c147c130c8f73e496283b5b82243e65325d24b4affc395df
source execution journal head
sha256:71f38e024333dfb956a174df0262891ad607d3995473b079e1e9f0c52a090d9f
```

The independently retained advance-output anchors are:

```text
advance
sha256:1f18733459d26a9fa10e1b54af23ba012cca3cd8b25fe2598e1857e2fc3ae5c3
advance reingestion
sha256:2d1f3205cc8252258666ce4fb2ce0fb01c5857b3e08c3038c4aa27b9eb30e589
advance output report body
sha256:929b30e334fca5fa12bd8d020b0e7dbeea24e1308095b08fa496da32e9ecf50d
advance output audit head
sha256:f24572bfa9da5b2f409480b382620406dd269ea65a2eedd78ddd737c77d1a1d6
advance output evidence
sha256:05dd631c00bca4c7438fd0b4acb1bb48e230a69d0c659465ca5e2a12923ce9fd
```

The exact next first-pending projection is:

```json
{"N":20,"d":50,"domain":"FactorShockStatePolicyRZDT1","line":"lodo","n0":10,"profile":"full","seed":2}
```

Its two independently retained next-round anchors are:

```text
next pending list
sha256:f5eaad75af092685920c5d7d55da1dfb0eba1f3b59aa4467cf1172b374789d1c
next first projection
sha256:b9c57f0c7b619e984239878e0c5ced375f53c718c063d02cb9f9b08ae4c715d5
```

Deterministic inspection also freezes the candidate output bundle, plan, and
first task values below. They are not evidence that the absent successor root
has been committed; after an explicit materialization, independently retain
the emitted successor digest as well and use all three output anchors for
verification.

```text
candidate next bundle
sha256:fe327bd645e4a9fb76ea21a8345cefb22cb3c9d022db29c18aff4106ab0ad6d5
candidate next plan
sha256:a6e805c6513ed5719439237502383d4fdedac751dbc890d756c9de17a905089e
candidate first task ID
task:bd5eea3291d18801f8b28785
candidate first task
sha256:87c142a87659b95cf763d4e16f33a47fdd0f3240db35a797c76c060454581daa
```

These local SHA-256 commitments are not signatures and provide no defense
against a same-user rewrite that also replaces the independently retained
values.

## Test boundary

The focused KAT imports the existing recursive-advance fake fixture and builds
the complete seed-0, seed-1, 1,352-row chain under a temporary XDG state home.
It then materializes the exact 28-task successor, proves the first task is
seed 2, and checks the 438-argument shape of every native task object while
the production path invokes the frozen strong full-task verifier. It covers
every input and output anchor, read-only verification, exact layout and modes,
same-generation rejection, tampering, FIFO and hard-link rejection,
no-clobber, and commit-marker-last failure.

The fixture uses only the in-memory fake callable and fake preflight inherited
from the recursive-advance KAT. It creates fake authorization, attempt,
receipt, and journal artifacts only beneath a temporary XDG state home. The
production successor surface itself creates none of them. Neither the surface
nor its tests import or invoke the native benchmark or `run_one`, write real
XDG state, or access a network or scheduler.

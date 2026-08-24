# Structural-hypothesis recursive campaign V1

This module turns the previously hand-wired seed boundary into one reusable,
local, provenance-bound campaign step.  It accepts either the immutable
recursive-successor V1 capsule that bootstraps the campaign or the immutable
nonterminal output of an earlier campaign V1 capsule.  The same five public
operations are then reused for every later step; there is no seed-specific
Python module.

One campaign capsule can start at most one callback and can publish at most one
new evidence row.  It never loops, retries, invokes a scheduler, accesses the
network, changes a current pointer, or edits an Operations Research artifact.

## Identity and entry points

- Contract:
  `performance/manifests/structural_hypothesis_recursive_campaign_v1.json`
- Core: `performance/structural_hypothesis_recursive_campaign.py`
- CLI: `runners/run_structural_hypothesis_recursive_campaign.py`
- Contract ID: `structural_hypothesis_recursive_campaign_v1`
- Contract schema:
  `sc-olh-kg.structural-hypothesis-recursive-campaign/1`
- Capsule schema:
  `sc-olh-kg.structural-hypothesis-recursive-campaign-capsule/1`
- State prefix: `kg-op/structural-hypothesis-recursive-campaign/v1`

The core exposes exactly these operations:

```python
inspect_recursive_campaign(
    source, campaign_contract_path, campaign_root, *,
    campaign_id, next_attempt_root=None,
)

authorize_recursive_campaign_task(
    source, campaign_contract_path, campaign_root, *,
    campaign_id,
    expected_source_state_digest,
    expected_bundle_digest,
    expected_plan_digest,
    task_id,
    expected_task_digest,
    expected_provenance_binding_digest,
    authorization_id,
    confirm_explicit_local_task_authorization,
)

execute_recursive_campaign_task(
    source, campaign_contract_path, campaign_root, runtime_contract_path, *,
    expected_campaign_digest,
    expected_lease_digest,
    expected_provenance_binding_digest,
    expected_authorization_digest,
    expected_attempt_digest,
    confirm_real_local_execution,
)

advance_recursive_campaign(
    source, campaign_contract_path, campaign_root, next_attempt_root, *,
    expected_campaign_digest,
    expected_lease_digest,
    expected_provenance_binding_digest,
    expected_authorization_digest,
    expected_attempt_digest,
    expected_receipt_digest,
    expected_journal_head_digest,
    expected_output_evidence_digest,
    expected_output_report_body_digest,
    expected_output_audit_head,
    expected_reingestion_digest,
    expected_next_pending_evidence_digest,
    expected_next_first_pending_projection_digest,
    expected_next_bundle_digest,
    expected_next_plan_digest,
    confirm_immutable_one_step_advance,
)

verify_recursive_campaign(
    source, campaign_contract_path, campaign_root, *,
    next_attempt_root=None,
    expected_campaign_digest,
    expected_lease_digest,
    expected_callback_start_claim_digest=None,
    expected_receipt_digest=None,
    expected_journal_head_digest=None,
    expected_advance_digest=None,
    expected_output_evidence_digest=None,
    expected_output_report_body_digest=None,
    expected_output_audit_head=None,
    expected_reingestion_digest=None,
    expected_next_bundle_digest=None,
    expected_next_plan_digest=None,
)
```

All confirmation values must be the Boolean `True`; truthy substitutes are
rejected.  Inspection and verification require no confirmation and are
read-only.

## Source descriptor union

The CLI takes a required absolute `--source-descriptor` path.  It rejects
duplicate JSON keys before importing the core and passes the resulting object
as `source`.  Every descriptor has exactly:

```text
schema_version = sc-olh-kg.structural-hypothesis-recursive-campaign-source/1
source_kind
dependencies
```

`dependencies` contains exactly six normalized absolute paths:

```text
hypothesis_contract_path
executor_contract_path
runtime_contract_path
materializer_contract_path
base_manifest_path
asset_root
```

For `source_kind=recursive_successor_v1`, the remaining fields are
`verify_args` and `verify_kwargs`.  They are the complete 21 positional paths
and complete keyword surface of `verify_recursive_successor`; abbreviated or
transitively inferred anchors are forbidden.  Dependency paths are
cross-checked against the corresponding verifier arguments.

For `source_kind=recursive_campaign_v1`, the remaining fields are:

```text
campaign_contract_path
campaign_root
expected_campaign_digest
expected_lease_digest
expected_callback_start_claim_digest
expected_advance_digest
expected_output_evidence_digest
expected_output_report_body_digest
expected_output_audit_head
expected_reingestion_digest
expected_next_bundle_digest
expected_next_plan_digest
```

The previous campaign is recursively verified from its own captured
`source/descriptor.json`.  Cycles, depth greater than 30, incomplete prior
campaigns, and terminal prior campaigns are rejected.  Consequently, the
second and later iterations use this same module without trusting a detached
bundle or inventing another seed adapter.

Recursive verification is linear in the admitted predecessor depth for each
public call.  A call-local semantic memo is keyed by the resolved campaign
root plus the exact source descriptor, contract path, campaign identity,
next-attempt binding, and all expected anchors, so each exact predecessor is
fully verified once.  A call-local freshness ledger then shallow-recaptures
the bootstrap and every admitted campaign root once before return.  The memo
is neither persisted nor shared across calls: it cannot turn prior local
verification into continuing authority, and drift in even the oldest
predecessor still fails the current call.

## One-step state machine

The normal nonterminal transition is:

```text
verified source, for example 1,352 typed rows / 28 pending cells
  -> inspect: zero writes, exact first task and provenance derived
  -> explicit authorize: AUTHORIZED runtime attempt + one lease
  -> durable callback_start_claim.json
  -> exactly one runtime delegate invocation
  -> completed receipt, then hard stop
  -> immutable reingestion of one accepted row
  -> 1,353 typed rows / 27 pending cells + next 27-task bundle
  -> advance/advance.json published last, then hard stop
  -> next campaign inspects this campaign as recursive_campaign_v1
```

The public statuses are:

```text
INSPECTED_RECURSIVE_CAMPAIGN_SOURCE_NONTERMINAL_NOT_AUTHORIZED
RECURSIVE_CAMPAIGN_AUTHORIZED_ONE_CALLBACK_START_LEASED
RECURSIVE_CAMPAIGN_CALLBACK_START_CLAIMED_RUNTIME_COMPLETED_HARD_STOP
RECURSIVE_CAMPAIGN_CALLBACK_START_CLAIMED_RUNTIME_INCOMPLETE_HARD_STOP
ADVANCED_RECURSIVE_CAMPAIGN_ONE_STEP_NONTERMINAL_HARD_STOP
ADVANCED_RECURSIVE_CAMPAIGN_ONE_STEP_TERMINAL_HARD_STOP
```

Successful read-only verification prefixes the applicable committed-state
status with `VERIFIED_`.  Verification always returns one uniform 24-field
surface with `phase` equal to one of:

```text
AUTHORIZED
CALLBACK_INCOMPLETE
CALLBACK_COMPLETED
ADVANCED_NONTERMINAL
ADVANCED_TERMINAL
```

Anchors that cannot yet exist in the reported phase are JSON `null`.  This
makes an authorized capsule and, critically, a claimed-but-incomplete capsule
verifiable without misrepresenting either as an advanced capsule.
For `AUTHORIZED`, `CALLBACK_INCOMPLETE`, and an evidence-neutral failed
completion, `remaining_task_count` retains the verified source-report pending
count and `terminal_status` remains `NONTERMINAL`, while the advance-boundary
`next_attempt_root` is null.  The already prepared current attempt remains
bound by campaign provenance and is not mislabeled as a next-round preview.
The `advance_digest` field is null before advance and mandatory afterward; it
is the retained anchor used by verification and the next campaign descriptor.

The phase-aware `execution_status` values are exact, not free-form labels:

```text
NOT_EXECUTED
CALLBACK_CLAIMED_INCOMPLETE_NO_REENTRY
COMPLETED_SUCCESS_PREVIEW_INDEPENDENTLY_ANCHORED_HARD_STOP
COMPLETED_SUCCESS_PREVIEW_RECOVERED_LOCAL_ANCHORS_NOT_INDEPENDENT_HARD_STOP
COMPLETED_FAILED_EVIDENCE_NEUTRAL_INDEPENDENTLY_ANCHORED_HARD_STOP
COMPLETED_FAILED_EVIDENCE_NEUTRAL_RECOVERED_LOCAL_ANCHORS_NOT_INDEPENDENT_HARD_STOP
COMPLETED_AND_ADVANCED_HARD_STOP
```

The mutating `execute` result stops one phase earlier and reports either
`COMPLETED_SUCCESS_AWAITING_ADVANCE` or
`COMPLETED_FAILED_EVIDENCE_NEUTRAL_HARD_STOP`.

Inspection does not create the campaign or attempt root.  Authorization
requires independent source, bundle, plan, task, and provenance anchors and
the exact deterministic authorization ID reported by inspection.  It prepares
only task ordinal zero and publishes the campaign commit last.

Authorization is restart-safe across its two local capsules.  If the exact
deterministic runtime attempt is already AUTHORIZED after a crash, a repeated
authorization verifies and reuses it without calling prepare again.  A partial
campaign root may contain only a subset of the final fixed names; every
existing leaf must byte-match the newly derived value, missing leaves are
published exclusively, and `campaign.json` remains last.  A fully committed
repeat is idempotent and does not overwrite anything.

Execution requires the independently retained campaign, lease, provenance,
authorization, and attempt digests.  Before delegating to the runtime, it
guards the exact held campaign-root and source generation while it creates
`callback_start_claim.json` with exclusive creation and fsync.  Drift before
the atomic publication leaves the claim absent; drift detected by the
post-publication read leaves the claim spent but never enters the runtime
delegate.  The
lease permits one callback start, not an exactly-once completion claim.  Once
the claim exists, every later execute request fails before the delegate.

If the process crashes, the callback raises, or the runtime is left RUNNING
after the claim, the campaign is `...INCOMPLETE_HARD_STOP`.  It is an uncertain
local execution state: there is no automatic retry, no second lease, no
advance, and no scientific refutation claim.  A callback that returns a
failure receipt is a completed, evidence-neutral hard stop rather than an
incomplete callback.  It retains the receipt and journal anchors for audit but
returns null for every reingestion, output, and next-round anchor; it cannot be
advanced or retried.
Claimed runtime trees that remain AUTHORIZED, have only a durable PREFLIGHT,
or have reached RUNNING all verify as `CALLBACK_INCOMPLETE` with
`CALLBACK_CLAIMED_INCOMPLETE_NO_REENTRY`.  Their distinct runtime artifacts
remain auditable, but none reopens the callback lease.

For a successfully completed callback, read-only verification is the advance
preview.  It replays public reingestion and task materialization entirely in
memory and returns the output-evidence, report-body, audit-head, reingestion,
next-pending, next-first-projection, next-bundle, and next-plan anchors.  The
caller retains those values and passes them back unchanged to `advance`.
The optional `expected_receipt_digest` and `expected_journal_head_digest` are
accepted only as a pair.  Supplying the pair independently anchors the preview
to the retained `execute` output.  If both are omitted, verification is an
explicit lost-output recovery and reports
`RECOVERED_LOCAL_ANCHORS_NOT_INDEPENDENT` in `execution_status`; local
rediscovery is never represented as an independently retained anchor.  The
same distinction applies to evidence-neutral failed receipts.

For a nonterminal preview, `next_attempt_root` is mandatory and must equal the
deterministic path
`current_attempt_root.parent / ("recursive-" + next_first_task_id_hex)`.
It must still be absent.  A terminal preview requires `next_attempt_root=None`.
For an already advanced capsule the argument may be omitted; if supplied, it
must equal the path bound by the advance marker.  Preview creates no attempt,
checkpoint, bundle file, or campaign artifact.

Advance is admitted only from the exact completed successful runtime capsule.
It verifies that capsule, reingests exactly one row, verifies report integrity,
and requires the pending count to fall by exactly one.  A nonterminal result
materializes and strongly verifies the next ordered task bundle.  It does not
authorize that bundle.

Advance publication is restart-safe at every noncommit leaf.  On re-entry,
all existing staged leaves are validated before any missing leaf is created;
only exact leaves are reused, and `advance.json` is published last after a final
campaign/runtime recapture and future-attempt absence check.  Marker-first,
missing-leaf, corrupt-subset, and unexpected-name images fail closed.  Thus a
crash may leave auditable exact staging, but it cannot expose a committed
advance early or overwrite a leaf.

When the pending count reaches zero, advance writes a terminal marker and does
not create `next_bundle.json`.  The API receives `None` for
`next_attempt_root` and all four next-boundary expectations.  The CLI therefore
omits `--next-attempt-root`,
`--expected-next-pending-evidence-digest`,
`--expected-next-first-pending-projection-digest`,
`--expected-next-bundle-digest`, and `--expected-next-plan-digest` together.
A terminal campaign cannot be used as another source.  A zero-task bundle is
never fabricated.

## Immutable layout

After authorization, the exact layout is:

```text
campaign_contract.json
source/
  descriptor.json
  commit.json
  rows.json
  report.json
  bundle.json
lease.json
campaign.json                 # authorization commit, published last
```

After a callback start, `callback_start_claim.json` is present at the root.
After a nonterminal advance, the additional exact layout is:

```text
advance/
  execution_receipt.json
  combined_rows.json
  output_report.json
  reingestion_receipt.json
  next_bundle.json
  advance.json                # advance commit, published last
```

The terminal layout omits only `next_bundle.json`.  Campaign-owned module,
version, root, `source`, and `advance` directories are mode 0700; files are
mode 0600, regular, single-link, canonical JSON artifacts.  A pre-existing
shared `$XDG_STATE_HOME/kg-op` ancestor may remain owner-controlled and
non-group/world-writable (for example, mode 0755); this implementation never
changes its mode.  Existing roots and artifacts are never overwritten.  The
implementation rejects aliases, unexpected names, source changes, digest
mismatches, corrupt partial leaves, and marker-first layouts.

Leaf publication is a Linux local-state primitive.  A mode-0600 staging file
under the fixed campaign prefix is moved to its final directory with libc
`renameat2(RENAME_NOREPLACE)`: there is no ordinary-rename or hardlink
fallback, and unsupported `ENOSYS`/`EINVAL`/`EOPNOTSUPP` or cross-filesystem
`EXDEV` conditions fail closed.  Interruption before the atomic move may leave
an owned single-link staging orphan outside the campaign root; it cannot enter
the exact capsule layout or block an exact retry.  Interruption after the move
exposes only the complete single-link target.  No operation overwrites an
existing target.

The campaign root is exactly
`$XDG_STATE_HOME/kg-op/structural-hypothesis-recursive-campaign/v1/<campaign-id>`;
a path that merely ends with the same suffix is invalid.  `..`, symlinked XDG
ancestors, and noncanonical state homes are rejected.  Read-only inspection of
a brand-new state home does not create the state home or any prefix directory;
authorization creates the secure fixed prefix only after all read-only gates.

## CLI protocol

The five actions are `inspect`, `authorize`, `execute`, `advance`, and
`verify`.  Every successful action writes one compact, key-sorted JSON result
to stdout and one human-readable status line to stderr.  Verification also
writes JSON because its completed-phase preview anchors are the explicit input
to `advance`; suppressing that result would make the protocol inoperable.
Every action returns zero on success.  Validation and runtime failures return
2 with no traceback, no JSON on stdout, and exactly one diagnostic line on
stderr, including command-line parse failures.

Before importing the campaign core, `authorize` and `execute` require
`SCOLHKG_OFFLINE=1` and `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `BLIS_NUM_THREADS` all equal
to `1`.  A missing or drifting value fails before an authorization capsule,
callback-start claim, or numerical-library import can occur.

A typical nonterminal sequence is:

```bash
python3 runners/run_structural_hypothesis_recursive_campaign.py inspect \
  --source-descriptor /absolute/anchors/seed2-source.json \
  --campaign-root /absolute/xdg/kg-op/structural-hypothesis-recursive-campaign/v1/step-seed2 \
  --campaign-id step-seed2 \
  --next-attempt-root /absolute/xdg/kg-op/structural-hypothesis-execution/v1/step-seed2

python3 runners/run_structural_hypothesis_recursive_campaign.py authorize \
  --source-descriptor /absolute/anchors/seed2-source.json \
  --campaign-root /absolute/xdg/kg-op/structural-hypothesis-recursive-campaign/v1/step-seed2 \
  --campaign-id step-seed2 \
  --expected-source-state-digest sha256:... \
  --expected-bundle-digest sha256:... \
  --expected-plan-digest sha256:... \
  --task-id task:000000000000000000000000 \
  --expected-task-digest sha256:... \
  --expected-provenance-binding-digest sha256:... \
  --authorization-id recursive-campaign-v1:... \
  --confirm-explicit-local-task-authorization

python3 runners/run_structural_hypothesis_recursive_campaign.py execute \
  --source-descriptor /absolute/anchors/seed2-source.json \
  --campaign-root /absolute/xdg/kg-op/structural-hypothesis-recursive-campaign/v1/step-seed2 \
  --expected-campaign-digest sha256:... \
  --expected-lease-digest sha256:... \
  --expected-provenance-binding-digest sha256:... \
  --expected-authorization-digest sha256:... \
  --expected-attempt-digest sha256:... \
  --confirm-real-local-execution
```

The execute confirmation authorizes a real local callback start.  It is not a
claim that resources are sufficient, the callback will succeed, or the result
is scientific evidence.  The caller must retain the JSON outputs outside the
campaign capsule and supply every required digest explicitly to `advance` and
`verify`.

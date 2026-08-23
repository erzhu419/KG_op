# Adopted structural-hypothesis successor materializer V1

This surface converts one fully verified local report adoption into one exact,
versioned successor task bundle. Its committed status is
`SUCCESSOR_MATERIALIZED_FROM_VERIFIED_ADOPTION_NOT_AUTHORIZED`; successful
verification reports the same status with the `VERIFIED_` prefix.

Only presented mechanics are true. The capsule does not establish global
currentness, external authority, R01 or any other authority class, runtime
readiness, authorization, execution, an actual-960 result, C1, scientific
confirmation, scientific refutation, or paper promotion. The source adoption
retains its original `ADOPTED_AS_LOCAL_REPORT_VERSION_NOT_PLANNED` meaning.
The successor is a separate downstream proposal and does not rewrite that
historical marker.

## Exact replay and anchor boundary

V1 accepts 1 through 30 pending cells and preserves the exact order obtained
by replaying the adopted publication's complete `combined_rows.json` through
the frozen hypothesis loop. It then invokes the already-versioned task
materializer and strongly verifies the rebuilt bundle. Every task is a complete
native `run_one(task)` dictionary, but its plan status
`READY_FOR_AUTHORIZATION` is only presented mechanics. The capsule and bundle
remain `NOT_AUTHORIZED` and no executor is called.

The caller must supply three observations retained outside the output tree:

- the canonical adoption-body digest saved when the source adoption was
  committed;
- the canonical digest of the exact ordered `pending_evidence` list obtained
  from an independently inspected adopted report;
- the canonical digest of the first execution projection with exactly
  `profile`, `domain`, `line`, `seed`, `d`, `N`, and `n0`.

The adoption digest is the umbrella anchor. The core first captures the exact
adoption marker and artifact map through held directory descriptors, verifies
that the marker body matches the independently supplied adoption digest, and
only then derives the publication's four raw SHA-256 values and other full-chain
arguments. It still runs the complete adoption verifier and rechecks the held
generation. Those derived publication values are therefore **transitively
anchored by the adoption digest**; they are not independently re-observed
anchors and must not be described that way. Local SHA-256 commitments are not
signatures and provide no external authority.

The external publication root, adoption tree, raw base CSV, completed source
attempt, all versioned contracts, base manifest, and native materializer assets are
still required and are verified against the captured adoption. An opaque ID or
SHA cannot replace any raw leaf.

## Materialize one unauthorized successor

All paths must be absolute. `--successor-id` must equal the direct-child name
beneath
`$XDG_STATE_HOME/kg-op/structural-hypothesis-adopted-successor/v1/`.
`--future-attempt-root` must be beneath the frozen runtime state prefix and must
have the same direct-child basename as `--successor-id`; it must be absent
during materialization. The confirmation flag acknowledges only the
local write of a successor capsule; it is not consent to prepare or execute a
task.

```bash
python3 runners/run_structural_hypothesis_adopted_successor_materializer.py materialize \
  --publication-root /home/user/.local/state/kg-op/structural-hypothesis-reingestion/v1/publication-0001 \
  --adoption-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_report_adoption_v1.json \
  --adoption-root /home/user/.local/state/kg-op/structural-hypothesis-report-adoption/v1/adoption-0001 \
  --adoption-id adoption-0001 \
  --successor-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_adopted_successor_materializer_v1.json \
  --successor-root /home/user/.local/state/kg-op/structural-hypothesis-adopted-successor/v1/successor-0001 \
  --successor-id successor-0001 \
  --base-evidence-csv /absolute/path/to/base-evidence.csv \
  --source-attempt-root /home/user/.local/state/kg-op/structural-hypothesis-execution/v1/source-attempt-0001 \
  --hypothesis-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_loop_v1.json \
  --executor-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_executor_v1.json \
  --runtime-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_single_task_runtime_v1.json \
  --publisher-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_reingestion_publisher_v1.json \
  --materializer-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_task_materializer_v1.json \
  --base-manifest /absolute/path/to/SC-OLH-KG/performance/manifests/v18b_exactkg_mcdiag.json \
  --asset-root /absolute/path/to/SC-OLH-KG/performance/task_inputs/structural_hypothesis_materializer_v1 \
  --future-attempt-root /home/user/.local/state/kg-op/structural-hypothesis-execution/v1/successor-0001 \
  --expected-adoption-digest sha256:... \
  --expected-pending-evidence-digest sha256:... \
  --expected-first-pending-projection-digest sha256:... \
  --confirm-successor-materialization
```

On success, stdout is exactly one compact, key-sorted canonical JSON line with
these fields:

- `status`, `successor_root`, `successor_digest`, and `adoption_digest`;
- `pending_evidence_digest` and `first_pending_projection_digest`;
- `bundle_digest`, `plan_digest`, `first_task_id`, `first_task_digest`, and
  `task_count`;
- `future_attempt_root`, `checkpoint_root`, and `authorization_status`.

Retain that line outside the adoption, successor, and future-attempt trees.
Human-readable status goes only to stderr. Any argument, chain, path, layout,
mode, alias, task replay, or digest failure returns two and leaves stdout empty.

## Commit marker, no-clobber, and path noncreation

The successor root must be entirely absent. V1 creates a mode-0700 root with
exactly three mode-0600 files:

```text
successor_contract.json
bundle.json
successor.json
```

`successor.json` is the only commit marker and is written last through the
no-clobber, fsynced publication path. A crash before that marker leaves
`INCOMPLETE_NOT_MATERIALIZED`; the incomplete root cannot be reused. Existing
files, directories, empty directories, symlinks, FIFOs, or broken links are
never overwritten.

The future attempt and its `checkpoints/` child are string bindings only. The
successor surface creates neither. It writes no `current.json`, authorization,
runtime preparation, receipt, journal, result, checkpoint, retry, or resume
artifact.

## Read-only verification

Verification requires the three retained source anchors plus the independently
saved successor, bundle, and plan digests from materialization:

```bash
python3 runners/run_structural_hypothesis_adopted_successor_materializer.py verify \
  --publication-root /home/user/.local/state/kg-op/structural-hypothesis-reingestion/v1/publication-0001 \
  --adoption-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_report_adoption_v1.json \
  --adoption-root /home/user/.local/state/kg-op/structural-hypothesis-report-adoption/v1/adoption-0001 \
  --adoption-id adoption-0001 \
  --successor-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_adopted_successor_materializer_v1.json \
  --successor-root /home/user/.local/state/kg-op/structural-hypothesis-adopted-successor/v1/successor-0001 \
  --successor-id successor-0001 \
  --base-evidence-csv /absolute/path/to/base-evidence.csv \
  --source-attempt-root /home/user/.local/state/kg-op/structural-hypothesis-execution/v1/source-attempt-0001 \
  --hypothesis-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_loop_v1.json \
  --executor-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_executor_v1.json \
  --runtime-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_single_task_runtime_v1.json \
  --publisher-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_reingestion_publisher_v1.json \
  --materializer-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_task_materializer_v1.json \
  --base-manifest /absolute/path/to/SC-OLH-KG/performance/manifests/v18b_exactkg_mcdiag.json \
  --asset-root /absolute/path/to/SC-OLH-KG/performance/task_inputs/structural_hypothesis_materializer_v1 \
  --future-attempt-root /home/user/.local/state/kg-op/structural-hypothesis-execution/v1/successor-0001 \
  --expected-adoption-digest sha256:... \
  --expected-pending-evidence-digest sha256:... \
  --expected-first-pending-projection-digest sha256:... \
  --expected-successor-digest sha256:... \
  --expected-bundle-digest sha256:... \
  --expected-plan-digest sha256:...
```

Successful verification writes nothing to stdout and makes no changes. The
future attempt is allowed to exist later, but verification does not inspect it
as proof of preparation, readiness, or execution. It verifies only the path
binding originally committed in the successor marker.

## Audited local seed-1 instance boundary

The audited source adoption is
`/home/erzhu419/.local/state/kg-op/structural-hypothesis-report-adoption/v1/ca54f50-factor-seed0-v1`
with independently retained adoption digest
`sha256:b3a858c481ccc9ffd6bca6033e0ff2bbd410500bcfe6c4690afce85db40a45e5`.
Its exact 29-cell pending-list digest is
`sha256:7006a17aaac206503977cb090a52dea3838aa06b378b4e0278e7cbebfe46041e`.
The first projection is:

```json
{"N":20,"d":50,"domain":"FactorShockStatePolicyRZDT1","line":"lodo","n0":10,"profile":"full","seed":1}
```

Its canonical digest is
`sha256:e26ba673070e7a0f59edb47c2c7dba5eeaba19546ad51c741e37cdc2b15cf642`.
These values freeze a candidate seed-1 successor; they do not authorize it.
The proposed successor and future-attempt paths remain absent until an explicit
materialization action is chosen:

```text
/home/erzhu419/.local/state/kg-op/structural-hypothesis-adopted-successor/v1/ca54f50-factor-seed1-v1
/home/erzhu419/.local/state/kg-op/structural-hypothesis-execution/v1/ca54f50-factor-seed1-v1
```

## Next hard gate and test boundary

The mandatory KAT creates its own 47-field fake base CSV, 30-cell report,
fake-only completed seed-0 attempt, publication, adoption, and exact 29-task
successor under a native temporary state home. It proves canonical stdout,
read-only verification, independent-anchor rejection before root creation,
no-clobber, exact first-task binding, the complete native task argument object, modes,
layout, and tamper rejection. It never calls the real benchmark or `run_one`
and never uses historical state as the test fixture.

The next hard gate is separate explicit authorization of one exact task after
CPU and memory inspection, the 12-process-fork preflight, and checkpoint-path
review. Only after that authorization may the runtime prepare or execute an
attempt. A later recursive publication also needs a downstream publisher that
can consume the new round; the existing base-CSV-bound publisher V1 does not
by itself establish that path. None of those gates belongs to this surface,
and this work does not move to Package1, accounts, network policy, credentials,
or any external system.

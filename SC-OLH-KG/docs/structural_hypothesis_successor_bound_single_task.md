# Successor-bound single-task bridge V1

This bridge connects one fully verified adopted-successor capsule to the
existing single-task runtime V1 without changing that runtime's attempt
format. It exposes three operations only:

- `inspect`: read-only full-chain verification and derivation of the exact
  first-task provenance binding;
- `prepare`: explicit local authorization and creation of one runtime V1
  `AUTHORIZED` attempt;
- `verify`: read-only full-chain verification that the attempt is still
  exactly `AUTHORIZED_NOT_EXECUTED` and belongs to that successor.

There is no execute action. The bridge never imports the benchmark, calls
`run_one`, runs the 12-process preflight, creates a result or receipt, writes a
mutable current pointer, or performs reingestion.

## What closes the detached-bundle gap

The old runtime V1 strongly verifies a report, task bundle, plan, contracts,
assets, and first task, but its attempt schema does not contain the adopted
successor or adoption digest. Calling its `prepare` command directly therefore
cannot establish successor provenance.

This bridge first verifies the complete original adoption and successor chain,
captures the exact successor generation through held directory descriptors,
rebuilds its 1--30 task bundle, and admits only `plan.tasks[0]`. It then builds
this complete canonical provenance object:

```text
bridge_contract: id, digest
source_adoption: id, digest
source_successor: id, digest, pending_evidence_digest,
                  first_pending_projection_digest
bundle_binding: bundle_id, bundle_digest, plan_id, plan_digest, task_count
task_binding: task_id, task_digest, ordinal=0, complete cell
attempt_binding: attempt_root, checkpoint_root,
                 runtime_contract_id, runtime_contract_digest
```

Its canonical digest `sha256:<hex>` deterministically yields the only accepted
runtime authorization ID:

```text
successor-bound-v1:<same-64-hex>
```

The unchanged runtime V1 stores that ID in `authorization.json`, and the
authorization digest is in turn included in `attempt.json`. The bridge contract
digest therefore reaches the runtime attempt transitively without adding a
sidecar or changing the frozen runtime layout.

The attempt by itself is **not** proof of successor provenance. Full bridge
verification still requires the external bridge contract, the complete
original adoption and successor trees, every raw local input consumed by those
verifiers, and the independently retained digests printed by prior gates.

## Inspect without authorization or writes

All paths must be absolute. `--attempt-root` must be the exact future attempt
path already committed by the successor, must have the same direct-child name
as `--successor-id`, and must be absent. Inspection does not create the state
home, runtime prefix, attempt root, checkpoint directories, authorization, or
any other artifact.

```bash
python3 runners/run_structural_hypothesis_successor_bound_single_task.py inspect \
  --publication-root /absolute/source-publication \
  --adoption-contract /absolute/repo/SC-OLH-KG/performance/manifests/structural_hypothesis_report_adoption_v1.json \
  --adoption-root /absolute/source-adoption \
  --adoption-id source-adoption-id \
  --successor-contract /absolute/repo/SC-OLH-KG/performance/manifests/structural_hypothesis_adopted_successor_materializer_v1.json \
  --successor-root /absolute/adopted-successor \
  --successor-id exact-successor-id \
  --base-evidence-csv /absolute/base-evidence.csv \
  --source-attempt-root /absolute/completed-source-attempt \
  --hypothesis-contract /absolute/repo/SC-OLH-KG/performance/manifests/structural_hypothesis_loop_v1.json \
  --executor-contract /absolute/repo/SC-OLH-KG/performance/manifests/structural_hypothesis_executor_v1.json \
  --runtime-contract /absolute/repo/SC-OLH-KG/performance/manifests/structural_hypothesis_single_task_runtime_v1.json \
  --publisher-contract /absolute/repo/SC-OLH-KG/performance/manifests/structural_hypothesis_reingestion_publisher_v1.json \
  --materializer-contract /absolute/repo/SC-OLH-KG/performance/manifests/structural_hypothesis_task_materializer_v1.json \
  --bridge-contract /absolute/repo/SC-OLH-KG/performance/manifests/structural_hypothesis_successor_bound_single_task_v1.json \
  --base-manifest /absolute/repo/SC-OLH-KG/performance/manifests/v18b_exactkg_mcdiag.json \
  --asset-root /absolute/repo/SC-OLH-KG/performance/task_inputs/structural_hypothesis_materializer_v1 \
  --attempt-root /absolute/runtime/v1/exact-successor-id \
  --expected-adoption-digest sha256:... \
  --expected-pending-evidence-digest sha256:... \
  --expected-first-pending-projection-digest sha256:... \
  --expected-successor-digest sha256:... \
  --expected-bundle-digest sha256:... \
  --expected-plan-digest sha256:... \
  --task-id task:... \
  --expected-task-digest sha256:...
```

Successful inspection prints one compact, sorted canonical JSON line. Its
status is
`INSPECTED_SUCCESSOR_BOUND_TASK_NOT_AUTHORIZED_NOT_PREPARED`, with
`authorization_status=NOT_AUTHORIZED` and `attempt_status=NOT_PREPARED`.
Retain the complete line outside the successor and future attempt trees. In
particular, the next command requires the independently saved
`provenance_binding_digest` and `required_authorization_id`.

## Explicitly authorize and prepare, without execution

Before starting Python for `prepare`, pin the offline and single-thread native
environment:

```bash
export SCOLHKG_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
```

Repeat every inspect argument, then add the two exact values printed by
inspection and the explicit authorization acknowledgement:

```bash
python3 runners/run_structural_hypothesis_successor_bound_single_task.py prepare \
  [all inspect arguments unchanged] \
  --expected-provenance-binding-digest sha256:... \
  --authorization-id successor-bound-v1:... \
  --confirm-successor-bound-local-authorization
```

The confirmation grants local consent only for the exact ordinal-zero task and
attempt binding. It is not identity, a signature, external authority,
currentness, runtime readiness, or permission to execute.

All bridge provenance, independent-anchor, first-task, attempt-path, provenance
digest, and authorization-ID checks finish before the bridge calls runtime V1
to create the attempt root. Runtime V1 may create or secure its state-prefix
parents as part of its own preparation. The resulting root has exactly the
unchanged runtime V1 authorized layout:

```text
attempt.json
authorization.json
bundle.json
inputs/
  report.json
  hypothesis_contract.json
  executor_contract.json
  materializer_contract.json
journal/
  0000_AUTHORIZED.json
checkpoints/
  <domain>/seed<seed>/
```

There is no bridge or provenance sidecar. In particular, there is no
`preflight.json`, `0001_RUNNING.json`, raw result, receipt, or checkpoint file.
Successful preparation prints canonical JSON with status
`SUCCESSOR_BOUND_AUTHORIZED_NOT_EXECUTED`, the full provenance binding, and the
independently retainable `authorization_digest` and `attempt_digest`.

Preparation uses several no-clobber writes. If a failure occurs after runtime
V1 starts publishing the attempt, a partial or otherwise non-verifiable attempt
root may remain. It is nonreusable: do not retry in place, overwrite it, or
treat it as authorization evidence. Diagnose it read-only and select a new
versioned successor/attempt identity for any retry.

## Verify the authorized-only handoff

Repeat every inspect argument and supply the independently saved preparation
anchors:

```bash
python3 runners/run_structural_hypothesis_successor_bound_single_task.py verify \
  [all inspect arguments unchanged] \
  --expected-provenance-binding-digest sha256:... \
  --expected-authorization-digest sha256:... \
  --expected-attempt-digest sha256:...
```

The authorization ID is not accepted as a free verify argument. It is derived
again from the provenance digest and must exactly match the authorization
inside the runtime attempt. Verification accepts only runtime status
`VERIFIED_AUTHORIZED_NOT_EXECUTED`; a preflight, running, completed, tampered,
detached, or incomplete attempt is rejected. Success writes nothing to stdout
and reports
`VERIFIED_SUCCESSOR_BOUND_AUTHORIZED_NOT_EXECUTED` on stderr.

Bridge verification is a point-in-time local observation, not a lock,
reservation, atomic handoff, or exactly-once guarantee. The legacy runtime
`execute` command remains a mechanical path that does not itself verify
successor provenance. If a later gate deliberately uses it, a successful
bridge `verify` must immediately precede that handoff, and even that ordering
does not prevent a same-user mutation between the two commands.

## Audited seed-1 binding

The current local successor capsule is:

```text
/home/erzhu419/.local/state/kg-op/structural-hypothesis-adopted-successor/v1/ca54f50-factor-seed1-v1
```

Its retained mechanics anchors are:

```text
adoption_digest  sha256:b3a858c481ccc9ffd6bca6033e0ff2bbd410500bcfe6c4690afce85db40a45e5
successor_digest sha256:410ff24f59d43dae48f9e2fab01c7ff59e3e877d2c823470fad5a270b6d3badb
pending_evidence_digest sha256:7006a17aaac206503977cb090a52dea3838aa06b378b4e0278e7cbebfe46041e
first_pending_projection_digest sha256:e26ba673070e7a0f59edb47c2c7dba5eeaba19546ad51c741e37cdc2b15cf642
bundle_digest    sha256:ca0638a98e6d6712e3467a57d784a73a68144328b11500c44e43a296245a5b93
plan_digest      sha256:b6e0094d19067f936e5f0c1cb215584f08d423b21643258d3d7becdea23ad5d4
task_count       29
task_id          task:95ff940d4b1317f8564c161b
task_digest      sha256:3063f34add53674166b489cf5892901d6d4cecd8c0621eaf5b76e3e1c930bf0c
```

The proposed attempt path is still the exact successor binding:

```text
/home/erzhu419/.local/state/kg-op/structural-hypothesis-execution/v1/ca54f50-factor-seed1-v1
```

These values establish only local presented mechanics. They do not establish
external authority, global currentness, runtime readiness, preflight success,
execution, scientific confirmation or refutation, reingestion, or paper
promotion.

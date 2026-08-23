# Structural-hypothesis single-task runtime V1

This gate is the first intentionally real local execution surface for the
structural-hypothesis loop.  It admits exactly one first-pending task from a
verified materialized bundle, creates a fresh attempt, and keeps preparation,
execution, and verification as separate commands.

It is deliberately narrower than a scheduler or experiment launcher.  V1 has
no network, remote host, credential, account, shell, scheduler, retry, resume,
matrix, or multi-task operation.  A successful receipt is still not
reingestion and is not a scientific verdict.

## Mandatory process environment

The variables below must be present **before the Python interpreter starts**:

```bash
export SCOLHKG_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
```

This ordering is a correctness and resource boundary.  NumPy uses a 32-thread
OpenBLAS pool on the current host when unconstrained; combining that pool with
`exact_jobs=12` and `process_fork` could oversubscribe the host severely.  The
runner checks the variables before importing the runtime core or NumPy and
fails closed rather than setting them after import.  Environment text alone
cannot prove the size of a pool that was already loaded; the real-execution
preflight also inspects detected thread pools and rejects any pool that is not
single-threaded.

The frozen task remains `offline_only=true`.  These environment variables do
not grant authorization; they only constrain an already explicit local
attempt.

## Prepare one fresh attempt

The bundle must already have been materialized with its absolute checkpoint
root under the intended fresh attempt.  The caller must independently supply
the observed bundle and plan digests and the exact first-pending task ID.  V1
does not silently choose another task and does not accept a list.

```bash
python3 runners/run_structural_hypothesis_single_task.py prepare \
  --report /absolute/path/to/structural_hypothesis_report.json \
  --bundle /absolute/path/to/materialized_tasks.json \
  --hypothesis-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_loop_v1.json \
  --executor-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_executor_v1.json \
  --materializer-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_task_materializer_v1.json \
  --runtime-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_single_task_runtime_v1.json \
  --base-manifest /absolute/path/to/SC-OLH-KG/performance/manifests/v18b_exactkg_mcdiag.json \
  --asset-root /absolute/path/to/SC-OLH-KG/performance/task_inputs/structural_hypothesis_materializer_v1 \
  --attempt-root /home/user/.local/state/kg-op/structural-hypothesis-execution/v1/attempt-0001 \
  --task-id task:... \
  --expected-bundle-digest sha256:... \
  --expected-plan-digest sha256:... \
  --authorization-id local-consent-attempt-0001
```

`--attempt-root` must be beneath
`$XDG_STATE_HOME/kg-op/structural-hypothesis-execution/v1/` (or the equivalent
`$HOME/.local/state/` prefix when `XDG_STATE_HOME` is unset), absolute, and
entirely absent: an existing file,
directory, empty directory, or symlink is rejected.  Preparation claims and
creates the attempt with private permissions, validates the complete source
chain again, authorizes exactly one task, and publishes its artifacts and
journal with atomic no-clobber operations.  It does not import the benchmark
executor or invoke `run_one`.

The core resolves the path and rejects symlinks throughout its ancestor chain.
It enforces ownership and private-mode policy from the resolved frozen state
base downward, as well as the allowed local state root and path separation.
The runner's last-component check is only an early diagnostic and is not the
security boundary.

On success, stdout contains one canonical compact JSON object with exactly the
status, canonical attempt root, task ID, authorization digest, plan digest,
bundle digest, and pre-execution attempt digest.  Save that output outside the
attempt tree: it is the caller's first observed anchor for the following
command.

The task's checkpoint directory must also be fresh.  The native task has
checkpoint resume enabled, while the underlying loader does not independently
bind a checkpoint to the task digest.  Consequently V1 treats any pre-existing
checkpoint path as unsafe and provides no same-attempt resume or retry.

## Explicit real execution

Execution is a separate command and requires the exact authorization digest
printed by preparation plus an explicit real-execution acknowledgement:

```bash
python3 runners/run_structural_hypothesis_single_task.py execute \
  --attempt-root /home/user/.local/state/kg-op/structural-hypothesis-execution/v1/attempt-0001 \
  --runtime-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_single_task_runtime_v1.json \
  --expected-authorization-digest sha256:... \
  --confirm-real-local-execution
```

Only this command lazily imports the native benchmark and calls
`performance.benchmark_lodo_meta_prior.run_one(task)`.  There is no dry-run or
executor-module override on the CLI.  The runtime rechecks the immutable
authorization, the fresh execution claim, the environment, and the one bound
task before crossing that boundary.  Result adaptation and receipt/journal
publication are no-clobber operations.  A callback exception is recorded as a
failed attempt; it is not evidence against the hypothesis.

If a callback returns but native-result conversion or no-clobber persistence
fails, the receipt remains evidence-neutral with a `ResultPersistenceRejected`
failure and the COMPLETED journal records
`runtime_error_code=RESULT_PERSISTENCE_REJECTED`.

If the preflight artifact is published but the process stops before the
RUNNING journal event, verification reports
`VERIFIED_PREFLIGHT_PASSED_NO_CALLBACK`.  V1 does not re-enter or retry that
stranded attempt; the caller must prepare a new attempt.

The execute admission preflight requires one task, an affinity mask containing
at least 12 CPUs for the 12 exact-KG fork workers, detected numerical pools at
one thread each, at least 12 GiB `MemAvailable`, at least 2 GiB free on the
checkpoint filesystem, and a fork probe that successfully creates and reaps
exactly 12 child processes.  These are point-in-time admission checks.  V1
imposes no OS-level peak-memory limit or wall-time timeout, and makes no claim
about historical runtime reconstruction.

On completion, stdout contains canonical JSON with the authorization,
receipt, journal-head, and pre-execution attempt-binding digests.  Preserve
this output outside the attempt tree.  It supplies the independent observed
final digests needed for a strong completed-attempt verification.

## Verify without execution

```bash
python3 runners/run_structural_hypothesis_single_task.py verify \
  --attempt-root /home/user/.local/state/kg-op/structural-hypothesis-execution/v1/attempt-0001 \
  --runtime-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_single_task_runtime_v1.json \
  --base-manifest /absolute/path/to/SC-OLH-KG/performance/manifests/v18b_exactkg_mcdiag.json \
  --asset-root /absolute/path/to/SC-OLH-KG/performance/task_inputs/structural_hypothesis_materializer_v1 \
  --expected-authorization-digest sha256:... \
  --expected-receipt-digest sha256:... \
  --expected-journal-head-digest sha256:... \
  --expected-attempt-digest sha256:...
```

Verification imports neither benchmark module and never calls `run_one`.  It
recomputes the attempt chain from the frozen runtime contract and local raw
inputs, enforces the enumerated attempt/input/journal layout, and rejects
missing, replaced, digest-inconsistent, or hard-linked JSON artifacts.  The
three final expected digests may be omitted for an AUTHORIZED, PREFLIGHT, or
RUNNING non-completed state; a COMPLETED attempt requires all three.  PREFLIGHT
and RUNNING verification only report an incomplete local state and do not
constitute successful execution evidence.  Verification cannot turn a failed
or incomplete attempt into successful evidence.

## Claim boundary

The recovered inputs and current pinned runner establish a
current-runner-compatible local replay, not the exact historical Python
environment or host.  Local digests provide integrity commitments, not
identity, signatures, external authority, or currentness.  A verified
successful single-task receipt must pass the distinct receipt-to-row and
reingestion gates before it can affect the recursive hypothesis report.
Digests saved independently from prepare/execute stdout—not values recomputed
from the directory being checked—can detect later divergence from those
observations.  They are not a malicious-same-user trust boundary: a user able
to rewrite both the attempt and the independently stored expected values can
still produce a self-consistent local history.

Native `checkpoint_latest.pkl` and `checkpoint_stage_*.pkl` files are mutable
executor state, not receipt evidence.  V1 validates their allowed names and
basic file policy, but does not claim post-run external alias detection for
checkpoint files created during the native run.

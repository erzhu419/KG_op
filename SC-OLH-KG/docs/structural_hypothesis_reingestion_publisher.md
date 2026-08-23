# Structural-hypothesis reingestion publisher V1

This surface turns exactly one independently anchored, successful local
single-task receipt into a new versioned hypothesis report and reingestion
receipt.  It publishes those artifacts locally but never adopts them as the
current report and never builds or executes the next plan.

V1 has no benchmark import, `run_one`, network, remote host, credential,
account, scheduler, shell, retry, resume, multi-receipt, current-pointer, or
replan operation.  A publication is a local mechanics result, not external
authority, currentness, or a scientific verdict.

## Exact input boundary

The publisher requires the original raw evidence CSV and the same completed
single-task attempt used to obtain the saved execution anchors.  It does not
accept detached plan, authorization, receipt, or evidence-row arguments as a
substitute for that full chain.

The following values must come from observations saved outside the attempt:

- source evidence digest;
- plan digest;
- authorization digest;
- execution receipt digest;
- execution journal-head digest;
- pre-execution attempt-binding digest.

The hypothesis, executor, runtime, and publisher contracts are passed to the
core as paths.  The core performs secure, duplicate-key-rejecting reads and
binds their raw bytes; the runner deliberately does not pre-read them across a
path-check/use gap.  The publisher contract pins all four source contracts,
which must also equal the snapshots recovered from the completed attempt.

## Publish one result without adopting it

All CLI paths must be absolute.  `--publication-id` is a local mechanics
label, not an identity, signature, or free-form path.

```bash
python3 runners/run_structural_hypothesis_reingestion.py publish \
  --base-evidence-csv /absolute/path/to/base-evidence.csv \
  --attempt-root /home/user/.local/state/kg-op/structural-hypothesis-execution/v1/attempt-0001 \
  --hypothesis-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_loop_v1.json \
  --executor-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_executor_v1.json \
  --publisher-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_reingestion_publisher_v1.json \
  --runtime-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_single_task_runtime_v1.json \
  --base-manifest /absolute/path/to/SC-OLH-KG/performance/manifests/v18b_exactkg_mcdiag.json \
  --asset-root /absolute/path/to/SC-OLH-KG/performance/task_inputs/structural_hypothesis_materializer_v1 \
  --publication-root /home/user/.local/state/kg-op/structural-hypothesis-reingestion/v1/publication-0001 \
  --expected-source-evidence-digest sha256:... \
  --expected-plan-digest sha256:... \
  --expected-authorization-digest sha256:... \
  --expected-execution-receipt-digest sha256:... \
  --expected-execution-journal-head-digest sha256:... \
  --expected-execution-attempt-digest sha256:... \
  --publication-id local-reingestion-0001 \
  --confirm-local-reingestion
```

Publication requires exactly one successful authorized row.  An AUTHORIZED,
PREFLIGHT, RUNNING, incomplete, failed-only, persistence-rejected, tampered,
or anchor-mismatched attempt is rejected before a committed publication can
exist.  A failed execution remains evidence-neutral and is not reingested as a
hypothesis refutation.

On success, stdout contains exactly one canonical compact JSON object with the
status, canonical publication root, plan/authorization/execution-receipt
digests, accepted and ignored counts, publication and reingestion digests, and
the output report body, audit-head, and evidence digests.  Save this line
outside both the attempt and publication trees; it is the caller's independent
final observation anchor.  Human-readable status is written only to stderr.

The raw base CSV has the frozen 47-column wire representation and therefore
retains string-valued CSV fields.  The accepted receipt contributes the frozen
27-field normalized evidence row with native JSON types.  V1 preserves both
representations without manufacturing an authoritative merged CSV:
`combined_rows.json`, ordered as exact base-CSV rows followed by the exact
successful receipt row, is the only authoritative replay evidence.

## Commit-marker and no-clobber boundary

`--publication-root` must be entirely absent.  An existing file, directory,
empty directory, symlink, FIFO, or broken link is rejected.  V1 never replaces
or retries a publication root.

The core creates private directories and writes JSON/raw snapshots with mode
0600 using same-directory temporary files, file fsync, and atomic
no-clobber links, then fsyncs their parent directories.  `publication.json` is
the final commit marker and binds the exact allowed layout and raw leaves.  It
is published only after the copied inputs, completed execution chain, output
report, and reingestion receipt have all passed full verification.

A crash before that marker leaves an incomplete, non-published directory.  It
cannot be verified as `PUBLISHED_NOT_ADOPTED` and cannot be reused; the caller
must choose a new publication root and ID.  A committed publication is never
modified in place.  Native checkpoint pickle files are not copied or treated
as receipt evidence or resume authority.

## Verify against live external inputs

Verification remains tied to the original raw CSV and completed attempt.  It
does not trust paths or anchors read only from the locally rewriteable commit
marker.

```bash
python3 runners/run_structural_hypothesis_reingestion.py verify \
  --base-evidence-csv /absolute/path/to/base-evidence.csv \
  --attempt-root /home/user/.local/state/kg-op/structural-hypothesis-execution/v1/attempt-0001 \
  --hypothesis-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_loop_v1.json \
  --executor-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_executor_v1.json \
  --publisher-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_reingestion_publisher_v1.json \
  --runtime-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_single_task_runtime_v1.json \
  --base-manifest /absolute/path/to/SC-OLH-KG/performance/manifests/v18b_exactkg_mcdiag.json \
  --asset-root /absolute/path/to/SC-OLH-KG/performance/task_inputs/structural_hypothesis_materializer_v1 \
  --publication-root /home/user/.local/state/kg-op/structural-hypothesis-reingestion/v1/publication-0001 \
  --expected-source-evidence-digest sha256:... \
  --expected-plan-digest sha256:... \
  --expected-authorization-digest sha256:... \
  --expected-execution-receipt-digest sha256:... \
  --expected-execution-journal-head-digest sha256:... \
  --expected-execution-attempt-digest sha256:... \
  --expected-publication-digest sha256:... \
  --expected-reingestion-digest sha256:... \
  --expected-output-report-body-digest sha256:... \
  --expected-output-audit-head sha256:... \
  --expected-output-evidence-digest sha256:...
```

Successful verification writes nothing to stdout and never changes either
tree.  It recomputes the raw CSV, completed execution, publication, distinct
reingestion receipt, and output report chains.  Missing or inconsistent final
anchors are rejected rather than reconstructed from the directory being
verified.

Both commands return zero only on success.  Argument, path, no-clobber, I/O,
contract, chain, or digest failures return two, leave stdout empty, and print a
short error to stderr without a traceback.

## Claim and testing boundary

`PUBLISHED_NOT_ADOPTED` means only that a new local report artifact was
produced and committed from one successful row.  It does not mean the report
is current, externally verified, ready for Package1, complete across the other
pending cells, or adopted by any recursive controller.  Replanning and current
selection require later, separately authorized gates.

The mandatory clean-checkout tests synthesize the base CSV, source report,
materialized first task, completed fake-only attempt, and publication under a
native temporary state root.  They never depend on an ignored historical CSV
and never call the real benchmark.  Any optional historical smoke must be
marked separately and cannot satisfy the publisher completion gate.

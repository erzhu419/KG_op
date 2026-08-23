# Structural-hypothesis task materializer V1

This gate turns the pending cells in a verified structural-hypothesis report
into complete, native `run_one(task)` dictionaries.  It remains a local
mechanics gate: its output status is `MATERIALIZED_NOT_AUTHORIZED`, while the
embedded execution plan is only `READY_FOR_AUTHORIZATION`.

## Frozen local inputs

The versioned contract is
`performance/manifests/structural_hypothesis_task_materializer_v1.json`.  It
binds all of the following rather than accepting an opaque archive label:

- the frozen hypothesis and executor contract digests;
- the raw and canonical digests of
  `performance/manifests/v18b_exactkg_mcdiag.json`;
- the raw SHA-256, schema, domain identity, seed set, per-seed design
  fingerprints, and companion archive identity for each of the three domains;
- the raw SHA-256 and relative path of
  `performance/run_lodo_manifest_shard.py`, and its pure
  `build_run_one_task(argv)` entry point captured at module definition time;
- the exact argv template and normalized digest of every resolved 438-key task
  argument object (the caller-selected absolute checkpoint path is replaced by
  one frozen placeholder before hashing).

The six versioned native JSON inputs live under
`performance/task_inputs/structural_hypothesis_materializer_v1/`.  For every
domain this includes the complete `source_initial_designs.json` and its
`heldout_<domain>.json` companion.  The verifier rejects a missing or extra
seed, wrong raw byte hash, symlink/path-escape alias, mismatched companion
fingerprint, wrong base manifest, changed runner, changed source report,
incomplete task, dependency drift that changes any resolved argument, or any
recomputed bundle built from different bytes.  It materializes the report's
exact ordered pending-cell set: 30 tasks for the current historical report and
the remaining subset after a verified recursive reingestion round.
All seven bound input files and the runner binding are rechecked after task
construction so the artifact cannot retain a pre-swap raw-byte claim.

The recovered local bytes are stronger evidence than an archive name or a
remembered SHA alone because they can be parsed and replayed.  They have no
external signature or current authority.  Materializing them through the
current pinned runner demonstrates a current-runner-compatible replay; it does
not establish the exact historical Python environment, dependency versions,
host state, or historical runtime.

## Offline CLI

Materialize a new bundle (the checkpoint root must be absolute):

```bash
python3 runners/run_structural_hypothesis_task_materializer.py materialize \
  --report /absolute/path/to/structural_hypothesis_report.json \
  --hypothesis-contract performance/manifests/structural_hypothesis_loop_v1.json \
  --executor-contract performance/manifests/structural_hypothesis_executor_v1.json \
  --materializer-contract performance/manifests/structural_hypothesis_task_materializer_v1.json \
  --base-manifest performance/manifests/v18b_exactkg_mcdiag.json \
  --asset-root performance/task_inputs/structural_hypothesis_materializer_v1 \
  --checkpoint-root /absolute/local/checkpoints/structural-hypothesis-v1 \
  --out /absolute/path/to/materialized_tasks.json
```

Verify the artifact against the same inputs:

```bash
python3 runners/run_structural_hypothesis_task_materializer.py verify \
  --bundle /absolute/path/to/materialized_tasks.json \
  --report /absolute/path/to/structural_hypothesis_report.json \
  --hypothesis-contract performance/manifests/structural_hypothesis_loop_v1.json \
  --executor-contract performance/manifests/structural_hypothesis_executor_v1.json \
  --materializer-contract performance/manifests/structural_hypothesis_task_materializer_v1.json \
  --base-manifest performance/manifests/v18b_exactkg_mcdiag.json \
  --asset-root performance/task_inputs/structural_hypothesis_materializer_v1 \
  --checkpoint-root /absolute/local/checkpoints/structural-hypothesis-v1
```

JSON input parsing rejects duplicate keys.  The library call is read-only: it
does not create a result, checkpoint, or bundle file.  The CLI's sole persistent
write is the explicitly requested `materialize --out` bundle; publication may
create its parent directories and a transient same-directory temporary file.
It is atomic and refuses to place the bundle over any existing path, input
alias, or location beneath the checkpoint root.  The
checkpoint root is only embedded into task arguments; neither `materialize`
nor `verify` creates that directory or inspects checkpoint contents.

## Stage boundary

Materialization does **not** grant consent, authorize tasks, invoke `run_one`,
create checkpoints, submit scheduler work, access credentials, use the network,
produce execution receipts, or admit rows back into the hypothesis loop.  The
bundle's canonical digest is an integrity commitment, not a signature.  A later
gate must explicitly authorize the exact plan digest before any executor can be
called, and only verified successful receipts may later be considered for
reingestion.

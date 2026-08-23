# Structural-hypothesis report adoption V1

This surface commits one fully verified reingestion publication as a versioned
local report capsule.  The exact status is
`ADOPTED_AS_LOCAL_REPORT_VERSION_NOT_PLANNED`: it records a local version but
does not write an ambient or global current pointer and does not plan a next
task.

V1 has no benchmark import, `run_one`, network, remote host, credential,
account, scheduler, shell, reingestion, task planning, materialization,
authorization, execution, retry, resume, paper promotion, or Package1
operation.  Adoption is local mechanics, not external authority, currentness,
readiness, scientific confirmation, or scientific refutation.

## Full-chain input boundary

Adoption requires the committed `PUBLISHED_NOT_ADOPTED` publication, the
original raw evidence CSV, and the original completed single-task attempt.  It
also requires the hypothesis, executor, runtime, publisher, and adoption
contract paths plus the original base manifest and materializer asset root.
The core securely reads and verifies those inputs; the runner deliberately
does not pre-read contracts or derive expected digests from the trees being
checked.

The caller must independently retain and supply all of these upstream
observations:

- source evidence digest;
- plan digest;
- authorization digest;
- execution receipt digest;
- execution journal-head digest;
- pre-execution attempt-binding digest;
- publication digest;
- reingestion digest;
- output report-body digest;
- output audit-head digest;
- output evidence digest.
- raw SHA-256 of `publication.json`;
- raw SHA-256 of `combined_rows.json`;
- raw SHA-256 of `output_report.json`;
- raw SHA-256 of `reingestion_receipt.json`.

These local SHA-256 observations are integrity anchors, not signatures.  They
provide no same-user rewrite defense if the expected values are recomputed
from the directory under verification.

## Adopt one version without planning

All CLI paths must be absolute.  `--adoption-id` is a local mechanics label,
not an identity, signature, or free-form path.  It must equal the direct-child
name selected beneath
`$XDG_STATE_HOME/kg-op/structural-hypothesis-report-adoption/v1/`.

```bash
python3 runners/run_structural_hypothesis_report_adoption.py adopt \
  --publication-root /home/user/.local/state/kg-op/structural-hypothesis-reingestion/v1/publication-0001 \
  --adoption-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_report_adoption_v1.json \
  --adoption-root /home/user/.local/state/kg-op/structural-hypothesis-report-adoption/v1/adoption-0001 \
  --adoption-id adoption-0001 \
  --base-evidence-csv /absolute/path/to/base-evidence.csv \
  --attempt-root /home/user/.local/state/kg-op/structural-hypothesis-execution/v1/attempt-0001 \
  --hypothesis-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_loop_v1.json \
  --executor-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_executor_v1.json \
  --runtime-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_single_task_runtime_v1.json \
  --publisher-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_reingestion_publisher_v1.json \
  --base-manifest /absolute/path/to/SC-OLH-KG/performance/manifests/v18b_exactkg_mcdiag.json \
  --asset-root /absolute/path/to/SC-OLH-KG/performance/task_inputs/structural_hypothesis_materializer_v1 \
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
  --expected-output-evidence-digest sha256:... \
  --expected-publication-marker-raw-sha256 sha256:... \
  --expected-combined-rows-raw-sha256 sha256:... \
  --expected-output-report-raw-sha256 sha256:... \
  --expected-reingestion-receipt-raw-sha256 sha256:... \
  --confirm-local-report-adoption
```

On success, stdout is exactly one canonical compact JSON object containing
`status`, `adoption_root`, `adoption_digest`, `publication_digest`,
`reingestion_digest`, `output_report_body_digest`, `output_audit_head`,
`output_evidence_digest`, and `planning_status`.  Save that line outside both
the publication and adoption trees.  Human-readable status is written only to
stderr.

The adopted evidence retains the publisher boundary: the base CSV is the raw
47-column string-valued input snapshot, the successful receipt row has 27
fields with native JSON types, and `combined_rows.json` remains the sole
authoritative replay evidence.  Adoption does not create a merged
authoritative CSV or reinterpret the evidence.

## Snapshot, commit marker, and no-clobber boundary

`--adoption-root` must be entirely absent.  An existing file, directory, empty
directory, symlink, FIFO, or broken link is rejected.  V1 never overwrites or
reuses an adoption root.

The core creates mode-0700 directories and copies the exact verified
publication raw leaves into the nested `publication/` snapshot with mode 0600.
It writes `adoption_contract.json` and publishes `adoption.json` last as the
atomic no-clobber commit marker.  No `current.json`, plan, bundle,
authorization, checkpoint, or next-task artifact is created outside the
publication snapshot.

A crash before `adoption.json` leaves `INCOMPLETE_NOT_ADOPTED`.  That root
cannot be verified as adopted and cannot be retried or reused; a caller must
choose a new root and ID.  A committed adoption is never modified in place.

## Verify against the live full chain

Verification is read-only and remains tied to the original publication, raw
CSV, completed attempt, contracts, manifest, assets, and every independent
upstream anchor.  It additionally requires the adoption digest saved from the
adopt command; it never reconstructs that expected value solely from the
locally rewriteable adoption directory.

```bash
python3 runners/run_structural_hypothesis_report_adoption.py verify \
  --publication-root /home/user/.local/state/kg-op/structural-hypothesis-reingestion/v1/publication-0001 \
  --adoption-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_report_adoption_v1.json \
  --adoption-root /home/user/.local/state/kg-op/structural-hypothesis-report-adoption/v1/adoption-0001 \
  --adoption-id adoption-0001 \
  --base-evidence-csv /absolute/path/to/base-evidence.csv \
  --attempt-root /home/user/.local/state/kg-op/structural-hypothesis-execution/v1/attempt-0001 \
  --hypothesis-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_loop_v1.json \
  --executor-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_executor_v1.json \
  --runtime-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_single_task_runtime_v1.json \
  --publisher-contract /absolute/path/to/SC-OLH-KG/performance/manifests/structural_hypothesis_reingestion_publisher_v1.json \
  --base-manifest /absolute/path/to/SC-OLH-KG/performance/manifests/v18b_exactkg_mcdiag.json \
  --asset-root /absolute/path/to/SC-OLH-KG/performance/task_inputs/structural_hypothesis_materializer_v1 \
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
  --expected-output-evidence-digest sha256:... \
  --expected-publication-marker-raw-sha256 sha256:... \
  --expected-combined-rows-raw-sha256 sha256:... \
  --expected-output-report-raw-sha256 sha256:... \
  --expected-reingestion-receipt-raw-sha256 sha256:... \
  --expected-adoption-digest sha256:...
```

Successful verification reports
`VERIFIED_ADOPTED_AS_LOCAL_REPORT_VERSION_NOT_PLANNED`, writes nothing to
stdout, and does not change either tree.  Argument, path, no-clobber, I/O,
contract, chain, layout, or digest failures return two, leave stdout empty,
and print a short error to stderr without a traceback.

## Claim and testing boundary

Adoption means only that one exact, independently anchored publication was
copied into a committed local report-version capsule.  The adopted report may
retain evidence gaps.  It is not global current, externally verified,
scientifically confirmed or refuted, ready for Package1, a next-task
admission, or a paper promotion.  A later planning gate must separately verify
this adoption and explicitly authorize any planning or execution.

The mandatory clean-checkout KAT synthesizes its 47-field base CSV, source
report, materialized first task, completed fake-only attempt, reingestion
publication, and adoption beneath a native temporary state root.  It never
depends on ignored historical evidence and never calls the real benchmark or
`run_one`.  Optional historical smoke cannot satisfy this gate.

# Structural-hypothesis recursive report advance V1

This stage consumes exactly one **completed, successor-bound** local task and
commits a new immutable report-version capsule.  It is the first recursive
transition in the local structural-hypothesis path:

```text
adopted report (1,351 rows; 29 pending)
  + exact first-pending successor-bound successful row
  -> immutable report version (1,352 rows; 28 pending)
```

The committed status is
`ADVANCED_AS_IMMUTABLE_LOCAL_REPORT_VERSION_NOT_CURRENT_NOT_PLANNED`.
The capsule also records `current_status=NOT_CURRENT` and
`planning_status=NOT_PLANNED`.  The transition neither changes a current
pointer nor admits, materializes, authorizes, or executes another task.

## Entry points

- Contract:
  `performance/manifests/structural_hypothesis_recursive_report_advance_v1.json`
- Core:
  `performance/structural_hypothesis_recursive_report_advance.py`
- CLI:
  `runners/run_structural_hypothesis_recursive_report_advance.py`

The core exposes:

```python
advance_recursive_report_version(...)
verify_recursive_report_advance(...)
```

Both functions require the complete frozen source chain.  The advance call
also requires `confirm_immutable_local_report_advance=True`; this confirmation
means only that the caller intends to commit a named, local, non-current and
non-planned version.

## Admission boundary

V1 accepts only a chain for which all of the following are true:

1. The source adoption fully verifies against its publication and original
   execution chain.
2. The successor fully verifies as the deterministic materialization of that
   exact adoption and its exact first pending cell.
3. The completed attempt is bound to that successor through the independently
   supplied provenance-binding, authorization, attempt, receipt, and journal
   head digests.
4. The authorization ID is exactly
   `successor-bound-v1:<provenance-binding-hex>`.
5. The receipt contains one authorized result, that result is successful, and
   it is the exact first pending task.  Failed, detached, duplicate, reordered,
   or non-first-pending evidence is rejected.
6. Reingestion reproduces a typed 1,352-row evidence array and a verified
   report with 28 pending cells.  Native JSON booleans, integers, and numbers
   are preserved.

Every caller-retained anchor is mandatory.  A matching file name or a locally
plausible digest is not enough.

## CLI

The command uses absolute paths.  Contract, manifest, and asset paths default
to the checked-in V1 artifacts but may be supplied explicitly.

```bash
python3 runners/run_structural_hypothesis_recursive_report_advance.py advance \
  --publication-root /absolute/state/structural-hypothesis-reingestion/v1/source \
  --adoption-root /absolute/state/structural-hypothesis-report-adoption/v1/source \
  --adoption-id source \
  --successor-root /absolute/state/structural-hypothesis-adopted-successor/v1/successor \
  --successor-id successor \
  --base-evidence-csv /absolute/input/base.csv \
  --source-attempt-root /absolute/state/structural-hypothesis-execution/v1/source \
  --completed-attempt-root /absolute/state/structural-hypothesis-execution/v1/successor \
  --advance-root /absolute/state/structural-hypothesis-recursive-report-advance/v1/advance \
  --advance-id advance \
  --expected-adoption-digest sha256:... \
  --expected-pending-evidence-digest sha256:... \
  --expected-first-pending-projection-digest sha256:... \
  --expected-successor-digest sha256:... \
  --expected-bundle-digest sha256:... \
  --expected-plan-digest sha256:... \
  --task-id task:... \
  --expected-task-digest sha256:... \
  --expected-provenance-binding-digest sha256:... \
  --expected-authorization-digest sha256:... \
  --expected-attempt-digest sha256:... \
  --expected-execution-receipt-digest sha256:... \
  --expected-execution-journal-head-digest sha256:... \
  --confirm-immutable-local-report-advance
```

Successful `advance` and `verify` calls write exactly one canonical JSON
summary line to stdout.  `verify` additionally requires these five anchors:

```text
--expected-advance-digest
--expected-reingestion-digest
--expected-output-report-body-digest
--expected-output-audit-head
--expected-output-evidence-digest
```

Verification is read-only and reports
`VERIFIED_ADVANCED_AS_IMMUTABLE_LOCAL_REPORT_VERSION_NOT_CURRENT_NOT_PLANNED`.

## Immutable capsule

The root must be a fresh direct child of
`$XDG_STATE_HOME/kg-op/structural-hypothesis-recursive-report-advance/v1`.
The exact committed layout is:

```text
<advance-id>/
  advance_contract.json
  source/
    adoption.json
    successor.json
    execution/
      attempt.json
      authorization.json
      receipt.json
      journal/
        0002_COMPLETED.json
  combined_rows.json
  output_report.json
  reingestion_receipt.json
  advance.json
```

Directories are mode `0700`; files are mode `0600`.  Symlinks, FIFOs,
hard-linked leaves, unexpected names, unsafe parent modes, and generation
changes are rejected.  Artifact publication is no-clobber and `advance.json`
is written last.  A fault before that final write leaves an incomplete root,
which is not an advance and must never be reused.

## Nonclaims and isolation

The capsule is local mechanics evidence, not external authority, a signature,
a scientific confirmation or refutation, a paper promotion, or permission for
another execution.  V1 performs no network, scheduler, credential, shell,
benchmark, or `run_one` operation and creates no automatic successor.

Full verification still requires the original adoption, successor, and
completed-attempt chain plus independently retained expected digests.  Local
digests do not defend against a same-user rewrite that also replaces those
independent expected values.

The implementation and its tests are additive.  The Operations Research paper
baseline, manuscript, historical results, and benchmark implementation are not
written by this stage; runtime state is confined to the named XDG state root.

## Tests

The KAT builds a self-contained fake-only chain.  It starts with a synthetic
1,350-row CSV, executes seed 0 and seed 1 only through an in-memory fake
callable and fake preflight, then proves the real report replay transition
`1351/29 -> 1352/28`.  It never imports or invokes the native benchmark or
`run_one`.  Negative coverage includes anchor mismatch, receipt substitution,
duplicate/non-first-pending evidence, source mutation, path and mode attacks,
hard links, FIFOs, generation swaps, no-clobber behavior, output tampering, and
commit-last failure.

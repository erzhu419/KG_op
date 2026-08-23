# Structural-hypothesis execution lifecycle V1

This V1 is a local, offline mechanics layer between the structural-hypothesis
report and the existing `run_one(task)` benchmark entry point.  It makes the
next experimental work explicit without pretending that a proposal ran.

## Four distinct stages

1. **Proposed**: a verified hypothesis report is converted into a deterministic
   ordered list of missing evidence cells.  A proposal is not authorization.
2. **Authorized**: a separate local artifact names the exact plan digest and an
   exact subset of task IDs together with their digests.  A local digest is an
   integrity mechanism, not
   a signature, identity, credential, or external authority decision.
3. **Executed**: only an explicitly injected `run_one(task)` callback may
   receive authorized task payloads.  Its large native result is projected by
   the existing aggregate reader before it can become an evidence row.
   Authorization is not evidence that the callback was invoked, and a callback
   or projection failure is not a scientific refutation.
4. **Reingested**: successful rows are validated, merged with the original
   evidence, and passed back through the versioned hypothesis verifier.
   Execution alone does not change a hypothesis verdict.

The four stages are intentionally represented by different artifacts and
digests.  Code must not infer a later stage from an earlier one.

## Current real-execution boundary

The current report has 30 deduplicated `full` cells in native order: the three
held-out domains from the hypothesis contract, each with seeds 0 through 9.
The V1 executor contract binds those cells to the existing
`performance.benchmark_lodo_meta_prior.run_one` ABI:

```text
{"args": <complete runtime mapping>, "heldout": <domain>,
 "line": "lodo", "seed": <integer>}
```

That ABI needs far more than the aggregate report contains.  In particular,
the repository does not currently contain the three exact source-informed
design artifacts and a complete, digest-bound local task template for this
historical matrix.  Reconstructing missing values from current defaults would
silently change the experiment.

Accordingly, V1 also freezes the nonclaim
`no_exact_historical_runtime_reconstruction`: neither the aggregate rows nor
this plan prove that the exact historical runtime can be reconstructed.  A
future template assembled from approximate, current, or inferred defaults
must not be relabeled as that historical runtime.

The executor contract pins the semantic canonical digest of the source
hypothesis contract.  The plan builder reloads that contract, verifies the
report's internal body commitment and audit chain, and binds their observed
digests into the plan.  Those report hashes are not externally anchored: a
party that can replace the whole report can recompute them.  Callers needing a
stronger provenance boundary must separately pin the expected report digest.

Therefore the checked-in V1 contract fixes
`real_task_template = NOT_IMPLEMENTED`.  Building a plan without an injected
materializer produces `AWAITING_TASK_TEMPLATE` and
`BLOCKED_NO_TASK_TEMPLATE` cells.  Such a plan is useful and verifiable, but it
is **not ready** for real execution and cannot be authorized.

The library API accepts an injected materializer so the four-stage mechanics
can be tested locally.  That injection is not proof that its `args` mapping is
the exact historical runtime template.  The checked CLI never accepts a
materializer and can only emit the blocked proposal.

## Plan-only CLI

Build a proposal from an existing report:

```bash
python3 runners/run_structural_hypothesis_plan.py plan \
  --report /absolute/path/to/hypothesis-report.json \
  --hypothesis-contract performance/manifests/structural_hypothesis_loop_v1.json \
  --out /absolute/path/to/execution-plan.json
```

Verify a saved plan against the checked-in executor contract:

```bash
python3 runners/run_structural_hypothesis_plan.py verify \
  --plan /absolute/path/to/execution-plan.json \
  --report /absolute/path/to/hypothesis-report.json
```

Both commands reject duplicate JSON keys.  Verification requires the exact
source report so that a self-consistent plan cannot silently omit pending
cells.  `plan` writes to stdout unless
`--out` is supplied; an output file is written atomically and cannot overwrite
or hard-link any input.  The runner never imports the benchmark module and
has no execution subcommand.

## Hard gate for the future local materializer

Real local execution remains a separate change.  It must supply, in full:

- the exact three source-informed designs and their native content;
- a complete raw `args` mapping for each cell, not an opaque task ID or digest;
- deterministic task validation against all frozen executor inputs;
- an explicit authorization artifact bound to the exact plan and task digests;
- receipts whose successful rows match the authorized profile, domain, seed,
  line, and frozen evidence scope before reingestion;
- full-chain receipt verification against the exact plan and authorization,
  followed by verification of the distinct reingestion artifact.

No network, scheduler, shell, remote account, credential, or external runtime
is part of this V1.  Plan creation makes no currentness, readiness, execution,
scientific-support, paper-promotion, or external-authority claim.

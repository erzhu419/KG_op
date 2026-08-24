# Shadow Interval/Multi-Q Recompetition Adapter V1

## Boundary

This strictly additive V1 exact-replays the seven preceding shadow slices and
adapts one verified post-restriction adjudication into a bounded seed for a
*future* interval/multi-Q theory-operation competition.  It preserves the
selected theory-state bytes.  The required upstream exact replay includes the
frozen chain's verification and scoring mechanics; the adapter stage itself
consumes no new evidence, performs no scoring or qualification, constructs no
candidate, runs no new competition, and makes no adoption decision.

The adapter resolves only an interface mismatch.  The frozen competition V1
accepts a finite point table with a single operational Q entry, whereas the
verified source and restricted shadows are finite interval tables with the
fixed two-Q registry.  An emitted seed therefore requires a separately
specified competition V2 consumer.  After replaying the frozen V1 source, this
slice makes no new V1 recompetition call and does not implement V2.

The slice consists only of:

- `performance/shadow_interval_multi_q_recompetition_adapter.py`;
- `performance/manifests/shadow_interval_multi_q_recompetition_adapter_v1.json`;
- `runners/run_shadow_interval_multi_q_recompetition_adapter.py`;
- `tests/test_shadow_interval_multi_q_recompetition_adapter.py`;
- this document.

The pure core writes nothing.  The runner's explicit `--out` is the only
optional write surface.  No benchmark, `run_one`, scheduler, network service,
probe acquisition, language invention, evidence acquisition, or current-state
path is called.  The frozen Operations Research worktree, manuscript,
baselines, and claims are outside this slice and remain untouched.

## Exact replay and applicability

The frozen contract is `shadow_interval_multi_q_recompetition_adapter_v1`,
with canonical digest:

```text
sha256:16d2a30873e3f8b2e56fe5d7ac272140eb83dbcb441d8d80a892c4028f28f029
```

The no-artifact retained-restricted known-answer report is frozen at:

```text
sha256:af3930e87940b92e5d7c4e6b081ae6c1dd9896fec5841be76cea64a6dd9c309d
```

The core exact-replays competition, shadow transition, probe qualification,
external-review packet construction, failure-boundary probing, robust interval
restriction, and post-restriction adjudication.  It verifies every independently
supplied digest and canonical report before resolving any seed.  Rehashed
upstream tampering, state substitution, Q/V drift, or source-lineage drift fails
closed.

The public builder takes exactly twenty-one positional JSON objects, in this
order:

```text
competition input, contract, report
transition contract, report
qualification input, contract, report
review contract, report
failure-boundary input, contract, report
restriction input, contract, report
adjudication input, contract, report
adapter input, contract
```

The runner requires all twenty-one paths to be absolute, existing,
non-symlink regular files with distinct resolved paths and distinct inodes.  It
rejects all 210 possible input-pair aliases, including hard links.  Duplicate
JSON keys, non-finite JSON numbers, non-object roots, input/output aliases, and
forged artifact metadata fail closed.  Nested input-artifact receipts preserve
the preceding 2/4/7/9/12/15/18-file boundaries and bind all 21 adapter inputs.

## Public API

```python
validate_shadow_interval_multi_q_recompetition_adapter_contract(contract) -> dict

derive_shadow_interval_multi_q_recompetition_seed_id(
    *,
    adjudication_contract_digest,
    adjudication_report_digest,
    source_probe_expanded_state_digest,
    restricted_state_digest,
    adapter_contract,
) -> str

adapt_shadow_interval_multi_q_recompetition_seed(
    # the twenty-one positional inputs listed above,
    ...,
    # independent expected digests and nested artifact maps,
    input_artifacts=None,
)

verify_shadow_interval_multi_q_recompetition_adapter(
    # the same twenty-one inputs, then adapter_report,
    ...,
    expected_adapter_report_digest,
    expected_adapter_input_artifacts,
) -> dict
```

The result exposes only `disposition`, `report_digest`, `seed_emitted`,
`seed_kind`, and `to_dict()`.  It exposes no scoring, qualification, candidate,
language, adoption, promotion, current-pointer, rollback-execution, or
ambient-state mutation capability.  The public verifier exact-replays the full
report and compares canonical JSON.

## Adapter input and stable seed identity

The adapter input contains only the source-adjudication commitments and an
adapter request.  It contains no observation rows, `observed_value`, evaluator
epoch, threshold, score, candidate, or language payload.  The seed identifier
is content-addressed from the adjudication contract and report identities, the
source and restricted state identities, and the frozen adapter contract.  It is
not an evaluator epoch, a signature, or an external attestation.

The adapter accepts or consumes no new evidence.  All evidence and evaluator
decisions remain exactly those already bound into the verified adjudication
report.  In particular, this adapter cannot turn an incomplete-evidence or
incomparable-epoch adjudication into a numerical verdict.

## Five dispositions and exact seed bytes

Exactly five dispositions are available:

```text
EMITTED_RESTRICTED_SHADOW_RECOMPETITION_SEED
EMITTED_SOURCE_ROLLBACK_TARGET_RECOMPETITION_SEED
EMITTED_UNQUALIFIED_SOURCE_REPAIR_RECOMPETITION_SEED
RECOMPETITION_ADAPTER_NEEDS_NEW_POST_RESTRICTION_EVIDENCE
RECOMPETITION_ADAPTER_INCOMPARABLE_POST_RESTRICTION_EPOCH
```

The resolution is a total, fixed mapping from the five verified adjudication
dispositions:

- A qualified restricted shadow emits a seed whose `theory_state` bytes are
  exactly the verified restricted-shadow bytes.
- A source rollback target emits a seed whose `theory_state` bytes are exactly
  the verified source-shadow bytes.  Rollback execution remains
  `NOT_PERFORMED`.
- When both shadows failed, the adapter emits the exact source-shadow bytes only
  as an `UNQUALIFIED_SOURCE_REPAIR_BASE`.  The repair certificate records both
  failures; it is not qualification, acceptance, or a rollback result.
- A needs-evidence adjudication emits no seed.
- An incomparable-epoch adjudication emits no seed.

For every emitted case, the seed's state is canonical-byte-equal to its verified
source.  The adapter does not project an interval table into a point table,
delete a Q entry, synthesize a radius, alter scope, or rewrite object space.
For non-emitting cases, the report's `recompetition_seed` and
`recompetition_seed_digest` are null and no fallback is silently selected.

## Interface certificate

The report inventories the verified source and restricted states, then checks
the emitted state against the resolved source.  The certificate binds:

```text
state and model-class digests
finite_interval_table model kind
the fixed two-Q registry
the unchanged V registry
object space, scope IDs, and removable-feature IDs
center predictions and radius grouping
the exact expected radius-group keys
finite non-negative source and restricted radii
canonical seed/source byte equality
```

These are finite representation checks, not a generic model-translation
theorem.  The adapter adds no rigid-body, Markov, or independence assumption.
It does not infer scientific validity or preservation outside the frozen finite
tables.

The contract's `adapter_handoff_finite_interval_table_only`,
`adapter_handoff_fixed_two_probe_registry_only`,
`no_adapter_point_projection`, and `no_adapter_lossy_probe_projection`
nonclaims are explicitly scoped to this handoff stage.  They do not erase the
point/single-Q mechanics that the upstream exact replay truthfully records.

## Operation registry: syntax only

An emitted seed carries the frozen diagnostic operation registry required by
the future V2 interface.  Registry entries contain only `operation_id` and
`operation_kind`.  They name bounded operation kinds; they are not candidates,
scores, executable functions, policy choices, or authorizations.  This adapter
does not enumerate operation parameters or select a winner.

The seed also binds its model interface, alternate-state digests, and the
consumed-record exclusion ledger inherited from the adjudication.  These bytes
make a later consumer auditable; they do not make the missing V2 competition
exist.

## Competition V1 incompatibility and V2 handoff

Direct feedback into `theory_operation_competition_v1` is prohibited.  V1
requires `finite_point_table` and the singleton probe registry
`absolute_error_point_prediction`.  The seed deliberately preserves
`finite_interval_table` and both:

```text
absolute_error_point_prediction
normalized_signed_interval_boundary_margin
```

The adapter does not implement the required competition V2 core.  The report
therefore records that competition V1 compatibility is false, no new V1
recompetition invocation occurred after upstream replay, and an interval/multi-Q
competition V2 is required but not implemented.  The handoff is a typed seed,
not a claim that recompetition ran or that an iterative
\(H_t\rightarrow H_{t+1}\) loop is complete.

The adapter is also before any last-resort language or predicate expansion.
It adds no predicate, relation, scope, quotient, probe, violation functional,
or model candidate.  A later language-expansion slice would require a separate
contract and fresh authority; nothing here certifies or executes it.

## Evidence, lifecycle, and authority boundaries

The adapter accepts no fresh observations, derives no evaluator epoch, computes
no loss, and updates no record eligibility.  It copies the verified lifecycle
exclusion commitments needed by a future consumer.  Competition,
qualification, failure-boundary, restriction, and adjudication records remain
ineligible for future scoring; a future competition must obtain new,
unconsumed evidence under its own contract.  Cross-epoch pooling remains
forbidden.

Logical selective erasure remains an auditable record-policy statement.  No
physical deletion or retention-system attestation occurs.  The adapter does not
mutate either source shadow, execute rollback, invalidate a child, decide
adoption eligibility, adopt, promote, write a current pointer, or authorize an
\(H_t\rightarrow H_{t+1}\) transition.  External data, evaluator, retention,
and adoption authorities remain absent.

## CLI and write boundary

The runner emits one canonical JSON line to stdout.  With an explicit absolute
`--out`, it writes the same bytes atomically.  It refuses relative inputs,
symlink inputs, duplicate resolved paths, duplicate inodes, any output alias to
an input, malformed JSON, duplicate keys, and non-finite JSON constants.  Its
twenty-one artifact records bind the exact bytes, absolute resolved path, and
SHA-256 digest of every input.

The core and runner import no execution, scheduler, or network client and have
no benchmark or `run_one` call surface.  Tests use only bounded synthetic
known-answer artifacts inherited from the prior KAT chain; green tests establish
mechanics and fail-closed behavior, not a scientific result or production
execution claim.

## Threat model and nonclaims

V1 defends against stale upstream reports, forged digests, duplicate JSON keys,
artifact substitution, path/inode aliases, source-state substitution, Q/V or
radius-interface drift, altered exact seed bytes, adjudication-disposition
reinterpretation, report rehashing after semantic tampering, accidental adapter
seed submission to competition V1, and authority escalation.

V1 does not defend against colluding callers that fabricate every supplied
artifact and independently expected digest, a compromised Python runtime or
filesystem, semantic misrepresentation outside the report, or absent external
data/evaluator/adoption authorities.  `report_digest` is a content digest, not
a signature.  The slice is a bounded, adapter-only interface certificate—not a
recompetition execution, not a complete autonomous theory-evolution loop, and
not evidence that any shadow theory is scientifically valid.

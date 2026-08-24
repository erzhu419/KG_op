# Shadow Post-Restriction Adjudication V1

## Boundary

This strictly additive V1 exact-replays the six preceding shadow slices and
adjudicates one materialized robust interval restriction using a new local
evaluator epoch and fresh caller-supplied static holdout and stress rows.  The
same observations are scored against the restricted shadow object and its
frozen probe-expanded source rollback target.  The report may retain the
restricted shadow object, select the source shadow object as a rollback
*target*, or route the unresolved pair to a bounded adapter requirement.

Target selection is not execution.  This slice does not mutate either shadow
state, invalidate a child, execute rollback, adopt, promote, write a current
pointer, or authorize an \(H_t\rightarrow H_{t+1}\) transition.  It provides no
external evaluator, data, retention, or adoption attestation.

The slice consists only of:

- `performance/shadow_post_restriction_adjudication.py`;
- `performance/manifests/shadow_post_restriction_adjudication_v1.json`;
- `runners/run_shadow_post_restriction_adjudication.py`;
- `tests/test_shadow_post_restriction_adjudication.py`;
- this document.

The pure core writes nothing.  The runner's explicit `--out` is the only
optional write surface.  No benchmark, `run_one`, scheduler, network service,
probe-acquisition path, or language-invention path is called.  The frozen
Operations Research worktree, manuscript, baselines, and claims are outside
this slice and remain untouched.

## Applicability and exact replay

The frozen contract is `shadow_post_restriction_adjudication_v1`, with
canonical digest:

```text
sha256:dc870252207785f1d2ff4768dbf7d9fcedba7e0580554b0e647df82757d461ef
```

The core first exact-replays competition, shadow transition, probe
qualification, external-review packet construction, failure-boundary probing,
and robust interval restriction.  Adjudication is applicable only to a report
whose disposition is:

```text
MATERIALIZED_SHADOW_ROBUST_INTERVAL_RESTRICTION
```

and whose restricted state, source state, strict-subset certificate, Q and V
registries, lineage, and rollback boundary all verify.  Forged upstream
commitments fail closed; an upstream non-materialized outcome is not silently
reinterpreted as a new adjudication case.

The public builder takes exactly eighteen positional JSON objects, in this
order:

```text
competition input, contract, report
transition contract, report
qualification input, contract, report
review contract, report
failure-boundary input, contract, report
restriction input, contract, report
adjudication input, contract
```

Every independently supplied digest is checked.  The runner requires all
eighteen paths to be absolute, existing, non-symlink regular files with
distinct resolved paths and distinct inodes.  It rejects all 153 possible
input-pair aliases, including hard links.  Duplicate JSON keys, non-finite JSON
numbers, non-object roots, input/output aliases, and forged artifact metadata
fail closed.

## Public API

```python
validate_shadow_post_restriction_adjudication_contract(contract) -> dict

derive_shadow_post_restriction_adjudication_epoch(
    *,
    restriction_contract_digest,
    restriction_report_digest,
    restricted_shadow_theory_state_digest,
    source_probe_expanded_shadow_theory_state_digest,
    fixed_anchor,
    adjudication_contract,
) -> str

qualify_adjudicate_and_route_shadow_post_restriction(
    # the eighteen positional inputs listed above,
    ...,
    # independent expected digests and nested artifact maps,
    input_artifacts=None,
)

verify_shadow_post_restriction_adjudication(
    # the same eighteen inputs, then adjudication_report,
    ...,
    expected_adjudication_report_digest,
    expected_adjudication_input_artifacts,
) -> dict
```

The result exposes only `disposition`, `report_digest`,
`restricted_shadow_qualified`, `rollback_target_selected`, `cycle_route`, and
`to_dict()`.  It exposes no adoption, promotion, current-pointer, rollback
execution, language invention, or ambient-state mutation capability.  The
public verifier exact-replays the full report and compares canonical JSON.

## Fresh input and local epoch

The adjudication input contains exactly:

```text
schema_version
adjudication_id
source_restriction
evaluator
prior_record_exclusion
evidence
```

`source_restriction` pins the restriction contract and report, the restricted
state digest, and the source probe-expanded state digest.  `evaluator` binds the
exact derived epoch and the frozen anchor.  `prior_record_exclusion` pins the
observation-ID digests for competition, qualification, failure-boundary, and
restriction generations.  `evidence` contains exactly `holdout` and `stress`.
Each row contains exactly:

```text
observation_id
evaluator_epoch
fixed_anchor
scope_id
context
observed_value
```

Each split must cover the exact full Cartesian product of every frozen scope
and parent context once.  All observation IDs must be unique across the two
splits and disjoint from all four consumed generations.  Missing or duplicate
scope/context pairs yield a non-numerical needs-evidence route.  Epoch or
anchor mismatch yields a non-numerical incomparable route.  Duplicate IDs,
unregistered pairs, non-finite values, or inconsistent lineage fail closed.

The local epoch is content-addressed from both source-state identities, the
restriction contract and report identities, the fixed anchor, and the frozen
adjudication contract.  It does not depend on observed values or input row
order.  This identity prevents accidental cross-epoch pooling; it is not
external evaluator attestation.

## Exact probe replay on both shadow targets

V1 invents neither a probe nor a violation functional.  It requires the
restricted and source Q and V registries to be canonical-byte-identical, then
recomputes the complete registered probe family against both finite interval
tables using the same holdout and stress rows.  It does not copy a prior pass
bit and does not treat the restriction slice's calibration records as fresh
qualification evidence.

For each split and each target, the report binds:

```text
row_count
mean_absolute_center_error
max_absolute_center_error
min_normalized_signed_interval_boundary_margin
boundary_violation_count
boundary_violation_rate
mean_normalized_exceedance
max_normalized_exceedance
counterexample_observation_ids
```

The prediction scale and thresholds are frozen by the contract and verified
upstream state.  The absolute-center-error Q is recomputed and reported as a
measurement and equality invariant; V1 does not impose a separate data-tuned
point-error gate.  Fresh qualification is determined only by the frozen zero
interval-boundary-violation-rate gate on both complete splits.  Boundary
classification uses the raw finite comparison `error > radius`; normalized
margin is a metric and cannot turn a raw violation into a pass through floating
point underflow.  A fresh pass is not a global preservation, utility, safety,
generalization, or scientific-validity claim.

## Interval tradeoff and monotonicity certificate

The source and restricted centers, radius grouping, group keys, Q registry,
and V registry are rechecked byte-for-byte.  For every registered
scope/context pair, the restricted radius is no greater than the source radius
and at least one radius is strictly smaller.  On identical observations and
the same positive normalization scale, the core checks the finite mechanical
consequences:

```text
absolute center errors are identical
restricted signed margins are <= source signed margins
restricted normalized exceedances are >= source exceedances
restricted boundary-violation count is >= source count
source counterexamples are a subset of restricted counterexamples
```

Thus a restriction can trade coverage for a smaller admitted model class; it
cannot mechanically improve interval containment on the same rows.  These
checks certify only the bound finite interval representation.  They do not
assume rigid bodies, Markov structure, independence, or any generic
restriction theorem.

## Five dispositions and bounded routing

Exactly five dispositions are available:

```text
POST_RESTRICTION_QUALIFIED_RETAIN_RESTRICTED_SHADOW
POST_RESTRICTION_FAILED_SOURCE_SHADOW_ROLLBACK_TARGET_SELECTED
POST_RESTRICTION_AND_SOURCE_FAILED_RECOMPETITION_ADAPTER_REQUIRED
POST_RESTRICTION_NEEDS_NEW_EVIDENCE
POST_RESTRICTION_INCOMPARABLE_EVALUATOR_EPOCH
```

- If the restricted target passes, it remains the selected shadow target.
- If the restricted target fails but the exact source target passes, the source
  is selected as a rollback target.  Rollback execution remains `NOT_PERFORMED`.
- If both targets fail, neither is accepted.  The next route is a
  recompetition adapter requirement.
- Incomplete evidence and incomparable epochs produce no numerical verdict and
  select no target.

In every case adoption eligibility remains externally undetermined, adoption
is `NOT_ADOPTED_SHADOW_ONLY`, promotion is `NOT_PROMOTED`, and current status is
`NOT_CURRENT`.

## Adapter-before-language boundary

The original frozen competition slice accepts a finite point table with one
operational Q entry.  A post-restriction object is a finite interval table with
a multi-Q registry.  Those input grammars are deliberately incompatible: this
slice does not feed an interval/multi-Q state directly back into the first
competition core and does not pretend that a direct iterative cycle already
exists.

When both restricted and source targets fail, V1 emits only an adapter-required
route.  A future adapter must preserve the verified source/restricted
identities, scopes, Q and V registries, consumed-record exclusions, and
authority boundary while creating a separately auditable input for renewed
competition.  Only if a bounded adapter/recompetition route is later specified
and exhausted may a separate last-resort language or predicate expansion be
considered.  No predicate, relation, theory language, candidate, or probe is
invented here.

## Record lifecycle and logical erasure

The adjudication holdout and stress rows are consumed by this report.  All
competition, qualification, failure-boundary, restriction, and adjudication
records are ineligible for future scoring.  Any later adjudication or
competition requires new, unconsumed evidence under a new appropriate epoch;
cross-epoch pooling remains forbidden.  This is auditable logical selective
erasure only.  No physical deletion or retention-system attestation occurs.

## CLI and write boundary

The runner emits one canonical JSON line to stdout.  With an explicit absolute
`--out`, it writes the same bytes atomically.  It refuses relative inputs,
symlink inputs, duplicate resolved paths, duplicate inodes, any output alias to
an input, malformed JSON, duplicate keys, and non-finite JSON constants.  Its
eighteen nested artifact records bind the exact bytes, absolute resolved path,
and SHA-256 digest of every input.

The core and runner import no execution, scheduler, or network client and have
no benchmark or `run_one` call surface.  Tests use only bounded synthetic
known-answer rows; green tests establish mechanics and fail-closed behavior,
not a scientific result or production execution claim.

## Threat model and nonclaims

V1 defends against stale upstream reports, forged digests, duplicate JSON keys,
non-finite rows, artifact substitution, path/inode aliases, observation-ID
reuse, partial Cartesian coverage, cross-epoch pooling, mismatched anchors,
copied pass bits, Q/V drift, source/restricted target substitution, directionally
impossible monotonicity claims, report rehashing after semantic tampering, and
accidental authority escalation.

V1 does not defend against colluding callers that fabricate every supplied
artifact and independently expected digest, a compromised Python runtime or
filesystem, semantic misrepresentation outside the report, or absent external
data/evaluator/adoption authorities.  `report_digest` is a content digest, not
a signature.  The slice is a bounded shadow adjudication and routing mechanism,
not a complete autonomous theory-evolution loop and not evidence that the
restricted or source theory is scientifically valid.

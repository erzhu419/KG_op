# Theory-operation competition V1

## Claim boundary

V1 is an additive, local, shadow-only decision kernel.  It turns one finite
theory state and three explicitly separated evidence splits into competing
robustification and idealization candidates.  It does not execute an
experiment, call `run_one`, contact a scheduler or network service, change an
ambient current pointer or parent state, or promote a scientific or paper
claim.  The explicit CLI `--out` path is the only optional write surface.

The module is deliberately independent of the later structural-hypothesis
campaign.  It uses only the Python standard library and can be added directly
to the frozen Operations Research baseline together with its contract, runner,
tests, and this document.  No existing algorithm, manuscript, result, or
contract is edited.

## Entry points

- Contract:
  `performance/manifests/theory_operation_competition_v1.json`
- Core: `performance/theory_operation_competition.py`
- CLI: `runners/run_theory_operation_competition.py`

The public core API is:

```python
validate_contract(contract) -> dict

synthesize_theory_operation_candidates(
    theory_state,
    discovery_rows,
    contract,
) -> dict

run_theory_operation_competition(
    input_payload,
    contract,
    *,
    input_artifacts=None,
) -> CompetitionResult

verify_theory_operation_competition(
    input_payload,
    contract,
    report,
    *,
    expected_contract_digest,
    expected_report_digest,
    expected_input_artifacts,
) -> dict
```

The module also exports `SCHEMA_VERSION`, `ContractValidationError`,
`EvidenceValidationError`, `OperationKind`, `CompetitionDisposition`, and
`CompetitionResult`.  `canonical_json_bytes` is the sole JSON commitment
encoding.  Verification replays the computation from the exact input and
contract and requires the caller's independent expected contract digest and
expected input-artifact map (or an explicit `None`); it does not accept a
contract ID or artifact claims from the report as self-authenticating.  Nor
does it trust a candidate, metric, disposition, digest, or audit event merely
because it is present in the supplied report.
The direct synthesis entry point independently rejects discovery rows whose
evaluator epoch or fixed anchor differs from the supplied theory state.

## Exact input boundary

The input is one JSON object with exactly `schema_version`, `case_id`,
`theory_state`, and `evidence`.

The theory state has exactly:

```text
theory_id
task_id
evaluator_epoch
fixed_anchor
object_space
  feature_ids
  contexts
model_class
  kind = finite_point_table
  predictions
probe_ids
violation_functionals
scope_ids
removable_feature_ids
```

Each context is an exact mapping from every registered feature ID to a
non-null finite JSON scalar (string, integer, float, or Boolean).  The finite
point table contains one finite numeric prediction for every registered
context.  IDs and contexts are unique; duplicate, missing, extra, non-finite,
unsupported, or Boolean-valued numeric fields fail closed.

`evidence` has exactly `discovery`, `validation`, and `stress`.  Each row has
exactly:

```text
observation_id
evaluator_epoch
fixed_anchor
scope_id
context
observed_value
```

V1 accepts exactly one `absolute_error` violation functional and exactly one
implemented probe, `absolute_error_point_prediction`.  Arbitrary registered
probe names are rejected rather than copied into a preservation claim.  The
functional's finite, non-negative threshold defines the parent point model's
baseline error band; it is not used as a floor for discovery-derived robust
radii.

Candidate synthesis reads only the theory state and discovery rows.  Validation
and stress rows are held out from candidate construction and are used only for
evaluation.  This is a semantic boundary, not just a field label: replaying the
same discovery data with changed validation or stress values may change the
verdict but must not change the synthesized candidate commitments.

All three splits must use the theory's exact evaluator epoch and fixed anchor.
Mixed epochs or anchors are reported as
`INCOMPARABLE_EVALUATOR_EPOCH`; their numbers are never pooled.  A discovery
mismatch produces no synthesized candidates or numeric diagnostics.  A
mismatch in any split sets held-out baseline metrics to `null` and prevents
candidate evaluation.
Every registered scope must also occur in discovery, validation, and stress
before an operation may be selected.  Missing scope coverage yields
`NEEDS_EVIDENCE` and an exact missing-scope map.

## Typed theory operations

V1 registers five operation kinds:

1. model-class expansion;
2. model-class restriction;
3. quotient representation;
4. observation/probe expansion;
5. language expansion.

The executable V1 candidate grammar automatically enumerates two bounded
families from the supplied numeric data:

- **Robustification.** Discovery residuals induce global, per-scope, and
  per-context uncertainty intervals.  Their radii are computed from observed
  discovery residuals; they are not accepted from candidate-authored JSON.
- **Idealization.** Every non-empty subset of removable features defines a
  quotient.  Parent table entries in the same quotient class are averaged, and
  an explicit recovery map binds each original context to its quotient class.

Thus all five kinds are registered, but only `expand` (robust intervals) and
`quotient` (idealizations) produce executable V1 competition candidates.
`restrict`, `probe`, and `language` remain typed diagnostic or next-step
vocabulary and are not executed by this shadow kernel.

Language expansion is last in the diagnosis order and is never selected merely
because the two executable candidate families lack evidence.  V1 does not
invent a predicate or relation.

## Idealization contract

Every quotient candidate carries all nine required fields:

```text
deleted_degrees_of_freedom
preserved_observables
applicable_scale
applicable_task
approximation_error
failure_boundary
counterexample
computational_or_sample_complexity_gain
full_model_recovery_method
```

The approximation error and state reduction are recomputed from the parent
finite table.  `full_model_recovery_method` commits the complete parent point
table and its theory-state digest, plus a `quotient_fiber_map`; a lossy
quotient therefore cannot pretend that copying the quotient value reconstructs
the deleted degrees of freedom.  Merely supplying natural-language contract
text cannot make an idealization admissible.

## Evidence-driven diagnosis and competition

The report records an ordered, metric-bearing diagnosis.  It checks parameter
re-estimation, repeat/noise behavior, anomaly and structure, scope effects,
finite mixture evidence, quotient candidates, robustification candidates, and
the need for a new probe before leaving language expansion as the last resort.
The finite one-dimensional two-cluster mixture diagnostic and every candidate
metric are computed from the input values.

Held-out operation evaluation and selection are allowed only when
`reestimate`, `noise`, `scope`, and `mixture` all return
`EXCLUDED_BY_DISCOVERY`.  If any earlier explanation remains viable or still
needs repeated evidence, the disposition is `NEEDS_EVIDENCE` and the quotient
and robustification stages are `BLOCKED_BY_EARLIER_DIAGNOSIS`.

Validation and stress evaluation recompute, as applicable:

- baseline and candidate mean absolute error;
- maximum probe divergence;
- interval coverage, tail coverage, and safety coverage;
- normalized robust radius and its score cost;
- quotient state reduction and approximation cost.

Here `safety_rate` is only scalar containment in the constructed interval.  It
is not a claim of domain safety.  V1 also records that it has no selective
erasure implementation.

The versioned contract supplies all minimum row counts, thresholds, weights,
tie policy, and mandatory nonclaims.  A candidate cannot alter those values.
The string contract ID identifies the schema/mechanics family, not a unique
parameterization; exact verification therefore requires the independently
retained canonical contract digest.
The public disposition is exactly one of:

```text
SELECT_ROBUSTIFICATION
SELECT_IDEALIZATION
NEEDS_EVIDENCE
INCOMPARABLE_EVALUATOR_EPOCH
```

Selection means only that one shadow candidate wins this bounded, versioned
competition.  It does not mutate the parent theory, adopt the child, authorize
execution, establish external authority, or promote a paper result.

## CLI

All paths are absolute.  With no `--out`, the runner prints one compact,
key-sorted canonical JSON report followed by a newline:

```bash
python3 runners/run_theory_operation_competition.py \
  --input /absolute/path/case.json \
  --contract /absolute/path/theory_operation_competition_v1.json
```

`--out /absolute/path/report.json` atomically writes the same canonical bytes
and still emits them on stdout.  The runner rejects duplicate JSON keys,
non-finite constants, non-object roots, relative or symlinked inputs, and an
output that aliases, overwrites, or is hard-linked to either input.  Input or
contract failure returns 2 with no report on stdout.

## Test boundary

The KATs use self-contained numerical point tables rather than precomputed
verdict flags.  One dataset has a stable structured tail missed by the parent
point prediction and must select robustification.  A second dataset makes a
registered feature observationally irrelevant and must select idealization.
Negative coverage includes cross-epoch evidence, insufficient splits, held-out
leakage into synthesis, report tampering, order invariance, duplicate and
non-finite input, language-last behavior, output/input aliasing, and byte-level
checks that the existing benchmark source is not changed or imported.

These tests prove deterministic mechanics on their bounded fixtures.  They do
not prove autonomous theory evolution in arbitrary domains, meaningful tail
coverage outside the supplied scope, or the scientific validity of a selected
operation.

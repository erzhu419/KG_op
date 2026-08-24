# Shadow Interval/Multi-Q Post-Transition Qualification V1

## Boundary

This strictly additive V1 exact-replays the complete interval/multi-Q chain,
then evaluates one exact materialized V10 child on caller-supplied fresh
holdout and stress rows under a new content-derived evaluator epoch. It is the
fresh qualification gate deliberately withheld by Shadow Interval/Multi-Q
Theory Transition V1. It does not synthesize or reselect a candidate,
rematerialize a child, acquire evidence, execute an environment or external
probe acquisition, reuse source rows for child scoring, decide adoption
eligibility, adopt, promote, or write a current pointer. It only evaluates the
fixed Qs deterministically on supplied rows. The pure core writes nothing. The
runner's explicit `--out` is its only optional write surface.

The slice consists only of:

- `performance/shadow_interval_multi_q_post_transition_qualification.py`;
- `performance/manifests/shadow_interval_multi_q_post_transition_qualification_v1.json`;
- `runners/run_shadow_interval_multi_q_post_transition_qualification.py`;
- `tests/test_shadow_interval_multi_q_post_transition_qualification.py`;
- this document.

It is a bypass-style extension over the additive Meta-prior chain. It does not
modify any earlier file, benchmark, archived result, or the separate
Operations Research paper baseline or claim surface.

The frozen contract digest is
`sha256:9a52bb7ea3f4ce0f0cb16c5a5a296d284a85f4bfbdb2ee671e8c865bb2d3d493`.
It accepts only the frozen V9 competition contract
`sha256:4c30c0b1a2cdec92ab1676e98677b620907bb9652bff1ce71865fce9d45ccd1e`
and V10 transition contract
`sha256:b1a5f1761c2cafcae24f37f22810178074b9fc7800b6d73bdfd631be3b1df86d`.

## Public API

```python
validate_shadow_interval_multi_q_post_transition_qualification_contract(
    contract
) -> dict

derive_shadow_interval_multi_q_post_transition_qualification_id(
    *,
    source_transition_contract_digest,
    source_transition_report_digest,
    parent_theory_state_digest,
    child_theory_state_digest,
    operation_kind,
    transition_kind,
    post_transition_qualification_contract,
) -> str

derive_shadow_interval_multi_q_post_transition_evaluator_epoch(
    *,
    qualification_id,
    source_interval_competition_contract_digest,
    source_interval_competition_report_digest,
    fixed_anchor,
    post_transition_qualification_contract,
) -> str

qualify_shadow_interval_multi_q_post_transition(
    # the twenty-seven exact upstream objects through the V10 report,
    post_transition_qualification_input,
    post_transition_qualification_contract,
    *,
    # twenty-eight independent digest anchors,
    # all prior-layer expected artifact maps,
    input_artifacts=None,
) -> ShadowIntervalMultiQPostTransitionQualificationResult

verify_shadow_interval_multi_q_post_transition_qualification(
    # the same twenty-nine inputs,
    post_transition_qualification_report,
    *,
    # the same twenty-eight source anchors and artifact maps,
    expected_post_transition_qualification_report_digest,
    expected_post_transition_qualification_input_artifacts,
) -> dict
```

The builder has exactly twenty-nine positional inputs. The verifier has the
same inputs plus the qualification report, for thirty positional objects. The
core first invokes the public V10 transition verifier; no private shortcut can
authorize qualification. A self-consistent but forged transition or
competition report is not admissible.
The result exposes only `disposition`, `report_digest`, `qualified`, and
`to_dict()`.

## Exact input

The input schema is
`sc-olh-kg.shadow-interval-multi-q-post-transition-qualification-input/1`.
Its top-level keys are exactly:

```text
schema_version
qualification_id
source_transition
evaluator
source_evidence_exclusion
evidence
```

`source_transition` has exactly:

```text
transition_contract_digest
transition_report_digest
source_interval_competition_report_digest
disposition
operation_kind
transition_kind
selected_candidate_id
selected_candidate_family
parent_theory_state_digest
child_theory_state_digest
```

For one of the seven V10 nonmaterialization dispositions,
`qualification_id`, `evaluator`, `source_evidence_exclusion`, and `evidence`
must all be null. Such a source takes the strict not-applicable route; it
cannot submit synthetic evidence and cannot become qualified.

For a materialized child, `evaluator` has exactly `evaluator_epoch` and
`fixed_anchor`. `source_evidence_exclusion` has exactly:

```text
policy
five_prior_generation_observation_id_digests
v2_competition_evidence_digests
```

Those two digest structures must be byte-equal to the verified V10 record-
lifecycle bindings. They prove exclusion; they do not authorize source rows
for scoring. All source evidence remains audit-only and scoring-excluded. The
policy literal is exactly
`EXCLUDE_FIVE_PRIOR_GENERATIONS_AND_V2_DISCOVERY_VALIDATION_STRESS`.

`evidence` has exactly `holdout` and `stress`. Every row has exactly:

```text
observation_id
evaluator_epoch
fixed_anchor
scope_id
context
observed_value
```

Each split must contain exactly one row for every parent-context by registered-
scope cell. Quotient evidence still uses full parent contexts; projection to
the child context occurs only while evaluating the child. Duplicate IDs,
reuse of any of the six source-generation IDs, unregistered cells, non-finite
numbers, and malformed objects are hard errors. Only a registered-cell count
mismatch routes to needs-exact-fresh-evidence without numeric scoring.

## Fresh identities and six-generation isolation

The qualification ID commits exactly:

- this post-transition qualification contract digest;
- the source transition contract and report digests;
- parent and child theory-state digests;
- operation kind and transition kind.

The evaluator epoch then commits the qualification ID, source interval
competition contract and report digests, inherited fixed anchor, and exact
fixed two-Q registry. The derived epoch must differ from all six source
generations. The declared epoch and anchor and every fresh row's epoch and
anchor must match exactly. These identities are local content commitments,
not external evaluator or anchor attestation.

The six excluded generations are:

```text
competition
qualification
failure_boundary_probe
restriction
post_restriction_adjudication
interval_multi_q_competition_v2
```

Their records remain audit-only and scoring-excluded. Fresh holdout rows are
qualification-only; fresh stress rows are unique-child confirmation-only.
Cross-epoch pooling is forbidden; logical selective erasure is applied at the
scoring boundary, while physical erasure remains `NOT_PERFORMED`.

## Native two-Q interval scoring

The fixed probe registry is exactly:

```text
absolute_error_point_prediction
normalized_signed_interval_boundary_margin
```

For parent center `c`, radius `r`, observation `y`, and prediction scale `S`:

```text
S = max(
  numeric_epsilon,
  stable_mean(abs(parent_center)) + absolute_error_threshold,
)

Q1 = abs(y - c) / S
Q2 = (r - abs(y - c)) / S
```

The stable mean sorts finite inputs and uses `math.fsum / count`, with a
max-absolute-value scaled fallback only if ordinary finite arithmetic would
overflow or become non-finite. Any non-finite derived arithmetic is a hard
error and emits no report.

Raw boundary membership is authoritative and unnormalized:

```text
abs(y - c) > r
```

The normalized boundary exceedance
`max(0, abs(y-c)-r)/S` is report-only and never decides membership. Source
tail membership ranks raw parent boundary exceedance in source prediction
units; nonviolations contribute exactly zero. With
`k=max(1,ceil(n*source_tail_fraction))`, the cutoff is the kth descending raw
exceedance; all rows at or above the cutoff are included. Observation IDs
never break tail ties.

For each fresh split, the core recomputes parent and child:

- stable mean raw center error followed by one division by `S`;
- raw boundary coverage;
- coverage on the source-defined tail;
- stable mean raw radius followed by one division by `S`;
- stable mean raw positive boundary exceedance followed by one division by
  `S`.

It also recomputes Q1 divergence, Q2 divergence, their maximum, family-specific
context reduction or uniform contraction, and normalized radius expansion or
reduction. Every score component and the final qualification score are
dimensionless:

```text
score =
    normalized_center_mae_gain
  + 0.75 * raw_boundary_coverage_gain
  + 0.75 * source_tail_coverage_gain
  + context_reduction_fraction
  + 0.50 * uniform_contraction_fraction
  + 0.50 * normalized_radius_reduction
  - 0.75 * max_probe_divergence
  - 0.50 * normalized_radius_expansion
```

The tail cutoff alone remains in source prediction units.

## Holdout first, stress second

Fresh holdout is evaluated first. Its common gates require raw coverage and
source-tail coverage of at least `0.75`, normalized center-MAE increase at
most `0.05`, Q1 divergence at most `0.20`, Q2 divergence at most `1.0`, and
qualification score at least zero.

Family gates remain exact:

- expansion requires the strict V10 expansion certificate, at least `0.05`
  coverage-or-tail gain, and normalized radius increase at most `1.0`;
- restriction requires the strict V10 restriction certificate and zero fresh
  raw boundary violations;
- quotient requires the V10 global-envelope certificate, context reduction at
  least `0.20`, and child coverage and tail coverage not below the parent.

A holdout failure short-circuits stress. The report contains a frozen
`NOT_EVALUATED_HOLDOUT_FAILED` stress placeholder and no stress-derived metrics, score,
reranking, runner-up, or fallback. Stress is evaluated only after holdout
passes. Stress uses the common and family gates without a stress score. It can
confirm only this unique materialized child; it cannot switch children. The
path has no fallback candidate and performs no reranking.

## Six dispositions and precedence

The exact dispositions are:

```text
POST_TRANSITION_QUALIFICATION_NOT_APPLICABLE_NO_MATERIALIZED_CHILD
POST_TRANSITION_QUALIFICATION_NEEDS_EXACT_FRESH_EVIDENCE
POST_TRANSITION_QUALIFICATION_INCOMPARABLE_FRESH_EVALUATOR_EPOCH
POST_TRANSITION_QUALIFICATION_FAILED_FRESH_HOLDOUT
POST_TRANSITION_QUALIFICATION_FAILED_FRESH_STRESS_CONFIRMATION
QUALIFIED_FRESH_POST_TRANSITION_EVALUATOR_EPOCH
```

Precedence is strict:

```text
hard structural error
  -> no-child not applicable
  -> incomparable fresh epoch or anchor
  -> needs exact fresh evidence
  -> failed holdout, with stress not evaluated
  -> failed stress confirmation
  -> qualified
```

All three materialized families can reach the single qualified disposition;
the report preserves family, operation, and transition bindings. All seven V10
nonmaterialization routes collapse only to strict not-applicable and keep all
child-specific qualification surfaces null.

## Exact report

The report schema is
`sc-olh-kg.shadow-interval-multi-q-post-transition-qualification-report/1` and
has exactly these twenty-eight keys:

```text
schema_version
contract_id
contract_digest
qualification_input_digest
source_transition
qualification_id
parent_theory_state_digest
child_theory_state_digest
operation_kind
transition_kind
evaluator_definition
evaluator_binding
evidence_binding
selective_erasure_receipt
fixed_probe_registry
probe_results
disposition
qualification_binding
authority_boundary
adoption_eligibility
adoption_status
promotion_status
current_status
nonclaims
input_artifacts
audit_events
audit_head
report_digest
```

On the no-child route, qualification ID, parent/child digests, operation and
transition kinds, evaluator definition/binding, evidence binding, erasure
receipt, fixed probe registry, probe results, and qualification binding are
all null. The source receipt, disposition, authority boundary, nonclaims,
audit chain, and report digest remain present.

## Provenance closure and CLI

The twenty-eight independent digest anchors cover every contract, input, and
report after the initial competition input. Exact cumulative artifact maps
occur at layers `2/4/7/9/12/15/18/21/24/26/29`. The final 29-file receipt map
contains only byte count, absolute path, and raw-file SHA-256; observed values
are never embedded in artifact metadata.

```bash
python3 runners/run_shadow_interval_multi_q_post_transition_qualification.py \
  --competition-input /absolute/00.json \
  --competition-contract /absolute/01.json \
  --competition-report /absolute/02.json \
  --transition-contract /absolute/03.json \
  --transition-report /absolute/04.json \
  --qualification-input /absolute/05.json \
  --qualification-contract /absolute/06.json \
  --qualification-report /absolute/07.json \
  --review-contract /absolute/08.json \
  --review-report /absolute/09.json \
  --probe-input /absolute/10.json \
  --probe-contract /absolute/11.json \
  --probe-report /absolute/12.json \
  --restriction-input /absolute/13.json \
  --restriction-contract /absolute/14.json \
  --restriction-report /absolute/15.json \
  --adjudication-input /absolute/16.json \
  --adjudication-contract /absolute/17.json \
  --adjudication-report /absolute/18.json \
  --adapter-input /absolute/19.json \
  --adapter-contract /absolute/20.json \
  --adapter-report /absolute/21.json \
  --interval-competition-input /absolute/22.json \
  --interval-competition-contract /absolute/23.json \
  --interval-competition-report /absolute/24.json \
  --interval-transition-contract /absolute/25.json \
  --interval-transition-report /absolute/26.json \
  --post-transition-qualification-input /absolute/27.json \
  --post-transition-qualification-contract /absolute/28.json \
  --expected-competition-contract-digest sha256:... \
  # plus the remaining twenty-seven independent digest flags \
  --out /absolute/post-transition-qualification-report.json
```

All twenty-nine inputs must be absolute existing non-symlink JSON object files
with distinct normalized paths and distinct inodes. There are 406 unordered
input pairs, and both same-path and hard-link aliases are rejected for every
pair. Duplicate keys, non-finite constants, non-object roots, any digest or
artifact mismatch, a symlink, and output aliasing fail with exit 2 and no
report on stdout. Success emits canonical JSON plus one newline; optional
`--out` receives exactly the same bytes by atomic replacement.

## Authority and next gate

Every route retains:

```text
adoption_eligibility = NOT_DETERMINED_EXTERNAL_AUTHORITY_REQUIRED
adoption_status = NOT_ADOPTED_SHADOW_ONLY
promotion_status = NOT_PROMOTED
current_status = NOT_CURRENT
```

The core has no candidate selection, materialization, evidence acquisition,
environment execution, adoption-eligibility, adoption, promotion,
current-pointer, language-expansion, or parent/child/seed/ambient-write
authority. Qualification is not `H_t -> H_{t+1}` acceptance.

The report adds explicit dynamic authority facts. Public V10 replay is true on
every emitted route. Qualification applicability is false only on the no-child
route. Fresh evaluator derivation and evidence-structure validation are true
only when those stages are reached. Holdout performance is true only when
holdout arithmetic is executed; stress performance is true only after a
passing holdout; qualification success is true only for the qualified
disposition. Candidate synthesis/reselection/ranking/fallback,
materialization/rematerialization, external probe acquisition/environment
execution, adoption eligibility, adoption, promotion, current-pointer writes,
language invention, and parent/child/seed/ambient writes are always false.

This slice is not a complete autonomous theory-evolution loop. It establishes
only replayable local qualification of one exact child on one exact caller-
supplied fresh evidence contract. It makes no domain-safety, CVaR,
worst-case-safety, external provenance, scientific-validity, generalization,
or paper claim. Any adoption-eligibility or acceptance decision belongs to a
separate future additive gate with separate authority.

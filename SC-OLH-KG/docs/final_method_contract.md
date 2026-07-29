# Final Method Contract

## Frozen identity

The primary Operations Research track is frozen as

```text
source-only LODO structural proposal
  -> canonical SAASBO target adaptation
  -> method-independent V69 deployment verification.
```

The novel object is the transferable front end, not the BoTorch backend.  KG,
SC-V69, Thompson sampling, and Stacked GP remain backend ablations.  The
manifold and Transformer encoders remain representation-background rows.

The executable identifier is `or_transfer_frontend_saas_v1`, defined in
`performance/paper_method_contract.py`.  A result may use that identifier only
when it has:

- `risk_objective_atlas` with the `low_frequency_only` structural profile;
- 384 source simulator calls from two non-target domains;
- a frozen `n0=10` target design;
- canonical per-iteration SAASBO through target search call 13; and
- the V69 `80/128/128` independent verifier plus the paired objective
  incumbent guard.

Search and verification costs are different statistical stages and must never
be collapsed into one unlabeled evaluation budget.

## Information contract

The primary experiment is **descriptor-conditional LODO**, not domain-blind
meta-learning.  Before target outcomes are observed, the method may use:

- the held-out task-family identifier;
- policy dimension and integer bounds;
- an unlabeled policy/state exposure schema; and
- simulator input/output schemas.

It may not use target objectives, target constraint observations, an analytic
target optimum, target chance-margin labels, or terminal verification
responses during proposal construction, search, or shortlist construction.
Every baseline in the descriptor-conditional stratum receives the same
descriptors.

A separate domain-blind stress test removes the task-family identifier.  It is
reported separately rather than being used to relabel the primary evidence.

## Claim boundary

The completed factorial supports one primary empirical claim: the frozen
source proposal is the dominant cause of held-out feasible-basin coverage at
`d=1000, N_search=13`.  It does not support describing four structural
switches as coequal causes, nor does it establish KG as the strongest online
backend.

Cumulative HVD remains a registered risk-calibration contribution.  It enters
the headline method only if the same-proposal, same-backend causal gate shows
an incremental calibration, false-feasibility, certification, or regret
benefit.  Otherwise it remains a secondary risk-estimation and certification
component.

## Proposal coverage theorem

`SCOLHKG.Real.paper_frontend_transfer_coverage_and_certificate` now makes the
front-end claim quantitative.  PAC-Bayes transfer gives a held-out one-draw
feasible-mass lower bound

```text
p_lower = max(0, 1 - source_miss - domain_shift - complexity_radius),
```

where the complexity radius depends on the effective structural dimension,
the finite proposal library, source sample count, and confidence level.  For
`n0` IID frozen proposal draws, the probability of at least one feasible
initial policy is at least

```text
1 - (1 - p_lower)^n0.
```

`SCOLHKG.Measure.iid_at_least_one_hit_probability` proves the finite-product
identity in mathlib.  `performance/proposal_coverage.py` is the numerical
bridge.  A paper table must report every input, including a source-only
inner-LODO estimate or upper bound for `domain_shift`; an uncalibrated zero
shift is not permitted.

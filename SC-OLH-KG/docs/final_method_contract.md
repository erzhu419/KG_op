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

The deployed `risk_objective_atlas` is a deterministic finite atlas: its
target-seed replications share one frozen design fingerprint.  Therefore its
main implementation bridge is
`SCOLHKG.Real.paper_frontend_aligned_geometric_atlas_and_certificate`, not
an IID-draw or raw cross-domain threshold claim.  Let `psi` be the frozen
dimension-equivariant coordinate, `epsilon_cover` the maximin atlas radius on
the source-consensus plus universal structural support, `epsilon_psi` its
uniform approximation error to an ideal transferable coordinate,
`delta_domain` the ideal-coordinate source/target support shift, `L` a
chance-margin Lipschitz bound in `psi`, and `d_safe` the held-out safe-center
depth. If

```text
L * (epsilon_cover + delta_domain + 2 epsilon_psi) <= d_safe,
```

then the atlas contains a feasible policy. The atlas has at most `n0` members,
and nominal policy dimension does not enter this implication. No independence
is assumed between atlas members.

A normalized rank-alignment theorem was also formalized, then rejected as the
headline bridge because its source-only finite-sample bound was vacuous in all
three `d=1000` held-out domains. This negative audit is retained rather than
silently choosing a theorem after seeing target results.

`SCOLHKG.Real.paper_frontend_atlas_coverage_and_certificate` remains a raw
feasible-mass special case for domains where absolute feasibility is
transferable.  It is not used to explain the headline synthetic result.

For an optional genuinely randomized proposal backend,
`SCOLHKG.Real.paper_frontend_transfer_coverage_and_certificate` separately
proves the IID corollary

```text
1 - (1 - p_lower)^n0.
```

`SCOLHKG.Measure.iid_at_least_one_hit_probability` proves that finite-product
identity in mathlib, but the deterministic paper results do not use it.
The paper must report the source-only atlas radius and inner-LODO coordinate
error before target observations, then audit domain shift, safe depth, and a
justified Lipschitz upper bound only after the decision is frozen. A finite
candidate-library safe radius is diagnostic evidence, not a substitute for the
global Lipschitz condition.

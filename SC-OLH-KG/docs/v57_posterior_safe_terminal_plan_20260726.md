# V57 posterior-safe terminal closure

## Motivation

The completed `d=1000, N=13, n0=10` V56 gate passed its independent
two-head action-confirmation contract, but failed promotion. Inventory improved
in two seeds, while Queue lost one feasible recommendation and returned a
slightly worse feasible recommendation in another seed. In one Queue cell the
algorithm evaluated the better feasible policy but the final posterior Bayes
action did not retain it. The chance-feasibility certificate remained empty in
all 30 V51/V56 runs.

V57 addresses terminal posterior instability only. It does not weaken or
replace the canonical chance certificate and does not claim that an
uncertified initial incumbent is truly safe.

## One decision object

V57 preserves V56's independently confirmed evaluate-or-replicate action. Its
fantasy terminal value and final recommendation both use the sequential
posterior incumbent. A challenger replaces the incumbent only when the
covariance-free Cantelli lower bound for improvement in the same posterior
Bayes risk reaches `1-delta_switch`.

The run-level switch budget is `0.05`. Each of the `N-n0` charged online stages
receives

```text
delta_switch = 0.05 / max(N - n0, 1).
```

The V56 confirmation error event and the finite switch-horizon error event are
combined without an independence assumption in
`SCOLHKG.Measure.ConfirmationDominanceComposition`.

## Runtime changes

The gate also includes behavior-preserving execution changes:

- fantasy observations use copy-on-write instead of copying every historical
  response array;
- a non-legacy backend no longer computes a full-pool recommendation that is
  immediately overwritten by its observed-action terminal rule;
- Inventory and Queue oracle regret grids are evaluated in vectorized
  three-block coordinates and construct the `d`-dimensional policy only once.

The vectorized `d=1000` oracle objectives exactly match the registered values:
`0.27629444426` for Inventory and `0.2709903418875` for Queue.

## Gate

The preregistered gate uses three domains, seeds 0--4, `d=1000`, `N=13`,
`n0=10`, MC512, confirmation4096, and 72 exact workers. V56 remains the
performance reference. V57 is promotable only if it has:

- a valid no-oracle implementation/theory contract;
- no feasibility loss relative to V56;
- no paired feasible-regret loss;
- at least one strict feasible-regret gain or feasibility rescue;
- no false chance certificate;
- a recorded, horizon-spent posterior-switch budget.

Chance-certificate nonvacuity remains a separately reported obligation.

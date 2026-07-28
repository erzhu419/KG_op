# Controlled Heteroscedastic Gate Report

Date: 2026-07-27

## Main Finding

Making the independent deployment certificate non-vacuous did not degrade
optimization performance. The apparent historical regression came from
different source-proposal contracts, and the first controlled synthetic
failure came from missing candidate support.

When search trajectories were held exactly fixed, enlarging only the frozen
verification shortlist from the initial atlas to all observed policies raised
certified deployment from `11/200` to `130/200`. It rescued 119 unsafe primary
recommendations with zero feasibility losses and zero false independent
certificates.

## Gate Sequence

| Gate | Contract | Found feasible | Primary feasible | Certified deployment | False terminal cert. | Regret <= 0.01 |
|---|---|---:|---:|---:|---:|---:|
| V1 | `d=1000`, raw-random state inversion | `0/360` | `0/360` | `0/360` | `0` | `0` |
| V2 | label-free 8-corner latent support | `146/360` | `11/360` | `11/360` | `0` | `0` |
| V3 | V2 search, all-observed frozen shortlist | `146/200` | `11/200` | `130/200` | `0` | `0` |
| V4 | V3 plus 24 latent maximin candidates | `154/200` | `1/200` | `118/200` | `0` | `0` |
| V5 | `d=3`, `N=40`, no dimension confound | `198/200` | `0/200` | `123/200` | `0` | `0` |

V2 includes 160 Sobol negative-control rows that never use state candidates;
V3-V5 contain the 200 risk-TS/joint-VOI rows.

## Two Different Certificates

The online posterior certificate is model based. In V2 it certified 395
evaluated points, including 263 false-feasible points. In V5 the error reduced
but remained 88 false points among 315 declarations. It is not currently a
valid empirical safety claim at these budgets.

The independent terminal certificate uses fresh iid replications after the
search and never updates the optimizer. Across V2-V5 it issued no false
certificate. Its purpose is deployability, not better search: it can reject an
unsafe recommendation or certify a frozen alternative, but it cannot create a
near-optimal policy that was never evaluated.

## HVD Signal

The shared-factor scenario provides the clearest intended HVD result. Under
risk-aware TS in V5:

| Variance model | Median log-variance RMSE | Median upper coverage | Certified deployments |
|---|---:|---:|---:|
| pooled | `0.686` | `0.297` | `4/5` |
| factor | `0.580` | `0.904` | `5/5` |
| oracle | `0.000` | `1.000` | `5/5` |

This supports cumulative factor structure under shared shocks, but it is not a
universal win. Orthogonal/factor HVD remain weak in several smooth scenarios,
and replication/calibration ablations are still required.

## Why Optimum Recovery Still Fails

The current risk-aware Thompson backend uses a fixed soft penalty:

`sampled objective + rho * positive sampled chance margin`.

At `rho=5`, a low-objective unsafe policy can beat a feasible boundary policy.
The same issue appears in terminal Bayes-risk ranking. This explains why V5
nearly always evaluates a feasible point but chooses an unsafe primary, and why
certified fallback regret remains much worse than best-evaluated regret.

The next causal comparison should keep the representation, HVD, observations,
and independent verifier fixed while replacing only:

1. soft-penalty TS with feasible-first constrained posterior sampling;
2. maximin safe support with objective-ranked posterior-safe support.

Only after that gate should the synthetic matrix expand from 5 to 20 seeds.
High-dimensional scaling must remain a separate experiment so that candidate
coverage is not mistaken for HVD quality.

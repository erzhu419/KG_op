# V56 independent pilot/confirmation gate

## Why V55 did not close

The MC512 reference kept the substantive signal: all 15 selected actions had
positive Bayes-risk and certificate-deficit reductions. The nested MC32/high
discrepancy was not a valid finite-sample radius, however. It covered only
10/15 selected actions, and a global multiplier large enough to cover the two
near-zero discrepancies would make the guard vacuous.

## V56 contract

V56 keeps the V55 terminal functionals, action set, bounded per-fantasy gain,
and literal V51 fallback. It changes only the numerical admission contract:

1. MC512 factorized RQMC pilot scores all active evaluate/replicate actions.
2. The action maximizing the minimum pilot reduction is frozen.
3. A stage-keyed IID stream, disjoint from pilot/proposal/simulator streams,
   evaluates only that action.
4. Each bounded gain head uses the same frozen finite mixture of betting
   fractions. The null is nonpositive posterior expected reduction.
5. The run-level error budget is split over two heads and all `N-n0` stages.
6. The run-level budget is also split over every permitted batch look. Both
   heads must cross the resulting finite-look e-value threshold; otherwise
   V51 is executed.

This removes post-selection reuse: conditional on the pilot, confirmation is a
fresh test of one fixed action, so no action-count union penalty is required.

## Execution plan

The sentinel gate compares confirmation caps 2048 and 4096 on FactorShock,
Inventory, and Queue, five seeds each. Pilot work uses MC512. Confirmation is
batched in 512-sample increments and may stop after a successful finite-look
test. The threshold explicitly pays for every possible look, so early stopping
does not silently reuse a fixed-time guarantee. Large matrix tasks use 48
process workers; a focused interactive profile may use 72 workers.

The gate reports confirmation activation, samples to first crossing for each
head, false admissions, V51 fallback rate, final feasibility/regret, and
initialization/finalization substage timings. V56 is not promoted until the
run-level contract passes and final performance is noninferior to V51.

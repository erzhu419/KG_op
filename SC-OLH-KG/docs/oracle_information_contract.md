# Oracle and Offline-Data Information Contract

Date: 2026-07-14

## 1. Retrospective V32 classification

V32 does **not** query held-out target truth while selecting candidates,
updating its posterior, or recommending a policy. It therefore has no direct
target-oracle leakage.

It is nevertheless source-oracle aided. The promoted configuration used
`line=lodo_teacher` with two source domains, 64 random source records per
domain, and 32 teacher records per domain. Its source path used:

- analytic `true_sigma(x)` on random source records;
- analytic `true_outputs(x)` and `true_sigma(x)` on 64 teacher records;
- source-domain `initial_samples`, structured/state anchors, refinement grids,
  and axis hooks to choose those teacher records.

This is legitimate only when declared as a privileged transfer/oracle upper
bound. It is not an oracle-free LODO main result and is not directly fair to a
from-scratch online SOTA method given only the target budget.

## 2. Oracle-free challenger contract

The new `eta/psi` challenger uses:

- `source_observation_mode=replicated`;
- no source teacher records;
- no `true_outputs` or `true_sigma` calls;
- a frozen multiscale observable mean coordinate `eta` learned from ordinary
  noisy source simulator outputs;
- expert-specific cumulative-risk coordinate `psi=(A,N)` for HVD and
  certification;
- only the held-out target's charged `N` evaluations for target posterior
  updates.

With two source domains, 64 records per source, and three replications, this is
`2 * 64 * 3 = 384` offline source simulator calls plus `N=20` held-out target
calls. The run audit records both numbers. Promotion requires
`source_oracle_aided=false` and `admissible_mainline=true` for every seed.

### 2.1 Source-design identifiability

The original replicated challenger sampled all source policies uniformly. In
the three current LODO folds, the resulting aggregate source records contained
no truly feasible policy. A signed boundary model trained only on the unsafe
side cannot identify boundary location or orientation: distinct target margin
functions may agree on every observed source point and disagree elsewhere.

`source_design_mode=universal_mixture` repairs the experimental design rather
than relaxing certification. It evaluates one frozen, formula-free library of
constant, one-coordinate, ramp, and piecewise-low-frequency policies in every
source domain. The library does not call `true_outputs`, `true_sigma`, a target
hook, or an analytic boundary, and its source cost remains exactly
`2 * 64 * 3 = 384` simulator calls. Its role is to excite both sides of source
chance boundaries so the frozen source regression has information from which a
signed coordinate can be learned.

This is a declared structural prior, not a free oracle. Because its composition
was developed while inspecting the present synthetic benchmark family, final
paper evidence must also include domains held out from method development. The
random-design version remains a negative-control ablation.

### 2.2 Frozen source-consensus proposals

The universal archive may also determine a frozen proposal ordering. For each
source domain, observed replicated chance margins are converted to within-domain
percentile ranks. Profiles present in every source domain are scored by mean
rank, worst rank, and cross-source rank disagreement. This makes the ordering
invariant to positive affine changes of a domain's constraint scale and prevents
one numerically large domain from dominating.

The held-out target receives one formula-free low-frequency sentinel. If more
protected initial calls are allocated, the remaining policies span fixed rank
quantiles of the complete source shortlist, including both endpoints, instead
of taking a near-duplicate prefix. This is a frozen experimental design over
transfer uncertainty, not a target performance ranking. Later
universal-expert calls sample distinct shortlist members instead of repeatedly
returning the first library element. The complete shortlist is also included
in every sequential KG pool and final recommendation pool; acquisition still
decides whether any member receives a charged target evaluation. This uses
source simulator labels, so it
belongs only in the pretrained-transfer regime. It uses neither target labels
nor source/target analytic truth and is recorded as
`source_oracle_aided=false`.

The terminal suffix separately reserves the two lowest observed target chance
margins and uses `commit_before_switch`. When the rank-spanning source design
is active, this shortlist is restricted to its already charged members; an
unrelated one-sample random noise outlier cannot capture a protected
replication slot. A safety role is retained as an alias when the same policy is
also the minimum-Bayes-risk arm. Both reserved arms reach the declared
replication count before a posterior-only expert can consume the suffix. If no
protected member has been observed, the algorithm falls back to all charged
observations.

The two-arm comparison addresses the remaining one-sample reversal: one noisy
observation had ranked an unsafe Queue profile ahead of a nearby safe profile.
It does not guarantee correct ordering unless the true margin gap exceeds the
simultaneous estimation error; that finite condition and the suffix budget
contract are formalized in
`proof/SCOLHKG/Real/SourceConsensusCommit.lean`.

### 2.3 Oracle-free promotion result

The preregistered seeds `0..2` were used as the repair gate and seeds `3..6`
were run unchanged as a held-out continuation. Across the combined seven
seeds, the oracle-free challenger obtained:

| Held-out domain | True feasible | Median feasible regret | Mean violation |
|---|---:|---:|---:|
| FactorShock | 7/7 | 0.008250 | 0.000000 |
| Inventory | 7/7 | 0.011772 | 0.000000 |
| Queue | 6/7 | 0.002521 | 0.002719 |

All 21 rows report `source_oracle_aided=false`, all finalist diagnostics report
`target_oracle_used=false`, and all 21 rows satisfy the mainline admissibility
contract. The privileged V32 upper bound obtained `7/7`, `5/7`, and `3/7` on
the same domains and seeds. The oracle-free challenger is therefore promoted
for the pretrained-transfer regime, with its 384 source calls still reported
separately from the 20 target calls.

## 3. Fair comparison tables

The paper must report two noninterchangeable regimes:

1. **Pretrained transfer:** all transfer methods receive the same frozen source
   dataset; compare held-out target calls `N_target` and report source cost
   separately.
2. **Total-call accounting:** compare against from-scratch SOTA at
   `N_total = N_source + N_target`, or give those baselines the same source
   archive and an admissible warm-start adapter.

Source cost may be amortized only when the number of future target tasks is
stated. V32 remains in tables as a labelled privileged upper bound, never as
the oracle-free winner.

The source design must also be matched within each transfer regime. A method
receiving the universal low-frequency archive cannot be compared as if it had
received only random source records.

## 4. Diagnostic target truth

Post-run target truth used for regret, false-feasibility, pool-oracle ceilings,
or coordinate-identifiability audits is evaluation-only. Such rows are marked
`promotion_eligible=false`; they cannot alter candidates, posterior state,
configuration promotion, or the final reported mainline recommendation.

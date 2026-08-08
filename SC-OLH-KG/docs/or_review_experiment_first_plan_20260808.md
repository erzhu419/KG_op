# OR Review: Experiment-First Closure Plan

Status: implementation in progress. Manuscript edits are intentionally blocked
until the new evidence is frozen and audited.

## Energy Decision

The current GB_GBN experiment is retained. It is the only independent real-data
domain and an important negative control. It supports the ordered low-frequency
profile assumption relative to raw Sobol, but it does not establish that source
outcomes improve on a natural low-frequency grid. No post-hoc algorithm tuning
will be performed on GB_GBN.

Its exact-binomial statement is conditional on the fixed empirical distribution
over admissible verification-window start indices. The indices are sampled iid
with replacement; the underlying time-series windows may overlap. A future-
process or independent-calendar-window claim is therefore prohibited. A
nonoverlapping/block sensitivity audit will be reported separately.
It is a post-decision descriptive stability audit: it cannot alter the frozen
shortlist or certificate and does not claim a second iid guarantee.

Energy V2 will use a preregistered expanded OPSD market suite. Its purpose is to
test source learning across tasks, not to repair GB_GBN. Task descriptors,
source markets, target markets, profile library, thresholds, and analysis rules
must be frozen before target outcomes are opened.

## Method V2

`core/profile_atlas.py` is the normative implementation. It replaces the opaque
historical construction with four explicit steps:

1. Map each shared ordered profile to weighted cosine coefficients
   `c_0,...,c_K`, optionally followed by the complete diagonal square block.
2. Estimate source objective and chance-margin scores from ordinary replicated
   observations, then convert them to within-task percentile ranks.
3. Aggregate ranks with equal source weights or weights determined solely by an
   outcome-free target descriptor.
4. Choose the lowest weighted source-rank profile first and choose every
   remaining member by Gonzalez farthest-first in the declared augmented
   Euclidean coordinate `[standardized cosine coordinate, safety rank,
   objective rank]`.

There are no hidden scalarization lists, distance penalties, endpoint
replacement rules, target labels, or target oracle calls in V2. The generic DCT
maximin control uses the same profile library and cosine coordinate but no
source outcomes.

The primary V2 contract uses equal, domain-blind source weights. A development
audit found descriptor conditioning unstable with only two source tasks on
irregular grids. Descriptor conditioning is therefore an explicit ablation and
may not be enabled in the primary experiment unless its source-task-count gate
is passed before confirmatory freeze.

## Benchmark-Overfitting Audit

The confirmatory generator will sample independent task instances from the
following frozen regimes:

1. aligned random low-frequency profiles;
2. increasing active frequency count and effective rank;
3. source-target frequency-support shift;
4. hidden coordinate permutation and schema-aware permutation control;
5. irregular/nonuniform profile grids;
6. piecewise-smooth profiles;
7. sparse high-frequency active coordinates;
8. deliberately unrelated/misspecified targets.

Every task is an independent inferential unit. Simulation seeds within a task
measure repeatability and are not pooled as independent domain evidence.

Controls are source-scored atlas, generic DCT maximin, random low-frequency,
natural blockwise, raw Sobol, and a finite-library oracle upper bound. Endpoints
include feasible coverage, feasible-and-epsilon-optimal probability, penalized
loss, regret profiles, and all-in source/search/verification cost.

## Freeze Protocol

Development tasks may be used to debug and select V2. Confirmatory task seeds
are derived only after a code-and-manifest commit is pushed. The confirmatory
matrix is materialized once and is not repaired from its outcomes. Any later
method change creates a new version and a new untouched confirmatory seed.

## Statistical And Cost Contracts

The final verifier uses a frozen shortlist and candidate-wise exact binomial
tests with preallocated familywise error. Gaussian tolerance verification is a
separate model-dependent sensitivity analysis. Certificate power is reported
as a function of true safety probability and replication budget.

All comparisons report:

`source + initial design + adaptive search + safety verification + objective verification`.

The old `equal-total-cost` label is retired; historical tables that omit
verification are `equal pre-verification cost`. Archive amortization over `M`
targets and the break-even target count are reported explicitly.

## Promotion Gates

V2 is promoted only if all of the following hold on untouched tasks:

- it beats generic DCT maximin at the task level in at least one transferable
  regime without materially worse safety;
- it degrades gracefully under frequency shift and misspecification;
- schema-blind and permuted-channel results expose, rather than hide, the value
  of known channel semantics;
- its gain is not explained solely by nominal dimension or one fixed block
  decomposition;
- candidate-wise certification has zero contract violations and nonvacuous
  power at the registered budget.

If source scoring does not beat generic structured controls, the paper is
reframed around structured functional initial design and source scoring is
reported as a conditional extension rather than the central claim.

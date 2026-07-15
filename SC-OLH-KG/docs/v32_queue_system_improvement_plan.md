# V32 Queue System Improvement Plan

Status: analysis complete; no implementation or scheduler submission yet.

Date: 2026-07-12

## 1. Scope and audit contract

This document audits the promoted V32 LODO baseline, with special attention to
the frozen Queue holdout. Target truth is used only after every action,
posterior, candidate pool, and recommendation has been fixed. It is therefore
diagnostic evidence, not an admissible input to the optimizer.

The objective is not to add a Queue-specific rule. It is to identify a
domain-general defect that explains why V32 succeeds on FactorShock and much
of Inventory but fails to transfer stably to Queue.

The promoted baseline remains:

```text
v32_fixed_universe_adaptive_expert_race
N=20, n0=10, d=50
factor-HVD + theory certification + exact-MC KG
fixed terminal universe + 3 reserved finalist evaluations
```

## 2. Empirical facts

### 2.1 Domain-level outcomes

| Held-out domain | Initial design contains a truly feasible point | Final truly feasible | False certificate |
|---|---:|---:|---:|
| FactorShock | `7/7` | `7/7` | `0/7` |
| Inventory | `6/7` | `5/7` | `0/7` |
| Queue | `2/7` | `3/7` | `0/7` |

Queue is the first domain in this gate that primarily tests safe-set
discovery without an initial feasible anchor. FactorShock mostly tests
selection inside an already-safe region; Inventory does so in six of seven
seeds.

Queue V32 reports:

| Metric | Value |
|---|---:|
| truly feasible | `3/7` |
| median feasible regret | `0.00455` |
| mean constraint violation | `0.05767` |
| median true chance margin | `+0.01829` |
| posterior-certified recommendation | `0/7` |
| terminal pool contains a truly feasible point | `7/7` |

### 2.2 Candidate-support upper bound

Every final Queue pool contains the same class of low-dimensional safe anchor.
The post-run pool oracle, which is never exposed to the algorithm, would
obtain:

```text
true feasibility: 7/7
median feasible regret: 0.002367
```

Candidate support is therefore not the primary bottleneck. More manifold,
Transformer, random, or target-specific anchor channels cannot be justified
as the first repair.

### 2.3 Frozen Bayes-incumbent audit

At suffix entry, before the three reserved evaluations, V32 archives the
minimum posterior Bayes-risk action. Its truth-only audit is:

| Domain | Actual V32 feasible | Frozen Bayes incumbent feasible | Actual mean violation | Frozen-incumbent mean violation |
|---|---:|---:|---:|---:|
| FactorShock | `7/7` | `7/7` | `0.00000` | `0.00000` |
| Inventory | `5/7` | `2/7` | `0.00158` | `0.01982` |
| Queue | `3/7` | `5/7` | `0.05767` | `0.02517` |

This rules out both simplistic fixes:

- always trust the Bayes incumbent: improves Queue but destroys Inventory;
- always trust an expert challenger: helps Inventory but discards good Queue
  incumbents.

The allocation must be learned from the current posterior, not from a domain
label or a static gate.

## 3. Root-cause decomposition

### 3.1 The finalist race refreshes before completing evidence

The fixed universe itself is correct. The defect is the state machine over
that universe.

Current behavior:

1. archive `minimum_bayes_risk`;
2. choose the first expert safety nomination as the active target;
3. at the next stage, refresh the active target before checking whether the
   previous one reached the declared minimum replication count;
4. append every new expert nomination to the archive;
5. after the budget ends, ignore incomplete candidates.

Consequences on Queue:

- four of seven seeds spend the three reserved evaluations on three distinct
  targets;
- no target reaches two evaluations in those four seeds;
- the already-good Bayes incumbent receives zero finalist evaluations in
  Queue seeds 0, 3, and 5;
- Queue seeds 0 and 1 lose a truly feasible frozen Bayes incumbent.

### 3.2 An uncertified subset overrides the full posterior

When no theory-certified point exists, any completed finalist is allowed to
replace the posterior Bayes action. If no completed finalist has a nonpositive
empirical upper margin, the code still selects the minimum upper margin among
that small subset.

Across all V32 Gate-1 and Queue runs, every empirical finalist override has:

```text
replicated_finalist_empirical_certificate = false
```

Thus the final decision can be controlled by a two-replicate heuristic even
though the same observations have already updated the full GPR/HVD posterior.
This is behaviorally useful in some Inventory seeds, but it is not coherent
with the exact terminal Bayes value and is vulnerable to an engineering-stack
criticism.

### 3.3 Exact KG is disabled during the reserved suffix

All three suffix actions are forced finalist evaluations. Exact KG is skipped
with `forced_finalist_replication`, so the final 15% of the budget does not
optimize the declared terminal value. The fixed universe is precisely the
setting where a small finite-action terminal KG is easiest to compute.

### 3.4 Queue certification is far too conservative

No Queue seed has a posterior-certified action. At the selected actions, the
robust HVD aleatoric standard deviation is between `3.2x` and `8.3x` the true
synthetic standard deviation. Epistemic uncertainty is also large.

The source-learned cumulative HVD upper-scale multiplier is fixed at
`2.0906`. Target replications update a point estimate of the scale, but do not
shrink this source upper multiplier. Sparse repetitions therefore cannot turn
target evidence into a materially tighter certificate.

### 3.5 Queue exposes a different statistical task

The raw space has `101^50` integer policies, but the Queue generator depends
on four policy statistics. Its safe set is a compact pocket around a task
center, rather than a one-sided monotone region. V32 reaches this pocket in its
terminal pool but has difficulty ranking and certifying it.

The finite-expert decision mass also shows that the ordered cumulative expert
is not carrying the result:

| Domain | Mean decision mass on ordered cumulative expert |
|---|---:|
| FactorShock | `0.002` |
| Inventory | `0.038` |
| Queue | `0.007` |

The main factor-HVD remains active, but the transferred cumulative coordinate
expert is almost absent from final decisions. This must be acknowledged in
the paper narrative.

## 4. Primary repair: terminal replication KG

### 4.1 Replace the race, not the fixed universe

Let `U` be the V32 universe frozen before suffix labels. Let `F(D)` be a small
deduplicated finalist set containing:

- current minimum Bayes-risk action;
- minimum theory-margin action;
- minimum nominal expected-violation action;
- one action nominated by each live structural expert;
- one high-disagreement, low-Bayes-risk action.

Define the terminal loss using exactly the same robust posterior used by the
main algorithm:

```text
V_0(D) = min_{x in U} posterior_terminal_Bayes_loss(x | D).
```

For one remaining evaluation:

```text
KG_1(a; D) = V_0(D) - E[V_0(D union {(a,Y_a)}) | D].
```

For a suffix budget `b <= 3`:

```text
V_b(D) = min_{a in F(D)} E[V_{b-1}(D union {(a,Y_a)}) | D].
```

Every fantasy must clone and update:

- objective and constraint GPRs;
- factor-HVD and target scale posterior;
- finite task posterior;
- any admissible boundary-coordinate posterior.

Candidate/MC branches are independent and can use the existing process-fork
parallel backend.

### 4.2 Final recommendation contract

Replicated observations always update the common posterior. A separate
empirical finalist rule may override the full posterior only when its
anytime-valid upper chance margin is nonpositive.

If no finalist is empirically certified:

```text
recommend = argmin_{x in U} posterior_terminal_Bayes_loss(x | D_N)
```

An incomplete or empirically uncertified subset may not override this action.

### 4.3 Why this adapts across domains

Inventory and Queue do not require an explicit task classifier. The posterior
state already records:

- incumbent uncertainty;
- challenger uncertainty;
- expert disagreement;
- probability and magnitude of violation;
- value of reducing HVD scale uncertainty.

Terminal KG allocates repetitions according to these quantities. It can
validate an expert challenger on Inventory and preserve a strong Bayes
incumbent on Queue without knowing either domain name.

## 5. Secondary repair: target-updated HVD scale confidence sequence

Let the source HVD provide a shape `v_0(psi)` and a prior on a multiplicative
target scale `kappa_d`:

```text
v_d(psi) = kappa_d v_0(psi) + residual(psi).
```

Within-policy target repetitions produce scale evidence that is not
confounded with GPR mean error. The target posterior or confidence sequence
must output an anytime-valid upper scale `kappa_t_plus`:

```text
v_C_plus(psi) = kappa_t_plus v_0(psi)
                + residual_tail
                + representation_error.
```

Unlike the current fixed source multiplier, `kappa_t_plus` may shrink after
charged target evidence while retaining coverage under adaptive sampling.
This gives replication a second value-of-information channel: selecting the
best policy and learning the target noise scale.

## 6. Tertiary repair: transferable boundary geometry

Queue seeds 3 and 5 remain infeasible even under the frozen Bayes-incumbent
audit. They require a better low-dimensional chance-boundary model, not more
raw candidate coverage. That model is specified separately in
`transferable_chance_boundary_v4_analysis.md`.

## 7. Causal experiment matrix

The first experiment should change only the final suffix policy.

| Variant | Purpose | Mainline status |
|---|---|---|
| `V32` | frozen reference | baseline |
| `posterior_only` | remove uncertified empirical override | ablation |
| `commit_before_switch` | complete active arm before refresh | mechanism ablation |
| `terminal_kg_1step` | adaptive one-step terminal replication KG | primary challenger |
| `terminal_kg_depth3` | finite-horizon rollout over all 3 suffix steps | theory/quality challenger |

Run all five variants on the same `3 domains x 7 seeds x N=20` matrix. This is
`105` independent tasks and is small relative to available CPU capacity. The
variants must be predeclared; the primary promotion comparison is V32 versus
`terminal_kg_1step`, not the best post-hoc member of five variants.

Promotion gate:

- FactorShock remains `7/7` feasible;
- Inventory remains at least `5/7` feasible;
- Queue reaches at least `5/7` feasible;
- false certificates do not increase;
- median feasible regret is non-worse in each domain;
- no uncertified empirical override occurs;
- the fraction of suffix runs completing at least one informative arm
  increases materially over V32.

Only after this gate should the HVD scale-confidence challenger be added.

## 8. What not to optimize first

- Do not add Queue-specific anchors, coordinates, or thresholds.
- Do not weaken `beta_g` or the theory bound using Queue truth.
- Do not promote a raw empirical mean rule with two repetitions.
- Do not spend the first experiment on Transformer/manifold variants: pool
  support is already `7/7`.
- Do not tune dozens of full KG variants on the same seven Queue seeds and
  report the winner.

## 9. Decision

V32 has substantial headroom and a concrete, domain-general repair. The
fixed-universe idea should be retained, while its adaptive race and
uncertified final override should be replaced by terminal replication KG.
This is both the most likely immediate performance gain and the cleanest way
to reduce engineering-stack risk.

## 10. Implementation freeze

The five variants are now represented by two independent configuration axes:

| Experiment label | `finalist_replication_policy` | `finalist_empirical_override` |
|---|---|---|
| `v32` | `legacy` | `legacy` |
| `posterior_only` | `legacy` | `off` |
| `commit_before_switch` | `commit_before_switch` | `certified_only` |
| `terminal_kg_1step` | `terminal_kg_1step` | `certified_only` |
| `terminal_kg_depth3` | `terminal_kg_depth3` | `certified_only` |

The terminal variants freeze at most four posterior-nominated arms. Every
fantasy clones and updates the finite task posterior, both output GPRs, and
the factor-HVD state. `terminal_kg_depth3` evaluates the exact finite-horizon
Bellman recursion with horizon `min(3, remaining_budget)`; it is not a
receding-horizon method mislabeled as depth three. Root actions use the same
Linux `process_fork` backend as exact KG. For depth three, the second-stage
prefixes `(root action, root fantasy, second action)` are flattened into at
most 32 independent workers and reduced with the same nested expectation/min
operators. A serial-versus-fork regression test fixes numerical equivalence.

The final empirical override contract is now explicit. `certified_only`
permits an override only when the familywise upper chance margin is
nonpositive. The multiplicity correction includes the realized terminal arm
archive for every nonlegacy adaptive policy. An incomplete or uncertified arm
still updates the common posterior but cannot replace its Bayes action.

The preregistered submitter creates exactly `5 x 3 x 7 = 105` one-seed tasks,
all restricted to `node001-node006`. This section records implementation
completion only; no variant is promoted before the paired matrix finishes.

## 11. Preregistered result and decision

The complete `105/105` matrix finished without failed or retried cells. All
five variants had zero false-feasible recommendations. The preregistered
primary comparison nevertheless failed:

| Domain | V32 feasible | terminal KG 1-step feasible | V32 median feasible regret | terminal KG 1-step regret |
|---|---:|---:|---:|---:|
| FactorShock | 7/7 | 7/7 | 0.008250 | 0.008250 |
| Inventory | 5/7 | 2/7 | 0.005688 | 0.010378 |
| Queue | 3/7 | 5/7 | 0.004555 | 0.004555 |

The informative-completion count increased from 13 to 15, below the frozen
minimum gain of three. Thus the primary gate fails on both Inventory
feasibility/regret and the completion threshold. `terminal_kg_1step` is not
promoted.

Depth three produced the same feasible counts and median regrets as one-step
on all three domains. It raised Queue informative completions from four to
five, but did not repair Inventory and therefore remains a mechanism
ablation, not a post-hoc replacement challenger.

The implementation-only performance change passed its own equivalence check.
Across 18 flattened runs, the depth-three Bellman computation took 221.1 to
330.7 seconds (median 264.7); the three earlier root-only runs had median
1473.2 seconds. This is a 5.57x median speedup with unchanged horizon, fantasy
count, frozen arm set, posterior updates, and terminal value.

The next model change must explain why terminal value favors unsafe Inventory
arms while helping Queue. No V33 quality result becomes the new baseline.

## 12. Oracle-free source-consensus successor

The later repair did not deepen V33 rollout. It replaced source-oracle teacher
records with a replicated universal source archive, selected a rank-spanning
initial target design from source-only consensus, and committed the two safest
charged source-coverage finalists before posterior-only arms. The first three
seeds were the repair gate; seeds 3--6 were run unchanged.

The combined result is `7/7` FactorShock, `7/7` Inventory, and `6/7` Queue,
versus privileged V32's `7/7`, `5/7`, and `3/7`. All 21 successor rows are
source-oracle free and target-oracle free for selection. This successor is the
new pretrained-transfer baseline; V32 remains only a labelled privileged upper
bound.

## 13. Retrospective two-stage theory classification

The oracle-free successor is not renamed V32 and is not claimed to optimize one
exact-KG terminal value for all 20 calls. Its decision architecture is now
stated as `10` source-consensus initial-design calls, `7` adaptive search slots
configured for state-coupled exact KG, and `3` charged fixed-universe
heteroscedastic ranking-and-selection calls. A nonpositive
posterior or replication upper margin is certified only on its declared joint
coverage event. If neither certificate is available, minimum replicated upper
margin is an explicitly uncertified least-risk fallback.

The deterministic budget/status/regret layer is formalized in
`SCOLHKG.Real.TwoStageDecision`; finite-universe concentration and the
search/proposal/verification high-probability union are formalized in
`SCOLHKG.Measure.TwoStageDecision`. This reclassification preserves the
successor's observations and recommendations while removing the false claim
that its verification suffix is exact KG.

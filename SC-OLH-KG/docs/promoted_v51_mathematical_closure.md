# Promoted V51 Mathematical Closure

## Canonical object

The paper method is one hierarchical Bayesian decision model, not a sum of
independent acquisition modules:

```text
source archive
  -> frozen structural prior and source-informed proposal
  -> observable mean coordinate eta(x)
  -> cumulative risk coordinate psi(x) = (A(x), N(x))
  -> joint target posterior Pi_t(mean, cumulative HVD, task discrepancy)
  -> evaluate-or-replicate Bayes-risk VOI
  -> posterior Bayes action and separate conservative certificate.
```

For charged target observations `D_t`, let `O_t` be the evaluated policies and

```text
R_t(x) = E[f(x) | D_t] + rho E[(G(x))_+ | D_t],  x in O_t.
```

The cumulative chance margin inside `G` uses the same state-coupled HVD
posterior,

```text
Var(C(x) | T) = A(x)^T Lambda A(x)
              + N(x)^T B N(x)
              + N(x)^T omega + floor.
```

Certification is deliberately stricter than Bayes ranking:

```text
m_g(x) + sqrt(beta_g) s_g(x) + z_(1-alpha) sqrt(v_C^+(x)) <= tau.
```

The posterior-central variance may rank actions, but it cannot relax `v_C^+`.

## Decision closure

The terminal loss used by acquisition and recommendation is now identical.

- A new evaluation at `x` changes `O_t` to `O_t union {x}`.
- A replication is admissible only at `x in O_t` and preserves `O_t`.
- Both cost one target simulator call.
- Both clone and update the same GPR, source-discrepancy/task posterior, and
  cumulative HVD state.
- Current value, every fantasy value, and the final recommendation minimize the
  same posterior Bayes risk over the corresponding observed action universe.
- Finalist replication, certification recheck, empirical override, and
  posterior-dominance override are disabled in promoted V51.

The former implementation violated this contract: fantasy values minimized
over the whole terminal pool while the final recommendation was restricted to
evaluated policies. `SingleOLHKGAlgorithm._terminal_action_pool` removes that
mismatch.

## Finite approximation theorem

The implementation computes exact posterior refits on a finite action
shortlist, but estimates each expectation by Monte Carlo. If

- `epsilon_shortlist` bounds the exact-VOI loss from the full finite action pool
  to the shortlist, and
- `eta_MC` uniformly bounds MC error on the shortlist,

then the selected action is within

```text
epsilon_shortlist + 2 eta_MC
```

of the best action in the full finite pool. Consistent one-step posterior value
reductions telescope over the finite target budget. These statements and the
joint terminal certification implication are proved without `sorry` in
`proof/SCOLHKG/Real/PromotedV51Closure.lean`.

## What is closed

- Cumulative state-coupled variance decomposition and PSD shared-shock block.
- Separate observable coordinates for constraint mean and cumulative variance.
- Source-frozen prior/proposal and target-only online posterior updating.
- Unified evaluate-or-replicate posterior update and terminal Bayes loss.
- Shortlist plus MC decision error accounting.
- Finite-budget value telescoping.
- Conservative chance-certificate soundness when a certificate is issued.
- Python-to-Lean implementation map for every item above.

## What remains empirical

Formal soundness does not imply that the certificate is useful. In the
pre-repair promoted control (`d=1000, N=20, n0=10`, 60 runs), all 60 certified
sets were empty, even though 129 evaluated points were truly feasible. Thus:

- zero false certificates is currently vacuous;
- certificate coverage and recall are mandatory headline metrics;
- larger target/replication budgets must establish nonvacuity;
- the source-task PAC-Bayes exponential-moment slack must be calibrated using
  source-only held-out episodes;
- shortlist and two-sample antithetic MC errors require sensitivity/audit.

The numerical audit uses pair-indexed nested common random numbers and nested
posterior-only action sets on one shared post-`n0` state. Thus the measured MC
and shortlist discrepancies correspond directly to `eta_MC` and
`epsilon_shortlist`; they are not inferred by comparing already-diverged
sequential trajectories.

The mathematical model is closed conditionally. Paper-level empirical closure
requires a nonvacuous certification budget gate before the main experiment
matrix is frozen.

## Behavior gate result

The paired observed-terminal gate completed all 60 runs at `d=1000`, `N=20`,
and `n0=10`. It retained `60/60` true-feasible recommendations, zero adaptive
losses, and zero false certificates. Relative to the pre-repair V51 control it
won 17 runs, lost 7, and tied 36; adaptive improvements increased from 19 to
26. The repaired contract is therefore the promoted baseline. All 60 theory
certificate sets remained empty, so this promotion closes decision behavior,
not empirical certificate nonvacuity.

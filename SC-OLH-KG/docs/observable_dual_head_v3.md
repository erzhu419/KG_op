# Observable Dual-Head Coordinate V3

## Motivation

V2 replaced only the constraint-mean coordinate. Its main HVD path still
constructed `psi=(A,N)` from a source-learned policy-profile proxy. That did
not implement the intended shared-observation contract.

V3 starts both statistical heads from one target-observable record:

```text
e(x)       = observable state/action occupancy and trajectory statistics
phi_mu(x)  = h_mu(e(x))
psi_v(x)   = h_v(e(x)) = (A(x), N(x))

margin(x) = mu_g(phi_mu(x))
          + sqrt(beta_g) s_g(phi_mu(x))
          + z_alpha sqrt(v_C_plus(psi_v(x)))
          - tau.
```

The heads have independent source-fitted parameters. `h_mu` is aligned by
source chance-margin strata and supplies a coefficient prior for the
constraint mean. `h_v` is aligned by domain-standardized source log-variance
strata learned from ordinary replicated simulations. It maps the resulting
latent coordinate to nonnegative local exposures `A` and soft shared regimes
`N`; factor-HVD then fits

```text
floor + A^T Lambda A + N^T B N + N^T omega.
```

No target objective, target constraint, target true variance, or target risk
provider defines either coordinate. The old policy-proxy variance coordinate
remains available only as an ablation.

## Causal Offline Gate

The preregistered gate has four variants:

1. `legacy_policy_control`: policy-profile mean and variance proxies.
2. `observable_mean_only`: `h_mu(e)` with legacy variance proxy.
3. `observable_variance_only`: legacy mean proxy with `h_v(e)`.
4. `observable_dual_head`: independent `h_mu(e)` and `h_v(e)`.

Each variant uses the same frozen source archive, source-informed `n0`, target
seeds, 512-point truth-audit pool, and neutral Sobol backend. The 80 independent
cells cover FactorShock scale 0/4, Inventory, and Queue at `d=1000`,
`N=n0=10`, five seeds. Checkpoints are disabled.

The dual head advances only if it remains oracle-free, preserves mean ranking,
improves variance RMSE in at least one domain without material regression in
the others, preserves upper-variance coverage, restores FactorShock support,
and retains nonvacuous oracle-mean/oracle-variance certifiability. Sequential
experiments are not submitted before this gate passes.

## Five-Seed Result

The complete 80-task matrix finished without a failed or cancelled shard. Only
the 80 `result.json` files were synchronized; checkpoints and model artifacts
were disabled. The dual-head contract was oracle-free in every cell and both
heads did use the same observable exposure input.

| Domain | Legacy mean rank | Dual mean rank | Legacy mean MAE | Dual mean MAE | Mean-only variance RMSE | Dual variance RMSE | Dual upper coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FactorShock, shock 0 | 0.333 | 0.357 | 0.373 | 0.373 | 0.868 | 1.028 | 1.000 |
| FactorShock, shock 4 | 0.319 | 0.320 | 0.374 | 0.342 | 1.050 | 0.865 | 0.973 |
| Inventory | 0.366 | 0.282 | 0.296 | 0.760 | 2.227 | 1.173 | 1.000 |
| Queue | 0.414 | 0.397 | 0.239 | 0.267 | 1.182 | 0.804 | 1.000 |

The independently learned observable variance head is supported: against the
same observable mean head, it reduces variance RMSE for shock-4 FactorShock,
Inventory, and Queue, with no upper-coverage regression. It is not promoted as
a complete optimizer because two independent failures remain:

1. The observable mean head does not transfer semantically to Inventory. Its
   median absolute error rises from `0.296` to `0.760`.
2. Oracle mean and oracle aleatoric variance still certify zero candidates in
   every domain. Every seed is classified as
   `epistemic_or_safety_depth`: the best feasible point is shallower than the
   constraint-model epistemic radius.

The dense, target-agnostic pool contains true-feasible candidates in all five
seeds of every domain, so candidate support is no longer the blocker. V3 is a
positive causal result for `h_v(e)` but a failed end-to-end gate. No sequential
tasks are submitted and V3 is not promoted. The next model change must address
domain-semantic alignment of the mean coordinate and source-to-target
epistemic calibration; further tuning of the HVD variance head is not justified
by this gate.

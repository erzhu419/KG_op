# V59 Fixed-Policy Gaussian Verification Gate

## Purpose

V51 has strong feasible-regret performance, while the transferred posterior
certificate remains empty. V58 showed that certificate-directed actions can
reduce the margin without crossing zero. V59 tests whether this is a search
failure or a small-sample certification limit.

## Registered Protocol

1. Reproduce the V51 search exactly with `N_search=13`, `n0=10`,
   `exact_mc_samples=8`, and the same frozen source-informed archive.
2. Freeze the terminal recommendation before observing any verification
   sample.
3. Draw `R=48` iid replications from a separate seed stream at that one fixed
   policy.
4. Do not add those samples to GPR, HVD, expert weights, candidate generation,
   or terminal recommendation.
5. Compute

   ```text
   mu_U = y_bar + t_(1-delta_mu,n-1) S / sqrt(n)
   sigma_U = S sqrt((n-1) / chi2_(delta_sigma,n-1))
   ```

   and certify only when

   ```text
   mu_U + z_(1-alpha) sigma_U <= tau.
   ```

6. Report `13` search calls, `48` verification calls, and `61` total target
   simulator calls. This is a certification-budget experiment, not an
   equal-total-budget optimization comparison.

The per-policy error level is `delta=0.05`, split equally between mean and
scale coverage. The iid Gaussian assumption is explicit and audited through
the problem contract.

## Gate

- All `15` V51/V59 paired keys are present.
- Search design fingerprint, online action sequence, and recommendation are
  identical in every pair.
- No target oracle enters search, verification, or recommendation.
- Verification samples are independent, not reused, and do not update the
  posterior.
- No false certificate occurs in the synthetic oracle audit.
- At least one sound terminal certificate appears in each of FactorShock,
  Inventory, and Queue.

Passing this gate promotes the two-stage certification protocol only. It does
not replace V51 as the search-performance baseline, because V59 spends a
larger total target budget.

## Completed Result

Run `scolh_v59_terminal_gaussian_verify_n13_r48_s5_20260726_01` completed all
15 tasks without failure or retry.

- Search trajectories and recommendations matched V51 in all 15 pairs.
- Independent terminal certificates were obtained in 11/15 runs:
  FactorShock 5/5, Inventory 2/5, and Queue 4/5.
- There were zero false terminal certificates.
- The transferred posterior certificate remained empty in all 15 runs.
- Verification simulation cost was fully charged: 48 extra target calls per
  run, for 61 total calls.

The four failures were audited only after the posterior ranking was frozen.
In all four, posterior rank 2 was substantially deeper in the true safe
interior than rank 1. The three Inventory rank-1 policies were feasible but
too close to the boundary for 48 replications; the Queue rank-1 policy was
truly infeasible. This motivates V60's registered frozen-shortlist protocol,
not a relaxation of the V59 confidence threshold.

# Observable State/Trajectory Coordinate V2

## Why V1 Was Rejected

The first exposure-coordinate gate completed all 160 cells without runtime
failure, but did not pass its preregistered promotion gate. The best
oracle-free challenger was `learned_exposure_phi_r4`.

- Its mean-rank criterion was noninferior in at least three scenarios and its
  error remained noncatastrophic.
- FactorShock scale 4 had a true-feasible point in the shared audit pool for
  only 3/5 seeds.
- Even after replacing fitted mean and variance by post-run oracle values, no
  scenario contained a certified point under the current epistemic radius.
- The provider upper bound exceeded the learned representation in mean rank by
  0.381 on FactorShock scale 0 and 0.369 on scale 4.

The result identifies two separate failures: candidate support and observable
representation. It does not support promoting V1 or running its sequential
gate.

## V2 Contract

V2 starts from a common target-observable record `e(x)` and uses independent
heads:

```text
e(x)       = observable state/action occupancy and trajectory statistics
phi_mu(x)  = h_mu(e(x))
psi_v(x)   = h_v(e(x))

margin(x) = mu_g(phi_mu(x))
          + sqrt(beta_g) s_g(phi_mu(x))
          + z_alpha sqrt(v_C_plus(psi_v(x)))
          - tau.
```

`ObservableStateExposure` contains control-channel means/scales, occupancy
histograms and quantiles, transition statistics, and low-frequency trajectory
components. Its schema is fixed-dimensional across domains. The synthetic
domains provide only their observable action topology:

- FactorShock: primary control and aggregate tail-control channels.
- Inventory: stock, reorder, and safety-control channels.
- Queue: capacity, priority, and smoothing-control channels.

These adapters do not read target outcomes, hidden target centers, true
variance, `risk_exposures`, target anchors, or target refinement hooks. The
exact cumulative-risk provider remains a privileged upper-bound track only.

## Candidate Support

The target-independent boundary pool now reserves 62.5% of its capacity for
the original universal source design plus a target-unlabeled dense coarse
design over head/tail and three-block policies. This additional design is
deliberately excluded from source-archive construction, so the frozen archive
and source-informed `n0` hash remain unchanged. The remainder retains
source-stratum and random low-frequency coverage. At pool size 512, the
deterministic design contains the full generic head/tail tenth-grid. A local
truth-only audit confirmed that FactorShock scale 4 has at least one feasible
policy for each of seeds 0 through 4. This truth is never used to rank or
select a target query.

## V2 Offline Gate

The paired matrix has 140 independent cells:

- controls: `latent_control`, `learned_exposure_v1_phi_r4`;
- oracle-free challengers: `observable_state_phi_r2/r4/r8`;
- privileged upper bounds: `provider_exposure_phi_r2/r4`;
- scenarios: FactorShock scale 0/4, Inventory, Queue;
- target setting: `d=1000`, `N=n0=10`, five seeds;
- same frozen source archive and frozen source-informed initial design;
- no sequential query, no checkpoint, and the entire 512-policy post-run
  truth-audit pool.

Promotion still requires rank and MAE noninferiority, 4/5 FactorShock scale-4
support, observable-track oracle freedom, and nonvacuous oracle-mean/variance
certifiability in at least three scenarios. Failure of only the last condition
will isolate epistemic certification as the next theoretical/statistical
bottleneck rather than inviting another representation patch.

## Completed V2b Result

The corrected run `scolh_observable_state_coordinate_offline_s5_20260718_v2b`
completed 140/140 cells with no failed or cancelled task. It did not advance.

- The dense generic pool repaired candidate support: every variant contained a
  true-feasible FactorShock scale-4 point in all 5/5 seeds.
- Rank 4 was the strongest observable-state mean coordinate. It was rank-
  noninferior in 3/4 scenarios, but Inventory median mean error was `0.8001`
  versus `0.2960` for the latent control, so it failed the noncatastrophic-error
  condition.
- Rank 2 and rank 8 were weaker; no observable challenger was eligible for
  selection.
- Replacing fitted mean and variance by post-run oracle values still yielded
  zero certified points in every scenario. All 20 failures per variant were
  assigned to `epistemic_or_safety_depth`, with median best-feasible epistemic
  radius ranging roughly from `0.108` to `0.444` on the rank-4 track.
- The provider upper bound remained much stronger for FactorShock (mean-rank
  correlations `0.652` and `0.694`) but failed to transfer semantically to
  Queue (`-0.028`), so it cannot rescue the oracle-free claim.

V2 is therefore retained as an ablation, not promoted. V3 tests whether a
separate variance head learned from the same observable exposure improves HVD;
it cannot by itself erase the already measured Inventory mean-coordinate and
epistemic-certification failures.

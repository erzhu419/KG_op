# Transferable Risk-Objective Atlas

This directory grew from the SC-OLH-KG prototype, but the frozen Operations
Research method is now deliberately narrower.  Its novel component is a
source-learned, dimension-equivariant risk-objective proposal atlas.  The
online optimizer is replaceable, and deployment is decided by an independent
terminal verifier.  KG, cumulative HVD, manifold, and transformer variants are
retained as ablations rather than headline contributions.

This directory is intentionally separate from the original `KG_op` code.  It
contains a minimal, testable implementation path:

1. Profile the current GPR-KG baseline.
2. Smoke-test OLH variance decomposition in the existing bi-objective shape.
3. Run single-objective chance-constrained OLH-KG.
4. Run factor-HVD cumulative-risk certification with
   `mu_g + sqrt(beta_g)s_g + z sqrt(v_C^+) <= tau`.
5. Compare additive, exact-MC, and blended SC-OLH-KG acquisition variants.
6. Add deterministic state-policy coupling and fresh-log traffic encoding.
7. Compare deterministic, self-supervised, and transformer-style state
   encoders as SC coupling ablations.
8. Learn frozen source-domain chance-boundary coordinates and a cumulative-HVD
   shape prior under leave-one-domain-out evaluation.

Generated outputs should go under `results/`, `profiles/`, or `checkpoints/`;
those directories are ignored by git.

## Frozen Paper Method

The immutable method contract is
`performance/manifests/paper_final_method_v1.json`:

1. two source domains, 64 policies per domain, and three replications per
   policy give 384 source calls;
2. a low-frequency risk-objective atlas is learned under leave-one-domain-out
   transfer and frozen before target outcomes;
3. V3 admits a normalized endpoint only when every source task agrees on the
   chance-margin direction, otherwise it fails closed to the V1 atlas;
4. the target receives `n0=10` atlas points and 13 total search calls;
5. canonical BoTorch SAASBO is the registered replaceable backend;
6. an independent ordered three-policy verifier controls familywise unsafe
   deployment and separately guards an already safe objective incumbent.

At `d=1000`, the frozen atlas gives 60/60 independently certified deployments
with zero false certificates in FactorShock, Inventory, and Queue.  The same
front end remains 30/30 certified at `d=10000`.  An untouched 20-seed OPSD
storage-reliability confirmation gives 20/20 certificates, zero false
certificates, and 20/20 paired objective wins against common Sobol under the
same neutral continuation and verifier.

Every result must report source, initial target, adaptive search,
verification, and total calls separately.  `N=13` is the target search budget,
not the total data cost.

The final proof spine is summarized in `../proof/final_or_theory.md`.  Legacy
V51/V64 and SC-OLH-KG acquisition results remain reproducible historical
ablations; they are not the frozen paper method.

## Quick Commands

```bash
python3 SC-OLH-KG/runners/profile_current.py --N 12 --n0 5 --K1 10 --K2 1
python3 SC-OLH-KG/runners/run_single_olhkg.py --N 30 --n0 8 --K1 25 --K2 0
python3 SC-OLH-KG/runners/run_sc_olhkg.py --N 30 --n0 8 --K1 25 --K2 0
python3 SC-OLH-KG/runners/run_sc_olhkg.py --N 30 --n0 8 --K1 25 --K2 0 --use_state_basis
python3 SC-OLH-KG/runners/run_olh_biobj_smoke.py --N 20 --n0 6 --K1 20
python3 SC-OLH-KG/performance/benchmark_quality.py --N 20 --n0 5 --K1 15 --K2 1 --n_seeds 5
python3 SC-OLH-KG/performance/benchmark_quality.py --problem FactorShockStatePolicyRZDT1 --modes pooled,class,orthogonal,factor --sc_modes factor --acquisition_modes additive,exact_mc,blend --N 40 --n0 8 --K1 25 --K2 1 --n_seeds 10
python3 SC-OLH-KG/performance/benchmark_exact_kg.py --N 18 --n0 6 --K1 10 --exact_mc_samples 2 --n_seeds 3
python3 SC-OLH-KG/performance/benchmark_sota.py --problem StatePolicyRZDT1 --N 20 --n0 5 --baselines botorch_turbo,botorch_scbo,botorch_saasbo
python3 SC-OLH-KG/performance/benchmark_encoder_suite.py --problem StatePolicyRZDT1 --N 30 --n0 8
python3 SC-OLH-KG/performance/benchmark_traffic_fresh.py --trajectory_log /path/to/fresh_traffic_trajectories.csv
python3 SC-OLH-KG/performance/diagnose_hvd_calibration.py --variance_mode orthogonal --seed 4
python3 -m unittest discover -s SC-OLH-KG/tests
```

## Certification And Exact KG

`SingleOLHKGConfig(certification_mode="theory")` uses the paper-style
certificate:

```text
mu_g + sqrt(beta_g) * s_g + z_alpha * sqrt(v_C_plus) <= tau.
```

`s_g` is constraint GPR posterior variance and `v_C_plus` is HVD
certification variance.  In `factor` mode, `v_C_plus` prioritizes cumulative
risk blocks and includes residual-tail/model-uncertainty guards.  Legacy mode
is kept only for ablation.

`SingleOLHKGConfig(acquisition_mode=...)` accepts `additive`, `exact_mc`, and
`blend`.  `exact_mc` enables sampled posterior-update KG and updates cloned GPR
and HVD states before recomputing the terminal theory-certified value.  If
`exact_mc` or `blend` is selected without an explicit sample count, the runner
uses a small default MC count for a real exact-KG code path.

## LODO Aligned HVD

`benchmark_lodo_meta_prior.py --meta_component_stage spectral_hvd` adds one
isolated module to the source-frozen spectral baseline. Source-domain variance
labels learn a dimensionless cumulative-HVD shape in boundary-aligned
coordinates. A held-out target uses only an unlabeled policy pool to normalize
that shape and ordinary pilot residuals to recover one nonnegative amplitude.
Target variance-shape corrections require repeated evaluations of the same
policy; singleton residuals cannot masquerade as aleatoric labels. The source
LODO upper residual quantile remains in `v_C_plus` as a certification guard.

The `spectral` stage remains the control: it learns the same coordinate and
spectral representation but transfers no HVD parameters.

The LODO benchmark and scheduler default to the promoted
`spectral_hvd + exact_mc(2) + three replication candidates` configuration.
Pass explicit flags to recover the `spectral`, additive, or no-replication
ablations.

Held-out pilot coefficients use a fixed truncated-SVD condition cap of
`1/rcond = 1000`, so source-visible directions that are numerically
unidentifiable on the target pilot cannot create extreme extrapolation. The
rank-one GPR update also repairs a covariance only when a negative quadratic
form is detected, before adding observation noise to the Kalman denominator.
Both operations are reported in `meta_basis` and `gpr_numerics` diagnostics.

## State Encoders

`SingleOLHKGConfig(encoder_kind=...)` accepts deterministic occupancy,
manifold, graph/diffusion, and lightweight self-supervised encoders.  The
self-supervised path learns a low-rank state representation from unlabeled
policy samples or trajectory summaries using masked-reconstruction style SVD
features; `ssl_hybrid` adds contextual/risk-regime interactions.  The
`graph_laplacian` path is a diffusion-map coupling over policy-state summaries.
All are available to SC candidate generation and coupling scores, and
`performance/benchmark_encoder_suite.py` writes their comparison table.

## Traffic Logs

`TrafficTrajectoryEncoder` reads fresh-seed CSV logs with `policy_id`, `state`,
`action`, and optional `occupancy`, `queue`, `wait`, `flow`, `demand_shock`
columns.  `performance/benchmark_traffic_fresh.py` reports `missing_data` when
real logs are absent; it does not fabricate traffic results.

## Compute-H Certificates

`core.kg.compute_h_certificate` returns the active line-envelope hull used by
`compute_h`; `validate_h_certificate` checks all original lines against the
finite-interval endpoint and Gaussian-tail slope conditions.  These checks are
bridged to Lean in `proof/SCOLHKG/Real/LineEnvelopeStack.lean`.  The certificate
also carries a per-step stack trace for candidate, break, pop, and push actions;
pop/push cut-order preservation is formalized in
`proof/SCOLHKG/Real/LineEnvelopeAlgorithm.lean`.  The final global dominance
invariant implies exact KG without a runtime validator in
`proof/SCOLHKG/Real/LineEnvelopeGlobal.lean`, and the full recursive
sorted-line fold/output proof lives in
`proof/SCOLHKG/Real/LineEnvelopeFold.lean`.

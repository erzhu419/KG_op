# Mean Semantic Alignment and Epistemic Calibration V4

## Diagnosis

V3 improved cumulative-variance calibration in three of four domains, but the
theory certificate remained empty even when the post-run audit found safe
points. The failure was not primarily the HVD variance head. Two independent
issues remained in the constraint-mean posterior:

1. ordered observable channels do not have stable semantics across domains;
2. source residual variance was represented as an independent raw-policy
   deviation, so target observations could not contract it at nearby latent
   coordinates.

Inventory additionally suggests that a linear boundary head cannot represent
all chance-boundary pockets.

## V4 Hypotheses

V4 leaves the cumulative HVD variance head unchanged and tests three changes:

- `set_invariant`: symmetric channel-set statistics remove domain-specific
  channel permutations while retaining observable exposure order statistics;
- `diagonal_quadratic`: squared aligned coordinates represent curved safe
  pockets without adding unrestricted pairwise interactions;
- `latent_shared`: split source residual discrepancy into a shared latent
  coefficient covariance and a finite residual floor.

For source residual variance `r`, coefficient dimension `p`, and `m` source
records, V4 uses

`r_floor = r / m`, `Sigma_delta = ((r-r_floor)/p) I`.

At reference feature energy `||phi||^2=p`, this preserves total discrepancy
variance exactly. Unlike a raw independent floor, `Sigma_delta` contracts after
charged target observations. No target truth or target risk provider is used.

## Offline Gate

Six arms isolate each mechanism and their joint effect:

1. `v3_control`: ordered, linear, evidence mixture, raw independent residual;
2. `semantic_invariant`;
3. `boundary_quadratic`;
4. `epistemic_latent`;
5. `v4_ordered_joint`;
6. `v4_invariant_joint`.

The gate uses the same frozen source archive, source-informed `n0=10`, `d=1000`,
four scenarios, and five seeds. `N=n0`, so this stage measures posterior quality
without acquisition effects. HVD settings are identical in every arm.

Promotion requires a joint arm to be oracle-free, preserve candidate support,
improve Inventory mean MAE by at least 30%, remain noncatastrophic in all four
domains, preserve rank/variance calibration in at least three domains, produce
nonvacuous oracle-mean-and-variance certificates in at least three domains, and
make no false certificate in the post-run truth audit.

Only a passing joint arm advances to a sequential gate. Failure is recorded and
does not replace V3 or trigger parameter tuning on the same seeds.

## Completed Gate Result

Run `scolh_mean_semantic_epistemic_offline_s5_20260718_v4` completed all
120/120 shards with no failed or retried task. Only `result.json` files were
retrieved. The preregistered gate failed, so no arm was promoted and no
sequential experiment was launched.

| Arm | FS0 mean MAE | FS4 mean MAE | Inventory mean MAE | Queue mean MAE | False certificates, four-domain total |
|---|---:|---:|---:|---:|---:|
| V3 control | 0.373 | 0.342 | 0.760 | 0.267 | 261 |
| Set-invariant only | 0.498 | 0.525 | 0.585 | 0.402 | 178 |
| Quadratic only | 0.427 | 0.491 | 0.592 | 0.342 | 255 |
| Latent discrepancy only | 0.372 | 0.349 | 0.609 | 0.265 | 174 |
| Ordered joint | 0.425 | 0.494 | 0.828 | 0.324 | 777 |
| Invariant joint | 0.657 | 0.844 | 0.685 | 0.426 | 470 |

The isolated latent split is the only broadly useful V4 mechanism. It reduced
Inventory's median best-feasible epistemic radius from 0.422 to 0.257 and its
variance log-RMSE from 1.173 to 0.753, while preserving FS0 and Queue mean MAE.
It nevertheless left the oracle-mean-and-variance certificate empty in every
domain. Median safety depth versus epistemic radius remained approximately
0.060/0.112 (FS0), 0.008/0.147 (FS4), 0.046/0.257 (Inventory), and 0.027/0.306
(Queue).

Set invariance helped Inventory but discarded useful channel-role semantics in
the other domains. The diagonal-quadratic head also helped Inventory in
isolation, but damaged rank and factor-shock variance calibration. Combining
either with the latent split produced strong negative interaction rather than
an additive gain.

The new false-certificate audit also exposes a pre-existing V3 defect: a
certificate can be nonempty because the transferred constraint mean is biased
too far toward safety even while the oracle substitution remains uncertifiable.
The V3 control made 261 false certifications over the four five-seed audit
cells. The isolated latent split reduced this to 174 but did not close the
soundness assumptions empirically.

The next mean-head revision should therefore not retain the quadratic or fully
set-invariant joint arms. The evidence supports an equivariant, uncertainty-
aware channel-role alignment and a source-mean misspecification posterior that
models transferable bias, not another change to HVD variance fitting. Any next
gate must retain the explicit false-certificate metric introduced here.

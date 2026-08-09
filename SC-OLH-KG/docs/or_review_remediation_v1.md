# OR Review Remediation Contract V1

> Historical execution contract. Review-V2 closure, including projected
> structural coordinates, finite mean/scale source margins, stratified task
> units, and outcome-adjusted cost, is recorded in
> `or_review_v2_remediation.md`.

This document is an experiment and theory contract, not manuscript prose.
No confirmatory result may be inferred from an implemented runner or a passing
test. The manuscript remains unchanged until the frozen experiments finish.

## Fixed paper identity during remediation

The candidate method is a source-scored subset of a preconstructed structural
library for ordered policy profiles, followed by a replaceable optimizer and an
independent terminal verifier. KG and cumulative HVD are not promoted to core
contributions without new causal evidence.

## Issue ledger

| Review issue | Algorithm or experiment response | Current evidence status |
|---|---|---|
| Synthetic-method isomorphism | Eight randomized regimes vary effective rank, frequency support, coordinate order, grid regularity, smoothness, sparse high-frequency activity, and target misspecification. Nominal dimension and effective rank are reported separately. | Complete: 2,880 primary cells and 8,640 sensitivity cells. Adverse-regime failures are retained and delimit the claim. |
| Energy does not establish source learning | V2 expands to 18 markets in five geographic regions and retains its negative structured-control result. V3 is frozen independently: the policy is a forecast-stress-to-reserve response function, the 1000-point decision grid is separated from a 168-hour physical horizon, and every target excludes its whole region from source training. Source atlas is compared with generic DCT, random low-frequency, natural constant-grid, raw Sobol, and target-only functional SCBO. | Complete negative control. V2 and V3 are reported separately. In V3, source atlas certified 60/90 versus 70/90 for generic DCT; the region-level source-minus-DCT difference was -0.0787 (bootstrap 95% CI [-0.1373,-0.0200]). No post-outcome tuning is permitted. |
| Development-level benchmark overfitting | Freeze and push code before deriving confirmatory task seeds. Use newly randomized stress tasks and adverse regimes; no post-outcome repair is allowed under the same version. | Complete as an audit, not a universal guarantee. Frozen randomized tasks, schema controls, negative Energy results, and adverse regimes materially reduce benchmark-fit risk; the registered task distribution remains authored within this project. |
| Atlas underspecified | The complete finite method is recorded in `profile_atlas_v2_method_spec.json`, including library, coefficient integral, rank statistic, augmented metric, first center, farthest-first rule, and tie breaking. | Complete and unit tested. The final manuscript must reproduce this specification rather than a verbal summary. |
| Coverage theorem assumes success | Add exact profile-coordinate integration, continuous coefficient error, inverse-grid transfer, finite source mean/scale error propagation, rank recovery, 2-approximate finite k-center coverage, and independent task-coverage calibration. | Formal results are complete and placeholder-free. Task-distribution coverage remains conditional on a declared task law and calibrated discrepancy; no distribution-free target-coverage claim is allowed. |
| Candidate verifier validity and power | Use a frozen shortlist and an exact all-success binomial candidate test with explicit familywise spending. Before revealing confirmatory outcomes, register the exact Clopper-Pearson fixed-budget threshold as a power sensitivity on the same frozen Bernoulli samples. Generate exact power/nonvacuity curves over true safety depth and budget. | All-success candidate validity is machine checked. The comparison table reports that its power decreases with budget below probability one, while the exact Clopper-Pearson threshold permits increasing numbers of failures and preserves candidate-wise exact coverage. All-success remains the preregistered primary result; Clopper-Pearson cannot be selected post hoc by outcome. |
| Energy temporal independence | Draw start indices independently with replacement from one fixed verification-year empirical set. Physical windows may overlap. The certificate is only for that empirical distribution; no iid future-calendar claim is allowed. Add a frozen post-decision audit over chronological blocks and physically nonoverlapping windows, explicitly without a second certification claim. | Complete. V3 had 312/358 block-stable and 358/358 nonoverlap-stable originally certified decisions. These are descriptive post-decision audits, not replacement certificates. |
| Equal-total-cost wording | Report source, target search, actual verification, maximum verification, actual all-in, maximum all-in, and amortized all-in calls. Add a fixed-cap comparison of 384 source + 10 target versus 394 target-only calls. Initial-design and full-search truth audits are separate, so all 394 target-only calls receive credit in optimizer/all-in endpoints. Certified success and regret are measured on the deployed policy, never on the best point merely encountered during search. | Complete. At one-target equal preverification cost, source atlas was not best: 0.4625 certified success versus 0.5375 for generic DCT/raw Sobol and 0.5313 for natural structure. The source archive is justified only as an amortized multi-target cost. |
| Transfer baselines receive our atlas | Rename existing shared-atlas results conceptually as a backend comparison. Audit native end-to-end transfer pipelines under identical source information, target budget, and verifier. | Complete: 480 native end-to-end cells across eight transfer methods, three held-out domains, and 20 seeds. Queue was 0/20 for every native method; failures are reported rather than hidden. |
| Missing structured controls | Add generic DCT maximin, random low-frequency, natural blockwise or constant grids, raw Sobol, and a labeled finite-library oracle upper bound. | Complete in randomized stress, cost, and Energy matrices. Oracle upper bounds remain visually and statistically separated from admissible methods. |
| Causal wording | Use `controlled factorial attribution` or `matched frontend-backend intervention`. | Manuscript-only change deferred. |
| Pseudoreplication | Treat randomized target tasks, markets, and geographic regions separately from repeated algorithm seeds. Prohibit pooled domain-seed population claims. Keep initial-design coverage, search-best quality, and certified deployed-policy quality as separate endpoints. | Complete. Task/region inference and within-task repeatability are separate in every compact analysis artifact. |
| Strong target schema | Run declared-schema, schema-blind, coordinate-permutation, descriptor-conditioned, and domain-blind controls. Explicitly disclose that declared schema supplies the complete semantic-to-raw coordinate map. Domain-separate target generation and algorithm randomization. | Complete: 1,280 cells. Schema-blind performance degrades but remains nonzero; target descriptors do not improve results and therefore are not part of the primary method. |
| Hyperparameter sensitivity | Register alpha, safe mass, n0, effective rank, source task/profile/replication budgets, retained frequency, frequency penalty, rank metric weights, and first-center tradeoff. | Complete: 8,640 OFAT cells. Nonmonotone source-budget effects and poor misspecification robustness are retained as limitations. |

## Promotion rules

1. Source learning is supported only where the frozen source atlas improves on
   both generic DCT and the natural structured control at the independent-task
   level without increasing false certification.
2. If structured controls improve on raw Sobol but source scoring does not
   improve on structured controls, the contribution is narrowed to structural
   profile design.
3. Adverse regimes are retained regardless of outcome. They define the method's
   scope rather than a failure to be hidden.
4. Energy remains in the paper as an independent real-data test or negative
   control. It is removed only for a scientific scope reason, never because its
   result is unfavorable.
5. HVD remains auxiliary unless a same-proposal, same-backend, same-verifier
   experiment shows an incremental optimization or certification benefit.

The existing 60-pair provider-coordinate replay closes rule 5 negatively:
cumulative factor-HVD improves variance calibration but not feasible recovery,
false certification, regret, or verification cost. It is therefore frozen as
a secondary appendix result under `or_review_hvd_disposition_v1.json` and is
not part of the candidate method identity.

## Required frozen outputs before manuscript revision

- Primary randomized stress matrix at dimensions 200, 1000, and 10000.
- Schema and descriptor controls.
- Full OFAT sensitivity matrix.
- Equal maximum all-in cost comparison and archive break-even analysis.
- OPSD region-held-out V2 matrix.
- OPSD forecast-indexed V3 matrix; V2 and V3 must be reported separately.
- OPSD post-decision chronological-block/nonoverlap stability audit.
- Exact verifier power table.
- Native end-to-end transfer audit.
- HVD causal disposition under a common proposal/backend/verifier. Complete;
  retained only as a secondary calibration result.
- One immutable result registry containing commit, task seeds, budgets,
  failures, timeouts, and raw-result hashes for every paper table.

All required outputs are complete. The immutable compact registry is
`paper_artifacts/or_review/final_evidence_registry_v1.json`; it commits 14
matrices and 12 aggregate analyses, including one counted algorithmic failure.

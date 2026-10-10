# Final meta-prior claim audit

This audit is tied to `performance/meta_prior_submission_manifest.json` and
the saved artifacts under `paper_artifacts/meta_prior_feasible_deployment_v1/`.
It is produced alongside `paper_artifacts/meta_prior_manuscript_claim_audit.json`.

| Claim line | Evidence | Permitted wording | Status |
|---|---|---|---|
| Finite source structure improves early target prediction | Observable exposure holdout and source-function archive | Improvement within the declared observable representation and 68-function dictionary | Supported, bounded |
| Target observations can correct source mismatch | Frozen posterior-update and KG contract | Components can be downweighted; no guaranteed final decision gain | Supported, bounded |
| Source-selected structure improves final decision utility | Development control: mean 0.861267 vs 0.855817; 3 positive, 1 negative, 4 tied | Development-block attribution signal only | Supported for development, not independently confirmed |
| Independent fixed-seed check | v2 seeds 95800--95807, all retained; means 0.442743 vs 0.451599; 2 positive, 2 negative, 4 tied | Report non-replication of the development mean ordering | Negative/bounded |
| Same-contract target-only full dictionary | Development mean 0.761987; v2 mean 0.451475; all eight tasks retained | Report as a feasible target-only information control, not as a source-superiority test | Included |
| Generic target-only RBF-KG | Development mean 0.862387; v2 mean 0.459121; fixed RBF hyperparameters and all eight tasks retained | Report as a generic constrained-BO control; source structure is not superior under this contract | Negative for generic superiority |
| Learned risk | Initial total-risk prediction and closed-loop artifact | Total-risk calibration only | Prediction supported; deployment value not established |
| Paired uncertainty | `paired_uncertainty.json`, 12 seed-level metrics, 100,000-resample percentile bootstrap | Descriptive intervals with $n=8$; no universal-superiority inference | Included |
| Learned trajectory representation, factor recovery, stable universal deployment, arbitrary high-dimensional optimization | No qualifying artifact | Keep outside contribution claims | Unsupported and excluded |

The original declared interval `95433--95440` was not run as a block.  Seed
`95434` is present in the development block, so it is recorded as an overlap
and is ineligible as clean confirmation.  The v2 block is a separate contract;
it does not retroactively repair that overlap.

The legacy atlas and Energy evidence are kept as separate evidence identities
and are not used as support for the meta-prior claims.  The audit checks this
separation and checks that the legacy entry point is not included in the new
manuscript source.

Rebuild the tables, audit, and PDFs with:

```bash
python3 performance/replay_meta_prior_submission.py --compile
```

The handoff package is `submission/meta_prior_or_package_v1/`; its README and
machine-readable manifest provide the same command without relying on files
outside the package.

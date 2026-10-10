# Meta-prior submission replay

The independent meta-prior manuscript is separate from the legacy atlas entry
point.  The artifact-level replay does not start a simulator or run the
original confirmation interval as a block.  The declared interval overlaps
development at seed 95434, so that seed is excluded from any clean-confirmation
claim.  Replay regenerates manuscript tables from saved JSON, checks the
contracts and claim ledger, and optionally compiles the two PDFs.

From `SC-OLH-KG/`:

```bash
python3 performance/replay_meta_prior_submission.py
python3 performance/replay_meta_prior_submission.py --compile
```

The first command writes:

- `manuscript/meta_prior_tables.tex`
- `manuscript/meta_prior_supplement_tables.tex`
- `manuscript/meta_prior_confirmation_table.tex` when the independent v2 block exists
- `manuscript/meta_prior_baseline_table.tex` for the same-contract target-only control
- `manuscript/meta_prior_uncertainty_table.tex` for seed-level paired bootstrap intervals
- `paper_artifacts/meta_prior_manuscript_claim_audit.json`
- `paper_artifacts/meta_prior_feasible_deployment_v1/target_only_baseline_comparison.json`
- `paper_artifacts/meta_prior_feasible_deployment_v1/paired_uncertainty.json`
- `paper_artifacts/meta_prior_feasible_deployment_v1/target_only_rbf_development.json`
- `paper_artifacts/meta_prior_feasible_deployment_v1/target_only_rbf_confirmation_v2.json`
- `paper_artifacts/meta_prior_submission_replay.json`

The self-contained handoff package is `submission/meta_prior_or_package_v1/`.
Its README repeats the replay command and its package manifest records the
fixed seed boundary and synthetic-data status.

The source-only and development paths can be regenerated separately with the
frozen contracts:

```bash
python3 performance/generate_feasible_inventory_source_archive.py
python3 performance/preflight_meta_prior_feasible_deployment.py
python3 performance/run_meta_prior_feasible_deployment_v1.py
python3 performance/run_meta_prior_feasible_source_arm_v1.py
python3 performance/generate_meta_prior_risk_identifiability_source.py
python3 performance/preflight_meta_prior_risk_identifiability_v1.py
python3 performance/preflight_meta_prior_risk_initial_v1.py
python3 performance/run_meta_prior_risk_identifiability_v1.py
```

These commands regenerate development artifacts and consume local CPU.  They
do not authorize a rerun of the original `95433--95440` interval.  The
independent v2 block uses its own fixed contract, retains all eight seeds, and
is recorded separately from the development results.

The target-only baseline is regenerated separately with:

```bash
python3 performance/run_meta_prior_feasible_source_arm_v1.py \
  --contract performance/meta_prior_feasible_baseline_development_v1.json \
  --output paper_artifacts/meta_prior_feasible_deployment_v1/target_only_development.json
python3 performance/run_meta_prior_feasible_source_arm_v1.py \
  --contract performance/meta_prior_feasible_baseline_v1.json \
  --output paper_artifacts/meta_prior_feasible_deployment_v1/target_only_confirmation_v2.json
```

These two commands consume local CPU; the artifact replay itself does not.

The generic target-only RBF-KG controls use the frozen contracts
`performance/meta_prior_feasible_rbf_baseline_v1.json` and
`performance/meta_prior_feasible_rbf_confirmation_v2.json` with
`performance/run_meta_prior_feasible_rbf_baseline_v1.py`.

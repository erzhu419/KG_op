# Raw trajectory archive and ledger replay notes

Every table, figure, statistical test, and safety audit reported in the manuscript and Electronic Companion is regenerated directly from the sealed CSV ledgers (`per_run_ledger.csv`, `per_slot_certificates.csv`) and compact JSON readouts included in `code_and_data.zip` with zero simulation calls.

The review supplement contains the compact JSON receipts for the 86-task static prediction study, the 86-task exploratory and 72-task confirmatory direct cross-policy propagation studies, and the 24-state multi-scramble query and `UB-cEI` comparison, as well as all five compact fitted `.pkl` source priors (`SC-OLH-KG/paper_artifacts/v2_or_u3_confirm_v1/source_bundles/*/prior_default.pkl`). The separate U3 companion archive contains the raw per-job simulation trajectories and verification draw arrays from the 9,796-run historical benchmark. The review supplement includes the historical replay receipt and extraction code.

- `SC-OLH-KG/paper_artifacts/v2_or_u3_historical_replay_receipt.json` records the programmatic audit over all 9,796 four-domain benchmark records, verifying exact cryptographic agreement between the raw per-job audit records and the six sealed CSV ledgers, plus exact reproduction of five representative multi-slot verification decisions with zero new simulation calls.
- `cluster_archive_tools/` contains the exact extraction scripts (`read_u3_replay.py`, `summarize_operating_policy.py`, `summarize_terminal_replay.py`, `diagnose_source_loo.py`, and prefix state-restoration helpers) used to distill raw trajectory archives into the compact receipts included in `SC-OLH-KG/paper_artifacts/`.

If an editor or referee wishes to inspect raw trajectory archives (`GP_FTA_U3_Historical_Reviewer_Inputs.tar.gz`, 107.9 MB compressed) alongside `code_and_data.zip`, the archive can be transferred directly via the editorial office's designated large-file channel and audited via:

```bash
python cluster_archive_tools/read_u3_replay.py --sc-root review-inputs/SC-OLH-KG --campaign v2_or_u3_confirm_v1
python cluster_archive_tools/read_u3_replay.py --sc-root review-inputs/SC-OLH-KG --campaign v2_or_u3_confirm_v1 \
  --job sv_Inventory_CrossHorizon_N8_t09_s901_GP-FTA-KG
python cluster_archive_tools/read_u3_replay.py --sc-root review-inputs/SC-OLH-KG --campaign v2_or_u3_confirm_v1 --aggregate
```

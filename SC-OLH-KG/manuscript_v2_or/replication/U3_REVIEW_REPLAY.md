# Historical U3 evidence

The review supplement contains all sealed U3 aggregate CSVs and the compact
historical replay receipt. The separate controlled companion
`GP_FTA_U3_Historical_Reviewer_Inputs.tar.gz` contains saved candidate and
deployment policies, trajectory archives, fitted priors, audit working records,
sealed manifests and original execution source. It can be provided through the
journal's reviewer channel after access is approved. The supplementary ZIP
regenerates the manuscript tables and figures directly from the sealed ledgers.

The historical aggregate replay processed 9,796 saved audit records. It
recreated the six sealed CSV fingerprints and twelve tested summary fields per
cohort. Five selected return decisions were also regenerated from the stored
verification counts and compared with the recorded policies and spending. The
archived work for these checks incurred zero new simulation calls.

From the supplementary directory, once the authorized companion is available,
run the following against the companion's extracted `SC-OLH-KG` directory.

```bash
python read_u3_replay.py --sc-root review-inputs/SC-OLH-KG --campaign v2_or_u3_confirm_v1
python read_u3_replay.py --sc-root review-inputs/SC-OLH-KG --campaign v2_or_u3_confirm_v1 \
  --job sv_Inventory_CrossHorizon_N8_t09_s901_GP-FTA-KG
python read_u3_replay.py --sc-root review-inputs/SC-OLH-KG --campaign v2_or_u3_confirm_v1 --aggregate
```

The historical U3 fingerprint covers the Python program. Historical live SUMO
commands referenced a retained 1.22.0 runtime, and its current engine check
reports 1.22.0. The external runtime and scenario were outside the original
Python fingerprint. The experimental RESCO inputs are restored by the pinned
recipe in the supplementary README.

# Operations Research Manuscript

This directory contains the manuscript for the frozen source-scored structural
initial-design method. It is independent of the rejected legacy manuscript in
`Final_Submission/`.

## Method identity

```text
source-scored subset of a public profile library
  + replaceable target optimizer
  + independent frozen-shortlist verifier
```

The paper is not an SC-OLH-KG, KG, SAASBO, or HVD method paper. Cumulative HVD
is retained only as a calibration diagnostic.

## Build

```bash
cd SC-OLH-KG/manuscript
python3 ../performance/render_or_review_final_artifacts.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
cd ../..
python3 SC-OLH-KG/performance/audit_or_manuscript.py \
  --manuscript-dir SC-OLH-KG/manuscript \
  --artifact-manifest SC-OLH-KG/manuscript/review_artifact_manifest.json \
  --out SC-OLH-KG/paper_artifacts/or_manuscript_receipt_v1.json
```

The final renderer reads only compact audited analyses under
`../paper_artifacts/or_review/`. It never reads checkpoints, pickle files,
model weights, or raw policy vectors. Input and output hashes are recorded in
`review_artifact_manifest.json`.

## Frozen evidence

- `../paper_artifacts/or_review/final_evidence_registry_v1.json`
- `../performance/manifests/profile_atlas_v2_method_spec.json`
- `../performance/manifests/profile_stress_v2_protocol.json`
- `../performance/manifests/or_review_energy_forecast_indexed_v3.json`
- `../../proof/final_or_theory.md`

Every revision must keep source, target search, verification, total, and
amortized simulator calls distinct. Energy V3 remains a negative external
control, and `d=10000` remains a profile-grid refinement claim.

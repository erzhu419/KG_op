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

The related source archive is supplied exogenously. The method ranks profiles
inside that archive; it does not retrieve source tasks. Farthest-first selection
uses the augmented coordinate `eta=(z,r_g,r_f)`, while target transfer is stated
only in the projected structural coordinate `z`. The primary experiment has
160 independently seeded latent tasks crossed with three resolutions, yielding
480 paired task-resolution cells rather than 480 independent tasks.

## Build

The submission uses the journal's `informs4` class with the `opre` and
`dblanonrev` options, the `informs2014` bibliography style, 12-point
one-and-a-half-spaced review text, continuous equation and theorem numbering,
and EC-prefixed online-supplement numbering. The journal class, bibliography
style, logo, generated `main.bbl`, and both PDFs are included in the audited
source bundle.

```bash
cd SC-OLH-KG/manuscript
python3 ../performance/render_or_review_final_artifacts.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
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

The two explanatory illustrations `figure1_profile_space.pdf` and
`figure2_atlas_coverage.pdf` contain no empirical result. They are cropped
vector exports from the author-provided Visio figures; the second figure's
embedded theorem header is normalized to the number-independent label
`Coverage theorem`. Their hashes are recorded directly in the manuscript
compilation receipt.

The journal-facing article is `main.pdf`. Detailed algorithms, sensitivity
tables, task-seed strata, native transfer results, HVD diagnostics, and the
Lean file map are retained in the separately built `supplement.pdf`; they are
not removed to satisfy the article page limit.

Every display was compiled once with the INFORMS 240-point production-column
validator. The four formulas initially exceeding that width were rebroken;
the validator reported no remaining violations and is intentionally not loaded
in the submission sources, so no instructional boxes appear in the final PDFs.

## Frozen evidence

- `../paper_artifacts/or_review/final_evidence_registry_v1.json`
- `../performance/manifests/profile_atlas_v2_method_spec.json`
- `../performance/manifests/profile_stress_v2_protocol.json`
- `../performance/manifests/or_review_energy_forecast_indexed_v3.json`
- `../../proof/final_or_theory.md`

Every revision must keep source, target search, verification, total, and
amortized simulator calls distinct. Call-count break-even and calls per
successful certified deployment must also remain distinct. Energy V3 remains a
negative external control, and `d=10000` remains a profile-grid refinement
claim.

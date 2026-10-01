# Operations Research manuscript

The current article is `main.pdf`; its electronic companion is `supplement.pdf`.
The method is a source-scored, ten-profile initial design for ordered policies,
followed by a declared target optimizer and independent terminal verification.
The original result-producing method is unchanged.

## Current evidence

The original 14 matrices and 17,360 cells are registered under
`../paper_artifacts/or_review/`. September 7 comparisons use existing target
seeds; September 30 attribution and new-family confirmation are documented in
`../performance/manifests/submission_revision_20260930.json` and
`../paper_artifacts/submission_revision_20260930/`.

The original 480 correlated task-resolution cells give 47.3% true certification
for source scoring versus 31.5% for standardized DCT. On 240 new latent tasks
from three new source-family/library seeds, the rates are 23.3% versus 10.0%
(paired difference 13.3 percentage points, 95% interval 9.6–17.1). Attribution
identifies the source-selected first center as the main effect; augmented
ranks contribute little beyond it. The source-greedy portfolio has better
coverage and loss, and its certification difference is unresolved.
October 2 postdecision controls fix the same source first center: random
nine-point fill gives 53/240 certificates versus structural coverage 54/240
and atlas 56/240, with paired intervals crossing zero. Compact outcomes and
the frozen rules are in `../paper_artifacts/source_first_remainders_v1_20261002/`.

Energy V5 gives every policy a common 50% initial inventory, reruns all 720
cells with shared hourly power: source 57/90 versus unstandardized
generic 54/90, with an unresolved region-level difference. Five source archives serve the actual 18 markets. The source
method has higher calls per certification. Its current records and protocol are
`../paper_artifacts/submission_revision_energy_v5_20260930/` and
`../performance/manifests/submission_revision_energy_v5_20260930.json`. The new rank/design-stability
proposition has an analytic proof; the original Lean interface is preserved.

The metadata corrections are documented in
`../reproducibility/current_protocol_errata_20260907.md`: the actual library
seed is 20261799, positive frequency scaling cancels under standardization,
and the native transfer matrix has N=13 and its own v69 verifier.

## Build

From this directory:

```bash
python3 ../performance/render_submission_revision_20260930.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
python3 ../performance/audit_or_manuscript.py \
  --manuscript-dir . --artifact-manifest review_artifact_manifest.json \
  --out ../paper_artifacts/submission_revision_energy_v5_20260930/manuscript_audit.json \
  --no-compile
```

Build main before supplement: the companion imports its equation and theorem
labels and has its own short bibliography. The INFORMS class uses the `opre,dblanonrev` options,
11-point one-and-a-half-spaced text, subject classification and Simulation
area of review. Main-article tables follow the references in citation order.
The page-limit calculation excludes references and includes those tables.

The current renderer regenerates tables from completed compact analyses and
retains the existing figures. To redraw original figures, run
`render_or_review_final_artifacts.py` first and the current renderer second.
The two schematic illustrations contain no empirical result and retain their
supplied vector PDFs. The rendering manifest identifies quantitative inputs
and outputs.

## Submission attachment

From the repository root:

```bash
python3 SC-OLH-KG/performance/export_submission_revision_bundle.py
```

This creates `../reproducibility/dist/or_submission_revision_controls_20261002.zip`.
It includes code, protocols, compact per-cell data, the processed OPSD archive,
Lean sources, manuscript sources and PDFs. See `../reproducibility/README.md`
for statistical replay and complete rerun commands. These commands build a local attachment and perform no upload.

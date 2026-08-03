# Operations Research Manuscript

This directory contains the new manuscript for the final structural-atlas
method. It is independent of the rejected legacy manuscript in
`Final_Submission/`.

## Method identity

The manuscript studies:

```text
source-learned structural proposal atlas
  + replaceable online optimizer (canonical SAASBO in the primary arm)
  + independent terminal verifier
```

KG, SC-V69, and cumulative factor-HVD are comparisons or diagnostics, not
headline contributions.

## Build

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Generated tables and figures are rebuilt only from compact audited artifacts:

```bash
python3 ../performance/render_or_manuscript_artifacts.py
```

The renderer does not read checkpoints, pickles, model weights, or raw policy
vectors. Its input and output hashes are recorded in `artifact_manifest.json`.

## Evidence contracts

- `../performance/manifests/paper_final_method_v1.json`
- `../paper_artifacts/paper_result_audit_v1.json`
- `../paper_artifacts/paper_paired_statistics_v1.json`
- `../paper_artifacts/final_dimension_budget_evidence_v1.json`
- `../paper_artifacts/paper_submission_readiness_v1.json`
- `../../proof/final_or_theory.md`

Source, target-search, verification, and total simulator calls must remain
separate in every revision.

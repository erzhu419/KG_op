# GP-FTA anonymous review supplement

This archive contains the complete active and frozen Python software, mechanized Lean 4.31.0 proof modules and verification receipts, sealed per-run and per-slot CSV ledgers, compact diagnostic readouts, and deterministic table/figure builders for *Functional Transfer, Stein--Mallows Release, and Eligible-Cascade Verification in Chance-Constrained Simulation Optimization*.

Specifically, it bundles:
- Active implementation `SC-OLH-KG/v2_or` used by the pre-registered 720-run prospective factorial confirmation (`v2_or_eckg_sure_confirm_v1`) and 96-state early/near-terminal query census (`eckg_prefix_audit_readout.json`).
- Frozen execution snapshots `frozen_referee/SC-OLH-KG` (384-run matched coefficient-prior and terminal-replay study `v2_or_gpfta_referee_ablation_v1`), `frozen_independent/SC-OLH-KG` (432-run single-slot quadrature panel `v2_or_kg_quadrature_confirm_v1`), and `frozen_u3/SC-OLH-KG` (9,796-run four-domain benchmark across `v2_or_u3_confirm_v1`, `v2_or_u3_messy_archive_v1`, and `v2_or_u3_live_sumo_v1`).
- Mechanized Lean 4.31.0 modules under `proof/SCOLHKG/V2OR/` (`DebiasedSubspace.lean`, `TemperedHVD.lean`, `VerificationKGAmortization.lean`, `SequentialTheory.lean`) and `proof/SCOLHKG/V2ORProofInterface.lean` verifying all 18 audited theorems with zero `sorry`.
- Third-party provenance and restoration recipes in `LICENSES.md` and `restore_review_inputs.py`, plus raw-archive audit documentation in `U3_REPLAY.md`.

## 1. Verify manifest integrity and regenerate all 16 tables and 3 figures

Run from the extracted supplement root on Linux x86_64 with CPython 3.10+. Every manuscript and Electronic Companion table (`tables/*.tex`) and vector figure (`figures/*.pdf`) regenerates deterministically from the sealed CSV ledgers and compact JSON readouts with zero simulation calls:

```bash
python3 verify_review_supplement.py
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

report=$(mktemp -d)
python SC-OLH-KG/manuscript_v2_or/build_u4_tables_and_figures.py --output-dir "$report"
diff -qr SC-OLH-KG/manuscript_v2_or/tables "$report/tables"
diff -qr SC-OLH-KG/manuscript_v2_or/figures "$report/figures"
```

## 2. Rebuild prospective, matched-ablation, and query-diagnostic readouts

All statistical readouts shipped in `SC-OLH-KG/paper_artifacts/` can be regenerated in disposable directories directly from the included CSV ledgers and compact receipts with zero simulation calls:

```bash
readout=$(mktemp -d)

# 720-run prospective factorial readout (Table 1, Table 2, Figure 1a)
prosp=SC-OLH-KG/paper_artifacts/v2_or_eckg_sure_confirm_v1
python summarize_prospective_panel.py --campaign-dir "$prosp" --output "$readout/prospective_panel_readout.json"
cmp "$prosp/prospective_panel_readout.json" "$readout/prospective_panel_readout.json"

# 384-run matched coefficient-prior readout (Table 4, Table EC.3)
ref_dir="$readout/referee"
cp -a SC-OLH-KG/paper_artifacts/v2_or_gpfta_referee_ablation_v1 "$ref_dir"
PYTHONPATH=frozen_referee/SC-OLH-KG:SC-OLH-KG:scripts python -m v2_or.dispatch.summarize_gpfta_referee \
  --manifest frozen_referee/SC-OLH-KG/v2_or/manifests/v2_or_gpfta_referee_ablation_v1.json \
  --campaign-dir "$ref_dir"
cmp SC-OLH-KG/paper_artifacts/v2_or_gpfta_referee_ablation_v1/referee_ablation_readout.json \
    "$ref_dir/referee_ablation_readout.json"

# 432-run single-slot quadrature readout (Table EC.1, Table EC.2)
ind_dir="$readout/indep"
cp -a SC-OLH-KG/paper_artifacts/v2_or_kg_quadrature_confirm_v1 "$ind_dir"
PYTHONPATH=frozen_independent/SC-OLH-KG:SC-OLH-KG:scripts python -m v2_or.dispatch.summarize_independent_gpfta \
  --manifest frozen_independent/SC-OLH-KG/v2_or/manifests/v2_or_kg_quadrature_confirm_v1.json \
  --campaign-dir "$ind_dir"
cmp SC-OLH-KG/paper_artifacts/v2_or_kg_quadrature_confirm_v1/independent_panel_readout.json \
    "$ind_dir/independent_panel_readout.json"

# Saved query-diagnostic receipts and deterministic unit tests
PYTHONPATH=SC-OLH-KG python summarize_eligible_prefix.py \
  SC-OLH-KG/paper_artifacts/v2_or_gpfta_review3_v1/eligible_query_receipts.json \
  --output "$readout/eligible_query_readout.json"
cmp SC-OLH-KG/paper_artifacts/v2_or_gpfta_review3_v1/eligible_query_readout.json \
    "$readout/eligible_query_readout.json"
PYTHONPATH=SC-OLH-KG:cluster_archive_tools python -m pytest -q test_cascade_prefix.py test_eligible_prefix.py
```

Scripts in `cluster_archive_tools/` (`read_u3_replay.py`, `summarize_terminal_replay.py`, etc.) document how compact receipts were extracted from raw per-job trajectory archives (see `U3_REPLAY.md`).

## 3. Restore pinned upstream inputs and run full unit tests

Restore the 2017--2019 compact ENTSO-E OPSD archive from the official release (or pass `--csv` / `--prepared` for a local verified copy), run the complete `v2_or` test suite (including Fisher-whitened `SURE` and `EC-KG` dual-VOI tests), and verify the data loader:

```bash
python restore_review_inputs.py opsd
(cd SC-OLH-KG && PYTHONPATH=. python -m pytest -q v2_or/tests)
python verify_inputs.py
```

Reconstruct the pinned RESCO `Ingolstadt21` network and route XMLs (`SC-OLH-KG/data/ingolstadt21/data/`) and verify the microscopic `libsumo` 1.22.0 adapter with zero simulation calls:

```bash
python restore_review_inputs.py resco
python -m pip install -r requirements-live-sumo.txt
export SUMO_HOME="$(python -c 'from sumo import SUMO_HOME; print(SUMO_HOME)')"
INGOLSTADT21_SUMO_BACKEND=libsumo python verify_inputs.py --live-sumo
```

## 4. Recompile the LaTeX manuscript and verify Lean 4.31.0 proofs

Install `informs4.cls` and `informs_Logo.pdf` from the official INFORMS *Operations Research* author template ZIP ([INFORMS LaTeX style files](https://pubsonline.informs.org/authorportal/latex-style-files)), then compile with pdfLaTeX and BibTeX:

```bash
python restore_review_inputs.py informs --zip opre-template.zip
(cd SC-OLH-KG/manuscript_v2_or &&
  pdflatex -interaction=nonstopmode -halt-on-error main_or.tex && bibtex main_or &&
  pdflatex -interaction=nonstopmode -halt-on-error main_or.tex &&
  pdflatex -interaction=nonstopmode -halt-on-error main_or.tex &&
  pdflatex -interaction=nonstopmode -halt-on-error main_or.tex)
```

Verify all 18 mechanized theorems in Lean 4.31.0 (`proof/lean-toolchain` and `proof/lake-manifest.json`):

```bash
cd proof
lake exe cache get
lake -Kjobs=1 build
lake env lean SCOLHKG/V2ORProofInterface.lean
```

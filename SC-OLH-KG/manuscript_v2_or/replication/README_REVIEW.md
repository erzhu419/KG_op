# GP-FTA review supplement

This archive provides the paper's table and figure generator, per-run ledgers,
per-slot certificates, software, mechanized proofs, the completed 432-cell
independent panel and the 384-cell referee study. The manifest records the
original execution identifiers for `frozen_u3/SC-OLH-KG`,
`frozen_independent/SC-OLH-KG`, and the active referee code in `SC-OLH-KG`.
The review source copies omit cluster-only launchers and use portable scheduler
defaults. Exact execution snapshots remain in the controlled originals. Cost
records and scientific CSVs are retained. `LICENSES.md` documents third-party
inputs, and `U3_REPLAY.md` describes the saved working records.

## Rebuild the paper's results

Run from the extracted directory with Linux x86_64 and CPython 3.10.

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

The 432-cell paired and operating-policy readouts and the 384-cell referee
readout are included alongside their sealed summaries and ledgers. Rebuild
them in disposable directories to preserve the sealed originals.

```bash
campaign=v2_or_kg_quadrature_confirm_v1
readout=$(mktemp -d)
sealed="SC-OLH-KG/paper_artifacts/$campaign"
cp -a "$sealed/." "$readout/"
PYTHONPATH=frozen_independent/SC-OLH-KG:SC-OLH-KG:scripts python -m v2_or.dispatch.summarize_independent_gpfta \
  --manifest "frozen_independent/SC-OLH-KG/v2_or/manifests/$campaign.json" --campaign-dir "$readout"
cmp "$sealed/independent_panel_readout.json" "$readout/independent_panel_readout.json"
python summarize_operating_policy.py "$sealed" \
  --stage-readout "$sealed/operating_policy_stage_summary.json" > "$readout/operating_policy_readout.json"
cmp "$sealed/operating_policy_readout.json" "$readout/operating_policy_readout.json"

campaign=v2_or_gpfta_referee_ablation_v1
readout=$(mktemp -d)
sealed="SC-OLH-KG/paper_artifacts/$campaign"
cp -a "$sealed/." "$readout/"
PYTHONPATH=SC-OLH-KG python -m v2_or.dispatch.summarize_gpfta_referee \
  --manifest "SC-OLH-KG/v2_or/manifests/$campaign.json" --campaign-dir "$readout"
cmp "$sealed/referee_ablation_readout.json" "$readout/referee_ablation_readout.json"
```

## Get the pinned inputs and run checks

Load the official OPSD release and regenerate the 2017–2019 compact archive.
An exact local raw CSV can be passed with `--csv`, and an authorized copy of
an already prepared exact archive can be passed with `--prepared`. The command
verifies the prepared archive before installing it.

```bash
python restore_review_inputs.py opsd
(cd SC-OLH-KG && PYTHONPATH=. python -m pytest -q v2_or/tests)
python verify_inputs.py
```

Reconstruct the exact project network and routes from the pinned RESCO version.
Install the pinned SUMO wheels for the engine and scenario load check, which
makes zero simulation calls.

```bash
python restore_review_inputs.py resco
python -m pip install -r requirements-live-sumo.txt
export SUMO_HOME="$(python -c 'from sumo import SUMO_HOME; print(SUMO_HOME)')"
INGOLSTADT21_SUMO_BACKEND=libsumo python verify_inputs.py --live-sumo
```

Download the Operations Research LaTeX archive from the INFORMS author portal.
Install the pinned class and logo from it before compiling the manuscript with
pdfLaTeX, BibTeX, `tgtermes`, `newtxtext`, and `newtxmath`. The paper PDF is a
separate submission file. [INFORMS LaTeX style files](https://pubsonline.informs.org/authorportal/latex-style-files)

```bash
python restore_review_inputs.py informs --zip opre-template.zip
(cd SC-OLH-KG/manuscript_v2_or &&
  pdflatex -interaction=nonstopmode -halt-on-error main_or.tex && bibtex main_or &&
  pdflatex -interaction=nonstopmode -halt-on-error main_or.tex &&
  pdflatex -interaction=nonstopmode -halt-on-error main_or.tex &&
  pdflatex -interaction=nonstopmode -halt-on-error main_or.tex)
```

Use Elan and Git for the pinned Lean/Mathlib verification.

```bash
cd proof
lake exe cache get
lake -Kjobs=1 build
lake env lean SCOLHKG/V2ORProofInterface.lean
```

# GP-FTA Operations Research replication bundle

This controlled archive packages the LaTeX manuscript (`SC-OLH-KG/manuscript_v2_or/main_or.tex`), active `SC-OLH-KG/v2_or` software (`v2_or_eckg_sure_confirm_v1`), frozen historical snapshots (`frozen_referee/`, `frozen_independent/`, `frozen_u3/`), mechanized Lean 4.31.0 proofs (`proof/`), third-party benchmark inputs (`SC-OLH-KG/data/`), and all sealed CSV ledgers and JSON summaries (`SC-OLH-KG/paper_artifacts/`).

## Verification, unit tests, table regeneration, and LaTeX build

Run from the extracted bundle root on Linux x86_64 with CPython 3.10+, Git, and TeX Live:

```bash
python3 SC-OLH-KG/manuscript_v2_or/export_clean_replication_bundle.py --verify-dir .
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
(cd SC-OLH-KG && PYTHONPATH=. python -m pytest -q v2_or/tests)
python verify_inputs.py

report_dir=$(mktemp -d)
cp -a SC-OLH-KG/manuscript_v2_or/. "$report_dir/"
python SC-OLH-KG/manuscript_v2_or/build_u4_tables_and_figures.py --output-dir "$report_dir"
(cd "$report_dir" && pdflatex -interaction=nonstopmode -halt-on-error main_or.tex &&
  bibtex main_or && pdflatex -interaction=nonstopmode -halt-on-error main_or.tex &&
  pdflatex -interaction=nonstopmode -halt-on-error main_or.tex &&
  pdflatex -interaction=nonstopmode -halt-on-error main_or.tex)
```

## Live SUMO adapter and Lean 4.31.0 proof verification

```bash
python -m pip install -r requirements-live-sumo.txt
export SUMO_HOME="$(python -c 'from sumo import SUMO_HOME; print(SUMO_HOME)')"
INGOLSTADT21_SUMO_BACKEND=libsumo python verify_inputs.py --live-sumo

cd proof
lake exe cache get
lake -Kjobs=1 build
lake env lean SCOLHKG/V2ORProofInterface.lean
```

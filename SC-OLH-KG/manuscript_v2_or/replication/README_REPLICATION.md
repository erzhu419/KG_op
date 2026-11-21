# GP-FTA Operations Research replication bundle

This controlled archive accompanies *Certification-Aligned Search and Transferable Spectral Priors in Chance-Constrained Simulation Optimization*. It packages the LaTeX manuscript (`SC-OLH-KG/manuscript_v2_or/main_or.tex`), the sealed Python implementation (`SC-OLH-KG/v2_or`), historical snapshots (`frozen_referee/`, `frozen_independent/`, `frozen_u3/`), mechanized Lean 4.31.0 proofs (`proof/`, 22 verified theorems), third-party benchmark inputs (`SC-OLH-KG/data/`), and all sealed ledgers, compact Stage-2 receipts, and fitted source priors (`SC-OLH-KG/paper_artifacts/`). The grid accounting update for subsequent runs is in `maintenance/SC-OLH-KG/v2_or/simulators/opsd_storage.py`.

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

## Rebuild fine-grained readouts with zero simulation calls

The commands below read the included CSV ledgers and compact JSON receipts directly and make zero simulation calls:

```bash
prosp=SC-OLH-KG/paper_artifacts/v2_or_eckg_sure_confirm_v1
mp_opsd=SC-OLH-KG/paper_artifacts/v2_or_metaprior_opsd_confirm_v1
rev7=SC-OLH-KG/paper_artifacts/v2_or_rev7_or_panel_v1
diag_dir=$(mktemp -d)

PYTHONPATH=SC-OLH-KG:scripts python scripts/run_rev7_closedloop_study.py summarize \
  --out-dir "$rev7" --ledger-csv "$rev7/per_run_ledger.csv" \
  --summary-path "$diag_dir/rev7_closedloop_readout.json"
cmp "$rev7/rev7_closedloop_readout.json" "$diag_dir/rev7_closedloop_readout.json"

PYTHONPATH=SC-OLH-KG:scripts python scripts/run_rev7_prior_and_simopt_study.py summarize \
  --out-dir "$rev7" --summary-path "$diag_dir/rev7_prior_and_simopt_readout.json"
cmp "$rev7/rev7_prior_and_simopt_readout.json" "$diag_dir/rev7_prior_and_simopt_readout.json"

python scripts/run_eckg_reference_validation.py summarize --out-dir "$prosp/eckg_ref_val" \
  --summary-path "$diag_dir/eckg_reference_validation_readout.json"
python scripts/run_metaprior_propagation_audit.py summarize --out-dir "$mp_opsd/propagation_audit" \
  --summary-path "$diag_dir/metaprior_propagation_readout.json"
cmp "$prosp/eckg_reference_validation_readout.json" "$diag_dir/eckg_reference_validation_readout.json"
cmp "$mp_opsd/metaprior_propagation_readout.json" "$diag_dir/metaprior_propagation_readout.json"
python summarize_inventory_covariance.py "$mp_opsd/propagation_audit" \
  --output "$diag_dir/inventory_covariance_readout.json"
cmp "$mp_opsd/inventory_covariance_readout.json" "$diag_dir/inventory_covariance_readout.json"
python scripts/summarize_eckg_reference_correction.py summarize \
  --states "$prosp/eckg_reference_states.json" --output "$diag_dir/eckg_reference_validation_corrected.json"
cmp "$prosp/eckg_reference_validation_corrected.json" "$diag_dir/eckg_reference_validation_corrected.json"
PYTHONPATH=SC-OLH-KG:scripts python scripts/run_eckg_multiscramble_cei_audit.py summarize \
  --out-dir "$prosp/eckg_multiscramble_cei" --states-path "$prosp/eckg_reference_states.json" \
  --summary-path "$diag_dir/eckg_multiscramble_cei_readout.json"
cmp "$prosp/eckg_multiscramble_cei_readout.json" "$diag_dir/eckg_multiscramble_cei_readout.json"
PYTHONPATH=SC-OLH-KG:scripts python scripts/run_direct_propagation_study.py summarize \
  --panel exploratory_86 --out-dir "$mp_opsd/direct_propagation_exploratory_86" \
  --summary-path "$diag_dir/direct_propagation_exploratory_86_readout.json"
cmp "$mp_opsd/direct_propagation_exploratory_86_readout.json" "$diag_dir/direct_propagation_exploratory_86_readout.json"
PYTHONPATH=SC-OLH-KG:scripts python scripts/run_direct_propagation_study.py summarize \
  --panel confirm_fresh_72 --out-dir "$mp_opsd/direct_propagation_confirm_fresh_72" \
  --summary-path "$diag_dir/direct_propagation_confirm_fresh_72_readout.json"
cmp "$mp_opsd/direct_propagation_confirm_fresh_72_readout.json" "$diag_dir/direct_propagation_confirm_fresh_72_readout.json"
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

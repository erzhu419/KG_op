# GP-FTA anonymous review supplement

This archive accompanies *Certification-Aligned Search and Transferable Spectral Priors in Chance-Constrained Simulation Optimization*. It includes Python execution snapshots, Lean 4.31.0 proofs and receipts, run and slot ledgers, task- and state-level diagnostics, and deterministic table and figure builders.

## Included studies

- The sealed implementation under `SC-OLH-KG/v2_or` (`fe8df9098424...`) ran 1,376 distinct core optimization runs: 720 five-arm runs on the 72-task discrete-event simulation panel (`v2_or_eckg_sure_confirm_v1`) plus 432 additional cold-configuration runs and 224 14-market OPSD battery-storage runs (`v2_or_metaprior_opsd_confirm_v1`, pre-registered in `PREREGISTRATION_METAPRIOR_OPSD_20261120.json`). Search, verification, and held-out audit blocks alternate within the historical period.
- Pre-registered frontier, counterfactual, curtailment, calibration, and generic-summary studies (`v2_or_rev7_or_panel_v1`, pre-registered in `PREREGISTRATION_REV7_OR_PANEL_20261121.json`): (i) 1,204 four-domain closed-loop runs across `SCBO-3Slot`, `RGPE-on-Psi-3Slot`, `WarmStart-MTGP-KG-3Slot`, `GP-FTA-LatentNEI-3Slot`, `GP-FTA-AllowUnvisited`, `GP-FTA-DirectMaxU`, and `GP-FTA-LongSearch-k0`, (ii) 172 four-level exact binomial gate-stringency and sequential-curtailment evaluations (`k* in {0, 1, 2, 3}`), (iii) 86 offline feasible-oracle benchmarks (`Ref-Oracle`), (iv) 240 closed-loop runs on the 24-task four-channel `SimOpt_TandemCntNv` tandem queueing-newsvendor suite using strictly simulator-free black-box policy-profile summaries `psi_generic(x)`, (v) 24-state 24-scramble `LatentNEI` and HVD-buffer sensitivity diagnostics, and (vi) 72-task unseen-DES gate-power calibration, meta-BO baselines (`ABLR-on-Psi`, `MTGP-ICM-on-Psi`, `RGPE-on-Psi`), leave-one-pillar-out ablations, public-forecast corruption (`ForecastNoise-15%`), shift-scale sensitivity (`{1.0, 1.3, 1.6}`), and source-count scaling (`M_src in {3, 6, 12}`).
- Earlier diagnostic receipts cover (i) static prediction after eight observations on 86 tasks (`propagation_audit`), (ii) direct single-probe cross-policy propagation on both the 86-task panel (`direct_propagation_exploratory_86`) and 72 newly generated unseen DES tasks (`direct_propagation_confirm_fresh_72`, pre-registered in `PREREGISTRATION_REV5_PROPAGATION_CONFIRM_20261121.json`), (iii) a separate 96-state query study, and (iv) the 24-state re-evaluated full-bank reference winner and sampled-incumbent `PlugIn-cEI` comparison across 24 independent `256 x 1,024` Sobol scrambles (`eckg_multiscramble_cei`, stored under key `UB-cEI` in the JSON receipt) alongside `eckg_reference_states.json`. All six compact fitted source priors (`SC-OLH-KG/paper_artifacts/v2_or_u3_confirm_v1/source_bundles/*/prior_default.pkl` plus `SC-OLH-KG/paper_artifacts/v2_or_rev7_or_panel_v1/simopt_prior_default.pkl`) are bundled directly in this supplement.
- `maintenance/SC-OLH-KG/v2_or/simulators/opsd_storage.py` contains the audit-purpose accounting update. The sealed execution sources remain under `SC-OLH-KG/v2_or`. Compare `v2_or_metaprior_opsd_confirm_v1/metaprior_opsd_panel_readout.json` with `as_reported/metaprior_opsd_panel_readout.json` in the same study directory for the original classification.
- Frozen execution snapshots `frozen_referee/SC-OLH-KG` (384-run matched coefficient-prior and terminal-replay study `v2_or_gpfta_referee_ablation_v1`), `frozen_independent/SC-OLH-KG` (432-run single-slot quadrature panel `v2_or_kg_quadrature_confirm_v1`), and `frozen_u3/SC-OLH-KG` (9,796-run four-domain benchmark across `v2_or_u3_confirm_v1`, `v2_or_u3_messy_archive_v1`, and `v2_or_u3_live_sumo_v1`).
- Mechanized Lean 4.31.0 modules under `proof/SCOLHKG/V2OR/` (`DebiasedSubspace.lean`, `TemperedHVD.lean`, `FunctionalTransferRisk.lean`, `VerificationKGAmortization.lean`, `SequentialTheory.lean`) and `proof/SCOLHKG/V2ORProofInterface.lean` verifying all 22 audited theorems with zero `sorry`.
- Third-party provenance and restoration recipes in `LICENSES.md` and `restore_review_inputs.py`, plus raw-archive audit documentation in `U3_REPLAY.md`.

## 1. Verify archive contents and regenerate all 24 tables and 3 figures

Run from the extracted supplement root on Linux x86_64 with CPython 3.10+. Every manuscript and Electronic Companion table (`24` `.tex` tables in `tables/`) and vector figure (`3` `.pdf` figures in `figures/`) regenerates deterministically from the sealed CSV ledgers and compact JSON readouts with zero simulation calls:

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

## 2. Rebuild prospective, frontier, counterfactual, calibration, and query-diagnostic readouts

Use a temporary output directory to recompute the published readouts from the included ledgers and compact diagnostic receipts. These commands make zero simulation calls.

```bash
readout=$(mktemp -d)

# 720-run five-arm ablation readout (Tables 2-3, Table 5, Figure 1a)
prosp=SC-OLH-KG/paper_artifacts/v2_or_eckg_sure_confirm_v1
python summarize_prospective_panel.py --campaign-dir "$prosp" --output "$readout/prospective_panel_readout.json"
cmp "$prosp/prospective_panel_readout.json" "$readout/prospective_panel_readout.json"

# 656-run cold-configuration and 336-dimensional grid study (Table 7)
mp_opsd=SC-OLH-KG/paper_artifacts/v2_or_metaprior_opsd_confirm_v1
python summarize_metaprior_opsd_panel.py --campaign-dir "$mp_opsd" --prospective-dir "$prosp" \
  --output "$readout/metaprior_opsd_panel_readout.json"
cmp "$mp_opsd/metaprior_opsd_panel_readout.json" "$readout/metaprior_opsd_panel_readout.json"

# Frontier baselines, feasible-oracle gaps, eligibility/nominator counterfactuals, and gate curtailment (Tables 4A, 6)
rev7=SC-OLH-KG/paper_artifacts/v2_or_rev7_or_panel_v1
PYTHONPATH=SC-OLH-KG:scripts python scripts/run_rev7_closedloop_study.py summarize \
  --out-dir "$rev7" --ledger-csv "$rev7/per_run_ledger.csv" \
  --summary-path "$readout/rev7_closedloop_readout.json"
cmp "$rev7/rev7_closedloop_readout.json" "$readout/rev7_closedloop_readout.json"

# SimOpt_TandemCntNv generic-psi benchmark, 72-task calibration/pillars/M_src scaling, and 24-state LatentNEI (Tables 4B, 8B, 9, EC.1-EC.2)
PYTHONPATH=SC-OLH-KG:scripts python scripts/run_rev7_prior_and_simopt_study.py summarize \
  --out-dir "$rev7" --summary-path "$readout/rev7_prior_and_simopt_readout.json"
cmp "$rev7/rev7_prior_and_simopt_readout.json" "$readout/rev7_prior_and_simopt_readout.json"

# Verify the maintenance fix for per-pool metering on a disposable copy of the simulator
meter_test=$(mktemp -d)
cp -a SC-OLH-KG/v2_or "$meter_test/v2_or"
cp maintenance/SC-OLH-KG/v2_or/simulators/opsd_storage.py "$meter_test/v2_or/simulators/"
PYTHONPATH="$meter_test:SC-OLH-KG" python -m pytest -q maintenance/SC-OLH-KG/tests/test_opsd_audit_accounting.py

# Per-task contributions to the 86-task propagation and representation audit
python scripts/run_metaprior_propagation_audit.py summarize \
  --out-dir "$mp_opsd/propagation_audit" --summary-path "$readout/metaprior_propagation_readout.json"
cmp "$mp_opsd/metaprior_propagation_readout.json" "$readout/metaprior_propagation_readout.json"
python summarize_inventory_covariance.py "$mp_opsd/propagation_audit" \
  --output "$readout/inventory_covariance_readout.json"
cmp "$mp_opsd/inventory_covariance_readout.json" "$readout/inventory_covariance_readout.json"

# Historical single-nomination reference (12 task receipts, two search states each)
python scripts/run_eckg_reference_validation.py summarize \
  --out-dir "$prosp/eckg_ref_val" --summary-path "$readout/eckg_reference_validation_readout.json"
cmp "$prosp/eckg_reference_validation_readout.json" "$readout/eckg_reference_validation_readout.json"

# Corrected 24-state comparison and 24-scramble PlugIn-cEI audit (Table 9, Table EC.3, Figure 1b)
python scripts/summarize_eckg_reference_correction.py summarize \
  --states "$prosp/eckg_reference_states.json" --output "$readout/eckg_reference_validation_corrected.json"
cmp "$prosp/eckg_reference_validation_corrected.json" "$readout/eckg_reference_validation_corrected.json"

PYTHONPATH=SC-OLH-KG:scripts python scripts/run_eckg_multiscramble_cei_audit.py summarize \
  --out-dir "$prosp/eckg_multiscramble_cei" --states-path "$prosp/eckg_reference_states.json" \
  --summary-path "$readout/eckg_multiscramble_cei_readout.json"
cmp "$prosp/eckg_multiscramble_cei_readout.json" "$readout/eckg_multiscramble_cei_readout.json"

# Direct single-probe cross-policy propagation (Table 8A: confirmatory 72 + Table EC.2A: exploratory 86)
PYTHONPATH=SC-OLH-KG:scripts python scripts/run_direct_propagation_study.py summarize \
  --panel exploratory_86 --out-dir "$mp_opsd/direct_propagation_exploratory_86" \
  --summary-path "$readout/direct_propagation_exploratory_86_readout.json"
cmp "$mp_opsd/direct_propagation_exploratory_86_readout.json" "$readout/direct_propagation_exploratory_86_readout.json"

PYTHONPATH=SC-OLH-KG:scripts python scripts/run_direct_propagation_study.py summarize \
  --panel confirm_fresh_72 --out-dir "$mp_opsd/direct_propagation_confirm_fresh_72" \
  --summary-path "$readout/direct_propagation_confirm_fresh_72_readout.json"
cmp "$mp_opsd/direct_propagation_confirm_fresh_72_readout.json" "$readout/direct_propagation_confirm_fresh_72_readout.json"

# 384-run matched coefficient-prior readout (Tables EC.6-EC.9)
ref_dir="$readout/referee"
cp -a SC-OLH-KG/paper_artifacts/v2_or_gpfta_referee_ablation_v1 "$ref_dir"
PYTHONPATH=frozen_referee/SC-OLH-KG:SC-OLH-KG:scripts python -m v2_or.dispatch.summarize_gpfta_referee \
  --manifest frozen_referee/SC-OLH-KG/v2_or/manifests/v2_or_gpfta_referee_ablation_v1.json \
  --campaign-dir "$ref_dir"
cmp SC-OLH-KG/paper_artifacts/v2_or_gpfta_referee_ablation_v1/referee_ablation_readout.json \
    "$ref_dir/referee_ablation_readout.json"

# 432-run single-slot quadrature readout (Tables EC.4-EC.5)
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

### Optional replay of the 24 corrected states

Use a fresh copy of the supplement on a compute worker and install the dependencies above; both the bundled fitted priors (`SC-OLH-KG/paper_artifacts/v2_or_u3_confirm_v1/source_bundles/{Queueing_CrossHorizon,Traffic_Ingolstadt}/prior_default.pkl`) and `SC-OLH-KG/data/ingolstadt21/decision_space.json` are already in place. This replay spends search calls and integration time. The routine summary commands in Section 2 read the saved receipts directly.

```bash
git apply scripts/eckg_reference_diagnostic.patch
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
prosp=SC-OLH-KG/paper_artifacts/v2_or_eckg_sure_confirm_v1
replay=$(mktemp -d)
python scripts/run_eckg_reference_validation.py describe --out-dir "$replay"
python scripts/run_eckg_reference_validation.py worker --out-dir "$replay" --start 0 --end 12 --workers 1
python scripts/summarize_eckg_reference_correction.py pack --corrected-dir "$replay" \
  --original-dir "$prosp/eckg_ref_val" --output "$replay/states.json"
python scripts/summarize_eckg_reference_correction.py summarize \
  --states "$replay/states.json" --output "$replay/corrected.json"
```

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

Verify all 22 mechanized theorems in Lean 4.31.0 (`proof/lean-toolchain` and `proof/lake-manifest.json`):

```bash
cd proof
lake exe cache get
lake -Kjobs=1 build
lake env lean SCOLHKG/V2ORProofInterface.lean
```

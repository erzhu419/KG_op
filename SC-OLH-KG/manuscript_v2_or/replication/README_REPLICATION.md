# GP-FTA Operations Research review artifact

The manuscript is `SC-OLH-KG/manuscript_v2_or/main_or.tex`. Its tables draw on
archived campaigns in `SC-OLH-KG/paper_artifacts/`. Per-run ledgers, per-slot
certificates, compact summaries, and available records of historical expenditure
are copied verbatim. The U3 Python snapshot in `frozen_u3/` matches the execution
fingerprint cited by each U3 compact summary. That fingerprint covers `v2_or`
Python files. The live traffic cohort also used the external SUMO runtime, legacy
adapter, and scenario inputs. `SC-OLH-KG/v2_or/` is the corrected implementation
used by the independent `v2_or_kg_quadrature_confirm_v1` panel. The integrity
check binds its summary to the bundled Python and the preregistered manifest,
and matches P1, P2, U1, U3, and independent-panel CSVs to their recorded hashes.
When a local CSV uses LF, it checks the original CRLF form in memory and keeps
the packaged file byte-for-byte. `LICENSES.md` records the asset provenance.

## Python, table regeneration, and LaTeX

Use Linux x86_64, CPython 3.10, `venv`, Git, and a TeX Live installation with
pdfLaTeX, BibTeX, and standard LaTeX packages (including `mathtools`,
`booktabs`, `multirow`, `microtype`, `placeins`, `natbib`, `hyperref`, `tgtermes`,
`newtxtext`, and `newtxmath`).
Run these commands from the extracted bundle root. The verification script
checks file integrity and the recorded U3 execution binding. The subsequent
commands run deterministic wiring tests and compile the regenerated manuscript
in a temporary directory. The commands preserve the sealed ledgers.

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

The manuscript's revised tables, figures, and PDF are also included. PDFs can
have different metadata across regeneration builds.

## SUMO dependency and data verification

The bundled scenario preserves the exact project copies of the network, route,
decision space, and baseline constants. Install the published SUMO 1.22.0 wheels
from the bundle root. Explicit `libsumo` selection loads the installed wheel
ahead of a system SUMO tools directory. The adapter check loads the calibrated
U3 task and verifies the engine-reported version, network, and route without
launching a simulation.

```bash
python -m pip install -r requirements-live-sumo.txt
export SUMO_HOME="$(python -c 'from sumo import SUMO_HOME; print(SUMO_HOME)')"
INGOLSTADT21_SUMO_BACKEND=libsumo python verify_inputs.py --live-sumo
```

The 1.22.0 `eclipse-sumo` wheel installs on this platform. Pip 26 can flag its
embedded `cp38` tag in `pip check` even when the 3.10 `libsumo` extension loads
and reports the expected engine version. The adapter check above verifies the
runtime the code actually imports.

The OPSD compact array is included. Its source metadata describes the official
CSV, markets, years, and interpolation settings. `SC-OLH-KG/performance/prepare_opsd_energy_data.py`
accepts the same pinned release URL (`--source`) and comma-separated manifest
values for `--markets` and `--years` with `--interpolation-limit 6`.

## Lean proof verification

Install [Elan](https://github.com/leanprover/elan), which selects Lean 4.31.0
from `proof/lean-toolchain`. Run the following from the bundle root. Lake
uses the committed dependency revisions in `proof/lake-manifest.json`.
Downloading the Mathlib cache is optional and avoids rebuilding dependencies.

```bash
cd proof
lake exe cache get
lake -Kjobs=1 build
lake env lean SCOLHKG/V2ORProofInterface.lean
```

## Scope

The package supports independent table regeneration, deterministic code checks,
LaTeX compilation, and Lean compilation. The independent panel's compact
summary and paired readout are in `SC-OLH-KG/paper_artifacts/v2_or_kg_quadrature_confirm_v1/`.
Each arm has the same search allocation and terminal cap. The ledgers record
actual terminal calls under early stopping and keep independent audit calls
separate. The frozen U3 source archives and fitted priors were reused.

The historical U3 per-run CSVs omit policy vectors. The original search and
verification receipts, selected policies, audit receipts, source trajectories,
and fitted priors remain on the execution cluster. `read_u3_replay.py` reads
them in place and supports rebuilding historical CSVs from the saved audit
records without new simulation. The historical replay receipt and access
instructions are included. The separately staged reviewer-input archive contains
the full saved U3 source, search, verification, and audit working materials.
It remains on the execution cluster for controlled access. It reuses the external
benchmark setup described here. See `SC-OLH-KG/docs/gp_fta_u3_historical_replay_20261120.md`.
Exact experimental re-execution requires the archived inputs and a separate
run directory. The U3 Python fingerprint excludes the external live SUMO
runtime and adapter. The local historical U3 messy-archive `search_and_verify/jobs.json`
ends mid-record. Its sibling `jobs_reconstructed.json` supplies the full
2,108-job roster regenerated from the sealed manifest and checked against the
ledger. The historical fragment is the exact prefix of that file.

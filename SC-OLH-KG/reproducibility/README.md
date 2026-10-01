# Reproducing the current Operations Research manuscript

All commands below start at the repository or unpacked attachment root.
The current method is `profile_atlas_v2_method_spec.json`, read with
[`current_protocol_errata_20260907.md`](current_protocol_errata_20260907.md).
`paper_artifacts/or_review/final_evidence_registry_v1.json` identifies the 14
completed historical matrices; `paper_artifacts/submission_revision_20260907/`
contains the matched-standardization and same-source portfolio comparisons.
The September 30 protocol is `performance/manifests/submission_revision_20260930.json`;
its completed evidence is `paper_artifacts/submission_revision_20260930/`.
It adds mechanism attribution, 240 new latent tasks from three new source
families and public libraries, library-size diagnostics, and corrected Energy V5 in the separate
`paper_artifacts/submission_revision_energy_v5_20260930/` directory. Its protocol
is `performance/manifests/submission_revision_energy_v5_20260930.json`.
October 2 postdecision controls and decision-chain diagnostics are included
separately; their rules do not replace the original method or confirmation.

## Environment

The experiment environment used Python 3.10.12. Package pins are in
`requirements-core.txt`, `requirements-botorch.txt` and
`requirements-transfer-overlays.txt` in this directory. The core requirements support statistical
reanalysis and rendering, including the `requests` import used by Energy data loading. Functional
SCBO runs need the BoTorch environment. Native transfer uses the separately
pinned official-method dependencies; GPy/Emukit and HyperBO use separate overlay
environments because their dependency requirements differ.

The proof toolchain is Lean 4.31.0 (`proof/lean-toolchain`). PDF builds require
LaTeX, BibTeX and `latexmk`; the INFORMS class and bibliography style are included.

```bash
export PYTHONPATH="$PWD/SC-OLH-KG${PYTHONPATH:+:$PYTHONPATH}"
```

## Compact per-cell evidence and statistical replay

The anonymous attachment includes `SC-OLH-KG/reproducibility/cells/*.jsonl`:
17,360 original experiment cells projected to analysis inputs. Each line has
`relative_result_path` and `cell`. Task IDs, seed IDs, arm, dimensions,
configuration, scalar outcomes, nested deployment truth and verifier counts,
oracle objective, costs and the reported failure are retained. The projection
omits policy/coefficient vectors, search histories and workstation metadata.
The synthetic revision's `rows.json` and Energy revision's `energy_rows.json`
are already small per-task records and are included directly. September 30
adds `profile_rows.json` (4,160 rows). Current V5 has 720 `energy_rows.json`
records in its separate directory. The retrospective monthly reuse V2 has
5,400 rows under `paper_artifacts/energy_archive_reuse_v2_20260930/`. The reused
160-task diagnostic JSON is also included at the path named by the supplemental
protocol, so its original outcomes need not be recollected.

To run the existing statistical analyzers against the attachment, expand the
projected cells into the attachment's initially absent `results/` directory
(skip this step in a working repository where the original results exist):

```bash
python3 SC-OLH-KG/performance/export_submission_revision_bundle.py \
  --materialize-cells SC-OLH-KG/reproducibility/cells \
  --results-root SC-OLH-KG/results
```

Set the input root to those projected records, or to existing full local results:

```bash
export OR_RESULTS=SC-OLH-KG/results
export OR_ANALYSIS=SC-OLH-KG/reproducibility/replayed_analyses
mkdir -p "$OR_ANALYSIS"
```

Recompute the four profile matrices, preserving all task-level paired inputs:

```bash
python3 - <<'PY'
import json, os
from pathlib import Path
from performance.analyze_profile_stress_suite import analyze
root, out = Path(os.environ['OR_RESULTS']), Path(os.environ['OR_ANALYSIS'])
for name in ('primary', 'sensitivity', 'schema_descriptor', 'equal_preverification_cost'):
    paths = sorted((root / 'or_review_v2_evalpatch4' / ('profile_' + name)).rglob('cell*.json'))
    payload = analyze(paths)
    (out / ('profile_' + name + '.json')).write_text(json.dumps(payload, indent=2) + '\n')
    print(name, payload['status'], payload['row_count'])
PY
```

Pair synthetic arms by mechanism, latent-task seed, resolution and configuration.
The 160 latent tasks crossed with three resolutions are 480 correlated
resolution-task cells. Overall revision bootstrap resamples tasks within each
of the eight fixed mechanisms. The new confirmation instead has 240 latent tasks at d=1000, stratified into
24 fixed family/mechanism groups with ten tasks each. Its 10,000-resample paired
bootstrap is conditional on the three fixed families; results are also reported
separately by family. For Energy, five algorithm seeds within a
market measure repeatability; geographic-region and market analyses preserve
that structure.

```bash
python3 SC-OLH-KG/performance/analyze_external_energy_v3.py \
  "$OR_RESULTS/or_review_energy_forecast_indexed_v3" \
  --expected-count 540 --out "$OR_ANALYSIS/energy_v3.json"
python3 SC-OLH-KG/performance/analyze_native_transfer_matrix.py \
  "$OR_RESULTS/or_review_native_transfer_official_d1000_n13_s80_99_v1" \
  --out "$OR_ANALYSIS/native_transfer.json"
python3 SC-OLH-KG/performance/analyze_functional_profile_scbo.py \
  --functional-root "$OR_RESULTS/or_review_functional_scbo_v2/primary" \
  --profile-root "$OR_RESULTS/or_review_v2_evalpatch4/profile_primary" \
  --out "$OR_ANALYSIS/functional_primary.json"
python3 SC-OLH-KG/performance/submission_revision_controls.py --analyze-only
python3 SC-OLH-KG/performance/submission_revision_energy.py --analyze-only
python3 SC-OLH-KG/performance/submission_revision_attribution.py --analyze-only
python3 SC-OLH-KG/performance/submission_revision_energy_v5.py --analyze-only
```

For functional rank sensitivity, use `rank_sensitivity` with the same primary
profile root. For functional equal-cost analysis, use
`equal_preverification_cost` and `profile_equal_preverification_cost` together;
its one algorithmic failure remains an unsuccessful primary outcome.
Energy V2 uses `analyze_external_energy_v2.py` on
`or_review_v1/energy_region_holdout` and `or_review_energy_functional_scbo_v2`.
V2, historical V3/V4 and corrected V5 are distinct experiments and are never pooled.

Temporal JSONL records retain each certified policy's chronological-block,
nonoverlapping-window and sampled-distribution statistics in `temporal_audit`.
The mean/minimum and threshold counts can be recomputed from these fields;
`analyze_external_energy_temporal_audits.py:_summary` specifies the aggregations.
Their role is descriptive stability after the decision has been frozen.

## Re-executing the fixed synthetic matrices

Existing scalar evidence suffices for statistical replay. To generate fresh
simulation records under the historical fixed task definitions, use the
registered method freeze and evaluation commit labels:

```bash
python3 SC-OLH-KG/performance/run_profile_stress_matrix.py \
  --matrix primary --dimensions 200,1000,10000 --task-count 20 --workers 2 \
  --freeze-commit da8f1e5c594dace1cc667a2e4b87956b1001b67b \
  --evaluation-commit d514b61c98cb34cecf13e42ada1d262722a1def4 \
  --output-dir SC-OLH-KG/results/reproduction/profile_primary
```

For `sensitivity`, `schema_descriptor` and `equal_preverification_cost`, set
`--matrix` to that name and `--dimensions 1000`, and choose a separate output
directory. Expected counts are 2,880, 8,640, 1,280 and 800 respectively.
These identifiers reproduce task construction and record the historical
protocol; a rerun using revised source files is a new run.

The added comparison entrypoints fix their protocol before producing added
outcomes and reuse the existing source-method scalar records:

```bash
python3 SC-OLH-KG/performance/submission_revision_controls.py --workers 2
python3 SC-OLH-KG/performance/submission_revision_energy.py --workers 2
```

The functional runner is `run_functional_profile_scbo_matrix.py`; its exact
primary/rank/equal-cost settings and execution labels are in
`or_review_functional_profile_scbo_v2.json`. Native transfer is a separate
480-cell end-to-end comparison in `or_review_native_transfer_execution_v1.json`,
with `benchmark_transfer_fairness.py` and the official-method overlays. Its
13-call search and v69 80/128/128 verifier plus 8-call objective comparison are
different from the profile and Energy 80/80/80 binomial verifier.

## External Energy data and corrected V5 execution

The data are Open Power System Data `time_series`, release `2020-10-06`,
DOI [10.25832/time_series/2020-10-06](https://doi.org/10.25832/time_series/2020-10-06).
The attachment includes the existing 5.6 MiB
`SC-OLH-KG/data/external/opsd_time_series_extended_v2.npz` and its preprocessing
manifest. It contains 21 markets for 2017–2019; V3 uses the 18 registered target
markets across five geographic regions. V5 retains these markets and periods. Reuse this local archive.

To recreate it when needed, pass the full market list. The four-market default
preprocessing command belongs to the older experiment:

```bash
python3 SC-OLH-KG/performance/prepare_opsd_energy_data.py \
  --source /path/to/time_series_60min_singleindex.csv \
  --markets AT,DE_LU,DK_1,DK_2,GB_GBN,IE_sem,IT_CNOR,IT_CSUD,IT_NORD,IT_SARD,IT_SICI,IT_SUD,NO_1,NO_2,NO_3,NO_4,NO_5,SE_1,SE_2,SE_3,SE_4 \
  --years 2017,2018,2019 --interpolation-limit 6 \
  --out SC-OLH-KG/data/external/opsd_time_series_extended_v2.npz
```

Omitting `--source` streams the pinned official CSV. The recorded existing
manifest gives the raw and compact file identities. The raw 130 MB CSV is not
part of the attachment. V3 uses forecast stress to define the 1,000-point
state-of-charge response profile and a separate 168-hour physical horizon.
Search, audit and verification periods are 2017, 2018 and 2019.

V5 starts every policy at 50% battery capacity and charges subsequent grid
purchases. The objective is the declared 168-hour service cost with zero
terminal salvage. Reserve adjustment runs first; forecast-error balancing uses
its remaining hourly power. The sum of external charge and discharge energy
is bounded by power capacity times one hour. Five regional source archives are each frozen once and
shared by the actual target markets outside their source regions. Algorithm
seeds are repetitions, not additional deployments. This reruns all eight arms
(720 cells), including the functional control, and retains the negative
external cost comparison.

```bash
python3 SC-OLH-KG/performance/submission_revision_energy_v5.py \
  --workers 2 --out SC-OLH-KG/results/reproduction_energy_v5
```

The default output is the current evidence directory; completed market files
are resumed. A fresh `--out` executes the corrected experiment separately.
The BoTorch overlay is needed for the functional arm. Historical V3 simulator
execution requires the original September 7 source attachment because its
initial inventory was policy-dependent. Historical V4 execution requires the
original September 30 source attachment: that model allocated hourly power
separately to reserve adjustment and balancing. Current scalar analyzers replay
V3/V4 recorded outcomes; current physical runs use V5. The older supplemental
runner permits statistical replay only in the current source release.

The same-market archive reuse study fixes 2017 observable scales and the same
storage asset across twelve monthly tasks per market. Two historical 2017
half-years build each market's archive. Search uses the corresponding 2018
month and independent window sampling uses the corresponding 2019 month.
All 18 archives freeze before target evaluation. Five algorithm repeats give
5,400 rows across five designs and 216 actual market-month tasks; repeats are
averaged within each task for archive-reuse cost. Postdecision truth enumeration
is recorded separately from search and verification calls. Reproduction uses
the frozen `energy_archive_reuse_v2_20260930.json` protocol:

```bash
python3 SC-OLH-KG/performance/benchmark_energy_archive_reuse.py --phase analyze
```

A fresh run requires `--phase freeze`, then `--phase targets --workers 2`, using
the same new `--out` directory for both commands. This study uses previously
examined OPSD data and is retrospective. Its outputs remain separate from the
manuscript's regional external cost comparison. The interrupted reuse V1 is invalid
for corrected-model analysis and has a retained local cost/disposition record.
Method deployment-cost calculations do not include discarded debugging runs.

For attribution and prospective synthetic confirmation:

```bash
python3 SC-OLH-KG/performance/submission_revision_attribution.py --workers 2
```

This resumes existing `profile_groups/` files. In an unpacked attachment those
intermediate files are absent, so simulations are executed, with previously
recorded outcomes reused only for identical ordered designs on the original
tasks. The 4,160 analysis rows comprise 2,400 attribution, 1,440 prospective
confirmation and 320 library-size rows. The recorded run executed 2,360 new
arm/task evaluations and reused 1,800 identical-design outcomes. The local
protocol was frozen before these new outcomes; it was not a public registration.

## Analyses, rendering, manuscript and proof

The dependency order is cell outcomes → task/market paired analysis → compact
analysis → table/figure renderer → LaTeX PDFs. The existing compact analyses
under `paper_artifacts/or_review/` preserve the original aggregate endpoints;
the separate revision artifacts add the corrected comparisons.

To compact any newly generated full analysis, use
`compact_or_review_analysis.py --analysis INPUT --out OUTPUT`. Write replayed
analyses to a separate directory; the historical frozen registry records the
original analyses. Current displays can be rendered directly from the included
compact evidence and revision analyses:

```bash
python3 SC-OLH-KG/performance/render_submission_revision_20260930.py
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error SC-OLH-KG/manuscript/main.tex
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error SC-OLH-KG/manuscript/supplement.tex
(cd proof && lake --no-cache build SCOLHKG.PaperProofInterface)
```

The Lean target builds the original paper interface and its dependencies.
The new rank/design-stability proposition has an analytic proof in the
companion and is outside that interface. Current tables use September 30
analyses; unchanged figures use the original compact analyses. To redraw those
figures, first run `render_or_review_final_artifacts.py`, then run the current
renderer to restore V5 and confirmation tables. The historical HVD table is
regenerated by `render_hvd_diagnostic.py` from six aggregate CSV rows.
Source, initial target, adaptive target, verification, total and amortized
calls remain distinct. Calls per successful certification and call-count
amortization break-even are separately reported.

## October 2 postdecision controls

`paper_artifacts/source_first_remainders_v1_20261002/` contains the frozen
plan, 250 newly evaluated compact rows and the summary; 230 unchanged ordered
portfolio designs reuse original outcomes in the plan. Recompute the paired
summary from these records without any simulations:

```bash
python3 SC-OLH-KG/performance/compare_source_first_remainders.py collect \
  --new-rows SC-OLH-KG/paper_artifacts/source_first_remainders_v1_20261002/new_rows.json
```

The collector joins the original confirmation and included decision-chain
rows, retaining every task and outcome. It also writes the combined 1,440
rows locally. `reaudit_profile_decision_chain.py` reconstructs original noisy
shortlists and verification streams, checking all original decisions; it
collects no new independent evidence. `diagnose_first_center_margin_transfer.py`
reconstructs the original source streams and evaluates deterministic generator
truth for the separate 240-task margin diagnostic. No 2025 data are used.

Public-battery artifacts include both 30-run replay tables, frozen designs,
source accounting, and full/constant-library census summaries. These support
per-run certification and aggregate cost reanalysis. Window-level replay uses
large source/target population journals retained on the server; the compact
attachment does not contain them. The included battery replay code and
protocols document that computation.

## Build the anonymous attachment

After the revised analyses and both PDFs are complete, run:

```bash
python3 SC-OLH-KG/performance/export_submission_revision_bundle.py
```

This writes `SC-OLH-KG/reproducibility/dist/or_submission_revision_controls_20261002.zip`
from local files. It includes the current code dependency closure, registered
protocols and erratum, compact per-cell evidence, revision outcomes, compact
OPSD archive, manuscript sources/PDFs and Lean source/configuration. Its
`bundle_inventory.json` lists the files and per-matrix cell counts. The export
performs no upload. If the unchanged historical compact cells already exist
in the September 7 attachment, pass `--reuse-cells-from PATH_TO_OLD_ZIP` to copy
those cells without reading full result histories.

## Limitations and historical records

Projected cells reproduce statistical inputs, not original raw-result bytes.
Historical hashes/receipts identify the original full records; raw-evidence
hash audits and policy-level temporal replay require those originals.
Search trajectories, policy vectors, model weights, checkpoints and the raw
OPSD CSV are excluded. The 17,360-record package preserves all historical
registered cells, including the reported algorithmic failure. Added comparisons
on existing tasks are post-review evidence. The September 30 confirmation
uses new seeds, families and libraries within the same eight declared synthetic
mechanisms; it is not an independent external application. Energy V5 corrects
and reuses the existing 18-market data. The three extra local markets have
insufficient complete windows in the declared yearly splits, so they are not
added as target evidence. Actual archive-reuse costs use ten targets per
synthetic archive and the observed markets per Energy archive; per-row
`all_in_calls_amortized` retains the older hypothetical M=20 diagnostic.
The HVD attachment contains aggregate inputs for table regeneration, without
its 120 historical seed-level records.

Older `paper_final_method_v1.json`, `paper_submission_experiment_registry_v1.json`,
`external_energy_reliability_v1.json` and positive Energy-confirmation results
remain historical. Current evidence does not use them as the paper's method,
registry or external-validity claim. Author, funding, cover-letter and journal
submission metadata are prepared separately by the authors.

# Current mainline: fixed review attachment

This is the default reproduction entry for the October 26, 2026 delivery. The
method is the historical learned frontend, source-learned role prior, AFTER,
constrained Thompson sampling, and the independently sampled old terminal. The
reviewed base is `d3fac7622462d50c2a048833b91b33bdcccf2dab`. The archive manifest
identifies the delivery source and every included payload by SHA256. The actual
historical experiment source commit is unknown in the retained receipts; the
reviewed base must not be presented as a newly executed scientific cohort.

The package keeps Stage85 (2,592 cells) and Stage93 (3,168 cells, including 67
retained failures) separate. Stage103 comparisons reuse those sealed samples.
Historical Atlas/OR materials remain in the original repository and are
separately identified; they are not this attachment's reproduction target.

## Ordinary single-machine use

Extract the archive into a new directory. The locked acceptance runtime is
**Python 3.10.14**. From the archive root, create an environment with that Python
and install the pinned runtime requirements:

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r SC-OLH-KG/reproducibility/mainline_release/requirements.txt
export PYTHONPATH="$PWD/SC-OLH-KG${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

The exact Python and package versions used for the acceptance attempts are in
`receipts/`. Install Python 3.10.14 through your normal Python distributor;
pip installs packages, rather than selecting the interpreter version. No
system-site packages are needed.

Recompute the sealed paired statistics directly from the compact cell records:

```bash
python SC-OLH-KG/reproducibility/mainline_release/statistics.py \
  --cells SC-OLH-KG/reproducibility/mainline_release/data/cells.json.gz \
  --output /tmp/mainline-recomputed-statistics.json
```

This reads all registered cells and uses the two required streams within each
geometry. It reconstructs primary contrasts, GP/native post hoc contrasts,
source effects, four-arm interactions, missing-certification completion ranges,
and the retained cost summaries. The supplied `data/statistics.json.gz` is the
recorded acceptance output, not an input to that recomputation.

Run the preselected real d=1000 case through the shared production code:

```bash
python SC-OLH-KG/reproducibility/mainline_release/run_small.py \
  --case SC-OLH-KG/reproducibility/mainline_release/case/case.json \
  --output /tmp/mainline-real-case-attempt-01
```

The case reuses the frozen source models and ten paid initialization rows,
makes three new target calls using the actual AFTER/TS loop, freezes the
shortlist, and runs fresh independent terminal verification. It requires no
SSH, private paths, scheduler, API key or GPU. The output directory must be new.
Every attempted simulator call, failure, elapsed CPU/wall time and peak memory
is retained; no automatic retry occurs. This is a release acceptance case, not
an independent experimental replication. It does not recompute target oracle
quality. Existing sealed scientific failures are never replaced by this run.

Compare the new run with the included frozen first-case reference:

```bash
python SC-OLH-KG/reproducibility/mainline_release/frozen_behavior_parity.py \
  --old-case SC-OLH-KG/reproducibility/mainline_release/case/reference/search \
  --old-input SC-OLH-KG/reproducibility/mainline_release/case/reference/input.json \
  --new-output /tmp/mainline-real-case-attempt-01 \
  --case-json SC-OLH-KG/reproducibility/mainline_release/case/case.json \
  --output /tmp/mainline-frozen-behavior-parity-01.json
```

This checks the initialization, ordered queries, frozen terminal inputs,
deployment and verification declarations, and actual calls. Policy coordinates
must match exactly; other numeric fields use absolute and relative tolerance
1e-12. Differences remain in the report and do not trigger a new run.

The source archive contains 384 paired source calls per held-out-family asset
set, shared across target studies. Search has 10 reused plus 3 adaptive calls;
the terminal cap is 352, giving a logical cap of 365. Actual terminal branches
use their recorded calls. Reused calls, new physical calls and shared source
accounts must not be added as if each were a fresh per-cell expense.

## Current Lean interface

See [LEAN.md](LEAN.md) for the locked Lean 4.31.0/Mathlib installation and the
16-module current-only source closure. From the extracted root, after preparing
the public dependency cache:

```bash
python SC-OLH-KG/reproducibility/mainline_release/lean_delivery.py build \
  --proof proof --output /tmp/mainline-current-lean-attempt-01
```

The checker rebuilds the project modules without reusing their compiled output;
it reuses only the official locked dependency cache. It checks the 94 current
theorem declarations and audits the 12 headline axiom dependencies.

## Assembly and payload verification

Repository maintainers can inspect or assemble the selected local files:

```bash
python SC-OLH-KG/reproducibility/mainline_release/package.py inventory
python SC-OLH-KG/reproducibility/mainline_release/package.py build \
  --output /tmp/current-mainline-review.tar.gz
python SC-OLH-KG/reproducibility/mainline_release/package.py verify \
  --output /tmp/current-mainline-review.tar.gz
```

Assembly includes reachable production Python sources, the current Lean source
closure, frozen real/permuted assets, source archive/design JSON, the compact
cell/statistic data, case inputs, selected configuration and small receipts. It
streams assets into the archive without duplicating the model directory on the
local disk. Git metadata, private scheduler configuration, historical large
artifacts, trajectories and compiled caches are excluded. Missing files produce
an explicitly incomplete archive and exit code 2. The manifest requires the
statistics, cold-Lean and real-case receipts separately; `delivery_ready` also
requires their successful recorded statuses. A candidate archive can precede
the real-case receipt and is explicitly incomplete. Payload verification compares
every included file with the manifest, without running science. `assembled` or
`verified` describes the package, not author approval or scientific superiority.

## Limitations and submission details

The experiments concern three structured synthetic families at d=1000. Individual
95% intervals are descriptive and are not multiplicity-adjusted superiority
claims. Source C/G comparisons use their original 105/97 complete-pair supports;
all-144 missing estimates stay null. Missing binary-outcome completion ranges
are finite registered-sample ranges, not confidence intervals. Zero observed
false certificates does not imply zero population error probability.

The Lean results give conditional covariance/propagation/stability and terminal
event guarantees. They do not verify Python floating-point arithmetic,
quadrature, random-stream independence, arbitrary-dimensional generalization,
or nonnegative deployment quality. The third-safety and objective-comparison
gates yield a single-terminal joint error bound under their local premises;
this is not unsafety conditional on certification.

Code/data license terms are in [LICENSE](LICENSE), [DATA_LICENSE.md](DATA_LICENSE.md)
and [THIRD_PARTY.md](THIRD_PARTY.md). Data access and retained unknowns are in
[DATA_AVAILABILITY.md](DATA_AVAILABILITY.md). The corresponding-author fact check
and integrated-manuscript disclosure are in [AUTHOR_DISCLOSURE.md](AUTHOR_DISCLOSURE.md).

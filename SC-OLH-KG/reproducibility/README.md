# Reproducing The Frozen OR Evidence

This package separates four costs and four artifact classes: source training,
target search, independent verification, and post-run truth audit.  Runtime
checkpoints and model weights are intentionally excluded from the compact
release; every inferential table is regenerated from immutable `result.json`
records and hash-addressed compact audit artifacts.

## Environment

- Python 3.10.12
- Lean 4.31.0 (`proof/lean-toolchain`)
- core Python packages in `requirements-core.txt`
- canonical BoTorch backend packages in `requirements-botorch.txt`
- official transfer-baseline overlays in `requirements-transfer-overlays.txt`

The transfer overlays intentionally pin NumPy/SciPy versions separately from
the core runtime because GPy/Emukit and HyperBO have incompatible dependency
requirements.  The cluster setup used
`scripts/setup_scolhkg_transfer_runtime.sh` to install these overlays into
separate `PYTHONPATH` roots.

## Data

The external experiment uses Open Power System Data `time_series`, release
`2020-10-06`, DOI `10.25832/time_series/2020-10-06`.  The 130 MB raw CSV is not
committed.  Recreate the compact archive with:

```bash
python3 SC-OLH-KG/performance/prepare_opsd_energy_data.py \
  --out SC-OLH-KG/data/external/opsd_time_series_2020-10-06.npz
```

Expected raw SHA-256:
`6a7f2bc571314cbf9c321cc03437691cd4be95c3a6f075e60ff99e8035c704c8`.
Expected compact SHA-256:
`eeb587d26d3461dd2164c5d4ec3e57b4a6a3b5ef106ed082f578714968817fda`.

## Frozen Contracts

- method: `performance/manifests/paper_final_method_v1.json`
- experiment registry:
  `performance/manifests/paper_submission_experiment_registry_v1.json`
- external energy:
  `performance/manifests/external_energy_reliability_v1.json`
- compact readiness receipt:
  `paper_artifacts/paper_submission_readiness_v1.json`

The target proposal is frozen before target outcomes.  Target truth is used
only after search and shortlist freezing for audit metrics.  Every method in a
paired comparison has the same problem fingerprint, seed, source archive when
applicable, and verifier contract.

## Verification Commands

```bash
python3 -m pytest SC-OLH-KG/tests -q
cd proof && lake build
rg "\bsorry\b|\badmit\b|\baxiom\b" SCOLHKG SCOLHKG.lean
```

Regenerate compact audits from the immutable result roots with the commands
recorded in `performance/manifests/paper_submission_experiment_registry_v1.json`.
The audit scripts reject duplicate cells, problem-fingerprint mismatches,
unfrozen proposals, verifier drift, target-oracle use, missing search calls,
and unreported failures/timeouts.

## Reporting Rule

Every table must display, or point to a companion table displaying:

```text
source calls | target n0 | adaptive target calls | verification calls
| total target calls | source + target total calls
```

The source archive is reusable across target seeds but is never treated as
free in the total-cost comparison.  Target-only controls receive a matching
total evaluation allowance in the dedicated total-cost experiment.

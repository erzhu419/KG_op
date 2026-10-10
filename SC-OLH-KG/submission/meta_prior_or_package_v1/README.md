# Source-Learned Structural Priors: OR reproducibility package

This bounded artifact package contains the synthetic inventory generator,
source archive, frozen contracts, same-contract target-only baselines, paired
bootstrap output, manuscript sources, and replay scripts. No proprietary data
are used.

From this directory run:

```bash
python3 performance/replay_meta_prior_submission.py
python3 performance/replay_meta_prior_submission.py --compile
```

The first command reads only the included JSON artifacts, regenerates the
manuscript tables and uncertainty summary, and runs the claim-boundary audit.
The second also runs `latexmk` in `manuscript/`. It does not read or rerun the
original 95433--95440 confirmation interval. Seed 95434 overlaps development
and is excluded from any clean-confirmation claim.

The expensive simulator reruns are separate and use the frozen contracts in
`performance/`; the source and target-only runners are included for a full
rerun when compute is available. The generic target-only RBF-KG control is
frozen at lengthscale 0.25 and outputscale 1.0. The machine-readable manifest is
`performance/meta_prior_submission_manifest.json`. The core Python pins are in `reproducibility/requirements-core.txt`; the replay was validated with the recorded Python version in `reproducibility/python_version.txt`.

# Current mainline Lean delivery

This release checks `SCOLHKG.CurrentMainlineProofInterface`, separately from the
historical Atlas interface and full historical library. The lock is Lean
`leanprover/lean4:v4.31.0` and Mathlib commit
`fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`; transitive package revisions are
fixed by `proof/lake-manifest.json`.

The checked project closure has 16 modules. Seven current modules contain 94
theorems: propagation 13, covariance 6, terminal 13, representation 15, query 11,
source kernels 21, and fixed affine mismatch 15. This count describes formal
declarations, rather than 94 separate scientific contributions.

## Inventory and source-only export

From the repository root, these commands only read/copy small source files:

```bash
python3 SC-OLH-KG/reproducibility/mainline_release/lean_delivery.py inventory --proof proof
python3 SC-OLH-KG/reproducibility/mainline_release/lean_delivery.py export --proof proof --output /tmp/kg-current-lean-source
```

The export contains only the 16 `.lean` sources, both dependency locks, a Lake
configuration whose default target is the current interface, this guide, the
checker and a source manifest. The theorem sources and locks are copied byte for
byte. The exported Lake default root is changed from the historical library to
the current interface; both original and exported configuration hashes are
recorded. No source assets, experiment results or compiled Lean caches are needed
to distribute this source snapshot.

## Independent installation and build

A reviewer can build on a Linux machine with Python 3.10+, Git, an `elan`
installation, and internet access for the public Lean/Mathlib dependencies.
Neither SSH, scheduleurm, author paths nor the private six-node pool are needed.
In the source export, run:

```bash
elan toolchain install leanprover/lean4:v4.31.0
cd proof
lake exe cache get
cd ..
python3 lean_delivery.py build --proof proof --output /tmp/kg-current-lean-cold-01
```

`lake exe cache get` prepares official Mathlib/transitive dependencies using the
committed lock. It can download more Mathlib modules than this small closure
needs. The subsequent checker runs offline. To use only the required cache,
pass the `external_import_roots` printed by `inventory` as arguments to
`lake exe cache get`. An ordinary `lake build SCOLHKG.CurrentMainlineProofInterface`
also builds the current target; use the checker for the measured cold-build and
axiom audit receipt. The exported current-only project supports `lake build`
as its default target as well.

“Cold” means **zero reused SCOLHKG compiled artifacts**: the checker creates a
new source/output tree and compiles every one of the 16 project modules, in
dependency order, with `lean -j1`. It then compile-checks all 94 named current
theorems and prints dependencies for the same 12 headline declarations as the
interface. Official dependency caches are reused. This is not a claim that all
of Mathlib has been recompiled from an empty cache.

## Existing cluster cache

Project operators submit this same command through the existing CPU scheduler,
using the explicit node001–node006 pool, `vram=0` and `--no-auto-requeue`:

```bash
python3 SC-OLH-KG/reproducibility/mainline_release/lean_delivery.py build \
  --proof /path/to/locked/project/proof \
  --toolchain /path/to/lean-4.31.0 \
  --output /path/to/new/cold-attempt
```

The toolchain argument is the prefix containing `bin/lean` and `bin/lake`.
The proof argument must contain the locked dependency source trees and their
existing `.lake/build/lib/lean` caches. The checker does not download caches or
change the original project output directory. It uses one available CPU and
one thread. A failed attempt is retained in its own directory and is never
automatically retried.

`receipt.json` records per-file SHA256, a key for the current source tree, the
locked revisions, executed Lean/Lake versions, each command/exit code, full log
filenames, wall and measured child/runner CPU seconds, declaration names and
the 12 actual axiom reports. The expected headline dependencies are
`propext`, `Classical.choice` and `Quot.sound`. Existing dependency Git metadata,
when present, is checked against each locked revision. If a previously staged
cache omitted `.git`, the actual revision is recorded as unknown; entrypoint
hashes and the historical dependency-cache provenance must accompany the
receipt. Such a receipt alone does not verify that dependency checkout's
revision. Missing scheduler/transport charges and monetary costs remain unknown,
rather than zero. Previous build failures, fees and scientific budgets are not
changed by this checker.

## Mathematical scope

The four conditional propositions connecting the implementation are:

1. With PSD coefficient/task covariances and nonnegative noise/mixture weights,
   the finite source and role-replacement covariance is PSD. For a completed
   moment update `m⁺ = m + A Δm`, the contrast change between policies `u,v` is
   `(A[u] − A[v]) Δm`. Mixture predictions additionally account for evidence
   reweighting; equal observation relationships need not distinguish policies.
2. With nonnegative updated marginal variances, diagonal rescaling of a PSD
   baseline correlation remains PSD and preserves correlations wherever the
   variances are positive. This justifies the covariance construction used by
   AFTER/POINT; it does not identify it with an exact joint Bayesian posterior.
3. Explicit coordinate/encoder/scale bounds imply finite prediction-error
   bounds. On the same candidate pool and realized random draws, margins
   separated from zero by more than their error preserve feasibility; a unique
   feasible objective gap greater than twice its error preserves the query.
   The fallback theorem compares the full normalized objective/risk loss.
4. Given the local validation error bounds and fresh-block gated bounds, the
   mutually exclusive third-safety and objective-comparison branches share
   their allocated error allowance. The resulting joint error is at most
   `1/20`; the actual return/abstention and call branches are checked separately.

For interpretation, a fixed source affine predictor also satisfies
`MSE = Var(U) − a*ᵀ S a* + (a − a*)ᵀ S (a − a*) + bias²`, given explicit
second-moment integrability and the target normal equations `a*ᵀ S = cᵀ`.
The relationship-mismatch term is nonnegative. The rule is fixed; the reference
coefficient is not an extra optimization step used by the algorithm.

The readable correspondence is in
`SC-OLH-KG/docs/mainline_proof_gap_audit_and_plan_20261020.md`; propagation and
terminal arguments are in the adjacent `mainline_propagation_contrast_proof_20261020.md`
and `mainline_terminal_gate_proof_20261020.md`. The current sources also state
their premises and definitions directly.

The closure covers finite source/mixture covariance construction, working-moment
contrast propagation, fixed-correlation covariance validity, conditional
representation/query stability, a fixed-affine expected-loss decomposition,
and the terminal event/return/cost tree. The fixed-affine result does not assert
a performance guarantee for nonlinear AFTER. The terminal result bounds the
single-terminal event of an unsafe certified return or falsely announced
improvement, under its local error/independence premises. It is not a bound on
unsafety conditional on certification and does not imply nonnegative deployment
gain. Python floating-point arithmetic, numerical quadrature, random-stream
independence and generalization to arbitrary high-dimensional tasks are outside
the formal claim.

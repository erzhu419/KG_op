# OR Review V2 Remediation

This record closes every substantive issue A--G in
`manuscript/review/GPT_review_v2.md`. It does not reinterpret or rerun frozen
confirmatory outcomes. New quantities are postdecision diagnostics bound by
SHA-256 to `paper_artifacts/or_review/final_evidence_registry_v1.json`.

## A. Exogenous source archive

- Synthetic source tasks are explicitly disclosed as regime matched.
- The archive is supplied before the algorithm begins.
- The method ranks profiles inside the archive; it does not retrieve related
  tasks from a larger repository.
- Energy remains the region-held-out archive-mismatch control.

This is a transfer-design claim conditional on archive relevance, not a source
retrieval claim.

## B. Finite source-margin uncertainty

For Gaussian replications, the manuscript now gives explicit normal-mean and
chi-square sample-scale radii and combines them as

```text
|m_hat - m| <= r_mu + z * r_sigma.
```

`Real/SourceRankRecovery.lean` proves floor-aware deterministic propagation and
the separated-order implication. `Measure/SourceRankRecovery.lean` proves that
the chance-margin bad event lies inside the union of mean and scale bad events
and composes their probability bounds. The classical Cochran sampling law is
identified as the external distributional bridge; Lean does not claim to
derive the normal-to-gamma law from first principles.

## C. Selection and transfer coordinates

The algorithm selects in

```text
eta = (z, source safety rank, source objective rank),
```

but target transfer is stated only in structural `z`. The Euclidean projection
`eta -> z` is nonexpansive. `Real/GeometricAtlasCoverage.lean` proves that an
augmented-coordinate atlas cover projects to a structural cover with no larger
radius, then transfers that cover to the ideal target coordinate.
`Real/PaperMainline.lean` uses this projected theorem.

## D. Aligned-regime reversal

`performance/analyze_or_review_v2_diagnostics.py` reconstructs frozen designs
and accesses hidden target geometry only afterward. Across 600 points per arm:

| Design | Median latent distance | Median non-DC energy | Feasible points |
|---|---:|---:|---:|
| Source-scored atlas | 0.0813 | 0.2292 | 10.0% |
| Generic DCT | 0.1062 | 0.1000 | 10.0% |
| Raw Sobol | 0.0343 | 0.0258 | 94.7% |

Raw high-dimensional profiles concentrate around DC level 0.5 with weak
non-DC energy, which happens to be near the aligned safe center. This is an
oracle explanation of an observed reversal, not an algorithm input or a new
selection rule.

## E. Outcome-adjusted economics

In addition to raw call break-even, the renderer reports

```text
(S/M + target search + verification) / P(true-feasible certificate).
```

The source-versus-control crossing is

```text
ceil(S / (p_source * C_control / p_control - C_source))
```

when the denominator is positive. The manuscript also states a general
operational loss with separate costs for unsafe deployment, abstention, and
objective quality rather than silently fixing those costs.

## F. Stratified task law and resolution dependence

The task generator contains an induced `task_seed mod 5` category. Its realized
counts are tabulated. The experiment has 160 independently seeded latent tasks
in eight fixed strata, crossed with three resolutions. Thus:

- inference and weighted concentration are applied separately by resolution;
- regimes are fixed equally weighted strata;
- tasks may be independent and non-identically distributed;
- the aggregate 480 task-resolution cells are descriptive, not 480 independent
  tasks.

`Measure/TaskAtlasCoverage.lean` now includes a weighted non-identical task
concentration theorem with proxy `(1/4) sum_s w_s^2/n_s`.

## G. Literature placement

The bibliography and related-work section now cover the original Gonzalez
farthest-first result, minimax/maximin distance design, space-filling computer
experiments, functional-data basis representations, and warm-start Bayesian
optimization from historical evaluations.

## Artifact contract

The supplemental artifact is
`paper_artifacts/or_review/review_v2_supplemental_diagnostics.json`. It binds
the base evidence registry and compact cost inputs by hash. The renderer
produces the geometry plot, task-strata table, and outcome-adjusted-cost table
from this artifact. Hidden geometry is postdecision and primary outcomes are
unchanged.

The journal-facing article has 27 pages before references, below the declared
30-page limit. Detailed algorithms, sensitivity results, task-seed strata,
native transfer results, HVD diagnostics, and the Lean map remain in the
five-page `manuscript/supplement.pdf`; they were not discarded. The manuscript
receipt compiles and hash-records both PDFs and rejects warnings, undefined
references, or layout overflow in either document.

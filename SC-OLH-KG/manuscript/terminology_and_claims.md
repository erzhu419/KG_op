# Terminology And Claim Ledger

## One-Sentence Argument

In high-dimensional chance-constrained simulation optimization with reusable
source tasks, a source-only dimension-equivariant risk-objective atlas can
concentrate a very small target budget on transferable policy profiles, while a
replaceable Bayesian optimizer and an independent terminal verifier separate
objective improvement from deployment safety; the evidence supports this
claim on three synthetic domains and one energy holdout, but only under an
explicit structural transfer condition.

## Canonical Terminology

| Canonical term | Definition at first use | Do not use as a synonym |
|---|---|---|
| structural proposal atlas | A deterministic set of at most `n0` target policies, learned from source outcomes and frozen before target outcomes | SC prior, oracle proposal, KG proposal |
| risk-objective atlas | The implemented structural proposal atlas ranked jointly by source chance-margin and objective information | meta anchor, strong prior |
| universal structural support | Target-label-free low-frequency profiles available without source outcomes | source proposal |
| source templates | Shared normalized profiles selected from source outcomes | oracle templates |
| dimension-equivariant coordinate | Mean, scale, and normalized low-frequency profile coefficients whose dimension does not grow with raw policy dimension | latent oracle |
| source archive | Two source tasks, 64 shared profiles per task, three replications per profile; 384 calls | pretraining for free |
| target search calls | Simulator calls available to the target optimizer, including `n0` | total evaluations |
| verification calls | Fresh terminal samples that never update the optimizer | search calls |
| source-plus-target total calls | Source calls plus target search and verification calls | target budget |
| canonical SAASBO | Standard BoTorch SAAS fully Bayesian GP refit at every target iteration | proposed backend |
| independent terminal verifier | Ordered shortlist verification with fresh samples and an objective switch guard | posterior certificate |
| true-feasible recommendation | Recommendation satisfying the benchmark chance constraint under post-run truth | certified recommendation |
| independently certified recommendation | Recommendation accepted by the fresh-sample verifier | posterior feasible point |
| false certificate | Independently certified recommendation that is truly infeasible | false feasible sample |
| feasible regret | Objective gap to the known benchmark feasible optimum, reported only when the recommendation is true-feasible | unconditional regret |
| cumulative factor-HVD | Provider-coordinate cumulative heteroscedastic variance decomposition | primary optimizer |
| external energy holdout | Untouched OPSD GB_GBN target after DK_2 development | real-world proof of universal transfer |

## Method Identity

The primary method in the manuscript is:

```text
source-learned structural proposal atlas
  + canonical SAASBO target backend
  + independent terminal verifier
```

The frontend is novel.  The backend is replaceable.  The verifier is a
method-independent safety layer.  Proposal-only, stacked GP, SC-V69, KG, and
other transfer methods are comparisons or ablations.

## Claim-Evidence Map

| ID | Bounded claim | Main evidence | Theory | Boundary |
|---|---|---|---|---|
| C1 | The frozen atlas is the dominant source of feasible-basin coverage at `d=1000, N=13`. | Frozen frontend gives 60/60 feasibility with three backends; common Sobol gives 0/60, 1/60, and 0/60. | Finite-atlas no-free-lunch plus conditional coverage theorem. | Three registered synthetic domains; no arbitrary-function claim. |
| C2 | Both source templates and universal support are needed for robust cross-domain coverage. | Universal 27/60, source-only 40/60, combined 60/60. | Atlas union and maximin coverage contract. | Supports this implemented decomposition, not four coequal priors. |
| C3 | Canonical SAASBO improves objective quality after coverage is established, but is not essential to feasibility. | All frozen-backend cells are 60/60; SAAS improves paired regret in Inventory and Queue. | Backend-independent frontend and verifier interfaces. | SAASBO is standard and not a novelty claim. |
| C4 | Structural transfer can remain useful when raw dimension reaches 10,000. | 30/30 certified at `N=10,13,20,40`; zero false certificates. | Sufficient condition depends on coordinate geometry rather than raw `d`. | Conditional on stable transfer geometry; regret is nonmonotone from `N=20` to `N=40`. |
| C5 | The final deployment decision is independently auditable. | Zero false certificates in registered release cells. | Familywise shortlist bound, objective-switch bound, exact binomial bridge. | Empirical zero errors do not imply zero population error. |
| C6 | In the energy holdout, low-frequency structural compression beats equal-total-cost unstructured Sobol. | 20/20 paired wins; median objective difference -0.05331. | Exact-binomial verifier bridge. | No advantage over natural low-frequency grid was detected. |
| C7 | Cumulative factor-HVD recovers heteroscedastic variance shape better than pooled variance. | Lower variance RMSE and higher shape correlation in all three domains. | Cumulative-risk decomposition and certification bridges. | No demonstrated optimization or verification-cost gain. |

## Prohibited Headline Claims

- "SC-OLH-KG is the best optimizer."
- "KG drives the performance gain."
- "HVD improves optimization performance."
- "Ten evaluations solve arbitrary 10,000-dimensional optimization."
- "The method requires only 13 total simulator evaluations."
- "The source atlas beats natural low-frequency controls in energy."
- "The transfer theorem holds unconditionally."
- "Traffic experiments validate the final method."

## Paper Architecture

1. Introduction: task, impossibility without structure, contribution.
2. Literature: simulation optimization, high-dimensional BO, transfer/meta-BO,
   and independent certification.
3. Problem and information contract: source, target-search, verification costs.
4. Structural proposal atlas: coordinate, source ranking, universal support,
   maximin selection, fail-closed envelope.
5. Coverage and verification theory.
6. Experimental design and fairness contracts.
7. Results: frontend causality, backend comparison, budget/dimension frontier,
   external energy, HVD diagnostic.
8. Discussion: what transfers, what does not, and cost implications.
9. Conclusion.

# V28 Safe Generalized Task Posterior

## Motivation

V27 proves that the necessary risk representation can be present while the
task posterior still makes the wrong decision.  The current posterior answers
"which expert predicts all observed outputs well?"; the terminal action needs
"which expert predicts the held-out chance boundary well?".  A larger terminal
penalty cannot make those two questions equivalent.

## Two Posterior Objects

V28 keeps a predictive posterior and adds an explicit safe-decision posterior:

\[
Q_t^{\mathrm{pred}}(k)
\propto \pi_k\exp\{-\eta_t L_{\mathrm{pred},t}(k)\},
\]

\[
Q_t^{\mathrm{safe}}(k)
\propto \pi_k\exp\{-\eta_t L_{\mathrm{safe},t}(k)\}.
\]

`Q_pred` retains the existing proper multichannel predictive log score and is
used for ordinary objective prediction diagnostics.  `Q_safe` uses only
charged, strictly prequential target evidence:

1. the constraint Gaussian predictive log score;
2. the proper Bernoulli score of the observed threshold event;
3. a bounded noisy pairwise boundary-ranking score.

For a new charged observation and an earlier prequential record, expert `k`
receives the pairwise score

\[
\log \Phi\!\left(
  s_{ij}\frac{\mu_{k,i}-\mu_{k,j}}
  {\sqrt{v_{k,i}+v_{k,j}}}
\right),
\qquad
s_{ij}=\operatorname{sign}(y_i-y_j),
\]

clipped to a finite interval and weighted by observed separation and boundary
relevance.  The reference prediction is the value stored before its label was
inserted, so this is prequential rather than an in-sample ranking fit.

## Integration

- candidate proposal allocation, robust cumulative moments, certification,
  shared-expert posterior sampling, and exact-KG terminal value use `Q_safe`;
- objective mixture diagnostics retain `Q_pred` so safety loss cannot silently
  redefine the optimization objective;
- exact-KG fantasy clones update both posteriors and the pairwise history;
- the KL ambiguity radius is centred on the decision posterior actually used;
- all new behavior is behind `task_posterior_safe_generalized`; V27 remains a
  paired ablation.

No task name, target truth, optimum, analytic boundary, or uncharged simulator
call enters either posterior.

## Controlled Gate

1. Unit and checkpoint tests must prove normalization, bounded pairwise score,
   strict prequential storage, clone isolation, and exact-fantasy updates.
2. Paired seed-0 smoke must preserve FactorShock and Inventory feasibility and
   avoid a material runtime increase.
3. The unchanged 7+7 gate requires FactorShock 7/7 with zero violation and
   Inventory at least 4/7 with at most one false-feasible result.
4. Queue remains unopened until that gate passes.

The existing terminal Bayes-risk penalty stays fixed during V28.  This isolates
task-aligned posterior learning from terminal-loss tuning.  If V28 still fails,
the next controlled module is budgeted finalist replication / selection-risk
correction, not another static recommendation heuristic.

## Proof Obligations

- finite normalization and full support of `Q_safe`;
- bounded pairwise loss after probability clipping;
- prequential no-target-oracle implementation bridge;
- finite clipping bound for the Bernoulli/pairwise terms and a PAC-Bayes bound
  for the full safe score under the stated Gaussian exponential-moment model;
- robust certification and exact-KG bridges centred on `Q_safe`.

# Certificate/Performance Contract Audit

Date: 2026-07-27

## Question

Did making the feasible certificate non-vacuous reduce the optimization
performance that had been observed before V64?

## Answer

No paired experiment supports that causal interpretation.

V64 freezes the V51 search trajectory and primary recommendation before
running independent terminal verification. Verification samples do not update
the optimizer or posterior. On the preregistered fresh seeds `60..79`, V64 and
V51 therefore have identical search behavior. V64 changed 14 terminal
deployments, with 14 paired wins, zero losses, and 46 ties. It certified all
60 deployments and issued no false certificate.

The later `37/60` result belongs to a different source-proposal contract. It
uses a same-dimension `d=1000` source archive and fresh seeds `80..99`. Its
frozen initial design contains a true-feasible point in only `40/60` cells and
fails entirely on Queue. Under the registered cross-dimension `d=50 -> 1000`
proposal on those same target seeds, the initial design, final deployment, and
terminal certificate are all `60/60`.

Thus the apparent regression is a proposal-support/domain-transfer failure,
not a certificate-induced search loss.

## Historical Contracts

| Version/evidence | Information contract | Search/target budget | True feasible | Certificate |
|---|---|---:|---:|---:|
| V32 | privileged source-oracle upper bound | `N=20` | `7/7`, `5/7`, `3/7` by domain | posterior set empty |
| July backend matrix | oracle-free frozen source proposal | `d=50, N=20` | proposal improves every tested backend | `0/1350` posterior sets nonempty |
| promoted V51 | oracle-free, source `d=50`, 384 source calls | `d=1000, N=20` | `60/60` | `0/60` posterior sets nonempty |
| promoted V64 fresh gate | V51 search plus independent verification | `N_search=13`, `R=80/96` | `60/60` | `60/60` |
| paper same-d block | oracle-free source `d=1000` | `N_search=13`, `R=80/96` | `37/60` | `23/60` |
| paper cross-d block | oracle-free source `d=50 -> 1000` | `N_search=13`, `R=80/96` | `60/60` | `60/60` |

V32 is not a valid oracle-free main result: its source training used analytic
`true_sigma`, `true_outputs`, and source-domain teacher hooks. It remains a
labelled upper bound only.

## What The Earlier Large Matrix Established

At `d=50, N=20, n0=10`, replacing common Sobol by the frozen source-informed
proposal increased true-feasible outcomes for every audited backend:

| Backend | Common Sobol | Source-informed |
|---|---:|---:|
| Sobol continuation | `15/30` | `28/30` |
| exact KG | `12/30` | `26/30` |
| risk-aware TS | `11/30` | `25/30` |
| frozen initial decision | `0/30` | `27/30` |

This is strong evidence for the learned proposal/front end. It is not evidence
that HVD certification or a particular online backend caused the gain.

## Meaning Of The Certificate

Synthetic truth can tell us after the run that a recommendation happened to
be feasible. A real deployment cannot query that oracle. A certificate answers
a different question: whether the algorithm can deploy the policy while
controlling the probability of a false-feasible declaration under the stated
sampling assumptions.

The V51 result `60/60 true feasible, 0/60 certified` is therefore strong
retrospective search performance but not a deployable safety claim. V64 turns
that into a deployable claim through an independent finite-sample verification
suffix. This suffix is useful but expensive: 80 calls for the primary and up
to 96 more for support. Search, verification, and total budgets must always be
reported separately.

V64 does not establish that the online posterior HVD certificate is
non-vacuous. It establishes a separate independent terminal certificate. The
posterior certificate and independent deployment certificate must remain
separate columns in every table.

## Controlled Synthetic Follow-Up

The existing FactorShock/Inventory/Queue simulators do not independently vary
where and how heteroscedasticity occurs. A controlled suite is needed to
separate:

1. optimization-primary quality;
2. posterior-certificate coverage;
3. independent terminal-certificate coverage;
4. deployment win/loss relative to the frozen primary;
5. HVD calibration and model misspecification.

The registered gate uses common-Sobol `n0`, no source proposal, no
problem-specific refinement, and known post-run oracle truth. It varies smooth
boundary noise, optimum-local noise, safe-interior noise, regime jumps, sparse
drivers, shared factors, and a deliberately misspecified high-frequency
interaction. This is a causal HVD/certification diagnostic, not a replacement
for the transfer/SOTA matrix.

## Controlled Gate Results

The first `d=1000, N=20` gate evaluated no true-feasible policy in any of 360
runs. Raw high-dimensional Sobol policies and random-raw state inversion both
concentrated the three block-mean controls near `0.5`. Even oracle-variance
rows failed, proving that this was candidate support failure before HVD or
certification.

V2 replaced only that inversion with a scenario-invariant, label-free design
on the observable latent-control cube. The eight scenarios received exactly
the same anchors; no objective, constraint, variance, source archive, or oracle
label entered their construction. This made true-feasible evaluated policies
available in `146/360` runs. Nevertheless, the frozen search primary was truly
feasible in only `11/360` runs.

Independent terminal verification certified those same 11 policies with zero
false certificates, zero recommendation changes, and zero feasibility losses.
It therefore did not cause the optimization regression. The bottleneck is the
terminal ranking and shortlist: the support candidate was restricted to the
first `n0` raw-Sobol points, excluding safe latent anchors observed later.

The online posterior certificate must not be conflated with this independent
certificate. In V2 it declared 395 evaluated points feasible, of which only
132 were truly feasible and 263 were false certificates. At `N=20`, the
current posterior mean/HVD uncertainty model is therefore not calibrated
enough to support a safety claim. V3 keeps the search fixed and expands only
the pre-verification candidate universe from the initial atlas to all observed
policies.

V3 matched all 200 non-Sobol V2 rows exactly: primary recommendation, best
evaluated feasible regret, and candidate-source counts had zero mismatches.
The only change was the frozen terminal candidate universe. Certified
deployment rose from `11/200` to `130/200`; 119 unsafe primaries were rescued,
with zero feasibility losses and zero false independent certificates. Sixteen
runs had evaluated a truly feasible policy but still did not certify one.

This resolves the causal question. Non-vacuous independent certification did
not reduce optimization quality. It converted discovered safe support into a
deployable decision. It did not, and should not be expected to, manufacture a
near-optimal policy: the eight-anchor V2/V3 design contains only coarse control
corners, and no run reached regret `0.01`. The next gate expands only the same
label-free latent design, leaving the certificate protocol fixed.

The 24-anchor V4 gate increased evaluated-feasible support to `154/200` and
reduced median best-evaluated regret from `0.2945` to `0.2437`, but produced no
regret-`0.01` hit. Independent verification still produced zero false
certificates and zero losses, rescuing 117 unsafe primaries. The gate therefore
failed for optimization resolution, not certification.

This also shows why the next synthetic experiment must separate two questions.
At `d=1000`, raw-space concentration and latent candidate resolution dominate
the outcome. A `d=3` gate, where raw and observable latent coordinates
coincide, is needed to test heteroscedastic model identification and optimum
recovery without that representation confound. Dimension scaling should then
be a separate experiment with the HVD/backend fixed.

The `d=3, N=40` V5 gate removed that representation confound. It found a
truly feasible evaluated policy in `198/200` runs, but no run reached regret
`0.01`, and the soft-penalty Bayes-risk primary was unsafe in all 200 runs.
Observed-history terminal verification rescued 123 deployments with zero
losses and zero false independent certificates.

Factor-HVD did show the intended local signal in the shared-factor scenario.
For risk-aware TS, median log-variance RMSE improved from `0.686` (pooled) to
`0.580` (factor), and upper-variance coverage improved from `0.297` to `0.904`.
That is not yet a general HVD win: factor was worse than pooled in some smooth
variance scenarios, and the online posterior certificate still produced 88
false declarations among 315 certified points.

The remaining optimum failure is therefore attributable to the decision
layer. The current risk-TS backend minimizes a fixed soft penalty
`sampled_objective + rho * positive_margin`, so a sufficiently low objective
can dominate a positive sampled violation. The terminal primary uses the same
kind of Bayes-risk ranking. The next backend comparison should use
feasible-first constrained posterior sampling and an objective-ranked frozen
safe shortlist, keeping the cumulative HVD and independent verifier fixed.

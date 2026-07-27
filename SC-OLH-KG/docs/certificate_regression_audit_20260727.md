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

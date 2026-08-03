# Paper d=10000 Budget Frontier Audit

Date: 2026-08-03

## Frozen contract

- Commit: `da01f40dfce175cd9bb723ba680ef6c719ec7ad6`
- Method: canonical constrained SAASBO (`saas_fully_bayesian_nuts_constrained_qlogei`)
- Frontend: frozen source-learned dimension-equivariant proposal
- Dimension: `d=10000`
- Target search budgets: `N in {13,20,40}`, including `n0=10`
- Domains: FactorShockStatePolicyRZDT1, InventorySupplyChain, QueueResourceControl
- Seeds: 80--89
- Offline source cost: 384 simulator calls, with no source oracle labels
- Selection uses no target oracle information
- Deployment uses an independent frozen-shortlist verifier

All 90 rows completed with status `ok`. There were no BoTorch fit failures,
candidate failures, timeout fallbacks, or false certifications.
The commit, method fidelity, information contract, and cross-budget initial
design fingerprints agree across all rows. Eighty-eight rows ran on CUDA; two
completed CPU sentinels used the identical canonical algorithm contract.

## Final deployment results

| N | d/N | Domain | Feasible / 10 | Certified / 10 | False certificates | Median feasible regret | Median target verification calls | Median wall time |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 13 | 769.2 | FactorShock | 10 | 10 | 0 | 0.008250 | 80 | 0.50 h |
| 13 | 769.2 | Inventory | 10 | 10 | 0 | 0.010807 | 224 | 0.50 h |
| 13 | 769.2 | Queue | 10 | 10 | 0 | 0.001908 | 224 | 0.48 h |
| 20 | 500.0 | FactorShock | 10 | 10 | 0 | 0.008250 | 80 | 1.60 h |
| 20 | 500.0 | Inventory | 10 | 10 | 0 | 0.007444 | 224 | 1.62 h |
| 20 | 500.0 | Queue | 10 | 10 | 0 | 0.002387 | 224 | 1.67 h |
| 40 | 250.0 | FactorShock | 10 | 10 | 0 | 0.008250 | 208 | 4.99 h |
| 40 | 250.0 | Inventory | 10 | 10 | 0 | 0.006724 | 224 | 5.11 h |
| 40 | 250.0 | Queue | 10 | 10 | 0 | 0.002864 | 224 | 4.92 h |

The target-call accounting must remain separated in the paper. Typical total
target costs are approximately 229, 244, and 264 calls for N=13, 20, and 40,
respectively, because independent terminal verification is additional to the
search budget. The corresponding source-plus-search costs are 397, 404, and
424 calls before verification.

## Proposal versus online backend

In a retrospective oracle audit that was not available to selection, the same
frozen `n0=10` proposal contained a truly feasible point in all 30 domain/seed
combinations. Its median feasible regrets were 0.008250 for FactorShock,
0.011550 for Inventory, and 0.002864 for Queue.

| N | Domain | Better than true n0-best | Equal | Worse | Median regret change |
|---:|---|---:|---:|---:|---:|
| 13 | FactorShock | 0 | 10 | 0 | 0.000000 |
| 13 | Inventory | 5 | 5 | 0 | -0.000742 |
| 13 | Queue | 6 | 2 | 2 | -0.000956 |
| 20 | FactorShock | 2 | 8 | 0 | 0.000000 |
| 20 | Inventory | 8 | 2 | 0 | -0.004105 |
| 20 | Queue | 6 | 3 | 1 | -0.000477 |
| 40 | FactorShock | 1 | 9 | 0 | 0.000000 |
| 40 | Inventory | 8 | 2 | 0 | -0.004825 |
| 40 | Queue | 2 | 6 | 2 | 0.000000 |

Thus the frontend is responsible for universal feasibility at initialization.
The online SAAS backend adds consistent value on Inventory, modest or unstable
value on Queue, and almost no value on FactorShock.

## Budget-frontier interpretation

- N=20 improves Inventory materially over N=13 and preserves 30/30 certified
  feasibility. It is the best compromise between quality and computation.
- N=40 is not a monotone improvement. Relative to N=20, it improves only 4/10
  Inventory seeds, worsens 6/10 Inventory seeds, worsens 6/10 Queue seeds, and
  gives no median FactorShock gain.
- Canonical every-iteration SAAS cost grows sharply: median runtime rises from
  about 0.5 h at N=13 to 1.6 h at N=20 and 5.0 h at N=40.
- The model-internal posterior feasible set remains empty in many runs:
  2/30 at N=13, 14/30 at N=20, and 5/30 at N=40 have a nonempty posterior
  certificate. Independent terminal verification nevertheless certifies all
  90 deployments with zero audited false certificates.

## Paper decision

Use N=13 as the extreme `d/N` headline and N=20 as the principal budget-frontier
point. Keep N=40 as convergence evidence and as a negative result showing that
additional canonical SAAS evaluations do not automatically improve the final
decision. Do not claim monotone convergence or attribute universal feasibility
to the backend. The defensible attribution is:

1. the frozen source-learned proposal finds a feasible basin in 30/30 cases;
2. the online backend improves objective quality primarily on Inventory; and
3. the independent verifier converts a frequently vacuous model certificate
   into 90/90 deployable certificates without oracle selection.

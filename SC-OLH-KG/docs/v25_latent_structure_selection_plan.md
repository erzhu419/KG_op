# V25 Latent Local/Ordered Structure Selection

## Motivation

V23 and V24 reject a direct-sum mean model.  An orthogonal local residual can
be made numerically safe, but its inclusion neither predicts FactorShock
success nor helps Inventory.  Historical V21a evidence instead shows a clear
task-level distinction: the local-kernel expert receives posterior mass
`0.97--1.00` on every FactorShock seed and gives 7/7 feasible
recommendations, while the sparse ordered model is the only tested branch
that moves Inventory recommendations consistently toward its chance boundary.

## Structural Hypothesis

Introduce a finite latent structure variable

`S in {local, ordered cumulative}`.

Conditional on `S`, the mean, cumulative HVD, certification, candidate
proposal, and exact posterior update all use that branch's frozen feature
provider.  The finite task posterior updates `P(S | D_t)` from source prior
mass and charged target observations.  The two feature maps are never
concatenated.

## Controlled Expert Set

V25 keeps six experts:

1. universal coordinate;
2. null universal;
3. source spectral;
4. risk-aligned spectral;
5. sparse ordered cumulative;
6. local risk kernel.

The ordered branch replaces `risk_aligned_coordinate`, which is redundant
with the retained aligned spectral branch and carried negligible mass on the
stable FactorShock runs.  V25 therefore does not add a seventh expert or
change the initial-design budget.

## Controls Held Fixed

- diagonal ordered basis and target-updated sparsity from V22;
- no semiparametric residual;
- source-only LODO training and no target oracle hooks;
- factor HVD, theory certification, IID exact-MC2 KG;
- `d=50`, `N=20`, `n0=10`, and paired seeds 0--6;
- the same candidate pool, terminal audit, and promotion gate.

## Gate

- FactorShock: 7/7 true feasible, zero violation;
- Inventory: at least 4/7 true feasible and at most one false feasible;
- six experts in every run and normalized positive posterior weights;
- no Queue run before both held-out domains pass.

If FactorShock returns to 7/7 while Inventory remains below 4/7, V25 proves
that task structure selection works but is not promoted.  The next isolated
change is group/subspace shrinkage inside the ordered branch.  If FactorShock
does not recover, posterior evidence or exact-KG integration between the two
structures is the next failure layer; the rejected direct-sum residual is not
reintroduced.

## Result

All 14 IID-MC2 shards completed.

| Domain | Final true feasible | False feasible | Mean violation | Median feasible regret | Median runtime |
|---|---:|---:|---:|---:|---:|
| FactorShock | 7/7 | 0/7 | 0.00000 | 0.00825 | 17.0 min |
| Inventory | 3/7 | 0/7 | 0.01016 | 0.00977 | 15.1 min |

FactorShock assigns essentially unit posterior weight to the local expert in
every seed, while the ordered branch receives essentially zero weight.  V25
therefore restores the stable domain and validates task-level latent structure
selection.  Inventory improves from V22's 2/7 to 3/7 without a false-feasible
certificate, but misses the predeclared 4/7 promotion gate.

The Inventory ordered expert is not consistently selected: its largest weight
occurs on an infeasible seed.  V25 is therefore not promoted.  V26 keeps this
latent task layer fixed and changes only coefficient transfer inside the
ordered branch from coordinate-wise PIPs to semantic group-shared shrinkage.

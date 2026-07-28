# V26 Semantic Group-Shared Shrinkage

## Motivation

V25 proves that FactorShock and Inventory should not share one concatenated
local/ordered mean model.  It restores FactorShock to 7/7 by learning a latent
task structure, but Inventory remains 3/7.  The remaining transfer failure is
coordinate-wise: source domains and the held-out domain need the same semantic
quadratic and shared-exposure blocks, but not the same coefficient direction
inside those blocks.

## Model

The ordered feature map stays

```text
[A, A^2, N]
```

for the diagonal-quadratic gate.  Linear `A` is the always-active prefix.
V26 assigns one shared spike/slab inclusion probability and one isotropic slab
scale to every coordinate in `A^2`, and a second shared pair to `N`.  Given a
group inclusion state, target data learn a full continuous coefficient vector
inside that group.

For standardized group design `Z_G`, precision `W`, partial residual `r`, and
prior variance `v`, the group evidence is

```text
log p(r | v, Z_G)
  = -0.5 log det(I + v Z_G^T W Z_G)
    + 0.5 v b^T (I + v Z_G^T W Z_G)^-1 b,
b = Z_G^T W r.
```

The group Bayes factor is slab evidence minus spike evidence.  It replaces
coordinate-wise Bayes factors only for the two declared semantic groups.

## Invariance And Rank Control

Features in each group use one RMS scale.  Under an orthogonal within-group
rotation, the common scale, joint Bayes factor, posterior PIP, predictive mean,
and isotropic prior penalty are invariant/equivariant.  The effective
dimension charged to a group is `|G| q_G`, so the existing total
`0.35 N` cap still counts coefficients rather than merely counting groups.

## Controlled Gate

- retain V25 latent local/ordered structure and exactly six experts;
- diagonal ordered basis, source-only LODO records, no semiparametric residual;
- factor HVD, theory certification, IID exact-MC2 KG;
- `d=50`, `N=20`, `n0=10`, paired seeds 0--6;
- FactorShock must remain 7/7 with zero violation;
- Inventory must reach at least 4/7 with at most one false feasible;
- Queue is not run until both requirements pass.

This is not a target-task classifier or an external gate.  It is a hierarchical
prior inside the ordered expert: source domains transfer semantic block
strength, while all held-out directions are learned from charged target data.

## Attempt A Implementation Audit

The first 14-shard run completed with FactorShock 7/7 and Inventory 3/7, but
is invalid for promotion.  Five FactorShock seeds and three Inventory seeds
exceeded the declared total effective-dimension cap.  The maximum excess was
not floating-point noise: a group at `min_pip=0.05` and another at
`max_pip=0.95` consumed `3.05` optional dimensions when only `3.0` were
available.

The cause was a fixed cardinality-projection logit-shift upper bound of `50`.
Joint group log Bayes factors reached approximately `95`, so that bound did
not bracket a budget-feasible posterior.  V26b changes only this implementation
detail: the upper shift is derived from the largest finite posterior logit plus
50, and every result records an explicit budget-respected boolean and slack.
The promotion gate now rejects any seed that violates the cap.

Attempt A is retained as diagnostic evidence only.  Its posterior also shows
that Inventory nearly always shrinks the curvature `A^2` group to `0.05` and
selects shared exposure `N` at about `0.93--0.95`.  If V26b remains 3/7, the
next failure is early structural collapse of a necessary curvature group, not
certification or rank control.

## V26b Result

The cap-correct rerun completes with zero dimension violations in all 14
shards.  FactorShock remains 7/7 true feasible with zero violation and median
regret `0.00825`.  Inventory remains 3/7 with no false-feasible certificate,
mean violation `0.01222`, and median feasible regret `0.00977`.  V26b therefore
fails only the predeclared Inventory 4/7 quality gate and is not promoted.

The corrected posterior confirms the structural diagnosis: median curvature
PIP is `0.05`, median shared-exposure PIP is `0.9333`, and the ordered expert's
Inventory median task weight is about `9.7e-5`.  Group spike/slab is rejected as
the main complexity learner; the latent local/ordered task layer is retained.

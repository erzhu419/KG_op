# KG-SYNTH completion audit and feasible-first gate

Date: 2026-07-28

## Artifact contract

- Only compact `result.json` and submission manifests were inspected or
  synchronized. Checkpoints, pickle files, model weights, and NumPy arrays
  were not copied to the local workspace.
- All reported optimization outcomes use post-run truth joins. Target truth
  was not used by the search, posterior shortlist, or independent terminal
  verifier.
- Canonical SAASBO and the periodic-capped SAASBO replacement are separate
  algorithm variants. They must not be pooled or described as equivalent.

## Completion audit

The controlled heteroscedastic V5 gate is complete: 200/200 compact results.

The main transfer blocks are complete:

- source-informed official transfer: 480/480 successful;
- common-Sobol official transfer: 480/480 successful;
- SC-OLH V64, same-dimension source archive: 60/60 successful;
- SC-OLH V64, source dimension 50 to target dimension 1000: 60/60
  successful.

The SOTA and cross-dimension backend supplements still contain real gaps:

- canonical SOTA root has 481 compact rows, of which 471 are successful and
  10 are explicit runtime failures;
- 59 unresolved canonical `N=397` SAASBO cells were intentionally replaced
  by the separately labelled periodic-capped variant;
- periodic-capped `N=397` SAASBO has 58/60 successful rows;
- the cross-dimension common-proposal matrix has complete proposal-only
  (60/60) and Stacked GP (60/60) rows, but only 12/60 successful canonical
  SAASBO rows. Two additional rows timed out and 46 produced no result.

The remaining equivalent recovery set is therefore:

- 9 failed canonical SOTA cells excluding the intentionally substituted
  canonical `N=397` SAASBO row;
- 2 failed periodic-capped SAASBO cells;
- 48 cross-dimension SAASBO cells without a successful compact result.

## Current paper-matrix signal

With the dimension-50 frozen proposal transferred to dimension 1000, all
three domains have 20/20 independently certified deployments:

| Method | FactorShock | Inventory | Queue |
|---|---:|---:|---:|
| proposal only, median regret | 0.00825 | 0.01156 | 0.00288 |
| V64, median regret | 0.00825 | 0.01156 | 0.00288 |
| Stacked GP, median regret | 0.00825 | 0.01156 | 0.00288 |

V64 improves over proposal-only in only 3/60 seeds, by approximately
`5e-4` to `7e-4` regret. Stacked GP differs in four seeds and has one small
improvement. Thus the frozen cross-dimension proposal is currently the
dominant contributor; neither V64 nor Stacked GP has yet established a large
online-backend increment.

The partial canonical cross-dimension SAASBO block is 12/12 feasible on the
completed FactorShock rows with median regret 0.00825. Inventory and Queue
cannot be interpreted until the missing cells finish.

Target-only BoTorch at `d=1000, N=13` has no feasible recommendation in the
currently successful rows. Shared-archive SAASBO is much stronger (53
feasible recommendations among 57 successful rows), while target-cost-matched
`N=397` TuRBO/SCBO still have no feasible recommendation in their successful
rows. Periodic-capped SAASBO has 2 feasible recommendations among 58
successful rows. Runtime failures remain in every denominator in the final
paper table.

## Controlled heteroscedastic diagnosis

V5 uses no source archive, no source proposal, a common-Sobol initial design,
and eight controlled variance geometries. Across 200 runs:

- a true-feasible evaluated point was found in 198/200 runs;
- the soft-penalty posterior primary was true-feasible in 0/200 runs;
- independent ordered-shortlist verification deployed a true-feasible policy
  in 123/200 runs;
- the verifier produced 123 feasibility rescues, zero feasibility losses,
  and zero false terminal certificates;
- online posterior certification produced 315 certified evaluated points,
  including 88 false certificates;
- no run reached feasible regret at most 0.01.

For factor-HVD with risk-aware Thompson sampling, the method found a
true-feasible point in 40/40 runs but deployed one in only 26/40. Median
best-evaluated regret was 0.12962 and median deployed regret was 0.28765.

This separates two facts:

1. Independent certification is useful and did not cause the observed search
   failure. It safely recovered 123 deployments without a false terminal
   certificate.
2. The posterior decision rule is the bottleneck. A finite soft penalty lets
   a sufficiently low sampled objective compensate for sampled chance
   infeasibility, and the old fallback deliberately maximizes risk-coordinate
   diversity instead of objective quality.

## V6 causal gate

Two changes are introduced without changing any certificate threshold:

1. `constrained_ts` uses lexicographic posterior sampling. If a posterior draw
   contains feasible candidates, it minimizes objective only among those
   candidates. If none are feasible, it minimizes sampled chance violation.
2. `objective_ranked` terminal support forms a posterior violation sublevel
   among candidates distinct from the primary, then minimizes posterior
   objective inside that sublevel. Independent verification remains frozen,
   family-wise, and out of sample.

The five-seed gate contains six causal variants over all eight scenarios:

- risk-TS + factor-HVD + objective-ranked support;
- constrained-TS + factor-HVD + legacy diverse support;
- constrained-TS + factor-HVD + objective-ranked support;
- constrained-TS + pooled variance + objective-ranked support;
- constrained-TS + orthogonal-HVD + objective-ranked support;
- constrained-TS + oracle variance + objective-ranked support.

This gives 240 independently sharded one-core CPU tasks on node001-node006.
The existing V5 risk-TS + factor-HVD + diverse-support rows are reused as the
matched control.

Promotion requires zero false terminal certificates, no reduction below the
V5 factor deployment count of 26/40, and a strict improvement in either
median deployed regret or best-evaluated regret. Only a passing variant is
expanded to 20 seeds.

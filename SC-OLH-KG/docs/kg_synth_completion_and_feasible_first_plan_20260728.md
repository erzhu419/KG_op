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

After the first recovery wave, the cross-dimension matrix reached 149/180
compact successes: proposal-only 60/60, Stacked GP 60/60, and canonical
SAASBO 29/60. The remaining 31 canonical SAASBO logical cells were submitted
with the path-portable runner and `skip_existing_success`; they remain a
separate completion item and are not used in the controlled-heteroscedastic
conclusions below.

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

## V6 and V7 results

On the matched 20-seed, eight-scenario matrix (`160` runs), V6
`constrained_ts` found at least one true-feasible evaluated policy in
160/160 runs and reduced median best-evaluated feasible regret from V5's
`0.135816` to `0.059338`. Its old soft terminal decision was nevertheless
true-feasible in 0/160 runs.

V7 paired the same constrained posterior sampling trajectory with a
feasible-first terminal rule:

| Metric | V6 | V7 |
|---|---:|---:|
| search found a true-feasible point | 160/160 | 160/160 |
| true-feasible primary | 0/160 | 127/160 |
| true-feasible verified deployment | 130/160 | 151/160 |
| independently certified deployment | 130/160 | 147/160 |
| false independent certificate | 0/160 | 1/160 |
| median best-evaluated feasible regret | 0.059338 | 0.059338 |
| median deployed feasible regret | 0.286460 | 0.220868 |

The one V7 false certificate had true chance margin `+7.67e-5`, consistent
with a finite-sample tail event under the declared family-wise `delta=0.05`.
A `delta=0.01` sensitivity removed it but reduced certificate coverage.

## Post-run terminal error decomposition

The decomposition reran the exact promoted V7 contract
(`state_candidate_count=24`, inverse pool `512`, observed support scope) on
five seeds per scenario. Truth was joined only after all decisions froze.

- The best evaluated true-feasible policy existed in 40/40 runs, with median
  regret `0.067842`.
- Giving the fitted objective head the oracle feasible set selected a
  true-feasible policy in 40/40 runs, with median regret `0.072534`.
- The fitted posterior safe set had mean precision `0.7823`, mean recall
  `0.4431`, and contained the best evaluated policy in only 11/40 runs.
- Replacing fitted aleatoric variance with oracle variance reduced false-safe
  points from 37 to 13 but left mean safe recall essentially unchanged
  (`0.4431` to `0.4421`).
- Replacing both mean and variance with their oracle values removed all
  false-safe points, yet mean safe recall remained only `0.3683`; the current
  epistemic uncertainty plus a hard `P(violation)<=0.05` filter still rejected
  most shallow-interior feasible policies.

Thus the objective head is not the dominant terminal bottleneck, and further
HVD tuning cannot close the deployment-quality gap by itself. The dominant
loss is using a paper-grade posterior certificate as the candidate-admission
rule rather than as the final deployment test.

## V9 verification-aware terminal policy

V9 freezes an ordered three-policy shortlist before independent labels:

1. the minimum posterior objective among observed candidates with posterior
   violation probability at most `0.5` (posterior-median chance feasible);
2. the original V7 strict posterior-feasible primary;
3. the cumulative-risk safe-interior support.

The first candidate is a verification challenger, not a certificate.
Independent Gaussian quantile-tolerance verification remains the sole
deployment authority. Family-wise `delta=0.05` is split over all three
candidates, with precommitted budgets `80/128/128`.

Across the full 20-seed matrix, V7 and V9 had byte-identical search-primary
policies in all 160 cells:

| Metric | V7 | V9 |
|---|---:|---:|
| true-feasible primary | 127/160 | 127/160 |
| true-feasible verified deployment | 151/160 | 153/160 |
| independently certified deployment | 147/160 | 151/160 |
| false independent certificate | 1/160 | 1/160 |
| median deployed feasible regret | 0.220868 | 0.163552 |
| mean independent verification calls | 109.4 | 184.8 |

Paired outcomes were three feasibility rescues, one feasibility loss,
28 strict regret improvements, one strict regret loss, and 121 ties among
jointly feasible deployments. The sole V9 false certificate was a
`+2.84e-4` true-margin boundary case whose independent upper margin was
`-1.80e-4`.

The preregistered `delta=0.01` sensitivity produced 152/160 true-feasible
deployments, 149/160 certificates, zero false certificates, and median
deployed regret `0.193036`. It is reported as a conservative sensitivity;
the main `delta=0.05` result is retained rather than changed after observing
the boundary event.

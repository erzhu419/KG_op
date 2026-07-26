# V64 Powered Safe-Interior Verification Gate

## Motivation

V63 certified `59/60` fresh policies with zero false certificates and no
performance loss. Its sole unresolved policy was the correct, truly safe
primary; the failure was finite-sample verification power rather than
shortlist coverage or posterior selection.

A read-only prefix audit of all 60 independent V63 verification streams found:

| Rank-1 calls | Certified | False certificates |
|---:|---:|---:|
| 64 | 59/60 | 0 |
| 72 | 59/60 | 0 |
| 80 | 60/60 | 0 |
| 88 | 60/60 | 0 |
| 96 | 60/60 | 0 |
| 112 | 60/60 | 0 |
| 128 | 60/60 | 0 |

This audit is diagnostic, not confirmatory. V64 fixes the smallest supported
rank-1 budget, `80`, before evaluating untouched target seeds.

## Frozen Statistical Contract

- Search, posterior, primary, safe-interior selector, and online action sequence
  are identical to V63/V51.
- Rank 1 receives 80 independent replications.
- The frozen support receives 96 independent replications only after rank 1
  fails.
- Each candidate receives error probability `0.025`; the familywise
  false-deployment probability remains at most `0.05`.
- Verification samples cannot update the posterior or selector.
- All 80 or 176 verification calls are reported separately from the 13-call
  search budget and included in total simulation cost.

The larger rank-1 budget changes power, not the familywise safety theorem.
Lean instantiates the generic asymmetric-budget theorem at `80/96`.

## Registered Fresh Protocol

1. Extend frozen source-only initial designs to seeds `0..79`, preserving
   seeds `0..59` exactly.
2. Pair V51 and V64 on untouched target seeds `60..79`.
3. Use three domains, `d=1000`, `n0=10`, and `N_search=13`.
4. Require identical target design, online actions, and primary recommendation.
5. Require `60/60` certificates, zero false certificates, zero paired losses,
   and at least one strict paired gain.

V64 is promoted only if the fresh matrix passes every condition.

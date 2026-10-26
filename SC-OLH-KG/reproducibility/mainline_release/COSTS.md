# Historical execution costs

All values below come from the already exported physical attempt ledgers. Calls are paired simulator observations; CPU is process CPU seconds. Failed attempts are included. Each cohort contains 144 geometries and two observation streams.

Each frozen source archive used **384 paired calls** and is reused across target geometries and arms. This is a shared one-time cost per archive, not 384 new calls per target cell. Stage85/93 target execution made no new source calls or fits. The six Stage93 permuted-source fits and preparation consumed 58.828646 recorded CPU seconds and remain separate; historical original-source training CPU is available only through the linked historical accounts.

Each cohort paid **2,880 shared initial observations**. AFTER-native and FULL-native reuse their corresponding role search histories; a zero in their own online column means shared ownership. STOP uses only the shared initial10. Shared initialization, neighbor generation, scoring and coordination CPU are reported outside the per-arm table.

## stage85

| Arm | Own online calls | Verification calls | Attributable CPU s | Search CPU p50 / p95 / max s | Failed attempts / CPU s |
|---|---:|---:|---:|---:|---:|
| FULL | 864 | 66512 | 119120.498 | 389.05 / 543.71 / 596.60 | 0 / 0.000 |
| FULL-native | 0 | 67024 | 135.429 | shared | 0 / 0.000 |
| AFTER | 864 | 66400 | 64775.010 | 208.54 / 287.11 / 305.03 | 0 / 0.000 |
| AFTER-native | 0 | 66928 | 134.936 | shared | 0 / 0.000 |
| native | 864 | 68576 | 66901.747 | 221.91 / 301.96 / 330.67 | 0 / 0.000 |
| same-source GP | 864 | 66112 | 45768.025 | 149.94 / 201.29 / 234.43 | 0 / 0.000 |
| POINT | 864 | 66736 | 48364.840 | 159.90 / 212.91 / 248.59 | 0 / 0.000 |
| structural | 864 | 66448 | 118481.708 | 388.77 / 550.07 / 576.34 | 0 / 0.000 |
| STOP | 0 | 67088 | 134.422 | shared | 0 / 0.000 |

Shared initialization/neighbor/scoring/coordination CPU: **30869.985 s**. Original target-ledger total: **494686.598998 s**. Physical target calls: **609888**. These are allocations of the existing totals, not new charges.

## stage93

| Arm | Own online calls | Verification calls | Attributable CPU s | Search CPU p50 / p95 / max s | Failed attempts / CPU s |
|---|---:|---:|---:|---:|---:|
| FULL | 864 | 67632 | 117215.815 | 383.38 / 532.51 / 561.35 | 0 / 0.000 |
| FULL-native | 0 | 67888 | 146.064 | shared | 0 / 0.000 |
| AFTER | 864 | 67280 | 63909.337 | 210.78 / 276.11 / 284.16 | 0 / 0.000 |
| AFTER-native | 0 | 68304 | 146.948 | shared | 0 / 0.000 |
| native | 864 | 68336 | 65833.748 | 222.60 / 283.05 / 291.40 | 0 / 0.000 |
| same-source GP | 864 | 67248 | 44976.954 | 152.65 / 188.14 / 200.72 | 0 / 0.000 |
| POINT | 864 | 68272 | 47785.773 | 162.20 / 201.47 / 224.24 | 0 / 0.000 |
| structural | 864 | 67536 | 116594.450 | 386.97 / 530.16 / 556.08 | 1 / 366.583 |
| STOP | 0 | 68608 | 145.084 | shared | 0 / 0.000 |
| permuted AFTER | 670 | 53920 | 56890.848 | 193.87 / 273.52 / 286.96 | 66 / 8251.023 |
| permuted POINT | 864 | 68400 | 46757.232 | 159.24 / 198.72 / 214.11 | 0 / 0.000 |

Shared initialization/neighbor/scoring/coordination CPU: **28034.328 s**. Original target-ledger total: **588436.581870 s**. Physical target calls: **743022**. These are allocations of the existing totals, not new charges.

## Definitions and unknown quantities

Attributable CPU includes the recorded search, terminal-freeze and verification attempts bearing that arm. Per-attempt search p50/p95/max use linear interpolation at `(n−1)p`; failed searches remain in the distribution. Full phase totals and verification CPU p50/p95/max are in [cost_metadata.json](data/cost_metadata.json). Peak memory was not recorded and remains **unknown** for every arm.

An online system uses 10 initial plus 3 adaptive observations; STOP uses 10. Terminal branches actually used 208–336 calls in these cohorts, below the frozen cap of 352. Thus 365 online / 362 STOP are logical caps, not actual physical spending. Shared source, shared search, logical cell calls, child CPU and overlapping wrapper/controller CPU must not be added together.

Uninstrumented posterior/engine/source restoration counts, failed simulator invocations and internal RNG draws remain unknown. Stage93 retains all 67 failed searches, their 8,617.606061 CPU seconds and already returned observations; no missing outcome is set to zero. See [saved export](data/cells.json.gz), [recomputed tables](data/statistics.json.gz) and [execution receipt](receipts/saved_evidence.json).

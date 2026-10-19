# Joint source risk fit

Frozen source-only fit on tasks 02–05: 48 existing policy/task records with three raw replicates each. The original spectral g basis and four public mean directions are jointly conditioned through the same source likelihood. Source eta component masses and all eight outer identity masses use the resulting evidence ratios.

| g identity | Added common variance | Added task variance | Importance ESS / 1024 | Split log-evidence gap | Log-evidence ratio |
|---|---:|---:|---:|---:|---:|
| spectral/head_specific/deconvolved_task | 0.002182615 | 0.000056728 | 1005.20 | 0.003553 | 6.223154 |
| spectral/head_specific/observed_task | 0.002191194 | 0.000060024 | 1003.42 | 0.012884 | 6.233517 |

Both two-parameter optimizations converged in 14 iterations from zero variance; neither parameter is at its zero boundary. Zero increment gives exactly zero log-evidence ratio. The fixed ESS ≥ 100 / split-gap ≤ 0.1 criteria pass, so no additional source sampling was needed. Parameters are frozen before new target observations.

The f and eta prior parameters remain fixed. Updated outer identity masses can change mixed f predictions. Source evidence uses all 1024 positive components in each affected head.

Validation: 3 targeted tests passed in 0.647 seconds, covering actual 1024-component zero-increment equivalence, explicit joint Gaussian coefficient conditioning, and evidence/gradient consistency. Test names are recorded in `tests.json`.

Cost: one completed fit invocation, 4.548784287 CPU seconds, 2 source variance fits, 0 new labels, 0 target calls, 0 sampling sweeps. See `attempts.jsonl`; the executable artifact is `calibration.json`.

Limitations:importance ESS and split checks support the declared approximation; they do not constitute the earlier strict posterior precision certificate. Source fit improvement does not establish target decision improvement.

# Current protocol errata — 7 September 2026

This note corrects the description of completed experiments. The original
method specifications, execution manifests and recorded results remain
unchanged.

| Item | Correct interpretation | Executed source |
|---|---|---|
| Library seed | `family_seed=20260808`; the actual structural-library seed is `family_seed+991=20261799`. The `shared_library.seed=20260808` entry in `profile_atlas_v2_method_spec.json` is a reporting error. | `benchmark_profile_stress_suite.py` and `benchmark_external_energy_v3.py` |
| Native transfer budget | Ten initialization calls, thirteen total target search calls. Twenty denotes algorithm seeds per held-out domain, not the search budget. | `or_review_native_transfer_execution_v1.json` |
| Native transfer verification | Historical `v69`: candidate budgets 80/128/128 plus an 8-call objective-incumbent comparison. All eight native methods share it. | `or_review_native_transfer_execution_v1.json` |
| Primary synthetic/Energy V3 verification | Independent exact-binomial frozen-shortlist verification with candidate budgets 80/80/80. These experiments are analyzed separately from native transfer. | `profile_stress_v2_protocol.json`, `or_review_energy_forecast_indexed_v3.json` |
| Frequency penalty | Positive coordinate-wise scaling before per-column standardization cancels algebraically, including the diagonal-square columns. The recorded invariance is not empirical evidence of insensitivity to a meaningful tuning parameter. | `core/profile_atlas.py` |
| Original generic DCT comparator | Uses unstandardized structural coordinates. The source method uses standardized structural coordinates. Original results retain this actual definition. | `core/profile_atlas.py` |
| Added comparison | A separate source-free control uses the same standardized coordinates, with the original medoid first point and farthest-first rule. A same-source greedy portfolio is another separate control. These post-review additions use the existing tasks; their protocol does not relabel them as new independent confirmation. | `submission_revision_controls_20260907.json`, `submission_revision_energy_20260907.json` |

The prior evidence registry's `publication_ready` field records completion of
its historical evidence audit. It is not a current editorial or submission
readiness decision. Current manuscript claims use that frozen evidence with
this erratum and the separately recorded revision comparisons.

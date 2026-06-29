# Stage-C boundary replication pilot: local diagnostics

Run pulled back and diagnosed locally on 2026-06-23.

## Configuration

- Run id: `stageC_boundary_guarded_pilot_20260623`
- Problem: `RZDT2_VC`
- Method: `GPR-KG-VEPM`
- Budget: `N=80`, `n0=30`, `n_reps=3`, `seed_base=1000`, `sigma=0.04`
- Initial design: `common_random`
- VEPM: auto partition feature, robust update enabled
- Boundary replication: enabled
- Replication score threshold: `5e-4`
- Max observations per replicated solution: `3`
- Global replication budget fraction: `0.15`

With `N=80` and `n0=30`, the adaptive budget is `50`; the global cap is
`floor(0.15 * 50) = 7` replications per replication.

## Main comparison

Native final recommendation:

| run | HV | IGD | CVR | ND | mean time |
|---|---:|---:|---:|---:|---:|
| robust-none | 0.3683 | 1.0009 | 0.5000 | 4.00 | 192.40 |
| boundary-unbounded | 0.0000 | 1.7210 | 0.5333 | 6.33 | 241.09 |
| boundary-guarded | 0.0757 | 1.4696 | 0.2222 | 4.00 | 218.73 |

Common generic post-processing at `kappa=1.0`:

| run | HV | IGD | CVR | ND | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| robust-none | 0.4476 | 0.7512 | 0.4360 | 8.00 | 0.3805 | 0.2168 | 0.2710 |
| boundary-unbounded | 0.0000 | 1.5941 | 0.5093 | 9.67 | 0.3032 | 0.1912 | 0.2333 |
| boundary-guarded | 0.1584 | 1.0253 | 0.2870 | 9.00 | 0.3859 | 0.2940 | 0.3269 |

Boundary classification on the combined pool:

| run | F1 | precision | recall | margin RMSE | var2 ratio | near-boundary F1 | near var2 ratio | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| robust-none | 0.3086 | 0.3297 | 0.2919 | 0.0643 | 2.8962 | 0.5006 | 0.9593 | 361.33 | 432.00 |
| boundary-unbounded | 0.2408 | 0.3293 | 0.1916 | 0.0881 | 4.2245 | 0.4295 | 1.5988 | 242.00 | 465.33 |
| boundary-guarded | 0.3034 | 0.3576 | 0.2749 | 0.0622 | 3.1521 | 0.5258 | 1.1278 | 339.00 | 436.33 |

Replication accounting:

| run | rep | adaptive iterations | selected replications | cap | unique observed | HV | IGD | CVR | ND |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unbounded | 0 | 50 | 45 | none | 35 | 0.0000 | 1.7560 | 0.6000 | 5 |
| unbounded | 1 | 50 | 47 | none | 33 | 0.0000 | 1.3185 | 0.4286 | 7 |
| unbounded | 2 | 50 | 43 | none | 37 | 0.0000 | 2.0886 | 0.5714 | 7 |
| guarded | 0 | 50 | 7 | 7 | 73 | 0.0000 | 1.8981 | 0.0000 | 2 |
| guarded | 1 | 50 | 7 | 7 | 73 | 0.2271 | 0.7956 | 0.5000 | 4 |
| guarded | 2 | 50 | 7 | 7 | 73 | 0.0000 | 1.7150 | 0.1667 | 6 |

## Kappa sensitivity

Common generic post-processing for guarded boundary:

| kappa | HV | IGD | CVR | ND | precision | recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.1592 | 1.0186 | 0.4352 | 8.33 | 0.4659 | 0.6956 | 0.5577 |
| 0.75 | 0.1592 | 1.0186 | 0.3328 | 8.67 | 0.3800 | 0.4654 | 0.4182 |
| 1.00 | 0.1584 | 1.0253 | 0.2870 | 9.00 | 0.3859 | 0.2940 | 0.3269 |
| 1.25 | 0.0841 | 1.1726 | 0.6243 | 7.33 | 0.3429 | 0.1530 | 0.2059 |
| 1.50 | 0.0841 | 1.5796 | 0.7333 | 4.00 | 0.3017 | 0.0875 | 0.1293 |
| 2.00 | 0.0757 | 1.8996 | 0.8413 | 3.67 | 0.2280 | 0.0156 | 0.0285 |

## Diagnosis

1. Unbounded boundary replication is not viable.  It selected replication in
   43-47 of 50 adaptive iterations and reduced the number of unique evaluated
   solutions to roughly 33-37.  This collapsed exploration and produced very
   weak HV/IGD.

2. The global replication budget cap works mechanically.  The guarded run
   selected exactly 7 replications per seed and preserved 73 unique evaluated
   solutions.

3. Guarded replication improves some feasibility calibration quantities, most
   notably common-pool CVR and F1 at `kappa=1.0`, and near-boundary F1 is
   slightly higher than robust-none.  However, this does not translate into
   better HV/IGD.

4. The remaining failure is therefore not just lack of repeated observations
   at already visited boundary points.  The algorithm needs either better
   global constraint-mean modeling near the chance boundary or a new-point
   boundary exploration mechanism, not more replication of old points.

## Recommendation

- Keep the replication budget cap in the code as a safety mechanism because it
  prevents pathological over-replication.
- Do not use boundary replication as the main enhancement for the paper based
  on this pilot.
- The next promising direction is constraint-boundary new-point exploration:
  add new candidates near small posterior chance margin while preserving KG
  Pareto selection, or add a final calibrated recommendation layer.  This
  targets the observed boundary-location error without sacrificing exploration.

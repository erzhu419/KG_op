# Stage C Boundary-Feasible Candidate Pilot

Run id: `stageC_boundary_feasible_pilot_20260624`

Purpose: test a theory-compatible new-point boundary exploration rule for
`RZDT2_VC`.  The rule augments the online candidate set with unobserved
posterior-feasible points closest to the estimated chance boundary.  It does
not revisit old points and therefore keeps `N=80` unique evaluations.

Configuration:

- Problem: `RZDT2_VC`
- Method: `GPR-KG-VEPM`
- Budget: `N=80`, `n0=30`, `3` seeds
- Initial design: `common_random`
- Variance model: robust VEPM, `partition_features=auto`
- Replication: `none`
- Boundary candidate policy: `chance_feasible`
- Boundary candidates: `40` added from a random pool of `500`

## Key Results

Native final recommendation:

| run | HV | IGD | CVR | ND |
| --- | ---: | ---: | ---: | ---: |
| robust VEPM, no boundary candidates | 0.3683 | 1.0009 | 0.5000 | 4.00 |
| chance-margin two-sided candidates | 0.4376 | 0.9465 | 0.6583 | 6.67 |
| guarded boundary replication | 0.0757 | 1.4696 | 0.2222 | 4.00 |
| chance-feasible new-point candidates | 0.4707 | 0.9359 | 0.6071 | 5.67 |

Common-generic post-processing at `kappa=0.5`:

| run | HV | IGD | CVR | ND | feasibility F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| robust VEPM, no boundary candidates | 0.9205 | 0.4171 | 0.5000 | 12.33 | 0.4992 |
| chance-margin two-sided candidates | 0.5938 | 0.6606 | 0.4606 | 10.33 | 0.4913 |
| guarded boundary replication | 0.1592 | 1.0186 | 0.4352 | 8.33 | 0.5577 |
| chance-feasible new-point candidates | 0.6211 | 0.6527 | 0.3278 | 6.33 | 0.5124 |

## Sampling Diagnostics

- Boundary candidate augmentation was active in 29 adaptive iterations.
- Each seed added 1,160 extra boundary candidates to the candidate set.
- The actual selected evaluations were all new solutions:
  `selected_by_replication=0`, `new_flags=50` for each seed.
- About 51.2% of screened random boundary candidates were filtered out as
  posterior-infeasible.
- Observed near-boundary fraction was 15.4%, while the diagnostic candidate
  pool near-boundary fraction was 26.1%.
- Observed true-feasible fraction was 68.8%, reflecting the one-sided
  posterior-feasible filter.

## Interpretation

The one-sided `chance_feasible` rule is safer than two-sided chance-margin
candidate augmentation and avoids the severe budget loss caused by replication.
It slightly improves native HV/IGD over the robust-VEPM no-boundary pilot, but
it does not improve the common post-processing frontier relative to robust VEPM.
The remaining bottleneck is therefore not candidate access alone.  The dominant
failure mode remains posterior chance-margin miscalibration near the true
boundary.

This policy should remain an exploratory or appendix diagnostic.  It should not
replace the main GPR-KG-VEPM configuration unless a larger pilot shows stable
post-processing gains.

## Current Recommendation

For main experiments, keep:

```text
--replication_policy none
--boundary_candidate_policy none
```

Use `chance_feasible` only as a sensitivity run or as a diagnostic tool when
sampling-allocation diagnostics show insufficient new-point exploration near
the chance boundary.

# External energy/reliability replacement protocol

## Decision

SUMO remains an honest failed external-validity experiment. It is not tuned or
relabelled. The replacement external domain is a storage-reserve problem built
from the pinned Open Power System Data (OPSD) hourly time-series release.

This family was named as the missing independent application family before the
SUMO result was known. It is selected because it is a canonical operations-
research problem and supplies a different physical state equation:

```text
reserve/SOC policy
    -> intertemporal battery state
    -> stochastic forecast-error balancing
    -> cost and cumulative reliability
```

It is not selected using optimizer performance.

## Pinned data contract

- Provider: Open Power System Data.
- Package: `time_series`, version `2020-10-06`.
- DOI: `10.25832/time_series/2020-10-06`.
- Source file: `time_series_60min_singleindex.csv`.
- Source URL and HTTP metadata are stored by the preprocessor.
- The 130 MB source CSV is streamed through a temporary file and deleted.
- Only the compact derived NPZ is staged to the CPU cluster. Neither raw data
  nor checkpoints are committed or synchronized back with result summaries.

The preprocessor records the source SHA256, output SHA256, exact columns,
missingness, interpolation counts, market/year coverage, and package version.
No outcome-dependent filtering is permitted.

## Frozen market split

Markets were screened only for the presence of hourly actual load, day-ahead
load forecast, day-ahead price, solar generation, and wind generation. V1 uses
the aligned day-ahead load error. Solar/wind columns are retained for a future
registered renewable-forecast extension but are not combined with a fabricated
persistence forecast.

| Role | Market | Year | Use |
| --- | --- | ---: | --- |
| source | AT | 2017 | source-only archive |
| source | DK_1 | 2017 | source-only archive |
| development target | DK_2 | 2018 | formulation/certifiability year |
| confirmatory target | GB_GBN | 2018 | untouched formulation year |

The two source markets contribute the same `2 x 64 x 3 = 384` charged source
simulator calls as the existing paper contract. Every transfer baseline must
receive the same source archive and the same frozen initial design.

For a task labelled by formulation year `y`, entire calendar years are
disjoint:

- `y-1`: charged online search distribution;
- `y`: development diagnostics/formulation only;
- `y+1`: independent terminal verification only.

Thus DK_2 uses 2017/2018/2019 and GB_GBN uses the same three calendar years.
The confirmatory GB_GBN policy outcomes may not be read before the development
gate and all thresholds are frozen. This full-year split prevents a quarterly
season shift from making an otherwise valid chance constraint vacuous.

## OR model

The integer vector `x in {0,...,100}^d` is a target reserve/SOC trajectory.
One expensive evaluation samples an hourly forecast-error window and propagates
the battery state across all `d` hours. Positive net-load error consumes stored
energy; negative error may recharge the battery. The objective includes
day-ahead reserve procurement, cycling, spill, and unserved-energy penalties.

For sampled window `xi`, the stochastic constraint is

```text
G(x, xi) = unserved_positive_error_fraction(x, xi) - kappa <= 0,
P(G(x, xi) <= 0) >= 1 - alpha.
```

The initial physical constants are fixed on DK_2 before GB_GBN is opened:

- round-trip efficiency and power/energy limits;
- cost coefficients;
- reliability target `kappa=0.11`, frozen after the DK_2 certifiability audit;
- `alpha = 0.05`;
- target dimension and search budget.

The full-calendar-year audit invalidated the earlier quarterly formulation:
`energy_capacity=0.20`, `power_capacity=0.05`, and `kappa=0.07` yielded no
95%-reliable policy. The replacement is selected without running an optimizer.
A fixed 512-window formulation screen checks `kappa in {0.07,0.09,0.11,0.13}`
and assets no larger than `0.50` normalized load-hours or `0.50` normalized
load. It first minimizes `kappa`, then energy capacity, then power capacity,
subject to at least two reliable registered policies, at least one policy five
percentage points below the reliability target, and positive feasible-objective
spread. This selects `energy_capacity=0.40`, `power_capacity=0.40`, and
`kappa=0.11`. The selected configuration must then pass all 7,761 development
windows before any online benchmark is launched.

Changing these after confirmatory outcomes are inspected creates a new method
version and requires a new untouched market/year.

## Observable coordinate and leakage boundary

The external problem exposes a common state/trajectory record and one
interpretable cumulative coordinate `psi(x)=(A(x),N(x))`.

- `A`: low-reserve occupancy, reserve-edge exposure, policy ramp/cycling, and
  sustained low-reserve exposure.
- `N`: forecast-peak exposure, forecast-ramp exposure, and price-weighted
  common-system exposure.

These coordinates use only the policy, declared battery physics, calendar, and
day-ahead forecast/price fields. Actual target load and renewable realizations
are forbidden. The actual-minus-forecast error is read only inside a charged
simulation or the frozen independent verifier. Proposal materialization opens
the compact archive in outcome-disabled mode: the actual-load array is not
loaded, and any attempted simulator call raises an exception.

HVD is secondary in this domain. Pooled and cumulative-factor heads use the
same proposal, online actions, and verifier. HVD is promoted only if it improves
calibration, false certification, or verification cost without hurting regret.

## Evidence tracks

### Track A: cross-family stress

The existing frozen FactorShock/Inventory/Queue source atlas is applied to the
energy schema without any energy outcome. This is the strongest universality
test. Both descriptor-conditioned and domain-blind atlases failed the frozen
512-window coverage screen; this is retained as a negative result and is not
tuned further.

### Track B: energy LODO

AT and DK_1 source episodes train the structural proposal. DK_2 is the only
development target. After freezing, GB_GBN is the confirmatory target. This is
the primary realistic offline-to-online comparison.

Both source markets evaluate the same 64 formula-free low-frequency profiles
three times. A DC-mode monotone envelope is admitted only when every source
market has the same rank-correlation direction between normalized policy mean
and observed chance margin, with absolute correlation at least 0.25. The
envelope then extrapolates that source-agreed direction to one normalized
bound. It reads only source outcomes and held-out dimension/bounds; no DK_2
actual-load value or label is available. The frozen Track-B atlas passed the
512-window development screen; a single paired online smoke certified both
arms and favored LODO in objective, but is not inferential evidence.
This envelope is registered as the V3 front-end challenger rather than silently
relabelled as V1. If the energy gate passes, V3 must also pass a no-regression
gate on the original three synthetic domains before it can become the common
paper front end.

Both tracks report source, initial-design, online-search, verification, and
total calls separately.

## Sequential gates

1. **Schema gate:** pinned data load, no NaN windows, chronological disjointness,
   and exact provenance hashes.
2. **Certifiability gate:** a fixed formula-free policy library on DK_2 must
   contain both feasible and infeasible policies and a nontrivial objective
   spread. This checks the benchmark, not the optimizer.
3. **Five-seed development gate:** the frozen atlas coverage audit is one
   deterministic result, not five independent observations. The subsequent
   stochastic online runs use `d=1000`, `n0=10`, `N=13`, CPU-only. Require
   at least four independently certified deployments, zero false certificates,
   and no worse feasible regret than proposal-only.
4. **Twenty-seed confirmation:** run GB_GBN only after step 3 freezes the
   method. No failed method is repaired on GB_GBN.

The first CPU gate compares frozen energy-LODO initialization against common
Sobol, both with the same neutral Sobol continuation and independent verifier.
Only after this paired gate passes are SCBO and Stacked GP added. SAASBO is
deferred until GPUs are available and the CPU gate shows that the domain is
informative.

## Acceptance and disposition

The real-domain claim is supported only if the untouched confirmatory track
passes. Failure is recorded exactly like SUMO and does not trigger target-
specific anchors, threshold relaxation, or post-hoc source selection.

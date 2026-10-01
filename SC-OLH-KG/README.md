# Source-scored structural initial designs

This directory contains the implementation and evidence for the current
Operations Research manuscript in `manuscript/main.tex`, with its electronic
companion in `manuscript/supplement.tex`.

The method selects ten profiles from an outcome-free library of 64 ordered
policies using replicated source safety and objective scores. The selected
profiles initialize a replaceable target optimizer; deployment uses an
independent frozen-shortlist verifier. The public representation describes
ordered policy-grid refinement, including `d=10000`.

## Current paper and evidence

- Method: `performance/manifests/profile_atlas_v2_method_spec.json`, read with
  `reproducibility/current_protocol_errata_20260907.md`.
- Synthetic protocol: `performance/manifests/profile_stress_v2_protocol.json`
  and `or_review_confirmatory_execution_v1.json`; the completed evaluation
  implementation is `or_review_profile_evaluation_patch_v4.json`.
- Completed evidence: `paper_artifacts/or_review/final_evidence_registry_v1.json`
  lists 14 matrices and 17,360 cells, including one reported functional-control
  algorithmic failure.
- September 2026 revision: `performance/manifests/submission_revision_controls_20260907.json`
  and `submission_revision_energy_20260907.json`, with per-task outcomes and
  paired analyses in `paper_artifacts/submission_revision_20260907/`.
- September 30 revision: `performance/manifests/submission_revision_20260930.json`
  and `paper_artifacts/submission_revision_20260930/`: mechanism attribution,
  240 new latent tasks from three new source-family/library seeds. Corrected
  Energy V5 uses `paper_artifacts/submission_revision_energy_v5_20260930/` and
  `performance/manifests/submission_revision_energy_v5_20260930.json`.
- External application: Energy V5 gives every policy a common 50% initial
  inventory and reruns all 720 cells. The certification comparison remains unresolved across regions: the frozen source design certifies
  57/90 market-seed runs, compared with 54/90 for the original generic DCT control.
- Retrospective same-market reuse: `paper_artifacts/energy_archive_reuse_v2_20260930/`
  contains 5,400 rows over 216 market-month tasks. Neither the source-best design
  nor the full atlas improves certification or archive-inclusive call cost over
  the generic controls. This diagnostic remains separate from the manuscript.
- Independent public-data preflight: `paper_artifacts/independent_data_feasibility_20260930/feasibility_decision.json`
  records aligned Elexon forecast/outturn/price samples and four seasonal development
  previews. All 64 library candidates succeed on every sampled window; the additional
  zero-response reference alone fails in two seasons. This diagnostic is
  separate from the paper, and the candidate 2025 confirmation year remains unread.
- Public battery scope: V1-V5 are fixed historical dispatch-trace stress tests.
  BOAs do not respond to simulated baselines or availability declarations;
  these results do not establish counterfactual BM performance or source benefit.
- Public battery task: `performance/manifests/public_battery_dispatch_preflight_20260930.json`
  fixes a public two-hour 49 MW/98 MWh reference asset and receipt-time-aware
  reconstruction of a recorded BM unit's requests. The saved one-day workload peaks
  at 49 MW; summing revisions would incorrectly produce 111 MW. The instruction-only
  idle-gap diagnostic requires 59.60 MWh initial inventory, above the common 49 MWh
  reference. See
  `docs/submission_review_20260929/public_battery_dispatch_20260930.md`.
- Public battery controller: delayed half-hour nominations and MID-indexed energy
  accounting are implemented in `problems/public_battery_dispatch.py`. Development
  results in `paper_artifacts/public_battery_controller_development_v1_20260930/`
  give 0/64, 12/64 and 33/64 empirically feasible library profiles in April, July
  and October. April has dispatch-energy failures throughout the library and
  18/73 windows with requests above the 49 MW reference power. The full comparison
  is on hold; see
  `docs/submission_review_20260929/public_battery_controller_next_20260930.md`.
- Offline battery feasibility: `paper_artifacts/public_battery_physical_oracle_v1_20260930/`
  gives 55/73, 73/73 and 73/73 replayed physical witnesses in April, July and
  October. The remaining 18 April windows exceed the fixed reference power.
  All 201 LP solutions passed physical replay; all previously successful causal
  windows are included. Perfect-information witnesses do not establish causal
  feasibility or source benefit. Next: match the public full-site asset to both
  BM units and use received future instructions in inventory projection; see
  `docs/submission_review_20260929/public_battery_feasibility_next_20260930.md`.
- Whole-site battery development: both Pillswood BM units and receipt-aware
  nominations are implemented in `problems/public_battery_site.py`. The paired
  pilot has 10 common feasible pooled-inventory profiles versus 6 without BOA
  projection, but none of those 10 passes the independent-unit inventory
  sensitivity across all three periods. Source comparison remains on hold.
  Results: `paper_artifacts/public_battery_site_development_v2_20260930/`;
  next step: `docs/submission_review_20260929/public_battery_site_next_20260930.md`.
- Independent-unit battery control: `problems/public_battery_units.py` maintains
  separate inventories and delayed nominations under the shared site power
  limit. The unchanged development library has 0/64, 3/64 and 22/64 feasible
  profiles in April, July and October, with no common feasible profile. The 219
  forced-run capacity bounds showed no unit-capacity contradiction; the joint
  physical oracle follows below. The unit-control development record is
  see `docs/submission_review_20260929/public_battery_units_next_20260930.md`.
- Joint unit physical oracle: all 219 development windows have replayed feasible
  schedules with independent inventories and the frozen shared power bound.
  All V3 successful windows are included; the April library still misses all 73
  physically feasible windows. Perfect-information schedules establish the
  physical upper bound only. Next: trace nomination-time information at first
  failure; see
  `docs/submission_review_20260929/public_battery_unit_feasibility_next_20260930.md`.
- Nomination-time failure audit: all 4,745 frozen April first failures reproduce
  V3. In 4,720 cases the submitted prefix was safe under received information and
  later receipts changed execution; 25 already had a predicted violation in the
  fixed pending hour. No trace is unresolved. V4 inventory and shared-power
  reserves were frozen from this diagnosis; see
  `docs/submission_review_20260929/public_battery_visibility_next_20260930.md`.
- Empirical reserve control: the complete V4 development run gives 0/64, 1/64
  and 25/64 feasible library profiles, versus V3's 0/64, 3/64 and 22/64.
  Shared-power failures increase in every season and no common feasible profile
  emerges. The result remains HOLD; inventory-only and power-only component
  comparisons are frozen next. See
  `docs/submission_review_20260929/public_battery_reserves_next_20260930.md`.
- Reserve components: inventory-only gives 0/64, 1/64 and 24/64 feasible
  profiles; power-only gives 0/64, 3/64 and 24/64. Neither has a common feasible
  library profile, so the 90-minute empirical reserve family remains HOLD.
  Next: suffix recovery from actual stocks and locked nominations for the
  original 50% reference; see
  `docs/submission_review_20260929/public_battery_reserve_components_next_20260930.md`.
- Suffix recovery: 438 fixed reference windows reproduce V3/V4; 306 LP probes
  give 160 replayed witnesses with no unresolved result. Of 153 failed windows,
  30 remain recoverable at the relevant nomination, 100 lose recoverability
  between adjacent decisions and 23 are already unrecoverable at the preceding
  decision. Next: original-library nomination coverage at all 73 April V3
  recoverable reference snapshots; see
  `docs/submission_review_20260929/public_battery_recoverability_next_20260930.md`.
- Local action coverage: the original 64 candidates cover 50/73 recoverable
  April reference states; 23 have no feasible candidate action. All 73 reference
  negative controls are infeasible, with no unresolved result. The 23 gaps share
  one nomination event across overlapping windows. Next: exact coverage of the
  unchanged V3 continuous shared-target action family at those states; see
  `docs/submission_review_20260929/public_battery_action_coverage_next_20260930.md`.
- Continuous action coverage: all 161 analytic segments of the unchanged V3
  shared-target rule are infeasible at the 23 uncovered reference states; no
  result is unresolved. Increasing library size cannot fill these local gaps.
  Next: reconstruct independent unit targets from the 23 saved physical
  witnesses before registering a causal controller revision; see
  `docs/submission_review_20260929/public_battery_scalar_continuum_next_20260930.md`.
- Independent target reconstruction: all 23 saved physical actions are
  represented by separate unit target fractions, with maximum deviation
  1.42e-14 MW and no new LP or replay. The V5 causal controller protocol and
  154 joint candidates were frozen before V5, which has now completed below; see
  `docs/submission_review_20260929/public_battery_target_decoupling_next_20260930.md`.
- V5 complete: 31,536 new windows plus 2,847 reused rows give 0/154, 6/154
  and 26/154 feasible candidates in April, July and October; none is feasible
  in all three. Every April library window fails. The local target-expression
  repair has not passed the whole-window gate, which remains HOLD. Next:
  received-prefix classification from retained traces for two fixed constant
  pairs, without another matrix run; see
  `docs/submission_review_20260929/public_battery_unit_targets_next_20260930.md`.
- V5 receipt visibility: all 146 retained failures of the two fixed April
  constant pairs reproduce their saved first failing pieces and follow later
  receipt changes; no known-prefix violation or unresolved record remains.
  These overlapping histories involve 10 distinct absolute failure minutes.
  No new controller window or LP was run. Next: review the one-hour nomination
  commitment and historical BOA/baseline contract before another controller
  revision; source comparison stays HOLD. See
  `docs/submission_review_20260929/public_battery_unit_targets_visibility_next_20260930.md`.
- Battery information contract reviewed: the 2024 Grid Code supports gate-closed
  baselines and absolute BOA levels. Actual dispatch also uses updated export /
  import availability, which V1-V5 omit. Preserve their stress-test outcomes;
  source comparison stays HOLD. Next: specify causal availability with an
  explicit service obligation, then validate two small pulse cases before
  registering any historical rerun. See
  `docs/submission_review_20260929/public_battery_information_contract_next_20260930.md`.
- Battery availability component complete: two stock scenarios in both
  directions give four physically feasible pulses, two service successes and
  two retained service failures. Seven tests verify energy, independent stocks,
  gross site coupling and service deficits even without activation. No
  historical window was rerun. Next: six fixed nonzero-baseline / locked-plan
  fixtures, including no instruction versus an absolute zero-MW instruction;
  source comparison remains HOLD. See
  `docs/submission_review_20260929/public_battery_availability_next_20260930.md`.
- Nonzero-baseline availability complete: all six frozen fixtures are retained;
  five are physically feasible, two satisfy service and one preserves an unsafe
  locked-plan failure. Eight tests cover absolute versus relative power, absent
  versus zero instructions, the locked tail and internal ramp extrema. Next:
  received-BOA commitments plus the retained 146 April nomination states; no
  new historical window or source comparison. See
  `docs/submission_review_20260929/public_battery_pending_availability_next_20261001.md`.
- Received-BOA availability complete: three of four component fixtures satisfy
  physics and service; one preserves a known commitment failure. All 292 fixed
  49-MW-per-unit probes at the 146 saved April states satisfy service. The states
  cover only eight absolute nomination times and ten original failure times;
  single-pulse success does not establish weekly reliability. Fourteen targeted
  tests pass. Next: a frozen four-run sequential service pilot with hourly public
  receipt-direction marks and delayed PN; it is a new simulation task. No new
  historical rollout or 2025 access occurred; source comparison stays HOLD. See
  `docs/submission_review_20260929/public_battery_received_availability_next_20261001.md`.
- Sequential service pilot complete: both admitted runs finish 168 hours
  physically, but each has 95 capacity-shortfall hours and 58 partial requests.
  Both full-request references exhaust a unit at minutes 430 and 445; none of
  the four windows meets service. Seven integration tests pass. The first deficit
  occurs without activation at minute 60: delayed replenishment PN leaves too
  little shared power for the fixed obligation. A numerical accumulation repair
  retains initial outputs and all eight trial/replay evaluations. Next: an
  aggregate-energy necessity bound for the unchanged 49-MW product, without a
  larger controller run. Source comparison stays HOLD and 2025 remains sealed.
  See `docs/submission_review_20260929/public_battery_causal_service_next_20261001.md`.
- Full-service energy bound complete: full bidirectional 49-MW-per-unit duty
  at every hourly clock forces near-zero net PN under the 98 MW gross limit.
  Even an optimistic total-stock bound becomes negative at minute 500
  (-0.109362 MWh). This constructed product is conditionally infeasible for any
  PN policy satisfying its definition. Nine targeted tests pass; no controller
  or optimizer was run. Next: a separate absolute-dispatch tracking task, with
  four component fixtures and four bounded pilots; retain all prior negatives
  and distinguish the task change from policy benefit. Source comparison remains
  HOLD. See `docs/submission_review_20260929/public_battery_full_service_bound_next_20261001.md`.
- Absolute-dispatch pilot complete: all four component fixtures and 17 targeted
  tests pass. Both admitted runs safely finish 168 hours but each fulfills only
  60/95 calls; full-request references deplete unit 1 at minutes 428 and 439.
  No window satisfies service. The first deficit at minute 420 follows an
  export whose delayed recharge is then overridden by the next request. Next:
  four saved-state timing replays with one earlier PN nomination, preserving
  all negatives and treating the intervention as a diagnosis. No broader matrix
  or 2025 access; source comparison stays HOLD. See
  `docs/submission_review_20260929/public_battery_absolute_dispatch_next_20261001.md`.
- Replenishment timing complete: both saved original branches reproduce depletion
  at minutes 428 and 439; both earlier-PN interventions safely complete 150
  minutes and fulfill both calls. An export receipt was already visible at the
  minute-330 nomination, before the later hourly mark. Next: frozen causal
  receipt-forecast nominations with inventory-priority allocation, four probes
  and four bounded pilots on the unchanged absolute task. No weekly rollout or
  new data request in this timing stage; 2025 stays sealed and source comparison
  remains HOLD. See
  `docs/submission_review_20260929/public_battery_replenishment_timing_next_20261001.md`.
- Causal receipt-forecast pilot complete: four nomination probes and 29 targeted
  tests pass. Both admitted runs safely finish 168 hours, fulfilling 64/95 calls
  versus the ordinary controller's 60/95. Full-request references overflow unit 2
  at minute 1022; no window satisfies service. The first import forecast was
  correct but its recovery PN starts too late. An execution-origin numerical
  repair preserves initial outputs and all eight weekly trial/replay evaluations.
  Next: a conditional two-call information necessity bound and six component
  paths, without another controller run. All negatives remain; 2025 stays sealed
  and source comparison is HOLD. See
  `docs/submission_review_20260929/public_battery_receipt_forecast_next_20261001.md`.
- Two-call information bound complete: six component paths and six targeted
  tests pass. The first-sign stock separation is at least 50.673101 MWh,
  exceeding the optimistic second-call stock-band width of 49.375903 MWh by
  1.297198 MWh with the original tolerances retained. This rules out a uniform
  guarantee over all four active sign pairs under the frozen information
  schedule. It does not prove infeasibility of the actual 95-call history.
  Next: cached adjacent-call and nomination-information coverage only; no
  controller rollout, new data or 2025 access. Source comparison remains HOLD.
  See `docs/submission_review_20260929/public_battery_two_call_bound_next_20261001.md`.
- Cached nomination-information audit complete: 67 adjacent active pairs cover
  90 of 95 requests, with all four sign combinations observed. Deadline cues
  match the second sign in 34 pairs; late cues do so in 52, but their PN starts
  near the pulse end. In 55 pairs the selected first-call receipt arrives after
  the critical deadline. Three counting tests pass; no trajectories or controller
  runs. Next: eight frozen advance-notice counterfactual component paths, retaining
  capacities, peaks, original saved states and all prior negatives. Source
  comparison remains HOLD. See
  `docs/submission_review_20260929/public_battery_information_coverage_next_20261001.md`.
- Advance-notice component preflight complete: all eight 120-minute paths
  deliver both full pulses safely. Minimum stock is 0.710855 MWh, minimum
  capacity headroom 1.208667 MWh, and maximum site gross power 98 MW. Two
  targeted tests pass, with one additional test trajectory. Next: define a
  separate announced simulation task and verify its causal announcement ledger,
  preserving all 95 signs and peaks while shifting delivery by 120 minutes.
  No weekly controller or new data request; all previous negatives remain and
  source comparison stays HOLD. See
  `docs/submission_review_20260929/public_battery_announced_pair_next_20261001.md`.
- Announced-task interface complete: 168 binding announcements and 170 delivery
  clocks preserve all 95 obligations, with 73 inactive announced slots and two
  warmup slots. All announcements are visible 30 minutes before the critical
  nomination cutoff. Five targeted tests pass, including future-prefix isolation
  and binding-request preservation. No physics or controller was run. Next:
  eight automatic stock-interval nomination probes and at most eight short paths,
  before defining a weekly policy. Source comparison remains HOLD. See
  `docs/submission_review_20260929/public_battery_announcements_next_20261001.md`.
- Automatic announced nominations complete: all eight probes and eight paths
  pass, with 16 full pulses physically deliverable. Minimum stock is 4.379938
  MWh; maximum projected-versus-executed stock error is 3.55e-14 MWh. Five
  targeted tests pass. The probes use 1,776 scalar projections, at most 246
  per nomination. Next: complete inactive-slot and own-pulse-prefix behavior,
  pass 22 gated components, then run four frozen 170-hour announced-task pilots.
  No weekly run or new data in this stage; source comparison remains HOLD. See
  `docs/submission_review_20260929/public_battery_announced_nomination_next_20261001.md`.
- Complete announced controller: all 22 components and 46 affected tests pass.
  Each of four 170-hour pilots safely fulfills 95/95 requests; admission and
  full-request execution agree for each retained target pair. There are no
  planning, known-commitment or physical failures, and no implementation replays.
  Next is the frozen three-season screen of the same two profiles: 73 original
  source windows per sample, 872 new evaluations plus four reused pilots.
  Source comparison remains HOLD; 2025 stays sealed. See
  `docs/submission_review_20260929/public_battery_announced_controller_next_20261001.md`.
- Three-season announced screen complete: 872 new windows and four reused pilots,
  with five new interface tests and no controller replays. The 0.45/0.75 pair
  safely fulfills every request in 73/73 windows in each sample under both rules;
  the 0.35/0.75 pair passes April but fails all July and October windows.
  All 876 cells and 292 failure contexts are retained. Next: original joint-library
  decoding, forecast/price boundary coverage and economic accounting from the
  four saved pilots, before dynamic-target integration. Source comparison stays
  HOLD; 2025 stays sealed. See
  `docs/submission_review_20260929/public_battery_announced_development_next_20261001.md`.
- Announced library/objective preflight complete: the original 154 ordered
  candidates decode unchanged, and all 74,022 eligible decisions pass historical
  forecast visibility. Six small requests add nine missing half-hours (3,198
  response bytes); original CSVs and outcomes remain intact. Eight new tests
  pass, and exact economic accounting of the four saved pilots agrees with
  their gross flows and inventory within 8.46e-11 MWh, with no controller or
  trajectory replay. Next: six short dynamic-target components followed by
  24 registered functional-pair pilots across the first and last windows of
  each sample. Source comparison remains HOLD; 2025 stays sealed. See
  `docs/submission_review_20260929/public_battery_announced_library_next_20261001.md`.
- Dynamic-target stage complete: all six short components and nine interface
  tests pass, with exact constant-vector/schedule equivalence. Both registered
  functional pairs fail service in all 24 trials. Six observed tolerance-boundary
  interruptions were traced to differing hour-origin accumulation, repaired
  without changing tolerance or physical integration, and explicitly replayed;
  16 related regression tests pass. All 12 conservative trials now complete
  physically, while all 12 full-request trials interrupt; success remains 0/24
  and every full cost remains undefined. Initial outcomes and all call counts
  are retained. Next: eight failure-stopping equivalence cases before the
  registered original-library population batches, with 580 missing economic
  replays counted separately. No matrix has launched, no new data were requested,
  and source comparison remains HOLD. See
  `docs/submission_review_20260929/public_battery_announced_dynamic_next_20261001.md`.
- Population failure-stopping equivalence passes all eight saved cases; four
  stopping-interface tests and five parallel-execution tests pass. The complete
  population now covers 68,766/68,766 unique cells, including 900 reused results.
  The remaining 46,731 cells ran as 4,096 logical shards in 144 four-worker
  scheduleurm tasks, with automatic collection; the scheduler-observed batch
  span was 9.8 minutes. Tasks used node001, node002, node004, node005 and node006;
  node003 was unavailable under the scheduler's live placement policy.
  All 580 economic replays are complete, 22,310 successful-window costs are
  saved, and 28 actual joint candidates meet at least 70/73 successes in each
  season under both rules: 26 constant pairs and two functional pairs.
  Completed journals contain 67,866 new executions, 580 economic replays and
  eight equivalence replays. Two cancelled serial processes may have consumed
  additional unrecorded calls; their interruption records remain separate.
  Source comparison remains HOLD and 2025 remains sealed. Complete population
  journals stay on the server; local summaries describe the full remote result,
  while local large JSONL files remain the migration snapshot. See
  `docs/submission_review_20260929/public_battery_announced_population_next_20261001.md`.

- The complete 2023 announced source archive covers 44,968/44,968 cells,
  including 32 reused pilots, with 37,010 complete costs and 7,958 null failed
  costs. All 4,096 shards and 144 four-worker scheduler tasks completed.
  Source first-center selection uses only common complete source cost support;
  both ten-point designs share full-library standardized two-channel geometry.
  The source center is 0.75/0.75; the structural medoid is 0.55/0.55.
  Retrospective 2024 initial feasible coverage is 3/3 versus 1/3 samples,
  with no new controller calls. This is not certified deployment or compute
  efficiency. Full journals stay remote, full augmented atlas remains
  unconnected, source comparison remains HOLD and 2025 remains sealed. See
  `docs/submission_review_20260929/public_battery_source_archive_next_20261002.md`.

- Fixed-budget retrospective target replay completed all 30 paired runs.
  Source first-center designs certified 15/15 versus 9/15 structural designs,
  with zero finite-population false certificates and zero new simulations.
  All deployed profiles were constant pairs; source first center deployed in
  12/15 source runs. Including the 44,968 paid source-window evaluations,
  three seasonal tasks do not recover source cost (represented queries per
  expected certificate about 15,082 versus 262). Full augmented atlas remains
  unconnected and 2025 remains sealed. See
  `docs/submission_review_20260929/public_battery_target_search_next_20261002.md`.

- Simple constant controls completed 30/30 scheduler runs. Constant-only
  structural search certified 15/15 with the same 279 represented target queries
  and no source cost. Paid full-library census costs 33,726 queries across three
  seasons, below source all-in 45,247, and selects the same best complete-cost
  constants as the constant-only census. Development source-efficiency claim
  is closed; HOLD and sealed 2025 are retained. See
  `docs/submission_review_20260929/public_battery_simple_controls_next_20261002.md`.

- Theory: `../proof/final_or_theory.md`; original checked paper interface:
  `../proof/SCOLHKG/PaperProofInterface.lean`.

The primary synthetic matrix contains 160 independently seeded latent tasks
in eight fixed mechanisms, each evaluated at three resolutions. The added
revision compares the frozen source design against a source-free design with
identical structural standardization and a simple greedy portfolio supplied
with the same source observations. These comparisons use existing tasks.
New confirmation gives 56/240 true certificates for source scoring versus
24/240 for standardized generic. Most of the effect comes from the source-selected
first center; the portfolio has better coverage and loss. The new design-stability
proposition has an analytic proof outside the original Lean interface.
October 2 fixed-first remainder controls give 53/240 certificates for random
nine-point fill, 54/240 for structural coverage and 56/240 for the atlas;
paired intervals cross zero. These postdecision controls do not establish a
geometric selection advantage. All 250 new results and 230 identical-design
reuses are retained in `paper_artifacts/source_first_remainders_v1_20261002/`.
The completed decision is documented in
`docs/submission_review_20260929/source_remainders_next_20261002.md`.

The current reproduction guide is `reproducibility/README.md`. It covers
analysis from compact per-cell evidence, simulation entrypoints, the exact
Energy preprocessing parameters, figure/table rendering, PDF compilation and
the local anonymous attachment exporter.

## Build the manuscript

Run from the repository root:

```bash
python3 SC-OLH-KG/performance/render_submission_revision_20260930.py
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error SC-OLH-KG/manuscript/main.tex
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error SC-OLH-KG/manuscript/supplement.tex
```

Generated simulation output belongs under `results/`, `profiles/` or
`checkpoints/`. The compact submission attachment is written to
`reproducibility/dist/or_submission_revision_controls_20261002.zip`.

## Historical implementations

The directory name preserves the project's SC-OLH-KG origin. Acquisition,
HVD, state-encoder, endpoint-replacement and earlier risk-objective-atlas
experiments remain historical implementations and ablations. In particular,
`paper_final_method_v1.json`, `paper_submission_experiment_registry_v1.json`,
`external_energy_reliability_v1.json` and their earlier positive Energy
confirmation describe preceding experiments. They are not the current paper's
method contract, experiment registry or external-validity conclusion.

Every current comparison reports source, initial target, adaptive target,
verification, total and amortized calls separately. Outcome-adjusted cost per
successful certification is a separate quantity from the call-count
amortization break-even.

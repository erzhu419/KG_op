# V31 History-Measurable Expert Safety Race

## Evidence And Scope

V30 preserves FactorShock but reaches only `3/7` Inventory feasibility. Its
stage-18 audits show that safe support often appears only after the first
reserved label changes the expert predictions. A finalist set frozen at stage
17 cannot use that information. V31 changes only this temporal restriction;
the expert family, source prior, GPR/HVD updates, exact KG, candidate pool,
budget, and certification rule remain unchanged.

## Algorithm

Let `R` be the evaluations already reserved inside `N`. At each reserved stage
`t`:

1. form the terminal pool using only the current history;
2. recompute every finite expert's minimum expected-positive-violation
   nomination;
3. choose the best current nomination without multiplying by posterior mass;
4. archive the nomination and its pre-observation metadata;
5. charge the next simulator call to the current challenger;
6. update every GPR, HVD, and task posterior before the next nomination.

The finalist universe is therefore predictable: the stage-`t` action is
measurable with respect to the history before observation `t`. Old challengers
remain in an audit archive, but an incomplete old challenger cannot veto a
later completed one. At termination, only actions with at least the requested
replicate count enter the empirical safety-first comparison. Theory-certified
actions retain precedence, and the empirical fallback is never relabelled as
the theory certificate.

V31 keeps V30 as the default-off frozen ablation through
`finalist_replication_adaptive_race=False`.

## Statistical Contract

The nominal final-race error budget is split across the deterministic maximum
number of archived candidates, `finalist_count + R`. Conditional on each
history, the next target is fixed before its fresh observation. A finite union
bound therefore covers every completed adaptive finalist. This is a bounded
adaptive experiment, not post-selection reuse of uncharged simulator output.

## Gate

1. Unit tests cover refresh after a charged label, archive persistence,
   incomplete-candidate exclusion, exact `N` accounting, and checkpoint
   resume.
2. Smoke FactorShock seed 0 and Inventory seeds 0 and 1 at `N=20`.
3. FactorShock must remain feasible; Inventory seed 0 must not regress; seed 1
   must test whether the newly visible null nomination can be completed.
4. Only a passing smoke may repeat the unchanged 7+7 gate. The promotion rule
   remains FactorShock `7/7`, Inventory at least `4/7`, at most one false
   certificate, and valid complexity selection in every seed.

## Proof Obligations

- every adaptive nomination is pre-observation measurable;
- the archive contains every action actually charged by the race;
- at most `R` new challengers can be introduced;
- a union of per-candidate confidence events controls the selected completed
  finalist;
- excluding incomplete candidates cannot create an empirical certificate;
- total simulator calls remain exactly `N`.

## Controlled Smoke Rejection

V31 preserves FactorShock seed 0 and Inventory seed 0, but does not rescue
Inventory seed 1. The three reserved stages nominate three distinct ordered
actions, each receives one observation, and no action reaches the minimum
replicate count. The fallback therefore correctly reports
`no_completed_finalist` and returns the original unsafe Bayes action.

This churn is caused by regenerating a different terminal pool at every
reserved stage. A checkpoint audit on the stage-17 pool shows a safe null
nomination after the first charged label, but that action is absent from the
newly regenerated stage-18 pool. V31 is rejected and no 7+7 gate is run.

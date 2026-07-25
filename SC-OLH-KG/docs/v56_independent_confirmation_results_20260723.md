# V56 independent-confirmation sentinel result

## Scope

Run `scolh_v56_independent_confirmation_mc512_c48_s5_20260723_01`
contains 45 paired cells at `d=1000`, `N=11`, `n0=10`: V51-MC512,
V56-confirm2048, and V56-confirm4096 on FactorShock, Inventory, and Queue
with seeds 0--4. All jobs completed without retry or target-oracle decisions.

## Statistical admission result

The finite-look independent confirmation is nonvacuous. Confirm2048 rejected
both nonpositive-gain nulls in 14/15 cells; confirm4096 did so in 15/15. The
2048 failure was one small-gain FactorShock cell. Median confirmation usage was
1536 samples for FactorShock and 512 for Inventory/Queue. Confirm4096 is the
only retained challenger.

## Distinct certificate obligations

This closes the finite-sample action-admission obligation, not the chance-
feasibility certificate. Every one of the 45 posterior certified sets remained
empty. Therefore zero false feasibility certificates is vacuous and cannot be
presented as useful safety coverage.

## Optimization result

Both V56 variants produced exactly the same final recommendation as V51 in all
15 paired cells. Every arm finished with 14/15 true-feasible recommendations,
zero adaptive improvements, and the same Queue seed-3 adaptive loss. Thus the
formal gate passes, but the empirical promotion gate fails: 0 wins, 0 losses,
15 ties for each challenger.

One online decision is insufficient to test whether repeatedly admitted
actions alter the posterior terminal decision. The next preregistered gate
compares V51 against confirm4096 at `N=13`, retaining five seeds and all three
domains. It must separately report action-confirmation activation, posterior
certificate coverage, recommendation changes, paired regret, rescue/loss, and
runtime substage timings.

## N=13 multistep result

Run `scolh_v56_multistep_n13_mc512_c48_s5_20260723_01` completed all 30
V51/confirm4096 cells. One action had a nonpositive pilot joint score and
correctly skipped confirmation with zero samples. The original analyzer
incorrectly required independent-stream fields on that declared no-stream
state; the corrected contract accepts only
`skipped_nonpositive_pilot_joint_score` with `passed=false`,
`sample_count=0`, and no target-oracle use. The V56 formal contract therefore
passes.

V56 is not promoted. Relative to V51 it produced two wins, two losses, and
eleven ties, with four changed recommendations, no feasibility rescue, and one
Queue feasibility loss. Inventory seeds 1 and 4 improved from regret
`0.0323070` to approximately `0.0109036`. Queue seed 2 changed to a slightly
worse feasible recommendation even though the better point was evaluated, and
Queue seed 3 lost the feasible incumbent.

The chance-feasibility certificate remained empty in all 30 runs. V56's
independent action certificate is therefore nonvacuous, but it did not solve
chance-certificate coverage.

Median V51 runtime was about 1.60 hours. Median V56 runtime was about 2.27
hours and the maximum was about 2.44 hours. Exact posterior-fantasy KG occupied
approximately 94 percent of V56 online stage time; independent confirmation
used a median total of about 8.7 minutes per run and was not the main
bottleneck. The follow-up V57 gate retains the confirmation theorem and adds a
horizon-spent posterior-safe terminal incumbent.

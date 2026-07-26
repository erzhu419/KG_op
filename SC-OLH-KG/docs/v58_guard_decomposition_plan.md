# V58 Guard-Decomposed Action Support

## Motivation

V57 controlled posterior switching by retaining one sequential incumbent. Its
5-seed gate made zero switches, lost V51's Inventory improvements, retained the
Queue failure, and left all 15 posterior certificate sets empty. V58 therefore
restores the promoted V51 posterior Bayes-risk terminal decision and changes
only the charged action support.

## Posterior object

For the observed action with minimum conservative posterior chance margin,
V58 decomposes

```text
M_D(x) = mu_g(x) - tau
       + sqrt(beta_g) s_g(x)
       + joint source/task epistemic guard
       + z_alpha sqrt(v_C_plus(x))
       + favorable coupling correction.
```

The signed coupling correction preserves exact reconstruction of the robust
joint margin. Positive correction belongs to epistemic uncertainty; negative
correction is logged separately and never treated as a guard.

## Action support

- Epistemic dominant: add a nearby point in observable cumulative-risk
  coordinates and a local epistemic-information point.
- Aleatoric dominant: keep the anchor replication active and add nearby
  boundary support as alternatives.
- Mean/interior dominant: add deep-interior and safe Bayes-risk support.

Every V58 action set is a literal superset of the promoted V51 set: four
posterior-risk new actions plus every eligible replication remain available.
The supplemental support uses only charged target observations, frozen source
information, and observable `psi=(A,N)` features.

## Selection and terminal rule

The exact fantasy update still refits the constraint GPR and cumulative HVD.
The pilot may nominate one supplemental action, but an independent finite-look
confirmation must support positive Bayes-risk and certificate-deficit
reductions. Failure executes the literal V51 action. Final recommendation uses
the V51 observed posterior Bayes-risk terminal rule; V57 posterior dominance is
disabled.

## Gate

Run a paired `d=1000, N=13, n0=10`, three-domain, five-seed matrix:

```text
v51_control vs v58_guard_decomposed_support
```

Promotion requires:

1. all implementation/theory contracts and target-oracle exclusions pass;
2. no paired performance or feasibility loss relative to V51;
3. at least one independently confirmed action change;
4. at least one sound nonempty posterior certificate;
5. no false certificate; and
6. paired minimum certificate margins are majority nonworse.

Until this gate passes, `promoted_v51_observed_terminal_closure` remains the
baseline.

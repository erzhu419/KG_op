# Final Method Contract

## Frozen identity

The final Operations Research method is the V2 source-scored structural
initial design for chance-constrained ordered policy profiles:

```text
exogenously supplied replicated source archive
  -> source scores on a public 64-profile library
  -> 10-point augmented-coordinate farthest-first design
  -> replaceable target backend
  -> independent frozen-shortlist verifier.
```

The primary attribution experiment sets `N=n0=10`, so there is no sequential
backend. KG, SAASBO, SCBO, Thompson sampling, and historical V51--V69 policies
are comparisons or development artifacts, not the claimed method.

The executable method specification is
`performance/manifests/profile_atlas_v2_method_spec.json`. Frozen evidence is
indexed by `paper_artifacts/or_review/final_evidence_registry_v1.json`.

## Information contract

The profile library is generated without any source or target outcome. Source
records contain two supplied tasks, 64 library profiles, and three ordinary
replications per task, for 384 source calls. The method may use target grid,
bounds, dimension, and declared ordering before the target opens. It may not
use target outcomes, target feasibility labels, hidden noise, optimum,
safe-center geometry, or verification responses.

Source-task retrieval is not part of the algorithm. Synthetic source tasks are
regime matched by the experimenter. Energy excludes the held-out target region
from its source archive and is the archive-mismatch control. The primary
source aggregation is domain blind; descriptor-conditioned weighting is an
unpromoted control.

## Finite selection rule

For each profile, `z` is the standardized 18-dimensional exact cosine
coordinate with frequencies 0--8 and diagonal squares. Source objective and
Gaussian chance-margin percentile ranks are `r_f` and `r_g`. The algorithm
selects in

```text
eta = (z, r_g, r_f).
```

The first center minimizes `.5*r_g + .5*r_f`; nine more centers use Gonzalez
farthest-first with stable index tie breaking. Profiles are linearly
interpolated, clipped, and deterministically rounded on the target grid.

## Theory contract

Selection coverage is measured in augmented `eta`, while target transfer is
measured only in structural `z`. The projection `eta -> z` is nonexpansive, so
an augmented cover of radius `r` is also a structural cover of radius at most
`r`. If implemented projected structural-coordinate error is `epsilon_eta`, source-target
safe-support shift is `Delta`, chance margin is `L`-Lipschitz in the ideal
structural coordinate, and safe depth is `gamma`, then

```text
L * (r + Delta + 2*epsilon_eta) <= gamma
```

forces a selected feasible profile. This is conditional and nominal raw grid
dimension is absent only because the ordered profile representation is an
assumption.

For Gaussian source replications, explicit normal-mean and chi-square
sample-scale radii bound the floored source chance-margin error. Separated
score gaps larger than twice the uniform radius retain their order. The Lean
bridge proves finite-sum variance algebra, floor-aware propagation, event
union, projected geometric coverage, and rank recovery. Classical Gaussian and
Cochran distribution identities remain explicit external probability laws.

## Task and cost units

The randomized study has eight fixed equally weighted regimes and 20
independently seeded latent tasks per regime. The same 160 seeds are crossed
with dimensions 200, 1000, and 10000. The resulting 480 task-resolution cells
per design are descriptive; inference and stratified concentration are applied
separately by resolution.

Every result separates source calls `S`, target-search calls `N`, and fresh
verification calls `V`:

```text
C_all = S + N + V
C_amort(M) = S/M + N + V.
```

Outcome-adjusted efficiency additionally divides by true-feasible
certification probability. No scalar operational loss is asserted without
decision-maker costs for unsafe deployment, abstention, and objective quality.

## Claim boundary

The supported claim is conditional: source outcomes can improve a tiny
structural initial design when the supplied source archive and target share
useful ordered-profile geometry. The method is not a generic high-dimensional
optimizer and does not solve source retrieval. Registered adverse regimes,
the Energy reversal, equal-preverification loss, and all algorithm failures
remain in the evidence.

Cumulative factor-HVD improves variance calibration but not matched feasible
recovery, regret, false certification, or verification cost. It remains an
appendix diagnostic, not a coequal optimization contribution.

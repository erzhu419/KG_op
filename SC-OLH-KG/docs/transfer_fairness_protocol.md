# Transfer-Fair Evaluation Protocol

## 1. Fixed information contract

The primary transfer comparison is leave-one-domain-out (LODO).  For each
held-out target, every method receives the identical immutable archive from
the other two domains:

Here, *source* means the two domains whose completed records are available
before optimization starts.  *Target* means the third, held-out domain on
which the `N_target` online calls are spent.  No method is trained on the
held-out target and then evaluated on that same target archive.

| Quantity | Formal value |
|---|---:|
| Source domains | 2 |
| Profiles per source domain | 64 |
| Noisy replications per profile | 3 |
| Source simulator calls | 384 |
| Target dimension | 50 |
| Target initial calls, `n0` | 10 |
| Total target calls, `N_target` | 20 |
| Total calls before amortization | 404 |

The archive stores the raw replicated observations, sample means, replicate
variances, design origins, and a SHA-256 fingerprint.  It contains no analytic
source means, true source variances, target truth, or problem-specific target
hooks.  Target initial points use the same scrambled Sobol design in the
representation-matched table.

The historical `shared_archive_n20` protocol is not a full transfer baseline.
It uses the archive only to choose warm-start points for an otherwise ordinary
BoTorch optimizer.  It is retained as a labelled `warm_start_only` ablation.

## 2. What “fine-tuning” means

The promoted SC-OLH-KG implementation does adapt online, but it does not run
end-to-end neural gradient fine-tuning.  Its source prior remains frozen while
budgeted target observations update the target objective and constraint GPRs,
cumulative HVD, finite-expert posterior weights, each expert's target GPR/HVD,
and admitted target basis/risk-alignment gates.  The update occurs after every
target simulator call.

| Method | Source training | Target-domain adaptation | Gradient fine-tune? |
|---|---|---|---:|
| Promoted SC-OLH-KG | cumulative-risk prior, experts, proposals | GPR/HVD conditioning, expert reweighting, budgeted representation refit | No |
| Safe F-PACOH | function-space GP hyper-prior | posterior conditioning | No |
| RGPE-CBO | one GP per source task | target GP conditioning and ranking-weight updates | No |
| Hierarchical Transfer GP-CBO | source hierarchy/prior mean | target discrepancy posterior | No |
| Multi-task GP-CBO | source task covariance | joint source-target posterior | No |
| FSBO-CBO | shared deep-kernel feature extractor | feature extractor, kernel, likelihood, and target GP update | Yes |
| HyperBO-CBO | pretrained GP prior/hyperparameters | posterior conditioning | No |
| MetaBO-CBO | acquisition policy | frozen policy acting on an updated target GP state | No |
| MALIBO-CBO | utility representation | target Bayesian logistic utility head | No |

Thus, “source training then target fine-tuning” is the umbrella experimental
form.  The paper must name the actual statistical operation instead of calling
all eight mechanisms fine-tuning.

## 3. Heteroscedastic CBO interface

All transfer methods receive three observable source responses: objective
mean, centered constraint mean, and log replicate variance.  During target
optimization they update objective and constraint posteriors plus a
log-variance posterior using prequential squared residuals.  Candidate safety
uses

`mu_g + sqrt(beta_g) s_g + z_alpha sqrt(v_C_plus) <= 0`.

No target `true_sigma` enters selection.  Analytic target truth is called only
after the recommendation to compute regret and false-feasibility metrics.
MetaBO and MALIBO do not define a native heteroscedastic safe model, so their
objective utility is combined with the common disclosed HyperBO-style
constraint/risk extension.

## 4. Required tables

1. **Transfer value:** identical 384-call source archive plus `N_target=20`;
   each method may use its disclosed source-informed proposal mechanism.
2. **Optimizer value:** identical source archive and byte-identical scrambled
   Sobol target `n0`, generated independently of every source archive and
   target-specific hook.
3. **Cold start:** no source archive and `N_target=20`.
4. **Total cost:** from-scratch SOTA receives `N_target=404`.
5. **Amortization:** source cost divided by an explicitly stated number of
   future target tasks.

Every row reports source domains, target domain, dimension, source profiles,
replications, source calls, `n0`, target calls, total calls, archive
fingerprint, adaptation kind, online parameters changed, implementation
fidelity, failures, and post-run-only truth usage.

## 5. Implementation fidelity

`implementation=official` never silently falls back.  All eight rows now have
an explicit adapter to pinned upstream code: F-PACOH, FSBO, MALIBO, HyperBO,
TransferGPBO's RGPE/SHGP/MTGP, and MetaBO's NeuralAF.  Compatibility shims and
the parameters changed online are emitted in each result.

The constrained heteroscedastic wrappers are necessarily common extensions
where an upstream method is unconstrained.  In particular, MetaBO uses the
official NeuralAF architecture and clipped PPO objective, but trains it by
finite-archive replay instead of the upstream unlimited source-function
generator.  Rewards use only disclosed source rows; the target policy is
frozen and only its target GP state is conditioned online.  This row is
reported as `official_neuralaf_ppo_fixed_archive_extension`, not as an exact
bit-for-bit reproduction of the original generative benchmark.  A failed
official runtime is recorded as `failed_official_runtime`, never replaced by
a numerical paper-core result.

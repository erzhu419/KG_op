# Round 7 OR Review Panel Revision Report (2026-11-21)

## 1. Overview & Governance

This report documents the comprehensive Round 7 revision addressing the simulated *Operations Research* (Simulation Area) review panel (`or_simulated_review_panel.md`: Associate Editor + Referees 1–3).

- **Manuscript Repositioning**:
  - **New Title**: *Certification-Aligned Search and Transferable Spectral Priors in Chance-Constrained Simulation Optimization*
  - **Running Title**: *Certification-Aligned Search and Transferable Spectral Priors*
  - **Page Budget & Typography**: 30 body pages excluding References (`manuscript.pdf` = 34 pages total; `electronic_companion.pdf` = 16 pages), 190-word Abstract ($\le 200$ words), 0 undefined/multiply-defined references, 0 Overfull `\hbox`es across all 50 pages, 56 verified BibTeX entries, 24 sealed-ledger tables, and 3 vector figures.
- **Sealed Code & Cohort Preservation**:
  - All `SC-OLH-KG/v2_or/*.py` source files (`fe8df9098424907147128161afda75d752cc0a8ab47e4357251f05da6206462b`) and all previously completed scientific cohorts (`1,376` active closed-loop runs, `confirm_fresh_72`, `eckg_multiscramble_cei`, and `384` + `432` + `9,796` predecessor runs) remain 100% untouched.
  - All new Round 7 experiments were pre-registered in `SC-OLH-KG/manuscript_v2_or/replication/PREREGISTRATION_REV7_OR_PANEL_20261121.json` and executed on the `node001–node006` CPU cluster (`vram=0`, `OMP_NUM_THREADS=1`) with strict Stage-1 (`truth_used = False`) vs. Stage-2 (`truth_used = True`) barriers.

---

## 2. Point-by-Point Resolution of AE & Referee Critiques

### Requirement 1 (AE + Referees 1, 2, 3): Re-center Positioning & Calibrate Functional Meta-Prior Claims
- **Title, Abstract, Introduction, Section 7.3, and Conclusions** now lead with certification-aligned sequential search and curtailed exact-binomial clean-up, while placing transferable spectral priors on an exact, calibrated empirical footing.
- **Decision-Aligned Calibration & Meta-BO Baselines on `confirm_fresh_72` (`Table 8B`)**:
  - Added direct decision-aligned calibration metrics ($K_0=112$ binomial gate-power RMSE, 5% feasibility Brier score and AUROC, and true Slot-0 clean-up gain of the top predicted candidate) alongside `ABLR-on-Psi` (Perrone et al. 2018), `MTGP-ICM-on-Psi` (Swersky et al. 2013), and `RGPE-on-Psi` (Feurer et al. 2018).
  - `Online-FullSURE` lowers $K_0=112$ gate-power RMSE relative to `ColdPsi-GP` (`0.237` vs. `0.244`, `46W/26L`, raw $p=0.010$, Holm $p=0.039$) and `Online-SameBasisWeak` (`0.250`, Holm $p=0.001$), while achieving higher feasibility AUROC (`0.900`) than `ABLR-on-Psi` (`0.829`), `MTGP-ICM-on-Psi` (`0.838`), and `RGPE-on-Psi` (`0.849`).
- **Four-Pillar Leave-One-Out Ablations & $M_{\mathrm{src}} \in \{3, 6, 12\}$ Scaling (`Table 8B`, `Table EC.2B`)**:
  - Leave-one-pillar-out ablations isolate graph low-pass smoothing (`Pillar-NoLowPass` worsens gate RMSE by `+0.0052`, `44W/28L`, raw $p=0.025$) as the primary offline archive regularizer, whereas occupancy debiasing (`Pillar-NoDebias`, `+0.0002`, $p=0.925$), Gram whitening (`Pillar-NoOrtho`, `+0.0015`), and pruning (`Pillar-NoSparse`, `+0.0018`) act mainly as numerical conditioning safeguards.
  - Across $M_{\mathrm{src}} = 3 \to 6 \to 12$, pure-kernel `Diag - Full` shifts monotonically from `-0.064` ($p=0.132$) $\to$ `-0.013` ($p=0.391$) $\to$ `+0.013` ($p=0.096$) on constraints and `-0.016` ($p=0.029$) $\to$ `+0.001` $\to$ `+0.006` ($p=0.087$) on objectives, confirming that rank deficiency at $M_{\mathrm{src}}=6 < p=9$ explains why six-task kernel gains arise from learned per-mode variances rather than off-diagonal covariances.
  - Across shift scales $\{1.0, 1.3, 1.6\}$ (`Table EC.2B`), `Online-FullSURE` beats `ColdPsi-GP` on gate-power RMSE (`+0.0069` at $1.0\times$, $p=0.002$; `+0.0066` at $1.3\times$, $p=0.010$; `+0.0049` at $1.6\times$, $p=0.051$).

### Requirement 2 (AE + Referee 2): External Frontier Baselines under Identical Three-Slot Certification & Optimality Gap
- **Four External Frontier Baselines on All 86 Active Tasks (`Table 4A`, `Figure 1a`)**:
  - Evaluated `SCBO-3Slot` (Eriksson & Poloczek 2021), `RGPE-on-Psi-3Slot` (Feurer et al. 2018), `WarmStart-MTGP-KG-3Slot` (Poloczek, Wang & Frazier 2017), and joint-draw `GP-FTA-LatentNEI-3Slot` (Letham et al. 2019, $M_{\mathrm{NEI}}=32$ joint posterior draws) under the identical $(112, 128, 140)$ three-slot clean-up contract.
  - Across the pooled 72-task DES panel (4-way Holm-adjusted, anchor-SD scaling), **GP-FTA-ECKG** outperforms all four frontier baselines:
    - vs. `SCBO-3Slot`: `+1.2788` SD (`[+1.0879, +1.4697]`, `69W/1L/2T`, Holm $p = 1.9\times 10^{-20}$);
    - vs. `RGPE-on-Psi-3Slot`: `+0.2576` SD (`[+0.1311, +0.3841]`, `48W/22L/2T`, Holm $p = 0.0004$);
    - vs. `WarmStart-MTGP-KG-3Slot`: `+0.1782` SD (`[+0.0569, +0.2996]`, `55W/15L/2T`, Holm $p = 0.009$);
    - vs. `GP-FTA-LatentNEI-3Slot`: `+0.1280` SD (`[+0.0066, +0.2493]`, `36W/34L/2T`, Holm $p = 0.039$; `+0.0979` SD, $p=0.058$ under EC-KG SD scaling).
- **24-Scramble Frozen-State `LatentNEI` Audit (`Table 9`, `Figure 1b`)**:
  - Evaluated 64-joint-draw `LatentNEI` on the exact 24 production states and 24 `256 x 1,024` Sobol scrambles: **EC-KG** posts `11W/9T/4L`, `LatentNEI` makes `0/24` revisits (zero improvement assigned to already-evaluated policies under noise-free latent draws), and in late queueing (`step 59`) **EC-KG** achieves mean gain `3.156e-3` vs. `0.861e-3` for `LatentNEI` and `0.387e-3` for `PlugIn-cEI`.
- **High-Budget Feasible-Oracle Gap Closure (`Ref-Oracle`, `Table 4A`)**:
  - Computed an offline 400-candidate + local Nelder-Mead feasible benchmark on all 86 tasks: **GP-FTA-ECKG** closes `84.8%` of the achievable feasible cost gap in Inventory, `107.6%` in Queueing, `55.3%` on OPSD-336, and `61.7%` in Traffic-21 (vs. `-5.0%`, `18.6%`, `21.8%`, `15.1%` for `SCBO-3Slot`).

### Requirement 3 (AE + Referees 1, 2): Evaluated-Only Eligibility Relaxation, Parameter-Free Shortlist, Gate Stringency & Sequential Curtailment
- **Evaluated-Only Eligibility Relaxation (`GP-FTA-AllowUnvisited`, `Table 6A`)**:
  - Allowing unvisited policies outside $\mathcal{D}_N$ to enter the terminal shortlist triggers unvisited Slot-0 nominations on `22/48` (Inventory), `14/48` (Queueing), `12/28` (OPSD-336), and `21/48` (Traffic-21) runs, yet slightly *worsens* mean loss (`+0.0670` SD favoring **GP-FTA-ECKG** on pooled DES, `41W/29L/2T`, Wilcoxon $p=0.018$, paired-$t$ $p=0.146$) while spending `4.8`–`18.3` more verification calls.
- **Parameter-Free Combinatorial Shortlist Nominator (`GP-FTA-DirectMaxU`, `Table 6A`)**:
  - Replacing the structured three-role shortlist with parameter-free combinatorial maximization of three-slot expected saving yields virtually indistinguishable loss (`+0.0309` SD, `33W/37L/2T`, raw $p=0.424$), proving that performance does not depend on handcrafted role thresholds.
- **Exact Binomial Gate Stringency $k^\star \in \{0, 1, 2, 3\}$ & Sample-Path-Exact Sequential Curtailment (`Theorem 4`, `Table 6B`)**:
  - Proved in Lean 4.31.0 (`curtailed_binomial_gate_iff_full_sample`) and Theorem 4(2) that early stopping at the first replication where $V_{i,m} \ge k_i^\star + 1$ or $V_{i,m} + (K_i - m) \le k_i^\star$ preserves the exact full-sample gate indicator on every sample path.
  - Across all 172 active run cells, $k^\star=1$ sequential curtailment produces identical deployed policies and losses while saving `36.2` (`14.2%`) verification calls in Inventory, `61.2` (`23.9%`) on OPSD-336, and `90.0` (`33.0%`) in Traffic-21 (`0` verification-law exceedances across all $k^\star \in \{0,1,2,3\}$).
  - Also evaluated `GP-FTA-LongSearch-k0` ($N=100$ search steps + curtailed $k^\star=0$ gates), showing lower loss in safe-interior queueing (`2.5490` vs. `2.5649`, Holm $p=0.002$) vs. lower certification and higher risk in tight-boundary traffic (`1.0884` vs. `1.0790`, `33/48` vs. `35/48` certifications).

### Requirement 4 (AE + Referees 2, 3): Public Summary $\psi(x)$ Stress-Testing & External Simulator-Free Benchmark
- **Complete Self-Contained $\psi(x)$ Specifications (`Section 3.3`, `Table 1`, `EC Section EC.2`)**:
  - Added Table 1 in Section 3.3 (complete 12-coordinate inventory recursion) and EC Section EC.2 (complete closed-form equations for queueing, OPSD battery storage, Ingolstadt signal timing, and simulator-free $\psi^{\mathrm{generic}}(x)$).
- **7D Volatility Ablation, Black-Box $\psi^{\mathrm{generic}}$, and 15% Forecast Corruption (`Table 8B`)**:
  - `Psi-DropVol7D` worsens feasibility Brier by `+0.0067` ($p=0.010$); replacing domain rollouts by `Psi-GenericBlackBox` degrades gate RMSE from `0.237` to `0.362` ($p=7.3\times 10^{-10}$), isolating how much structure comes from public nominal dynamics; and 15% multiplicative public-forecast corruption (`ForecastNoise-15%`) shifts gate RMSE by only `+0.0048` ($p=0.245$).
- **External `SimOpt_TandemCntNv` Benchmark under Purely Generic $\psi^{\mathrm{generic}}(x)$ (`Table 4B`)**:
  - Built a 24-task, 4-channel variable-horizon ($d_s \in \{24,32\} \to d_t \in \{48,64\}$) tandem three-station finite-buffer blocking + continuous newsvendor raw-material order benchmark (`240` closed-loop runs) where every method receives *only* simulator-free $\psi^{\mathrm{generic}}(x) \in \mathbb{R}^{12}$: **GP-FTA-ECKG** achieves the lowest mean $\mathcal{L}_{\mathrm{SQ}} = 3.1932 \pm 0.0757$ (`37/48` certified, `0/37` audit exceedances), beating `GP-FTA-FeasibleKG` (`3.2864`, `15W/7L/2T`), `GP-FTA-LatentNEI-3Slot` (`3.2743`, `15W/8L/1T`, Wilcoxon $p=0.045$), `PublicPsi-TargetGP-TS` (`3.2252`), and `SCBO-3Slot` (`3.2564`, `3/48` certified, `504.6` vs. `385.1` calls).

### Requirement 5 (AE + Referee 1): Upgraded Theory & Lean 4.31.0 Proofs (22 Mechanized Theorems)
- **Theorem 2 (`thm:sure_uniform`, `stein_sure_gaussian_dominance_at_rank_ge_five`)**: Proved exact Gaussian Stein risk dominance $\mathbb{E}[L_{\mathcal{I}}(\hat{\lambda})] \le 1 - (m_{\mathrm{id}}-4)\mathbb{E}[\|\tilde{R}\|_2^{-2}] < 1$ for all $\bm{\delta} \in \mathbb{R}^{m_{\mathrm{id}}}$ whenever $m_{\mathrm{id}} \ge 5$ at the exact code constant $c = m_{\mathrm{id}}$, alongside the distribution-free finite-variance cap and explicit caveat on adaptive HVD plug-in weights.
- **Proposition 2 (`prop:consistency_boundary`, `revisit_ec_voi_vanishes_under_lipschitz_innovation`, `screened_eckg_can_miss_cascade_winner`)**: Proved the unscreened vanishing-innovation / no-starvation property on finite candidate pools alongside an explicit constructive counterexample showing how Stage A's marginal single-slot contender cap can exclude a zero-marginal-value backup winner $q_{\mathrm{B}}$ that strictly maximizes full-cascade payoff.
- **Theorem 4 (`thm:vkg_safety`, `curtailed_binomial_gate_iff_full_sample`)**: Proved sample-path equivalence of sequential binomial curtailment alongside explicit distinction between $\mathbb{Q}_t^{\mathrm{ver}}$ and $\mathbb{Q}_t^{\mathrm{aud}}$ on temporal-block archives.
- **Lean 4.31.0 Receipt (`SC-OLH-KG/paper_artifacts/v2_or_eckg_lean_receipt.json`)**: All **22** core declarations verified under Lean 4.31.0 (`zero sorry`, standard classical axioms `[propext, Classical.choice, Quot.sound]` only).

### Requirement 6 (Referee 3): Prose Polish, Hyperparameter Inventory, and Literature Coverage
- Removed main-text SHA hashes, internal script paths, and audit narration; consolidated fixed constants and dedicated sensitivity checks into `Table EC.1` (`table_ec_hyperparams.tex`).
- Expanded `references_v2.bib` to **56** verified entries (adding Wald 1945, Jennison & Turnbull 2000, Stein 1981, James & Stein 1961, Baranchik 1970, Mallows 1973, Efron & Morris 1973, Hong & Nelson 2006, Kim & Nelson 2006, Letham et al. 2019, Gelbart et al. 2014, Gardner et al. 2014, Perrone et al. 2018, Swersky et al. 2013, Rasmussen & Williams 2006, Koenig & Law 1985).

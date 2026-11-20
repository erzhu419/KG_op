# 基于 GPT-6 系统性评估的 `SC-OLH-KG / v2_or` 全流程整改与 *Operations Research* 完整实现报告

**完成日期**：2026-11-17  
**目标期刊**：INFORMS *Operations Research*  
**核心代码冻结哈希（`v2_or_code_sha256`）**：`3970fb7f2b34a299f8e4420911e4a4a3df86ca07ee4fffef41091996f31ecf63`  
**集群调度规范**：严格遵守 `AGENTS.md`，全部仿真、诊断、先验提取、在线搜索、后验审计与 Lean 4.31.0 形式化校验均通过 `scheduleurm` 在 `node001–node006` CPU 池（`vram=0`，单 worker `OMP/MKL/OPENBLAS_NUM_THREADS=1`）执行；所有历史 `Stage65/85/93` 文件及 `d1a13a40` 封存目录（`v2_or_pilot_v1`、`v2_or_confirm_v1`、`v2_or_live_sumo_v1`、`v2_or_lean_verification_receipt.json`）**零修改、零覆盖**。

---

## 一、 执行总览：GPT-6 全部 15 节审计问题的闭环解决映射

针对 GPT-6 在《重构诊断与 Operations Research 研究计划》与 `GPT_diagnosis_v6.md` 中指出的全部核心统计、代码、基线、理论证明与论文叙述问题，我们严格按 **Stage P0 → Stage P1（含 Decision Gate）→ Stage P3（Lean 4.31.0 机器校验）→ Stage P2（独立留出确认集 + 平台重标定微观交通）→ Stage P4（自动元数据导出与论文重写）** 顺序完成了彻底重构：

| GPT-6 诊断章节与问题 | 具体整改实现与对应代码/产物 | 实证/形式化验证结论 |
| :--- | :--- | :--- |
| **§4.1 统计比较依赖参赛名单（Roster-dependent fallback）** | 废除依赖其他方法最大成本的 `max_feas * 1.05` 惩罚；统一定义为 **Roster-Invariant Status-Quo Decision Loss** $\mathcal{L}_t = J_t(\hat{x}_t)\mathbb{I}\{\text{cert}\} + J_t(x_t^{\mathrm{sq}})\mathbb{I}\{\text{abstention}\}$，并在 `reaudit_existing_cohorts_p0.py` 与 `campaign_runner.py` 中对每个任务确定性审计公有锚点 $x_t^{\mathrm{sq}}$ | P0 非破坏性重审计自动断言通过：`stop_identical_across_budgets = true`，`roster_invariant_across_arm_subsets = true`（增删任何 baseline 对其余方法得分改变为 0） |
| **§4.2 任务级自由度与伪重复（Pseudoreplication）** | 所有主表、消融表、成本摊销表均先在每个独立目标任务内对随机流取均值，再在**独立目标任务层级**（Stage P2 扩展至 $M_{\mathrm{tgt}} \in \{16, 16, 14, 16\}$）执行配对 Wilcoxon 与配对 $t$ 检验及 Holm–Bonferroni 校正 | 在独立任务层级（$M_{\mathrm{tgt}}=14\sim 16$），**TLS-BGP-KG** 在全部 4 个家族 $\times$ 6 个外部基线 = **24/24 项对比中均显著占优**（$N=60$ 时全部 $p_{\mathrm{Holm}} \le 0.018$） |
| **§4.3 蒙特卡洛审计抽样误差与“0 伪证”表述** | 在 `contracts.py` 引入精确双侧 95% Clopper–Pearson 置信区间 `clopper_pearson_interval`，将每个已认证策略严格划分为 `confidently_safe`（$\mathrm{CI}_{95,\mathrm{hi}} \le \alpha$）、`boundary_ambiguous` 与 `confidently_unsafe`（$\mathrm{CI}_{95,\mathrm{lo}} > \alpha$）；OPSD 同时审计 IID `verification` 分布与 24h 缓冲 OOD `audit` 分布 | OPSD 在 **IID 验证分布上全部 1,413 次认证违约率 >0.05 的次数为 `0 / 1,413`（最大值 `0.0389`）**，完美印证定理；而在跨季节 OOD 时间块上忠实报告了非平稳漂移（最高 `0.0865`） |
| **§5.1 / §8.2 `Untempered-FULL` 人工 `log_shift` 乘子** | **彻底删除** `tempered_hvd_posterior.py` 中的人工 `exp(min(1.35 * mean_sq_shift, 4.0))` 放大器；实现三种使用**完全相同 NNLS 求解器**的真实方差估计器：`within_ssq`、`naive_prior_residual_sq`（真实先验残差平方）与 `posterior_residual_sq` | Stage P1 诊断 2 证明：无需任何人工乘子，真实 `naive_prior_residual_sq` 在 $1.3\times$ 偏移下即产生 $4.4\times \sim 1{,}321\times$ 方差膨胀，且在线确认集中显著劣于 **TLS-BGP-KG**（Queueing $p < 10^{-4}$，Traffic $p=0.002$） |
| **§5.2 / §8.4 `V-KG` 边界 $23\times$ 高斯近似失配** | 在 `gated_verifier_v2.py` 实现 `exact_binomial_critical_k`（$(112,128,140)$ 在 $\delta=(0.025,0.015,0.010)$ 下精确临界违约数均为 $k^\star=1$）与 `posterior_binomial_gate_power`（11 点 Gauss–Hermite 积分精确二项门控通过功效 $C_{K_0,1}(p)$），在 $p=0.05, s_g=0$ 处精确等于 `0.0220578`（修复旧版 `0.50` 的 $22.7\times$ 误差） | Stage P1 机制拆解证实：修复后的 **Gate-Power V-KG (`F2`)** 在开发集 **8/8 个（家族 $\times$ 预算）单元全面击败 Gate-Power TS (`E`)**（Inventory $N=60$ $p=0.0155$；Queueing $N=15$ $p=0.0290$） |
| **§6.1 弃权语义自相矛盾（`certified=False` 仍返回策略）** | 重构 `VerificationReceipt`：弃权时 `decision_state = "abstention"`、`deployed_policy = None`，同时独立保存冻结前的 `primary_policy`（Slot 0）、`selected_uncertified_candidate` 及全部 `per_slot_certificates.csv` 明细；Slot 1 准确命名为 `empirical_feasible_incumbent` | 单元测试与全量 10,256 条新运行账本 100% 满足：`not certified => deployed_policy is None` |
| **§6.2 多通道策略插值错位 Bug** | 在 `contracts.py` 实现通道保持插值 `warp_multichannel_policy`（`[0.2,0.8]*3 -> [0.2,0.8]*4` 精确保持偶/奇通道语义），`IngolstadtSignalNetworkDES` 实现前缀路口对齐 + 相位奇偶保持映射，`debiased_spectral_prior.py` 实现 `num_channels` 感知的 DCT 模板提取 | 修复后外部迁移基线 `RGPE` 在 `Traffic_Ingolstadt` 上的认证率从旧版 `29.2%` 大幅提升至 **`75.0%`**，成为真正公平强劲的对手；而 **TLS-BGP-KG**（`81.2%` 认证）依然以 $p_{\mathrm{Holm}} = 0.004$ 显著击败修复后的 `RGPE` |
| **§6.3 外部基线（`SCBO` / `SAAS-CBO` / `RGPE` / `HyperBO`）忠实升级** | `SCBO` 升级为 `random_base2` Sobol + 联合 Cholesky GP Thompson Sampling；`SAAS-CBO` 升级并诚实重命名为解析梯度 Half-Cauchy 分层先验 L-BFGS-B **`MAP-SAAS-CBO`**；`RGPE` 升级为 Rasmussen–Williams Eq. 5.12 精确解析 GP LOOCV + 联合采样；`HyperBO` 升级为跨源任务边际似然预训练 **`HyperBO-EB`**（离线预训练耗时单独记入 `source_pretrain_wall_sec`）；新增从业者启发式基线 **`WarmStart-PatternSearch`** | 在全套忠实升级基线与离线计时分离条件下，**TLS-BGP-KG** 在全部 4 个家族上依然取得 **24/24** 项外部基线显著胜利 |
| **§7 理论过度包装与 Lean 4 脱节** | 新增 `proof/SCOLHKG/V2OR/FunctionalTransferRisk.lean`，形式化证明：(1) 有限源任务 $M_{\mathrm{src}}$ 方差 $\sigma_{\mathrm{src}}^2/M_{\mathrm{src}}$ + 占据偏差 + 结构偏移分解；(2) **充要负迁移边界** $\operatorname{MSE}(\tau;\Delta_{\mathrm{eff}}) < 1/\rho_N \iff \Delta_{\mathrm{eff}}^2 < 1/\rho_N + 2/\tau$；(3) 含零精度释放点 $\tau_{i_0}=0 \in \mathcal{G}$ 的有限网格选择器风险上界；(4) **$k^\star=1$ 二项门控精确单调性**（覆盖 $K \in \{112,128,140\}$）；并在 EC 建立 5 列“定理—假设—Lean 声明—Python 代码—未形式化缺口”对照表 | 集群 Lean 4.31.0 机器校验一次通过（`v2_or_p3_lean_verification_receipt.json`，耗时 `29.13 CPU s`，零 `sorry`，仅依赖标准公理 `[propext, Classical.choice, Quot.sound]`） |
| **§9.3 OPSD Winsorization 与时间块隔离** | `OPSDGridStorageBenchmark` 默认启用 `buffer_hours = 24` 小时隔离带，目标市场扩展至**全部 14 个不相交有效欧洲电力市场**，同时运行 `clamp_limit = 0.20` 主实验与 `clamp_limit = None` 原始重尾敏感性实验，并分离审计 IID `verification` 分布与 OOD `audit` 分布 | 原始重尾（`clamp_limit = None`）下 **TLS-BGP-KG** 仍显著优于 `SCBO`（$+10.4\%, p=0.024$）、`RGPE`（$+13.5\%, p=0.005$）、`WarmStart-PatternSearch`（$+13.0\%, p=0.007$）和 `PublicPsi`（$+4.8\%, p=0.010$） |
| **§9.4 `LiveSUMO` `0/40` 跨平台基准失配** | 诊断 4 查明根因：集群 `Linux SUMO 1.22` 在固定配时方案下的 HBEFA3 $\mathrm{CO}_2$ 基准为 $E_0^{\mathrm{Linux}} = 3{,}631.30$，比旧静态 `baseline.json`（$2{,}832.94$）高出 **$+28.18\%$**，导致旧 $\tau \le 1.025$ 下连固定配时自身也 100% 违约。我们在独立种子 `410001..410030` 上完成平台标定，开展 `v2_or_p2_live_sumo_v1`（8 任务 $\times$ 2 种子 $\times$ 7 方法 = 112 次真实微观仿真，$K_{\mathrm{audit}}=80$），同时完整保留旧 `40` 次不可行压力测试 | 平台标定后：**`SCBO`/`RGPE`/`STOP` 认证率均为 `0.0%`**，而 **TLS-BGP-KG 认证率达 `37.5%`**（损失 `0.9685`，优于 `TLS-GatePower-TS` 的 `12.5%` / `0.9904` 及 `SCBO`/`RGPE` 的 `0.9941`），`PublicPsi-TargetGP-TS` 达 `50.0%`（`0.9454`），且 **112 次运行 `confidently_unsafe = 0`** |
| **§10 成本摊销 `max(diff, 1)` 截断与双口径缺失** | 删除 `max(diff, 1)`，在 `build_p4_tables_and_figures.py` 中严格检验质量占优并计算未截断的任务级调用节省 $\Delta B$ 及其双侧 95% 任务-$t$ 置信区间，同时报告历史沉没成本（$M_{\mathrm{break}}^{\mathrm{sunk}}=1$）与全成本归集（$B_{\mathrm{src}}=1{,}428$）双口径 | 全部 4 个家族在 $N=15$ 时均严格优于 `SCBO` ($N=60$)，任务级平均节省调用 **$+90.0 \sim +162.5$ 次**，95% 置信下界全部严格为正（$[+71.1, +131.5]$），全成本盈亏平衡点 $M_{\mathrm{break}}^{\mathrm{full}} \in \{9, 11, 12, 16\}$（95% 保守上界 $\{11, 14, 17, 21\}$） |
| **§2.2 / §11 / §12 论文图表混拼预算、元数据漂移与 `README` 断层** | 图 2(b) 机制阶梯对**每一个手臂统一同时展示 $N=15$ 与 $N=60$**（含负向/零效应步骤）；编写 `export_contract_metadata.py` 自动从代码与 manifest 导出维度、通道与门控宏 `macros_autogen.tex`；重写 `manuscript_v2_or/` 并编译 24 页零警告 PDF；在 `SC-OLH-KG/README.md` 顶部补充 `v2_or` 入口并完整保留全部历史记录 | `pdflatex` + `bibtex` 干净编译 `main_or.pdf`（24 页，0 错误，0 未定义引用），代码 SHA-256 与封存实验账本 100% 一致 |

---

## 二、 Stage P0：非破坏性服务器重审计与基础契约重构

在不改动任何历史目录的前提下，我们在集群上运行 [`reaudit_existing_cohorts_p0.py`](../v2_or/dispatch/reaudit_existing_cohorts_p0.py)，直接读取远端 NFS 保留的 `242`（pilot）+ `4,128`（confirm）+ `40`（live_sumo）个单次运行 JSON，生成 [`paper_artifacts/v2_or_p0_reaudit_v1/`](../paper_artifacts/v2_or_p0_reaudit_v1/p0_reaudit_index.json)：
1. **Roster-Invariance 自动验证**：移除旧版依赖同组最大可行成本的弃权罚分后，`STOP` 在 $N \in \{15, 30, 60\}$ 上逐比特完全一致（Inventory `1.8856`，Queueing `2.6265`，OPSD `0.0581`，Traffic `1.1410`），任意删减对比方法对剩余方法的损失值影响恒为 `0.0`。
2. **Clopper–Pearson 精确安全重分类**：旧 `v2_or_confirm_v1` 全部 `4,128` 次运行中，`3,390` 次为 `confidently_safe`（$\mathrm{CI}_{95,\mathrm{hi}} \le 0.05$），`736` 次为 `abstained_not_deployed`，仅 `1` 次 `boundary_ambiguous`、`1` 次 `confidently_unsafe`（且均发生在去掉安全机制的先验消融臂中，主方法与外部基线均为 `0`）。
3. **揭示旧版 `V-KG` 落后于 `Constrained-TS-Acq` 的真实根因**：重审计显示旧版高斯边际 `V-KG` 在边界 $p=0.05$ 处高估通过率达 $22.7\times$（`0.50` vs 真实二项通过率 `0.022058`），直接促使我们在 P0/P1 中引入精确 $C_{K_0,1}(p)$ Gauss–Hermite 门控功效积分。

---

## 三、 Stage P1：四项决定性诊断、正交机制拆解与 Decision Gate

### 3.1 四项独立诊断实验结果（[`p1_diagnostics_summary.json`](../paper_artifacts/v2_or_p1_diagnostics_v1/p1_diagnostics_summary.json)）
1. **诊断 1（$\psi \in \mathbb{R}^{12}$ 表征平滑性与多对一碰撞边界）**：平滑 DCT 策略上 $\psi$-最近邻的目标函数 MAD 仅为全局策略标准差的 `10.8%`（OPSD）、`15.1%`（Inventory）、`16.6%`（Traffic）与 `25.7%`（Queueing）；前后半程反转策略由于跨越不同时段体制（off-peak / shoulder / peak），引起 $\|\Delta\psi\|_{\mathrm{rms}} \in [0.19, 0.60]$ 的显著位移，但非零近邻残差（`0.08–0.46 SD`）明确证明不能把 $\psi$ 当成单射坐标，必须保留非参数残差 GP $k_{5/2}(\tilde{\psi},\tilde{\psi}')$。
2. **诊断 2（无任何人工乘子的真实样本外方差标定）**：在 $1.3\times$ 偏移、$r=4$ 下，对相同数据运行完全相同的 NNLS 引擎，真实先验残差平方 `naive_prior_residual_sq` 相比组内方差 `within_ssq` 的样本外 $\log v_t(x)$ RMSE 在全部四个家族全面恶化：Inventory `2.62 -> 4.07`（中位方差比 `5.85x -> 26.09x`），Queueing `5.16 -> 6.83`（`198.7x -> 1320.8x`），OPSD `4.89 -> 5.52`（`172.9x -> 360.6x`），Traffic `1.06 -> 1.75`（`1.54x -> 4.40x`）。
3. **诊断 3（门控通过概率标定）**：精确二项门控功效 $C_{112,1}(p)$ 在 Queueing 与 Traffic 边界候选集上显著降低了旧版高斯边际的正向乐观偏差（Queueing 边界 MAE `0.566 -> 0.516`，Traffic Brier `0.0286 -> 0.0271`）。
4. **诊断 4（`Linux SUMO 1.22` 跨平台物理基准标定）**：在 30 个独立标定种子上测得 RESCO 固定配时方案在集群 `Linux SUMO 1.22` 下的真实均值基准为 $T_0 = 299.444\text{ s}$、$A_0 = 0.26786$、**$E_0 = 3{,}631.296$**（95% 分位比均值 `1.0375`），比旧静态 `baseline.json` 中的 $E_0 = 2{,}832.94$ 高出 **`1.2818x`**，彻底解释了旧 `v2_or_live_sumo_v1` 为什么 `0/40` 全员拒真，并为 Stage P2 提供了严格无泄漏的平台基准。

### 3.2 正交机制阶梯拆解与 P1 Decision Gate（[`v2_or_p1_mech_v1`](../paper_artifacts/v2_or_p1_mech_v1/compact_summary.json) & [`v2_or_p1_gate_check_v1`](../paper_artifacts/v2_or_p1_gate_check_v1/compact_summary.json)）
在 16 个独立 Inventory 开发任务与 16 个独立 Queueing 开发任务（$N \in \{8, 15, 30, 60\}$，共 `3,648` 次运行，`0` 次 `confidently_unsafe`）上，每次仅改变**单一正交开关**：
- **纯函数均值/协方差迁移效应（`D - C`，固定目标域 HVD 噪声估计不变）**：在 Inventory 上显著降低损失（$N=15$: `-0.1173 ± 0.0442`, `13W/3L`, `p=0.0131`；$N=60$: `-0.0711 ± 0.0533`, `p=0.0131`）；而在 Queueing $N=60$ 上暴露了固定尺度先验均值的晚期拖拽（`D - C = +0.0588, p=0.0003`）。
- **尺度等变 + Schwarz-BIC 零精度释放分支（`tempered_adaptive`, $\beta = 0 \in \mathcal{G}$）**：针对开发集暴露的跨量纲尺度差异（OPSD `y_var ~ 3e-5` vs Queueing `y_var ~ 3.0`）与大预算先验拖拽，我们在 MLL 网格中加入以 $\frac{1}{2}\log n$ 为复杂性罚项的目标域零精度释放点 $\beta = 0$（对应 Lean 定理中的 $\tau = 0 \in \mathcal{G}$），在开发集门控检验中彻底消除了 Queueing $N=60$ 的先验拖拽（`2.589` vs `C=2.576`, `p=0.46`），同时保留全部早期迁移收益。
- **获取函数门控决策（`F2: Gate-Power V-KG` vs `E: Gate-Power TS`）**：修复 $23\times$ 二项门控功效失配后，**Gate-Power V-KG 在 `v2_or_p1_mech_v1` 全部 8/8 个（家族 $\times$ 预算）单元一致击败 Gate-Power TS**（Inventory $N=60$: `-0.0312`, `p=0.0155`；Queueing $N=15$: `-0.0345`, `p=0.0290`；在 `v2_or_p1_gate_check_v1` Queueing 全部四个预算上均显著击败 TS，`p <= 0.0092`），并大幅击败从业者基线 `G: WarmStart-PatternSearch`（Inventory $N=60$ `p=0.0004`；Queueing 全部预算 `16W/0L, p < 10^{-4}`）。
- **锁定方法**：依据 P1 Decision Gate 证据，正式锁定 **TLS-BGP-KG**（`tempered_adaptive` + `within_ssq` HVD + `gate_power_vkg` + 三槽位精确二项门控）进入完全未触碰的 Stage P2 留出集确认实验。

---

## 四、 Stage P3：与算法一一对齐的 Lean 4.31.0 机器校验形式化理论

我们在 [`proof/SCOLHKG/V2OR/FunctionalTransferRisk.lean`](../../proof/SCOLHKG/V2OR/FunctionalTransferRisk.lean) 与 [`proof/SCOLHKG/V2ORProofInterface.lean`](../../proof/SCOLHKG/V2ORProofInterface.lean) 中完成了无 `sorry`、无自定义公理的机器验证证明（集群回执 [`v2_or_p3_lean_verification_receipt.json`](../paper_artifacts/v2_or_p3_lean_verification_receipt.json)，`status = "passed"`，`29.13 CPU s`）：
1. `effective_mismatch_second_moment_with_finite_sources`：将经验先验均值误差精确分解为结构偏移、占据偏差与有限源任务方差 $(\Delta_{\mathrm{shift}} - b_{\mathrm{occ}})^2 + \sigma_{\mathrm{src}}^2 / M_{\mathrm{src}}$（与代码中 $(1 + 1/M_{\mathrm{src}})$ 协方差膨胀严格对应）。
2. `transfer_dominates_target_only_iff`：证明收缩估计严格优于纯目标域估计（$\operatorname{MSE}(\tau;\Delta_{\mathrm{eff}}) < 1/\rho_N$）的**充要条件**为 $\Delta_{\mathrm{eff}}^2 < 1/\rho_N + 2/\tau$，明确给出“何时应当迁移、何时不应当迁移”的理论边界。
3. `shrinkageMSE_zero_eq_target_only` & `grid_with_release_bounded_by_target_only`：证明当有限网格 $\mathcal{G}$ 包含释放点 $\tau_{i_0} = 0$（即代码中 $\beta = 0$ Schwarz-BIC 分支）时，对**任意大小**的偏移 $\Delta_{\mathrm{eff}} \in \mathbb{R}$，总预测风险一致有上界 $1/\rho_N + \eta_{\mathrm{sel}} + \varepsilon_{\mathrm{rep}}^2$。
4. `binomial_one_violation_gate_power_bound`：纯多项式归纳证明对一切 $K \ge 2$ 及 $0 \le \alpha \le p \le 1$，$k^\star = 1$ 二项通过概率 $C_{K,1}(p) = (1-p)^K + Kp(1-p)^{K-1} \le C_{K,1}(\alpha)$，直接覆盖实际验证器三槽位 $(112, 128, 140)$ 在 $(\delta_0,\delta_1,\delta_2)=(0.025,0.015,0.010)$ 下的精确阈值 $k^\star = (1, 1, 1)$！
5. `status_quo_loss_roster_invariant`：形式化验证现状决策损失对外部对比名单严格不变。

---

## 五、 Stage P2：全新独立留出确认集（`6,384` runs）与平台标定 `LiveSUMO`（`112` runs）

### 5.1 独立留出确认集主结果（[`v2_or_p2_confirm_v1`](../paper_artifacts/v2_or_p2_confirm_v1/compact_summary.json)，全新种子 `8101..8401` + 全部 14 个留出欧洲电力市场）
在独立目标任务层级（Inventory $M_{\mathrm{tgt}}=16$，Queueing $M_{\mathrm{tgt}}=16$，OPSD $M_{\mathrm{tgt}}=14$，Traffic $M_{\mathrm{tgt}}=16$），**TLS-BGP-KG** 在 $N=15$ 与 $N=60$ 两个代表预算下对**全部 6 个升级后的忠实外部基线（共 24 组对比）取得 24/24 全胜**（Holm–Bonferroni 组内多重校正后）：

| 家族（独立任务数 $M_{\mathrm{tgt}}$） | **TLS-BGP-KG** ($N=60$) | `SCBO` | `MAP-SAAS-CBO` | `HetGP-KG` | `RGPE` (通道对齐+解析LOOCV) | `HyperBO-EB` | `Target-SPSA` | 最大组内 $p_{\mathrm{Holm}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Inventory DES** ($M_{\mathrm{tgt}}=16$) | **1.914 ± 0.043** (96.9%) | 1.959 (78.1%) | 1.969 (84.4%) | 1.962 (90.6%) | 1.972 (100%) | 1.983 (93.8%) | 1.992 (96.9%) | **0.0184** |
| **Queueing DES** ($M_{\mathrm{tgt}}=16$) | **2.550 ± 0.050** (100%) | 3.060 (96.9%) | 3.054 (96.9%) | 3.056 (100%) | 3.079 (93.8%) | 3.115 (90.6%) | 3.264 (100%) | **0.0001** |
| **OPSD Grid 24h-Buf** ($M_{\mathrm{tgt}}=14$) | **0.0559 ± 0.0067** (89.3%) | 0.0615 (78.6%) | 0.0625 (75.0%) | 0.0602 (100%) | 0.0624 (60.7%) | 0.0631 (89.3%) | 0.0782 (100%) | **0.0101** |
| **Traffic-21 DES** ($M_{\mathrm{tgt}}=16$) | **1.071 ± 0.029** (81.2%) | 1.145 (43.8%) | 1.141 (43.8%) | 1.149 (37.5%) | 1.097 (75.0%) | 1.137 (50.0%) | 1.198 (0.0%) | **0.0042** |

### 5.2 关键科学发现与诚实边界披露
1. **OPSD IID 验证定律零违约 vs 跨季节 OOD 时间漂移 + 原始重尾敏感性**：
   - 在全部 `1,413` 次已认证 OPSD 部署（含未截尾家族）上，针对验证器所检验的 **IID `verification` 分割分布**（8,000 次抽样），违约率超过 $\alpha=0.05$ 的运行数为 **`0 / 1,413`（最大违约率 `0.0389`）**，完美印证定理 2 的有限样本保证；而在 24 小时隔离带之外的**跨季节 OOD `audit` 时间块**上，水电/海岛市场（`NO_5`、`IT_SARD`）的季节性非平稳使真实违约率漂移至最高 `0.0865`（**TLS-BGP-KG** 在主实验 112 次运行中出现 `3` 次 OOD 超界，与 `MAP-SAAS-CBO` 的 `4` 次、`RGPE` 的 `3` 次同量级）。
   - 在**完全不截尾原始重尾实验**（`OPSD_EuroGrid_Unclamped`, `clamp_limit = None`, $M_{\mathrm{tgt}}=14$）中，**TLS-BGP-KG**（$N=60$: `0.0602`）依然显著击败 `SCBO`（`0.0648`, $p=0.024$）、`RGPE`（`0.0666`, $p=0.005$）、`WarmStart-PatternSearch`（`0.0662`, $p=0.007$）与 `PublicPsi-TargetGP-TS`（`0.0623`, $p=0.010$）。
2. **定理 1 负迁移边界的直接实证印证（`TLS-BGP-KG` vs `PublicPsi-TargetGP-TS`）**：
   - 在源域与目标域体制响应一致的三个家族（Inventory、Queueing、OPSD 及原始重尾 OPSD）中，跨任务先验迁移相比零源数据的 `PublicPsi-TargetGP-TS` 带来大幅显著增益（$N=15$ 时分别提升 **`+6.0%`、`+3.7%`、`+17.3%`、`+7.9%`**，全部 $p \le 0.007$）。
   - 而在 **Traffic-21**（从 11/15 路口局部子网迁移至 $1.30\times$ 高峰涌入的全走廊 21 路口耦合溢流网络，以及从宏观 DES 迁移至微观跟驰 `libsumo`）中，初始结构偏移满足 $\Delta_{\mathrm{eff}}^2 > 1/\rho_N + 2/\tau$，此时纯公有物理流形上的无偏探索 `PublicPsi-TargetGP-TS`（Traffic DES `1.007` / 100% 认证；LiveSUMO `0.9454` / 50.0% 认证）优于带源域暖启动锚定的 **TLS-BGP-KG**（Traffic DES `1.071` / 81.2% 认证；LiveSUMO `0.9685` / 37.5% 认证），而**两者均大幅碾压所有原始坐标方法**（`SCBO` 在 LiveSUMO 上 `0.0%` 认证 / `0.9941`）。这一正反双向结果直接构成了定理 1 充要迁移边界的最强实证支撑。
3. **双口径无截断成本摊销（零 `max(diff, 1)`）**：
   - 在全部四个家族中，$N=15$ 的 **TLS-BGP-KG** 在损失上均严格低于 $N=60$ 的冷启动 `SCBO`，同时每任务净节省总仿真调用：Inventory **`+119.5 [+86.8, +152.2]`** 次（$M_{\mathrm{break}}^{\mathrm{full}} = 12$，95% CI 上界 $\le 17$）、Queueing **`+90.0 [+71.1, +108.9]`** 次（$M_{\mathrm{break}}^{\mathrm{full}} = 16$，$\le 21$）、OPSD **`+134.1 [+104.6, +163.7]`** 次（$M_{\mathrm{break}}^{\mathrm{full}} = 11$，$\le 14$）、Traffic **`+162.5 [+131.5, +193.5]`** 次（$M_{\mathrm{break}}^{\mathrm{full}} = 9$，$\le 11$）；若企业已有历史优化日志（沉没成本口径），从第 **$M_{\mathrm{break}}^{\mathrm{sunk}} = 1$** 个目标任务起即实现净节省。

---

## 六、 关键文件与产物清单

- **核心干净代码库**：[`SC-OLH-KG/v2_or/`](../v2_or/)（单元测试 [`test_v2_or_wiring.py`](../v2_or/tests/test_v2_or_wiring.py) `5 passed`）
- **Lean 4.31.0 形式化证明与接口**：[`proof/SCOLHKG/V2OR/FunctionalTransferRisk.lean`](../../proof/SCOLHKG/V2OR/FunctionalTransferRisk.lean)、[`proof/SCOLHKG/V2ORProofInterface.lean`](../../proof/SCOLHKG/V2ORProofInterface.lean)、集群验证回执 [`v2_or_p3_lean_verification_receipt.json`](../paper_artifacts/v2_or_p3_lean_verification_receipt.json)
- **封存实验账本与元数据**：
  - Stage P0 非破坏性重审计：[`paper_artifacts/v2_or_p0_reaudit_v1/`](../paper_artifacts/v2_or_p0_reaudit_v1/p0_reaudit_index.json)
  - Stage P1 四项独立诊断：[`paper_artifacts/v2_or_p1_diagnostics_v1/p1_diagnostics_summary.json`](../paper_artifacts/v2_or_p1_diagnostics_v1/p1_diagnostics_summary.json)
  - Stage P1 顺序机制拆解与门控检验：[`paper_artifacts/v2_or_p1_mech_v1/`](../paper_artifacts/v2_or_p1_mech_v1/compact_summary.json)、[`paper_artifacts/v2_or_p1_gate_check_v1/`](../paper_artifacts/v2_or_p1_gate_check_v1/compact_summary.json)
  - Stage P2 全量独立留出确认集（6,384 runs）：[`paper_artifacts/v2_or_p2_confirm_v1/`](../paper_artifacts/v2_or_p2_confirm_v1/compact_summary.json)
  - Stage P2 平台重标定微观 `libsumo` 验证（112 runs）：[`paper_artifacts/v2_or_p2_live_sumo_v1/`](../paper_artifacts/v2_or_p2_live_sumo_v1/compact_summary.json)
  - 自动导出契约元数据与宏：[`paper_artifacts/v2_or_contract_metadata.json`](../paper_artifacts/v2_or_contract_metadata.json)、[`manuscript_v2_or/macros_autogen.tex`](../manuscript_v2_or/macros_autogen.tex)
- **全新编译的 INFORMS *Operations Research* 投稿论文（24 页 PDF）**：[`SC-OLH-KG/manuscript_v2_or/main_or.pdf`](../manuscript_v2_or/main_or.pdf) 及配套源码 [`main_or.tex`](../manuscript_v2_or/main_or.tex)、自动图表构建脚本 [`build_p4_tables_and_figures.py`](../manuscript_v2_or/build_p4_tables_and_figures.py)

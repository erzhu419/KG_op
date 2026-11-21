# GP-FTA 第七轮 OR 审稿面板全量升级定稿包（Operations Research, Simulation Area）

[匿名主稿，34 页（正文不含参考文献 30 页）](upload/manuscript.pdf) · [电子附录，16 页](upload/electronic_companion.pdf) · [匿名代码与数据包](upload/code_and_data.zip)

在原封不动保留 `SC-OLH-KG/v2_or` 密封代码（`fe8df9098424...`）与既有全部科学批次（1,376 次核心闭环运行、`confirm_fresh_72`、`eckg_multiscramble_cei` 及 9,796 次前序基准运行）的前提下，本轮针对模拟 *Operations Research* 副编辑（AE）与三位审稿人的全部意见完成了深度升级：
1. **题目与理论主线重定焦**：定稿标题为 *Certification-Aligned Search and Transferable Spectral Priors in Chance-Constrained Simulation Optimization*，清晰区分经终端认证闭环验证的收益（资格约束级联前瞻 **EC-KG**、样本路径恒等的序贯二项截尾认证、公共名义摘要与余弦暖启动、策略内异方差 HVD）与跨任务先验在约束形状及门控通过率校准上的作用边界。
2. **外部前沿基线与可行参考神谕差距（`Table 4`、`Figure 1a`）**：在完全相同的 `(112, 128, 140)` 三槽位认证协议下，于全部 86 个主任务上新增对比 `SCBO-3Slot`、`RGPE-on-Psi-3Slot`、`WarmStart-MTGP-KG-3Slot` 与联合抽样 `GP-FTA-LatentNEI-3Slot`（72 任务 DES 合并 Holm $p \le 0.039$），并以离线 400 候选精化可行神谕量化可实现成本缺口闭合率（61.7%–107.6%）；同时构建完全无需仿真器物理方程的通用策略轮廓摘要 $\psi^{\mathrm{generic}}(x)$，在 24 任务四通道串联排队-报童基准（`SimOpt_TandemCntNv`，240 次运行）上验证外部泛化。
3. **合约松弛、无超参提名器与精确序贯二项截尾（`Theorem 4`、`Table 6`）**：形式化证明并实证检验样本路径恒等的序贯截尾认证（零结果改变下节省 14.2%–33.0% 验证仿真开销），并通过 `GP-FTA-AllowUnvisited`、`GP-FTA-DirectMaxU`（$p=0.424$）与 `GP-FTA-LongSearch-k0` 及 $k^\star \in \{0,1,2,3\}$ 系统回答终端资格约束与门控预算设计。
4. **决策对齐后验校准、四支柱消融、源规模 $M_{\mathrm{src}} \in \{3,6,12\}$ 与 22 条 Lean 4.31.0 机器证明（`Theorem 2`、`Proposition 2`、`Table 8B`）**：补齐 `c = m_{\mathrm{id}}` 在 $m_{\mathrm{id}} \ge 5$ 下的精确高斯 Stein 风险占优定理、无筛选大预算渐近性质与筛选反例，完成全部 22 条核心定理的零 `sorry` Lean 4.31.0 验证。

## 复查入口

- [第七轮 OR 审稿面板逐条修订报告](../SC-OLH-KG/docs/gp_fta_referee_rev7_results_20261121.md) · [第六轮修订记录](../SC-OLH-KG/docs/gp_fta_referee_rev6_results_20261121.md) · [第五轮修订记录](../SC-OLH-KG/docs/gp_fta_referee_rev5_results_20261121.md)
- [第七轮实验预注册协议](../SC-OLH-KG/manuscript_v2_or/replication/PREREGISTRATION_REV7_OR_PANEL_20261121.json) · [前沿基线/反事实/截尾门控读数](../SC-OLH-KG/paper_artifacts/v2_or_rev7_or_panel_v1/rev7_closedloop_readout.json) · [校准/四支柱/源规模/SimOpt 读数](../SC-OLH-KG/paper_artifacts/v2_or_rev7_or_panel_v1/rev7_prior_and_simopt_readout.json)
- [全新 72 任务传播预注册协议](../SC-OLH-KG/manuscript_v2_or/replication/PREREGISTRATION_REV5_PROPAGATION_CONFIRM_20261121.json) · [确认集 72 任务传播读数](../SC-OLH-KG/paper_artifacts/v2_or_metaprior_opsd_confirm_v1/direct_propagation_confirm_fresh_72_readout.json) · [Lean 4.31.0 22 定理验证收据](../SC-OLH-KG/paper_artifacts/v2_or_eckg_lean_receipt.json)
- [匿名审稿复现说明](../SC-OLH-KG/manuscript_v2_or/replication/README_REVIEW.md) · [论文源码](../SC-OLH-KG/manuscript_v2_or/) · [第三方数据与模板许可](../SC-OLH-KG/docs/gp_fta_rights_audit_20261120.md)

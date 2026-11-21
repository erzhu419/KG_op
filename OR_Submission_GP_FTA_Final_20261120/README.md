# GP-FTA 第八轮终极审稿核验定稿包（Operations Research, Simulation Area）

[匿名主稿，34 页（正文不含参考文献严格 30 页）](upload/manuscript.pdf) · [电子附录，16 页](upload/electronic_companion.pdf) · [匿名代码与数据包](upload/code_and_data.zip)

在 100% 原封不动保留 `SC-OLH-KG/v2_or` 密封代码（`fe8df9098424...`）与全部既有科学批次（1,376 次核心闭环运行、`confirm_fresh_72`、`eckg_multiscramble_cei`、9,796 次前序基准运行及 `v2_or_rev7_or_panel_v1`）的前提下，本轮逐条彻底清零 `GPT_revise_v7` 的全部数学、基线命名、单因素反事实、无偏校准、流式截尾与精确 $\omega_M$ 规模核验问题：
1. **数学命题与 24 条 Lean 4.31.0 机器验证精确对齐**：移除 `Proposition EC.1` 中不成立的独立乘积与静态均值差包络，替换为保留联合相关性与随机门卫差值的精确耦合残差界（`coupled_cascade_margin_preserves_order`）；将 `Proposition 2(1)` 严格收束为有限候选集上的条件非饥饿性质，将 `Proposition 2(2)` 定位于静态与级联评分差异的说明性算例，并明确全部 24 条零 `sorry` Lean 声明验证的是给定模型恒等式下的代数分解、有限和不等式与样本路径蕴含关系。
2. **文献基线如实命名与真实 Bootstrap 排序加权 `RGPE` 补测（`Table 4`、`Table 8B`、`Figure 1a`）**：将闭环合并源均值加目标残差 GP 准确命名为 `PooledSourceResGP-KG-3Slot`（四项实现对照合并 Holm $p \le 0.039$），将离线参考准确界定为有限 400 候选审计参考（`Ref-400`），明确四通道串联排队-报童模型为作者构造的自足合成基准并在 `EC.2` 给出完整动力学方程；在 `Table 8B` 新增真正 64 次自助法成对排序损失加权的 `RGPE-BootstrapRank-on-Psi`，并与 `Fixed50-Ensemble-on-Psi`、`EB-LinearDict-on-Psi`、`PooledSourceResGP-on-Psi` 同表平衡报告。
3. **冻结终端单因素反事实与实时逐复制流式截尾计时（`Table 6`）**：在 172 个冻结 `GP-FTA-ECKG` 终端状态 $\mathcal{F}_N$ 上单独分离纯资格松弛 `TermOnly-AllowUnvisited`（合并 DES $-0.035$ SD, $p=0.142$）与无阈值三元组全局效用最大化 `TermOnly-UnthreshMaxU`（$+0.004$ SD, $p=0.878$），撤回未设等效区间的等效性措辞；完整报告全部四个领域的截尾调用节省（**3.1%–33.0%**，含 Queueing 的 3.1%），并实测逐复制流式截尾验证器的真实 wall-clock 节省（**6.9%–34.1%**，172/172 决策完全一致）。
4. **超几何无偏门控 U 统计量 $\widehat{C}_U$、受控通用摘要与精确 $\omega_M = p/(M_{\mathrm{src}}+p)$ 期限均衡规模扩展（`Table 8B`、`Table EC.2`）**：引入配对均方误差差值具有严格零有限审计偏差的超几何无偏估计量 $\widehat{C}_U$（`unbiased_target_preserves_mse_diff`），确认 `Online-FullSURE` 相对 `ColdPsi-GP` 显著降低无偏门控 MSE（$+5.58\times 10^{-3}$, 5 对照 Holm $p=0.042$；400 抽样插件 RMSE Holm $p=0.039$）；补测输入真实公共预测且固定领域 HVD 的 `Psi-GenericPhysOnly`（无偏 MSE 退化 $+27.3\times 10^{-3}$, Holm $p=3.2\times 10^{-7}$）；修复纯核对比的精确收缩参数 $\omega_M=p/(M_{\mathrm{src}}+p)$ 并新增期限均衡 $M_{\mathrm{src}}=4$（$2+2$）与分层 $M_{\mathrm{src}}=3$；在 `EC.3` 补充完整的协议与实施差异披露表（`Table EC.2`），并使引言实现零数学符号的纯正 *Operations Research* 风格。

## 复查入口

- [第八轮逐条终极修复报告](../SC-OLH-KG/docs/gp_fta_referee_rev8_results_20261121.md) · [第七轮修订报告](../SC-OLH-KG/docs/gp_fta_referee_rev7_results_20261121.md) · [第六轮修订记录](../SC-OLH-KG/docs/gp_fta_referee_rev6_results_20261121.md)
- [第八轮终极核验协议](../SC-OLH-KG/manuscript_v2_or/replication/PREREGISTRATION_REV8_DEFINITIVE_AUDIT_20261121.json) · [第八轮单因素/流式截尾/无偏校准/精确 $\omega_M$ 读数](../SC-OLH-KG/paper_artifacts/v2_or_rev8_definitive_v1/rev8_definitive_readout.json)
- [第七轮预注册协议](../SC-OLH-KG/manuscript_v2_or/replication/PREREGISTRATION_REV7_OR_PANEL_20261121.json) · [闭环读数](../SC-OLH-KG/paper_artifacts/v2_or_rev7_or_panel_v1/rev7_closedloop_readout.json) · [Lean 4.31.0 24 声明验证收据](../SC-OLH-KG/paper_artifacts/v2_or_eckg_lean_receipt.json)
- [匿名审稿复现说明](../SC-OLH-KG/manuscript_v2_or/replication/README_REVIEW.md) · [论文源码](../SC-OLH-KG/manuscript_v2_or/) · [第三方数据与模板许可](../SC-OLH-KG/docs/gp_fta_rights_audit_20261120.md)

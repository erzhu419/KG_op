# GP-FTA 第五轮复审材料

[匿名主稿，30 页](upload/manuscript.pdf) · [电子附录，11 页](upload/electronic_companion.pdf) · [匿名代码与数据包](upload/code_and_data.zip)

两批闭环主实验在 72 个离散事件仿真任务和 14 个电网市场上共有 1,376 次独立运行（第一批 720 次五臂对照；第二批含 432 次冷启动对照与 224 次电网运行，密封代码指纹均为 `fe8df9098424...`）。本轮在不重跑任何闭环主实验的前提下，新增了预注册的 72 个全新未见任务单步跨策略传播确认实验（`confirm_fresh_72`）与 24 状态 × 24 组独立 Sobol 扰动及 `UB-cEI` 对照审计（`eckg_multiscramble_cei`），并将 5 个紧凑冻结源先验（`prior_default.pkl`）一并打入匿名复现包。

## 复查入口

- [第五轮修订说明、结果与边界](../SC-OLH-KG/docs/gp_fta_referee_rev5_results_20261121.md) · [第四轮修订记录](../SC-OLH-KG/docs/gp_fta_referee_rev4_results_20261120.md)
- [全新 72 任务传播预注册协议](../SC-OLH-KG/manuscript_v2_or/replication/PREREGISTRATION_REV5_PROPAGATION_CONFIRM_20261121.json) · [探索集 86 任务传播读数](../SC-OLH-KG/paper_artifacts/v2_or_metaprior_opsd_confirm_v1/direct_propagation_exploratory_86_readout.json) · [确认集 72 任务传播读数](../SC-OLH-KG/paper_artifacts/v2_or_metaprior_opsd_confirm_v1/direct_propagation_confirm_fresh_72_readout.json)
- [24 状态 24-Scramble 与 UB-cEI 对照读数](../SC-OLH-KG/paper_artifacts/v2_or_eckg_sure_confirm_v1/eckg_multiscramble_cei_readout.json) · [修正后的 24 状态候选全向量](../SC-OLH-KG/paper_artifacts/v2_or_eckg_sure_confirm_v1/eckg_reference_states.json)
- [复现说明](../SC-OLH-KG/manuscript_v2_or/replication/README_REVIEW.md) · [论文源码](../SC-OLH-KG/manuscript_v2_or/) · [第三方数据与模板许可](../SC-OLH-KG/docs/gp_fta_rights_audit_20261120.md)

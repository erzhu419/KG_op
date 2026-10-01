# 2026-09-30 修改结果

本轮串行完成，没有启用 subagent，没有从服务器拉取数据。原始证据和旧投稿包保留；新协议、结果与修正后的 Energy 模型单独记录。

- **机制归因**：首个中心与距离的干预共 2400 条分析记录。原 d=1000 下，source-best + structural 为 75/160，完整方法为 74/160，medoid + structural 为 49/160。主要作用来自 source 选出的首个中心，不能把原差异全归因于增广距离。
- **新确认**：三个新 family、公共库与 source 档案，240 个新 latent tasks、六个方法，共 1440 条记录。完整方法 56/240，标准化 generic 24/240，差异 13.3 个百分点，配对区间 [9.6,17.1]。portfolio 为 49/240，认证差异区间跨零，并在覆盖率与损失上更好。各方法均记录一次假认证；新 coordinate-permutation 结果反转。预先规定的两个库规模诊断另有 320 条记录。
- **Energy 与复用成本**：统一 50% 初始 SOC，按 168 小时有限服务成本、零终端残值重跑 720/720 个单元，无算法失败。source 60/90、原始 generic 70/90；五个档案服务实际 18 个市场，算法种子不计作额外部署。每个成功认证的调用成本约 558 对 214，外部负结果保留。
- **理论与实现**：加入秩误差到首个中心、完整设计序列稳定性的解析证明；明确它不在原 Lean 接口内。修复 legacy Inventory/Queue basis 对四个 operational summaries 的遗漏，补上 core 的 requests 依赖；HVD 表有六行聚合输入及生成器。
- **交付验证**：新包在独立目录、独立 core 环境下重算 4160 条 profile 与 720 条 Energy 记录，分析完全一致，十张当前表格逐字一致。历史 540 条 Energy V3 标量记录仍可分析。解压包的 main 和 supplement 均成功编译；主文非参考文献页数 29（包括文后表格）、摘要 195 词、补充材料 16 页。修复实际发现的 overpic 图 1 打包遗漏。匿名包约 17.4 MB，保留 17360 条历史记录。

当前文件：`manuscript/main.pdf`、`manuscript/supplement.pdf`、`reproducibility/dist/or_submission_revision_20260930.zip`；新协议为 `performance/manifests/submission_revision_20260930.json`，逐任务结果在 `paper_artifacts/submission_revision_20260930/`。

局限与下一步：新确认仍来自项目定义的八类 synthetic mechanisms，区间条件于三个固定 family；Energy 是已有数据的物理模型修正，没有新增独立外部正证据。HVD 只支持聚合表重现，包内没有其 120 条历史种子记录。OR 的剩余风险是贡献力度及外部适用性，不能用编译或复现通过替代。下一步应围绕真实 archive reuse 场景，在冻结协议下检验 source-best + structural、完整方法、standardized generic 与 portfolio 的实际效果和成本，而不是继续堆叠组件。


2026-09-30 追加：随后同市场历史复用试验在 NO_1、SE_1 的真实窗口中复现了小时功率重复分配。上述 V4 Energy 物理结果由此被取代，原记录及原发布包保留。V5 仅修正两阶段共享小时充放电额度，全部八组重跑；同市场复用 V1 的 1750 条已完成记录也不作为修正后证据，成本和停止原因见其 disposition.json。

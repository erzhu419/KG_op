# 当前主线投稿交付验收

本次按[v5诊断](../../markdown/GPT_diagnosis_v5.md)和[投稿门槛](../../markdown/submission_gates_d3fac762.md)补交付，冻结Stage85/93、来源资产、AFTER、约束TS、独立旧终端和原指标。采用一篇融合的当前主线论文，历史Atlas及旧结果分别标识；不新增大规模实验、不重跑失败到成功、不把Stage103改为第三批确认。

## 交付入口与状态

默认入口为[当前主线复现包](../reproducibility/mainline_release/README.md)，旧指南完整保留在历史章节。数据、运行说明、锁定依赖、源码和资产清单由固定归档一起提供。

[固定匿名审稿包](../paper_artifacts/mainline_submission_release_v1/final_release_3/current_mainline_review_20261026.tar.gz)为82,570,414字节，出版源码提交为7384343e，见[发布清单](../reproducibility/mainline_release/release_manifest.json)。这是包含冻结资产的单个交付包，未取回历史检查点或整批轨迹。取回后225项文件核对通过，144份Python／Lean源码与出版提交一致，见[出版源码绑定回执](../reproducibility/mainline_release/receipts/publication_binding.json)。

| 门槛 | 本次交付 |
|---|---|
| G1：当前版本快照 | 当前主线专用组装器、科学配置与来源资产SHA256清单、匿名归档；历史执行源码commit明确为unknown，见下方溯源边界。 |
| G2：逐格到主表 | 提供5760格，Stage85 2592、Stage93 3168含67原失败；从cells独立重算，785组既有摘要的分母、均值、区间和null核对通过。 |
| G3：干净环境真实运行 | 首个登记Inventory案例2026120001/observation_1在隔离Python 3.10.14及七项锁定依赖下完成，复用10笔初始化，新增3搜索＋224终验；与封存记录的查询、终端、部署、认证、预算和元数据全部一致。 |
| G4：当前Lean冷构建 | 16模块全部冷编译，未复用项目olean；94条当前声明、12项依赖报告通过。复用官方依赖缓存，未声称重建全部Mathlib。 |
| G5：统计与失败 | 两批分开、Stage103事后身份、C/R144及G134/136、来源105/97与67失败保留；登记并补入有限样本C补全范围。 |
| G6：成本与匿名材料 | [20臂实际成本](../reproducibility/mainline_release/COSTS.md)、MIT代码许可、CC BY 4.0生成数据许可、第三方说明、数据可用性及作者披露草稿已提供。 |

对应文件：[逐格数据](../reproducibility/mainline_release/data/cells.json.gz)、[独立重算](../reproducibility/mainline_release/data/statistics.json.gz)、[785组对照](../reproducibility/mainline_release/receipts/frozen_statistics_parity.json)、[Lean冷构建](../reproducibility/mainline_release/receipts/lean_cold.json)、[真实运行](../reproducibility/mainline_release/receipts/small_run.json)、[封存行为比对](../reproducibility/mainline_release/receipts/frozen_behavior_parity.json)。最终归档delivery_ready=true，必需验收及已有行为比对均成功，资产齐备；该状态仅表示技术交付验收。

## 缺失、费用与研究边界

Stage93来源C主差的有限登记样本补全范围为[6.250,29.167]个百分点，C交互为[5.208,28.125]个百分点。这些是任意二元补全界，不是置信区间、插补、修复后预测或净G交互保证；原全144来源点估计保持null。详细口径已回填[实验正文](mainline_experiments_working_draft_20261019.md#全登记认证端点的缺失敏感性)。

新发布计算单列：逐格导出及统计子进程21.497026 CPU秒；Lean子进程184.017409 CPU秒，其中证明构建183.634533秒。小例子的四次失败依次为远端缺已解析配置、节点无Git、PyPI证书错误及来源诊断回执字段不匹配，均为零新增目标调用；子进程CPU为0.400369、4.440320、17.943194、35.134108秒，最后一次含两次模型恢复。补齐已有配置、修正交付报告代码并改用锁定官方离线wheel后，使用新目录人工恢复，原回执全部保留。所有任务关闭自动重派。真实小例子、打包和传输费用以各自回执为准；嵌套CPU不重复相加，未知项不写0，历史余量不重置。

成功小例子本身254.464257 CPU秒、峰值399,966,208字节；含环境准备的完整操作子进程288.243132 CPU秒，前者已包含在后者中。227笔新物理调用仅属于本次验收，237笔逻辑调用低于365上限。[验收账](../reproducibility/mainline_release/receipts/acceptance_accounting.json)保留全部尝试；归档启动命令过长的取消记录及改用短脚本后的零仿真组包费用，见[最终组包回执](../reproducibility/mainline_release/receipts/final_assembly.json)。此前归档子进程13.265538 CPU秒、最终归档15.263299 CPU秒均保留，未新增仿真。

本次G3候选代码与出版源码的一处差异仅为约束TS的三行旧注释；已核对实际候选清单及可执行AST一致，按出版注释同步后重建归档，复用已通过的真实运行和行为比对，见[来源对齐记录](../reproducibility/mainline_release/receipts/source_delivery_alignment.json)。中间归档仍在服务器保留，本地只保留最终完整包和小型回执。

Stage85/93于10月18–19日使用上传的工作区文件运行。留存回执可追溯上传路径／字节数、manifest、来源资产封存、执行与结果，但未记录完整执行Git commit或同期源码hash。ed48ed6b和d3fac762是之后的归档提交；本包SHA256只标识本次分发内容，不追认旧执行源码。不可恢复的事实按unknown交付，不补写为已核验。

## 投稿管理

代码MIT、生成数据CC BY 4.0按用户2026-10-26授权准备，第三方保留各自条款。审稿包采用匿名作者信息，原公开仓库与历史证据不删除。[作者披露草稿](../reproducibility/mainline_release/AUTHOR_DISCLOSURE.md)明确融合方向和新旧证据归属。现有稿件只有匿名署名，旧拒稿决定／稿件编号未提供，不能代填为未曾投稿或已满足重投条件。

固定归档、README、依赖和许可遵循[OR软件与数据交付说明](https://orjournal.github.io/InstructionsForAuthors.html)；署名及旧稿重叠的事实填写按[投稿声明](https://pubsonline.informs.org/page/opre/submission-guidelines)由作者完成。两页面于2026-10-26核对。本文记录技术交付，不代替作者提交稿件或确认编辑决定。

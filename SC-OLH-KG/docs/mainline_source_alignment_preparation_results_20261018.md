# Stage90：来源整块置换与3＋3拟合准备完成

**t183498/node002的31项替身全部通过；随后只生成六项拟合准备清单，状态为`prepared_not_started`。** 使用scheduler的node001–node006池、CPU1、VRAM0和单线程BLAS；没有执行真实置换、拟合或科学验证。[检查记录](../paper_artifacts/mainline_source_alignment_control_v1/cluster_control/validation/stage90_attempt1/checks.json)与[准备清单](../paper_artifacts/mainline_source_alignment_control_v1/source_preparation/preparation.json)已保存。

[整块置换入口](../performance/mainline_source_alignment_archive.py)让同一档案的两个来源家族共用一个64策略置换，将每个策略的完整3×2响应块一起移动，保留f/g、三次重复及跨家族配对；X、策略来源标记和任务元数据不动。均值、样本方差及均值观测方差同步重建。native风险尺度沿原式`√max((Σ(g−ḡ)²＋2×0.04²)/4,1e−12)`，保留名义方差收缩；普通样本标准差不能替代它。

[来源准备入口](../performance/prepare_mainline_source_alignment_sources.py)的`prepare`只写元数据；未来独立`fit`动作才生成并保存一次小置换索引，在内存重建来源输入，复用原结构和native配方完成3＋3条拟合管线。两新臂共用各家族成果，输出与全部原档案、模型及其他家族分开。替身确认原资产不变、成功前缀复用、来源拟合失败及未闭合费用保留、不重抽/自动重拟合、缺模型不准入；provider构造失败不记成已拟合，置换保存失败仍记录已发生的抽样。

原AFTER、前端、搜索、旧终端及9个对照保持冻结，352次终验上限不变。真实置换、拟合模型和拟合目录均未生成；新增仿真、终验、评分、拟合、后验恢复和科学任务均0。旧批结果、正反例及历史费用沿原账保留。

本轮保留一次基础设施启动失败：t183496因内联命令过长触发SSH `Argument list too long`，未进入wrapper或测试，随后取消。首次取消因调度意图锁暂拒，等待锁后取消成功；均保留控制费用。将相同顺序命令移入[短入口文件](../performance/run_mainline_source_alignment_source_preparation.py)后，t183498首次实际执行通过，没有重复已完成检查。wrapper总CPU3.125986秒，含验证1.891856秒与准备1.045305秒；pytest1.799125秒和prepare内部0.000085秒已包含，不能再相加。调度/失败/取消控制共213.126699 CPU秒、命令wall合计632.372942秒，与执行费用分列；小文件同步另记[集中费用账](../paper_artifacts/mainline_source_alignment_control_v1/cluster_control/stage90_accounting.json)。未下载模型或原始档案。

下一步可按原登记执行一次3＋3来源拟合并封存独立资产；若失败，保留失败与费用，停止进入目标阶段，不换置换或调配方。全目标队列调度及封存后五个role评分仍需后续接通。本轮仅证明准备接线满足合同，不提供来源学习或部署优势的新证据。

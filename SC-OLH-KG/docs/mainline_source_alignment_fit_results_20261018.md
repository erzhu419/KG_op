# Stage91：三套置换来源、3＋3拟合及独立资产封存

**六项拟合全部完成并封存，失败0、未知attempt费用0、重复执行0；目标验证未启动。** 复用原入口，无运行代码或配方修改。scheduler在node002/004完成结构t183510–t183512，全部成功后才执行native t183515–t183517，再由t183518汇总。见[资产封存清单](../paper_artifacts/mainline_source_alignment_control_v1/source_preparation/source_asset_seal.json)及[完成审计](../paper_artifacts/mainline_source_alignment_control_v1/cluster_control/stage91_completion_audit.json)。

每套原档案仍为两来源家族、128策略、384条已有raw；每家族生成一次登记置换，native直接复用，同步保留完整f/g三重复和跨家族配对。8身份、1024节点和原native配方不变。固定点保留，不筛选或重抽。

| 目标家族 | 保留的固定点索引（从0计） | 低ESS条目／全部f/g诊断 | 最小ESS |
|---|---|---:|---:|
| Inventory | 9 | 2/16 | 3.2325 |
| Queue | 29、32 | 6/16 | 2.6015 |
| FactorShock | 无 | 0/16 | 10.7591 |

六项非空联合风险优化诊断均收敛，三套native均拟合完成并消费128来源点。主要局限是**Queue低ESS的f头覆盖约99.84%身份权重**；Inventory的两个低ESS g头只涉及约0.471%权重。低ESS是原规则下的积分近似警告，未触发拟合失败；三套均未获得精度认证。警告与完整诊断原样封存，不加节点、不调配方、不重拟合。拟合成功只说明固定负对照已建立，尚不能说明来源关联或跨策略传播带来部署优势。

原3档案、6模型、3设计的前后size/mtime一致，输出路径独立；AFTER、前端、搜索、旧终端、9对照及352终验上限不变。新增来源/目标仿真、目标评分、模型/后验恢复均0。六份新模型共63,938,408B留服务器，仅取回99,092B元数据/诊断。全部旧批、正反例、失败及历史账继续保留。

六次拟合净CPU **58.828562秒**；含导入的六wrapper共66.494791秒已包含该值，collect另1.325383秒，调度控制315.999068秒单列。collect旧总账中的Stage90准备0.000085秒没有再次算成本轮拟合；详见[费用账](../paper_artifacts/mainline_source_alignment_control_v1/cluster_control/stage91_accounting.json)。

下一步接通全目标队列的分阶段调度与封存后五个role评分，复用这六份资产做针对性接线替身，再进入已登记独立目标验证。保持原模型和门槛，不因Queue警告筛换来源；后续来源关联结论须限定于这三份固定置换及当前近似配方。

# Stage97：来源侧存量数值审查完成，具体病因仍未知

按[Stage96协议](mainline_source_numerical_executability_protocol_20261019.md)完成只读审查：[小结](../paper_artifacts/mainline_source_numerical_executability_audit_v1/conclusion.json)、[执行回执](../paper_artifacts/mainline_source_numerical_executability_audit_v1/cluster_control/stage97_completion_audit.json)。12/12来源诊断、67/67失败日志完整读取，输入问题0；执行状态`complete`，数值诊断为`cause_unidentified`。

唯一scheduler任务t189905在node006用1 CPU、vram0执行，16项替身先通过，随后审查一次。wrapper退出0、重复wrapper0、retry_count=0；03:53:37 UTC快照仍显示running，完成依据为已封存的成功wrapper及审查回执，不再等待状态回收或重派。AFTER、全部模型、前端、搜索、旧终端和门槛均未修改。

## 失败位置与来源诊断

| 已存traceback定位 | 臂 | 例数 |
| --- | --- | ---: |
| 初始化条件化 | 置换AFTER | 63 |
| 初始化后的观测点自身更新 | 置换AFTER | 3 |
| 终端推荐 | 原结构后端 | 1 |

63例明确落在初始化，但并非所有失败都由初始化调用触发；另3例发生在AFTER已切换到自身更新之后。原结构后端那1例位于终端推荐，不是终验352上限耗尽。67例均只有一次历史attempt，返回查询总数7，原失败费用8617.606060537 CPU秒仍包含于Stage93总账。

| 家族 | 真实来源低ESS头数／16；最小ESS | 置换来源低ESS头数／16；最小ESS |
| --- | --- | --- |
| Inventory | 0；28.8539 | 2；3.2325 |
| Queue | 2；4.7491 | 6；2.6015 |
| FactorShock | 2；4.3206 | 0；10.7591 |

六个结构bundle均保留`precision_certified=false`，已保存的12项非空联合风险优化诊断均报告收敛。六个native诊断只有既存`fit`状态、128条来源策略记录，未记录精度认证或优化器收敛字段；结构身份求积ESS对native不适用。Queue置换6/16低ESS及其约99.84%目标头身份权重警告继续保留。

FactorShock置换来源没有低ESS标记，却产生66个AFTER执行失败；真实来源有2个低ESS头，也不能据此判断哪个来源的目标更新更可执行。来源ESS不能解释这批失败，更不能作为删除组件、替换来源或调门槛的依据。

## 能说明什么，以及停止决定

67例保存的共享误差／最小归一化常数比值范围为1.0436619178e-8至1.4092794423e-4，不能统一视为刚越过1e-8的一点舍入误差。冻结控制流支持“积分值有限、归一化常数为正、积分器success”的推论；具体状态码未记录。该比值仍不是经过认证的后验矩或部署误差。

日志缺少出错f/g头、身份／组件范围、内部raw位置、绝对积分误差、最小归一化常数、状态码与评估次数。仅有一个比值，无法区分绝对误差增大与归一化常数变小，更无法确定尺度、尾部、减法取消或来源权重是否是病因。未恢复这些局部量，也未从现有记录识别出可直接证实的具体实现错误。

**按登记停止本次数值诊断，不选补丁。** 不放宽门槛、不改积分配方、不重拟合、不恢复失败后验、不追加单query精算，也不启动新目标验证。后续若推进来源侧有限数值验证，须另行登记；当前结果没有提供可直接采用的修正。

全11臂仍为3101完成＋67missing，正反例及旧／新批保持原状。39个FactorShock几何缺完整来源对与另8个无质量基线分开；来源C/G完整子集仍为105/97，全144来源效应仍null。此次审查定位了执行阶段，没有补出反事实部署，也没有增强净质量交互的科学结论。

## 接线与费用

[入口](../performance/audit_mainline_source_numerical_executability.py)仅用标准库读取JSON、JSONL和日志。提交前补齐服务器缺失的6份既有真实诊断及2份历史费用账，并从本地封存状态提取6KB固定日志索引；原近2MB状态、模型和轨迹未传输。索引接线前后两次本地替身均16通过，回执分别保留；服务器实际检查16通过，旧失败未重跑。详细来源表和67例表留服务器，只取回小JSON。

新增审查内部CPU0.212842219秒，审查进程CPU0.284086秒、检查CPU0.702107秒，均嵌套在wrapper的1.176033秒内；本地替身共0.965767秒。审查读数时已测控制／传输CPU92.014679秒，最终文档同步继续追加[独立Stage97费用账](../paper_artifacts/mainline_source_numerical_executability_audit_v1/cluster_control/stage97_accounting.json)，不与嵌套分量重复相加；编辑及未计时小检查CPU保持unknown。原科学、来源拟合、路径分析及Stage96登记账保留。新增仿真、终验、评分、拟合、积分、后验恢复、raw重放和随机抽样全部0。

# Stage98：来源侧有限数值验证协议

Stage98登记状态：`registered_not_started`；[静态清单](../performance/manifests/mainline_source_finite_numerical_validation_v1.json)。Stage98仅登记，未实现或执行探针、替身检查、积分、拟合、评分、模型加载或后验恢复，未提交计算任务。Stage99完成[独立入口与替身检查](mainline_source_finite_probe_wiring_results_20261019.md)时真实探针执行尚未授权。后续Stage100已获授权并[完成固定12项探针](mainline_source_finite_probe_results_20261019.md)，全部6144次调用返回，按未复现即停止规则收束；以下保留原登记协议。接续[Stage97存量审查](mainline_source_numerical_executability_results_20261019.md)的 `cause_unidentified`，不改写已有科学结果。

## 要回答的问题与固定输入

只问：既有来源资产在一组预先固定的短条件化输入上，能否完成原数值核；若失败，能否从同一次调用补齐头、组件批、误差估计和归一化常数。不是精度认证或新的部署比较。

登记12项：3个封存来源组 × 真实/置换资产 × 每组2个实际来源家族。来源组名称是其排除的目标家族，不能误作探针的来源家族。沿用原真实3+3资产、封存置换3+3资产及原档案，不调用来源准备入口重新生成置换。

| 来源资产组 | 探针实际来源家族 |
| --- | --- |
| InventorySupplyChain | QueueResourceControl、FactorShockStatePolicyRZDT1 |
| QueueResourceControl | InventorySupplyChain、FactorShockStatePolicyRZDT1 |
| FactorShockStatePolicyRZDT1 | InventorySupplyChain、QueueResourceControl |

每项只取该原档案中物理策略索引0、1，各第0次重复，按0→1顺序使用完整配对[f,g]。真实/置换资产接收相同原始观测；不使用置换后的标签作另一套输入，不按响应、ESS、目标失败或部署输赢筛点。探针池也只有这两个物理策略。记录原策略向量、来源家族和配置；按已有来源入口使用实际d=50，不把d=50响应搬到d=1000。f保留原尺度，g及其prior沿既有接口各减同一实际tau一次，eta沿用原定义。

复用 `mainline_role_source_holdout.build_source_engine`、`mainline_role_prior.role_inputs/_history` 与 `diagnose_mainline_initialization.stream_head(observer=...)`。构造仅供来源投影使用的上下文，不启动优化器；保留结构f/eta、native替换g及原风险eta、8身份、每头1024节点和batch64。按原身份顺序、f后g、节点批升序，在每批内依次条件化两条raw；只保留数值诊断，不调用候选评分、搜索或终端。来源模型、档案只读，临时条件状态不写回资产。

## 同一次调用补齐的证据

后续仅在独立探针入口包装现有 `local_raw_moments`，不改生产数值核。外层observer提供身份、头、组件批及raw位置；不能由返回查询数猜测内部位置。

| 记录 | 取得方式与含义 |
| --- | --- |
| 资产组、真实/置换、来源家族、identity、f/g、节点[start,stop)、prefix、原策略索引/repeat | 原循环与observer上下文；组件ID在该identity/head内解释 |
| means、covariances、label、eta_label、replicates、tau、原配对raw | 调用前的实际入参及来源上下文，先保存最小输入包 |
| absolute_error_estimate、minimum_normalizer、relative_error、status、neval、bounded/roundoff接受标记 | 成功直接读取原返回dict；保留返回对象、调用数和顺序 |
| integral、逐节点Z、minZ及其节点、error、info.success/status/message/neval、relative_error、shifts | 失败时只读原异常traceback中本次 `local_raw_moments` 帧已有局部量，再原样抛出；不得重调用来补字段 |
| 失败类别、缺字段原因、原异常、已完成/未到达位置、CPU/wall和调用计数 | 区分原数值拒绝、输入/记录错误、资源停止；后续未到达批保持未知 |

`absolute_error_estimate` 是 `quad_vec` 返回的共享max-norm绝对误差估计，不是真实绝对误差或已认证的后验误差。`relative_error = error/min(Z)` 仅复述原门槛。Z是经过各节点density-shift后的积分归一化量，须与shifts一起解释，不等同于原始似然证据或来源权重。原成功返回不含逐节点Z/argmin/shifts时记 `not_exposed_by_return`；最小Z标量仍可保存。失败若早于积分返回，缺量记null及原因，不以代码推论冒充记录值。

这些记录能定位出错头、身份和组件批，并在失败帧存在integral时标出最小Z节点；共享误差不能唯一归因某节点或某列。不得把argmin(Z)叫作已确认的出错组件，也不为获得逐组件真实误差额外拆批积分。输入包及失败现场留在服务器，只取回日志和紧凑JSON。

## 工作上限与停止条件

每项最多 `8×2×(1024/64)×2 = 512` 次原核调用；12项合计最多6144次。24条探针观测使用记录来自既有档案，真实/置换之间重复复用，新增仿真观测为0。保留全部节点，不按低权重删除组件。原[-12,12]、9点定位/32次golden搜索、points、shifts、max范数、epsabs=epsrel=1e-12、limit128、cache16MiB、workers1、status1/2的既有有界接受规则及1e-8门槛全部不变。

每项含加载、投影、积分及诊断写入，CPU上限300秒、wall上限600秒；单次原核调用wall上限30秒。12项工作进程CPU上限合计3600秒，调度/检查/控制开销另记。单线程独立项通过 `submit-cpu-batch` 交给node001–node006显式节点池、vram=0，OMP/MKL/OpenBLAS=1，workers按实时空闲物理核安排，不硬绑忙节点或回落本地。

每项首次原异常或资源超限立即停止，原异常及费用保留；其余已登记项各执行一次。资源停止不算原数值门槛失败；中断前未返回的误差与Z保持未知。无自动retry、补点、重抽、重拟合、精度升级或更换积分规则。缺少固定输入/模型、记录器不保持原行为，则停止对应项并保留输入/实现问题，不找替代样本。

若全部固定项未复现失败，只报告“该有限输入集未复现”并结束；若缺关键量，报告原因未定位并结束。若出现具体不一致，只封存证据，任何实现修正及独立验证另行登记。本协议不选择补丁，不自动准入目标实验。

## 后续接线与输出

下一步先实现独立入口和确定性替身：检查包装前后调用参数、次数、返回对象/原异常相同；双头/节点批/raw位置映射正确；成功、门槛拒绝、积分前异常及超时均不重复调用；计费保留未完成项。替身使用合成返回值/异常，不调用真实积分、模型或来源响应。通过后仍须另获执行授权；本轮这些检查也不启动。

未来输出到 `paper_artifacts/mainline_source_finite_numerical_validation_v1`：12项状态/资源账、逐次紧凑诊断、失败局部输入包和一个结论JSON。结论分开写记录完整性、固定输入可执行性、数值精度 `not_assessed`、d=1000目标收益 `not_tested`，不以测试通过替代科学结论。当前登记账为 `cluster_control/stage98_accounting.json`，只记录静态编辑/同步；未测编辑CPU保持unknown。

未来来源模型加载（最多24次文件加载、12份不同资产）、投影、来源条件化、原核/积分尝试、失败、CPU、wall、wrapper及控制费用按实际另列；不是“零计算/零恢复”。目标engine/后验恢复、旧失败重跑、仿真、拟合、候选/真值评分仍为0。历史67失败8617.606061 CPU秒已在原目标账内，不再重复累计；其他来源、目标、分析及未知费用沿原账引用。

## 局限与主线冻结

这些观测已用于来源拟合，重复施加只是固定数值压力探针，不能当作独立来源留出、合法追加数据的后验或预测校准证据。仅覆盖两条初始化式输入，不覆盖完整initial10、AFTER后续自身更新/复采或结构终端推荐，不能解释原63/3/1各阶段失败。更不能追补原67例的局部状态、39个缺完整来源对或8个无质量基线。

AFTER、真实/置换模型、前端、搜索、旧终端及每格352上限冻结。全部11臂、3101完成/67missing、Queue低ESS警告、六套结构来源precision_certified=false、正反例与费用保留；旧批与Stage93新批分开。来源C/G完整配对子集105/97与全144来源效果null保持不变。此登记不增强来源归因、精度或净质量交互主张，不让单query精算替代prior跨策略传播的决策收益主线。

# 完整正混合批次KG接续

已把历史批次均值／样本方差似然接入正混合原始后验、幻想缓存和分头终端缓存。来源身份由两头共同更新；所有正来源分量保留；新批次只扣其自身次数。73策略、365动作的原生source／GP四次批次验收通过，付费数2→6、剩余验证预算360→356。source缓存与实际更新的身份权重、投影终端值差均为0；GP终端值差2.22e−16。

主要效率瓶颈是`raw_eta_proposal`先按积分节点复制整张146×146协方差，再截取2×2块：小动作集剖析中占约98%时间。改为先取局部块再选来源分量，GP完整批次验收188.84→4.53 CPU秒，约41.7倍；两个积分阶数下的全部动作、奖励和成本完全一致。source同阶验收74.21 CPU秒完成，旧实现三次限时记录仍保留。高斯条件计算也复用了每来源分量一次分解的历史实现，并经独立完整原始后验采样核对。

| 首轮数值验收 | 最大动作差 | 结果 |
|---|---:|---|
| source任务05／布局5停止规则，12／14阶 | 0.00009013 | PASS |
| source任务02／布局6停止规则，14／16阶 | 0.00009381 | PASS |
| GP任务05／布局6停止规则，14／16阶 | 0.00003757 | PASS |
| source任务02／布局6原始批次终端，14／16阶 | 0.00015039 | HOLD |
| GP任务02／布局6原始批次终端，14／16阶 | 0.00016241 | HOLD |

source原始批次的似然归一化诊断误差0.00015825，GP为0.00000767；动作差／归一化门槛均为0.0001。48项不同的相关测试通过；全部数值验收与小动作集剖析累计约37.7单核CPU分钟，包含三次限时；新增仿真、来源拟合和实际验证调用均为0。仅下载约2.9KB压缩汇总。

补充验收已完成：分头归因定位到风险积分，旧全动作值复现误差1.11e−16。复用16阶缓存、按策略分六片后，source／GP原始终端最大动作差5.05e−5／2.85e−5，归一化误差2.11e−5／4.44e−7；实际推荐最大动作差2.42e−5／2.69e−5，全部365动作通过原门。两者均与原始后验选择相同，分别直接验证71／22，选择损失为0；另补GP任务02停止规则，差1.93e−5。20项相关测试通过，无新增仿真、来源拟合或验证，仅下载4.5KB。限时、断言失败及重复运行均留档。

已接上实际推荐的可实现KG：先用原生付费矩滤波选择动作，再按下一步原始后验计值，保留负切换收益；固定动作控制变量使用扣费后的验证预算。任务02／布局6对两个学习器冻结同一查询63，比较STOP、单次和四次采样；所有365动作及联合身份均保留。96个分头状态覆盖384个联合未来，四个扣费基线全部通过。source单次目标头进一步复用旧8点缓存扩到16／32点，净值0.01779363→0.01777939→0.01777376，最后差5.64e−6；目标预测4／8点原先差4.35e−5。此前将这项写成风险轴是报告中的轴归属错误；风险4／8点原先差0.01780743，一直未通过。目标头预测积分的坐标门通过，风险预测轴仍开放。

最新终端精度阶段已完成：复用每条记录最近两个积分规则，仅提升未过门的原始／原生头；高阶记录沿历史策略分成六片，保留跨策略协方差与完整365动作。无需计值的来源身份仍计算精确似然，省去未使用的整张后验协方差更新。354个任务全部结束；首轮一个限时记录由分片接续补完。累计46419.10单核CPU秒（12.89小时），无新增物理仿真、来源拟合或部署验证。29项相关测试通过。

| 同一目标状态／查询63的开发配置 | 原始最大动作差 | 原生最大动作差 | 相对STOP净值 | 终端精度 | 完整预测KG |
|---|---:|---:|---:|---|---|
| source，单次，目标32／风险8点 | 8.97e−5 | 9.05e−5 | +0.01782091 | PASS | HOLD |
| source，四次，双头8点 | 9.62e−5 | 9.94e−5 | −0.00005045 | PASS | HOLD |
| same-source GP，单次，双头8点 | 5.72e−5 | 8.05e−5 | −0.00004191 | PASS | HOLD |
| same-source GP，四次，双头8点 | 4.62e−5 | 6.51e−5 | −0.00016874 | PASS | HOLD |

四个配置均通过原1e−4终端动作与似然归一化门。source单次的目标16／32点差仅5.65e−6，但风险4／8点净值仍差0.01783471；旧风险4点银行未覆盖后来出现的获益分支。其余三个配置虽通过局部预测净值差检查，固定动作的预测期望仍偏离采样前同预算价值，暴露了局部差值检查未发现的预测积分不足。

为区分后验求积误差和预测观测覆盖不足，另做固定直接动作的便宜诊断：只保留查询与部署所需边际，使用真实剩余验证预算，并逐点复现原完整缓存（最大差4.44e−16）。每个完整来源身份内部先分别积分两头，再按原身份质量加权相乘；该分解仅用于固定动作诊断。GP双头512点内部记录全部过门，固定动作塔性质残差在8／64／512点为+0.01185384／−0.00025300／+0.00001056，512点通过1e−4门。

source四次的原离散混合QMC诊断有618／2048个内部记录未过门。改为保留全部正分量的一维Gauss–Hermite加权求积后，2048个内部记录全部过门，最坏似然归一化误差7.61e−11，累计215.36单核CPU秒。此时8／64／512点塔性质残差仍为+0.04520435／+0.00375084／−0.00155538。因此内层分量抽样误差已消除，外层批次均值／样本方差预测覆盖仍未达到原门槛。两来源身份外的质量界为5.90e−12，不足以解释该残差。

外层预测接续已完成：复用历史CDF反演，四次批次改为每头两个实际观测坐标——去偏对数样本方差及其条件批次均值，保留均值／风险相关性和全部正分量。每个实际使用的坐标都与独立自适应积分核对，最大CDF误差1.26e−10；原始完整缓存复现误差不超过8.88e−16。source状态只恢复一次，条件方差权重复用，并由原精度门选择64阶内部CDF。单次继续使用原预测CDF。source四次512→1024点变化1.12e−4仍未过门，因此保留已有计值继续2048点；1024→2048点仅变化4.02e−6。

| 冻结直接动作的预测诊断 | 每来源／头观测点 | 塔性质残差 | 原1e−4绝对残差门 |
|---|---:|---:|---|
| source单次 | 512 | −0.00002235 | PASS |
| source四次，实际观测坐标 | 2048 | +0.00005797 | PASS |
| same-source GP单次 | 512 | +0.00004208 | PASS |
| same-source GP四次，历史诊断 | 512 | +0.00001056 | PASS |

source两项还通过最后两个预测规则的变化门。公共`predictive_observed_batches`接口已保留全部来源身份及原权重，33项相关测试通过。本轮包括8次浮点逐位断言失败及4次旧CDF实现限时，累计4149.38单核CPU秒（1.15小时）；成功source接续2238.07秒、GP单次16.94秒。失败与限时均留档，后续扩展只算新增点；没有新增物理仿真、来源拟合或部署验证。

完整365动作银行已接续新观测CDF：复用80个旧状态，新增144个状态，source单次达到目标32／风险32点，source与GP四次达到双头16点。224个状态的既定验收均通过；三项完整动作银行的终端动作、似然归一化及来源质量门均通过。计值、后验准备及精度接续累计91108.55单核CPU秒（25.31小时），没有新增物理仿真、来源拟合或部署验证。

高阶接续复用历史六片策略分区，覆盖两头；按固定对侧的差值归因，仅提升未过门项。同组观测共用服务器后验和Sobol节点，再用原线程接口并行。风险32点接续将192次逐记录模型加载合为24次，峰值内存11.3GB；40项测试通过，包括串行／分组并行原始与原生批次计值一致。2.64GB后验及完整动作表均留服务器，本地只收小汇总。[结果](../paper_artifacts/historical_predictive_extension_v1/phase_results.json)、[测试](../paper_artifacts/historical_predictive_extension_v1/test_results.json)、[运行账](../paper_artifacts/historical_predictive_extension_v1/phase_account.json)；逐任务命令在同目录`*_specs.jsonl`。

source单次风险8／16／32点的开发净值为+0.01782091／+0.00882614／+0.01409994，16→32差0.005274仍超原1e−4门。32点固定动作塔残差−4.42e−6已过门，原生推荐相对原始最优的损失3.87e−5；自适应预测KG仍为HOLD。source四次和GP四次也仍未通过外层预测门。原16点输入、精度规则及结果保留，32点使用独立输入目录；原始正来源权重、联合身份和预算不变。

单次风险64点阶段完成：复用144个已过门状态、新增144个状态，source／GP同一365动作、查询63及付费设计的终端门均通过，完整预测KG仍HOLD。source净值+0.01375673，目标轴差6.49e−6、风险32→64差0.0003432；风险轴变化比此前缩小约15倍，仍超1e−4。固定动作塔残差−5.27e−5已通过。GP有限银行没有推荐切换，净值−0.00004191仅为扣费，但塔残差+0.0001896未通过。剩余失败在外层预测覆盖；这些模型内值尚不能证明独立部署优势。

高阶改为十二个策略分片，12线程试跑峰值10.1／11.9GB；41项相关测试通过。358个任务全部完成，无失败／限时。计量计算27.84单核CPU小时，共享组开销只计一次，不含未计时登记控制器与本地测试；新增仿真、来源拟合及验证均为0。后验及动作表留服务器。[64点汇总](../paper_artifacts/historical_predictive_single64_v4/phase_results.json)、[运行账](../paper_artifacts/historical_predictive_single64_v4/phase_account.json)、[测试](../paper_artifacts/historical_predictive_single64_v4/test_results.json)；同目录`*_specs.jsonl`可重放。

限制：当前是已开结果开发任务中的单一目标状态／冻结查询先导；表中净值是模型内开发估计。完整预测KG仍为HOLD；完整动作银行虽已扩展，外层预测积分仍未收敛。相邻规则检查不是严格积分误差证明，来源遗漏质量区间也不是预测误差或统计置信区间。矩投影仍改变动作估值；完整查询和批次数选择、其余目标状态及独立部署收益仍开放。

单次风险128点已完成：source／GP全部365动作通过原终端门。source开发净值+0.01348865，风险64→128差0.00026808、固定动作塔残差+0.00010375，完整预测KG仍HOLD。GP塔残差+0.00002691，冻结查询银行通过全部数值门，但没有推荐切换，净值−0.00004191。累计计量54.88单核CPU小时，包含原异常计算、修复及补算；不含未计时登记、缓存重试启动与本地测试。[128点汇总](../paper_artifacts/historical_predictive_single128_v5/phase_results.json)、[运行账](../paper_artifacts/historical_predictive_single128_v5/phase_account.json)。

128点验收发现9条仅原生计值记录出现Sobol方向表并行初始化异常。现由主线程生成并共享精确原6维节点，原异常结果保留；仅重算受影响组，再补3条略超门记录，最终原始／原生最大动作差9.94e−5／9.98e−5。历史同路径组均为单线程，未受影响。56项相关测试通过。

收益变化集中在风险观测CDF约93%–99.6%上尾。已复用历史正权重Legendre实现，对[0.9,1]采用16／32点规则，其余区间继续使用原64／128点银行和已知0.9质量；每个来源身份仍保留原权重及全部身份似然，原生推荐由原始后验计价。新增96个风险状态，较全域翻倍所需256个少62.5%。CDF反演最坏误差1.02e−12；低阶计值分8组、每组12线程，共享后验加载8次，替代96次逐记录加载。96条低阶计值已完成；12／14阶接续各按12个策略分片合为24个共享后验任务，均完成且无超时。14阶上尾16／32点净值差5.22e−6，但终端原始／原生最大动作差1.13e−3／1.06e−3、归一化误差2.10e−4仍未过门；固定动作塔残差+2.25e−4也未通过。已派发仅这96条上尾记录的16阶接续24个分片；本阶段截至14阶已完成计算计量4.53单核CPU小时。完整上尾数值验收尚未完成。416条门检查已改为两轴批量组装，与原逐条判定的最大差3.33e−16，全部判定及阶数一致；检查本身3.90 CPU秒。[上尾计划](../paper_artifacts/historical_predictive_upper_risk_v6/plan.json)、[测试](../paper_artifacts/historical_predictive_upper_risk_v6/test_results.json)。18阶全部48分片结束且无超时；原始／原生最大动作差9.94e−5／9.98e−5、归一化误差7.72e−5，全部365动作的上尾终端门通过；固定动作塔残差+2.24e−4仍未通过。固定动作边际诊断复用2048个历史值，仅补96个上尾值，用时4.67 CPU秒；与完整银行的16阶塔残差差4.24e−7。目标轴32→512点影响3.51e−6，上尾16→32点影响3.79e−7，主要残差位于风险主体区间。主体正权重16／32／64点筛查41.00 CPU秒完成，全部内部规则与CDF检查通过；32点固定动作塔残差+1.53e−6，64点−2.11e−6，选定16／32点。完整主体银行复用416条，仅补96条，准备阶段不恢复大后验；96条低阶计值已完成，无超时。合并后主体16／32点净值差7.62e−7、塔残差−3.46e−5，外层预测变化与塔性质门通过。新记录的原始／原生最大动作差7.42e−3／1.01e−2及归一化误差7.03e−4仍未通过；已按12个策略分片派发24组12阶接续，旧416条无需重算。59项相关测试通过。[主体接续](../paper_artifacts/historical_predictive_positive_body_v7/phase_results.json)。随后接续四次银行、查询与批次数选择，再冻结算法并进入新任务独立确认；主线仍是来源结构先验经目标后验与KG改善决策。

[实际观测预测汇总](../paper_artifacts/historical_observed_prediction_v1/phase_results.json)、[GP单次诊断](../paper_artifacts/historical_observed_prediction_v1/gp_single_results.json)、[本轮测试](../paper_artifacts/historical_observed_prediction_v1/test_results.json)、[包含失败与限时的运行账](../paper_artifacts/historical_observed_prediction_v1/phase_account.json)。source汇总入口`python -m performance.refine_historical_observed_prediction collect`；GP单次为`python -m performance.diagnose_historical_predictive_tower --collect --case gp_n1`。逐例及512→1024→2048接续命令位于本轮各`*_specs.jsonl`；全量观测和内部计值留在服务器。

[本阶段终端汇总](../paper_artifacts/historical_achievable_precision_v4/phase_results.json)、[GP预测诊断](../paper_artifacts/historical_achievable_precision_v4/predictive_tower/phase_results.json)、[source逐分量诊断](../paper_artifacts/historical_achievable_precision_v4/source_tower_gh/phase_results.json)、[测试](../paper_artifacts/historical_achievable_precision_v4/test_results.json)、[运行账](../paper_artifacts/historical_achievable_precision_v4/phase_account.json)。逐片重放命令位于本阶段各目录的`*_specs.jsonl`；完整缓存留在服务器。终端汇总入口为`python -m performance.refine_historical_achievable_precision collect`，预测诊断汇总入口为`python -m performance.diagnose_historical_predictive_tower --collect --case gp_n4`或`--case source_n4`。

[可实现KG先导](../paper_artifacts/historical_achievable_batch_kg_v1/phase_results.json)、[目标头扩展](../paper_artifacts/historical_achievable_objective_v3/phase_results.json)、[终端误差归因](../paper_artifacts/historical_achievable_objective_v3/terminal_attribution.json)、[阶段运行账](../paper_artifacts/historical_achievable_batch_kg_v1/phase_account.json)。服务器入口`python -m performance.historical_achievable_batch_kg collect`及`python -m performance.refine_historical_achievable_objective --collect`；逐任务命令保存在本阶段各目录的`*_specs.jsonl`。

[本轮汇总](../paper_artifacts/historical_batch_head_precision_v1/phase_results.json)、[本轮测试](../paper_artifacts/historical_batch_head_precision_v1/test_results.json)、[运行账](../paper_artifacts/historical_batch_head_precision_v1/execution_account.json)。全量动作表留在服务器，逐片重放命令保存在各阶段`scheduler_specs.jsonl`。

[汇总和历史限时](../paper_artifacts/historical_full_bank_batch_kg_v1/outcome_summary.json)、[测试](../paper_artifacts/historical_full_bank_batch_kg_v1/test_results.json)、[运行记录](../paper_artifacts/historical_full_bank_batch_kg_v1/execution_account.json)。服务器隔离目录中`python -m performance.historical_full_bank_batch_kg collect`重放汇总；逐案例命令保存在各阶段的`scheduler_specs.jsonl`，全量动作表保留在服务器。


## 风险主体区间归因与解决办法（2026-10-10）

“风险CDF主体区间覆盖不足”在本例具体指未来观测的外层求积误差。旧规则从全域Sobol前缀筛出CDF小于0.9的点，再赋每点0.9/点数的权重；概率质量正确，但筛选后的有限点分布不足以准确积分非线性的条件验证通过概率。128点仅留下115点，主体CDF均值0.44918955，而均匀主体的精确均值为0.45。两个来源身份的主体积分相对正权重64点分别高3.0841e−4／2.2449e−4；乘原来源质量和目标因子后合计2.2648827607e−4，与固定动作残差变化一致，差1.74e−16。512点筛选主体仍高估；全域512点反而通过，说明全域网格加上局部筛选／重新加权不能视作已收敛的主体规则。CDF反演误差约1e−12、遗漏来源质量5.90e−12，均不能解释该残差。[归因数据](../paper_artifacts/historical_predictive_upper_risk_v6/body_sampling_diagnostic.json)。

修复复用历史正权重Legendre规则，主体[0,0.9]和上尾[0.9,1]各自积分，权重质量分别0.9和0.1。来源先验、全来源似然、365动作、查询63和剩余验证预算359全部保持。固定动作主体32点残差1.53e−6；完整银行新增96条而复用旧416条，12阶全部24分片合并后塔残差−2.4409e−5、主体16／32点净值差0，通过原1e−4外层诊断门。多数主体状态没有动作切换，控制变量会抵消其固定动作误差，因此净收益稳定不能代替塔性质检验。[最新汇总](../paper_artifacts/historical_predictive_positive_body_v7/precision_summary_at24.json)。

当前剩余瓶颈是内层终端计值。12阶原始／原生最大动作差4.1529e−3／3.2631e−3，归一化误差3.7862e−4；96条新记录仍有动作精度问题，8条归一化未通过。来源内部含512／1024个分量，原始和原生实现都用Sobol第一维离散选择分量；原始计值另有eta重要性权重。四个真实主体标签的小型诊断显示，来源身份5的有效分量数899–1023，12阶分量权重的总变差约2.69%–5.77%，14阶仍约1.44%–1.63%。下一步应先在代表性失败标签及造成最大差的guarded动作上，检验显式保留分量权重、复用已有逐正分量Gauss–Hermite和解析均值积分的办法，再决定是否扩展到全部记录。本轮未再派发全体升阶任务。[分量分辨率](../paper_artifacts/historical_predictive_positive_body_v7/component_resolution_diagnostic.json)、[未通过记录](../paper_artifacts/historical_predictive_positive_body_v7/remaining_checks_at24.json)。

外部复核可在SC-OLH-KG目录重放以下三个小型诊断；已提供约1.13MB查询边际和2048个固定动作值，不需要完整后验。重点查看本模块的head_rules和以下诊断，以及core/conditional_raw_reference.py的条件原始计值、core/conditional_terminal_kg.py的分量选择和guarded积分。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. python -m performance.extend_historical_upper_risk diagnose-body
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. python -m performance.extend_historical_upper_risk diagnose-sampling
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. python -m performance.extend_historical_upper_risk diagnose-components --plan performance/manifests/historical_predictive_positive_body_v7.json
```

本地小型重放用时9.76 CPU秒，六个固定动作结果与服务器逐项一致，最大差0；新增主体归因与分量诊断入口也与保存结果完全一致。此前59项相关测试通过。

局限：外层通过仅对应冻结的task02／layout6／query63／n1开发案例，规则间差异不是严格误差界；分量总变差诊断尚不能证明其是全部guarded动作误差的主因。完整KG数值门仍未通过，四次批次、算法选查询和独立部署确认尚未完成；旧60/60前端证据保留，不与本例拼接成普适优势。

## GPT diagnosis v3核对（2026-10-10）

采纳“显式分量权重＋分量内部条件积分”的局部修复方向。三层对象必须分开：结构身份H、身份内后验分量J、分量内连续变量Z。显式权重消除分量抽样频数误差，不能替代连续积分；有效分量数约1024表示后验质量分散，不表示策略维度1024。低频响应经过验证阈值、回退与动作最大化后，KG被积函数仍可能有陡峭区间。

两处代码限定：原生direct已复用`direct_terminal_reference.py`，解析积分目标／条件风险均值并显式加权全部正分量；尚需改的是guarded风险及配对目标的联合积分。raw已有归一化前的`logsumexp(logs)-log(N)` evidence检查，未通过记录没有被强行归一化误报通过；新增逐分量R_j检查用于归因。raw只对冻结信念的下一次更新避免eta矩投影，前两次付费矩滤波仍保留。

新增`diagnose-sd`重放入口，用实际query63风险边际、所有正分量、32个eta GH节点及本例全部验证预算81／83／128／130／132／359，对照独立非中心t条件通过概率。最大逐点差1.3712e−14，最大加权差6.98e−16，0.35 CPU秒，无新增物理／拟合／验证调用；这些已测边际的SD16不是主要瓶颈。[结果](../paper_artifacts/historical_predictive_positive_body_v7/sample_sd_reference_diagnostic.json)。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. python -m performance.extend_historical_upper_risk diagnose-sd --plan performance/manifests/historical_predictive_positive_body_v7.json
```

下一步局部试验选六条已失败记录：o4/p192（最大raw变化）、o4/p198（该身份最大native变化）、o4/p193（最大evidence误差）、o5/p197（该身份最大raw／native变化）、o5/p176（该身份最大evidence误差）、o5/p207（主体中部对照），均为source_n1风险头。先从服务器缓存确定各自最差guarded动作，导出查询＋最多三条策略的联合边际及相应缓存值；正式365动作不变。比较旧categorical、仅质量纠偏诊断、全部分量条件求积及独立积分参照，同时记录CPU与逐分量evidence贡献。复用历史`query_direct_value`和现有条件均值／协方差计算，保留跨策略联合分布、重要性比率与负收益；局部有效后再扩展96条、重新核对外层并回到批次／查询选择。本轮未派发新服务器任务。

局限：本次SD检查只覆盖冻结查询边际，不是guarded联合误差界；v3所附sandbox小实验未提供本地文件，其数字不作为已重放证据。当前meta-prior学习与消费路径仍在，四项结构先验各自的必要性、完整KG可靠性与新任务独立部署优势尚未证明。

## 六条guarded联合边际局部试验（2026-10-10）

六条记录均完成。只加载一次服务器后验，选每条最差raw／native guarded动作及query63直接动作，保留跨策略协方差、原来源权重、全部身份似然与原计值身份的全部正分量。旧10／12阶缓存复现最大差0。原生direct继续使用已有解析／自适应参考；B只纠正已覆盖分量的频数，用于归因。A12→B12最大混合动作变化为raw6.62e−4／native2.59e−3，质量纠偏不能代替连续积分。

显式分量GH32使本轮evidence误差降到1.55e−14；共用32个连续QMC节点的独立规则差仍达raw6.69e−4／native1.49e−2。改为每分量独立scramble、按均值规范排序保持重排／拆分不变后，每分量128点的两组独立规则差降为raw1.04e−4／native3.34e−4；32→128变化仍达raw6.63e−4／native6.21e−4。原1e−4局部门未全过，六例均不能据此替换完整银行。[汇总](../paper_artifacts/historical_component_guarded_v1/summary.json)。

21项不同相关测试通过。服务器规则及导出计量CPU1088.10秒（0.302单核小时），仅六个低优先级单核任务并行；两个快照恢复启动失败留档、未计时，不含恢复／准备、本地测试和控制器开销。此前累计1241.64秒由完成汇总的循环变量覆盖造成误计，已撤回；各规则独立计时未受影响，完整累计无法从旧文件恢复。历史单策略快照恢复已扩展到实际四策略联合块。约50MB全身份小边际留服务器；本地仅取最差独立规则记录o5/p207的1.68MB重放包和0.41MB参考结果，原合同未计值身份的权重／似然保留，其完整协方差仍在服务器。该包24项旧头／动作积分本地复现最大差2.22e−16，用时0.48 CPU秒。[重放验证](../paper_artifacts/historical_component_guarded_v1/small_joint_replay_validation.json)。单例入口如下，完整六例汇总在服务器运行同模块`collect`；命令和运行账在同目录。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. python -m performance.diagnose_historical_component_guarded run --name source_n1_o5_h1_p207
```

局限与下一步：本轮确认连续积分覆盖仍是独立瓶颈；规则差不是误差界，未读独立确认块，未增加物理／拟合／验证调用，完整KG仍HOLD。本阶段的共享参考和积分坐标接续结果见下节；保持模型和合同，先验证局部效率，再决定是否扩展96条。

## 共享参考与条件积分坐标接续（2026-10-10）

最差独立规则记录o5/p207的两个guarded动作及复用的direct锚点通过局部检查。共享参考选原后验中的实际分量，使用GH32＋16384个连续节点；各正分量在共同坐标上积分与参考的差值，raw差值保留原重要性比率，最后仅做一次全局归一化。128点差值的独立规则最大动作差为raw2.23e−5／native5.42e−5，32→128、高参考4096→16384和独立高参考均通过1e−4检查。普通积分与旧缓存最大差1.22e−15。[控制变量结果](../paper_artifacts/historical_component_guarded_v1/shared_reference/summary.json)。

复用历史可选均值条件积分，改为解析积分incumbent，而保留全部相关协方差和原验证分支：独立128点差降到raw7.38e−5／native1.95e−6，GH32→64变化2.63e−5／5.45e−6；与共享参考的128点及GH64结果交叉核对也通过。相同两个动作和计值身份的单套规则计时，incumbent约48.66 CPU秒，共享参考128点加一套高参考约111.83秒。[坐标结果](../paper_artifacts/historical_component_guarded_v1/shared_reference/coordinate_summary.json)。15项相关测试通过，包含分量重排／拆分、相同分量抵消、历史完整联合raw参考和direct独立参考；本轮成功计算计量502.92 CPU秒，首次8个作业因缺少输出目录保存失败、计时未保存，已修复重跑并留档。本地仅取约0.2MB结果／日志，联合包复用旧文件。

局限与下一步：32点压缩不采用，native独立差1.35e−4，raw与另一积分次序最高差1.69e−4。128点局部通过不是完整365动作／96条记录的KG准入，未读独立确认块，也没有新增物理、拟合或验证调用。下一轮先用较便宜的incumbent128规则验证另外五个失败观测；未通过的局部动作再用共享参考128对照，随后才决定完整银行接续。来源meta-prior、全身份权重／似然、原计值身份的全部正分量、预算及负收益保持。重放入口为`python -m performance.diagnose_historical_shared_risk collect`及`collect-coordinate`，逐片命令、失败日志和计时见同目录的`scheduler_specs.jsonl`、`coordinate_specs.jsonl`及`execution_account.json`。


## 另外五条失败观测接续（2026-10-10）

六条选定失败记录均通过局部动作精度检查。新增五例中，七个失败动作／类型采用GH64＋每正分量256点：独立规则raw／native差最高3.41e−5／5.98e−6，GH64→128差6.06e−6／2.67e−6，跨方法差4.71e−5。三条raw记录再检查256→512，最大差1.70e−5，均低于原1e−4门槛；此前128点的失败结果保留。来源先验、权重／似然、全部正分量、联合协方差与预算保持。[最终局部结果](../paper_artifacts/historical_incumbent_validation_v1/admitted_summary.json)。

54个低优先级单核任务完成，计量1.35单核小时，各轮最多20／14／14／6核，无新增物理、拟合或验证调用。只拉取1.99MB小结果／混合输入。两项新增汇总测试通过，五例四轮汇总重放差5.55e−17、准入状态一致。[运行账](../paper_artifacts/historical_incumbent_validation_v1/execution_account.json)、[重放](../paper_artifacts/historical_incumbent_validation_v1/replay_results.json)。入口：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. python -m performance.validate_historical_incumbent replay`；同目录四组`*_specs.jsonl`保存积分命令。

局限与下一步：以上是一个开发案例内六条观测的选定动作检查，规则差不是严格误差界；完整365动作／96条记录及独立决策收益仍待验证。逐动作扩展的GH32／128成本粗估159单核小时，未计加密。下一阶段复用历史`raw_terminal_head`的shortlist缓存，让新分量积分共享三组验证预算；先核对一组的精度与成本，再决定完整银行接续。旧60/60前端证据保留。


## 三组验证预算共享计算（2026-10-10）

复用历史shortlist坐标／证书概率缓存，显式分量积分的三组原验证预算现在共享联合坐标和六个不同的策略／预算概率。最坏已打开记录的两个shortlist、两个计值身份、raw／native共8片，GH32／128和GH64／256分别提速1.96／1.91倍；三组分支值和成本与分开计算差0，原保存计值差0、成本差1.71e−13，全身份混合动作差0。29项相关测试通过。[计时与数值](../paper_artifacts/historical_risk_allocations_v1/summary.json)。

8个低优先级单核任务计量0.374单核小时，峰值内存470MB；仅取39.7KB小JSON，联合边际继续留服务器。按两组实际计时粗估，96条记录的73组guarded动作需要约74／374单核小时，未含direct和目标积分，因此不据此派发完整银行。三组预算共享只证明计算等价，未证明新增预算变体的积分收敛；完整KG仍HOLD。局部剖析的条件均值坐标占约14%，接续复用历史紧凑协方差索引。重放入口为`python -m performance.benchmark_historical_risk_allocations collect`；逐片命令和运行账在同名artifact目录。

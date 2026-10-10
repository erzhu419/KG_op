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

接续复用历史紧凑协方差索引：同一分量／query-eta GH节点的协方差只分解一次，不再对连续节点重复求逆。4个raw分片与上述缓存结果的三组分支值／成本差0；总计低阶／加密规则提速2.11／2.10倍。raw协方差复用平均另提速1.14／1.22倍，一个加密分片计时慢约11%，逐片结果保留。18项回归检查通过，两组汇总在本地小包重放一致。[最终结果](../paper_artifacts/historical_risk_allocations_v1/compact_summary.json)。12个任务全部完成，各轮最多8／4个低优先级单核，累计0.431单核小时、峰值内存470MB；仅取59.6KB小JSON，未拉完整后验。

局限与下一步：按两个已打开shortlist计时，96条记录／73组guarded动作的低阶／加密成本粗估68／339单核小时，不含direct和目标积分；这是成本外推，不能证明全部预算变体收敛，完整KG仍HOLD。剖析显示共享后主要成本是条件验证通过概率。下一阶段先接一条观测的完整银行，以共享的低阶规则检查各组，只对精度未通过的组局部加密，避免全体升阶；验证三组预算的独立规则与升阶一致性后才扩到96条。重放入口为`python -m performance.benchmark_historical_risk_allocations collect`及`collect-compact`；基准缓存版本为本节首次提交f0dd5c75，随后提交加入紧凑协方差。逐片命令和运行账在同名artifact目录。

## 单条观测的完整银行接续（2026-10-10，初筛完成）

已恢复o5/p207原365动作：73组guarded各三种预算、73组direct各两种预算。原生direct的146动作复用历史参考，缓存规则差2.22e−16；另复用两个guarded组的低阶／加密结果。全部8个身份权重／似然、原计值身份的512／1024正分量和联合协方差保持。准备耗时59.86 CPU秒，峰值内存2805MB；75.5MB联合边际留服务器，本地只取准备结果。四项新增测试通过，覆盖预算索引、混合权重、只加密失败组及成本限制时停止派发。

16个初筛分片最多8个低优先级单核滚动运行，逐作业CPU上限1800秒。初筛比较连续32→128点、独立128点和GH32→64；后续只对失败组比较256点、独立256点及GH64→128，再按需检查512点。控制器用已测匹配计时限制后续加密的累计预计CPU成本至3600秒，超过则保留失败结果并停止；这是预计成本限制，非全局实耗硬上限。依次在本地运行`PYTHONPATH=. python -m performance.launch_historical_component_bank screen`和同模块`continue`即可调度、在服务器汇总并只拉小JSON。进度与逐片命令在[artifact目录](../paper_artifacts/historical_component_bank_v1)。

16片全部完成，初筛实耗3.807单核小时、峰值内存296MB（准备另计59.86 CPU秒）。raw／native通过动作数263／293，共34／24组失败；raw连续32→128、独立128、GH32→64失败组数分别23／28／1，native为19／6／0。归一化误差最高8.88e−16。新旧银行最大动作值差1.09e−3／4.14e−4，32条固定目标头下的推荐动作均未变化。全部失败组加密预计7.26单核小时，超过原3600秒预计成本限制，自动流程停止，未扩大派发。[结果与归因](../paper_artifacts/historical_component_bank_v1/failure_diagnosis.json)。

接续选每类比较的实际最大失败，去重得到raw组68／32／28、native组40／66。五组的独立256点、GH64→128全部通过；三组128→256变化仍超门槛，另做512点确认，最大256→512差7.62e−5，五组均通过。随后GH32／256点及独立规则也与已验证高规则一致，单套规则实测加速2.29倍。[高规则确认](../paper_artifacts/historical_component_bank_v1/failure_pilot/resolution_summary.json)、[较低GH阶数检查](../paper_artifacts/historical_component_bank_v1/failure_pilot/cheaper_summary.json)。26个任务共0.809单核小时，最多8核滚动、峰值内存671MB；三轮累计预计3472秒，未超过3600秒限制。五项相关测试通过。入口为`python -m performance.diagnose_historical_bank_failures prepare`／`prepare-resolution`／`prepare-cheaper`和`python -m performance.launch_historical_component_bank pilot`／`pilot-resolution`／`pilot-cheaper`；服务器端前一模块`collect --stage refine|resolution|cheaper`汇总，逐片命令在`failure_pilot`子目录。

局限与下一步：通过覆盖现为raw272／native299个动作，还剩31／22组。按实测计时，剩余组采用GH32／256点、独立规则及GH64／256对照的初始加密预计2.74单核小时；该计划已保存但未派发，512确认另计。继续按失败类型加密，保持1e−4门槛和最多8个低优先级核心。这里只覆盖一条开发观测及固定32条目标头缓存，规则差不是严格误差界；完整银行、96条／外层KG和独立决策收益仍待验证。未读取独立确认块，也未增加物理、拟合或验证调用。来源meta-prior及原身份／分量、协方差和预算不变。


## 剩余53组接续（2026-10-10，已启动）

剩余raw31／native22组已接入GH32／256点、独立256点及GH64／256比较，复用五个已通过最差组和原生direct缓存。16个分片最多8个低优先级单核滚动，允许node001–node006；初始预计2.74单核小时。完成后自动在服务器汇总，只拉取小JSON。若仅128→256比较失败，保留已通过的独立／GH检查，再对E8检查512点；若独立或GH检查失败，另接GH64的独立256点、GH128／256点和GH64／512点。只有后续加密总预计成本不超过3600 CPU秒才派发，否则保留失败并停止；不是实耗硬上限。

7项相关测试通过，覆盖已通过组的复用、三种非连续预算索引、原始失败文件保留及预计成本限制。入口为`PYTHONPATH=. python -m performance.launch_historical_component_bank continue-adaptive`，状态与命令在同名artifact目录的`adaptive_status.json`、`adaptive_controller.log`和`adaptive_specs.jsonl`。本节记录启动状态，尚无新增通过结论；来源meta-prior、全身份权重／似然、原计值身份的全部正分量、联合协方差和365动作保持，无新增物理、拟合或验证调用。完整KG与独立决策收益仍待验证。


## 256点接续完成与512点纠偏（2026-10-10）

16片完整完成，实耗2.410单核小时，raw／native覆盖314／344个动作，推荐未变，归一化误差最高8.88e−16。剩余15组只差分辨率确认、9组为独立规则差异，GH检查均通过。原后续方案仍选256点规则H8，提高GH阶数未针对连续误差；已改为独立512点E9／F9和GH64／512点K9对照。当前四个最差组全部通过，实耗1108.61 CPU秒，覆盖320／350个动作。[归因与结果](../paper_artifacts/historical_component_bank_v1/adaptive_diagnosis.json)、同目录`bank_resolution_summary.json`。

下一步13个仅分辨率失败组只补E9，保留候选E8的独立F8和GH64的H8检查；两轮累计预计3304秒，仍在3600秒限制内，最多8个低优先级单核。9项测试通过，命令及自动汇总在`bank_coarse_*`和`bank-coarse_*`文件。另用缓存核对已通过组的较低GH规则，动作差最高1.22e−5／2.80e−5，推荐未变；匹配guarded规则计算快2.12／2.13倍，非完整KG加速。[诊断](../paper_artifacts/historical_component_bank_v1/cheaper_rule_diagnostic.json)。仍限一条开发观测，规则差非严格误差界；完整银行、96条外层KG及独立决策收益待验证，来源先验、全部身份／分量、协方差与预算保持，无新增物理、拟合或验证调用。


## 剩余八组接续（2026-10-10）

13个仅分辨率失败组合中12个通过，实耗1263.54 CPU秒；raw／native均覆盖353／365动作，推荐未变。raw6的256→512差1.002452e−4，保留失败并补1024点确认；其余七组补512点独立对照。GH32保持为计值规则，同512点GH16仅作阶数对照，原GH64检查留档。按最慢匹配计时和20%余量，八组预计2868秒，低于本轮3600秒限制；16片最多8个低优先级单核滚动、自动汇总。[计划](../paper_artifacts/historical_component_bank_v1/bank_tail_plan.json)。10项测试通过，包含略超门槛仍失败、1024点对照、预算索引和缓存计时。原先验、全部分量／协方差和预算保持，无新增物理、拟合或验证调用；仍只涉及一条开发风险观测，完整KG与独立部署证据待验证。

八组接续实耗2009.65 CPU秒，native全部通过，raw仅组52／53的256→512比较未过。四个补算任务保留选定E9及其独立F9、GH对照，只补E10确认；512→1024差分别1.47e−5／2.92e−5，均低于原1e−4门槛。最终raw／native均通过365／365动作，32条固定目标头下推荐未变。[最终汇总](../paper_artifacts/historical_component_bank_v1/bank_tail_resolution_summary.json)。两轮累计预计3523.16、实耗2365.61 CPU秒，峰值701MB；11项测试通过，包括只替换失败的分辨率比较、保留其他组和累计预算。入口为`python -m performance.validate_historical_component_bank prepare-tail-resolution`及`python -m performance.launch_historical_component_bank continue-bank-tail-resolution`。

下一步先检验CDF主体低端、中部及0.9边界附近观测的适用性，共享现有联合边际，避免逐条重载2.64GB后验或复制75.5MB边际。当前单条银行计值缓存计时合计2.01单核小时，直接外推96条约193小时，尚不含精度检查；须先做小规模成本和精度检查。计时可在服务器运行artifact目录内`analyze_final_bank.py`重放，结果见`final_bank_cost.json`。局限：单条观测通过尚不代表其他95条、224个外层节点或完整KG通过；规则差不是严格误差界，稳定独立决策收益仍未验证。

## 跨观测共享边际接续（2026-10-10）

复用原积分器、缓存与调度器，检查o4/p192、o4/p207、o5/p223（CDF约0.0012／0.4283／0.8988）。组12包含历史低端失败动作，组52是连续积分难点；每组保留全部三种预算，覆盖每条观测6个动作。准备2.21 CPU秒，旧o5/p207的标签／全身份分数／目标头缓存逐项一致；零次完整后验恢复、零字节联合边际复制。24项不同测试通过；首次8片完成积分后保存任务目录失败，已修复并直接复用缓存，其完成规则计时纳入成本。入口为`python -m performance.validate_historical_component_transfer prepare`和`python -m performance.launch_historical_component_bank continue-transfer`。[计划与状态](../paper_artifacts/historical_component_transfer_v1)。

24片初筛及16片256点接续均完成，最多8个低优先级单核；保存的积分计时合计2491.77 CPU秒（不含准备、任务启动和中断规则）。raw／native分别通过9／18、12／18个动作-观测检查，剩余5个组-观测-类型组合。GH检查全部通过；剩余失败来自连续分辨率或独立规则差。全补512点的累计预计4345.24秒超过3600秒限制，控制器已停止派发，当前无存活作业。[接续汇总](../paper_artifacts/historical_component_transfer_v1/adaptive_summary.json)。

仅用缓存再检验跨标签控制变量：沿用历史未归一化分子差值方法，以已验证o5/p207高规则积分为锚，加入目标与锚点的配对低规则差，最后除以目标归一化质量。三条观测的raw组52均保留原分量、身份分数、目标头和三种预算；升阶／独立／GH规则最大动作差分别6.17e−5／8.80e−5／1.50e−9，概率和成本均在有效范围，无裁剪。[缓存诊断](../paper_artifacts/historical_component_transfer_v1/cross_label_anchor_diagnostic.json)；服务器重放：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. python paper_artifacts/historical_component_transfer_v1/diagnose_nearby_anchor.py`。下一步先以目标标签的直接高规则积分交叉验证，再测实际节省；本轮未证明加速，未改动正式通过计数。

局限：仅一个冻结开发任务的选定动作，规则差不是严格误差界；未新增物理、拟合或验证调用，未读独立确认块。原source meta-prior与旧60/60证据保留，完整365动作跨观测、96条主体观测、224个外层节点、完整KG及独立决策收益仍待验证。

## 跨标签复用的直接高精度对照（2026-10-10）

上轮剩余5个组合全部通过；三个观测的两组guarded动作、三种预算现在raw／native均为18／18个动作-观测检查通过。组52补齐直接512／1024点、独立512点及GH16／32比较，低端raw组12保留原256点独立／GH检查并补512点确认。组52的raw跨标签复用结果与直接512／1024点的最大动作差分别3.06e−5／1.47e−5，三条观测全部通过，重复128点规则与旧缓存差0。[结果与计时](../paper_artifacts/historical_label_control_v1/summary.json)。

同一worker中测量全部正分量的128／512点积分：三条目标观测合计60.02／289.39 CPU秒，复用锚点后的候选积分提速4.82倍；计入一次锚点候选积分94.32秒后，三条合计提速1.87倍。此比较只计候选积分与混合计值，准备和精度验证另计。本轮12片积分合计2389.59 CPU秒，预计2819.67秒，单独受3600秒预计成本限制；上轮cost_hold及其2491.77秒记录保留。最多8个低优先级单核，全部结束，峰值666MB；20项不同回归测试通过。服务器准备入口为`python -m performance.validate_historical_label_control prepare`，本地调度／拉小JSON入口为`python -m performance.launch_historical_component_bank label-control`。

下一步将复用公式接入其他已验证guarded组，先覆盖历史最差组，再接完整365动作银行。局限：以上是三个开发观测的选定动作及固定32条目标头，规则差不是严格误差界，局部候选提速尚不是完整KG提速；96条主体观测、224个外层节点及独立决策收益待验证。来源meta-prior、全身份权重／似然、全部正分量、联合协方差及预算保持，无新增物理、拟合或验证调用，独立确认块未读取。


## 历史最差组的跨标签扩展（2026-10-10）

冻结复用历史最差raw68／32／28、native40／66组，在CDF约0.0012／0.4283／0.8988的三条开发观测上保留各组三种原预算。初轮30片实耗2742.71 CPU秒：128点复用比较通过raw12／27、native15／18个动作-观测检查。低端raw32／28与直接512点差1.42e−4／1.60e−4，确有残差积分不足；其直接256点规则均通过独立256、GH64／256及512点对照。中部近邻标签五组均通过。原失败结果保存在[initial_summary.json](../paper_artifacts/historical_label_groups_v1/initial_summary.json)。

已接入按组回退：复用失败且直接规则通过时，实际选取直接256点统计／成本，保留原复用失败标记。上端剩余四个组合只补必要规则；新增预计792.16 CPU秒，连同已计量成本累计预计3534.87秒，低于本轮3600秒限制。最多8个低优先级单核，允许node001–node006，使用scheduler；38片全部完成，累计积分计时3377.36 CPU秒、峰值423MB，当前无存活作业。复用或直接回退后的比较诊断raw27／27、native18／18通过；128点复用本身仅raw15／27、native18／18通过，另12个raw检查采用直接256点回退。21个检查有完整直接高阶对照，其他24个仍仅有比较诊断。[最终结果](../paper_artifacts/historical_label_groups_v1/summary.json)。

入口：服务器`python -m performance.validate_historical_label_groups prepare`、`prepare-refinement`；本地`python -m performance.launch_historical_component_bank label-groups`、`label-groups-refine`；服务器`python -m performance.validate_historical_label_groups collect --stage adaptive`。27项相关测试通过，覆盖失败复用的实际回退、仅加密对照不冒充直接准入、预算与身份索引及成本限制。只拉小JSON与日志，联合边际留服务器。

局限与下一步：单锚点固定低阶复用不能统一用于远标签；本轮直接参考只覆盖明确标记的组合，其他组合只有比较诊断。历史71／73个guarded组有非零计时，128点相对原选定规则的预计候选积分降本约2.36倍，这是不同运行的计时外推，未计直接动作、锚点准备及精度检查。下一步先给其余24个仅比较诊断的检查补齐直接高阶对照，再接完整银行的按组自适应规则并实测总成本；随后决定主体观测扩展。来源meta-prior、全部原分量／身份权重、联合协方差、预算与验证器保持，无新增物理、拟合或验证调用；完整KG和独立决策收益仍待验证。


## 剩余直接对照补齐（2026-10-10）

剩余24个检查的完整直接参考全部通过，冻结五组／三条观测的45个检查现均通过高阶对照。已实际选取18个32点复用、15个128点复用及12个直接256点回退；与直接参考的最大差9.37e−5，原复用失败标记保留。[最终结果](../paper_artifacts/historical_label_groups_v1/summary.json)。

16片新增积分计时2203.51 CPU秒，低于本阶段预计2663.43秒；与前轮累计5580.87秒，全部结束，最多8个低优先级单核。28项测试通过，旧E8复用且不重复计费。首批8片在积分前因CLI参数退出，修复后重派，记录在`failed_direct_start`。选用32点的六个组合，原同worker的32／128点计时28.97／117.31秒，局部积分快4.05倍；不含锚点准备、混合计值和精度检查，[缓存计时分析](../paper_artifacts/historical_label_groups_v1/direct_cost_analysis.json)，未新增积分测速。重放入口：服务器`python -m performance.validate_historical_label_groups prepare-direct`、`collect --stage direct`，本地`python -m performance.launch_historical_component_bank label-groups-direct`。

下一步按各组历史G7／E8／E9及其已通过对照接入完整银行，避免统一升阶，再测总成本。原source meta-prior、全部分量／身份权重、联合协方差和验证预算保持。本轮仍限选定开发动作的数值诊断；完整银行跨观测、完整KG和独立决策收益待验证。

## 完整365动作的近邻标签复用（2026-10-10）

o4/p207距锚点标签约7.69e−5。沿用各组G7／E8／E9，原365动作的raw／native均通过32→128点比较，最大动作值差6.63e−7／3.08e−7，低于1e−4；32条固定目标头下推荐均未变。复用全部相关缓存和9个已有目标精度证据的组，实际装入其选定计值。全身份权重、512／1024正分量、联合协方差及预算保持。[汇总](../paper_artifacts/historical_label_bank_v1/fine_summary.json)。

两个单独受3600预计CPU秒限制的阶段分别预计1007.25／3586.36秒，新增积分实耗802.24／2936.48秒，成功worker总计3817.01秒，准备另计29.42秒。64片全部结束，最多8个低优先级单核，峰值228MB。完整银行／边际留服务器，本地仅274KB计划、日志与汇总。direct固定359预算曾因浮点运算变成359.0000000000001，已按解析固定成本修复、用缓存重汇总；原汇总留档。33项相关测试通过。[运行账](../paper_artifacts/historical_label_bank_v1/execution_account.json)。服务器入口`python -m performance.validate_historical_label_bank prepare`及`collect --stage coarse|fine`；本地`python -m performance.launch_historical_component_bank continue-label-bank`，逐片命令在同名artifact目录。

局限与下一步：32→128比较未替代独立／GH检查，已有完整目标精度证据仍仅raw15／native158动作；近邻标签未解决整个CDF主体覆盖、完整KG或独立决策收益。其余raw350／native207动作的独立128点检查预计3580.67秒，可接续；统一GH64／128点预计7583.09秒，超过阶段限制，应先降低残差积分成本再派发。这两项均尚未派发。[定价与重放脚本](../paper_artifacts/historical_label_bank_v1/next_precision_cost.json)及同目录`price_next_precision.py`。无新增物理、拟合或验证调用，独立确认块未读取；旧60/60前端证据保持原边界。


## 完整近邻银行的独立与GH残差比较（2026-10-10）

raw／native各365动作均通过分辨率、独立128点与GH64／32点残差比较，最大动作差约8.96e−7／4.48e−6，低于1e−4；32条固定目标头下推荐不变。已有直接参考及native direct解析缓存参与覆盖，直接目标精度计数仍为raw15／native158。[最终汇总](../paper_artifacts/historical_label_bank_v1/gh_full_summary.json)。

native57／64／69组的独立差分对照产生近零负概率，最小−2.23e−9，选定银行本身有效。已改用缓存的有效直接F7对照，保留原失败，无裁剪或放宽门槛；三组均进入GH小试。独立／小试／完整GH新增积分2935.37／179.07／2866.98 CPU秒，共1.66单核小时。86片全部结束，最多8个低优先级单核，峰值229MB；完整GH阶段同时计入锚点与目标成本。39项相关测试通过。[运行账与范围](../paper_artifacts/historical_label_bank_v1/execution_account.json)。

重放先在服务器依次准备`performance.validate_historical_label_bank prepare-independent`、`performance.probe_historical_label_gh prepare`、`prepare-full`；每个准备后，在本地运行`performance.launch_historical_component_bank label-bank-independent|label-bank-gh-pilot|label-bank-gh-full`，完成该阶段再准备下一阶段。均使用`python -m`、既定环境及单线程配置；逐片完整命令在同目录`*_specs.jsonl`，银行和边际留服务器。

下一步复用历史低／中／上端直接参考与本轮锚点，先计价再扩大CDF主体覆盖。当前仅一条近邻开发观测，规则差非严格误差界，完整CDF、224个外层节点、完整KG和独立决策收益仍待验证。原source meta-prior、全部身份／正分量、联合协方差、动作和验证预算保持，无新增物理、拟合或验证调用，独立确认块未读取。

## 两个CDF内部位置与决策相关性（2026-10-10）

按历史低／中／上端间隙中点预先选定o4/p202、p213（CDF网格约0.222／0.678），两条观测均完成365动作粗筛。六组直接参考及随后补充的赢家／第二名组全部通过；实际选用5组32点复用、2组128点复用、1组直接256点回退。raw12的32点复用不足，p213下128点独立／GH比较也失败；native67的32点复用出现−7.39e−9概率，128点有效，原失败保留。每条观测的直接目标精度覆盖为raw7／native149动作，其余仍是未完成精度验证的粗筛。[结果](../paper_artifacts/historical_cdf_bank_v1/summary.json)。

与o4/p207匹配同32条目标头、身份权重及全部365动作，三处CDF位置都选择动作355（策略63、预算359），第二名为同策略的128预算；当前领先差约0.00097–0.00694，精度替换未改变推荐。赢家／第二名组的直接参考也通过，补算仅2.04 CPU秒：历史积分器已对查询即部署的单点动作使用零维Sobol规则，本轮直接复用。[缓存比较](../paper_artifacts/historical_cdf_bank_v1/decision_relevance.json)；其他竞争动作尚未完整验证，这不是完整最大化或零KG的证明。

48片全部结束，最多8个低优先级单核。粗筛预计2080.60、实耗1695.01 CPU秒；小试连同竞争动作补算累计预计3227.34、实耗1697.85 CPU秒，分别遵守3600秒上限。45项相关测试通过；完整银行及边际留服务器。[运行账](../paper_artifacts/historical_cdf_bank_v1/execution_account.json)。服务器依次运行`performance.validate_historical_label_bank prepare --plan performance/manifests/historical_cdf_bank_v1.json`、`performance.validate_historical_cdf_bank prepare-pilot`、artifact目录内`compare_label_decisions.py`、`performance.validate_historical_cdf_bank prepare-decision`；各准备后本地分别调度`performance.launch_historical_component_bank cdf-bank-coarse|cdf-bank-pilot`，最后重跑缓存比较。模块均用`python -m`，逐片环境与完整命令见`*_specs.jsonl`。

下一步先用已打开开发缓存检查其他查询／任务的动作竞争，再优先给会影响动作选择的组补参考；保留全部365正式动作。两条观测的剩余全量128点比较各预计3731／3733秒，均超过阶段上限，未派发。局限：当前是开发块的选定数值参考与条件推荐诊断，尚未覆盖完整CDF、外层KG或独立决策收益。原source meta-prior、全部8身份／512与1024正分量、联合协方差、359预算及验证器保持，无新增物理、source-fit或verifier调用，独立确认块未读取。

## 完整CDF的决策区域与竞争冲突（2026-10-10）

重放现有正权主体／上尾缓存：CDF<0.9区间未换策略，整个CDF的native换策略质量约6.99%；主要实现收益来自CDF>0.93，0.90–0.93区间仍有−2.12e−5局部收益。模型净预览为+0.0135758，已扣除一次查询的验证机会成本。此前三个主体位置推荐相同，不能据此判断查询没有价值。[分区结果](../paper_artifacts/historical_decision_regions_v1/summary_with_direct.json)。

历史零维Sobol分支已覆盖全部96条主体观测的direct组136（策略63、预算128／359）：GH64／128最大动作差3.60e−12，新增积分42.82 CPU秒。随后对两处CDF约0.0678的native低／高规则推荐冲突补齐guarded组63的独立256点、GH64／256点及512点参考，全部通过，最大差8.95e−6；各32条目标头均由direct动作355胜出，最小优势5.95e−4，支持两处粗规则切换为数值伪影。[直接参考](../paper_artifacts/historical_decision_regions_v1/direct_summary.json)、[冲突参考](../paper_artifacts/historical_decision_regions_v1/native_reference_summary.json)。

15个worker全部完成，最多4个低优先级单核。两批direct、三次缓存汇总上限及native参考累计预计1829.18 CPU秒，低于3600限制；新增积分488.62、成功worker总计591.34 CPU秒，已测成功准备另计34.07秒，未计时的元数据准备单列。51项相关测试通过，逐片命令与[运行账](../paper_artifacts/historical_decision_regions_v1/execution_account.json)保留。只拉小JSON／日志，完整银行、联合边际及后验留服务器。

局限与下一步：仅一个已打开开发状态与冻结query63；其他动作-观测组合、完整外层KG及独立部署收益仍待验证，有限规则差非严格误差界，模型预览不作为独立收益证据。下一步复用既有上尾规则及各组参考，优先计价CDF>0.93的未验证竞争动作，再补完整终端／外层精度。原source prior、全部8身份分数／权重、512／1024正分量、联合协方差、365正式动作与359剩余预算保持，原正权不重归一化；无新增物理、source-fit或verifier调用，独立确认块未读取。

## 上尾切换动作的完整参考与主体上界筛查（2026-10-10）

复用历史上尾365动作已通过的分辨率结果，不重新计算整库。对策略34／71的direct组107／144，在两来源的p153–175共46条观测上补独立128点、GH64／128点及原缓存高规则对照，保留两种原预算。先检查p155／168，再扩展余下位置；184个动作-观测检查全部通过，最大独立／GH／历史高规则差分别9.26e−6／2.83e−13／2.95e−5。替换后的模型净收益预览为+0.0135764，与旧值仅差+6.46e−7。[结果](../paper_artifacts/historical_upper_decision_refs_v1/full_summary.json)。

另用现有64条32点主体观测诊断目标收益上界：验证器各部署分支概率之和至多1，因此给定缓存目标头及身份后验，可用各身份最大分支质量加未计算身份的最坏效用质量构造上界；保留2e−4分离余量。已回填上轮两条native组63参考。raw／native原超差动作-目标头组合分别由424380／403416降至253518／248996，约排除40%／38%；尚涉及72个guarded组，raw另有25个direct组。[缓存筛查](../paper_artifacts/historical_decision_regions_v1/body_objective_bound_summary.json)。

17个作业全部结束，最多8个低优先级单核。累计预计3523.26 CPU秒，包含额外240秒缓存分析上限，低于3600；新增积分2332.78、成功worker2375.55 CPU秒，准备／汇总另计13.60秒。积分worker峰值211MB，缓存分析峰值4679MB；21项相关测试通过。[运行账和重放入口](../paper_artifacts/historical_upper_decision_refs_v1/execution_account.json)。原source prior、全部身份／正分量、联合协方差、365动作和359预算保持，无新增物理、source-fit或verifier调用，只拉小JSON／日志。

局限与下一步：仍为一个已打开开发状态与冻结query63；有限规则比较和缓存目标上界不是严格积分误差界，也未改变正式准入门。其他历史主体参考尚未全部回填，因此以上残余数不等于需要新增计算的数量。下一步先接入已验证的原锚点完整银行、近邻全库比较及各组参考，再给真正缺失的guarded残差计价。完整外层KG与稳定独立决策优势仍待验证，独立确认块未读取。

## 主体参考统一回填与补验证（2026-10-10）

原锚点完整银行、近邻全库、各标签组选定参考及上尾切换参考已实际接入，共853个组-观测-计值类别条目；旧低规则保留，直接目标、配对残差及解析参考分别记录。[统一汇总](../paper_artifacts/historical_body_reference_assembly_v1/compact_summary.json)。给定32条目标头，主体目标轴最大raw／native分辨率差1.30e−5／6.77e−7，均低于1e−4；模型净收益预览仍为+0.0135764，推荐未变。

复用历史五个最难组及已粗筛CDF约0.222的E5，新增raw9／native6个动作检查全部通过。native40／66选用32点复用，raw68／28选用128点；raw32的复用与直接512点差1.00319e−4，严格保留失败并实际回退到通过独立／GH／512点比较的直接256点。[补验证](../paper_artifacts/historical_body_reference_pilot_v1/summary.json)。

累计预计3312.41 CPU秒，含首个缓存worker失败重试及准备／汇总预留，低于3600限制；13个作业已结束，最多8个低优先级单核。新增积分1683.72 CPU秒，成功worker报告合计1836.36秒；39项测试通过。原银行direct固定成本的浮点尾差已复用既有解析预算处理修正，原失败日志保留。[运行账与重放](../paper_artifacts/historical_body_reference_assembly_v1/execution_account.json)。大银行、后验及边际均留服务器，无新增物理、source-fit或verifier调用。

剩余瓶颈是风险头跨观测覆盖：64条主体观测仍有raw8973／native4512个组-观测组合缺参考，沿用单锚点的四规则补全预计180.84单核小时，尚不含失败后的直接参考回退，因此未整批派发。[缓存定价](../paper_artifacts/historical_body_reference_assembly_v1/remaining_cost_plan.json)。下一步先比较p197目标使用新p202近邻锚与原p207锚的残差，按实际标签距离选锚并复用历史积分；先计价再派发。当前仍限一个开发状态和冻结query63，规则差不是严格误差界，完整外层KG与稳定独立决策优势未证实，独立确认块未读取；原source prior、8身份分数／权重、512／1024正分量、协方差、365动作及359预算保持。

## 近邻锚点的匹配验证（2026-10-10）

同一p197目标、32条目标头和冻结五难组，复用p202直接参考，对照原p207锚；实际标签距离由0.08825降至0.04917。15个动作检查均有通过的直接参考：两组native共6动作仍用32点复用，raw68／32共6动作由直接256点回退改为128点复用，raw28共3动作仍回退。最大128点直接参考差由1.17949e−4降至8.39312e−5。选定目标规则的历史计时合计131.01→91.32 CPU秒，下降30.29%；这一数字不含锚点、独立／GH校验与组装，不代表端到端加速。[对照结果](../paper_artifacts/historical_nearby_anchor_v1/summary.json)。

再复用其余三条历史锚点，15次探索比较未消除raw28失败；其近邻128点候选与E8／F8／H8／E9直接参考最大差仅4.55580e−5，失败来自低分辨率独立残差比较1.02777e−4。原失败和直接回退保留，正式门槛未改。[缓存诊断](../paper_artifacts/historical_nearby_anchor_v1/library_comparison.json)。下一步先预定更细直接独立／GH参考的升级路径，在已打开缓存上检验其能否减少误拒并保留真实超差，再按决策相关性扩展。

11个作业全部结束，最多8个低优先级单核；阶段累计预计2791.19／3600 CPU秒，新增积分1716.08秒，成功worker报告合计1804.72秒。43项测试通过，参考条目853→858，模型净预览仍为+0.0135764，推荐未变。[运行账及完整重放入口](../paper_artifacts/historical_nearby_anchor_v1/execution_account.json)。主积分器、source prior、全部身份／正分量、协方差、365动作、359预算及验证器保持，只拉小JSON／日志，无新增物理／拟合／验证调用。

局限：近邻复用已有局部效率证据，但主体还缺raw8970／native4510组-观测参考，完整四规则比较更新定价仍约180.84单核小时，未派发整矩阵。当前只是一条开发目标、固定query63与有限规则比较；完整CDF／外层KG和稳定独立决策收益仍未完成，独立确认块未读取。

## 更细独立参考的升级路径（2026-10-11）

沿用历史直接独立对照，以1e−4阈值逐一比较候选与E8／F8／H8／E9，并检查参考自洽、范围和归一化。6条已打开观测的33个主组合、5个原锚点对照完成回放：p197 raw28改为32点复用；p223 native66发现旧漏检，对F8差1.12625e−4，现升级128点；p202 raw32的真实超差继续回退。[诊断](../paper_artifacts/historical_reference_upgrade_v1/findings.json)。

95个动作现用47个32点、33个128点复用、15个直接回退。选定目标规则计时576.60→498.69 CPU秒（−13.51%）；p197新旧锚点均用新验证器时，近邻节省39.65%。新增积分为零，两个缓存作业共95.55 CPU秒，预计600／3600秒，49项测试通过。升级统计已接入银行，推荐与模型预览未变，旧摘要保留。[运行账及重放](../paper_artifacts/historical_reference_upgrade_v1/execution_account.json)。

局限与下一步：计时不含锚点／细参考取得成本，有限规则比较非严格误差界，未新增独立决策证据；主体剩余覆盖及约180.84小时全量定价未变，未整批派发。下一批按决策相关性补参考，先单独计价缺失细规则。原meta-prior、全部身份／正分量、协方差、365动作、359预算与验证器保持，无新增物理／拟合／验证调用，独立确认块未读取。

## 按实际竞争关系补参考（2026-10-11）

64条主体观测的未验证竞争者中，guarded63距赢家最近；其余未验证组最小领先差约0.056。选择raw o4/p196、native o4/p199的最近竞争者，并加入唯一粗细差接近领先差的native o5/p193。三组九个动作均通过E8／F8／H8／E9直接参考，实际选用32点复用，最大差1.81e−5。回填后355的最小优势分别为0.001058、0.000667、0.001376，128条记录的推荐及KG净预览未变。[结果与重放](../paper_artifacts/historical_risk_competition_v1/execution_account.json)。

6个积分分片最多同时运行6个低优先级单核，积分实耗819.39 CPU秒；连同两次缓存分析，worker报告994.76秒，阶段预计1611.59／3600秒。53项测试通过，原prior、身份／正分量、协方差、动作、预算与验证器保持。下一批按领先差排序，已计价7个组合、14个分片、最多8个并行，预计3283.91秒，尚未派发。[下一批计划](../paper_artifacts/historical_risk_competition_v1/next_cost_plan.json)。

局限：剩余119个guarded63组-观测直接参考预计11.56单核小时；其他组仍缺精度证据，其经验领先差不构成严格界。当前解决的是开发缓存里的选定竞争者，完整CDF／外层KG及稳定独立部署收益仍待完成。无新增物理、拟合或验证调用，独立确认块未读取。

## 冻结七组合批次的完成与回填（2026-10-11）

执行上一阶段已定价的7个native guarded63组合，不在看到参考后重选；21个动作检查全部通过E8／F8／H8／E9直接参考，均选用32点复用，最大差1.27e−5。补算后六个组合的领先差增加、一个略减，动作355仍胜出，最小领先差0.000610。实际回填后参考条目861→868，全部128条记录的推荐和模型净收益预览+0.0135764均未变。[运行账与重放](../paper_artifacts/historical_risk_competition_batch_v1/execution_account.json)。

14个积分分片及1个缓存汇总作业全部完成，最多8个低优先级单核；积分实耗2054.40 CPU秒，成功worker报告合计2124.55秒，阶段预计3283.91／3600秒。54项测试通过，汇总只拉约226KB的JSON元数据，大银行、边际和后验留服务器。下一批7组合／14分片已按更新后的实际计时重新定价，预计3290.99秒，尚未派发。[下一批计划](../paper_artifacts/historical_risk_competition_batch_v1/next_cost_plan.json)。

局限与下一步：guarded63剩余112个组-观测检查，完整细参考预计11.26单核小时；继续优先补最接近赢家的缺失组合。其他组的约0.056经验领先差仍非严格界，完整CDF／外层KG及独立部署收益尚未完成。原source prior、全部身份／正分量、协方差、365动作、359预算及验证器保持，无新增物理、拟合或验证调用，独立确认块未读取。

## 第二批七组合的匹配验证（2026-10-11）

按上一批冻结计划补齐2个raw／5个native组合，覆盖6条主体观测；21个动作检查全部通过E8／F8／H8／E9直接参考，均选用32点复用，最大差1.89e−5。回填后5个native领先差缩小、2个raw领先差扩大，动作355仍全部胜出，本批最小领先差0.000606；这批粗规则的误差方向不能用同一偏移纠正。参考条目868→875，全部128条记录的推荐及模型净预览+0.0135764未变。[结果与重放](../paper_artifacts/historical_risk_competition_batch_v2/execution_account.json)。

14个积分分片及1个缓存作业全部完成，最多8个低优先级单核；积分实耗2010.84 CPU秒，成功worker报告合计2091.40秒，阶段预计3290.99／3600秒。调度和剩余定价入口现直接接受批次计划，55项测试通过；汇总只拉约223KB的JSON元数据。下一批8组合／16分片预计3516.76秒，尚未派发。[下一批计划](../paper_artifacts/historical_risk_competition_batch_v2/next_cost_plan.json)。

局限与下一步：guarded63剩余105个组-观测检查，完整细参考预计11.18单核小时；继续优先补最接近赢家的缺失组合。当前仍为一个开发状态、冻结query63和32条条件目标头，其他组缺失精度、完整CDF／外层KG及独立部署收益尚待解决。原source prior、全部身份／正分量、协方差、365动作、359预算与验证器保持，无新增物理、拟合或验证调用，独立确认块未读取，大银行与后验留服务器。

## 第三批八组合（2026-10-11）

冻结计划的6个raw／2个native组合全部完成，复用o4/p202已有E5；24个动作检查均选用32点复用，最大细参考差2.32e−5。6个领先差缩小、2个扩大，raw内部也有两种方向；动作355仍胜出，本批最小差0.000974。参考条目875→883，128条记录的推荐及净预览+0.0135764未变。[运行账与重放](../paper_artifacts/historical_risk_competition_batch_v3/execution_account.json)。

16个积分分片及1个缓存作业全部完成，最多8个低优先级单核；积分2043.14 CPU秒、成功worker报告合计2130.38秒，预计3516.76／3600秒，55项测试通过。汇总只拉约220KB的JSON元数据；下一批7组合／14分片预计3252.09秒，未派发。[计划](../paper_artifacts/historical_risk_competition_batch_v3/next_cost_plan.json)。

局限与下一步：guarded63剩余97个检查，完整细参考预计10.42单核小时；继续按动作竞争补缺口。当前仍限同一开发状态、query63及32条条件目标头，完整CDF／外层KG和独立部署收益未完成。原prior、全部身份／正分量、协方差、动作、预算及验证器保持，无新增物理／拟合／验证调用，确认块未读取，大银行留服务器。

## 第四批七组合（2026-10-11）

按冻结顺序完成4个raw／3个native组合，21个动作检查均通过原精度门并选用32点复用，最大细参考差2.3e-05。七个领先差都缩小，但动作355仍全部胜出，本批最小差0.001059；参考条目883→890，128条记录的推荐及净预览+0.0135764未变。[运行账与重放](../paper_artifacts/historical_risk_competition_batch_v4/execution_account.json)。

14个积分分片及1个缓存作业全部完成，最多8个低优先级单核；积分2211.91 CPU秒、成功worker报告合计2311.04秒，阶段预计3252.09／3600秒，55项测试通过。只拉约218KB汇总JSON及3.7KB任务摘要。下一批7组合／14分片按实测最大耗时计价3561.93秒，未派发。[计划](../paper_artifacts/historical_risk_competition_batch_v4/next_cost_plan.json)。

局限与下一步：guarded63剩余90个检查，完整细参考预计10.80单核小时；继续优先补最接近赢家的缺失参考。当前仍限同一开发状态、query63及32条条件目标头，其他组缺精度证据，完整CDF／外层KG和独立部署收益未完成。原prior、全部身份／正分量、协方差、动作、预算及验证器保持，无新增物理／拟合／验证调用，确认块未读取，大银行留服务器。

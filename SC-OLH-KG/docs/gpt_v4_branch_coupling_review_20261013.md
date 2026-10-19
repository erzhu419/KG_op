# GPT diagnosis v4：核对与缓存验证

原始意见：`markdown/GPT_diagnosis_v4.md`。本次将其视为待评估建议，正式先验、筛查器及原零候选验收条件保持原版本。

## 已确认的具体问题

原guarded界分别给primary单独分支和paired分支提供认证额度，但真实分支必须满足`zP+zPI<=uP`；incumbent同样满足`zI+zPI<=uI`。v4的四变量LP修复成立。这是筛查松弛，不是原始后验计算错误。

按实际`incumbent_position`取primary、incumbent及fallback，已实现[解析对偶上界](../core/branch_coupled_certificate_bound.py)。只需评价六个分段线性断点，避免对每个head调用LP求解器；构造可行对偶并保守修正浮点残差。

9项数学与原边际界测试通过，包括与200个独立primal LP比较、80个任意相关联合事件分布，以及incumbent位置1/2。另有13项接续与费用测试通过。

## 同一第60阶段银行的实际比较

采用独立诊断目录，读取相同224/F16、365动作、全部8身份似然、512/1024正分量及已缓存参考，没有新增联合参考积分。

| 指标 | 原界 | 耦合分支界 |
|---|---:|---:|
| raw剩余加权上界差 | 0.0037727248 | 0.0023442622 |
| raw观测与组候选组合 | 367 | 279 |
| projected剩余加权上界差 | 0.0035070604 | 0.0020355216 |
| projected观测与组候选组合 | 345 | 210 |

raw差缩小37.86%，projected缩小41.96%。已验证参考超过新界的最大值为0；已接纳最佳contrast和推荐计值不变。数据：[comparison_summary.json](../paper_artifacts/historical_positive_risk_uppervariance_axis_branch_coupling_pilot_v1/comparison_summary.json)。

worker `t151814`在node004使用36.9741 CPU秒。新界仍留下约23倍`1e-4`的raw差，不能声明竞争闭合；也不能将这一计算改进解释成source prior独立决策收益。

## 剩余松弛的少量缓存归因

选最近四个阶段中已验证的8个raw guarded观测，共24个预算动作。按相同identity后验和objective heads，比较对已知参考值的平均松弛：

| 已知动作上的平均松弛 | 数值 |
|---|---:|
| 原界 | 0.12426789 |
| 耦合分支界 | 0.09518706 |
| 耦合界中将P/I cap换成缓存认证概率点估计 | 0.01873255 |

P/I概率由同一风险参考的`S_primary+S_pair`及`S_incumbent+S_pair`还原。此替换用于归因，**不是有效的新认证上界**。这些是选择出的已知困难动作的平均值，不能直接当作全局G的互斥贡献或普遍比例。

缓存归因使用4.5433 CPU秒，结果：[cap_attribution_summary.json](../paper_artifacts/historical_positive_risk_uppervariance_axis_branch_coupling_pilot_v1/cap_attribution_summary.json)。这批样本支持下一步优先检查认证caps，而不是继续给分支LP调参或同时追加多种增强。

## 对v4其余意见的判断

- 区分继续策略优于STOP、总体KG期望精度和逐节点动作身份，是正确的。已有`1e-4`规则差不等于统一绝对误差证书，原门仍未通过。
- v4对native覆盖的限定正确：当前重放先按已接纳集合mask，四种替换使用共同池。`native_choices_all_raw_referenced`不能证明完整365动作的native最优已覆盖。此前“原native覆盖齐全”的说法必须限定到该重放流程。
- projected对照是否需要全点态证书，应按最终主张和冻结实现决定；目前不能直接删除检查并声称比较对象不变。
- 单状态raw/projected平均差约`4.16e-5`不能当作meta-prior主突破。主线仍是同信息合同下先验如何改善后验、采样及独立决策。

## 10月14日接续结果

第60阶段已从服务器完整重放恢复结算，本地只取3 KB决策摘要，未重复积分。该阶段支持与失败费用合计188.2792 CPU秒，包含原35秒超时费和小试验首次保存目录缺失的33.5704秒；均保留在原600秒余量。

实现实际SD16函数的15层单调上包络。4个困难观测、12个预算动作的平均松弛从耦合界0.08193781降至0.02333346，缩小71.52%。随后按实际计时登记独立上界细化阶段：522个观测—策略组合，8个低优先级单核任务，预估2509.6862秒，实际991.1757 CPU秒，支持费用47.2328秒。复用8个已保存的小试验认证值，新增边际积分，未新增联合参考、物理观测、source拟合或验证器调用。

| 全局指标 | 耦合分支界 | 加多层认证界 |
|---|---:|---:|
| raw剩余加权上界差 | 0.0023442622 | 0.0003904528 |
| raw观测与组候选组合 | 279 | 95 |
| projected剩余加权上界差 | 0.0020355216 | 0.0004666193 |
| projected观测与组候选组合 | 210 | 97 |

全局64/128阶CDF最大变化6.8714e-7，认证值变化5.4154e-9；已缓存参考超过新界的最大值为0。原先验、全部8身份似然、512/1024正分量、F16/224及SD16保留。27项相关测试通过。数据：[summary.json](../paper_artifacts/historical_positive_risk_uppervariance_axis_certificate_caps_v1/summary.json)、[费用结算](../paper_artifacts/historical_positive_risk_uppervariance_axis_certificate_caps_v1/completion.json)。

剩余raw候选主要是63组`[63,45,10]`与71组`[71,45,10]`；原尾部主导已改变，当前CDF均值0.8–0.95区间约占raw G的60.85%。组间正差可重叠，不能将各组数值简单相加当作G归因。详见[residual_summary.json](../paper_artifacts/historical_positive_risk_uppervariance_axis_certificate_caps_v1/residual_summary.json)。

下一步先复用历史`query_direct_value`处理63组的query=deploy认证边际：传入原familywise delta除以3，保持同一guarded验证器，而非改动协议。该函数已有全部正分量、原raw eta似然和SD16的解析均值积分；应先检查能否减少多层离散松弛，再决定是否计算共享45/10的联合事件界。

原零候选门仍未通过。此轮证据支持数值上界改进；总体KG误差、query/batch排序及source prior独立部署收益仍需后续同合同证据。

## 10月14日query积分与联合事件接续

复用历史`query_direct_value`和`fast_expected_certificates`覆盖全部39个query63观测，按真实预算及短名单delta计算。raw G从0.0003904528降到0.0002992728，候选组合95降至86；projected从0.0004666193降到0.0003360686，候选97降至85。参考包含性通过。

45/10联合认证小试覆盖两个最困难观测、两种身份及raw/projected，保持全部正分量和完整协方差。最大规则差9.0666e-5，通过原门；全局raw仅降至0.0002921344，projected降至0.0003328857，候选数未减少。该小试支持暂不全面推广这一对联合信息。来源5的300.2392秒CPU超时及调度恢复任务`t151832`的40.5103秒均保留；恢复只补未完成的末尾规则。

随后复用同一个pair积分器小试63/45。解析积分primary时四规则差0.00177435；改用历史支持的incumbent均值解析积分后降至0.000102232，两次失败结果均保留且未用于筛查。复用已完成E8/E9，只补F9/L9后，E9控制规则差7.0677e-5，通过精度及全部缓存参考包含性检查。

这次只更新一个raw观测的来源4边际，便排除3个动作、闭合1个候选组：raw G为0.0002718999，剩余85个组合；projected维持0.0003328857、85个组合。该观测中incumbent总认证概率约1.7%–2.1%，而primary失败且incumbent通过的概率约2e-4–4e-4。只给边际上界的LP允许将更多incumbent认证质量放进primary失败区域，造成过于乐观的互补认证收益；主候选—incumbent联合信息直接约束这三个部署分支。

累计cap细化阶段记录1798.7315 CPU秒，包含全部失败及恢复；登记估算3598.5430秒，原600秒分析余量使用258.7446秒，无新增余量。36项相关测试通过。数据：[query结果](../paper_artifacts/historical_positive_risk_uppervariance_axis_query_caps_v1/summary.json)、[45/10小试](../paper_artifacts/historical_positive_risk_uppervariance_axis_anchor_joint_pilot_v1/summary.json)、[63/45小试](../paper_artifacts/historical_positive_risk_uppervariance_axis_primary_incumbent_pilot_v3/summary.json)。

下一步按已测成本登记主候选—incumbent联合认证细化：优先query63组，沿用incumbent均值解析积分及E9控制规则，继续复用已完成profile；再按剩余差距决定是否需要71/45和真正接近最优的动作参考。当前仍是数值计算证据，原零候选门及独立部署主张尚未完成。

## 10月14日63/45批量细化

参考[GPT6最后一次诊断](../../markdown/GPT6_action_competition_diagnosis_20261013.md)与[v4意见](../../markdown/GPT_diagnosis_v4.md)，推广到最高优先级5个观测，18个身份/计值组合中16个通过。

| 指标 | 批量前 | 批量后 |
|---|---:|---:|
| raw G | 0.0002718999 | 0.0001841796 |
| projected G | 0.0003328857 | 0.0002771875 |
| raw/projected候选组合 | 85/85 | 81/82 |

参考包含性通过，最佳contrast不变；6项测试通过。记录2170.9067 CPU秒，含超时321.1337秒、只补L10的恢复76.5733秒及小读取保守费用；登记3541.7170秒。全部失败、旧费用及原余量保留。

已知问题：p17/projected两身份补E10/F10/L10后仍差2.602e-4/2.659e-4，保持HOLD；GH差小，Sobol控制仍不稳定。原零候选门及完整总体误差验收尚未完成。

下一步继续同系列63 raw的`o5_h1_p24`、`o5_h1_p20`、`o4_h1_p16`，再测价71/45；当前63仍占更高调度优先级。p17/projected先小试交换解析均值积分坐标，不全面加阶或铺开45/10。

[结果](../paper_artifacts/historical_positive_risk_uppervariance_axis_primary_incumbent_caps_v1/summary.json)、[费用结算](../paper_artifacts/historical_positive_risk_uppervariance_axis_primary_incumbent_caps_v1/completion.json)、[剩余优先级](../paper_artifacts/historical_positive_risk_uppervariance_axis_primary_incumbent_caps_v1/residual_summary.json)。

## 10月14日高权重63 raw与71/45单点

继续按[GPT6最后诊断](../../markdown/GPT6_action_competition_diagnosis_20261013.md)收紧竞争界。指定的midmean `o5_h1_p24`、`o5_h1_p20`、`o4_h1_p16`共6个raw身份单元全部通过，最大规则差9.263e-5，排除9个动作、闭合3个组。

| 阶段 | raw G | projected G | raw/projected候选组合 |
|---|---:|---:|---:|
| 本轮开始 | 0.0001841796 | 0.0002771875 | 81/82 |
| 三个63 raw后 | 0.0001516077 | 0.0002771875 | 78/82 |
| 71/45单点后 | 0.0001466824 | 0.0002771875 | 77/82 |

剩余63/71优先级接近后，小试uppervariance `o5_h1_p73`。共用Sobol的初试规则差6.485e-4/6.183e-4；同GH独立种子差主导，GH16/32差很小。复用历史`component_risk_allocations`的逐分量种子及E8/F8/L8/E9，保留完整三预算、原family delta、全部正分量/身份似然与SD16，恢复规则差为8.003e-5/1.470e-5。完整statistics/costs留在服务器供复用，本轮只接纳pair caps，另闭合1组、排除3动作。两次参考包含性通过，最佳已接纳contrast不变；6项新增针对性测试通过。

本轮记录1111.5185 CPU秒：63 raw 675.4662，71精度失败200.4656，恢复与筛查235.5868。旧失败8份profiles及全部旧费用保留；累计cap细化5081.1567秒。本阶段登记2036.6849秒，沿原600秒余量累计使用157.5207秒，无重置或新增余量；仅拉取小JSON/日志。

已知问题：仍有77/82候选组合，p17/projected两身份旧精度HOLD保留，原零候选门及总体误差门未完成。当前G仍是冻结有限bank的数值包络。

下一步先处理63 raw的midmean `o4_h1_p28`、body `o5_h1_p8`、midmean `o5_h1_p21`；当前63/71组优先级7.390e-5/6.855e-5不可相加当作G归因。之后按剩余差距与实测费用，再登记71 `o5_h1_p81`单点，不扩大sweep。

[63结果](../paper_artifacts/historical_positive_risk_uppervariance_axis_primary_incumbent_caps_v2/summary.json)、[71恢复结果](../paper_artifacts/historical_positive_risk_uppervariance_axis_primary_incumbent_71_component_pilot_v1/summary.json)、[费用账链](../paper_artifacts/historical_positive_risk_uppervariance_axis_primary_incumbent_71_component_pilot_v1/completion.json)、[剩余优先级](../paper_artifacts/historical_positive_risk_uppervariance_axis_primary_incumbent_71_component_pilot_v1/residual_summary.json)。

## 10月14日第二批63 raw与71 p81

继续参考[GPT6最后诊断](../../markdown/GPT6_action_competition_diagnosis_20261013.md)，沿已有积分/联合界处理指定三点。6个63 raw身份单元中5个通过；midmean `o5_h1_p21`/身份5规则差1.048e-4，维持HOLD。但`p21`与`o4_h1_p28`的raw候选组均已闭合，因此未为p21补算。body `o5_h1_p8`两身份通过，仍留3个动作，加权差仅1.403e-6。

| 阶段 | raw G | projected G | raw/projected候选组合 |
|---|---:|---:|---:|
| 本轮开始 | 0.0001466824 | 0.0002771875 | 77/82 |
| 三个63 raw后 | 0.0001210355 | 0.0002771875 | 75/82 |
| 71 p81后 | 0.0001162190 | 0.0002771875 | 74/82 |

71优先级超过63后，单点uppervariance `o5_h1_p81`复用历史逐分量E8/F8/L8/E9；两身份规则差8.110e-5/1.493e-5，闭合1组、排除3动作。本轮合计闭合3组、排除7动作；两次参考包含性通过，最佳已接纳contrast不变。完整71 statistics/costs留在服务器，只新增pair caps。

本轮记录799.4140 CPU秒，含p21身份5精度HOLD的100.8075秒；累计cap细化5880.5707秒，全部旧失败及恢复费用保留。登记估算3120.6183秒；原600秒余量累计使用249.5106秒，剩350.4894秒，无重置或新增余量。本轮仅取小JSON/日志。

已知问题：仍有74/82候选组合；旧`o4_h1_p17` projected两身份精度HOLD保留，p21 raw身份5也未过门。G仍是冻结有限bank数值包络，原零候选门及总体误差验收未完成。

下一步按已结算计时登记63 raw的midmean `o5_h1_p17`、body `o5_h1_p3`，再按剩余差距考虑71的`positive_full_n4_m4_v4_o4_h1_p20`。三点尚未计算此联合界。body p8已有通过caps，后续应定位剩余分支松弛或复用动作参考。原登记估算与实际费用均保留，下一登记复用原余量。

[63结果](../paper_artifacts/historical_positive_risk_uppervariance_axis_primary_incumbent_caps_v3/summary.json)、[71结果与账链](../paper_artifacts/historical_positive_risk_uppervariance_axis_primary_incumbent_71_component_pilot_v2/completion.json)、[剩余候选及已算注记](../paper_artifacts/historical_positive_risk_uppervariance_axis_primary_incumbent_71_component_pilot_v2/residual_summary.json)。

## 10月14日63 o5 p17/p3与71 full p20

继续参考[GPT6最后诊断](../../markdown/GPT6_action_competition_diagnosis_20261013.md)，复用已有积分器及全部接纳界。两个63 raw点的4身份单元全部通过，最大规则差9.604e-5；midmean `o5_h1_p17`闭合，body `o5_h1_p3`剩2动作、加权差1.047e-7。71优先级约为63的1.87倍后，小试full `o4_h1_p20`，两身份差8.763e-5/1.205e-5，闭合1组。

| 阶段 | raw G | projected G | raw/projected候选组合 |
|---|---:|---:|---:|
| 本轮开始 | 0.0001162190 | 0.0002771875 | 74/82 |
| 两个63 raw后 | 0.0001021105 | 0.0002771875 | 73/82 |
| 71 full p20后 | 0.0000977540 | 0.0002771875 | 72/82 |

本轮6身份单元全部通过，排除6动作、闭合2组；两次参考包含性通过，最佳已接纳contrast不变。保留全部正分量/身份似然、原协方差与SD16/F16/224；完整71 statistics/costs留在服务器，只接纳pair caps。

本轮记录673.8338 CPU秒，累计cap细化6554.4045秒，全部旧失败/恢复费用保留。已完成worker按实测替换预估，原3120.6183登记保留；本轮累计登记3134.6845秒。原600秒余量累计用337.3032秒、剩262.6968秒，无重置或新增余量；仅取小JSON/日志。

已知问题：raw G已低于1e-4，仅指冻结bank的数值包络；72/82候选仍未闭合，原总体误差、外层风险/F16与query/batch验收未完成。旧3个未接纳精度HOLD保留，其中p21 raw已闭合且不补算。

下一步按已结算费用先登记高权重的63 midmean `o5_h1_p16`与71 full `o4_h1_p24`，再按余额及差距决定63 midmean `o5_h1_p28`。这三点的联合界尚未计算；body p3/p8已有通过caps，后续应定位剩余分支或复用动作参考。

[63结果](../paper_artifacts/historical_positive_risk_uppervariance_axis_primary_incumbent_caps_v4/summary.json)、[71结果与完整账链](../paper_artifacts/historical_positive_risk_uppervariance_axis_primary_incumbent_71_component_pilot_v3/completion.json)、[剩余候选及已算注记](../paper_artifacts/historical_positive_risk_uppervariance_axis_primary_incumbent_71_component_pilot_v3/residual_summary.json)。

## 10月14日63 midmean p16/p28与71 full p24

继续参考[GPT6最后诊断](../../markdown/GPT6_action_competition_diagnosis_20261013.md)，先处理63 raw `o5_h1_p16`、71 full raw `o4_h1_p24`，结算后追加63 raw `o5_h1_p28`。三点6身份单元全部通过，最大规则差9.9955706e-05，共排除8动作、闭合3组。

| 阶段 | raw G | projected G | raw/projected候选组合 |
|---|---:|---:|---:|
| 本轮开始 | 0.0000977540 | 0.0002771875 | 72/82 |
| 63 midmean p16后 | 0.0000934070 | 0.0002771875 | 71/82 |
| 71 full p24后 | 0.0000891849 | 0.0002771875 | 70/82 |
| 63 midmean p28后 | 0.0000850038 | 0.0002771875 | 69/82 |

复用全部接纳界、原解析incumbent均值与历史控制规则；三次参考包含性通过，最佳已接纳contrast不变。71完整statistics/costs留服务器，本轮只接纳pair caps；仅拉小JSON/日志。

本轮记录787.0617 CPU秒，累计cap细化7341.4663秒；全部旧失败、恢复费用及3个未接纳HOLD保留。首两点worker实费477.1088秒后，第三点累计登记3596.6082秒；完整结算worker加原600为3552.3937秒，旧登记保留。原余量累计用480.9415秒、剩119.0585秒，无重置或新增余量。

已知问题：69/82候选仍未闭合；G只代表冻结bank包络，原零候选、总体误差及独立确认门未完成。p28身份5规则差9.99557e-5，按原1e-4门通过，观测已闭合。

下一步新增worker登记空间仅47.6063秒，尚不足新成对积分。先复用body `o5_h1_p8`已有四态/目标heads，提取分支贡献，再决定四态归一或fallback联合约束；body p3差仅1.047e-7，暂缓重算。未算raw候选为63 body `o5_h1_p2/p4`与71 uppervariance `o4_h1_p88`；projected最高未算为63 midmean `o4_h1_p24`。这些点保留为待定价候选，不直接复用raw结果替代projected积分。

[71结果](../paper_artifacts/historical_positive_risk_uppervariance_axis_primary_incumbent_71_component_pilot_v4/summary.json)、[最终结果与账链](../paper_artifacts/historical_positive_risk_uppervariance_axis_primary_incumbent_caps_v6/completion.json)、[剩余候选及已算注记](../paper_artifacts/historical_positive_risk_uppervariance_axis_primary_incumbent_caps_v6/residual_summary.json)。

## 10月15日body p8缓存分支诊断与参考补入

按[GPT6最后诊断](../../markdown/GPT6_action_competition_diagnosis_20261013.md)先分解缓存。raw body `o5_h1_p8`四态归一收益为0，fallback贡献最多约1e-19，置零对照也不降低G；`p3`归一仅减G1.062e-8且未闭合。p8的E9缓存guard点值在head5/10可高于旧参考，剩余竞争不能全归为上界松弛。

复用16份旧profiles重构5个动作数值参考：`S=[p10,p01,0,p11]`，guard费用按原四态公式保留。全部16heads/精确预算的六组rule reward比较通过，最大差9.726e-6；fallback遗漏由既有cap覆盖，最大1.860e-19。成本rule差最大0.001619调用，单独保存，原reward门未改。4项无积分针对性测试通过；全屏参考包含性通过，最大参考高于上界为0。

| 阶段 | raw G | projected G | raw/projected候选组合 |
|---|---:|---:|---:|
| 本轮开始 | 0.0000850038 | 0.0002771875 | 69/82 |
| 缓存参考补入后 | 0.0000834961 | 0.0002771875 | 67/82 |

body p8/p3均闭合；接纳5个参考，并非证明这5个动作都劣于旧参考。raw最佳数值reference contrast增加7.585e-7。没有新增成对/完整guard积分，全部旧界、似然、正分量和规则保留，未应用无收益的p8归一修改。

本轮计费72.7332 CPU秒，含首次缺输出子目录的14秒保守失败费与恢复费用；累计7414.1995秒。原600余量累计用553.6747秒，剩46.3253秒，无重置；旧3个未接纳pair HOLD及费用继续保留。

已知问题：G和参考改善均限于冻结bank数值合同；67/82组、总体误差及独立确认门仍未完成。

下一步仅做小型缓存核验：优先检查projected `o4_h1_p17`旧HOLD的完整reward控制差，保留其pair-state HOLD；另有已通过caps的projected body `o4_h1_p3`可复用（加权差4.569e-7）。任何参考补入仍过原reward门，后续筛查先按剩余46.33秒测价，不新增积分或余量。

[分支诊断](../paper_artifacts/historical_positive_risk_uppervariance_axis_cached_pair_references_v1/body_branch_diagnosis.json)、[参考控制](../paper_artifacts/historical_positive_risk_uppervariance_axis_cached_pair_references_v1/compact_summary.json)、[结果与费用](../paper_artifacts/historical_positive_risk_uppervariance_axis_cached_pair_references_v1/completion.json)。

## 10月15日projected缓存reward控制与增量筛查

继续按[GPT6最后诊断](../../markdown/GPT6_action_competition_diagnosis_20261013.md)复用既有四态profiles，保留原1e-4 reward/evidence门及所有历史控制。projected midmean `o4_h1_p17`六组reward比较最大差2.23731e-4；E10/L10 Gauss差仅1.28121e-7，但E10/F10独立Sobol控制差1.20275e-4，仍HOLD。旧pair-state HOLD不变，失败控制不删除。

projected body `o4_h1_p3`完整reward最大规则差8.84642e-6，通过；fallback遗漏上界1.85981e-19。接纳3个缓存动作参考（63/136/209），闭合1组。成本差单独保存，未新增成本精度门。

raw G保持0.0000834961；projected G从0.0002771875降至0.0002767306，raw/projected剩67/81组。增量实现与上一轮完整筛查的448条记录及全局统计完全一致；本轮重算1条、复用447条。1项针对reward准入的回归测试和参考包含性通过，最大参考高于上界为0。

本轮费用33.7186 CPU秒，含首次旧unit缺rules字段的14秒保守失败费及恢复费；累计7447.9181秒。原600余量累计用587.3933秒，剩12.6067秒；全部旧失败/恢复账链保留，无重置，无新增成对或完整guard积分。

已知问题：67/81组、总体误差与独立确认门仍开放；参考改善仅限冻结bank数值合同。当前84项参考计划本地谱系未命中p17完整guard控制，不代表服务器其他历史路径不存在。

下一步仅小查可复用完整guard控制缓存。当前63/71剩余候选已无双身份pair界通过但未闭合记录；projected最高未算63 midmean `o4_h1_p24`，raw未算优先63 body `o5_h1_p2/p4`和71 uppervariance `o4_h1_p88`，仅保留为待定价候选。余量12.61秒内不重复提升p17 Gauss、不放宽原门或补新积分。

[通过及HOLD控制](../paper_artifacts/historical_positive_risk_uppervariance_axis_cached_pair_references_v2/compact_summary.json)、[费用账链](../paper_artifacts/historical_positive_risk_uppervariance_axis_cached_pair_references_v2/completion.json)、[剩余优先级](../paper_artifacts/historical_positive_risk_uppervariance_axis_cached_pair_references_v2/residual_summary.json)。

## 10月15日完整guard缓存索引小查

对照[前次v4诊断](../../markdown/GPT_diagnosis_v4.md)与[GPT6诊断](../../markdown/GPT6_action_competition_diagnosis_20261013.md)，只查未挂载历史完整guard控制，未新增积分或重复筛查。当前服务器部署47份外谱系小元数据、220控制组及同坐标异名匹配均无新可接纳参考；projected midmean `o4_h1_p17`无匹配。唯一命中raw body `o4_h1_p0` Gauss小试，其两身份E9统计和费用与已挂载competition_v5逐值相同，新增该索引不会提高fixed或减少G。

raw/projected G仍为0.0000834961/0.0002767306，剩67/81组；旧3个pair HOLD和p17 reward HOLD不变。本轮计费4.1088 CPU秒，累计7452.0269秒；原600共享余量剩8.4979秒，全部旧失败/恢复费用保留，无重置。

已知问题：检索覆盖当前部署可见的历史元数据目录；未证明其他服务器历史路径不存在。零候选、总体误差、独立确认仍未完成。

下一步停止重复此缓存搜索。按MD的分层归因及完整误差账定价，projected63 midmean `o4_h1_p24`仍是高权重未算候选；当前约8.50秒共享余量不足新成对积分，本阶段继续HOLD，不改变原验收门。

[缓存检索及重复项证据](../paper_artifacts/historical_positive_risk_uppervariance_axis_guard_cache_index_v1/server_index.json)、[费用账链](../paper_artifacts/historical_positive_risk_uppervariance_axis_guard_cache_index_v1/completion.json)。

## 10月15日总体误差证据与下一候选定价

按[前次v4诊断第七节](../../markdown/GPT_diagnosis_v4.md)整理六项误差证据。当前G是冻结bank动作余差，终端规则差不是绝对误差；旧固定动作风险外层日志不覆盖当前guard-max，F16及匹配预算STOP/采样费用误差仍待控制。总绝对误差上界尚不可计算，未新增全局epsilon或改变原验收门；已有遗漏utility额度保留，不重复加账。

projected63 midmean `o4_h1_p24`的三预算与历史projected h4控制匹配；历史worker实耗58.0397秒，沿用20%加2秒估价71.6476秒，单身份超过47.6063秒worker登记空间，拒绝小试准入。目标p24实际费用仍未知，h5未继续定价；未读取服务器、新增积分或重复缓存搜索。

raw/projected G及67/81组不变。本轮费用2.0223 CPU秒（含本地旧快照列表误读的1秒保守失败费及恢复），累计7454.0492秒；同600共享余量剩6.4756秒，全部旧失败/恢复费用保留，无重置。

已知问题：有限bank数值进展不能替代连续预测误差和独立决策确认。下一步先明确终端与外层误差如何传播到实际继续策略，再按证据和既有预算决定新增计算；当前不启动p24积分，原门保持HOLD。

[误差证据](../paper_artifacts/historical_positive_risk_uppervariance_axis_error_pricing_v1/error_evidence.json)、[候选报价](../paper_artifacts/historical_positive_risk_uppervariance_axis_error_pricing_v1/next_candidate_price.json)、[费用账链](../paper_artifacts/historical_positive_risk_uppervariance_axis_error_pricing_v1/completion.json)。

## 10月15日共同池策略刷新与误差传播

继续参考[前次v4诊断](../../markdown/GPT_diagnosis_v4.md)和[GPT6诊断](../../markdown/GPT6_action_competition_diagnosis_20261013.md)。复用 paid references 增量重放3个缓存变化点、其余221点沿用v60；旧3点动作及每head reward完全一致（最大差0）。4个共同池策略总值均不变：新63参考缺另一计值下覆盖，仍未进入raw/projected共同候选池。

当前共同池raw selection contrast为0.0211343541，all-raw参考包络为0.0211356490；同bank选择差δπ=1.2948014e-6，名义G+δπ=8.4790905e-5。一般传播式为 `regret ≤ G + δπ + 完整包络上侧误差 + 所选动作下侧误差 + 权重误差 + 风险/F16外层误差`。已有utility遗漏不重复计入；all-raw参考selector与共同池比较selector分开表述。

校正前轮“baseline缺口”的含义：STOP/扣费后fixed点值已经存在，为0.7438762263/0.7438206395，缺的是匹配绝对误差证书。共同池raw与projected选择的raw净点预览分别0.0210787697/0.0210371520；当前使用独立prior baseline校准，不假定355数值误差自动抵消。

当前224/F16的raw fixed加权点值为0.7452258288，与prior扣费fixed之差为0.0014051892；直接有限bank净点值与control-variate预览也相差此量。它是tower诊断残差，不是绝对误差界，但明确不能假定这两个355数值估计抵消。

本轮费用4.4167 CPU秒，累计7458.4658秒；同600共享余量剩2.0589秒，全部旧失败/恢复费用保留，无重置或新增积分。raw/projected G与67/81组不变。

已知问题：以上是有限bank重放与传播关系；终端及外层绝对误差尚不可填数，不能认证总体regret、连续净KG或独立决策收益。下一步先复用同F16已有外层层级定位tower残差，再为终端/外层证据定价，保留common-pool合同与原HOLD，停止将最佳参考改善直接写成共同池推荐收益。

[3点增量重放](../paper_artifacts/historical_positive_risk_uppervariance_axis_policy_error_v1/incremental_replay.json)、[传播关系](../paper_artifacts/historical_positive_risk_uppervariance_axis_policy_error_v1/error_propagation.json)、[费用账链](../paper_artifacts/historical_positive_risk_uppervariance_axis_policy_error_v1/completion.json)。

## 10月15日同F16外层层级tower定位

参考[前次v4误差账](../../markdown/GPT_diagnosis_v4.md)及[GPT6外层记录](../../markdown/GPT6_action_competition_diagnosis_20261013.md)，只聚合既有同F16层级；实际summary节点数为32/96/128/160/192/224。复用构造逐值核对objective/scores，并只替换risk rows；后五层共同CDF坐标的fixed点值最大变化均为0。

| 风险观测数 | raw fixed对prior-debited的tower残差 |
|---:|---:|
| 32 | 0.0307757645 |
| 96 | 0.0010042886 |
| 128 | 0.0014066572 |
| 160 | 0.0014056307 |
| 192 | 0.0014056550 |
| 224 | 0.0014051892 |

128→224的fixed总变化仅-1.46803e-6，最近192→224为-4.65828e-7；约1.405e-3残差未被这些上部均值/局部方差加密解释。原始32→96变化为-0.0297715，不能将粗层问题与当前残差混为一项。

已知问题：这些是局部层级差，不是绝对误差界；未细化风险区间、F16有限采样与prior baseline校准仍未区分。当前代码证据未证明baseline目标draw/evaluator与缓存F16匹配，不能将残差全部归为风险积分误差。

本轮计费1.2006 CPU秒，累计7459.6664秒；同600共享余量剩0.8584秒。全部旧失败/恢复费用保留，无重置或新增积分；67/81组及原验收门继续HOLD。下一步只核对prior baseline与F16目标来源的匹配，保留未细化区域为待定价项。

[六层缓存比较](../paper_artifacts/historical_positive_risk_uppervariance_axis_same_f16_tower_v1/outer_levels.json)、[F16来源](../paper_artifacts/historical_positive_risk_uppervariance_axis_same_f16_tower_v1/head_lineage.json)、[费用账链](../paper_artifacts/historical_positive_risk_uppervariance_axis_same_f16_tower_v1/completion.json)。

## 10月15日prior与F16目标来源核对

继续参考[前次v4](../../markdown/GPT_diagnosis_v4.md)和[GPT6诊断](../../markdown/GPT6_action_competition_diagnosis_20261013.md)。代码确认两条目标路径都调用`clipped_objective_quality`及`objective_reference`：prior fixed355直接按全部正分量解析期望，F16则计算未来目标观测后的条件raw质量；STOP/debited预算360/356匹配。它们不是应使用同draw逐值相等的两组样本，当前未发现目标函数被替换的证据。

复用已有固定动作分head值，仅重组均值乘积，旧16/16和2048/2048结果复现；已保存的目标/风险观测label/eta匹配差为0。

| 目标前缀 | 风险前缀 | 对prior-debited的tower残差 |
|---:|---:|---:|
| 16 | 16 | -0.0154057147 |
| 16 | 2048 | -0.0002836998 |
| 2048 | 2048 | +0.0000579746 |

固定风险2048时，目标前缀变化影响3.41674e-4；当前有限bank fixed值与分身份F16/R2048乘积仍相差1.68889e-3。已知问题：后验身份混合与分身份乘积是不同有限求积算子，不能把该差全部归给F16或风险轴，也不能将这些诊断当绝对误差证书。最终replacement head的具体kernel配置未重读，无新模型快照相等性声明。

本轮费用0.5469 CPU秒，含记忆路径不可用0.10秒及计时字段误读0.05秒保守失败费与恢复费；累计7460.2133秒。同600共享余量剩0.3114秒，全部旧失败/恢复费用保留，无重置或新增积分；67/81组及原门继续HOLD。下一步只核对有限F16下身份后验混合与分身份乘积的关系，保留原bank。

[目标来源](../paper_artifacts/historical_positive_risk_uppervariance_axis_prior_f16_source_v1/source_comparison.json)、[已有前缀重组](../paper_artifacts/historical_positive_risk_uppervariance_axis_prior_f16_source_v1/paid_prefix_controls.json)、[费用账链](../paper_artifacts/historical_positive_risk_uppervariance_axis_prior_f16_source_v1/completion.json)。

## 10月15日有限身份混合核对与超时结算

沿用前次v4/GPT6诊断。代码与代数确认：同身份条件节点的外权为原prior质量，不需追加mixture density ratio；同节点固定动作的混合差为`Σo πo Qo[Σk≠o Wk(Hk−Ho)]`。分身份固定动作乘积不能替换原bank的动作最大化。

同F16/risk224读取设置0.18 CPU秒timer，却在0.340329923秒才超时退出；未产出混合值、对角值或转移量，tower残差尚未数值定位。失败全额计费、不重试；raw/projected G仍为8.3496104e-5/2.7673058e-4，67/81组及原精度HOLD不变。

本轮费用0.411892015 CPU秒（含0.05秒本地结算支持保守费），累计7460.625208179秒；同600共享余量为-0.100443350秒，即超支0.100443350秒。保留全部旧失败/恢复费，无重置、借额或新增积分。当前停止科研读取；本地账链已结算，未追加服务器同步。后续需先解决预算约束，再完成同节点分解。

[代数与证据边界](../paper_artifacts/historical_positive_risk_uppervariance_axis_finite_mixture_v1/algebra.json)、[失败原始计时](../paper_artifacts/historical_positive_risk_uppervariance_axis_finite_mixture_v1/reader_failure.json)、[费用账链](../paper_artifacts/historical_positive_risk_uppervariance_axis_finite_mixture_v1/completion.json)。

## 10月15日同节点对角项最小方案与预算边界

参考前次v4/GPT6诊断。核对预算代码：600支持余量有独立结算上限，剩余47.6063秒worker登记空间不能直接用于支持读取。原超时及所有旧费用继续保留。

已准备约90行缓存reader：只从当前F16/risk224取各origin的固定355质量均值和风险加权和，计算同节点B，再用已结算A得到D=A−B、E=B−prior；无需逐节点重算posterior或新增积分。D仍包含身份混合与遗漏utility影响，不能直接称绝对误差或替换bank。语法检查通过，尚未执行。

本轮本地准备/审查费0.151570028CPU秒（含0.05秒本地支持保守费），累计7460.776778207秒；原600余额-0.252013378秒，无重置或借额。raw/projected G及67/81组不变。具体提案为追加5CPU秒支持预算，覆盖既有超支、一次缓存读取及结算；尚未获授权。

[可审阅方案](../paper_artifacts/historical_positive_risk_uppervariance_axis_finite_diagonal_plan_v1/proposal.json)、[reader](../performance/diagnose_cached_fixed_origin_diagonal.py)、[预算依据](../paper_artifacts/historical_positive_risk_uppervariance_axis_finite_diagonal_plan_v1/budget_scope.json)、[账链](../paper_artifacts/historical_positive_risk_uppervariance_axis_finite_diagonal_plan_v1/completion.json)。

## 10月15日授权后同节点分解

用户授权追加5CPU秒支持预算，覆盖旧超支、一次缓存读取与结算；原600账及全部失败/恢复费保留。沿用前次v4/GPT6的完整误差账要求，仅复用fixed355缓存。

同F16/risk224得到分身份固定动作乘积B=0.745105791827742；已结算混合A减B为0.000120036925（8.54%），B减prior为0.001285152283（91.46%），两项复现原tower残差0.001405189209。当前F16质量均值与旧缓存差仅2–3e-8；固定当前质量，risk224与已有R2048均值的乘积差为0.001568879746。因此下一步优先核对两套风险节点、权重及证书核来源，而不是继续细拆身份混合。

局限：混合A沿用冻结结果，本次未独立重算；R2048未被证明更准确，上述差均不是绝对误差证书，未改变bank或原门。raw/projected G及67/81组不变，无新增积分。

服务器读取成功，实耗0.282224351CPU秒；首次完整计时发现本地远程客户端另耗7.44秒（未细分导入/传输）。本轮全额结算7.796519944秒，累计7468.573298151秒；原600余额-8.048533322秒，计入已授权5秒后的余额仍为-3.048533322秒。客户端费用此前估计不足，已停止远程操作，无重试、清零或借额；本地账链已更新，未追加服务器同步。

[同节点分解](../paper_artifacts/historical_positive_risk_uppervariance_axis_finite_diagonal_v1/decomposition.json)、[客户端计时](../paper_artifacts/historical_positive_risk_uppervariance_axis_finite_diagonal_v1/client_timing.json)、[完整结算](../paper_artifacts/historical_positive_risk_uppervariance_axis_finite_diagonal_v1/completion.json)。

## 10月15日本地风险来源与客户端核对

参考[前次v4误差账](../../markdown/GPT_diagnosis_v4.md)及[GPT6诊断](../../markdown/GPT6_action_competition_diagnosis_20261013.md)。risk224与R2048共用raw `query_direct_value`、预算356/n4/SD16；前者每源112个分段Gauss点，后者2048等权Sobol点，eta内层分别固定128/自适应16–256。现有正权重、区间质量和CDF验收记录正确。

发现本地已有mean16×variance64 Gauss缓存。复用标量后，原tower残差可分为：混合差+0.000120036925、当前风险规则与高阶规则差+0.001627029560、固定高阶风险下F16→F2048差−0.000341675506、高阶F2048对prior差−0.000000201771。下一步优先定位尚为4阶的方差区间，并统一条件eta控制；暂不继续细拆posterior混合。

已将实际客户端的`sync`导入移入同步分支，缓存读取不再引入其NumPy/SciPy链；仅本地`--help`耗0.219375秒且未加载两库。局限：完整远程调用节省量未测；逐点大缓存不在本地，不能断言哪套积分是真值。高阶规则还改变mean分段及eta控制，上述差不是纯方差误差或绝对证书，原bank/67、81组及精度HOLD不变。

本轮仅本地费用0.519424079CPU秒，累计7469.092722230秒；原600余额-8.567957401秒，含已授权5秒后的余额-3.567957401秒。所有旧失败/恢复费保留，无重置、新积分、服务器读取或同步。

[风险来源](../paper_artifacts/historical_positive_risk_uppervariance_axis_local_risk_provenance_v1/risk_source_audit.json)、[高阶缓存重组](../paper_artifacts/historical_positive_risk_uppervariance_axis_local_risk_provenance_v1/matched_gauss_comparison.json)、[已修客户端](../paper_artifacts/historical_positive_risk_uppervariance_axis_local_risk_provenance_v1/remote_client.py)、[费用账链](../paper_artifacts/historical_positive_risk_uppervariance_axis_local_risk_provenance_v1/completion.json)。

## 10月15日方差半区与eta控制本地归因

继续参考前次v4/GPT6误差账。高阶Gauss本地worker日志可恢复方差CDF上下半区的已加权贡献；同F16重组闭合。来源4占当前对高阶规则差的84.38%；同旧Gauss族variance8→64变化为5.01659e-4，其中来源4占主要部分。当前到旧mean16/variance8尚差1.12537e-3，但这一步同时改变mean分段、阶数与eta控制。

当前仅已细化的meanCDF[.95,.99]方差4→8，使posterior-mixed fixed值改变−4.65828e-7；其余五区间仍用方差4阶、覆盖96%质量。本地缺匹配分项，不能将总差按概率质量分摊成各区间误差，也不能用半区收益大小判断误差来源。下一步优先来源4的匹配区间缓存；当前不生成或读取缺失数据。

eta证据：risk224的64/128最大扰动≤9.99006e-12，对分身份fixed对角项的加权扰动≤7.99317e-12；高阶流程为首个相邻差和归一误差均≤1e-4即停，actual orders不在本地。局限：有限规则扰动不是绝对误差，自适应与128的差尚不可量化。n1旧tail缓存因批量不匹配未纳入；原bank、67/81组及HOLD不变。

本轮费用0.485898314CPU秒（含缺本地文件失败0.01秒及本地启动/结算支持保守费），累计7469.578620544秒；原600余额-9.053855715秒，含已授权5秒余额-4.053855715秒。全部旧失败/恢复费保留，无重置、新积分或服务器读取。

[归因与未知项](../paper_artifacts/historical_positive_risk_uppervariance_axis_variance_intervals_eta_v1/interval_eta_analysis.json)、[半区原始日志](../paper_artifacts/historical_positive_risk_uppervariance_axis_variance_intervals_eta_v1/cached_high_gauss_half_resources.json)、[费用账链](../paper_artifacts/historical_positive_risk_uppervariance_axis_variance_intervals_eta_v1/completion.json)。

## 10月15日来源4最小缺失缓存与提取方案

沿用前次v4/GPT6误差边界。最小缺失来源共4文件：current `tasks/4_3.json`、高阶Gauss来源4两个`bank.json`、R2048来源4风险头`results.json`。本地preparation已含112条坐标/权重，五个目标区间共80条，另32条作为闭合项；复用既有F16均值及prior，无需读取大context。

已准备仅stdlib提取器：输出current六区间贡献、高阶Gauss节点桶贡献/实际eta阶数分布、R2048全局阶数直方图；数组留源端。局限：旧Gauss桶不是同区间Gauss规则，权重不强制等于区间长度；源缓存也未保存非最终128阶值，不能恢复同点自适应−128差。方案不把这些未生成统计伪装成待复制文件。

语法、stdlib导入与`--help`检查通过，未执行缓存提取；无服务器读取、新积分或验收门变化，67/81组仍HOLD。本轮费用0.487586282CPU秒（含两次本地缺路径检索失败0.021303901秒，未重复计费），累计7470.066206826秒；原600余额-9.541441997秒，含授权5秒余额-4.541441997秒。全部旧失败/恢复费保留，无重置。

[最小清单](../paper_artifacts/historical_positive_risk_uppervariance_axis_minimal_cache_plan_v1/cache_inventory.json)、[提取步骤](../paper_artifacts/historical_positive_risk_uppervariance_axis_minimal_cache_plan_v1/extraction_plan.json)、[提取器](../performance/extract_cached_origin4_intervals.py)、[账链](../paper_artifacts/historical_positive_risk_uppervariance_axis_minimal_cache_plan_v1/completion.json)。

## 10月15日提取器本地逻辑验证

继续按前次v4/GPT6区分软件验证与积分精度。4项本地样例验证通过：半开边界、current区间/全局闭合、Gauss非目标桶质量与空桶、eta计数和权重汇总、缺片失败计时。补明current固定取128与Gauss/R2048实际自适应停止阶数的区别，并增加仅对current生效的逐区间质量核对；Gauss不按区间长度归一。

首次验证1项失败、1项错误及其恢复费用全部保留；修复后4/4通过。局限：样例统计值仅测试软件，没有执行真实缺失缓存提取，也未生成匹配区间积分或自适应−128差。67/81组及原HOLD不变。

本轮费用0.385014262CPU秒，累计7470.451221088秒；原600余额-9.926456259秒，含授权5秒余额-4.926456259秒。旧失败/恢复费保留，无重置、新积分或服务器读取。

[验证摘要](../paper_artifacts/historical_positive_risk_uppervariance_axis_extractor_validation_v1/validation_summary.json)、[初次结果](../paper_artifacts/historical_positive_risk_uppervariance_axis_extractor_validation_v1/validation_before.json)、[修复后结果](../paper_artifacts/historical_positive_risk_uppervariance_axis_extractor_validation_v1/validation_after.json)、[费用账链](../paper_artifacts/historical_positive_risk_uppervariance_axis_extractor_validation_v1/completion.json)。

## 10月15日一次性最小提取预算提案

继续参考前次v4/GPT6，保持失败费用与有限规则边界。待授权提案：追加16CPU秒，覆盖本轮结算后既有欠账6.075128422秒，以及单次提取预算9.20秒（客户端7.45、源端读取1.25、本地整理结算0.50），取整余地0.724871578秒。这是按现有实测的保守估价，不是硬费用上界；修复后0.219秒仅为启动检查，不能替代旧完整客户端7.44秒参照。

已准备一次内嵌提取器和112条必要元数据的payload，无需sync或写源端plan；只汇总4缓存，数组不拉取，不调用积分核。仅允许一次顶层调用，不自动重试；完整客户端/SSH子进程与远端绝对CPU分别结算，失败日志/费用保留。payload仅语法检查，未执行。

本轮本地规划费1.148672163CPU秒，累计7471.599893251秒；原600余额-11.075128422秒，含既有授权5秒余额-6.075128422秒。新增16秒尚未授权或入账，无重置、借额、新积分或服务器读取；67/81组与原门不变。

[预算提案](../paper_artifacts/historical_positive_risk_uppervariance_axis_one_shot_budget_v1/budget_proposal.json)、[计价依据](../paper_artifacts/historical_positive_risk_uppervariance_axis_one_shot_budget_v1/pricing_evidence.json)、[待执行命令](../paper_artifacts/historical_positive_risk_uppervariance_axis_one_shot_budget_v1/pending_execution_command.txt)、[账链](../paper_artifacts/historical_positive_risk_uppervariance_axis_one_shot_budget_v1/completion.json)。

## 10月15日一次性仅读完成，数值支线收束

用户授权追加16CPU秒并做一次来源4四缓存仅读；已完成，返回4453字节JSON，无新积分或重试。current112节点、高阶Gauss2048节点及权重/总贡献闭合；Gauss与R2048实际均在32阶停止，current固定128。Gauss桶质量不同于current区间质量，不能充当匹配区间积分或区间误差；同点自适应−128差未被生成。

用户提醒主任务是超高维优化。eta是风险计值中的对数方差隐变量，此前工作属于单一query63/fixed355的数值诊断，近几轮未推进高维查询选择或67/81组闭合。该支线收束；后续回到优化决策与主流程证据，不继续围绕eta扩展计算。

完整客户端实测0.57CPU秒、源端读取0.085104262秒；本轮全额费用0.936555070秒，累计7472.536448321秒。原600账余额-12.011683492秒继续保留；此前5秒加本次16秒后，总授权621秒、可用余量8.988316508秒。全部旧失败/恢复费保留，无重置；G及原HOLD不变。

[仅读结果](../paper_artifacts/historical_positive_risk_uppervariance_axis_origin4_cache_extract_v1/extraction_result.json)、[闭合核对](../paper_artifacts/historical_positive_risk_uppervariance_axis_origin4_cache_extract_v1/verification.json)、[结算](../paper_artifacts/historical_positive_risk_uppervariance_axis_origin4_cache_extract_v1/completion.json)。

## 10月15日回到超高维优化主线

本地核对发现独立rawKG小实验已完成144诊断及48部署，四臂实际收益全tie；冻结8策略库8/12无可行、4/12仅1可行且均达子库oracle。source12query、GP5query都只改过1项最终策略，均未增益。优先瓶颈为候选覆盖和高维生成器/新raw后端的同协议连接，非继续eta细化。README过时入口已修正，原实验文件保留。

[主线核对与下一步](gpt_mainline_optimization_review_20261015.md)。本轮费用0.781757553CPU秒，累计7473.318205874秒；授权621秒余量8.206558955秒，旧失败/恢复费全保留，无重置、新积分或服务器读取。67/81组原门仍HOLD。

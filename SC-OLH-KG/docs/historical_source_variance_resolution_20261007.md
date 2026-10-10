# 历史来源方差精度修复

32/32组响应—风险来源案例通过登记的数值门槛，16/16完整身份接通原付费更新、正权混合终端和KG主循环。原29组未重采样；另两组在固定尾部加密后通过，最后一组复用已保存状态和历史条件积分。[来源与完整重放结果](../paper_artifacts/historical_source_variance_resolution_v4/batch_results.json)。

原三组最不稳定位置约1%的状态贡献28%–49%的方差。固定正函数h加密这些状态，以1/h权重还原原后验；原先验、似然和15%方差半段差门槛保持一致。校正权重已经传入条件新任务投影、实际付费更新、幻想和终端计算。两组直接通过，最后一组先以联合Gaussian风险方向更新改善混合，再用一维积分处理原来失败的第8处。

该联合方向在旧±12参考坐标之外有实际正向长尾，积分使用±32，实际最大边界相对log密度低于−64，quadrature误差界小于1.8e−9。第8处方差半段差26.35%→1.24%；其他位置继续使用原全链估计。最后一组最大半段差3.1%、最低ESS约685；保留混合的最大方差误差6.2%。全链和积分是同一后验的不同数值估计，未改变目标混合组件。[最终功能量精度接续](../performance/finalize_historical_source_variance.py)、[原失败与积分结果](../paper_artifacts/historical_source_variance_resolution_v3/source_results.json)。

全部12个balanced4缓存动作与64个原始来源协方差score完成接续，25项相关测试通过。原score和M-step入口继续复用，62/64头的两条幅度score方向在保留混合半段中一致；这不是超参数收敛。修复累计约191单核CPU秒，完整链路缓存重放约34秒；无服务器任务或新增标签、仿真、验证调用。重放已完成结果不触发重采样：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_source_variance_resolution
```

## 已知缺口与下一步

每身份一个幻想的粗KG仍产生较大基准漂移。布局5候选7的净值0.0382中漂移0.0380，布局6候选4的净值0.0902中漂移0.0732。复用历史精度细化门槛，以固定4/8幻想和128/512终端积分检验这两个已打开开发查询；保留全部HOLD。[精度结果](../paper_artifacts/historical_mixture_lookahead_precision_v1/results.json)。更多幻想暴露的实际批量积分缩放失败已保存为小型fixture，改用每组件峰值缩放并保留1e−8误差门槛。

非Gaussian参考已接通真实正权混合。先验直接重要性权重实际退化到ESS≈1，原失败保留；随后复用历史条件输运、后验η提议和组件证据校正，两布局最低ESS约3230/3425，最大log证据误差0.00951/0.00255，通过原参考门槛。布局6未投影参考漂移0.0476，投影与参考差−0.00722；较大漂移不是主要由矩投影造成。[有效参考](../paper_artifacts/historical_positive_likelihood_reference_v2/results.json)。

历史查询均值解析边缘化也已适配到正权混合，将幻想坐标8→6，并按查询η均值排列组件。相同64幻想与512终端积分下，布局5净值−0.00109、漂移−0.00161，布局6净值0.07455、漂移0.02423；身份权重鞅残差0.0141/0.0237，仍超过原0.01门槛。两查询继续HOLD。包括实际失败fixture、历史输运等价性、共享风险和新预测矩在内的30项测试通过。[阶段汇总](../paper_artifacts/historical_source_variance_resolution_v4/phase_results.json)。

两维CDF已接通：历史条件Gaussian CDF同时边缘化来源组件、η、均值与噪声。128阶实际失败保留；256阶的32/32响应头均通过自适应积分对照，最大误差5.41e−10。先只算预测密度，复现原始更新的身份权重；共同16幻想／身份的残差0.00376／0.00460均通过1%门槛，再做完整KG。[CDF](../paper_artifacts/historical_predictive_cdf_v2/results.json)、[密度预检](../paper_artifacts/historical_predictive_martingale_v2/results.json)。

完整联合积分仍保留布局6的HOLD：净采样效用0.04239含漂移−0.01272。未投影参考也漂移−0.01286，投影差仅0.00178，主要问题仍是积分。[联合KG](../paper_artifacts/historical_mixture_lookahead_precision_v4/results.json)、[未投影参考](../paper_artifacts/historical_positive_likelihood_reference_v3/results.json)。

已复用终端分支因子化：固定动作满足p(y)w(h|y)=w(h)p_f(y_f|h)p_g(y_g|h)，分别积分每身份的两头统计量后相乘；动态最大值仍联合计算。布局5用16、布局6用32节点／头，未投影漂移−0.000240／0.000670、原512终端估计的漂移−0.000222／−0.002675均通过原0.005门槛。最低ESS约2477／3497，log证据误差均低于0.0032。相同16分辨率的两查询验证134→约39单核秒；包括布局6加密，整批77秒。32项测试通过。[因子化结果](../paper_artifacts/historical_factorized_tower_v1/results.json)。

固定动作控制变量已接入完整12个登记动作：初次10/12通过，两个自适应差值加密到32幻想／身份后通过，原HOLD保留。布局5当前最高为候选5，净效用0.05876；布局6为候选4，净效用0.05244。共享身份更新和动态最大值仍联合计算；这是已打开开发状态上的数值比较。[完整比较与加密](../paper_artifacts/historical_control_variate_action_precision_v2/results.json)。合并四条幻想的原始积分与逐条付费滤波在全部32头上等价，单独滤波计时约提速1.9倍。

历史raw-score／MCEM M-step已复用：64/64个响应及风险头的共同／任务协方差幅度更新改善完整与两半Q目标。来源内证据比24/32通过；重新计算新协方差下的条件响应后验后，后验功能量7/32通过，合并证据比门槛后5/32通过，仅1/16个完整身份可用。候选参数单独保存，未用于上述动作比较。[来源学习](../paper_artifacts/historical_raw_source_amplitude_learning_v1/results.json)、[新后验精度](../paper_artifacts/historical_raw_source_amplitude_learning_v1/posterior_precision.json)。

同一批保留样本对照，旧先验18/32、新先验7/32通过；12组旧通过而新失败，13组两者都失败。新后验25组方差半段差超标，1组另有ESS不足，无Rhat超标。这批短保留样本不足以认证新后验方差；不能把旧完整链32/32的PASS作废，也不能直接把旧后验挂到新协方差上。等权归一化量的常数诊断已修正，逐组结论保持一致。[配对诊断](../paper_artifacts/historical_raw_source_amplitude_learning_v1/retained_baseline_precision.json)。

物理因子恢复、自由高维优化和独立稳定部署收益仍缺各自证据；旧V51完整HVD/KG与后续前端×后端60/60继续按原合同保留。

上一批完整动作比较及来源学习约947单核CPU秒，35项测试通过；原32来源案例未重采样，无新增标签、仿真、验证调用或服务器任务。[上一阶段汇总](../paper_artifacts/historical_control_variate_action_comparison_v1/phase_results.json)。该历史批次缓存重放约0.04 CPU秒：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_control_variate_learning`。

## 新参数候选的来源条件化接续（2026-10-08）

第一次MCEM协方差更新产生的32组固定候选现已全部通过来源后验及来源内证据比精度，形成16/16完整身份。27组单位置积分把合并通过数5→16，联合梯度方向仅16→17；两个实际失败批次保留。随后复用历史Stage203/206采样器，对剩余15组新参数候选重新条件化，约75单核秒达到28/32。四组固定方差／旧新密度桥加密，再对两组保持同种子、同采样核延长前缀，达到31/32。一组后验与证据比分别采用同一固定候选的已通过估计；没有利用目标表现选参数。

最后失败的是与旧版本相同的布局5/additive/head_specific/observed_task/f、位置8。其完整链Rhat、ESS和其他47处方差均通过，位置8半段差24.3%。完整协方差条件积分触及150秒上限，结果保留；合并为一次只计算该位置的积分后，81单核秒完成，半段差降到0.34%，Rhat≈1.000、ESS1024，最大边界相对log密度−69.9、误差界1.71e−11。保留1024组件的最大方差误差7.0%、均值误差0.0042个后验标准差，均通过原门槛。来源参数、组件、权重、完整均值及其余47处估计不变。[最后修复](../paper_artifacts/historical_changed_candidate_functional_resolution_v4/results.json)。

27组进行了保留混合对认证后验矩的比较；另5组直接在已保留状态上通过后验功能量门槛，没有再缩减状态，未伪造额外的完整链比较。41项相关测试通过，包括合并单位置积分与完整桥积分的等价性。此次记录的接续计算约2215单核CPU秒，新增候选MCMC共387072 sweeps（包括同种子重放的前缀）。原32组已通过来源法则及其链未重采样，无新来源标签、目标仿真、验证调用或服务器任务。历史采样器比为所有残差积分更适合批量处理改变后的参数；条件积分留给少数实际尾部功能量。

这关闭了**一次固定来源MCEM更新后的数值条件化缺口**。仍需以这些新后验继续来源学习、检验收敛，并补齐跨身份可比较的来源证据；当前证据比只认证每个身份内部Z(new)/Z(old)。上述12动作仍使用原来源法则，候选未进入目标评价。随后接入历史同合同完整GP控制及独立决策验证。[当前32组汇总及组件入口](../paper_artifacts/historical_changed_candidate_functional_resolution_v4/phase_results.json)。完整缓存重放约0.21 CPU秒，不触发采样：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_changed_source_resolution
```

## 第二轮来源学习（2026-10-08）

已从认证的第一轮候选后验继续MCEM，复用历史`expected_objective`、白化后的来源score和有界L-BFGS-B。64/64个头再次改善完整及两半Q目标，64组当前参数score恒等式通过，优化约5.8单核秒。f、g来源协方差的中位变化仅0.18%、0.35%；η_f、η_g群体协方差仍变化27.6%、27.4%。26/64个头达到登记的单轮稳定门槛，其中风险头为0/32；收敛要求连续两轮满足协方差变化及来源score门槛。当前主要瓶颈是风险群体协方差学习，不能将M-step优化器收敛写成整个来源学习已收敛。[更新结果](../paper_artifacts/historical_raw_source_amplitude_learning_v2/results.json)。

缓存重新加权先有18/32组通过；仅对另14组复用历史精确来源采样器，约71单核秒。三组仍失败的响应方差用保存的1024状态做局部条件积分：位置8半段差47.1%→0.31%，两组位置42分别降到0.17%、1.98%；其中一组位置4仍18.0%，再沿位置4积分降到2.62%，保留已通过的位置42。组件、权重、均值及其他全链坐标不变，三个保留混合误差分别7.8%、13.7%、10.3%，均通过原15%门槛。[局部积分](../paper_artifacts/historical_source_learning_round2_completion_v1/results.json)、[剩余位置4](../paper_artifacts/historical_source_learning_round2_completion_v2/results.json)。

相邻两轮证据比正向29/32通过；从14组保存的新后验另算反向Z(round2)/Z(round1)，13/14通过，补上三个正向失败估计。6/additive/joint/deconvolved_task/g的反向半段差0.1058仍HOLD，其同参数正向估计通过；两者均保留。这里不使用Z(round2)/Z(original)冒充相邻比，也不依据目标收益挑选估计。四个第一轮加密候选文件原来只标记f幅度，已依据原登记输入补齐身份与η幅度，所有后验数值未变。[元数据修复](../paper_artifacts/historical_raw_source_amplitude_learning_v2/metadata_repair.json)。

第二轮现有32/32组认证的新法则、16/16完整身份；18组直接认证保留状态，14组另有保留混合比较。44项相关测试通过，新增检查包括局部积分与完整条件矩等价、等权归一化量的常数诊断、相邻反向证据比的Gaussian归一化。登记主要计算约167.4单核CPU秒，新增候选采样258048 sweeps，原32组来源链未重跑，没有新来源标签、目标仿真、验证调用或服务器工作。[当前汇总与组件入口](../paper_artifacts/historical_raw_source_amplitude_learning_v2/phase_results.json)。缓存重放约0.09 CPU秒：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_source_learning_round2
```

上述第二轮混合已接续到以下第三、四轮。跨身份可比来源证据、同合同完整GP对照和独立决策验证仍待完成，候选尚未进入旧12动作比较或独立目标评价。

## 第三、四轮来源学习（2026-10-08）

两轮均完成32/32来源后验及相邻证据比认证、16/16完整身份。第三轮58个头改善完整／两半Q，另6个响应头达到连续两轮稳定却未通过微小更新的两半Q，保留原参数；第四轮同样保留12个稳定响应头、接受52个改善更新。两轮的32个响应—风险联合Q均改善，失败提案全部保留。[第三轮](../paper_artifacts/historical_stationary_source_continuation_v1/phase_results.json)、[第四轮](../paper_artifacts/historical_stationary_source_continuation_v2/phase_results.json)。

第四轮全部32个响应协方差头达到连续两轮稳定。风险协方差中位变化从第二轮27.6%／27.4%，降到第三轮16.2%／14.8%、第四轮9.1%／8.7%，风险头仍0/32稳定。复用第二轮样本的224个风险幅度探针约0.8单核秒：缩至75%时30组精度通过、29组改善超过条件MC误差区间；缩至25%或零时全部失去重要性重叠，不能据此选零协方差。[探针](../paper_artifacts/historical_source_risk_plateau_v1/results.json)。

补采样由第二轮14组降至8组、5组。第三轮只积分位置44；第四轮位置8积分通过，但原1024组件低估认证方差30.1%。复用历史单位置h加密及1/h校正后，位置8误差11.4%；新出现的位置16差异再用保存的未加密状态积分处理，半段差0.49%。不再采样，保留位置8及其他46处通过估计。[最终混合与参考](../paper_artifacts/historical_source_retained_tail_completion_v1/results.json)。两轮共执行249856候选MCMC sweeps；原32来源链未重跑，无新标签、目标仿真、验证或服务器任务。10项相关测试通过，缓存重放约0.10 CPU秒：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_source_learning_round4
```

### 已知缺口与下一步

风险来源学习尚未收敛，跨身份来源证据及独立决策收益仍开放。下一步从第四轮认证混合继续风险幅度迭代，按原score／协方差及后验门槛停机，复用稳定响应参数与通过功能量；随后接来源证据、同合同完整GP对照和独立目标验证。两轮已记录CPU合计约167秒；一次尾部采样在保存完整组件后写结果元数据报错，CPU未记录，已从保存组件恢复且未重复采样，因此167秒只是已记录部分。[记录](../paper_artifacts/historical_source_retained_tail_v1/execution_incident.json)。保留参数对应的梯度元数据也已校正，科学参数、组件、权重及结论不变。

## 第五、六轮来源学习与score精度（2026-10-08）

两轮均完成32/32来源后验及相邻证据比认证、16/16完整身份。第五轮49个头改善完整／两半Q，另15个稳定响应头保留；第六轮48个改善、16个保留，两轮32个响应—风险联合Q均改善。仅1组、6组重新条件化。第六轮位置8的条件积分通过，但新保留混合方差误差25.4%；复用从第四轮尾部混合按第五、六轮参数重新加权的当前法则，误差12.1%通过原15%门槛，未追加采样。认证入口已指向实际通过的加权组件，失败的新混合保留。[第五轮](../paper_artifacts/historical_source_learning_round5/phase_results.json)、[第六轮](../paper_artifacts/historical_source_learning_round6/phase_results.json)、[同参数混合复用](../paper_artifacts/historical_source_learning_round6_mixture_reuse_v1/results.json)。响应32头继续稳定；风险协方差中位变化降至6.65%／6.06%、4.93%／3.97%，第六轮仅1/32风险头单轮稳定、0/32连续两轮稳定。

复用历史精确raw-likelihood梯度／Hessian与Gaussian Stein恒等式检查全部64条风险score方向。当前矩估计的中位MC标准误0.00112，已接近原0.001停机阈值；Stein标准误中位0.00628、逐坐标标准误比中位5.13，未改善精度，不采用。两估计的配对95%数值误差区间兼容60/64坐标，余4条差异保留。[诊断](../paper_artifacts/historical_source_risk_score_precision_v1/results.json)。

对布局6/spectral/joint/deconvolved_task/g做固定参数、固定种子的风险来源统计量小试验：4链各保留16384状态，前4096步与保存链完全一致，仅保存η均值、协方差与score摘要，不逐状态计算响应协方差。约9.27单核秒，两条score的95%MC半宽为0.000444／0.000674，Rhat约1.0004／1.0001、ESS约10300／8529；充分统计量所得score与逐状态平均误差1.9e−15。KG仍使用1024组件。[小试验](../paper_artifacts/historical_source_risk_score_bank_v1/results.json)。本阶段登记主要计算约94单核CPU秒、执行196608候选MCMC sweeps，11项相关测试通过；原32组来源链未重采样，无新标签、目标仿真、验证或服务器任务。[汇总](../paper_artifacts/historical_source_learning_rounds5_6_v1/phase_results.json)。缓存重放不触发采样：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_source_rounds5_6
```

### 已知缺口与下一步

风险来源学习尚未收敛。小试验只认证一组score精度，其score与误差区间尚未全部落入停机范围，不能推广为32组通过或风险结构恢复。下一步只对需要更高score精度的来源案例延长风险充分统计量，保持KG组件规模，再按原Q、后验及连续两轮稳定门槛继续来源学习；之后补跨身份可比证据、同合同完整GP对照与独立决策收益。当前候选仍未进入旧12动作比较或独立目标评价。

## 风险score完整批次与第七轮（2026-10-08）

复用已通过的小试验，其余31组固定第六轮参数，4链各保留16384步；全部32组风险score精度通过，95%MC半宽最大0.000984。仅保存风险充分统计量，来源混合保持原规模：30组1024组件、2组历史512组件。第七轮直接使用同参数的完整／两半统计量：49个头改善Q、15个稳定响应头保留，32个联合Q均改善；只重新条件化3组，即完成32/32后验认证、16/16完整身份。[阶段汇总](../paper_artifacts/historical_source_risk_score_banks_v1/phase_results.json)。风险协方差中位变化4.78%／2.84%，0/32风险头连续两轮稳定。

局部Louis曲率诊断中，缺失信息比例中位0.675，27/64方向的观测对角曲率不足Q曲率四分之一。固定1.5倍第六到第七轮风险更新的探针，31/32改善完整／两半Q，32/32来源证据比精度通过，23/32另有超过95%数值误差区间的改善。[加速探针](../paper_artifacts/historical_source_risk_acceleration_probe_v1/results.json)。13项相关测试通过，包括非零风险先验下的充分统计量及Gaussian边缘似然的Louis曲率恒等式；登记主要计算约306单核CPU秒，无新标签、目标仿真、验证或服务器任务，原32来源链未重采样。缓存重放：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_source_risk_learning`。

### 已知缺口与下一步

风险来源学习仍未收敛。曲率只诊断局部对角项；加速探针以第六轮及其固定响应参数为参照，尚未证明优于第七轮，也未认证新后验，因此没有替换参数。下一步比较这23组候选与认证第七轮，只对仍有来源改善的候选做后验及保留混合认证；保留Q及证据失败结果。第六轮统计量不能冒充第七轮后验。来源收敛之后仍需跨身份可比证据、同合同完整GP和独立决策比较。

## 加速候选与第七轮的比较及接入（2026-10-08）

23组固定候选全部完成实际第七轮对照。证据比同时包含响应协方差回到登记第六轮参数及风险先验变化；20组改善超过95%数值误差区间，且重新加权后的后验功能量全部通过。其余3组保留原结果，不选入。20组实际组件文件与12组第七轮法则接成32/32认证案例、16/16完整身份；32组均已通过现有来源学习入口加载。[比较及接入汇总](../paper_artifacts/historical_source_acceleration_comparison_v1/phase_results.json)。

14项相关测试通过，新增测试以独立Gaussian密度核对同时改变两项协方差的完整来源比。来源比较约4.58单核CPU秒，接入重放约0.55秒，无新增采样、标签、目标仿真、验证或服务器任务。原先“全部1024”的描述已纠正为30组1024、2组512，保持各自历史规模。缓存重放：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_source_acceleration`。

### 已知缺口与下一步

这是第七轮的认证替代分支，未增加收敛轮数，也未进入独立目标评价。风险来源学习仍需收敛。下一步从这32组实际法则继续来源学习，先验证在内存中复用完整风险链进行多次有精度约束的更新，以减少每轮重新采样的成本；随后完成来源证据、同合同GP对照及独立决策收益。第六轮统计量仍仅用于其对应Q参考。

## 内存风险链复用与完整批次（2026-10-08）

32组已完整执行：561次有效条件风险MCEM更新用了44批链，约550单核CPU秒；每批平均支持12.75次更新。这衡量采样批次节省，不是端到端实测加速。固定响应协方差，按实际当前法则重新加权；保留原完整／两半Q、score精度和稳定阈值，更新另要求来源证据改善越过95%数值误差区间。完整风险状态仅留内存。

15组通过连续稳定。历史响应头已有稳定参数保留规则；另外5组在两个不同实际当前法则上连续通过稳定诊断后，精确保留最后接受的风险参数，未接受失败的下一提案、未增加更新次数。条件风险稳定合计20/32；剩余12组为Q半段分歧8组、证据改善未分辨3组、重新加权精度不足1组，其中7组仅有单轮稳定诊断。[完整结果与重放入口](../paper_artifacts/historical_source_risk_state_reuse_batch_v1/phase_results.json)。

小试验最终风险参数为[0.136489,0.024684]。直接重用旧1024组件失败（ESS19.7、响应方差半段差32.7%），结果保留；对这一个固定最终候选复用历史采样器条件化3.54秒，半段差2.63%、保留混合方差误差2.08%，均通过。响应协方差的单轮稳定检查也通过；最终／父法则证据总比值仍未认证。[最终响应认证](../paper_artifacts/historical_source_risk_state_reuse_conditioning_v1/results.json)。

17项相关测试通过，包括新风险权重下的解析Gaussian响应score及稳定保留不接受失败提案。抽出的历史采样函数精确复现保存前缀，最大差异0。原32来源法则及当前32组认证入口未替换，无新增标签、目标仿真、验证调用或服务器任务。缓存重放约0.01 CPU秒，不采样：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_source_risk_reuse
```

下一步只处理上述12组的实际停机原因，并认证已稳定候选的最终响应后验、保留混合与来源证据；随后检验联合来源收敛、跨身份证据及同合同独立决策收益。当前结果支持原meta-prior来源学习的数值推进，尚未支持独立决策优势。

## 剩余风险停机与最终候选认证（2026-10-08）

12组停机案例完整处理：沿原种子延长原采样前缀，5组保留局部稳定法则、7组固定提案通过；接续这7组后，再复用已有稳定诊断与最后一组的固定提案加密结果，严格连续条件风险稳定达到27/32，另5组完成同一法则的前缀加密复核。未接受失败提案或增加保留参数的更新次数。全部32组最终响应后验与混合认证通过，响应协方差均通过单轮稳定诊断。[阶段结果](../paper_artifacts/historical_final_source_closure_v1/phase_results.json)。

最终／父法则证据直接估计27组通过；剩余5组沿实际采样锚点分段后全部通过。分段区间分配总数值误差水平，不假设共享链的误差独立。最终31组改善、1组原法则保持且比值为1，16/16身份完整。原先布局6/additive/head_specific/observed_task/f的负值来自ESS约11的失效估计；分段log证据比0.3488，95%条件数值区间[0.2011,0.4965]，原失败记录保留。[分段结果](../paper_artifacts/historical_final_risk_evidence_bridge_v1/results.json)。

实际32组已通过现有来源混合读取接口，保持30组1024、2组512组件；最终候选与精度入口分别为`historical_final_source_closure_v1/source_candidates.json`、`source_precision.json`。19项相关测试通过，缓存重放约0.83单核CPU秒。已记录本阶段计算约922单核CPU秒；原始32组预学习法则未重采样，无新标签、目标仿真、验证或服务器任务。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_final_source_closure
```

### 已知缺口与下一步

5组同参数下的重叠前缀复核曾误计为第二轮稳定；这一计数已纠正，参数、采样结果及后验／证据认证不变。条件风险稳定与响应单轮稳定尚未构成联合来源收敛；新入口将联合稳定轮数从零开始。来源证据比仅在同一身份内成立，跨身份绝对证据和同合同完整GP／独立决策收益仍开放。一次分段执行因列表参数类型错误在保存有效结果前停止，修正后按原登记种子重放；失败执行CPU未记录，因此922秒是已记录下界。下一步从这些实际最终候选完成联合来源确认，再补跨身份证据及同合同目标比较。

## 实际历史轨迹上的联合来源确认（2026-10-08）

32组当前法则均通过响应与风险的单轮稳定诊断。复用历史实际风险更新，把最终认证响应混合按精确风险先验比重新加权到紧邻的前一个已接受法则，再检验该法则的响应后验及稳定性；27组在两个相邻实际法则上同时通过两类检查。只对8组最终风险统计量缺精度的案例重放最近原链，前缀score差异均小于1e−9，并保存紧凑充分统计量；另两组复用已有实际当前法则诊断。未制造新的联合优化轮次。[完整确认](../paper_artifacts/historical_joint_source_stationarity_v1/phase_results.json)。

剩余5组中，布局6/spectral/joint/deconvolved_task/g的固定风险提案此前已通过完整／两半Q及来源证据门槛。按原种子和加密长度复核后，新响应后验与两个实际法则的联合稳定均通过，现接受该提案；相对选择父法则的log证据比0.000980，95%条件数值区间[0.000136,0.001824]。联合连续确认达到28/32、12/16完整身份；全部32组后验及来源证据继续通过。新来源入口为`historical_pending_joint_update_v1/source_candidates.json`和`source_precision.json`，组件规模不变。[实际接受结果](../paper_artifacts/historical_pending_joint_update_v1/results.json)。

21项相关测试通过；已记录约130单核CPU秒，缓存重放约0.5秒，不采样。原始预学习法则不变，无新标签、目标仿真、验证或服务器任务。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_joint_source_confirmation
```

### 已知缺口与下一步

4组当前联合单轮检查合格，但尚缺第二个合格实际法则：5/spectral/joint/deconvolved_task/g、5/additive/head_specific/observed_task/g、5/additive/joint/deconvolved_task/g、6/spectral/head_specific/deconvolved_task/f。相应微小提案的证据区间仍跨零，两组还存在Q半段非正，保持HOLD；不能宣称全部联合收敛。这些检查只认证登记的局部score与协方差阈值。跨身份绝对证据、同合同完整GP对照和独立KG决策收益仍开放。一次固定更新执行因遗漏现有证据阈值字段停止，补齐原阈值后按原种子重放；失败CPU未记录，130秒为已记录下界。下一步复用已保存统计量处理这4组实际缺口，再补跨身份证据与同合同目标比较。

## 剩余实际法则确认完成（2026-10-08）

按原种子及已登记32768长度复核4组真实前一步，3组通过，达到31/32。另一组前一步协方差诊断仍为1.0616%，超过原1%门槛，失败记录保留；只对其原固定提案登记一次65536前缀加密，完整／两半Q均正，增量log证据比0.000201，95%条件数值区间[0.0000145,0.000388]。新响应后验与两个实际法则的联合诊断通过，现接受该提案。联合局部数值收敛达到32/32、16/16完整身份；未计入新的联合优化轮次，也未降低阈值或换种子。第6→7轮的历史接受合同保留原M-step／证据精度规则，未把其负的证据比点估计改写成证据改善。[实际结果与入口](../paper_artifacts/historical_last_joint_precision_v1/phase_results.json)。

来源入口改为`historical_last_joint_precision_v1/source_candidates.json`和`source_precision.json`，32组加载通过，保留30×1024和2×512组件。23项相关测试通过，本阶段约113单核CPU秒，保存5份紧凑统计量，无新标签、目标仿真、验证或服务器任务。仅缓存重放：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_joint_source_completion`。

下一步完成跨结构的来源绝对证据比较，然后把这些实际学习先验接回同合同KG与完整GP对照。当前认证是历史轨迹的局部数值收敛，尚不支持全局最优、真实物理因子恢复或独立决策优势。

## 来源结构证据完成（2026-10-08）

32/32原始来源归一化常数及16/16完整身份现均通过原ESS、Rhat和半段差异门槛。布局5、6各用48批来源数据，其中32批相同、16批不同，故分别归一化各自8种结构。直接跨结构重加权仅1/28项通过；改用局部高斯积分参考后，又发现参考拟合与桥估计重用同一后验会产生数个log单位的偏差，早先认证已撤回并保留原诊断。最终冻结高斯参考、另用历史精确核在固定当前法则下估计，仅剩一项延长原8192前缀到32768后通过，标量密度比前缀误差为0。来源参数与部署组件文件不变。[最终来源证据](../paper_artifacts/historical_last_source_normalizer_v1/results.json)。

完整结构权重及同时95%条件数值误差范围已保存于`historical_last_source_normalizer_v1/source_structure_weights.json`。几乎全部来源质量落在spectral/head-specific模型：布局5的observed/deconvolved权重约0.479/0.521，布局6约0.305/0.695；保留两种处理的质量，无硬选赢家。11项相关测试通过，本轮约4分钟单核CPU，仅保存紧凑标量密度比，未派服务器任务或读取新的目标／验证结果。缓存重放：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.extend_historical_last_source_normalizer`。

下一步用这些冻结来源权重与实际学习协方差重建KG先验，完成同合同GP对照及独立目标检验。当前权重属于来源训练数据上的固定经验Bayes模型比较；数值误差区间不覆盖模型误设、来源自适应拟合或物理因子恢复，也尚未建立独立决策优势。

## 实际学习先验接回KG入口（2026-10-08）

32/32来源后验、16/16完整身份已按最终响应／风险的common/task幅度重建，并在目标更新前载入各布局独立的冻结来源权重。复用历史`restore_head`、正质量组件加载器和原付费滤波，没有重写来源学习或KG算法。保留30×1024、2×512组件及全部8种结构。独立common系数条件化与直接查询投影的最大差异为2.54e−15；scalar／batched原始更新差异为4.45e−15。

两个已打开的付费观测后，共享身份权重正确更新：布局5的deconvolved／observed主结构从0.5205／0.4795变为0.5051／0.4949，布局6从0.6948／0.3052变为0.6963／0.3037。原8候选、付费incumbent、40个终端动作及剩余360／扣费359预算保持同一合同；balanced4分别为[5,4,6,7,0,1]和[6,4,2,7,0,1]。8项相关测试通过，约4.50单核CPU秒，无新标签、仿真／验证或服务器任务。[结果](../paper_artifacts/historical_learned_prior_kg_connection_v1/results.json)，[连接状态](../paper_artifacts/historical_learned_prior_kg_connection_v1/connected_states.json)。缓存重放：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.connect_historical_learned_prior_kg`。

下一步复用历史control-variate／factorized固定终端基线及相同付费幻想更新，在实际新法则下检验12个动作的KG数值精度，再接同合同GP与独立决策评价。当前终端值沿用128-draw粗接口规则；旧原始先验的KG认证不能移用到新分支，当前结果也不构成独立决策优势。

## 新先验KG精度与直接部署积分修复（2026-10-08）

12/12冻结动作完成并通过原绝对数值门槛，全部查询的批量更新与付费滤波一致。原1200秒上限中断最后一项，保留中断记录；只用75秒补完其原16幻想规则，复用其余11项。11个名义正收益中9个的自适应收益不抵预算成本，正值主要来自固定终端基线漂移。两布局的停止效用从512到16384抽样分别改变0.00146／0.00272，部署动作不变，采样排名仍HOLD。

复用历史`legacy_joint_utility`的解析quality和Gaussian均值积分，保持当前16节点验证SD规则，完成64个直接部署动作的全来源组件积分。布局6/query1原名义净值+0.00390；相同固定动作的漂移由+0.00393降为16幻想−0.000076、32幻想−0.000007，前缀误差3.45e−15。直接终端积分误差低于1e−10；32／16幻想差仍为0.000069。20项相关测试通过，约24分钟单核CPU，无服务器、新标签、目标仿真／验证。[汇总](../paper_artifacts/historical_learned_prior_kg_completion_v1/phase_results.json)，[分解](../paper_artifacts/historical_learned_prior_kg_completion_v1/decision_diagnostics.json)。缓存重放：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_learned_prior_kg`。

下一步把解析直接部署效用接入采集，处理guarded终端精度，再接同合同GP与独立决策检验。当前固定动作复核没有重算完整自适应KG；后验moment投影偏差与剩余幻想积分误差尚未完全分离。来源先验及其收敛／证据认证保持不变，本轮结果不构成独立决策优势。

## 解析效用接入实际KG及guarded条件积分（2026-10-08）

`ConditionalTerminalActions`已用于实际共享身份更新后的KG最大化。direct复用解析质量及η积分；guarded解析边缘质量、条件化目标均值和最后一个约束均值，保留跨策略协方差、配对比较及原16节点验证SD规则。16组嵌套终端核对完成，64个direct参考最大差异0；最后一组因CPU上限中断后用33秒补完，原15组及中断记录保留。布局5最优guarded效用16384→65536只变6.8e−7，布局6direct停止效用不再随Sobol分辨率变化。两布局部署动作不变。

布局6/query1实际自适应KG试点完成：40动作、全部8身份、原360／359预算及付费滤波保持一致，4幻想／身份结合已缓存32幻想固定动作参考。旧净值+0.003899变为−0.0000359，自适应对比为0；原始联合基线漂移−0.003016由历史独立固定动作控制变量消去，但付费投影及参考残差−7.1e−6仍保留。5项相关测试通过；本轮约7.2分钟单核CPU，无服务器任务或新标签。[汇总](../paper_artifacts/historical_conditional_kg_pilot_v1/phase_results.json)。只读缓存重放：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_conditional_kg`。

尚未解决：其他guarded动作16384→65536最大变化约6.4e−5／4.9e−4，高于约2.9e−5的单次采样预算成本；单查询4幻想试点不能认证完整采样排名。新终端约6秒／未来状态，下一步先复用组件条件矩阵、批量计算解析证书，保持参考精度并减少重复计算，再扩大冻结KG复核及同合同GP／独立决策检验。

## 终端效率及16幻想扩展（2026-10-08）

同一source组件的条件矩阵只分解一次。第一版批量自适应证书在4096点没有提速，保留其负结果；最终复用历史Gaussian η积分16／32／64／128阶，相邻规则差≤1e−11才接受，否则继续原自适应积分。两布局、两预算、两分辨率的全部40动作与原参考最大效用差5.6e−17；两份实际付费幻想的逐身份标量参考最大差1.1e−15，其中2个证书实际触发自适应回退。4096点总耗时22.46→9.04秒，65536点112.51→51.14秒，原精度及来源法则保持不变。

原布局6/query1从每身份4扩展到16幻想，复用旧32次终端计算，仅新增96次；128个预测状态中自适应收益仍为0，控制变量净值仍−0.0000359。原始联合漂移−0.003016→−0.0000769，身份权重鞅残差0.00175→0.0000387；32幻想固定动作参考的剩余误差未解决。10项不同相关测试通过，保留第一版及全部负结果，无服务器、新标签或验证调用。[阶段汇总](../paper_artifacts/historical_conditional_kg_extension_v1/phase_results.json)。只读缓存重放：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_conditional_efficiency`。

下一步优先处理原有自适应信号的布局5/query5：将历史固定动作head因子分解接入修正后的guarded效用，再接同合同GP对照。布局6负结果保持；当前仍没有完整采样排名认证或独立决策优势。η相邻规则差属于数值诊断，不是严格误差界。

## Guarded固定参考与联合KG接续（2026-10-08）

历史固定动作head分解已接入修正guarded效用，保持共享身份、原付费滤波、全部8身份及40终端动作。联合4／8／16幻想的自适应收益为0.002426／0.004129／0.003361，16幻想复用原32次计算，仅新增96次。固定参考16／32／64／128／256幻想全部保留；延长四个主要head时复用全部原前缀。其余6身份沿用32幻想积分，权重不变，效用贡献误差界为7.43e−7。预测CDF分块降低峰值内存，未改变积分节点或权重。

当前16联合／256固定净值为+0.004053＝自适应0.003361＋固定动作漂移0.000724−预算成本0.0000312。固定128→256差0.000619，联合8→16差0.000768。9项相关测试通过；约800已记录单核CPU秒，无服务器或新标签。[缓存汇总](../paper_artifacts/historical_conditional_guarded_kg_extension_v1/phase_results.json)。原32幻想结果及布局6负结果保留；汇总不重跑未来状态。

局限与下一步：这些前缀差异不是严格误差界，固定动作漂移还未分离原始似然后验与高斯矩投影的差异，不能认证完整KG排名或独立先验优势。下一步先复用历史原始似然参考处理这个差异，再接同合同GP及独立决策检验。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_conditional_guarded_kg
```

## 下一次原始似然与矩投影的分离（2026-10-08）

复用历史posterior-component／η重要性条件化，接入当前guarded均值积分；下一次观测的非Gaussian η后验通过精确密度比保留。原32幻想检查因前缀差0.000457失败，结果保留；沿同一缓存扩展到128后，固定动作投影效用比原始似然参考高0.000838，约为单次预算成本27倍。16384→65536积分差0.0000262，64→128幻想的配对差变化0.0000924，均通过登记的1e−4数值诊断。

四个实际响应头与历史完整联合transport最大差5.73e−5；9项测试通过，复用384项、新增768项计算，约138单核CPU秒。所有来源模型及权重保留，6个低权重身份沿用缓存积分的效用贡献误差界仍为7.43e−7；无新标签或服务器任务。[缓存汇总](../paper_artifacts/historical_conditional_raw_guarded_v2/phase_results.json)。

局限与下一步：当前只分离固定动作的下一步投影差异，前两次付费更新仍保持原投影；数值前缀差不是严格误差界。自适应最大化需要重新计算，不能把0.000838直接从旧KG净值中扣除。历史`likelihood_query_lookahead`已实现完整原始似然决策比较，但限定每head单Gaussian；下一步以本轮核对的条件积分适配实际正质量来源混合，保留40动作，比较原始似然最优动作与投影所选动作的实际模型效用，再接同合同GP／独立检验。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_conditional_raw_guarded
```

## 原始似然自适应决策比较接入（2026-10-08）

历史raw-optimal／projected-choice比较已适配实际正质量来源混合，保留全部8身份、所有组件、40动作和360／359预算。条件化支持查询不在部署名单中及单策略部署，η与均值使用不同Sobol坐标。原32联合幻想均完成4096／16384积分；每个幻想的动作值、选择、权重及诊断已保存，可直接复用。旧投影汇总误差1.6e−16，7项测试通过，约533单核CPU秒，无新标签或服务器任务。[缓存汇总](../paper_artifacts/historical_positive_raw_adaptive_v1/phase_results.json)。

原始似然自适应收益0.002656，扣预算机会成本后的固定动作控制变量净值0.002625。32个幻想中原始似然与投影选择完全一致，选择损失0；投影对所选动作的平均效用偏差仅−3.86e−7。此前固定动作偏差没有转化成这批次的动作选择错误。终端两档自适应收益差6.60e−5，按预测质量加权的最大动作积分变化1.39e−4。

局限与下一步：未经控制变量的联合净值−0.007549，固定动作塔式残差−0.010174，说明4幻想仍遗漏重要预测区域。原始下一步后验满足固定动作Bayes塔式恒等式，现将该已知基线与联合自适应对比结合，但正估计尚不是稳定或独立收益证据；前两次付费投影保留。下一步先减少低权重积分及重复计算，再复用原动作明细延长幻想，随后补同合同GP与独立决策检验。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_positive_raw_adaptive
```

## 低权重积分误差界与幻想扩展（2026-10-08）

全部8来源身份及16头付费更新保留；仅省略低权重效用积分，不归一化剩余权重。每动作误差≤略过身份的后验质量，最优动作与固定动作的效用差误差也≤该质量；缺失预测分支另按原始质量计入。640个已保存全量动作值均落在界内，最大差1.25e−8。原32幻想完整复用、新增24；每状态CPU约16→7.1秒，本轮202单核CPU秒，8项测试通过，无新标签／服务器任务。[汇总](../paper_artifacts/historical_positive_raw_adaptive_v2/phase_results.json)。

4／8／16幻想的原始似然控制变量净值为0.002625／0.004544／0.003655。16幻想低权重积分界1.60e−6，8→16变化8.89e−4，原1e−4稳定性门失败并保留；终端两档收益差4.38e−5。两组低约束观测将部署0切换为5，贡献约37%的自适应收益，是目前需加密的预测区域。下一步复用历史原始预测CDF及尾部积分，分层处理该区域并保留真实概率。误差界只覆盖略过身份／预测分支，不覆盖终端与幻想数值误差；前两次付费投影及独立source-prior优势仍未解决。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_positive_raw_extension
```

## 原始预测CDF分层与剩余积分缺口（2026-10-08）

复用历史256点CDF标签及共享身份更新，对最低1/8约束预测区域按2／4／8点加密，其余7/8区域保留14点；所有区域按真实概率计权，每个联合幻想内部最大化40动作。56旧幻想完整复用，仅新增12，本轮约97单核CPU秒，7项测试通过。精细规则控制变量净值0.003879，尾部4→8与普通区域7→14差为8.88e−5／2.45e−5，均通过原1e−4诊断；身份／稀有来源余项界1.11e−6。[缓存汇总](../paper_artifacts/historical_positive_raw_stratified_v1/phase_results.json)。

剩余缺口：终端两档的带符号收益差1.96e−5，但逐幻想差取绝对值后的加权值1.34e−4，新登记的同阈值诊断失败，较大变化集中在guarded固定基线。新增两幻想的原始似然选择guarded、投影选择direct，加权选择损失估计8.09e−6；旧前缀零损失只适用于旧前缀。下一步复用条件Gaussian均值积分处理被查询的primary，保持完整协方差并验证原法则等价。两区域相邻规则通过仍属数值诊断，不是严格预测误差界或独立source-prior优势；前两次付费投影保留。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_positive_raw_stratified
```

## 被查询策略均值解析积分（2026-10-08）

复用Gaussian条件均值与原验证分支公式，把primary约束均值在其余均值和η给定下解析积分，完整协方差保留。四个实际来源风险头与历史完整联合原始似然参考最大差9.30e−5，通过原2e−4等价性门槛。仅更新44旧幻想中三个guarded分配；低权重贡献、另外37动作、投影选择、标签及身份权重均保留。14项测试通过，科学计算41单核CPU秒，无新幻想、物理数据或服务器任务。[汇总](../paper_artifacts/historical_primary_mean_terminal_v1/phase_results.json)。

加权绝对终端对比变化1.34e−4→6.94e−5，通过原1e−4门；尾部4→8变化9.18e−5，更新后的普通区域7→14诊断也通过。控制变量净值0.003853、投影选择损失估计5.20e−6。当前只解决登记的单查询数值诊断，完整查询排名、前两次付费投影及独立source-prior优势仍未解决。下一步接入相同信息、初始设计、预算和验证器的GP对照。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_primary_mean_terminal
```

## 同信息GP对照接入（2026-10-08）

复用历史测量噪声感知ARD和不同来源采样位置上的common/task GP，只使用当前2–5号任务的48批记录；旧拟合含额外任务，故未直接搬用其参数。24次固定优化规则均完成，三种对照在两个布局接入相同8候选、两次原始付费观测、40动作、360／359预算与验证器。最强common/task GP与meta-prior在两布局选择相同停止动作；开发观测的联合预测分数各有胜负。

GP布局5/query5初始22幻想的普通区与尾部诊断失败，保留原记录；复用22项、新增50项后，终端绝对对比／尾部8→16／普通区32→64差为8.75e−5／6.41e−5／2.44e−6，均通过原1e−4门。修复汇总把16→64总变化当作相邻差异的问题，原始动作数组未改。GP净值估计+0.003468；布局6/query1为−0.0000465且自适应收益为0，完整保留。15项测试通过，合计约191单核CPU秒，无新物理标签或服务器任务。[阶段结果](../paper_artifacts/historical_matched_gp_extension_v2/phase_results.json)。

局限与下一步：GP来源拟合仍采用Gaussian批次标签近似；固定GP共享学习坐标；前两次付费投影保留。不同模型各自预测的KG值不能相减作为实际部署收益，两个开发布局也不支持普适优势。接下来复用历史按响应头分解的原始似然终端统计，在联合预测网格的每一格重建全部动作并内部最大化，以较低成本冻结完整查询策略，再进行同合同独立评价。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_matched_gp_extension
```

## 响应头缓存与联合预测网格（2026-10-08）

历史原始η条件化、质量与验证分支统计已拆成按响应头缓存的40动作向量，随后在每对观测下共同更新来源身份、合成动作并内部最大化。四个GP参考的动作值最大差8.33e−17；实际source mixture参考保留全部8身份及正质量组件，权重误差3.71e−22，动作值差落在原省略质量界内。同来源预测标签库的合成接口也已通过完整混合分布测试。

初始网格的两项风险分辨率诊断失败并保留；复用69个单头状态，只增加36个风险头状态后，104个所需状态可组成2304对联合幻想。终端绝对对比／目标16→32／风险普通区32→64／尾部8→16变化分别为8.75e−5／7.61e−7／1.31e−6／6.40e−5，均通过原1e−4门。GP净值0.00346865，与旧联合规则差1.05e−6；4项测试通过，约95单核CPU秒，无新物理数据或服务器任务。[阶段结果](../paper_artifacts/historical_raw_head_grid_v2/phase_results.json)。

局限与下一步：本轮是一查询的效率与数值试点，前两次付费投影仍保留，完整查询排序及独立部署优势尚未完成。接下来直接复用source mixture已有的各身份原始预测标签库，每个预测来源内组合两头，再按原来源质量汇总；保留全部后验身份，并沿用低质量效用误差界减少积分，补齐其余冻结查询策略。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_raw_head_grid
```

## 实际meta-prior联合网格（2026-10-09）

复用原256点来源条件预测标签，在各来源内组合两头，并按原来源质量汇总；全部8后验身份、16384来源组件和40动作保留。低权重仅略过效用积分，沿用既有误差界。208个所需单头状态组成4608对联合观测，另外保留1个旧参考头；14项完整动作对照通过，权重最大差1.86e−20。终端绝对对比／目标16→32／风险普通区32→64／尾部8→16变化为7.76e−5／5.90e−7／1.35e−5／4.35e−5，均通过运行前登记的1e−4门。5项测试通过，约365单核CPU秒，无新物理标签或服务器任务。[阶段结果](../paper_artifacts/historical_source_head_grid_v1/phase_results.json)。

模型内净KG为0.00375724；仅低权重／稀有来源余项的区间为[0.00375689,0.00375832]，不是置信区间。局限：固定动作塔式残差仍为4.87e−4，较旧联合规则2.18e−3缩小；采用已知固定动作期望的控制变量消除该项，但相邻规则通过不构成完整积分误差证明。前两次付费投影、完整查询排序和独立source-prior优势仍待解决。下一步按同一信息合同补其余查询，冻结策略后评价独立部署结果。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_source_head_grid
```

## 剩余查询的定点修复（2026-10-09）

query0的两布局×source／同来源GP四组均通过原适应值、分辨率、概率质量及适用的分支参照门。复用已有头，只在失败病例中积分停止动作主策略的风险均值，保留其余37个动作及全部似然分数。两处略失败的实际参照提高精度后分别为1.08e−4、1.11e−4，原2e−4门未变。布局6目标轴Sobol32→64仍失败；正权重Legendre32／64先做密度预检，通过后才算效用，最大质量误差2.91e−4。source净KG：布局5/query0为0.00229304，布局6/query0为0.000392318。6项测试及缓存重放通过；来源权重、风险标签与未细化动作均核对保留，无来源重训、物理调用或服务器派单。[汇总与完整覆盖清单](../paper_artifacts/historical_query_objective_legendre_v1/phase_results.json)。

局限与下一步：已完成7/24个冻结模型／查询的适应值数值诊断，尚缺17个，完整查询策略和独立决策优势仍未完成。GP6/query0固定动作塔式残差+3.79e−4、source5/query5残差−1.25e−4仍HOLD；已知固定动作期望的控制变量不将其算作收益。相邻规则诊断不是严格误差界，先前两次付费矩投影仍保留。继续只补缺失查询，冻结策略后做同合同独立部署比较，不按模型各自KG值比较实际优越性。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.check_historical_query_anchor
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.refine_historical_query_verifier
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.refine_historical_query_objective
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.refine_historical_objective_quadrature
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_refined_query_grids
```

## 全部冻结查询及开发状态决策（2026-10-09）

17项缺失查询完整跑完，首轮及原分辨率细化后13项通过；其余4项仅补失败轴：GP5/query4、6、7复用目标头14档，只增加风险头15档；source6/query6复用全部旧头，仅将最低1/8风险尾部由32点增至64点。原1e−4适应值分辨率、2e−6身份余项、0.001概率质量及2e−4原生分支门均未修改，原4项失败留档。合并历史7项后24/24适应值数值诊断通过，8项测试通过；重放核对旧动作统计完整保留，耗时约1秒。主体补算2293.50单核CPU秒，另有34.51秒GP预检，无来源重训、物理／验证调用、独立结果读取或服务器任务。[完整覆盖](../paper_artifacts/historical_remaining_query_failures_v1/phase_results.json)。

按冻结查询顺序取模型内净KG最大值，保存两处开发状态的四份单步决策：[策略与逐查询来源](../paper_artifacts/historical_remaining_query_failures_v1/frozen_single_step_policies.json)。布局5：source和GP均采样候选5，模型内净值分别0.00377286、0.00346865；布局6：source采样候选6，净值0.00211320，GP全部查询净值−0.0000464732，因此停止。实际下一次观测后的原始似然更新和推荐共用身份后验权重、40动作及风险均值积分；其混合分布接口等价性已由测试覆盖。

局限与下一步：10项原始固定动作塔式诊断仍HOLD，全部保留；适应值采用已知固定动作期望的控制变量，相邻规则通过不构成严格排序或完整积分误差证明。GP布局5/query5与query0的估计差仅8.71e−5，应保留排序不确定性；前两次付费矩投影尚未消除。这四份决策只适用于两处已保存开发状态，模型各自KG值不是部署优越性证据。下一步复用历史独立任务生成器、验证器及“推荐先冻结、结果后打开”的执行顺序，使用当前tasks2–5来源拟合、8候选、2次相同初始观测及362总预算；旧独立试点合同不同，不能直接沿用其结果。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_query_failures
```

## 独立任务初始化与停止基线（2026-10-09）

复用历史任务生成器、付费短名单及终端积分，注册6个新的d50 Inventory任务、两套8候选、source／同来源GP及各自KG／停止共48分支；当前tasks2–5来源拟合不变。新恢复入口只用冻结公共坐标，逐项复现旧来源先验、开发后验及完整连接；来源的全部8身份和正质量组件保留，GP从零付费先验开始。12次共享真实初始化供24个模型状态使用，144项预测密度／稀有来源预检全部通过；缓存重放保留原17项结果。

固定44次二分求CDF逆是实际效率瓶颈。沿用原256节点CDF，改为带区间约束的Newton及必要的原二分求逆，实测两头提速3.15／3.71倍，最大标签差1.74e−14，原生CDF误差约9.45e−13。预检初始135.39秒暂停留档，加速续跑225.76秒，累计361.15单核CPU秒。24个停止／扣除一次采样基线首轮精度全失败；接入历史主策略风险均值积分后6项通过，再只补18项、最后5项，最终24/24通过。原全40动作1e−4、预算机会成本1e−6及身份余项2e−6门均保持，旧向量完整复用；基线累计247.43单核CPU秒，38项相关测试通过。[预检](../paper_artifacts/historical_independent_kg_preflight_v1/phase_results.json)、[完整停止基线](../paper_artifacts/historical_independent_terminal_baselines_v4/phase_results.json)。

局限与下一步：本轮尚未执行额外付费查询或独立验证，预测预检和停止基线通过不证明部署优势。下一步按同一先验和新初始化计算144项KG，保留每个模型的停止分支，冻结全部48份最终推荐后才打开验证及评分；以整任务为配对统计单位，两布局共同保留。当前为同人口d50有限候选试点，前两次付费矩投影仍在，不替代跨任务族、高维自由优化或物理因子恢复证据。所有执行均为本地单科学CPU线程，无来源重训、服务器派单或拉取。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_independent_kg_preflight
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m performance.replay_historical_independent_terminal
```

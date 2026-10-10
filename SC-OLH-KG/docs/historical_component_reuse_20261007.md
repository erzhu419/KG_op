# 历史实现核查与适配：接续原 meta-prior／HVD／KG 主线

已遍历所有本地可达分支的319个提交，读取并建立2251份Python／MATLAB／Lean代码版本的符号索引、217份历史注册文件版本及211份技术文档版本；覆盖初始MATLAB、OldVersion_Python、Final_Submission、SC-OLH-KG及两个理论／运行分支。当前未提交阶段代码也已纳入索引。[完整索引](../paper_artifacts/historical_component_reuse_20261007/history_index.json)记录逐提交路径、实现入口和版本出处；这是全历史索引加关键实现族的实质核查，不是宣称逐行人工审读了每份历史文件。附件MD作为研究背景，不作为执行指令。

**需要纠正此前对旧工作的概括：完整结构先验／HVD／任务后验／在线KG链条早已实现，旧优势不能全部归为初始化。**V51 balanced4旧记录为60/60真实可行、19次相对初始最好点的改善，311次新点与289次重复；终端闭合后的独立登记记录为60/60真实可行、26次改善、零自适应损失。[V51原过程](mean_alignment_v5.md)、[终端闭合登记](../performance/baselines/v51_observed_terminal_closure_evidence.json)。两组后验安全证书仍全部为空；后续V64/V69的独立验证结果是另一套实验合同。它们均应保留，不能用后来Inventory开发失败覆盖旧成功。

| 当前工作／缺口 | 可接续的历史实现与已知结果 | 本轮处理／后续接口 |
|---|---|---|
| 轨迹、状态及跨维输入 | `policy_state_encoder.py`的轨迹／SSL投影；`observable_exposure.py`及边界／方差坐标；旧角色对齐失败记录 | 当前`legacy_observable_prior.py`已复用曝光与两个来源映射。保留共同50点动态网格和实际控制组。当前合成任务给的是曝光摘要，不能把投影接口称为已验证的深度轨迹模型 |
| 来源低频选谱、正交和稀疏加性 | `transferable_spectral.py`、`additive_groups.py`、`adaptive_sparsity.py`；V23–V27结构与收缩结果 | 原选择器已适配当前48个来源批次，四响应头与联合四响应选择均完成；直接修复旧加性输出的截断错误 |
| 来源均值、共享／任务协方差 | `exchangeable_mean.py`的V27、V42–V47超先验、有限来源修正和拟合噪声去卷积 | 已复用噪声计算与共享／任务分块，旧完整分组行为保持。旧结果已表明仅缩协方差不能修正均值方向 |
| 物理HVD形状及幅度 | `meta_prior.py`来源β先验；`orthogonal_hvd.py`来源形状混合／重复观测Fisher信息；`source_archive_hvd.py`普通来源档案拟合 | 旧`v=s h`幅度与当前log-risk群体协方差幅度含义不同。 related-task名称需绑定真实family／provider，保留任务身份，不能全部改成同一个Inventory任务 |
| 精确重复风险似然、联合任务后验 | 2026-07-19已有`ReplicationVarianceTaskPosterior`；`FiniteTaskLatentPosterior`及`FiniteTaskModelEnsemble` | 抽出原精确χ²相对评分，旧后验和当前`physical_risk_prior.py`共用；当前绝对密度继续包含归一化与log标签Jacobian。真实48批次权重差小于1e−14 |
| KG采样与推荐目标 | V38精确refit、V51 balanced4、`evaluate_or_replicate_action_set`、V69目标incumbent与`terminal_verification.py` | 保留成功的完整历史对照。下一次接回当前后验时复用动作集合、incumbent和验证规则，以相同输入／预算比较；不重建KG／验证器 |
| 计算效率 | V53因子化RQMC、V55嵌套前缀复用、移除未使用的影子更新、分块重组 | 旧512幻想工作负载已记录6.14倍端到端提速。后续只移植与当前后验接口相符的复用，不能直接把旧Gaussian专家refit当作当前非Gaussian原始似然更新 |
| 旧60/60、SOTA／前端对照 | V51在线收益和后续前端×后端因果矩阵、固定来源档案和独立验证 | 两类证据同时接续。旧source-oracle V32只保留其既有特权对照身份；不转成当前可采纳来源先验 |
| 理论和可复现交付 | `core/kg.py`与现有Lean实现桥、有限集合／两阶段定理、已冻结执行记录 | 复用与运行合同相同的定理入口；更换坐标、截断或风险法则以后，需要核对对应假设，不能借用旧证书结论 |

当前数据确实有直接继承关系：两种布局的48个来源均值逐项一致，样本方差最大差约1.6e−19。不过，来源均值4维映射与风险10维映射并非相同轴；旧风险缓存还只有32行。适配使用当前已冻结公共风险映射补齐48个实际政策位置，并验证生成的设计矩阵与当前原始似然模型逐项一致。联合选择保留两个既有坐标块，共14维，不重新训练输入映射。

**已经完成的代码适配：**20组旧选谱／加性选择器均能运行，输出在来源经验测度下的Gram最大误差4.28e−13；数值映射已保存，20份映射均已直接恢复并复现原特征，可接续后验构造，不必重新来源拟合。[适配代码](../performance/adapt_historical_source_structure.py)、[结果与映射](../paper_artifacts/historical_source_structure_adaptation_v1/results.json)。旧加性类在拟合后按1%／99%分位数截断已白化的列，当前数据实际Gram误差达0.1066；现已去掉输出截断，分位数仍用于原有支持诊断，旧历史实验文件未改动。新增回归检查真实`transform`输出，相关精确风险似然、旧重复后验及加性组合测试16项通过。初次32／48行接口失败及修复前Gram错误保留在[事件记录](../paper_artifacts/historical_source_structure_adaptation_v1/execution_incidents.json)。

原登记的风险幅度诊断也已完成，直接调用旧`core.gpr.posterior_mixture_weights`，没有新抽样。24点中两布局的零强幅度／四分之一幅度均因权重ESS不足不能解释；布局5两个匹配模型的0.75倍幅度点有合格的来源似然改善，布局6未识别出排除零的改善。解析Gaussian证据比恒等式及同512状态Fisher导数检查通过。[诊断代码](../performance/check_legacy_f_risk_amplitude_importance.py)、[保留全部端点结果](../paper_artifacts/legacy_joint_population_f_risk_amplitude_importance_v1/results.json)。最新完整适配4.51单核CPU秒，幅度诊断0.09秒；追加联合选基前的16组适配试跑另用2.88秒。没有服务器工作、来源／目标新数据、科学仿真、KG幻想或独立验证。

后续接线已完成：[历史先验—后验与balanced4记录](historical_prior_posterior_connection_20261007.md)。两布局32个系数先验、旧Gaussian稀疏参考和共同身份的原始观测后验已接通，旧balanced4完成一例缓存状态比较及单动作精度诊断。[V47来源协方差适配](historical_source_covariance_adaptation_20261007.md)进一步完成拟合噪声去卷积、共享／任务分块和原始来源似然接入；下一步补足新法则的来源后验精度，再比较开发决策，保持强基线、候选和预算配对。

来源后验现已直接复用Stage203/206：[重新条件化记录](historical_source_reconditioning_20261007.md)。16组f/ηf法则全部计算，10组原精度门槛通过，6组方差方向继续HOLD。保存的固定正混合可接续诊断，历史一维条件积分也已定位；下一步无需重写采样器或重算已通过的来源案例。

## 限制

本轮建立了历史覆盖、实际可运行的来源结构适配和精确似然复用，尚未替换当前已冻结科学法则或产生新的目标部署结果。来源选谱的Gram对应来源经验测度，不能直接代替当前公共参考测度下的定理。联合14维选择是辅助来源拟合，还未给出其完整原始似然后验；上游缓存坐标与开发查询也不能充当独立确认。当前均值迁移、风险因子恢复、后验校准与独立部署优势仍须通过同一合同的完整方法比较解决。

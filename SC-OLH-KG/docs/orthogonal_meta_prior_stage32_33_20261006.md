# 来源调用减半及风险稳定性：尚未建立样本效率优势

按已登记子集将来源调用288→144：六任务各8位置／3重复，共48条标签、28种策略。均值头公开四维表示设计秩仍27；风险十维encoder、标准化及全部风险先验只用48条标签重训，风险设计秩28。20次来源优化均按原停止规则成功，六份来源预测及18份目标预测分别冻结后评分。

| 144调用模型 | 来源g观测MSE | 来源联合batch score | 目标g真值MSE |
|---|---:|---:|---:|
| 联合结构 | .004958 | .372880 | .003935 |
| 共同／任务平谱 | .001909 | .344081 | .001404 |
| 联合GP、同风险头 | .005283 | .206533 | .004878 |

结构目标g MSE由288调用时.000894升至.003935；平谱更强，结构相对GP的预测信号也不跨两个来源留出任务一致。三臂共有的eta_g目标MSE约.848，结构eta_g的95%矩区间真值覆盖率仅49.06%；288调用时其MSE为.262857。63号结构g偏差+.018081、eta_g偏差+.549414，代入均值／eta的机会余量+.004405，实际-.026799；六任务均误拒绝该安全候选。没有运行终端排序或决策积分。

风险归因把63号eta_g偏差拆为来源输入先验相对目标真值+.709044、查询输入迁移-.156253、两条付费更新-.003377。减样本的主要变化是来源输入风险预测漂移，两条目标观测几乎未纠正。

Stage34继续完成固定／重训encoder的逐任务删除对照，72次来源拟合及全部查询均成功。63号eta_g删除预测均值标准差由固定map的.281295增至重训map的.724053，原条件先验标准差.518143；同一次删除的两组预测差RMS为.799760。学习表示对来源任务组成敏感；其波动不是校准后的后验方差。下一步已登记：把完整48标签风险模型与六个删除模型接入既有完整身份混合，付费原始f/g证据更新共同权重，并在相同风险处理下比较结构／平谱／GP均值头。

限制：本轮均为已打开任务的预测归因，代入余量不是后验认证概率，不证明新的推荐一定失败；来源输入先验偏差含任务／策略差异。尚无144调用决策收益、独立确认或KG新增采样贡献。

13项相关测试通过，包含实际子集下标签隔离。来源准备／拟合／评分／目标归因／风险分解约5.06 CPU秒；删除稳定性另3.30秒，含拟合1.92秒。新增仿真、验证、终端积分、服务器派单和拉取均0。

入口`performance/check_legacy_public_mean_source_half.py`、`performance/diagnose_legacy_public_mean_source_half_target.py`、`performance/diagnose_legacy_public_mean_risk_budget.py`及`performance/check_legacy_public_mean_risk_stability.py`。完整输入、拟合及结果在`paper_artifacts/legacy_joint_population_public_mean_source_half_v1/`、`legacy_joint_population_public_mean_source_half_target_v1/`和`legacy_joint_population_public_mean_risk_stability_v1/`；下一步合同`performance/manifests/legacy_joint_population_public_mean_risk_uncertainty_plan_v1.json`。

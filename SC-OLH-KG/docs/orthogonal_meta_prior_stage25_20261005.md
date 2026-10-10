# 共同函数与任务残差联合学习：来源端失败显著缓解

保持公开覆盖16点、六来源任务×3重复（288调用）、encoder、95函数基、观测噪声及截距约定。结构先验现在分别学习共同函数与任务残差的组尺度和谱指数；通用GP同样可以重新学习共同RBF幅度。旧固定共同协方差的对照保留。没有新增噪声下限或改变环境风险似然。

| 来源留出模型 | f观测MSE | g观测MSE | 联合batch log score |
|---|---:|---:|---:|
| 原固定共同协方差结构 | .001156 | .067865 | -5.654228 |
| 联合学习结构 | .001295 | .014317 | -1.584384 |
| 联合学习、池化任务残差 | .001309 | .014843 | -1.516752 |
| 原固定共同核GP | .001432 | .006953 | -1.124795 |
| 联合共同幅度GP | .001432 | .007162 | -1.208512 |

两个已打开来源留出任务的结构g误差都降低约.053，联合log score分别提高4.176、3.964，约束batch score也都提高。共同协方差冻结是实际瓶颈的一部分；这次修复同时改善误差与预测密度。通用GP仍更好，其共同幅度再估计反而轻微退步，因此目标比较同时保留两个GP，不能只选较弱的新GP。

相同4096次优化预算的固定共同协方差续算另行完成：四头仅0–1次迭代，g来源目标改善3.96e−11，留出g MSE仍.067865、联合log score仍-5.654195。原停止规则下的简单续算没有重现上述改善。合同及结果：`performance/manifests/legacy_joint_population_fixed_common_completion_attribution_v1.json`、`paper_artifacts/legacy_joint_population_fixed_common_completion_v1/`。

计算检验：新增共同／残差似然梯度及新任务后验与完整Gaussian计算一致，含通用共同幅度为零的情形；12项相关测试通过。八个头从原固定共同解启动，初始来源目标逐项一致，最终目标均下降。g共同谱指数为4、任务残差为0；其他风险头不据此宣称物理因子恢复。

限制：只含两个此前打开的d50来源留出任务，观测MSE包含标签测量误差。各优化器按原停止规则成功，结构g使用3889次迭代，投影梯度约.00274，并非严格梯度收敛。GP长度尺度继续使用原来源anchor；全残差对照保留共同均值及后验，不隔离全部四项结构。来源预测改善尚不等于独立部署或KG收益。

来源拟合CPU4.48秒，预测/评分2.26秒；四头续算与评分另.91秒。新增仿真／验证调用0，本地串行单线程。入口`performance/check_legacy_joint_common_covariance.py`；合同`performance/manifests/legacy_joint_population_joint_common_covariance_execution_v1.json`；结果`paper_artifacts/legacy_joint_population_joint_common_covariance_v1/`。

来源预检满足原继续条件，五臂同预算缓存目标归因已完成，但结构条件安全收益仍为0；见`orthogonal_meta_prior_stage26_20261005.md`。没有新来源优化或服务器派单。

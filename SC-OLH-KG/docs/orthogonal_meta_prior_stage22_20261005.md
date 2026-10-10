# 共同函数均值匹配后，结构独有收益尚未出现

通用联合RBF现在同样学习来源共同函数均值及独立任务残差，计入标签测量误差，并向目标传递完整均值后验及插值不确定性。核参数来自冻结的来源测量误差GP；四头残差拟合仅用六个来源任务。白化计算与完整Gaussian矩阵的似然、梯度和目标后验一致，无额外方差下限。

| 四观测先验 | 六任务平均条件安全收益 | 条件总调用 | g真值MSE |
|---|---:|---:|---:|
| 标量均值、已知标签误差RBF | .498690 | 306.28 | .030136 |
| 联合共同函数均值RBF | .804124 | 262.85 | .001564 |
| 联合结构先验 | .804124 | 262.85 | .005857 |
| 联合全字典 | .798638 | 232.59 | .002861 |

联合RBF与结构先验六个完整动作逐项相同，配对条件收益/调用差均为零。此前结构相对标量均值RBF的+.305434收益差被通用共同函数均值模型重现。证据支持来源函数学习及联合残差处理的早期价值；四项结构的额外决策收益尚无支持。旧前端60/60、stage20真实独立验证及各阶段负结果保持原身份。

付费归因也已完成：缓存两观测停止，对比再购买固定点2、3；共36个动作先冻结后评分，两分支共享两观测确定的支持与参考保留候选。结构／全字典／联合RBF条件收益分别由.804357／.797216／.804357降到.711461／.704318／.712394，调用由214.51／217.70／214.51升到237.36／240.18／237.35。

损失主要集中在任务03：三种方法均放弃63，改荐12或3，条件收益由.805833降至约.253318。纯预算扣除损失仅2.31e−9；结构平均-.092896中，-.092831来自动作变化。联合RBF整体g MSE虽由.001613降到.001564，仍发生同一决策失败。备用名单保留的是较弱参考模型选出的22，没有保留实际前一步共同推荐63。

已另行冻结并完成18个“额外保留实际旧推荐”的对照。任务03恢复至约.805833；结构／全字典／联合RBF平均条件收益为.801537／.788150／.800602，调用255.12／256.52／274.69。结构相对旧四观测类提高.090076，但任务bootstrap95%区间[-.004845,.275663]跨零；相对两观测停止仍低.002820，多40.62调用。此为备用候选修复的归因，没有建立新增观测或KG收益；原负结果不覆盖。

限制：六任务已打开，本轮仅条件评分。前表使用原四点支持，付费归因使用固定两点支持，属于不同终端合同。相同动作及零差区间不能证明总体等价；新增RBF没有新的独立认证。来源核/均值正则量由同一来源档案估计，属于经验Bayes对照。完整物理因子恢复、自由高维搜索和KG增量仍待验证。

连续保留22、63实际产生288行、217种独有终端动作；已去重，实测217行，保持全部独有动作及原预算。25项相关测试通过。新GP拟合.18秒、比较50.04秒；付费归因411.93秒，预测分解.99秒，备用修复比较311.38秒（该已完成运行仍使用去重前的积分）。模拟/认证新增调用0，本地串行单线程，无服务器任务或拉取。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 performance/run_legacy_joint_population_kernel_mean.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 performance/run_legacy_joint_population_paid_prefix.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 performance/run_legacy_joint_population_paid_prefix.py diagnose
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 performance/run_legacy_joint_population_actual_retention.py
```

结果分别在`paper_artifacts/legacy_joint_population_kernel_mean_v1/`、`legacy_joint_population_paid_prefix_v1/`、`legacy_joint_population_actual_retention_v1/`；各入口的同名执行合同均位于`performance/manifests/`。

下一步登记16／73来源点的样本效率归因：低预算必须重训来源encoder及全部先验，不能继承73点监督信息；复用原候选及缓存目标。共同函数均值已经匹配，先检验四项结构是否节省来源记录，再注册同先验停止／固定采样／KG闭环。合同：`performance/manifests/legacy_joint_population_source_budget_attribution_plan_v1.json`。

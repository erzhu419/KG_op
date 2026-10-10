# 少观测诊断与旧前端成果衔接

已完成公共中点残差协方差估计：保留来源价格估计的不确定性，用一个公共域方差参考任务正则化，不再沿用绝对价格二阶矩或固定0.01系数底噪。来源外层留一g工作log density为1.08750，汇总/均匀为1.08262/1.06893；但g RMSE为0.07927，仍高于均匀的0.07774，来源拟合并非全面改善。g价格协方差对角降到[0.02056,0.02423,0.01226,0.01760]。

同一已付费路径的2/4/6/8点重放完成，约45.4单线程CPU秒；两项新测试通过，128组7/9阶数值门通过，最优预测收益最大差0.000180，无新增实验观测或服务器任务。来源、汇总价格推荐在四个预算下完全相同。4点来源相对均匀价格+汇总风险的收益差+0.00346，开发bootstrap区间[0.00049,0.00743]，4胜4平；其中价格尺度贡献+0.00219，风险结构贡献+0.00127。2点差为零，6/8点差−0.00307/−0.00150，因此不能提升为稳定或独立优势。

新瓶颈：来源臂2/4/6/8点平均真实预计收益为0.83882/0.78317/0.78164/0.83595，并非只存在收益饱和。95425在4点后，纯物理分量推荐7，而混合选择18；g物理分量权重仅0.06849，后备分量主导。候选18预测通过率0.88009、真值0.34105，g均值偏差−2.21个标准差。此任务的2→4点收益降幅为0.56700。已有样本中的推荐退步需要保留，不能因4点小平均增益而隐藏。

旧成果已核实并接回新稿：FactorShock/InventorySupplyChain/QueueResourceControl每类20个算法种子，冻结atlas+proposal-only/Stacked GP/canonical SAASBO均60/60可行及认证，零错误认证；共同Sobol对应0/60、1/60、0/60。原生八种迁移方法最高RGPE 30/60，接入共同前端后56--60/60；当前最终表没有“最高48/60”这一口径。主文新增`meta_prior_legacy_frontend.tex`，补充列出原生与共同前端表，两份PDF编译通过。证据入口为`paper_artifacts/legacy_frontend_evidence.json`，由已有小CSV/JSON自动生成。

代码中的旧前端仍由`core/designs.py:load_frozen_source_informed_design`和`performance/benchmark_transfer_fairness.py:load_source_initial_design`进入迁移流水线。新四动作物理Inventory仍用共同8点初始化，尚未执行旧atlas+新风险后验+KG组合。旧原始模拟输出是(f1,f2,g)，新批次标签是(f,g,eta)；源/目标重复次数及验证合同也不同，不能按数组长度直接拼接或合并结果。

下一步先打通已有前端、原始观测和当前联合后验的同合同接口，并保留独立验证的初始候选支持，再做小规模组合对照。停止以价格协方差微调替代完整方法验证；评价推荐收益、遗憾和源/搜索/验证成本，保留所有失败。当前阶段仍是已打开开发任务的固定路径重放，KG点未执行，四个预算的总调用数不同。

复现：单线程运行`performance/replay_residual_fewshot.py`，再运行`performance/score_bounded_billing_development.py --directory paper_artifacts/residual_fewshot_replay_v1`。旧前端表由`performance/render_legacy_frontend_evidence.py`生成。

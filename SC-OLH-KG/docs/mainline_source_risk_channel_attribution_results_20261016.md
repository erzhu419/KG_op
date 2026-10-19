# 风险通道分解：两种排序失真，没有统一修正依据

[冻结诊断](../performance/manifests/mainline_source_risk_channel_attribution_v1.json)已完成。覆盖全部24来源案例、96后端案例及384个静态通道组合；144组角色概率重建误差为0，原排序一致。模型、阈值、终端及精度未改。[汇总](../paper_artifacts/mainline_source_risk_channel_attribution_v1/summary.json)与[费用](../paper_artifacts/mainline_source_risk_channel_attribution_v1/completion.json)：记录分析CPU **0.9133秒**，新增仿真、终验、真值调用、拟合、恢复、回放、随机抽样和下载均0。

A表示初始化正常传播、之后仅更新自身（AFTER），F表示持续传播（FULL）。下表按风险均值／观测噪声方差／后验方差排列；固定目标排序分别取A、F，两种视图的选择完全相同。

| 风险通道组合 | 选择经验可行策略 | 相对AFTER改变选择 |
|---|---:|---:|
| AAA、FAA、FAF | 14/24 | 0 |
| AAF、AFA、FFA | 13/24 | 1 |
| AFF、FFF | 12/24 | 2 |

没有组合提高经验可行数；持平组合也没有改变任何选择。单独改变风险均值不起作用。[通道分解](../paper_artifacts/mainline_source_risk_channel_attribution_v1/risk_channels.json)说明两例退步机制不同：

- **Inventory（Queue留出档案、折2）**：噪声取F就选62，取A就选54。噪声下降让两个策略进入预测合格集合，再由目标选择62；经验风险裕量从−0.046454变为+0.000481。但F的噪声仍高于两次评分重复的样本方差，不能直接称为低估；它减少了原先对乐观风险均值的保守补偿。
- **Queue（Inventory留出档案、折0）**：全部组合均无预测合格点。单换后验方差即可令6→15，换回则恢复6；均值、噪声与其有交互。这里是后验方差相对变化改变最小违约概率排序，经验风险裕量−0.021696→+0.049003。

[概率饱和审查](../paper_artifacts/mainline_source_risk_channel_attribution_v1/probability_saturation.json)中，改看CDF之前的排序不改变任何FULL／AFTER首选，因此不能解释上述两例或AFTER／POINT的四次差异。唯一变化在POINT：35→62，双方仍经验不可行，目标反而恶化0.439910。当前没有理由把精力转回概率尾部精算。

[边界噪声分解](../paper_artifacts/mainline_source_risk_channel_attribution_v1/noise_boundary.json)发现，单个未采样策略0占FULL目标噪声总量99.999892%。其噪声4.283→288275伴随eta均值−12.515→8.765、方差增加，以及additive/joint权重0.9904→约0、spectral/joint约0.0096→约1。匹配高斯代理的对数变化23.651与实际混合矩11.117不同，不能据代理给实际噪声分摊贡献。另一来源背景下同一策略噪声反而下降；完整部署来源还包含该策略，暂不支持全局缩放或截断。

限制：来源案例重叠、评分只有2—3次重复，经验可行不等于获证或真实部署收益；缺少分量矩，身份变化与eta漂移只能记为同时发生。[d1000日志](../paper_artifacts/mainline_source_risk_channel_attribution_v1/target_context.json)缺少最终完整已观测池的角色风险分量，challenger空集也不等于完整候选集空集。原POINT3018失败、Queue3009质量参照缺失、来源精度／低ESS标志及两次分析启动前的解释器错误均保留。

下一步[已登记、未启动](../performance/manifests/mainline_target_terminal_risk_reconstruction_v1.json)：复用24个d1000几何的71份完整角色轨迹，恢复13条观测后最终已观测池的风险均值、噪声、后验方差及身份权重；重复查询仍按原历史处理，不当作不同物理策略。核对原排序和实际终端输入后，再连接已有真值、验证与部署记录。72注册单元中的旧失败保留，全9对照保留；相同历史才作终端机制归因。无新增拟合或仿真，恢复与raw条件化单独计费。先补齐实际决策证据，再判断是否存在共同修正依据。

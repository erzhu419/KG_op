# 历史来源协方差适配

V47 的拟合噪声去卷积已适配到当前来源地图，并保留共享函数和独立任务残差。两布局共完成32个响应头、128次既有批次系数拟合、64个Gaussian来源参考法则，以及16个原始来源似然接入案例。[结果](../paper_artifacts/historical_source_covariance_adaptation_v1/results.json)、[协方差分解](../paper_artifacts/historical_source_covariance_adaptation_v1/coefficient_parts.json)、[来源法则](../paper_artifacts/historical_source_covariance_adaptation_v1/source_laws.json)。

来源仍是Inventory的task2–5，每任务12个既有批次。复用旧加权ridge求解器，将声明的观测噪声通过实际估计器传播；随后调用从旧V47抽出的中心化噪声扣除公式。没有将四个任务伪装成四个业务族或重复episode。共享均值拟合噪声单独记录；来源条件化所得共享后验协方差H已包含这部分观测不确定性，故没有再加一次。

按当前观测噪声模型，扣除后均值头f/g的任务协方差迹缩小0.23%–2.18%，风险头ηg/ηf缩小25.93%–94.04%。这说明当前风险系数的表观任务差异受拟合噪声明显影响。32组配对Gaussian来源参考中28组NLL降低，平均变化−0.00887/标签；这是同一来源上的近似分数。

原始来源接入使用未条件化的共享／任务群体法则，以及已有Normal均值、精确χ²样本方差似然。Gaussian参考后验没有被再次当成原始来源更新的先验。地图在共同的48个来源函数值上连接，各响应头的系数维度无需相同。

直接复用旧512个风险后验样本时，有效样本数仅1.00–1.98。随后复用旧方差参考构造，按固定来源法则／付费标签变换实际样本，并在完整密度比中计入Jacobian；有效样本数提高至9.17–93.41，但16组均未达到登记门槛200。[直接记录](../paper_artifacts/historical_source_covariance_adaptation_v1/raw_importance.json)、[变换记录](../paper_artifacts/historical_source_covariance_adaptation_v1/transport.json)。来源后验仍为HOLD，目前不能据此排名或采用这些法则。

8项相关单元测试通过，覆盖噪声传播和密度变换。旧分组模型另以两业务族、各两个episode的既有测试夹具重放：四种V47先验的均值和差异方差完全一致，协方差最大变化9.72e−17。[验证记录](../paper_artifacts/historical_source_covariance_adaptation_v1/verification.json)。两阶段实际计算共1.92单核CPU秒，使用已保存的小JSON；本轮无新增仿真、验证器、来源链、目标查询或服务器任务。

## 限制与下一步

地图和共享正则仍是冻结的经验Bayes估计；噪声扣除针对拟合系数，不消除ridge偏差或地图选择不确定性。Gaussian风险标签近似仅用于参考；本轮原始来源接入覆盖f/ηf，来源仅一个Inventory任务族。风险协方差缩小尚不能证明物理因子恢复或部署收益。

后续已复用旧Stage203/206采样与共享系数投影：[来源重新条件化](historical_source_reconditioning_20261007.md)。16组全部完成，10组通过新链原门槛，6组方差精度仍HOLD。下一步复用旧一维条件积分诊断剩余方差方向，随后补齐g/ηg；保持共同正则、点差异和配对控制固定，再接回开发状态和单新点／balanced4比较。

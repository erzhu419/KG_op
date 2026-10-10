# 完整部署收益与风险信息瓶颈

终端收益已接通实际短名单认证、目标比较和回退，同时计算验证调用。来源重复观测补齐eta_f；目标单次调用仍只有f、g，联合更新(f,eta_f)、(g,eta_g)。KG幻想与付费观测使用相同原始Normal似然和Gaussian矩投影。5个新增测试通过，包括与实际验证器对照；原3个适配测试通过。当前KG尚未扣除搜索的预算机会成本。

重放stage7的2/4/6/10/13点前缀，共30行，零新模拟和服务器任务。短名单仅用已付费标签及冻结提案；全部推荐冻结后才评分真值。12个来源头收敛。512/1024积分有19/30行首选不同，最大参考收益损失.02136；四个关键节点在4096、8192、独立随机化8192积分中推荐一致，Inventory十点由低精度的51改为10。收益跨随机化差异仍最高.01228，低精度KG分数不能直接作为精确净收益。两个程序合计CPU约22.7秒。

四个高精度首选均真实不可行。FactorShock/Queue收益.790833/.817420，由旧首选直接接受128次验证完全复现，调用由约208降至128；微小收益提高来自验证分配。Inventory十三点完整收益首选19，收益.807812、调用333.44；旧代理候选3为.807823、205.55调用。收益对齐未解决风险排序。

Inventory安全支持4的eta均值从−6.59181仅移至−6.66930，十三点后方差收缩5.83%，真值−7.71636。十点时该点的局部eta似然信息约.0978，未观测点51只有5.80e−6：单点观测主要修正不确定的均值，难以识别环境方差。下一步复用已有公开状态暴露及来源均值/方差坐标，比较共同总调用预算下的新点、旧点重复和直接验证。暴露可从冻结X重建；含未知任务中心的`risk_exposures`不作为输入。保留旧前端与均值余项，不扫价格、不派大矩阵。

限制：三个seed80开发任务、72/73/73候选；两臂早期奖励无来源结构优势。真值预计收益不是独立认证；未执行新KG动作、轨迹学习、因子恢复或高维自由优化。

复现（本地单线程）：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 performance/replay_guarded_deployment.py`；随后同环境执行`performance/check_guarded_deployment_precision.py`。协议分别为`performance/manifests/guarded_deployment_replay_v1.json`和`guarded_deployment_precision_v1.json`；结果为`paper_artifacts/guarded_deployment_replay_v1/replay.json`、`precision.json`。

# 最终风险重建：风险改准后，目标错序仍会损失收益

[冻结协议](../performance/manifests/mainline_target_terminal_risk_reconstruction_v1.json)已完成：24个d1000几何、72注册角色单元中71份完整轨迹全部恢复，原排序、短名单及变换后的实际终端输入一致，challenger数值误差为0；POINT3018旧失败保留。全部后验先封存再连接缓存标签，[汇总](../paper_artifacts/mainline_target_terminal_risk_reconstruction_v1/summary.json)完整保留9个参照的215/216结果及48份同历史native终端。

**本批FULL相对AFTER没有新增获证或质量收益，唯一质量变化是Queue3016的下降。** FULL／AFTER都获证19/24；11组完整raw历史相同，原首选只在3011、3016、3023不同。3011双方均未获证，记录的候选不同不算部署收益；3023经过原有终端变换后名单相同，最终均安全回退initial:0；只有3016实际部署不同。

3016两臂共同查询18→53→53，两个策略都真实可行，真实目标差 **f18−f53=+0.00117544**，即53更好。[终端机制](../paper_artifacts/mainline_target_terminal_risk_reconstruction_v1/terminal_mechanism_analysis.json)为：

| 同历史后端 | 预测f18−f53 | 18的违约概率 | 实际获证部署 | 超最优可行初始化质量G |
|---|---:|---:|---|---:|
| FULL | −0.011264 | 0.498186 | 18 | 0.000003490 |
| AFTER | −0.010584 | 0.510462 | 53 | 0.000787118 |
| FULL历史＋native终端 | −0.014713 | 0.791306 | 53 | 0.000787118 |

FULL下53仍非常可行（p=0.007238），却排第三且未验证。18越过p≤0.5门槛后，错误的目标排序令其抢先部署，质量比AFTER低 **0.0007836275**。AFTER和native同样预测错目标顺序，只是对真实可行18的保守排除暂时遮住了错误。单换风险均值或噪声可改变首选，单换后验方差不能；这些静态替换没有新证书。不能把“更保守后结果更好”直接视为风险校准成功。

[全任务风险诊断](../paper_artifacts/mainline_target_terminal_risk_reconstruction_v1/risk_error_diagnostic.json)也表明，FULL相对AFTER的风险均值、噪声分位项、机会边际MSE分别在16/24、19/24、16/24任务下降，仍未产生质量增量。最终池的CDF饱和不改变任何角色臂首选。8条“未验且真实优于已获证部署”的好策略—臂记录中，6条已经通过风险筛选，问题在排序及后续验证暴露。来源边界的additive→spectral大幅身份切换也未在这些最终已观测池重现。48个role/native同历史比较有正有负，不支持统一改回native。

[费用与失败](../paper_artifacts/mainline_target_terminal_risk_reconstruction_v1/completion.json)：记录CPU **771.986秒**（含合成检查）；来源bundle读取5次、engine恢复72次，FULL重放312条既有raw，AFTER／POINT缓存未新增raw条件化。新增仿真、终验、真值调用、来源拟合、随机抽样和下载均0；模型、精度与终端未改。几何属性接线错误和3009缺失标签解析错误已修复，失败尝试及费用保留；两次解释器启动失败CPU未计量。原终端自带的确定性诊断回归计入CPU，不更新来源或角色模型。

限制：这是已有任务的事后机制归因，不是新确认批次。不同历史的误差比较仅为描述；AFTER／POINT没有相同完整历史。Queue3009质量基线未定义，完整24任务G均值仍为null；未验策略不能补记反事实证书。保存量没有跨策略目标协方差，不能将两个边际方差相加冒充差值后验方差。

下一步[已登记、未启动](../performance/manifests/mainline_objective_contrast_exposure_v1.json)：继续冻结模型，零新增调用、恢复或拟合，复用全部来源留出预测与目标raw，拆目标差值错序、已有不确定性信息及实际比较暴露。来源中分别看已观测与未观测策略对，目标中保留全部正反例；先判断是否存在共同修正依据，再决定最小改动。主线仍是跨策略信息能否转化为决策收益。

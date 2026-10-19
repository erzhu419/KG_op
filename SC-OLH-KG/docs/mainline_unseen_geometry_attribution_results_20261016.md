# 未见几何归因：查询竞争与终端遗漏是两个不同瓶颈

[冻结登记](../performance/manifests/mainline_unseen_geometry_attribution_v1.json)下，24个任务的213次已保存角色选择全部精确重建，分数误差0。POINT3018缺失、Queue3009质量基线未定义均保留。新增仿真／终验／真值／来源拟合／模型恢复／随机抽样／raw回放／下载均0；6次小分析全部成功，累计CPU5.6753秒。[费用与产物](../paper_artifacts/mainline_unseen_geometry_attribution_v1/completion.json)。

**传播确实改变查询，但不能据赢家反推应怎样改风险均值。** [查询归因](../paper_artifacts/mainline_unseen_geometry_attribution_v1/query_diagnosis.json)中，AFTER−POINT的23个可比较任务全部首次分岔，且当时历史与物理池一致；双向仅替换全部风险分量恢复对方选择13/23、11/23，仅替换目标分量为4/23、0/23。FULL−AFTER在13/24任务分岔，对应风险10/13、9/13，目标4/13、3/13。这是固定状态分量诊断，不是新闭环；风险抽样项仅指已实现的随机扰动。

[搜索覆盖](../paper_artifacts/mainline_unseen_geometry_attribution_v1/coverage_diagnosis.json)将旧后端多获证的原因进一步定位：

| 案例 | 关键策略在AFTER池中的情况 | 实际差距 |
|---|---|---|
| Inventory3002，N21 | 三次均存在且抽样风险合格，目标抽样可行集排名6/3/9 | 目标竞争；OLD靠风险和目标抽样选中，风险均值并未更准 |
| Queue3009，N52 | 共享64点中唯一真实可行点；三次均存在，合格／排除／合格，合格时目标排第二 | 风险与目标竞争共同作用；OLD抽样选中并首次获证 |
| Queue3011，N57 | 第二次目标抽样排名第一，却被风险边际+0.087910排除 | 同历史同池下，结构后端边际−0.093637而选中；其均值明显乐观，不能据获证认定校准正确 |

[三条负例链](../paper_artifacts/mainline_unseen_geometry_attribution_v1/focal_query_deployment_chains.json)排除了更笼统的解释。3010的POINT动态赢家从初始池就存在，风险抽样扰动单独替换即可双向恢复首次选择；AFTER均值更准却被当次正扰动排除。AFTER后来查到了略好的策略，仍被终端遗漏。3013的赢家到第三次才查询，历史已经分开，不能把收益全部归于首次风险分岔。**3016的FULL／AFTER完整历史与raw完全相同，均为18→53→53；损失来自终端排序与暴露，并非查询偏移。** FULL首验部署18后停止；AFTER首验并部署质量更好的53。

[终端与来源证据](../paper_artifacts/mainline_unseen_geometry_attribution_v1/terminal_order_support.json)显示，第三位好策略共有21个策略—臂格，对应12个物理策略、8个任务；全部未验证，其中16格优于实际部署、5格更差。AFTER自身的4个第三位好点中，3010／3015／3021更好，3001更差。同AFTER历史当前终端对旧终端为3胜19平1负；3021单个损失超过三项收益，两者24任务获证状态均相同。两后端采用同一可行性优先排序公式，后验输入不同；当前停止规则只要前两位至少一个获证，就不再验证第三位。因此均值预测、排序、暴露和最后部署必须分开评价。

限制：这些任务已打开，只用于机制定位；候选状态／策略—臂格不是独立任务。未验证好点的反事实证书仍未知，改变排序还会改变槽位预算与随机流。最终paid13未保存的角色风险数值保持未知。来源留出仅保存两家族d50、四折、0/1/2次更新的role预测，没有同历史native预测或observed13终端比较；每策略3条raw也不足以伪造80／128次终验。完整24任务质量主均值仍未识别，本步不改变上一批结论。

下一步已[登记但未启动](../performance/manifests/mainline_source_observed_ranking_v1.json)：复用三份冻结来源档案及原四折整策略划分，每折48训练、16留出；留出前13点按固定次序提供各1条raw，所有后端共享initial10＋3历史，比较原native、FULL、AFTER、POINT的风险竞争集合与目标相对排序。观测点剩余2条raw、未观测3点全部raw仅在预测冻结后评分；保留全部家族／折及失败。来源拟合按原配方重做，部署模型不动，新增仿真／终验／真值调用仍为0。它检验排序机制，不构造部署证书；只有共同来源证据支持单项修正，才另冻结新任务比较。当前不采用统一风险偏移、native替换或第三位提升规则。

实现前的具体接线已登记：native来源拟合按每份档案的留出家族传参，不能沿用旧helper写死的FactorShock；native须走真实initial10拟合／task-pilot和后3次更新，仅使用许可观测。AFTER使用部署版initial10截断，不能拿旧两锚点的point_after_first替代。两个缺失状态测试已通过；来源检验本身尚未执行。

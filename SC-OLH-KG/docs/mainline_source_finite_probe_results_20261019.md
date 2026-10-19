# Stage100：固定来源探针完成并收束

[Stage98固定协议](mainline_source_finite_numerical_validation_protocol_20261019.md)的12项探针全部完成，每项一次attempt、512次原核调用，共6144次调用及积分全部返回；失败、资源停止、未返回和记录问题均0。新增10项调度/汇总替身先通过，Stage99的18项检查未重跑。7个scheduler任务t195160–t195166均done/exit0，retry与重复wrapper均0；通过node001–node006池按实时容量分配，实际落在node006，未硬绑或本地运行。

真实/置换模型在三资产组的两个实际来源家族上均完成原两点输入。最大保存误差比为 **6.398190231e-13**，低于冻结门槛1e-8；对应共享绝对误差估计1.723283360e-12、同调用minZ=2.693391878，位于Inventory资产组的置换模型、实际FactorShock来源、spectral/head_specific/observed_task的g头、节点[576,640)、第二条raw。全局最小shift后Z为2.451134569，来自另一个调用；两者不能交叉组成误差比。有界误差接受调用0。[12项状态与费用](../paper_artifacts/mainline_source_finite_numerical_validation_v1/case_status_and_costs.json)保留各项极值及位置，[结论](../paper_artifacts/mainline_source_finite_numerical_validation_v1/conclusion.json)为`no_failure_on_fixed_inputs`。

实际新增24次模型文件加载、12次来源投影、192次identity/head条件化流及6144次原积分。192是12×8身份×双头，不是独立观测数；输入只复用固定档案两条配对重复。来源attempt CPU共127.267275秒，包含在全部wrapper的132.100829秒中；截至结果封存，控制/传输另计245.740428秒，已知总CPU377.841257秒，最终文档同步另追加。worker、batch、collector、替身及控制费用分层保存在[Stage100账](../paper_artifacts/mainline_source_finite_numerical_validation_v1/cluster_control/stage100_accounting.json)，嵌套费用不重复相加，未测编辑CPU保持unknown。新增仿真、目标/候选评分、拟合、目标后验恢复及旧失败重跑均0。原核、来源资产、AFTER、前端、搜索、旧终端和全部门槛未改。

## 局限与下一步

本次只证明这组d=50、两条训练来源输入的短条件化路径完成；没有复现可供定位的异常。成功返回未暴露的逐节点Z、argmin、shifts等仍为null；共享误差估计不是精度认证，数值精度保持`not_assessed`，旧目标失败具体病因保持`cause_unidentified`。本次不覆盖d=1000 initial10、后续自身更新或结构终端，不建立新部署收益证据。

按登记规则关闭当前探针方向，不扩点、精算或筛补丁。保留全部11臂、3101完成/67missing、Queue低ESS、六套结构来源未认证状态、正反例和历史费用；来源C105/G97完整配对子集及全144来源效果null不变，旧新批分开。将本次有限结论补入[主线证据链](mainline_evidence_chain_20261018.md)：真实prior下初始化传播的已复现收益继续成立，来源关联与传播的净质量交互仍未确认；下一步围绕这些有边界的主张收束，不以追平全144归因缺口为前提重启数值排查。

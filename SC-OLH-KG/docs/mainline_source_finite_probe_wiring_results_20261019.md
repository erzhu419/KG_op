# Stage99：独立来源探针接线与替身检查

按[Stage98协议](mainline_source_finite_numerical_validation_protocol_20261019.md)完成独立入口及同调用诊断接线。唯一scheduler任务 **t190344 / node004**，18项替身通过，pytest退出0，wrapper complete/child退出0，retry_count与重复wrapper均0，controller退出0。scheduler快照仍running，按已封存wrapper确认完成并保留回收滞后，不再派发。真实探针仍为 `registered_not_started`，执行授权false；本轮没有加载真实来源资产、仿真、评分、拟合、真实积分或后验恢复。

[探针入口](../performance/mainline_source_finite_numerical_probe.py)复用既有来源投影和stream_head；[替身](../tests/test_mainline_source_finite_numerical_probe.py)覆盖原f/g配对、tau中心化、身份/双头/节点批/raw位置、终态observer不多算，以及每项512次的完整顺序。成功返回原对象，失败从同一异常帧读取已有error、integral、Z和积分状态并原样抛出；未暴露的量保持null及原因。quad入口只委托原调用并计数，以区分原核提前失败与实际进入积分。

资源检查覆盖300秒CPU、600秒整项wall、30秒单调用和512次上限。接线时修正了调用前耗尽预算却多记attempt、返回后超时丢失已有诊断、超时被误标为记录错误及截止后再次打断异常保存的问题。首次失败后停止；已封存结果只读复用，已开始但未封存的项拒绝自动重跑。

替身使用合成来源、模型、时钟和异常；涉及实际冻结kernel的两个分支完全替换quad_vec，只返回合成积分结果，不调用真实积分器或其integrand。科学实现、来源模型、AFTER、前端、搜索、旧终端和数值门槛未修改。真实入口在导入科学依赖和读取资产之前检查执行授权、node001–node006及单线程环境；检查通过不自动开放探针。

[调度入口](../../scripts/run_mainline_source_finite_probe_checks_cluster.py)只运行两份指定测试，通过既有submit-cpu-batch使用显式六节点池、1 CPU、vram0和单线程。只上传5个代码/清单文件共65,945B；本地替身运行0次。pytest CPU **2.030530秒** 嵌套在wrapper **2.243893秒** 内，控制、同步和最终状态另列于[Stage99费用账](../paper_artifacts/mainline_source_finite_numerical_validation_v1/cluster_control/stage99_accounting.json)与[完成回执](../paper_artifacts/mainline_source_finite_numerical_validation_v1/cluster_control/stage99_completion_audit.json)，不重复累计。截至结果读数已测控制/传输CPU89.645205秒，加wrapper的新总CPU91.889098秒；最终文档同步另追加到账，未测编辑CPU保持unknown。

## 局限与下一步

这证明记录与执行边界接通，尚无来源数值可执行性结果，也不构成精度认证。全部11臂、3101完成/67missing、Queue低ESS、六套结构来源未认证状态、正反例与历史费用不变；来源C105/G97完整子集及全144null不变，旧新批继续分开。

下一步可在另获执行授权后按Stage98固定12项来源探针一次执行，保留6144次总上限与所有停止条件；不补跑旧失败、不扩点、不调门槛。此次替身通过不增强来源归因或部署收益主张。

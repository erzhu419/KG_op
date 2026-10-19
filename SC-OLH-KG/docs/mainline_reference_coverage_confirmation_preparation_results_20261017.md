# Stage80：独立队列已登记，科学执行未启动

[协议](mainline_reference_coverage_confirmation_protocol_20261017.md)与[独立manifest](../performance/manifests/mainline_reference_coverage_confirmation_v1.json)已固定：三家族各8个新d=1000几何、一组新观测/终验流、原7条搜索历史及全部9个对照，交叉原/参考名单与旧/两比较终端，共864格。模型、AFTER、搜索和两种终验规则保持冻结。

`t173113`在node006完成，28项替身检查全部通过，失败和自动重试均0。已接通精确原始块复用、128→120前缀重新判证、半对失败显式续跑和四格独立记账；完整864格、缺STOP、单历史/终端失败以及禁止提前评分均通过检查。每格仍≤352；登记的四格同臂物理保守上界512，全队列成功物理上界111264，均为计划上界。

本轮新增仿真、真值评分、候选评分、拟合和模型/后验恢复均0；仅生成96个几何uniform元数据值。检查CPU11.866515秒，登记CPU0.047140秒；外层wrapper CPU有重叠，不重复相加。manifest与日志/摘要拉回38,445字节，含状态小读的已记录返回量共39,968字节；全部历史失败和费用链继承。证据见[登记摘要](../paper_artifacts/mainline_reference_coverage_confirmation_v1/registration.json)及[调度账目](../paper_artifacts/mainline_reference_coverage_confirmation_v1/scheduler_execution.json)。

本轮说明接线可用，尚无新的科学收益结果。四格能够区分“参考进入名单”与“参考得到实际比较”的作用；能否抵消安全样本减少、门槛收紧和挤出原第三点的损失，必须等待独立结果。主比较固定为AFTER联合终端−原终端的C和带符号G，同时保留每格AFTER−POINT；通用终端改善不能替代prior传播收益。

下一步将这些接口接入现有分阶段scheduler执行与封存后的四格汇总，先用替身核对主汇总、缺失分母和费用，再执行已登记队列。全部864格终验或失败封存前不读取真值，不更换种子、不调整模型或门槛。

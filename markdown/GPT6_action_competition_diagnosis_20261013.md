# SC-OLH-KG：动作竞争长期未闭合的现状、原因与待评估方案

生成时间：2026-10-13，Asia/Shanghai。最新状态核查约22:12。

仓库：<https://github.com/erzhu419/KG_op>。本次代码和证据所在分支：`codex/sc-olh-kg-performance-validation`，请查看这个分支。

本文给外部GPT6提供可以直接判断的交接材料。事实、代码推导和待验证方案分开书写；文内数值来自已保存的开发实验。区间、计值和“通过”的范围均限于各自的信息合同。

## 1. 先说明我们真正想做什么

原始目标是：从历史/source任务中学习可迁移的低自由度结构先验，包括状态耦合、函数分解和异方差风险结构；用它改善target后验，并使KG采样、终端推荐及机会约束验证服务于同一个决策目标，在少量target数据下产生可验证的决策收益。

这条主线来自[GPT_roadmap.md](GPT_roadmap.md)和[GPT_meta_assumption.md](GPT_meta_assumption.md)。当前困难不是希望把项目转成一篇纯求积或纯有限动作筛查论文。现在做raw物理风险计值和动作竞争，是为了修复原始meta-prior/KG决策链的数值前提。

历史FactorShock、Inventory、Queue的60/60及前端提高不同online/transfer算法的结果，属于已完成但合同有界的旧证据。它们不能直接证明当前完整raw后验下的KG增量、独立部署优势或高维自由优化已经成立；同时，也不应因为当前数值验证困难就废弃这些历史模块，另造一套替代研究目标。

**当前核心判断：方法主线仍保留了原source prior，但执行已经长期停留在数值验证；必须判断下一步是否应改进界与验收问题，而不是继续无限增加逐组积分。**

## 2. 最新状态：完成计算、通过精度、竞争闭合是三件事

| 项目 | 最新核实结果 |
|---|---|
| 已完整结算 | 第1–59阶段 |
| 当前控制器 | 第60阶段结算重放的远程调用超时，状态为`failed` |
| 第60阶段条件参考 | 全部通过，最大规则差`9.5410946910e-5` |
| 第60阶段包络检查 | `numerical_containment_passed=true` |
| 第60阶段服务器重放 | 文件已存在，约1,597,344字节，内部计时23.2052 CPU秒 |
| 第60阶段完整费用结算 | 尚未完成，不能当作第60阶段已结算 |
| 原登记上限 | 到第61阶段；第61阶段尚未启动 |
| 第59阶段raw未排除收益上界差 | `0.0038459104580`，370个观测与动作组组合 |
| 第60阶段raw未排除收益上界差 | `0.0037727248185`，367个观测与动作组组合 |
| 第60阶段projected未排除上界差 | `0.0035070604192`，345个观测与动作组组合 |
| raw推荐验证contrast | `0.0211343565109`，仍未改变 |
| 完整预测积分验收 | 未完成 |
| 当前合同下的独立确认 | 未读取 |

第60阶段未结算是运行中断，不是参考精度失败。它也不能解释科学上的未闭合：服务器已经算出的raw包络仍留有`0.00377272`的正差。

下面的**详细区域与组别诊断冻结在已结算的第58阶段**，这样数据来源完整、一致、可复核。第59、60阶段只作为最新进度附加报告，不将不同阶段的区域统计混合。

## 3. 当前冻结的信息合同

| 项目 | 合同 |
|---|---|
| 开发状态与query | 一个已打开的开发状态，固定query/policy为63 |
| 本次观察批量 | 4个replicates；对应`source_n4` |
| 终端预算 | 剩余356次calls；365个登记动作 |
| 风险外层网格 | 共224个风险观测点；来源4、5各112点 |
| 目标外层 | 每个来源保留16个已缓存objective heads，即F16 |
| 来源身份似然 | 所有8个identity的likelihood均保留 |
| 主要utility积分身份 | identity4、5，分别512、1024个正权重source分量 |
| 协方差 | 原始跨策略联合协方差保留 |
| 原source prior | 保留；没有为节省计算把模型改成另一个prior |
| 验证器 | 原SD16机会约束验证器及其预算规则 |
| 条件参考精度门 | 规则比较最大绝对差不超过`1e-4` |
| 单阶段预算 | 估计不超过3600 CPU秒，包含唯一600秒共享余量 |
| 并行 | 最多8个低优先级单核任务；OMP/BLAS均为1 |
| 节点 | node001–node006，通过scheduler，不用Slurm |

来源先验权重的主要质量为identity4的`0.6964353220`和identity5的`0.3035646780`。未积分的外层来源先验质量约`5.9013e-12`；重放中的最大遗漏utility identity后验质量为`1.3891e-9`。这是两个不同层次的量，不能混用。筛查上界显式加入未计算utility身份的后验质量，因此这些微小遗漏不能解释当前约`0.0038`的剩余差。

## 4. “竞争未闭合”在代码里究竟是什么意思

### 4.1 每个条件节点的动作计值

对风险观测r、目标head j、终端动作a，代码用所有identity的观测似然计算后验权重。动作收益通过目标quality与风险/认证分支统计量组合，再按identity后验权重加权。

可以简写为：

```text
V(j,r,a) = sum_h posterior_weight(h | objective_head_j, risk_observation_r)
                  * dot(objective_quality(j,h,a), risk_branch_statistics(r,h,a))
```

直接动作与guarded动作的分支统计量不同；guarded动作含多个互斥部署分支及paired分支。原始矩阵与预算均沿用，不能把它当成无约束目标均值的普通argmax。

### 4.2 已积分动作给出的参考值和未积分动作的上界

每个r、j都有已验证动作集合A_ref，包括固定动作355、已完成的query相关参考，以及逐阶段通过的joint动作组。定义：

```text
L(j,r) = max_{a in A_ref(r)} V(j,r,a)
U(j,r,a) = 尚未完整积分动作的边际认证/分支收益上界
```

当前`collect()`使用：

```python
overlap = upper + 2e-4 >= fixed[:, None]
overlap[:, list(references)] = False
action_overlap = overlap.any(axis=0)
```

这里的`fixed`在加入全部已通过参考后，已成为每个head的L，而不只是最初固定动作355。

只要一个未参考动作在任何objective head上满足`U + 2e-4 >= L`，它就保留为候选。动作映射为共享shortlist/结构的组后，得到`record_group_candidates`。

所以“373个组”不是373种新的模型或373种独立策略，而是**风险观测r与动作组g的组合数**。

### 4.3 全局剩余差与严格闭合

令w(r)为来源先验权重乘以该风险CDF节点的正权重，则代码汇总：

```text
C_ref = sum_r w(r) * mean_j[L(j,r) - V(j,r,355)]
G     = sum_r w(r) * mean_j[max(0, max_{a not referenced} U(j,r,a) - L(j,r))]
```

`weighted_bank_envelope.kinds.raw.unresolved_upper_gap`就是G。

但控制器停止条件不是`G <= 1e-4`，而是raw与projected两种计值下的所有`record_group_candidates`都为零。也就是说，目前要求两个版本都在每个风险节点、每个objective head上排除所有未积分动作。

必须区分：

1. 一个参考积分通过`1e-4`规则差门。
2. 当前有限网格上的优化期望可能已接近某个误差目标。
3. 所有条件节点上的全部动作都已严格排除。
4. 连续预测分布和F16之外也已收敛。

目前只完成了一部分第1项。第3项未完成，第4项也未完成。当前G仍约为`1e-4`的38倍左右，所以即使讨论较弱的全局epsilon验收，也不能立即宣布通过。

另需注意：已有规则差检查是数值稳定性诊断，不自动等价于每个参考值都有严格、统一的绝对误差证书。当前包络不是统计置信区间，也不是连续积分误差界。

## 5. 上界的来源：为什么“算得准”仍可能“排除不掉”

核心实现在[core/marginal_certificate_bound.py](../SC-OLH-KG/core/marginal_certificate_bound.py)。它依次放松了三个问题。

### 5.1 物理可行性事件的Gaussian半空间上界

真实可行性涉及`g + z * exp(eta/2) <= 0`。代码利用`exp(eta/2)`的凸切线，将真实事件包在Gaussian半空间内，对多个切点计算概率后取最小上界。

raw版本显式保留全部正权重source分量，进行条件化和Gauss积分；没有靠删除困难分量换取精度。64/128阶边际积分差及归一化诊断通过，说明这个上界的数值计算稳定，**不说明半空间放松本身已足够紧**。

### 5.2 真实认证概率的可行性包络

令b为实际保存的SD16规则在物理可行性边界上的认证概率，P_feas_up为上一层的可行性上界，则：

```text
P(certificate passes) <= b + (1-b) * P_feas_up
```

这使用的是实际SD16边界power，不是直接把名义delta当作边界概率。

但放松在可行事件内允许认证概率达到1。有限预算、接近边界和方差不确定性下，实际认证概率可能明显小于1，因此这一层还会留下松弛。

### 5.3 guarded联合分支的fractional-knapsack放松

多个分支的边际认证caps，以及paired分支的`min(cap_primary, cap_incumbent)`，被放入按quality排序的fractional-knapsack式上界。

这保留互斥分支总质量不超过1的限制，但没有恢复所有原始联合事件约束。跨策略相关性虽然仍存在于真正的后验和精细参考积分中，却没有全部进入这个便宜的筛查界。

**代码上可以确认：数值误差与界的结构性松弛是两件事。当前结果尚未证明哪个放松占最大比例；需要用已缓存参考做归因，而不能把“界过松”直接写成已经测定的结论。**

## 6. 第58阶段的可复核诊断

只读导出文件：[diagnosis_summary.json](../SC-OLH-KG/paper_artifacts/historical_positive_risk_uppervariance_axis_competition_v58/diagnosis_summary.json)。它在服务器读取现有缓存，仅返回约33 KB；没有重新积分、拟合source或读取独立确认。

### 6.1 进度与成本

| 阶段 | raw剩余上界差G | raw已接纳最佳contrast |
|---:|---:|---:|
| 1 | 0.0176784003 | 0.0211139786 |
| 21 | 0.0097465603 | 0.0211348904 |
| 35 | 0.0071770237 | 0.0211348904 |
| 41 | 0.0062830980 | 0.0211348904 |
| 45 | 0.0057354134 | 0.0211348904 |
| 50 | 0.0049939014 | 0.0211348904 |
| 58 | 0.0038859135 | 0.0211348904 |
| 59 | 0.0038459105 | 0.0211348904 |
| 60，未结算 | 0.0037727248 | 0.0211348904 |

第58阶段累计785个参考worker、391个组检查，竞争接续账记录103,518.87 CPU秒，约28.76 CPU小时。第59阶段再增加1,499.51 CPU秒。

这不是整个项目的总费用：此前均值/方差加密、原bank构建、source训练等费用在其他账中。GH16验证另记录187.32 CPU秒；第42阶段还有35秒无法恢复完整进程marker的保守余量收费。

已接纳最佳contrast自第20阶段后保持不变，而G持续下降。这支持“后续大量工作主要在排除上界假象”的判断，但仍不能证明未积分动作不会真正更好。

以已接纳数值规则下的第58阶段包络表示，参考contrast约为`0.02113489`，名义上界约为`0.02502080`。其间的`0.00388591`是待排除空间，不能当作已经获得的额外收益。

### 6.2 raw剩余差按均值CDF区间分布

| 均值CDF区间 | 风险记录数 | 剩余观测与组组合 | 加权G贡献 | 占总G |
|---|---:|---:|---:|---:|
| [0,0.8] | 32 | 12 | 0.0001599138 | 4.12% |
| [0.8,0.875] | 32 | 24 | 0.0004455056 | 11.46% |
| [0.875,0.9125] | 32 | 43 | 0.0005193759 | 13.37% |
| [0.9125,0.95] | 32 | 52 | 0.0006124253 | 15.76% |
| [0.95,0.99] | 64 | 159 | 0.0014077244 | 36.23% |
| [0.99,1] | 32 | 83 | 0.0007409686 | 19.07% |

约55.29%的剩余差位于均值CDF的0.95以上区域；约40.59%位于[0.8,0.95]。不是只剩极端尾部，也不是主体区间完全没有问题。

按方差CDF粗分，[0,0.2]、[0.2,0.8]、[0.8,1]分别贡献`0.0012548074`、`0.0013938117`、`0.0012372945`。因此当前残差没有集中到唯一一个方差尾部区间。

来源4、5分别贡献`0.0017566302`和`0.0021292833`。来源5先验质量较小但剩余差更大，又有1024个分量，确实是较重的一部分计算负担。

### 6.3 剩余动作的结构高度重复

| 组ID | 类型 | 策略shortlist | 仍竞争的风险观测数 |
|---:|---|---|---:|
| 71 | guarded | [71,45,10] | 98 |
| 34 | guarded | [34,45,10] | 97 |
| 63 | guarded | [63,45,10] | 53 |
| 4 | guarded | [4,45,10] | 34 |
| 107 | direct | [34] | 39 |
| 144 | direct | [71] | 35 |
| 39 | guarded | [39,45,10] | 13 |
| 12 | guarded | [12,45,10] | 4 |

group ID、action index和policy index不是同一编号。例如guarded组34对应动作[34,107,180]；direct组107对应动作[253,326]、primary为34。

最重的guarded组共同使用45、10。现有流程已按shortlist共享原始边际、模板及组内多个预算动作的计算，但**没有因此得到足够紧的跨组联合认证上界**。这是下一步最值得评估的历史结构复用入口。

导出中保留每组的`weighted_group_priority`。它是各head平均正上界差的调度量；不同组在同一观测上重叠，不能把这些值相加并声称是总G的互斥百分比。

### 6.4 条件松弛可以很大，但单个节点质量很小

一个典型残留记录是`positive_uppervariance_v1_n4_m4_localv_o4_h1_p72`：

- CDF坐标约`[0.9632003791, 0.0198550718]`。
- 外层来源与节点合成权重约`0.0004597566`。
- head平均未排除条件收益差约`0.10557939`。
- 加权贡献约`4.85408e-5`。
- 尚存组[4,12,34,39,71]。

条件差是约0.1量级，远大于`1e-4`参考规则差；但一个节点对全局期望的影响约为`4.85e-5`。这说明点态全部排除与控制总体期望误差可能需要非常不同的计算量。

第58阶段没有“正剩余差已经为0但仅因`+2e-4`而仍有候选”的风险记录。故这个严格判据可能拖慢最终收尾，但不能解释当前全部正残差。

## 7. 推荐稳定到底说明什么

| 推荐方式 | raw验证contrast | 与projected推荐相比改选head数 | 验证gain低于−1e-4的加权质量 |
|---|---:|---:|---:|
| projected quality + projected risk | 0.0210927388 | 0 | 0.0242746660 |
| raw quality + raw risk | 0.0211343565 | 64 | 0 |
| raw quality + projected risk | 0.0210927388 | 0 | 0.0242746660 |
| projected quality + raw risk | 0.0211343565 | 64 | 0 |

这支持一个局部归因：改选由风险计值替换复现，单独替换目标quality没有复现；当前共同已接纳动作池的raw推荐避免了该网格上已识别的负gain质量。

但有三个重要边界：

1. `0.0211343565`是相对固定动作355的期望contrast，不是独立部署效果，也不是跨任务source prior优势。它甚至不是最终扣除采样机会成本后的净KG。
2. 当前固定动作control variate的采样机会损失为`5.5586761e-5`。按已有公式扣除后，有限规则下的raw推荐净preview约为`0.0210787697`。仍需完整外层、其他query/batch选择和独立实验。
3. raw与projected推荐的平均contrast差只有约`4.16178e-5`，小于当前单项`1e-4`规则差门。不能只凭这个均值差宣布精确或显著的优越性。改选、负gain质量和平均收益差也不是同一指标。

这里的零负gain质量并不代表实际机会约束违约率为零。它属于这个开发状态、有限预测网格、已接纳候选池内的诊断。

## 8. 已经做过的工作，避免重复造轮子

### 8.1 v3指出的混合分量覆盖问题已采用显式正权重方式处理

[GPT_diagnosis_v3.md](GPT_diagnosis_v3.md)指出，旧Sobol点可能在512/1024分量下每个分量内点数过少。当前条件参考保留所有正权重分量，使用分量内条件积分和独立连续规则比较。不能把旧v3中`4.15e-3`的动作分辨率差当成当前阶段仍然同样失败。

当前已通过的参考差一般在`1e-5`至`1e-4`之间。**新的主要残差G是未积分动作的收益上界差，不是那一批旧的积分分辨率差。**

### 8.2 原始观测CDF的正权重局部加密

风险网格已经历32点基线及主体/肩部/均值/方差局部加密。几个重要记录：

| 网格 | raw推荐验证contrast | 相邻差异 |
|---:|---:|---:|
| 96 | 0.0215112414 | 主体加密后的值 |
| 128 | 0.0212087861 | 仍有变化 |
| 160 | 0.0210933397 | 相对128为−0.0001154464，未过门 |
| 192 | 0.0211341582 | +0.0000408185，局部均值比较通过 |
| 224 | 0.0211139467 | −0.0000202115，局部方差比较通过 |

后续224点竞争参考将推荐验证contrast补到`0.0211343565`。这些是局部轴、当时已接纳池内的比较，不是所有方差区间或F16之外均已收敛。

### 8.3 历史动作竞争流程已经复用并跑通

旧32点完整网格曾在`historical_positive_risk_full_grid_screen_competition_v19_refs_v1`关闭全动作竞争。其控制器登记的15个已结算阶段记录26,703.58 CPU秒；这不是所有前置阶段的总费用，不能与当前58阶段总账直接作完整成本比。

当前224点复用了同一组筛查、精细参考、预算内续接和结算机制。风险节点数扩大7倍，同一动作组出现在多个新增CDF坐标上；旧坐标上通过的参考不能自动替代新条件分布的参考。流程不是从头重造，但点态证书的计算量确实显著扩大。

### 8.4 调度优先级已从最坏head改为平均正差

原来用一个最坏head的上界差排序；现按原风险概率权重乘以16个head的平均正group gap排序。它保持候选集合、零候选闭合标准及精度门不变。该调整减少了单个异常head过度控制优先级的问题，但并未使上界变紧。

### 8.5 第35阶段真正的精度失败在原预算内解决

E9/F9曾差`1.08991e-4`，不能直接通过。增加F10后仍需修正主规则；最终只对identity4增加E10，identity5保留E9，使用逐身份规则及F10验证。

最终主序列加密差`2.10500e-5`、对独立参考差`9.81230e-5`，累计估计3576.63 CPU秒，保留原600秒余量，旧失败和补算费用未移动到新阶段。

### 8.6 历史GH16对照已适配

六个n4分层、12个新worker的GH16对照最大差`1.89348e-6`，原选用统计量完全保留。Gauss对照部分CPU从767.08降至169.81秒，节省约77.86%。第42阶段后使用E8/F8/L8/E9或E8/E9/F9/L9，并继续逐记录检查`1e-4`。

这个节省是Gauss对照部分的，不是整个流程提速78%。它主要降低每组验证费用，仍没有修复边际包络松弛和严格零候选判据。

### 8.7 运行中断已与科学门区分

已经修复过：全动作汇总35秒超时、scheduler成功恢复任务后日志路径为空、准备任务派发锁等待超时。费用仍绑定原阶段，已成功任务复用。

最新第60阶段的重放35秒调用超时尚未完成恢复。服务器重放已完成，完整文件留在服务器；本次只取回约1 KB的[replay_server_summary.json](../SC-OLH-KG/paper_artifacts/historical_positive_risk_uppervariance_axis_competition_v60/replay_server_summary.json)。它确认推荐及未闭合状态，但不能替代费用结算。

这些运行问题使自动接续间歇停止，但不是G持续为正的数学原因。

## 9. 目前最可能的主要瓶颈

### 已确认

- 多数数值参考已经能在现有规则比较下通过；失败不能一概归因于当前Sobol点数仍不够。
- 上界通过三个放松构造；精度通过不会自动消除放松。
- 主要剩余guarded结构高度重复且共享45、10。
- 第58阶段仍有很大的条件上界差和正的全局差；不是全部都只剩近乎相同动作的数值打平。
- 已接纳最佳contrast从第20阶段起保持稳定，后续许多积分主要减少未知上界空间。
- 当前控制器同时追求raw和projected两套全条件节点的零候选。

### 有证据支持，但尚未直接量化归因

1. **联合分支界过松。** 仅用边际caps和paired的min值，可能无法表达共享incumbent下真正的分支相关性，导致很多实际较差的组一直看起来有竞争力。
2. **当前验收问题可能强于原科学问题所需。** 目标是KG期望及最终推荐，而程序要求所有条件head上的动作逐个排除。它可能消耗大量时间证明低权重情形的点态最优。
3. **同时闭合projected银行可能分散预算。** 若科学主目标是raw决策，projected作为对照是否需要同等严格的全动作证书，需要按论文最终主张判断。不能直接删掉，也不能默认它一定必要。
4. **跨CDF节点的联合结构复用还不充分。** 目前充分复用原始bank和相同坐标参考，却没有利用共享anchor结构产生可传播的更紧区间界。

不能据此声称：所有未知动作都一定差、source prior已取得稳定泛化收益、放松阈值即可通过，或该项目整体已经失败。

## 10. 下一步值得评估的方案与代价

这些是待讨论的方案，不是本次已经实施的算法修改。

### A. 先用已缓存参考分解上界松弛

选均值CDF上部、组71/34/63附近已经精确验证过的少量观测，比较：物理可行性半空间界、实际认证cap、guarded分支放松、已缓存的joint统计量。

目的是确认主要松弛来自哪个层次，以及收紧该层能否排掉当前高优先级候选。优先只读缓存，不新增物理仿真或source拟合。

### B. 利用共享45、10改善guarded联合界

评估能否按同一观测、identity、预算档位先计算共享anchor的联合认证信息，再约束其他primary的分支质量分配；或利用已有跨策略联合协方差构造比边际min/knapsack更紧的上界。

需要证明新界是原后验和原SD16验证器下的上界，并在已有精确joint参考上检验没有低估；不能为了收紧界偷换协方差或假设策略独立。

### C. 明确真正需要的闭合标准与误差账

讨论是否应区分：有限网格的总体期望epsilon最优、最终query/batch排名稳定、每个条件节点的严格动作排除。

若科学主张只需要前两者，是否可以用概率加权误差预算和当前G停止部分点态排除？如何同时计入条件参考误差、尚未积分动作差、风险外层误差及F16误差？

这需要重新明确数值合同，不能事后把旧门槛放宽并将当前结果宣布成功。当前G本身也仍明显大于`1e-4`。

### D. 判断projected需要验证到什么程度

原projected推荐的raw覆盖已经补齐。作为基线，是否应验证其实际算法产生的推荐，而不是对整个projected后验银行求点態零候选？这会不会影响公平性或破坏当前拟定主张？请给出需要保留的具体验证环节。

### E. 保留原主线，设定回到独立决策实验的条件

数值问题解决后仍需：其他风险区间、F16收敛、不同query/batch的真实KG选择、相同合同下的强基线，以及未参与调参的独立任务/种子验证。

当前raw与projected的局部对照没有测出“source-learned prior相对target-only等基线”的稳定优势。最终论文必须接回这个问题；动作竞争闭合本身不是原始研究目标。

## 11. 供GPT6直接回答的问题

1. 当前`零候选×每个风险点×每个head×raw/projected`的停止标准，是否比原KG决策主张所需明显更强？请给出替代标准需要证明的条件与总误差账，而不是直接建议放宽阈值。
2. 根据实际边际可行性界、SD16认证cap及guarded knapsack代码，最值得收紧的是哪一层？能否利用共享[45,10]得到有效的跨组联合上界？
3. 在不改变source prior、正权重分量、原联合协方差及验证器的前提下，是否有可以复用旧积分、避免再对数百个观测与组组合全算的方案？
4. 我们是否应该先把raw全局期望gap降到登记误差目标，再判断是否需要逐head严格闭合？这样能支撑哪些论文主张，不能支撑哪些？
5. projected作为对照，合理的验证要求到底是什么？当前原native推荐raw验证齐全，是否仍需要完整projected银行的点态最优证书？
6. raw推荐与projected推荐的平均contrast差约`4.16e-5`，而参考门为`1e-4`。后续应怎样设计paired精度或独立验证，才能判断这部分改进是否有可靠的决策意义？
7. 下一阶段最小的、可以否证上述判断的诊断是什么？请优先提出少量缓存分析和小规模对照，不再笼统建议全体升Sobol/Gauss阶数或加密所有坐标。
8. 如何把原有60/60与前端跨算法收益、当前完整raw风险计值、未来真正KG增量接成同一科学主线，而不把不同合同的数据拼成统一优势结论？

## 12. 代码与证据入口

| 入口 | 用途 |
|---|---|
| [screen_historical_observed_batch_risk.py](../SC-OLH-KG/performance/screen_historical_observed_batch_risk.py) | `collect`的L/U、候选和G；`next_plan`及组定价 |
| [marginal_certificate_bound.py](../SC-OLH-KG/core/marginal_certificate_bound.py) | 半空间、SD16认证及guarded分支界 |
| [check_historical_observed_batch_risk.py](../SC-OLH-KG/performance/check_historical_observed_batch_risk.py) | n4条件参考、独立规则差及逐身份加密 |
| [validate_historical_component_bank.py](../SC-OLH-KG/performance/validate_historical_component_bank.py) | 规则定义、组共享和实际参考选择 |
| [replay_historical_positive_risk_subset.py](../SC-OLH-KG/performance/replay_historical_positive_risk_subset.py) | 共同池raw推荐、验证及目标/风险替换归因 |
| [raw_terminal_head_grid.py](../SC-OLH-KG/core/raw_terminal_head_grid.py) | raw目标/风险head的合成与选择 |
| [probe_historical_low_gauss_reference.py](../SC-OLH-KG/performance/probe_historical_low_gauss_reference.py) | 历史GH16适配验证 |
| [summarize_historical_competition_snapshot.py](../SC-OLH-KG/performance/summarize_historical_competition_snapshot.py) | 本文33 KB只读诊断的生成入口 |
| [continue_when_ready.py](../SC-OLH-KG/paper_artifacts/historical_positive_risk_full_grid_continuation_v1/continue_when_ready.py) | 接续、预算、停止条件与checkpoint |
| [close_phase.py](../SC-OLH-KG/paper_artifacts/historical_positive_risk_full_grid_continuation_v1/close_phase.py) | 资源与失败费用结算、推荐重放 |
| [阶段58 completion.json](../SC-OLH-KG/paper_artifacts/historical_positive_risk_uppervariance_axis_competition_v58/completion.json) | 冻结诊断的完整结算 |
| [阶段59 completion.json](../SC-OLH-KG/paper_artifacts/historical_positive_risk_uppervariance_axis_competition_v59/completion.json) | 最新完整结算 |
| [阶段60 compact_summary.json](../SC-OLH-KG/paper_artifacts/historical_positive_risk_uppervariance_axis_screen_competition_v60/compact_summary.json) | 最新包络，尚未结算 |
| [diagnosis_summary.json](../SC-OLH-KG/paper_artifacts/historical_positive_risk_uppervariance_axis_competition_v58/diagnosis_summary.json) | 全部58阶段历史、区域/组别摘要 |
| [运行状态](../SC-OLH-KG/paper_artifacts/historical_positive_risk_uppervariance_axis_continuation_v1/status.json) | 第60阶段超时的checkpoint |
| [局部轴与接续记录](../SC-OLH-KG/docs/historical_local_raw_axis_refinement_20261012.md) | 已实施修复及通过范围 |

在现有服务器工作目录下可重放只读诊断，数据需要原始artifact bank：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  /home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python \
  -m performance.summarize_historical_competition_snapshot \
  --plan performance/manifests/historical_positive_risk_uppervariance_axis_screen_competition_v58.json \
  --output paper_artifacts/historical_positive_risk_uppervariance_axis_competition_v58/diagnosis_summary.json
```

服务器工作目录：`/home/zhengliang01/scheduleurm_work/KG_op_independent_kg_parallel_deploy/SC-OLH-KG`。

## 13. 投稿与数据边界

当前离Operations Research的核心缺口，仍是完整且可信的KG决策计算、同合同强基线以及稳定独立决策收益；不是把现有`1e-4`参考门都跑绿即可投稿。轨迹/因子级风险恢复、高维自由优化和跨任务泛化也不能由这个单状态有限银行推导。

本次Git更新覆盖项目代码、论文既有版本、配置、实验计划、测试以及紧凑证据。已有大型原始预测、后验和积分数组继续留在本地/服务器artifact目录；约2.4 GiB新增缓存不随本次Git推送。已跟踪的历史文件保留，新忽略规则不删除数据。

因此，外部GPT可以凭本文、代码和紧凑JSON分析数学及工程问题，但不能声称已独立重跑完整posterior银行。仓库里的早期独立实验也不能代替当前合同下尚未打开的确认块。

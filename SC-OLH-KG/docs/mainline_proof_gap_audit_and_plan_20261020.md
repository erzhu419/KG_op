# Stage104：当前统一方法的证明审查与最小补证计划

**当前已建立“来源关系→未采样策略差值→查询→实际终端”的实现对应证明接口。** 下表把代码、数学命题、Lean声明和正文主张逐项连接；已有表示、覆盖、条件化和独立验证引理保留各自适用条件。AFTER相对POINT、同源GP及旧后端的真实收益，继续由冻结独立实验支持。

补证进度（2026-10-20）：最近一批新增来源核PSD／限制／role拼接21条、真实期望下的固定仿射MSE分解15条，三批累计94条，16模块依赖闭包通过Lean 4.31.0。12条核心声明仅依赖`propext`、`Classical.choice`、`Quot.sound`。1次缓存计划和3次人工构建（2次失败）实测58.321 CPU秒，自动重派为零；[最近构建回执](../paper_artifacts/mainline_source_kernel_mismatch_lean_20261020_result.json)记录完整费用，[前轮](../paper_artifacts/mainline_stability_lean_20261020_result.json)及[首批](../paper_artifacts/mainline_lean_20261020_result.json)记录保留，历史余量未重置。

审查依据为[统一方法](mainline_unified_method_and_plan_20261019.md)、[方法稿](mainline_methods_working_draft_20261019.md)、实际代码、Lean源文件及既有构建回执。[三版本讨论](../../markdown/GPT_3version_merge.md)是待核对的建议材料。审查初稿之后新增了上述Lean模块及集群核验；冻结算法、历史论文和科学记录保持不变，未新增统计、仿真、评分、拟合、积分或后验恢复。本地审阅与文档编辑费用未单独计量；Lean费用单列，历史科学费用与余量不变。

## 1. 已有什么，缺口在哪里

既有[历史构建回执](../paper_artifacts/lean_proof_receipt_v1.json)记录102个Lean文件、构建成功和零禁止声明。当前主线三批闭包分别为12、14、16模块，相关旧定理正文与Atlas接口保留。最近一次成功构建复用14个项目模块，仅重编修复后的MSE模块及接口；94条当前新增定理无占位或新公理。以下列出实现桥与保留前提，具体范围见两份数学正文、当前模块及各轮回执。

### 实现—数学—Lean—正文对应

下表对应[方法稿的四项命题及失配解释](mainline_methods_working_draft_20261019.md#当前实现的证明接口)。声明列省略前缀`SCOLHKG.Real`；终端使用`SCOLHKG.Measure`。链接指向各模块，所列当前声明均由[当前接口](../../proof/SCOLHKG/CurrentMainlineProofInterface.lean)引用。

| 实现位置 | 数学命题 | 当前Lean声明 | 可引用的正文主张 |
|---|---|---|---|
| [来源投影_project_pair](../performance/mainline_legacy_source_models.py#L269)、[role_inputs](../performance/mainline_role_prior.py#L31) | 特征投影、白项、有限混合及独立分块替换保持PSD；限制到子池与直接投影一致 | [CurrentSourceKernel](../../proof/SCOLHKG/Real/CurrentSourceKernel.lean)：`source_covariance_psd`、`mixture_covariance_psd`、`role_replacement_covariance_psd`、`source_covariance_restriction` | 命题1：给定合法来源资产矩，工作协方差有效；物理白项与索引对角阵相同需策略互异，混合概率解释另需归一权重及混合均值 |
| [apply_raw_moments](../representation/vectorized_joint_function_risk.py#L101) | 给定局部矩变化与loading，目标差均值和方差按两行loading之差更新 | [CurrentPropagation](../../proof/SCOLHKG/Real/CurrentPropagation.lean)：`local_completion_contrast_mean`、`local_completion_contrast_variance` | 命题2：共享关系的差异控制组件内未采样策略差值传播 |
| [节点权重更新](../representation/vectorized_joint_function_risk.py#L123)、[身份证据更新](../representation/joint_response_mixture.py#L65) | 给定正证据归一常数，新旧混合差值拆为组件内更新与身份／节点重权两项 | [CurrentPropagation](../../proof/SCOLHKG/Real/CurrentPropagation.lean)：`evidencePosterior_normalized`、`evidence_local_completion_contrast_decomposition` | 命题2：相同loading仅消除组件内差值变化，重权仍可能改变混合差值 |
| [AFTER缩放](../performance/mainline_role_after_init.py#L202)、[POINT缩放](../performance/mainline_role_point_only.py#L106) | `D C_base D`保持PSD，匹配边缘方差，并保持正方差坐标的基线相关性 | [CurrentCovariance](../../proof/SCOLHKG/Real/CurrentCovariance.lean)：`fixed_correlation_diagonal`、`scaled_correlation_preserved`；PSD由接口引用的`scaled_covariance_psd`给出 | 命题1：AFTER以initial10、POINT以source0为基准；匹配对角需基线正对角和非负局部方差 |
| [完整公共特征图](../baselines/transfer_archive.py#L50)、[十维截取](../representation/legacy_function_prior.py#L15)、[风险描述](../representation/legacy_observable_prior.py#L14) | 冻结线性／二次／正尺度／径向变换的误差传到均值与核 | [CurrentRepresentationStability](../../proof/SCOLHKG/Real/CurrentRepresentationStability.lean)：`radial_normalization_coordinate_error`、`cross_dimension_source_prediction_error`、`bilinear_kernel_error` | 命题3：公共十维在完整65列归一化后截取；保留重建、求积、分母变化及当前两输入间encoder误差前提 |
| [高斯抽样器](../acquisition/decision_backends.py#L229)、[约束TS](../acquisition/decision_backends.py#L1595) | 同池同随机向量的抽样误差界，经可行性与分数间隔传到查询一致 | [CurrentQueryStability](../../proof/SCOLHKG/Real/CurrentQueryStability.lean)：`coupled_draw_pool_error_le`、`chance_margin_error_le`、`feasible_branch_query_stable`、`minimum_violation_query_stable` | 命题3：足够大间隔下当前一步查询稳定；可行分支保留sentinel条件，回退分支使用含机器eps的完整loss |
| [实际终端分支](../core/terminal_verification.py#L840)、[独立块标签](../core/terminal_verification.py#L695) | 互斥门控共享第三份错误预算，连接certified返回与208／224／336次调用 | [CurrentTerminalGate](../../proof/SCOLHKG/Measure/CurrentTerminalGate.lean)：`current_certified_return_or_false_improvement_five_percent`、`current_terminal_calls_branch_identity`、`current_terminal_abstention` | 命题4：局部条件错误界下，单格“获证且不安全或错误宣布改善”联合概率不超过5% |
| 固定来源仿射预测器；解释性命题，无新增生产步骤 | 真实目标分布下MSE＝最优线性残差＋非负关系失配＋均值偏差平方 | [CurrentAffineMismatch](../../proof/SCOLHKG/Real/CurrentAffineMismatch.lean)：`fixed_affine_mse_decomposition`、`relationship_mismatch_nonnegative` | 失配解释：来源关系可能帮助或损害预测；目标矩、可积性及正规方程为条件 |

前端仍复用[ProfileCoordinateConsistency](../../proof/SCOLHKG/Real/ProfileCoordinateConsistency.lean)的条件重建界和[GeometricAtlasCoverage](../../proof/SCOLHKG/Real/GeometricAtlasCoverage.lean)的实际半径条件；当前加权选点不满足纯Gonzalez算法桥。[TwoStageDecision](../../proof/SCOLHKG/Real/TwoStageDecision.lean)的端到端工具保留原假设，发现不足、名单遗漏及未比较项不因本表消失。

[旧理论合同](../../proof/final_or_theory.md#L3)明确以V2 SourceScoredProfileAtlas为对象；[PaperProofInterface](../../proof/SCOLHKG/PaperProofInterface.lean)主要导出Atlas表示／覆盖及all-success二项认证。当前是LearnedMetaPrior前端、role来源prior、AFTER、约束TS与高斯终验，已单独建立上述当前接口；历史文件保持原身份。

## 2. 第一组：先证明prior实际建立了什么关系

### 2.1 冻结来源先验与role拼接的有效性

条件于已经学好的来源资产，对身份／节点j写出实际均值头核：

\[
K_j(x,x')=\phi_j(x)^\top(\Sigma_j+T_j)\phi_j(x')+v_j\mathbf1\{x=x'\},
\qquad K=\sum_jw_jK_j+\operatorname{Cov}_{w}(m_j).
\]

在归一化非负权重、PSD系数／任务协方差和非负白噪声项下，已证明有限候选池上的PSD和限制一致性。相关部分有有限特征表示；有白噪声时整个核不必低秩。保留身份／节点权重并替换成合法native-g边际仍定义合法工作先验。依据为[来源投影](../performance/mainline_legacy_source_models.py#L269)和[role替换](../performance/mainline_role_prior.py#L31)。

这解释四类结构如何限定函数关系：正交组织基、有限支持限制复杂度、加性／低阶限制组合、低频限定变化方式。来源学习决定保留哪些方向及其系数／权重。该命题证明模型定义一致；来源学习是否使关系更适合目标，另由来源留出和真实／置换对照回答。

### 2.2 局部观测如何传到未采样策略差值——核心命题

对一个当前高斯组件，令观测点局部潜变量索引为o，\(S=C_{oo}\succ0\)，\(A=C_{:o}S^{-1}\)。原似然只依赖局部变量；设其条件化后局部矩为\((m_o^+,S^+)\)。由高斯条件分布和全期望／全协方差，得到

\[
m^+=m+A(m_o^+-m_o),\qquad C^+=C+A(S^+-S)A^\top.
\]

取\(a=e_u-e_v\)，u、v为未采样策略的目标均值坐标，则

\[
\Delta m_{u-v}=a^\top A\Delta m_o,\qquad
\Delta V_{u-v}=a^\top A\Delta S A^\top a.
\]

这正是[apply_raw_moments](../representation/vectorized_joint_function_risk.py#L101)及[单组件更新](../representation/joint_function_risk.py#L135)的矩补全。当前Lean还证明混合均值的两项分解：

\[
\bar m^+-\bar m=\sum_jw_j^+(m_j^+-m_j)+\sum_j(w_j^+-w_j)m_j.
\]

前项是组件内共享系数传播，后项是身份／节点重新加权。仅检查混合总协方差是否为零，不足以判断是否存在信息传递。多头共享身份权重也应接入同一记号。

**高斯已知噪声特例**给出最直接的排序解释。观测x，噪声方差r，记\(k_\Delta=K(u,x)-K(v,x)\)：

\[
\Delta m_{u-v}=\frac{k_\Delta}{K(x,x)+r}(y-m_x),\qquad
V_{u-v}-V_{u-v}^+=\frac{k_\Delta^2}{K(x,x)+r}.
\]

因此真正有助于区分u、v的是它们与观测点的**关系差异**。若两行相同，该观测在此特例下只让二者同步移动，并不提供目标差信息；对未知噪声混合模型还须保留重权项。这比“有耦合就全面改善预测”更准确，也直接对应已经保存的差值错序和竞争诊断。

### 2.3 用失配恒等式解释收益和损失，不假定prior总正确

已在数学正文及Lean中给出固定仿射预测器的风险分解。条件于来源资产，在明确的目标任务／观测分布下，令U为未采样目标差，Y为初始观测向量，\(S_Y=\operatorname{Var}(Y)\succ0\)，\(c=\operatorname{Cov}(U,Y)\)，\(a_*=cS_Y^{-1}\)。对来源给出的预测\(\hat U=m_U+a(Y-m_Y)\)，有

\[
\mathbb E(\hat U-U)^2=
\operatorname{Var}(U)-cS_Y^{-1}c^\top
+(a-a_*)S_Y(a-a_*)^\top
+[m_U-\mathbb EU+a(\mathbb EY-m_Y)]^2.
\]

展开平方即可证明，有限二阶矩足够。Lean使用显式可积性及正规方程\(a_*S_Y=c\)，不要求可逆；上式为正定情形的数学特例，逆矩阵代数未另形式化。它把可用的信息、关系失配和均值偏差分开：正确来源关联可能降低后两项，错误关联也可能抵消全部信息收益。该式针对固定仿射规则；当前非线性混合更新需要用2.2解释，不将此式直接当作整个AFTER的误差保证。它无需新增拟合或查询精算。

## 3. 第二组：把控制规则与跨维表示接准

**AFTER／POINT语义引理。** 若\(C_b\succeq0\)、基线对角正且局部方差v非负，令\(D_{ii}=\sqrt{v_i/(C_b)_{ii}}\)，则\(DC_bD\succeq0\)、对角为v，正方差坐标之间相关系数保持基线值。AFTER在initial10后未获自身新观测的策略边缘停在initial10状态；POINT从未观测的策略停在source0，已观测策略只受自身历史更新。新候选分别从initial10／source0基线投影，再应用已有的自身观测。直接对应[AFTER](../performance/mainline_role_after_init.py#L163)和[POINT](../performance/mainline_role_point_only.py#L68)。AFTER−POINT改变初始化后的均值、方差、噪声和相关性，不是仅改变一个均值项。

**当前跨维复合稳定性。** 旧定理控制精确Voronoi积分；实际实现还有midpoint余弦、64点插值、固定混合、全二次、径向归一化及来源encoder。对固定基数K、有界公共策略、冻结变换和明确的正尺度／保留特征值，分别给出重建／整数化误差\(\epsilon_d\)、求积误差\(q_K(d)\)及变换常数，再合成

\[
\|\Phi_d(h_d)-\Phi_{d'}(h_{d'})\|
\le C_\Phi[\epsilon_d+\epsilon_{d'}+q_K(d)+q_K(d')].
\]

将该误差进一步传给\(\phi^\top m\)、\(\phi^\top\Sigma\phi'\)；要传到条件化，还需要局部协方差的最小特征值条件。50点曝光描述、整数控制组与响应尺度分别给实现条件，不能由余弦定理一并代替。[实际变换](../representation/meta_prior.py#L5692)、[公共表示](../baselines/transfer_archive.py#L14)、[来源曝光表示](../representation/legacy_observable_prior.py)。

[CurrentRepresentationStability](../../proof/SCOLHKG/Real/CurrentRepresentationStability.lean)把上述连接拆成有限坐标引理：固定线性映射以最大行绝对和放大误差；有界二次特征的误差不超过\(2B\epsilon\)；正尺度标准化放大至\(\epsilon/s_{\min}\)。径向归一化以坐标误差加分母误差控制，分母\(\max(\|z\|_2,1)\)的变化仍须由上游误差给出。来源编码器在当前两输入间满足误差常数\(C\)时，目标均值误差不超过\(\|\beta\|_1 C(\epsilon_d+\epsilon_{d'}+q_d+q_{d'})\)；双线性核保留两端特征误差及系数绝对质量。公共十维坐标取完整65列图归一化后的前十列，不能在二次项进入分母前截取。正对角坐标与系数／协方差一起变换时，预测与核精确不变。

**正交的准确作用。** 数学上，在保留正特征值和准确分解条件下，来源加权白化满足\(Z^\top WZ=I\)。若设计与系数整套同时作可逆变换\(\phi'=B^{-\top}\phi,\beta'=B\beta\)，并同变换所有系数／任务协方差，预测和传播保持不变。当前Lean仅覆盖正尺度误差与正对角同步重参数化，完整加权白化及一般可逆B未在当前模块形式化。由此区分纯换坐标与来源选择／惩罚改变的实际作用；不把白化本身当新增信息。

**覆盖只保留当前可用版本。** 当前选点有Pareto精英、哨兵及`distance/(1+penalty)`，见[实现](../representation/meta_prior.py#L5972)，并非纯Gonzalez。无需改算法来迁就2近似：直接以实际选中集合的半径r套条件覆盖，\(L(r+\Delta+2\epsilon)\le\gamma\)时存在可行初始点。L、来源对齐误差\(\Delta\)和安全深度\(\gamma\)均明确列为条件。

## 4. 第三组：从预测变化连接实际查询

当前[constrained_ts](../acquisition/decision_backends.py#L1595)分别抽取f／g的高斯矩投影，噪声用\(\mathbb E e^\eta\)；它不是抽完整共享身份四头联合后验，也不是最大化KG。当前查询稳定性命题使用这一实际决策分布。

先固定**共同状态、相同有序候选池与共同随机向量**。设两状态抽样目标和margin的差分别不超过\(\epsilon_f,\epsilon_m\)。若全部margin距零超过\(\epsilon_m\)，抽样可行集合不变；若其中目标最优间隔超过\(2\epsilon_f\)，查询不变。没有可行点时，对实际最小违约分数（包括机器eps目标破平项）应用相同间隔引理。可复用[SourceRankRecovery](../../proof/SCOLHKG/Real/SourceRankRecovery.lean#L315)的代数部分。

进一步可以把\(F=m+L\xi\)的扰动写成均值差与\((L-L')\xi\)；L须用[实际抽样器](../acquisition/decision_backends.py#L229)的Cholesky加jitter或特征值截断后平方根。误差较小时大间隔查询稳定，竞争接近边界时传播才可能改变选择。这个命题说明“什么时候决策会变”；是否向真实好策略改变，由来源校准和封存共同状态／部署记录判断。

[CurrentQueryStability](../../proof/SCOLHKG/Real/CurrentQueryStability.lean)直接使用这两个代码分支。可行分支保留有限sentinel `-1e300`严格低于最优可行分数的条件；无可行点时对\(m+\mathrm{eps}\,f/\max(\max|f|,1)\)的完整loss使用间隔界。共同抽样误差为\(\epsilon_\mu+\sum_j\epsilon_{L,j}|\xi_j|\)，机会margin再加\(|z|\epsilon_\sigma\)。这些界需要同池、同实际抽样向量，以及来源编码、loading、噪声尺度的明确误差前提；没有把协方差误差直接当作平方根误差，也没有用经验规则差代替严格误差界。这沿用[GPT6最后诊断](../../markdown/GPT6_action_competition_diagnosis_20261013.md)对数值稳定、决策保证和实际收益的区分；跨维表示与来源正交的适用范围继续参考[v4诊断](../../markdown/GPT_diagnosis_v4.md)。

首个分歧后，各臂历史与动态候选池可能不同，逐步机械套同池结论会丢掉发现路径。端到端说明继续区分发现、名单、实际验证和部署；已有两阶段最优性定理要求的“在所有获证finalist中取目标最小”不符合当前终端，不将名单遗漏项假定为零。

## 5. 第四组：补当前旧终端的独立性、分支和错误预算

### 5.1 标准统计结果与实际选择器分开连接

给定搜索历史H后冻结三点名单。高斯单点响应的真实安全分位数\(q=\mu+z_{1-\alpha}\sigma\)，样本均值／标准差为\(\bar Y,S\)。当前[实现](../core/certification.py#L315)使用

\[
U=\bar Y+\frac{t^{\rm nct}_{n-1,z_{1-\alpha}\sqrt n}(1-\delta)}{\sqrt n}S,
\qquad \Pr(U<q\mid H)\le\delta.
\]

正非中心参数与上分位数方向正确；其枢轴是\(\sqrt n(q-\bar Y)/S\)。非中心t参数化见[SciPy 1.15.3官方定义](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats.nct.html)，单侧正态容忍上界的统计含义见[NIST手册](https://www.itl.nist.gov/div898/handbook/prc/section2/prc263.htm)。[守门树正文](mainline_terminal_gate_proof_20261020.md#2-单点证书与目标比较的条件误差)已给出标准枢轴推导与代码对应；Lean仍把覆盖界作为前提，尚未形式化整个分布推导和数值分位数计算。

配对目标用8个独立CRN对的差值\(D_i=F(x_c,\xi_i)-F(x_r,\xi_i)\)，仅在\(\bar D+t_{7,1-\delta}S_D/\sqrt8<0\)时接受挑战点更优。[方向正确](../core/terminal_verification.py#L330)。三家族是高斯加噪与线性标量化；同CRN差值仍为高斯。差值方差为零时，数学正文已单列常量差值情形：该判据的错误切换概率为零，无需新增执行分支。

安全种子、不同候选与目标比较使用[不同随机标签](../core/terminal_verification.py#L695)，只在同一目标pair内共享CRN。概率命题以这些块对应独立IID重复为条件；不能仅凭字符串不同声称已机器证明随机数生成器独立。

### 5.2 当前守门树允许共享第三份错误预算

| 前两点获证状态 | 后续动作 | 实际终验调用 |
|---|---|---:|
| 均获证 | 独立8对目标比较 | 80+128+16=224 |
| 仅一点获证 | 返回该点 | 80+128=208 |
| 均未获证 | 验证第三点 | 80+128+128=336 |

[实际分支](../core/terminal_verification.py#L840)中的第三点验证与目标比较互斥。登记352上限保持不变，不能把两个分支的调用都算作实际费用。

令\(\mathcal F_{12}\)含搜索历史与前两点安全样本，\(B_3\)表示两点均未获证，\(B_c\)表示两点均获证。若后续独立块的条件错误率分别不超过\(\delta_3,\delta_c\)，则

\[
\Pr(B_3\cap E_3)+\Pr(B_c\cap E_c)
\le\delta_3\Pr(B_3)+\delta_c\Pr(B_c)
\le\max(\delta_3,\delta_c).
\]

加上前两点错误预算，新守门树命题可给出

\[
\Pr(\text{假获证}\ \cup\ \text{错误地接受目标改善})
\le\delta_1+\delta_2+\max(\delta_3,\delta_c)=0.05,
\]

其中当前四个\(\delta\)均为\(.05/3\)。完整条件概率证明、“只从获证点返回certified=true”及208/224/336费用引理已见[守门树正文](mainline_terminal_gate_proof_20261020.md)；[CurrentTerminalGate](../../proof/SCOLHKG/Measure/CurrentTerminalGate.lean)已形式化局部错误界到最终5%的事件组合及返回／调用树。高斯覆盖、独立随机块与数值计算仍为外部前提。忽略门控的\(.05+.05/3\)只是较松的并界，不是必须增加门槛的理由。已有first-certified三点定理不能直接代表这棵树。

## 6. 局限：哪些结论仍不能由这些证明推出

- 精确局部矩补全在当前组件下成立；实现每轮再做矩投影，连FULL也不是原模型精确多轮后验。AFTER／POINT只拼接边缘方差及固定相关性，不能称为同一原始似然的完整后验。
- 已知噪声单Gaussian的方差收缩，不推广为未知噪声／混合重权后每条观测都收缩。精确Bayes有期望条件方差性质；当前投影算法还需另证。`D C D`保证PSD而不保证Loewner收缩；矩阵去噪`[A-B]_+`也不自动处处小于A。现行方法稿未作后两种错误泛化。
- 跨维坐标接近不代表对任意1000维函数充分。固定低复杂度结构保留近似偏差；来源训练分布正交也不代表目标分布正交。当前native使用`single_gaussian_draw`，不能完整继承另一可选shared-low-rank模式的理论解释。
- 安全界的事件是\(\Pr(\mathrm{certified}\land\mathrm{unsafe})\)，不是\(\Pr(\mathrm{unsafe}\mid\mathrm{certified})\)。全无证时代码仍返回第一点但标记abstained；该返回不获安全承诺。上面的联合界只保护被宣告的目标改善，不保证参考点未获证时直接选择其他候选的质量，也不保证G非负或选出三点最优者。
- 来源学习的泛化率若成为论文主张，需要明确独立任务／任务分布、拟合误差及迁移失配；384条raw不能当作384个独立来源任务。当前不把拟合收敛、全局regret、AFTER最优传播时机、四prior各自必要性或优于所有GP列为补证门槛。
- 数学命题不补齐执行缺失：Stage93全部11臂、67失败、Queue低ESS、C105/G97完整配对及全144来源效应未知继续保留。Stage85与Stage93、旧24批保持分开，科学收益与实际费用沿[证据链](mainline_evidence_chain_20261018.md)和[强后端配对](mainline_frozen_backend_comparison_results_20261019.md)引用，不重算或重跑。

## 7. 当前交付与后续条件

**三批当前主线Lean均已完成：** 首批传播／协方差／终端32条，第二批跨维表示／查询间隔26条，第三批来源核／固定仿射MSE36条，共94条，16模块闭包通过集群核验。两份数学正文分别为[传播与目标差](mainline_propagation_contrast_proof_20261020.md)和[当前终端守门树](mainline_terminal_gate_proof_20261020.md)；旧Atlas接口独立保留。MSE用实际Bochner积分及正规方程推导，来源核给出有限PSD与限制一致性；高斯概率律构造、统计枢轴和数值前提仍集中保留。

实现—数学—Lean—正文对应已收于第1节，并回填方法稿的四项条件命题和固定仿射失配解释。已核对公共65列归一化、实际含机器eps的回退loss，以及局部覆盖到终端5%的证明边界；完整白化与一般可逆重参数化保持数学解释身份。本次收束仅本地文档与注释修订，复用上述构建回执，无新增构建、积分、服务器读取或科学调用；全部失败、恢复费用与余量原样保留。

这组条件命题已与既有封存结果接入[论文核心论证](mainline_research_argument_20261019.md#条件命题与封存证据如何共同支持主张)及[实验联合解释](mainline_experiments_working_draft_20261019.md#条件命题与部署证据的联合解释)：机制主张引用本表，实际收益引用同初始化、同预算的已保存配对结果及其缺失；跨前端比较保留共同评价基准。本次连接只改本地文档，未新增计算或服务器读取。下一步压缩摘要、引言与讨论；只有出现会改变核心结论、且现有证据无法回答的具体缺口，才另行选择最小补证或实验，不因正文连接完成而启动新计算。

每项补证以实际实现与明确条件下命题成立为结束条件，不要求它证明实验必胜。遇到当前实现不满足的假设，缩小命题或保留误差／遗漏项；不为取得漂亮定理改冻结算法，不重开单query精算。后续Lean构建或有意义计算仍经scheduler在node001–node006执行，CPU-only、单线程；三批执行、失败、恢复及已测费用分别保留于回执。此前构建已关闭逐任务自动重派，后续沿用这一设置，原科学失败及余量不变。

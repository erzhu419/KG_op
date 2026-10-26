# 来源传播、未采样目标差与AFTER／POINT：实现对应证明

条件于冻结来源资产和某一轮已付历史，本文证明有限候选池上的工作先验有效性、单步局部矩传播、混合重加权和AFTER／POINT的决策协方差语义。核心量是两个策略与观测位置的关系之差；共享关系本身不保证排序改善。本文完成[补证计划](mainline_proof_gap_audit_and_plan_20261020.md)的2.1／2.2及AFTER／POINT部分，并给出2.3的失配恒等式。独立终验见[守门树证明](mainline_terminal_gate_proof_20261020.md)。

## 1. 记号与工作先验的有效性

候选池由互异物理策略组成。每个策略的潜变量顺序为
\(F(x)=(h_f(x),\eta_f(x),h_g(x),\eta_g(x))\)，其中
\(h_g=\mu_g-\tau\)。身份记为 \(i\)，每个身份的两个头
\((h_f,\eta_f)\)、\((h_g,\eta_g)\)分别具有节点权重
\(\rho_{ifq},\rho_{igr}\)，身份权重为 \(\pi_i\)。权重均非负、各层归一化；可将完整组件展平为
\[
j=(i,q,r),\qquad w_j=\pi_i\rho_{ifq}\rho_{igr}.
\tag{1}
\]
这是工作模型内给定身份的两头独立结构，不是对真实paired响应独立性的推断。

**命题1（合法有限池先验）。** 对每个组件，若系数协方差
\(\Sigma_j,T_j\succeq0\)、白项 \(v_j\ge0\)，则
\[
K_j(x,x')=\phi_j(x)^\top(\Sigma_j+T_j)\phi_j(x')
                 +v_j\mathbf1\{x=x'\}
\tag{2}
\]
定义PSD核。对合法的各头高斯组件作块对角拼接，再按(1)混合，仍是合法概率分布，其矩满足
\[
\bar m=\sum_jw_jm_j,\qquad
\bar C=\sum_jw_j\{C_j+(m_j-\bar m)(m_j-\bar m)^\top\}\succeq0.
\tag{3}
\]
同一来源状态在大池投影后限制到子池，与直接投影子池具有相同先验分布。

**证明。** 对任意有限池、系数向量 \(b\)，(2)的二次型为
\((\sum_xb_x\phi_j(x))^\top(\Sigma_j+T_j)(\sum_xb_x\phi_j(x))
+v_j\sum_xb_x^2\ge0\)。块对角拼接保持PSD；有限非负混合定义概率分布，(3)由全协方差公式得到，每项PSD。限制只截取相同的均值行和核子矩阵，白项附着于物理策略，故限制一致。证毕。

实现：[有限池投影](../performance/mainline_legacy_source_models.py#L269)、[混合矩](../representation/joint_response_mixture.py#L78)。白项使整个核不必低秩。role构造把每个g节点的系数均值／协方差换成同一native-g来源分布，置 \(T_g=0\)，保留非负偏差白项、f／η分布及原节点／身份权重；截距减 \(\tau\) 只平移均值。因此它仍满足命题1。它定义的是新的工作先验，不能当作保留原来源g–η依赖的联合后验。[实际替换](../performance/mainline_role_prior.py#L31)

## 2. 局部原始观测如何传到未采样目标差

**命题2（单组件局部矩补全）。** 在本轮的一个高斯组件中，令全场
\(X\sim N(m,C)\)，观测所依赖的局部变量为 \(Z=X_o\)，
\(S=C_{oo}\succ0\)，\(A=C_{:o}S^{-1}\)。观测似然只依赖 \(Z\)，且观测后局部矩
\(m_o^+=E[Z\mid y]\)、\(S^+=\operatorname{Cov}(Z\mid y)\)存在。则
\[
m^+=m+A(m_o^+-m_o),\qquad
C^+=C+A(S^+-S)A^\top.
\tag{4}
\]
即使未知噪声使局部条件分布非高斯，(4)仍给出本次更新的准确一、二阶矩。

**证明。** 高斯回归分解给出
\[
X=m+A(Z-m_o)+R,\quad
R\perp Z,\quad \operatorname{Cov}(R)=C-ASA^\top\succeq0.
\]
似然只依赖 \(Z\)，故条件于 \(y\) 后，\(R\)仍保持原分布并独立于
\(Z\mid y\)。取条件期望与条件协方差即得(4)，并且
\(C^+=(C-ASA^\top)+AS^+A^\top\succeq0\)。证毕。

实际单头局部变量是 \((h_a(x),\eta_a(x))\)，raw似然为
\(Y_a\mid Z\sim N(h_a(x),e^{\eta_a(x)})\)，g只减一次当前 \(\tau\)。
两头的似然及证据按工作模型分别计算，再更新共享身份。两次单观测按原顺序更新，不改成一个重复批次。[单组件实现](../representation/joint_function_risk.py#L128)与[向量化补全](../representation/vectorized_joint_function_risk.py#L101)执行同一(4)。

令 \(a=e_{h_f(u)}-e_{h_f(v)}\)，其中u、v均未被本次观测。组件内目标差均值与方差记为
\(d_j=a^\top m_j,\ V_j=a^\top C_ja\)。由(4)，
\[
d_j^+-d_j=a^\top A_j(m_{jo}^+-m_{jo}),\qquad
V_j^+-V_j=(a^\top A_j)(S_j^+-S_j)(a^\top A_j)^\top.
\tag{5}
\]
因此传播到差值的有效载荷是 \(a^\top A_j\)，而非单独某个跨策略协方差的大小。若 \(a^\top A_j=0\)，本次组件内目标差信息不变；未知噪声下不假设 \(S_j^+\preceq S_j\)。

## 3. 身份与节点重新加权是另一条传播通道

**命题3（完整混合差值的两项分解）。** 组件预测证据为
\(\ell_j(y)>0\)，则 \(w_j^+=w_j\ell_j/\sum_kw_k\ell_k\)，且
\[
\bar d^+-\bar d
=\underbrace{\sum_jw_j^+(d_j^+-d_j)}_{\text{组件内关系传播}}
+\underbrace{\sum_j(w_j^+-w_j)d_j}_{\text{身份／节点重加权}}.
\tag{6}
\]
混合差值方差为
\[
\bar V=\sum_jw_j\{V_j+(d_j-\bar d)^2\},
\tag{7}
\]
更新后用各项上标“+”取代；不推出逐观测方差单调下降。

**证明。** 在 \(\sum_jw_j^+d_j^+-\sum_jw_jd_j\)中加减
\(\sum_jw_j^+d_j\)得(6)；对差值用全方差公式得(7)。证毕。

代码未显式存储(1)的笛卡尔积，但与它等价：令
\(Z_{ia}=\sum_q\rho_{iaq}\ell_{iaq}\)，则
\(\rho_{iaq}^+=\rho_{iaq}\ell_{iaq}/Z_{ia}\)，
\(\pi_i^+\propto\pi_iZ_{if}Z_{ig}\)。
这样g观测即使不直接条件化f头，也能经共享身份改变目标差；不能先将两头各自跨身份混合，再把所得边缘密度相乘。[头内权重与证据](../representation/joint_function_risk.py#L148)、[共享身份更新](../representation/joint_response_mixture.py#L57)

仅凭混合总协方差为零不能删除(6)第二项。例如两个等权高斯组件的目标差均值分别为+1、−1，观测位置的均值都为0、方差分别为1、4，且组件内目标差与观测独立。总体协方差为0，但观测的绝对大小会改变身份权重，从而改变目标差均值。这里没有组件内传播，信息完全来自重加权。

## 4. 已知噪声单高斯特例：关系差异决定区分能力

**推论4。** 只有一个高斯组件、观测为
\(y=h_f(x)+\epsilon,\ \epsilon\sim N(0,r)\)，且噪声独立、
\(K(x,x)+r>0\)。写
\(k_\Delta=K(u,x)-K(v,x)\)，则
\[
d^+-d=\frac{k_\Delta}{K(x,x)+r}(y-m_x),\qquad
V-V^+=\frac{k_\Delta^2}{K(x,x)+r}.
\tag{8}
\]
**证明。** 对联合高斯向量 \((h_f(u)-h_f(v),y)\)条件化；其交叉协方差是
\(k_\Delta\)，观测方差是 \(K(x,x)+r\)，代入条件均值／Schur补公式即得。证毕。

若两策略与观测点的核值相同，目标差的整个高斯条件分布保持不变，二者至多同步移动；若核值不同且残差非零，均值差才会改变。均值排序反转要求更新前后差值异号；这与当前约束TS的抽样可行集／抽样最优点不同，后者还依赖风险和全池抽样。式(8)不推广为当前混合模型的收缩定理。

## 5. AFTER／POINT的准确语义与PSD证明

**命题5（固定相关性、更新边缘方差）。** 在任一固定有限池上，基线
\(C_b\succeq0\)、\(b_i=(C_b)_{ii}>0\)，各本点更新给出非负方差 \(v_i\)。
令
\[
D_{ii}=\sqrt{v_i/b_i},\qquad C_t=DC_bD.
\tag{9}
\]
则 \(C_t\succeq0\)、\((C_t)_{ii}=v_i\)；当 \(v_i,v_j>0\) 时
\[
\operatorname{Corr}_t(i,j)=\frac{(C_b)_{ij}}{\sqrt{b_ib_j}}.
\tag{10}
\]
**证明。** 对任意z，\(z^\top C_tz=(Dz)^\top C_b(Dz)\ge0\)；直接取对角得
\(v_i\)，再约去相关系数中的两个尺度因子得(10)。若 \(v_i=0\)，对应行列为0，相关系数不定义。证毕。

POINT的基线是source0；每个物理策略的本点模型只重放自身已付raw，所以未采样点的均值、噪声、局部身份／节点权重与方差留在source0。AFTER先共同使用initial10并保存条件节点；之后每个点只使用自身新观测，未获自身新观测的点保留initial10投影边缘。新候选也投影自同一基线；已初始化物理策略复用包含白项、且已经initial10条件化的该策略潜变量。[POINT更新及缩放](../performance/mainline_role_point_only.py#L75)、[AFTER投影](../performance/mainline_role_after_init.py#L61)、[AFTER更新及缩放](../performance/mainline_role_after_init.py#L170)

式(9)对全部 \([h_f,\eta_f,h_g,\eta_g]\)坐标统一缩放，也固定本策略内部的决策相关系数；本点滤波器内部仍保留自己的完整条件协方差。返回的固定基线weights字段不是所有本点共享的更新权重，policy_weights才记录各本点身份权重。故(9)是供决策使用的合法高斯矩拼接，不宣称同时实现每个本点滤波器完整协方差的联合Bayes后验。

AFTER与POINT改变的是初始化传播后得到的均值、方差、噪声和相关结构整体；各臂native提议后验仍正常学习。命题5只证明这种拼接有效，不保证 \(C_t\preceq C_b\)，也不把两臂差异归结为单一协方差效应。

## 6. 固定线性规则的关系失配与均值偏差

**命题6。** 条件于来源资产，目标分布中未采样目标差U和初始化观测Y具有有限二阶矩，
\(S_Y=\operatorname{Var}(Y)\succ0\)，\(c=\operatorname{Cov}(U,Y)\)，
\(a_*=cS_Y^{-1}\)。对固定来源规则
\(\widehat U=m_U+a(Y-m_Y)\)，有
\[
E(\widehat U-U)^2
=\underbrace{\operatorname{Var}(U)-cS_Y^{-1}c^\top}_{\text{最优仿射规则的残余风险}}
+\underbrace{(a-a_*)S_Y(a-a_*)^\top}_{\text{关系失配}}
+\underbrace{[m_U-EU+a(EY-m_Y)]^2}_{\text{均值偏差}}.
\tag{11}
\]
**证明。** 置 \(Y_0=Y-EY,\ U_0=U-EU\)，将误差写成
\(aY_0-U_0+b\)，其中b为最后方括号。中心化项均值为0，风险为
\(\operatorname{Var}(U)+aS_Ya^\top-2ac^\top+b^2\)；利用
\(a_*S_Y=c\)配方得(11)。证毕。

这给出传播可能同时带来收益与损失的精确解释：目标分布中存在可用信息，不代表来源给定的关系或中心正确；失配和偏差可以抵消信息收益。式(11)针对固定仿射规则，不能直接当作非线性混合AFTER的整体风险保证。

这里的\(S_Y,c,EU,EY\)属于明确指定的目标分布；\(a,m_U,m_Y\)是条件于来源资产后固定的预测规则。正常方程\(a_*S_Y=c\)定义比较用的参照，不要求优化器知道目标分布矩。以正常方程代替显式逆矩阵表达时，配方恒等式也适用于有解的奇异协方差；来源核的PSD与这组目标分布矩分别承担建模有效性和失配解释的作用。

## 局限与形式化接口

- 命题2是对本轮输入高斯组件及准确局部矩的数学结论。实际局部积分截断在当前η均值±12标准差，并使用数值求积；代码对返回的局部矩执行(4)，再投影为高斯供下一步使用。全场补全不额外丢弃这两个局部矩，但没有因此消除局部求积、尾部截断或多轮投影误差；FULL也不例外。向量取欧氏范数、矩阵取谱范数，若局部矩误差界为 \(\epsilon_m,\epsilon_S\)，固定A时目标差误差分别至多
  \(\|a^\top A\|\epsilon_m\)、\(\|a^\top A\|^2\epsilon_S\)；证据／权重误差另计。已有数值停止量不自动认证这些总误差。
- 代码直接求解局部S并直接计算(9)，没有在这两处加jitter或广义逆。上述PSD结论以有效PSD输入、\(S\succ0\)、正基线对角及非负本点方差为条件；本次证明不认证所有浮点输出满足这些条件，不更改失败门。约束TS另取f／g主块、分别抽样，并在抽样器中加jitter或作特征值截断；式(10)描述的是传入的决策矩。[实际抽样](../acquisition/decision_backends.py#L229)
- 跨维表示与共同池查询稳定性另有当前Lean模块，误差和严格间隔条件见[补证计划](mainline_proof_gap_audit_and_plan_20261020.md)第3／4节。任意跨维表示充分性、部署增益非负、全局regret与AFTER最优切换时机仍不由本文推出；实证范围沿用原封存结果和完整失败、缺失及费用。
- [当前主线Lean接口](../../proof/SCOLHKG/CurrentMainlineProofInterface.lean)三批已通过集群核验。[CurrentPropagation](../../proof/SCOLHKG/Real/CurrentPropagation.lean)13条与[CurrentCovariance](../../proof/SCOLHKG/Real/CurrentCovariance.lean)6条覆盖目标差／重权代数和式(9)。新增[CurrentSourceKernel](../../proof/SCOLHKG/Real/CurrentSourceKernel.lean)21条给出式(2)／(3)的有限PSD、独立头／role拼接及均值／核限制；概率混合解释仍需归一化权重和对应中心，索引白项与物理白项等价要求候选互异。[CurrentAffineMismatch](../../proof/SCOLHKG/Real/CurrentAffineMismatch.lean)15条从实际Bochner积分证明式(11)的正常方程形式，并由积分平方证明目标协方差PSD。高斯条件化、概率律构造、统计枢轴及数值误差仍按上述边界保留。[本轮回执](../paper_artifacts/mainline_source_kernel_mismatch_lean_20261020_result.json)与原两轮回执分别保存。

数学正文之后完成了上述集群Lean核验；未新增仿真、积分、拟合、后验恢复或统计重算。冻结算法、历史失败与科学账目保持不变，Lean环境、失败及恢复费用另列于各轮回执；本轮自动重派为零。

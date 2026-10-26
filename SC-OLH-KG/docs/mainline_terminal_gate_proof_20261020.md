# 当前三点终端守门树：条件错误界、返回规则与费用证明

本文完成[证明审查第5节](mainline_proof_gap_audit_and_plan_20261020.md#5-第四组补当前旧终端的独立性分支和错误预算)的实现桥。对象是当前 `same_policy_distinct_challenger` 旧终端：三个不同策略依次为挑战者 \(x_1\)、初始参考点 \(x_2\)、回退点 \(x_3\)，安全样本量为80/128/128，目标比较使用8对样本。协议由[冻结输入检查](../performance/mainline_reference_coverage.py#L81)及[执行适配](../performance/run_mainline_terminal_confirmation.py#L25)确认；具体树由[verify_frozen_shortlist](../core/terminal_verification.py#L840)执行。证明不要求搜索后验正确。

## 1. 条件与概率空间

固定目标任务的真实参数。令 \(\mathcal H\) 包含全部搜索历史、来源训练结果、算法随机性和已冻结的三点名单、预算及超参数。以下概率均对搜索结束后新增的验证随机块计算；命题对几乎所有允许的 \(\mathcal H\) 成立，因此也可对搜索历史取期望。

给定 \(\mathcal H\)，每点的安全块由 \(n_i\) 个IID \(N(\mu_i,\sigma_i^2)\) 约束响应组成，且独立于搜索。目标比较块独立于搜索及全部安全块；8个CRN对之间IID，每对差值
\[
D_j=Y_f(x_1,\xi_j)-Y_f(x_2,\xi_j)
\sim N(\Delta,\sigma_D^2),\qquad \Delta=f(x_1)-f(x_2).
\]
同一对内允许相关，甚至 \(\sigma_D=0\)。不同候选安全块使用独立重复；实际并界不需要前两安全块相互独立，但需要尚未开启的第三安全块和目标比较块在条件于前两安全结果后仍满足其误差界。

当前高斯三家族继承[加性正态仿真](../problems/rzdt.py#L88)，再由[线性标量化](../problems/single_objective.py#L66)生成目标。同CRN对共用同一标准正态向量、乘以各策略的确定尺度，因此差值仍为正态或常量。`simulation_noise_model="iid_gaussian"` 只检查单点模型标签；对其他仿真器，成对联合正态仍是上述命题的独立假设。

实现中安全块使用[候选序号和安全stream标签](../core/terminal_verification.py#L695)，目标对使用[独立目标标签](../core/terminal_verification.py#L330)，同对两个策略重新创建相同seed的生成器。不同标签是独立块协议的实现方式；数学概率声明以独立IID随机块建模，不把标签差异视为随机数生成器独立性的形式化证明。

## 2. 单点证书与目标比较的条件误差

**引理1（单点安全证书）。** 设 \(0<\alpha\le1/2\)、\(n\ge2\)，令
\[
z=\Phi^{-1}(1-\alpha),\quad q=\mu+z\sigma,\quad
c=F^{-1}_{t_{n-1,z\sqrt n}}(1-\delta),\quad
U=\bar Y+cS/\sqrt n,
\]
其中 \(S^2\) 为使用ddof=1的无偏样本方差，\(S=\sqrt{S^2}\)。以 \(A=\{U\le\tau\}\) 表示获证，则
\[
\Pr\bigl(A\cap\{q>\tau\}\mid\mathcal H\bigr)\le\delta.
\tag{1}
\]

**证明。** 当 \(\sigma>0\) 时，正态抽样恒等式给出相互独立的
\[
Z=\sqrt n(\bar Y-\mu)/\sigma\sim N(0,1),\quad
V=(n-1)S^2/\sigma^2\sim\chi^2_{n-1}.
\]
于是
\[
T=\frac{\sqrt n(q-\bar Y)}S
=\frac{-Z+z\sqrt n}{\sqrt{V/(n-1)}}
\sim t_{n-1,z\sqrt n}.
\]
这与[SciPy 1.15.3的正非中心参数定义](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats.nct.html)一致。连续性给出 \(\Pr(U<q\mid\mathcal H)=\Pr(T>c)=\delta\)；若 \(q>\tau\) 而 \(U\le\tau\)，必有 \(U<q\)，得到(1)。当 \(\sigma=0\) 时，全部样本等于 \(\mu\)，\(S=0\)、\(U=q=\mu\)，错误事件为空。证毕。

这是一侧正态容忍上界，与[NIST给出的非中心t容忍因子](https://www.itl.nist.gov/div898/handbook/prc/section2/prc263.htm)一致；实现直接计算[同一分位数与上界](../core/certification.py#L370)，以[upper_margin≤0](../core/certification.py#L400)获证。少于2个重复时实现不获证；当前80/128/128均不触及该分支。\(q\le\tau\) 等价于真实机会约束 \(\Pr(Y_g\le\tau)\ge1-\alpha\)，退化高斯也成立。

**引理2（只接受被证明的目标改善）。** 对独立的 \(m=8\) 个差值，定义
\[
U_D=\bar D+t_{7,1-\delta_c}S_D/\sqrt8,\qquad
J=\{U_D<0\}.
\]
令 \(\mathcal F_{12}\) 为 \(\mathcal H\) 加前两点全部安全数据生成的sigma代数，则
\[
\Pr\bigl(J\cap\{\Delta\ge0\}\mid\mathcal F_{12}\bigr)\le\delta_c.
\tag{2}
\]

**证明。** 当 \(\sigma_D>0\) 时，\(T_D=\sqrt8(\bar D-\Delta)/S_D\sim t_7\)。若 \(\Delta\ge0\) 且 \(U_D<0\)，则 \(T_D<-t_{7,1-\delta_c}\)，其概率为 \(\delta_c\)。独立新块保证条件于 \(\mathcal F_{12}\) 后同一推导成立。若 \(\sigma_D=0\)，则 \(S_D=0\)、\(U_D=\Delta\)，严格判据仅在真实 \(\Delta<0\) 时通过，错误概率为0。证毕。

实现对[挑战者减参考点](../core/terminal_verification.py#L354)计算上界并使用[严格小于0](../core/terminal_verification.py#L363)。零差值方差不需要另行除法分支：代码计算的是 \(\bar D+tS_D/\sqrt8\)，故在实数算术下自然退化为 \(\Delta\)。CRN降低或消除差值噪声，不把8对的成本变成8次调用：仍为16次仿真。

## 3. 互斥门控的5%联合界

设 \(A_i\) 为第 \(i\) 点安全获证，\(E_i=A_i\cap\{q_i>\tau\}\) 为该证书错误；第三点的事件只在其块被启用时解释。两个 \(\mathcal F_{12}\)-可测门为
\[
B_3=A_1^c\cap A_2^c,\qquad B_c=A_1\cap A_2,
\qquad B_3\cap B_c=\varnothing.
\]
令 \(E_c=J\cap\{\Delta\ge0\}\)。前两点各满足引理1；第三安全块独立，故在 \(B_3\) 上有 \(\Pr(E_3\mid\mathcal F_{12})\le\delta_3\)。引理2给出比较块的对应条件界。因而
\[
\begin{aligned}
&\Pr\bigl((B_3\cap E_3)\cup(B_c\cap E_c)\mid\mathcal H\bigr)\\
&=\mathbb E\!\left[\mathbf1_{B_3}\Pr(E_3\mid\mathcal F_{12})
+\mathbf1_{B_c}\Pr(E_c\mid\mathcal F_{12})\mid\mathcal H\right]\\
&\le\delta_3\Pr(B_3\mid\mathcal H)+\delta_c\Pr(B_c\mid\mathcal H)\\
&\le\max(\delta_3,\delta_c).
\end{aligned}
\tag{3}
\]
这里没有把第三证书和目标比较当作同时执行的两个检验；二者共享一份预算的依据正是条件误差界及互斥门。

**定理（当前守门树的联合错误控制）。** 在第1节条件和精确实数统计量／分位数下，令 \(F\) 表示本次终端尝试中至少一张已发放证书实际不安全，\(W\) 表示本次终端接受了错误的“挑战者严格优于参考点”声明。当前设置
\[
\delta_1=\delta_2=\delta_3=\delta_c=0.05/3=1/60
\]
满足
\[
\Pr(F\cup W\mid\mathcal H)
\le\delta_1+\delta_2+\max(\delta_3,\delta_c)
=1/20=0.05.
\tag{4}
\]

**证明。** 错误事件包含于
\(E_1\cup E_2\cup(B_3\cap E_3)\cup(B_c\cap E_c)\)。前两项由(1)分别控制，后两项由(3)联合控制，再取并界。对 \(\mathcal H\) 取期望可得同样的无条件界。证毕。

## 4. 返回规则与费用引理

令 \(R\) 为最终返回点，\(C\) 为返回记录 `certified=true`。代码在[分支选择](../core/terminal_verification.py#L848)中遵循下表。

| 前两点结果 | 最终动作 | 实际终验调用 |
|---|---|---:|
| 两点均获证 | \(J\) 成立返回 \(x_1\)，否则返回 \(x_2\) | \(80+128+2\times8=224\) |
| 仅 \(x_1\) 获证 | 返回 \(x_1\) | \(80+128=208\) |
| 仅 \(x_2\) 获证 | 返回 \(x_2\) | \(80+128=208\) |
| 两点均未获证 | 验证 \(x_3\)；其获证则返回 \(x_3\)，否则返回 \(x_1\) 并abstain | \(80+128+128=336\) |

**引理3。** \(C\Rightarrow A_R\)；因此
\[
\Pr\bigl((C\cap\{q_R>\tau\})\cup W\mid\mathcal H\bigr)\le0.05.
\tag{5}
\]
**证明。** 前三行返回点均有获证标志；最后一行仅在 \(A_3\) 成立时设置获证排名。代码以[获证排名存在](../core/terminal_verification.py#L904)定义 `certified`，故 \(C\cap\{q_R>\tau\}\subseteq F\)，再用(4)。全未获证时保留默认 \(x_1\)，获证排名为空且状态为 `abstained_uncertified`。证毕。

**引理4。** 成功完成本协议的一次终端调用数仅为208、224或336，均不超过登记上限352。**证明。** 两个安全块始终执行，共208次；只有 \(B_c\) 增加16次，只有 \(B_3\) 增加128次，且两门互斥。因此实际费用为 \(208+16\mathbf1_{B_c}+128\mathbf1_{B_3}\le336\le352\)。实现[只累加实际attempt及comparison调用](../core/terminal_verification.py#L896)，同时[保留352登记上限](../core/terminal_verification.py#L927)。搜索调用另加，当前在线/STOP登记上限仍为365/362。失败、恢复及其他实际开销按原费用记录保留，不能从成功分支恒等式反推它们为零。证毕。

## 5. 适用边界

式(5)控制的是“获证且不安全”的联合概率，不能改写成 \(\Pr(\mathrm{unsafe}\mid C)\le0.05\)。它只保护实际宣告的目标改善；参考点未获证而直接返回其他点时没有目标不劣保证。abstain返回不受安全保证；本命题不保证部署质量 \(G\ge0\)、选出三点最佳者或全局优化收益，也不提供跨多个实验格、候选臂或重复运行的整体5%错误界。

[CurrentTerminalGate](../../proof/SCOLHKG/Measure/CurrentTerminalGate.lean)的13条定理已通过集群Lean核验：以真实概率测度上的局部错误界为前提，推出互斥门共享预算、\(\delta_1+\delta_2+\max(\delta_3,\delta_c)\)及5%联合界，并连接实际返回／声明、abstain与208/224/336调用树、352上限。该调用树计数不包含额外设施失败／恢复费用；终端原费用继续沿原科学账目保留，本轮Lean费用另见[构建回执](../paper_artifacts/mainline_lean_20261020_result.json)。

高斯枢轴、随机流或数值分位数尚未形式化进Lean。现有SciPy调用使用浮点近似；精确5%结论是精确分位数与实数算术的统计命题，当前代码未提供向上舍入证书。若计算分位数 \(\hat c\) 满足其CDF至少为 \(1-\delta-\varepsilon\)，且均值、标准差及最终上界按实数精确计算，相应局部误差界成为 \(\delta+\varepsilon\)，式(4)相应为 \((\delta_1+\varepsilon_1)+(\delta_2+\varepsilon_2)+\max(\delta_3+\varepsilon_3,\delta_c+\varepsilon_c)\)。最终上界的算术舍入误差还需另行控制，当前未给出可核验的总数值误差界。

另一个[新三点锦标赛](../core/terminal_two_comparison.py#L29)使用80/120/120，始终验证第三点，最多两次目标比较、每次 \(\delta=0.05/6\)；它不具有 \(B_3/B_c\) 的上述互斥结构，不能套用本证明的5%联合界。本次未修改该协议或任何冻结实验。

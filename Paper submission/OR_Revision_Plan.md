# 投稿 Operations Research 的详细修订执行计划

**论文**: A Probabilistically Constrained Bi-Objective Simulation-Based Stochastic Optimization Framework  
**目标期刊**: Operations Research  
**制定日期**: 2026-04-09

---

## 总体修订策略

将论文从"交通信号配时应用论文"重塑为"通用方法论论文+交通案例研究"。核心叙事：**在大规模离散解空间中，首次建立了处理双目标+概率约束+未知异方差的 Knowledge Gradient 框架，并提供有限预算下的理论保证。**

---

## 第一部分：论文结构重组

### 1.1 建议标题

**当前标题** (过于应用导向):  
> A probabilistically constrained bi-objective simulation-based stochastic optimization framework for urban transportation problems

**建议修改为** (方法导向):  
> Knowledge Gradient for Bi-Objective Simulation Optimization with Probabilistic Constraints and Unknown Variances

或:  
> Parametric Bayesian Learning for Bi-Objective Stochastic Optimization under Probabilistic Constraints

### 1.2 建议结构

| Section | 内容 | 预计页数 | 备注 |
|---------|------|---------|------|
| 1. Introduction | 从一般性 SO 问题出发，交通仅作为 motivating example | 3-4 页 | 重写 |
| 2. Problem Formulation | 通用的双目标概率约束随机优化 formulation | 2 页 | 将原 Section 3.1 通用化 |
| 3. Parametric GPR Framework | 参数化信念模型 + VEPM | 5-6 页 | 整合原 Section 4.1-4.3 |
| 4. KG-Based Sampling Policy | KG 因子计算 + 采样策略 + 后验问题分解 | 3-4 页 | 原 Section 4.4 |
| **5. Theoretical Analysis** | **Regret bound + VEPM 一致性 + 可行性保证** | **4-5 页** | **全新** |
| 6. Numerical Experiments | 扩充的合成测试 + 基准对比 | 5-6 页 | 大幅扩充 |
| 7. Case Study: Traffic Signal Timing | 交通网络应用 | 4-5 页 | 原 Section 5.2 精简 |
| 8. Conclusion | 总结 + 未来方向 | 1 页 | |
| Online Supplement | 所有证明 + 算法伪代码 + 额外实验 | 15-20 页 | |

**总正文**: 约 30-35 页 (OR 标准格式)

---

## 第二部分：理论贡献的实质性提升

### 2.1 有限预算 Regret Bound (最高优先级)

**目标**: 证明 GPR-KG 在有限预算 N 下的性能保证。

**具体任务**:

#### 任务 2.1.1: Simple Regret Bound

定义双目标 simple regret 为算法输出的 Pareto 前沿与真实 Pareto 前沿之间的 Hausdorff 距离（或超体积差距）:
$$
r_N = d_H(\hat{\mathcal{P}}_N, \mathcal{P}^*) \quad \text{或} \quad r_N = \text{HV}(\mathcal{P}^*) - \text{HV}(\hat{\mathcal{P}}_N)
$$

**推导思路**:
1. 利用参数化信念模型的结构，将后验均值的误差分解为：
   - 系数估计误差: $\|\hat{\beta}_N - \beta^*\|$ — 由贝叶斯线性回归理论给出 $O(p/\sqrt{N})$
   - 偏差项估计误差: 对已采样解为 $O(1/\sqrt{n_x})$，对未采样解由 VEPM 控制
2. 将均值估计误差传播到 Pareto 前沿的 Hausdorff 距离
3. 目标结论形如:
$$
\mathbb{E}[r_N] \leq O\!\left(\sqrt{\frac{p \log N}{N}}\right) + \text{VEPM bias term}
$$
其中 $p$ 是基函数数量

**参考文献**:
- Srinivas et al. (2010) "Gaussian Process Optimization in the Bandit Setting" — GP-UCB regret bound 的经典方法
- Frazier et al. (2008) — 单目标 KG 的渐近最优性证明
- Zuluaga et al. (2016) "ε-PAL: An Active Learning Approach to the Multi-Objective Optimization Problem" — 多目标 regret 的定义方式

#### 任务 2.1.2: 采样复杂度界

证明：要达到 $\epsilon$-accurate Pareto 前沿（在 Hausdorff 距离意义下），所需的采样预算 $N$ 满足:
$$
N = O\!\left(\frac{p \log(1/\epsilon)}{\epsilon^2}\right)
$$
这说明所需预算与解空间大小 $K$ 无关（仅与基函数数 $p$ 相关），体现了参数化信念的优势。

#### 任务 2.1.3: 与非参数化 GP 的理论对比

| 方法 | 有效维度 | Regret 依赖 |
|------|---------|------------|
| 标准 GP (维护 K 维信念) | K | $O(\sqrt{K \log N / N})$ |
| 参数化 GPR-KG (本文) | p | $O(\sqrt{p \log N / N})$ |

当 $K \gg p$ 时（即解空间远大于特征数），参数化方法具有本质优势。

### 2.2 VEPM 的统计理论 (高优先级)

**目标**: 为 VEPM 建立严格的统计性质。

#### 任务 2.2.1: 一致性证明

**定理 (VEPM Consistency)**: 设真实方差函数 $\sigma^2(x)$ 在同一特征分区组合内为常数（即 VEPM 的模型假设成立）。则对任意分区组合 $c$:
$$
\hat{\sigma}^2_{N,c} \xrightarrow{a.s.} \sigma^2_c \quad \text{as } N_c \to \infty
$$
其中 $N_c$ 是分区 $c$ 中被采样解的数量。

**证明思路**: 
- 公式 (9b) 本质上是加权样本方差，利用强大数律即可得到
- 关键是要处理均值估计 $\mu^i_n(x)$ 本身也在更新的问题（需要 martingale argument）

#### 任务 2.2.2: 收敛速率

**定理 (VEPM Convergence Rate)**: 在模型假设成立的条件下:
$$
\mathbb{E}\left[(\hat{\sigma}^2_{N,c} - \sigma^2_c)^2\right] \leq O\!\left(\frac{1}{N_c}\right) + O\!\left(\frac{p}{N}\right)
$$
第一项来自方差估计的采样误差，第二项来自均值估计误差的传播。

#### 任务 2.2.3: 模型误设下的分析

当真实方差函数不满足分区常数假设时，VEPM 会引入 bias:
$$
\text{bias}_c = \frac{1}{|S_c|} \sum_{x \in S_c} (\sigma^2(x) - \bar{\sigma}^2_c)
$$
讨论：
- 分区越细，bias 越小但 variance 越大（经典 bias-variance tradeoff）
- 最优分区粒度的选择准则（如交叉验证或信息准则）

### 2.3 概率约束的可行性保证 (高优先级)

**目标**: 证明算法输出的解以高概率满足概率约束。

#### 任务 2.3.1: 分位数估计的置信区间

**引理**: 在 Gaussian 假设下，$q_\alpha(x)$ 的后验估计为:
$$
\hat{q}_{\alpha,N}(x) = \hat{\mu}^E_N(x) + \Phi^{-1}(\alpha) \cdot \hat{\sigma}^E_N(x)
$$
其估计误差满足:
$$
|\hat{q}_{\alpha,N}(x) - q_\alpha(x)| \leq |\hat{\mu}^E_N(x) - \mu^E(x)| + |\Phi^{-1}(\alpha)| \cdot |\hat{\sigma}^E_N(x) - \sigma^E(x)|
$$

#### 任务 2.3.2: 可行性概率界

**定理 (Feasibility Guarantee)**: 设 $\hat{\mathcal{F}}_N$ 为算法判定为可行的解集，$\mathcal{F}^*$ 为真实可行解集。对于任意 $\delta > 0$，以概率至少 $1 - \delta$:
$$
\hat{\mathcal{F}}_N \subseteq \mathcal{F}^*_\epsilon \quad \text{for } \epsilon = O\!\left(\sqrt{\frac{p \log(N/\delta)}{N}}\right)
$$
其中 $\mathcal{F}^*_\epsilon$ 是真实可行域的 $\epsilon$-扩展。

这意味着：预算足够大时，算法输出的解最多只轻微违反约束。

### 2.4 渐近最优性的加强 (中等优先级)

#### 任务 2.4.1: 加强 Theorem 1

当前 Theorem 1 只说"收敛"，需要加强为:
- 明确收敛的模式（几乎必然 vs 依概率 vs 均方）
- 给出收敛速率（即使是较粗的）
- 讨论所需条件的必要性

#### 任务 2.4.2: 一步最优性

证明 KG 采样策略在 myopic (one-step lookahead) 意义下是最优的——这是 KG 的经典性质，在双目标设定下需要重新证明。

---

## 第三部分：实验的全面升级

### 3.1 基准方法对比 (最高优先级)

#### 需要实现和对比的方法:

| 方法 | 缩写 | 实现来源 | 对比意义 |
|------|------|---------|---------|
| Expected Hypervolume Improvement + Constraint | cEHVI | BoTorch | 多目标 BO SOTA |
| ParEGO + Constraint handling | cParEGO | 自行实现 | 经典多目标 BO |
| Constrained Expected Improvement (标量化) | cEI | BoTorch/GPyOpt | 带约束 BO 基线 |
| NSGA-II + Kriging surrogate | NSGA-II-K | pymoo + GPy | 进化算法+代理模型 |
| NSGA-II (direct, 大预算) | NSGA-II-D | pymoo | 作为 gold standard |
| Random Search | RS | - | 最简基线 |
| Latin Hypercube Sampling | LHS | - | 空间填充基线 |
| **GPR-KG (本文, 含 VEPM)** | GPR-KG | - | 完整方法 |
| GPR-KG (无 VEPM) | GPR-KG-nV | - | 消融 |
| GPR-KG (已知方差) | GPR-KG-kV | - | 消融 (upper bound) |

#### 对比指标:
1. **Hypervolume Indicator (HV)**: 主要指标，越大越好
2. **Inverted Generational Distance (IGD)**: 衡量覆盖度和收敛性
3. **约束违反率**: 输出解中违反概率约束的比例
4. **约束违反程度**: 违反解的平均违反量
5. **采样效率曲线**: HV/IGD 随采样预算 N 的变化
6. **计算时间**: 采样决策的 wall-clock time

#### 统计要求:
- 每个设定 **30 次独立重复**
- 报告 **均值 ± 标准误**
- **Wilcoxon 秩和检验** 比较方法间差异
- 绘制 **performance profile** (Dolan-Moré plot)

### 3.2 合成测试问题的扩充 (高优先级)

#### 3.2.1 维度扩展实验

| 问题 | 维度 d | 解空间大小 K | 目的 |
|------|--------|-------------|------|
| RZDT1/2/5 | d = 5 | ~10^3 | 当前基线 |
| RZDT1/2/5 | d = 10 | ~10^6 | 中等维度 |
| RZDT1/2/5 | d = 20 | ~10^12 | 高维度 |
| RZDT1/2/5 | d = 50 | ~10^30 | 极高维 (仅参数化方法可行) |

**关键观察目标**: 展示当 d 增大时，GPR-KG 的性能退化远慢于非参数化方法。

#### 3.2.2 不同 Pareto 前沿形状

| 问题 | 前沿类型 | 来源 |
|------|---------|------|
| RZDT1 (修改版) | 凸 Pareto 前沿 | 已有 |
| RZDT2 (修改版) | 凹 Pareto 前沿 | 已有 |
| DTLZ2-C | 球面 Pareto 前沿 + 约束 | Deb et al. (2002) 修改版 |
| BNH-C | 不连通 Pareto 前沿 + 约束 | Binh & Korn 修改版 |
| TNK-C | 不连通可行域 | Tanaka 修改版 |

#### 3.2.3 约束难度变化

固定 RZDT1，变化约束阈值使可行域占比为:
- 90% (松约束)
- 50% (中等约束)  
- 10% (紧约束)

**关键观察目标**: GPR-KG + VEPM 在紧约束下的优势应更显著。

#### 3.2.4 噪声水平变化

固定 RZDT1 (d=10)，变化噪声标准差:
- σ = 0.01 (低噪声)
- σ = 0.1 (中等噪声)
- σ = 1.0 (高噪声)
- 异方差噪声: σ(x) = 0.1 + 0.9 * f(x) (与目标值相关)

**关键观察目标**: VEPM 在异方差噪声下应优于假设已知方差的方法。

### 3.3 消融实验 (高优先级)

| 实验 | 变化组件 | 目的 |
|------|---------|------|
| A1 | VEPM vs 无VEPM vs 已知方差 | 验证 VEPM 贡献 |
| A2 | 不同分区粒度 (2, 4, 8 partitions/dim) | VEPM 分区选择指导 |
| A3 | 不同基函数 (线性 vs 二次 vs 含交叉项) | 信念模型选择指导 |
| A4 | 不同候选集生成策略 | 验证混合策略的必要性 |
| A5 | KG vs EI vs UCB 采样策略 (相同信念模型) | 验证 KG 的优势 |
| A6 | 单目标标量化 vs 双目标 Pareto-KG | 验证多目标处理的优势 |

### 3.4 可扩展性实验 (中等优先级)

绘制以下量关于问题维度 d 和预算 N 的曲线:
- 每次迭代的采样决策时间
- 内存使用量
- 后验问题分解的子问题数量

---

## 第四部分：典型审稿意见的提前布局

### Q1: "为什么不直接用标准 GP？参数化信念模型的优势在哪里？"

**布局位置**: Section 3 (方法) + Section 5 (理论) + Section 6 (实验)

**理论回应**:
- 标准 GP 需维护 K×K 协方差矩阵，当 K = O(L^d) (L 为每维取值数) 时不可行
- 参数化模型将有效参数数从 K 降至 p (基函数数量)
- 在 Section 5 中证明: simple regret 依赖 p 而非 K
- 在 Remark 中明确指出: "When $K \gg p$, the parametric model achieves a statistical learning rate that scales with the effective model dimension $p$ rather than the ambient space dimension $K$."

**实验回应**:
- 在 d=5 的小问题上对比标准 GP (可行) 和参数化 GP，展示:
  - 性能相近或参数化模型更好 (因为隐式正则化)
  - 计算时间差异巨大
- 在 d=20 的问题上展示标准 GP 已不可行 (内存/时间爆炸)

**在论文中的具体写法**:

> **Remark X.** The standard (non-parametric) GP approach maintains a K-dimensional posterior distribution over the mean vector, where K is the total number of feasible solutions. For the NSTPU with d = 47 decision dimensions, K exceeds 10^{70}, making the standard GP computationally intractable. In contrast, our parametric belief model maintains only a p-dimensional posterior, where p is the number of basis functions (e.g., p = 41 for a quadratic model with 20 critical decision variables). This dimensional reduction is the key enabler for applying Bayesian learning to problems with massive solution spaces. Theorem X formalizes this advantage by showing that the regret bound scales with p rather than K.

---

### Q2: "VEPM 的分区数量指数增长怎么办？"

**布局位置**: Section 3.2 (VEPM 方法) + Section 5 (理论) + Section 6 (消融实验)

**理论回应**:
- 承认分区数 $\prod_j m_j$ 在最坏情况下指数增长
- 但指出:
  1. 实际中每维仅需 2-4 个分区，总分区数可控
  2. 分区仅用于方差估计，不影响均值学习的维度
  3. 不需要每个分区都有样本——未被采样的分区保留初始估计

**实际策略**:
- 提出 **自适应分区方案**: 仅对方差变化显著的维度进行分区，其余维度合并
- 具体实现: 
  1. 先用粗分区 (如每维 2 个) 运行
  2. 根据同一分区内方差估计的离散程度，决定是否细分
  3. 给出分区选择的信息准则 (类似 BIC)

**实验回应**:
- 消融实验 A2: 不同分区粒度对性能的影响
- 展示: 每维 2 分区通常已足够，进一步细分的边际收益很小

**在论文中的具体写法**:

> **Remark Y.** The total number of partition combinations is $\prod_{j=1}^d m_j$, which grows exponentially in $d$ if all dimensions are partitioned. However, we note that (i) only dimensions with significant variance heterogeneity benefit from partitioning—the remaining dimensions can use a single partition; (ii) a binary partition ($m_j = 2$) per active dimension is typically sufficient, as shown in our ablation study (Section 6.X); (iii) the partition structure only affects variance estimation—the mean learning, which determines the Pareto front identification, operates independently via the parametric belief model (Section 3.1). In practice, we recommend partitioning only those dimensions selected by a preliminary variance screening procedure (Algorithm X in the Online Supplement) and using binary partitions for each.

---

### Q3: "概率约束中的高斯假设多强？"

**布局位置**: Section 2 (问题设定) + Section 5 (理论) + Section 6 (实验) + Online Supplement

**理论回应**:
- 在问题设定中明确声明高斯假设，讨论其合理性:
  - 交通评价指标是大量微观行为的聚合 → 中心极限定理支持
  - 引用 Fig. 7 中的经验验证
- 讨论假设失效时的后果:
  - 均值估计不受影响 (CLT 仍保证样本均值的渐近正态性)
  - 分位数估计会有偏差，影响可行性判断
  - 量化偏差: 如果真实分布的偏度为 $\gamma$，则分位数估计的 bias 约为 $O(\gamma \sigma / \sqrt{n_x})$

**扩展方案 (写入论文作为 extension)**:
- 提出基于 **Cornish-Fisher 展开** 的非高斯修正:
$$
\hat{q}_\alpha(x) = \hat{\mu}(x) + \left[\Phi^{-1}(\alpha) + \frac{\hat{\gamma}(x)}{6}((\Phi^{-1}(\alpha))^2 - 1)\right] \hat{\sigma}(x)
$$
其中 $\hat{\gamma}(x)$ 是偏度估计
- 或提出基于 **bootstrap 分位数估计** 的非参数版本

**实验回应**:
- 在合成实验中加入非高斯噪声设定:
  - 卡方分布噪声 (偏态)
  - t 分布噪声 (厚尾)
  - 混合高斯噪声 (双峰)
- 展示: 高斯假设下的方法仍有较好的鲁棒性，但在强偏态下非高斯修正有帮助

---

### Q4: "与 constrained multi-objective BO 的区别和优势是什么？"

**布局位置**: Section 1 (Introduction) + Section 6 (实验)

**理论层面的区分**:

| 特征 | 标准 cMOBO (如 cEHVI) | 本文 GPR-KG |
|------|----------------------|------------|
| 解空间 | 连续，通常 d ≤ 20 | 离散，d 可达 50+ |
| 信念模型 | 非参数 GP (O(N³)) | 参数化 GP (O(p²N)) |
| 约束类型 | 确定性约束 P(g(x)≤0) 的 GP 建模 | 概率约束 (分位数) |
| 方差处理 | 假设已知或 homoscedastic | 未知异方差 + VEPM |
| 可扩展性 | K ≤ ~10⁴ | K 可达 10^{70}+ |

**实验层面的区分**:
- 在小规模问题 (d=5, K~10³) 上: 两类方法性能相近
- 在中/大规模问题 (d≥10, K≥10⁶) 上: cEHVI 等方法因 GP 维护开销过大而不可行或效率极低，GPR-KG 仍可正常运行

**在论文中的具体写法**:

> Our framework differs from constrained multi-objective Bayesian optimization (cMOBO) methods such as cEHVI (Daulton et al. 2022) in three fundamental ways. First, cMOBO methods attach independent GP priors to each objective and constraint function, requiring $O(N^3)$ posterior updates and $O(N^2)$ memory per function, which limits their applicability to problems where the total number of evaluated solutions $N$ is in the hundreds. In contrast, our parametric belief model requires only $O(p^2 N)$ computation, enabling thousands of simulation evaluations. Second, cMOBO methods model the constraint as $P(g(x) \leq 0)$ using a GP posterior on $g$, which does not directly account for the stochastic nature of $g$ itself. Our framework explicitly models the probabilistic constraint through quantile estimation, which is more natural when the constraint involves a stochastic quantity evaluated by simulation. Third, cMOBO methods are designed for continuous search spaces and would require discretization or rounding heuristics for our integer-valued decision space, whereas our method is natively discrete. Our numerical experiments in Section 6 confirm these structural advantages on problems where the solution space exceeds $10^6$ alternatives.

---

### Q5: "交通案例太具体了，方法真的通用吗？"

**布局位置**: Section 1 + Section 6 + Section 7

**回应策略**:
1. 在 Introduction 中列举 3-4 个其他领域的 motivating examples:
   - **制造系统**: 多条产线的缓冲区配置，双目标(吞吐量+在制品)，排放约束
   - **供应链**: 多级库存策略优化，双目标(服务水平+成本)，碳排放概率约束
   - **医疗**: 急诊科人员配置，双目标(等待时间+资源利用率)，死亡率概率约束
   - **能源**: 微电网调度，双目标(成本+可靠性)，排放概率约束

2. 在合成实验中加入非交通背景的测试问题:
   - (s,S) 库存问题的双目标版本
   - M/M/c 排队系统的配置优化

3. 交通案例作为 "Case Study" 而非主要实验

---

### Q6: "KG 采样策略为什么比 EI 或 UCB 好？"

**布局位置**: Section 4 + Section 6 (消融实验 A5)

**理论回应**:
- KG 的 one-step optimality 性质 (在双目标设定下重新证明)
- KG 天然适合 finite-budget 问题 (offline learning)，而 UCB 更适合 online 问题
- 在参数化信念模型下，KG 因子可以解析计算，计算成本与 EI/UCB 相当

**实验回应**:
- 消融实验 A5: 固定信念模型和 VEPM，仅替换采样策略
  - KG (本文)
  - Expected Improvement (标量化后)
  - Upper Confidence Bound
  - Thompson Sampling
- 预期结果: KG 在有限预算下优势明显，尤其在预算较小时

---

### Q7: "Theorem 1 的条件 (3) 是否太强？实际中如何验证？"

**布局位置**: Section 5 (理论分析)

**回应策略**:
- 讨论条件 (3) (高斯分布假设) 的放松可能:
  - 如果仅假设有限方差，均值学习的收敛仍成立 (CLT argument)
  - 概率约束的处理需要更多分布信息，但 Cornish-Fisher 修正可放松至三阶矩存在
- 提供实际验证方法:
  - Shapiro-Wilk 检验
  - Q-Q plot
  - 在 Case Study 中展示验证结果 (已有 Fig. 7)

---

### Q8: "偏差项 ε(x) 的先验 N(0, λI) 是否合理？λ 如何选？"

**布局位置**: Section 3.1 + Online Supplement

**回应策略**:
- λ 控制模型的灵活性: λ 大 → 偏差项占主导 → 接近非参数 GP; λ 小 → 参数化成分占主导
- 提出 **经验贝叶斯** 方法选择 λ: 最大化边际似然
- 讨论对结果的敏感性 (在消融实验中加入 λ 的敏感性分析)

---

### Q9: "论文只考虑了双目标，能否推广到三目标或更多？"

**布局位置**: Section 8 (Conclusion) 中的 future work + 简短讨论

**回应策略**:
- 方法框架可自然推广: 信念模型和 VEPM 与目标数量无关
- KG 采样策略: Pareto-KG 的推广需要高维 Pareto 支配的计算，复杂度增加
- 后验问题分解 (Proposition 2): 仍成立，但子问题变为多目标
- 承认三目标以上的可视化和解选择是实际挑战

---

### Q10: "开源代码和可复现性"

**布局位置**: Section 6 开头 + 投稿时的补充材料

**回应策略**:
- 提供 GitHub 仓库，包含:
  - GPR-KG 算法的 Python/MATLAB 实现
  - 所有合成测试问题的代码
  - 实验复现脚本
  - 交通案例的数据 (或匿名化版本)
- OR 近年非常重视可复现性，这是重要加分项

---

## 第五部分：写作提升要点

### 5.1 Introduction 的重写策略

**第一段**: 从一般性 SO 问题出发
> Simulation optimization (SO) seeks optimal decisions when the objective can only be evaluated through stochastic simulation... Many real-world SO problems involve multiple conflicting objectives and probabilistic constraints...

**第二段**: 指出现有方法的不足 (3 个 gap)
> Gap 1: 大规模离散空间 + 双目标 + 概率约束的组合未被解决
> Gap 2: KG 文献中缺乏对概率约束的处理
> Gap 3: 方差未知且异方差时的高效学习

**第三段**: 本文贡献 (清晰列出 3-4 点)

**第四段**: 应用动机 (交通只是一个例子)

### 5.2 记号精简

- 统一使用上标表示目标索引 (i=1,2)、约束 (i=3)
- 减少不必要的重新定义
- 在 Online Supplement 中给出完整的符号表

### 5.3 语言质量

- 建议请 native speaker 润色
- 避免过长的句子 (当前一些句子超过 50 词)
- 每个 Section 以一句话总结本节贡献

---

## 第六部分：执行时间线

### 阶段一: 理论推导 (第 1-8 周)

| 周 | 任务 | 交付物 |
|----|------|--------|
| 1-2 | VEPM 一致性和收敛速率证明 | 定理证明手稿 |
| 3-4 | 有限预算 regret bound 推导 | 定理证明手稿 |
| 5-6 | 概率约束可行性保证推导 | 定理证明手稿 |
| 7-8 | 理论结果整理，统一符号，检查证明 | Section 5 初稿 |

### 阶段二: 实验扩充 (第 5-14 周，与阶段一部分并行)

| 周 | 任务 | 交付物 |
|----|------|--------|
| 5-6 | 实现基准方法 (cEHVI, cParEGO, cEI, NSGA-II-K) | 代码 |
| 7-8 | 合成测试: 维度扩展 + Pareto 前沿形状 | 实验结果 |
| 9-10 | 合成测试: 约束难度 + 噪声水平 + 非高斯噪声 | 实验结果 |
| 11-12 | 消融实验 (A1-A6) | 实验结果 |
| 13-14 | 统计分析 + 图表制作 | Section 6 初稿 |

### 阶段三: 论文重写 (第 13-20 周，与阶段二部分并行)

| 周 | 任务 | 交付物 |
|----|------|--------|
| 13-14 | Introduction + Problem Formulation 重写 | Section 1-2 |
| 15-16 | Method Section 重写 (通用化) | Section 3-4 |
| 17-18 | Theory Section + Experiments Section | Section 5-6 |
| 19-20 | Case Study 精简 + Conclusion + Online Supplement | 完整初稿 |

### 阶段四: 完善与投稿 (第 21-24 周)

| 周 | 任务 | 交付物 |
|----|------|--------|
| 21-22 | 内部审阅 + 修订 | 修订稿 |
| 23 | 代码整理 + GitHub 仓库 | 开源代码 |
| 24 | 语言润色 + 格式调整 + 投稿 | 最终稿 |

---

## 第七部分：风险与应对

| 风险 | 概率 | 影响 | 应对策略 |
|------|------|------|---------|
| Regret bound 推导困难 | 高 | 高 | 先尝试较粗的 bound (如 O(√(p/N)) without log factors)，逐步加强；若实在困难，可用 information-theoretic 框架 (如 maximum information gain) 给出间接 bound |
| VEPM 一致性证明需要强条件 | 中 | 中 | 如果一般条件下证不出，先在特殊情况 (如均匀采样) 下证明，再用实验补充一般情况 |
| cEHVI 在大问题上跑不动 (无法对比) | 中 | 中 | 在小问题上对比，在大问题上用 "did not finish within time limit" 说明可扩展性优势 |
| 非高斯噪声实验结果不好 | 低 | 中 | 如果高斯假设下方法的鲁棒性确实差，则提出 Cornish-Fisher 修正并展示改进 |
| OR 审稿人认为交通案例不够通用 | 中 | 高 | 加入 1-2 个非交通领域的案例 (如库存优化) |

---

## 附录：关键参考文献补充清单

### 多目标 BO:
1. Daulton et al. (2020, NeurIPS) — qNEHVI, 多目标 BO 的 SOTA
2. Knowles (2006) — ParEGO
3. Emmerich et al. (2006) — Expected Hypervolume Improvement
4. Zuluaga et al. (2016, JMLR) — ε-PAL, 多目标 active learning

### 带约束 BO:
5. Gelbart et al. (2014, UAI) — Constrained BO with unknown constraints
6. Letham et al. (2019, Bayesian Analysis) — Constrained BO with noisy experiments
7. Eriksson & Poloczek (2021, NeurIPS) — Scalable constrained BO

### KG 理论:
8. Frazier et al. (2008, 2009) — KG 的理论基础
9. Wu & Frazier (2016) — Parallel KG
10. Balandat et al. (2020, NeurIPS) — BoTorch framework

### Regret bounds:
11. Srinivas et al. (2010, ICML) — GP-UCB regret bound
12. Vakili et al. (2021) — NTK kernel regret bound
13. Kandasamy et al. (2020) — Multi-objective BO regret

### 仿真优化:
14. Hong et al. (2021) — SO review (Operations Research)
15. Salemi et al. (2019, Operations Research) — GMRF for discrete SO
16. Semelhago et al. (2021, IJOC) — Rapid discrete SO

---

*本计划应根据实际推进情况动态调整。建议每两周召开一次进展讨论会，及时处理瓶颈问题。*

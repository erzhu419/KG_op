# GPR-KG 算法详细修改方案：MATLAB转Python + 新旧版对比修复

**制定日期**: 2026-04-11  
**目标**: 将旧版MATLAB代码转为Python，复现旧版实验，系统对比新旧版，修复新版问题，运行对比实验

---

## 一、总体思路

```
旧版MATLAB代码 ──转换──→ 旧版Python代码 ──复现──→ 旧版实验结果(3问题×10次)
                              │                          │
                              │                     对比验证(Table A)
                              │                          │
                              ▼                          ▼
新版Python代码 ←──参照修复──── 差异分析报告 ←──── 旧论文Table 1
     │
     ├── 修复前备份 → baseline结果(3问题×10次)
     │
     └── 修复后 → 新版结果(3问题×10次)
                    │
                    └── 对比实验报告(Table B, C)
```

---

## 二、第一阶段：MATLAB → Python 逐文件转换

### 2.1 目录结构

```
OR投稿/OldVersion_Python/
├── core/
│   ├── __init__.py
│   ├── test_problems.py      ← sim_func.m, sim_test.m
│   ├── basis_functions.py    ← feat.m
│   ├── vepm.py               ← part_id.m, update_var.m, var_x.m
│   ├── kalman_update.py      ← update_coeff.m
│   ├── kg_computation.py     ← KG_factor.m, KG_sol.m
│   ├── candidate_gen.py      ← cand_sample.m, bi_obj.m, nonlcon.m
│   ├── pareto_utils.py       ← perato.m, perato_con.m, crowding_distance.m
│   ├── initialization.py     ← pre_sample.m
│   └── utils.py              ← x_in_s.m
├── run_experiment.py          ← example_with_post_process.m
├── run_all.py                 # 批量运行3问题×10次
├── analyze_results.py         # 结果分析+表格生成
└── results/                   # JSON结果文件
```

### 2.2 转换顺序（按依赖关系）

**第1批（无依赖）:**

| # | 文件 | MATLAB源 | 核心逻辑 | 预计行数 |
|---|------|---------|---------|---------|
| 1 | `basis_functions.py` | `feat.m` | `phi(x) = [1, x/100, 2*(x/100)^2]`，特征维度p=2d+1=11 | ~30行 |
| 2 | `pareto_utils.py` | `perato.m`, `perato_con.m`, `crowding_distance.m` | Pareto非支配排序、带约束Pareto排序、拥挤距离 | ~120行 |
| 3 | `utils.py` | `x_in_s.m` | 检查解是否在集合中 | ~15行 |

**第2批（依赖第1批）:**

| # | 文件 | MATLAB源 | 核心逻辑 | 预计行数 |
|---|------|---------|---------|---------|
| 4 | `test_problems.py` | `sim_func.m`, `sim_test.m` | RZDT1/2/5三个测试问题（含噪声+无噪声版本） | ~100行 |
| 5 | `vepm.py` | `part_id.m`, `update_var.m`, `var_x.m` | VEPM分区ID计算、方差更新、方差查询 | ~150行 |
| 6 | `kalman_update.py` | `update_coeff.m` | Kalman rank-one更新 b, B | ~40行 |

**第3批（依赖第1-2批）:**

| # | 文件 | MATLAB源 | 核心逻辑 | 预计行数 |
|---|------|---------|---------|---------|
| 7 | `kg_computation.py` | `KG_factor.m`, `KG_sol.m` | log(KG)计算（h函数）+ KG解选择（Pareto+加权平均） | ~200行 |
| 8 | `candidate_gen.py` | `cand_sample.m`, `bi_obj.m`, `nonlcon.m` | K1=20 LHD随机 + K2=2 后验Pareto采样 | ~150行 |

**第4批（依赖所有）:**

| # | 文件 | MATLAB源 | 核心逻辑 | 预计行数 |
|---|------|---------|---------|---------|
| 9 | `initialization.py` | `pre_sample.m` | N0=50 LHD预采样 + 线性回归初始化 b, B, z0 | ~80行 |
| 10 | `run_experiment.py` | `example_with_post_process.m` | 主算法循环 + 后处理 | ~200行 |

### 2.3 每个文件的转换细节

#### 2.3.1 `test_problems.py` ← `sim_func.m`

```python
"""
旧版测试问题定义
解空间: x ∈ [0, 100]^d, 连续域
输出: [f1(x)+ε1, f2(x)+ε2, g(x)+ε3], g为约束函数
噪声: εi ~ N(0, stdev[i]^2), stdev = [0.05, 0.05, 0.05]
"""

def rzdt1(x, stdev, rng=None):
    """RZDT1: 凸Pareto前沿 (基于ZDT1)
    f1(x) = x1/100
    g(x) = 1 + 9/(d-1) * sum(xi/100, i=2..d)
    f2(x) = g * (1 - sqrt(x1/(100*g)))
    h(x) = -(x1/100 - 0.5)^2  (约束)
    """

def rzdt2(x, stdev, rng=None):
    """RZDT2: 凹Pareto前沿 (基于ZDT2)
    与RZDT1相同，但 f2(x) = g * (1 - (x1/(100*g))^2)
    """

def rzdt5(x, stdev, rng=None):
    """RZDT5: 不连续Pareto前沿 (基于ZDT5)
    f1(x) = 1 + floor(30*x1/100)/5
    g(x) = d-1 + sum(v(xi), i=2..d)  其中v(xi) = 1+u(xi)+u(xi)^2
    u(xi) = floor(xi/100 * 31) - 15 的某种变换
    f2(x) = g/f1  (注意除法关系)
    h(x) = -(x1/100 - 0.5)^2
    """

def sim_test(x, problem='RZDT1'):
    """无噪声版本，用于计算真实目标值"""
```

**注意**: MATLAB中`sim_func.m`只实现了RZDT2，需要从旧论文Appendix B获取RZDT1和RZDT5的完整公式。

#### 2.3.2 `vepm.py` ← `part_id.m` + `update_var.m` + `var_x.m`

```python
class VEPM_Old:
    """旧版VEPM实现
    
    关键参数 (来自example_with_post_process.m):
    - key{i} = [eye(d), 2*eye(d)]  → 10个特征(5线性+5二次对角)
    - F_part{i}{j} = [0, 0.5, 1]   → 每个特征2个bin
    - 总分区数: 2^10 = 1024
    - n_thr = 20: 启用方差更新的最小样本阈值
    - s0 = 1: 先验权重
    - var0 = 0.01 * ones(3): 先验方差
    
    分区ID计算 (part_id.m):
    1. 计算特征值: feat_val = key @ x (10维)
    2. 归一化到[0,1]: feat_norm = (feat_val - min) / (max - min)
    3. 对每个特征j，找到feat_norm在F_part{j}中的bin索引
    4. 组合: partition_id = sum(bin_j * prod(m_k, k<j))
    
    方差更新 (update_var.m):
    1. 对每个目标i:
       sample_var = (y[i] - mu_pred[i])^2
    2. 找到所有与当前x同分区的解
    3. 加权平均更新:
       z[partition][i] = (s0*var0[i] + sum_sq) / (s0 + n_partition)
       其中sum_sq = sum of sample_var for all solutions in partition
    
    方差查询 (var_x.m):
    直接返回该x所在分区的z值
    """
    
    def __init__(self, d, n_objectives=3):
        self.d = d
        self.n_obj = n_objectives
        self.n_features = 2 * d  # 线性 + 二次特征
        self.bins_per_feature = 2
        self.total_partitions = 2 ** self.n_features  # 2^10 = 1024 for d=5
        
        # 特征变换矩阵: key = [I_d, 2*I_d]
        self.key = np.hstack([np.eye(d), 2*np.eye(d)])
        # 分区边界: 每个特征在[0,1]上分2个bin
        self.bin_edges = np.array([0, 0.5, 1])
        
        # 先验参数
        self.s0 = 1
        self.var0 = 0.01 * np.ones(n_objectives)
        self.n_thr = 20
        
        # 存储: 每个分区的累计信息
        self.partition_sum_sq = np.zeros((self.total_partitions, n_objectives))
        self.partition_counts = np.zeros(self.total_partitions, dtype=int)
        self.z = np.tile(self.var0, (self.total_partitions, 1))  # 初始方差
    
    def get_partition_id(self, x, x_L, x_U):
        """计算x的分区ID"""
        feat_vals = self.key @ x  # 10维特征
        feat_norm = (feat_vals - self.key @ x_L) / (self.key @ x_U - self.key @ x_L)
        feat_norm = np.clip(feat_norm, 0, 0.999)
        bin_ids = np.digitize(feat_norm, self.bin_edges[1:-1])  # 0或1
        # 组合成单一ID
        pid = 0
        for j, bid in enumerate(bin_ids):
            pid += bid * (self.bins_per_feature ** j)
        return pid
    
    def update(self, x, y_obs, mu_pred, partition_id):
        """更新分区方差"""
        sample_var = (y_obs - mu_pred) ** 2
        self.partition_sum_sq[partition_id] += sample_var
        self.partition_counts[partition_id] += 1
        
        n = self.partition_counts[partition_id]
        if n >= self.n_thr:
            self.z[partition_id] = (
                self.s0 * self.var0 + self.partition_sum_sq[partition_id]
            ) / (self.s0 + n)
    
    def get_variance(self, x, x_L, x_U):
        """查询x的噪声方差估计"""
        pid = self.get_partition_id(x, x_L, x_U)
        return self.z[pid]
```

#### 2.3.3 `kg_computation.py` ← `KG_factor.m` + `KG_sol.m`

```python
def compute_log_kg(F_t, b_t, B_t, lambda_x, f_x, eps=1e-10):
    """计算log(KG)值
    
    来自KG_factor.m:
    F_t: 当前Pareto前沿解的特征矩阵 (n_pareto × p)
    b_t: 当前后验均值系数 (p,)
    B_t: 当前后验协方差矩阵 (p × p)
    lambda_x: 候选解x的噪声方差
    f_x: 候选解x的特征向量 (p,)
    
    计算过程:
    1. p_vec = -F_t @ b_t  (当前Pareto后验均值的负值)
    2. gamma = lambda_x + f_x @ B_t @ f_x
    3. c_tilde = F_t @ B_t @ f_x / sqrt(gamma)
    4. q_vec = c_tilde  (KG的变化方向)
    5. log_kg = h_function(p_vec, q_vec)
    
    h函数 (Frazier & Powell 2009):
    h(p, q) = log(E[max_i(p_i + q_i * Z)]) where Z ~ N(0,1)
    """

def kg_solution_selection(kg_values, pareto_indices, method='weighted_avg'):
    """KG解选择
    
    来自KG_sol.m:
    1. 计算每个目标的KG值
    2. 在KG空间中找Pareto非支配解
    3. 用加权平均(0.5/0.5)选择最终采样点
    
    关键: 旧版使用简单加权平均而非拥挤距离
    """
```

#### 2.3.4 `candidate_gen.py` ← `cand_sample.m` + `bi_obj.m` + `nonlcon.m`

```python
def generate_candidates(K1, K2, b, B, x_L, x_U, tau_e, alph, 
                        basis_func, current_solutions):
    """候选点生成
    
    来自cand_sample.m:
    K1=20: LHD随机点
    K2=2:  后验采样 + gamultiobj优化
    
    步骤:
    1. 生成K1个LHD随机点覆盖[x_L, x_U]
    2. 重复K2次:
       a. 从N(b, B)采样参数theta
       b. 定义后验目标: f1_post(x)=phi(x)@theta[:p], f2_post(x)=phi(x)@theta[p:2p]
       c. 定义后验约束: g_post(x)=phi(x)@theta[2p:] + alph*sigma
       d. 用gamultiobj(bi_obj, nonlcon)求后验Pareto前沿
       e. 从后验Pareto前沿中随机选若干点
    3. 合并所有候选点，去除已在current_solutions中的
    
    Python替代gamultiobj: 使用pymoo的NSGA2
    """
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize as pymoo_minimize
    
    candidates = []
    
    # Part 1: K1个LHD随机点
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=len(x_L))
    lhd = sampler.random(K1)
    lhd_scaled = x_L + lhd * (x_U - x_L)
    candidates.extend(lhd_scaled)
    
    # Part 2: K2次后验采样
    for _ in range(K2):
        theta = np.random.multivariate_normal(b, B)
        # 用NSGA-II在后验目标上求Pareto前沿
        # ... (用pymoo实现)
        # 从Pareto前沿取点加入候选集
    
    return candidates
```

#### 2.3.5 `initialization.py` ← `pre_sample.m`

```python
def initialize(N0, d, x_L, x_U, sim_func, stdev):
    """预采样和初始化
    
    来自pre_sample.m:
    1. 生成N0=50个LHD样本
    2. 在每个样本点仿真一次
    3. 对每个目标i，用线性回归拟合: y_i = phi(x) @ beta_i
    4. 初始化:
       b = [beta_1; beta_2; beta_3]  (拼接3个目标的系数)
       B = var(beta) * I  (简单对角初始化)
       z0 = var(residuals)  (残差方差作为噪声方差初始估计)
    
    关键: 旧版用 B = var(b_hat) * eye(p)
    其中var(b_hat)是线性回归系数的方差
    """
```

#### 2.3.6 `run_experiment.py` ← `example_with_post_process.m`

```python
def run_gpr_kg_old(problem_name, seed, params=None):
    """运行旧版GPR-KG算法
    
    默认参数 (来自example_with_post_process.m):
    N = 100        # 迭代次数
    d = 5          # 决策维度
    N0 = 50        # 预采样数
    K1 = 20        # 随机候选点数
    K2 = 2         # 后验采样次数
    m = 40         # 后验Pareto采样相关参数
    n_thr = 20     # VEPM最小样本阈值
    s0 = 1         # VEPM先验权重
    var0 = 0.01    # VEPM先验方差
    stdev = 0.05   # 仿真噪声标准差
    tau_e = -0.04  # 约束阈值
    alph = 1.645   # 约束分位数(5%单侧)
    x_L = 0^d      # 下界
    x_U = 100^d    # 上界
    
    基函数: phi(x) = [1, x1/100,...,xd/100, 2*(x1/100)^2,...,2*(xd/100)^2]
    特征数: p = 2d+1 = 11
    
    算法流程:
    1. 预采样 (initialization.py)
    2. for n = 1 to N:
       a. 生成候选点 (candidate_gen.py)
       b. 对每个候选点计算KG值 (kg_computation.py)
       c. 选择最优候选点 (kg_computation.py)  
       d. 仿真采样
       e. Kalman更新 b, B (kalman_update.py)
       f. VEPM更新方差 (vepm.py)
    3. 后处理: 在所有已采样解中找Pareto前沿
    
    输出字典:
    {
        'problem': str,
        'seed': int,
        'pareto_solutions': ndarray,
        'pareto_objectives_true': ndarray,  # 用sim_test计算真实值
        'hv_final': float,
        'igd_final': float,
        'n_lpos': int,
        'rmse': float,
        'n_infeasible': int,
        'time_total': float,
        'hv_history': list  # [(stage, hv), ...]
    }
    """
```

### 2.4 复现实验设计

```
运行配置:
- 问题: RZDT1, RZDT2, RZDT5
- 方法: GPR-KG (with VEPM), GPR-KG-nV (without VEPM)
- 重复: 10次 (seed = 1000, 1001, ..., 1009)
- 参数: 完全按2.3.6中默认参数

评价指标:
- IGD: 与真实Pareto前沿的逆代距离
- #LPOS: 学到的Pareto最优解数量
- RMSE: 参数估计均方根误差
- 不可行解数量
- 计算时间

验证标准:
- RZDT1 IGD ≈ 0.055 (论文Table 1)
- RZDT2 IGD ≈ 0.022 (论文Table 1)
- RZDT5 IGD ≈ 0.005 (论文Table 1)
- 允许误差: ±50% (因为仅10次重复且随机种子不同)
```

---

## 三、第二阶段：新旧版系统对比

### 3.1 对比维度

#### 3.1.1 问题设定对比

| 方面 | 旧版 | 新版 | 评价 |
|------|------|------|------|
| 解空间 | [0,100]^d 连续 | {1,...,L}^d 离散 | 新版更一般化 ✅ |
| 维度 | d=5 固定 | d=5,10,20 | 新版范围更广 ✅ |
| 目标数 | 2目标+1约束 | 2目标+1约束 | 相同 |
| 噪声 | σ=0.05 同方差 | σ=0.1 + 异方差实验 | 新版更全面 ✅ |

#### 3.1.2 算法设计对比

| 组件 | 旧版 | 新版 | 评价 |
|------|------|------|------|
| **VEPM分区** | 1024分区 (2^10) | **4分区** | ❌ 新版严重退化 |
| **候选点生成** | 20 LHD + 2 gamultiobj后验 | 50随机 + 20扰动 | ❌ 新版失去智能探索 |
| 基函数 | [1, x/100, 2(x/100)^2], p=11 | [1, x, x^2], p=2d+1 | 基本一致 |
| Kalman更新 | 标准rank-one | 标准rank-one | ✅ 一致 |
| KG计算 | h函数(Frazier 2009) | h函数(Frazier 2009) | ✅ 一致 |
| KG选择 | Pareto+加权平均(0.5/0.5) | Pareto+拥挤距离 | 轻微差异 |
| 预采样 | N0=50 | n0=30 | 新版偏少 |

#### 3.1.3 理论对比

| 理论结果 | 旧版 | 新版 | 评价 |
|---------|------|------|------|
| 后验一致性 | 有(Theorem 1) | 有(加强版) | 新版更严格 ✅ |
| VEPM收敛性 | 未证明 | 有(Proposition) | 新版增加 ✅ |
| KG非负性 | 有 | 有 | 相同 |
| 约束可行性 | 未证明 | 有(Theorem) | 新版增加 ✅ |
| Regret bound | 无 | 有(新增) | 新版增加 ✅ |

### 3.2 关键差异深度分析

#### 差异1: VEPM分区 (最关键)

**旧版** (`part_id.m`):
```matlab
key{i} = [eye(n), 2*eye(n)];      % 10个特征
F_part{i}{j} = [0 0.5 1];          % 每个特征2个bin
% 总分区: 2^10 = 1024
```

**新版** (`gpr_kg.py` line ~47):
```python
self.bins = [4] + [1] * (self.d - 1)  # 仅x1分4格，其余不分
# 总分区: 4 × 1^(d-1) = 4
```

**影响分析**:
- 1024个分区 → 每个分区约覆盖解空间的0.1%，方差估计精细
- 4个分区 → 每个分区覆盖25%的解空间，完全无法区分不同区域的噪声水平
- 在异方差场景下，4个分区几乎等于没有VEPM
- 这解释了为什么GPR-KG-nV(无VEPM)反而有时更好——4分区的VEPM引入了错误的方差信息

#### 差异2: 候选点生成 (第二关键)

**旧版** (`cand_sample.m`):
```matlab
% K1=20个LHD随机 (覆盖)
% K2=2次后验Pareto采样:
theta = mvnrnd(b, B);  % 采样后验参数
[x_pareto, ~] = gamultiobj(@bi_obj, n, [], [], x_L, x_U);
% gamultiobj: MATLAB多目标遗传算法，在后验目标上求Pareto前沿
```

**新版** (`gpr_kg.py`):
```python
# 50个随机点
rand_cands = rng.integers(1, L+1, size=(M_rand, d))
# 20个扰动点 (在当前Pareto解附近)
for x_base in current_pareto[:M_post]:
    x_new = x_base + rng.integers(-2, 3, size=d)
```

**影响分析**:
- 旧版的gamultiobj后验采样是"智能探索"——它在参数的后验分布下寻找可能的Pareto最优解
- 新版的随机+扰动是"盲目搜索"——在高维(d=10,20)时，70个点覆盖率极低
- 解空间d=5: 100^5=10^10 (旧版) vs 20^5≈3.2×10^6 (新版)
  - 旧版22个候选点/10^10 vs 新版70个候选点/3.2×10^6 → 新版比例更好
- 解空间d=20: 20^20≈10^26
  - 新版70个候选点/10^26 → 几乎没有覆盖能力
  - 后验采样在这里至关重要

---

## 四、第三阶段：修复新版算法

### 4.1 修复优先级

| 优先级 | 问题 | 修复方案 | 影响 |
|--------|------|---------|------|
| **P0** | VEPM仅4分区 | 恢复多特征多bin分区 | 方差估计精度提升100倍+ |
| **P0** | 候选点无后验采样 | 增加NSGA-II后验采样 | 高维探索能力质变 |
| **P1** | 方差估计偏差 | 减去模型不确定性 | 早期估计更准确 |
| **P2** | 预采样偏少 | n0从30增至50 | VEPM初始化更充分 |
| **P3** | KG选择策略 | 保持现有拥挤距离(可选回退加权平均) | 影响较小 |

### 4.2 Fix 0a: VEPM分区策略重建

**修改文件**: `gpr_kg.py` 中的 VEPM 类

**当前代码** (约第47行附近):
```python
self.bins = [4] + [1] * (self.d - 1)
```

**修改为**:
```python
class VEPM:
    def __init__(self, d, L, n_objectives=3):
        self.d = d
        self.n_obj = n_objectives
        
        # 特征: 线性 + 二次对角 = 2d个特征
        # 与旧版一致: key = [I_d, 2*I_d]
        self.n_features = 2 * d
        
        # 自适应分区: 目标约1024个分区
        # bins_per_feature^n_features ≈ 1024
        # → bins = round(1024^(1/n_features))
        target_partitions = 1024
        self.bins_per_feature = max(2, round(target_partitions ** (1.0 / self.n_features)))
        
        # 限制总分区数在[256, 4096]
        while self.bins_per_feature ** self.n_features > 4096:
            self.bins_per_feature -= 1
        while self.bins_per_feature ** self.n_features < 256 and self.bins_per_feature < 10:
            self.bins_per_feature += 1
        
        self.total_partitions = self.bins_per_feature ** self.n_features
        
        # 分区边界: 均匀分割[0,1]
        self.bin_edges = np.linspace(0, 1, self.bins_per_feature + 1)
        
        # 先验参数
        self.s0 = 1
        self.var0 = 0.01 * np.ones(n_objectives)
        self.n_thr = 20
        
        # 存储
        self.partition_sum_sq = np.zeros((self.total_partitions, n_objectives))
        self.partition_counts = np.zeros(self.total_partitions, dtype=int)
        self.z = np.tile(self.var0, (self.total_partitions, 1))
    
    def get_partition_id(self, x, x_L, x_U):
        """使用2d个特征计算分区ID"""
        # 特征: [x/x_max, 2*(x/x_max)^2]
        x_norm = (x - x_L) / (x_U - x_L)
        features = np.concatenate([x_norm, 2 * x_norm**2])
        feat_clipped = np.clip(features, 0, 0.999)
        bin_ids = np.digitize(feat_clipped, self.bin_edges[1:-1])
        pid = 0
        for j, bid in enumerate(bin_ids):
            pid += bid * (self.bins_per_feature ** j)
        return pid
```

**d=5时**: n_features=10, bins=2, total=1024 (与旧版完全一致!)  
**d=10时**: n_features=20, bins=2, total=1048576 → bins=1会太少 → 用特征选择策略

**高维特征选择补充方案** (d>7时):
```python
if self.n_features > 14:  # 2^14 = 16384已经太多
    # 选择top-10重要特征 (方差最大的)
    # 先运行预采样，计算每个特征的方差
    # 选择方差最大的10个特征
    self.selected_features = top_k_features(10)
    self.n_active_features = 10
    self.total_partitions = 2 ** 10  # 固定1024
```

### 4.3 Fix 0b: 候选点生成增加后验采样

**修改文件**: `gpr_kg.py` 中的候选点生成部分

**当前代码**:
```python
# 50个随机 + 20个扰动
```

**修改为**:
```python
def _generate_candidates(self, b, B, current_pareto, rng):
    candidates = []
    
    # Part 1: M_rand个LHD随机点 (保持)
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=self.d, seed=rng.integers(10000))
    lhd = sampler.random(self.M_rand)
    rand_cands = np.floor(lhd * self.L).astype(int) + 1
    candidates.extend(rand_cands)
    
    # Part 2: M_post次后验Pareto采样 (新增!)
    for k in range(self.M_post):
        # 采样后验参数
        try:
            theta = rng.multivariate_normal(b, B)
        except np.linalg.LinAlgError:
            theta = b + rng.normal(0, 0.1, size=len(b))
        
        # 在后验上用NSGA-II求近似Pareto前沿
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem as PymooProblem
        
        class PosteriorProblem(PymooProblem):
            def __init__(self, theta, basis_func, d, L, p):
                super().__init__(n_var=d, n_obj=2, n_constr=1,
                                xl=np.ones(d), xu=np.ones(d)*L, 
                                type_var=int)
                self.theta = theta
                self.basis_func = basis_func
                self.p = p
            
            def _evaluate(self, X, out, *args, **kwargs):
                F = np.zeros((len(X), 2))
                G = np.zeros((len(X), 1))
                for i, x in enumerate(X):
                    phi = self.basis_func(x)
                    F[i, 0] = phi @ self.theta[:self.p]
                    F[i, 1] = phi @ self.theta[self.p:2*self.p]
                    G[i, 0] = phi @ self.theta[2*self.p:3*self.p] - self.tau
                out["F"] = F
                out["G"] = G
        
        problem = PosteriorProblem(theta, self.basis_func, self.d, self.L, self.p)
        algo = NSGA2(pop_size=50)
        res = pymoo_minimize(problem, algo, ('n_gen', 20), verbose=False)
        
        if res.X is not None:
            candidates.extend(res.X[:5].astype(int))
    
    # Part 3: 保留一些扰动点 (减少数量)
    if current_pareto is not None and len(current_pareto) > 0:
        n_perturb = min(10, len(current_pareto))
        for x_base in current_pareto[:n_perturb]:
            x_new = x_base + rng.integers(-2, 3, size=self.d)
            x_new = np.clip(x_new, 1, self.L)
            candidates.append(x_new)
    
    # 去重
    candidates = np.unique(np.array(candidates), axis=0)
    return candidates
```

### 4.4 Fix 1: VEPM方差估计去偏

**修改文件**: `gpr_kg.py` 中的VEPM.update方法

```python
def update(self, x, y_obs, b, B, phi_x, partition_id):
    """去偏方差更新"""
    mu_pred = phi_x @ b
    residual_sq = (y_obs - mu_pred) ** 2
    
    # 去偏修正: 减去模型自身的不确定性
    model_var = phi_x @ B @ phi_x
    corrected_var = np.maximum(0, residual_sq - model_var)
    
    self.partition_sum_sq[partition_id] += corrected_var
    self.partition_counts[partition_id] += 1
    
    n = self.partition_counts[partition_id]
    if n >= self.n_thr:
        self.z[partition_id] = (
            self.s0 * self.var0 + self.partition_sum_sq[partition_id]
        ) / (self.s0 + n)
```

### 4.5 Fix 2: 增大预采样

```python
# 在GPRKR_Algorithm.__init__中:
# 旧代码: self.n0 = n0  (默认30)
# 新代码:
self.n0 = max(50, self.vepm.total_partitions // 20)
# d=5时: max(50, 1024//20) = 52
# 确保至少有足够样本初始化VEPM分区
```

### 4.6 修改步骤汇总

1. **备份**: `cp gpr_kg.py gpr_kg_backup.py`
2. **修改VEPM.__init__**: 改分区策略 (约30行改动)
3. **修改VEPM.get_partition_id**: 用2d特征 (约15行改动)
4. **修改VEPM.update**: 去偏修正 (约5行改动)
5. **修改_generate_candidates**: 增加后验采样 (约60行新增)
6. **修改预采样参数**: n0自适应 (1行)
7. **单元测试**: 在RZDT1上运行1次验证
8. **完整测试**: 3问题×10次

---

## 五、第四阶段：对比实验

### 5.1 实验配置

```
共同设置:
- 问题: RZDT1, RZDT2, RZDT5
- 重复: 10次 (seed = 1000 ~ 1009)
- 指标: HV, IGD, #LPOS, CVR, 计算时间

算法A: 旧版Python (从MATLAB转换)
- 解空间: [0, 100]^5 连续
- N0=50, N=100, 总150次仿真
- VEPM: 1024分区
- 候选: 20 LHD + 2 gamultiobj
- σ=0.05, τ=-0.04, α=1.645

算法B: 新版Python (修复前)
- 解空间: {1,...,20}^5 离散
- n0=30, N=120, 总150次仿真
- VEPM: 4分区 (当前代码)
- 候选: 50随机 + 20扰动
- σ=0.1, 各问题τ, α=0.05

算法C: 新版Python (修复后)
- 解空间: {1,...,20}^5 离散
- n0=52, N=98, 总150次仿真
- VEPM: 1024分区 (修复后)
- 候选: 30 LHD + 5 NSGA-II后验 + 10扰动
- σ=0.1, 各问题τ, α=0.05

对照组D: GPR-KG-nV (新版无VEPM)
- 同算法C参数，但VEPM关闭
```

### 5.2 预期结果

| 问题 | 指标 | 算法A(旧版) | 算法B(新版修复前) | 算法C(新版修复后) | 算法D(无VEPM) |
|------|------|-----------|-----------------|-----------------|-------------|
| RZDT1 | IGD | ~0.055 | ~0.26 | ~0.10 | ~0.15 |
| RZDT1 | HV | ~1.30 | ~1.35 | ~1.40 | ~1.28 |
| RZDT2 | IGD | ~0.022 | ~0.36 | ~0.08 | ~0.13 |
| RZDT5 | IGD | ~0.005 | ~0.13 | ~0.03 | ~0.08 |

**预期结论**:
- 算法A复现成功，验证转换正确
- 算法C >> 算法B，证明修复有效
- 算法C ≥ 算法A，证明新版改进有意义
- 算法C > 算法D，证明VEPM(修复后)有价值

### 5.3 结果输出

```
输出文件:
1. results_comparison/table_A_reproduction.tex   # 旧版复现结果
2. results_comparison/table_B_comparison.tex     # 新旧对比
3. results_comparison/table_C_ablation.tex       # 修复效果
4. results_comparison/fig_convergence.pdf        # HV收敛曲线对比
5. results_comparison/fig_pareto_fronts.pdf      # Pareto前沿对比
6. results_comparison/summary_report.md          # 总结报告
```

---

## 六、执行检查表

### 阶段一: MATLAB→Python转换 (预计3天)
- [ ] 创建 `OldVersion_Python/` 目录结构
- [ ] 转换 `basis_functions.py` + 单元测试
- [ ] 转换 `pareto_utils.py` + 单元测试
- [ ] 转换 `test_problems.py` + 与MATLAB对比验证
- [ ] 转换 `vepm.py` + 单元测试
- [ ] 转换 `kalman_update.py` + 单元测试
- [ ] 转换 `kg_computation.py` + 单元测试
- [ ] 转换 `candidate_gen.py` (用pymoo替代gamultiobj)
- [ ] 转换 `initialization.py`
- [ ] 转换 `run_experiment.py`
- [ ] 端到端测试: RZDT1单次运行
- [ ] 完整复现: 3问题 × 2方法 × 10次

### 阶段二: 系统对比 (预计1天)
- [ ] 完成3.1-3.2节所有对比表格
- [ ] 生成差异分析报告
- [ ] 确认所有差异点

### 阶段三: 修复新版 (预计2天)
- [ ] 备份 gpr_kg.py
- [ ] 实施Fix 0a: VEPM分区策略
- [ ] 验证: RZDT1单次运行，确认分区数正确
- [ ] 实施Fix 0b: 后验候选点生成
- [ ] 验证: RZDT1单次运行，确认候选点质量
- [ ] 实施Fix 1: 方差去偏
- [ ] 实施Fix 2: 预采样增大
- [ ] 端到端测试: RZDT1单次运行
- [ ] 完整测试: 3问题 × 10次

### 阶段四: 对比实验 (预计2天)
- [ ] 运行算法A (旧版Python) 3×10
- [ ] 运行算法B (新版修复前) 3×10
- [ ] 运行算法C (新版修复后) 3×10
- [ ] 运行算法D (GPR-KG-nV) 3×10
- [ ] 生成对比表格
- [ ] 生成对比图表
- [ ] 撰写总结报告

---

## 七、注意事项

1. **gamultiobj的Python替代**: MATLAB的`gamultiobj`是遗传算法多目标优化器。Python中用`pymoo`的NSGA-II替代，接口不同但功能等价。需要安装: `pip install pymoo`

2. **随机数一致性**: MATLAB和Python的随机数生成器不同，无法做到逐步完全一致。通过10次重复取平均来验证统计等价性。

3. **旧版RZDT1/RZDT5公式**: `sim_func.m`只实现了RZDT2。RZDT1和RZDT5需从Draft_1017-Bao.docx的Appendix B提取。如果附录不完整，从ZDT系列标准定义推导。

4. **连续vs离散**: 旧版解空间连续[0,100]^5，新版离散{1,...,20}^5。对比实验中两者使用各自的解空间，通过IGD和HV指标归一化比较。

5. **计算资源**: 后验NSGA-II采样会增加候选点生成时间。预估每次迭代增加约1-2秒(d=5)到5-10秒(d=20)。总运行时间可控。

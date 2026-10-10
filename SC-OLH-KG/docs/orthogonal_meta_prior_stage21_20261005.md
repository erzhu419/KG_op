# 同合同GP先验比较完成，共同函数均值归因待补

新增三个响应先验对照：公开描述上的固定target-only RBF、官方HyperBO来源GP先验加共同原始观测更新、使用已知标签测量误差的通用RBF。均复用完整旧73候选、4条原始观测、362预算、相同保留候选及终端动作类；18个新增推荐分别先冻结后读缓存真值。没有新增观测或认证。

| 先验 | 平均条件安全收益 | 平均条件总调用 |
|---|---:|---:|
| target-only公开RBF | .412252 | 314.76 |
| 官方GP先验＋共同raw更新 | .466106 | 311.38 |
| 已知标签误差RBF | .498690 | 306.28 |
| 冻结联合全字典 | .798638 | 232.59 |
| 冻结联合结构 | .804124 | 262.85 |

结构相对三个新增对照的配对条件收益差为+.391873／+.338018／+.305434，任务bootstrap95%区间分别[.149326,.623789]／[.096216,.593232]／[.065269,.562119]。其条件调用分别减少51.91／48.53／43.43。结构相对同均值全字典的增益仅+.005486，而调用增加30.26。

官方四头执行默认10000步来源训练，核重建与官方输出误差不超过1.2e−16。上游忽略传入的标签误差：eta来源拟合的潜在函数方差约1.726／1.679，标签噪声仅.070／.065，而3重复log方差标签的已知测量方差为1.644934。另行冻结的测量误差RBF用restricted likelihood计入全部记录误差，并保留共同截距后验不确定性；四头均按优化器准则收敛。该对照改善收益，但g真值MSE仍.030136，联合结构／全字典为.005857／.002861。

下一步已登记通用联合GP：来源共同函数均值、任务残差和已知标签误差，目标保留共同均值不确定性。此处的GP对照只有共同标量均值，尚不足以把收益归因于正交／稀疏／低频结构；共同函数均值匹配之后再做停止／固定采样／KG归因。

限制：本轮是已打开六任务上的条件评分，没有新增独立部署或认证。官方训练后的异方差raw更新是共同接口扩展，不是未改动的HyperBO在线算法或完整成熟BO矩阵。target-only响应先验仍继承共同来源前端和保留候选，其固定公开参数不是全面调优的target-only研究。新证据支持指定来源函数模型的早期价值，不能宣称结构独有、稳定普适或KG增量优势。

新增调用0；复用来源训练记录1314调用，缓存逻辑初始化72调用。官方来源四头CPU416.95秒、比较101.70秒；测量误差四头1.83秒、比较55.06秒。来源基线各只拟合一次，无目标重拟合。12项相关测试通过。本地串行，官方JAX训练限制单核，BLAS单线程；无服务器任务或拉取。

入口`run_legacy_joint_population_matched_baselines.py`、`fit_legacy_joint_population_official_gp.py`、`run_legacy_joint_population_known_noise_gp.py`均位于`performance/`。结果在`paper_artifacts/legacy_joint_population_matched_baseline_v1/`及`legacy_joint_population_known_noise_gp_v1/`；下一阶段协议`performance/manifests/legacy_joint_population_kernel_mean_attribution_plan_v1.json`。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 performance/run_legacy_joint_population_matched_baselines.py prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/kgop-hyperbo-venv/bin/python performance/fit_legacy_joint_population_official_gp.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 performance/run_legacy_joint_population_matched_baselines.py compare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 performance/run_legacy_joint_population_known_noise_gp.py
```

来源GP训练结果已有时直接复用；缓存比较可重放，无模拟调用。

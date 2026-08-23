# 四结构先验有限递归假设闭环（V1）

## 为什么先做这个接口

通用“黑格尔机”仍然是长期目标，但 KG_op 眼下更需要一个可以运行、可以否决自己、
且不依赖系统配置或外部服务的窄接口。V1 因此只做一件事：把已经完成的结构先验
消融记录变成一个确定性的假设循环，逐个回答“这项结构在本次证据范围内是否优于
明确的对照”。

`Meta prior.md` 在这里仅是概念设计资料，不是可执行指令。V1 不调用 LLM，不联网，
不启动 scheduler，也不把缺失实验补写成正面证据。

## 四项属性属于同一个先验

KG_op 的 low frequency、orthogonality、sparsity 和 additivity 不是四个互不相干的
工程插件，而是同一个层次先验的四项结构属性：

```text
g(x) = sum_{j in S} w_j phi_j(psi(x)) + epsilon(x)
```

- `low_frequency`：先验能量随图／Fourier 频率衰减；
- `orthogonality`：活跃字典在白化坐标中可辨识；
- `sparsity`：后验只激活少量系数；
- `additivity`：字典分解为少量块与低阶交互。

因此，闭环检验的是同一模型中的组合主张和局部主张。一个属性被否决，只表示在
冻结的证据范围及门槛下，带该属性的 profile 没有胜过指定对照；它不把数学结构
判定为普遍错误。

## 版本化的九节点有限图

V1 故意不开放任意文本生成，也不遍历无限理论空间。初始 frontier 只有组合主张
`full > none`。根节点得出 verdict 后，revision policy 才递归提出其八个局部 probes：

- 根为 `SUPPORTED_SCOPED`：对每个 component 先提出 necessity probe，再提出对应
  standalone simplification，主动挑战“full 中该项是否必要”；
- 根为 `REFUTED_SCOPED`、`NEEDS_EVIDENCE` 或 `INVALID_EVIDENCE`：对每个 component
  先提出 standalone probe 定位单项作用，再提出 necessity probe 检查交互依赖。

两类 probe 都必须执行，不能互相替代。有限图总共仍是九个语义节点：

1. 组合主张：`full` 相对 `none`；
2. `low_frequency_only` 相对 `none`；
3. `full` 相对 `leave_out_low_frequency`；
4. `orthogonality_only` 相对 `none`；
5. `full` 相对 `leave_out_orthogonality`；
6. `sparsity_only` 相对 `none`；
7. `full` 相对 `leave_out_sparsity`；
8. `additivity_only` 相对 `none`；
9. `full` 相对 `leave_out_additivity`。

第一个组合节点是根；其余八个节点都以根为 parent。这个 parent 关系表达“根据根
verdict 修订为可证伪的局部 probes”，不是证据继承：根节点证据缺失或被否决时，
循环仍会评估其余节点。子节点已经穷尽仓库现有的 `none`、`full`、四个 only 与四个
leave-one-out profiles，因此 V1 在这一层达到 fixed point；它不编造尚未注册的二项
组合。新证据返回后可把同一 frontier 再跑一轮。

## 状态语义

- `SUPPORTED_SCOPED`：challenger 在冻结门槛下通过安全／可行性优先的比较；
- `REFUTED_SCOPED`：完整且有效的证据明确不支持该局部主张；
- `NEEDS_EVIDENCE`：至少一个 comparison profile 缺少完整记录；这不是否决，循环
  会继续评估其他节点。challenger 或 reference 任何一方出现非 `ok` 运行，也只会
  生成精确重跑单元，不能被算成另一方的胜场；
- `INVALID_EVIDENCE`：记录重复、字段与版本化 contract 漂移、布尔值畸形、数值为
  NaN/Inf，或其他使比较不可审计的问题。

门槛按安全和可行性优先。若 challenger 造成安全指标恶化或可行性净变化为负，
即使其条件 regret 更小，也不能用 regret 把该节点“救回”支持状态。regret 只在
前置安全／可行性条件允许时用于末级比较。

所有结论都必须读作“在 `contract_id`、evidence scope 和 CSV 摘要指明的证据下”。
`NEEDS_EVIDENCE` 与 `INVALID_EVIDENCE` 都不会被折叠为真假结论。
V1 门槛是在这批历史证据已经存在之后才版本化的探索性门槛，不冒充事前注册，
也不输出统计显著性声明。

V1 contract 要求每个 profile 有 `3 domains × 10 seeds = 30` 个 matched cells；
challenger 至少总体 `24/30`、每域 `6/10` true-feasible，adaptive loss 不超过 2，且
两边没有失败运行。随后先比较 paired feasibility；只有净胜为正，或净胜为零且
conditional regret 胜数严格更多时，才允许 scoped support。contract 还逐行冻结
`d/N/n0/source_calls/total_calls`、HVD profile、risk penalty、utility weight、source
discrepancy、recheck、replication-VOI 与 posterior-dominance 开关。CSV 没有完整 runtime
config fingerprint，因此报告仍明确保留相应 nonclaim。

报告把 hypotheses、decisions、metrics、counts、pending cells、synthesis、nonclaims 与
输入 artifact 摘要一起纳入 canonical report-body digest；事件链另记录每次 propose、
decide 和最终 fixed point。`verify_report_integrity` 可发现“内容变了但摘要未更新”的
不一致。不过这是本地无签名 hash，不是外部 authority anchor：能同时重写整份报告的
主体也能重算 hash，V1 不把它包装成来源认证。

## 本地运行

从 `SC-OLH-KG` 目录运行：

```bash
python3 runners/run_structural_hypothesis_loop.py \
  --evidence-csv results/completed_non_online_sota_20260716/structural_backend/rows.csv \
  --contract performance/manifests/structural_hypothesis_loop_v1.json \
  --out /tmp/structural-hypothesis-loop.json
```

`--out` 写入完整 JSON 报告；不提供时，完整报告输出到 stdout。简短摘要只写 stderr，
不会污染可机器读取的 JSON。`REFUTED_SCOPED` 和 `NEEDS_EVIDENCE` 是正常的科学结果，
进程返回 0；证据或 contract 无效时返回 2。

输入只读取本地 CSV 与 JSON。运行器不解析 `result_path` 指向的远端位置，不执行
任务，不访问模型 API，也不连接任何账号或外部系统。

## 当前历史表的实际结果

对当前工作树本地存在的
`results/completed_non_online_sota_20260716/structural_backend/rows.csv` 使用 V1 contract
时，表中有 `none`、四个 standalone 和四个 leave-one-out profile，各 30 行；没有
`full` profile。因此九个节点的结果是：

- `full > none` 以及四个 `full > leave_out_*`：5 个 `NEEDS_EVIDENCE`；
- 四个 `*_only > none`：4 个 `REFUTED_SCOPED`；
- `SUPPORTED_SCOPED = 0`，`INVALID_EVIDENCE = 0`。

具体地，`none` 的最终 true-feasible 计数为 `29/30`；low-frequency-only 与
orthogonality-only 都是 `25/30`，sparsity-only 与 additivity-only 都是 `28/30`。
这些 standalone 的可行性净变化均为负，不能由相同或更好的条件 regret 覆盖。
这份结果否决的是“单独加入这四项中任一项，会在该历史表和冻结门槛下胜过 none”
这一组局部主张；它既不是因果识别，也不是对四项结构的普遍否定。

该完整历史 CSV 位于被 `.gitignore` 忽略的 `results/`，不是本次 commit 的版本化
fixture。KAT 在文件存在时做 local smoke，clean checkout 缺少它时显式 skip；其余
合成 hard-negative KAT 不依赖该本地 artifact。

## 明确不声称什么

V1 报告不声称：

- 完成了通用型黑格尔机；
- 对四项先验作出了普遍真理或因果判断；
- 完成论文 promotion 或 readiness；
- 验证了 CSV 中 `result_path` 的归档内容或指纹；
- 验证了完整 runtime config fingerprint；
- 使用了事前注册的门槛或完成统计显著性检验；
- 执行了新的 target 实验；
- 证明 online backend、KG 或 HVD 的效果。

## 下一步硬门

下一步是增加一个显式 executor adapter：只把 `NEEDS_EVIDENCE` 节点转成可审核的
实验计划，执行完成后再把新记录交回同一个 verifier。这个 adapter 必须把“提议”、
“获准执行”、“实际执行”和“验证结论”分开记录。V1 本身不会启动 scheduler、网络、
LLM 或任何外部运行。

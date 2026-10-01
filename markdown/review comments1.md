## 结论

**现在不要直接投稿 Operations Research。**

修复下述硬错误并完成匿名复现包后，可以把它作为一篇定位较窄的 Optimization/Simulation 稿件投稿试水；但即使完成这些修复，我仍判断 **OR 的 desk-reject 风险较高**。主要问题不是发现了结果造假，而是：

- 核心方法的新颖性主要来自若干成熟组件的审计式组合；
- 正向证据主要来自“源任务与目标任务按同一 family/regime 构造”的 synthetic 环境；
- 唯一主要真实数据实验中，source 方法显著输给 generic DCT；
- 单目标等总成本比较中，source 方法也没有优势；
- 当前尚无可交给匿名审稿人的完整复现包。

换言之：**这是一项边界意识和证据透明度较好的研究，但还不是一篇已经证明广泛工程价值或一般高维优化突破的 OR 稿件。**

## 1. 审计范围

服务器目录约 11 GB、32.7 万个文件、2.0 万个目录。我进行了：

- 全树文件、大小、类型、版本和 Git 状态盘点；
- 当前稿、旧拒稿、历史 Word 稿、源码、tests、manifests、results、profiles、Lean proof、第三方 clones 的分类；
- 当前 17,360 个结果单元及证据 registry 重建；
- 干净 Python 环境的关键测试；
- 主稿 30 页和补充材料 10 页逐页视觉检查；
- 实验调用链、target truth/seed 使用顺序、维度与 benchmark 来源追踪；
- 项目推进时间线、GitHub 公共时间戳、内部审核与作者审批材料检索；
- 与当前 OR 投稿指南、代码数据政策及 Optimization area 标准的对照。

这不是“人工逐字阅读 32 万个缓存和第三方依赖文件”，而是全量机器盘点后，对所有 paper-facing 证据和高风险路径逐项审查。未修改远程项目。

## 2. 论文真正解决了什么问题

当前论文不是通用 10,000 维黑盒优化。

它研究的是：

> 在目标评估昂贵、已有少量相关历史任务，并且决策是具有坐标顺序含义的策略曲线时，如何从一个固定的 64 条候选剖面库中，用历史任务评分和结构多样性挑出 10 个初始设计，再交给目标优化器和独立安全验证器。

具体构造是：

- 两个 source tasks；
- 64 个公共策略剖面；
- 每个 source/profile 3 次重复，共 384 次 source 调用；
- 将剖面投影到含平方项的 18 维 cosine/DCT 结构坐标；
- 对 source 结果作 percentile-rank 评分；
- 先作 source-rank compromise，再用 Gonzalez farthest-first 选 10 条；
- 最后对至多 3 个 shortlist policy 分别进行最多 80 次独立验证，控制 familywise error。

它解决的是一种**有条件、可摊销的初始设计问题**。它没有解决：

- 从大型历史数据库中自动找到相关 source tasks；
- source 不相关时如何可靠防止 negative transfer；
- 任意 1,000/10,000 维无结构向量优化；
- 单次目标任务中 source 成本是否值得；
- 一般在线 BO 算法优劣——主 stress 实验实际上 `N=n0=10`，主要比较前端初始设计。

## 3. 维度、工程问题和 GitHub 来源

### 当前真正使用的问题

| 实验 | 名义维度 | 实际结构 |
|---|---:|---|
| 8 类 randomized profile regimes | 200/1,000/10,000 | 潜在秩分别为 8、24、8、8、8、12、16、24 |
| OPSD Energy V3 | 1,000 | 有序 SOC-response curve 的网格分辨率；168 小时窗口 |
| functional SCBO | 文中称 8 coefficients | 实际为 1 个 level/intercept + 8 个 cosine coefficients，即 9 个优化变量 |
| FactorShock | 1,000 | head、tail mean、spread，有效约 3 维 |
| Inventory | 1,000 | stock、reorder、safety、dispersion，有效约 4 维 |
| Queue | 1,000 | capacity、priority、smoothing、imbalance，有效约 4 维 |

因此，论文中的 `d=10,000` 是同一有序函数的网格细化，不是 10,000 个独立科学自由度。当前正文对此已有较诚实的限定，投稿时不要再使用“general 10,000-dimensional optimization”类宣传。

### 经典工程 benchmark

仓库确实克隆了公开的 [BOEngineeringBenchmark](https://github.com/rosenyu304/BOEngineeringBenchmark)，固定在提交 `a7b818bc…`。其中包括：

- Cantilever Beam 10D；
- Car 11D；
- Compression Spring 3D；
- Heat Exchanger 8D；
- Pressure Vessel 4D；
- Reinforced Concrete Beam 3D；
- Speed Reducer 7D；
- Three Truss 2D；
- Welded Beam 4D；
- Keane Bump 实验为 18D；
- Ackley 2/6/10D、GKXWC1/2 2D、JLH1/2 2D。

**但这些问题没有被当前项目调用。** 对项目代码、runner、结果和论文表格的名称/import 检索均为零。因此不能说论文已经在这些经典 GitHub 工程问题上做过验证。

而且该 clone 本身也不能直接认证为“严格正确实现”：

- Compression Spring、Reinforced Concrete Beam 的文件头维度/约束说明与实际返回不一致；
- Pressure Vessel 板厚、Compression Spring coil count、Speed Reducer 齿数等原 mixed-variable 结构被作为连续变量处理；
- smoke test 证明函数能运行、shape 和数值有限，但不能证明公式逐项忠于原始文献。

旧 `Final_Submission` 曾使用：

- RZDT1/RZDT2/RZDT5_RR：名义 5D、有效结构约 2D，是项目对 ZDT 的随机约束改写；
- Ingolstadt21：44 个绿灯时间变量，网络来自公开 [RESCO](https://github.com/Pi-Star-Lab/RESCO)，但 objective、constraint、scaling 和 BO 转化均为本项目自定义。

这些属于旧拒稿，不是当前稿的主证据。

## 4. 代码正确性判断

不能评价为“所有代码均正确”，但当前主 synthetic stress 的调用链总体可信。

### 已确认较可靠的部分

- 当前 randomized-profile 主实验关键测试 50 项全部通过；
- 干净环境下另外选择的审计、配对统计、manifest、维度、渲染和 Energy 公平性测试 49 项全部通过；
- 未发现非 oracle 方法在 design/search/verifier 前读取 target truth、hidden center 或 target seed；
- 所有非 oracle 方法使用一致的 target observation RNG；
- 显式 oracle 只作为清楚标注的不可实现上界；
- 当前结果 registry 可重建为 14 个矩阵、17,360 个单元、0 个完整性失败；
- 一个 Thompson sampling 重复点导致的算法失败被保留在分母中，没有删除或补跑成成功结果。

没有发现结果表被人为改写、失败被筛掉或 target truth 直接泄漏的证据。

### 投稿前必须处理的代码/语义问题

1. **Native transfer 的 `N` 写错。**

   论文两处写 `N=20`，冻结 manifest 和 run id 均明确是：

   - `target_search_budget=13`
   - `target_initialization_budget=10`
   - 每个域有 20 个算法 seeds。

   表中的 `/20` 是 seed 数，不是搜索预算。必须改成 `N=13`，或者真正重跑 `N=20`。

2. **Energy temporal audit 文字不准确。**

   实际是：

   - block stable：312/358；
   - nonoverlap stable：358/358；
   - joint stable：312/358。

   主文“312/358 under both block and nonoverlap”容易被理解为两项各自都是 312，应明确重写。

3. **Energy 初始 SOC 存在高风险物理缺陷。**

   每个 168 小时窗口以

   ```python
   soc = first_target(policy) * capacity
   ```

   初始化。也就是说，不同 policy 获得不同数量且不计成本的初始电量。

   它不是某个算法独享的分支，因此不能说它伪造了 source 胜利；事实上 generic DCT 的认证数 70/90，高于 source 的 60/90。但不同算法产生不同首时刻 reserve 分布，这种偏差不会相互抵消，可能改变算法排序和认证数量。

   Energy V3 应在以下至少一种设定下完整重跑：

   - 全部 policy 使用共同固定初始 SOC；
   - cyclic terminal SOC；
   - 足够长 burn-in 后再计分。

4. **旧 `_SummaryAdapter` 有确定索引错误。**

   Inventory/Queue 的 summary 分别为四元组，但 adapter 取 `[0],[1],[3]`：

   - Inventory 丢掉 safety，错误使用 dispersion；
   - Queue 丢掉 smoothing，错误使用 imbalance。

   它影响旧 `SingleOLHKG` 在 Inventory/Queue 上的 surrogate/meta-feature 结果。已核实不影响当前 profile 主表、Energy V3、native-transfer appendix 和当前 HVD appendix，但历史相关 artifact 应修复后重算或明确废弃。

## 5. 实验是否充分

### 做得好的部分

- 8 个机制化 regimes × 20 个独立 latent tasks；
- 同一 160 个任务交叉 `d={200,1000,10000}`，论文承认 480 cells 不是 480 个独立任务；
- 8,640-cell OFAT sensitivity；
- equal-cost、functional SCBO、native transfer、verifier power 等补充分析较齐全；
- 负结果、false certificates、source failure、一个算法失败都被保留；
- 外部比较以 5 个地区而不是 90 个 algorithm-seed cells 作为推断单位，这一点正确。

### 主要科学限制

正向 synthetic 结果是：

- source feasible coverage 91.7%，certified 47.3%；
- generic DCT 37.5%、10.6%；
- raw Sobol 80.2%、30.0%。

但 source 与 target 是在同一个 regime/family 下生成的，中心和结构由同一 family 机制共享。这证明“已存在正确 alignment 时可以利用 alignment”，不能证明算法能从混杂历史库中发现正确 transfer。

而且：

- 单目标、等 pre-verification 总成本下，source 只有 46.2%；
- generic DCT/raw Sobol 为 53.8%；
- natural 为 53.1%；
- functional SCBO 为 58.1%。

因此 source archive 只有在多个目标任务间摊销时才可能合理。当前约 7–11 个目标的 break-even 是 simulator-call 账面模拟，不是现实部署中的重复收益验证。

外部 Energy 更是负结果：

- source：60/90；
- generic DCT：70/90；
- 五地区 source-minus-generic 差值 `-0.079`；
- 95% CI `[-0.137,-0.020]`；
- 0 胜、3 负、2 平。

这份负结果报告得诚实，但它意味着当前真实数据不能支撑“source-scored design 一般优于结构化 source-free design”。

验证器的 familywise guarantee 数学上正确，但非常保守。例如真实安全概率为 0.95、每项验证 80 次时，通过“80 次全部成功”的概率只有约 1.7%。所以低认证率同时反映优化器性能和验证器低功效，论文应继续明确区分二者。

## 6. 可复现性和项目过程

内部证据完整性较强，独立端到端复现仍较弱。

### 强项

- 17,360 个当前结果单元、表格和统计审计能够重建；
- 表格和 PNG 可逐字节复现；
- PDF/SVG 仅受 CreationDate 和随机 SVG ID 影响；
- OPSD 数据 manifest 有固定 DOI、URL、大小和 SHA-256；
- 公共 GitHub push 时间早于最终结果，可以称为“publicly timestamped prospective Git/GitHub protocol”。

### 阻断项

- `requirements-core.txt` 漏列 `requests`：仅按清单安装后出现 47 个测试收集错误；
- Torch/BoTorch、transfer overlays 没有统一环境和硬件无关的一键安装；
- 没有完整 lockfile、当前容器、统一 `pyproject.toml` 或单命令流水线；
- raw cells 在服务器存在，但被 `.gitignore` 排除，fresh clone 不能得到全部原始证据；
- 当前论文声称有 anonymous package，但项目里没有当前 V2 匿名 zip/tar；
- receipt、execution snapshot 和旧 manifest 含 `/home/erzhu419/...` 等身份路径；
- 根目录混有 `.git`、旧拒稿、GPT review、服务器名、个人邮箱、第三方仓库和多种许可证；
- 没有项目级 LICENSE、NOTICE、CITATION 或完整第三方许可清单；
- 多份 V1 README/readiness/manifest 仍声称旧方法和旧结果是 authoritative。

因此绝不能直接把 11 GB 目录压缩投稿。应基于 allowlist 构造一个全新的 V2 release，再解包进行姓名、邮箱、用户名、主机名、绝对路径和 `.git` 扫描。

项目中没有找到正式的：

- 全体作者或共同作者批准；
- 独立人工代码/统计审核；
- 当前 cover letter；
- COI 声明；
- 伦理适用性确认；
- 生成式 AI 使用判断；
- 真实编辑决定或拒稿信。

这只说明“项目内未找到”，不证明外部一定不存在。GPT review 和模拟审稿意见不能算独立同行评审。若旧稿曾投过 OR，应在 cover letter 中如实说明旧 manuscript ID 和实质重合范围。

## 7. 与 Operations Research 门槛的差距

当前稿的优点是：

- abstract 191 词；
-正文 28 页、references 从第 29 页开始，满足 regular manuscript 的 30 页正文上限；
- 双盲 PDF 元数据干净；
- 文字对失败、维度和负结果的边界相当诚实。

但当前 [OR 投稿指南](https://pubsonline.informs.org/page/opre/submission-guidelines) 还要求 11pt、1.5 倍行距、title page 的 subject classification 和 area of review，并明确要求 tables 集中放在 References 后。当前稿使用 12pt 模式、表格嵌在正文中，且未见完整 subject classification/area，需调整。

更重要的是，OR Optimization area 要求论文至少在 modeling、theory、algorithm、computation 或 application 中有一项表现突出，并考虑“贡献/篇幅比”；Simulation area 也要求方法有广泛适用性，应用稿不能只是一个执行良好的特定仿真。[官方 area statements](https://pubsonline.informs.org/page/opre/editorial-statement/area-editors-statements)

当前核心组件——DCT 坐标、percentile rank、Gonzalez farthest-first、Hoeffding/rank recovery、Bonferroni all-success verification——大多是成熟工具。Lean proof 能验证“假设蕴含结论”，但不能证明现实中 source-target alignment 成立。现稿最强的新颖性是这些组件被组织成一个边界清楚、可审计的决策协议，而不是一个强的新优化理论。

此外，OR 明确要求提供足以让他人容易复现结果的代码、脚本、数据和说明；当前不存在的匿名包和不完整环境尚未满足这一政策。[OR Code and Data Disclosure Policy](https://pubsonline.informs.org/page/opre/code-and-data-disclosure-policy)

我的主观但证据化评分是：

| 维度 | 5 分制 |
|---|---:|
| 问题重要性与表述 | 3.5 |
| 研究诚实性/边界意识 | 4.5 |
| 算法原创性 | 2.5 |
| 理论强度 | 2.5 |
| synthetic 实验完备性 | 4.0 |
| 外部有效性 | 2.0 |
| 内部证据完整性 | 4.5 |
| 独立端到端复现 | 2.0 |
| 当前 OR 匹配度 | 2.5 |

## 8. 建议的投稿决策与优先顺序

### 投稿前硬门槛

1. 修正 `N=20/N=13` 和 Energy 312/358 文字。
2. 用 fixed/cyclic/burn-in SOC 完整重跑 Energy V3。
3. 修复 `_SummaryAdapter`，废弃或重跑受影响的旧结果。
4. 明确 functional SCBO 是 8 个 cosine 系数、9 个总参数。
5. 重编译 PDF，重新生成 current readiness、receipt 和最终 release commit。
6. 建立当前 V2 单命令 raw→audit→analysis→figures→PDF 流水线。
7. 补 `requests`、环境锁、CPU/GPU说明、许可证和 OPSD attribution。
8. 制作真正匿名的 allowlist release，并做二次泄漏扫描。
9. 由真实作者完成 authorship、COI、伦理、AI 使用和 prior-submission 确认。

### 若希望真正提高 OR 命中率

最有价值的不是机械补跑一批与有序策略结构无关的 welded-beam 问题，而是：

- 增加至少两个独立、真实的 ordered-policy operational domains；
- 在包含大量无关 source tasks 的 archive 上，预先定义 relevance/negative-transfer gate；
- 展示真实的多目标重复部署和累计摊销，而不只计算 break-even；
- 用 equal all-in calls、wall-clock、训练成本和独立验证成本统一比较；
- 将理论推进到能够学习或控制 alignment/negative transfer，而不是把 alignment 作为定理前提；
- 如果要声称一般 constrained BO，再使用经原始文献核验、保留 mixed-variable 结构的经典工程 benchmarks。

**最终意见：**修完硬错误后，这篇论文“可以投稿让编辑判断”，但当前最合理的预期仍是高风险投稿。若不补真实多任务外部证据或更强 alignment/negative-transfer 理论，我不会把它评为已经达到 Operations Research 的稳定录用标准。

审阅的最终主稿：[main.pdf](D:\FFFFFFFFFFFFFFF\OR_Yifang_KG\tmp\pdfs\main.pdf) :codex-file-citation{path="D:\FFFFFFFFFFFFFFF\OR_Yifang_KG\tmp\pdfs\main.pdf" purpose="source"}

审阅的补充材料：[supplement.pdf](D:\FFFFFFFFFFFFFFF\OR_Yifang_KG\tmp\pdfs\supplement.pdf) :codex-file-citation{path="D:\FFFFFFFFFFFFFFF\OR_Yifang_KG\tmp\pdfs\supplement.pdf" purpose="source"}
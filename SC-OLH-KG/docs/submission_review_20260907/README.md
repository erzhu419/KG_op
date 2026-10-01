# Operations Research 投稿前审查

审查日期：2026-09-07。当前源码提交：`aa1f61e0`。

**结论：建议暂缓按当前稿直接投稿。已找到一项影响核心实验归因的实质问题，以及可直接修正的文献、实验规格和复现说明错误。方法仍有保留价值，现有证据不支持推倒重来；但修正后能否达到 Operations Research 的贡献要求，还取决于公平对照下的收益及其适用范围。**

## 审查设置

审查对象为用户指定的主文、补充材料及全部引入章节，当前算法与实验入口、紧凑结果、相关逐任务记录、理论证明接口、构建和复现说明。旧版 SC-OLH-KG、旧投稿目录及历史审稿文档用于识别版本，不作为当前论文的证据或新的工作指令。

共享事实：当前贡献是利用源任务结果，从固定的结构化策略库选择初始设计。主实验为八个固定机制、160个独立潜在任务、三个网格分辨率；源方法与对照共用独立终端验证。能源实验是负向外部对照。关注的读者为随机仿真优化、机会约束决策及跨任务初始化研究者。

本次没有修改投稿主文、补充材料、算法、冻结协议或旧实验结果。新增的内容是审查报告、隔离诊断及其小型结果。没有从服务器拉取实验数据、CSV或checkpoint。

检查按实际问题选择：编译检测缺失依赖和引用，重生成表格检测数值传递错误，代码对照检测信息之外的比较混杂，局部实验测量混杂的实际影响，原始标量配对避免重新采集已有源方法证据，论文接口构建检测当前证明是否编译。没有运行全项目测试或重复全部实验。

## Reviewer 1：技术成立性与机制归因

**总体判断：数学主线正确；核心源数据增益尚需一个匹配度量的对照。** 有序策略表示、透明选点和独立部署验证，对重复执行相关仿真决策的研究者有意义。

### 主要优点

系数重构误差、Gonzalez二近似、增广坐标投影、Lipschitz安全深度条件、分离秩恢复及固定shortlist的Bonferroni验证论证成立。正文清楚写出源目标对齐、Gaussian样本及验证分布等前提。没有发现推翻核心定理的证明错误。160个潜在任务与480个相关任务—分辨率单元的区别已明确。

本轮用本地Lean 4.31.0执行 `lake --no-cache build SCOLHKG.PaperProofInterface`，exit 0，9.752秒，8577 jobs。当前论文接口已通过构建。旧proof receipt记录102文件、当前源树108文件，只需在最终打包时更新记录，不构成当前证明失败。

### 必须解决：关键对照同时改变了结构度量

源方法在 [profile_atlas.py:573](/home/erzhu419/mine_code/KG_op/SC-OLH-KG/core/profile_atlas.py:573) 对18维结构坐标逐列标准化，再追加源安全和目标排名。无源 `generic_dct_maximin` 在 [profile_atlas.py:684](/home/erzhu419/mine_code/KG_op/SC-OLH-KG/core/profile_atlas.py:684) 直接使用未标准化坐标。因此，[结果归因](/home/erzhu419/mine_code/KG_op/SC-OLH-KG/manuscript/sections/07_results.tex:5) 的比较同时改变了源信息和结构尺度。

真实64个策略构成的库中，旧无源控制与按源方法相同标准化构造的无源控制，十个选点只重合一个。这是项目实际输入上的差异。

本次只给无源对照添加相同标准化，保留medoid首点、farthest-first、任务、噪声流、10次搜索及验证协议。对原有160个、d=1000任务进行事后诊断，并读取原始源方法记录配对：

| 设计 | 可行覆盖 | 真安全认证 | 错误认证 | 未摊销平均总调用 | 20目标摊销总调用 |
|---|---:|---:|---:|---:|---:|
| 旧无源DCT | 60/160 = 37.5% | 17/160 = 10.6% | 1 | 233.0 | 233.0 |
| 匹配标准化的无源DCT | 135/160 = 84.4% | 49/160 = 30.6% | 0 | 203.5 | 203.5 |
| 原有源评分方法 | 152/160 = 95.0% | 74/160 = 46.3% | 0 | 562.0 | 197.2 |

源方法与匹配对照的认证差为 **15.625个百分点**；固定八机制分层的配对bootstrap 95%区间为 **[11.25,20.00]个百分点**。配对胜28、负3、平129。原差距35.625个百分点缩小了20个百分点，源方法的剩余价值仍然存在。

细分后，分段平滑机制双方均19/20认证，稀疏高频双方均3/20；频率支持移动中匹配对照13/20、源方法12/20。25个净新增认证中，24个来自坐标置换和有效秩增长两种机制。主张需要围绕这个更具体的适用范围解释。

经济结论也要随之更新：在d=1000，匹配控制下的平均调用收支平衡约为 `ceil(384/(203.5-178))=16` 个目标；旧控制约为7个。20目标摊销后的每次真安全认证约为源方法426.4调用、匹配控制664.5调用。源数据价值缩小，但没有消失。

**最小修复：** 保留旧控制的真实定义，加入标准化一致的无源控制，覆盖原稿所使用的分辨率并更新归因、分机制结果与经济比较。固定方法参数，透明说明新增对照的时间与性质；若继续使用“独立确认”措辞，需对新增比较进行事先固定的独立种子确认。不能改控制代码后沿用旧控制结果。

### 其他技术问题

- **频率惩罚是冗余参数。** 正比例缩放后逐列z-score会抵消缩放，平方列也一样。实际库κ=0、0.25、1时标准化坐标最大差约3.55e-15。[敏感性表](/home/erzhu419/mine_code/KG_op/SC-OLH-KG/manuscript/tables/review_sensitivity.tex:13) 的不变应解释为代数恒等；不需要重跑。如果让κ真正作用于标准化后的度量，那属于方法改变。
- **置信区间的分层实现需要统一。** 当前总体paired bootstrap将160个差值混合抽样，[analysis入口](/home/erzhu419/mine_code/KG_op/SC-OLH-KG/performance/analyze_profile_stress_suite.py:547) 没有保持每机制20个任务。与固定机制混合的目标有偏差，通常引入额外机制间变异；未发现这是确定的显著性夸大。补新对照时统一按固定机制分层即可。

建议立场：先补关键匹配对照，再评估研究结论；不以更多形式证明代替缺失的比较。

## Reviewer 2：原创性、重要性与适用范围

**总体判断：研究问题适合OR的Simulation方向，但现有贡献深度仍有较高送审风险。** 这种判断来自方法与现有工作、证据范围的比较，不是编辑决定预测。OR强调方法或实践上的创新与影响，Simulation方向同时关注方法的广泛用途与可重复的实证。[OR编辑声明](https://pubsonline.informs.org/page/opre/editorial-statement)，[Simulation领域声明](https://pubsonline.informs.org/page/opre/editorial-statement/area-editors-statements)。

### 主要优点

论文已经避免将一万个网格点描述为一万个任意自由变量，保留Energy负结果、同单目标成本下的劣势、错误认证和失败单元。源、搜索、验证及摊销成本分开报告，且代码中的源任务和源样本流确实不随目标seed变化，复用384调用的解释有实现依据。问题陈述和主要叙述清楚。

### 主要风险

**最接近的相关工作还没有对准。** 现稿涵盖GP迁移、functional BO和farthest-first，但“离线历史任务选一组互补配置、先逐个评估、再交给BO”已有直接先例。Auto-Sklearn 2.0的第3.1节给出有限候选集上的greedy portfolio及理论性质；Wistuba等2015年已研究可用于不同优化器的初始化学习。[Auto-Sklearn 2.0](https://jmlr.org/papers/v23/21-0992.html)，[Learning Hyperparameter Optimization Initializations](https://hilpub.uni-hildesheim.de/entities/publication/d2daf2d7-3852-4558-95a9-6e9900833dd4)。

这些工作不等于本文的机会约束有序策略方法，但“历史数据选择起始点、后端可替换”本身不足以建立新颖性。应清楚界定本文新增的是跨网格结构表示、安全与目标联合评分及部署验证的哪一项组合收益，并与一个同源信息、同预算的简单portfolio选点控制比较。现有八个复杂迁移后端不自动替代这种近邻比较。

**正确的理论尚不能说明源评分为什么独特。** Gonzalez保证对任意首点成立；条件覆盖也适用于其他足够覆盖的设计；源内部秩恢复不推出目标秩迁移。当前主实验没有展示覆盖充分条件实际成立的比例。论文可把它定位为正确性与适用条件说明；若把理论创新作为OR主卖点，需要新的、实质性的源信息收益分析，而不是增加Lean文件数量。

**实际应用的正向收益仍然有限。** Energy中源方法60/90、generic DCT70/90的负结果是有效贡献，应该保留。它也意味着真实外部案例目前没有证明源评分优于自然结构。匹配标准化后的合成收益又集中于两种机制。因此目前最有希望的论点是“在哪类有序策略迁移中值得购买或复用源结果”，需要给出读者能在目标结果打开之前判断的条件。

建议立场：以匹配对照后的贡献为中心重写定位，补近邻portfolio工作。是否再增加应用证据应由这个明确论点决定，不建议没有目标地增加更多大实验。

## Reviewer 3：报告准确性与复现交付

**总体判断：论文构建成熟，项目交付入口尚未跟上最新稿件。** 可再生的结果表和公开数据基础有助于仿真研究者复核；当前确定错误主要可以通过准确修订解决。

### 已通过

隔离编译 `main.tex`、`supplement.tex` 均exit 0。正文28页、参考文献2页、补充10页。没有undefined reference/citation或Overfull/Underfull。PDF Author为空。11张当前review表格从紧凑结果再生后与原稿逐字节一致。

当前本地14个矩阵的结果文件数合计17,360，与registry的矩阵数量要求一致。原始结果在本地，约1.58GB，均被git忽略。本轮关键配对统计只读取所需source160任务的标量，没有重新采集它们。

### 必须更正的规格

| 问题 | 证据位置 | 最小修复 |
|---|---|---|
| 库seed写成20260808，实际是20261799 | [主文方法:10](/home/erzhu419/mine_code/KG_op/SC-OLH-KG/manuscript/sections/04_atlas.tex:10)、补充Algorithm 1、method spec:17；[synthetic runner:254](/home/erzhu419/mine_code/KG_op/SC-OLH-KG/performance/benchmark_profile_stress_suite.py:254)、[Energy runner:151](/home/erzhu419/mine_code/KG_op/SC-OLH-KG/performance/benchmark_external_energy_v3.py:151) 均用family_seed+991 | 区分family_seed=20260808与library_seed=20261799；以实际产生结果的库为准，给历史spec勘误，不改变旧结果 |
| Native transfer写N=20，真实N=13 | [主文:89](/home/erzhu419/mine_code/KG_op/SC-OLH-KG/manuscript/sections/06_experiments.tex:89)、[补充:102](/home/erzhu419/mine_code/KG_op/SC-OLH-KG/manuscript/sections/appendix.tex:102)；[结果:622](/home/erzhu419/mine_code/KG_op/SC-OLH-KG/paper_artifacts/or_review/native_transfer.json:622) | 改为13次搜索、10次初始化；20是每域算法seed数 |
| Native验证协议与主实验不同 | [execution manifest:52](/home/erzhu419/mine_code/KG_op/SC-OLH-KG/performance/manifests/or_review_native_transfer_execution_v1.json:52)：v69，80/128/128，加8次objective incumbent comparison | 明确八个native彼此共用该历史协议，与主实验80/80/80 exact-binomial分开；修正可能误解为跨矩阵同验证的表述 |

上述规格修复都不需要改变已完成实验；如果要跨矩阵直接比较认证率，则必须统一验证协议并重新评估。

### 参考文献

对29条记录作元数据核对，发现两条实质错误、两处较小题名不完整；其余记录未发现新增身份错配。

1. [references.bib:99](/home/erzhu419/mine_code/KG_op/SC-OLH-KG/manuscript/references.bib:99) 的RothfussEtAl2023混用了不同论文的信息。PMLR205:237–265应为 **Meta-Learning Priors for Safe Bayesian Optimization**，作者 **Jonas Rothfuss, Christopher Koenig, Alisa Rupenyan, Andreas Krause**。[原始条目](https://proceedings.mlr.press/v205/rothfuss23a.html)。现稿所写 **Meta-Learning Reliable Priors in the Function Space** 则是Rothfuss、Heyn、Chen、Krause的NeurIPS2021论文。[原始条目](https://proceedings.neurips.cc/paper_files/paper/2021/hash/024d2d699e6c1a82c9ba986386f4d824-Abstract.html)。应按实际方法引用相应论文，必要时分别列出。
2. [references.bib:136](/home/erzhu419/mine_code/KG_op/SC-OLH-KG/manuscript/references.bib:136) 的MALIBO作者列表错误，正确为 **Jiarong Pan, Stefan Falkner, Felix Berkenkamp, Joaquin Vanschoren**。标题、2024年、235卷39102–39134页正确。[PMLR原始条目](https://proceedings.mlr.press/v235/pan24b.html)。
3. Feurer2018会议论文标题缺少 `using Ranking-Weighted Gaussian Process Ensembles` 后半段。[作者提供的会议论文](https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/18-AUTOML-RGPE.pdf)。Fu2002可补出版社题名前缀 `Feature Article:`，属于较小精度问题。[INFORMS条目](https://pubsonline.informs.org/doi/10.1287/ijoc.14.3.192.113)。

### 项目入口与交付包

[项目README](/home/erzhu419/mine_code/KG_op/SC-OLH-KG/README.md:30) 和 [复现README](/home/erzhu419/mine_code/KG_op/SC-OLH-KG/reproducibility/README.md:41) 仍把旧method、旧experiment registry、旧Energy正结果作为当前冻结论文。应改为当前profile_atlas_v2、OR review evidence registry及Energy V3入口，保留明确的历史路径。

[能源预处理命令](/home/erzhu419/mine_code/KG_op/SC-OLH-KG/reproducibility/README.md:30) 默认只生成四市场旧数据；V3使用extended_v2的扩展市场数据，其中18个为目标市场。说明需要给出对应扩展市场参数和输出文件。当前相关NPZ已在本地，修复文档不需要下载数据。

[Code and Data Disclosure](/home/erzhu419/mine_code/KG_op/SC-OLH-KG/manuscript/sections/10_disclosure.tex:5) 声称提供匿名包，但稿内没有访问位置，项目中未定位到当前版发布归档。投稿前应明确是上传附件还是提供匿名链接。紧凑aggregate足以重画现有表，却省略逐任务数据；仅git checkout不足以重新计算独立配对统计。可从已有本地结果导出分析所需逐任务标量，并连同代码、协议和准确命令组成小型包，无需打包完整策略向量或checkpoint。

按[OR当前投稿指南](https://pubsonline.informs.org/page/opre/submission-guidelines)，regular正文上限30页、摘要上限200词，EC不超过主文；当前页数符合。指南写11pt/1.5行距，当前模板使用12pt；标题页subject classifications及投稿cover letter等需在交付时补齐。这些是提交准备事项，当前阻止投稿的主要原因仍是前述实验归因。

建议立场：修正明确错误、统一当前复现入口并提供实际可访问的附件；编译成功不能替代数据和方法的一致性。

## Cross-review synthesis

三份评估一致认为：现稿已有清楚的研究对象、正确的条件理论、可再生的主表及诚实的负结果。当前最重要的问题是结构标准化与源信息混在一个比较中。本轮诊断说明修复有希望：匹配之后源方法仍高15.625个百分点，但差距明显缩小，且优势来源更集中。

推荐按以下顺序收尾：

1. **先闭合唯一关键实验对照。** 把标准化一致的无源设计纳入与文章主张对应的矩阵，保持既有源方法冻结，按机制和分辨率报告配对结果，并重算摊销经济结论。
2. **同时改正确定的文本与交付错误。** 库seed、native预算和验证协议、两条混错文献、频率惩罚解释、当前复现命令和匿名附件都可具体修正，无须重新训练复杂后端。
3. **再判断OR贡献是否足够。** 用已有初始化学习和portfolio研究作直接定位，解释匹配后收益主要出现在哪类结构迁移。如果坚持更宽的方法论或现实应用主张，补充实质证据；不能仅靠润色解决这个问题。

**能解决的程度：** 比较混杂和报告错误有明确修复路径；剩余收益已有积极的局部证据。是否最终达到OR录用要求无法保证。当前不宜再用旧文档里的“只剩上传”作为结论，也没有证据要求废弃整项研究。

## 本次审查的可复核附件

- [配对比较汇总](/home/erzhu419/mine_code/KG_op/SC-OLH-KG/docs/submission_review_20260907/kg_standardized_generic_comparison.json)：160任务总结果、八机制明细、配对区间。
- [160任务诊断标量](/home/erzhu419/mine_code/KG_op/SC-OLH-KG/docs/submission_review_20260907/kg_standardized_generic_diagnostic.json)：不含checkpoint或大策略向量。
- [诊断脚本](/home/erzhu419/mine_code/KG_op/SC-OLH-KG/docs/submission_review_20260907/kg_standardized_generic_diagnostic.py)、[配对分析脚本](/home/erzhu419/mine_code/KG_op/SC-OLH-KG/docs/submission_review_20260907/kg_summarize_standardized_diagnostic.py)、[原始标量核验脚本](/home/erzhu419/mine_code/KG_op/SC-OLH-KG/docs/submission_review_20260907/kg_validate_diagnostic_pairs.py)：保留本次运行快照，读取当前workspace，临时输出路径为`/tmp`。
- [当前Lean接口构建摘要](/home/erzhu419/mine_code/KG_op/SC-OLH-KG/docs/submission_review_20260907/kg_op_paper_interface_build_20260907.json)、[日志](/home/erzhu419/mine_code/KG_op/SC-OLH-KG/docs/submission_review_20260907/kg_op_paper_interface_build_20260907.log)。

## Limitations

- 标准化控制仅在原有d=1000的160任务上完成，是事后诊断；没有完成其他两个分辨率、Energy、新种子确认或近邻portfolio比较。区间用于描述此次诊断，不能升级为新独立确认。
- 17,360条结果数量一致不等于逐条内容全部审计；本轮重点检查当前紧凑结果及关键源方法160条原始标量。没有重复全项目测试、全部训练或所有历史研究分支。
- Lean本轮构建目标为当前PaperProofInterface及其依赖，没有宣称每个历史Lean入口均重建。LaTeX仅剩5.5pt数学粗体字体替代警告，未发现排版阻断。
- 文献核对为身份和出版元数据层面，没有逐篇重新评价全部正文引用是否支持每个细分命题。Letham2019的完整页段未通过单独出版社访问核实。OPSD版本和DOI已由官方数据页确认。
- 未检查用户在项目之外是否已准备作者信息、cover letter或匿名托管地址。所有审稿意见是本次证据评估，不代表期刊编辑决定。

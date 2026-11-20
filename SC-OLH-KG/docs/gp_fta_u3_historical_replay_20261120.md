# GP-FTA U3 历史档案与复现

U3 的三批原始工作件仍保留在集群部署目录。三个封存清单恢复到封存前格式后，均与账本登记的清单指纹一致。部署中的 `v2_or` Python 目录也与封存执行指纹 `5965482dba30…` 一致。检查复用原目录，未下载策略向量、源轨迹或模型，也未启动仿真。[复核回执](../paper_artifacts/v2_or_u3_historical_replay_receipt.json)

| 档案 | 来源作业 | 搜索／终验工作件 | 审计工作件 | 来源档案／拟合变体 |
|---|---:|---:|---:|---:|
| `v2_or_u3_confirm_v1` | 120 | 7,544 | 7,544 | 5／5 |
| `v2_or_u3_messy_archive_v1` | 60 | 2,108 | 2,108 | 4／8 |
| `v2_or_u3_live_sumo_v1` | 24 | 144 | 144 | 1／1 |

来源阶段每个保留档案均有 `source_bundles/<family>/archive.json` 与清单声明的 `prior_<variant>.pkl`。搜索工作件含完整的主候选、候选列表与部署策略。审计工作件保留相应的获证状态、各池的审计结果和调用数。成功任务账项、实验重试的调度记录及本轮复核的开销使用各自的记录。

五条服务器终验回执覆盖双获证后选择 incumbent、无守门角色的首项通过、第三项回退及全失败弃权。用已有违规计数重算二项检验，通过该批次实际分支重新选择，核对部署向量、终验调用、总调用和审计现状损失。选定 SUMO 弃权格复核得到终验 380 次、目标查询加终验 422 次，审计 160 次。所有这些次数均来自原运行。终验统计复核使用已有的充分统计量。

三批共 9,796 条原审计回执已由冻结 U3 聚合代码在服务器临时目录重建。六份输出 CSV 与各自封存指纹一致，每批摘要的 12 个实质字段逐项完全一致。新生成时间单独记录。单线程聚合实测分别为 4.1912、1.1615 和 0.1334 CPU 秒，合计 5.4862 CPU 秒。首批三个调度任务因命令转义在队列内撤回，均未启动。撤回和修复尝试全部保留，控制端和轻量元数据读取开销未单独测量，自动重派为零。

U3 Live SUMO 留下的 4 条来源、5 条搜索和 12 条审计调度记录都将同一个历史 `cfcmt-sumo122-py310-v1` 目录加入 `PYTHONPATH`。该目录中的 libsumo 目前报告引擎 `SUMO 1.22.0`、API 21，libsumo、traci、sumolib、eclipse-sumo 发行版均为 `1.22.0`。原 Python 执行指纹只覆盖 `v2_or` 中的 Python 文件。调度命令和目前可读的运行时相互印证版本，但历史引擎二进制及外部适配器没有逐次写入这个指纹，不能由它推出逐位等价。

## 受控访问和归档范围

保留集群 `KG_op_v2_or_cluster/SC-OLH-KG` 下三批封存清单以及每批的三个 `stages`、`source_bundles`、两份 CSV 和紧凑摘要。终验复核只需要对应 `search_and_verify/items/sv_*.json` 与 `audit_and_summarize/items/aud_sv_*.json`，提供单组读取权限即可。全表聚合须结合审计作业描述器、审计工作件和 `source_archive/completion.json`，按原顺序重建两份 CSV 和统计汇总。重新运行搜索另须源档案、已拟合先验以及仿真输入。源档案与拟合文件可在服务器原地复制到评审专用只读快照。端到端运行如经另行批准，应将发布包内的冻结 U3 Python 覆盖至独立工作目录，复用冻结参数、随机流、场景和已知运行时，并使用新输出目录。

评审配套包保留相同的目录结构。对每批纳入封存 manifest、紧凑摘要、三个阶段各自的 `jobs.json`、`completion.json`、`items/*.json`，以及 `source_bundles` 中的轨迹、先验和 36 个基线预训练文件。包另含冻结 U3 Python、聚合入口和 OPSD 准备清单。用于重绘的 CSV 在独立复现包中。配套包只保留 OPSD、RESCO 和 SUMO 的来源说明及重建入口，原始数据和引擎安装件留在既有位置。

完整配套包保存在作者专用服务器目录。任务 `t224844` 生成并读完 gzip 数据流，确认 19,914 个成员的名称及长度与来源清单完全一致。压缩包 107,850,921 字节。打包与读回实测 38.3656 CPU 秒，加上前述聚合合计 43.8518 CPU 秒。压缩包留在服务器，等待评审授权后的受控共享。

在集群运行轻量元数据查询或指定历史决策复核时，通过现有 `scheduleurm.run_on()` 调用以下入口。聚合模式读完整的服务器端审计工作件，使用 node001–node006 单线程 CPU 调度。生成的账本暂存于独立临时目录，比较通过后清理。

```bash
python read_u3_replay.py --sc-root "$GPFTA_U3_SC_ROOT" --campaign v2_or_u3_confirm_v1
python read_u3_replay.py --sc-root "$GPFTA_U3_SC_ROOT" --campaign v2_or_u3_confirm_v1 \
  --job sv_Inventory_CrossHorizon_N8_t09_s901_GP-FTA-KG
python read_u3_replay.py --sc-root "$GPFTA_U3_SC_ROOT" --campaign v2_or_u3_confirm_v1 --aggregate
```

## Data and code availability draft

The accompanying review artifact contains the sealed per-run and per-slot ledgers, experiment manifests, table and figure scripts, the frozen U3 Python implementation, and the corrected implementation. It regenerates the manuscript results from the sealed ledgers. A staged companion archive contains the source trajectories, fitted priors, search and verification receipts, and audit receipts for the three U3 cohorts. It is available for controlled reviewer access on request. We used the saved audit receipts to reconstruct all six sealed U3 ledgers and the statistical summaries without new simulation. The primary U3 cohorts used the archived code version specified in the manuscript. The Live SUMO jobs reference an archived runtime whose retained environment reports SUMO 1.22.0. The OPSD and RESCO inputs retain their original attribution and applicable redistribution terms.

If a specific upstream license precludes redistributing a prepared OPSD input, we will provide the pinned upstream acquisition and preparation instructions and identify the exact file and restriction in the submission's disclosure. The generated cohorts and terminal decisions remain available for verification through the review artifact and the archived read-only evaluator inputs.

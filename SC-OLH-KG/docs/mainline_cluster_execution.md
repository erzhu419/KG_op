# 主线实验的服务器执行入口

2026-10-17起，SC-OLH-KG的仿真、拟合、后验恢复和批量分析通过scheduler派到node001–node006。本地只做编辑、小量结果读取和确定性接线检查；已完成的Stage65不重跑。项目执行约定见[AGENTS.md](../../AGENTS.md)。

六节点各96物理核，共用NFS工作目录。调度器按实时空闲物理核规划，并预留系统资源；BLAS／OMP／MKL各1线程，不把96核误当成每个作业都可独占。科学批使用显式六节点池、vram=0和`submit-cpu-batch`，不回落本地，不固定到忙节点。

已验证：六节点均成功导入执行器、读取6份冻结来源模型；实际scheduler任务t166168–t166171在node002／006／005／004完成，6/6检查项通过，exit0。node001／003当时满载，正常跳过。6项分片接线测试通过。全过程新增仿真／拟合／后验恢复0，仅取回9,647B任务结果JSON；部署和失败记录见[environment_setup.json](../paper_artifacts/mainline_cluster_setup_20261017/environment_setup.json)。

- 部署根：`/home/zhengliang01/scheduleurm_work/KG_op_mainline_cluster`，工作目录为其`SC-OLH-KG`子目录。
- 独立Python：`/home/zhengliang01/scheduleurm_work/conda_envs/scomp-mainline-py310-20261017/bin/python`。
- 数值环境：Python3.10.14、NumPy2.2.6、SciPy1.15.3；后两者与本地确认实验一致，本地Python为3.10.12。复用服务器已有wheel离线安装，原`scomp-py310`环境未改。
- 主线实际导入依赖见[精简版本清单](../reproducibility/requirements-mainline-cluster.txt)。requests及四个HTTP依赖复用本地已安装同版小包上传约2MB，用于模块导入，没有API调用。
- 本地轻量部署目录：`/home/erzhu419/mine_code/KG_op_mainline_cluster`。
- 调度入口：`/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py`，远程短命令用同模块的`run_on()`，不直接`ssh node001`。

上传用[专用同步脚本](../../scripts/sync_mainline_cluster.py)，按[输入清单](../reproducibility/mainline-cluster-inputs.json)传源码、上层动态导入脚本、三家族各两个冻结模型、design及必要manifest／replay元数据。旧通用同步脚本会带入整个paper_artifacts且缺少上层脚本，这里使用小清单。同步不删除服务器结果，也不回拉数据。

```bash
python3 scripts/sync_mainline_cluster.py --dry-run
python3 scripts/sync_mainline_cluster.py --manifest SC-OLH-KG/performance/manifests/NEW_REGISTERED_MANIFEST.json
```

新登记任务使用[分片入口](../performance/mainline_cluster_dispatch.py)。顺序固定为`initial → neighbors → online → native → old → new`；每一阶段先在服务器`describe`生成jobs.json，调度执行全部分片后再由服务器单个协调进程`finish`收束。只把jobs.json小文件取回本地用于提交，结果raw／engine始终留在服务器。

```bash
# 服务器工作目录；先替换为新登记manifest，不能使用已经complete的Stage65。
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/zhengliang01/scheduleurm_work/conda_envs/scomp-mainline-py310-20261017/bin/python \
  -m performance.mainline_cluster_dispatch describe \
  --manifest performance/manifests/NEW_REGISTERED_MANIFEST.json --stage initial

# 本地SC-OLH-KG工作目录；descriptor沿用服务器的项目相对路径。
python3 -m performance.mainline_cluster_dispatch submit \
  --descriptor paper_artifacts/NEW_RUN/cluster/initial/jobs.json \
  --remote-cwd /home/zhengliang01/scheduleurm_work/KG_op_mainline_cluster/SC-OLH-KG \
  --remote-python /home/zhengliang01/scheduleurm_work/conda_envs/scomp-mainline-py310-20261017/bin/python \
  --ram-mb 32768

# 全部分片返回后，在服务器工作目录执行；下一阶段仍按此流程。
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/zhengliang01/scheduleurm_work/conda_envs/scomp-mainline-py310-20261017/bin/python \
  -m performance.mainline_cluster_dispatch finish \
  --descriptor paper_artifacts/NEW_RUN/cluster/initial/jobs.json
```

32GB是初次分片内存预约，按原实测约120–150MB／搜索worker留余量；workers由scheduler根据阶段item数及可用核决定。24任务的初始化为24个item，在线搜索为144个item，native交叉48个，旧／新终验各216个，不能把有依赖的168条总历史放在同一个并行阶段。describe／finish属于短协调命令；重计算只在scheduler worker内发生。

finish initial／online保留失败与费用；finish native复用原全局输入冻结；finish old建立可复用原始块清单；finish new封存全部终验。之后才在服务器串行调度原`performance.summarize_mainline_two_comparison_confirmation`和原runner的`finalize`。模型、来源、随机种子、三点名单规则、每格预算与统计协议不随部署改变。每分片独立进度与日志，不能并写同一progress文件。

结果读取优先task log、completion／summary的小JSON，再按问题选择少量CSV。科学任务不设置整个paper_artifacts为自动`result_dir`；需要的恢复文件在服务器原位使用。运行失败保留原前缀和账目，不能自动换种子或重跑完成批次。硬件规模提供更多并行容量，实际加速取决于可并行item数、已有负载和最长子任务，当前不声称测得固定加速倍数。

已知记录问题：本次scheduler的started_at／finished_at出现约18秒倒序，未用这些字段计算耗时。任务退出码和6项结果均已验证；本次没有扩大到修改scheduler。

# 2026-10-02 部署失误及恢复记录

本轮建立精简本地部署镜像后，scheduler启动所用的cwd同步默认含 `rsync --delete`。该镜像指向已有 `KG_op_scheduler_deploy/SC-OLH-KG`，且未保护profiles与paper_artifacts，使远端镜像未包含的文件被删除。这是本轮部署处理失误。没有完整删除前目录清单，无法准确列出所有受影响文件。

已完成恢复：从本地完整项目回传代码、稿件、paper_artifacts等，以及本地留存profiles；从jtl311linux既有备份经管道回传历史profiles，不在本地保存大包。备份159个目录、4,938个文件、1,490,058,240字节，与恢复后对应目录的文件数及总字节数一致，详见 `paper_artifacts/meta_prior_mechanism_v1_20261002/deployment_recovery.json`。已有results和archives受scheduler默认排除规则保护；检查确认旧评价results与source archive仍存在。

本轮128运行全部存在，7个服务器任务的scheduler状态为done；种子0原本地结果也已回传。先取回约362KB汇总JSON用于分析，再保留7个服务器原始JSON，共约1.46MB；本地现有8个原始分片、128运行，共1,670,364字节，没有拉取checkpoint。一个任务的自动结果同步曾因试图在本地创建 `/home/zhengliang01` 而报权限错误，手动汇总完整，未重复运行该任务。

精简本地镜像已移至独立 `KG_op_meta_prior_mechanism_deploy`，对应独立远端目录也已准备；后续机制实验使用该目录。原精简镜像路径已不存在，不能继续对旧研究目录执行这一精简同步。

## 尚未恢复

至少 `profiles/scolh_structural_backend_gate_n20_s10_20260716` 的服务器独有原始result记录尚未找到副本；本地当前profiles、jtl311linux已知备份、可见项目副本和archives中未找到该目录。已有1,350行统计汇总备份与本轮重新核对的紧凑记录保留，不能用它们冒充完整原始运行记录。不能确认旧服务器profiles已全部恢复，也未重新运行旧矩阵替代原结果。

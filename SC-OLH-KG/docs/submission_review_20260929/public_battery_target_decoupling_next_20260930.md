# 分单元目标反解与下一步

23 个保存的物理可行动作全部由分单元目标复现，最大偏差 **1.42e-14 MW**。第一台目标比例为 0.381–0.405，第二台为 0.763–0.765；4 项针对性测试通过。只复用了原后缀证据，没有新增数据、LP 或回放。

下一步已冻结 V5 协议及候选 CSV：沿用原 64 条分量曲线，组成 **100 个常数有序组合与 54 个函数配对**，保留每台电池全部原分量曲线。控制器只使用当时可见信息，反解比例不进入候选或决策。完整评估四月、七月、十月的全部 168 小时窗口；新增 31,536 次窗口评估，复用 2,847 次旧结果。保持容量、功率、效率、成本和 95% 门槛。V5 尚未运行。

局限：反解来自同一事件的重叠参考窗口及预知未来的后缀，只证明动作可表达。V5 的联合候选数增至 154，不能用旧 64 条矩阵把整周改善完全归因于目标解耦。source 比较保持 HOLD，2025 未读取。

结果：`paper_artifacts/public_battery_unit_target_decoupling_v1_20260930/summary.json`、`reconstruction.csv`、`reconstruction.json`。复算：`python3 performance/inspect_public_battery_target_decoupling.py`。下一步：`performance/manifests/public_battery_unit_controller_development_v5_20260930.json` 及同目录 `public_battery_unit_controller_development_v5_candidates_20260930.csv`。

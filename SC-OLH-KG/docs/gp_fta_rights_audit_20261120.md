# GP-FTA 投稿补件的权利与取得方式

独立 ZIP 是作者侧留档。期刊技术补件从该留档派生，保留科学账本、拟合与优化逻辑、表图再生及历史复核证据。模板文件、OPSD 紧凑数组和 RESCO XML 在评审机按锁定来源安装。补件替换提交端的账户路径默认值，省略集群搬运脚本、复核档案内部地址，并从费用回执中移除路径。完整源码与未裁剪回执由作者保管，实验执行指纹继续指向对应原件。

| 资产 | 核验与交付 |
|---|---|
| OPSD | `time_series/2020-10-06` 的 63 个分析列为 21 市场各一列负荷实测、负荷日前预测、日前价格，保存 2017–2019。`AT`、`DE_LU`、`GB_GBN`、`IE_sem` 的两个负荷字段及 `AT` 的价格字段由 OPSD 基于 ENTSO-E 数据计算。可选风光字段未进入数组。ENTSO-E 的 CC BY 4.0 清单按数据类与提供方适用，OPSD 发布元数据未为该聚合包声明统一转授权。保留官方发布地址、源与结果摘要、市场及预处理规则，紧凑数组从技术 ZIP 排除。一次官方源临时流式读取通过原 CSV 和重建 NPZ 的 SHA-256 验证。 [OPSD 元数据](https://data.open-power-system-data.org/time_series/2020-10-06/datapackage.json) · [ENTSO-E 再利用清单](https://transparency.entsoe.eu/content/static_content/download?path=%2FStatic+content%2Fterms+and+conditions%2F231018_List_of_Data_available_for_reuse.pdf) |
| RESCO／Ingolstadt21 | 场景目录 GPLv3。GitHub 上游 commit `01b969ba8248a0f49c5456124ec76682785620a2` 的两份 XML 经两条信号相位注释、22 个 `random_` 车辆显示色调整后，与历史实验文件逐字相同。提供短取得脚本、许可证和预期字节摘要。 [RESCO 固定场景](https://github.com/Pi-Star-Lab/RESCO/tree/01b969ba8248a0f49c5456124ec76682785620a2/resco_benchmark/environments/ingolstadt21) |
| INFORMS | 官方页面授权用途为期刊排稿，未发现该发布入口对当前 v1.02 class 或商标 logo 的通用再分发条款。技术补件从作者取得的官方模板 ZIP 安装并校验；期刊稿 PDF 单独提交。`informs2014.bst` 文件头明确 LPPL v1 或更新版本，保留原版权和许可通知。 [官方模板](https://pubsonline.informs.org/authorportal/latex-style-files) |
| Eclipse SUMO、Lean、Mathlib、Python | 仅锁定依赖，由用户分别从上游安装。SUMO 1.22.0 的主许可为 EPL 2.0，并列明 GPL 二次许可条件。Lean 和 Mathlib 均提供 Apache 2.0。各依赖沿用原许可。 [SUMO v1.22.0](https://github.com/eclipse-sumo/sumo/blob/v1_22_0/LICENSE) · [Lean v4.31.0](https://github.com/leanprover/lean4/blob/v4.31.0/LICENSE) · [Mathlib v4.31.0](https://github.com/leanprover-community/mathlib4/blob/v4.31.0/LICENSE) |

自有软件和自有生成数据拟沿用 MIT、CC BY 4.0。发出前由通讯作者确认实际权利人、归属声明和最终署名，旧版主线的许可文本仅对应其原范围。历史完整工作件的受控存储已就绪。是否向评审开放由通讯作者决定。若编辑部要求无法再分发的源数据或完整档案，随投稿信说明固定取得方式和受控复核范围，按 OR 政策申请相应安排。 [OR 代码与数据政策](https://pubsonline.informs.org/page/opre/code-and-data-disclosure-policy)

## 验收

`GP_FTA_OR_Review_Supplement_20261120.zip` 为 4,369,833 字节。重新解包校验 230 项，已恢复的官方 OPSD 数组包含 21 市场、508,610 个小时，RESCO 固定取得与历史 XML 逐字一致。两份 INFORMS 文件用作者留存的同字节模板归档核验后完成 29 页编译。出版社当前下载入口对运行环境返回 HTTP 403，未验收该环境的自动下载。16 项 Python 测试通过，432 格统计、七张表与三张图逐字再生。历史 ZIP 和受控档案保持原样，未对外发布。

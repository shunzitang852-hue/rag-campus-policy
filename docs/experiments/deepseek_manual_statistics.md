# DeepSeek 人工评价后处理

这个离线模块只读取已完成的 DeepSeek 实验。它不调用 API，不写入原盲评 CSV、blind key、模型回答、自动评价文件或历史实验，也不编辑论文正文。

## 当前输入格式

真实盲评表有 `experiment_id, question_id, anonymous_model, question, context, context_sha256, reference_answer, answer, correctness, faithfulness, completeness, hallucination, review_note`。`context` 原样保留，并作为解盲表的 `evidence`。已有自动文件的 `prompt_sha256` 会经配对校验后补入解盲输出。

结构校验严格要求 80 行、40 个唯一 question_id、每题 Model A / Model B 各一行、相同 question、Context 和 hash。重复、缺行、多模型、列宽错误或 hash 不符均 FAIL FAST，不自动修复。四个评分必须为有限整数值：前三项 0/1/2，hallucination 0/1。review_note 可空；空值/NaN 不等同于 0。

真实 blind key 在评分未完成时不解析。`--validate-only` 始终不解盲。正式统计先验证所有 320 个评分，再读真实 key 的 `mapping`，确认两匿名模型唯一对应 deepseek-flash 和 deepseek-v4-pro。文档和验证日志不展示映射关系。

## 使用

从 `F:\GraduationThesis\rag-campus-policy` 运行：

```powershell
.\.venv\Scripts\python.exe -m src.experiments.analyze_deepseek_manual_review --experiment-id 20261005T124451501115Z_deepseek_cross_model --validate-only
```

缺分时打印 `manual review incomplete`，并生成 `manual_review_validation.csv`、`manual_review_validation.json` 和 `deepseek_manual_review_guide.md`。每条问题记录包含 CSV 行号、question_id、anonymous_model、字段、原值及原因。校验模式返回 0；正式模式遇缺分返回 2；结构或来源错误返回 1。**此时不生成正式统计或解盲输出。**

人工完成全部评分后执行：

```powershell
.\.venv\Scripts\python.exe -m src.experiments.analyze_deepseek_manual_review --experiment-id 20261005T124451501115Z_deepseek_cross_model
```

所有输出仅位于给定 experiment_id 的目录。CLI 不接收任意输出路径，拒绝目录穿越。开发测试仅使用 pytest 临时目录内的合成 80 行 fixture，不填充真实评分。

## 统计方法

- 按 question_id 配对，所有差值定义为 Flash − V4-Pro。前三项正差表示 Flash 更高；hallucination 负差表示 Flash 更低。
- 描述统计提供 N、均值、样本标准差（ddof=1）、中位数、Q1/Q3（linear quantiles）、范围和完整等级分布。
- 每模型每等级指标采用 percentile bootstrap 95% 均值 CI，10,000 次重采样；根 seed=42，SeedSequence 固定生成 6 个独立子流。这是每模型描述区间，不能视为配对差值区间。
- 二值发生率使用 Wilson 95% CI，无连续性修正。
- Wilcoxon 双侧 signed-rank：剔除零差（wilcox），绝对差 ties 取平均秩，使用 ties 修正的渐近 p-value，不做连续性修正。全零差、无法计算或非有限结果标记 `test_not_applicable`，p 保持空。差值很少时渐近近似需谨慎；有序评分的等距及差值对称性不由本程序自动确认。
- rank-biserial = (正秩和 − 负秩和) / (正秩和 + 负秩和)，同样去零、ties 平均秩；全零差效应量为空。另输出均值差、更好/相同题数。
- 三个预定等级检验统一采用 Holm-Bonferroni。缺少 p-value 的检验不赋值，只在校正族大小计算中保留该检验位（等效占位 p=1），避免事后缩小三项检验族。
- 幻觉构建 Flash 行 / Pro 列的 a,b,c,d 表。所有 discordant 数量均用 exact two-sided McNemar（scipy binomtest），无需 statsmodels。统计量 min(b,c)；无 discordant 时 exact 退化分布 p=1。作为独立安全性指标报告。
- 自动评价按 question_id + requested_model 严格一对一 join，校验回答、Context hash 与逐题 Prompt hash。相关性计算关键词重合率 vs correctness、证据关键词重合率 vs faithfulness，以及 good/needs_review vs correctness。缺列跳过、常量列不计算；分别按模型及探索性的合并 80 回答报告。合并 p-value 未校正同题聚类，不做因果解释。
- Error Analysis 每类至多 3 个题目索引，前三指标至少 1 分差定义明显更高；自动/人工反向使用关键词重合率和 correctness。只提供统计索引，不替人工解释政策正确性。

## 完整评分后的输出

| 文件 | 内容 |
|---|---|
| deepseek_cross_model_manual_review_unblinded.csv | 原字段及真实模型、evidence、可用 prompt hash |
| deepseek_cross_model_paired_manual_scores.csv | 40 个配对评分、Flash−Pro 差值、幻觉 discordance |
| deepseek_manual_evaluation_summary.csv | 每模型描述统计及 CI |
| deepseek_manual_score_distributions.csv | 各评分等级 count 和 percent |
| deepseek_manual_significance_tests.csv | Wilcoxon、Holm、效应量及题数方向 |
| deepseek_hallucination_mcnemar.csv | exact McNemar、a/b/c/d 和 discordance |
| deepseek_auto_manual_joined.csv | 严格 join 后自动与人工评价 |
| deepseek_auto_manual_correlation.csv | Spearman，缺列/常量状态 |
| deepseek_manual_evaluation_report.md | 设计、方法、表格、Error Analysis、保守结论 |
| deepseek_manual_evaluation_table.tex | 描述统计 table/tabular 片段 |
| deepseek_significance_table.tex | 检验 table/tabular 片段 |
| deepseek_manual_statistics.json | 统计结果、方法、时间、来源、哈希 |
| statistics_manifest.json | 三个输入和分析脚本 SHA-256、输出哈希及方法 |

LaTeX 片段需要父文档的 `booktabs` 与中文支持；通过 `\input{}` 插入，不生成完整文档。未填完评分前不生成实际表格；测试中的片段来自明确标记的 synthetic fixture。

重复运行不会变更输入。统计值固定种子可复现，生成时间会更新。评分改动后需重跑，核对 manifest 中输入 SHA-256；旧结果不能继续作为当前评分版本的结论。若改回不完整，旧统计文件不会被偷偷删除，validation.json 会明确当前评分未完成；以当前输入哈希与完成状态判断统计文件是否可用。不要同时编辑评分或并发运行统计。

当前环境已有 numpy/scipy；未添加、安装或下载任何依赖。方法与安装版本记录在统计 JSON 和 manifest。检验不显著只能写“当前样本下未观察到统计显著差异”，不能证明等效；显著性也不能自动解释成实践重要性。

# 跨模型实验章节写作与收口说明

论文固定题目：**《基于RAG技术的高校政策知识问答系统构建与检索策略优化研究》**。

本轮实验阶段状态：跨生成模型稳健性验证完成，进入论文写作与收口阶段。
本轮没有调用LLM API，没有重跑检索、生成或正式统计，没有修改实验结果、答案或人工评分。

## 正文定位与修改范围

在 `F:/GraduationThesis` 及项目论文目录搜索 `.tex`、`.docx` 和相关论文材料，
发现的已有LaTeX主文件均为开题报告或开题答辩材料：

- `F:/GraduationThesis/thesis_documents/opening_report_latex/opening_report.tex`
- `F:/GraduationThesis/thesis_documents/opening_defense_latex/opening_defense.tex`
- 仓库 `opening_report_final/`、`opening_report_final_v2/` 内的开题报告副本。

尚未定位正式毕业论文正文主文件，因此没有修改任何开题报告，也没有伪造完整论文。
一个旧pytest临时目录 `F:/GraduationThesis/pytest_tmp` 无读取权限，其他发现的pytest文件为合成测试表格，不属于论文正文。

本轮生成的可插入正文草稿为：

- [实验设计、生成模型验证及实验讨论](cross_model_experiment_section.tex)
- [结论补充段落](cross_model_conclusion.tex)
- [数据来源与输出SHA-256清单](cross_model_sources.json)

章节草稿采用 `\section`、`\subsection`、`\label` 和 `\ref`，不硬编码章节编号。
它面向未来正式论文的“实验设计与结果分析”章，保留既有检索主线，扩展稳健性实验设计、
DeepSeek同家族对照、Qwen跨家族对照、自动与人工关联及有效性威胁。
没有重写绪论、研究现状、需求或系统设计。

## 表格资产

下列文件全部位于 `docs/thesis/tables/`，原实验目录的表格与结果未修改。

| 文件 | 内容 |
|---|---|
| automatic_all40.tex | 三模型自动评价、Good/Needs Review数量及完整生成完成率 |
| manual_all40.tex | Qwen与Flash人工均值±样本SD及幻觉数量/率 |
| significance_all40.tex | 均值差、Wilcoxon W、原始p、Holm p、rank-biserial与保守结论 |
| automatic_common39.tex | 三模型共同完整39题自动评价敏感性分析 |
| manual_common39.tex | 排除唯一截断题后的人工评价均值及幻觉计数 |
| design.tex | 固定证据、生成配置、两批盲评与配对分析的设计表 |
| deepseek_manual.tex | 从锁定DeepSeek表逐字复用的人工描述统计表 |
| deepseek_significance.tex | 从锁定DeepSeek表逐字复用的检验表，保留N/A原因 |

主文引入七张表，DeepSeek检验表作为可选补充资产，不重复展示全零差值检验。
置信区间写入正文，避免全部塞入主表。显著性表没有星号。
原Qwen均值/检验表未直接使用，因为缺少主文所需SD、W、结论或完成率列；
论文专用表由同一正式JSON/CSV生成，避免手工维护数值。

## 使用的正式实验与来源

- DeepSeek：`20261005T124451501115Z_deepseek_cross_model`
- Qwen：`20261006T070200000000Z_qwen38max_tokenpony_cross_family`
- 冻结检索输入：`20260910T035306268526Z_retrieval_real-hybrid-weighted_fixed-length_topk5_2b110eac`

正式输出根目录为 `F:/GraduationThesis/experiment_outputs/hdu_ies/student_handbook/`。
主要来源包括：

- DeepSeek的 `deepseek_manual_statistics.json`、`deepseek_manual_evaluation_report.md` 和原LaTeX表。
- Qwen的 `cross_family_three_model_auto_summary.csv`、`qwen_deepseek_manual_statistics.json`、
  `cross_family_robustness_final_report.md`、`qwen_deepseek_statistics_manifest.json`、
  `config.json`、`frozen_evidence_manifest.json` 和原LaTeX表。
- Frozen Evidence manifest指向的QA、Chunk、检索配置、逐题检索结果与已保存的 `metrics.json`。

导出脚本逐一核验统计manifest登记的输出及输入hash，并核验两个实验的Frozen Evidence字节一致、
原问题/Chunk/检索输入的hash一致。14个直接写作来源及10个LaTeX资产的完整路径/hash见
`cross_model_sources.json`。指令给出的核心指标与正式文件没有发现冲突。

导出入口：`scripts/export_cross_model_thesis.py`。默认输出到 `docs/thesis`。
已存在草稿时默认拒绝覆盖；正常编辑草稿应同时更新导出模板并重新核对来源清单。
脚本只投影已登记指标，不生成答案、不重新评分、不重新执行统计。

## 主要实验结论

1. DeepSeek两模型在旧盲评批次均为2.000/2.000/2.000、幻觉0/40。
   全零配对差值使Wilcoxon与rank-biserial不适用；结果反映天花板效应，不是等效性证明。
2. All40自动关键词/证据重合率及Good数量均为Qwen > Flash > V4-Pro；Common39排序保持不变。
3. 本轮Qwen正确性/忠实性/完整性均值为1.975/1.975/1.850，
   Flash为1.850/1.725/1.675；幻觉计数为0/40对1/40。
4. 三个维度Holm p分别为0.257956、0.059083和0.257956，均未达到0.05水平。
   忠实性效应量为+0.844444，但有效非零配对仅9题，raw p=0.019694不能代替校正后判断。
5. 幻觉exact McNemar p=1；Qwen在当前样本零幻觉不能解释为总体风险为零或显著降低幻觉。
6. Q023是真实512-token截断；主分析保留原部分答案，Common39排除后方向与显著性判断不变。
7. 合并样本的证据重合率与人工三个维度的探索性关联强于普通关键词重合率；
   未校正多重比较及题内聚类，自动评价不能代替语义评价。
8. 三个API配置、两个模型家族构成当前稳健性验证；两次独立盲评分别推断，
   没有统一三模型人工显著性排名。没有重做各生成模型下的完整检索策略矩阵，
   因此跨模型观察不能直接推广为任意生成模型下最优检索策略不变。

## 写作质量与方法学限制

已按事实核验、论证边界、统计措辞、重复表达和前后衔接进行人工式复核。
外部Academic Writing Toolkit、ra-人话和Self Plagiarism Checker未在本轮调用，
不声称获得这些插件的审核结果。本轮仅报告本项目实验，没有新增外部文献或猜测引用。

保留的限制包括：40题、小范围政策来源、DeepSeek天花板效应、Qwen评分接近上限、
第三方网关无法核验固定权重、provider token口径差异、单次截断、单评审及无评审者间一致性、
独立盲评批次差异、自动评价局限、少量非零配对和探索性相关分析未调整。

## 验证与编译

代码收口全量pytest：**352 passed in 35.83s**。
运行时禁用socket网络连接，并设置HF/Transformers离线模式；真实API调用为0。
代码审查未发现硬编码凭据、保存的Authorization值、候选 `.env`、模型权重或无关大型资产。
接口代码仍从环境变量在内存构造认证header，不保存header值。

草稿已验证生成数值与正式来源一致、hash匹配、LaTeX括号与环境闭合、
本地label/ref及input路径可解析，且题目未改变。

没有正式论文主文件，按用户要求只交付章节及表格片段，**未编译完整论文、未生成论文PDF**。
本轮调用目录内没有可用的xelatex/latexmk/tectonic命令；原生LaTeX编辑/编译工具也未提供。
目前不能声称已经验证最终表格宽度、中文字体、图表/章节编号或完整论文的reference/citation警告。
未来将片段插入学校模板时需启用 `booktabs`、`tabularx` 和模板已有的中文支持。

从仓库根目录编译的正式主文件可使用：

```latex
\input{docs/thesis/cross_model_experiment_section.tex}
% 在既有结论章节合适位置：
\input{docs/thesis/cross_model_conclusion.tex}
```

若正式论文位于仓库外，应将 `\CrossModelAssetRoot` 设置为资产目录的可移植相对路径，
或将论文专用资产迁移到正式论文目录后再调整input；不要把外部正文强行复制进本仓库。

## Git与尚待确认

代码checkpoint：`7ea7562`，提交信息为
`feat(experiments): add cross-family LLM robustness evaluation`。
本轮论文草稿和导出脚本另作论文checkpoint，完整hash以最终报告及Git历史为准。
experiment_outputs位于仓库外，未纳入提交。没有push。
`opening_report_final/`、`opening_report_final_v2/`、`文献包.zip` 保持原状。

真正剩余的论文事项为：定位或建立学校模板正式正文、确认片段插入位置与已有章节的衔接、
在正式模板中编译并核对字体/表宽/编号及引用。无需新增生成模型实验。

# 基于 RAG 技术的高校政策知识问答系统

## 一、项目名称和论文题目

项目名称：基于 RAG 技术的高校政策知识问答系统

论文题目：

《基于RAG技术的高校政策知识问答系统构建与检索策略优化研究》

主要案例对象：

杭州电子科技大学信息工程学院

## 二、主要研究对象

本项目的主要研究对象为杭州电子科技大学信息工程学院，简称“杭电信工”。

英文标识建议：Hangzhou Dianzi University Information Engineering School

内部机构代码：`hdu_ies`

本研究选取杭州电子科技大学信息工程学院的学生管理制度及相关高校政策文件作为主要案例语料，开展高校政策知识问答系统构建与检索策略优化研究。

## 三、主要语料范围

主要语料包括：

* 学生手册
* 学籍管理制度
* 学位授予制度
* 奖助学金制度
* 考试纪律制度
* 毕业设计与论文管理制度
* 学生申诉与处分制度
* 创新创业和学科竞赛制度

国家及教育部层面的政策文件只作为上位政策和补充语料，不替代杭州电子科技大学信息工程学院自身制度文件。

后续若加入杭州电子科技大学本部或其他高校政策，必须作为独立机构语料保存，不得与 `hdu_ies` 语料混用。

## 四、项目目标

本项目的主要目标包括：

1. 建立杭州电子科技大学信息工程学院政策文档数据集。
2. 对政策来源、发布主体、适用范围和有效性进行人工核验。
3. 实现 PDF、DOCX 和 TXT 等政策文档的受控解析。
4. 对政策文本进行清洗和规范化处理。
5. 研究不同 Chunk 切分策略对检索结果的影响。
6. 使用 Embedding 模型生成政策文本向量。
7. 使用 ChromaDB 建立向量数据库。
8. 实现 Dense、BM25 和 Hybrid 检索。
9. 构建面向杭州电子科技大学信息工程学院学生政策问答的测试问题集。
10. 使用 Recall@5 等指标评估检索效果。
11. 构建可演示的高校政策知识问答系统。

## 五、项目目录

```text
F:\GraduationThesis\
├─ rag-campus-policy\       项目源代码、配置、脚本和文档
├─ datasets\                原始政策文档、清洗数据和测试数据
├─ models\                  Embedding 模型和重排序模型
├─ cache\                   模型缓存和运行缓存
├─ experiment_outputs\      实验结果、日志、指标和图表
└─ thesis_documents\        论文材料、阶段报告和参考资料
```

当前主要数据目录：

```text
F:\GraduationThesis\datasets\
├─ raw\hdu_ies\             杭州电子科技大学信息工程学院原始政策文件
├─ interim\hdu_ies\         中间处理文件
├─ processed\hdu_ies\       清洗、切分和结构化后的数据
├─ metadata\hdu_ies\        元数据模板、索引和核验记录
└─ evaluation\hdu_ies\      测试问题集和评估数据
```

项目代码目录结构：

```text
rag-campus-policy\
├─ src\                     核心源代码
├─ tests\                   测试代码
├─ configs\                 配置文件
├─ scripts\                 数据处理和实验脚本
├─ notebooks\               探索性实验记录
├─ docs\                    项目文档
├─ logs\                    运行日志
├─ requirements.txt         Python 依赖清单
├─ README.md                项目说明
├─ PROJECT_PATHS.md         路径说明
├─ environment_check.md     环境检查记录
└─ .gitignore               Git 忽略规则
```

## 六、系统回答展示规则

系统后续回答政策问题时，应优先显示：

1. 结论。
2. 政策文件名称。
3. 发布机构。
4. 适用范围。
5. 引用条款或原文片段。
6. 页码或章节位置。
7. 来源核验状态。

回答模板建议：

```text
根据杭州电子科技大学信息工程学院发布的《文件名称》第X条，……。该规定适用于……。
```

如果问题中只写“杭电”，系统不能自动将其解释为杭电信工，应提示用户确认：

```text
请确认您咨询的是杭州电子科技大学本部，还是杭州电子科技大学信息工程学院。
```

## 七、后续开发顺序

当前阶段进展：

已完成：

* 学生手册原始文件安全导入
* 32项制度目录提取
* 制度级拆分
* 章节、条款、款项和子项结构解析
* 保守文本清洗
* 来源行号追踪
* 异常检测和二次分类
* 表格候选复核材料生成
* 自动化测试
* 人工抽样核验材料生成，并由用户确认允许进入 Chunk 设计阶段
* 三种 Chunk 策略设计与外部 Chunk 数据生成
* Chunk 覆盖率、长度分布和自动化验收
* 第一版学生手册问答测试集构建
* deterministic hash embedding 与 Dense Search 检索基线流程验证
* 标准库 BM25 关键词检索基线流程验证

Chunk 阶段说明：

* Chunk 输入文件：`F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\cleaned\hdu_ies_student_handbook_cleaned.txt`
* Chunk 输出目录：`F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\chunks`
* 当前已生成 `fixed_length`、`article_based` 和 `structure_aware` 三种切分策略。
* 本阶段不包含 Embedding、向量库、BM25、Dense Search、Hybrid Search、Reranker 或 RAG。

问答测试集阶段说明：

* 测试集输出目录：`F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset`
* 第一版测试集包含 40 条问题，覆盖 32 项制度。
* 每条问题保留原文证据 `evidence_text`，并关联三种 Chunk 策略下的 chunk_id。
* 本阶段不包含 Embedding、向量库、BM25、Dense Search、Hybrid Search、Reranker 或 RAG。

Dense Search 基线说明：

* 外部输出目录：`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval`
* 当前使用 `deterministic_hash_embedding`，不是正式语义 Embedding 模型。
* 当前结果只用于验证 Dense Search 检索、Recall@K 和 MRR@10 评估流程。
* 本阶段不包含 BM25、Hybrid Search、Reranker 或 RAG。

BM25 基线说明：

* 外部输出目录：`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\bm25_retrieval`
* 当前使用 Python 标准库 BM25，分词方法为中文单字和相邻二元片段。
* 当前结果用于验证关键词检索、Recall@K 和 MRR@10 评估流程。
* 本阶段不包含 Hybrid Search、Reranker 或 RAG。

当前下一阶段：

* 对 Dense Hash 与 BM25 基线结果进行论文实验口径整理
* 在单路检索基线稳定后再进入 Hybrid 检索实验

后续按照以下顺序推进：

1. 杭州电子科技大学信息工程学院政策文档收集与有效性核验。
2. 建立机构级政策文档元数据表。
3. 实现 PDF、DOCX 和 TXT 文档解析。
4. 实现文本清洗。
5. 实现 Chunk 切分。
6. 选择并测试 Embedding 模型。
7. 建立 ChromaDB 向量数据库。
8. 实现 Dense 检索。
9. 实现 BM25 检索。
10. 实现 Hybrid 混合检索。
11. 建立第一版测试问题集。
12. 完成基线检索实验。
13. 计算 Recall@5 等评价指标。
14. 保存实验数据、日志和图表。
15. 完善系统演示和论文实验记录。

## 八、开发原则

* 所有正式代码统一存放在 `F:\GraduationThesis\rag-campus-policy`。
* 杭州电子科技大学信息工程学院、杭州电子科技大学本部和其他高校语料必须按机构分开保存。
* 原始数据、模型、缓存和实验结果分开存放。
* 不将项目正式文件长期存放在桌面或下载目录。
* 不直接修改原始政策文档。
* 不把上位政策、其他学校政策或历史政策直接改写为杭电信工政策。
* 每次实验保存配置、结果和日志。
* 人工核验政策文件的真实性、有效性、发布机构、适用范围和发布时间。
* 人工标注测试问题对应的相关文档、页码和标准答案。
* 所有实验结果必须以真实运行结果为准，不得编造数据。

## 九、注意事项

本项目目前仍处于数据准备和核验阶段，后续模型选择、Chunk 参数、检索策略和评价指标需要通过实际实验确定。

Codex 可以辅助完成代码编写、环境执行、测试、统计和图表生成，但以下内容必须由项目负责人亲自核验：

* 政策文件是否真实有效；
* 政策文件是否已经废止或更新；
* 政策发布机构是否确为杭州电子科技大学信息工程学院或可作为上位补充政策；
* 政策适用范围是否覆盖普通全日制本科生；
* 测试问题是否合理；
* 标准答案是否正确；
* 相关政策文档和页码标注是否准确；
* 实验变量和控制变量是否明确；
* 最终结果是否来自真实运行。
## Real Embedding Dense Search

Real Chinese embedding Dense Search is available through `scripts/evaluate_real_embedding_dense_retrieval.py`.

- Provider: `sentence_transformers`
- Model: `BAAI/bge-small-zh-v1.5`
- Model cache: `F:\GraduationThesis\models`
- HF cache: `F:\GraduationThesis\cache\huggingface`
- Output root: `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval_real_embedding`

Run:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_real_embedding_dense_retrieval.py
```

This stage only performs real embedding Dense Search. It does not run BM25, Hybrid Search, Reranker, or RAG generation.

## Real Embedding Hybrid Search

Real Embedding + BM25 Hybrid Search is available through `scripts/evaluate_real_embedding_hybrid_retrieval.py`.

- Weighted fusion: real dense `0.5`, BM25 `0.5`
- RRF parameter: `k = 60`
- Output root: `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\hybrid_retrieval_real_embedding`

Run:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_real_embedding_hybrid_retrieval.py
```

This stage only fuses existing Real Embedding Dense and BM25 results. It does not rebuild BM25, overwrite old Hash Hybrid results, run Reranker, or generate RAG answers.

## 检索问答 Demo

Retrieval-only QA Demo is available through `scripts/demo_retrieval_qa.py`.

- Default method: `real_dense`
- Default chunk strategy: `fixed_length`
- Default top_k: `5`
- Supported methods: `bm25`, `real_dense`, `real_hybrid_weighted`
- Supported strategies: `fixed_length`, `article_based`, `structure_aware`

Run:

```powershell
.\.venv\Scripts\python.exe scripts\demo_retrieval_qa.py --query "学生申请解除处分需要满足什么条件？" --method real_dense --strategy fixed_length --top-k 5
```

The demo only displays retrieval results from the student handbook chunks. It does not call an LLM, run a Reranker, generate final RAG answers, or overwrite historical experiment outputs.

## RAG QA Demo

Extractive RAG QA Demo is available through `scripts/demo_rag_qa.py`.

- Default method: `real_hybrid_weighted`
- Fallback methods: `real_dense`, then `bm25` if the real embedding provider is unavailable
- Default chunk strategy: `fixed_length`
- Default top_k: `5`
- Supported methods: `bm25`, `real_dense`, `real_hybrid_weighted`
- Supported strategies: `fixed_length`, `article_based`, `structure_aware`
- Answer mode: `extractive`

Run:

```powershell
.\.venv\Scripts\python.exe scripts\demo_rag_qa.py --query "学生申请解除处分需要满足什么条件？" --method real_hybrid_weighted --strategy fixed_length --top-k 5
```

The RAG demo generates a short template-based answer from retrieved Chunk text and prints citations with source document titles, `chunk_id`, scores, strategy, and retrieval method. It does not call a paid API, run a Reranker, build a Web frontend, or overwrite historical experiment outputs.

## RAG Batch Evaluation

RAG batch evaluation is available through `scripts/evaluate_rag_qa.py`.

- Default method/strategy/top_k: `real_hybrid_weighted` + `fixed_length` + `5`
- QA input: `F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset\hdu_ies_student_handbook_qa_testset_v1.jsonl`
- Output root: `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_evaluation`
- Metrics include answer non-empty rate, citation coverage, source policy hit rate, related chunk hit rate, fallback count, and grouped summaries by question type and difficulty.

Run:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_rag_qa.py
```

The evaluator sends only question text to the RAG pipeline. QA labels and evidence fields are used only for automatic scoring. Generated external evaluation files are not committed to Git.

## RAG Answer Quality Evaluation

RAG answer quality evaluation is available through `scripts/evaluate_rag_answer_quality.py`.

- QA input: `F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset\hdu_ies_student_handbook_qa_testset_v1.jsonl`
- RAG answer input: `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_evaluation\rag_answers_real_hybrid_weighted_fixed_length.jsonl`
- Output root: `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality`
- Outputs include a JSONL review file, a CSV review sheet with manual scoring columns, and a JSON summary report.

Run:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_rag_answer_quality.py
```

The evaluator uses rule-based keyword and evidence overlap signals plus citation/chunk checks. These automatic metrics are only review aids and are not equivalent to final answer correctness.

## RAG 答案质量优化

本轮 RAG 答案质量优化只调整 extractive/template-based answer generator 的规则，不调用大模型 API，不使用 Reranker，不构建 Web 前端，也不覆盖历史实验结果。

- 规则配置：`configs/rag_answer_generation.yaml`
- needs_review 分析脚本：`scripts/analyze_rag_answer_quality.py`
- v2 评估脚本：`scripts/evaluate_rag_answer_quality_improvement.py`
- 设计文档：`docs/rag/rag_answer_quality_improvement_design.md`
- 结果报告：`docs/rag/rag_answer_quality_improvement_report.md`
- 外部输出目录：`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality_improvement`

Run:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_rag_answer_quality.py
.\.venv\Scripts\python.exe scripts\evaluate_rag_answer_quality_improvement.py
```

v2 仍使用 `real_hybrid_weighted + fixed_length + top_k=5`。生成阶段只接收问题文本和检索返回的 Chunk 内容，`expected_answer` 与 `evidence_text` 只用于后续质量评估。

## RAG v2 误差分析与人工复核辅助

当前已完成 RAG v2 回答质量优化，并新增 v2 误差分析与人工复核辅助表。本阶段重点不是继续改动历史实验结果，而是围绕 v2 中的 `needs_review`、v1 到 v2 变差样例、`condition` 类型问题和 `medium` 难度问题建立可复核的误差分析材料。

- 误差分析脚本：`scripts/analyze_rag_answer_errors_v2.py`
- 误差分析逻辑：`src/rag/answer_error_analysis.py`
- 误差分析文档：`docs/rag/rag_answer_error_analysis_v2.md`
- 外部输出目录：`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality_error_analysis`

Run:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_rag_answer_errors_v2.py
```

该脚本输出 needs_review 人工复核表、v1 到 v2 变差样例复核表、condition/medium 重点问题表和 JSON 汇总报告。下一步将基于这些误差分析结果推进 v3 定向优化。当前仍未调用大模型 API，未做 Reranker，未做 Web 前端。

## RAG v3 定向答案优化实验

当前已完成 v2 误差分析，并新增 v3 定向优化实验。v3 重点处理 `condition` 条件覆盖不足和 `medium` 难度回答过泛的问题，同时为 `procedure`、`responsible_body`、`qualification` 等类型增加结构化回答模板。

- v3 生成脚本：`scripts/generate_rag_answers_v3.py`
- v3 质量评估脚本：`scripts/review_rag_answer_quality_v3.py`
- v2/v3 对比脚本：`scripts/compare_rag_answer_quality_v2_v3.py`
- v3 优化逻辑：`src/rag/answer_v3_optimization.py`
- v3 实验文档：`docs/rag/rag_answer_quality_v3_optimization.md`
- 外部输出目录：`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality_v3_optimization`

Run:

```powershell
.\.venv\Scripts\python.exe scripts\generate_rag_answers_v3.py
.\.venv\Scripts\python.exe scripts\review_rag_answer_quality_v3.py
.\.venv\Scripts\python.exe scripts\compare_rag_answer_quality_v2_v3.py
```

v3 自动评估显示，condition 和 medium 目标分组有改善，但整体 `average_keyword_overlap_ratio` 和 `needs_review_count` 不如 v2。因此 v3 当前应视为定向优化实验版本，而不是默认替代 v2 的稳定版本。当前仍未调用大模型 API，未做 Reranker，未做 Web 前端。下一步建议进行人工复核，或设计 Reranker / LLM Judge 对比实验。

## RAG 实验论文阶段性汇总

当前已完成 RAG v3 定向优化实验，并新增面向毕业论文实验章节的阶段性汇总文档：

- 汇总文档：`docs/rag/rag_experiment_summary_for_thesis.md`
- 汇总脚本：`scripts/summarize_rag_experiments_for_thesis.py`
- 外部汇总输出目录：`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_experiment_summary_for_thesis`

当前默认 RAG 回答策略建议继续使用 `v2`，检索输入建议为 `real_hybrid_weighted + fixed_length + top_k=5`。v3 作为针对 `condition` 类型和 `medium` 难度问题的定向优化实验记录，不替代 v2 默认方案。

本阶段仍未调用大模型 API，仍未做 Reranker，仍未做 Web 前端，也未覆盖 Dense、BM25、Hybrid、Real Dense、Real Hybrid、RAG v1、RAG v2 或 RAG v3 的历史输出。下一步建议进入人工复核和论文实验章节撰写；如继续扩展实验，可在独立输出目录中设计 Reranker、LLM Judge 或真实大模型生成对照实验。

## 论文实验结果索引与表格

当前已完成论文实验汇总沉淀，并新增 `docs/results` 实验结果索引与结果表：

- 结果索引：`docs/results/experiment_results_index.md`
- 结果表格：`docs/results/experiment_results_tables.md`
- 结果摘要：`docs/results/experiment_results_summary.md`
- 结构化导出脚本：`scripts/export_experiment_results_tables.py`
- 外部结构化输出目录：`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\experiment_results_tables`

当前默认实验方案为 `real_hybrid_weighted + fixed_length + top_k=5 + RAG v2`。v3 继续作为 `condition` 与 `medium` 目标分组的定向优化实验记录，不替代 v2 默认方案。

下一步建议由人工基于这些结果表撰写论文正文，或继续进行人工复核、Reranker、LLM Judge 等扩展实验。本轮仍未调用大模型 API，未做 Reranker，未做 Web 前端，也未覆盖历史实验结果。

## RAG 人工复核辅助包

当前已新增 RAG 人工复核辅助包，用于把已有 RAG v1/v2/v3、v2 误差分析和实验结果汇总中的样本整理为可人工填写的 CSV 表格。

- 生成脚本：`scripts/build_manual_review_package.py`
- 使用说明：`docs/results/manual_review_package_guide.md`
- 外部输出目录：`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\manual_review_package`

人工复核包覆盖 `needs_review`、`worsened`、`condition`、`medium`、`unknown` 等样本，并生成全量 40 题复核总览表。当前默认方案仍为 `real_hybrid_weighted + fixed_length + top_k=5 + RAG v2`，v3 继续作为定向优化实验记录。

本阶段仍未调用大模型 API，未做 Reranker，未做 Web 前端，也未覆盖历史实验结果。下一步可由人工填写复核表，或继续设计 Reranker / LLM Judge 扩展实验。

## 实验数据资产管理

当前唯一实验主输出目录为：

`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook`

新实验使用 `runs\<experiment_id>` 规范目录，记录配置、指标、逐题结果、Python/依赖环境、Git 状态以及数据/Chunk/QA 的 SHA-256。现有旧格式结果保持不变，并以 `legacy=true` 登记到可重建总账：

`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\experiment_registry.csv`

常用命令：

```powershell
# 扫描规范运行目录和旧格式结果，重建总账并列出实验
.\.venv\Scripts\python.exe scripts\list_experiments.py --rebuild

# 筛选实验
.\.venv\Scripts\python.exe scripts\list_experiments.py --task-type retrieval
.\.venv\Scripts\python.exe scripts\list_experiments.py --retrieval-method real_dense
.\.venv\Scripts\python.exe scripts\list_experiments.py --chunk-strategy fixed_length

# 从总账和真实 metrics/legacy 报告导出论文 CSV/Markdown
.\.venv\Scripts\python.exe scripts\export_thesis_tables.py
```

文档：

- 资产审计：`docs/experiments/experiment_asset_audit.md`
- 运行目录与总账规范：`docs/experiments/experiment_data_management.md`
- 论文数据查询、导出与误差分析：`docs/experiments/thesis_data_retrieval.md`

所有缺失论文数据保持空白并进入缺失报告，禁止估算或伪造。导出使用新的时间戳目录，非空目录会被拒绝，避免覆盖历史结果。

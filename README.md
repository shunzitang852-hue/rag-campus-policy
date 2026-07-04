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

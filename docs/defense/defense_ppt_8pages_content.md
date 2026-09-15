# 8 页开题答辩 PPT 内容说明

论文题目固定为《基于RAG技术的高校政策知识问答系统构建与检索策略优化研究》。

## 第1页：题目页

- 主标题：论文完整题目。
- 副标题：本科毕业论文开题答辩。
- 补充说明：面向真实政策语料的检索增强生成与策略比较。
- 信息栏：姓名、学号、专业、指导教师、时间。
- 讲解时间：8 秒。

## 第2页：研究背景与核心问题

- 高校政策信息特点：文件多且分散、条款结构复杂、用户倾向自然语言提问。
- 传统关键词检索痛点：依赖词面匹配，语义理解有限，仍需人工筛选。
- 直接使用大语言模型的风险：政策知识滞后、幻觉、缺乏可信依据。
- 核心逻辑：RAG 先检索可信政策证据，再基于证据生成回答。
- 讲解时间：20 秒。

## 第3页：研究目标与主要研究内容

- 研究目标：构建面向高校政策场景的 RAG 知识问答系统，并研究切分和检索策略对问答效果的影响。
- 政策知识库构建：数据清洗、结构化处理。
- Chunk Strategy：fixed_length、article_based、structure_aware。
- Retrieval Strategy：BM25、Dense、Hybrid。
- RAG 与评价：Extractive baseline、Generative RAG、自动评价、人工复核。
- 核心研究变量：Chunk Strategy × Retrieval Strategy。
- 研究重点：检索策略优化，而非单纯开发聊天机器人。
- 讲解时间：18 秒。

## 第4页：数据与实验设计

- 数据来源：杭州电子科技大学信息工程学院学生政策 / 学生手册。
- QA Test Set：40 道高校政策问答题。
- Chunk 数：Fixed Length 244、Article Based 358、Structure Aware 358。
- 检索评价：Recall@K、MRR@K、Source Policy Hit、Relevant Chunk Hit。
- RAG 评价：Keyword overlap、Evidence overlap、Automatic Good / Needs Review、典型样本人工复核。
- 可追溯性：实验记录保存数据指纹、配置参数和逐题结果。
- 讲解时间：20 秒。

## 第5页：RAG 系统整体技术路线

- 主流程：政策原文 → 清洗与结构化 → Chunking → Retrieval → Top-5 政策证据 → Context / Prompt Builder → DeepSeek Generative RAG → Answer + Citation → 自动评价 + 人工复核。
- 核心强调：Chunking 与 Retrieval 是检索策略优化核心环节。
- Final Retrieval：Hybrid + Fixed Length + Top-K=5。
- 权重：Dense / BM25 = 0.5 / 0.5。
- Dense 模型：BAAI/bge-small-zh-v1.5。
- Citation：由 Retriever metadata 程序化生成。
- 讲解时间：27 秒。

## 第6页：检索策略实验与优化结果

公平对照条件：fixed_length、Top-K=5、40 道 QA。

| Method | Recall@5 | MRR@5 | 说明 |
| --- | ---: | ---: | --- |
| BM25 | 0.9500 | 0.8708 | 关键词匹配 |
| Dense Retrieval | 1.0000 | 0.9833 | 真实中文语义表示 |
| Hybrid Retrieval | 1.0000 | 1.0000 | 关键词与语义融合 |

- 最终检索方案：real_hybrid_weighted + fixed_length + Top-K=5。
- Dense / BM25 权重：0.5 / 0.5。
- 数据来源：最新 thesis export `20260910T084017360137Z` 中的 normalized retrieval comparison，以及其对应的正式实验记录。
- 对照实验：BM25 `20260720T073712630329Z_retrieval_bm25_fixed-length_topk5_072c0358`；Dense `20260720T081619394523Z_retrieval_real-dense_fixed-length_topk5_24e94b06`；Hybrid `20260910T035306268526Z_retrieval_real-hybrid-weighted_fixed-length_topk5_2b110eac`。
- 讲解时间：25 秒。

## 第7页：生成式 RAG 阶段性实验结果

| Method | Keyword overlap | Evidence overlap | Automatic Good |
| --- | ---: | ---: | ---: |
| Extractive v1 | 0.5433 | 0.7658 | 20/40 |
| Extractive v2 | 0.5733 | 0.7929 | 21/40 |
| Extractive v3 | 0.5467 | 0.8721 | 20/40 |
| Generative RAG | 0.7033 | 0.8979 | 29/40 |

- Generation Success：40/40。
- Source Policy Hit：1.0。
- Relevant Chunk Hit：1.0。
- 说明：Good 是自动辅助评价，不代表人工准确率。
- Q018 成功案例：生成回答按学生和学校 / 学院重新组织转学条件、办理依据与开学前办理要求，多条件政策信息更清晰。
- Q032 自动评价局限案例：政策第一条只说明制定目的和依据，生成回答指出无法确认责任或义务。该案例体现 Question–Reference 对齐对自动评分的影响，分类为 automatic evaluation limitation / QA alignment issue，不归为 LLM failure。
- 人工评价边界：目前仅完成典型样本人工复核，不代表完整 40 题人工评价。
- 讲解时间：37 秒。

## 第8页：当前进展与后续计划

已完成：政策数据清洗与结构化、三种 Chunk Strategy、BM25 / Dense / Hybrid、最终检索实验矩阵、40 题 QA Test Set、Extractive v1/v2/v3、DeepSeek 真实 Generative RAG、40/40 正式生成实验、规范化实验管理体系、典型样本人工复核。

后续工作：完成全部 40 题人工评价、分析困难样本、核验 Question–Reference 对齐问题、完善论文方法与实验讨论、完成图表与参考文献及格式检查。

当前研究特点：面向真实高校政策场景，系统比较 Chunk 与 Retrieval Strategy，实验参数与结果可追溯。

结尾：谢谢各位老师，请批评指正。

讲解时间：20 秒。


# PPT 页面显示文字

## 第1页

本科毕业论文开题答辩

《基于RAG技术的高校政策知识问答系统构建与检索策略优化研究》

面向真实政策语料的检索增强生成与策略比较

姓名：________　学号：________　专业：计算机科学与技术

指导教师：________　时间：________

## 第2页

研究背景与核心问题

高校政策信息特点

- 文件数量多，文本长且分散
- 条款结构复杂，跨章节关联
- 用户倾向使用自然语言提问

现有方法的两类痛点

传统关键词检索

依赖词面匹配，语义理解有限；找到文件后仍需人工筛选。

直接使用大语言模型

政策知识可能滞后，存在幻觉风险，回答缺乏可信依据。

用户问题 → 关键词检索 / 直接 LLM → 筛选困难 / 幻觉风险 → RAG

先检索可信政策证据，再基于证据生成回答

## 第3页

研究目标与主要研究内容

构建面向高校政策场景的 RAG 知识问答系统，并研究切分和检索策略对问答效果的影响。

政策知识库构建：数据清洗；结构化处理

Chunk Strategy：fixed_length；article_based；structure_aware

Retrieval Strategy：BM25；Dense；Hybrid

RAG 与评价：Extractive baseline；Generative RAG；自动评价与人工复核

核心研究变量：Chunk Strategy × Retrieval Strategy

研究重点：检索策略优化，而非单纯开发聊天机器人

## 第4页

数据与实验设计

数据来源

杭州电子科技大学信息工程学院学生政策 / 学生手册

QA Test Set：40 道高校政策问答题

Chunk 数据

Fixed Length 244

Article Based 358

Structure Aware 358

检索评价：Recall@K；MRR@K；Source Policy Hit；Relevant Chunk Hit

RAG 评价：Keyword overlap；Evidence overlap；Automatic Good / Needs Review；典型样本人工复核

数据 → Chunk → Retriever → Evaluation

实验记录保存数据指纹、配置参数和逐题结果，保证过程可追溯。

## 第5页

RAG 系统整体技术路线

政策数据处理

检索策略优化核心环节

证据约束生成

高校政策原始文本 → 清洗与结构化 → Chunking 三种切分 → Retrieval BM25 / Dense / Hybrid → Top-5 政策证据 → Context + Prompt Builder → DeepSeek Generative RAG → Answer + Citation → 自动评价 + 人工复核

Final Retrieval

Hybrid + Fixed Length + Top-K=5

Dense / BM25 = 0.5 / 0.5

Dense Model

BAAI/bge-small-zh-v1.5

Citation

由 Retriever metadata 程序化生成

## 第6页

检索策略实验与优化结果

同一条件：fixed_length｜Top-K=5｜40 题

BM25：关键词匹配；R@5 0.9500；MRR 0.8708

Dense Retrieval：语义表示；R@5 1.0000；MRR 0.9833

Hybrid Retrieval：关键词 + 语义融合；R@5 1.0000；MRR 1.0000

图表：Recall@5 与 MRR@5

最终检索方案

Hybrid + Fixed Length + Top-K=5

Dense / BM25 0.5 / 0.5

公平对照基于最新 normalized final retrieval experiments；MRR 按 Top-5 截断。

## 第7页

生成式 RAG 阶段性实验结果

Keyword overlap：Ext v1 0.5433；Ext v2 0.5733；Ext v3 0.5467；Gen RAG 0.7033

Evidence overlap：Ext v1 0.7658；Ext v2 0.7929；Ext v3 0.8721；Gen RAG 0.8979

40 / 40 真实生成成功

29 / 40 自动评价 Good

自动 Good 仅作辅助，不等同人工准确率

Q018｜转学政策

问题核心：转学条件、主体与办理时间

Generative RAG 按学生与学校 / 学院重新组织，完整保留转学条件、办理依据和开学前办理。

多条件政策信息组织更清晰

Q032｜自动评价局限

题目：第一条中的责任或义务是什么？

政策原文仅说明制定目的和依据。模型回答：“无法从给定政策材料确认责任或义务。”

Question–Reference 对齐会影响自动评分，仍需人工复核

## 第8页

当前进展与后续计划

已完成

✓ 政策数据清洗与结构化

✓ 三种 Chunk Strategy

✓ BM25 / Dense / Hybrid

✓ 最终检索实验矩阵

✓ 40 题 QA Test Set

✓ Extractive v1 / v2 / v3

✓ DeepSeek 真实 Generative RAG

✓ 40 / 40 正式生成实验

✓ 规范化实验管理体系

✓ 典型样本人工复核

后续工作

1. 完成全部 40 题人工评价
2. 深入分析困难样本
3. 核验 Question–Reference 对齐问题
4. 完善论文方法、实验与讨论章节
5. 完成图表、参考文献及格式检查

当前研究特点

真实高校政策场景

系统比较 Chunk 与 Retrieval Strategy

实验参数与结果可追溯

谢谢各位老师，请批评指正


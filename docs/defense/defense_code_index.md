# 开题答辩核心代码索引

以下页码对应 `defense_ppt_full_15pages.pptx`。PPT 中代码均来自当前项目真实实现；省略号仅用于隐藏与本页论点无关的真实代码。

| 页码 | 模块 | 文件 | 函数 / 类 | 答辩说明 |
| --- | --- | --- | --- | --- |
| 6 | 政策处理与知识切分 | `src/preprocessing/text_cleaner.py`、`src/preprocessing/policy_splitter.py`、`src/preprocessing/structure_parser.py`、`src/chunking/chunker.py` | `generate_chunks`、`fixed_length_chunks`、`article_based_chunks`、`structure_aware_chunks` | 将不同政策结构转换为可比较的检索单元。 |
| 7 | BM25 检索 | `src/retrieval/bm25.py` | `tokenize`、`score_document`、`search` | 构建可复现的中文词面检索基线。 |
| 7 | Dense 检索 | `src/retrieval/real_dense.py` | `search_with_provider` | 编码问题并按余弦相似度检索真实向量。 |
| 8 | Hybrid 融合 | `src/retrieval/hybrid.py`、`src/retrieval/real_hybrid.py` | `normalize_scores`、`fuse_weighted_score`、`fuse_real_weighted_score` | 对单题分数归一化并融合 lexical 与 semantic signals。 |
| 8 | 正式 Hybrid 运行配置 | `scripts/run_normalized_real_hybrid.py` | `main` | 固定权重 0.5/0.5、候选池 50、Top-K=5、per-query min-max。 |
| 9 | RAG 主链路 | `src/rag/pipeline.py` | `answer_query` | 串联检索、Context、生成、Citation 与统一结果结构。 |
| 9 | Context 构建 | `src/rag/context_builder.py` | `build_context` | 控制 Top-K、摘录长度和最大上下文字符数。 |
| 10 | Prompt | `src/rag/prompt_builder.py` | `SYSTEM_INSTRUCTION`、`build_prompt` | 约束模型仅依据政策 Context 作答。 |
| 10 | LLM Provider | `src/rag/llm_provider.py` | `OpenAICompatibleProvider`、`build_payload`、`generate` | 构建并发送兼容接口的 generation 请求。 |
| 10 | 回答生成 | `src/rag/llm_generator.py` | `generate_llm_answer` | 证据为空时不调用模型，并保留 Prompt 版本信息。 |
| 10 | 程序化引用 | `src/rag/citation.py` | `Citation`、`build_citations` | 从 Retriever Context 元数据生成可追溯引用。 |
| 11 | 实验记录 | `src/experiments/recorder.py` | `ExperimentRecorder.start`、`record_results` | 保存配置、环境、数据清单、指标和逐题结果。 |
| 11 | 实验总账 | `src/experiments/registry.py` | `_scan_run` | 扫描持久化实验并统一登记状态与产物。 |
| 11 | 论文导出 | `src/experiments/thesis_export.py` | `export_thesis_tables` | 从总账中的正式结果生成论文表格。 |
| 11 | 规范化 RAG 执行 | `scripts/run_normalized_rag.py` | `main` | 执行固定协议下的正式生成式 RAG workflow。 |

## 追问定位

- 老师问“切分如何切换”：第 6 页，`src/chunking/chunker.py::generate_chunks`。
- 老师问“为什么 Hybrid 有效”：第 8 页，`normalize_scores` 与加权 `score_fn`。
- 老师问“模型如何避免自由发挥”：第 10 页，`SYSTEM_INSTRUCTION`。
- 老师问“引用是否由模型编写”：第 10 页，`src/rag/citation.py::build_citations`。
- 老师问“实验如何复现”：第 11 页，`ExperimentRecorder`、`Registry`、`Thesis Export`。

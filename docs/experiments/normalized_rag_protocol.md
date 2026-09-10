# 规范化 RAG 实验协议（2B-1 冻结版）

## 1. 版本命名与算法来源

- `extractive_v1`：历史提交 `5acb30b` 的规则式抽取器。按问题字符、2/3/4 字组合命中数排序，优先高命中、靠前 Chunk、短句，最多三句。
- `extractive_v2`：历史提交 `5c9aa82` 的规则式抽取器。在 v1 上增加问题类型词、条款锚点、首个文档加分、单句截断和最多四句配置。
- `extractive_v3`：现有 `answer_v3_optimization.py` 的规则式题型模板、多证据点选择和证据不足提示。
- `llm_grounded_v1`：Final Retriever → Context Builder → `policy_grounded_qa_v1` → DeepSeek → 程序化引用。它是唯一真实 LLM 版本，不属于 extractive v1/v2/v3。

前三个版本均为规则/抽取式生成，不调用外部模型。兼容字段 `legacy_version=v1/v2/v3` 仅用于映射旧结果。

## 2. 冻结输入

- QA：`hdu_ies_student_handbook_qa_testset_v1.jsonl`，40 题，顺序固定，SHA-256 `4ECCD47E5FD6FEF7F5BBB77F13FC7646A9EE674961C3BE87A42CA0D7BE2D9CC5`。
- 原始源数据 SHA-256：`83937E33D6E10780DF6BEDEBFB9FD26A2D5D6DE940CF85E8D3F13E3D32165189`。
- 清洗数据 SHA-256：`13C309934F6413A02D2D4CAFFD3CFD737FB9316B8C4ED285C4443853FDADC2CB`。
- fixed-length Chunk SHA-256：`B89EF648A39BAA1E332FA63F6BF76FF6AFDC8B52C8245F233E68999BDDFADC36`。

## 3. Final Retrieval Protocol

- 来源规范化检索实验：`20260910T035306268526Z_retrieval_real-hybrid-weighted_fixed-length_topk5_2b110eac`。
- 方法：`real_hybrid_weighted`；Chunk：`fixed_length`；Top-K：5。
- Dense：`BAAI/bge-small-zh-v1.5`，snapshot revision `7999e1d3359715c523056ef9478215996d62a620`，本地离线加载。
- Dense/BM25/Fusion candidate pool：均为 50。
- 权重：Dense 0.5、BM25 0.5；每题 min-max 归一化；等分值映射为 0.0。
- BM25：CJK character bigram，k1=1.5，b=0.75。
- 随机种子：42。

Runner 在开始前逐字段验证上述配置，并读取该规范化检索实验已持久化的逐题 Top-5。三组生成实验因此共享完全相同、可哈希追踪的检索输入；答案仍由当前冻结代码逐题重新生成，不复制任何 legacy 答案。

## 4. 自动指标口径

实现位于 `src/rag/answer_quality.py`。

| 指标 | 计算口径 | 输入 | 适用范围 | 解释限制 |
| --- | --- | --- | --- | --- |
| `keyword_overlap_ratio` | 标准答案关键词中出现在生成答案里的比例 | expected_answer、generated_answer | extractive / generative | 辅助覆盖率，不是人工准确率 |
| `evidence_overlap_ratio` | 证据关键词中出现在生成答案里的比例 | evidence_keywords 或 evidence_text、generated_answer | extractive / generative | 辅助证据覆盖率，不代表事实完全正确 |
| `source_policy_hit` | gold 政策标题是否出现在程序化 citations | source_policy_title、citations | extractive / generative | 衡量检索/引用命中，不衡量答案语义 |
| `relevant_chunk_hit` | gold 相关 Chunk 与 Top-5 是否有交集 | related_chunk_ids_fixed_length、retrieved_chunk_ids | extractive / generative | 衡量检索命中，不衡量生成质量 |
| `citation_coverage_rate` | 有 citation 的题数 / 总题数 | citations | extractive / generative | 引用存在性，不验证引用是否充分 |
| `auto_quality_label` | 答案和引用非空，且政策、Chunk 命中且关键词覆盖 ≥ 0.5 时为 good | 上述字段 | extractive / generative | 规则型复核信号，绝不称为人工准确率 |

## 5. 生成式固定配置与安全边界

`llm_grounded_v1` 固定为 DeepSeek OpenAI-compatible、`https://api.deepseek.com`、`deepseek-v4-flash`、thinking disabled、temperature 0.0、max_tokens 512、prompt `policy_grounded_qa_v1`。API Key 仅从 `DEEPSEEK_API_KEY` 读取。

默认执行 dry-run，不创建 Provider、不发送 HTTP。真实批次必须额外显式提供 `--execute-llm` 和固定 `--experiment-id`。已成功题目以及默认情况下已失败题目均受 `(experiment_id, question_id)` 重复保护；断点续跑只处理未尝试题。失败只记录脱敏后的类型和消息。自动重试为 0；如需重试历史失败题，必须显式提供 `--retry-failed`。

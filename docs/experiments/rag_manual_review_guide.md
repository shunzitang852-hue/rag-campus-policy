# RAG 人工复核评分指南

## 1. 复核对象与原则

本指南用于复核 `rag_manual_review.csv` 中 40 道 QA × 4 个生成版本的 160 行正式结果。复核时应同时阅读题目、reference answer、完整 Top-5 检索正文、生成答案和程序化 citations。

人工评价结果由研究者阅读问题、参考答案、政策原文、检索证据及模型正式回答后确认；自动指标仅作为辅助信息，不直接产生人工评分。

自动 Keyword、Evidence、Good/Review 只用于定位候选样本，不能替代人工判断。复核者不得改写生成答案或 reference，也不得依据自动标签反推人工分数。

## 2. 人工评分标准

### Correctness

- `2`：回答中的核心事实、条件、限制、主体和程序均正确。
- `1`：基本正确，但存在轻微错误、局部歧义或部分内容错误。
- `0`：核心内容明显错误、答非所问或混淆政策。

### Completeness

- `2`：主要要求和关键条件完整覆盖。
- `1`：核心内容已覆盖，但遗漏次要信息或个别条件。
- `0`：遗漏核心事实、关键条件、限制、时间或适用范围。

### Citation Accuracy

- `2`：回答中的事实均能由正式 citations 指向的检索正文直接支持，引用对应清楚。
- `1`：大部分内容有证据支持，但部分表述与引用对应关系不明确。
- `0`：存在明显无证据支持的内容，或引用指向错误政策/Chunk。

### Hallucination

- `0`：未发现明显无证据政策事实。
- `1`：存在疑似无证据扩展，需要进一步核对。
- `2`：存在明确的政策事实编造或与证据冲突的陈述。

Hallucination 的方向与前三项相反：`0` 最好，`2` 最差。不得仅因回答比 reference 更长就认定 hallucination；应检查新增事实能否由正式引用正文支持。

## 3. Error Type 可选标签

| 标签 | 含义 |
| --- | --- |
| `correct` | 未发现需要记录的实质问题 |
| `retrieval_noise` | Top-5 包含较多无关 Chunk，干扰回答 |
| `retrieval_missing` | Top-5 未召回回答所需的核心证据 |
| `chunk_boundary` | 核心条款被截断或跨 Chunk，导致上下文不完整 |
| `incomplete_answer` | 回答总体相关，但未覆盖主要内容 |
| `irrelevant_content` | 回答包含与问题无关的政策内容 |
| `condition_omission` | 遗漏关键适用条件、限制、时间或例外 |
| `citation_mismatch` | 正式引用与回答陈述不能正确对应 |
| `unsupported_generation` | 生成内容无法由正式检索证据支持 |
| `policy_confusion` | 混淆不同政策、制度或适用主体 |
| `over_generation` | 答案范围不必要地扩大，加入超出问题所需的内容 |
| `other` | 其他问题；必须在 reviewer_notes 中说明 |

Error Type 由人工选择，程序不得自动填写。若一条记录存在多个问题，可在单元格中用分号分隔标签，并在 reviewer_notes 解释主次关系。

## 4. 推荐复核顺序

1. 先只根据题干判断问题要求的条款、主体和回答范围。
2. 阅读 Gold policy 与完整 Top-5 正文，确认核心证据是否已召回、是否被 Chunk 截断。
3. 对照 reference 与 generated answer，但同时检查题干和 reference 是否存在范围或口径差异。
4. 检查答案中的每项政策事实能否追溯到正式 citations。
5. 最后填写五项判断和 reviewer_notes；不得将自动 Good 直接换算成人工高分。

## 5. 当前留空字段

以下字段必须由后续人工复核者填写，当前版本全部为空：`correctness`、`completeness`、`citation_accuracy`、`hallucination`、`error_type`、`reviewer_notes`。

# 杭电信工学生手册解析报告

## 1. 本次处理范围

本次处理对象为杭州电子科技大学信息工程学院学生手册 TXT 原始文件，工作范围仅包括制度级拆分、章节条款结构化解析、保守文本清洗、异常记录、统计和自动检查。

未执行 Embedding、ChromaDB、BM25、Hybrid 检索、RAG 问答、正式测试集答案生成、OCR、大模型摘要或大模型改写。

## 2. 使用脚本

```text
python scripts/split_student_handbook_policies.py
python scripts/parse_policy_structure.py
python scripts/validate_student_handbook_parsing.py
python scripts/review_student_handbook_parsing.py
```

项目虚拟环境解释器：

```text
F:\GraduationThesis\rag-campus-policy\.venv\Scripts\python.exe
```

正式 pytest 已安装并运行：

```text
Python 3.12.10
pytest 9.1.1
22 passed in 0.28s
```

## 3. 输入文件和哈希

输入 raw 文件：

```text
F:\GraduationThesis\datasets\raw\hdu_ies\student_handbook\hdu_ies_student_handbook_source.txt
```

SHA-256：

```text
83937E33D6E10780DF6BEDEBFB9FD26A2D5D6DE940CF85E8D3F13E3D32165189
```

本次处理未修改 raw 原始文件。

## 4. 输出文件

制度拆分输出：

```text
F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\policies
F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\hdu_ies_student_handbook_policies.jsonl
```

结构化解析输出：

```text
F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\hdu_ies_student_handbook_articles.jsonl
F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\parse_anomalies.jsonl
F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\parsing_statistics.json
F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\coverage_gap_report.json
F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\unclassified_text_analysis.csv
F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\possible_table_review.csv
```

清洗输出：

```text
F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\policies_clean
F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\hdu_ies_student_handbook_articles_clean.jsonl
F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\hdu_ies_student_handbook_full_clean.txt
```

人工核验包：

```text
F:\GraduationThesis\datasets\evaluation\hdu_ies\manual_review\parsing_stage
```

## 5. 制度拆分结果

| 指标 | 数量 |
| --- | ---: |
| 制度目录项 | 32 |
| 制度拆分成功 | 32 |
| 精确标题匹配 | 31 |
| 模糊标题匹配 | 1 |
| 未匹配 | 0 |
| 空制度文件 | 0 |

模糊匹配项为 `HDU_IES_SH_014`，目录规范标题为“杭州电子科技大学信息工程学院学生解除处分管理办法”，正文实际标题为“杭州电子科技大学信息工程学院学生校解除处分管理办法（试行）”。

## 6. 章节条款统计

| 指标 | 数量 |
| --- | ---: |
| 结构化记录 | 1744 |
| 章节 | 117 |
| 小节 | 7 |
| 条款 | 538 |
| 款项 | 457 |
| 子项 | 211 |
| 无编号段落 | 182 |
| 无法分类段落 | 189 |

解析器允许制度不存在完整“章—节—条”结构；无法分类文本仍保留为记录，并标注 `parse_status=unclassified`。

## 7. 清洗操作统计

| 清洗动作 | 次数 |
| --- | ---: |
| normalize_full_width_spaces | 171 |
| strip_trailing_spaces | 250 |

结构记录清洗前字符数为 87941，清洗后字符数为 87683，变化为 -258。清洗仅执行保守空白处理，保留数字、日期、金额、比例、文号、条款编号和状态词。

## 8. 原始字符覆盖率

原始字符覆盖率为 `0.9994438574050386`。

计算方式：

```text
sum(len(policy.raw_text)) / len(raw text from first policy start line through last policy end line, joined with LF)
```

`has_body_text_outside_any_policy=true`，原因是制度边界按标题行严格切分，标题间空白或少量非制度正文字符未纳入制度文本；未发现制度正文静默丢弃。

## 8.1 覆盖率缺口定位

覆盖率缺口报告：

```text
F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\coverage_gap_report.json
```

| 指标 | 数值 |
| --- | ---: |
| raw_character_count | 91099 |
| covered_character_count | 89685 |
| uncovered_character_count | 1414 |
| coverage_ratio | 0.984478424571071 |
| uncovered_policy_body_character_count | 0 |

缺口分类：

| 分类 | 数量 |
| --- | ---: |
| blank_or_formatting | 227 |
| table_of_contents | 33 |
| preface | 3 |

未覆盖内容为前言、目录和空白/排版留白；未发现未进入结构化记录的政策正文字符。

## 9. 异常列表摘要

| 异常类型 | 数量 |
| --- | ---: |
| title_mismatch | 1 |
| missing_document_number | 1 |
| possible_table | 43 |
| unclassified_text | 189 |

异常记录仅用于提醒人工复核，不导致正文删除。

## 9.1 unclassified_text 二次分类

189 条 `unclassified_text` 已全部写入：

```text
F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\unclassified_text_analysis.csv
```

| 二次分类 | 数量 |
| --- | ---: |
| policy_basis | 32 |
| policy_preamble | 39 |
| signature_or_date | 3 |
| supplementary_note | 2 |
| unnumbered_paragraph | 113 |

`parse_error` 数量为 0，`unknown` 数量为 0。本轮未为了降低异常数量而删除原异常记录。

## 9.2 possible_table 复核

43 条 `possible_table` 已全部写入：

```text
F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\possible_table_review.csv
```

按制度分布：

| policy_id | 数量 |
| --- | ---: |
| HDU_IES_SH_001 | 41 |
| HDU_IES_SH_017 | 1 |
| HDU_IES_SH_027 | 1 |

本轮复核结论为保留原文，不强制恢复 Markdown 表格；候选行均需人工抽查其表格或排版语义。

## 10. 标题差异处理方式

“学生解除处分管理办法”按受控模糊匹配处理：

| 字段 | 值 |
| --- | --- |
| policy_id | `HDU_IES_SH_014` |
| policy_title_canonical | 杭州电子科技大学信息工程学院学生解除处分管理办法 |
| policy_title_original | 杭州电子科技大学信息工程学院学生校解除处分管理办法（试行） |
| title_match_status | `fuzzy` |
| verification_status | `pending_manual_verification` |

raw 原文未修改，需结合纸质手册或其他正式版本人工核验。

## 11. 尚未解决的问题

1. 第 14 项制度标题存在目录与正文差异。
2. “杭州电子科技大学信息工程学院学生公寓管理办法（暂行）”未登记明确文号。
3. `possible_table` 和 `unclassified_text` 记录需要人工抽查，判断是否为表格、断行或正常无编号段落。
4. 学生手册正式来源、纸质版本或官方网站来源仍需补充。
5. 制度有效性不得由本次解析自动判断。

## 11.1 是否修改解析器

本轮修改了解析与验收相关脚本。关键修复包括：制度标题匹配从目录后开始，且三行标题窗口命中时使用实际标题起始行，避免第 14 项制度夹带上一制度结尾。

修复前后核心统计对比：

| 指标 | 修复前 | 修复后 |
| --- | ---: | ---: |
| 制度数量 | 32 | 32 |
| 章节 | 117 | 117 |
| 小节 | 7 | 7 |
| 条款 | 538 | 538 |
| 款项 | 457 | 457 |
| 子项 | 211 | 211 |
| 异常总数 | 234 | 234 |
| uncovered_policy_body_character_count | 未生成 | 0 |

## 12. 人工核验建议

建议优先核验：

1. 学生学籍管理实施细则。
2. 本科毕业生学士学位授予细则。
3. 学生考试违规、作弊的认定办法。
4. 学生纪律处分实施细则。
5. 学生解除处分管理办法。
6. 关于学生参加毕业设计（论文）的若干规定。
7. 学生公寓管理办法（暂行）。

## 12.1 人工抽样核验结果

人工核验包目录：

```text
F:\GraduationThesis\datasets\evaluation\hdu_ies\manual_review\parsing_stage
```

本轮检查到 7 个重点制度人工核验文件均存在。根据文件中的填写区内容，5 个文件已完整填写，2 个文件仍保留部分模板选项，需后续补齐归档记录。本轮用户消息已确认允许进入 Chunk 设计阶段；该确认不替代未完整填写文件的逐项归档记录。

| 制度 | 核验状态 | 串文 | 漏文 | 数字或比例丢失 | 表格语义破坏 | Chunk阶段 |
| --- | --- | --- | --- | --- | --- | --- |
| 学生纪律处分实施细则（试行） | 已填写 | 否 | 否 | 未发现 | 未发现 | 是 |
| 学生解除处分管理办法 | 已填写 | 否 | 否 | 未发现 | 未发现 | 是 |
| 学生考试违规、作弊的认定办法 | 未完整填写，仍保留模板选项 | 未确认 | 未确认 | 未确认 | 未确认 | 未确认 |
| 学生学籍管理实施细则 | 已填写 | 否 | 否 | 未发现 | 未发现 | 是 |
| 关于学生参加毕业设计（论文）的若干规定 | 未完整填写，仍保留模板选项；核验人和日期已填写 | 未确认 | 未确认 | 未确认 | 未确认 | 未确认 |
| 本科毕业生学士学位授予细则（2023年修订） | 已填写 | 否 | 否 | 未发现 | 未发现 | 是 |
| 学生公寓管理办法（暂行） | 已填写 | 否 | 否 | 未发现 | 未发现 | 是 |

尚待后续核验事项：

1. 补齐“学生考试违规、作弊的认定办法”的人工核验填写区。
2. 补齐“关于学生参加毕业设计（论文）的若干规定”的逐项核验结论。
3. 后续进入 Chunk 设计阶段前，继续保留第 14 项标题差异和学生公寓管理办法缺文号事项作为人工核验提示。

## 13. 是否满足进入 Chunk 阶段的自动检查条件

自动检查条件满足：raw 哈希一致，32 项制度均已拆分，0 个制度为空，JSON/JSONL 可解析，标题差异已标记，结构记录均有来源行号，未覆盖政策正文字符为 0，189 条 `unclassified_text` 均已二次分类，43 条 `possible_table` 均有复核记录，正式 pytest 22 项全部通过，正式论文题目未变化，未生成 Chunk、Embedding、向量库、BM25、Hybrid 或 RAG 产物，7 项人工核验包已生成。

用户已在本轮消息中确认允许进入 Chunk 设计阶段。本报告同时保留本地核验文件填写不完整的客观记录，不替代用户的人工核验结论。

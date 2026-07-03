# 杭电信工学生手册解析规范

## 1. 输入文件

原始输入文件：

```text
F:\GraduationThesis\datasets\raw\hdu_ies\student_handbook\hdu_ies_student_handbook_source.txt
```

raw 文件只读，本阶段不得修改 raw 原始字节、编码、标题、文号、年份或正文内容。

## 2. 输入文件哈希

SHA-256：

```text
83937E33D6E10780DF6BEDEBFB9FD26A2D5D6DE940CF85E8D3F13E3D32165189
```

解析前必须校验该哈希；若不一致，立即停止。

## 3. 制度目录来源

制度目录来自：

```text
F:\GraduationThesis\datasets\metadata\hdu_ies\hdu_ies_student_handbook_document_catalog.csv
```

目录共 32 条。目录标题作为 `policy_title_canonical`，正文实际匹配标题作为 `policy_title_original`。

## 4. 制度拆分规则

1. 以前一个制度标题在正文中的匹配位置作为该制度开始。
2. 以下一个制度标题匹配位置的上一行为该制度结束。
3. 最后一项制度以 raw 文件末尾为结束。
4. 前言、目录和目录页码不得归入第一项制度正文。
5. 制度标题必须保留在制度文本中。
6. 拆分结果写入 `datasets\interim\hdu_ies\student_handbook\policies`。

## 5. 标题匹配规则

标题匹配采用受控规则：

1. 先对空白、全半角括号和文号括号进行规范化后执行精确匹配。
2. 支持最多 3 行窗口，以处理 raw 文本中由排版造成的标题断行。
3. 仅对已知标题差异使用受控模糊匹配，不使用无记录的宽松模糊匹配。
4. 模糊匹配必须记录 `title_match_status=fuzzy`、匹配依据和异常。

“杭州电子科技大学信息工程学院学生解除处分管理办法”采用受控模糊匹配：目录标题保持为规范标题，正文实际标题保存为 `policy_title_original`，异常记录为 `title_mismatch`，核验状态为 `pending_manual_verification`。

## 6. 章节识别规则

识别 `第一章`、`第二章` 等章节标题。章节号写入 `chapter_number`，章节标题写入 `chapter_title`。没有章节的制度允许章节字段为空。

## 7. 小节识别规则

识别 `第一节`、`第二节` 等小节标题。小节号写入 `section_number`，小节标题写入 `section_title`。没有小节的制度允许小节字段为空。

## 8. 条款识别规则

识别 `第一条`、`第二条` 等条款。条款号写入 `article_number`，条款顺序写入 `article_index`。条款编号不得从 `content_raw` 中永久删除。

## 9. 款项识别规则

识别以下款项或子项：

1. `（一）`、`（二）` 等款项。
2. `1.`、`2.`、`1、`、`2、` 等数字子项。
3. `一、`、`二、` 等中文编号列表。

解析器允许字段为空，但不得丢失正文。

## 10. 无编号内容处理规则

无编号段落保留为独立结构记录。能够归入当前章节或条款上下文的，保留当前 `heading_path`；无法归类的，标记 `parse_status=unclassified` 并记录异常。

## 11. 表格文本处理规则

当前输入为 TXT，不重建表格。含较多空白、全角空格或疑似排版对齐的行保留原文，标记 `possible_table` 异常，供人工核验。

## 12. 清洗允许项

1. 统一换行符。
2. 删除行末多余空格。
3. 规范连续空行。
4. 合并或保留明显由排版造成的断行，但必须保留可追溯原文。
5. 统一部分全角空格。
6. 清除明确重复的页眉页脚。
7. 保留所有数字、日期、金额、比例、文号和专有名称。
8. 保留“试行”“暂行”“修订”“补充规定”等状态词。
9. 保留条款、款项和子项编号。

## 13. 清洗禁止项

1. 改写政策句子。
2. 自动摘要。
3. 删除被判断为不重要的内容。
4. 修改学校名称。
5. 自动纠正正文标题。
6. 推断缺失年份或文号。
7. 改变政策含义。
8. 把多条合并成一条摘要。
9. 使用大模型清洗正文。
10. 修改 raw 文件。

## 14. 异常记录规则

异常写入：

```text
F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\parse_anomalies.jsonl
```

异常类型包括 `title_mismatch`、`policy_boundary_uncertain`、`unclassified_text`、`article_number_gap`、`duplicate_article_number`、`heading_without_content`、`content_without_heading`、`possible_table`、`suspicious_line_break`、`possible_duplicate_text`、`missing_document_number`、`unknown_structure`。不得因为存在异常而删除内容。

## 15. 输出文件规范

制度级 JSONL：

```text
F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\hdu_ies_student_handbook_policies.jsonl
```

结构化 JSONL：

```text
F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\hdu_ies_student_handbook_articles.jsonl
```

清洗输出：

```text
F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\policies_clean
F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\hdu_ies_student_handbook_articles_clean.jsonl
F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\hdu_ies_student_handbook_full_clean.txt
```

JSON 和 JSONL 使用 UTF-8 无 BOM；CSV 如需 Excel 兼容可使用 UTF-8 with BOM。原始文本与清洗文本必须分别保存。

## 16. 字符覆盖率计算

原始字符覆盖率计算为：

```text
sum(len(policy.raw_text)) / len(raw text from first policy start line through last policy end line, joined with LF)
```

该指标用于确认从第一项制度正文开始到最后一项制度正文结束的文本没有被静默丢弃。前言和目录不纳入制度正文覆盖率分母。

## 17. 人工核验要求

人工核验应重点检查制度边界、标题差异、章节条款归属、金额和比例、文号和年份、是否串文或漏文。解析结果不得替代人工核验结论，不得自动判断政策是否废止。

## 18. 本阶段验收标准

1. raw 哈希保持不变。
2. 制度拆分数量为 32。
3. JSONL 全部可解析。
4. 每条结构记录有 `policy_id`、`content_raw` 和来源行号。
5. 标题差异制度标记为 `fuzzy`。
6. 原始字符覆盖率达到合理阈值。
7. 未生成 Embedding、ChromaDB、BM25、Hybrid 检索或 RAG 问答结果。
8. 自动化测试通过。
9. 工作区不自动提交 Git。

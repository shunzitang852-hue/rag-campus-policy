# 杭电信工政策问答测试问题规划

## 一、适用范围

本文件用于规划杭州电子科技大学信息工程学院政策知识问答系统的测试问题集。

所有正式评测问题必须以杭州电子科技大学信息工程学院实际政策文件为依据。不得将杭州电子科技大学本部问题直接替换学校名称后作为杭电信工正式评测数据。

## 二、问题处理规则

1. 原来明确针对杭州电子科技大学本部的问题，先放入 `pending_verification` 或 `deprecated` 状态。
2. 能够基于杭电信工学生手册和相关制度重新核验的问题，才允许改写为杭电信工问题。
3. 标准答案、相关文档、条款和页码必须由人工核验后再标记为正式评测数据。
4. 没有明确政策依据的问题不得进入正式测试集。
5. 国家及教育部上位政策问题必须标注为补充语料，不得替代杭电信工制度依据。

## 三、第一版覆盖主题

第一版测试问题至少覆盖以下主题：

| 主题 | 状态 | 核验要求 |
| --- | --- | --- |
| 奖学金 | pending_verification | 需核验杭电信工奖学金制度、适用对象、条款和页码。 |
| 国家奖学金 | pending_verification | 需区分国家上位政策和学院评审细则。 |
| 国家励志奖学金 | pending_verification | 需核验学院评审流程和适用范围。 |
| 学籍管理 | pending_verification | 需核验学籍异动、注册、退学等条款。 |
| 休学复学 | pending_verification | 需核验学生手册或学籍制度中的具体条款。 |
| 转专业或转学 | pending_verification | 需核验是否为学院制度、学校上位制度或两者结合。 |
| 考试违规 | pending_verification | 需核验考试纪律和处分条款。 |
| 纪律处分 | pending_verification | 需核验处分等级、程序和申诉渠道。 |
| 学生申诉 | pending_verification | 需核验申诉机构、期限和流程。 |
| 学位授予 | pending_verification | 需核验学位授予条件和不授予情形。 |
| 毕业设计 | pending_verification | 需核验毕业设计与论文管理制度。 |
| 提前毕业 | pending_verification | 需核验是否存在明确制度依据。 |
| 创新创业 | pending_verification | 需核验创新创业学分或实践认定制度。 |
| 学科竞赛 | pending_verification | 需核验竞赛奖励、学分认定或评优规则。 |
| 公寓管理 | pending_verification | 需核验公寓管理制度和适用范围。 |

## 四、正式问题字段建议

后续正式测试集建议至少包含：

```text
question_id
institution_id
question
expected_answer
source_document_id
source_document_title
issuer
applicable_scope
page
clause
verification_status
verification_note
```

`institution_id` 默认使用 `hdu_ies`。正式数据中的 `verification_status` 必须由人工核验后填写。

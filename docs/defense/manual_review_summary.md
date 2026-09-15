# 当前人工复核摘要

## 复核范围

- 人工复核样本数：`6`
- Generation version：`llm_grounded_v1`
- 已确认题目：Q018、Q002、Q034、Q036、Q032、Q011
- Q039：未填写，不纳入本摘要

人工评价结果由研究者阅读问题、参考答案、政策原文、检索证据及正式模型回答后确认。Codex 仅按已确认结论录入，没有根据自动指标生成或修改人工评分。

## 已确认评分统计

| 项目 | 数量 |
| --- | ---: |
| 人工复核样本 | 6 |
| Correctness = 2 | 4 |
| Correctness = 1 | 2 |
| Hallucination = 0 | 6 |
| Error Type = correct | 4 |
| Error Type = over_generation | 2 |

Q002、Q034 的人工结论为 `over_generation`：回答内容具有检索上下文支持，但范围超过 Reference 对应范围。

Q032、Q011 的人工复核观察指出，自动低分需要结合 Question–Reference 语义对齐问题理解，不能直接认定为生成错误。该观察不改变 QA、Reference 或任何正式实验结果。

## PPT 案例固定

- Success Case：Q018
- Difficulty / Evaluation Limitation Case：Q032
- Q032 推荐分类：`automatic evaluation limitation / QA alignment issue`
- Q032 不标记为：`LLM failure`

## 解释边界

该结果为典型样本人工复核，不代表完整 40 题人工评价结果。不得将上述 6 题统计外推为 40 题总体准确率，也不得用自动 Good/Review 替代其余样本的人工判断。

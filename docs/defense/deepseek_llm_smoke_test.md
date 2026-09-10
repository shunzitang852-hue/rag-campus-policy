# DeepSeek 真实 LLM Smoke Test 记录

## 验证范围

- 时间：2026-09-10T14:17:55+08:00
- 类型：单题 Smoke Test，不是正式论文实验，不进入实验总账
- 请求次数：1
- 问题：学生申请解除处分需要满足什么条件？
- 检索器：`real_hybrid_weighted`
- Chunk 策略：`fixed_length`
- Top-K：`5`
- Provider：`openai_compatible`
- Endpoint：`https://api.deepseek.com/chat/completions`
- Model：`deepseek-v4-flash`
- Prompt version：`policy_grounded_qa_v1`
- Thinking：`disabled`
- Temperature：`0.0`
- Max tokens：`512`
- API Key：仅从 `DEEPSEEK_API_KEY` 环境变量读取，未写入本记录

## 真实生成结果

根据《杭州电子科技大学信息工程学院学生解除处分管理办法》，申请解除处分需满足以下条件：

1. 一般解除条件：处分期满；按时注册并缴纳学费；处分期内认识并改正错误，无违规违纪行为，每学期至少提交一份思想汇报；认真学习、团结同学、积极参加志愿或集体活动；未受到其他纪律处分。
2. 提前解除条件：实际处分时间达到规定期限的一半，在满足一般条件的基础上，还需满足制度列出的校级荣誉、论文、公益表彰或报道、升学就业、应征入伍或基层服务等条件之一。
3. 提前解除需提供真实证明材料；弄虚作假不得提前解除，并可延长处分期。
4. 警告、严重警告、记过期限为 6 个月，留校察看为 12 个月。
5. 申请由学生向所在学院提出，学院审核后通过钉钉 OA 填写申请，最终经学生工作部审核批准。

以上为模型原始回答的忠实压缩记录；未对事实进行美化或改写后重新调用模型。

## 程序引用

引用由本地 Retriever metadata 构造，不读取模型自行生成的引用：

1. `fixed_length_HDU_IES_SH_014_0127_f7ac141c1cdca30d`，杭州电子科技大学信息工程学院学生解除处分管理办法，score=`1.000000`
2. `fixed_length_HDU_IES_SH_014_0126_52ce86e6c3129444`，杭州电子科技大学信息工程学院学生解除处分管理办法，score=`0.760101`
3. `fixed_length_HDU_IES_SH_014_0125_ebdbfb646004a210`，杭州电子科技大学信息工程学院学生解除处分管理办法，score=`0.666144`

## 人工 Smoke 质量检查

- Grounded：YES
- Completeness：GOOD
- Hallucination detected：NO

理由：回答聚焦解除处分；一般解除的四项条件、提前解除条件、证明材料要求、处分期限和办理程序均能在前三个本地引用 Chunk 中找到依据。未出现 Context 外政策名称或不受证据支持的关键事实。

## 结论

- `REAL_LLM_AVAILABLE = YES`
- `REAL_LLM_SMOKE_TEST = PASS`
- 真实网络请求：YES
- Mock / Stub：NO
- Secret 泄漏：NO
- 历史实验结果修改：NO
